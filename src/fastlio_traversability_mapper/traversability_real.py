#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
REAL-TIME LIVOX TRAVERSABILITY NODE (ROS1)
============================================================================
Subscribes:
    /livox/lidar                        sensor_msgs/PointCloud2

Publishes:
    /traversability/traversable_cloud   PointCloud2 (traversable ground only, green)
    /traversability/cloud               PointCloud2 (ground colored red->green by T,
                                                    obstacles red)
    /traversability/obstacles           PointCloud2 (obstacle points, red)

NOTE: the sensor frame must be gravity aligned, OR set ~target_frame to a
gravity-aligned frame published by your LIO (e.g. odom / camera_init).
"""

import threading

import numpy as np
import rospy

from scipy.ndimage import (
    grey_opening,
    maximum_filter,
    minimum_filter,
    binary_dilation,
    distance_transform_edt,
)
from scipy.spatial import cKDTree

from sensor_msgs.msg import PointCloud2, PointField
import tf2_ros


# ============================================================
# POINTCLOUD2 <-> NUMPY
# ============================================================

def cloud_to_xyz(msg):
    """Fast extraction of x/y/z from PointCloud2 (works with Livox driver)."""
    fields = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields:
            rospy.logwarn_throttle(5.0, "PointCloud2 has no '%s' field." % name)
            return np.zeros((0, 3), np.float32)

    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [np.float32, np.float32, np.float32],
        "offsets": [fields["x"].offset, fields["y"].offset, fields["z"].offset],
        "itemsize": msg.point_step,
    })

    raw = bytes(msg.data)
    n = len(raw) // msg.point_step
    if n <= 0:
        return np.zeros((0, 3), np.float32)

    data = np.frombuffer(raw, dtype=dtype, count=n)
    pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    return pts[np.all(np.isfinite(pts), axis=1)]


def make_xyzrgb_cloud(pts, colors, frame_id, stamp):
    """Build an XYZRGB PointCloud2 message."""
    n = len(pts)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.is_dense = True
    msg.point_step = 16
    msg.row_step = 16 * n

    if n == 0:
        msg.data = b""
        return msg

    rgb = ((colors[:, 0].astype(np.uint32) << 16) |
           (colors[:, 1].astype(np.uint32) << 8) |
            colors[:, 2].astype(np.uint32))

    rec = np.zeros(n, dtype=np.dtype([
        ("x", np.float32), ("y", np.float32),
        ("z", np.float32), ("rgb", np.float32),
    ]))
    rec["x"] = pts[:, 0]
    rec["y"] = pts[:, 1]
    rec["z"] = pts[:, 2]
    rec["rgb"] = rgb.view(np.float32)
    msg.data = rec.tobytes()
    return msg


def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ============================================================
# PIPELINE HELPERS (vectorized for real time)
# ============================================================

def grid_extent(points, res):
    xmin = float(points[:, 0].min())
    xmax = float(points[:, 0].max())
    ymin = float(points[:, 1].min())
    ymax = float(points[:, 1].max())
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    return xmin, ymin, nx, ny


def voxel_downsample(points, voxel):
    idx = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((inverse.max() + 1, 3), np.float64)
    counts = np.zeros(inverse.max() + 1, np.int32)
    np.add.at(sums, inverse, points.astype(np.float64))
    np.add.at(counts, inverse, 1)
    return (sums / counts[:, None]).astype(np.float32)


def build_lower_grid(points, res):
    xmin, ymin, nx, ny = grid_extent(points, res)
    ix = np.clip(np.floor((points[:, 0] - xmin) / res).astype(np.int32), 0, nx - 1)
    iy = np.clip(np.floor((points[:, 1] - ymin) / res).astype(np.int32), 0, ny - 1)
    lower = np.full((ny, nx), np.inf, np.float32)
    np.minimum.at(lower, (iy, ix), points[:, 2])
    observed = np.isfinite(lower)          # inf -> not observed
    lower[~observed] = np.nan
    return lower, observed, xmin, ymin, nx, ny


def fill_grid(grid):
    """Fill NaN cells with nearest valid value (EDT)."""
    valid = np.isfinite(grid)
    if valid.all() or not valid.any():
        return grid.copy()
    _, indices = distance_transform_edt(~valid, return_indices=True)
    filled = grid.copy()
    filled[~valid] = grid[indices[0][~valid], indices[1][~valid]]
    return filled


def estimate_ground_surface(lower, observed, windows, thresholds):
    working = fill_grid(lower)
    for w, th in zip(windows, thresholds):
        opened = grey_opening(working, size=(w, w))
        remove = (working - opened) > th
        working[remove] = opened[remove]
        working[~observed] = np.nan
        working = fill_grid(working)
    surface = working.copy()
    surface[~observed] = np.nan
    return surface


def classify_points(points, surface, observed, xmin, ymin, res,
                    ground_tol, obstacle_h, max_ref_dist):
    valid = observed & np.isfinite(surface)
    if not valid.any() or len(points) == 0:
        return points[:0], points[:0]

    ys, xs = np.where(valid)
    cx = xmin + (xs.astype(np.float32) + 0.5) * res
    cy = ymin + (ys.astype(np.float32) + 0.5) * res
    tree = cKDTree(np.column_stack((cx, cy)))
    ref_z = surface[ys, xs]

    dist, idx = tree.query(points[:, :2], k=1)
    residual = points[:, 2] - ref_z[idx]

    supported = dist <= max_ref_dist
    ground = supported & (np.abs(residual) <= ground_tol)
    obstacle = supported & (residual >= obstacle_h)
    return points[ground], points[obstacle]


def build_terrain(ground, xmin, ymin, nx, ny, res):
    terrain = np.full((ny, nx), np.nan, np.float32)
    if len(ground) == 0:
        return terrain
    ix = np.floor((ground[:, 0] - xmin) / res).astype(np.int64)
    iy = np.floor((ground[:, 1] - ymin) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not ok.any():
        return terrain
    ix, iy, z = ix[ok], iy[ok], ground[ok, 2].astype(np.float64)
    flat = iy * nx + ix
    order = np.lexsort((z, flat))
    flat_s, z_s = flat[order], z[order]
    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    ends = np.r_[starts[1:], len(z_s)]
    for s, e in zip(starts, ends):          # median height per cell
        terrain.flat[flat_s[s]] = np.median(z_s[s:e])
    return terrain


def points_to_grid_mean(xy, values, shape, xmin, ymin, res):
    out = np.full(shape, np.nan, np.float32)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    ix = np.floor((xy[ok, 0] - xmin) / res).astype(np.int64)
    iy = np.floor((xy[ok, 1] - ymin) / res).astype(np.int64)
    inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    ix, iy, v = ix[inside], iy[inside], values[ok][inside]
    sums = np.zeros(shape, np.float64)
    counts = np.zeros(shape, np.int32)
    np.add.at(sums, (iy, ix), v)
    np.add.at(counts, (iy, ix), 1)
    m = counts > 0
    out[m] = (sums[m] / counts[m]).astype(np.float32)
    return out


def estimate_normals(points, k):
    """Fully vectorized PCA normals (smallest-eigenvalue eigenvector, +Z up)."""
    n = len(points)
    normals = np.full((n, 3), np.nan, np.float32)
    if n < 4:
        return normals
    k_eff = min(k + 1, n)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_eff)
    if k_eff == 1:
        idx = idx[:, None]
    neigh = points[idx]                              # (n, k, 3)
    centered = neigh - neigh.mean(axis=1)[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k_eff - 1, 1)
    _, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    normal = eigvecs[:, :, 0]
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    flip = normal[:, 2] < 0
    normal[flip] *= -1.0
    normals[:] = normal.astype(np.float32)
    return normals


def compute_roughness(points, normals, scales):
    """Multi-scale normal variation (vectorized version of the paper metric)."""
    n = len(points)
    if n == 0 or not np.isfinite(normals).any():
        return np.full(n, np.nan, np.float32)
    tree = cKDTree(points)
    per_scale = []
    for s in scales:
        k_eff = min(s + 1, n)
        _, idx = tree.query(points, k=k_eff)
        if k_eff == 1:
            idx = idx[:, None]
        nn = normals[idx]                            # (n, k, 3)
        n0 = normals[:, None, :]
        valid = np.isfinite(nn).all(axis=2) & np.isfinite(normals).all(axis=1)[:, None]
        signs = np.sign((nn * n0).sum(axis=2))
        signs[signs == 0] = 1.0
        signs[~valid] = 0.0
        diff = np.linalg.norm(nn * signs[:, :, None] - n0, axis=2)
        diff[~valid] = 0.0
        cnt = valid.sum(axis=1)
        vals = np.where(cnt > 0, diff.sum(axis=1) / np.maximum(cnt, 1), np.nan)
        per_scale.append(vals)
    rough = np.nanmean(np.vstack(per_scale), axis=0)
    return np.clip(rough, 0.0, 2.0).astype(np.float32)


def compute_features(terrain, ground_pts, xmin, ymin, res,
                     use_roughness, normal_k, rough_scales, max_rough_pts):
    valid = np.isfinite(terrain)
    slope = np.full(terrain.shape, np.nan, np.float32)
    rough = np.full(terrain.shape, np.nan, np.float32)
    step = np.full(terrain.shape, np.nan, np.float32)
    if not valid.any():
        return slope, rough, step, valid

    filled = fill_grid(terrain)

    # SLOPE
    dx = np.gradient(filled, res, axis=1)
    dy = np.gradient(filled, res, axis=0)
    sl = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    slope[valid] = sl[valid]

    # PAPER MULTI-SCALE ROUGHNESS
    if use_roughness and len(ground_pts) >= 10:
        rp = ground_pts[:, :3]
        if len(rp) > max_rough_pts:
            sel = np.random.choice(len(rp), max_rough_pts, replace=False)
            rp = rp[sel]
        normals = estimate_normals(rp, normal_k)
        pr = compute_roughness(rp, normals, rough_scales)
        rough = points_to_grid_mean(rp[:, :2], pr, terrain.shape, xmin, ymin, res)
        if np.isfinite(rough).any():
            filled_rough = fill_grid(rough)
            need = valid & ~np.isfinite(rough)
            rough[need] = filled_rough[need]
    elif not use_roughness:
        rough[valid] = 0.0

    # STEP HEIGHT
    lmax = maximum_filter(filled, size=3)
    lmin = minimum_filter(filled, size=3)
    st = lmax - lmin
    step[valid] = st[valid]

    return slope, rough, step, valid


def build_obstacle_grid(obstacle_pts, shape, xmin, ymin, res, dilation_radius):
    grid = np.zeros(shape, bool)
    if len(obstacle_pts) == 0:
        return grid
    ix = np.floor((obstacle_pts[:, 0] - xmin) / res).astype(np.int64)
    iy = np.floor((obstacle_pts[:, 1] - ymin) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    grid[iy[ok], ix[ok]] = True
    r = max(1, int(np.ceil(dilation_radius / res)))
    if r > 1:
        grid = binary_dilation(grid, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return grid


def compute_traversability(slope, rough, step, obstacle_grid, valid,
                           max_slope, max_rough, max_step,
                           w_slope, w_rough, w_step, non_trav_threshold):
    T = np.full(slope.shape, np.nan, np.float32)
    if not valid.any():
        return T
    slope_cost = np.clip(slope / max_slope, 0.0, 1.0)
    rough_cost = np.clip(rough / max_rough, 0.0, 1.0)
    step_cost = np.clip(step / max_step, 0.0, 1.0)
    cost = w_slope * slope_cost + w_rough * rough_cost + w_step * step_cost
    T[valid] = np.clip(1.0 - cost[valid], 0.0, 1.0)

    impossible = (slope > max_slope) | (rough > max_rough) | (step > max_step)
    T[valid & impossible] = 0.0
    T[obstacle_grid & valid] = 0.0
    T[valid & (T <= non_trav_threshold)] = 0.0
    return T


# ============================================================
# NODE
# ============================================================

class LivoxTraversabilityNode:
    def __init__(self):
        gp = rospy.get_param
        # topics / frames
        self.input_topic = gp("~input_topic", "/livox/lidar")
        self.target_frame = gp("~target_frame", "")
        self.process_rate = float(gp("~process_rate", 10.0))
        # cloud
        self.voxel_size = float(gp("~voxel_size", 0.10))
        self.max_range = float(gp("~max_range", 250.0))
        self.min_z = float(gp("~min_z", -2.0))
        self.max_z = float(gp("~max_z", 3.0))
        # ground
        self.ground_res = float(gp("~ground_grid_resolution", 0.12))
        self.ground_windows = list(gp("~ground_window_sizes", [3, 5, 7]))
        self.ground_thresholds = list(gp("~ground_filter_thresholds", [0.20, 0.35, 0.50]))
        self.ground_tolerance = float(gp("~ground_tolerance", 0.05))
        self.obstacle_height = float(gp("~obstacle_height_threshold", 0.30))
        self.max_ref_dist = float(gp("~max_ground_reference_distance", 2.0))
        # terrain
        self.terrain_res = float(gp("~terrain_resolution", 0.5))
        # roughness
        self.normal_k = int(gp("~normal_k", 20))
        self.rough_scales = list(gp("~roughness_scales", [10, 20, 40]))
        self.max_roughness_points = int(gp("~max_roughness_points", 15000))
        self.use_roughness = bool(gp("~use_roughness", True))
        # traversability
        self.max_slope = float(gp("~max_slope", 30.0))
        self.max_roughness = float(gp("~max_roughness", 0.05))
        self.max_step = float(gp("~max_step", 0.30))
        self.non_trav_threshold = float(gp("~non_traversable_threshold", 0.60))
        self.w_slope = float(gp("~slope_weight", 0.45))
        self.w_rough = float(gp("~roughness_weight", 0.30))
        self.w_step = float(gp("~step_weight", 0.25))
        self.obstacle_dilation = float(gp("~obstacle_dilation_radius", 0.50))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.lock = threading.Lock()
        self.latest = None

        self.sub = rospy.Subscriber(
            self.input_topic, PointCloud2, self.cloud_cb,
            queue_size=1, buff_size=2 ** 24)
        self.pub_trav = rospy.Publisher(
            "/traversability/traversable_cloud", PointCloud2, queue_size=1)
        self.pub_col = rospy.Publisher(
            "/traversability/cloud", PointCloud2, queue_size=1)
        self.pub_obs = rospy.Publisher(
            "/traversability/obstacles", PointCloud2, queue_size=1)

        period = rospy.Duration(1.0 / max(self.process_rate, 0.1))
        self.timer = rospy.Timer(period, self.process_cb)

        rospy.loginfo("=" * 60)
        rospy.loginfo("Livox traversability node started")
        rospy.loginfo("  input  : %s", self.input_topic)
        rospy.loginfo("  outputs: /traversability/traversable_cloud, "
                      "/traversability/cloud, /traversability/obstacles")
        rospy.loginfo("  target_frame: '%s'", self.target_frame)
        rospy.loginfo("=" * 60)

    # --------------------------------------------------------
    def cloud_cb(self, msg):
        with self.lock:
            self.latest = msg        # keep only the newest frame

    # --------------------------------------------------------
    def process_cb(self, _event):
        with self.lock:
            msg = self.latest
            self.latest = None
        if msg is None:
            return
        t0 = rospy.get_time()
        try:
            self.process(msg)
        except Exception as e:
            rospy.logerr_throttle(2.0, "Traversability failed: %s" % e)
        rospy.loginfo_throttle(5.0, "frame processed in %.1f ms",
                               1000.0 * (rospy.get_time() - t0))

    # --------------------------------------------------------
    def transform_points(self, pts, src_frame):
        if not self.target_frame or src_frame == self.target_frame:
            return pts, self.target_frame or src_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, src_frame, rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "TF %s -> %s failed: %s",
                                   src_frame, self.target_frame, e)
            return pts, src_frame
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        out = pts.astype(np.float64) @ R.T + np.array([t.x, t.y, t.z])
        return out.astype(np.float32), self.target_frame

    # --------------------------------------------------------
    def publish_empty(self, frame, stamp):
        empty = np.zeros((0, 3), np.float32)
        col = np.zeros((0, 3), np.uint8)
        self.pub_trav.publish(make_xyzrgb_cloud(empty, col, frame, stamp))
        self.pub_col.publish(make_xyzrgb_cloud(empty, col, frame, stamp))
        self.pub_obs.publish(make_xyzrgb_cloud(empty, col, frame, stamp))

    # --------------------------------------------------------
    def process(self, msg):
        pts = cloud_to_xyz(msg)
        if len(pts) < 10:
            return

        stamp = msg.header.stamp
        pts, frame = self.transform_points(pts, msg.header.frame_id)

        # ---- crop ----
        rng = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[rng <= self.max_range]
        pts = pts[(pts[:, 2] >= self.min_z) & (pts[:, 2] <= self.max_z)]
        if len(pts) < 10:
            self.publish_empty(frame, stamp)
            return

        pts = voxel_downsample(pts, self.voxel_size)
        if len(pts) < 10:
            self.publish_empty(frame, stamp)
            return

        # ---- [1] adaptive ground surface ----
        lower, observed, gxmin, gymin, _, _ = build_lower_grid(pts, self.ground_res)
        surface = estimate_ground_surface(
            lower, observed, self.ground_windows, self.ground_thresholds)

        # ---- [2] ground / obstacle classification ----
        ground, obstacle = classify_points(
            pts, surface, observed, gxmin, gymin, self.ground_res,
            self.ground_tolerance, self.obstacle_height, self.max_ref_dist)

        if len(ground) == 0:
            rospy.logwarn_throttle(5.0, "No ground points in frame.")
            self.publish_empty(frame, stamp)
            return

        # ---- [3] terrain map ----
        txmin, tymin, tnx, tny = grid_extent(pts, self.terrain_res)
        terrain = build_terrain(ground, txmin, tymin, tnx, tny, self.terrain_res)

        # ---- [4] features ----
        slope, rough, step, valid = compute_features(
            terrain, ground, txmin, tymin, self.terrain_res,
            self.use_roughness, self.normal_k, self.rough_scales,
            self.max_roughness_points)

        # ---- [5] obstacle grid ----
        obstacle_grid = build_obstacle_grid(
            obstacle, terrain.shape, txmin, tymin,
            self.terrain_res, self.obstacle_dilation)

        # ---- [6] traversability ----
        T = compute_traversability(
            slope, rough, step, obstacle_grid, valid,
            self.max_slope, self.max_roughness, self.max_step,
            self.w_slope, self.w_rough, self.w_step, self.non_trav_threshold)

        # ---- [7] per-point colors -> publish ----
        ix = np.clip(np.floor((ground[:, 0] - txmin) / self.terrain_res).astype(np.int64),
                     0, tnx - 1)
        iy = np.clip(np.floor((ground[:, 1] - tymin) / self.terrain_res).astype(np.int64),
                     0, tny - 1)
        T_pts = T[iy, ix]

        ok = np.isfinite(T_pts)
        g_pts, g_T = ground[ok], T_pts[ok]

        colors = np.zeros((len(g_T), 3), np.uint8)   # red -> green
        if len(g_T):
            colors[:, 0] = (255.0 * (1.0 - g_T)).astype(np.uint8)
            colors[:, 1] = (255.0 * g_T).astype(np.uint8)

        obs_pts = obstacle[:, :3] if len(obstacle) else np.zeros((0, 3), np.float32)
        obs_col = np.zeros((len(obs_pts), 3), np.uint8)
        obs_col[:, 0] = 255

        self.pub_col.publish(make_xyzrgb_cloud(
            np.vstack([g_pts, obs_pts]), np.vstack([colors, obs_col]), frame, stamp))

        trav_mask = g_T > self.non_trav_threshold
        trav_pts = g_pts[trav_mask]
        trav_col = np.zeros((len(trav_pts), 3), np.uint8)
        trav_col[:, 1] = 255
        self.pub_trav.publish(make_xyzrgb_cloud(trav_pts, trav_col, frame, stamp))

        self.pub_obs.publish(make_xyzrgb_cloud(obs_pts, obs_col, frame, stamp))

        rospy.loginfo_throttle(5.0,
            "pts=%d ground=%d obstacle=%d traversable=%d",
            len(pts), len(ground), len(obstacle), len(trav_pts))


# ============================================================
# MAIN
# ============================================================

def main():
    rospy.init_node("livox_traversability")
    LivoxTraversabilityNode()
    rospy.spin()


if __name__ == "__main__":
    main()
