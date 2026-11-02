#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate persistent traversability mapper for FAST-LIO2 (ROS1 / Noetic).

Input cloud is FAST-LIO2's registered cloud in camera_init. The synchronized
Odometry pose is used only to express the cloud in the body-local frame for the
per-scan terrain classifier. Classified points are transformed back to
camera_init and accumulated into a persistent 2-D global grid.

This node does NOT modify FAST-LIO2 and does NOT save or overwrite scans.pcd.
"""
import os
import struct
import threading

import numpy as np
import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger, TriggerResponse

from fastlio_traversability_mapper import traversability_real as alg


class FastLIOTraversabilityMapper:
    def __init__(self):
        gp = rospy.get_param
        self.cloud_topic = gp('~cloud_topic', '/cloud_registered')
        self.odom_topic = gp('~odom_topic', '/Odometry')
        self.world_frame = gp('~world_frame', 'camera_init')
        self.map_resolution = float(gp('~map_resolution', 0.5))
        self.max_range = float(gp('~max_range', 60.0))
        self.min_z = float(gp('~min_z', -2.0))
        self.max_z = float(gp('~max_z', 300.0))
        self.voxel_size = float(gp('~voxel_size', 0.10))
        self.ground_res = float(gp('~ground_grid_resolution', 0.12))
        self.ground_windows = list(gp('~ground_window_sizes', [3, 5, 7]))
        self.ground_thresholds = list(gp('~ground_filter_thresholds', [0.20, 0.35, 0.50]))
        self.ground_tolerance = float(gp('~ground_tolerance', 0.05))
        self.obstacle_height = float(gp('~obstacle_height_threshold', 0.30))
        self.max_ref_dist = float(gp('~max_ground_reference_distance', 2.0))
        self.use_roughness = bool(gp('~use_roughness', True))
        self.terrain_res = float(gp('~terrain_resolution', 0.5))
        self.normal_k = int(gp('~normal_k', 20))
        self.rough_scales = list(gp('~roughness_scales', [10, 20, 40]))
        self.max_roughness_points = int(gp('~max_roughness_points', 15000))
        self.max_slope = float(gp('~max_slope', 30.0))
        self.max_roughness = float(gp('~max_roughness', 0.05))
        self.max_step = float(gp('~max_step', 0.30))
        self.non_trav_threshold = float(gp('~non_traversable_threshold', 0.60))
        self.w_slope = float(gp('~slope_weight', 0.45))
        self.w_rough = float(gp('~roughness_weight', 0.30))
        self.w_step = float(gp('~step_weight', 0.25))
        self.obstacle_dilation = float(gp('~obstacle_dilation_radius', 0.50))
        self.save_path = os.path.abspath(os.path.expanduser(
            gp('~save_path', '~/sri/trav.pcd')))
        self.npz_save_path = os.path.splitext(self.save_path)[0] + '_grid.npz'
        self.sync_slop = float(gp('~sync_slop', 0.10))
        self.max_map_cells = int(gp('~max_map_cells', 2000000))

        # key=(global ix, global iy), value=[sum_x,sum_y,sum_z,sum_score,count]
        self.cells = {}
        self.lock = threading.RLock()
        self.pub_colored = rospy.Publisher('/traversability/registered_colored', PointCloud2, queue_size=1)
        self.pub_ground = rospy.Publisher('/traversability/ground_world', PointCloud2, queue_size=1)
        self.pub_obstacles = rospy.Publisher('/traversability/obstacles_world', PointCloud2, queue_size=1)

        cloud_sub = message_filters.Subscriber(self.cloud_topic, PointCloud2, queue_size=2, buff_size=2**24)
        odom_sub = message_filters.Subscriber(self.odom_topic, Odometry, queue_size=10)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [cloud_sub, odom_sub], queue_size=10, slop=self.sync_slop, allow_headerless=False)
        self.sync.registerCallback(self.callback)
        self.save_srv = rospy.Service('/traversability/save_map', Trigger, self.save_service)
        rospy.on_shutdown(self.save_on_shutdown)

        rospy.loginfo('FAST-LIO2 separate traversability mapper started')
        rospy.loginfo('cloud=%s odom=%s world_frame=%s', self.cloud_topic, self.odom_topic, self.world_frame)
        rospy.loginfo('trav PCD output=%s (FAST-LIO2 scans save behavior unchanged)', self.save_path)
        rospy.loginfo('trav NPZ output=%s', self.npz_save_path)

    @staticmethod
    def pose_matrix(odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        R = alg.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    @staticmethod
    def transform(pts, T):
        if len(pts) == 0:
            return pts.astype(np.float32)
        return (pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    def callback(self, cloud_msg, odom_msg):
        try:
            if cloud_msg.header.frame_id and cloud_msg.header.frame_id.lstrip('/') != self.world_frame.lstrip('/'):
                rospy.logerr_throttle(5.0, 'Expected cloud frame %s, received %s; skipping to avoid wrong map coordinates',
                                      self.world_frame, cloud_msg.header.frame_id)
                return
            world_pts = alg.cloud_to_xyz(cloud_msg)
            if len(world_pts) < 10:
                return
            T_world_body = self.pose_matrix(odom_msg)
            T_body_world = np.linalg.inv(T_world_body)
            # cloud_registered is already in camera_init; inverse pose puts it into body-local coordinates
            local = self.transform(world_pts, T_body_world)
            rng = np.linalg.norm(local[:, :2], axis=1)
            keep = (rng <= self.max_range) & (local[:, 2] >= self.min_z) & (local[:, 2] <= self.max_z)
            local = local[keep]
            if len(local) < 10:
                return
            local = alg.voxel_downsample(local, self.voxel_size)
            if len(local) < 10:
                return

            lower, observed, gxmin, gymin, _, _ = alg.build_lower_grid(local, self.ground_res)
            surface = alg.estimate_ground_surface(lower, observed, self.ground_windows, self.ground_thresholds)
            ground_local, obs_local = alg.classify_points(
                local, surface, observed, gxmin, gymin, self.ground_res,
                self.ground_tolerance, self.obstacle_height, self.max_ref_dist)
            if len(ground_local) == 0:
                rospy.logwarn_throttle(5.0, 'No ground points classified in this registered scan')
                return

            txmin, tymin, tnx, tny = alg.grid_extent(local, self.terrain_res)
            terrain = alg.build_terrain(ground_local, txmin, tymin, tnx, tny, self.terrain_res)
            slope, rough, step, valid = alg.compute_features(
                terrain, ground_local, txmin, tymin, self.terrain_res,
                self.use_roughness, self.normal_k, self.rough_scales, self.max_roughness_points)
            obstacle_grid = alg.build_obstacle_grid(
                obs_local, terrain.shape, txmin, tymin, self.terrain_res, self.obstacle_dilation)
            Tgrid = alg.compute_traversability(
                slope, rough, step, obstacle_grid, valid,
                self.max_slope, self.max_roughness, self.max_step,
                self.w_slope, self.w_rough, self.w_step, self.non_trav_threshold)

            ix = np.clip(np.floor((ground_local[:, 0] - txmin) / self.terrain_res).astype(np.int64), 0, tnx - 1)
            iy = np.clip(np.floor((ground_local[:, 1] - tymin) / self.terrain_res).astype(np.int64), 0, tny - 1)
            scores = Tgrid[iy, ix]
            good = np.isfinite(scores)
            ground_local = ground_local[good]
            scores = scores[good].astype(np.float32)
            ground_world = self.transform(ground_local, T_world_body)
            obs_world = self.transform(obs_local, T_world_body)

            # Per-scan RGB visualization: red=low traversability, green=high.
            gcolors = np.zeros((len(scores), 3), dtype=np.uint8)
            gcolors[:, 0] = np.clip(255.0 * (1.0 - scores), 0, 255).astype(np.uint8)
            gcolors[:, 1] = np.clip(255.0 * scores, 0, 255).astype(np.uint8)
            ocolors = np.zeros((len(obs_world), 3), dtype=np.uint8)
            ocolors[:, 0] = 255
            self.pub_colored.publish(alg.make_xyzrgb_cloud(
                np.vstack((ground_world, obs_world)), np.vstack((gcolors, ocolors)),
                self.world_frame, cloud_msg.header.stamp))
            self.pub_ground.publish(alg.make_xyzrgb_cloud(
                ground_world, gcolors, self.world_frame, cloud_msg.header.stamp))
            self.pub_obstacles.publish(alg.make_xyzrgb_cloud(
                obs_world, ocolors, self.world_frame, cloud_msg.header.stamp))

            self.accumulate(ground_world, scores)
            if len(obs_world):
                self.accumulate(obs_world, np.zeros(len(obs_world), dtype=np.float32))
            rospy.loginfo_throttle(5.0, 'traversability mapper: input=%d ground=%d obstacles=%d persistent_cells=%d',
                                   len(world_pts), len(ground_world), len(obs_world), len(self.cells))
        except Exception as exc:
            rospy.logerr_throttle(2.0, 'Traversability mapping failed: %s', str(exc))

    def accumulate(self, points, scores):
        if len(points) == 0:
            return
        keys = np.floor(points[:, :2] / self.map_resolution).astype(np.int64)
        with self.lock:
            for p, score, key in zip(points, scores, keys):
                k = (int(key[0]), int(key[1]))
                v = self.cells.get(k)
                if v is None:
                    if len(self.cells) >= self.max_map_cells:
                        rospy.logwarn_throttle(10.0, 'Persistent grid reached max_map_cells=%d; ignoring new cells', self.max_map_cells)
                        continue
                    self.cells[k] = [float(p[0]), float(p[1]), float(p[2]), float(score), 1]
                else:
                    # Running mean gives stable map points and score across repeated observations.
                    n = v[4]
                    v[0] += float(p[0]); v[1] += float(p[1]); v[2] += float(p[2]); v[3] += float(score); v[4] = n + 1

    @staticmethod
    def score_color(score):
        score = float(np.clip(score, 0.0, 1.0))
        return int(round(255 * (1.0 - score))), int(round(255 * score)), 0

    def snapshot(self):
        # Copy the cells under one lock so all exported arrays describe the
        # same map state, even while new scans are being accumulated.
        with self.lock:
            rows = list(self.cells.items())

        if not rows:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )

        pts = np.empty((len(rows), 3), dtype=np.float32)
        scores = np.empty(len(rows), dtype=np.float32)
        colors = np.empty((len(rows), 3), dtype=np.uint8)
        cell_indices = np.empty((len(rows), 2), dtype=np.int64)
        observation_count = np.empty(len(rows), dtype=np.int64)

        for i, (key, v) in enumerate(rows):
            n = max(1, int(v[4]))
            pts[i] = [v[0] / n, v[1] / n, v[2] / n]
            scores[i] = np.clip(v[3] / n, 0.0, 1.0)
            colors[i] = self.score_color(scores[i])
            cell_indices[i] = [int(key[0]), int(key[1])]
            observation_count[i] = n

        return pts, scores, colors, cell_indices, observation_count

    def write_pcd(self, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()
        pts, scores, colors, _, _ = snapshot
        if len(pts) == 0:
            return False, 'No accumulated cells yet; trav.pcd not written.'
        parent = os.path.dirname(self.save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # PCL-compatible binary PCD: rgb is packed RGB bits reinterpreted as float.
        header = ('# .PCD v0.7 - Point Cloud Data file format\n'
                  'VERSION 0.7\nFIELDS x y z score rgb\nSIZE 4 4 4 4 4\n'
                  'TYPE F F F F F\nCOUNT 1 1 1 1 1\n'
                  'WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n' % (len(pts), len(pts)))
        tmp = self.save_path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(header.encode('ascii'))
            for p, score, color in zip(pts, scores, colors):
                packed = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
                rgb_float = struct.unpack('<f', struct.pack('<I', packed))[0]
                f.write(struct.pack('<fffff', float(p[0]), float(p[1]), float(p[2]), float(score), rgb_float))
        os.replace(tmp, self.save_path)
        return True, 'Saved %d persistent grid cells to %s' % (len(pts), self.save_path)

    def write_npz(self, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()

        pts, scores, colors, cell_indices, observation_count = snapshot
        if len(pts) == 0:
            return False, 'No accumulated cells yet; NPZ not written.'

        parent = os.path.dirname(self.npz_save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Save atomically: a failed write will not replace the previous map.
        tmp_path = self.npz_save_path + '.tmp'
        try:
            with open(tmp_path, 'wb') as f:
                np.savez_compressed(
                    f,
                    points_xyz=pts.astype(np.float32),
                    traversability=scores.astype(np.float32),
                    colors_rgb=colors.astype(np.uint8),
                    cell_indices=cell_indices.astype(np.int64),
                    observation_count=observation_count.astype(np.int64),
                    resolution=np.float32(self.map_resolution),
                    origin_xy=np.zeros(2, dtype=np.float64),
                    frame_id=np.asarray(self.world_frame),
                )
            os.replace(tmp_path, self.npz_save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return True, 'Saved %d persistent grid cells to %s' % (
            len(pts), self.npz_save_path
        )

    def save_service(self, _req):
        # Use the same snapshot for both files, so they describe the same map.
        snapshot = self.snapshot()
        pcd_ok, pcd_message = self.write_pcd(snapshot)
        npz_ok, npz_message = self.write_npz(snapshot)
        success = pcd_ok and npz_ok
        return TriggerResponse(
            success=success,
            message=pcd_message + '; ' + npz_message
        )

    def save_on_shutdown(self):
        try:
            snapshot = self.snapshot()
            for save_fn in (self.write_pcd, self.write_npz):
                ok, message = save_fn(snapshot)
                if ok:
                    rospy.loginfo(message)
                else:
                    rospy.logwarn(message)
        except Exception as exc:
            rospy.logerr('Could not save traversability map files on shutdown: %s', str(exc))


def main():
    rospy.init_node('fastlio_traversability_mapper')
    FastLIOTraversabilityMapper()
    rospy.spin()


if __name__ == '__main__':
    main()
