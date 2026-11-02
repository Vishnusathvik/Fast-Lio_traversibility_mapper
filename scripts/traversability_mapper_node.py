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
from visualization_msgs.msg import Marker, MarkerArray
from scipy.ndimage import binary_dilation
from scipy.spatial import Delaunay, QhullError

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
        self.export_gazebo = bool(gp('~export_gazebo_model', True))
        self.mesh_max_edge = float(gp('~mesh_max_triangle_edge', 0.75))
        self.gazebo_model_name = str(gp('~gazebo_model_name', 'fastlio_terrain'))
        # Temporal static/dynamic separation. An obstacle is persisted only after
        # it is observed in this many distinct scans within the temporal window.
        self.temporal_window_frames = int(gp('~temporal_window_frames', 30))
        self.dynamic_min_hits = int(gp('~dynamic_min_hits', 4))
        self.dynamic_max_age_frames = int(gp('~dynamic_max_age_frames', 20))
        self.frame_count = 0
        self.obstacle_tracks = {}  # key -> [hits, last_frame, x, y, z]
        self.persistent_obstacle_keys = set()
        # Vehicle footprint parameters (metres). Conservative rectangular footprint.
        self.vehicle_length = float(gp('~vehicle_length', 4.5))
        self.vehicle_width = float(gp('~vehicle_width', 1.9))
        self.vehicle_wheelbase = float(gp('~vehicle_wheelbase', 2.7))
        self.vehicle_ground_clearance = float(gp('~vehicle_ground_clearance', 0.18))
        self.vehicle_margin = float(gp('~vehicle_safety_margin', 0.15))
        self.layers = {}  # global terrain-cell key -> [sum_z,slope,rough,step,score,ground_n,total_n]

        # key=(global ix, global iy), value=[sum_x,sum_y,sum_z,sum_score,count]
        self.cells = {}
        self.lock = threading.RLock()
        self.pub_colored = rospy.Publisher('/traversability/registered_colored', PointCloud2, queue_size=1)
        self.pub_ground = rospy.Publisher('/traversability/ground_world', PointCloud2, queue_size=1)
        self.pub_obstacles = rospy.Publisher('/traversability/obstacles_world', PointCloud2, queue_size=1)
        self.pub_voxels = rospy.Publisher('/traversability/voxels', MarkerArray, queue_size=1)
        self.pub_elevation = rospy.Publisher('/traversability/elevation', PointCloud2, queue_size=1)
        self.pub_roughness = rospy.Publisher('/traversability/roughness', PointCloud2, queue_size=1)
        self.pub_slope = rospy.Publisher('/traversability/slope', PointCloud2, queue_size=1)
        self.pub_step = rospy.Publisher('/traversability/step', PointCloud2, queue_size=1)
        self.pub_ground_class = rospy.Publisher('/traversability/ground_class', PointCloud2, queue_size=1)

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
        rospy.loginfo('vehicle footprint L=%.2f W=%.2f wheelbase=%.2f clearance=%.2f m',
                      self.vehicle_length, self.vehicle_width, self.vehicle_wheelbase, self.vehicle_ground_clearance)

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
            self.frame_count += 1
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
            # Inflate obstacles by the rectangular vehicle footprint plus margin.
            # The local grid is in body coordinates, so x maps to length and y to width.
            rx = max(1, int(np.ceil((0.5 * self.vehicle_length + self.vehicle_margin) / self.terrain_res)))
            ry = max(1, int(np.ceil((0.5 * self.vehicle_width + self.vehicle_margin) / self.terrain_res)))
            obstacle_grid = binary_dilation(
                obstacle_grid, structure=np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool))
            # Low obstacles may fit below the underbody; the configured clearance
            # is saved as metadata and used by downstream footprint/clearance checks.
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
            self.accumulate_features(ground_world, ground_local, scores, slope, rough, step,
                                    txmin, tymin, tnx, tny, T_world_body)
            stable_obs = self.temporal_obstacles(obs_world)
            if len(stable_obs):
                self.accumulate(stable_obs, np.zeros(len(stable_obs), dtype=np.float32))
            self.publish_voxels(ground_world, scores, cloud_msg.header.stamp)
            self.publish_feature_clouds(ground_world, ground_local, slope, rough, step,
                                        txmin, tymin, tnx, tny, T_world_body, cloud_msg.header.stamp)
            rospy.loginfo_throttle(5.0, 'traversability mapper: input=%d ground=%d obstacles=%d persistent_cells=%d',
                                   len(world_pts), len(ground_world), len(obs_world), len(self.cells))
        except Exception as exc:
            rospy.logerr_throttle(2.0, 'Traversability mapping failed: %s', str(exc))

    def temporal_obstacles(self, obs_world):
        """Return only temporally persistent obstacle representatives; transient
        obstacle observations remain visible in the live obstacle topic but do not
        permanently poison the static traversability map."""
        if len(obs_world) == 0:
            self._prune_obstacle_tracks()
            return np.zeros((0, 3), dtype=np.float32)
        keys = np.floor(obs_world[:, :2] / self.map_resolution).astype(np.int64)
        current = {}
        for p, key in zip(obs_world, keys):
            k = (int(key[0]), int(key[1]))
            if k not in current:
                current[k] = p
        stable = []
        for k, p in current.items():
            old = self.obstacle_tracks.get(k)
            if old is None or self.frame_count - old[1] > self.temporal_window_frames:
                track = [1, self.frame_count, float(p[0]), float(p[1]), float(p[2])]
            else:
                track = [old[0] + 1, self.frame_count, float(p[0]), float(p[1]), float(p[2])]
            self.obstacle_tracks[k] = track
            if track[0] >= self.dynamic_min_hits:
                stable.append([track[2], track[3], track[4]])
                self.persistent_obstacle_keys.add(k)
        self._prune_obstacle_tracks()
        return np.asarray(stable, dtype=np.float32).reshape((-1, 3))

    def _prune_obstacle_tracks(self):
        cutoff = max(self.temporal_window_frames, self.dynamic_max_age_frames)
        stale = [k for k, v in self.obstacle_tracks.items() if self.frame_count - v[1] > cutoff]
        for k in stale:
            del self.obstacle_tracks[k]
            # Expire stale obstacle-only cells so a moving object does not remain
            # forever in the accumulated traversability layer. A later ground
            # observation can repopulate the same grid cell.
            if k in self.persistent_obstacle_keys:
                with self.lock:
                    self.cells.pop(k, None)
                self.persistent_obstacle_keys.discard(k)

    def accumulate_features(self, world_pts, local_pts, scores, slope, rough, step,
                            xmin, ymin, nx, ny, T_world_body):
        if len(world_pts) == 0:
            return
        ix = np.clip(np.floor((local_pts[:, 0] - xmin) / self.terrain_res).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.floor((local_pts[:, 1] - ymin) / self.terrain_res).astype(np.int64), 0, ny - 1)
        with self.lock:
            for i, p in enumerate(world_pts):
                vals = [slope[iy[i], ix[i]], rough[iy[i], ix[i]], step[iy[i], ix[i]]]
                vals = [float(v) if np.isfinite(v) else np.nan for v in vals]
                key = tuple(np.floor(p[:2] / self.terrain_res).astype(np.int64).tolist())
                rec = self.layers.get(key)
                if rec is None:
                    if len(self.layers) >= self.max_map_cells:
                        continue
                    rec = [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]
                    self.layers[key] = rec
                rec[0] += float(p[2]); rec[4] += float(scores[i]); rec[5] += 1; rec[6] += 1
                for j, v in enumerate(vals, start=1):
                    if np.isfinite(v): rec[j] += v

    def publish_voxels(self, pts, scores, stamp):
        """Publish traversability as cube markers so RViz can display true voxels."""
        ma = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = stamp
        marker.ns = 'traversability_voxels'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.terrain_res
        marker.scale.y = self.terrain_res
        marker.scale.z = max(0.05, self.voxel_size)
        marker.color.a = 1.0
        from geometry_msgs.msg import Point
        for p, score in zip(pts, scores):
            q = Point(); q.x = float(p[0]); q.y = float(p[1]); q.z = float(p[2]) + marker.scale.z * 0.5
            marker.points.append(q)
            c = __import__('std_msgs.msg', fromlist=['ColorRGBA']).ColorRGBA()
            c.r = float(np.clip(1.0 - score, 0.0, 1.0)); c.g = float(np.clip(score, 0.0, 1.0)); c.b = 0.0; c.a = 1.0
            marker.colors.append(c)
        ma.markers.append(marker)
        self.pub_voxels.publish(ma)

    def publish_feature_clouds(self, world_pts, local_pts, slope, rough, step,
                               xmin, ymin, nx, ny, T_world_body, stamp):
        if len(world_pts) == 0: return
        ix = np.clip(np.floor((local_pts[:, 0] - xmin) / self.terrain_res).astype(np.int64), 0, nx - 1)
        iy = np.clip(np.floor((local_pts[:, 1] - ymin) / self.terrain_res).astype(np.int64), 0, ny - 1)
        for pub, grid, scale in ((self.pub_elevation, None, 1.0), (self.pub_roughness, rough, 0.10),
                                 (self.pub_slope, slope, self.max_slope), (self.pub_step, step, self.max_step)):
            values = world_pts[:, 2] if grid is None else grid[iy, ix]
            good = np.isfinite(values)
            if not np.any(good): continue
            v = np.clip(values[good] / max(scale, 1e-6), 0.0, 1.0)
            cols = np.zeros((len(v), 3), dtype=np.uint8)
            cols[:, 0] = (255 * v).astype(np.uint8); cols[:, 1] = (255 * (1.0 - v)).astype(np.uint8)
            pub.publish(alg.make_xyzrgb_cloud(world_pts[good], cols, self.world_frame, stamp))
        # Ground class: green=ground (published points), obstacles are on their own topic.
        cols = np.tile(np.array([[0, 200, 0]], dtype=np.uint8), (len(world_pts), 1))
        self.pub_ground_class.publish(alg.make_xyzrgb_cloud(world_pts, cols, self.world_frame, stamp))

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

    def layer_snapshot(self):
        with self.lock:
            rows = list(self.layers.items())
        if not rows:
            return {k: np.zeros((0,), dtype=np.float32) for k in
                    ('points_xyz','elevation','roughness','slope','step','traversability','ground_class','observation_count')}
        pts=[]; elev=[]; rough=[]; slope=[]; step=[]; trav=[]; counts=[]
        for (ix, iy), v in rows:
            n=max(1, int(v[5]))
            pts.append([(ix+0.5)*self.terrain_res, (iy+0.5)*self.terrain_res, v[0]/n])
            elev.append(v[0]/n); slope.append(v[1]/n); rough.append(v[2]/n); step.append(v[3]/n); trav.append(v[4]/n); counts.append(n)
        return {'points_xyz':np.asarray(pts,dtype=np.float32), 'elevation':np.asarray(elev,dtype=np.float32),
                'roughness':np.asarray(rough,dtype=np.float32), 'slope':np.asarray(slope,dtype=np.float32),
                'step':np.asarray(step,dtype=np.float32), 'traversability':np.asarray(trav,dtype=np.float32),
                'ground_class':np.ones(len(pts),dtype=np.uint8), 'observation_count':np.asarray(counts,dtype=np.int32)}

    def write_layer_pcd(self, name, points, values):
        path = os.path.splitext(self.save_path)[0] + '_' + name + '.pcd'
        if len(points) == 0: return False, 'empty ' + name
        parent=os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        header=('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z value\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n' % (len(points),len(points)))
        tmp=path+'.tmp'
        with open(tmp,'wb') as f:
            f.write(header.encode('ascii'))
            for p,v in zip(points,values): f.write(struct.pack('<ffff',float(p[0]),float(p[1]),float(p[2]),float(v)))
        os.replace(tmp,path)
        return True, path

    def write_npz(self, snapshot=None):
        if snapshot is None: snapshot=self.snapshot()
        pts,scores,colors,cell_indices,observation_count=snapshot
        if len(pts)==0: return False,'No accumulated cells yet; NPZ not written.'
        parent=os.path.dirname(self.npz_save_path)
        if parent: os.makedirs(parent,exist_ok=True)
        layers=self.layer_snapshot(); tmp_path=self.npz_save_path+'.tmp'
        try:
            with open(tmp_path,'wb') as f:
                np.savez_compressed(f,points_xyz=pts.astype(np.float32),traversability=scores.astype(np.float32),colors_rgb=colors.astype(np.uint8),
                    cell_indices=cell_indices.astype(np.int64),observation_count=observation_count.astype(np.int64),resolution=np.float32(self.map_resolution),
                    terrain_resolution=np.float32(self.terrain_res),origin_xy=np.zeros(2,dtype=np.float64),frame_id=np.asarray(self.world_frame),
                    vehicle_length=np.float32(self.vehicle_length),vehicle_width=np.float32(self.vehicle_width),vehicle_wheelbase=np.float32(self.vehicle_wheelbase),
                    vehicle_ground_clearance=np.float32(self.vehicle_ground_clearance),vehicle_safety_margin=np.float32(self.vehicle_margin),
                    terrain_points_xyz=layers.get('points_xyz'),elevation=layers.get('elevation'),roughness=layers.get('roughness'),slope=layers.get('slope'),
                    step=layers.get('step'),ground_class=layers.get('ground_class'),terrain_traversability=layers.get('traversability'),
                    terrain_observation_count=layers.get('observation_count'))
            os.replace(tmp_path,self.npz_save_path)
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        return self.export_all_layers()

    @staticmethod
    def colorize(values, layer_name):
        """Return deterministic RGB colors for a terrain feature layer."""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.tile(np.array([[128, 128, 128]], dtype=np.uint8), (len(v), 1))
        good = np.isfinite(v)
        if not np.any(good): return out
        x = v[good]
        if layer_name == 'traversability':
            r, g, b = 255*(1.0-x), 255*x, np.zeros_like(x)
        elif layer_name == 'ground_class':
            palette = {0:(128,128,128), 1:(40,200,40), 2:(240,40,40), 3:(40,100,240)}
            good_idx = np.flatnonzero(good)
            for idx, cls in zip(good_idx, x.astype(int)):
                out[idx] = palette.get(int(cls), (200,0,200))
            return out
        elif layer_name == 'elevation':
            lo, hi = np.percentile(x, [2,98]) if len(x)>1 else (x[0]-0.5,x[0]+0.5)
            t = np.clip((x-lo)/max(hi-lo,1e-6),0,1)
            r, g, b = 255*t, 255*(1-np.abs(2*t-1)), 255*(1-t)
        else:
            hi = {'roughness':0.10,'slope':30.0,'step':0.30}.get(layer_name, max(float(np.nanpercentile(x,95)),1e-6))
            t = np.clip(x/max(hi,1e-6),0,1)
            r, g, b = 255*t, 255*(1-t), np.zeros_like(t)
        out[good,0] = np.clip(r,0,255).astype(np.uint8)
        out[good,1] = np.clip(g,0,255).astype(np.uint8)
        out[good,2] = np.clip(b,0,255).astype(np.uint8)
        return out

    def write_colored_value_pcd(self, path, points, values, colors):
        """Write PCD with both numeric feature value and packed RGB color."""
        points=np.asarray(points,dtype=np.float32).reshape((-1,3)); values=np.asarray(values,dtype=np.float32).reshape(-1)
        colors=np.asarray(colors,dtype=np.uint8).reshape((-1,3))
        if len(points)==0: return False, 'empty '+os.path.basename(path)
        parent=os.path.dirname(path)
        if parent: os.makedirs(parent,exist_ok=True)
        header=('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z value rgb\n'
                'SIZE 4 4 4 4 4\nTYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\n'
                'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n'%(len(points),len(points)))
        tmp=path+'.tmp'
        with open(tmp,'wb') as f:
            f.write(header.encode('ascii'))
            for p,val,c in zip(points,values,colors):
                packed=(int(c[0])<<16)|(int(c[1])<<8)|int(c[2])
                rgb_float=struct.unpack('<f',struct.pack('<I',packed))[0]
                f.write(struct.pack('<fffff',float(p[0]),float(p[1]),float(p[2]),float(val),rgb_float))
        os.replace(tmp,path)
        return True,path

    def write_layer_npz(self,path,name,points,values,colors,cell_indices):
        parent=os.path.dirname(path)
        if parent: os.makedirs(parent,exist_ok=True)
        tmp=path+'.tmp'
        units='metres' if name in ('elevation','roughness','step') else ('degrees' if name=='slope' else 'unitless')
        with open(tmp,'wb') as f:
            np.savez_compressed(f,points_xyz=np.asarray(points,dtype=np.float32),values=np.asarray(values,dtype=np.float32),
                colors_rgb=np.asarray(colors,dtype=np.uint8),cell_indices=np.asarray(cell_indices,dtype=np.int64),
                layer_name=np.asarray(name),resolution=np.float32(self.terrain_res),frame_id=np.asarray(self.world_frame),units=np.asarray(units),
                vehicle_length=np.float32(self.vehicle_length),vehicle_width=np.float32(self.vehicle_width),
                vehicle_wheelbase=np.float32(self.vehicle_wheelbase),vehicle_ground_clearance=np.float32(self.vehicle_ground_clearance),
                vehicle_safety_margin=np.float32(self.vehicle_margin))
        os.replace(tmp,path)

    def export_gazebo_mesh(self, points):
        """Export terrain elevation samples as a triangulated Gazebo static model."""
        if not self.export_gazebo: return True,'Gazebo mesh export disabled'
        points=np.asarray(points,dtype=np.float64).reshape((-1,3))
        if len(points)<4: return False,'Not enough terrain cells to triangulate Gazebo mesh'
        _, unique_idx=np.unique(np.round(points[:,:2],4),axis=0,return_index=True)
        points=points[np.sort(unique_idx)]
        if len(points)<4: return False,'Not enough unique XY terrain cells for mesh'
        try: tri=Delaunay(points[:,:2])
        except QhullError as exc: return False,'Terrain triangulation failed: '+str(exc)
        faces=tri.simplices
        edges=np.stack([points[faces[:,0],:2]-points[faces[:,1],:2],points[faces[:,1],:2]-points[faces[:,2],:2],points[faces[:,2],:2]-points[faces[:,0],:2]],axis=1)
        faces=faces[np.max(np.linalg.norm(edges,axis=2),axis=1)<=self.mesh_max_edge]
        if len(faces)==0: return False,'No triangles remain; increase mesh_max_triangle_edge or collect denser terrain'
        center=(np.min(points,axis=0)+np.max(points,axis=0))*0.5
        local_pts=points-center
        model_dir=os.path.join(os.path.dirname(self.save_path) or '.', self.gazebo_model_name); os.makedirs(model_dir,exist_ok=True)
        with open(os.path.join(model_dir,'terrain.obj'),'w') as f:
            f.write('# FAST-LIO2 terrain mesh; model.sdf pose restores map coordinates\n')
            for p in local_pts: f.write('v %.6f %.6f %.6f\n'%(p[0],p[1],p[2]))
            for face in faces: f.write('f %d %d %d\n'%(face[0]+1,face[1]+1,face[2]+1))
        sdf="""<?xml version="1.0"?>
<sdf version="1.6"><model name="%s"><static>true</static><pose>%.6f %.6f %.6f 0 0 0</pose>
<link name="terrain"><collision name="terrain_collision"><geometry><mesh><uri>model://%s/terrain.obj</uri><scale>1 1 1</scale></mesh></geometry></collision>
<visual name="terrain_visual"><geometry><mesh><uri>model://%s/terrain.obj</uri><scale>1 1 1</scale></mesh></geometry>
<material><ambient>0.45 0.48 0.42 1</ambient><diffuse>0.45 0.48 0.42 1</diffuse></material></visual></link></model></sdf>
"""%(self.gazebo_model_name,center[0],center[1],center[2],self.gazebo_model_name,self.gazebo_model_name)
        with open(os.path.join(model_dir,'model.sdf'),'w') as f: f.write(sdf)
        config="""<?xml version="1.0"?><model><name>%s</name><version>1.0</version><sdf version="1.6">model.sdf</sdf><author><name>FAST-LIO2 Traversability Mapper</name><email>noreply@example.com</email></author><description>Terrain mesh generated from LiDAR elevation data.</description></model>\n"""%self.gazebo_model_name
        with open(os.path.join(model_dir,'model.config'),'w') as f: f.write(config)
        with open(os.path.join(model_dir,'terrain_info.txt'),'w') as f:
            f.write('source_frame=%s\nmap_pose_center_xyz=%.6f %.6f %.6f\nterrain_resolution_m=%.6f\nmax_triangle_edge_m=%.6f\nvertices=%d\ntriangles=%d\n'%(self.world_frame,center[0],center[1],center[2],self.terrain_res,self.mesh_max_edge,len(local_pts),len(faces)))
            f.write('Verify map/world alignment and collision behavior in Gazebo before use.\n')
        return True,'Gazebo model: %s (%d triangles)'%(model_dir,len(faces))

    def export_all_layers(self):
        layers=self.layer_snapshot(); points=np.asarray(layers.get('points_xyz',np.zeros((0,3))),dtype=np.float32)
        if len(points)==0: return False,'No terrain feature cells accumulated yet'
        base=os.path.splitext(self.save_path)[0]
        specs=[('elevation','elevation'),('roughness','roughness'),('slope','slope'),('step','step'),('traversability','traversability'),('ground_class','ground_class')]
        indices=np.floor(points[:,:2]/self.terrain_res).astype(np.int64); written=[]
        for name,key in specs:
            vals=np.asarray(layers.get(key,np.zeros(len(points))),dtype=np.float32).reshape(-1)
            if len(vals)!=len(points): vals=np.resize(vals,len(points))
            colors=self.colorize(vals,name)
            pcd=base+'_'+name+'.pcd'; npz=base+'_'+name+'.npz'
            self.write_colored_value_pcd(pcd,points,vals,colors); self.write_layer_npz(npz,name,points,vals,colors,indices)
            written.append('%s (.pcd + .npz)'%name)
        bundle=base+'_layers.npz'; tmp=bundle+'.tmp'; payload={'points_xyz':points,'terrain_resolution':np.float32(self.terrain_res),'frame_id':np.asarray(self.world_frame)}
        for name,key in specs:
            vals=np.asarray(layers.get(key,np.zeros(len(points))),dtype=np.float32).reshape(-1)
            if len(vals)!=len(points): vals=np.resize(vals,len(points))
            payload[name]=vals; payload[name+'_colors_rgb']=self.colorize(vals,name)
        with open(tmp,'wb') as f: np.savez_compressed(f,**payload)
        os.replace(tmp,bundle)
        ok,msg=self.export_gazebo_mesh(points)
        return True,'Exported '+', '.join(written)+'; combined='+bundle+'; '+msg

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
