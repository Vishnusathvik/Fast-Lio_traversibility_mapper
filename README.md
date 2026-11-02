# FAST-LIO2 separate traversability mapper (ROS1 Noetic)

This is a separate ROS1 package. It does not patch, rebuild, or change FAST-LIO2 source. The original FAST-LIO2 PCD save behavior and filenames remain untouched; this package only writes `trav.pcd`.

## Inputs

- `/cloud_registered` (`sensor_msgs/PointCloud2`), expected in `camera_init`
- `/Odometry` (`nav_msgs/Odometry`), pose of the body in `camera_init`

The cloud and odometry are approximately synchronized. Since `/cloud_registered` is already in world coordinates, it is **not** transformed into world coordinates again. The pose is used to express the scan in the body-local frame for the per-scan classifier, then classified points are transformed back to `camera_init` for persistent accumulation.

## Outputs

The mapper now includes a first-pass temporal obstacle filter, vehicle-footprint inflation, terrain-feature outputs, and RViz cube markers. These are research/engineering features and require validation on recorded data before vehicle control.

- `/traversability/registered_colored`: current classified scan in world frame, packed `rgb`
- `/traversability/ground_world`: current ground points, score mapped red-to-green
- `/traversability/obstacles_world`: current obstacle points, red
- `/traversability/save_map` (`std_srvs/Trigger`): write accumulated `trav.pcd`
- `~save_path`, default `~/.ros/trav.pcd`; also saved on orderly shutdown

`trav.pcd` has PCD fields `x y z score rgb`. `score` is the accumulated traversability score from 0 to 1; RGB is red for low and green for high. It contains one point per persistent global XY grid cell. Only obstacles seen in multiple scans are added to the persistent obstacle map; the live obstacle topic still shows current scan detections.

The save service also writes `trav_grid.npz` with `points_xyz`, `traversability`, `colors_rgb`, `cell_indices`, `observation_count`, `terrain_points_xyz`, `elevation`, `roughness`, `slope`, `step`, `ground_class`, `terrain_traversability`, and metadata including grid resolution, frame, and vehicle parameters. Separate PCD layers are written beside the main file: `trav_elevation.pcd`, `trav_roughness.pcd`, `trav_slope.pcd`, `trav_step.pcd`, `trav_traversability.pcd`, and `trav_ground_class.pcd`. Each layer PCD uses `x y z value` fields.

Live voxel markers are published on `/traversability/voxels` as `visualization_msgs/MarkerArray` (`CUBE_LIST`). Feature point clouds are published on `/traversability/elevation`, `/traversability/roughness`, `/traversability/slope`, `/traversability/step`, and `/traversability/ground_class`.

### Temporal filtering and vehicle parameters

- `dynamic_min_hits`: number of distinct scans needed before an obstacle enters the persistent map (default 4).
- `temporal_window_frames`: a track is reset if it has not been observed within this number of frames (default 30).
- `dynamic_max_age_frames`: stale tracks are removed after this age (default 20; pruning uses the larger of the two age settings).
- `vehicle_length`, `vehicle_width`, `vehicle_wheelbase`, `vehicle_ground_clearance`, `vehicle_safety_margin`: configurable vehicle metadata. Length, width, and margin are used for conservative obstacle inflation. Wheelbase and ground clearance are stored in the NPZ for downstream footprint/clearance checks; the current node does not yet model suspension, steering angle, wheel tracks, or a full swept-volume vehicle model.

## Install

Copy this folder into your ROS1 catkin workspace `src/`, then:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Runtime dependencies:

```bash
sudo apt install ros-noetic-message-filters ros-noetic-tf2-ros ros-noetic-visualization-msgs python3-numpy python3-scipy
```

The reference `traversability_real.py` is included unchanged under `src/fastlio_traversability_mapper/` and imported as a Python package module. The executable is named `traversability_mapper_node.py` to avoid colliding with the ROS package name.

## Run live Horizon data

Start the Livox Horizon driver and FAST-LIO2 as usual, confirm the actual topics and frames, then run:

```bash
roslaunch fastlio_traversability_mapper fastlio_traversability.launch
```

The selected sensor is Livox **Horizon**; this node subscribes to FAST-LIO2's registered output, not `/livox/lidar`, so the LiDAR model does not change the registered-cloud topic choice.

## Run with a bag

Run FAST-LIO2 against the bag first, with simulated time if the bag uses it:

```bash
rosparam set use_sim_time true
rosbag play --clock your_recording.bag
```

Start the mapper alongside FAST-LIO2. Ensure the bag contains the raw Horizon topics needed by FAST-LIO2 and that FAST-LIO2 republishes `/cloud_registered` and `/Odometry`. The mapper can also be launched before playback. For a new bag replay, restart the mapper to start with an empty persistent map.

## Save

Save without stopping the node:

```bash
rosservice call /traversability/save_map
```

The node also attempts to save on orderly shutdown. It does not touch `scans.pcd` and does not alter FAST-LIO2's existing PCD save settings.

## Verify before a long run

```bash
rostopic echo -n 1 /cloud_registered/header
rostopic echo -n 1 /Odometry/header
rostopic info /cloud_registered
```

The cloud header frame must be `camera_init` (or the launch parameter must match the actual registered frame). Odometry pose must represent the body pose in that same world frame. If your FAST-LIO2 fork publishes odometry on a different topic, override `odom_topic`.

## Important algorithm caveats

This implementation adapts the uploaded algorithm's per-scan ground surface, slope, roughness, step and obstacle logic, then accumulates the score in a global grid. It is a practical integration baseline, not a validated safety-grade traversability estimator. In particular, the source algorithm's ground classifier assumes local-frame height limits and a gravity-aligned body frame; check the Horizon extrinsic/body axes and verify score maps in RViz before using it for vehicle decisions. The saved score is a per-cell mean of observations, not a probabilistic confidence estimate. A fresh process is needed for a clean new map.

## Separate terrain exports and Gazebo mesh

When `/traversability/save_map` is called (or the node shuts down cleanly), the mapper writes each terrain layer as **both PCD and NPZ** beside the main traversability map:

- `trav_elevation.pcd` + `trav_elevation.npz`
- `trav_roughness.pcd` + `trav_roughness.npz`
- `trav_slope.pcd` + `trav_slope.npz`
- `trav_step.pcd` + `trav_step.npz`
- `trav_traversability.pcd` + `trav_traversability.npz`
- `trav_ground_class.pcd` + `trav_ground_class.npz`
- `trav_layers.npz` contains all numeric feature arrays and corresponding `*_colors_rgb` arrays in one bundle.

Each layer PCD has `x y z value rgb` fields, so it preserves the numeric feature and packed RGB for PCL/Open3D visualization. Each layer NPZ includes `points_xyz`, `values`, `colors_rgb`, `cell_indices`, `layer_name`, `resolution`, `frame_id`, units and vehicle configuration. Colors are display aids; use the numeric `value`/`values` arrays for analysis. Color ramps: traversability red-to-green (low-to-high), roughness/slope/step green-to-red (low-to-high), elevation height ramp, and ground class categorical colors.

The default Gazebo export creates `~/sri/fastlio_terrain/` (relative to the directory containing `save_path`) with `terrain.obj`, `model.sdf`, `model.config`, and `terrain_info.txt`. The OBJ is triangulated from accumulated terrain elevation points and is used for both visual and collision geometry. Long triangles are rejected using `mesh_max_triangle_edge` to reduce bridging across gaps. This is a static mesh model, not a heightmap, so it preserves non-square map dimensions and uneven terrain without stretching the map into a square.

To make the model discoverable by Gazebo Classic, add the directory containing `fastlio_terrain/` to `GAZEBO_MODEL_PATH`, for example:

```bash
export GAZEBO_MODEL_PATH=$HOME/sri:$GAZEBO_MODEL_PATH
```

Then include it in a world or spawn it with `model://fastlio_terrain`. The SDF model pose restores the center of the source map coordinates. Confirm that the map's `camera_init` frame aligns with the Gazebo world frame before using it for navigation. Mesh collision behavior and scale must be validated in Gazebo; generated terrain is not guaranteed to be watertight, and steep overhangs/vertical obstacles are not represented well by a single-valued ground elevation surface.

Optional parameters:

- `export_gazebo_model` (default `true`)
- `gazebo_model_name` (default `fastlio_terrain`; should match the model directory name)
- `mesh_max_triangle_edge` (default `0.75` m; smaller values preserve holes but can leave gaps in sparse areas)
