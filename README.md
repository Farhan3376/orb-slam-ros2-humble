# ORB-SLAM Python (ROS 2 Humble)

A CPU-optimized, Python-based Monocular Visual SLAM system running as a ROS 2 Humble node.

## Features

- **Pure Python** implementation using OpenCV and NumPy
- **ORB features** for robust feature extraction and matching
- **Two-view initialization** using Essential matrix decomposition
- **Frame-to-frame tracking** via PnP with RANSAC
- **Keyframe-based mapping** with triangulation
- **Lightweight loop closure** using descriptor similarity
- **Real-time visualization** in RViz

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- OpenCV 4.x
- NumPy
- SciPy

## Installation

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws/src

# Clone or copy the package
cp -r /path/to/orb_slam_py .

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select orb_slam_py
source install/setup.bash
```

## Usage

### Running with a camera

```bash
# Terminal 1: Start camera driver (example with usb_cam)
ros2 run usb_cam usb_cam_node_exe --ros-args \
    -p camera_info_url:=file:///path/to/calibration.yaml

# Terminal 2: Start ORB-SLAM
ros2 launch orb_slam_py orb_slam.launch.py

# Terminal 3: Visualize in RViz
rviz2
```

### Running with a ROS 2 bag

```bash
# Terminal 1: Start ORB-SLAM
ros2 launch orb_slam_py orb_slam.launch.py \
    image_topic:=/camera/image_raw \
    camera_info_topic:=/camera/camera_info

# Terminal 2: Play bag
ros2 bag play /path/to/bag
```

### Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/image_raw` | Input image topic |
| `camera_info_topic` | `/camera/camera_info` | Camera info topic |
| `num_features` | `1000` | Number of ORB features |
| `scale_factor` | `1.2` | ORB pyramid scale factor |
| `num_levels` | `8` | ORB pyramid levels |
| `min_parallax` | `1.0` | Min parallax (degrees) for init |
| `min_init_matches` | `100` | Min matches for initialization |
| `min_track_matches` | `30` | Min matches for tracking |
| `log_trajectory` | `true` | Log trajectory to file |
| `trajectory_file` | `trajectory.txt` | Trajectory output path |
| `map_frame` | `map` | Map frame ID |
| `camera_frame` | `camera` | Camera frame ID |

## Topics

### Subscribed
- `/camera/image_raw` (sensor_msgs/Image) - Input camera image
- `/camera/camera_info` (sensor_msgs/CameraInfo) - Camera intrinsics

### Published
- `/orb_slam/pose` (geometry_msgs/PoseStamped) - Current camera pose
- `/orb_slam/path` (nav_msgs/Path) - Camera trajectory
- `/orb_slam/map_points` (sensor_msgs/PointCloud2) - Sparse 3D map

### TF
- `map` → `camera` transform

## RViz Configuration

Add the following displays in RViz2:

1. **PoseStamped**: Topic `/orb_slam/pose`
2. **Path**: Topic `/orb_slam/path`
3. **PointCloud2**: Topic `/orb_slam/map_points`
4. **TF**: Enable to see coordinate frames

Set **Fixed Frame** to `map`.

## Output

### Trajectory File (TUM format)

The trajectory is logged in TUM format:
```
timestamp tx ty tz qx qy qz qw
```

This format is compatible with TUM benchmark evaluation tools.

## Performance

Target performance on 8GB RAM:
- **Frame rate**: ≥10 FPS
- **Features**: 1000 ORB features per frame
- **Memory**: <2GB typical usage

## Architecture

```
orb_slam_py/
├── node.py          # Main ROS 2 node
├── tracking.py      # ORB extraction, matching, pose estimation
├── mapping.py       # Keyframe and map point management
├── geometry.py      # SE(3), projection, triangulation
├── visualization.py # RViz publishers
└── utils.py         # Data structures and helpers
```

## Limitations

- **Monocular only**: No stereo or RGB-D support
- **No bundle adjustment**: Simplified optimization for CPU performance
- **Simple loop closure**: Uses descriptor similarity instead of full BoW
- **Scale ambiguity**: Monocular SLAM cannot recover absolute scale

## License

MIT License
