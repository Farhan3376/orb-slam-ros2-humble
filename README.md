# ORB-SLAM Python (ROS 2 Humble)

A CPU-optimized, Python-based Visual SLAM system supporting **Monocular** and **Stereo** cameras.

## Features

- **Monocular SLAM** - Two-view initialization with Essential matrix
- **Stereo SLAM** - Instant initialization with direct depth from disparity
- **ORB Features** - Robust feature extraction and matching
- **Real-time Visualization** - RViz integration with trajectory and map points
- **PLY Export** - Save 3D map as point cloud
- **Dataset Converters** - Convert images/video to ROS 2 bags

## Installation

```bash
cd ~/ros2_ws/src
git clone https://github.com/Farhan3376/orb-slam-ros2-humble.git

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select orb_slam_py
source install/setup.bash
```

## KITTI Dataset

Download the KITTI raw dataset for testing:

1. Visit: https://www.cvlibs.net/datasets/kitti/raw_data.php
2. Download a synced+rectified sequence (e.g., `2011_09_26_drive_0001`)

**Camera calibration** for KITTI:
- fx, fy: 718.856
- cx: 607.1928
- cy: 185.2157
- Baseline: 0.54m

## Quick Start

### Stereo SLAM (KITTI)

```bash
# Terminal 1: Launch stereo SLAM
ros2 launch orb_slam_py kitti_stereo.launch.py

# Terminal 2: Play bag
ros2 bag play kitti_bag --rate 0.5
```

### Monocular SLAM

```bash
ros2 launch orb_slam_py kitti.launch.py
```

### Custom Dataset

```bash
ros2 launch orb_slam_py custom.launch.py \
    fx:=500 fy:=500 cx:=320 cy:=240 \
    width:=640 height:=480
```

## Dataset Converters

Convert your own data to ROS 2 bags:

```bash
# Images → Bag (Monocular)
python3 scripts/images_to_bag.py --images /path/to/images --output my_bag

# Images → Bag (Stereo)
python3 scripts/images_to_bag.py --left /path/left --right /path/right --output stereo_bag

# Video → Bag
python3 scripts/video_to_bag.py --video video.mp4 --output video_bag --fps 10
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/orb_slam/pose` | PoseStamped | Current camera pose |
| `/orb_slam/path` | Path | Camera trajectory |
| `/orb_slam/map_points` | PointCloud2 | 3D map points |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_features` | 1000 | ORB features per frame |
| `baseline` | 0.54 | Stereo baseline (meters) |
| `min_init_matches` | 100 | Matches for initialization |
| `log_trajectory` | true | Save trajectory to file |

## Output Files

After running SLAM:

- `Output/stereo_trajectory.txt` - TUM format trajectory
- `Output/stereo_map.ply` - 3D point cloud (open with MeshLab)

## Architecture

```
orb_slam_py/
├── node.py            # Monocular SLAM node
├── stereo_node.py     # Stereo SLAM node
├── stereo.py          # Stereo matching
├── tracking.py        # ORB + pose estimation
├── mapping.py         # Keyframe management
├── geometry.py        # SE3, projection
├── visualization.py   # RViz publishers
└── scripts/           # Dataset converters
```

## Stereo vs Monocular

| Feature | Monocular | Stereo |
|---------|-----------|--------|
| Initialization | Needs motion | Instant |
| Scale | Up to scale | Absolute |
| Depth | Triangulation | Disparity |

## License

MIT License
