"""
Launch file for ORB-SLAM with KITTI dataset.

Publishes default KITTI camera intrinsics since the bag doesn't include CameraInfo.
Also launches RViz for visualization.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for ORB-SLAM with KITTI."""
    
    # Get package share directory
    pkg_share = get_package_share_directory('orb_slam_py')
    rviz_config = os.path.join(pkg_share, 'config', 'orb_slam.rviz')
    
    # Declare launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/kitti/camera/left/image_raw',
        description='Input image topic for KITTI'
    )
    
    launch_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    # Camera info publisher (KITTI default calibration)
    camera_info_node = Node(
        package='orb_slam_py',
        executable='kitti_camera_info',
        name='kitti_camera_info_publisher',
        output='screen',
        parameters=[{
            'fx': 718.856,
            'fy': 718.856,
            'cx': 607.1928,
            'cy': 185.2157,
            'width': 1242,
            'height': 375,
        }]
    )
    
    # ORB-SLAM node
    orb_slam_node = Node(
        package='orb_slam_py',
        executable='orb_slam_node',
        name='orb_slam_py',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': '/kitti/camera/camera_info',
            'num_features': 1000,
            'min_parallax': 1.0,
            'min_init_matches': 100,
            'min_track_matches': 30,
            'log_trajectory': True,
            'trajectory_file': 'kitti_trajectory.txt',
            'map_frame': 'map',
            'camera_frame': 'camera',
        }]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )
    
    return LaunchDescription([
        image_topic_arg,
        launch_rviz_arg,
        camera_info_node,
        orb_slam_node,
        rviz_node,
    ])

