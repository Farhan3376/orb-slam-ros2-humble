"""
Launch file for Stereo ORB-SLAM with KITTI dataset.

Uses both left and right cameras for depth estimation.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Stereo ORB-SLAM with KITTI."""
    
    # Get package share directory
    pkg_share = get_package_share_directory('orb_slam_py')
    rviz_config = os.path.join(pkg_share, 'config', 'orb_slam.rviz')
    
    # Launch arguments
    launch_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    baseline_arg = DeclareLaunchArgument(
        'baseline',
        default_value='0.54',
        description='Stereo baseline in meters'
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
    
    # Stereo ORB-SLAM node
    stereo_slam_node = Node(
        package='orb_slam_py',
        executable='stereo_orb_slam_node',
        name='stereo_orb_slam_py',
        output='screen',
        parameters=[{
            'left_image_topic': '/kitti/camera/left/image_raw',
            'right_image_topic': '/kitti/camera/right/image_raw',
            'camera_info_topic': '/kitti/camera/camera_info',
            'baseline': LaunchConfiguration('baseline'),
            'num_features': 1000,
            'min_init_matches': 50,
            'min_track_matches': 30,
            'log_trajectory': True,
            'trajectory_file': 'Outputstereo_trajectory.txt',
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
        launch_rviz_arg,
        baseline_arg,
        camera_info_node,
        stereo_slam_node,
        rviz_node,
    ])
