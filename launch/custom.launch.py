"""
Launch file for custom datasets with ORB-SLAM.

Configure your camera parameters and topic names.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for custom dataset."""
    
    pkg_share = get_package_share_directory('orb_slam_py')
    rviz_config = os.path.join(pkg_share, 'config', 'orb_slam.rviz')
    
    # Mode: mono or stereo
    mode_arg = DeclareLaunchArgument(
        'mode', default_value='mono',
        description='SLAM mode: mono or stereo'
    )
    
    # Camera parameters - CUSTOMIZE THESE!
    fx_arg = DeclareLaunchArgument('fx', default_value='500.0', description='Focal length X')
    fy_arg = DeclareLaunchArgument('fy', default_value='500.0', description='Focal length Y')
    cx_arg = DeclareLaunchArgument('cx', default_value='320.0', description='Principal point X')
    cy_arg = DeclareLaunchArgument('cy', default_value='240.0', description='Principal point Y')
    width_arg = DeclareLaunchArgument('width', default_value='640', description='Image width')
    height_arg = DeclareLaunchArgument('height', default_value='480', description='Image height')
    baseline_arg = DeclareLaunchArgument('baseline', default_value='0.1', description='Stereo baseline (m)')
    
    # Topics - CUSTOMIZE THESE!
    image_topic_arg = DeclareLaunchArgument(
        'image_topic', default_value='/camera/image_raw',
        description='Monocular image topic'
    )
    left_topic_arg = DeclareLaunchArgument(
        'left_image_topic', default_value='/camera/left/image_raw',
        description='Left stereo image topic'
    )
    right_topic_arg = DeclareLaunchArgument(
        'right_image_topic', default_value='/camera/right/image_raw',
        description='Right stereo image topic'
    )
    
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    
    # Camera info publisher (custom calibration)
    camera_info_node = Node(
        package='orb_slam_py',
        executable='kitti_camera_info',
        name='custom_camera_info',
        output='screen',
        parameters=[{
            'fx': LaunchConfiguration('fx'),
            'fy': LaunchConfiguration('fy'),
            'cx': LaunchConfiguration('cx'),
            'cy': LaunchConfiguration('cy'),
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
        }],
        remappings=[
            ('/kitti/camera/camera_info', '/camera/camera_info'),
            ('/kitti/camera/left/camera_info', '/camera/left/camera_info'),
            ('/kitti/camera/right/camera_info', '/camera/right/camera_info'),
        ]
    )
    
    # Monocular SLAM node
    mono_slam_node = Node(
        package='orb_slam_py',
        executable='orb_slam_node',
        name='orb_slam_mono',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': '/camera/camera_info',
            'num_features': 1000,
            'min_init_matches': 100,
            'min_track_matches': 30,
            'log_trajectory': True,
            'trajectory_file': 'Output/mono_trajectory.txt',
        }],
        condition=IfCondition("True")  # Always run for now, add mode condition later
    )
    
    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )
    
    return LaunchDescription([
        mode_arg,
        fx_arg, fy_arg, cx_arg, cy_arg, width_arg, height_arg, baseline_arg,
        image_topic_arg, left_topic_arg, right_topic_arg,
        rviz_arg,
        camera_info_node,
        mono_slam_node,
        rviz_node,
    ])
