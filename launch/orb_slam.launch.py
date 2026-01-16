"""
Launch file for ORB-SLAM Python node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for ORB-SLAM node."""
    
    # Declare launch arguments
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Camera info topic'
    )
    
    num_features_arg = DeclareLaunchArgument(
        'num_features',
        default_value='1000',
        description='Number of ORB features to extract'
    )
    
    scale_factor_arg = DeclareLaunchArgument(
        'scale_factor',
        default_value='1.2',
        description='ORB scale factor between pyramid levels'
    )
    
    num_levels_arg = DeclareLaunchArgument(
        'num_levels',
        default_value='8',
        description='Number of pyramid levels'
    )
    
    min_parallax_arg = DeclareLaunchArgument(
        'min_parallax',
        default_value='1.0',
        description='Minimum parallax in degrees for initialization'
    )
    
    min_init_matches_arg = DeclareLaunchArgument(
        'min_init_matches',
        default_value='100',
        description='Minimum matches required for initialization'
    )
    
    min_track_matches_arg = DeclareLaunchArgument(
        'min_track_matches',
        default_value='30',
        description='Minimum matches for successful tracking'
    )
    
    log_trajectory_arg = DeclareLaunchArgument(
        'log_trajectory',
        default_value='true',
        description='Whether to log trajectory to file'
    )
    
    trajectory_file_arg = DeclareLaunchArgument(
        'trajectory_file',
        default_value='trajectory.txt',
        description='Trajectory log file path'
    )
    
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='Map frame ID'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera',
        description='Camera frame ID'
    )
    
    # ORB-SLAM node
    orb_slam_node = Node(
        package='orb_slam_py',
        executable='orb_slam_node',
        name='orb_slam_py',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'num_features': LaunchConfiguration('num_features'),
            'scale_factor': LaunchConfiguration('scale_factor'),
            'num_levels': LaunchConfiguration('num_levels'),
            'min_parallax': LaunchConfiguration('min_parallax'),
            'min_init_matches': LaunchConfiguration('min_init_matches'),
            'min_track_matches': LaunchConfiguration('min_track_matches'),
            'log_trajectory': LaunchConfiguration('log_trajectory'),
            'trajectory_file': LaunchConfiguration('trajectory_file'),
            'map_frame': LaunchConfiguration('map_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),
        }]
    )
    
    return LaunchDescription([
        image_topic_arg,
        camera_info_topic_arg,
        num_features_arg,
        scale_factor_arg,
        num_levels_arg,
        min_parallax_arg,
        min_init_matches_arg,
        min_track_matches_arg,
        log_trajectory_arg,
        trajectory_file_arg,
        map_frame_arg,
        camera_frame_arg,
        orb_slam_node,
    ])
