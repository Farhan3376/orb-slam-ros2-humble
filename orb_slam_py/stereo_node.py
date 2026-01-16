"""
stereo_node.py - Stereo ORB-SLAM ROS 2 node

Uses both left and right camera images for depth estimation
and improved localization.
"""

import numpy as np
import cv2
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

from .geometry import SE3, CameraIntrinsics
from .tracking import StereoTracker, TrackingState, ORBConfig
from .mapping import Map, LocalMapper, LoopDetector
from .visualization import SLAMVisualizer
from .utils import Frame, Keyframe, TUMLogger, Timer


class StereoORBSLAMNode(Node):
    """
    Stereo ORB-SLAM ROS 2 Node.
    
    Subscribes to synchronized left and right camera images,
    uses stereo matching for depth, and publishes SLAM results.
    """
    
    def __init__(self):
        super().__init__('stereo_orb_slam_py')
        
        # Declare parameters
        self.declare_parameter('left_image_topic', '/kitti/camera/left/image_raw')
        self.declare_parameter('right_image_topic', '/kitti/camera/right/image_raw')
        self.declare_parameter('camera_info_topic', '/kitti/camera/camera_info')
        self.declare_parameter('baseline', 0.54)  # KITTI stereo baseline
        self.declare_parameter('num_features', 1000)
        self.declare_parameter('scale_factor', 1.2)
        self.declare_parameter('num_levels', 8)
        self.declare_parameter('min_init_matches', 50)
        self.declare_parameter('min_track_matches', 30)
        self.declare_parameter('log_trajectory', True)
        self.declare_parameter('trajectory_file', 'Output/stereo_trajectory.txt')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera')
        
        # Get parameters
        left_topic = self.get_parameter('left_image_topic').value
        right_topic = self.get_parameter('right_image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.baseline = self.get_parameter('baseline').value
        num_features = self.get_parameter('num_features').value
        scale_factor = self.get_parameter('scale_factor').value
        num_levels = self.get_parameter('num_levels').value
        min_init_matches = self.get_parameter('min_init_matches').value
        min_track_matches = self.get_parameter('min_track_matches').value
        self.log_trajectory = self.get_parameter('log_trajectory').value
        trajectory_file = self.get_parameter('trajectory_file').value
        map_frame = self.get_parameter('map_frame').value
        camera_frame = self.get_parameter('camera_frame').value
        
        # ORB configuration
        self.orb_config = ORBConfig(
            num_features=num_features,
            scale_factor=scale_factor,
            num_levels=num_levels
        )
        
        self.min_init_matches = min_init_matches
        self.min_track_matches = min_track_matches
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics
        self.camera: Optional[CameraIntrinsics] = None
        self.camera_info_received = False
        
        # SLAM components
        self.tracker: Optional[StereoTracker] = None
        self.slam_map: Optional[Map] = None
        self.local_mapper: Optional[LocalMapper] = None
        self.loop_detector: Optional[LoopDetector] = None
        self.visualizer: Optional[SLAMVisualizer] = None
        
        # Trajectory logger
        if self.log_trajectory:
            self.tum_logger = TUMLogger(trajectory_file)
        else:
            self.tum_logger = None
        
        # Counters
        self.frame_count = 0
        self.last_keyframe: Optional[Keyframe] = None
        self.process_times = []
        
        # Store frame IDs
        self.map_frame = map_frame
        self.camera_frame = camera_frame
        
        # Camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # QoS for image topics
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Synchronized stereo subscribers
        self.left_sub = message_filters.Subscriber(
            self, Image, left_topic, qos_profile=image_qos
        )
        self.right_sub = message_filters.Subscriber(
            self, Image, right_topic, qos_profile=image_qos
        )
        
        # Time synchronizer
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=5,
            slop=0.1  # Allow 100ms time difference
        )
        self.sync.registerCallback(self.stereo_callback)
        
        self.get_logger().info('Stereo ORB-SLAM Node initialized')
        self.get_logger().info(f'  Left image topic: {left_topic}')
        self.get_logger().info(f'  Right image topic: {right_topic}')
        self.get_logger().info(f'  Baseline: {self.baseline}m')
        self.get_logger().info(f'  Num features: {num_features}')
        self.get_logger().info('Waiting for camera info...')
    
    def camera_info_callback(self, msg: CameraInfo):
        """Process camera info message."""
        if self.camera_info_received:
            return
        
        self.camera = CameraIntrinsics.from_camera_info(msg)
        self.camera_info_received = True
        
        self.get_logger().info(f'Camera info received: {self.camera}')
        
        self._initialize_slam()
    
    def _initialize_slam(self):
        """Initialize SLAM components."""
        # Stereo Tracker
        self.tracker = StereoTracker(
            camera=self.camera,
            baseline=self.baseline,
            orb_config=self.orb_config,
            min_init_matches=self.min_init_matches,
            min_track_matches=self.min_track_matches
        )
        
        # Map
        self.slam_map = Map()
        
        # Local mapper
        self.local_mapper = LocalMapper(
            slam_map=self.slam_map,
            camera=self.camera,
            min_parallax_deg=1.0
        )
        
        # Loop detector
        self.loop_detector = LoopDetector()
        
        # Visualizer
        self.visualizer = SLAMVisualizer(
            node=self,
            map_frame=self.map_frame,
            camera_frame=self.camera_frame
        )
        
        self.get_logger().info('Stereo SLAM components initialized')
    
    def stereo_callback(self, left_msg: Image, right_msg: Image):
        """Process synchronized stereo image pair."""
        if not self.camera_info_received:
            return
        
        timer = Timer("stereo_frame")
        timer.start()
        
        # Convert images
        try:
            left_image = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_image = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert images: {e}')
            return
        
        # Get timestamp
        timestamp = Time.from_msg(left_msg.header.stamp)
        timestamp_sec = timestamp.nanoseconds / 1e9
        
        # Process stereo frame
        state, pose = self.tracker.process_stereo_frame(
            left_image, right_image, timestamp_sec
        )
        
        self.frame_count += 1
        
        # Handle states
        if state == TrackingState.NOT_INITIALIZED:
            if self.frame_count % 30 == 0:
                self.get_logger().info('Waiting for stereo initialization...')
        
        elif state == TrackingState.OK:
            self._handle_tracking_success(pose, timestamp)
        
        elif state == TrackingState.LOST:
            self.get_logger().warn('Tracking lost!')
        
        # Timing
        elapsed = timer.stop()
        self.process_times.append(elapsed)
        
        if self.frame_count % 100 == 0:
            avg_time = np.mean(self.process_times[-100:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'Frame {self.frame_count}: {fps:.1f} FPS, '
                f'Map: {self.slam_map.num_keyframes} KFs, '
                f'{self.slam_map.num_map_points} MPs'
            )
    
    def _handle_tracking_success(self, pose: SE3, timestamp: Time):
        """Handle successful tracking."""
        # Log trajectory
        if self.tum_logger is not None:
            timestamp_sec = timestamp.nanoseconds / 1e9
            self.tum_logger.log(timestamp_sec, pose.inverse())
        
        # Add current stereo points to map for visualization
        current_frame = self.tracker.current_frame
        stereo_points, valid_indices = self.tracker.get_current_stereo_points()
        
        if len(stereo_points) > 0 and current_frame is not None and current_frame.pose is not None:
            from .utils import MapPoint
            
            # Transform stereo points to world frame and add to map
            for i, (pt_3d, kp_idx) in enumerate(zip(stereo_points[:50], valid_indices[:50])):  # Limit for performance
                world_pt = current_frame.pose.inverse().transform_point(pt_3d)
                
                # Only add if not too close to existing points
                mp = MapPoint(
                    id=self.slam_map.num_map_points,
                    position=world_pt,
                    descriptor=current_frame.descriptors[kp_idx] if kp_idx < len(current_frame.descriptors) else None
                )
                self.slam_map.add_map_point(mp)
        
        # Check for keyframe
        if self.tracker.should_create_keyframe():
            self._create_keyframe(timestamp)
        
        # Publish visualization with all map points
        if self.visualizer is not None:
            map_points = self.slam_map.get_map_points_array()
            self.visualizer.publish_all(pose, map_points, timestamp)
    
    def _create_keyframe(self, timestamp: Time):
        """Create a new keyframe from stereo data."""
        current_frame = self.tracker.current_frame
        
        if current_frame is None or current_frame.pose is None:
            return
        
        # First keyframe - use stereo initialization
        if self.slam_map.num_keyframes == 0:
            points_3d, kp_indices, depths = self.tracker.get_stereo_init_data()
            
            if len(points_3d) > 0:
                self._create_initial_stereo_map(current_frame, points_3d, kp_indices)
                return
        
        # Create regular keyframe
        keyframe = Keyframe(current_frame, self.camera)
        
        # Get current stereo points for this frame
        stereo_points, valid_indices = self.tracker.get_current_stereo_points()
        
        # Add stereo points directly as map points
        if len(stereo_points) > 0:
            for i, (pt_3d, kp_idx) in enumerate(zip(stereo_points, valid_indices)):
                # Transform to world frame
                world_pt = current_frame.pose.inverse().transform_point(pt_3d)
                
                from .utils import MapPoint
                mp = MapPoint(
                    id=self.slam_map.num_map_points + i,
                    position=world_pt,
                    descriptor=current_frame.descriptors[kp_idx]
                )
                mp.add_observation(keyframe.id, kp_idx)
                self.slam_map.add_map_point(mp)
                keyframe.add_map_point(kp_idx, mp)
        
        # Add keyframe to map
        self.slam_map.add_keyframe(keyframe)
        
        # Update references
        self.last_keyframe = keyframe
        self.tracker.set_reference_keyframe(keyframe)
        
        # Loop detection
        self.loop_detector.add_keyframe(keyframe)
        loop_kf_id = self.loop_detector.detect_loop(keyframe)
        if loop_kf_id is not None:
            self.get_logger().info(f'Loop detected with keyframe {loop_kf_id}!')
        
        self.get_logger().debug(
            f'Keyframe {keyframe.id}: {len(keyframe.map_points)} stereo points'
        )
    
    def _create_initial_stereo_map(self, frame: Frame, points_3d: np.ndarray,
                                    kp_indices: np.ndarray):
        """Create initial map from stereo points."""
        # Create first keyframe
        keyframe = Keyframe(frame, self.camera)
        
        # Add 3D points as map points
        from .utils import MapPoint
        
        for i, (pt_3d, kp_idx) in enumerate(zip(points_3d, kp_indices)):
            mp = MapPoint(
                id=i,
                position=pt_3d,  # Already in camera frame = world frame at first KF
                descriptor=frame.descriptors[kp_idx]
            )
            mp.add_observation(keyframe.id, kp_idx)
            self.slam_map.add_map_point(mp)
            keyframe.add_map_point(kp_idx, mp)
        
        self.slam_map.add_keyframe(keyframe)
        
        self.last_keyframe = keyframe
        self.tracker.set_reference_keyframe(keyframe)
        self.loop_detector.add_keyframe(keyframe)
        
        self.get_logger().info(
            f'Stereo map initialized: {len(points_3d)} points'
        )
    
    def _save_map_to_ply(self, filename: str):
        """Save map points to PLY file format."""
        if self.slam_map is None:
            return
        
        points = self.slam_map.get_map_points_array()
        
        if len(points) == 0:
            self.get_logger().warn('No map points to save')
            return
        
        with open(filename, 'w') as f:
            # PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            # Write points (white color)
            for pt in points:
                f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 255 255 255\n')
        
        self.get_logger().info(f'Saved {len(points)} points to {filename}')
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        # Save map to PLY
        if self.slam_map is not None and self.slam_map.num_map_points > 0:
            self._save_map_to_ply('Output/stereo_map.ply')
        
        if self.tum_logger is not None:
            self.tum_logger.close()
        
        self.get_logger().info(
            f'Shutting down. Processed {self.frame_count} frames, '
            f'{self.slam_map.num_keyframes if self.slam_map else 0} keyframes'
        )
        
        super().destroy_node()


def main(args=None):
    """Main entry point for stereo node."""
    rclpy.init(args=args)
    
    node = StereoORBSLAMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
