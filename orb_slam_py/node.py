"""
node.py - Main ROS 2 node for ORB-SLAM

Implements the orb_slam_py node that subscribes to camera images,
processes them through the SLAM pipeline, and publishes results.
"""

import numpy as np
import cv2
import os
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from .geometry import SE3, CameraIntrinsics
from .tracking import Tracker, TrackingState, ORBConfig
from .mapping import Map, LocalMapper, LoopDetector
from .visualization import SLAMVisualizer
from .utils import Frame, Keyframe, TUMLogger, Timer


class ORBSLAMNode(Node):
    """
    Main ORB-SLAM ROS 2 Node.
    
    Subscribes to camera images and camera info,
    runs the SLAM pipeline, and publishes results.
    """
    
    def __init__(self):
        super().__init__('orb_slam_py')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('num_features', 1000)
        self.declare_parameter('scale_factor', 1.2)
        self.declare_parameter('num_levels', 8)
        self.declare_parameter('min_parallax', 1.0)
        self.declare_parameter('min_init_matches', 100)
        self.declare_parameter('min_track_matches', 30)
        self.declare_parameter('log_trajectory', True)
        self.declare_parameter('trajectory_file', 'trajectory.txt')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera')
        
        # Get parameters
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        num_features = self.get_parameter('num_features').value
        scale_factor = self.get_parameter('scale_factor').value
        num_levels = self.get_parameter('num_levels').value
        min_parallax = self.get_parameter('min_parallax').value
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
        
        self.min_parallax = min_parallax
        self.min_init_matches = min_init_matches
        self.min_track_matches = min_track_matches
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics (set when camera_info received)
        self.camera: Optional[CameraIntrinsics] = None
        self.camera_info_received = False
        
        # SLAM components (initialized after camera info)
        self.tracker: Optional[Tracker] = None
        self.slam_map: Optional[Map] = None
        self.local_mapper: Optional[LocalMapper] = None
        self.loop_detector: Optional[LoopDetector] = None
        self.visualizer: Optional[SLAMVisualizer] = None
        
        # Trajectory logger
        if self.log_trajectory:
            self.tum_logger = TUMLogger(trajectory_file)
        else:
            self.tum_logger = None
        
        # Frame counter
        self.frame_count = 0
        self.last_keyframe: Optional[Keyframe] = None
        
        # Timing
        self.process_times = []
        
        # QoS for image topics
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            image_qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Store frame IDs
        self.map_frame = map_frame
        self.camera_frame = camera_frame
        
        self.get_logger().info(f'ORB-SLAM Node initialized')
        self.get_logger().info(f'  Image topic: {image_topic}')
        self.get_logger().info(f'  Camera info topic: {camera_info_topic}')
        self.get_logger().info(f'  Num features: {num_features}')
        self.get_logger().info(f'Waiting for camera info...')
    
    def camera_info_callback(self, msg: CameraInfo):
        """Process camera info message."""
        if self.camera_info_received:
            return
        
        self.camera = CameraIntrinsics.from_camera_info(msg)
        self.camera_info_received = True
        
        self.get_logger().info(f'Camera info received: {self.camera}')
        
        # Initialize SLAM components
        self._initialize_slam()
    
    def _initialize_slam(self):
        """Initialize SLAM components after camera info is received."""
        # Tracker
        self.tracker = Tracker(
            camera=self.camera,
            orb_config=self.orb_config,
            min_init_matches=self.min_init_matches,
            min_track_matches=self.min_track_matches,
            min_parallax=self.min_parallax
        )
        
        # Map
        self.slam_map = Map()
        
        # Local mapper
        self.local_mapper = LocalMapper(
            slam_map=self.slam_map,
            camera=self.camera,
            min_parallax_deg=self.min_parallax
        )
        
        # Loop detector
        self.loop_detector = LoopDetector()
        
        # Visualizer
        self.visualizer = SLAMVisualizer(
            node=self,
            map_frame=self.map_frame,
            camera_frame=self.camera_frame
        )
        
        self.get_logger().info('SLAM components initialized')
    
    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        if not self.camera_info_received:
            return
        
        timer = Timer("frame")
        timer.start()
        
        # Convert ROS image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Get timestamp
        timestamp = Time.from_msg(msg.header.stamp)
        timestamp_sec = timestamp.nanoseconds / 1e9
        
        # Process frame through tracker
        state, pose = self.tracker.process_frame(cv_image, timestamp_sec)
        
        self.frame_count += 1
        
        # Handle different tracking states
        if state == TrackingState.NOT_INITIALIZED:
            if self.frame_count % 30 == 0:
                self.get_logger().info('Waiting for initialization...')
        
        elif state == TrackingState.INITIALIZING:
            self.get_logger().debug(
                f'Initializing: {self.tracker.num_matches} matches'
            )
        
        elif state == TrackingState.OK:
            self._handle_tracking_success(pose, timestamp)
        
        elif state == TrackingState.LOST:
            self.get_logger().warn('Tracking lost!')
        
        # Timing
        elapsed = timer.stop()
        self.process_times.append(elapsed)
        
        # Log FPS periodically
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
            # Log camera position in world frame
            self.tum_logger.log(timestamp_sec, pose.inverse())
        
        # Check if we should create a keyframe
        if self.tracker.should_create_keyframe():
            self._create_keyframe(timestamp)
        
        # Publish visualization
        if self.visualizer is not None:
            map_points = self.slam_map.get_map_points_array()
            self.visualizer.publish_all(pose, map_points, timestamp)
    
    def _create_keyframe(self, timestamp: Time):
        """Create a new keyframe."""
        current_frame = self.tracker.current_frame
        
        if current_frame is None or current_frame.pose is None:
            return
        
        # Check if this is the first keyframe (after initialization)
        if self.slam_map.num_keyframes == 0:
            # Create initial map from two-view initialization
            init_frame, matches, inliers = self.tracker.get_init_data()
            
            if init_frame is not None and len(matches) > 0:
                kf1, kf2 = self.local_mapper.create_initial_map(
                    init_frame, current_frame, matches, inliers
                )
                
                self.last_keyframe = kf2
                self.tracker.set_reference_keyframe(kf2)
                
                # Add to loop detector
                self.loop_detector.add_keyframe(kf1)
                self.loop_detector.add_keyframe(kf2)
                
                self.get_logger().info(
                    f'Initial map created: {self.slam_map.num_map_points} points'
                )
                return
        
        # Create new keyframe
        keyframe = Keyframe(current_frame, self.camera)
        
        # Match with last keyframe for triangulation
        if self.last_keyframe is not None:
            from .tracking import FeatureMatcher
            matcher = FeatureMatcher(ratio_threshold=0.75)
            
            matches = matcher.match(
                self.last_keyframe.descriptors,
                keyframe.descriptors,
                max_distance=50
            )
            
            inlier_mask = np.ones(len(matches), dtype=bool)
            
            self.local_mapper.process_new_keyframe(
                keyframe, self.last_keyframe, matches, inlier_mask
            )
        else:
            self.local_mapper.process_new_keyframe(keyframe)
        
        # Update reference
        self.last_keyframe = keyframe
        self.tracker.set_reference_keyframe(keyframe)
        
        # Add to loop detector
        self.loop_detector.add_keyframe(keyframe)
        
        # Check for loop closure
        loop_kf_id = self.loop_detector.detect_loop(keyframe)
        if loop_kf_id is not None:
            self.get_logger().info(f'Loop detected with keyframe {loop_kf_id}!')
            loop_kf = self.slam_map.keyframes.get(loop_kf_id)
            if loop_kf is not None:
                self.loop_detector.correct_loop(keyframe, loop_kf, self.slam_map)
        
        self.get_logger().debug(
            f'Keyframe {keyframe.id} created: {len(keyframe.map_points)} map points'
        )
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        if self.tum_logger is not None:
            self.tum_logger.close()
        
        self.get_logger().info(
            f'Shutting down. Processed {self.frame_count} frames, '
            f'{self.slam_map.num_keyframes if self.slam_map else 0} keyframes'
        )
        
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = ORBSLAMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
