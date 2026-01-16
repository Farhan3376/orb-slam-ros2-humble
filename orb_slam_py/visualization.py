"""
visualization.py - ROS 2 visualization publishers for ORB-SLAM

Publishes pose, path, map points, and TF transforms.
"""

import numpy as np
from typing import List, Optional
import struct

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from .geometry import SE3
from .utils import MapPoint


class SLAMVisualizer:
    """
    ROS 2 visualization publisher for SLAM results.
    
    Publishes:
    - Camera pose as PoseStamped
    - Trajectory as Path
    - Map points as PointCloud2
    - TF transform (map -> camera)
    """
    
    def __init__(self, node: Node,
                 pose_topic: str = '/orb_slam/pose',
                 path_topic: str = '/orb_slam/path',
                 points_topic: str = '/orb_slam/map_points',
                 map_frame: str = 'map',
                 camera_frame: str = 'camera'):
        """
        Initialize visualizer.
        
        Args:
            node: ROS 2 node
            pose_topic: Topic for current pose
            path_topic: Topic for trajectory path
            points_topic: Topic for map point cloud
            map_frame: Frame ID for map
            camera_frame: Frame ID for camera
        """
        self.node = node
        self.map_frame = map_frame
        self.camera_frame = camera_frame
        
        # Publishers
        self.pose_pub = node.create_publisher(PoseStamped, pose_topic, 10)
        self.path_pub = node.create_publisher(Path, path_topic, 10)
        self.points_pub = node.create_publisher(PointCloud2, points_topic, 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(node)
        
        # Path accumulator
        self.path = Path()
        self.path.header.frame_id = map_frame
        
        # Publish rate limiting for point cloud
        self.last_points_time = node.get_clock().now()
        self.points_publish_period = Duration(seconds=0.5)  # 2 Hz max
    
    def publish_pose(self, pose: SE3, timestamp: Optional[Time] = None):
        """
        Publish current camera pose.
        
        Args:
            pose: Camera pose (world to camera)
            timestamp: ROS timestamp (uses current time if None)
        """
        # Always use current time for visualization (bag timestamps cause TF issues)
        current_time = self.node.get_clock().now()
        
        # Convert to ROS pose (camera w.r.t. world)
        # pose is T_wc (world to camera), we need T_cw (camera in world)
        pose_inv = pose.inverse()
        
        msg = PoseStamped()
        msg.header.stamp = current_time.to_msg()
        msg.header.frame_id = self.map_frame
        
        # Position
        msg.pose.position.x = float(pose_inv.t[0])
        msg.pose.position.y = float(pose_inv.t[1])
        msg.pose.position.z = float(pose_inv.t[2])
        
        # Orientation (quaternion)
        quat = pose_inv.to_quaternion()  # x, y, z, w
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        self.pose_pub.publish(msg)
        
        # Add to path
        self.path.header.stamp = timestamp.to_msg()
        self.path.poses.append(msg)
        
        # Limit path length
        max_path_length = 1000
        if len(self.path.poses) > max_path_length:
            self.path.poses = self.path.poses[-max_path_length:]
    
    def publish_path(self, timestamp: Optional[Time] = None):
        """Publish accumulated trajectory path."""
        # Always use current time for visualization
        current_time = self.node.get_clock().now()
        self.path.header.stamp = current_time.to_msg()
        self.path_pub.publish(self.path)
    
    def publish_map_points(self, points: np.ndarray, timestamp: Optional[Time] = None):
        """
        Publish map points as PointCloud2.
        
        Args:
            points: Nx3 array of 3D points
            timestamp: ROS timestamp
        """
        if timestamp is None:
            timestamp = self.node.get_clock().now()
        
        # Rate limiting
        if (timestamp - self.last_points_time) < self.points_publish_period:
            return
        self.last_points_time = timestamp
        
        if len(points) == 0:
            return
        
        msg = self._create_pointcloud2(points, timestamp)
        self.points_pub.publish(msg)
    
    def _create_pointcloud2(self, points: np.ndarray, timestamp: Time) -> PointCloud2:
        """Create PointCloud2 message from numpy array."""
        msg = PointCloud2()
        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = self.map_frame
        
        msg.height = 1
        msg.width = len(points)
        
        # Fields: x, y, z as float32
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Pack point data
        data = points.astype(np.float32).tobytes()
        msg.data = list(data)
        
        return msg
    
    def publish_tf(self, pose: SE3, timestamp: Optional[Time] = None):
        """
        Broadcast TF transform from map to camera.
        
        Args:
            pose: Camera pose (world to camera)
            timestamp: ROS timestamp
        """
        if timestamp is None:
            timestamp = self.node.get_clock().now()
        
        # We want map -> camera transform
        # pose is T_wc, so camera position in world is pose.inverse()
        pose_inv = pose.inverse()
        
        t = TransformStamped()
        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.camera_frame
        
        # Translation
        t.transform.translation.x = float(pose_inv.t[0])
        t.transform.translation.y = float(pose_inv.t[1])
        t.transform.translation.z = float(pose_inv.t[2])
        
        # Rotation
        quat = pose_inv.to_quaternion()
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_all(self, pose: SE3, map_points: np.ndarray, 
                    timestamp: Optional[Time] = None):
        """
        Publish all visualization data.
        
        Args:
            pose: Current camera pose
            map_points: Nx3 array of map points
            timestamp: ROS timestamp
        """
        if timestamp is None:
            timestamp = self.node.get_clock().now()
        
        # Use current time for TF (required for RViz to show it)
        current_time = self.node.get_clock().now()
        
        self.publish_pose(pose, timestamp)
        self.publish_path(timestamp)
        self.publish_map_points(map_points, timestamp)
        # TF must use current time, not bag timestamp
        self.publish_tf(pose, current_time)
    
    def reset(self):
        """Reset path and visualization state."""
        self.path = Path()
        self.path.header.frame_id = self.map_frame


def create_marker_array_for_keyframes(keyframes, node: Node, 
                                       frame_id: str = 'map'):
    """
    Create visualization markers for keyframe poses.
    
    Returns a MarkerArray (requires visualization_msgs).
    """
    try:
        from visualization_msgs.msg import Marker, MarkerArray
        
        markers = MarkerArray()
        
        for i, kf in enumerate(keyframes):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = "keyframes"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            pose = kf.pose_se3.inverse()
            marker.pose.position.x = float(pose.t[0])
            marker.pose.position.y = float(pose.t[1])
            marker.pose.position.z = float(pose.t[2])
            
            quat = pose.to_quaternion()
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])
            
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            markers.markers.append(marker)
        
        return markers
    
    except ImportError:
        return None
