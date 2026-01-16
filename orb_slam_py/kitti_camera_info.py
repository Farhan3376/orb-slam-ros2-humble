#!/usr/bin/env python3
"""
KITTI Camera Info Publisher

Publishes CameraInfo messages for KITTI dataset since the bag 
doesn't include camera calibration. Also publishes TF for camera frames.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class KITTICameraInfoPublisher(Node):
    """Publishes CameraInfo and TF for KITTI dataset."""
    
    def __init__(self):
        super().__init__('kitti_camera_info_publisher')
        
        # Declare parameters with KITTI defaults
        self.declare_parameter('fx', 718.856)
        self.declare_parameter('fy', 718.856)
        self.declare_parameter('cx', 607.1928)
        self.declare_parameter('cy', 185.2157)
        self.declare_parameter('width', 1242)
        self.declare_parameter('height', 375)
        
        # Get parameters
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        
        # Create camera info for left camera
        self.camera_info_left = self._create_camera_info(
            fx, fy, cx, cy, width, height, 'camera_left'
        )
        
        # Create camera info for right camera (same intrinsics for KITTI)
        self.camera_info_right = self._create_camera_info(
            fx, fy, cx, cy, width, height, 'camera_right'
        )
        
        # Publishers for camera info
        self.pub_left = self.create_publisher(
            CameraInfo, '/kitti/camera/camera_info', 10
        )
        self.pub_left_raw = self.create_publisher(
            CameraInfo, '/kitti/camera/left/camera_info', 10
        )
        self.pub_right = self.create_publisher(
            CameraInfo, '/kitti/camera/right/camera_info', 10
        )
        
        # Static TF broadcaster for camera frames
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()
        
        # Publish camera info at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_camera_info)
        
        self.get_logger().info(
            f'Publishing KITTI camera info: fx={fx}, fy={fy}, '
            f'cx={cx}, cy={cy}, {width}x{height}'
        )
        self.get_logger().info('Publishing TF: map -> camera_left, map -> camera_right')
    
    def _create_camera_info(self, fx, fy, cx, cy, width, height, frame_id):
        """Create CameraInfo message."""
        info = CameraInfo()
        info.header.frame_id = frame_id
        info.width = width
        info.height = height
        
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        return info
    
    def _publish_static_transforms(self):
        """Publish static transforms for camera frames."""
        transforms = []
        
        # map -> camera_left (identity, camera at origin)
        t_left = TransformStamped()
        t_left.header.stamp = self.get_clock().now().to_msg()
        t_left.header.frame_id = 'map'
        t_left.child_frame_id = 'camera_left'
        t_left.transform.translation.x = 0.0
        t_left.transform.translation.y = 0.0
        t_left.transform.translation.z = 0.0
        t_left.transform.rotation.x = 0.0
        t_left.transform.rotation.y = 0.0
        t_left.transform.rotation.z = 0.0
        t_left.transform.rotation.w = 1.0
        transforms.append(t_left)
        
        # map -> camera_right (offset by stereo baseline ~0.54m)
        t_right = TransformStamped()
        t_right.header.stamp = self.get_clock().now().to_msg()
        t_right.header.frame_id = 'map'
        t_right.child_frame_id = 'camera_right'
        t_right.transform.translation.x = 0.54  # KITTI stereo baseline
        t_right.transform.translation.y = 0.0
        t_right.transform.translation.z = 0.0
        t_right.transform.rotation.x = 0.0
        t_right.transform.rotation.y = 0.0
        t_right.transform.rotation.z = 0.0
        t_right.transform.rotation.w = 1.0
        transforms.append(t_right)
        
        self.tf_broadcaster.sendTransform(transforms)
    
    def publish_camera_info(self):
        """Publish camera info with current timestamp."""
        now = self.get_clock().now().to_msg()
        
        self.camera_info_left.header.stamp = now
        self.camera_info_right.header.stamp = now
        
        self.pub_left.publish(self.camera_info_left)
        self.pub_left_raw.publish(self.camera_info_left)
        self.pub_right.publish(self.camera_info_right)


def main(args=None):
    rclpy.init(args=args)
    node = KITTICameraInfoPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

