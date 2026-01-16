#!/usr/bin/env python3
"""
Convert image folder(s) to ROS 2 bag for SLAM.

Supports:
- Monocular: Single image folder
- Stereo: Left and right image folders

Usage:
    # Monocular
    python3 images_to_bag.py --images /path/to/images --output my_bag
    
    # Stereo
    python3 images_to_bag.py --left /path/to/left --right /path/to/right --output stereo_bag
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# ROS 2 imports
import rclpy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata


def get_image_files(folder: str, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    """Get sorted list of image files from folder."""
    folder = Path(folder)
    files = []
    
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def create_image_msg(image_path: str, timestamp_ns: int, frame_id: str = 'camera') -> Image:
    """Create ROS Image message from file."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    msg = Image()
    msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = 'mono8'
    msg.is_bigendian = False
    msg.step = img.shape[1]
    msg.data = img.tobytes()
    
    return msg


def create_camera_info_msg(width: int, height: int, fx: float, fy: float, 
                            cx: float, cy: float, timestamp_ns: int, 
                            frame_id: str = 'camera') -> CameraInfo:
    """Create CameraInfo message."""
    msg = CameraInfo()
    msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.distortion_model = 'plumb_bob'
    
    return msg


def convert_images_to_bag(args):
    """Convert image folder(s) to ROS 2 bag."""
    
    # Determine mode
    is_stereo = args.left is not None and args.right is not None
    
    if is_stereo:
        left_files = get_image_files(args.left)
        right_files = get_image_files(args.right)
        
        if len(left_files) != len(right_files):
            print(f"Warning: Left ({len(left_files)}) and right ({len(right_files)}) have different counts")
        
        num_images = min(len(left_files), len(right_files))
        print(f"Stereo mode: {num_images} image pairs")
    else:
        image_folder = args.images or args.left
        image_files = get_image_files(image_folder)
        num_images = len(image_files)
        print(f"Monocular mode: {num_images} images")
    
    if num_images == 0:
        print("Error: No images found!")
        return
    
    # Get image dimensions from first image
    first_image = args.left if is_stereo else (args.images or args.left)
    first_file = get_image_files(first_image)[0]
    sample = cv2.imread(str(first_file))
    height, width = sample.shape[:2]
    
    print(f"Image size: {width}x{height}")
    
    # Setup bag writer
    storage_options = StorageOptions(uri=args.output, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)
    
    # Create topics
    if is_stereo:
        writer.create_topic(TopicMetadata(
            name='/camera/left/image_raw', type='sensor_msgs/msg/Image', 
            serialization_format='cdr'))
        writer.create_topic(TopicMetadata(
            name='/camera/right/image_raw', type='sensor_msgs/msg/Image',
            serialization_format='cdr'))
    else:
        writer.create_topic(TopicMetadata(
            name='/camera/image_raw', type='sensor_msgs/msg/Image',
            serialization_format='cdr'))
    
    # Calculate timestamps (based on fps)
    fps = args.fps
    frame_interval_ns = int(1_000_000_000 / fps)
    start_time_ns = 1_000_000_000_000  # Arbitrary start time
    
    print(f"Writing at {fps} FPS...")
    
    # Write images
    for i in range(num_images):
        timestamp_ns = start_time_ns + i * frame_interval_ns
        
        if is_stereo:
            left_msg = create_image_msg(left_files[i], timestamp_ns, 'camera_left')
            right_msg = create_image_msg(right_files[i], timestamp_ns, 'camera_right')
            
            writer.write('/camera/left/image_raw', serialize_message(left_msg), timestamp_ns)
            writer.write('/camera/right/image_raw', serialize_message(right_msg), timestamp_ns)
        else:
            img_msg = create_image_msg(image_files[i], timestamp_ns, 'camera')
            writer.write('/camera/image_raw', serialize_message(img_msg), timestamp_ns)
        
        if (i + 1) % 100 == 0 or i == num_images - 1:
            print(f"  Processed {i + 1}/{num_images} images")
    
    del writer
    
    print(f"\nBag created: {args.output}")
    print(f"\nTo play:")
    print(f"  ros2 bag play {args.output}")
    
    if is_stereo:
        print(f"\nStereo topics:")
        print(f"  /camera/left/image_raw")
        print(f"  /camera/right/image_raw")
    else:
        print(f"\nMonocular topic:")
        print(f"  /camera/image_raw")
    
    # Print camera calibration reminder
    print(f"\n⚠️  Remember to set your camera intrinsics!")
    print(f"    Example: fx={width}, fy={width}, cx={width/2}, cy={height/2}")


def main():
    parser = argparse.ArgumentParser(description='Convert images to ROS 2 bag')
    
    # Image sources
    parser.add_argument('--images', '-i', type=str, help='Image folder (monocular)')
    parser.add_argument('--left', '-l', type=str, help='Left image folder (stereo)')
    parser.add_argument('--right', '-r', type=str, help='Right image folder (stereo)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, required=True, help='Output bag name')
    
    # Options
    parser.add_argument('--fps', type=float, default=10.0, help='Playback FPS (default: 10)')
    
    args = parser.parse_args()
    
    # Validate input
    if args.images is None and (args.left is None or args.right is None):
        parser.error("Provide either --images (monocular) or --left and --right (stereo)")
    
    convert_images_to_bag(args)


if __name__ == '__main__':
    main()
