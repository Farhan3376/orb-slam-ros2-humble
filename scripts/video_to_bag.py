#!/usr/bin/env python3
"""
Convert video file to ROS 2 bag for SLAM.

Usage:
    python3 video_to_bag.py --video /path/to/video.mp4 --output my_bag
    
    # With custom FPS
    python3 video_to_bag.py --video video.mp4 --output bag --fps 15
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
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata


def create_image_msg(frame: np.ndarray, timestamp_ns: int, frame_id: str = 'camera') -> Image:
    """Create ROS Image message from numpy array."""
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    msg = Image()
    msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.height = gray.shape[0]
    msg.width = gray.shape[1]
    msg.encoding = 'mono8'
    msg.is_bigendian = False
    msg.step = gray.shape[1]
    msg.data = gray.tobytes()
    
    return msg


def convert_video_to_bag(args):
    """Convert video file to ROS 2 bag."""
    
    video_path = args.video
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use specified FPS or original
    fps = args.fps if args.fps else original_fps
    
    print(f"Video: {video_path}")
    print(f"  Size: {width}x{height}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Output FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Calculate frame skip if downsampling
    frame_skip = 1
    if fps < original_fps:
        frame_skip = int(original_fps / fps)
        print(f"  Skipping every {frame_skip} frames")
    
    # Setup bag writer
    storage_options = StorageOptions(uri=args.output, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)
    
    # Create topic
    writer.create_topic(TopicMetadata(
        name='/camera/image_raw', type='sensor_msgs/msg/Image',
        serialization_format='cdr'))
    
    # Calculate timestamps
    frame_interval_ns = int(1_000_000_000 / fps)
    start_time_ns = 1_000_000_000_000
    
    print("\nConverting...")
    
    frame_idx = 0
    written_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Skip frames if downsampling
        if frame_idx % frame_skip == 0:
            timestamp_ns = start_time_ns + written_count * frame_interval_ns
            img_msg = create_image_msg(frame, timestamp_ns, 'camera')
            writer.write('/camera/image_raw', serialize_message(img_msg), timestamp_ns)
            written_count += 1
            
            if written_count % 100 == 0:
                print(f"  Written {written_count} frames...")
        
        frame_idx += 1
    
    cap.release()
    del writer
    
    duration_sec = written_count / fps
    
    print(f"\n✅ Bag created: {args.output}")
    print(f"   Frames: {written_count}")
    print(f"   Duration: {duration_sec:.1f} seconds")
    print(f"\nTo play:")
    print(f"  ros2 bag play {args.output}")
    print(f"\nTopic:")
    print(f"  /camera/image_raw")
    print(f"\n⚠️  Remember to set your camera intrinsics!")
    print(f"    Example: fx={width}, fy={width}, cx={width/2}, cy={height/2}")


def main():
    parser = argparse.ArgumentParser(description='Convert video to ROS 2 bag')
    
    parser.add_argument('--video', '-v', type=str, required=True, help='Input video file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output bag name')
    parser.add_argument('--fps', type=float, default=None, help='Output FPS (default: same as video)')
    
    args = parser.parse_args()
    
    convert_video_to_bag(args)


if __name__ == '__main__':
    main()
