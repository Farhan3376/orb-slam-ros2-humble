"""
stereo.py - Stereo processing for ORB-SLAM

Provides stereo matching and depth computation from stereo image pairs.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .geometry import CameraIntrinsics


@dataclass
class StereoConfig:
    """Configuration for stereo processing."""
    baseline: float = 0.54  # KITTI stereo baseline in meters
    min_disparity: int = 0
    max_disparity: int = 128  # Maximum disparity search range
    block_size: int = 11  # Block matching window size
    min_depth: float = 0.5  # Minimum valid depth (meters)
    max_depth: float = 100.0  # Maximum valid depth (meters)
    match_threshold: int = 50  # Max Hamming distance for stereo match


class StereoMatcher:
    """
    Stereo matching for depth estimation.
    
    Uses ORB feature matching across stereo pairs with epipolar constraint.
    For rectified stereo (like KITTI), matching features must be on the same row.
    """
    
    def __init__(self, camera: CameraIntrinsics, config: StereoConfig = None):
        """
        Initialize stereo matcher.
        
        Args:
            camera: Camera intrinsics (same for both cameras in rectified stereo)
            config: Stereo configuration
        """
        self.camera = camera
        self.config = config or StereoConfig()
        
        # Precompute depth conversion factor: depth = factor / disparity
        self.depth_factor = camera.fx * self.config.baseline
        
        # BFMatcher for ORB
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def compute_stereo_matches(self, 
                                keypoints_left: tuple,
                                descriptors_left: np.ndarray,
                                keypoints_right: tuple,
                                descriptors_right: np.ndarray,
                                row_tolerance: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match ORB features between stereo pair with epipolar constraint.
        
        For rectified stereo images, matching points must be on the same row
        (horizontal epipolar lines).
        
        Args:
            keypoints_left: Keypoints from left image
            descriptors_left: Descriptors from left image
            keypoints_right: Keypoints from right image
            descriptors_right: Descriptors from right image
            row_tolerance: Maximum row difference for valid match (pixels)
        
        Returns:
            left_indices: Indices of matched keypoints in left image
            disparities: Disparity values for each match
        """
        if len(keypoints_left) == 0 or len(keypoints_right) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        if descriptors_left is None or descriptors_right is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # Get coordinates
        pts_left = np.array([kp.pt for kp in keypoints_left])
        pts_right = np.array([kp.pt for kp in keypoints_right])
        
        # KNN match
        matches = self.matcher.knnMatch(descriptors_left, descriptors_right, k=2)
        
        left_indices = []
        disparities = []
        
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            
            m, n = match_pair
            
            # Ratio test
            if m.distance >= 0.8 * n.distance:
                continue
            
            # Distance threshold
            if m.distance > self.config.match_threshold:
                continue
            
            left_idx = m.queryIdx
            right_idx = m.trainIdx
            
            left_pt = pts_left[left_idx]
            right_pt = pts_right[right_idx]
            
            # Epipolar constraint: same row
            row_diff = abs(left_pt[1] - right_pt[1])
            if row_diff > row_tolerance:
                continue
            
            # Disparity: left_x - right_x (should be positive for valid depth)
            disparity = left_pt[0] - right_pt[0]
            
            # Valid disparity range
            if disparity < self.config.min_disparity:
                continue
            if disparity > self.config.max_disparity:
                continue
            
            left_indices.append(left_idx)
            disparities.append(disparity)
        
        return np.array(left_indices, dtype=np.int32), np.array(disparities, dtype=np.float32)
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity to depth.
        
        depth = (fx * baseline) / disparity
        
        Args:
            disparity: Disparity values (can be array)
        
        Returns:
            depth: Depth values in meters
        """
        disparity = np.asarray(disparity, dtype=np.float32)
        
        # Avoid division by zero
        valid_mask = disparity > 0.5
        depth = np.zeros_like(disparity)
        depth[valid_mask] = self.depth_factor / disparity[valid_mask]
        
        # Clamp to valid range
        depth = np.clip(depth, self.config.min_depth, self.config.max_depth)
        
        return depth
    
    def compute_3d_points(self, 
                          keypoints: tuple,
                          left_indices: np.ndarray,
                          disparities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D points from stereo matches.
        
        Args:
            keypoints: Keypoints from left image
            left_indices: Indices of matched keypoints
            disparities: Disparity values
        
        Returns:
            points_3d: Nx3 array of 3D points in camera frame
            valid_indices: Indices of valid 3D points
        """
        if len(left_indices) == 0:
            return np.empty((0, 3)), np.array([], dtype=np.int32)
        
        # Get 2D coordinates
        pts_2d = np.array([keypoints[i].pt for i in left_indices])
        
        # Compute depth
        depths = self.disparity_to_depth(disparities)
        
        # Filter valid depths
        valid_mask = (depths > self.config.min_depth) & (depths < self.config.max_depth)
        
        if not np.any(valid_mask):
            return np.empty((0, 3)), np.array([], dtype=np.int32)
        
        pts_2d_valid = pts_2d[valid_mask]
        depths_valid = depths[valid_mask]
        valid_indices = left_indices[valid_mask]
        
        # Backproject to 3D
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        X = (pts_2d_valid[:, 0] - self.camera.cx) * depths_valid / self.camera.fx
        Y = (pts_2d_valid[:, 1] - self.camera.cy) * depths_valid / self.camera.fy
        Z = depths_valid
        
        points_3d = np.column_stack([X, Y, Z])
        
        return points_3d, valid_indices
    
    def process_stereo_frame(self,
                              keypoints_left: tuple,
                              descriptors_left: np.ndarray,
                              keypoints_right: tuple,
                              descriptors_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a stereo frame to get 3D points.
        
        Args:
            keypoints_left: Left image keypoints
            descriptors_left: Left image descriptors
            keypoints_right: Right image keypoints
            descriptors_right: Right image descriptors
        
        Returns:
            points_3d: Nx3 3D points in camera frame
            valid_indices: Indices of keypoints with valid depth
            depths: Depth values for valid keypoints
        """
        # Match features
        left_indices, disparities = self.compute_stereo_matches(
            keypoints_left, descriptors_left,
            keypoints_right, descriptors_right
        )
        
        if len(left_indices) == 0:
            return np.empty((0, 3)), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # Compute 3D points
        points_3d, valid_indices = self.compute_3d_points(
            keypoints_left, left_indices, disparities
        )
        
        # Get depths for valid points
        depths = self.disparity_to_depth(disparities[np.isin(left_indices, valid_indices)])
        
        return points_3d, valid_indices, depths


class DenseDisparityMatcher:
    """
    Dense disparity computation using block matching.
    
    Slower but provides depth for all pixels, not just features.
    """
    
    def __init__(self, camera: CameraIntrinsics, config: StereoConfig = None):
        self.camera = camera
        self.config = config or StereoConfig()
        
        self.depth_factor = camera.fx * self.config.baseline
        
        # Create StereoBM or StereoSGBM
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=self.config.max_disparity,
            blockSize=self.config.block_size,
            P1=8 * 3 * self.config.block_size ** 2,
            P2=32 * 3 * self.config.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
    
    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Compute dense disparity map.
        
        Args:
            img_left: Left grayscale image
            img_right: Right grayscale image
        
        Returns:
            disparity: HxW disparity map (values in pixels, divide by 16 for true disparity)
        """
        disparity = self.stereo.compute(img_left, img_right)
        
        # SGBM returns disparity * 16
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def compute_depth_map(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Compute dense depth map.
        
        Args:
            img_left: Left grayscale image
            img_right: Right grayscale image
        
        Returns:
            depth: HxW depth map in meters
        """
        disparity = self.compute_disparity(img_left, img_right)
        
        # Convert to depth
        valid_mask = disparity > 0.5
        depth = np.zeros_like(disparity)
        depth[valid_mask] = self.depth_factor / disparity[valid_mask]
        
        # Clamp
        depth = np.clip(depth, 0, self.config.max_depth)
        depth[~valid_mask] = 0
        
        return depth
    
    def get_depth_at_points(self, depth_map: np.ndarray, 
                            points_2d: np.ndarray) -> np.ndarray:
        """
        Get depth values at specific 2D points.
        
        Args:
            depth_map: HxW depth map
            points_2d: Nx2 array of (u, v) coordinates
        
        Returns:
            depths: N depth values
        """
        points_2d = np.asarray(points_2d).astype(np.int32)
        
        h, w = depth_map.shape
        
        # Clamp to valid range
        u = np.clip(points_2d[:, 0], 0, w - 1)
        v = np.clip(points_2d[:, 1], 0, h - 1)
        
        depths = depth_map[v, u]
        
        return depths
