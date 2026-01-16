"""
tracking.py - Feature extraction and tracking for ORB-SLAM

Contains ORB feature extractor, feature matcher, and the main Tracker class
that handles frame-to-frame motion estimation.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from .geometry import (
    SE3, CameraIntrinsics, compute_essential_matrix,
    recover_pose_from_essential, solve_pnp_ransac, compute_parallax,
    compute_reprojection_error
)
from .utils import Frame, MapPoint, Keyframe, Timer


class TrackingState(Enum):
    """State of the tracking system."""
    NOT_INITIALIZED = 0
    INITIALIZING = 1
    OK = 2
    LOST = 3


@dataclass
class ORBConfig:
    """Configuration for ORB feature extractor."""
    num_features: int = 1000
    scale_factor: float = 1.2
    num_levels: int = 8
    edge_threshold: int = 31
    first_level: int = 0
    wta_k: int = 2
    patch_size: int = 31
    fast_threshold: int = 20


class ORBExtractor:
    """
    ORB feature extractor with configurable parameters.
    
    ORB (Oriented FAST and Rotated BRIEF) provides:
    - FAST corner detection
    - Harris corner measure for keypoint selection
    - BRIEF descriptor with rotation invariance
    """
    
    def __init__(self, config: ORBConfig = None):
        if config is None:
            config = ORBConfig()
        
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=config.num_features,
            scaleFactor=config.scale_factor,
            nlevels=config.num_levels,
            edgeThreshold=config.edge_threshold,
            firstLevel=config.first_level,
            WTA_K=config.wta_k,
            patchSize=config.patch_size,
            fastThreshold=config.fast_threshold
        )
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[tuple, np.ndarray]:
        """
        Detect ORB keypoints and compute descriptors.
        
        Args:
            image: Grayscale image
            
        Returns:
            keypoints: Tuple of cv2.KeyPoint
            descriptors: Nx32 array of ORB descriptors
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        if descriptors is None:
            return tuple(), np.empty((0, 32), dtype=np.uint8)
        
        return tuple(keypoints), descriptors
    
    def detect(self, image: np.ndarray) -> tuple:
        """Detect keypoints only."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints = self.orb.detect(image, None)
        return tuple(keypoints)
    
    def compute(self, image: np.ndarray, keypoints: tuple) -> Tuple[tuple, np.ndarray]:
        """Compute descriptors for given keypoints."""
        keypoints, descriptors = self.orb.compute(image, keypoints)
        
        if descriptors is None:
            return tuple(), np.empty((0, 32), dtype=np.uint8)
        
        return tuple(keypoints), descriptors


class FeatureMatcher:
    """
    Feature matcher using BFMatcher with Hamming distance.
    
    Uses cross-check and ratio test for robust matching.
    """
    
    def __init__(self, ratio_threshold: float = 0.75, cross_check: bool = False):
        """
        Initialize matcher.
        
        Args:
            ratio_threshold: Lowe's ratio test threshold
            cross_check: Use cross-check (slower but more robust)
        """
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        
        # BFMatcher with Hamming distance for ORB
        if cross_check:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match(self, desc1: np.ndarray, desc2: np.ndarray,
              max_distance: int = 50) -> List[cv2.DMatch]:
        """
        Match descriptors between two frames.
        
        Args:
            desc1: Nx32 descriptors from frame 1
            desc2: Mx32 descriptors from frame 2
            max_distance: Maximum Hamming distance for valid match
            
        Returns:
            List of cv2.DMatch objects
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        if self.cross_check:
            matches = self.matcher.match(desc1, desc2)
            # Filter by distance
            matches = [m for m in matches if m.distance <= max_distance]
        else:
            # KNN matching with ratio test
            matches_knn = self.matcher.knnMatch(desc1, desc2, k=2)
            
            matches = []
            for match_pair in matches_knn:
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                # Lowe's ratio test
                if m.distance < self.ratio_threshold * n.distance:
                    if m.distance <= max_distance:
                        matches.append(m)
        
        return matches
    
    def match_with_projection(self, desc1: np.ndarray, pts1: np.ndarray,
                               desc2: np.ndarray, pts2: np.ndarray,
                               max_distance: int = 50,
                               search_radius: float = 50.0) -> List[cv2.DMatch]:
        """
        Match descriptors with spatial constraint.
        
        Only matches features that are within search_radius of each other.
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        matches = []
        
        for i in range(len(desc1)):
            # Find candidates within radius
            dists = np.linalg.norm(pts2 - pts1[i], axis=1)
            candidates = np.where(dists < search_radius)[0]
            
            if len(candidates) == 0:
                continue
            
            # Match with candidates only
            best_dist = max_distance + 1
            best_idx = -1
            second_dist = max_distance + 1
            
            for j in candidates:
                dist = cv2.norm(desc1[i], desc2[j], cv2.NORM_HAMMING)
                if dist < best_dist:
                    second_dist = best_dist
                    best_dist = dist
                    best_idx = j
                elif dist < second_dist:
                    second_dist = dist
            
            # Ratio test
            if best_idx >= 0 and best_dist < self.ratio_threshold * second_dist:
                if best_dist <= max_distance:
                    matches.append(cv2.DMatch(i, best_idx, best_dist))
        
        return matches


class Tracker:
    """
    Main tracking class for ORB-SLAM.
    
    Handles:
    - Frame processing and feature extraction
    - Initialization from two views
    - Frame-to-frame tracking via PnP
    - Motion model prediction
    - Keyframe decision
    """
    
    def __init__(self, camera: CameraIntrinsics, 
                 orb_config: ORBConfig = None,
                 min_init_matches: int = 100,
                 min_track_matches: int = 30,
                 min_parallax: float = 1.0):
        """
        Initialize tracker.
        
        Args:
            camera: Camera intrinsic parameters
            orb_config: ORB extractor configuration
            min_init_matches: Minimum matches for initialization
            min_track_matches: Minimum matches for successful tracking
            min_parallax: Minimum parallax (degrees) for initialization
        """
        self.camera = camera
        self.extractor = ORBExtractor(orb_config)
        self.matcher = FeatureMatcher(ratio_threshold=0.75)
        
        self.min_init_matches = min_init_matches
        self.min_track_matches = min_track_matches
        self.min_parallax = min_parallax
        
        # State
        self.state = TrackingState.NOT_INITIALIZED
        self.frame_id = 0
        
        # Frames
        self.current_frame: Optional[Frame] = None
        self.last_frame: Optional[Frame] = None
        self.reference_keyframe: Optional[Keyframe] = None
        self.init_frame: Optional[Frame] = None
        
        # Motion model (constant velocity)
        self.velocity: Optional[SE3] = None
        
        # Map point associations for current frame
        self.current_map_points: dict = {}  # kp_idx -> MapPoint
        
        # Statistics
        self.num_matches = 0
        self.num_inliers = 0
    
    def process_frame(self, image: np.ndarray, timestamp: float) -> Tuple[TrackingState, Optional[SE3]]:
        """
        Process a new frame through the tracking pipeline.
        
        Args:
            image: Input image (grayscale or BGR)
            timestamp: Frame timestamp
            
        Returns:
            state: Current tracking state
            pose: Estimated camera pose (or None if not available)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract ORB features
        keypoints, descriptors = self.extractor.detect_and_compute(gray)
        
        # Create frame
        frame = Frame(
            id=self.frame_id,
            timestamp=timestamp,
            image=gray,
            keypoints=keypoints,
            descriptors=descriptors,
            pose=None
        )
        
        self.frame_id += 1
        
        # Process based on state
        if self.state == TrackingState.NOT_INITIALIZED:
            self._try_initialize_monocular(frame)
        
        elif self.state == TrackingState.INITIALIZING:
            self._try_initialize_monocular(frame)
        
        elif self.state == TrackingState.OK:
            success = self._track_frame(frame)
            if not success:
                self.state = TrackingState.LOST
        
        elif self.state == TrackingState.LOST:
            # Try to relocalize
            success = self._try_relocalize(frame)
            if success:
                self.state = TrackingState.OK
        
        self.current_frame = frame
        
        return self.state, frame.pose
    
    def _try_initialize_monocular(self, frame: Frame) -> bool:
        """
        Try to initialize the map from two views.
        
        Uses the Essential matrix to recover relative pose and
        triangulates initial map points.
        """
        if self.state == TrackingState.NOT_INITIALIZED:
            # Store first frame
            self.init_frame = frame
            self.state = TrackingState.INITIALIZING
            return False
        
        # Match with init frame
        matches = self.matcher.match(
            self.init_frame.descriptors,
            frame.descriptors,
            max_distance=50
        )
        
        self.num_matches = len(matches)
        
        if len(matches) < self.min_init_matches:
            # Not enough matches, reset if too few
            if len(matches) < 50:
                self.init_frame = frame
            return False
        
        # Get matched points
        pts1 = np.array([self.init_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Check parallax
        parallax = compute_parallax(pts1, pts2)
        if parallax < self.min_parallax * self.camera.fx / 57.3:  # Convert to pixels
            return False
        
        # Compute Essential matrix
        E, mask = compute_essential_matrix(pts1, pts2, self.camera.K)
        
        if E is None or mask is None:
            return False
        
        # Recover pose
        relative_pose, pose_mask = recover_pose_from_essential(
            E, pts1, pts2, self.camera.K, mask
        )
        
        # Count inliers
        inlier_mask = mask & pose_mask
        self.num_inliers = np.sum(inlier_mask)
        
        if self.num_inliers < self.min_init_matches // 2:
            return False
        
        # Set poses
        self.init_frame.pose = SE3()  # First frame at origin
        frame.pose = relative_pose
        
        # Initialization successful
        self.state = TrackingState.OK
        self.last_frame = frame
        
        # Store matched points and inlier mask for mapping
        self._init_matches = matches
        self._init_inliers = inlier_mask
        
        return True
    
    def _track_frame(self, frame: Frame) -> bool:
        """
        Track current frame against last frame / reference keyframe.
        
        Uses PnP if we have 3D map points, otherwise Essential matrix.
        """
        # Predict pose using motion model
        if self.velocity is not None and self.last_frame.pose is not None:
            predicted_pose = self.last_frame.pose @ self.velocity
        else:
            predicted_pose = self.last_frame.pose
        
        # Try to track with map points first
        if self.reference_keyframe is not None and len(self.current_map_points) > 0:
            success = self._track_with_map_points(frame, predicted_pose)
            if success:
                self._update_velocity(frame)
                return True
        
        # Fall back to frame-to-frame tracking
        success = self._track_with_last_frame(frame, predicted_pose)
        
        if success:
            self._update_velocity(frame)
        
        return success
    
    def _track_with_map_points(self, frame: Frame, predicted_pose: SE3) -> bool:
        """
        Track using 3D map points from reference keyframe.
        
        Projects map points to current frame, matches, and solves PnP.
        """
        if self.reference_keyframe is None:
            return False
        
        # Get map points with valid 3D positions
        kf_map_points = self.reference_keyframe.get_map_point_indices()
        
        if len(kf_map_points) < 10:
            return False
        
        # Prepare 3D points and descriptors
        points_3d = []
        descriptors = []
        kp_indices = []
        
        for kp_idx, mp in kf_map_points.items():
            if not mp.is_bad:
                points_3d.append(mp.position)
                descriptors.append(mp.descriptor)
                kp_indices.append(kp_idx)
        
        if len(points_3d) < 10:
            return False
        
        points_3d = np.array(points_3d)
        descriptors = np.array(descriptors)
        
        # Match with current frame
        matches = self.matcher.match(descriptors, frame.descriptors, max_distance=50)
        
        if len(matches) < self.min_track_matches:
            return False
        
        # Prepare PnP data
        matched_3d = np.array([points_3d[m.queryIdx] for m in matches])
        matched_2d = np.array([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Solve PnP
        pose, inliers = solve_pnp_ransac(
            matched_3d, matched_2d,
            self.camera.K, self.camera.dist_coeffs,
            initial_pose=predicted_pose
        )
        
        if pose is None:
            return False
        
        self.num_inliers = np.sum(inliers)
        
        if self.num_inliers < self.min_track_matches:
            return False
        
        frame.pose = pose
        
        # Update current map point associations
        self.current_map_points.clear()
        for i, (m, is_inlier) in enumerate(zip(matches, inliers)):
            if is_inlier:
                mp_idx = kp_indices[m.queryIdx] if m.queryIdx < len(kp_indices) else -1
                if mp_idx >= 0 and mp_idx in kf_map_points:
                    self.current_map_points[m.trainIdx] = kf_map_points[mp_idx]
        
        self.num_matches = len(matches)
        
        return True
    
    def _track_with_last_frame(self, frame: Frame, predicted_pose: SE3) -> bool:
        """
        Track against last frame using Essential matrix or simple matching.
        """
        if self.last_frame is None or self.last_frame.pose is None:
            return False
        
        # Match with last frame
        matches = self.matcher.match(
            self.last_frame.descriptors,
            frame.descriptors,
            max_distance=50
        )
        
        self.num_matches = len(matches)
        
        if len(matches) < self.min_track_matches:
            return False
        
        # Get matched points
        pts1 = np.array([self.last_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.array([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Compute Essential matrix
        E, mask = compute_essential_matrix(pts1, pts2, self.camera.K)
        
        if E is None or mask is None:
            return False
        
        # Recover relative pose
        relative_pose, pose_mask = recover_pose_from_essential(
            E, pts1, pts2, self.camera.K, mask
        )
        
        inlier_mask = mask & pose_mask
        self.num_inliers = np.sum(inlier_mask)
        
        if self.num_inliers < self.min_track_matches:
            return False
        
        # Compose with last frame pose
        frame.pose = self.last_frame.pose @ relative_pose
        
        return True
    
    def _try_relocalize(self, frame: Frame) -> bool:
        """
        Try to relocalize when tracking is lost.
        
        Matches against all keyframes and tries PnP.
        """
        # Simplified: just try to initialize again
        self.state = TrackingState.NOT_INITIALIZED
        self.init_frame = None
        return False
    
    def _update_velocity(self, frame: Frame):
        """Update motion model velocity."""
        if self.last_frame is not None and self.last_frame.pose is not None and frame.pose is not None:
            # velocity = T_last^-1 @ T_current
            self.velocity = self.last_frame.pose.inverse() @ frame.pose
        
        self.last_frame = frame
    
    def should_create_keyframe(self) -> bool:
        """
        Decide if current frame should become a keyframe.
        
        Criteria:
        - Enough frames since last keyframe
        - Sufficient parallax from reference keyframe
        - Not too many tracked points (exploring new area)
        """
        if self.current_frame is None or self.current_frame.pose is None:
            return False
        
        if self.state != TrackingState.OK:
            return False
        
        # First keyframe after initialization
        if self.reference_keyframe is None:
            return True
        
        # Minimum frames between keyframes
        if self.current_frame.id - self.reference_keyframe.id < 10:
            return False
        
        # Check tracking quality
        if self.num_inliers < self.min_track_matches:
            return True  # Low tracking = need new keyframe
        
        # Check displacement from reference keyframe
        if self.current_frame.pose is not None:
            ref_pose = self.reference_keyframe.pose_se3
            curr_pose = self.current_frame.pose
            
            # Translation magnitude
            rel_pose = ref_pose.inverse() @ curr_pose
            translation = np.linalg.norm(rel_pose.t)
            
            # Create keyframe if moved significantly
            if translation > 0.1:  # Threshold in world units
                return True
        
        return False
    
    def get_init_data(self) -> Tuple[Optional[Frame], List, np.ndarray]:
        """
        Get initialization data for mapping.
        
        Returns:
            init_frame: First frame of initialization
            matches: Feature matches
            inlier_mask: Inlier mask
        """
        if hasattr(self, '_init_matches') and hasattr(self, '_init_inliers'):
            return self.init_frame, self._init_matches, self._init_inliers
        return None, [], np.array([])
    
    def set_reference_keyframe(self, keyframe: Keyframe):
        """Set the reference keyframe for tracking."""
        self.reference_keyframe = keyframe
    
    def get_current_map_points(self) -> dict:
        """Get map points associated with current frame."""
        return self.current_map_points


class StereoTracker(Tracker):
    """
    Stereo tracker that uses both cameras for depth estimation.
    
    Extends the monocular Tracker with:
    - Stereo feature matching for depth
    - Direct 3D point creation from stereo
    - Faster initialization (no parallax requirement)
    """
    
    def __init__(self, camera: CameraIntrinsics,
                 baseline: float = 0.54,
                 orb_config: ORBConfig = None,
                 min_init_matches: int = 50,
                 min_track_matches: int = 30):
        """
        Initialize stereo tracker.
        
        Args:
            camera: Camera intrinsics (same for rectified left/right)
            baseline: Stereo baseline in meters
            orb_config: ORB configuration
            min_init_matches: Minimum stereo matches for initialization
            min_track_matches: Minimum matches for tracking
        """
        super().__init__(camera, orb_config, min_init_matches, min_track_matches, min_parallax=0)
        
        self.baseline = baseline
        
        # Import stereo module (lazy import to avoid circular)
        from .stereo import StereoMatcher, StereoConfig
        
        stereo_config = StereoConfig(baseline=baseline)
        self.stereo_matcher = StereoMatcher(camera, stereo_config)
        
        # Stereo data for current frame
        self.current_3d_points: Optional[np.ndarray] = None
        self.current_depths: Optional[np.ndarray] = None
        self.stereo_valid_indices: Optional[np.ndarray] = None
    
    def process_stereo_frame(self, image_left: np.ndarray, image_right: np.ndarray,
                             timestamp: float) -> Tuple[TrackingState, Optional[SE3]]:
        """
        Process a stereo frame pair.
        
        Args:
            image_left: Left image
            image_right: Right image
            timestamp: Frame timestamp
        
        Returns:
            state: Tracking state
            pose: Estimated pose (or None)
        """
        # Convert to grayscale
        if len(image_left.shape) == 3:
            gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = image_left
        
        if len(image_right.shape) == 3:
            gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = image_right
        
        # Extract ORB from both images
        keypoints_left, descriptors_left = self.extractor.detect_and_compute(gray_left)
        keypoints_right, descriptors_right = self.extractor.detect_and_compute(gray_right)
        
        # Compute stereo depth
        points_3d, valid_indices, depths = self.stereo_matcher.process_stereo_frame(
            keypoints_left, descriptors_left,
            keypoints_right, descriptors_right
        )
        
        # Store stereo data
        self.current_3d_points = points_3d
        self.current_depths = depths
        self.stereo_valid_indices = valid_indices
        
        # Create frame (uses left image)
        frame = Frame(
            id=self.frame_id,
            timestamp=timestamp,
            image=gray_left,
            keypoints=keypoints_left,
            descriptors=descriptors_left,
            pose=None
        )
        
        self.frame_id += 1
        
        # Process based on state
        if self.state == TrackingState.NOT_INITIALIZED:
            self._try_initialize_stereo(frame)
        
        elif self.state == TrackingState.OK:
            success = self._track_frame(frame)
            if not success:
                self.state = TrackingState.LOST
        
        elif self.state == TrackingState.LOST:
            # Try stereo initialization again
            self._try_initialize_stereo(frame)
        
        self.current_frame = frame
        
        return self.state, frame.pose
    
    def _try_initialize_stereo(self, frame: Frame) -> bool:
        """
        Initialize from stereo (instant, no motion required).
        
        Stereo gives us 3D points directly, so we can initialize
        immediately without waiting for parallax.
        """
        if self.current_3d_points is None or len(self.current_3d_points) < self.min_init_matches:
            return False
        
        # Set initial pose at origin
        frame.pose = SE3()
        
        # Store data for mapping
        self._stereo_init_3d_points = self.current_3d_points.copy()
        self._stereo_init_indices = self.stereo_valid_indices.copy()
        self._stereo_init_depths = self.current_depths.copy()
        
        self.state = TrackingState.OK
        self.last_frame = frame
        
        self.num_matches = len(self.current_3d_points)
        self.num_inliers = len(self.current_3d_points)
        
        return True
    
    def get_stereo_init_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get stereo initialization data for mapping.
        
        Returns:
            points_3d: Nx3 3D points in camera frame
            kp_indices: Indices of keypoints with valid depth
            depths: Depth values
        """
        if hasattr(self, '_stereo_init_3d_points'):
            return (self._stereo_init_3d_points,
                    self._stereo_init_indices,
                    self._stereo_init_depths)
        return np.empty((0, 3)), np.array([]), np.array([])
    
    def get_current_stereo_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D points from current stereo frame.
        
        Returns:
            points_3d: Nx3 3D points
            valid_indices: Indices of keypoints with valid depth
        """
        if self.current_3d_points is not None:
            return self.current_3d_points, self.stereo_valid_indices
        return np.empty((0, 3)), np.array([])

