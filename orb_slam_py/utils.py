"""
utils.py - Helper classes and data structures for ORB-SLAM

Contains Frame, MapPoint, Keyframe classes and utility functions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import cv2
from threading import Lock
import time
import os

from .geometry import SE3, CameraIntrinsics


@dataclass
class Frame:
    """
    Represents a single camera frame with extracted features.
    """
    id: int
    timestamp: float
    image: np.ndarray  # Grayscale image
    keypoints: tuple  # cv2.KeyPoint tuple
    descriptors: np.ndarray  # Nx32 ORB descriptors
    pose: Optional[SE3] = None  # Camera pose (world to camera)
    
    @property
    def num_features(self) -> int:
        return len(self.keypoints) if self.keypoints else 0
    
    @property
    def keypoints_array(self) -> np.ndarray:
        """Get keypoint coordinates as Nx2 array"""
        if not self.keypoints:
            return np.empty((0, 2))
        return np.array([kp.pt for kp in self.keypoints])
    
    def get_keypoint(self, idx: int) -> np.ndarray:
        """Get single keypoint coordinate"""
        return np.array(self.keypoints[idx].pt)


@dataclass  
class MapPoint:
    """
    Represents a 3D map point with associated descriptor and observations.
    
    Attributes:
        id: Unique identifier
        position: 3D position in world frame
        descriptor: Representative ORB descriptor (32 bytes)
        observations: Dict mapping frame_id to keypoint_idx
        num_visible: Times point was visible in tracking
        num_found: Times point was successfully matched
    """
    id: int
    position: np.ndarray  # 3D position in world frame
    descriptor: np.ndarray  # 32-byte ORB descriptor
    observations: Dict[int, int] = field(default_factory=dict)  # frame_id -> kp_idx
    num_visible: int = 0
    num_found: int = 0
    is_bad: bool = False
    
    def add_observation(self, frame_id: int, keypoint_idx: int):
        """Add an observation of this point in a frame."""
        self.observations[frame_id] = keypoint_idx
    
    def remove_observation(self, frame_id: int):
        """Remove observation from a frame."""
        if frame_id in self.observations:
            del self.observations[frame_id]
    
    @property
    def num_observations(self) -> int:
        return len(self.observations)
    
    def get_found_ratio(self) -> float:
        """Ratio of times point was matched vs visible."""
        if self.num_visible == 0:
            return 0.0
        return self.num_found / self.num_visible
    
    def update_descriptor(self, descriptors: np.ndarray):
        """Update descriptor to be the one with minimum median distance."""
        if len(descriptors) == 0:
            return
        
        if len(descriptors) == 1:
            self.descriptor = descriptors[0]
            return
        
        # Compute median distance to all other descriptors
        distances = []
        for i, desc in enumerate(descriptors):
            dists = [cv2.norm(desc, d, cv2.NORM_HAMMING) 
                     for j, d in enumerate(descriptors) if i != j]
            distances.append(np.median(dists))
        
        # Choose descriptor with minimum median distance
        best_idx = np.argmin(distances)
        self.descriptor = descriptors[best_idx]


class Keyframe:
    """
    A keyframe in the SLAM map with associated map points.
    
    Keyframes are selected frames that anchor the map and are used
    for loop closure detection.
    """
    
    def __init__(self, frame: Frame, camera: CameraIntrinsics):
        self.id = frame.id
        self.timestamp = frame.timestamp
        self.image = frame.image
        self.keypoints = frame.keypoints
        self.descriptors = frame.descriptors
        self.pose = frame.pose.matrix.copy() if frame.pose else np.eye(4)
        self.camera = camera
        
        # Map point associations (keypoint_idx -> MapPoint)
        self.map_points: Dict[int, MapPoint] = {}
        
        # Covisibility graph connections
        self.connected_keyframes: Dict[int, int] = {}  # kf_id -> num_shared_points
        
        # BoW representation (for loop closure)
        self.bow_vector: Optional[np.ndarray] = None
        
        self._lock = Lock()
    
    @property
    def pose_se3(self) -> SE3:
        """Get pose as SE3 object."""
        return SE3.from_matrix(self.pose)
    
    def set_pose(self, pose: SE3):
        """Set keyframe pose."""
        with self._lock:
            self.pose = pose.matrix.copy()
    
    def add_map_point(self, keypoint_idx: int, map_point: MapPoint):
        """Associate a map point with a keypoint."""
        with self._lock:
            self.map_points[keypoint_idx] = map_point
            map_point.add_observation(self.id, keypoint_idx)
    
    def remove_map_point(self, keypoint_idx: int):
        """Remove map point association."""
        with self._lock:
            if keypoint_idx in self.map_points:
                mp = self.map_points[keypoint_idx]
                mp.remove_observation(self.id)
                del self.map_points[keypoint_idx]
    
    def get_map_points(self) -> List[MapPoint]:
        """Get all associated map points."""
        with self._lock:
            return [mp for mp in self.map_points.values() if not mp.is_bad]
    
    def get_map_point_indices(self) -> Dict[int, MapPoint]:
        """Get keypoint indices to map points mapping."""
        with self._lock:
            return {k: v for k, v in self.map_points.items() if not v.is_bad}
    
    @property  
    def keypoints_array(self) -> np.ndarray:
        """Get keypoint coordinates as Nx2 array"""
        if not self.keypoints:
            return np.empty((0, 2))
        return np.array([kp.pt for kp in self.keypoints])
    
    def update_connections(self, keyframes: Dict[int, 'Keyframe']):
        """Update covisibility connections with other keyframes."""
        with self._lock:
            self.connected_keyframes.clear()
            
            # Count shared map points with each keyframe
            my_points = set(id(mp) for mp in self.map_points.values() if not mp.is_bad)
            
            for kf_id, kf in keyframes.items():
                if kf_id == self.id:
                    continue
                    
                other_points = set(id(mp) for mp in kf.map_points.values() if not mp.is_bad)
                shared = len(my_points & other_points)
                
                if shared >= 15:  # Minimum shared points for connection
                    self.connected_keyframes[kf_id] = shared
    
    def get_connected_keyframes(self, min_weight: int = 15) -> List[int]:
        """Get IDs of connected keyframes with at least min_weight shared points."""
        with self._lock:
            return [kf_id for kf_id, weight in self.connected_keyframes.items() 
                    if weight >= min_weight]


class TUMLogger:
    """
    Logger for TUM trajectory format.
    
    Format: timestamp tx ty tz qx qy qz qw
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        
    def open(self):
        """Open log file for writing."""
        os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)
        self.file = open(self.filepath, 'w')
        self.file.write("# TUM trajectory format: timestamp tx ty tz qx qy qz qw\n")
        
    def log(self, timestamp: float, pose: SE3):
        """Log a pose."""
        if self.file is None:
            self.open()
        
        line = pose.to_tum_format(timestamp)
        self.file.write(line + "\n")
        self.file.flush()
        
    def close(self):
        """Close log file."""
        if self.file is not None:
            self.file.close()
            self.file = None


class Timer:
    """Simple timer for profiling."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
        
    def start(self):
        self.start_time = time.perf_counter()
        
    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def hamming_distance(desc1: np.ndarray, desc2: np.ndarray) -> int:
    """Compute Hamming distance between two ORB descriptors."""
    return cv2.norm(desc1, desc2, cv2.NORM_HAMMING)


def compute_descriptor_distance_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """
    Compute distance matrix between two sets of descriptors.
    
    Args:
        desc1: NxD descriptors
        desc2: MxD descriptors
    
    Returns:
        NxM distance matrix
    """
    n = len(desc1)
    m = len(desc2)
    
    distances = np.zeros((n, m), dtype=np.int32)
    
    for i in range(n):
        for j in range(m):
            distances[i, j] = cv2.norm(desc1[i], desc2[j], cv2.NORM_HAMMING)
    
    return distances


def draw_keypoints(image: np.ndarray, keypoints: tuple, 
                   color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw keypoints on image."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    return cv2.drawKeypoints(vis, keypoints, None, color=color,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches(img1: np.ndarray, kp1: tuple, 
                 img2: np.ndarray, kp2: tuple,
                 matches: List, mask: np.ndarray = None) -> np.ndarray:
    """Draw feature matches between two images."""
    if mask is not None:
        matches = [m for m, valid in zip(matches, mask) if valid]
    
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
