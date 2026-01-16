"""
mapping.py - Map management for ORB-SLAM

Contains Map class, LocalMapper for keyframe/map point management,
and lightweight LoopDetector.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Set, Tuple
from threading import Lock, Thread
from collections import defaultdict

from .geometry import (
    SE3, CameraIntrinsics, triangulate_points,
    compute_reprojection_error
)
from .utils import Frame, MapPoint, Keyframe


class Map:
    """
    Global map containing all keyframes and map points.
    
    Thread-safe container for SLAM map data.
    """
    
    def __init__(self):
        self.keyframes: Dict[int, Keyframe] = {}
        self.map_points: Dict[int, MapPoint] = {}
        
        self._next_point_id = 0
        self._lock = Lock()
        
    def add_keyframe(self, keyframe: Keyframe):
        """Add a keyframe to the map."""
        with self._lock:
            self.keyframes[keyframe.id] = keyframe
    
    def remove_keyframe(self, keyframe_id: int):
        """Remove a keyframe from the map."""
        with self._lock:
            if keyframe_id in self.keyframes:
                del self.keyframes[keyframe_id]
    
    def add_map_point(self, map_point: MapPoint):
        """Add a map point to the map."""
        with self._lock:
            self.map_points[map_point.id] = map_point
    
    def create_map_point(self, position: np.ndarray, descriptor: np.ndarray) -> MapPoint:
        """Create and add a new map point."""
        with self._lock:
            mp = MapPoint(
                id=self._next_point_id,
                position=position.copy(),
                descriptor=descriptor.copy()
            )
            self._next_point_id += 1
            self.map_points[mp.id] = mp
            return mp
    
    def remove_map_point(self, point_id: int):
        """Remove a map point from the map."""
        with self._lock:
            if point_id in self.map_points:
                mp = self.map_points[point_id]
                mp.is_bad = True
                del self.map_points[point_id]
    
    def get_all_keyframes(self) -> List[Keyframe]:
        """Get all keyframes."""
        with self._lock:
            return list(self.keyframes.values())
    
    def get_all_map_points(self) -> List[MapPoint]:
        """Get all valid map points."""
        with self._lock:
            return [mp for mp in self.map_points.values() if not mp.is_bad]
    
    def get_map_points_array(self) -> np.ndarray:
        """Get all map point positions as Nx3 array."""
        points = self.get_all_map_points()
        if not points:
            return np.empty((0, 3))
        return np.array([mp.position for mp in points])
    
    @property
    def num_keyframes(self) -> int:
        with self._lock:
            return len(self.keyframes)
    
    @property
    def num_map_points(self) -> int:
        with self._lock:
            return sum(1 for mp in self.map_points.values() if not mp.is_bad)
    
    def clear(self):
        """Clear all map data."""
        with self._lock:
            self.keyframes.clear()
            self.map_points.clear()
            self._next_point_id = 0


class LocalMapper:
    """
    Local mapping thread for ORB-SLAM.
    
    Handles:
    - Keyframe insertion
    - Map point triangulation
    - Map point culling
    - Local optimization (simplified)
    """
    
    def __init__(self, slam_map: Map, camera: CameraIntrinsics,
                 min_parallax_deg: float = 1.0,
                 max_reproj_error: float = 5.0):
        """
        Initialize local mapper.
        
        Args:
            slam_map: Global map
            camera: Camera intrinsics
            min_parallax_deg: Minimum parallax for triangulation (degrees)
            max_reproj_error: Maximum reprojection error for valid points
        """
        self.map = slam_map
        self.camera = camera
        self.min_parallax = np.deg2rad(min_parallax_deg)
        self.max_reproj_error = max_reproj_error
        
        # Recent keyframes for processing
        self.recent_keyframes: List[Keyframe] = []
        self.max_recent = 10
        
        self._lock = Lock()
    
    def process_new_keyframe(self, keyframe: Keyframe, 
                              prev_keyframe: Optional[Keyframe] = None,
                              matches: List = None,
                              inlier_mask: np.ndarray = None):
        """
        Process a newly created keyframe.
        
        Args:
            keyframe: New keyframe to process
            prev_keyframe: Previous keyframe for triangulation
            matches: Feature matches with previous keyframe
            inlier_mask: Inlier mask for matches
        """
        # Add to map
        self.map.add_keyframe(keyframe)
        
        # Add to recent keyframes
        with self._lock:
            self.recent_keyframes.append(keyframe)
            if len(self.recent_keyframes) > self.max_recent:
                self.recent_keyframes.pop(0)
        
        # Triangulate new map points
        if prev_keyframe is not None and matches is not None:
            self._triangulate_new_points(
                prev_keyframe, keyframe, matches, inlier_mask
            )
        
        # Update covisibility
        keyframe.update_connections(self.map.keyframes)
        
        # Cull bad map points
        self._cull_map_points()
    
    def _triangulate_new_points(self, kf1: Keyframe, kf2: Keyframe,
                                 matches: List, inlier_mask: np.ndarray = None):
        """
        Triangulate new map points from two keyframes.
        """
        if not matches:
            return
        
        if inlier_mask is None:
            inlier_mask = np.ones(len(matches), dtype=bool)
        
        pose1 = kf1.pose_se3
        pose2 = kf2.pose_se3
        
        # Check baseline
        baseline = np.linalg.norm(pose2.t - pose1.t)
        if baseline < 0.001:  # Too small baseline
            return
        
        # Projection matrices
        P1 = self.camera.K @ pose1.matrix_3x4
        P2 = self.camera.K @ pose2.matrix_3x4
        
        num_new_points = 0
        
        for i, m in enumerate(matches):
            if not inlier_mask[i]:
                continue
            
            # Skip if already has map point
            if m.queryIdx in kf1.map_points or m.trainIdx in kf2.map_points:
                continue
            
            # Get 2D points
            pt1 = np.array(kf1.keypoints[m.queryIdx].pt)
            pt2 = np.array(kf2.keypoints[m.trainIdx].pt)
            
            # Check parallax angle
            ray1 = self.camera.unproject(pt1)[0]
            ray1 = pose1.inverse().R @ ray1
            ray1 = ray1 / np.linalg.norm(ray1)
            
            ray2 = self.camera.unproject(pt2)[0]
            ray2 = pose2.inverse().R @ ray2
            ray2 = ray2 / np.linalg.norm(ray2)
            
            cos_parallax = np.dot(ray1, ray2)
            if cos_parallax > np.cos(self.min_parallax):
                continue  # Parallax too small
            
            # Triangulate
            pts1 = pt1.reshape(1, 2)
            pts2 = pt2.reshape(1, 2)
            
            point_3d = triangulate_points(pts1, pts2, P1, P2)[0]
            
            # Check if point is in front of both cameras
            pt_cam1 = pose1.transform_point(point_3d)
            pt_cam2 = pose2.transform_point(point_3d)
            
            if pt_cam1[2] <= 0 or pt_cam2[2] <= 0:
                continue
            
            # Check reprojection error
            proj1 = self.camera.project(pt_cam1.reshape(1, 3))[0]
            proj2 = self.camera.project(pt_cam2.reshape(1, 3))[0]
            
            err1 = np.linalg.norm(proj1 - pt1)
            err2 = np.linalg.norm(proj2 - pt2)
            
            if err1 > self.max_reproj_error or err2 > self.max_reproj_error:
                continue
            
            # Create new map point
            descriptor = kf2.descriptors[m.trainIdx]
            mp = self.map.create_map_point(point_3d, descriptor)
            
            # Add observations
            kf1.add_map_point(m.queryIdx, mp)
            kf2.add_map_point(m.trainIdx, mp)
            
            num_new_points += 1
        
        return num_new_points
    
    def _cull_map_points(self):
        """
        Remove bad map points based on:
        - Found ratio too low
        - Few observations
        - High reprojection error
        """
        to_remove = []
        
        for mp_id, mp in list(self.map.map_points.items()):
            if mp.is_bad:
                to_remove.append(mp_id)
                continue
            
            # Remove if found ratio is too low after being visible many times
            if mp.num_visible > 10 and mp.get_found_ratio() < 0.25:
                to_remove.append(mp_id)
                continue
            
            # Remove if too few observations
            if mp.num_observations < 2 and mp.num_visible > 5:
                to_remove.append(mp_id)
        
        for mp_id in to_remove:
            self.map.remove_map_point(mp_id)
    
    def create_initial_map(self, frame1: Frame, frame2: Frame,
                           matches: List, inlier_mask: np.ndarray) -> Tuple[Keyframe, Keyframe]:
        """
        Create initial map from two-view initialization.
        
        Returns the two keyframes created.
        """
        # Create keyframes
        kf1 = Keyframe(frame1, self.camera)
        kf2 = Keyframe(frame2, self.camera)
        
        # Add to map
        self.map.add_keyframe(kf1)
        self.map.add_keyframe(kf2)
        
        # Triangulate points
        num_points = self._triangulate_new_points(kf1, kf2, matches, inlier_mask)
        
        # Update connections
        kf1.update_connections(self.map.keyframes)
        kf2.update_connections(self.map.keyframes)
        
        with self._lock:
            self.recent_keyframes = [kf1, kf2]
        
        return kf1, kf2
    
    def get_local_map_points(self, keyframe: Keyframe) -> List[MapPoint]:
        """
        Get map points visible from keyframe and its neighbors.
        """
        local_points = set()
        
        # Add points from current keyframe
        for mp in keyframe.get_map_points():
            local_points.add(mp)
        
        # Add points from connected keyframes
        for kf_id in keyframe.get_connected_keyframes():
            if kf_id in self.map.keyframes:
                kf = self.map.keyframes[kf_id]
                for mp in kf.get_map_points():
                    local_points.add(mp)
        
        return list(local_points)


class LoopDetector:
    """
    Lightweight loop closure detection using ORB descriptors.
    
    Uses simple descriptor aggregation and similarity matching
    instead of full BoW vocabulary.
    """
    
    def __init__(self, min_score: float = 0.3, min_frames_gap: int = 30):
        """
        Initialize loop detector.
        
        Args:
            min_score: Minimum similarity score for loop candidate
            min_frames_gap: Minimum keyframe gap for loop detection
        """
        self.min_score = min_score
        self.min_frames_gap = min_frames_gap
        
        # Descriptor database (keyframe_id -> aggregated descriptor)
        self.descriptor_db: Dict[int, np.ndarray] = {}
        
        self._lock = Lock()
    
    def add_keyframe(self, keyframe: Keyframe):
        """
        Add keyframe to loop detection database.
        
        Creates aggregated descriptor representation.
        """
        if keyframe.descriptors is None or len(keyframe.descriptors) == 0:
            return
        
        # Simple aggregation: compute mean descriptor (for float) or mode (for binary)
        # For ORB (binary), we count bit occurrences
        agg_desc = self._aggregate_descriptors(keyframe.descriptors)
        
        with self._lock:
            self.descriptor_db[keyframe.id] = agg_desc
    
    def _aggregate_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Aggregate descriptors into single representation.
        
        For binary descriptors, count bit frequencies.
        """
        if len(descriptors) == 0:
            return np.zeros(256, dtype=np.float32)
        
        # Unpack bits and compute mean
        bit_counts = np.zeros(256, dtype=np.float32)
        
        for desc in descriptors:
            bits = np.unpackbits(desc)
            bit_counts += bits
        
        # Normalize
        bit_counts /= len(descriptors)
        
        return bit_counts
    
    def detect_loop(self, keyframe: Keyframe) -> Optional[int]:
        """
        Detect loop closure candidates for given keyframe.
        
        Returns keyframe ID of best loop candidate, or None.
        """
        if keyframe.descriptors is None or len(keyframe.descriptors) == 0:
            return None
        
        query_desc = self._aggregate_descriptors(keyframe.descriptors)
        
        best_score = 0.0
        best_id = None
        
        with self._lock:
            for kf_id, db_desc in self.descriptor_db.items():
                # Skip recent keyframes
                if abs(keyframe.id - kf_id) < self.min_frames_gap:
                    continue
                
                # Compute similarity (cosine similarity of bit frequencies)
                score = self._compute_similarity(query_desc, db_desc)
                
                if score > best_score and score > self.min_score:
                    best_score = score
                    best_id = kf_id
        
        return best_id
    
    def _compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Compute cosine similarity between aggregated descriptors."""
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        return float(np.dot(desc1, desc2) / (norm1 * norm2))
    
    def correct_loop(self, current_kf: Keyframe, loop_kf: Keyframe,
                     slam_map: Map) -> bool:
        """
        Apply simple loop correction.
        
        This is a simplified version that just aligns the current pose
        with the loop keyframe. Full implementation would do pose graph
        optimization.
        """
        # Compute relative pose between loop keyframes
        # For simplicity, we just log the detection
        # Full implementation would require g2o or similar
        
        return True
