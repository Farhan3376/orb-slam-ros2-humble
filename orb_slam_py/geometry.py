"""
geometry.py - Mathematical utilities for ORB-SLAM

Contains SE(3) pose representation, camera projection, and 
triangulation functions.
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (x, y, z, w).
    
    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R).reshape(3, 3)
    
    # Compute trace
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    # Return as (x, y, z, w) to match scipy convention
    return np.array([x, y, z, w])


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    """
    x, y, z, w = quat
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    return R


class SE3:
    """
    SE(3) pose representation using 4x4 homogeneous transformation matrix.
    
    Represents a rigid body transformation in 3D space:
    T = [R | t]
        [0 | 1]
    
    Where R is a 3x3 rotation matrix (SO(3)) and t is a 3x1 translation vector.
    """
    
    def __init__(self, R: np.ndarray = None, t: np.ndarray = None):
        """
        Initialize SE(3) pose.
        
        Args:
            R: 3x3 rotation matrix (default: identity)
            t: 3x1 translation vector (default: zero)
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)
        
        self._R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self._t = np.asarray(t, dtype=np.float64).reshape(3)
        
    @property
    def R(self) -> np.ndarray:
        """3x3 rotation matrix"""
        return self._R
    
    @property
    def t(self) -> np.ndarray:
        """3x1 translation vector"""
        return self._t
    
    @property
    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self._R
        T[:3, 3] = self._t
        return T
    
    @property
    def matrix_3x4(self) -> np.ndarray:
        """3x4 projection matrix [R|t]"""
        return np.hstack([self._R, self._t.reshape(3, 1)])
    
    def inverse(self) -> 'SE3':
        """
        Compute the inverse transformation.
        T^-1 = [R^T | -R^T * t]
        """
        R_inv = self._R.T
        t_inv = -R_inv @ self._t
        return SE3(R_inv, t_inv)
    
    def __matmul__(self, other: 'SE3') -> 'SE3':
        """
        Compose two SE(3) transformations: T1 @ T2
        """
        R_new = self._R @ other._R
        t_new = self._R @ other._t + self._t
        return SE3(R_new, t_new)
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point: p' = R * p + t
        
        Args:
            point: 3D point (3,) or points (N, 3)
        Returns:
            Transformed point(s)
        """
        point = np.asarray(point)
        if point.ndim == 1:
            return self._R @ point + self._t
        else:
            return (self._R @ point.T).T + self._t
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple 3D points.
        
        Args:
            points: Nx3 array of 3D points
        Returns:
            Nx3 array of transformed points
        """
        return (self._R @ points.T).T + self._t
    
    @classmethod
    def from_matrix(cls, T: np.ndarray) -> 'SE3':
        """Create SE3 from 4x4 homogeneous matrix"""
        T = np.asarray(T)
        return cls(T[:3, :3], T[:3, 3])
    
    @classmethod
    def from_quaternion(cls, quat: np.ndarray, t: np.ndarray) -> 'SE3':
        """
        Create SE3 from quaternion (x, y, z, w) and translation.
        """
        R = quaternion_to_rotation_matrix(quat)
        return cls(R, t)
    
    def to_quaternion(self) -> np.ndarray:
        """
        Convert rotation to quaternion (x, y, z, w).
        """
        return rotation_matrix_to_quaternion(self._R)
    
    def to_tum_format(self, timestamp: float) -> str:
        """
        Convert to TUM trajectory format:
        timestamp tx ty tz qx qy qz qw
        """
        quat = self.to_quaternion()
        return f"{timestamp:.6f} {self._t[0]:.6f} {self._t[1]:.6f} {self._t[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
    
    def __repr__(self) -> str:
        return f"SE3(t={self._t}, R=\n{self._R})"



class CameraIntrinsics:
    """
    Camera intrinsic parameters (pinhole model).
    
    K = [fx  0  cx]
        [0  fy  cy]
        [0   0   1]
    """
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int = 640, height: int = 480,
                 dist_coeffs: np.ndarray = None):
        """
        Initialize camera intrinsics.
        
        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point
            width, height: Image dimensions
            dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def K_inv(self) -> np.ndarray:
        """Inverse of camera intrinsic matrix"""
        return np.linalg.inv(self.K)
    
    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points (in camera frame) to 2D image coordinates.
        
        Args:
            points_3d: Nx3 array of 3D points in camera frame
        Returns:
            Nx2 array of 2D image coordinates
        """
        points_3d = np.asarray(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
        
        # Perspective division
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        
        # Apply intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        
        return np.column_stack([u, v])
    
    def unproject(self, points_2d: np.ndarray, depth: np.ndarray = None) -> np.ndarray:
        """
        Unproject 2D points to normalized camera coordinates (or 3D if depth given).
        
        Args:
            points_2d: Nx2 array of 2D image coordinates
            depth: Optional Nx1 array of depths
        Returns:
            Nx3 array of 3D points (z=1 if no depth)
        """
        points_2d = np.asarray(points_2d)
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, 2)
        
        # Remove intrinsics
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy
        z = np.ones(len(points_2d))
        
        points_3d = np.column_stack([x, y, z])
        
        if depth is not None:
            depth = np.asarray(depth).flatten()
            points_3d = points_3d * depth.reshape(-1, 1)
        
        return points_3d
    
    def is_in_frame(self, points_2d: np.ndarray, margin: int = 0) -> np.ndarray:
        """Check if 2D points are within image bounds."""
        points_2d = np.asarray(points_2d)
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, 2)
        
        valid = (
            (points_2d[:, 0] >= margin) &
            (points_2d[:, 0] < self.width - margin) &
            (points_2d[:, 1] >= margin) &
            (points_2d[:, 1] < self.height - margin)
        )
        return valid
    
    @classmethod
    def from_camera_info(cls, msg) -> 'CameraIntrinsics':
        """Create from sensor_msgs/CameraInfo message."""
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        return cls(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
            width=msg.width, height=msg.height,
            dist_coeffs=D
        )
    
    def __repr__(self) -> str:
        return f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f})"


def compute_essential_matrix(pts1: np.ndarray, pts2: np.ndarray,
                             K: np.ndarray, 
                             method: int = cv2.RANSAC,
                             threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Essential matrix from point correspondences.
    
    The Essential matrix E encodes the relative pose between two views:
    x2^T @ E @ x1 = 0
    
    Args:
        pts1: Nx2 points in first image
        pts2: Nx2 corresponding points in second image  
        K: 3x3 camera intrinsic matrix
        method: RANSAC or LMEDS
        threshold: RANSAC threshold in pixels
    
    Returns:
        E: 3x3 Essential matrix
        mask: Inlier mask (N,)
    """
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=method,
        prob=0.999,
        threshold=threshold
    )
    return E, mask.ravel().astype(bool) if mask is not None else None


def recover_pose_from_essential(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                                 K: np.ndarray, mask: np.ndarray = None) -> Tuple[SE3, np.ndarray]:
    """
    Recover relative pose (R, t) from Essential matrix.
    
    The Essential matrix has 4 possible decompositions, but only one
    gives points in front of both cameras (positive depth).
    
    Args:
        E: 3x3 Essential matrix
        pts1, pts2: Corresponding points
        K: Camera intrinsic matrix
        mask: Optional inlier mask
    
    Returns:
        pose: SE3 relative pose (camera 2 w.r.t camera 1)
        mask: Updated inlier mask (points with positive depth)
    """
    if mask is None:
        mask = np.ones(len(pts1), dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask.copy())
    
    return SE3(R, t.ravel()), pose_mask.ravel().astype(bool)


def triangulate_points(pts1: np.ndarray, pts2: np.ndarray,
                       P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from two views.
    
    Uses DLT (Direct Linear Transform) method via cv2.triangulatePoints.
    
    Args:
        pts1: Nx2 points in first image
        pts2: Nx2 corresponding points in second image
        P1: 3x4 projection matrix for camera 1 (K @ [R|t])
        P2: 3x4 projection matrix for camera 2
    
    Returns:
        points_3d: Nx3 array of triangulated 3D points
    """
    pts1 = pts1.T.astype(np.float64)  # 2xN
    pts2 = pts2.T.astype(np.float64)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)  # 4xN
    
    # Convert from homogeneous
    points_3d = points_4d[:3, :] / points_4d[3:, :]
    
    return points_3d.T  # Nx3


def compute_reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray,
                                pose: SE3, K: np.ndarray) -> np.ndarray:
    """
    Compute reprojection error for 3D-2D correspondences.
    
    Error = ||project(T @ P_3d) - P_2d||
    
    Args:
        points_3d: Nx3 3D points in world frame
        points_2d: Nx2 observed 2D points
        pose: SE3 camera pose (world to camera)
        K: Camera intrinsic matrix
    
    Returns:
        errors: N reprojection errors (Euclidean distance in pixels)
    """
    # Transform to camera frame
    points_cam = pose.transform_points(points_3d)
    
    # Project to image
    projected = cv2.projectPoints(
        points_cam.reshape(-1, 1, 3),
        np.zeros(3),  # Already in camera frame
        np.zeros(3),
        K,
        None
    )[0].reshape(-1, 2)
    
    # Compute error
    errors = np.linalg.norm(projected - points_2d, axis=1)
    
    return errors


def solve_pnp_ransac(points_3d: np.ndarray, points_2d: np.ndarray,
                     K: np.ndarray, dist_coeffs: np.ndarray = None,
                     initial_pose: SE3 = None) -> Tuple[Optional[SE3], np.ndarray]:
    """
    Solve PnP problem with RANSAC.
    
    Given 3D-2D correspondences, find the camera pose.
    
    Args:
        points_3d: Nx3 3D points in world frame
        points_2d: Nx2 2D observations
        K: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        initial_pose: Initial pose estimate for refinement
    
    Returns:
        pose: SE3 camera pose (world to camera), or None if failed
        inliers: Boolean mask of inliers
    """
    if len(points_3d) < 4:
        return None, np.zeros(len(points_3d), dtype=bool)
    
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    # Initial guess
    use_extrinsic_guess = initial_pose is not None
    if use_extrinsic_guess:
        rvec, _ = cv2.Rodrigues(initial_pose.R)
        tvec = initial_pose.t.reshape(3, 1)
    else:
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.reshape(-1, 1, 3).astype(np.float64),
        points_2d.reshape(-1, 1, 2).astype(np.float64),
        K.astype(np.float64),
        dist_coeffs.astype(np.float64),
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=use_extrinsic_guess,
        iterationsCount=100,
        reprojectionError=4.0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success or inliers is None:
        return None, np.zeros(len(points_3d), dtype=bool)
    
    R, _ = cv2.Rodrigues(rvec)
    pose = SE3(R, tvec.ravel())
    
    inlier_mask = np.zeros(len(points_3d), dtype=bool)
    inlier_mask[inliers.ravel()] = True
    
    return pose, inlier_mask


def compute_parallax(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """
    Compute median parallax (in pixels) between two sets of points.
    
    Parallax is a measure of how much points have moved between views,
    which indicates baseline for triangulation quality.
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0
    
    displacement = np.linalg.norm(pts1 - pts2, axis=1)
    return float(np.median(displacement))


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize points for numerical stability.
    
    Returns normalized points and the normalization matrix T such that:
    points_normalized = T @ points_homogeneous
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(2) / (mean_dist + 1e-8)
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    normalized = centered * scale
    
    return normalized, T
