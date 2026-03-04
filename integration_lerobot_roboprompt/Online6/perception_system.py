"""
Perception System for RoboPrompt Inference

This handles REAL-TIME object detection at inference time.
This is separate from demo creation where object poses can be approximate!
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class ObjectPose:
    """3D pose of a detected object."""
    name: str
    position: np.ndarray  # [x, y, z] in meters
    rotation: np.ndarray  # [rx, ry, rz] in degrees
    confidence: float


class PerceptionSystem:
    """
    Base class for perception systems.
    Subclass this for different detection methods (ArUco, YOLO, SAM, etc.)
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        camera_to_robot_transform: np.ndarray = None
    ):
        """
        Args:
            camera_id: Camera device index
            camera_to_robot_transform: 4x4 transform from camera to robot base
        """
        self.camera_id = camera_id
        
        # Default transform: camera 40cm in front, 50cm above robot base
        if camera_to_robot_transform is None:
            self.camera_to_robot = np.array([
                [1, 0, 0, 0.4],   # 40cm forward
                [0, 1, 0, 0.0],   # centered
                [0, 0, 1, 0.5],   # 50cm up
                [0, 0, 0, 1]
            ])
        else:
            self.camera_to_robot = camera_to_robot_transform
    
    def detect_objects(self, rgb_image: Optional[np.ndarray] = None) -> Dict[str, ObjectPose]:
        """
        Detect objects in scene. Override in subclasses.
        
        Args:
            rgb_image: RGB image [H, W, 3]. If None, captures new frame.
        
        Returns:
            Dict of {object_name: ObjectPose}
        """
        raise NotImplementedError("Subclass must implement detect_objects()")
    
    def discretize_poses(
        self,
        detected_objects: Dict[str, ObjectPose],
        scene_bounds: np.ndarray
    ) -> Dict[str, List[int]]:
        """
        Discretize detected object poses into RoboPrompt bins.
        
        Args:
            detected_objects: Objects from detect_objects()
            scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
        
        Returns:
            Dict of {object_name: [x_bin, y_bin, z_bin]} (0-99)
        """
        discretized = {}
        
        for name, obj_pose in detected_objects.items():
            position = obj_pose.position
            
            # Normalize to [0, 1]
            pos_normalized = (position - scene_bounds[:3]) / (scene_bounds[3:] - scene_bounds[:3])
            
            # Discretize to bins (0-99)
            pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
            
            discretized[name] = pos_bins.tolist()
        
        return discretized


class ArucoPerception(PerceptionSystem):
    """
    ArUco marker-based perception.
    Simplest and most reliable for controlled environments.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        marker_size: float = 0.05,  # 5cm markers
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        marker_to_object_names: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        super().__init__(camera_id, **kwargs)
        
        self.marker_size = marker_size
        
        # Camera intrinsics (use calibration for better accuracy)
        if camera_matrix is None:
            # Default for typical webcam (640x480)
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
        
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5, dtype=np.float32)
        else:
            self.dist_coeffs = dist_coeffs
        
        # Marker ID to object name mapping
        if marker_to_object_names is None:
            self.marker_to_object = {
                0: 'tape',
                1: 'target_zone',
                2: 'table',
                3: 'obstacle',
                4: 'container'
            }
        else:
            self.marker_to_object = marker_to_object_names
        
        # ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Open camera
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        print(f"✓ ArUco perception initialized")
        print(f"  Marker size: {marker_size}m")
        print(f"  Marker mappings: {self.marker_to_object}")
    
    def capture_frame(self) -> np.ndarray:
        """Capture RGB frame from camera."""
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return frame
    
    def detect_objects(
        self,
        rgb_image: Optional[np.ndarray] = None,
        visualize: bool = False
    ) -> Dict[str, ObjectPose]:
        """
        Detect objects using ArUco markers.
        
        Args:
            rgb_image: Input image. If None, captures new frame.
            visualize: Show detection visualization
        
        Returns:
            Dict of detected objects
        """
        # Capture or use provided image
        if rgb_image is None:
            frame = self.capture_frame()
        else:
            frame = rgb_image.copy()
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params
        )
        
        detected_objects = {}
        
        if ids is not None:
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            for i, marker_id in enumerate(ids.flatten()):
                # Get marker pose in camera frame
                tvec = tvecs[i][0]
                rvec = rvecs[i][0]
                
                # Transform to robot frame
                marker_in_camera = np.eye(4)
                marker_in_camera[:3, :3] = cv2.Rodrigues(rvec)[0]
                marker_in_camera[:3, 3] = tvec
                
                marker_in_robot = self.camera_to_robot @ marker_in_camera
                
                # Extract position and rotation
                position = marker_in_robot[:3, 3]
                rotation_matrix = marker_in_robot[:3, :3]
                rotation = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
                
                # Get object name
                object_name = self.marker_to_object.get(marker_id, f'marker_{marker_id}')
                
                # Create ObjectPose
                obj_pose = ObjectPose(
                    name=object_name,
                    position=position,
                    rotation=rotation,
                    confidence=1.0  # ArUco markers have high confidence
                )
                
                detected_objects[object_name] = obj_pose
                
                # Visualize
                if visualize:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                    rvec, tvec, self.marker_size * 0.5)
        
        if visualize:
            # Add detection info
            y_offset = 30
            for name, pose in detected_objects.items():
                text = f"{name}: [{pose.position[0]:.2f}, {pose.position[1]:.2f}, {pose.position[2]:.2f}]"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow('ArUco Detection', frame)
            cv2.waitKey(1)
        
        return detected_objects
    
    def close(self):
        """Release resources."""
        self.camera.release()
        cv2.destroyAllWindows()


class DepthCameraPerception(PerceptionSystem):
    """
    Depth camera based perception (RealSense, etc.)
    More general than ArUco but requires depth camera.
    """
    
    def __init__(self, camera_id: int = 0, **kwargs):
        super().__init__(camera_id, **kwargs)
        
        try:
            import pyrealsense2 as rs
            self.rs = rs
            
            # Configure RealSense pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            self.pipeline.start(config)
            
            print("✓ RealSense depth camera initialized")
            
        except ImportError:
            raise ImportError("pyrealsense2 required. Install with: pip install pyrealsense2")
    
    def detect_objects(
        self,
        rgb_image: Optional[np.ndarray] = None,
        visualize: bool = False
    ) -> Dict[str, ObjectPose]:
        """
        Detect objects using depth camera + segmentation.
        
        TODO: Implement with YOLO/SAM for object detection
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return {}
        
        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # TODO: Implement object detection (YOLO, SAM, etc.)
        # For now, return empty dict
        print("⚠️  Depth camera perception not fully implemented yet")
        print("    Use ArUco markers or implement YOLO/SAM detection")
        
        return {}
    
    def close(self):
        """Release resources."""
        self.pipeline.stop()


def test_perception():
    """Test perception system standalone."""
    
    print("="*80)
    print("TESTING PERCEPTION SYSTEM")
    print("="*80)
    print("\nPlace ArUco markers in camera view")
    print("Press 'q' to quit, 's' to save detection\n")
    
    # Initialize ArUco perception
    perception = ArucoPerception(
        camera_id=0,
        marker_size=0.05,  # 5cm markers
        marker_to_object_names={
            0: 'tape',
            1: 'target_zone',
            2: 'table'
        }
    )
    
    # Scene bounds for discretization
    scene_bounds = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
    
    try:
        while True:
            # Detect objects
            detected = perception.detect_objects(visualize=True)
            
            if detected:
                print(f"\n📍 Detected {len(detected)} objects:")
                for name, pose in detected.items():
                    print(f"  {name}:")
                    print(f"    Position: [{pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}]")
                    print(f"    Rotation: [{pose.rotation[0]:.1f}°, {pose.rotation[1]:.1f}°, {pose.rotation[2]:.1f}°]")
                
                # Show discretized poses
                discretized = perception.discretize_poses(detected, scene_bounds)
                print(f"\n🎯 Discretized (RoboPrompt format):")
                for name, bins in discretized.items():
                    print(f"  '{name}': {bins}")
            
            # Check for key press
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current detection
                import json
                timestamp = int(time.time())
                with open(f'detection_{timestamp}.json', 'w') as f:
                    detection_data = {
                        name: {
                            'position': pose.position.tolist(),
                            'rotation': pose.rotation.tolist(),
                            'confidence': pose.confidence
                        }
                        for name, pose in detected.items()
                    }
                    json.dump(detection_data, f, indent=2)
                print(f"\n💾 Saved detection to detection_{timestamp}.json")
    
    finally:
        perception.close()
        print("\n✓ Test complete!")


if __name__ == "__main__":
    import time
    test_perception()
