"""
Action Discretization Module
Converts continuous 6-DoF poses to discrete bins as per RoboPrompt paper.
"""

import numpy as np
from typing import Tuple, List, Dict
import yaml
from pathlib import Path


class ActionDiscretizer:
    """
    Discretizes continuous 6-DoF poses (x, y, z, roll, pitch, yaw) into discrete bins.
    
    Following RoboPrompt paper:
    - Translation: 100 bins per dimension
    - Rotation: 72 bins per dimension (5° each)
    - Gripper: Binary (open/closed)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize discretizer with workspace and rotation limits.
        
        Args:
            config: Configuration dictionary with discretization parameters
        """
        disc_config = config['discretization']
        
        # Number of bins
        self.translation_bins = disc_config['translation_bins']
        self.rotation_bins = disc_config['rotation_bins']
        
        # Workspace limits
        self.x_range = np.array(disc_config['x_range'])
        self.y_range = np.array(disc_config['y_range'])
        self.z_range = np.array(disc_config['z_range'])
        
        # Rotation limits (radians)
        self.roll_range = np.array(disc_config['roll_range'])
        self.pitch_range = np.array(disc_config['pitch_range'])
        self.yaw_range = np.array(disc_config['yaw_range'])
        
        # Compute bin edges
        self.x_bins = np.linspace(self.x_range[0], self.x_range[1], self.translation_bins + 1)
        self.y_bins = np.linspace(self.y_range[0], self.y_range[1], self.translation_bins + 1)
        self.z_bins = np.linspace(self.z_range[0], self.z_range[1], self.translation_bins + 1)
        
        self.roll_bins = np.linspace(self.roll_range[0], self.roll_range[1], self.rotation_bins + 1)
        self.pitch_bins = np.linspace(self.pitch_range[0], self.pitch_range[1], self.rotation_bins + 1)
        self.yaw_bins = np.linspace(self.yaw_range[0], self.yaw_range[1], self.rotation_bins + 1)
        
        # Gripper threshold (if gripper > threshold, considered "closed")
        self.gripper_threshold = 0.5
        
        print(f"Initialized ActionDiscretizer:")
        print(f"  Translation bins: {self.translation_bins}")
        print(f"  Rotation bins: {self.rotation_bins}")
        print(f"  Workspace: x={self.x_range}, y={self.y_range}, z={self.z_range}")
    
    def discretize_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Discretize a continuous 6-DoF pose into discrete bins.
        
        Args:
            pose: Continuous pose [x, y, z, roll, pitch, yaw] or [N, 6]
            
        Returns:
            discrete_pose: Discretized pose with bin indices [6,] or [N, 6]
        """
        single_pose = False
        if pose.ndim == 1:
            pose = pose.reshape(1, -1)
            single_pose = True
        
        # Extract components
        x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
        roll, pitch, yaw = pose[:, 3], pose[:, 4], pose[:, 5]
        
        # Discretize each component
        x_discrete = np.digitize(x, self.x_bins) - 1
        y_discrete = np.digitize(y, self.y_bins) - 1
        z_discrete = np.digitize(z, self.z_bins) - 1
        
        roll_discrete = np.digitize(roll, self.roll_bins) - 1
        pitch_discrete = np.digitize(pitch, self.pitch_bins) - 1
        yaw_discrete = np.digitize(yaw, self.yaw_bins) - 1
        
        # Clip to valid range (handle edge cases)
        x_discrete = np.clip(x_discrete, 0, self.translation_bins - 1)
        y_discrete = np.clip(y_discrete, 0, self.translation_bins - 1)
        z_discrete = np.clip(z_discrete, 0, self.translation_bins - 1)
        
        roll_discrete = np.clip(roll_discrete, 0, self.rotation_bins - 1)
        pitch_discrete = np.clip(pitch_discrete, 0, self.rotation_bins - 1)
        yaw_discrete = np.clip(yaw_discrete, 0, self.rotation_bins - 1)
        
        # Stack
        discrete_pose = np.stack([
            x_discrete, y_discrete, z_discrete,
            roll_discrete, pitch_discrete, yaw_discrete
        ], axis=1)
        
        if single_pose:
            return discrete_pose[0].astype(int)
        return discrete_pose.astype(int)
    
    def discretize_gripper(self, gripper_values: np.ndarray) -> np.ndarray:
        """
        Discretize gripper values to binary (0 = open, 1 = closed).
        
        Args:
            gripper_values: Continuous gripper values [N,]
            
        Returns:
            discrete_gripper: Binary gripper values [N,]
        """
        return (gripper_values > self.gripper_threshold).astype(int)
    
    def discretize_action(self, action: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Discretize full action including pose and gripper.
        
        Args:
            action: Full action [x, y, z, roll, pitch, yaw, gripper]
            
        Returns:
            discrete_pose: Discretized pose [6,]
            discrete_gripper: Binary gripper value (0 or 1)
        """
        pose = action[:6]
        gripper = action[6] if len(action) > 6 else 0
        
        discrete_pose = self.discretize_pose(pose)
        discrete_gripper = int(gripper > self.gripper_threshold)
        
        return discrete_pose, discrete_gripper
    
    def continuous_to_discrete(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert continuous pose to discrete representation.
        Alias for discretize_pose for clarity.
        
        Args:
            pose: Continuous pose [x, y, z, roll, pitch, yaw] or [N, 6]
            
        Returns:
            discrete_pose: Discretized pose [6,] or [N, 6]
        """
        return self.discretize_pose(pose)
    
    def discrete_to_continuous(self, discrete_pose: np.ndarray) -> np.ndarray:
        """
        Convert discrete bin indices back to continuous pose (bin centers).
        
        Args:
            discrete_pose: Discretized pose with bin indices [6,] or [N, 6]
            
        Returns:
            continuous_pose: Continuous pose [6,] or [N, 6]
        """
        single_pose = False
        if discrete_pose.ndim == 1:
            discrete_pose = discrete_pose.reshape(1, -1)
            single_pose = True
        
        # Get bin centers
        x = self._bin_to_continuous(discrete_pose[:, 0], self.x_bins)
        y = self._bin_to_continuous(discrete_pose[:, 1], self.y_bins)
        z = self._bin_to_continuous(discrete_pose[:, 2], self.z_bins)
        
        roll = self._bin_to_continuous(discrete_pose[:, 3], self.roll_bins)
        pitch = self._bin_to_continuous(discrete_pose[:, 4], self.pitch_bins)
        yaw = self._bin_to_continuous(discrete_pose[:, 5], self.yaw_bins)
        
        continuous_pose = np.stack([x, y, z, roll, pitch, yaw], axis=1)
        
        if single_pose:
            return continuous_pose[0]
        return continuous_pose
    
    def _bin_to_continuous(self, bin_indices: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Convert bin indices to continuous values (bin centers).
        
        Args:
            bin_indices: Bin indices
            bins: Bin edges
            
        Returns:
            continuous_values: Continuous values at bin centers
        """
        # Compute bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers[bin_indices.astype(int)]
    
    def format_discrete_pose_as_text(self, discrete_pose: np.ndarray, 
                                    discrete_gripper: int = None) -> str:
        """
        Format discrete pose as text for ICL prompt.
        
        Args:
            discrete_pose: Discretized pose [6,]
            discrete_gripper: Binary gripper value (optional)
            
        Returns:
            text: Formatted text representation
        """
        text = f"[{discrete_pose[0]}, {discrete_pose[1]}, {discrete_pose[2]}, " \
               f"{discrete_pose[3]}, {discrete_pose[4]}, {discrete_pose[5]}"
        
        if discrete_gripper is not None:
            text += f", {discrete_gripper}"
        
        text += "]"
        return text
    
    def parse_discrete_pose_from_text(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Parse discrete pose from text format.
        
        Args:
            text: Text representation like "[10, 20, 30, 5, 10, 15, 1]"
            
        Returns:
            discrete_pose: Discretized pose [6,]
            discrete_gripper: Binary gripper value
        """
        # Remove brackets and split
        text = text.strip("[]")
        values = [int(v.strip()) for v in text.split(",")]
        
        discrete_pose = np.array(values[:6])
        discrete_gripper = values[6] if len(values) > 6 else 0
        
        return discrete_pose, discrete_gripper


def test_discretizer():
    """Test the discretizer with sample poses."""
    from pathlib import Path
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create discretizer
    discretizer = ActionDiscretizer(config)
    
    # Test single pose
    pose = np.array([0.1, 0.05, 0.2, 0.5, -0.3, 1.0])
    discrete = discretizer.discretize_pose(pose)
    continuous = discretizer.discrete_to_continuous(discrete)
    
    print("\nTest single pose:")
    print(f"Original:    {pose}")
    print(f"Discrete:    {discrete}")
    print(f"Continuous:  {continuous}")
    print(f"Error:       {np.abs(pose - continuous)}")
    
    # Test text formatting
    text = discretizer.format_discrete_pose_as_text(discrete, discrete_gripper=1)
    print(f"\nFormatted text: {text}")
    
    parsed_pose, parsed_gripper = discretizer.parse_discrete_pose_from_text(text)
    print(f"Parsed pose: {parsed_pose}")
    print(f"Parsed gripper: {parsed_gripper}")
    
    # Test batch
    poses_batch = np.random.uniform(-0.2, 0.2, (5, 6))
    discrete_batch = discretizer.discretize_pose(poses_batch)
    print(f"\nBatch test:")
    print(f"Input shape: {poses_batch.shape}")
    print(f"Output shape: {discrete_batch.shape}")


if __name__ == "__main__":
    test_discretizer()
