"""
Forward Kinematics Module for SO101 Robot
Converts joint space positions to end-effector poses using PyBullet
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Tuple, Optional
from pathlib import Path


class ForwardKinematics:
    """
    Forward kinematics solver using PyBullet physics simulation.
    Converts joint positions to end-effector poses (position + orientation).
    """
    
    def __init__(self, urdf_path: str, end_effector_link: str = "jaw"):
        """
        Initialize the FK solver.
        
        Args:
            urdf_path: Path to the robot URDF file
            end_effector_link: Name of the end-effector link
        """
        self.urdf_path = Path(urdf_path)
        self.end_effector_link = end_effector_link
        
        # Initialize PyBullet in DIRECT mode (no GUI)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load robot
        self.robot_id = p.loadURDF(str(self.urdf_path), [0, 0, 0], useFixedBase=True)
        
        # Get joint and link information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        
        # Map joint and link names to indices
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            
            # Only consider revolute joints (type 0)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
            
            # Find end-effector link index
            if link_name == end_effector_link:
                self.ee_link_index = i
        
        print(f"Initialized FK solver for {len(self.joint_indices)} joints")
        print(f"Joint names: {self.joint_names}")
        print(f"End-effector link: {end_effector_link} (index: {self.ee_link_index})")
    
    def compute_fk(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint positions.
        
        Args:
            joint_positions: Array of joint positions [6,] or [N, 6]
            
        Returns:
            position: End-effector position [x, y, z] or [N, 3]
            orientation: End-effector orientation as Euler angles [roll, pitch, yaw] or [N, 3]
        """
        # Handle single configuration
        single_config = False
        if joint_positions.ndim == 1:
            joint_positions = joint_positions.reshape(1, -1)
            single_config = True
        
        positions = []
        orientations = []
        
        for joints in joint_positions:
            # Set joint positions
            for joint_idx, joint_pos in zip(self.joint_indices, joints):
                p.resetJointState(self.robot_id, joint_idx, joint_pos)
            
            # Get end-effector pose
            link_state = p.getLinkState(self.robot_id, self.ee_link_index)
            position = np.array(link_state[0])  # World position
            orientation_quat = np.array(link_state[1])  # Quaternion
            
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            orientation_euler = np.array(p.getEulerFromQuaternion(orientation_quat))
            
            positions.append(position)
            orientations.append(orientation_euler)
        
        positions = np.array(positions)
        orientations = np.array(orientations)
        
        if single_config:
            return positions[0], orientations[0]
        return positions, orientations
    
    def compute_fk_full(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics and return full 6D pose.
        
        Args:
            joint_positions: Array of joint positions [6,] or [N, 6]
            
        Returns:
            poses: End-effector poses [x, y, z, roll, pitch, yaw] or [N, 6]
        """
        positions, orientations = self.compute_fk(joint_positions)
        
        if positions.ndim == 1:
            return np.concatenate([positions, orientations])
        return np.concatenate([positions, orientations], axis=1)
    
    def compute_fk_batch(self, joint_positions_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute FK for a batch of joint position sequences.
        Useful for processing entire episodes.
        
        Args:
            joint_positions_list: List of joint position arrays, each [T, 6]
            
        Returns:
            poses_list: List of pose arrays, each [T, 6]
        """
        poses_list = []
        for joint_positions in joint_positions_list:
            poses = self.compute_fk_full(joint_positions)
            poses_list.append(poses)
        return poses_list
    
    def close(self):
        """Disconnect PyBullet physics client."""
        p.disconnect(self.physics_client)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close()
        except:
            pass


def test_fk():
    """Test the FK solver with sample joint positions."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create FK solver
    urdf_path = Path(__file__).parent.parent / config["robot"]["urdf_path"]
    fk = ForwardKinematics(str(urdf_path))
    
    # Test with sample joint positions (all zeros)
    joint_positions = np.zeros(6)
    position, orientation = fk.compute_fk(joint_positions)
    
    print(f"\nTest FK with zero joint positions:")
    print(f"Position (x, y, z): {position}")
    print(f"Orientation (r, p, y): {orientation}")
    
    # Test with batch
    joint_positions_batch = np.random.uniform(-1, 1, (5, 6))
    poses = fk.compute_fk_full(joint_positions_batch)
    print(f"\nBatch FK test:")
    print(f"Input shape: {joint_positions_batch.shape}")
    print(f"Output shape: {poses.shape}")
    
    fk.close()


if __name__ == "__main__":
    test_fk()
