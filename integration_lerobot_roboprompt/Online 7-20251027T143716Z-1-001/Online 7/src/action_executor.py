"""
Action Executor Module
Converts predicted discrete end-effector actions to joint commands for execution.
"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import yaml


class ActionExecutor:
    """
    Converts discrete EE actions to joint commands using nearest neighbor search.
    
    The challenge: RoboPrompt predicts discrete EE poses, but the robot needs joint positions.
    
    Solution options:
    1. Nearest Neighbor: Find closest EE pose in dataset, use those joints
    2. IK Solver: Use PyBullet/pinocchio IK (may fail for some poses)
    3. Learned Mapping: Train small MLP (requires training)
    
    We implement option 1 (nearest neighbor) as it's simplest and doesn't require IK.
    """
    
    def __init__(self, config: Dict, discretizer, fk_solver):
        """
        Initialize action executor.
        
        Args:
            config: Configuration dictionary
            discretizer: ActionDiscretizer instance
            fk_solver: ForwardKinematics instance (optional, for IK)
        """
        ik_config = config['ik']
        exec_config = config['execution']
        
        self.method = ik_config['method']
        self.distance_metric = ik_config.get('nearest_neighbor', {}).get('distance_metric', 'euclidean')
        
        self.control_frequency = exec_config['control_frequency']
        self.interpolation_steps = exec_config['interpolation_steps']
        self.use_simulation = exec_config['use_simulation']
        
        self.discretizer = discretizer
        self.fk_solver = fk_solver
        
        # Database for nearest neighbor search
        self.ee_pose_database = None
        self.joint_position_database = None
        
        print(f"Initialized ActionExecutor:")
        print(f"  Method: {self.method}")
        print(f"  Control frequency: {self.control_frequency} Hz")
        print(f"  Interpolation steps: {self.interpolation_steps}")
    
    def build_database(self, demonstrations: List[Dict]):
        """
        Build database of EE poses and corresponding joint positions from demonstrations.
        
        Args:
            demonstrations: List of demonstration dictionaries from dataset
        """
        print("Building EE pose database from demonstrations...")
        
        all_ee_poses = []
        all_joint_positions = []
        
        # Load original dataset to get joint positions
        # For now, this is a placeholder - you'd actually load from LeRobot dataset
        # and compute FK for all frames
        
        # Placeholder: generate some sample data
        print("WARNING: Using placeholder database! Load actual data from LeRobot dataset.")
        
        # Generate synthetic database for testing
        num_samples = 1000
        for _ in range(num_samples):
            # Random joint positions within limits
            joint_pos = np.random.uniform(-1, 1, 6)
            
            # Compute corresponding EE pose
            ee_pose = self.fk_solver.compute_fk_full(joint_pos)
            
            # Discretize and store
            discrete_ee_pose = self.discretizer.discretize_pose(ee_pose)
            
            all_ee_poses.append(discrete_ee_pose)
            all_joint_positions.append(joint_pos)
        
        self.ee_pose_database = np.array(all_ee_poses)
        self.joint_position_database = np.array(all_joint_positions)
        
        print(f"Built database with {len(self.ee_pose_database)} samples")
    
    def discrete_to_joint(self, discrete_pose: np.ndarray) -> np.ndarray:
        """
        Convert discrete EE pose to joint positions using nearest neighbor.
        
        Args:
            discrete_pose: Discrete EE pose [6,]
            
        Returns:
            joint_positions: Joint positions [6,]
        """
        if self.method == 'nearest_neighbor':
            return self._nn_discrete_to_joint(discrete_pose)
        elif self.method == 'pybullet_ik':
            # Convert discrete to continuous first
            continuous_pose = self.discretizer.discrete_to_continuous(discrete_pose)
            return self._ik_continuous_to_joint(continuous_pose)
        else:
            raise ValueError(f"Unsupported IK method: {self.method}")
    
    def _nn_discrete_to_joint(self, discrete_pose: np.ndarray) -> np.ndarray:
        """
        Nearest neighbor search to find joint positions.
        
        Args:
            discrete_pose: Discrete EE pose [6,]
            
        Returns:
            joint_positions: Joint positions [6,]
        """
        if self.ee_pose_database is None:
            raise ValueError("Database not built! Call build_database() first.")
        
        # Compute distances to all database entries
        if self.distance_metric == 'euclidean':
            distances = np.linalg.norm(
                self.ee_pose_database - discrete_pose, axis=1
            )
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Find nearest neighbor
        nearest_idx = np.argmin(distances)
        
        # Return corresponding joint positions
        return self.joint_position_database[nearest_idx]
    
    def _ik_continuous_to_joint(self, continuous_pose: np.ndarray) -> np.ndarray:
        """
        Use IK solver to convert continuous EE pose to joint positions.
        
        Args:
            continuous_pose: Continuous EE pose [6,]
            
        Returns:
            joint_positions: Joint positions [6,]
        """
        # TODO: Implement actual IK using PyBullet or pinocchio
        raise NotImplementedError("IK solver not yet implemented")
    
    def execute_action_sequence(self,
                               discrete_actions: List[np.ndarray],
                               discrete_grippers: List[int]) -> List[np.ndarray]:
        """
        Convert sequence of discrete actions to joint trajectories.
        
        Args:
            discrete_actions: List of discrete EE poses
            discrete_grippers: List of gripper states
            
        Returns:
            joint_trajectories: List of joint position arrays [T, 6]
        """
        # Convert each discrete action to joint positions
        joint_keyframes = []
        for discrete_action in discrete_actions:
            joint_pos = self.discrete_to_joint(discrete_action)
            joint_keyframes.append(joint_pos)
        
        # Interpolate between keyframes
        joint_trajectories = self._interpolate_trajectory(
            joint_keyframes, discrete_grippers
        )
        
        return joint_trajectories
    
    def _interpolate_trajectory(self,
                               joint_keyframes: List[np.ndarray],
                               gripper_keyframes: List[int]) -> np.ndarray:
        """
        Interpolate smooth trajectory between keyframes.
        
        Args:
            joint_keyframes: List of joint positions at keyframes
            gripper_keyframes: List of gripper states at keyframes
            
        Returns:
            trajectory: Interpolated trajectory [T, 7] (6 joints + gripper)
        """
        if len(joint_keyframes) < 2:
            return np.array(joint_keyframes)
        
        trajectory = []
        
        for i in range(len(joint_keyframes) - 1):
            start_joints = joint_keyframes[i]
            end_joints = joint_keyframes[i + 1]
            
            start_gripper = gripper_keyframes[i]
            end_gripper = gripper_keyframes[i + 1]
            
            # Linear interpolation
            for t in range(self.interpolation_steps):
                alpha = t / self.interpolation_steps
                
                # Interpolate joints
                interp_joints = (1 - alpha) * start_joints + alpha * end_joints
                
                # Interpolate gripper (step function at midpoint)
                interp_gripper = start_gripper if t < self.interpolation_steps // 2 else end_gripper
                
                # Combine
                waypoint = np.concatenate([interp_joints, [interp_gripper]])
                trajectory.append(waypoint)
        
        # Add final keyframe
        final_waypoint = np.concatenate([joint_keyframes[-1], [gripper_keyframes[-1]]])
        trajectory.append(final_waypoint)
        
        return np.array(trajectory)
    
    def send_to_robot(self, joint_trajectory: np.ndarray):
        """
        Send joint trajectory to robot for execution.
        
        Args:
            joint_trajectory: Trajectory [T, 7] (6 joints + gripper)
        """
        if self.use_simulation:
            print("Simulation mode: Would send trajectory to robot")
            print(f"Trajectory shape: {joint_trajectory.shape}")
            return
        
        # TODO: Integrate with actual robot control
        # This depends on your LeRobot/SO101 setup
        
        print("Sending trajectory to robot...")
        print(f"Trajectory length: {len(joint_trajectory)} waypoints")
        print(f"Control frequency: {self.control_frequency} Hz")
        
        # Example pseudo-code:
        # robot = SO101Robot()
        # for waypoint in joint_trajectory:
        #     joint_pos = waypoint[:6]
        #     gripper = waypoint[6]
        #     robot.set_joint_positions(joint_pos)
        #     robot.set_gripper(gripper)
        #     time.sleep(1.0 / self.control_frequency)
        
        raise NotImplementedError("Robot control interface not yet implemented")


class NearestNeighborIK:
    """
    Standalone nearest neighbor IK solver.
    Can be used independently of ActionExecutor.
    """
    
    def __init__(self, ee_poses: np.ndarray, joint_positions: np.ndarray):
        """
        Initialize with database of (EE pose, joint position) pairs.
        
        Args:
            ee_poses: Array of EE poses [N, 6]
            joint_positions: Array of joint positions [N, 6]
        """
        self.ee_poses = ee_poses
        self.joint_positions = joint_positions
        
        print(f"Initialized NearestNeighborIK with {len(ee_poses)} samples")
    
    def solve(self, target_pose: np.ndarray) -> np.ndarray:
        """
        Find joint positions for target EE pose.
        
        Args:
            target_pose: Target EE pose [6,]
            
        Returns:
            joint_positions: Corresponding joint positions [6,]
        """
        # Compute distances
        distances = np.linalg.norm(self.ee_poses - target_pose, axis=1)
        
        # Find nearest
        nearest_idx = np.argmin(distances)
        
        return self.joint_positions[nearest_idx]
    
    def solve_batch(self, target_poses: np.ndarray) -> np.ndarray:
        """
        Find joint positions for multiple target poses.
        
        Args:
            target_poses: Target EE poses [M, 6]
            
        Returns:
            joint_positions: Corresponding joint positions [M, 6]
        """
        results = []
        for pose in target_poses:
            joints = self.solve(pose)
            results.append(joints)
        return np.array(results)


def test_action_executor():
    """Test action executor with sample actions."""
    from forward_kinematics import ForwardKinematics
    from action_discretizer import ActionDiscretizer
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    urdf_path = Path(__file__).parent.parent / config["robot"]["urdf_path"]
    fk_solver = ForwardKinematics(str(urdf_path))
    discretizer = ActionDiscretizer(config)
    
    # Create executor
    executor = ActionExecutor(config, discretizer, fk_solver)
    
    # Build database (using placeholder)
    executor.build_database([])
    
    # Test with sample discrete actions
    discrete_actions = [
        np.array([10, 20, 30, 5, 10, 15]),
        np.array([20, 30, 40, 10, 15, 20]),
        np.array([30, 40, 50, 15, 20, 25])
    ]
    discrete_grippers = [0, 1, 0]
    
    print("\nTesting action execution...")
    joint_trajectory = executor.execute_action_sequence(discrete_actions, discrete_grippers)
    
    print(f"Generated trajectory with {len(joint_trajectory)} waypoints")
    print(f"Trajectory shape: {joint_trajectory.shape}")
    
    # Test sending to robot (simulation mode)
    executor.send_to_robot(joint_trajectory)


if __name__ == "__main__":
    test_action_executor()
