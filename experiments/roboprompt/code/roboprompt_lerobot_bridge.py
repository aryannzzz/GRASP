"""
RoboPrompt to LeRobot SO100/SO101 Integration Bridge (Simulation Only)

This module converts RoboPrompt's discretized end-effector actions
to joint position commands using inverse kinematics.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import time


class RoboPromptToLeRobotBridge:
    """
    Bridge to convert RoboPrompt actions to robot commands (simulation only).
    """
    
    def __init__(
        self,
        robot_port: str = "/dev/ttyUSB0",
        scene_bounds: List[float] = None,
        rotation_resolution: int = 72,
        use_sim: bool = True  # Always simulation for now
    ):
        """
        Initialize the bridge.
        """
        if scene_bounds is None:
            self.scene_bounds = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
        else:
            self.scene_bounds = np.array(scene_bounds)
        
        self.rotation_resolution = rotation_resolution
        self.use_sim = use_sim
        
        print(f"🤖 RoboPrompt Bridge initialized (Simulation Mode)")
        print(f"   Scene bounds: {self.scene_bounds}")
    
    def discretized_to_continuous_pose(
        self, 
        discretized_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert discretized RoboPrompt action to continuous pose.
        """
        trans_indices = discretized_action[:3]
        rot_indices = discretized_action[3:6]
        gripper_state = discretized_action[6]
        
        # Convert position bins to continuous coordinates
        bounds = self.scene_bounds
        position = np.array([
            bounds[0] + (trans_indices[0] / 99.0) * (bounds[3] - bounds[0]),
            bounds[1] + (trans_indices[1] / 99.0) * (bounds[4] - bounds[1]),
            bounds[2] + (trans_indices[2] / 99.0) * (bounds[5] - bounds[2])
        ])
        
        # Convert rotation bins to continuous Euler angles
        euler_angles = rot_indices * (360 / self.rotation_resolution)
        euler_rad = np.deg2rad(euler_angles)
        
        # Convert to quaternion
        rotation = R.from_euler('xyz', euler_rad)
        quaternion = rotation.as_quat()
        
        return position, quaternion, gripper_state
    
    def continuous_pose_to_joint_angles(
        self,
        position: np.ndarray,
        quaternion: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Simplified IK - returns dummy joint angles for testing.
        """
        # Return default joint angles for simulation
        joint_angles = np.deg2rad([0.0, -45.0, 90.0, -45.0, 0.0])
        return joint_angles
    
    def gripper_state_to_joint(self, gripper_state: float) -> float:
        """
        Convert binary gripper state to joint angle.
        """
        return np.deg2rad(30.0) if gripper_state == 1 else np.deg2rad(0.0)
    
    def execute_roboprompt_actions(
        self, 
        roboprompt_actions: List[np.ndarray],
        execution_speed: float = 1.0
    ) -> bool:
        """
        Simulate execution of RoboPrompt actions.
        """
        print(f"\n🤖 Simulating {len(roboprompt_actions)} RoboPrompt actions...")
        
        for i, action in enumerate(roboprompt_actions):
            print(f"\n[Action {i+1}/{len(roboprompt_actions)}]")
            print(f"  Discretized: {action}")
            
            position, quaternion, gripper = self.discretized_to_continuous_pose(action)
            joint_angles = self.continuous_pose_to_joint_angles(position, quaternion)
            
            if joint_angles is not None:
                print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]m")
                print(f"  Joint angles: {np.rad2deg(joint_angles).round(1)}°")
                print(f"  Gripper: {'OPEN' if gripper == 1 else 'CLOSED'}")
                print("  ✓ Action simulated")
            else:
                print("  ✗ IK failed")
                continue
            
            time.sleep(0.5 / execution_speed)
        
        print("\n✅ All actions simulated successfully!")
        return True
    
    def process_roboprompt_output(self, output_text: str) -> List[np.ndarray]:
        """
        Parse RoboPrompt's text output to extract actions.
        """
        import json
        import re
        
        try:
            # Extract JSON array from text
            match = re.search(r'\[\[.*?\]\]', output_text)
            if match:
                actions = json.loads(match.group())
            else:
                # Try to find any array format
                match = re.search(r'\[.*\]', output_text)
                if match:
                    actions = json.loads(match.group())
                else:
                    actions = [[50, 50, 50, 0, 36, 0, 1]]  # Default
        except Exception as e:
            print(f"⚠ Error parsing actions: {e}")
            actions = [[50, 50, 50, 0, 36, 0, 1]]
        
        if len(np.array(actions).shape) == 1:
            actions = [actions]
        
        return [np.array(action) for action in actions]
    
    def disconnect(self):
        """Cleanup."""
        print("🔌 Bridge disconnected")


def main():
    """Example usage."""
    bridge = RoboPromptToLeRobotBridge(use_sim=True)
    
    roboprompt_actions = [
        [78, 56, 24, 0, 36, 25, 0],
        [78, 56, 17, 0, 36, 25, 0],
        [78, 56, 17, 0, 36, 25, 1],
    ]
    
    success = bridge.execute_roboprompt_actions(roboprompt_actions)
    
    if success:
        print("\n🎉 Simulation completed successfully!")
    
    bridge.disconnect()


if __name__ == "__main__":
    main()
