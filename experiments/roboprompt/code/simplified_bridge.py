"""
Simplified RoboPrompt to SO101 Conversion Bridge

This module converts RoboPrompt's discretized actions to SO101 joint angles
using ONLY math (no robot control libraries needed).

Updated to handle new format and realistic scene bounds.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import json
import re


class SimplifiedRoboPromptBridge:
    """
    Pure math bridge: RoboPrompt actions → SO101 joint angles
    
    NO robot control - just conversion math!
    """
    
    def __init__(
        self,
        scene_bounds: List[float] = None,
        rotation_resolution: int = 72
    ):
        """
        Initialize the bridge.
        
        Args:
            scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] in meters
            rotation_resolution: Number of rotation bins (default: 72 = 5° resolution)
        """
        # Scene bounds - should match what was used in dataset creation
        if scene_bounds is None:
            # Use the same REALISTIC bounds as in the dataset creation
            self.scene_bounds = np.array([-0.2, -0.2, 0.3, 0.2, 0.2, 0.7])
        else:
            self.scene_bounds = np.array(scene_bounds)
        
        self.rotation_resolution = rotation_resolution
        
        # SO100/SO101 joint limits (from URDF)
        self.joint_limits = {
            'shoulder_pan': (-2.0, 2.0),
            'shoulder_lift': (0.0, 3.5),
            'elbow_flex': (-3.14159, 0.0),
            'wrist_flex': (-2.5, 1.2),
            'wrist_roll': (-3.14159, 3.14159),
            'gripper': (-0.2, 2.0)  # In radians, but dataset uses degrees
        }
        
        print("✅ Simplified Bridge initialized (NO robot control)")
        print(f"   Scene bounds: {self.scene_bounds}")
        print(f"   Rotation resolution: {self.rotation_resolution} bins")
    
    def discretized_to_continuous_pose(
        self, 
        discretized_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert discretized RoboPrompt action to continuous pose.
        
        Args:
            discretized_action: [x, y, z, rx, ry, rz, gripper] (all integers)
                - x, y, z: bins (0-99)
                - rx, ry, rz: rotation bins (0-71 for 72 bins)
                - gripper: 0 (closed) or 1 (open)
        
        Returns:
            position: [x, y, z] in meters
            quaternion: [qx, qy, qz, qw]
            gripper_state: 0 or 1
        """
        # Extract components
        trans_indices = discretized_action[:3]
        rot_indices = discretized_action[3:6]
        gripper_state = discretized_action[6]
        
        # Convert translation from bins to continuous (0-99 bins)
        bounds = self.scene_bounds
        resolution = (bounds[3:] - bounds[:3]) / 100
        position = bounds[:3] + resolution * trans_indices + resolution / 2
        
        # Convert rotation from bins to Euler angles
        euler_degrees = rot_indices * (360 / self.rotation_resolution)
        euler_rad = np.deg2rad(euler_degrees)
        
        # Convert Euler to quaternion
        rotation = R.from_euler('xyz', euler_rad)
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
        
        return position, quaternion, gripper_state
    
    def continuous_pose_to_joint_angles(
        self,
        position: np.ndarray,
        quaternion: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Convert Cartesian pose to SO101 joint angles using analytical IK.
        
        This is a simplified 5-DOF analytical IK for SO100/SO101.
        For production, use a proper IK solver (PyBullet, MoveIt, etc.)
        
        Args:
            position: [x, y, z] target position in meters
            quaternion: [qx, qy, qz, qw] target orientation
        
        Returns:
            joint_angles: [j1, j2, j3, j4, j5] in radians, or None if unreachable
        """
        x, y, z = position
        
        # Link lengths from SO100 URDF
        L1 = 0.1025   # shoulder to upper_arm
        L2 = 0.11257  # upper_arm to lower_arm
        L3 = 0.1349   # lower_arm to wrist
        L4 = 0.0601   # wrist to gripper
        base_height = 0.0165
        
        # Joint 1: shoulder_pan (rotation around Z)
        j1 = np.arctan2(y, x)
        
        # Planar 2D IK in the arm's plane
        r = np.sqrt(x**2 + y**2)  # Radial distance
        z_adj = z - base_height    # Adjust for base height
        
        # Target distance in 2D plane
        d = np.sqrt(r**2 + z_adj**2)
        
        # Check reachability - UPDATED for realistic reach
        max_reach = 0.41  # SO100/SO101 maximum reach
        min_reach = 0.15  # Minimum reach (can't get too close to base)
        if d > max_reach or d < min_reach:
            print(f"   ⚠️ Target unreachable: d={d:.3f}m, range=[{min_reach:.3f}, {max_reach:.3f}]")
            return None
        
        # Solve 2-link planar IK for joints 2 and 3
        # Using law of cosines
        effective_reach = d - L4  # Account for gripper length
        
        cos_j3 = (effective_reach**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_j3 = np.clip(cos_j3, -1, 1)
        j3 = -np.arccos(cos_j3)  # Elbow down configuration
        
        # Joint 2 calculation
        alpha = np.arctan2(z_adj, r)
        beta = np.arctan2(L2 * np.sin(-j3), L1 + L2 * np.cos(-j3))
        j2 = alpha - beta
        
        # Convert quaternion to Euler for wrist orientation
        rotation = R.from_quat(quaternion)
        euler = rotation.as_euler('xyz')
        
        # Joints 4 and 5: wrist orientation
        # j4 controls pitch, j5 controls roll
        j4 = euler[1]  # Pitch
        j5 = euler[2]  # Roll
        
        # Clip to joint limits
        j1 = np.clip(j1, *self.joint_limits['shoulder_pan'])
        j2 = np.clip(j2, *self.joint_limits['shoulder_lift'])
        j3 = np.clip(j3, *self.joint_limits['elbow_flex'])
        j4 = np.clip(j4, *self.joint_limits['wrist_flex'])
        j5 = np.clip(j5, *self.joint_limits['wrist_roll'])
        
        joint_angles = np.array([j1, j2, j3, j4, j5])
        return joint_angles
    
    def gripper_state_to_joint(self, gripper_state: float) -> float:
        """
        Convert binary gripper state to joint angle.
        
        Args:
            gripper_state: 0 (closed) or 1 (open)
        
        Returns:
            gripper_joint_angle: angle in degrees (for SO101 dataset format)
        """
        # SO101 dataset uses degrees for gripper
        # From HuggingFace dataset: gripper ranges 0-30 degrees
        min_angle_deg = 0.0   # Closed
        max_angle_deg = 30.0  # Open
        
        gripper_angle_deg = min_angle_deg + gripper_state * (max_angle_deg - min_angle_deg)
        return gripper_angle_deg
    
    def convert_roboprompt_action(
        self,
        discretized_action: np.ndarray,
        verbose: bool = True
    ) -> Optional[dict]:
        """
        Convert single RoboPrompt action to SO101 format.
        
        Args:
            discretized_action: [x, y, z, rx, ry, rz, gripper]
            verbose: Print conversion details
        
        Returns:
            Dictionary with conversion results or None if failed
        """
        if verbose:
            print(f"   Input (discretized): {discretized_action}")
        
        # Step 1: Discretized → Continuous pose
        position, quaternion, gripper = self.discretized_to_continuous_pose(discretized_action)
        
        if verbose:
            print(f"   Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]m")
            print(f"   Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            print(f"   Gripper: {'OPEN' if gripper == 1 else 'CLOSED'}")
        
        # Step 2: Continuous pose → Joint angles
        joint_angles = self.continuous_pose_to_joint_angles(position, quaternion)
        
        if joint_angles is None:
            if verbose:
                print("   ❌ IK failed")
            return None
        
        # Step 3: Add gripper
        gripper_angle = self.gripper_state_to_joint(gripper)
        
        # SO101 format: [j1, j2, j3, j4, j5, gripper] all in DEGREES
        full_joint_angles_deg = np.rad2deg(joint_angles).tolist() + [gripper_angle]
        
        if verbose:
            print(f"   Joints [1-5]: {np.rad2deg(joint_angles).round(1)}°")
            print(f"   Gripper [6]: {gripper_angle:.1f}°")
            print(f"   ✅ Conversion successful!")
        
        return {
            'joint_angles_deg': full_joint_angles_deg,
            'joint_angles_rad': joint_angles.tolist() + [np.deg2rad(gripper_angle)],
            'position': position.tolist(),
            'quaternion': quaternion.tolist(),
            'gripper_state': int(gripper)
        }
    
    def convert_roboprompt_sequence(
        self,
        actions: List[np.ndarray]
    ) -> List[dict]:
        """
        Convert sequence of RoboPrompt actions to SO101 format.
        
        Args:
            actions: List of discretized actions
        
        Returns:
            List of conversion results (only successful ones)
        """
        print(f"\n🔄 Converting {len(actions)} RoboPrompt actions to SO101 format...")
        
        results = []
        for i, action in enumerate(actions):
            print(f"\n[Action {i+1}/{len(actions)}]")
            result = self.convert_roboprompt_action(action, verbose=True)
            if result is not None:
                results.append(result)
        
        print(f"\n✅ Successfully converted {len(results)}/{len(actions)} actions")
        return results
    
    def parse_llm_output(self, llm_output: str) -> List[np.ndarray]:
        """
        Parse RoboPrompt LLM output to extract action sequence.
        
        Handles various output formats:
        - Plain JSON array
        - Markdown code blocks with ```json or ```
        - Text with embedded arrays
        
        Args:
            llm_output: Raw text from LLM
        
        Returns:
            List of discretized actions as numpy arrays
        """
        try:
            # Clean the output - remove any extra text
            llm_output = llm_output.strip()
            
            # Try to extract from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
            if json_match:
                actions_str = json_match.group(1)
            else:
                # Try without json tag
                code_match = re.search(r'```\s*(.*?)\s*```', llm_output, re.DOTALL)
                if code_match:
                    actions_str = code_match.group(1)
                else:
                    # Try to find the outermost array
                    array_match = re.search(r'\[.*\]', llm_output, re.DOTALL)
                    if array_match:
                        actions_str = array_match.group(0)
                    else:
                        # Fallback: use the whole output
                        actions_str = llm_output
            
            # Parse JSON
            actions = json.loads(actions_str)
            
            # Convert to numpy arrays
            if isinstance(actions, list):
                if len(actions) > 0 and isinstance(actions[0], list):
                    # Already in correct format: [[x, y, z, rx, ry, rz, gripper], ...]
                    return [np.array(action, dtype=int) for action in actions]
                elif len(actions) == 7:
                    # Single action: [x, y, z, rx, ry, rz, gripper]
                    return [np.array(actions, dtype=int)]
            
            print(f"⚠️ Unexpected action format, using default")
            return [np.array([50, 50, 50, 0, 36, 0, 1])]
            
        except Exception as e:
            print(f"⚠️ Failed to parse LLM output: {e}")
            print(f"Raw output: {llm_output[:200]}...")
            # Return default action
            return [np.array([50, 50, 50, 0, 36, 0, 1])]
    
    def save_results(
        self,
        results: List[dict],
        save_path: str
    ):
        """
        Save conversion results to JSON.
        
        Args:
            results: List of conversion result dictionaries
            save_path: Path to save JSON file
        """
        output = {
            'num_actions': len(results),
            'scene_bounds': self.scene_bounds.tolist(),
            'actions': results
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Saved {len(results)} converted actions to: {save_path}")


def main():
    """Test the simplified bridge."""
    
    print("\n" + "="*80)
    print("TESTING SIMPLIFIED ROBOPROMPT → SO101 BRIDGE")
    print("="*80 + "\n")
    
    # Initialize bridge
    bridge = SimplifiedRoboPromptBridge(
        scene_bounds=[-0.2, -0.2, 0.3, 0.2, 0.2, 0.7]  # Match REALISTIC dataset bounds
    )
    
    # Test with example RoboPrompt actions - UPDATED to realistic positions
    print("\n📋 Testing with example actions...")
    
    test_actions = [
        np.array([50, 50, 50, 0, 36, 0, 1]),  # Center, gripper open
        np.array([60, 60, 40, 0, 36, 0, 1]),  # Move right/forward/down
        np.array([60, 60, 30, 0, 36, 0, 0]),  # Lower and close gripper
        np.array([60, 60, 50, 0, 36, 0, 0]),  # Lift with object
    ]
    
    # Convert actions
    results = bridge.convert_roboprompt_sequence(test_actions)
    
    # Save results
    bridge.save_results(results, "test_conversion.json")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
