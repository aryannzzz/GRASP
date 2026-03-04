"""
Complete RoboPrompt Inference + SO-101 Execution Pipeline

This script:
1. Loads ICL demonstrations (created from dataset)
2. Uses real perception to detect objects in NEW scenes
3. Queries RoboPrompt LLM for actions
4. Executes on SO-101 using LeRobot's motor control
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation as R
import time

try:
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    LEROBOT_AVAILABLE = True
except ImportError:
    print("⚠️  LeRobot not installed. Install with: pip install lerobot")
    LEROBOT_AVAILABLE = False

from perception_system import PerceptionSystem, ArucoPerception


class RoboPromptSO101Pipeline:
    """
    Complete pipeline for RoboPrompt inference + SO-101 execution.
    """
    
    def __init__(
        self,
        icl_demos_path: str,
        perception_system: PerceptionSystem,
        robot_port: str = "/dev/ttyUSB0",
        simulation_mode: bool = False
    ):
        """
        Args:
            icl_demos_path: Path to ICL demonstrations JSON
            perception_system: Perception system for object detection
            robot_port: Serial port for SO-101
            simulation_mode: If True, don't actually move robot
        """
        # Load ICL demonstrations
        with open(icl_demos_path) as f:
            self.icl_data = json.load(f)
        
        self.demonstrations = self.icl_data['demonstrations']
        self.scene_bounds = np.array(self.icl_data['scene_bounds'])
        self.task = self.icl_data['task']
        
        print(f"✓ Loaded {len(self.demonstrations)} ICL demonstrations")
        print(f"  Task: {self.task}")
        
        # Perception
        self.perception = perception_system
        
        # Robot
        self.simulation_mode = simulation_mode
        self.robot_port = robot_port
        self.robot = None
        
        if not simulation_mode and LEROBOT_AVAILABLE:
            self._initialize_robot()
    
    def _initialize_robot(self):
        """Initialize SO-101 robot via LeRobot."""
        try:
            print(f"\n🤖 Initializing SO-101 on {self.robot_port}...")
            
            # Create motor bus
            self.motor_bus = FeetechMotorsBus(
                port=self.robot_port,
                motors={
                    # Follower arm motors (STS3215)
                    "shoulder_pan": (1, "sts3215"),
                    "shoulder_lift": (2, "sts3215"),
                    "elbow_flex": (3, "sts3215"),
                    "wrist_flex": (4, "sts3215"),
                    "wrist_roll": (5, "sts3215"),
                    "gripper": (6, "sts3215"),
                }
            )
            
            # Connect to motors
            self.motor_bus.connect()
            
            # Create robot interface
            self.robot = ManipulatorRobot(
                robot_type="so101",
                motor_names=["shoulder_pan", "shoulder_lift", "elbow_flex", 
                           "wrist_flex", "wrist_roll", "gripper"],
                motor_bus=self.motor_bus
            )
            
            print("✓ SO-101 initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize robot: {e}")
            print("   Running in simulation mode instead")
            self.simulation_mode = True
            self.robot = None
    
    def query_roboprompt_llm(
        self,
        detected_objects: Dict[str, List[int]],
        instruction: str,
        num_demos: int = 5
    ) -> List[np.ndarray]:
        """
        Query RoboPrompt LLM for action predictions.
        
        Args:
            detected_objects: Discretized object poses from perception
            instruction: Task instruction
            num_demos: Number of ICL demonstrations to use
        
        Returns:
            List of predicted actions (discretized)
        """
        # Format test input
        obs_parts = [f"'{name}': {bins}" for name, bins in detected_objects.items()]
        obs_str = "{" + ", ".join(obs_parts) + "}"
        test_input = f"{{{obs_str}, '{instruction}'}}>>"
        
        # Format ICL prompt with demonstrations
        icl_demos_subset = self.demonstrations[:num_demos]
        prompt = ", ".join(icl_demos_subset) + ", " + test_input
        
        print(f"\n📝 RoboPrompt LLM Query:")
        print(f"  Using {num_demos} demonstrations")
        print(f"  Test input: {test_input[:100]}...")
        
        # TODO: Call actual LLM (OpenAI, Claude, etc.)
        # For now, we'll use a placeholder
        print("\n⚠️  LLM integration not implemented yet!")
        print("    For demo purposes, using actions from first ICL example...")
        
        # Parse actions from first demonstration as placeholder
        first_demo = self.demonstrations[0]
        actions_str = first_demo.split('>>')[1]
        
        # Parse action arrays
        import ast
        actions_list = ast.literal_eval(actions_str)
        predicted_actions = [np.array(a) for a in actions_list]
        
        print(f"✓ Got {len(predicted_actions)} predicted actions")
        
        return predicted_actions
    
    def undiscretize_action(
        self,
        action: np.ndarray
    ) -> Dict:
        """
        Convert discretized RoboPrompt action to continuous pose.
        
        Args:
            action: [x, y, z, rx, ry, rz, gripper] discretized (0-99, 0-71, 0-1)
        
        Returns:
            dict with 'position', 'quaternion', 'gripper'
        """
        # Undiscretize position
        pos_bins = action[:3]
        pos_normalized = pos_bins / 100.0
        position = self.scene_bounds[:3] + pos_normalized * (self.scene_bounds[3:] - self.scene_bounds[:3])
        
        # Undiscretize rotation
        rot_bins = action[3:6]
        euler_normalized = rot_bins / 72.0
        euler_deg = euler_normalized * 360.0
        euler_rad = np.deg2rad(euler_deg)
        
        rotation = R.from_euler('xyz', euler_rad)
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
        
        # Gripper
        gripper = float(action[6])
        
        return {
            'position': position,
            'quaternion': quaternion,
            'gripper': gripper
        }
    
    def pose_to_joint_angles(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper: float
    ) -> np.ndarray:
        """
        Inverse kinematics: end-effector pose → joint angles.
        
        Args:
            position: [x, y, z] in meters
            quaternion: [qx, qy, qz, qw]
            gripper: Gripper state (0 or 1)
        
        Returns:
            Joint angles [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper]
        """
        # SO-100/101 link lengths
        L1 = 0.1025  # shoulder to upper_arm
        L2 = 0.11257  # upper_arm to lower_arm
        L3 = 0.1349  # lower_arm to wrist
        L4 = 0.0601  # wrist to gripper
        base_height = 0.0165
        
        x, y, z = position
        z_relative = z - base_height  # Relative to shoulder
        
        # Convert quaternion to Euler angles
        rotation = R.from_quat(quaternion)
        yaw, pitch, roll = rotation.as_euler('xyz')
        
        # Joint 1: Shoulder pan (rotation around Z)
        j1 = np.arctan2(y, x)
        
        # Calculate reach in XY plane
        r = np.sqrt(x**2 + y**2)
        
        # Wrist position (accounting for wrist roll)
        r_wrist = r - L4 * np.cos(pitch)
        z_wrist = z_relative - L4 * np.sin(pitch)
        
        # Distance from shoulder to wrist
        d = np.sqrt(r_wrist**2 + z_wrist**2)
        
        # Check if reachable
        if d > (L1 + L2 + L3) or d < abs(L1 - L2 - L3):
            print(f"⚠️  Target position unreachable: distance = {d:.3f}m")
            print(f"   Workspace: [{abs(L1-L2-L3):.3f}, {L1+L2+L3:.3f}]m")
            # Clamp to workspace
            d = np.clip(d, abs(L1-L2-L3) + 0.01, L1 + L2 + L3 - 0.01)
        
        # Joint 3: Elbow (using law of cosines)
        cos_j3 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_j3 = np.clip(cos_j3, -1.0, 1.0)
        j3 = -np.arccos(cos_j3)  # Negative for elbow down
        
        # Joint 2: Shoulder lift
        alpha = np.arctan2(z_wrist, r_wrist)
        beta = np.arctan2(L2 * np.sin(-j3), L1 + L2 * np.cos(-j3))
        j2 = alpha - beta
        
        # Joint 4: Wrist flex (to achieve desired pitch)
        j4 = pitch - j2 - j3
        
        # Joint 5: Wrist roll
        j5 = roll
        
        # Gripper: Convert 0/1 to degrees (0-30°)
        j6 = gripper * 30.0
        
        joint_angles = np.array([j1, j2, j3, j4, j5, np.deg2rad(j6)])
        
        return joint_angles
    
    def execute_actions(
        self,
        actions: List[np.ndarray],
        speed: float = 0.5,
        pause_between: float = 0.5
    ):
        """
        Execute predicted actions on SO-101.
        
        Args:
            actions: List of discretized RoboPrompt actions
            speed: Movement speed multiplier (0.1-2.0)
            pause_between: Pause between actions (seconds)
        """
        print(f"\n🤖 Executing {len(actions)} actions...")
        print(f"  Speed: {speed}x")
        print(f"  Simulation mode: {self.simulation_mode}")
        
        for i, action in enumerate(actions):
            print(f"\n[Action {i+1}/{len(actions)}]")
            
            # Undiscretize action
            pose = self.undiscretize_action(action)
            
            print(f"  Discretized: {action}")
            print(f"  Position: [{pose['position'][0]:.3f}, {pose['position'][1]:.3f}, {pose['position'][2]:.3f}]m")
            print(f"  Gripper: {'OPEN' if pose['gripper'] > 0.5 else 'CLOSED'}")
            
            # Inverse kinematics
            joint_angles = self.pose_to_joint_angles(
                pose['position'],
                pose['quaternion'],
                pose['gripper']
            )
            
            print(f"  Joints: [{np.degrees(joint_angles[0]):.1f}°, {np.degrees(joint_angles[1]):.1f}°, "
                  f"{np.degrees(joint_angles[2]):.1f}°, {np.degrees(joint_angles[3]):.1f}°, "
                  f"{np.degrees(joint_angles[4]):.1f}°, {np.degrees(joint_angles[5]):.1f}°]")
            
            # Execute on robot
            if not self.simulation_mode and self.robot is not None:
                try:
                    # Send command to robot
                    self.robot.teleop_step(
                        joint_positions=joint_angles,
                        velocity=speed
                    )
                    print("  ✓ Executed on robot")
                    
                except Exception as e:
                    print(f"  ❌ Execution failed: {e}")
            else:
                print("  ✓ Simulated (no robot movement)")
            
            # Pause between actions
            if i < len(actions) - 1:
                time.sleep(pause_between / speed)
        
        print(f"\n✅ All {len(actions)} actions completed!")
    
    def run_inference(
        self,
        num_icl_demos: int = 5,
        speed: float = 0.5,
        visualize_perception: bool = True
    ):
        """
        Run complete inference pipeline.
        
        Args:
            num_icl_demos: Number of ICL demonstrations to use
            speed: Robot movement speed
            visualize_perception: Show perception visualization
        """
        print("\n" + "="*80)
        print("ROBOPROMPT INFERENCE PIPELINE")
        print("="*80)
        
        # Step 1: Perceive objects in current scene
        print("\n📷 Step 1: Detecting objects in scene...")
        detected_objects = self.perception.detect_objects(visualize=visualize_perception)
        
        if not detected_objects:
            print("❌ No objects detected!")
            print("   Make sure objects with ArUco markers are visible to camera")
            return
        
        print(f"✓ Detected {len(detected_objects)} objects:")
        for name, pose in detected_objects.items():
            print(f"  - {name}: {pose.position}")
        
        # Step 2: Discretize object poses
        print("\n🎯 Step 2: Discretizing object poses...")
        discretized_objects = self.perception.discretize_poses(detected_objects, self.scene_bounds)
        print(f"✓ Discretized object poses:")
        for name, bins in discretized_objects.items():
            print(f"  '{name}': {bins}")
        
        # Step 3: Query RoboPrompt LLM
        print("\n🧠 Step 3: Querying RoboPrompt LLM...")
        predicted_actions = self.query_roboprompt_llm(
            discretized_objects,
            self.task,
            num_demos=num_icl_demos
        )
        
        # Step 4: Execute actions
        print("\n🤖 Step 4: Executing actions on SO-101...")
        self.execute_actions(predicted_actions, speed=speed)
        
        print("\n" + "="*80)
        print("✅ INFERENCE COMPLETE!")
        print("="*80)
    
    def close(self):
        """Clean up resources."""
        if self.perception:
            self.perception.close()
        
        if self.robot:
            self.motor_bus.disconnect()


def main():
    """Run complete pipeline."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='RoboPrompt + SO-101 Inference Pipeline')
    parser.add_argument('--icl-demos', type=str, default='./roboprompt_data/icl_demos.json',
                       help='Path to ICL demonstrations')
    parser.add_argument('--robot-port', type=str, default='/dev/ttyUSB0',
                       help='Serial port for SO-101')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--marker-size', type=float, default=0.05,
                       help='ArUco marker size in meters')
    parser.add_argument('--speed', type=float, default=0.5,
                       help='Robot movement speed (0.1-2.0)')
    parser.add_argument('--sim', action='store_true',
                       help='Run in simulation mode (no robot movement)')
    parser.add_argument('--num-demos', type=int, default=5,
                       help='Number of ICL demonstrations to use')
    
    args = parser.parse_args()
    
    # Check if ICL demos exist
    if not Path(args.icl_demos).exists():
        print(f"❌ ICL demonstrations not found: {args.icl_demos}")
        print("\nPlease run the demo creation script first:")
        print("  python lerobot_roboprompt_dataset.py")
        return
    
    # Initialize perception
    print("Initializing perception system...")
    perception = ArucoPerception(
        camera_id=args.camera_id,
        marker_size=args.marker_size,
        marker_to_object_names={
            0: 'tape',
            1: 'target_zone',
            2: 'table'
        }
    )
    
    # Initialize pipeline
    pipeline = RoboPromptSO101Pipeline(
        icl_demos_path=args.icl_demos,
        perception_system=perception,
        robot_port=args.robot_port,
        simulation_mode=args.sim
    )
    
    try:
        # Run inference
        pipeline.run_inference(
            num_icl_demos=args.num_demos,
            speed=args.speed,
            visualize_perception=True
        )
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
