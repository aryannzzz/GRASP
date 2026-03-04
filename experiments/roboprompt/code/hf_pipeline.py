"""
RoboPrompt Pipeline Using HuggingFace Dataset

This script uses pre-recorded demonstrations from HuggingFace instead of
recording from the leader arm.

Usage:
    # Process HuggingFace dataset to ICL format
    python pipeline_with_hf_dataset.py --mode process --episodes 5
    
    # Run RoboPrompt inference
    python pipeline_with_hf_dataset.py --mode inference
    
    # Test single action conversion
    python pipeline_with_hf_dataset.py --mode test
"""

import argparse
import numpy as np
from pathlib import Path
import json
from typing import List, Dict

# Import custom modules
from hf_dataset_loader import HuggingFaceDatasetLoader
from roboprompt_lerobot_bridge import RoboPromptToLeRobotBridge
from integrated_roboprompt_agent import LeRobotRoboPromptAgent


class RoboPromptHFPipeline:
    """
    Pipeline for RoboPrompt using HuggingFace dataset.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize HuggingFace dataset loader
        self.loader = HuggingFaceDatasetLoader(
            dataset_name=config['dataset_name'],
            scene_bounds=config['scene_bounds']
        )
        
        # Initialize bridge for action execution
        self.bridge = RoboPromptToLeRobotBridge(
            robot_port=config['robot_port'],
            scene_bounds=config['scene_bounds'],
            use_sim=config['use_sim']
        )
        
        print("✅ Pipeline initialized")
    
    def process_hf_dataset(
        self,
        episodes_to_use: List[int],
        object_poses: Dict[str, np.ndarray],
        instruction: str
    ) -> str:
        """
        Process HuggingFace dataset and create ICL demonstrations.
        
        Args:
            episodes_to_use: List of episode indices from dataset
            object_poses: Object poses for the task
            instruction: Task instruction
        
        Returns:
            Path to saved ICL demonstrations
        """
        print("\n" + "="*80)
        print("📦 PROCESSING HUGGINGFACE DATASET")
        print("="*80 + "\n")
        
        icl_path = self.loader.save_demonstrations(
            episodes_to_process=episodes_to_use,
            object_poses=object_poses,
            instruction=instruction,
            save_dir=self.config['save_dir']
        )
        
        return icl_path
    
    def run_roboprompt_inference(
        self,
        test_object_poses: Dict[str, np.ndarray],
        mask_id_to_name: Dict[int, str],
        execution_speed: float = 0.5
    ) -> List[np.ndarray]:
        """
        Run RoboPrompt inference to get predicted actions.
        
        NOTE: This requires implementing camera capture for your setup.
        For now, we'll simulate the LLM output.
        
        Args:
            test_object_poses: Current object poses in test scene
            mask_id_to_name: Mapping from mask IDs to object names
            execution_speed: Execution speed
        
        Returns:
            List of predicted actions
        """
        print("\n" + "="*80)
        print("🤖 RUNNING ROBOPROMPT INFERENCE")
        print("="*80 + "\n")
        
        print("⚠️  Note: Real inference requires camera implementation.")
        print("    For now, we'll load example actions from ICL demos.\n")
        
        # Load ICL demonstrations to get example actions
        icl_file = Path(self.config['save_dir']) / "icl_demos.json"
        if icl_file.exists():
            with open(icl_file, 'r') as f:
                icl_data = json.load(f)
            
            # Parse actions from first demonstration
            demo = icl_data['demonstrations'][0]
            
            # Extract actions between >> and ]
            import re
            match = re.search(r'>>\[(.*?)\]', demo)
            if match:
                actions_str = match.group(1)
                # Parse the actions
                actions = eval(f"[{actions_str}]")
                print(f"✓ Loaded {len(actions)} example actions from ICL demo")
                return [np.array(a) for a in actions]
        
        # Fallback: return default actions
        print("⚠️  Using default example actions")
        return [
            np.array([50, 50, 50, 0, 36, 0, 1]),  # Example action
        ]
    
    def test_action_conversion(
        self,
        test_actions: List[np.ndarray]
    ):
        """
        Test converting RoboPrompt actions to SO101 format.
        
        Args:
            test_actions: List of discretized RoboPrompt actions
        """
        print("\n" + "="*80)
        print("🧪 TESTING ACTION CONVERSION")
        print("="*80 + "\n")
        
        for i, action in enumerate(test_actions):
            print(f"\n[Action {i+1}/{len(test_actions)}]")
            print(f"Discretized input: {action}")
            
            # Convert to continuous pose
            position, quaternion, gripper = self.bridge.discretized_to_continuous_pose(action)
            print(f"Continuous pose:")
            print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]m")
            print(f"  Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            print(f"  Gripper: {'OPEN' if gripper == 1 else 'CLOSED'}")
            
            # Convert to joint angles
            joint_angles = self.bridge.continuous_pose_to_joint_angles(position, quaternion)
            if joint_angles is not None:
                print(f"Joint angles:")
                print(f"  [j1, j2, j3, j4, j5] = {np.rad2deg(joint_angles).round(1)}°")
                
                gripper_joint = self.bridge.gripper_state_to_joint(gripper)
                print(f"  j6 (gripper) = {np.rad2deg(gripper_joint):.1f}°")
                
                print("✅ Conversion successful!")
            else:
                print("❌ IK failed - position unreachable")
    
    def execute_actions(
        self,
        actions: List[np.ndarray],
        execution_speed: float = 0.5,
        simulate: bool = True
    ) -> bool:
        """
        Execute RoboPrompt actions on SO101 arm.
        
        Args:
            actions: List of discretized actions
            execution_speed: Execution speed multiplier
            simulate: If True, only simulate execution
        
        Returns:
            success: Whether execution was successful
        """
        print("\n" + "="*80)
        print("🎯 EXECUTING ACTIONS")
        print("="*80 + "\n")
        
        if simulate:
            print("🔄 SIMULATION MODE - No actual robot movement\n")
        
        success = self.bridge.execute_roboprompt_actions(
            actions,
            execution_speed=execution_speed
        )
        
        return success
    
    def visualize_trajectory(
        self,
        actions: List[np.ndarray],
        save_path: str = None
    ):
        """
        Visualize the planned trajectory.
        
        Args:
            actions: List of discretized actions
            save_path: Path to save visualization
        """
        print("\n" + "="*80)
        print("📊 VISUALIZING TRAJECTORY")
        print("="*80 + "\n")
        
        positions = []
        for action in actions:
            position, _, gripper = self.bridge.discretized_to_continuous_pose(action)
            positions.append(position)
        
        positions = np.array(positions)
        
        print(f"Trajectory summary:")
        print(f"  Start: [{positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f}]m")
        print(f"  End:   [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}]m")
        print(f"  Distance: {np.linalg.norm(positions[-1] - positions[0]):.3f}m")
        print(f"  Waypoints: {len(positions)}")
        
        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', linewidth=2, label='Trajectory')
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                      c='g', s=100, marker='o', label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                      c='r', s=100, marker='*', label='End')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('RoboPrompt Predicted Trajectory')
            ax.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 Saved visualization to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️  matplotlib not available for visualization")
    
    def disconnect(self):
        """Cleanup."""
        self.bridge.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="RoboPrompt Pipeline with HuggingFace Dataset"
    )
    parser.add_argument('--mode', type=str, required=True,
                       choices=['process', 'inference', 'test', 'visualize', 'execute'],
                       help='Operation mode')
    parser.add_argument('--dataset', type=str, 
                       default='aadarshram/pick_place_tape',
                       help='HuggingFace dataset name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to use from dataset')
    parser.add_argument('--sim', action='store_true',
                       help='Run in simulation mode')
    parser.add_argument('--speed', type=float, default=0.5,
                       help='Execution speed (0.1 to 2.0)')
    parser.add_argument('--robot-port', type=str, default='/dev/ttyUSB0',
                       help='Robot serial port')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'dataset_name': args.dataset,
        'robot_port': args.robot_port,
        'scene_bounds': [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],  # ADJUST THIS!
        'use_sim': args.sim,
        'save_dir': './roboprompt_data'
    }
    
    # Initialize pipeline
    pipeline = RoboPromptHFPipeline(config)
    
    # Execute based on mode
    if args.mode == 'process':
        # Process HuggingFace dataset
        print("\n📦 Processing HuggingFace dataset...\n")
        
        # Use first N episodes from dataset
        episodes_to_use = list(range(min(args.episodes, pipeline.loader.total_episodes)))
        
        # Define object poses (REPLACE with your perception system)
        object_poses = {
            'tape': np.array([0.3, 0.0, 0.7, 0, 0, 0]),
            'target_zone': np.array([0.5, 0.2, 0.7, 0, 0, 0]),
            'table': np.array([0.0, 0.0, 0.6, 0, 0, 0])
        }
        
        icl_path = pipeline.process_hf_dataset(
            episodes_to_use=episodes_to_use,
            object_poses=object_poses,
            instruction="pick up the tape and place it in the target zone"
        )
        
        print("\n" + "="*80)
        print("✅ DATASET PROCESSING COMPLETE")
        print("="*80)
        print(f"\nICL demonstrations saved to: {icl_path}")
        print("\nNext step: Run inference with --mode inference")
    
    elif args.mode == 'test':
        # Test action conversion
        print("\n🧪 Testing action conversion pipeline...\n")
        
        # Load example actions from ICL demos if available
        icl_file = Path(config['save_dir']) / "icl_demos.json"
        if icl_file.exists():
            with open(icl_file, 'r') as f:
                icl_data = json.load(f)
            
            # Parse actions from first demo
            import re
            demo = icl_data['demonstrations'][0]
            match = re.search(r'>>\[(.*?)\]', demo)
            if match:
                actions_str = match.group(1)
                test_actions = [np.array(a) for a in eval(f"[{actions_str}]")]
                print(f"✓ Loaded {len(test_actions)} actions from ICL demo\n")
            else:
                test_actions = [np.array([50, 50, 50, 0, 36, 0, 1])]
        else:
            print("⚠️  No ICL demos found. Using default test action.\n")
            test_actions = [
                np.array([50, 50, 50, 0, 36, 0, 1]),  # Center, gripper open
                np.array([30, 40, 20, 0, 36, 0, 0]),  # Lower left, gripper closed
            ]
        
        pipeline.test_action_conversion(test_actions)
    
    elif args.mode == 'visualize':
        # Visualize trajectory
        print("\n📊 Visualizing planned trajectory...\n")
        
        # Load actions from ICL demos
        icl_file = Path(config['save_dir']) / "icl_demos.json"
        if not icl_file.exists():
            print("❌ No ICL demos found. Run --mode process first.")
            return
        
        with open(icl_file, 'r') as f:
            icl_data = json.load(f)
        
        # Parse actions
        import re
        demo = icl_data['demonstrations'][0]
        match = re.search(r'>>\[(.*?)\]', demo)
        if match:
            actions_str = match.group(1)
            actions = [np.array(a) for a in eval(f"[{actions_str}]")]
            
            save_path = Path(config['save_dir']) / "trajectory_visualization.png"
            pipeline.visualize_trajectory(actions, save_path=str(save_path))
        else:
            print("❌ Failed to parse actions from ICL demo")
    
    elif args.mode == 'inference':
        # Run RoboPrompt inference
        print("\n🤖 Running RoboPrompt inference...\n")
        
        # Test object poses (REPLACE with perception system)
        test_object_poses = {
            'tape': np.array([0.3, 0.0, 0.7, 0, 0, 0]),
            'target_zone': np.array([0.5, 0.2, 0.7, 0, 0, 0]),
            'table': np.array([0.0, 0.0, 0.6, 0, 0, 0])
        }
        
        mask_id_to_name = {
            1: 'tape',
            2: 'target_zone',
            3: 'table'
        }
        
        # Get predicted actions
        actions = pipeline.run_roboprompt_inference(
            test_object_poses=test_object_poses,
            mask_id_to_name=mask_id_to_name,
            execution_speed=args.speed
        )
        
        print(f"\n📋 RoboPrompt predicted {len(actions)} actions:")
        for i, action in enumerate(actions):
            print(f"  [{i}] {action}")
        
        print("\nNext step: Run with --mode execute to execute on robot")
    
    elif args.mode == 'execute':
        # Execute actions on robot
        print("\n🎯 Executing actions on SO101 arm...\n")
        
        # Load actions from ICL demos
        icl_file = Path(config['save_dir']) / "icl_demos.json"
        if not icl_file.exists():
            print("❌ No ICL demos found. Run --mode process first.")
            return
        
        with open(icl_file, 'r') as f:
            icl_data = json.load(f)
        
        # Parse actions
        import re
        demo = icl_data['demonstrations'][0]
        match = re.search(r'>>\[(.*?)\]', demo)
        if match:
            actions_str = match.group(1)
            actions = [np.array(a) for a in eval(f"[{actions_str}]")]
            
            # Execute
            success = pipeline.execute_actions(
                actions,
                execution_speed=args.speed,
                simulate=args.sim
            )
            
            if success:
                print("\n" + "="*80)
                print("🎉 EXECUTION SUCCESSFUL!")
                print("="*80)
            else:
                print("\n" + "="*80)
                print("❌ EXECUTION FAILED")
                print("="*80)
        else:
            print("❌ Failed to parse actions from ICL demo")
    
    # Cleanup
    pipeline.disconnect()


if __name__ == "__main__":
    main()
