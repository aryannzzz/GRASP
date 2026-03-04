"""
RoboPrompt for SO101 - Main Pipeline
Orchestrates the complete workflow from dataset to robot execution.
"""

import numpy as np
from typing import List, Dict
from pathlib import Path
import yaml
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


# Import all modules
from forward_kinematics import ForwardKinematics
from keyframe_extractor import KeyframeExtractor
from action_discretizer import ActionDiscretizer
from pose_estimator import PoseEstimator
from icl_constructor import ICLConstructor
from llm_interface import LLMInterface
from lerobot_to_roboprompt import LeRobotDatasetConverter
from action_executor import ActionExecutor


console = Console()


class RoboPromptPipeline:
    """
    Complete RoboPrompt pipeline for SO101.
    
    Pipeline stages:
    1. [Offline] Load and convert LeRobot dataset
    2. [Offline] Extract keyframes and compute EE poses
    3. [Offline] Estimate object poses
    4. [Offline] Format as ICL demonstrations
    5. [Inference] Capture test image and estimate object poses
    6. [Inference] Construct ICL prompt with demonstrations
    7. [Inference] Query GPT-4 for action predictions
    8. [Inference] Convert predicted actions to joint commands
    9. [Inference] Execute on robot
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        console.print("[bold blue]Initializing RoboPrompt Pipeline...[/bold blue]")
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize all components
        self._init_components()
        
        # Storage for demonstrations
        self.demonstrations = None
        
        console.print("[bold green]✓ Pipeline initialized successfully![/bold green]\n")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        console.print("Initializing components...")
        
        # Forward kinematics
        urdf_path = Path(self.config["robot"]["urdf_path"])
        self.fk_solver = ForwardKinematics(
            str(urdf_path),
            end_effector_link=self.config["robot"]["end_effector_link"]
        )
        
        # Keyframe extractor
        self.keyframe_extractor = KeyframeExtractor(self.config)
        
        # Action discretizer
        self.discretizer = ActionDiscretizer(self.config)
        
        # Pose estimator
        self.pose_estimator = PoseEstimator(self.config)
        
        # ICL constructor
        self.icl_constructor = ICLConstructor(self.config, self.discretizer)
        
        # LLM interface
        self.llm_interface = LLMInterface(self.config)
        
        # Dataset converter
        self.dataset_converter = LeRobotDatasetConverter(
            self.config,
            self.fk_solver,
            self.keyframe_extractor,
            self.discretizer,
            self.pose_estimator
        )
        
        # Action executor
        self.action_executor = ActionExecutor(
            self.config,
            self.discretizer,
            self.fk_solver
        )
    
    def run_offline_processing(self, 
                              task_instruction: str = None,
                              num_episodes: int = None,
                              save_path: str = None):
        """
        Run offline processing stage: convert dataset to demonstrations.
        
        Args:
            task_instruction: Task instruction text
            num_episodes: Number of episodes to process
            save_path: Path to save processed demonstrations
        """
        console.print("\n[bold blue]Starting Offline Processing...[/bold blue]")
        
        # Convert dataset
        self.demonstrations = self.dataset_converter.convert_dataset(
            task_instruction=task_instruction,
            num_episodes=num_episodes
        )
        
        # Build action executor database
        console.print("\nBuilding action executor database...")
        self.action_executor.build_database(self.demonstrations)
        
        # Save if requested
        if save_path:
            self.dataset_converter.save_demonstrations(
                self.demonstrations,
                save_path
            )
        
        console.print("[bold green]✓ Offline processing complete![/bold green]")
        
        return self.demonstrations
    
    def load_demonstrations(self, path: str):
        """
        Load pre-processed demonstrations from disk.
        
        Args:
            path: Path to saved demonstrations
        """
        console.print(f"\nLoading demonstrations from {path}...")
        self.demonstrations = self.dataset_converter.load_demonstrations(path)
        
        # Build action executor database
        self.action_executor.build_database(self.demonstrations)
        
        console.print("[bold green]✓ Demonstrations loaded![/bold green]")
    
    def run_inference(self,
                     test_image: np.ndarray,
                     task_instruction: str,
                     num_demonstrations: int = None) -> Dict:
        """
        Run inference stage: predict and execute actions for test input.
        
        Args:
            test_image: Test RGB image [H, W, 3]
            task_instruction: Task instruction for test
            num_demonstrations: Number of demonstrations to use (default: from config)
            
        Returns:
            result: Dictionary with predictions and execution info
        """
        console.print("\n[bold blue]Starting Inference...[/bold blue]")
        
        if self.demonstrations is None:
            console.print("[bold red]Error: No demonstrations loaded![/bold red]")
            console.print("Run offline processing or load demonstrations first.")
            return None
        
        # Step 1: Estimate object poses from test image
        console.print("\n1. Estimating object poses from test image...")
        object_poses = self.pose_estimator.estimate_poses(test_image)
        console.print(f"   Detected objects: {list(object_poses.keys())}")
        
        # Discretize object poses
        object_poses_discrete = {}
        for obj_name, pose in object_poses.items():
            discrete_pose = self.discretizer.discretize_pose(pose)
            object_poses_discrete[obj_name] = discrete_pose
        
        # Step 2: Construct ICL prompt
        console.print("\n2. Constructing ICL prompt...")
        
        # Select demonstrations
        if num_demonstrations is None:
            num_demonstrations = self.config['icl']['num_demonstrations']
        
        selected_demos = self.demonstrations[:num_demonstrations]
        
        # Prepare test data
        test_data = {
            'object_poses': object_poses_discrete,
            'instruction': task_instruction
        }
        
        # Construct prompt
        system_msg, user_prompt = self.icl_constructor.construct_icl_prompt_with_system(
            selected_demos,
            test_data
        )
        
        console.print(f"   Using {len(selected_demos)} demonstrations")
        console.print(f"   Prompt length: {len(user_prompt)} characters")
        
        # Step 3: Query LLM
        console.print("\n3. Querying GPT-4 for action predictions...")
        response = self.llm_interface.query(system_msg, user_prompt)
        
        console.print(f"   Received response ({len(response)} characters)")
        
        # Parse response
        predicted_actions, predicted_grippers = self.icl_constructor.parse_llm_response(response)
        
        if not predicted_actions:
            console.print("[bold red]   Error: Failed to parse LLM response![/bold red]")
            return None
        
        console.print(f"   Predicted {len(predicted_actions)} actions")
        
        # Step 4: Convert to joint commands
        console.print("\n4. Converting predicted actions to joint commands...")
        joint_trajectory = self.action_executor.execute_action_sequence(
            predicted_actions,
            predicted_grippers
        )
        
        console.print(f"   Generated trajectory with {len(joint_trajectory)} waypoints")
        
        # Step 5: Execute (or simulate)
        console.print("\n5. Executing trajectory...")
        if self.config['execution']['use_simulation']:
            console.print("   [Simulation mode - not sending to real robot]")
        else:
            console.print("   Sending commands to robot...")
        
        try:
            self.action_executor.send_to_robot(joint_trajectory)
            console.print("[bold green]   ✓ Execution complete![/bold green]")
        except NotImplementedError:
            console.print("   [Note: Robot interface not yet implemented]")
        
        # Prepare result
        result = {
            'test_image': test_image,
            'object_poses': object_poses,
            'object_poses_discrete': object_poses_discrete,
            'predicted_actions': predicted_actions,
            'predicted_grippers': predicted_grippers,
            'joint_trajectory': joint_trajectory,
            'llm_response': response,
            'system_message': system_msg,
            'user_prompt': user_prompt
        }
        
        console.print("\n[bold green]✓ Inference complete![/bold green]")
        
        return result
    
    def visualize_result(self, result: Dict, save_dir: str = None):
        """
        Visualize inference result.
        
        Args:
            result: Result dictionary from run_inference
            save_dir: Directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save object pose visualization
            pose_vis = self.pose_estimator.visualize_poses(
                result['test_image'],
                result['object_poses'],
                save_path=str(save_dir / "detected_poses.png")
            )
            
            # Save trajectory plot
            # TODO: Add trajectory visualization
            
            console.print(f"\n[bold green]✓ Visualizations saved to {save_dir}[/bold green]")


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(description="RoboPrompt for SO101")
    parser.add_argument(
        '--mode',
        choices=['offline', 'inference', 'full'],
        default='full',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--task',
        default='Pick up the tape and place it on the target',
        help='Task instruction'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Number of episodes to process (offline mode)'
    )
    parser.add_argument(
        '--demo-path',
        default='output/demonstrations/processed_demonstrations.pkl',
        help='Path to save/load demonstrations'
    )
    parser.add_argument(
        '--test-image',
        default=None,
        help='Path to test image (inference mode)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RoboPromptPipeline(args.config)
    
    # Run based on mode
    if args.mode in ['offline', 'full']:
        # Offline processing
        pipeline.run_offline_processing(
            task_instruction=args.task,
            num_episodes=args.num_episodes,
            save_path=args.demo_path
        )
    
    if args.mode in ['inference', 'full']:
        # Load demonstrations if not already loaded
        if pipeline.demonstrations is None:
            pipeline.load_demonstrations(args.demo_path)
        
        # Load test image
        if args.test_image:
            import cv2
            test_image = cv2.imread(args.test_image)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        else:
            # Use dummy image for testing
            console.print("\n[yellow]Warning: No test image provided, using dummy image[/yellow]")
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run inference
        result = pipeline.run_inference(
            test_image,
            task_instruction=args.task
        )
        
        # Visualize
        if result:
            pipeline.visualize_result(result, save_dir='output/results')
    
    console.print("\n[bold blue]Pipeline execution complete![/bold blue]")


if __name__ == "__main__":
    main()
