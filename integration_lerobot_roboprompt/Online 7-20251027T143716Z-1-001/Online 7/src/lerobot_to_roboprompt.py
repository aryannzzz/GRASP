"""
LeRobot to RoboPrompt Dataset Converter
Loads LeRobot dataset and converts it to RoboPrompt format with keyframes and discretized actions.
"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import yaml
from tqdm import tqdm
import pickle


class LeRobotDatasetConverter:
    """
    Converts LeRobot dataset to RoboPrompt format.
    
    Pipeline:
    1. Load LeRobot dataset
    2. Extract keyframes from each episode
    3. Compute forward kinematics (joint -> EE poses)
    4. Discretize poses
    5. Estimate object poses from first frame
    6. Format as ICL demonstrations
    """
    
    def __init__(self, config: Dict, fk_solver, keyframe_extractor, 
                 discretizer, pose_estimator):
        """
        Initialize converter.
        
        Args:
            config: Configuration dictionary
            fk_solver: ForwardKinematics instance
            keyframe_extractor: KeyframeExtractor instance
            discretizer: ActionDiscretizer instance
            pose_estimator: PoseEstimator instance
        """
        self.config = config
        self.fk_solver = fk_solver
        self.keyframe_extractor = keyframe_extractor
        self.discretizer = discretizer
        self.pose_estimator = pose_estimator
        
        self.dataset_name = config['dataset']['name']
        self.local_path = Path(config['dataset']['local_path'])
        
        print(f"Initialized LeRobotDatasetConverter:")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Local path: {self.local_path}")
    
    def load_dataset(self):
        """
        Load LeRobot dataset from HuggingFace.
        
        Returns:
            dataset: Loaded dataset
        """
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            # Load dataset
            print(f"Loading dataset: {self.dataset_name}")
            dataset = LeRobotDataset(self.dataset_name, root=str(self.local_path))
            
            print(f"Loaded dataset with {len(dataset)} frames")
            print(f"Number of episodes: {dataset.num_episodes}")
            
            return dataset
            
        except ImportError:
            print("Error: lerobot library not installed!")
            print("Install with: pip install lerobot")
            return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def process_episode(self, 
                       episode_data: Dict,
                       episode_index: int) -> Dict:
        """
        Process a single episode to RoboPrompt format.
        
        Args:
            episode_data: Episode data dictionary with:
                - 'joint_positions': [T, 6]
                - 'gripper_states': [T,]
                - 'images': [T, H, W, 3]
                - 'timestamps': [T,]
            episode_index: Episode index
            
        Returns:
            processed_episode: Dictionary with RoboPrompt formatted data
        """
        # Extract keyframes
        keyframe_indices = self.keyframe_extractor.extract_keyframes(
            episode_data['joint_positions'],
            episode_data['gripper_states']
        )
        
        # Get keyframe data
        kf_joint_positions = episode_data['joint_positions'][keyframe_indices]
        kf_gripper_states = episode_data['gripper_states'][keyframe_indices]
        kf_images = [episode_data['images'][i] for i in keyframe_indices]
        
        # Compute forward kinematics to get EE poses
        ee_poses = self.fk_solver.compute_fk_full(kf_joint_positions)
        
        # Discretize EE poses
        discrete_poses = self.discretizer.discretize_pose(ee_poses)
        discrete_grippers = self.discretizer.discretize_gripper(kf_gripper_states)
        
        # Estimate object poses from first frame
        first_image = episode_data['images'][0]
        object_poses_continuous = self.pose_estimator.estimate_poses(first_image)
        
        # Discretize object poses
        object_poses_discrete = {}
        for obj_name, pose in object_poses_continuous.items():
            discrete_pose = self.discretizer.discretize_pose(pose)
            object_poses_discrete[obj_name] = discrete_pose
        
        # Skip first keyframe action (as per RoboPrompt paper)
        actions = discrete_poses[1:]
        grippers = discrete_grippers[1:]
        
        processed_episode = {
            'episode_index': episode_index,
            'object_poses': object_poses_discrete,
            'actions': actions,
            'grippers': grippers.tolist(),
            'keyframe_indices': keyframe_indices,
            'num_keyframes': len(keyframe_indices),
            'original_length': len(episode_data['joint_positions']),
            'first_image': first_image,
            'keyframe_images': kf_images
        }
        
        return processed_episode
    
    def convert_dataset(self, 
                       task_instruction: str = None,
                       num_episodes: int = None) -> List[Dict]:
        """
        Convert entire dataset to RoboPrompt format.
        
        Args:
            task_instruction: Task instruction text (if None, uses default)
            num_episodes: Number of episodes to process (if None, process all)
            
        Returns:
            demonstrations: List of demonstration dictionaries
        """
        # Load dataset
        dataset = self.load_dataset()
        if dataset is None:
            return []
        
        # Default task instruction
        if task_instruction is None:
            task_instruction = "Pick up the tape and place it on the target"
        
        # Determine number of episodes to process
        total_episodes = dataset.num_episodes
        if num_episodes is None:
            num_episodes = total_episodes
        else:
            num_episodes = min(num_episodes, total_episodes)
        
        print(f"\nProcessing {num_episodes} episodes...")
        
        demonstrations = []
        
        # Process each episode
        for ep_idx in tqdm(range(num_episodes)):
            # Get episode data
            episode_data = self._get_episode_data(dataset, ep_idx)
            
            # Process episode
            processed_ep = self.process_episode(episode_data, ep_idx)
            
            # Add task instruction
            processed_ep['instruction'] = task_instruction
            
            demonstrations.append(processed_ep)
        
        print(f"Processed {len(demonstrations)} demonstrations")
        
        # Print statistics
        self._print_statistics(demonstrations)
        
        return demonstrations
    
    def _get_episode_data(self, dataset, episode_index: int) -> Dict:
        """
        Extract episode data from LeRobot dataset.
        
        Args:
            dataset: LeRobot dataset
            episode_index: Episode index
            
        Returns:
            episode_data: Dictionary with episode data
        """
        # Get episode frame indices
        episode_frame_indices = dataset.episode_data_index['from'][episode_index].item()
        episode_length = dataset.episode_data_index['to'][episode_index].item() - episode_frame_indices
        
        # Load frames for this episode
        joint_positions = []
        gripper_states = []
        images = []
        timestamps = []
        
        for i in range(episode_length):
            frame_idx = episode_frame_indices + i
            frame = dataset[frame_idx]
            
            # Extract data
            joint_pos = frame['observation.state'].numpy()
            gripper = frame['action'].numpy()[-1]  # Last dimension is gripper
            
            # Get image (if available)
            if 'observation.images.top_phone' in frame:
                image = frame['observation.images.top_phone'].numpy()
                images.append(image)
            
            timestamp = frame['timestamp'].item()
            
            joint_positions.append(joint_pos)
            gripper_states.append(gripper)
            timestamps.append(timestamp)
        
        episode_data = {
            'joint_positions': np.array(joint_positions),
            'gripper_states': np.array(gripper_states),
            'images': images,
            'timestamps': np.array(timestamps)
        }
        
        return episode_data
    
    def _print_statistics(self, demonstrations: List[Dict]):
        """Print statistics about converted demonstrations."""
        num_keyframes = [d['num_keyframes'] for d in demonstrations]
        original_lengths = [d['original_length'] for d in demonstrations]
        compression_ratios = [d['num_keyframes'] / d['original_length'] 
                             for d in demonstrations]
        
        print("\nDataset Statistics:")
        print(f"  Total demonstrations: {len(demonstrations)}")
        print(f"  Avg keyframes per episode: {np.mean(num_keyframes):.1f} ± {np.std(num_keyframes):.1f}")
        print(f"  Min/Max keyframes: {np.min(num_keyframes)}/{np.max(num_keyframes)}")
        print(f"  Avg original length: {np.mean(original_lengths):.1f}")
        print(f"  Avg compression ratio: {np.mean(compression_ratios):.2%}")
    
    def save_demonstrations(self, demonstrations: List[Dict], 
                           output_path: str):
        """
        Save processed demonstrations to disk.
        
        Args:
            demonstrations: List of demonstration dictionaries
            output_path: Path to save file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save without images (too large)
        demos_to_save = []
        for demo in demonstrations:
            demo_copy = demo.copy()
            # Remove images for now (save separately if needed)
            demo_copy.pop('first_image', None)
            demo_copy.pop('keyframe_images', None)
            demos_to_save.append(demo_copy)
        
        with open(output_path, 'wb') as f:
            pickle.dump(demos_to_save, f)
        
        print(f"Saved demonstrations to {output_path}")
    
    def load_demonstrations(self, input_path: str) -> List[Dict]:
        """
        Load demonstrations from disk.
        
        Args:
            input_path: Path to saved file
            
        Returns:
            demonstrations: List of demonstration dictionaries
        """
        with open(input_path, 'rb') as f:
            demonstrations = pickle.load(f)
        
        print(f"Loaded {len(demonstrations)} demonstrations from {input_path}")
        return demonstrations


def convert_dataset_main():
    """Main function to convert dataset."""
    from forward_kinematics import ForwardKinematics
    from keyframe_extractor import KeyframeExtractor
    from action_discretizer import ActionDiscretizer
    from pose_estimator import PoseEstimator
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    print("Initializing components...")
    urdf_path = Path(__file__).parent.parent / config["robot"]["urdf_path"]
    fk_solver = ForwardKinematics(str(urdf_path))
    keyframe_extractor = KeyframeExtractor(config)
    discretizer = ActionDiscretizer(config)
    pose_estimator = PoseEstimator(config)
    
    # Create converter
    converter = LeRobotDatasetConverter(
        config, fk_solver, keyframe_extractor, 
        discretizer, pose_estimator
    )
    
    # Convert dataset
    demonstrations = converter.convert_dataset(
        task_instruction="Pick up the tape and place it on the target",
        num_episodes=None  # Process all episodes
    )
    
    # Save demonstrations
    output_path = Path(__file__).parent.parent / "output" / "demonstrations" / "processed_demonstrations.pkl"
    converter.save_demonstrations(demonstrations, str(output_path))
    
    print("\nDataset conversion complete!")
    
    return demonstrations


if __name__ == "__main__":
    convert_dataset_main()
