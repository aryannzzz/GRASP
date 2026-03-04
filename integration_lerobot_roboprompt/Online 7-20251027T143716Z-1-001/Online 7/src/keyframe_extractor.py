"""
Keyframe Extraction Module
Identifies critical frames in robot episodes based on velocity and gripper state changes.
Following RoboPrompt paper Section IV-A.
"""

import numpy as np
from typing import List, Tuple, Dict
import yaml
from pathlib import Path


class KeyframeExtractor:
    """
    Extracts keyframes from robot episodes based on:
    1. Near-zero joint velocities (direction changes)
    2. Gripper state changes (object interactions)
    
    Following equation (2) from RoboPrompt paper:
    ||S_tk||_2 < δ or A^gripper_tk != A^gripper_{tk+1}
    """
    
    def __init__(self, config: Dict):
        """
        Initialize keyframe extractor.
        
        Args:
            config: Configuration dictionary with keyframe parameters
        """
        keyframe_config = config['keyframes']
        
        self.velocity_threshold = keyframe_config['velocity_threshold']
        self.method = keyframe_config['method']
        
        print(f"Initialized KeyframeExtractor:")
        print(f"  Velocity threshold (δ): {self.velocity_threshold}")
        print(f"  Method: {self.method}")
    
    def extract_keyframes(self, 
                         joint_positions: np.ndarray,
                         gripper_states: np.ndarray,
                         timestamps: np.ndarray = None) -> List[int]:
        """
        Extract keyframe indices from an episode.
        
        Args:
            joint_positions: Joint positions over time [T, 6]
            gripper_states: Gripper states over time [T,]
            timestamps: Optional timestamps [T,]
            
        Returns:
            keyframe_indices: List of keyframe indices
        """
        T = len(joint_positions)
        keyframe_indices = [0]  # Always include first frame
        
        # Compute joint velocities (finite differences)
        velocities = self._compute_velocities(joint_positions)
        velocity_norms = np.linalg.norm(velocities, axis=1)
        
        # Detect gripper state changes
        gripper_changes = self._detect_gripper_changes(gripper_states)
        
        # Extract keyframes based on method
        for t in range(1, T - 1):  # Skip first and last frame for now
            is_keyframe = False
            
            # Check velocity criterion
            if self.method in ['velocity_only', 'velocity_and_gripper']:
                if velocity_norms[t] < self.velocity_threshold:
                    is_keyframe = True
            
            # Check gripper criterion
            if self.method in ['gripper_only', 'velocity_and_gripper']:
                if gripper_changes[t]:
                    is_keyframe = True
            
            if is_keyframe:
                # Avoid adding consecutive keyframes (reduce redundancy)
                if t - keyframe_indices[-1] > 1:
                    keyframe_indices.append(t)
        
        # Always include last frame
        if keyframe_indices[-1] != T - 1:
            keyframe_indices.append(T - 1)
        
        return keyframe_indices
    
    def extract_keyframes_batch(self,
                               episodes_data: List[Dict]) -> List[List[int]]:
        """
        Extract keyframes from multiple episodes.
        
        Args:
            episodes_data: List of episode dictionaries with 'joint_positions',
                          'gripper_states', and optionally 'timestamps'
            
        Returns:
            keyframes_list: List of keyframe indices for each episode
        """
        keyframes_list = []
        
        for episode in episodes_data:
            keyframes = self.extract_keyframes(
                episode['joint_positions'],
                episode['gripper_states'],
                episode.get('timestamps', None)
            )
            keyframes_list.append(keyframes)
        
        return keyframes_list
    
    def _compute_velocities(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute joint velocities using finite differences.
        
        Args:
            joint_positions: Joint positions [T, 6]
            
        Returns:
            velocities: Joint velocities [T, 6]
        """
        # Use central differences for interior points
        velocities = np.zeros_like(joint_positions)
        
        # Forward difference for first point
        velocities[0] = joint_positions[1] - joint_positions[0]
        
        # Central differences for interior points
        velocities[1:-1] = (joint_positions[2:] - joint_positions[:-2]) / 2.0
        
        # Backward difference for last point
        velocities[-1] = joint_positions[-1] - joint_positions[-2]
        
        return velocities
    
    def _detect_gripper_changes(self, gripper_states: np.ndarray) -> np.ndarray:
        """
        Detect when gripper state changes.
        
        Args:
            gripper_states: Gripper states [T,]
            
        Returns:
            changes: Boolean array indicating changes [T,]
        """
        # Binarize gripper states (threshold at 0.5)
        binary_gripper = (gripper_states > 0.5).astype(int)
        
        # Detect changes
        changes = np.zeros(len(binary_gripper), dtype=bool)
        changes[1:] = binary_gripper[1:] != binary_gripper[:-1]
        
        return changes
    
    def visualize_keyframes(self,
                           joint_positions: np.ndarray,
                           gripper_states: np.ndarray,
                           keyframe_indices: List[int],
                           save_path: str = None):
        """
        Visualize keyframes on joint trajectory plot.
        
        Args:
            joint_positions: Joint positions [T, 6]
            gripper_states: Gripper states [T,]
            keyframe_indices: List of keyframe indices
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        T = len(joint_positions)
        time = np.arange(T)
        
        # Compute velocities
        velocities = self._compute_velocities(joint_positions)
        velocity_norms = np.linalg.norm(velocities, axis=1)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Plot joint positions
        for i in range(6):
            axes[0].plot(time, joint_positions[:, i], label=f'Joint {i+1}', alpha=0.7)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        for kf in keyframe_indices:
            axes[0].axvline(x=kf, color='r', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Joint Position (rad)')
        axes[0].legend(ncol=6, fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Keyframe Extraction Visualization')
        
        # Plot velocity norms
        axes[1].plot(time, velocity_norms, color='blue', linewidth=2)
        axes[1].axhline(y=self.velocity_threshold, color='orange', 
                       linestyle='--', label=f'Threshold (δ={self.velocity_threshold})')
        for kf in keyframe_indices:
            axes[1].axvline(x=kf, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Velocity Norm')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot gripper states
        axes[2].plot(time, gripper_states, color='green', linewidth=2)
        axes[2].axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
        for kf in keyframe_indices:
            axes[2].axvline(x=kf, color='r', linestyle='--', alpha=0.5, 
                          label='Keyframe' if kf == keyframe_indices[0] else '')
        axes[2].set_ylabel('Gripper State')
        axes[2].set_xlabel('Time Step')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved keyframe visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_keyframe_statistics(self, keyframes_list: List[List[int]],
                               episodes_lengths: List[int]) -> Dict:
        """
        Compute statistics about keyframe extraction.
        
        Args:
            keyframes_list: List of keyframe indices for each episode
            episodes_lengths: List of episode lengths
            
        Returns:
            stats: Dictionary with statistics
        """
        num_keyframes = [len(kf) for kf in keyframes_list]
        compression_ratios = [len(kf) / length for kf, length 
                             in zip(keyframes_list, episodes_lengths)]
        
        stats = {
            'total_episodes': len(keyframes_list),
            'avg_keyframes_per_episode': np.mean(num_keyframes),
            'std_keyframes_per_episode': np.std(num_keyframes),
            'min_keyframes': np.min(num_keyframes),
            'max_keyframes': np.max(num_keyframes),
            'avg_compression_ratio': np.mean(compression_ratios),
            'avg_original_length': np.mean(episodes_lengths),
            'avg_compressed_length': np.mean(num_keyframes)
        }
        
        return stats


def test_keyframe_extractor():
    """Test keyframe extraction with synthetic data."""
    import yaml
    import matplotlib.pyplot as plt
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create extractor
    extractor = KeyframeExtractor(config)
    
    # Generate synthetic episode
    T = 300
    t = np.linspace(0, 4*np.pi, T)
    
    # Joint positions with pauses (near-zero velocity regions)
    joint_positions = np.zeros((T, 6))
    for i in range(6):
        joint_positions[:, i] = np.sin(t * (i+1) / 2) * 0.5
        # Add plateaus (near-zero velocity)
        joint_positions[50:70, i] = joint_positions[50, i]
        joint_positions[150:170, i] = joint_positions[150, i]
        joint_positions[250:270, i] = joint_positions[250, i]
    
    # Gripper states with changes
    gripper_states = np.zeros(T)
    gripper_states[100:200] = 1.0  # Closed
    
    # Extract keyframes
    keyframes = extractor.extract_keyframes(joint_positions, gripper_states)
    
    print(f"\nExtracted {len(keyframes)} keyframes from {T} frames")
    print(f"Keyframe indices: {keyframes}")
    print(f"Compression ratio: {len(keyframes)/T:.2%}")
    
    # Visualize
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "keyframe_test.png"
    extractor.visualize_keyframes(joint_positions, gripper_states, 
                                 keyframes, str(save_path))


if __name__ == "__main__":
    test_keyframe_extractor()
