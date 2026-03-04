"""
Simplified LeRobot to RoboPrompt converter - No LeRobot dependency needed!
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

# Just use HuggingFace datasets directly
from datasets import load_dataset

@dataclass
class RoboPromptAction:
    """Single RoboPrompt action (discretized)."""
    position_bins: np.ndarray  # [x, y, z] bins (0-99)
    rotation_bins: np.ndarray  # [rx, ry, rz] bins (0-71)
    gripper_binary: int  # 0 or 1
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.position_bins,
            self.rotation_bins,
            [self.gripper_binary]
        ])

class SimpleLeRobotConverter:
    """
    Direct HuggingFace dataset to RoboPrompt converter - no lerobot needed!
    """
    
    def __init__(self, repo_id: str, scene_bounds: List[float] = None):
        self.repo_id = repo_id
        
        # Load dataset directly with HuggingFace datasets
        print(f"📦 Loading dataset directly: {repo_id}")
        self.dataset = load_dataset(repo_id)
        
        # Usually the dataset has 'train' split
        if 'train' in self.dataset:
            self.data = self.dataset['train']
        else:
            self.data = self.dataset[list(self.dataset.keys())[0]]
        
        # Scene bounds for discretization
        if scene_bounds is None:
            self.scene_bounds = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
        else:
            self.scene_bounds = np.array(scene_bounds)
        
        print(f"✓ Loaded {len(self.data)} samples")
        print(f"  Available keys: {list(self.data[0].keys())}")
    
    def extract_episodes_simple(self) -> List[Dict]:
        """
        Simple episode extraction - assuming dataset is already episodic
        or we can use episode_id field if available.
        """
        episodes = []
        
        # Look for episode segmentation
        if 'episode_index' in self.data.column_names:
            # Group by episode
            episode_indices = set(self.data['episode_index'])
            for ep_idx in episode_indices:
                episode_data = self.data.filter(lambda x: x['episode_index'] == ep_idx)
                episodes.append({
                    'episode_index': ep_idx,
                    'states': np.array(episode_data['observation.state']),
                    'actions': np.array(episode_data['action']),
                    'length': len(episode_data)
                })
        else:
            # Treat entire dataset as one episode (or implement custom segmentation)
            episodes.append({
                'episode_index': 0,
                'states': np.array(self.data['observation.state']),
                'actions': np.array(self.data['action']),
                'length': len(self.data)
            })
        
        return episodes
    
    # Keep the same conversion methods as before...
    def joint_state_to_ee_pose(self, joint_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Same FK implementation as before
        L1 = 0.1025
        L2 = 0.11257  
        L3 = 0.1349
        L4 = 0.0601
        base_height = 0.0165
        
        j1, j2, j3, j4, j5 = joint_state[:5]
        
        r = L1 * np.cos(j2) + L2 * np.cos(j2 + j3) + L3 * np.cos(j2 + j3 + j4) + L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = base_height + L1 * np.sin(j2) + L2 * np.sin(j2 + j3) + L3 * np.sin(j2 + j3 + j4)
        
        position = np.array([x, y, z])
        
        pitch = j2 + j3 + j4
        roll = j5
        yaw = 0.0
        
        rotation = R.from_euler('xyz', [yaw, pitch, roll])
        quaternion = rotation.as_quat()
        
        return position, quaternion
    
    def discretize_pose_roboprompt(self, position: np.ndarray, quaternion: np.ndarray, gripper_state: float) -> RoboPromptAction:
        # Same discretization as before
        pos_normalized = (position - self.scene_bounds[:3]) / (self.scene_bounds[3:] - self.scene_bounds[:3])
        pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
        
        rotation = R.from_quat(quaternion)
        euler_rad = rotation.as_euler('xyz')
        euler_deg = np.degrees(euler_rad)
        
        euler_normalized = (euler_deg % 360) / 360
        rot_bins = (euler_normalized * 72).astype(int) % 72
        
        gripper_binary = 1 if gripper_state > 15.0 else 0
        
        return RoboPromptAction(
            position_bins=pos_bins,
            rotation_bins=rot_bins,
            gripper_binary=gripper_binary
        )

    def create_icl_demonstrations_simple(self, object_poses: Dict[str, np.ndarray], instruction: str, save_dir: str = "./roboprompt_data") -> str:
        """
        Simplified ICL creation without lerobot dependency.
        """
        print("🔧 CREATING ICL DEMONSTRATIONS (SIMPLIFIED)")
        
        episodes = self.extract_episodes_simple()
        icl_demos = []
        
        for episode in episodes[:5]:  # Use first 5 episodes
            print(f"Processing episode {episode['episode_index']} with {episode['length']} frames")
            
            # Simple keyframe extraction: just sample every N frames
            keyframe_indices = list(range(0, episode['length'], max(1, episode['length'] // 10)))
            
            actions = []
            for kf_idx in keyframe_indices:
                joint_state = episode['states'][kf_idx]
                position, quaternion = self.joint_state_to_ee_pose(joint_state)
                gripper_state = joint_state[-1]
                action = self.discretize_pose_roboprompt(position, quaternion, gripper_state)
                actions.append(action)
            
            # Format as ICL demonstration
            icl_demo = self._format_icl_demo(actions, object_poses, instruction)
            icl_demos.append(icl_demo)
        
        # Save to file
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        icl_file = save_path / "icl_demos_simple.json"
        
        icl_data = {
            'repo_id': self.repo_id,
            'task': instruction,
            'demonstrations': icl_demos,
            'object_names': list(object_poses.keys()),
            'scene_bounds': self.scene_bounds.tolist(),
        }
        
        with open(icl_file, 'w') as f:
            json.dump(icl_data, f, indent=2)
        
        print(f"💾 Saved {len(icl_demos)} ICL demonstrations to {icl_file}")
        return str(icl_file)
    
    def _format_icl_demo(self, actions: List[RoboPromptAction], object_poses: Dict[str, np.ndarray], instruction: str) -> str:
        # Same formatting as before
        obs_parts = []
        for obj_name, pose in object_poses.items():
            position = pose[:3]
            pos_normalized = (position - self.scene_bounds[:3]) / (self.scene_bounds[3:] - self.scene_bounds[:3])
            pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
            obs_parts.append(f"'{obj_name}': [{pos_bins[0]}, {pos_bins[1]}, {pos_bins[2]}]")
        
        obs_str = "{" + ", ".join(obs_parts) + "}"
        action_arrays = [a.to_array().tolist() for a in actions]
        actions_str = "[" + ", ".join([str(a) for a in action_arrays]) + "]"
        
        return f"{{{obs_str}, '{instruction}'}}>>{actions_str}"

def main_simple():
    """Simplified example without lerobot dependency."""
    print("=" * 80)
    print("SIMPLIFIED LEROBOT → ROBOPROMPT CONVERTER")
    print("=" * 80)
    
    # Initialize simple converter
    converter = SimpleLeRobotConverter(
        repo_id="aadarshram/pick_place_tape",
        scene_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    )
    
    # Approximate object poses
    object_poses = {
        'tape': np.array([0.3, 0.0, 0.7, 0, 0, 0]),
        'target_zone': np.array([0.5, 0.2, 0.7, 0, 0, 0]),
    }
    
    # Create ICL demonstrations
    icl_path = converter.create_icl_demonstrations_simple(
        object_poses=object_poses,
        instruction="pick up the tape and place it in the target zone",
        save_dir="./roboprompt_data"
    )
    
    print(f"\n✅ Created ICL file: {icl_path}")

if __name__ == "__main__":
    main_simple()