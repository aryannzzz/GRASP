"""
HuggingFace Dataset Loader for RoboPrompt Integration
Updated to match exact RoboPrompt ICL format from 0.txt
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from datasets import load_dataset


@dataclass
class DemonstrationFrame:
    """Single frame in a demonstration."""
    timestamp: float
    joint_positions: np.ndarray  # [6] joint angles
    gripper_state: float  # normalized 0-1
    
    
class HuggingFaceDatasetLoader:
    """
    Load demonstrations from HuggingFace LeRobot datasets.
    """
    
    def __init__(
        self,
        dataset_name: str = "aadarshram/pick_place_tape",
        scene_bounds: List[float] = None,
    ):
        """
        Initialize loader.
        
        Args:
            dataset_name: HuggingFace dataset name
            scene_bounds: Workspace bounds [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.dataset_name = dataset_name
        
        # Scene bounds for discretization - UPDATED to realistic SO100/SO101 bounds
        if scene_bounds is None:
            # REALISTIC bounds for SO100/SO101 - much smaller workspace
            self.scene_bounds = np.array([-0.2, -0.2, 0.3, 0.2, 0.2, 0.7])
        else:
            self.scene_bounds = np.array(scene_bounds)
        
        print(f"📦 Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, split="train")
        
        # Get dataset info
        self.total_episodes = len(set(self.dataset['episode_index']))
        self.total_frames = len(self.dataset)
        
        print(f"✓ Dataset loaded:")
        print(f"   Total episodes: {self.total_episodes}")
        print(f"   Total frames: {self.total_frames}")
        print(f"   Scene bounds: {self.scene_bounds}")
    
    def load_episode(self, episode_idx: int) -> List[DemonstrationFrame]:
        """
        Load a single episode from the dataset.
        
        Args:
            episode_idx: Episode number (0 to total_episodes-1)
        
        Returns:
            List of demonstration frames
        """
        # Filter dataset for this episode
        episode_data = self.dataset.filter(lambda x: x['episode_index'] == episode_idx)
        
        frames = []
        for i in range(len(episode_data)):
            row = episode_data[i]
            
            # Extract joint positions (observation.state)
            joint_positions = np.array(row['observation.state'], dtype=np.float32)
            
            # Last joint is gripper - normalize to [0, 1]
            gripper_joint = joint_positions[5]
            gripper_normalized = self._normalize_gripper(gripper_joint)
            
            frame = DemonstrationFrame(
                timestamp=float(row['timestamp']),
                joint_positions=joint_positions[:5],  # First 5 joints
                gripper_state=gripper_normalized
            )
            frames.append(frame)
        
        print(f"📂 Loaded episode {episode_idx}: {len(frames)} frames ({frames[-1].timestamp:.2f}s)")
        return frames
    
    def _normalize_gripper(self, gripper_joint: float) -> float:
        """
        Normalize gripper joint angle to [0, 1].
        
        Args:
            gripper_joint: Joint angle in degrees (from dataset)
        
        Returns:
            Normalized gripper state (0=closed, 1=open)
        """
        # From dataset, gripper ranges approximately 0-30 degrees
        min_angle, max_angle = 0.0, 30.0
        normalized = (gripper_joint - min_angle) / (max_angle - min_angle)
        return np.clip(normalized, 0, 1)
    
    def extract_keyframes(
    	self,
    	frames: List[DemonstrationFrame],
    	velocity_threshold: float = 2.0,  # Lower threshold for more keyframes
    	min_frames_between: int = 3,      # Reduced minimum gap
    	gripper_change_threshold: float = 0.1  # More sensitive gripper detection
    ) -> List[int]:
    	"""
    	Improved keyframe extraction.
    	"""
    	keyframe_indices = []
    	last_keyframe = -min_frames_between
    	
    	for i in range(1, len(frames) - 1):
    	    if i - last_keyframe < min_frames_between:
    	        continue
    	    
    	    dt = frames[i].timestamp - frames[i-1].timestamp
    	    if dt < 1e-6:
    	        continue
        
    	    # Calculate joint velocities
    	    velocity = np.linalg.norm(
    	        (frames[i].joint_positions - frames[i-1].joint_positions) / dt
    	    )
    	    
    	    # Improved criteria
    	    near_zero_velocity = velocity < velocity_threshold
    	    gripper_changed = abs(frames[i].gripper_state - frames[i-1].gripper_state) > gripper_change_threshold
    	    significant_movement = velocity > velocity_threshold * 5  # Add high-velocity keyf	rames
    	    
    	    if near_zero_velocity or gripper_changed or significant_movement:
    	        keyframe_indices.append(i)
    	        last_keyframe = i
    	
    	# Always include first and last frames
    	if len(frames) > 0:
    	    if 0 not in keyframe_indices:
    	        keyframe_indices.insert(0, 0)
    	    if len(frames)-1 not in keyframe_indices:
    	        keyframe_indices.append(len(frames) - 1)
    	
    	print(f"📌 Extracted {len(keyframe_indices)} keyframes from {len(frames)} total frames")
    	return keyframe_indices
    
    def joint_positions_to_ee_pose(
        self,
        joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert joint positions to end-effector pose using forward kinematics.
        
        This is a simplified FK for SO100/SO101. For production, use proper FK library.
        
        Args:
            joint_positions: [5] joint angles in degrees
        
        Returns:
            position: [x, y, z] in meters
            quaternion: [qx, qy, qz, qw]
        """
        # Convert to radians
        joints_rad = np.deg2rad(joint_positions)
        
        # Link lengths from SO100 URDF (approximate)
        L1 = 0.1025  # shoulder to upper_arm
        L2 = 0.11257  # upper_arm to lower_arm  
        L3 = 0.1349  # lower_arm to wrist
        L4 = 0.0601  # wrist to gripper
        base_height = 0.0165
        
        # Simplified forward kinematics
        j1, j2, j3, j4, j5 = joints_rad
        
        # Position calculation
        # XY plane (joints 1 controls rotation around Z)
        r = L1 * np.cos(j2) + L2 * np.cos(j2 + j3) + L3 * np.cos(j2 + j3 + j4) + L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        
        # Z height
        z = base_height + L1 * np.sin(j2) + L2 * np.sin(j2 + j3) + L3 * np.sin(j2 + j3 + j4)
        
        position = np.array([x, y, z])
        
        # Orientation (simplified)
        pitch = j2 + j3 + j4
        roll = j5
        yaw = 0.0  # Simplified
        
        rotation = R.from_euler('xyz', [yaw, pitch, roll])
        quaternion = rotation.as_quat()
        
        return position, quaternion
    
    def convert_to_roboprompt_format(
        self,
        frames: List[DemonstrationFrame],
        keyframe_indices: List[int]
    ) -> List[np.ndarray]:
        """
        Convert keyframes to RoboPrompt's discretized action format.
        
        Args:
            frames: All demonstration frames
            keyframe_indices: Indices of keyframes
        
        Returns:
            List of [x, y, z, rx, ry, rz, gripper] discretized actions
        """
        actions = []
        
        for idx in keyframe_indices:
            frame = frames[idx]
            
            # Convert joint positions to end-effector pose
            position, quaternion = self.joint_positions_to_ee_pose(frame.joint_positions)
            
            # Discretize position (100 bins per dimension)
            bounds = self.scene_bounds
            pos_normalized = (position - bounds[:3]) / (bounds[3:] - bounds[:3])
            pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
            
            # Convert quaternion to Euler angles and discretize (72 bins = 5° resolution)
            rotation = R.from_quat(quaternion)
            euler_deg = rotation.as_euler('xyz', degrees=True)
            euler_normalized = (euler_deg % 360) / 360  # Normalize to [0, 1]
            euler_bins = (euler_normalized * 72).astype(int) % 72
            
            # Binarize gripper (0 or 1)
            gripper_binary = 1 if frame.gripper_state > 0.5 else 0
            
            # Combine into RoboPrompt action format
            action = np.array([
                pos_bins[0], pos_bins[1], pos_bins[2],  # x, y, z bins
                euler_bins[0], euler_bins[1], euler_bins[2],  # rx, ry, rz bins
                gripper_binary  # gripper state
            ], dtype=int)
            
            actions.append(action)
        
        return actions
    
    def format_as_icl_demonstration(
        self,
        actions: List[np.ndarray],
        object_poses: Dict[str, np.ndarray]
    ) -> str:
        """
        Format demonstration in RoboPrompt's ICL format.
        MATCHES EXACT FORMAT FROM 0.txt:
        {object1: [x, y, z], object2: [x, y, z]}>>[action1, action2, ...]
        
        Args:
            actions: List of discretized actions
            object_poses: Dict mapping object names to [x, y, z] poses (position only)
        
        Returns:
            Formatted ICL demonstration string
        """
        # Format object observations - ONLY POSITION, no rotation
        obs_parts = []
        for obj_name, pose in object_poses.items():
            # Extract only position (first 3 elements)
            position = pose[:3]
            
            # Discretize object position
            bounds = self.scene_bounds
            pos_normalized = (position - bounds[:3]) / (bounds[3:] - bounds[:3])
            pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
            
            # Format as in example: 'object_name': [x, y, z]
            obs_parts.append(f"'{obj_name}': [{pos_bins[0]}, {pos_bins[1]}, {pos_bins[2]}]")
        
        # Format exactly like the example
        obs_str = "{" + ", ".join(obs_parts) + "}"
        
        # Format actions as list of lists
        action_strs = []
        for action in actions:
            action_list = action.tolist()
            action_strs.append(f"[{', '.join(map(str, action_list))}]")
        
        actions_str = "[" + ", ".join(action_strs) + "]"
        
        # Final format: {objects}>>[actions]
        demo = f"{obs_str}>>{actions_str}"
        
        return demo
    
    def save_demonstrations(
        self,
        episodes_to_process: List[int],
        object_poses: Dict[str, np.ndarray],
        save_dir: str = "./roboprompt_data"
    ) -> str:
        """
        Process multiple episodes and save as ICL demonstrations.
        
        Args:
            episodes_to_process: List of episode indices to process
            object_poses: Object poses for ICL format (position only)
            save_dir: Directory to save ICL demonstrations
        
        Returns:
            Path to saved ICL demonstrations file
        """
        print("\n" + "="*80)
        print("📝 CREATING ICL DEMONSTRATIONS FROM HUGGINGFACE DATASET")
        print("="*80 + "\n")
        
        icl_demos = []
        
        for ep_idx in episodes_to_process:
            print(f"\nProcessing episode {ep_idx}...")
            
            # Load episode
            frames = self.load_episode(ep_idx)
            
            # Extract keyframes
            keyframe_indices = self.extract_keyframes(frames)
            
            # Convert to RoboPrompt format
            actions = self.convert_to_roboprompt_format(frames, keyframe_indices)
            
            print(f"📋 Episode {ep_idx} - {len(actions)} keyframe actions:")
            for i, action in enumerate(actions[:3]):  # Show first 3
                print(f"  [{i}] {action}")
            if len(actions) > 3:
                print(f"  ... and {len(actions) - 3} more")
            
            # Format as ICL (NO INSTRUCTION in the format)
            icl_demo = self.format_as_icl_demonstration(
                actions=actions,
                object_poses=object_poses
            )
            
            icl_demos.append(icl_demo)
            print(f"✓ Formatted demonstration {ep_idx}")
        
        # Save ICL demonstrations in the exact format from 0.txt
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save as .txt file in the exact format from the example
        txt_file = save_path / "0.txt"
        
        with open(txt_file, 'w') as f:
            # Write each demo separated by commas, ending with comma
            for i, demo in enumerate(icl_demos):
                if i < len(icl_demos) - 1:
                    f.write(demo + ",\n")
                else:
                    f.write(demo + ",\n")  # End with comma like the example
        
        # Also save as JSON for metadata
        json_file = save_path / "icl_demos.json"
        
        icl_data = {
            'dataset_name': self.dataset_name,
            'demonstrations': icl_demos,
            'episodes_used': episodes_to_process,
            'object_names': list(object_poses.keys()),
            'object_poses': {k: v.tolist() for k, v in object_poses.items()},
            'scene_bounds': self.scene_bounds.tolist(),
            'format': 'RoboPrompt ICL - {objects}>>[actions]'
        }
        
        with open(json_file, 'w') as f:
            json.dump(icl_data, f, indent=2)
        
        print(f"\n💾 Saved {len(icl_demos)} ICL demonstrations to:")
        print(f"   {txt_file} (main format)")
        print(f"   {json_file} (with metadata)")
        
        # Print sample to verify format
        print(f"\n📋 Sample ICL Demonstration (first 150 chars):")
        sample = icl_demos[0]
        print(sample[:150] + "..." if len(sample) > 150 else sample)
        
        return str(txt_file)


def main():
    """Example usage."""
    
    # Initialize loader
    loader = HuggingFaceDatasetLoader(
        dataset_name="aadarshram/pick_place_tape",
        # REALISTIC bounds for SO100/SO101
        scene_bounds=[-0.2, -0.2, 0.3, 0.2, 0.2, 0.7]
    )
    
    # Use first few episodes as demonstrations
    episodes_to_use = [0, 1, 2, 3, 4]
    
    # Define object poses - ONLY POSITION (x, y, z) - UPDATED to realistic positions
    # These should be adjusted based on your actual scene
    object_poses = {
        'tape': np.array([0.1, -0.05, 0.4]),      # Closer to base, reachable
        'target_zone': np.array([0.15, 0.1, 0.4]), # Closer to base, reachable
        'table': np.array([0.0, 0.0, 0.3])         # Lower table
    }
    
    # Save demonstrations
    txt_path = loader.save_demonstrations(
        episodes_to_process=episodes_to_use,
        object_poses=object_poses,
        save_dir="./roboprompt_data"
    )
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"\nICL demonstrations saved to: {txt_path}")
    print("\nFormat matches 0.txt example:")
    print("  {object1: [x, y, z], object2: [x, y, z]}>>[action1, action2, ...]")
    print("\nNext steps:")
    print("1. Review the generated 0.txt file")
    print("2. Adjust object poses and scene bounds if bin values look wrong")
    print("3. Use with RoboPrompt inference")


if __name__ == "__main__":
    main()
