"""
LeRobot Dataset Loader for RoboPrompt Integration

This script loads demonstrations from a LeRobot dataset (HuggingFace or local)
and converts them to RoboPrompt ICL format.

Key insight: Object poses for ICL demos can be approximate/hardcoded because
they're just for demonstration formatting. Real perception is only needed at inference time!
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import torch
import cv2

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    print("⚠️  LeRobot not installed. Install with: pip install lerobot")
    LEROBOT_AVAILABLE = False


@dataclass
class RoboPromptAction:
    """Single RoboPrompt action (discretized)."""
    position_bins: np.ndarray  # [x, y, z] bins (0-99)
    rotation_bins: np.ndarray  # [rx, ry, rz] bins (0-71)
    gripper_binary: int  # 0 or 1
    
    def to_array(self) -> np.ndarray:
        """Convert to RoboPrompt action array."""
        return np.concatenate([
            self.position_bins,
            self.rotation_bins,
            [self.gripper_binary]
        ])


class LeRobotRoboPromptDataset:
    """
    Load LeRobot dataset and convert to RoboPrompt format.
    """
    
    def __init__(
        self,
        repo_id: str = "aadarshram/pick_place_tape",
        scene_bounds: List[float] = None,
        local_files_only: bool = False,
        root: Optional[str] = None
    ):
        """
        Initialize dataset loader.
        
        Args:
            repo_id: HuggingFace dataset ID or local path
            scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] in meters
            local_files_only: If True, only use local cache
            root: Local root directory (for local datasets)
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot library required. Install with: pip install lerobot")
        
        self.repo_id = repo_id
        
        # Scene bounds for discretization
        if scene_bounds is None:
            # Default bounds for SO-100/101
            self.scene_bounds = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
        else:
            self.scene_bounds = np.array(scene_bounds)
        
        print(f"📦 Loading LeRobot dataset: {repo_id}")
        
        # Load dataset
        kwargs = {"local_files_only": local_files_only}
        if root is not None:
            kwargs["root"] = root
            
        self.dataset = LeRobotDataset(repo_id, **kwargs)
        
        # Get dataset info
        self.fps = self.dataset.fps
        self.total_episodes = len(self.dataset.episodes)
        self.total_frames = len(self.dataset)
        self.robot_type = self.dataset.robot_type
        
        # Check what observation keys are available
        sample = self.dataset[0]
        self.available_keys = list(sample.keys())
        
        # Find camera keys
        self.camera_keys = [k for k in self.available_keys if 'image' in k]
        
        print(f"✓ Dataset loaded:")
        print(f"   Robot: {self.robot_type}")
        print(f"   Episodes: {self.total_episodes}")
        print(f"   Frames: {self.total_frames}")
        print(f"   FPS: {self.fps}")
        print(f"   Cameras: {self.camera_keys}")
        print(f"   Scene bounds: {self.scene_bounds}")
    
    def load_episode(
        self,
        episode_idx: int,
        load_images: bool = False
    ) -> Dict:
        """
        Load a complete episode from the dataset.
        
        Args:
            episode_idx: Episode index
            load_images: If True, decode video frames (slower)
        
        Returns:
            dict with 'states', 'actions', 'timestamps', optionally 'images'
        """
        # Get episode info
        episode_info = self.dataset.episodes.iloc[episode_idx]
        start_idx = episode_info['dataset_from_index']
        end_idx = episode_info['dataset_to_index']
        length = episode_info['length']
        
        print(f"📂 Loading episode {episode_idx}: {length} frames ({length/self.fps:.2f}s)")
        
        # Load all frames in episode
        states = []
        actions = []
        timestamps = []
        images = [] if load_images else None
        
        for idx in range(start_idx, end_idx):
            sample = self.dataset[idx]
            
            # Extract state (joint positions)
            if 'observation.state' in sample:
                state = sample['observation.state'].numpy()
                states.append(state)
            
            # Extract action (target joint positions)
            if 'action' in sample:
                action = sample['action'].numpy()
                actions.append(action)
            
            # Extract timestamp
            if 'timestamp' in sample:
                timestamp = sample['timestamp'].item()
                timestamps.append(timestamp)
            
            # Extract image (if requested)
            if load_images and len(self.camera_keys) > 0:
                # Use first available camera
                camera_key = self.camera_keys[0]
                if camera_key in sample:
                    img = sample[camera_key]  # Already a tensor [C, H, W]
                    # Convert to numpy [H, W, C]
                    img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
                    images.append(img_np)
        
        result = {
            'states': np.array(states),
            'actions': np.array(actions),
            'timestamps': np.array(timestamps),
            'episode_index': episode_idx,
            'length': length
        }
        
        if load_images and images:
            result['images'] = np.array(images)
        
        return result
    
    def extract_keyframes_roboprompt(
        self,
        episode: Dict,
        velocity_threshold: float = 0.1,  # radians/second
        min_frames_between: int = None
    ) -> List[int]:
        """
        Extract keyframes using RoboPrompt's criteria:
        1. Near-zero joint velocity
        2. Gripper state change
        
        Args:
            episode: Episode data from load_episode()
            velocity_threshold: Velocity threshold (rad/s)
            min_frames_between: Minimum frames between keyframes
        
        Returns:
            List of keyframe indices
        """
        states = episode['states']
        timestamps = episode['timestamps']
        
        if min_frames_between is None:
            min_frames_between = max(1, int(self.fps / 6))  # ~6 keyframes per second max
        
        keyframe_indices = []
        last_keyframe = -min_frames_between
        
        # Gripper is typically the last joint
        gripper_states = states[:, -1]
        
        for i in range(1, len(states) - 1):
            # Skip if too close to last keyframe
            if i - last_keyframe < min_frames_between:
                continue
            
            # Calculate joint velocities (rad/s)
            dt = timestamps[i] - timestamps[i-1]
            if dt < 1e-6:
                continue
            
            joint_vel = (states[i, :-1] - states[i-1, :-1]) / dt
            velocity_magnitude = np.linalg.norm(joint_vel)
            
            # Check criteria from RoboPrompt paper
            near_zero_velocity = velocity_magnitude < velocity_threshold
            gripper_changed = abs(gripper_states[i] - gripper_states[i-1]) > 0.1
            
            if near_zero_velocity or gripper_changed:
                keyframe_indices.append(i)
                last_keyframe = i
        
        # Always include first and last frames
        if 0 not in keyframe_indices:
            keyframe_indices.insert(0, 0)
        if len(states)-1 not in keyframe_indices:
            keyframe_indices.append(len(states) - 1)
        
        print(f"📌 Extracted {len(keyframe_indices)} keyframes from {len(states)} frames")
        return keyframe_indices
    
    def joint_state_to_ee_pose(
        self,
        joint_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert joint state to end-effector pose (forward kinematics).
        
        For SO-100/SO-101: [shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper]
        
        Args:
            joint_state: Joint angles in radians
        
        Returns:
            position: [x, y, z] in meters
            quaternion: [qx, qy, qz, qw]
        """
        # SO-100/101 link lengths (meters) - from URDF
        L1 = 0.1025  # shoulder to upper_arm
        L2 = 0.11257  # upper_arm to lower_arm
        L3 = 0.1349  # lower_arm to wrist
        L4 = 0.0601  # wrist to gripper_center
        base_height = 0.0165
        
        # Extract joints (first 5 are arm, last is gripper)
        j1, j2, j3, j4, j5 = joint_state[:5]
        
        # Forward kinematics
        # j1: shoulder pan (rotation around Z)
        # j2: shoulder lift
        # j3: elbow flex
        # j4: wrist flex
        # j5: wrist roll
        
        # Calculate end-effector position
        r = L1 * np.cos(j2) + L2 * np.cos(j2 + j3) + L3 * np.cos(j2 + j3 + j4) + L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = base_height + L1 * np.sin(j2) + L2 * np.sin(j2 + j3) + L3 * np.sin(j2 + j3 + j4)
        
        position = np.array([x, y, z])
        
        # Calculate end-effector orientation
        pitch = j2 + j3 + j4  # Combined pitch from arm joints
        roll = j5  # Wrist roll
        yaw = 0.0  # Simplified (no yaw from arm structure)
        
        rotation = R.from_euler('xyz', [yaw, pitch, roll])
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
        
        return position, quaternion
    
    def discretize_pose_roboprompt(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_state: float
    ) -> RoboPromptAction:
        """
        Discretize continuous pose into RoboPrompt bins.
        
        Args:
            position: [x, y, z] in meters
            quaternion: [qx, qy, qz, qw]
            gripper_state: Gripper joint angle
        
        Returns:
            RoboPromptAction with discretized values
        """
        # Discretize position (100 bins per dimension)
        pos_normalized = (position - self.scene_bounds[:3]) / (self.scene_bounds[3:] - self.scene_bounds[:3])
        pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
        
        # Convert quaternion to Euler angles
        rotation = R.from_quat(quaternion)
        euler_rad = rotation.as_euler('xyz')
        euler_deg = np.degrees(euler_rad)
        
        # Discretize rotation (72 bins = 5° resolution)
        euler_normalized = (euler_deg % 360) / 360
        rot_bins = (euler_normalized * 72).astype(int) % 72
        
        # Binarize gripper (threshold at 50% of range)
        # For SO-100/101, gripper ranges ~0-30 degrees
        gripper_binary = 1 if gripper_state > 15.0 else 0
        
        return RoboPromptAction(
            position_bins=pos_bins,
            rotation_bins=rot_bins,
            gripper_binary=gripper_binary
        )
    
    def create_icl_demonstrations(
        self,
        episodes_to_use: List[int],
        object_poses: Dict[str, np.ndarray],
        instruction: str,
        save_dir: str = "./roboprompt_data"
    ) -> str:
        """
        Create ICL demonstrations from episodes.
        
        IMPORTANT: object_poses can be approximate/hardcoded for demo creation!
        They're just for ICL formatting. Real perception is only needed at inference.
        
        Args:
            episodes_to_use: List of episode indices
            object_poses: Dict of {object_name: [x, y, z, rx, ry, rz]}
                         These can be approximate! They're just for ICL format.
            instruction: Task instruction string
            save_dir: Where to save demonstrations
        
        Returns:
            Path to saved demonstrations file
        """
        print("\n" + "="*80)
        print("🔧 CREATING ICL DEMONSTRATIONS FROM LEROBOT DATASET")
        print("="*80)
        print(f"\n💡 Note: Object poses are approximate for ICL demo formatting.")
        print("    Real perception is only needed at inference time!\n")
        
        icl_demos = []
        all_actions = []  # Store all actions for analysis
        
        for ep_idx in episodes_to_use:
            print(f"\n--- Processing Episode {ep_idx} ---")
            
            # Load episode
            episode = self.load_episode(ep_idx, load_images=False)
            
            # Extract keyframes using RoboPrompt criteria
            keyframe_indices = self.extract_keyframes_roboprompt(episode)
            
            # Convert keyframes to RoboPrompt actions
            actions = []
            for kf_idx in keyframe_indices:
                joint_state = episode['states'][kf_idx]
                
                # Forward kinematics
                position, quaternion = self.joint_state_to_ee_pose(joint_state)
                
                # Discretize
                gripper_state = joint_state[-1]  # Last joint is gripper
                action = self.discretize_pose_roboprompt(position, quaternion, gripper_state)
                actions.append(action)
            
            print(f"📋 Episode {ep_idx} - {len(actions)} keyframe actions:")
            for i, action in enumerate(actions[:5]):
                arr = action.to_array()
                print(f"  [{i}] {arr}")
            if len(actions) > 5:
                print(f"  ... and {len(actions) - 5} more")
            
            # Format as ICL demonstration
            icl_demo = self._format_icl_demo(actions, object_poses, instruction)
            icl_demos.append(icl_demo)
            all_actions.extend(actions)
            
            print(f"✓ Formatted demonstration {ep_idx}")
        
        # Save to file
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        icl_file = save_path / "icl_demos.json"
        
        icl_data = {
            'repo_id': self.repo_id,
            'robot_type': self.robot_type,
            'task': instruction,
            'demonstrations': icl_demos,
            'episodes_used': episodes_to_use,
            'num_keyframes': [len(d.split('>>')[1].split('],')) for d in icl_demos],
            'object_names': list(object_poses.keys()),
            'object_poses_approximate': {k: v.tolist() for k, v in object_poses.items()},
            'scene_bounds': self.scene_bounds.tolist(),
            'fps': self.fps,
            'note': 'Object poses are approximate for ICL formatting. Use perception at inference time.'
        }
        
        with open(icl_file, 'w') as f:
            json.dump(icl_data, f, indent=2)
        
        print(f"\n💾 Saved {len(icl_demos)} ICL demonstrations to {icl_file}")
        print(f"\n📋 Sample ICL Demonstration:")
        print(icl_demos[0][:200] + "...")
        
        # Print statistics
        action_arrays = [a.to_array() for a in all_actions]
        print(f"\n📊 Action Statistics:")
        print(f"   Total keyframes: {len(all_actions)}")
        print(f"   Avg per episode: {len(all_actions) / len(episodes_to_use):.1f}")
        
        return str(icl_file)
    
    def _format_icl_demo(
        self,
        actions: List[RoboPromptAction],
        object_poses: Dict[str, np.ndarray],
        instruction: str
    ) -> str:
        """
        Format demonstration in RoboPrompt's ICL format.
        
        Format: {objects, instruction} >> [actions]
        """
        # Format object observations (discretized)
        obs_parts = []
        for obj_name, pose in object_poses.items():
            position = pose[:3]
            euler = pose[3:] if len(pose) > 3 else [0, 0, 0]
            
            # Discretize object pose
            pos_normalized = (position - self.scene_bounds[:3]) / (self.scene_bounds[3:] - self.scene_bounds[:3])
            pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
            
            obs_parts.append(f"'{obj_name}': [{pos_bins[0]}, {pos_bins[1]}, {pos_bins[2]}]")
        
        obs_str = "{" + ", ".join(obs_parts) + "}"
        
        # Format actions
        action_arrays = [a.to_array().tolist() for a in actions]
        actions_str = "[" + ", ".join([str(a) for a in action_arrays]) + "]"
        
        # Combine in ICL format
        demo = f"{{{obs_str}, '{instruction}'}}>>{actions_str}"
        
        return demo
    
    def visualize_episode(
        self,
        episode_idx: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize an episode with keyframes marked.
        
        Args:
            episode_idx: Episode to visualize
            save_path: Where to save video (optional)
        """
        import matplotlib.pyplot as plt
        
        print(f"\n🎬 Visualizing episode {episode_idx}...")
        
        # Load episode with images if available
        episode = self.load_episode(episode_idx, load_images=len(self.camera_keys) > 0)
        keyframe_indices = self.extract_keyframes_roboprompt(episode)
        
        # Plot joint trajectories with keyframes
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Episode {episode_idx} - Joint Trajectories (Keyframes Marked)', fontsize=16)
        
        timestamps = episode['timestamps']
        states = episode['states']
        
        joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist Flex', 'Wrist Roll', 'Gripper']
        
        for i, ax in enumerate(axes.flat):
            if i < states.shape[1]:
                # Plot full trajectory
                ax.plot(timestamps, states[:, i], 'b-', alpha=0.5, label='Trajectory')
                
                # Mark keyframes
                kf_times = timestamps[keyframe_indices]
                kf_values = states[keyframe_indices, i]
                ax.scatter(kf_times, kf_values, c='red', s=50, zorder=5, label='Keyframes')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Angle (rad)' if i < 5 else 'Position')
                ax.set_title(joint_names[i])
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        else:
            plt.savefig(f'./roboprompt_data/episode_{episode_idx}_visualization.png', dpi=150)
            print(f"💾 Saved to ./roboprompt_data/episode_{episode_idx}_visualization.png")
        
        plt.close()


def main():
    """Example usage."""
    
    print("=" * 80)
    print("LEROBOT → ROBOPROMPT DEMONSTRATION CREATOR")
    print("=" * 80)
    
    # Initialize loader
    loader = LeRobotRoboPromptDataset(
        repo_id="aadarshram/pick_place_tape",
        scene_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    )
    
    # Use first 5 episodes as demonstrations
    episodes_to_use = [0, 1, 2, 3, 4]
    
    # Define APPROXIMATE object poses for ICL formatting
    # These don't need to be exact! They're just for the demonstration format.
    # Real perception happens at inference time.
    object_poses = {
        'tape': np.array([0.3, 0.0, 0.7, 0, 0, 0]),  # [x, y, z, rx, ry, rz]
        'target_zone': np.array([0.5, 0.2, 0.7, 0, 0, 0]),
        'table': np.array([0.0, 0.0, 0.6, 0, 0, 0])
    }
    
    print("\n💡 Important: Object poses above are APPROXIMATE for demo formatting only!")
    print("   Real object detection will happen at inference time.\n")
    
    # Create ICL demonstrations
    icl_path = loader.create_icl_demonstrations(
        episodes_to_use=episodes_to_use,
        object_poses=object_poses,
        instruction="pick up the tape and place it in the target zone",
        save_dir="./roboprompt_data"
    )
    
    # Visualize first episode
    loader.visualize_episode(0, save_path='./roboprompt_data/episode_0_trajectory.png')
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION CREATION COMPLETE")
    print("="*80)
    print(f"\nCreated files:")
    print(f"  1. {icl_path} - ICL demonstrations")
    print(f"  2. ./roboprompt_data/episode_0_trajectory.png - Visualization")
    print("\nNext steps:")
    print("  1. Review ICL demonstrations")
    print("  2. Add perception system for inference (see perception guide)")
    print("  3. Integrate with RoboPrompt LLM")
    print("  4. Execute on SO-101 robot")


if __name__ == "__main__":
    main()
