#!/usr/bin/env python
"""
MetaWorld ACT Training & Evaluation - Complete Working Pipeline

This module provides a complete pipeline for:
1. Dataset generation with expert demonstrations
2. ACT policy training 
3. Proper inference with normalization

Key fixes from original implementation:
- Proper policy.reset() calls
- Correct use of preprocessor/postprocessor pipelines
- Proper normalization handling
- Valid MetaWorld task names

Author: GitHub Copilot
Date: December 2024
"""

import os
import sys
import shutil
import numpy as np
import torch
from pathlib import Path
from collections import deque

# ============================================================================
# VALID METAWORLD TASK NAMES (v3)
# ============================================================================
VALID_TASKS = {
    # Your requested tasks
    "pick-place-v3": "Pick and place a puck to a goal",
    "push-v3": "Push the puck to a goal",
    "reach-v3": "Reach a goal position",
    "shelf-place-v3": "Pick and place a puck onto a shelf",
    "pick-place-wall-v3": "Pick a puck, bypass a wall and place the puck",
    
    # Pull-related tasks (there's no simple "pull-v3")
    "handle-pull-v3": "Pull a handle up",
    "handle-pull-side-v3": "Pull a handle up sideways",
    "lever-pull-v3": "Pull a lever down 90 degrees",
    "coffee-pull-v3": "Pull a mug from a coffee machine",
    "stick-pull-v3": "Grasp a stick and pull a box with the stick",
    
    # Other useful tasks
    "drawer-open-v3": "Open a drawer",
    "drawer-close-v3": "Push and close a drawer",
    "door-open-v3": "Open a door with a revolving joint",
    "button-press-v3": "Press a button",
}

# Task difficulties
DIFFICULTY_GROUPS = {
    "easy": ["reach-v3", "push-v3", "drawer-open-v3", "drawer-close-v3", 
             "handle-pull-v3", "handle-pull-side-v3", "lever-pull-v3"],
    "hard": ["pick-place-v3", "shelf-place-v3"],
    "very_hard": ["pick-place-wall-v3", "stick-pull-v3"],
}


def setup_environment():
    """Set up environment variables before any imports."""
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['LEROBOT_VIDEO_BACKEND'] = 'pyav'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['SVT_LOG'] = '0'


class MetaWorldDatasetGenerator:
    """Generate expert demonstrations for MetaWorld tasks."""
    
    def __init__(
        self,
        task_name: str,
        num_episodes: int = 50,
        max_episode_steps: int = 500,
        observation_width: int = 480,
        observation_height: int = 480,
        fps: int = 20,
        root_dir: Path = None,
    ):
        if task_name not in VALID_TASKS:
            raise ValueError(f"Invalid task: {task_name}. Valid tasks: {list(VALID_TASKS.keys())}")
        
        self.task_name = task_name
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.fps = fps
        self.root_dir = root_dir or Path("/kaggle/working/data")
        
    def _create_custom_env(self):
        """Create MetaWorld environment with raw observation capture."""
        from lerobot.envs.metaworld import MetaworldEnv
        
        class MetaworldEnvWithRawObs(MetaworldEnv):
            """Extended MetaworldEnv that captures raw internal state for expert policy."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._raw_obs = None
                # CRITICAL: Ensure randomization is enabled
                self._env._freeze_rand_vec = False

            def reset(self, seed=None, **kwargs):
                """Reset with randomization. Captures raw state for expert policy."""
                observation, info = super().reset(seed=seed, **kwargs)
                self._raw_obs = self._env._get_obs()
                return observation, info

            def step(self, action):
                """Execute action without auto-reset on termination."""
                if action.ndim != 1:
                    raise ValueError(f"Expected 1-D action, got shape {action.shape}")

                raw_obs, reward, done, truncated, info = self._env.step(action)
                
                is_success = bool(info.get("success", 0))
                terminated = done or is_success
                
                info.update({
                    "task": self.task,
                    "done": done,
                    "is_success": is_success,
                })
                
                observation = self._format_raw_obs(raw_obs)
                self._raw_obs = self._env._get_obs()
                
                return observation, reward, terminated, truncated, info
        
        return MetaworldEnvWithRawObs
    
    def generate(self, hf_username: str = None) -> Path:
        """Generate expert demonstration dataset."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.envs.metaworld import TASK_DESCRIPTIONS
        
        repo_id = f"lerobot/{self.task_name}"
        task_description = TASK_DESCRIPTIONS.get(self.task_name, f"perform {self.task_name}")
        dataset_dir = self.root_dir / repo_id
        
        print(f"\n{'='*70}")
        print(f"🎬 Generating Expert Dataset: {self.task_name}")
        print(f"{'='*70}")
        print(f"   Description: '{task_description}'")
        print(f"   Target episodes: {self.num_episodes}")
        print(f"   Max steps/episode: {self.max_episode_steps}")
        print(f"{'='*70}\n")
        
        # Clean up existing dataset
        if dataset_dir.exists():
            print(f"   ⚠️  Removing existing dataset at {dataset_dir}")
            shutil.rmtree(dataset_dir)
        
        # Create environment
        EnvClass = self._create_custom_env()
        env = EnvClass(
            task=self.task_name,
            camera_name="corner2",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=self.observation_width,
            observation_height=self.observation_height,
        )
        
        # Get expert policy
        expert_policy = env.expert_policy
        print(f"   ✅ Expert policy loaded: {type(expert_policy).__name__}")
        
        # Define features
        features = {
            "observation.images.image": {
                "dtype": "video",
                "shape": (self.observation_height, self.observation_width, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["x", "y", "z", "gripper"],
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["dx", "dy", "dz", "gripper"],
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "next.success": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["success"],
            },
        }
        
        # Create dataset
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=self.fps,
            root=dataset_dir,
            features=features,
            robot_type="sawyer",
            use_videos=True,
        )
        
        # Recording
        success_count = 0
        total_attempts = 0
        total_frames = 0
        
        print(f"🚀 Starting recording...\n")
        
        try:
            while success_count < self.num_episodes:
                total_attempts += 1
                seed = np.random.randint(0, 1_000_000)
                
                print(f"🎬 Episode attempt {total_attempts} (seed={seed})...", end=" ")
                
                obs, info = env.reset(seed=seed)
                dataset.episode_buffer = dataset.create_episode_buffer()
                
                done = False
                step_count = 0
                episode_success = False
                
                while not done and step_count < self.max_episode_steps:
                    image = obs["pixels"]
                    agent_pos = obs["agent_pos"]
                    
                    # Get expert action using RAW internal state
                    action = expert_policy.get_action(env._raw_obs)
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    
                    next_obs, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    
                    if step_info.get("is_success", False) or step_info.get("success", 0) > 0.5:
                        episode_success = True
                    
                    frame = {
                        "observation.images.image": image,
                        "observation.state": agent_pos.astype(np.float32),
                        "action": action,
                        "next.reward": np.array([reward], dtype=np.float32),
                        "next.success": np.array([episode_success]),
                        "task": task_description,
                    }
                    dataset.add_frame(frame)
                    
                    obs = next_obs
                    step_count += 1
                
                # Quality filtering - only save successful episodes
                if episode_success:
                    success_count += 1
                    total_frames += step_count
                    dataset.save_episode()
                    print(f"✅ SUCCESS | Steps: {step_count:3d} | Progress: {success_count}/{self.num_episodes}")
                else:
                    dataset.episode_buffer = None
                    print(f"❌ FAILED  | Steps: {step_count:3d} | Retrying...")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Recording interrupted by user")
        
        finally:
            print(f"\n{'='*70}")
            print("💾 Finalizing dataset...")
            dataset.finalize()
            env.close()
            
            print(f"\n📊 Dataset Summary:")
            print(f"   ✅ Successful episodes: {success_count}")
            print(f"   🎯 Total attempts: {total_attempts}")
            if total_attempts > 0:
                print(f"   📈 Success rate: {100 * success_count / total_attempts:.1f}%")
            if success_count > 0:
                print(f"   📏 Avg frames/episode: {total_frames / success_count:.1f}")
            print(f"   💾 Total frames: {total_frames:,}")
            print(f"   📁 Saved to: {dataset_dir}")
            print(f"{'='*70}\n")
        
        return dataset_dir


class MetaWorldACTInference:
    """Proper inference with ACT policy on MetaWorld."""
    
    def __init__(
        self,
        policy_repo_id: str,
        dataset_repo_id: str = None,
        device: str = "cuda",
    ):
        """
        Initialize inference.
        
        Args:
            policy_repo_id: HuggingFace repo ID for the trained policy
            dataset_repo_id: HuggingFace repo ID for dataset (for stats)
            device: Device to run inference on
        """
        self.policy_repo_id = policy_repo_id
        self.dataset_repo_id = dataset_repo_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self._load_policy()
        self._setup_processors()
    
    def _load_policy(self):
        """Load the ACT policy."""
        from lerobot.policies.act.modeling_act import ACTPolicy
        
        print(f"📥 Loading policy from: {self.policy_repo_id}")
        self.policy = ACTPolicy.from_pretrained(self.policy_repo_id)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        print(f"✅ Policy loaded on {self.device}")
    
    def _setup_processors(self):
        """Set up pre/post processors for proper normalization."""
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        
        # Get dataset stats for normalization
        if self.dataset_repo_id:
            print(f"📊 Loading dataset stats from: {self.dataset_repo_id}")
            try:
                dataset_meta = LeRobotDatasetMetadata(self.dataset_repo_id)
                dataset_stats = dataset_meta.stats
            except Exception as e:
                print(f"⚠️  Could not load dataset stats: {e}")
                dataset_stats = None
        else:
            dataset_stats = None
        
        # Create processors
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            dataset_stats=dataset_stats,
        )
        print("✅ Processors initialized")
    
    def _prepare_observation(self, obs: dict) -> dict:
        """
        Convert raw environment observation to policy format.
        
        CRITICAL: This properly formats and normalizes observations.
        """
        import torch
        import numpy as np
        
        batch = {}
        
        # Convert pixels -> observation.images.image
        if 'pixels' in obs:
            image = obs['pixels']
            # Convert to tensor, normalize to [0, 1], channel-first
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float() / 255.0
            if image.shape[-1] == 3:  # HWC -> CHW
                image = image.permute(2, 0, 1)
            batch['observation.images.image'] = image.to(self.device)
        
        # Convert agent_pos -> observation.state
        if 'agent_pos' in obs:
            state = obs['agent_pos']
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            batch['observation.state'] = state.to(self.device)
        
        return batch
    
    def evaluate(
        self,
        task_name: str,
        num_episodes: int = 10,
        max_steps: int = 500,
        render_videos: bool = False,
        video_dir: Path = None,
    ) -> dict:
        """
        Evaluate the policy on MetaWorld.
        
        CRITICAL FIXES:
        1. Call policy.reset() after each env.reset()
        2. Use preprocessor for proper normalization
        3. Use postprocessor for action unnormalization
        """
        from lerobot.envs.metaworld import MetaworldEnv
        
        print(f"\n{'='*70}")
        print(f"🧪 Evaluating Policy: {self.policy_repo_id}")
        print(f"   Task: {task_name}")
        print(f"   Episodes: {num_episodes}")
        print(f"{'='*70}\n")
        
        env = MetaworldEnv(
            task=task_name,
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        
        successes = 0
        total_rewards = []
        episode_lengths = []
        video_frames = []
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep)
            
            # CRITICAL: Reset policy action queue!
            self.policy.reset()
            
            done = False
            steps = 0
            episode_reward = 0
            frames = []
            
            while not done and steps < max_steps:
                # Prepare observation
                batch = self._prepare_observation(obs)
                
                # Apply preprocessor (normalization)
                processed_batch = self.preprocessor(batch)
                
                # Get action from policy
                with torch.no_grad():
                    action = self.policy.select_action(processed_batch)
                
                # Apply postprocessor (unnormalization)
                action = self.postprocessor(action)
                
                # Convert to numpy
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy().squeeze()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                if render_videos:
                    frames.append(env.render())
            
            episode_lengths.append(steps)
            total_rewards.append(episode_reward)
            
            is_success = info.get('is_success', False)
            if is_success:
                successes += 1
                status = "✅ SUCCESS"
            else:
                status = "❌ FAILED"
            
            print(f"   Episode {ep+1}/{num_episodes}: {status} | Steps: {steps} | Reward: {episode_reward:.2f}")
            
            if render_videos and frames:
                video_frames.append(frames)
        
        env.close()
        
        # Compute statistics
        success_rate = 100 * successes / num_episodes
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        results = {
            "success_rate": success_rate,
            "successes": successes,
            "total_episodes": num_episodes,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_episode_length": avg_length,
            "total_rewards": total_rewards,
            "episode_lengths": episode_lengths,
        }
        
        print(f"\n{'='*70}")
        print(f"📊 Evaluation Results")
        print(f"{'='*70}")
        print(f"   Success Rate: {success_rate:.1f}% ({successes}/{num_episodes})")
        print(f"   Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Avg Episode Length: {avg_length:.1f}")
        print(f"{'='*70}\n")
        
        return results


def get_training_command(
    task_name: str,
    dataset_repo_id: str,
    output_dir: str = "/kaggle/working/outputs",
    steps: int = 100000,
    batch_size: int = 8,
    eval_freq: int = 10000,
    wandb_project: str = "metaworld-act",
) -> str:
    """Generate the training command for LeRobot CLI."""
    
    cmd = f"""python /kaggle/working/lerobot/src/lerobot/scripts/lerobot_train.py \\
    --policy.type=act \\
    --policy.repo_id=act-{task_name} \\
    --env.type=metaworld \\
    --env.task={task_name} \\
    --dataset.repo_id={dataset_repo_id} \\
    --steps={steps} \\
    --batch_size={batch_size} \\
    --optimizer.lr=0.0001 \\
    --eval_freq={eval_freq} \\
    --save_freq=25000 \\
    --log_freq=100 \\
    --policy.chunk_size=100 \\
    --policy.n_obs_steps=1 \\
    --wandb.enable=true \\
    --wandb.project={wandb_project} \\
    --output_dir={output_dir}"""
    
    return cmd


# ============================================================================
# MULTI-TASK UTILITIES
# ============================================================================

def generate_all_datasets(
    tasks: list[str],
    num_episodes_per_task: int = 50,
    hf_username: str = None,
    root_dir: Path = None,
) -> dict[str, Path]:
    """Generate datasets for multiple tasks."""
    
    results = {}
    for task in tasks:
        print(f"\n{'#'*70}")
        print(f"# Generating dataset for: {task}")
        print(f"{'#'*70}")
        
        generator = MetaWorldDatasetGenerator(
            task_name=task,
            num_episodes=num_episodes_per_task,
            root_dir=root_dir,
        )
        dataset_path = generator.generate(hf_username=hf_username)
        results[task] = dataset_path
    
    return results


def evaluate_all_tasks(
    policy_repo_id: str,
    tasks: list[str],
    num_episodes: int = 10,
) -> dict:
    """Evaluate a policy on multiple tasks."""
    
    results = {}
    for task in tasks:
        print(f"\n{'#'*70}")
        print(f"# Evaluating on: {task}")
        print(f"{'#'*70}")
        
        try:
            inferencer = MetaWorldACTInference(
                policy_repo_id=policy_repo_id,
            )
            task_results = inferencer.evaluate(
                task_name=task,
                num_episodes=num_episodes,
            )
            results[task] = task_results
        except Exception as e:
            print(f"❌ Error evaluating {task}: {e}")
            results[task] = {"error": str(e)}
    
    return results


def print_results_table(results: dict):
    """Print results in a formatted table."""
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Task':<25} {'Success Rate':>15} {'Avg Reward':>15} {'Std':>10}")
    print("-"*80)
    
    for task, res in results.items():
        if "error" in res:
            print(f"{task:<25} {'ERROR':>15}")
        else:
            sr = f"{res['success_rate']:.1f}%"
            ar = f"{res['avg_reward']:.2f}"
            std = f"±{res['std_reward']:.2f}"
            print(f"{task:<25} {sr:>15} {ar:>15} {std:>10}")
    
    print("="*80)


if __name__ == "__main__":
    # Example usage
    setup_environment()
    
    # Your 6 target tasks
    TARGET_TASKS = [
        "pick-place-v3",          # Pick and place
        "push-v3",                # Push
        "reach-v3",               # Reach
        "shelf-place-v3",         # Place onto shelf
        "pick-place-wall-v3",     # Pick & place with wall
        "handle-pull-v3",         # Pull (using handle-pull since there's no simple pull-v3)
    ]
    
    print("Available tasks:", TARGET_TASKS)
