"""
DQN Training with GradCAM Video Recording

Train a DQN agent on Visual Frozen Lake and record GradCAM videos
showing where the agent focuses when making decisions.

This is where RL-GradCAM becomes meaningful:
- See attention evolve as agent learns
- Understand what trained agent "looks at"
- Create interpretable videos of decision-making
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_gradcam import (
    make_visual_frozen_lake,
    create_policy_network,
    RLGradCAM
)


@dataclass
class Transition:
    """Single transition in replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    """
    DQN Trainer with integrated GradCAM video recording.
    
    During evaluation episodes, records videos showing:
    - The visual observation
    - GradCAM heatmap for the chosen action
    - Q-values for all actions
    """
    
    def __init__(
        self,
        env,
        policy_net: nn.Module,
        target_net: nn.Module,
        gradcam: RLGradCAM,
        device: str = 'cpu',
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update: int = 10,
        buffer_size: int = 10000
    ):
        self.env = env
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        
        self.gradcam = gradcam
        self.device = device
        
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self) -> Optional[float]:
        """Single training step on batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        transitions = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32).to(self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in transitions])).float().to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self) -> Tuple[float, int]:
        """Train for one episode."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            action = self.select_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store transition
            self.buffer.push(state, action, reward, next_state, done or truncated)
            
            # Train
            loss = self.train_step()
            if loss is not None:
                self.losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return episode_reward, episode_length
    
    def update_target_network(self):
        """Copy weights from policy to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def evaluate_episode(self, record_gradcam: bool = False) -> Tuple[float, List[Dict]]:
        """
        Run evaluation episode (no exploration, no training).
        Optionally record GradCAM data for video.
        """
        state, info = self.env.reset()
        episode_reward = 0
        frames = []
        
        while True:
            # Get Q-values and action (greedy)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            action = np.argmax(q_values)
            
            # Record GradCAM if requested
            if record_gradcam:
                cam = self.gradcam.compute_cam(state, action)
                frames.append({
                    'state': state.copy(),
                    'action': action,
                    'q_values': q_values.copy(),
                    'cam': cam.copy(),
                    'info': info.copy() if info else {}
                })
            
            # Take step
            next_state, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                # Record final frame
                if record_gradcam:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                        q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                    cam = self.gradcam.compute_cam(state, np.argmax(q_values))
                    frames.append({
                        'state': state.copy(),
                        'action': np.argmax(q_values),
                        'q_values': q_values.copy(),
                        'cam': cam.copy(),
                        'info': info.copy() if info else {},
                        'terminal': True
                    })
                break
        
        return episode_reward, frames


class GradCAMVideoRecorder:
    """
    Records GradCAM videos showing agent's attention during episodes.
    
    Each frame shows:
    - Left: Original observation with agent position
    - Right: GradCAM heatmap overlay showing attention
    - Bottom: Q-values bar for all actions
    """
    
    def __init__(
        self,
        action_names: List[str],
        frame_size: Tuple[int, int] = (640, 480),
        fps: int = 2  # Slow for interpretability
    ):
        self.action_names = action_names
        self.frame_size = frame_size
        self.fps = fps
    
    def create_frame(
        self,
        state: np.ndarray,
        action: int,
        q_values: np.ndarray,
        cam: np.ndarray,
        step: int,
        info: Dict = None,
        terminal: bool = False
    ) -> np.ndarray:
        """Create single video frame with observation, GradCAM, and Q-values."""
        
        # Convert state to displayable format (H, W, C)
        if state.shape[0] == 3:  # (C, H, W)
            obs_img = (state.transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            obs_img = (state * 255).astype(np.uint8)
        
        # Resize observation
        obs_size = 200
        obs_resized = cv2.resize(obs_img, (obs_size, obs_size), interpolation=cv2.INTER_NEAREST)
        
        # Create GradCAM overlay
        cam_resized = cv2.resize(cam, (obs_size, obs_size), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(obs_resized, 0.5, heatmap, 0.5, 0)
        
        # Create frame canvas
        frame = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 255
        
        # Place observation (left)
        y_offset = 50
        x_offset = 50
        frame[y_offset:y_offset+obs_size, x_offset:x_offset+obs_size] = obs_resized
        
        # Place GradCAM overlay (right)
        x_offset_cam = 300
        frame[y_offset:y_offset+obs_size, x_offset_cam:x_offset_cam+obs_size] = overlay
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Observation', (x_offset + 50, y_offset - 10), font, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, 'GradCAM Attention', (x_offset_cam + 30, y_offset - 10), font, 0.6, (0, 0, 0), 2)
        
        # Add step info
        cv2.putText(frame, f'Step: {step}', (520, 80), font, 0.7, (0, 0, 0), 2)
        
        # Add action info
        action_name = self.action_names[action]
        color = (0, 128, 0)  # Green
        cv2.putText(frame, f'Action: {action_name}', (520, 120), font, 0.7, color, 2)
        
        # Draw Q-values bar chart
        bar_y = 300
        bar_height = 100
        bar_width = 80
        bar_spacing = 20
        bar_x_start = 100
        
        # Normalize Q-values for visualization
        q_min, q_max = q_values.min(), q_values.max()
        if q_max > q_min:
            q_norm = (q_values - q_min) / (q_max - q_min)
        else:
            q_norm = np.ones_like(q_values) * 0.5
        
        cv2.putText(frame, 'Q-Values:', (bar_x_start, bar_y - 20), font, 0.6, (0, 0, 0), 2)
        
        for i, (name, q, qn) in enumerate(zip(self.action_names, q_values, q_norm)):
            x = bar_x_start + i * (bar_width + bar_spacing)
            
            # Bar background
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
            
            # Bar fill
            fill_height = int(qn * bar_height)
            bar_color = (0, 200, 0) if i == action else (100, 100, 200)
            cv2.rectangle(frame, (x, bar_y + bar_height - fill_height), 
                         (x + bar_width, bar_y + bar_height), bar_color, -1)
            
            # Border for chosen action
            if i == action:
                cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (0, 128, 0), 3)
            
            # Label
            cv2.putText(frame, name, (x + 10, bar_y + bar_height + 25), font, 0.5, (0, 0, 0), 1)
            cv2.putText(frame, f'{q:.2f}', (x + 15, bar_y + bar_height + 45), font, 0.4, (0, 0, 0), 1)
        
        # Add terminal indicator
        if terminal:
            if info and info.get('reached_goal'):
                cv2.putText(frame, 'GOAL!', (520, 200), font, 1.0, (0, 200, 0), 3)
            elif info and info.get('fell_in_hole'):
                cv2.putText(frame, 'FELL IN HOLE', (480, 200), font, 0.8, (200, 0, 0), 2)
            else:
                cv2.putText(frame, 'DONE', (520, 200), font, 1.0, (100, 100, 100), 2)
        
        return frame
    
    def create_video(
        self,
        frames_data: List[Dict],
        output_path: str,
        title: str = "GradCAM Episode"
    ):
        """Create video from recorded frame data."""
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        
        for step, data in enumerate(frames_data):
            frame = self.create_frame(
                state=data['state'],
                action=data['action'],
                q_values=data['q_values'],
                cam=data['cam'],
                step=step,
                info=data.get('info', {}),
                terminal=data.get('terminal', False)
            )
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            # Hold terminal frame longer
            if data.get('terminal'):
                for _ in range(self.fps * 2):  # 2 seconds
                    out.write(frame_bgr)
        
        out.release()
        print(f"Video saved: {output_path}")


def train_and_record(
    n_episodes: int = 500,
    record_every: int = 50,
    output_dir: str = 'outputs/training'
):
    """
    Main training loop with periodic GradCAM video recording.
    """
    print("=" * 60)
    print("DQN Training with GradCAM Video Recording")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("\n1. Creating environment...")
    env = make_visual_frozen_lake(
        map_name="4x4",
        is_slippery=False,
        tile_size=16,
        highlight_danger=True
    )
    print(f"   Environment: FrozenLake 4x4 (deterministic)")
    
    # Create networks
    print("\n2. Creating networks...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    policy_net = create_policy_network(env)
    target_net = create_policy_network(env)
    
    # Create GradCAM
    gradcam = RLGradCAM(policy_net, policy_net.gradcam_target_layer, device=device)
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        policy_net=policy_net,
        target_net=target_net,
        gradcam=gradcam,
        device=device,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        batch_size=32,
        target_update=10
    )
    
    # Create video recorder
    video_recorder = GradCAMVideoRecorder(
        action_names=env.ACTION_NAMES,
        fps=2
    )
    
    # Training loop
    print("\n3. Starting training...")
    print(f"   Episodes: {n_episodes}")
    print(f"   Recording video every {record_every} episodes")
    print()
    
    best_reward = -float('inf')
    wins = 0
    
    for episode in range(1, n_episodes + 1):
        # Train
        reward, length = trainer.train_episode()
        
        if reward > 0:
            wins += 1
        
        # Update target network
        if episode % trainer.target_update == 0:
            trainer.update_target_network()
        
        # Print progress
        if episode % 10 == 0:
            recent_rewards = trainer.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            print(f"Episode {episode:4d} | "
                  f"Reward: {reward:.2f} | "
                  f"Avg(10): {avg_reward:.2f} | "
                  f"Win Rate: {win_rate:.0%} | "
                  f"Epsilon: {trainer.epsilon:.3f}")
        
        # Record GradCAM video
        if episode % record_every == 0 or episode == 1:
            print(f"\n   📹 Recording GradCAM video for episode {episode}...")
            eval_reward, frames = trainer.evaluate_episode(record_gradcam=True)
            
            video_path = os.path.join(output_dir, f'gradcam_episode_{episode:04d}.mp4')
            video_recorder.create_video(frames, video_path, title=f"Episode {episode}")
            
            result = "WIN" if eval_reward > 0 else "LOSS"
            print(f"   Eval reward: {eval_reward} ({result}), Steps: {len(frames)}")
            print()
        
        # Track best
        if reward > best_reward:
            best_reward = reward
    
    # Final evaluation videos
    print("\n4. Recording final evaluation videos...")
    
    for i in range(5):
        eval_reward, frames = trainer.evaluate_episode(record_gradcam=True)
        video_path = os.path.join(output_dir, f'gradcam_final_{i+1}.mp4')
        video_recorder.create_video(frames, video_path)
        result = "WIN" if eval_reward > 0 else "LOSS"
        print(f"   Final eval {i+1}: reward={eval_reward} ({result}), steps={len(frames)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTotal episodes: {n_episodes}")
    print(f"Total wins: {wins} ({100*wins/n_episodes:.1f}%)")
    print(f"Final epsilon: {trainer.epsilon:.3f}")
    print(f"Final avg reward (last 50): {np.mean(trainer.episode_rewards[-50:]):.3f}")
    
    print(f"\n📁 Videos saved in: {output_dir}/")
    print("   - gradcam_episode_XXXX.mp4: Training progress videos")
    print("   - gradcam_final_X.mp4: Final trained agent videos")
    
    print("\n💡 What to look for in videos:")
    print("   - Early episodes: Random/scattered attention")
    print("   - Later episodes: Focused attention on relevant tiles")
    print("   - Trained agent: Should attend to holes and goal")
    
    return trainer, video_recorder


if __name__ == "__main__":
    trainer, recorder = train_and_record(
        n_episodes=300,
        record_every=50,
        output_dir='outputs/training'
    )
