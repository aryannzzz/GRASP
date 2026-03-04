"""
Atari DQN Training with GradCAM Video Recording

Train a DQN agent on Atari Pong and record GradCAM videos showing
where the agent focuses (should be ball and paddle!).

This is where RL-GradCAM becomes meaningful:
- Agent should focus on the BALL to predict where it's going
- Agent should focus on PADDLE to know current position
- Attention should NOT be on score, background, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_gradcam.atari_wrappers import make_atari_env
from rl_gradcam.atari_policy import AtariCNN, create_atari_policy
from rl_gradcam.rl_gradcam import RLGradCAM


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Store as uint8 to save memory
        state = (state * 255).astype(np.uint8)
        next_state = (next_state * 255).astype(np.uint8)
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        transitions = random.sample(self.buffer, batch_size)
        # Convert back to float
        return [Transition(
            t.state.astype(np.float32) / 255.0,
            t.action, t.reward,
            t.next_state.astype(np.float32) / 255.0,
            t.done
        ) for t in transitions]
    
    def __len__(self):
        return len(self.buffer)


class AtariDQNTrainer:
    """DQN Trainer for Atari with GradCAM recording."""
    
    def __init__(
        self,
        env,
        policy_net: nn.Module,
        target_net: nn.Module,
        gradcam: RLGradCAM,
        device: str = 'cpu',
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 100000,
        batch_size: int = 32,
        target_update: int = 1000,
        buffer_size: int = 100000,
        learning_starts: int = 10000
    ):
        self.env = env
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        
        self.gradcam = gradcam
        self.device = device
        
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.learning_starts = learning_starts
        
        self.buffer = ReplayBuffer(buffer_size)
        
        # Stats
        self.total_steps = 0
        self.episode_rewards = []
        self.losses = []
    
    def get_epsilon(self) -> float:
        """Linear epsilon decay."""
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, state: np.ndarray) -> int:
        self.epsilon = self.get_epsilon()
        
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.learning_starts:
            return None
        
        transitions = self.buffer.sample(self.batch_size)
        
        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32).to(self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in transitions])).float().to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self) -> Tuple[float, int]:
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = self.select_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            
            self.buffer.push(state, action, reward, next_state, done or truncated)
            
            loss = self.train_step()
            if loss is not None:
                self.losses.append(loss)
            
            # Update target network
            self.total_steps += 1
            if self.total_steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        self.episode_rewards.append(episode_reward)
        return episode_reward, episode_length
    
    def evaluate_episode(
        self, 
        record_gradcam: bool = False,
        max_steps: int = 5000
    ) -> Tuple[float, List[Dict]]:
        """Evaluate one episode with optional GradCAM recording."""
        state, _ = self.env.reset()
        episode_reward = 0
        frames = []
        
        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            action = np.argmax(q_values)
            
            if record_gradcam:
                cam = self.gradcam.compute_cam(state, action)
                frames.append({
                    'state': state.copy(),
                    'action': action,
                    'q_values': q_values.copy(),
                    'cam': cam.copy(),
                    'step': step,
                    'reward': 0  # Updated after step
                })
            
            next_state, reward, done, truncated, info = self.env.step(action)
            
            if record_gradcam and len(frames) > 0:
                frames[-1]['reward'] = reward
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                if record_gradcam:
                    frames[-1]['terminal'] = True
                break
        
        return episode_reward, frames


class AtariGradCAMRecorder:
    """Record GradCAM videos for Atari games."""
    
    def __init__(
        self,
        action_names: List[str],
        frame_size: Tuple[int, int] = (900, 500),
        fps: int = 15
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
        reward: float = 0,
        terminal: bool = False
    ) -> np.ndarray:
        """Create visualization frame."""
        
        # State is (4, 84, 84) - use last frame for display
        if state.ndim == 3 and state.shape[0] == 4:
            # Use the most recent frame (last in stack)
            display_frame = (state[-1] * 255).astype(np.uint8)
        else:
            display_frame = (state * 255).astype(np.uint8)
        
        # Resize for display
        display_size = 300
        frame_resized = cv2.resize(display_frame, (display_size, display_size), 
                                   interpolation=cv2.INTER_NEAREST)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
        
        # Create GradCAM overlay
        cam_resized = cv2.resize(cam, (display_size, display_size), 
                                 interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame_rgb, 0.5, heatmap, 0.5, 0)
        
        # Create canvas
        canvas = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 30
        
        # Place images
        y_start = 80
        x_frame = 30
        x_overlay = 360
        
        canvas[y_start:y_start+display_size, x_frame:x_frame+display_size] = frame_rgb
        canvas[y_start:y_start+display_size, x_overlay:x_overlay+display_size] = overlay
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        yellow = (255, 255, 100)
        green = (100, 255, 100)
        red = (255, 100, 100)
        
        cv2.putText(canvas, 'GAME FRAME', (x_frame + 80, y_start - 15), font, 0.7, white, 2)
        cv2.putText(canvas, 'GRADCAM ATTENTION', (x_overlay + 50, y_start - 15), font, 0.7, yellow, 2)
        
        # Info panel
        info_x = 700
        cv2.putText(canvas, f'Step: {step}', (info_x, 100), font, 0.7, white, 2)
        
        action_name = self.action_names[action] if action < len(self.action_names) else str(action)
        cv2.putText(canvas, f'Action: {action_name}', (info_x, 140), font, 0.6, green, 2)
        
        reward_color = green if reward > 0 else (red if reward < 0 else white)
        cv2.putText(canvas, f'Reward: {reward:+.0f}', (info_x, 180), font, 0.6, reward_color, 2)
        
        # Epsilon
        cv2.putText(canvas, f'Eps: {self.epsilon:.3f}' if hasattr(self, 'epsilon') else '', 
                    (info_x, 220), font, 0.5, white, 1)
        
        # Q-values bar (vertical)
        bar_x = 700
        bar_y = 260
        bar_w = 25
        bar_h_max = 150
        
        cv2.putText(canvas, 'Q-Values:', (bar_x, bar_y - 10), font, 0.5, white, 1)
        
        # Normalize Q-values for display
        q_min, q_max = q_values.min(), q_values.max()
        q_range = max(q_max - q_min, 0.001)
        
        for i, (q, name) in enumerate(zip(q_values, self.action_names)):
            if i >= 4:  # Only show first 4 actions for space
                break
            
            x = bar_x + i * (bar_w + 15)
            
            # Normalize to [0, 1]
            q_norm = (q - q_min) / q_range
            bar_h = int(q_norm * bar_h_max)
            
            # Background
            cv2.rectangle(canvas, (x, bar_y), (x + bar_w, bar_y + bar_h_max), (60, 60, 60), -1)
            
            # Fill
            color = green if i == action else (100, 100, 180)
            cv2.rectangle(canvas, (x, bar_y + bar_h_max - bar_h), 
                         (x + bar_w, bar_y + bar_h_max), color, -1)
            
            # Highlight chosen
            if i == action:
                cv2.rectangle(canvas, (x-2, bar_y-2), (x + bar_w + 2, bar_y + bar_h_max + 2), green, 2)
            
            # Short name (first 4 chars)
            short_name = name[:4] if len(name) > 4 else name
            cv2.putText(canvas, short_name, (x-5, bar_y + bar_h_max + 20), font, 0.35, white, 1)
        
        # Colorbar
        cb_x, cb_y = 700, 440
        cb_w, cb_h = 150, 15
        for i in range(cb_w):
            color = cv2.applyColorMap(np.array([[int(i * 255 / cb_w)]], dtype=np.uint8), 
                                      cv2.COLORMAP_JET)[0][0]
            color = tuple(int(c) for c in color[::-1])
            cv2.line(canvas, (cb_x + i, cb_y), (cb_x + i, cb_y + cb_h), color)
        
        cv2.putText(canvas, 'Low', (cb_x, cb_y + cb_h + 15), font, 0.35, white, 1)
        cv2.putText(canvas, 'High', (cb_x + cb_w - 25, cb_y + cb_h + 15), font, 0.35, white, 1)
        cv2.putText(canvas, 'Attention', (cb_x + 40, cb_y - 5), font, 0.4, yellow, 1)
        
        # Terminal indicator
        if terminal:
            cv2.putText(canvas, 'EPISODE END', (info_x - 30, 480), font, 0.7, red, 2)
        
        return canvas
    
    def create_video(self, frames_data: List[Dict], output_path: str, epsilon: float = 0):
        """Create video from recorded frames."""
        self.epsilon = epsilon
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        
        for data in frames_data:
            frame = self.create_frame(
                state=data['state'],
                action=data['action'],
                q_values=data['q_values'],
                cam=data['cam'],
                step=data['step'],
                reward=data.get('reward', 0),
                terminal=data.get('terminal', False)
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"   📹 Saved: {output_path}")


def train_atari_with_gradcam(
    env_name: str = "ALE/Pong-v5",
    n_steps: int = 500000,
    record_every_episodes: int = 100,
    output_dir: str = 'outputs/atari_pong'
):
    """
    Train DQN on Atari and record GradCAM videos showing attention.
    """
    print("=" * 70)
    print(f"Atari DQN Training with GradCAM - {env_name}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("\n1. Creating environment...")
    env = make_atari_env(env_name, frame_stack=4)
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Actions: {env.action_space.n} - {env.ACTION_NAMES}")
    
    # Create networks
    print("\n2. Creating networks...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    policy_net = create_atari_policy(env)
    target_net = create_atari_policy(env)
    print(f"   Network params: {sum(p.numel() for p in policy_net.parameters()):,}")
    
    # Create GradCAM
    gradcam = RLGradCAM(policy_net, policy_net.gradcam_target_layer, device=device)
    
    # Create trainer
    trainer = AtariDQNTrainer(
        env=env,
        policy_net=policy_net,
        target_net=target_net,
        gradcam=gradcam,
        device=device,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=100000,
        batch_size=32,
        target_update=1000,
        buffer_size=100000,
        learning_starts=10000
    )
    
    # Create video recorder
    recorder = AtariGradCAMRecorder(action_names=env.ACTION_NAMES, fps=15)
    
    # Record untrained behavior
    print("\n3. Recording UNTRAINED agent...")
    _, frames = trainer.evaluate_episode(record_gradcam=True, max_steps=500)
    recorder.create_video(frames, os.path.join(output_dir, 'gradcam_untrained.mp4'), 
                         epsilon=trainer.epsilon)
    
    # Training loop
    print("\n4. Training...")
    print(f"   Target steps: {n_steps:,}")
    print(f"   Recording every {record_every_episodes} episodes\n")
    
    episode = 0
    start_time = time.time()
    
    while trainer.total_steps < n_steps:
        reward, length = trainer.train_episode()
        episode += 1
        
        # Progress logging
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = trainer.total_steps / elapsed
            avg_reward = np.mean(trainer.episode_rewards[-10:]) if len(trainer.episode_rewards) >= 10 else reward
            avg_loss = np.mean(trainer.losses[-100:]) if len(trainer.losses) >= 100 else 0
            
            print(f"Ep {episode:4d} | "
                  f"Steps: {trainer.total_steps:7,} | "
                  f"Reward: {avg_reward:+6.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {trainer.epsilon:.3f} | "
                  f"SPS: {steps_per_sec:.0f}")
        
        # Record GradCAM video
        if episode % record_every_episodes == 0:
            print(f"\n   📹 Recording episode {episode}...")
            eval_reward, frames = trainer.evaluate_episode(record_gradcam=True, max_steps=2000)
            recorder.create_video(
                frames,
                os.path.join(output_dir, f'gradcam_ep{episode:04d}_r{eval_reward:+.0f}.mp4'),
                epsilon=trainer.epsilon
            )
            print(f"   Eval reward: {eval_reward:+.0f}\n")
        
        # Save checkpoint
        if episode % 500 == 0:
            torch.save({
                'episode': episode,
                'total_steps': trainer.total_steps,
                'policy_net': policy_net.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'epsilon': trainer.epsilon
            }, os.path.join(output_dir, f'checkpoint_ep{episode}.pt'))
    
    # Final videos
    print("\n5. Recording final evaluation...")
    for i in range(5):
        eval_reward, frames = trainer.evaluate_episode(record_gradcam=True, max_steps=5000)
        recorder.create_video(
            frames,
            os.path.join(output_dir, f'gradcam_final_{i+1}_r{eval_reward:+.0f}.mp4'),
            epsilon=trainer.epsilon
        )
        print(f"   Final {i+1}: reward={eval_reward:+.0f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTotal episodes: {episode}")
    print(f"Total steps: {trainer.total_steps:,}")
    print(f"Final epsilon: {trainer.epsilon:.4f}")
    print(f"Mean reward (last 100): {np.mean(trainer.episode_rewards[-100:]):+.1f}")
    
    print(f"\n📁 Videos saved in: {output_dir}/")
    print("\n💡 What to look for in Pong GradCAM videos:")
    print("   - Agent should focus on the BALL (moving object)")
    print("   - Agent should focus on OWN PADDLE position")
    print("   - Should NOT focus on score, background, or opponent")
    
    return trainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--record-every', type=int, default=100)
    parser.add_argument('--output', type=str, default='outputs/atari_pong')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    args = parser.parse_args()
    
    trainer = train_atari_with_gradcam(
        env_name="ALE/Pong-v5",
        n_steps=args.steps,
        record_every_episodes=args.record_every,
        output_dir=args.output
    )
