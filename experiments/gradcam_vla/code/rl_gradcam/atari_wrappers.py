"""
Atari Environment Wrappers for DQN Training

Standard preprocessing for Atari games:
- Frame stacking (4 frames for motion)
- Grayscale conversion
- Resizing to 84x84
- Frame skipping
- Episodic life (game over on life loss)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque
from typing import Tuple, Optional
import ale_py


def register_atari():
    """Register ALE environments with gymnasium."""
    gym.register_envs(ale_py)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset."""
    
    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) and max over last 2."""
    
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over."""
    
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = info.get('lives', 0)
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = info.get('lives', 0)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Take FIRE action on reset for environments that require it."""
    
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 grayscale."""
    
    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    """Stack k last frames for motion perception."""
    
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1]."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class TransposeImage(gym.ObservationWrapper):
    """Transpose from (H, W, C) to (C, H, W) for PyTorch."""
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.float32
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""
    
    def reward(self, reward):
        return np.sign(reward)


def make_atari_env(
    env_name: str = "ALE/Pong-v5",
    frame_stack: int = 4,
    clip_rewards: bool = True,
    episodic_life: bool = True
) -> gym.Env:
    """
    Create a fully-wrapped Atari environment.
    
    Args:
        env_name: Atari environment name (e.g., 'ALE/Pong-v5', 'ALE/Breakout-v5')
        frame_stack: Number of frames to stack
        clip_rewards: Whether to clip rewards to {-1, 0, +1}
        episodic_life: Whether to reset on life loss
    
    Returns:
        Wrapped environment with shape (4, 84, 84) float observations
    """
    register_atari()
    
    env = gym.make(env_name, render_mode=None)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    # Fire reset for games that need it
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = WarpFrame(env)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    env = FrameStack(env, k=frame_stack)
    env = ScaledFloatFrame(env)
    env = TransposeImage(env)
    
    # Store action meanings for reference
    env.ACTION_NAMES = env.unwrapped.get_action_meanings()
    
    return env


def make_atari_env_for_recording(
    env_name: str = "ALE/Pong-v5",
    frame_stack: int = 4
) -> Tuple[gym.Env, gym.Env]:
    """
    Create both a training env and a recording env (with render_mode='rgb_array').
    
    Returns:
        (train_env, record_env) tuple
    """
    train_env = make_atari_env(env_name, frame_stack=frame_stack)
    
    # Recording env - same wrappers but with render mode
    register_atari()
    record_env = gym.make(env_name, render_mode='rgb_array')
    record_env = NoopResetEnv(record_env, noop_max=30)
    record_env = MaxAndSkipEnv(record_env, skip=4)
    
    if 'FIRE' in record_env.unwrapped.get_action_meanings():
        record_env = FireResetEnv(record_env)
    
    record_env = WarpFrame(record_env)
    record_env = FrameStack(record_env, k=frame_stack)
    record_env = ScaledFloatFrame(record_env)
    record_env = TransposeImage(record_env)
    record_env.ACTION_NAMES = record_env.unwrapped.get_action_meanings()
    
    return train_env, record_env
