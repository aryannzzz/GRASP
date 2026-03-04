"""
Visual Frozen Lake Wrapper
Converts discrete state indices to visual grid representations for GradCAM analysis.

The key insight: GradCAM needs spatial structure to show "what the agent looks at".
We create a visual representation where attention patterns become interpretable.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from typing import Tuple, Optional, Dict, Any


class VisualFrozenLakeWrapper(gym.Wrapper):
    """
    Wraps FrozenLake to output visual observations instead of state indices.
    
    This wrapper is essential for RL-GradCAM because:
    1. Original FrozenLake returns discrete state index (0-15)
    2. GradCAM requires spatial structure (H, W, C) to visualize attention
    3. Visual representation makes attention patterns interpretable
    
    Features:
    - Renders grid with color-coded tiles (Safe, Hole, Goal, Start)
    - Shows agent position clearly
    - Optional: highlights dangerous neighbors, distance to goal
    - Outputs normalized float tensor ready for CNN
    """
    
    # Color definitions (RGB)
    COLORS = {
        'S': np.array([144, 238, 144]),   # Start - Light green
        'F': np.array([200, 200, 200]),   # Frozen - Light gray  
        'H': np.array([65, 105, 225]),    # Hole - Royal blue
        'G': np.array([255, 215, 0]),     # Goal - Gold
        'agent': np.array([220, 20, 60]), # Agent - Crimson red
        'grid_line': np.array([50, 50, 50]),  # Grid lines - Dark gray
    }
    
    # Action names for interpretability
    ACTION_NAMES = ['Left', 'Down', 'Right', 'Up']
    ACTION_ARROWS = ['←', '↓', '→', '↑']
    
    def __init__(
        self, 
        env: gym.Env,
        tile_size: int = 16,
        show_grid_lines: bool = True,
        highlight_danger: bool = False,
        show_goal_gradient: bool = False,
        normalize: bool = True
    ):
        """
        Args:
            env: FrozenLake gym environment
            tile_size: Pixels per grid tile (affects resolution)
            show_grid_lines: Draw lines between tiles
            highlight_danger: Mark tiles adjacent to holes
            show_goal_gradient: Add distance-to-goal overlay
            normalize: Output [0,1] floats instead of [0,255] uint8
        """
        super().__init__(env)
        
        self.tile_size = tile_size
        self.show_grid_lines = show_grid_lines
        self.highlight_danger = highlight_danger
        self.show_goal_gradient = show_goal_gradient
        self.normalize = normalize
        
        # Get grid info from environment
        self.desc = self.env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape
        self.grid_size = self.nrow  # Assuming square grid
        
        # Calculate image dimensions
        self.img_height = self.nrow * tile_size
        self.img_width = self.ncol * tile_size
        
        # Update observation space
        if normalize:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(3, self.img_height, self.img_width),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(3, self.img_height, self.img_width),
                dtype=np.uint8
            )
        
        # Precompute tile positions and types
        self._precompute_grid_info()
        
        # Store current state for rendering
        self.current_state = 0
        
    def _precompute_grid_info(self):
        """Precompute grid layout for efficient rendering."""
        self.tile_types = {}
        self.goal_pos = None
        self.hole_positions = []
        self.danger_zones = set()
        
        for row in range(self.nrow):
            for col in range(self.ncol):
                tile_type = self.desc[row, col].decode('utf-8')
                state_idx = row * self.ncol + col
                self.tile_types[state_idx] = tile_type
                
                if tile_type == 'G':
                    self.goal_pos = (row, col)
                elif tile_type == 'H':
                    self.hole_positions.append((row, col))
        
        # Compute danger zones (tiles adjacent to holes)
        if self.highlight_danger:
            for hole_row, hole_col in self.hole_positions:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = hole_row + dr, hole_col + dc
                    if 0 <= nr < self.nrow and 0 <= nc < self.ncol:
                        state_idx = nr * self.ncol + nc
                        if self.tile_types[state_idx] not in ['H', 'G']:
                            self.danger_zones.add(state_idx)
    
    def _state_to_rowcol(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) position."""
        return state // self.ncol, state % self.ncol
    
    def _render_tile(self, image: np.ndarray, row: int, col: int, tile_type: str):
        """Render a single tile on the image."""
        y1 = row * self.tile_size
        y2 = (row + 1) * self.tile_size
        x1 = col * self.tile_size
        x2 = (col + 1) * self.tile_size
        
        # Base color
        color = self.COLORS[tile_type].copy()
        
        # Apply danger zone tint if enabled
        state_idx = row * self.ncol + col
        if self.highlight_danger and state_idx in self.danger_zones:
            # Add reddish tint to danger zones
            color = (color * 0.7 + np.array([255, 100, 100]) * 0.3).astype(np.uint8)
        
        # Fill tile
        image[y1:y2, x1:x2] = color
        
        # Add goal gradient overlay if enabled
        if self.show_goal_gradient and self.goal_pos is not None:
            goal_row, goal_col = self.goal_pos
            distance = abs(row - goal_row) + abs(col - goal_col)
            max_dist = self.nrow + self.ncol - 2
            # Closer to goal = slight golden tint
            if distance > 0:
                gold_factor = 1.0 - (distance / max_dist) * 0.3
                gold_tint = np.array([255, 215, 0]) * (1 - gold_factor) * 0.2
                image[y1:y2, x1:x2] = np.clip(
                    image[y1:y2, x1:x2].astype(float) + gold_tint, 0, 255
                ).astype(np.uint8)
    
    def _render_agent(self, image: np.ndarray, state: int):
        """Render agent as a circle at current position."""
        row, col = self._state_to_rowcol(state)
        
        # Calculate center of tile
        center_x = col * self.tile_size + self.tile_size // 2
        center_y = row * self.tile_size + self.tile_size // 2
        radius = self.tile_size // 3
        
        # Draw filled circle for agent
        cv2.circle(
            image, 
            (center_x, center_y), 
            radius, 
            self.COLORS['agent'].tolist(), 
            thickness=-1  # Filled
        )
        
        # Add white border for visibility
        cv2.circle(
            image,
            (center_x, center_y),
            radius,
            [255, 255, 255],
            thickness=1
        )
    
    def _render_grid_lines(self, image: np.ndarray):
        """Draw grid lines between tiles."""
        color = self.COLORS['grid_line'].tolist()
        
        # Vertical lines
        for col in range(1, self.ncol):
            x = col * self.tile_size
            cv2.line(image, (x, 0), (x, self.img_height), color, 1)
        
        # Horizontal lines
        for row in range(1, self.nrow):
            y = row * self.tile_size
            cv2.line(image, (0, y), (self.img_width, y), color, 1)
    
    def _state_to_image(self, state: int) -> np.ndarray:
        """
        Convert state index to visual grid image.
        
        Args:
            state: State index (0 to nrow*ncol-1)
            
        Returns:
            image: (C, H, W) tensor, normalized to [0,1] if self.normalize
        """
        # Create blank image
        image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # Render all tiles
        for row in range(self.nrow):
            for col in range(self.ncol):
                tile_type = self.desc[row, col].decode('utf-8')
                self._render_tile(image, row, col, tile_type)
        
        # Draw grid lines
        if self.show_grid_lines:
            self._render_grid_lines(image)
        
        # Render agent
        self._render_agent(image, state)
        
        # Convert to (C, H, W) format for PyTorch
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Normalize if requested
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return visual observation."""
        result = self.env.reset(**kwargs)
        
        # Handle both old and new gym API
        if isinstance(result, tuple):
            state, info = result
        else:
            state, info = result, {}
        
        self.current_state = state
        visual_obs = self._state_to_image(state)
        
        # Add useful info
        info['state_index'] = state
        info['agent_position'] = self._state_to_rowcol(state)
        info['grid_size'] = (self.nrow, self.ncol)
        
        return visual_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return visual observation."""
        result = self.env.step(action)
        
        # Handle both old and new gym API
        if len(result) == 4:
            next_state, reward, done, info = result
            truncated = False
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        self.current_state = next_state
        visual_obs = self._state_to_image(next_state)
        
        # Add useful info for interpretability
        info['state_index'] = next_state
        info['agent_position'] = self._state_to_rowcol(next_state)
        info['action_taken'] = action
        info['action_name'] = self.ACTION_NAMES[action]
        
        # Check what tile agent landed on
        tile_type = self.tile_types[next_state]
        info['landed_on'] = tile_type
        info['fell_in_hole'] = (tile_type == 'H')
        info['reached_goal'] = (tile_type == 'G')
        
        return visual_obs, reward, done, truncated, info
    
    def get_state_image(self, state: int) -> np.ndarray:
        """Get visual representation for any state (useful for analysis)."""
        return self._state_to_image(state)
    
    def get_all_state_images(self) -> Dict[int, np.ndarray]:
        """Get visual representations for all states."""
        return {s: self._state_to_image(s) for s in range(self.nrow * self.ncol)}
    
    def get_action_info(self) -> Dict[str, Any]:
        """Return action space information for interpretability."""
        return {
            'n_actions': self.env.action_space.n,
            'action_names': self.ACTION_NAMES,
            'action_arrows': self.ACTION_ARROWS,
            'action_meanings': {
                0: 'Move Left (decrease column)',
                1: 'Move Down (increase row)',
                2: 'Move Right (increase column)', 
                3: 'Move Up (decrease row)'
            }
        }
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Return grid information for interpretability."""
        return {
            'shape': (self.nrow, self.ncol),
            'tile_types': self.tile_types,
            'goal_position': self.goal_pos,
            'hole_positions': self.hole_positions,
            'danger_zones': list(self.danger_zones),
            'tile_size_pixels': self.tile_size,
            'image_size': (self.img_height, self.img_width)
        }


def make_visual_frozen_lake(
    map_name: str = "4x4",
    is_slippery: bool = False,
    tile_size: int = 16,
    **wrapper_kwargs
) -> VisualFrozenLakeWrapper:
    """
    Factory function to create a visual FrozenLake environment.
    
    Args:
        map_name: "4x4" or "8x8"
        is_slippery: Whether ice is slippery (stochastic transitions)
        tile_size: Pixels per tile
        **wrapper_kwargs: Additional args for VisualFrozenLakeWrapper
        
    Returns:
        Wrapped environment with visual observations
    """
    env = gym.make(
        'FrozenLake-v1',
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=None  # We do our own rendering
    )
    
    return VisualFrozenLakeWrapper(env, tile_size=tile_size, **wrapper_kwargs)


# Quick test
if __name__ == "__main__":
    print("Testing VisualFrozenLakeWrapper...")
    
    # Create environment
    env = make_visual_frozen_lake(
        map_name="4x4",
        is_slippery=False,
        tile_size=16,
        highlight_danger=True
    )
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"Initial info: {info}")
    
    # Test step
    action = 2  # Right
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nAfter action '{env.ACTION_NAMES[action]}':")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Info: {info}")
    
    # Get grid info
    print(f"\nGrid info: {env.get_grid_info()}")
    print(f"Action info: {env.get_action_info()}")
    
    print("\n✓ VisualFrozenLakeWrapper test passed!")
