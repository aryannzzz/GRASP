# envs/metaworld_wrapper.py

import gymnasium as gym 
import numpy as np
import metaworld

class MetaWorldACTWrapper:
    """Wraps MetaWorld for ACT training (Updated for MetaWorld 3.0)"""
    
    def __init__(self, task_name='shelf-place-v3', camera_names=None):
        self.ml1 = metaworld.ML1(task_name)
        # ✓ MUST specify render_mode at construction
        self.env = self.ml1.train_classes[task_name](render_mode='rgb_array')
        self.task = self.ml1.train_tasks[0]
        self.env.set_task(self.task)
        
        # Default cameras for multiple views
        # Note: MetaWorld 3.0 camera names may differ, check with env.sim
        self.camera_names = camera_names or ['corner', 'corner2', 'topview', 'behindGripper']
    
    def reset(self):
        obs_state, info = self.env.reset()  # ✓ Returns (obs, info)
        obs_dict = self._get_observation(obs_state)
        return obs_dict
    
    def step(self, action):
        # ✓ Returns 5 values (terminated, truncated separate)
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        obs_dict = self._get_observation(obs_state)
        return obs_dict, reward, done, info
    
    def _get_observation(self, state):
        """Convert state to ACT-compatible observation"""
        # Render images from multiple cameras
        images = {}
        
        # Get default render (from render_mode)
        default_img = self.env.render()
        images['default'] = default_img
        
        # For multiple camera views, we need to access MuJoCo directly
        # This requires the env to have mujoco_renderer
        if hasattr(self.env, 'mujoco_renderer'):
            for cam_name in self.camera_names:
                try:
                    # Try to render from specific camera
                    img = self.env.mujoco_renderer.render(
                        render_mode='rgb_array',
                        camera_name=cam_name
                    )
                    images[cam_name] = img
                except:
                    # If camera doesn't exist, skip
                    pass
        
        # Extract proprioceptive state
        # MetaWorld state includes:
        # - gripper position (3)
        # - gripper velocity (3)  
        # - gripper state (1)
        # - object position (3)
        # - object rotation (4)
        # - goal position (3)
        # Action space is 4D: [dx, dy, dz, gripper]
        joints = state[:4]  # First 4 values for compatibility
        
        return {
            'images': images,
            'state': state,  # Full state for ground truth
            'joints': joints,  # For action space
        }
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()