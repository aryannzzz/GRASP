"""
Modified RoboPrompt Agent that outputs actions to LeRobot SO101 arm.

This integrates your original roboprompt_agent.py with the LeRobot bridge.
"""

from typing import List
import re
import json
import numpy as np
from PIL import Image
import os
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
from openai import OpenAI

# Import the bridge
from roboprompt_lerobot_bridge import RoboPromptToLeRobotBridge


def openai_call(model_name, messages):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content


def huggingface_call(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


class LeRobotRoboPromptAgent:
    """
    RoboPrompt agent that executes actions on LeRobot SO101 arm.
    
    This replaces the YARR Agent interface with direct robot control.
    """
    
    def __init__(
        self, 
        task_name: str,
        model_config: dict,
        robot_port: str = "/dev/ttyUSB0",
        use_robot: bool = True,
        scene_bounds: List[float] = None
    ):
        """
        Initialize the agent.
        
        Args:
            task_name: Name of the task (e.g., "button_target")
            model_config: Config dict with 'llm_call_style' and 'name'
            robot_port: Serial port for SO101 arm
            use_robot: If False, run in simulation mode
            scene_bounds: Workspace bounds [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.episode_id = -1
        self.task_name = task_name
        self.model_config = model_config
        self.step = 0
        self.actions = []
        self._prev_action = None
        
        # Initialize LeRobot bridge
        self.bridge = RoboPromptToLeRobotBridge(
            robot_port=robot_port,
            scene_bounds=scene_bounds or SCENE_BOUNDS,
            rotation_resolution=ROTATION_RESOLUTION,
            use_sim=not use_robot
        )
        
        print(f"✓ Initialized LeRobotRoboPromptAgent for task: {task_name}")
        
    def setup(self, savedir: str):
        """Setup the agent (called once before execution)."""
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        
        # Create task handler for ICL demonstrations
        self.handler = create_task_handler(self.task_name)
        
        # Setup LLM
        if self.model_config.llm_call_style == "openai":
            self.llm_call = lambda messages: openai_call(self.model_config.name, messages)
        elif self.model_config.llm_call_style == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Loading model from HuggingFace...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
            self.llm_call = lambda messages: huggingface_call(model, tokenizer, messages)
        
        print("✓ Agent setup complete")
    
    def get_observation(self, cameras: List[str] = None) -> dict:
        """
        Capture observations from cameras (to be implemented based on your setup).
        
        For LeRobot integration, you'll need to capture RGB-D images from your cameras.
        This is a placeholder that you should replace with your camera capture code.
        
        Args:
            cameras: List of camera names (e.g., ['front', 'wrist'])
        
        Returns:
            obs: Dictionary with RGB images, masks, and point clouds
        """
        if cameras is None:
            cameras = CAMERAS
        
        # PLACEHOLDER: Replace with your actual camera capture code
        # This should return RGB-D images from your camera setup
        obs = {
            f'{cam}_rgb': None,  # Replace with actual RGB image
            f'{cam}_mask': None,  # Replace with segmentation mask
            f'{cam}_point_cloud': None  # Replace with depth/point cloud
        }
        
        raise NotImplementedError(
            "You need to implement camera capture for your SO101 setup. "
            "This should capture RGB-D images from your cameras and return them "
            "in the same format as RLBench observations."
        )
        
        return obs
    
    def preprocess_observation(self, obs: dict, mask_id_to_sim_name: dict) -> str:
        """
        Preprocess observation and get LLM prediction.
        
        Args:
            obs: Observation dictionary with RGB, masks, point clouds
            mask_id_to_sim_name: Mapping from mask IDs to object names
        
        Returns:
            output_text: LLM response with predicted actions
        """
        rgb_dict = {}
        mask_dict = {}
        point_cloud_dict = {}
        
        for camera in CAMERAS:
            # Extract and process RGB
            rgb_img = obs[f'{camera}_rgb']
            # Process as needed (convert to numpy, normalize, etc.)
            rgb_dict[camera] = rgb_img
            
            # Save for debugging
            rgb_dir = os.path.join(self.savedir, 'rgb_dir', camera, str(self.episode_id))
            os.makedirs(rgb_dir, exist_ok=True)
            img = Image.fromarray(rgb_img)
            img.save(os.path.join(rgb_dir, f'{self.step}.png'))
            
            # Extract masks and point clouds
            mask_dict[camera] = obs[f'{camera}_mask']
            point_cloud_dict[camera] = obs[f'{camera}_point_cloud']
        
        # Get ICL prompt from handler
        user_prompt = self.handler.get_user_prompt(
            mask_dict, 
            mask_id_to_sim_name, 
            point_cloud_dict
        )
        
        print("\n" + "="*80)
        print("SYSTEM PROMPT:")
        print(SYSTEM_PROMPT)
        print("\n" + "="*80)
        print("USER PROMPT:")
        print(user_prompt)
        print("="*80 + "\n")
        
        # Get LLM prediction
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        output_text = self.llm_call(messages)
        
        print(f"🤖 LLM Prediction:\n{output_text}\n")
        
        return output_text
    
    def postprocess_and_execute(self, output_text: str, execution_speed: float = 0.5):
        """
        Parse LLM output and execute on SO101 arm.
        
        Args:
            output_text: Raw LLM response
            execution_speed: Execution speed multiplier (0.1 to 2.0)
        """
        # Parse actions from LLM output
        actions = self.bridge.process_roboprompt_output(output_text)
        
        print(f"\n📋 Parsed {len(actions)} actions from LLM output")
        
        # Execute on robot
        success = self.bridge.execute_roboprompt_actions(
            actions,
            execution_speed=execution_speed
        )
        
        return success
    
    def run_episode(
        self, 
        mask_id_to_sim_name: dict,
        execution_speed: float = 0.5,
        capture_observation_fn = None
    ) -> bool:
        """
        Run one episode: capture observation, get LLM prediction, execute on robot.
        
        Args:
            mask_id_to_sim_name: Mapping from mask IDs to object names
            execution_speed: How fast to execute actions
            capture_observation_fn: Optional custom function to capture observations
        
        Returns:
            success: Whether episode executed successfully
        """
        self.episode_id += 1
        self.step = 0
        
        print(f"\n{'='*80}")
        print(f"🎬 Starting Episode {self.episode_id}")
        print(f"{'='*80}\n")
        
        try:
            # Step 1: Capture observation
            if capture_observation_fn is not None:
                obs = capture_observation_fn()
            else:
                obs = self.get_observation()
            
            # Step 2: Get LLM prediction
            output_text = self.preprocess_observation(obs, mask_id_to_sim_name)
            
            # Step 3: Execute on robot
            success = self.postprocess_and_execute(output_text, execution_speed)
            
            if success:
                print(f"\n✅ Episode {self.episode_id} completed successfully!")
            else:
                print(f"\n❌ Episode {self.episode_id} failed!")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Episode {self.episode_id} error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def disconnect(self):
        """Disconnect from robot."""
        self.bridge.disconnect()


def main():
    """Example usage."""
    
    # Configuration
    model_config = {
        'llm_call_style': 'openai',  # or 'huggingface'
        'name': 'gpt-4'  # or your HuggingFace model
    }
    
    # Initialize agent
    agent = LeRobotRoboPromptAgent(
        task_name="button_target",  # Your task name
        model_config=model_config,
        robot_port="/dev/ttyUSB0",
        use_robot=False,  # Set to True for real robot
        scene_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]  # Adjust for your workspace
    )
    
    # Setup
    agent.setup(savedir="./lerobot_runs/button_target")
    
    # Example object mapping (from your camera/segmentation)
    mask_id_to_sim_name = {
        1: "button_red",
        2: "button_blue",
        3: "table"
    }
    
    # Run episode (you need to implement capture_observation_fn)
    success = agent.run_episode(
        mask_id_to_sim_name=mask_id_to_sim_name,
        execution_speed=0.5,
        # capture_observation_fn=your_camera_capture_function
    )
    
    # Cleanup
    agent.disconnect()


if __name__ == "__main__":
    main()
