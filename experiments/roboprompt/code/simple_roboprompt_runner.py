# simple_roboprompt_runner.py
import os
import numpy as np
from hf_dataset_loader import HuggingFaceDatasetLoader
import json

class SimpleRoboPromptRunner:
    """
    Simplified RoboPrompt runner for HuggingFace datasets.
    This bypasses the RLBench dependencies and works directly with your generated demonstrations.
    """
    
    def __init__(self, demonstrations_path="./roboprompt_data/0.txt"):
        self.demonstrations_path = demonstrations_path
        self.system_prompt = """You are a Franka Panda robot with a parallel gripper. We provide you with some demos in the format of observation>[action_1, action_2, ...]. Then you will receive a new observation and you need to output a sequence of actions that match the trends in the demos. Do not output anything else."""
        
    def load_demonstrations(self):
        """Load the generated demonstrations"""
        if not os.path.exists(self.demonstrations_path):
            raise FileNotFoundError(f"Demonstrations not found at {self.demonstrations_path}")
        
        with open(self.demonstrations_path, 'r') as f:
            content = f.read().strip()
        
        # Parse the demonstrations (they're comma-separated)
        demonstrations = []
        demo_strings = content.split('>,')
        
        for demo_str in demo_strings:
            demo_str = demo_str.strip()
            if demo_str:
                if not demo_str.endswith('>'):
                    demo_str += '>'
                demonstrations.append(demo_str)
        
        print(f"📂 Loaded {len(demonstrations)} demonstrations")
        return demonstrations
    
    def create_observation(self, object_poses):
        """
        Create observation string from object poses.
        
        Args:
            object_poses: Dict mapping object names to [x, y, z] positions
        
        Returns:
            Observation string in RoboPrompt format
        """
        obs_parts = []
        for obj_name, pose in object_poses.items():
            obs_parts.append(f"'{obj_name}': [{pose[0]}, {pose[1]}, {pose[2]}]")
        
        return "{" + ", ".join(obs_parts) + "}"
    
    def format_prompt(self, demonstrations, current_observation):
        """
        Format the complete prompt for the LLM.
        
        Args:
            demonstrations: List of demonstration strings
            current_observation: Current observation string
        
        Returns:
            Complete prompt string
        """
        # Join demonstrations with commas
        demos_str = ",\n".join(demonstrations)
        
        # Full prompt format
        prompt = f"{self.system_prompt}\n\nDemonstrations:\n{demos_str},\n{current_observation}>"
        
        return prompt
    
    def run_inference(self, current_object_poses, model_name="gpt-4"):
        """
        Run RoboPrompt inference with current object poses.
        
        Args:
            current_object_poses: Dict of current object positions
            model_name: LLM model to use
        
        Returns:
            Generated action sequence
        """
        # Load demonstrations
        demonstrations = self.load_demonstrations()
        
        # Create current observation
        current_obs = self.create_observation(current_object_poses)
        
        # Format complete prompt
        prompt = self.format_prompt(demonstrations, current_obs)
        
        print("🚀 RoboPrompt Prompt Ready!")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        # Here you would send the prompt to your LLM
        # For now, we'll return the formatted prompt
        return prompt

def main():
    """Test the RoboPrompt runner with your HuggingFace dataset"""
    
    # Step 1: Generate demonstrations if not already done
    if not os.path.exists("./roboprompt_data/0.txt"):
        print("📦 Generating demonstrations from HuggingFace dataset...")
        loader = HuggingFaceDatasetLoader(
            dataset_name="aadarshram/pick_place_tape",
            scene_bounds=[-0.4, -0.4, 0.5, 0.4, 0.4, 1.2]
        )
        
        episodes_to_use = [0, 1, 2]
        object_poses = {
            'tape': np.array([0.2, -0.1, 0.8]),
            'target_zone': np.array([0.3, 0.2, 0.7]),
            'table': np.array([0.0, 0.0, 0.6])
        }
        
        loader.save_demonstrations(
            episodes_to_process=episodes_to_use,
            object_poses=object_poses,
            save_dir="./roboprompt_data"
        )
    
    # Step 2: Run RoboPrompt inference
    runner = SimpleRoboPromptRunner()
    
    # Define current object poses (this would come from your perception system)
    current_object_poses = {
        'tape': [75, 37, 42],      # Example bin values
        'target_zone': [87, 75, 28],
        'table': [50, 50, 14]
    }
    
    # Run inference
    prompt = runner.run_inference(current_object_poses)
    
    print("\n✅ RoboPrompt ready for LLM inference!")
    print("\n📋 Next steps:")
    print("1. Copy the prompt above and send it to your preferred LLM (GPT-4, Claude, etc.)")
    print("2. The LLM should return a sequence of actions in the format: [[x, y, z, rx, ry, rz, gripper], ...]")
    print("3. Parse the actions and execute them on your robot")

if __name__ == "__main__":
    main()