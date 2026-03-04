"""
Complete RoboPrompt Inference Pipeline - UPDATED FOR NEW FORMAT

End-to-end pipeline:
1. Load ICL demonstrations (from 0.txt format)
2. Create RoboPrompt prompt with current observation
3. Call OpenAI API to get predicted actions
4. Convert actions to SO101 joint angles
5. Save results

Usage:
    python run_inference_pipeline.py --icl-file ./roboprompt_data/0.txt
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

from simplified_bridge import SimplifiedRoboPromptBridge


# System prompt from RoboPrompt
SYSTEM_PROMPT = """You are a Franka Panda robot with a parallel gripper. We provide you with some demos in the format of observation>[action_1, action_2, ...]. Then you will receive a new observation and you need to output a sequence of actions that match the trends in the demos. Do not output anything else."""


class RoboPromptInferencePipeline:
    """
    Complete inference pipeline for RoboPrompt → SO101.
    Updated for new format: {objects}>>[actions]
    """
    
    def __init__(
        self,
        icl_file: str,
        openai_model: str = "gpt-4-turbo",
        scene_bounds: List[float] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            icl_file: Path to ICL demonstrations file (0.txt format)
            openai_model: OpenAI model name
            scene_bounds: Scene bounds (if None, uses default)
        """
        self.icl_file = Path(icl_file)
        self.openai_model = openai_model
        
        # Load ICL demonstrations from .txt file
        print(f"\n📂 Loading ICL demonstrations from: {self.icl_file}")
        with open(self.icl_file, 'r') as f:
            icl_content = f.read().strip()
        
        # Parse the .txt file format (comma-separated demos)
        self.demonstrations = [demo.strip() for demo in icl_content.split(',\n') if demo.strip()]
        
        print(f"✅ Loaded {len(self.demonstrations)} demonstrations")
        
        # Scene bounds (use provided or default)
        if scene_bounds is None:
            # Default bounds - should match what was used in dataset creation
            scene_bounds = [-0.4, -0.4, 0.5, 0.4, 0.4, 1.2]
        
        # Initialize conversion bridge
        self.bridge = SimplifiedRoboPromptBridge(scene_bounds=scene_bounds)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")
        self.client = OpenAI(api_key=api_key)
        
        print(f"✅ Pipeline initialized")
    
    def create_prompt(
        self,
        test_object_poses: Dict[str, np.ndarray]
    ) -> str:
        """
        Create RoboPrompt prompt with ICL demos + test observation.
        
        NEW FORMAT: {objects}>>[actions] (no instruction)
        
        Args:
            test_object_poses: Dictionary mapping object names to positions
                Format: {'object_name': [x, y, z]}  # POSITION ONLY
        
        Returns:
            Complete prompt string for LLM
        """
        # Format demonstrations from ICL file
        prompt = ", ".join(self.demonstrations)
        
        # Add test observation - ONLY POSITION, no instruction
        # Discretize object poses to match ICL format
        test_obs_parts = []
        for obj_name, pose in test_object_poses.items():
            position = pose[:3]  # Only use position
            
            # Discretize position (same as in dataset creation)
            bounds = np.array(self.bridge.scene_bounds)
            pos_normalized = (position - bounds[:3]) / (bounds[3:] - bounds[:3])
            pos_bins = np.clip((pos_normalized * 100).astype(int), 0, 99)
            
            test_obs_parts.append(f"'{obj_name}': [{pos_bins[0]}, {pos_bins[1]}, {pos_bins[2]}]")
        
        test_obs = "{" + ", ".join(test_obs_parts) + "}"
        
        # Final prompt format: demos + {observation}>
        user_prompt = prompt + f", {test_obs}>>"
        
        return user_prompt
    
    def call_llm(
        self,
        user_prompt: str
    ) -> str:
        """
        Call OpenAI API with RoboPrompt prompt.
        
        Args:
            user_prompt: User prompt with demos + observation
        
        Returns:
            LLM response text
        """
        print("\n🤖 Calling OpenAI API...")
        print(f"   Model: {self.openai_model}")
        print(f"   Prompt length: {len(user_prompt)} chars")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.0  # Deterministic for robotics
            )
            
            response = completion.choices[0].message.content
            print(f"✅ Got LLM response ({len(response)} chars)")
            
            return response
            
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            raise
    
    def run_inference(
        self,
        test_object_poses: Dict[str, np.ndarray],
        save_dir: str = "./results"
    ) -> dict:
        """
        Run complete inference pipeline.
        
        Args:
            test_object_poses: Current object positions in test scene
            save_dir: Directory to save results
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*80)
        print("🚀 RUNNING ROBOPROMPT INFERENCE PIPELINE")
        print("="*80)
        
        # Step 1: Create prompt
        print("\n📝 Step 1: Creating RoboPrompt prompt...")
        user_prompt = self.create_prompt(test_object_poses)
        
        # Print prompt preview
        print("\n" + "-"*80)
        print("PROMPT PREVIEW:")
        print("-"*80)
        print("SYSTEM:", SYSTEM_PROMPT[:100] + "...")
        print("\nUSER (last 300 chars):")
        print(user_prompt[-300:])
        print("-"*80)
        
        # Step 2: Call LLM
        llm_response = self.call_llm(user_prompt)
        
        print("\n" + "-"*80)
        print("LLM RESPONSE:")
        print("-"*80)
        print(llm_response)
        print("-"*80)
        
        # Step 3: Parse LLM output
        print("\n📋 Step 2: Parsing LLM output...")
        discretized_actions = self.bridge.parse_llm_output(llm_response)
        print(f"✅ Parsed {len(discretized_actions)} actions")
        
        for i, action in enumerate(discretized_actions):
            print(f"   [{i}] {action}")
        
        # Step 4: Convert to SO101 format
        print("\n🔄 Step 3: Converting to SO101 joint angles...")
        converted_actions = self.bridge.convert_roboprompt_sequence(discretized_actions)
        
        # Step 5: Save results
        print("\n💾 Step 4: Saving results...")
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        results = {
            'test_object_poses': {k: v.tolist() for k, v in test_object_poses.items()},
            'llm_prompt': user_prompt,
            'llm_response': llm_response,
            'discretized_actions': [a.tolist() for a in discretized_actions],
            'converted_actions': converted_actions,
            'num_actions': len(converted_actions),
            'scene_bounds': self.bridge.scene_bounds.tolist(),
            'icl_file': str(self.icl_file)
        }
        
        results_file = save_path / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved complete results to: {results_file}")
        
        # Save just joint angles for easy execution
        joint_angles_only = [
            action['joint_angles_deg'] for action in converted_actions
        ]
        
        joint_angles_file = save_path / "joint_angles.json"
        with open(joint_angles_file, 'w') as f:
            json.dump({
                'joint_angles_deg': joint_angles_only,
                'description': 'SO101 joint angles in degrees: [j1, j2, j3, j4, j5, gripper]'
            }, f, indent=2)
        print(f"✅ Saved joint angles to: {joint_angles_file}")
        
        # Save as numpy for easy loading
        np_file = save_path / "joint_angles.npy"
        np.save(np_file, np.array(joint_angles_only))
        print(f"✅ Saved numpy array to: {np_file}")
        
        print("\n" + "="*80)
        print("🎉 INFERENCE COMPLETE!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   - Predicted {len(discretized_actions)} actions")
        print(f"   - Successfully converted {len(converted_actions)} actions")
        print(f"   - Results saved to: {save_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run RoboPrompt inference and convert to SO101 joint angles"
    )
    parser.add_argument(
        '--icl-file',
        type=str,
        default='./roboprompt_data/0.txt',
        help='Path to ICL demonstrations .txt file (0.txt format)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4-turbo',
        help='OpenAI model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RoboPromptInferencePipeline(
        icl_file=args.icl_file,
        openai_model=args.model
    )
    
    # Define test object poses - ONLY POSITION, no rotation
    # IMPORTANT: Replace these with actual object poses from your perception system!
    print("\n⚠️  Using example object poses - replace with real perception!")
    test_object_poses = {
        'tape': np.array([0.2, -0.1, 0.8]),  # Only [x, y, z] - no rotation!
        'target_zone': np.array([0.3, 0.2, 0.7]),
        'table': np.array([0.0, 0.0, 0.6])
    }
    
    # Run inference
    results = pipeline.run_inference(
        test_object_poses=test_object_poses,
        save_dir=args.save_dir
    )
    
    # Print final joint angles for easy copying
    print("\n" + "="*80)
    print("📋 FINAL SO101 JOINT ANGLES (degrees)")
    print("="*80)
    for i, action in enumerate(results['converted_actions']):
        angles = action['joint_angles_deg']
        print(f"[{i}] {[round(a, 1) for a in angles]}")
    print("="*80)


if __name__ == "__main__":
    main()
