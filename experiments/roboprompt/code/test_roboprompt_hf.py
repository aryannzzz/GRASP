# test_roboprompt_hf.py
import numpy as np
from hf_dataset_loader import HuggingFaceDatasetLoader
from form_icl_demonstrations import create_task_handler, SYSTEM_PROMPT

def test_hf_demonstrations():
    """Test the HuggingFace dataset integration with RoboPrompt"""
    
    # 1. Generate demonstrations
    loader = HuggingFaceDatasetLoader(
        dataset_name="aadarshram/pick_place_tape",
        scene_bounds=[-0.4, -0.4, 0.5, 0.4, 0.4, 1.2]
    )
    
    episodes_to_use = [0, 1, 2]  # Start with fewer episodes for testing
    
    # Adjust object poses based on your actual scene
    object_poses = {
        'tape': np.array([0.2, -0.1, 0.8]),
        'target_zone': np.array([0.3, 0.2, 0.7]),
        'table': np.array([0.0, 0.0, 0.6])
    }
    
    # Save demonstrations
    txt_path = loader.save_demonstrations(
        episodes_to_process=episodes_to_use,
        object_poses=object_poses,
        save_dir="./roboprompt_data"
    )
    
    print("✅ Demonstrations generated successfully!")
    return txt_path

def run_roboprompt():
    """Run RoboPrompt with the generated demonstrations"""
    
    # Create task handler for your task
    handler = create_task_handler("pick_place_tape")
    
    # For testing, you'll need to provide current observation data
    # This would come from your current robot/camera setup
    print("🚀 RoboPrompt ready to use with HuggingFace demonstrations!")
    print("System Prompt:", SYSTEM_PROMPT)
    
    # In practice, you would call:
    # user_prompt = handler.get_user_prompt(current_mask_dict, current_mask_id_to_sim_name, current_point_cloud_dict)
    # Then feed this to your LLM

if __name__ == "__main__":
    # Generate demonstrations
    demo_path = test_hf_demonstrations()
    
    # Set up RoboPrompt
    run_roboprompt()
    
    print(f"\n📋 Next steps:")
    print(f"1. Review generated demonstrations in: {demo_path}")
    print(f"2. Adjust object poses in the loader if bin values look incorrect")
    print(f"3. Run RoboPrompt inference with: handler.get_user_prompt(...)")