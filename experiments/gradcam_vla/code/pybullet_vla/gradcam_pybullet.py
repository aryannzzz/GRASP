"""
GradCAM Extraction and Visualization for Trained VLA in PyBullet

This is the PAYOFF: After training the VLA on demonstrations, we can now extract
meaningful GradCAM heatmaps that show where the model "looks" for each instruction.

Expected behavior (if training was successful):
- "pick up the Shoe" -> heatmap focuses on the Shoe
- "pick up the Bus" -> heatmap shifts to the Bus
- "move to top right corner" -> heatmap focuses on that workspace region

This mirrors the Atari GradCAM success: train first, then visualize learned attention.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict

from .env import setup_env, TableTopEnv
from vla_gradcam.clip_vla_attn import load_clip_vla_attn
from vla_gradcam.gradcam_engine import VLAGradCAMEngine
from vla_gradcam.visualizer import VLAGradCAMVisualizer


def load_trained_model(checkpoint_path: str, device:str = "cpu"):
    """Load trained VLA model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Load base model
    model, processor = load_clip_vla_attn(device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")

    return model, processor


def generate_comparison_image(
    env: TableTopEnv,
    model,
    engine: VLAGradCAMEngine,
    viz: VLAGradCAMVisualizer,
    instructions: List[str],
    action_dim: str = "delta_x",
    output_path: str = "comparison.png"
):
    """
    Generate side-by-side comparison of GradCAM for different instructions.

    This is analogous to robot_scene_comparison_dx.png, but should show
    MEANINGFUL heatmaps now that the model is trained.
    """
    print(f"\nGenerating comparison image for action dimension: {action_dim}")

    # Capture scene
    image = env.render(view="topdown")

    # Compute GradCAM for each instruction
    results = {}
    for inst in instructions:
        print(f"  Computing GradCAM for: '{inst}'")
        result = engine.compute_full(image, inst, action_dims=[0, 1, 2, 6])
        results[inst] = result

    # Generate comparison plot
    fig = viz.plot_instruction_comparison(
        list(results.values()),
        action_dim_name=action_dim,
        save_path=output_path
    )
    plt.close(fig)

    print(f"✓ Saved comparison: {output_path}")


def generate_per_instruction_breakdown(
    env: TableTopEnv,
    model,
    engine: VLAGradCAMEngine,
    viz: VLAGradCAMVisualizer,
    instruction: str,
    output_dir: Path
):
    """Generate per-action-dimension breakdown for one instruction."""
    print(f"\nGenerating per-action breakdown for: '{instruction}'")

    image = env.render(view="topdown")
    result = engine.compute_full(image, instruction, action_dims=[0, 1, 2, 3, 4, 5, 6])

    # Save per-action saliency
    output_path = output_dir / f"breakdown_{instruction.replace(' ', '_')[:40]}.png"
    fig = viz.plot_per_action_saliency(result, save_path=str(output_path))
    plt.close(fig)

    print(f"✓ Saved: {output_path}")


def generate_gradcam_video(
    env: TableTopEnv,
    model,
    engine: VLAGradCAMEngine,
    instruction: str,
    n_frames: int = 100,
    output_path: str = "gradcam_video.mp4"
):
    """
    Generate video showing GradCAM at each timestep as the robot executes.

    This is analogous to the Atari GradCAM videos that show attention shifting
    as the agent plays.
    """
    print(f"\nGenerating GradCAM video for: '{instruction}'")

    frames = []

    # Capture frames as robot acts
    for step in range(n_frames):
        # Get current image
        image = env.render(view="topdown")

        # Compute GradCAM
        result = engine.compute_full(image, instruction, action_dims=[0, 1, 2])

        # Create overlay
        saliency = result.combined_saliency
        saliency = (saliency * 255).astype(np.uint8)
        saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]))

        heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

        # Create side-by-side
        canvas = np.hstack([image, overlay])
        frames.append(canvas)

        # Take a random action (just for demonstration)
        action = np.random.randn(7) * 0.1  # Small random movements
        action[6] = 1.0 if step < n_frames // 2 else -1.0  # Open then close gripper
        env.step(action, steps=5)

    # Save video
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 15, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

    print(f"✓ Saved video: {output_path}")


def main():
    """Main GradCAM extraction and visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract and visualize GradCAM from trained VLA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--gso-path", type=str, default=None, help="Path to GSO dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/pybullet_vla/gradcam", help="Output directory")
    parser.add_argument("--n-objects", type=int, default=3, help="Number of objects in scene")
    parser.add_argument("--generate-video", action="store_true", help="Generate GradCAM video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 80)
    print("GRADCAM EXTRACTION FOR TRAINED VLA IN PYBULLET")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    model, processor = load_trained_model(args.checkpoint, device=args.device)

    # Create GradCAM engine
    engine = VLAGradCAMEngine(model, target_layer_idx=-1)
    viz = VLAGradCAMVisualizer(figsize_scale=1.2)

    # Set up environment
    print("\nSetting up PyBullet environment...")
    env, object_names = setup_env(
        n_objects=args.n_objects,
        gui=False,
        image_size=224,
        gso_path=args.gso_path
    )
    print(f"✓ Spawned objects: {object_names}")

    # Map object names to natural language
    object_name_map = {
        "duck_vhacd": "duck",
        "sphere_small": "sphere",
        "cube_small": "cube",
        "cylinder": "cylinder",
        "jenga/jenga": "block",
    }

    # Generate instructions
    instructions = []
    for obj in object_names:
        natural_name = object_name_map.get(obj, obj.replace("_", " "))
        instructions.append(f"pick up the {natural_name}")

    # Add a location-based instruction
    instructions.append("move to the top right corner")

    print(f"\nInstructions to visualize:")
    for i, inst in enumerate(instructions, 1):
        print(f"  {i}. {inst}")

    # ==================================================================
    # 1. COMPARISON IMAGE (like robot_scene_comparison_dx.png)
    # ==================================================================
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON IMAGE")
    print("=" * 60)

    comparison_path = output_dir / "comparison_delta_x.png"
    generate_comparison_image(
        env, model, engine, viz,
        instructions=instructions[:min(4, len(instructions))],  # Max 4 for layout
        action_dim="delta_x",
        output_path=str(comparison_path)
    )

    if len(instructions) > 4:
        comparison_path_2 = output_dir / "comparison_delta_y.png"
        generate_comparison_image(
            env, model, engine, viz,
            instructions=instructions[:min(4, len(instructions))],
            action_dim="delta_y",
            output_path=str(comparison_path_2)
        )

    # ==================================================================
    # 2. PER-INSTRUCTION BREAKDOWNS
    # ==================================================================
    print("\n" + "=" * 60)
    print("GENERATING PER-ACTION-DIMENSION BREAKDOWNS")
    print("=" * 60)

    for inst in instructions[:2]:  # First 2 instructions
        generate_per_instruction_breakdown(
            env, model, engine, viz, inst, output_dir
        )

    # ==================================================================
    # 3. GRADCAM VIDEO (optional)
    # ==================================================================
    if args.generate_video:
        print("\n" + "=" * 60)
        print("GENERATING GRADCAM VIDEO")
        print("=" * 60)

        video_path = output_dir / "gradcam_demo.mp4"
        generate_gradcam_video(
            env, model, engine,
            instruction=instructions[0],
            n_frames=100,
            output_path=str(video_path)
        )

    # Clean up
    engine.remove_hooks()
    env.disconnect()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("GRADCAM EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {comparison_path.name}")
    if len(instructions) > 4:
        print(f"  - {comparison_path_2.name}")
    print(f"  - Per-instruction breakdowns")
    if args.generate_video:
        print(f"  - {video_path.name}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
If training was successful, you should see:

1. FOCUSED HEATMAPS for each instruction:
   - "pick up the Shoe" -> heatmap highlights the Shoe region
   - "pick up the Bus" -> heatmap shifts to the Bus
   - Different instructions produce DIFFERENT spatial patterns

2. PER-ACTION-DIMENSION PATTERNS:
   - delta_x (horizontal): highlights left/right object position
   - delta_y (forward/back): highlights depth/distance
   - delta_z (vertical): highlights object height
   - gripper: highlights the grasp target itself

3. CONTRAST WITH UNTRAINED MODEL:
   - Untrained: random dots, everywhere heatmaps (what you saw before)
   - Trained: focused, object-specific attention

This is the key insight: GradCAM reveals LEARNED attention, not random activations.
The Atari DQN model learned to attend to ball/paddle through training.
Your VLA model now learned to attend to objects mentioned in language.
    """)


if __name__ == "__main__":
    main()
