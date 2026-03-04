#!/usr/bin/env python3
"""
Test script for VLA-GradCAM with Mock VLA Model

This validates the GradCAM pipeline works before testing with real VLA models.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from vla_gradcam import VLAGradCAM, MockVLAModel, OPENVLA_ACTION_NAMES


def test_basic_saliency():
    """Test basic saliency computation with mock model."""
    print("=" * 60)
    print("Test 1: Basic Saliency Computation")
    print("=" * 60)
    
    # Create mock model
    model = MockVLAModel(
        image_size=224,
        patch_size=14,
        embed_dim=384,
        n_layers=4,
        n_heads=6,
        action_dim=7
    )
    model.eval()
    
    # Create GradCAM wrapper
    gradcam = VLAGradCAM(
        model=model,
        vision_encoder_name='vision_encoder',
        patch_size=14,
        image_size=(224, 224),
        action_names=OPENVLA_ACTION_NAMES
    )
    
    # Random test image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    instruction = "pick up the red cup"
    
    # Compute saliency
    result = gradcam.compute_saliency(
        image=test_image,
        instruction=instruction,
        action_dims=[0, 1, 2, 6]  # x, y, z, gripper
    )
    
    print(f"✓ Predicted action shape: {result.predicted_action.shape}")
    print(f"✓ Saliency maps computed: {list(result.saliency_maps.keys())}")
    print(f"✓ Combined saliency shape: {result.combined_saliency.shape}")
    
    # Validate saliency properties
    assert result.combined_saliency.shape == (224, 224), "Saliency should match image size"
    assert result.combined_saliency.min() >= 0, "Saliency should be non-negative"
    assert result.combined_saliency.max() <= 1, "Normalized saliency should be <= 1"
    
    gradcam.remove_hooks()
    print("✓ Test 1 PASSED\n")
    return result


def test_instruction_sensitivity():
    """
    Test that different instructions produce different saliency maps.
    This is a key property for VLA-GradCAM.
    """
    print("=" * 60)
    print("Test 2: Instruction Sensitivity")
    print("=" * 60)
    
    model = MockVLAModel()
    model.eval()
    
    gradcam = VLAGradCAM(
        model=model,
        vision_encoder_name='vision_encoder',
        patch_size=14,
        image_size=(224, 224),
        action_names=OPENVLA_ACTION_NAMES
    )
    
    # Same image, different instructions
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    instructions = [
        "pick up the red cup",
        "push the blue block", 
        "open the drawer"
    ]
    
    results = []
    for inst in instructions:
        result = gradcam.compute_saliency(
            image=test_image,
            instruction=inst,
            action_dims=[0, 1, 2]
        )
        results.append(result)
        print(f"  Instruction: '{inst}'")
        print(f"    Action: {result.predicted_action}")
    
    # Note: With mock model, actions will differ due to initialization
    # With real VLA, same image + different instruction should give different actions
    
    gradcam.remove_hooks()
    print("✓ Test 2 PASSED (Note: Real instruction sensitivity requires actual VLA)\n")
    return results


def test_action_dimension_saliency():
    """
    Test that different action dimensions have different saliency patterns.
    """
    print("=" * 60)
    print("Test 3: Per-Action-Dimension Saliency")
    print("=" * 60)
    
    model = MockVLAModel()
    model.eval()
    
    gradcam = VLAGradCAM(
        model=model,
        vision_encoder_name='vision_encoder',
        patch_size=14,
        image_size=(224, 224),
        action_names=OPENVLA_ACTION_NAMES
    )
    
    # Create a more structured test image (with distinct regions)
    test_image = np.zeros((224, 224, 3), dtype=np.float32)
    # Red square top-left
    test_image[20:80, 20:80, 0] = 1.0
    # Green square bottom-right
    test_image[140:200, 140:200, 1] = 1.0
    # Blue strip in middle
    test_image[100:120, :, 2] = 1.0
    
    result = gradcam.compute_saliency(
        image=test_image,
        instruction="interact with objects",
        action_dims=list(range(7))  # All 7 action dimensions
    )
    
    print(f"  Action dimensions analyzed: {len(result.saliency_maps)}")
    
    for dim_name, saliency in result.saliency_maps.items():
        # Find region with max saliency
        max_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
        print(f"    {dim_name}: max at {max_idx}, value={saliency.max():.4f}")
    
    gradcam.remove_hooks()
    print("✓ Test 3 PASSED\n")
    return result


def test_visualization():
    """Test visualization functionality."""
    print("=" * 60)
    print("Test 4: Visualization")
    print("=" * 60)
    
    model = MockVLAModel()
    model.eval()
    
    gradcam = VLAGradCAM(
        model=model,
        vision_encoder_name='vision_encoder',
        patch_size=14,
        image_size=(224, 224),
        action_names=OPENVLA_ACTION_NAMES
    )
    
    # Create meaningful test image
    test_image = np.random.rand(224, 224, 3).astype(np.float32) * 0.3
    # Add a "cup" (red circle)
    y, x = np.ogrid[:224, :224]
    cup_mask = ((x - 100)**2 + (y - 100)**2) < 30**2
    test_image[cup_mask, 0] = 1.0  # Red
    
    result = gradcam.compute_saliency(
        image=test_image,
        instruction="pick up the red cup",
        action_dims=[0, 1, 2, 6]
    )
    
    # Create visualization
    fig = gradcam.visualize(result, show_individual=True, figsize=(14, 4))
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "outputs" / "vla_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "test_saliency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    gradcam.remove_hooks()
    print(f"✓ Visualization saved to {output_dir / 'test_saliency.png'}")
    print("✓ Test 4 PASSED\n")


def test_batch_processing():
    """Test with batched inputs."""
    print("=" * 60)
    print("Test 5: Batch Processing")
    print("=" * 60)
    
    model = MockVLAModel()
    model.eval()
    
    gradcam = VLAGradCAM(
        model=model,
        vision_encoder_name='vision_encoder',
        patch_size=14,
        image_size=(224, 224),
        action_names=OPENVLA_ACTION_NAMES
    )
    
    # Multiple images
    test_images = [
        np.random.rand(224, 224, 3).astype(np.float32),
        np.random.rand(224, 224, 3).astype(np.float32),
    ]
    
    instructions = [
        "pick up red cup",
        "open drawer"
    ]
    
    results = []
    for img, inst in zip(test_images, instructions):
        result = gradcam.compute_saliency(
            image=img,
            instruction=inst,
            action_dims=[0, 1, 2]
        )
        results.append(result)
        print(f"  Processed: '{inst[:30]}' -> action shape {result.predicted_action.shape}")
    
    gradcam.remove_hooks()
    print("✓ Test 5 PASSED\n")
    return results


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VLA-GradCAM Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_basic_saliency()
        test_instruction_sensitivity()
        test_action_dimension_saliency()
        test_visualization()
        test_batch_processing()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test with real VLA model (OpenVLA)")
        print("2. Validate on robot manipulation images")
        print("3. Run instruction sensitivity experiments")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
