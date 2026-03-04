# PyBullet VLA GradCAM: Meaningful Attention visualization

**Problem**: Your VLA GradCAM heatmaps showed random dots and unclear attention patterns. Why?

**Root Cause**: GradCAM can only visualize what a model has **learned**. Your VLA had random weights, so it produced random heatmaps.

**Solution**: Train the VLA on real PyBullet demonstrations first, *then* extract GradCAM. This mirrors how Atari GradCAM works: train the DQN agent, then visualize its learned attention.

---

## Why This Works: Learning from Atari

### Atari GradCAM (✓ Works)
```
1. Train DQN for 500K steps on Pong gameplay
2. Agent learns: ball & paddle positions matter for Q-values
3. GradCAM reveals learned attention:
   → Heatmaps focus on ball (predicting trajectory)
   → Heatmaps focus on paddle (knowing current position)
   → Clean, interpretable visualizations
```

### VLA GradCAM Before (✗ Failed)
```
1. Random initialization of attention_pool + action_head
2. "Training" on 4 hand-coded (image, action) pairs from ONE synthetic image
3. 300 epochs × 4 datapoints = severe overfitting, no spatial learning
4. GradCAM on random weights:
   → "pick up red cup" = random dots everywhere
   → "pick up blue block" = heatmaps cover entire image
   → No interpretable patterns
```

### VLA GradCAM After (✓ Fixed)
```
1. Collect 50+ expert demonstrations in PyBullet simulation
   - Scripted pick-and-place trajectories
   - Record (image, instruction, action) at each timestep
   - Multiple objects, varied positions
2. Train VLA on real demonstrations (100 epochs)
   - Learns: "pick up Shoe" → attend to Shoe location
   - Learns: spatial correspondence between language & vision
3. GradCAM reveals learned attention:
   → "pick up the Shoe" = focused heatmap on Shoe
   → "pick up the Bus" = heatmap shifts to Bus
   → Clean, object-specific attention patterns
```

---

## Installation

```bash
# Core dependencies (if not already installed)
pip install torch torchvision transformers pillow matplotlib opencv-python tqdm

# PyBullet
pip install pybullet

# Navigate to GradCAM directory
cd /home/aryannzzz/GRASP/GradCAM
```

### Optional: Google Scanned Objects (GSO)

For more realistic objects, download GSO:
```bash
# Follow instructions at: https://github.com/google-research/google-scanned-objects
# Mount in Colab or provide path via --gso-path
```

Without GSO, the pipeline will use PyBullet's built-in objects (plane, table, simple shapes).

---

## Quick Start

### Option 1: Full Pipeline (Recommended for first run)

```bash
# Quick test (10 episodes, 20 epochs - ~10 minutes on CPU)
python pybullet_vla/run_pipeline.py --quick --device cpu

# Full pipeline (50 episodes, 100 epochs - ~1-2 hours)
python pybullet_vla/run_pipeline.py \
    --n-episodes 50 \
    --epochs 100 \
    --device cuda    # Use GPU if available

# With GSO objects
python pybullet_vla/run_pipeline.py \
    --gso-path /path/to/GSO \
    --n-episodes 50 \
    --epochs 100
```

This will:
1. Collect demonstrations in PyBullet (headless simulation)
2. Train VLA on demonstrations
3. Generate GradCAM visualizations

**Output**: `outputs/pybullet_vla/gradcam/comparison_delta_x.png` - Compare this with your original `robot_scene_comparison_dx.png`!

### Option 2: Step-by-Step

```bash
# Step 1: Collect demonstrations
python -m pybullet_vla.data_collector \
    --output-dir outputs/pybullet_vla/demos \
    --n-episodes 50 \
    --image-size 224

# Step 2: Train VLA
python -m pybullet_vla.train \
    --data-dir outputs/pybullet_vla/demos \
    --output-dir outputs/pybullet_vla/training \
    --epochs 100 \
    --batch-size 32 \
    --device cuda

# Step 3: Extract GradCAM
python -m pybullet_vla.gradcam_pybullet \
    --checkpoint outputs/pybullet_vla/training/best_model.pt \
    --output-dir outputs/pybullet_vla/gradcam \
    --n-objects 3 \
    --generate-video    # Optional: create video like Atari demos
```

---

## Understanding the Output

### 1. Comparison Image (`comparison_delta_x.png`)

Side-by-side GradCAM heatmaps for different instructions on the same scene.

**What to look for** (if training succeeded):
- "pick up the Shoe" → heatmap highlights **Shoe region**
- "pick up the Bus" → heatmap shifts to **Bus region**
- "move to top right corner" → heatmap focuses on **that workspace area**

**Contrast with old output**:
- Old (untrained): random dots, unclear patterns
- New (trained): focused, object-specific attention

### 2. Per-Action Breakdowns (`breakdown_*.png`)

Shows GradCAM for each action dimension (delta_x, delta_y, delta_z, gripper, etc.)

**Expected patterns**:
- **delta_x** (horizontal): highlights left/right object position
- **delta_y** (depth): highlights forward/backward distance
- **delta_z** (vertical): highlights object height
- **gripper**: highlights the grasp target itself

These patterns emerge from training - the model learns that different action dimensions depend on different spatial features.

### 3. Training Curves (`training_curves.png`)

- **Loss should decrease** over epochs
- **Per-dimension losses**: delta_x, delta_y, delta_z should all improve
- If loss plateaus early or doesn't decrease, collect more data or train longer

### 4. GradCAM Video (if `--generate-video`)

Shows attention shifting over time as the robot executes actions.

Like the Atari videos: attention should track relevant objects/locations as the robot moves.

---

## Architecture

```
pybullet_vla/
├── __init__.py              # Package initialization
├── env.py                   # TableTopEnv (adapted from your notebook)
├── data_collector.py        # Collects expert demonstrations
├── train.py                 # Trains VLA on demonstrations
├── gradcam_pybullet.py      # Extracts and visualizes GradCAM
└── run_pipeline.py          # End-to-end pipeline script
```

### Data Flow

```
[PyBullet Simulation]
       ↓
   Scripted trajectories (pick & place)
       ↓
   Record: (image, "pick up Shoe", [dx,dy,dz,...])
       ↓
   Dataset: demonstrations.json + images/
       ↓
   Train CLIPVLA_AttnPool
   (CLIP frozen, attention_pool + action_head trained)
       ↓
   Trained model checkpoint
       ↓
   GradCAM extraction
       ↓
   Meaningful heatmaps! ✓
```

---

## Key Differences from Original Approach

| Aspect | Original (Failed) | New (Fixed) |
|--------|------------------|-------------|
| **Training data** | 4 hand-coded pairs from 1 image | 1000s of (image, instruction, action) from 50+ episodes |
| **Scene diversity** | 1 static OpenCV-drawn image | Multiple PyBullet scenes, varied object positions |
| **Training epochs** | 300 (severe overfitting) | 100 (generalizes to new scenes) |
| **Model state** | Random/overfit weights | Learned spatial-language correspondence |
| **GradCAM output** | Random dots, unclear | Focused on mentioned objects |

---

## Troubleshooting

### Issue: GradCAM still looks random after training

**Possible causes**:
1. **Not enough data**: Collect more episodes (try 100+)
2. **Not enough training**: Train for more epochs (150-200)
3. **Learning rate too high**: Try `--lr 5e-5`
4. **Check training curves**: If loss didn't decrease, there's a training problem

**Solution**: Run with more data and monitor training loss.

### Issue: "Object not found" errors during data collection

**Cause**: PyBullet can't find object URDF files (GSO not available)

**Solution**:
- If using GSO: provide correct `--gso-path`
- Without GSO: Edit `env.py` to use PyBullet's built-in objects (duck, sphere, cube, etc.)

### Issue: Training is slow

**Solutions**:
- Use GPU: `--device cuda`
- Reduce batch size: `--batch-size 16`
- Quick test first: `--quick` flag

---

## Theory: Why Training Enables GradCAM

GradCAM computes: **∇(action) / ∇(vision_features)**

```python
# Simplified GradCAM formula
saliency = (gradients * activations).sum(dim=feature_axis)
```

**Without training** (random weights):
- Gradients are random → saliency is random
- No learned correspondence between vision and actions
- Heatmaps show noise, not attention

**With training** (learned weights):
- Model learned: vision_features[Shoe_region] → pick_action
- Gradients flow through learned pathways
- Heatmaps reveal where model "looks" for that action

**Analogy**:
- Untrained model = Random forest of connections → GradCAM shows noise
- Trained model = Highway from Shoe pixels to pick action → GradCAM shows highway

This is **exactly** why Atari GradCAM works: the DQN learned Q(s,a) through RL, so GradCAM reveals learned attention to ball/paddle.

---

## Comparison with Atari GradCAM

| Aspect | Atari| VLA |
|--------|------|-----|
| **Model** | DQN (CNN → Q-values) | CLIPVLA (ViT → actions) |
| **Training** | RL on Pong gameplay (500K steps) | Imitation learning on demos (50 episodes) |
| **What model learns** | Ball position → Q(move_up) | "Shoe" pixels → pick_action |
| **GradCAM reveals** | Attention to ball & paddle | Attention to mentioned object |
| **Video shows** | Attention tracking ball motion | Attention shifting between objects |

Both work because: **Train first, visualize learned attention second.**

---

## Citation & Acknowledgments

This implementation builds on:
- GradCAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization" (ICCV 2017)
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
- OpenVocab Pick-Place: Your Colab notebook `Copy_of_OpenVocabPickPlace_Classical_(7).ipynb`
- PyBullet: Coumans & Bai, "PyBullet, a Python module for physics simulation for games, robotics and machine learning"

---

## Next Steps

1. **Compare outputs**:
   - Old: `outputs/multi_scene_validation/robot_scene/robot_scene_comparison_dx.png`
   - New: `outputs/pybullet_vla/gradcam/comparison_delta_x.png`

2. **Experiment**:
   - Try more objects
   - Try longer training
   - Try different instructions ("push", "slide", etc.)

3. **Real robot** (future):
   - Collect demonstrations on real robot
   - Train VLA on real data
   - GradCAM will show attention on real images

The pipeline is modular - swap PyBullet for real robot data collection and the rest works the same!

---

## FAQ

**Q: Why not just use CLIP's attention maps?**

A: CLIP attention is language-image similarity, not action-specific. GradCAM on the VLA shows which vision features matter for *predicting actions*, not just matching text. This is task-specific attention.

**Q: Can I use a pre-trained VLA like OpenVLA?**

A: Yes! The `openvla_integration.py` module exists. But you'd still need to extract GradCAM from a model that was trained on robot data. The key insight remains: GradCAM visualizes learned attention, not random weights.

**Q: Why train an imitation learning model instead of using the SayCan affordance scores?**

A: SayCan's affordance is binary (object exists or not). VLA's learned spatial predictor is continuous and fine-grained (where exactly is the object, what's its pose, etc.). GradCAM reveals this spatial reasoning.

**Q: How does this compare to attention visualization in Transformers?**

A: Transformer attention (self-attention) shows token-to-token relationships. GradCAM shows input-to-output gradients (which pixels matter for which actions). Both are useful, different perspectives.

---

**Bottom line**: GradCAM is a tool for visualizing what a model has *learned*. Train your VLA properly, and GradCAM will show you meaningful, interpretable attention patterns - just like it does for Atari agents.
