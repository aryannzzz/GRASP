# Complete Implementation Guide for RoboPrompt SO101

## Table of Contents
1. [Understanding the Pipeline](#understanding-the-pipeline)
2. [Step-by-Step Implementation](#step-by-step-implementation)
3. [Component Deep Dive](#component-deep-dive)
4. [Integration Guide](#integration-guide)
5. [Testing & Debugging](#testing-debugging)
6. [Advanced Topics](#advanced-topics)

---

## Understanding the Pipeline

### What is RoboPrompt?

RoboPrompt uses **In-Context Learning (ICL)** with Large Language Models to predict robot actions. Instead of training a model on your data, you show the LLM a few examples (demonstrations) and ask it to predict actions for a new scenario.

**Key Insight**: By discretizing continuous robot actions into bins (like "bin 10 for x-position", "bin 5 for roll"), we can use text-based LLMs for robot control!

### The Complete Workflow

```
[Dataset] → [Keyframes] → [FK] → [Discretize] → [Demonstrations]
                                                        ↓
[Test Image] → [Object Poses] → [ICL Prompt] → [GPT-4] → [Actions]
                                                            ↓
                                               [IK] → [Joints] → [Robot]
```

---

## Step-by-Step Implementation

### Step 1: Install Dependencies

```bash
# Clone/download the project
cd roboprompt_so101

# Run setup script
./setup.sh

# OR manually:
pip install -r requirements.txt
cp /path/to/your/so100.urdf config/so100.urdf
export OPENAI_API_KEY=your_key_here
```

### Step 2: Prepare Your Dataset

Your dataset should be in LeRobot format with:
- Joint positions (6 DOF)
- Gripper states
- RGB images from camera

```python
# Example: Your dataset structure on HuggingFace
dataset_name = "aadarshram/act_pick_place_tape"

# Features:
# - action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# - observation.state: same as action
# - observation.images.top_phone: RGB video [480, 640, 3]
```

### Step 3: Run Offline Processing

This converts your LeRobot dataset to RoboPrompt format:

```bash
python src/main_pipeline.py \
    --mode offline \
    --task "Pick up the tape and place it on the target" \
    --num-episodes 50 \
    --demo-path output/demonstrations/processed_demonstrations.pkl
```

**What happens:**
1. Loads dataset from HuggingFace
2. Extracts keyframes (5-15 per episode vs 200+ original frames)
3. Computes forward kinematics (joint positions → EE poses)
4. Discretizes poses (continuous → discrete bins)
5. Estimates object poses (placeholder - needs FoundationPose integration)
6. Saves processed demonstrations

### Step 4: Test Inference

Run prediction on a test image:

```bash
python src/main_pipeline.py \
    --mode inference \
    --demo-path output/demonstrations/processed_demonstrations.pkl \
    --test-image path/to/test_image.jpg \
    --task "Pick up the tape and place it on the target"
```

**What happens:**
1. Loads test image
2. Estimates object poses
3. Constructs ICL prompt with demonstrations
4. Queries GPT-4
5. Parses predicted actions
6. Converts to joint trajectory
7. Executes (or simulates)

---

## Component Deep Dive

### 1. Forward Kinematics (`forward_kinematics.py`)

**Purpose**: Convert joint positions to end-effector poses

**How it works**:
- Uses PyBullet physics engine
- Loads SO101 URDF
- Computes pose of "jaw" link (end-effector)
- Returns [x, y, z, roll, pitch, yaw]

**Usage**:
```python
from forward_kinematics import ForwardKinematics

fk = ForwardKinematics("config/so100.urdf", end_effector_link="jaw")
joint_positions = np.array([0.1, 0.5, -0.3, 0.2, 0.0, 0.8])
position, orientation = fk.compute_fk(joint_positions)
# position: [x, y, z]
# orientation: [roll, pitch, yaw] in radians
```

### 2. Keyframe Extractor (`keyframe_extractor.py`)

**Purpose**: Identify critical moments in demonstrations

**Criteria** (from RoboPrompt paper):
1. Joint velocity near zero: `||velocity|| < δ` (direction change)
2. Gripper state change: open ↔ closed

**Usage**:
```python
from keyframe_extractor import KeyframeExtractor

extractor = KeyframeExtractor(config)
keyframe_indices = extractor.extract_keyframes(
    joint_positions,  # [T, 6]
    gripper_states    # [T,]
)
# Returns: [0, 15, 42, 87, 120, 199] (example)
```

**Compression**: Typical 200-frame episode → 5-15 keyframes (93-97% reduction!)

### 3. Action Discretizer (`action_discretizer.py`)

**Purpose**: Convert continuous poses to discrete bins

**Discretization** (from RoboPrompt paper):
- **Translation** (x, y, z): 100 bins each
- **Rotation** (roll, pitch, yaw): 72 bins each (5° per bin)
- **Gripper**: Binary (0=open, 1=closed)

**Usage**:
```python
from action_discretizer import ActionDiscretizer

discretizer = ActionDiscretizer(config)

# Continuous pose
pose = np.array([0.15, 0.05, 0.2, 0.5, -0.3, 1.0])  # [x,y,z,r,p,y]

# Discretize
discrete = discretizer.discretize_pose(pose)
# Returns: [45, 67, 82, 35, 20, 58] (bin indices)

# Convert back
continuous = discretizer.discrete_to_continuous(discrete)
# Returns: [0.1501, 0.0498, 0.2003, 0.502, -0.301, 0.998] (bin centers)
```

### 4. Pose Estimator (`pose_estimator.py`)

**Purpose**: Detect object poses in images

**Current Status**: Placeholder implementation
**Integration Needed**: FoundationPose

**How to integrate**:
```python
# In pose_estimator.py, replace _init_foundationpose():
from foundationpose import FoundationPose

self.model = FoundationPose(
    model_path=str(self.model_path),
    use_rgb_only=True  # Per author's email
)

# In estimate_poses():
pose = self.model.estimate_pose(image, object_name)
```

**Note**: Based on author's email, use RGB images only (no depth required)

### 5. ICL Constructor (`icl_constructor.py`)

**Purpose**: Build prompts for GPT-4

**Prompt Format** (from RoboPrompt paper):
```
x1 > y1, x2 > y2, ..., xn > yn, xTest >
```

Where:
- `xi` = Input: {[object1]: [pose], [object2]: [pose], [Task]: instruction}
- `yi` = Output: {[action1], [action2], ...}

**Example prompt**:
```
{[tape]: [10, 20, 30, 5, 10, 15], [Task]: pick tape} > {[15, 25, 35, 8, 12, 18, 0], [10, 20, 30, 5, 10, 15, 1]},
{[tape]: [12, 22, 32, 6, 11, 16], [Task]: pick tape} > {[17, 27, 37, 9, 13, 19, 0], [12, 22, 32, 6, 11, 16, 1]},
{[tape]: [11, 21, 31, 5, 10, 16], [Task]: pick tape} >
```

GPT-4 completes the last part!

### 6. Action Executor (`action_executor.py`)

**Purpose**: Convert predicted EE poses to joint commands

**The Challenge**: RoboPrompt predicts discrete EE poses, but robot needs joint positions

**Solution: Nearest Neighbor IK**
1. Build database: (EE pose, joint positions) pairs from dataset
2. For new EE pose: find closest pose in database
3. Use those joint positions

**Why not traditional IK?**
- IK solvers can fail for some poses
- May produce unexpected joint configurations
- Nearest neighbor guarantees feasible joints (from actual demonstrations)

**Usage**:
```python
from action_executor import ActionExecutor

executor = ActionExecutor(config, discretizer, fk_solver)

# Build database from demonstrations
executor.build_database(demonstrations)

# Convert discrete actions to joints
discrete_actions = [[10, 20, 30, 5, 10, 15], ...]
discrete_grippers = [0, 1, 0]

joint_trajectory = executor.execute_action_sequence(
    discrete_actions,
    discrete_grippers
)
# Returns: [T, 7] trajectory (6 joints + gripper)
```

---

## Integration Guide

### Integrating FoundationPose

1. **Install FoundationPose**:
```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
# Follow their installation instructions
```

2. **Update `pose_estimator.py`**:
```python
# Add at top
from foundationpose import FoundationPose

# In _init_foundationpose():
self.model = FoundationPose(
    model_path=str(self.model_path),
    device='cuda:0'
)

# In estimate_poses():
pose_matrix = self.model.estimate_pose(
    rgb=image,
    object_name=object_name
)
# Convert pose_matrix to [x, y, z, roll, pitch, yaw]
```

3. **Prepare object models**:
- Get CAD model or reference images of objects
- Place in `models/objects/`

### Integrating SO101 Control

1. **Find control interface**:
```python
# LeRobot usually provides something like:
from lerobot.common.robot_devices.robots.so100 import SO100Robot

robot = SO100Robot()
```

2. **Update `action_executor.py`**:
```python
# In send_to_robot():
robot = SO100Robot()

for waypoint in joint_trajectory:
    joint_positions = waypoint[:6]
    gripper_position = waypoint[6]
    
    robot.set_joint_positions(joint_positions)
    robot.set_gripper(gripper_position)
    
    time.sleep(1.0 / self.control_frequency)
```

---

## Testing & Debugging

### Test Individual Components

Each module has a `if __name__ == "__main__":` section for testing:

```bash
# Test forward kinematics
python src/forward_kinematics.py

# Test keyframe extraction
python src/keyframe_extractor.py

# Test action discretization
python src/action_discretizer.py

# Test ICL construction
python src/icl_constructor.py

# Test LLM interface
python src/llm_interface.py
```

### Common Issues

**Issue**: "Dataset not found"
```bash
# Check dataset exists
huggingface-cli login
huggingface-cli repo info aadarshram/act_pick_place_tape
```

**Issue**: "OpenAI API error"
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test API
python -c "from openai import OpenAI; client = OpenAI(); print('OK')"
```

**Issue**: "PyBullet errors"
```bash
# Reinstall PyBullet
pip uninstall pybullet
pip install pybullet
```

### Debugging Tips

1. **Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Visualize keyframes**:
```python
extractor.visualize_keyframes(
    joint_positions,
    gripper_states,
    keyframe_indices,
    save_path="output/keyframes.png"
)
```

3. **Check discretization**:
```python
# Verify round-trip
original = np.array([0.1, 0.05, 0.2, 0.5, -0.3, 1.0])
discrete = discretizer.discretize_pose(original)
recovered = discretizer.discrete_to_continuous(discrete)
error = np.abs(original - recovered)
print(f"Max error: {error.max()}")  # Should be < bin_size
```

---

## Advanced Topics

### Improving Performance

1. **Better demonstrations**: Collect more varied, high-quality demonstrations
2. **Tune discretization**: Adjust bin counts for your workspace
3. **Better IK**: Train learned mapping or use better IK solver
4. **More shots**: Use more demonstrations in ICL (5-10 works well)

### Scaling to Multiple Tasks

Create task-specific demonstration libraries:

```python
demonstrations = {
    'pick_place_tape': load_demonstrations('tape.pkl'),
    'stack_blocks': load_demonstrations('blocks.pkl'),
    'pour_water': load_demonstrations('pour.pkl')
}

# Select based on task
task = "pick and place tape"
demo_lib = demonstrations['pick_place_tape']
result = pipeline.run_inference(test_image, task, demo_lib)
```

### Using Different LLMs

The code supports any OpenAI-compatible API:

```python
# In config.yaml
icl:
  llm_model: "gpt-4-turbo"  # or "gpt-4o", "gpt-3.5-turbo"
  
# Or use Claude
# Modify llm_interface.py to use Anthropic API
```

### Adding More Objects

```yaml
# In config.yaml
pose_estimation:
  objects:
    - name: "tape"
    - name: "bowl"
    - name: "target_location"
```

---

## Next Steps

1. ✅ **You have**: Complete codebase
2. ⏳ **To do**: Integrate FoundationPose
3. ⏳ **To do**: Integrate SO101 control
4. ⏳ **To do**: Test on real robot
5. ⏳ **To do**: Tune and improve

**Start here**:
1. Run offline processing on your dataset
2. Inspect the processed demonstrations
3. Try inference with dummy test image
4. Integrate real components incrementally

Good luck! 🚀
