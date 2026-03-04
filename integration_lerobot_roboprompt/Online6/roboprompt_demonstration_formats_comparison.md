# RoboPrompt Demonstration Formats: CoppeliaSim vs HuggingFace Approach

## 🎯 Understanding Your Confusion

You're trying to bridge **two different approaches** to creating RoboPrompt demonstrations:

### Traditional RoboPrompt (CoppeliaSim/RLBench)
- Records demonstrations in simulator
- Creates specific folder structure with multiple camera views
- Uses `form_icl_demonstrations.py` to process

### Your Approach (HuggingFace Dataset)
- Uses pre-recorded real robot demonstrations
- Loads from HuggingFace LeRobot format
- Converts directly to RoboPrompt format

---

## 📁 Original RoboPrompt Demonstration Structure

### What the Image Shows

Your screenshot shows the **original RoboPrompt folder structure** from RLBench/CoppeliaSim:

```
episode1/
├── front_depth/          # Front camera depth images
├── front_mask/           # Front camera segmentation masks
├── front_rgb/            # Front camera RGB images
├── left_shoulder_depth/  # Left shoulder camera depth
├── left_shoulder_mask/   # Left shoulder camera segmentation
├── left_shoulder_rgb/    # Left shoulder camera RGB
├── overhead_depth/       # Overhead camera depth
├── overhead_mask/        # Overhead camera segmentation  
├── overhead_rgb/         # Overhead camera RGB
├── right_shoulder_depth/ # Right shoulder camera depth
├── right_shoulder_mask/  # Right shoulder camera segmentation
├── right_shoulder_rgb/   # Right shoulder camera RGB
├── wrist_depth/          # Wrist-mounted camera depth
├── wrist_mask/           # Wrist camera segmentation
├── wrist_rgb/            # Wrist camera RGB
├── low_dim_obs.pkl       # Joint positions, gripper states
├── variation_descriptions.pkl  # Task variations
└── variation_number.pkl  # Which variation this episode is
```

### Why So Many Cameras?

RoboPrompt was originally designed for **RLBench simulator** which provides:
- 5 camera viewpoints (front, overhead, 2 shoulders, wrist)
- Perfect segmentation masks for all objects
- Perfect depth information
- Consistent lighting and setup

This gives the LLM comprehensive 3D understanding of the scene from multiple angles.

---

## 🔄 How Original RoboPrompt Works

### Step 1: Generate Data in RLBench
```bash
cd RLBench/tools/
python dataset_generator.py \
  --tasks=pick_and_lift \
  --save_path=./demos \
  --episodes_per_task=100
```

**Output:** Creates the folder structure shown in your image

### Step 2: Form ICL Demonstrations
```bash
python form_icl_demonstrations.py
```

**What it does:**
1. Loads episodes from folder structure
2. Extracts keyframes (important moments)
3. Gets object poses from segmentation masks
4. Extracts robot actions from `low_dim_obs.pkl`
5. Discretizes actions into bins
6. Formats as text: `{objects, instruction} >> [actions]`

**Output:** Creates ICL prompt files like:
```
{'cube': [45, 52, 48], 'target': [67, 43, 55], 'pick up the cube'} >> [[45, 52, 48, 0, 36, 0, 1], [50, 55, 45, 0, 38, 0, 0], ...]
```

### Step 3: Run Inference
```bash
python main.py --task=pick_and_lift
```

**What it does:**
1. Captures new scene with all 5 cameras
2. Segments objects and gets their poses
3. Forms a query with ICL examples
4. LLM predicts actions
5. Executes on robot/simulator

---

## 🆕 Your Approach: HuggingFace Dataset

### What's Different?

Instead of simulator demonstrations, you have **real robot demonstrations** in HuggingFace format:

```python
dataset = load_dataset("aadarshram/pick_place_tape")

# Each row has:
{
  'episode_index': 0,
  'timestamp': 0.033,
  'observation.state': [j1, j2, j3, j4, j5, gripper],  # Joint angles
  'action': [j1, j2, j3, j4, j5, gripper],  # Next joint angles
  # No RGB images! Just joint data
}
```

### Your Code Does This:

```python
# 1. Load episode from HuggingFace
frames = loader.load_episode(episode_idx=0)

# 2. Extract keyframes
keyframes = loader.extract_keyframes(frames)

# 3. Forward kinematics: joints → end-effector pose
for frame in keyframes:
    position, quaternion = loader.joint_positions_to_ee_pose(frame.joint_positions)

# 4. Discretize poses into RoboPrompt bins
actions = loader.convert_to_roboprompt_format(frames, keyframes)

# 5. Format as ICL demonstration
icl_demo = loader.format_as_icl_demonstration(
    actions=actions,
    object_poses=object_poses,  # You need to provide these!
    instruction="pick up the tape"
)
```

---

## 🔑 Key Differences Summary

| Aspect | Original RoboPrompt | Your HuggingFace Approach |
|--------|-------------------|---------------------------|
| **Data Source** | RLBench/CoppeliaSim simulator | Real robot recordings |
| **Camera Data** | 5 cameras × (RGB + Depth + Mask) | None (joint data only) |
| **Object Poses** | From segmentation masks | **You must provide** |
| **Robot Type** | Any RLBench-supported robot | SO100 (dataset) → SO101 (yours) |
| **Folder Structure** | Complex multi-camera folders | Single HuggingFace dataset |
| **Processing Script** | `form_icl_demonstrations.py` | Your `hf_dataset_loader.py` |
| **Advantages** | Perfect perception, multi-view | Real-world data, no sim2real gap |
| **Disadvantages** | Sim-to-real gap | No camera data, need perception |

---

## ⚠️ Critical Missing Piece: Object Poses

### The Problem

RoboPrompt needs **object poses** to form demonstrations:
```python
{'tape': [45, 52, 48], 'target': [67, 43, 55], 'pick tape'} >> [actions]
         ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
         Where are the objects in the scene?
```

**Original RoboPrompt:** Gets these from segmentation masks (all 5 cameras)

**Your approach:** Dataset only has joint angles, **no object poses!**

### Solution: You Must Provide Object Poses

In your code, you're **hardcoding** object poses:

```python
# In pipeline_with_hf_dataset.py
object_poses = {
    'tape': np.array([0.3, 0.0, 0.7, 0, 0, 0]),  # ← HARDCODED!
    'target_zone': np.array([0.5, 0.2, 0.7, 0, 0, 0]),
}
```

This works for creating ICL demonstrations from the dataset, but **won't work for inference** on new scenes because object positions will be different!

---

## 🎥 Camera Situation: 5 vs 1-2 Cameras

### Original RoboPrompt Expects:
- ✅ Front camera
- ✅ Overhead camera  
- ✅ Left shoulder camera
- ✅ Right shoulder camera
- ✅ Wrist camera

### Your Setup:
- ✅ Front camera (you have this)
- ❓ Maybe one more camera?
- ❌ No overhead, shoulders, wrist

### Does This Break RoboPrompt?

**For creating ICL demos:** ❌ No, you don't need cameras at all! Your dataset only has joint data, and that's enough to create demonstrations.

**For inference:** ⚠️ **Yes**, but you can adapt:
1. RoboPrompt needs object poses for new scenes
2. Original: Gets from 5-camera segmentation
3. Your adaptation: Get from your 1-2 cameras

---

## 🔧 Adapting for Your SO101 Setup

### What You Have:
- ✅ HuggingFace dataset (50 episodes, SO100 robot)
- ✅ SO101 robot (similar to SO100)
- ✅ 1-2 cameras
- ✅ Code to convert dataset → ICL demos

### What You Need to Add:

#### 1. Perception System (For Inference)

You need to detect objects in your camera view:

```python
# perception.py
import cv2
import numpy as np

def detect_objects(rgb_image):
    """
    Detect objects in camera image.
    
    Options:
    1. ArUco markers (simplest)
    2. YOLOv8 + Depth camera
    3. SAM (Segment Anything Model)
    4. Template matching
    """
    
    # Example with ArUco markers
    markers = detect_aruco_markers(rgb_image)
    
    object_poses = {}
    for marker_id, corners in markers.items():
        # Estimate 3D pose from marker
        position = estimate_marker_position(corners, camera_params)
        object_poses[f'object_{marker_id}'] = position
    
    return object_poses
```

#### 2. Camera Calibration

You need to know your camera's position relative to robot:

```python
# Camera calibration
camera_to_robot_transform = np.array([
    [1, 0, 0, 0.4],   # Camera is 0.4m in front
    [0, 1, 0, 0.0],   # Centered
    [0, 0, 1, 0.6],   # 0.6m above base
    [0, 0, 0, 1]
])
```

#### 3. Update Your Pipeline

Modify `integrated_roboprompt_agent.py`:

```python
def get_observation(self, cameras=['front']):
    """Capture current scene and detect objects."""
    
    # 1. Capture image from your camera(s)
    rgb_image = self.capture_camera('front')
    
    # 2. Detect objects
    object_poses = self.perception.detect_objects(rgb_image)
    
    # 3. Return in RoboPrompt format
    return {
        'object_poses': object_poses,
        'rgb': rgb_image,  # For visualization
    }
```

---

## 🎬 Complete Workflow Comparison

### Original RoboPrompt (Simulator)

```
1. Generate demos in RLBench
   ↓ (form_icl_demonstrations.py)
2. Extract keyframes + object poses from masks
   ↓
3. Discretize actions
   ↓
4. Form ICL demos
   ↓
5. [INFERENCE TIME]
   ↓
6. Capture 5-camera view of new scene
   ↓
7. Segment objects → get poses
   ↓
8. Query LLM with ICL examples
   ↓
9. LLM predicts actions
   ↓
10. Execute in simulator/robot
```

### Your Approach (HuggingFace → SO101)

```
1. Load demos from HuggingFace ✅ (DONE)
   ↓ (hf_dataset_loader.py)
2. Extract keyframes from joint trajectories ✅ (DONE)
   ↓
3. Forward kinematics: joints → poses ✅ (DONE)
   ↓
4. Discretize poses ✅ (DONE)
   ↓
5. Form ICL demos with HARDCODED object poses ✅ (DONE)
   ↓
6. [INFERENCE TIME]
   ↓
7. Capture 1-2 camera views ⚠️ (TODO)
   ↓
8. Detect objects → get poses ⚠️ (TODO)
   ↓
9. Query LLM with ICL examples ⚠️ (PARTIAL)
   ↓
10. LLM predicts actions ⚠️ (PARTIAL)
   ↓
11. Un-discretize actions ✅ (DONE)
   ↓
12. Inverse kinematics: poses → joints ✅ (DONE)
   ↓
13. Execute on SO101 ✅ (DONE)
```

---

## 📋 What Your Current Code Does

### ✅ Working (Demo Creation)

1. **Load HuggingFace dataset** - Works perfectly
2. **Extract keyframes** - Identifies important waypoints
3. **Forward kinematics** - Converts joints → end-effector poses
4. **Discretization** - Converts continuous → bins
5. **ICL formatting** - Creates text demonstrations
6. **Inverse kinematics** - Converts poses → joints for SO101
7. **Execution** - Can command SO101 arm

### ⚠️ Not Implemented (Inference)

1. **Camera capture** - Need to implement for your cameras
2. **Object detection** - Need perception system
3. **RoboPrompt LLM call** - Need to integrate with OpenAI/LLM
4. **Dynamic object poses** - Currently hardcoded

---

## 🚀 Your Next Steps

### Phase 1: Verify Demo Creation ✅
```bash
# This should already work:
python pipeline_with_hf_dataset.py --mode process --episodes 5
python pipeline_with_hf_dataset.py --mode test
python pipeline_with_hf_dataset.py --mode visualize
python pipeline_with_hf_dataset.py --mode execute --sim
```

### Phase 2: Add Perception System 🎯 (PRIORITY)

**Option A: ArUco Markers (Easiest)**
```python
# Install
pip install opencv-contrib-python

# Attach ArUco markers to objects
# Detect and estimate 3D pose from camera
```

**Option B: YOLO + Depth Camera**
```python
pip install ultralytics realsense2

# Use YOLO to detect objects
# Use depth camera to get 3D positions
```

**Option C: Segment Anything Model**
```python
pip install segment-anything

# Use SAM to segment objects
# Estimate 3D position from depth/stereo
```

### Phase 3: Integrate LLM Inference

```python
# In integrated_roboprompt_agent.py

def predict_actions(self, observation, icl_demos):
    """Use LLM to predict actions."""
    
    # 1. Format query
    prompt = self.format_prompt(observation, icl_demos)
    
    # 2. Call LLM (OpenAI/Claude/etc)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 3. Parse predicted actions
    actions = self.parse_llm_response(response)
    
    return actions
```

### Phase 4: Test End-to-End

```python
# Full pipeline with real perception
python pipeline_with_hf_dataset.py \
    --mode inference \
    --use-camera \
    --camera-index 0
```

---

## 🔍 Answering Your Specific Questions

### Q1: "Why did CoppeliaSim create those folders?"

**A:** CoppeliaSim (via RLBench) creates those folders because RoboPrompt was originally designed to use **multi-view RGB-D-Mask data** for:
- Segmenting objects from multiple angles
- Getting accurate 3D object poses
- Providing rich visual context to the LLM

### Q2: "My arm is different (SO101 vs SO100)"

**A:** Minor issue! SO100 and SO101 are very similar:
- Both are 5-DOF + gripper
- Same kinematic structure
- Your IK solver in `roboprompt_lerobot_bridge.py` should work for both
- You may need to adjust link lengths slightly

### Q3: "I only have 1-2 cameras"

**A:** This is fine! You can:
- Use fewer cameras if they give good object views
- Focus on quality object detection rather than quantity of views
- RoboPrompt's LLM is robust to different setups
- You might get slightly lower accuracy than 5-camera setup

### Q4: "How do I create similar demo files?"

**A:** You don't need the same folder structure! Your approach is actually **better** because:
- ✅ Real robot data (no sim2real gap)
- ✅ Simpler pipeline (no multi-camera recording)
- ✅ Works with existing HuggingFace datasets

You just need to add perception for **inference time**.

---

## 💡 Recommended Approach

### For Demo Creation (What You Have)
Keep using your HuggingFace approach! It's working well:
- ✅ Loads real robot demonstrations
- ✅ Converts to RoboPrompt format
- ✅ Creates ICL demos
- ✅ Can execute on SO101

### For Inference (What You Need)

**Minimum Viable Perception:**
1. **Use ArUco markers** on objects (simplest)
2. **Single RGB-D camera** (RealSense D435 or similar)
3. **Detect markers → estimate 3D poses**
4. **Pass to RoboPrompt → predict actions**

**Code Structure:**
```python
# main_inference.py

# 1. Load ICL demos (your existing code works)
icl_demos = load_icl_demos('./roboprompt_data/icl_demos.json')

# 2. Capture scene (NEW - add this)
camera = Camera(device_id=0)
rgb, depth = camera.capture()

# 3. Detect objects (NEW - add this)
perception = PerceptionSystem(camera_params)
object_poses = perception.detect_objects(rgb, depth)

# 4. Query RoboPrompt (NEW - add this)
agent = RoboPromptAgent()
predicted_actions = agent.predict(
    observation={'objects': object_poses, 'instruction': 'pick tape'},
    icl_demos=icl_demos
)

# 5. Execute (your existing code works)
bridge = RoboPromptLeRobotBridge()
bridge.execute_roboprompt_actions(predicted_actions, speed=0.3)
```

---

## 📚 Summary

### Your Confusion Was Valid!

You saw two different approaches:
1. **Original RoboPrompt:** Complex folder structure, multi-camera, simulator
2. **Your approach:** HuggingFace dataset, real robot, no cameras

### Key Insight

You're doing a **clever adaptation** of RoboPrompt:
- ✅ Skip the simulator recordings
- ✅ Use real robot demonstrations
- ✅ Convert from joint space to RoboPrompt format
- ⚠️ Need to add perception for inference

### What Works Now

Your demo creation pipeline is **complete and working**! You successfully:
- Load HF demos → Extract keyframes → Discretize → Create ICL format

### What to Add Next

Focus on **perception system** for inference:
1. Camera capture
2. Object detection (ArUco markers = easiest start)
3. LLM integration
4. End-to-end testing

---

## 🎉 You're 70% Done!

Your code already handles the hard parts:
- ✅ Dataset loading and processing
- ✅ Forward/inverse kinematics
- ✅ Action discretization/continuous conversion
- ✅ SO101 execution

You just need perception and LLM integration for the final 30%!

**Focus on:** Getting object poses from your 1-2 cameras, and you'll have a complete system.

---

## 📞 Next Steps

1. **Test what you have:**
   ```bash
   python pipeline_with_hf_dataset.py --mode process --episodes 5
   python pipeline_with_hf_dataset.py --mode execute --sim
   ```

2. **Choose perception method:**
   - ArUco markers (recommended for first version)
   - YOLO + depth camera (more general)
   
3. **Implement object detection:**
   - Create `perception.py`
   - Test with your camera setup
   
4. **Integrate with LLM:**
   - Set up OpenAI API or local LLM
   - Implement `predict_actions()` method
   
5. **Test end-to-end:**
   - Place objects in view
   - Run inference
   - Execute on SO101

Good luck! 🚀
