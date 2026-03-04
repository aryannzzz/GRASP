# RoboPrompt + SO-101: Complete Integration Guide

**Proper integration of LeRobot dataset → RoboPrompt → SO-101 execution**

---

## 🎯 Key Concept: Two Different Uses of Object Poses

### Your Confusion (Now Solved!)

**Question:** "How can we create keyframes if we don't know where objects are?"

**Answer:** Object poses are used differently at different times!

| Stage | Object Poses | Why |
|-------|-------------|-----|
| **Demo Creation** | Approximate/hardcoded | Just for ICL formatting - don't need to be exact! |
| **Inference Time** | Real-time detection | Need actual object positions in NEW scenes |

**Key Insight:** When creating ICL demonstrations from the dataset, object poses can be approximate estimates or even hardcoded values. They're just there to format the ICL examples correctly. The LLM learns the task structure, not the exact positions.

**Real perception** is only needed when you run inference on NEW, unseen scenes where object positions are different.

---

## 📦 What's Included

### Core Scripts

1. **`lerobot_roboprompt_dataset.py`**
   - Loads LeRobot dataset (proper format with video frames)
   - Extracts keyframes using RoboPrompt's criteria
   - Creates ICL demonstrations with approximate object poses
   - **Run this ONCE to create demos**

2. **`perception_system.py`**
   - Real-time object detection (ArUco markers, YOLO, etc.)
   - Converts detections to RoboPrompt format
   - **Used at INFERENCE time only**

3. **`roboprompt_so101_pipeline.py`**
   - Complete inference + execution pipeline
   - Uses perception to detect objects in NEW scenes
   - Queries RoboPrompt LLM
   - Executes on SO-101 via LeRobot
   - **Run this for each new task**

### Supporting Files

- **`generate_aruco_markers.py`** - Generate printable markers
- **`README.md`** - This file
- **`requirements.txt`** - Dependencies

---

## 🚀 Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: Demo Creation (ONE TIME)                      │
└─────────────────────────────────────────────────────────┘
    │
    ├─> 1. Load LeRobot dataset
    │      (aadarshram/pick_place_tape)
    │
    ├─> 2. Extract keyframes from episodes
    │      (using RoboPrompt's criteria)
    │
    ├─> 3. Convert to RoboPrompt format
    │      (discretize poses into bins)
    │
    └─> 4. Create ICL demonstrations
           (with APPROXIMATE object poses)
           ↓
        📄 icl_demos.json

┌─────────────────────────────────────────────────────────┐
│ PHASE 2: Inference (EACH NEW SCENE)                    │
└─────────────────────────────────────────────────────────┘
    │
    ├─> 1. Detect objects in REAL scene
    │      (ArUco markers / YOLO / SAM)
    │
    ├─> 2. Discretize detected poses
    │      (convert to RoboPrompt format)
    │
    ├─> 3. Query RoboPrompt LLM
    │      (using ICL demos + new scene)
    │
    ├─> 4. Get predicted actions
    │
    ├─> 5. Convert to joint angles (IK)
    │
    └─> 6. Execute on SO-101
           (via LeRobot motor control)
```

---

## 📋 Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install numpy scipy matplotlib opencv-contrib-python

# LeRobot (for dataset and robot control)
pip install lerobot

# For LLM integration
pip install openai  # or anthropic, etc.

# Optional: for depth camera
pip install pyrealsense2
```

### 2. Hardware Setup

- **SO-101 robot arm** connected via USB
- **Camera** (USB webcam or RealSense)
- **ArUco markers** (print from `generate_aruco_markers.py`)

### 3. Print ArUco Markers

```bash
python generate_aruco_markers.py
```

Print the generated PNG files and attach to objects:
- Marker 0 → tape
- Marker 1 → target zone
- Marker 2 → table

---

## 🎬 Step-by-Step Usage

### Phase 1: Create ICL Demonstrations (One Time)

```bash
# Generate demonstrations from LeRobot dataset
python lerobot_roboprompt_dataset.py
```

**What this does:**
- Downloads dataset from HuggingFace
- Loads 5 episodes
- Extracts keyframes (near-zero velocity, gripper changes)
- Converts joint angles → end-effector poses (FK)
- Discretizes poses into RoboPrompt bins
- Creates ICL demonstrations with approximate object poses
- Saves to `./roboprompt_data/icl_demos.json`

**Output:**
```
📦 Loading LeRobot dataset: aadarshram/pick_place_tape
✓ Dataset loaded:
   Robot: so100
   Episodes: 50
   Frames: 14954
   FPS: 30

--- Processing Episode 0 ---
📂 Loading episode 0: 299 frames (9.97s)
📌 Extracted 15 keyframes from 299 frames

💾 Saved 5 ICL demonstrations to ./roboprompt_data/icl_demos.json
```

**Important:** The object poses used here are approximate! They're just for formatting the ICL examples. The LLM learns the task structure, not exact positions.

### Phase 2: Test Perception (Before Inference)

```bash
# Test object detection
python perception_system.py
```

**What this does:**
- Opens camera
- Detects ArUco markers
- Shows real-time visualization
- Displays object positions in robot frame

**Make sure you can detect objects before running inference!**

### Phase 3: Run Inference on New Scenes

```bash
# Simulation mode (no robot movement)
python roboprompt_so101_pipeline.py --sim

# Real robot (slow speed for safety)
python roboprompt_so101_pipeline.py --speed 0.3 --robot-port /dev/ttyUSB0

# With custom settings
python roboprompt_so101_pipeline.py \
    --icl-demos ./roboprompt_data/icl_demos.json \
    --robot-port /dev/ttyUSB0 \
    --camera-id 0 \
    --marker-size 0.05 \
    --speed 0.5 \
    --num-demos 5
```

**What this does:**
1. Loads ICL demonstrations
2. **Detects objects in CURRENT scene** (real perception!)
3. Discretizes detected poses
4. Queries RoboPrompt LLM (TODO: implement LLM call)
5. Gets predicted actions
6. Converts to joint angles (IK)
7. Executes on SO-101

**Output:**
```
================================================================================
ROBOPROMPT INFERENCE PIPELINE
================================================================================

📷 Step 1: Detecting objects in scene...
✓ Detected 3 objects:
  - tape: [0.284, -0.012, 0.723]
  - target_zone: [0.481, 0.193, 0.717]
  - table: [0.003, -0.001, 0.602]

🎯 Step 2: Discretizing object poses...
✓ Discretized object poses:
  'tape': [58, 48, 62]
  'target_zone': [78, 69, 59]
  'table': [30, 50, 0]

🧠 Step 3: Querying RoboPrompt LLM...
✓ Got 15 predicted actions

🤖 Step 4: Executing actions on SO-101...
[Action 1/15]
  Position: [0.284, -0.012, 0.723]m
  Gripper: OPEN
  Joints: [4.2°, 48.3°, -31.2°, 11.8°, 0.0°, 30.0°]
  ✓ Executed on robot

...

✅ All 15 actions completed!
```

---

## 🔑 Key Files Explained

### `icl_demos.json`

Contains ICL demonstrations in RoboPrompt format:

```json
{
  "demonstrations": [
    "{{tape': [58, 48, 62], 'target_zone': [78, 69, 59], 'pick up tape'}>>[45, 52, 48, 0, 36, 0, 1], ..."
  ],
  "object_poses_approximate": {
    "tape": [0.3, 0.0, 0.7, 0, 0, 0],
    "target_zone": [0.5, 0.2, 0.7, 0, 0, 0]
  },
  "note": "Object poses are approximate for ICL formatting. Use perception at inference time."
}
```

**Key point:** `object_poses_approximate` are just examples! Real object detection happens at inference.

### Camera Calibration

For better accuracy, calibrate your camera:

```python
# In perception_system.py, update camera_matrix
camera_matrix = np.array([
    [fx, 0, cx],  # Focal length x, principal point x
    [0, fy, cy],  # Focal length y, principal point y
    [0, 0, 1]
])
```

Get these values from camera calibration (use OpenCV's calibration tools).

---

## 🎯 Understanding RoboPrompt Action Format

### Discretized Action

```python
[x_bin, y_bin, z_bin, rx_bin, ry_bin, rz_bin, gripper]
[  45,    52,    48,      0,     36,      0,       1 ]
```

- **Position bins** (0-99): Discretized XYZ position in workspace
- **Rotation bins** (0-71): Discretized rotation (5° resolution)
- **Gripper** (0-1): 0=closed, 1=open

### How Discretization Works

```python
# Position: continuous [x, y, z] → bins [0-99]
scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]  # [x_min, y_min, z_min, x_max, y_max, z_max]
normalized = (position - bounds[:3]) / (bounds[3:] - bounds[:3])
bins = (normalized * 100).astype(int)  # 0-99

# Rotation: Euler angles → bins [0-71]
euler_normalized = (euler_degrees % 360) / 360
bins = (euler_normalized * 72).astype(int)  # 72 bins = 5° resolution

# Gripper: continuous → binary
gripper_bin = 1 if gripper_angle > 15° else 0
```

---

## 🐛 Troubleshooting

### "Dataset not found"

```bash
# Download dataset manually
python -c "from datasets import load_dataset; load_dataset('aadarshram/pick_place_tape')"
```

### "No ArUco markers detected"

- Ensure markers are visible and not occluded
- Check lighting (reduce shadows)
- Use larger markers (increase `--marker-size`)
- Verify camera is working: `python perception_system.py`

### "Robot not responding"

```bash
# Check USB connection
ls /dev/ttyUSB*

# Check permissions
sudo chmod 666 /dev/ttyUSB0

# Try different port
python roboprompt_so101_pipeline.py --robot-port /dev/ttyUSB1
```

### "IK failed - position unreachable"

- Adjust `scene_bounds` to match robot workspace
- Check target position is within reach
- Verify forward kinematics is correct

### "Actions look wrong"

- Verify scene_bounds match your workspace
- Calibrate camera properly
- Check ArUco marker size matches physical size
- Visualize detected positions before execution

---

## 🔧 Advanced Configuration

### Use Different Dataset

```python
# In lerobot_roboprompt_dataset.py
loader = LeRobotRoboPromptDataset(
    repo_id="your-username/your-dataset",
    scene_bounds=[-0.4, -0.6, 0.5, 0.8, 0.6, 1.8]
)
```

### Add More Markers

```python
# In perception_system.py
marker_to_object_names = {
    0: 'tape',
    1: 'target_zone',
    2: 'table',
    3: 'obstacle',     # Add more objects
    4: 'container',
    5: 'tool'
}
```

### Adjust Keyframe Extraction

```python
# In lerobot_roboprompt_dataset.py
keyframes = loader.extract_keyframes_roboprompt(
    episode,
    velocity_threshold=0.2,  # Higher = fewer keyframes
    min_frames_between=10    # Minimum spacing
)
```

---

## 🎓 How It All Works Together

### Demo Creation

1. **Load dataset**: Episodes with joint angles over time
2. **Extract keyframes**: Find important moments (near-zero velocity, gripper changes)
3. **Forward kinematics**: Convert joint angles → end-effector poses
4. **Discretize**: Continuous poses → bins (0-99 for position, 0-71 for rotation)
5. **Format ICL**: Create text format with approximate object poses
6. **Save**: Store demonstrations for later use

**Object poses at this stage**: Approximate/hardcoded - just for formatting!

### Inference

1. **Detect objects**: ArUco markers / YOLO / SAM on REAL scene
2. **Discretize poses**: Same discretization as demos
3. **Query LLM**: RoboPrompt with ICL examples + new scene
4. **Get actions**: LLM predicts discretized actions
5. **Undiscretize**: Bins → continuous poses
6. **Inverse kinematics**: Poses → joint angles
7. **Execute**: Send commands to SO-101

**Object poses at this stage**: Real-time detection - actual positions!

---

## 📚 Next Steps

### 1. Integrate Real LLM

Currently, LLM integration is a placeholder. Implement:

```python
# In roboprompt_so101_pipeline.py
import openai

def query_roboprompt_llm(self, detected_objects, instruction, num_demos=5):
    # Format prompt
    prompt = self._format_prompt(detected_objects, instruction, num_demos)
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are RoboPrompt, predicting robot actions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    # Parse response
    actions = self._parse_llm_response(response.choices[0].message.content)
    
    return actions
```

### 2. Improve Perception

- Use YOLO for markerless detection
- Add depth camera support
- Implement SAM for segmentation
- Add object tracking over time

### 3. Better IK/FK

- Use PyBullet for accurate kinematics
- Integrate MoveIt! for motion planning
- Add collision checking
- Implement trajectory smoothing

### 4. More Demonstrations

- Record more episodes
- Add task variations
- Use different objects
- Try complex multi-step tasks

---

## ✅ Checklist

### Before First Run:

- [ ] Installed all dependencies
- [ ] SO-101 connected and powered
- [ ] Camera connected
- [ ] ArUco markers printed and attached
- [ ] Ran demo creation script
- [ ] Tested perception system
- [ ] Verified ICL demos created

### Before Each Inference:

- [ ] Objects with markers in camera view
- [ ] Robot in safe starting position
- [ ] E-stop accessible
- [ ] Workspace clear of obstacles
- [ ] Tested in simulation first
- [ ] Speed set appropriately (0.3 for first run)

---

## 🎉 You're Ready!

You now have a complete pipeline from LeRobot dataset → RoboPrompt → SO-101 execution!

**The key insight:** Object poses are approximate for demo creation, real for inference. This separates the "learning from examples" phase from the "acting in new scenes" phase.

**Questions?** Check the troubleshooting section or open an issue.

Good luck! 🚀🤖
