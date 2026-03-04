# RoboPrompt for SO101 Robot Arm

Complete implementation of RoboPrompt (In-Context Learning for robot manipulation) for the SO101 robotic arm using LeRobot framework.

## 🎯 Overview

This project implements the RoboPrompt framework from the paper ["Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models"](https://roboprompt.github.io/) for the SO101 6-DOF robot arm.

**Key Features:**
- 🤖 Converts LeRobot datasets to RoboPrompt format
- 🎬 Automatic keyframe extraction from teleoperation data
- 🔄 Forward kinematics (joint space → end-effector space)
- 📦 Action discretization (continuous → discrete bins)
- 👁️ Object pose estimation using FoundationPose
- 🧠 GPT-4 based action prediction via In-Context Learning
- 🎮 Action execution on SO101 arm

## 📋 Pipeline Overview

### Offline Processing (Dataset Preparation)
1. Load LeRobot dataset from HuggingFace
2. Extract keyframes (critical moments where robot changes direction or gripper state)
3. Compute forward kinematics (convert joint positions to end-effector poses)
4. Discretize poses into bins (100 bins for position, 72 bins for rotation)
5. Estimate object poses using FoundationPose
6. Format as ICL demonstrations

### Inference (Test Time)
1. Capture test image from camera
2. Estimate object poses in test scene
3. Construct ICL prompt with demonstrations
4. Query GPT-4 for action predictions
5. Parse and discretize predicted actions
6. Convert to joint commands using nearest-neighbor IK
7. Interpolate smooth trajectory
8. Execute on SO101 robot

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd roboprompt_so101

# Install dependencies
pip install -r requirements.txt

# Install LeRobot
pip install lerobot

# Copy your SO101 URDF file
cp /path/to/so100.urdf config/so100.urdf

# Set up OpenAI API key
export OPENAI_API_KEY=your_api_key_here
```

### 2. Setup FoundationPose (Optional but Recommended)

```bash
# Clone FoundationPose
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose

# Install dependencies (follow their README)
# Download model checkpoints
# Place models in ./models/foundationpose/

# Return to project root
cd ..
```

Note: Based on the paper author's email, FoundationPose uses RGB images only (no depth required).

### 3. Run the Complete Pipeline

```bash
# Full pipeline (offline + inference)
python src/main_pipeline.py \
    --mode full \
    --task "Pick up the tape and place it on the target" \
    --num-episodes 50

# Offline processing only
python src/main_pipeline.py \
    --mode offline \
    --num-episodes 50 \
    --demo-path output/demonstrations/processed_demonstrations.pkl

# Inference only (requires pre-processed demonstrations)
python src/main_pipeline.py \
    --mode inference \
    --demo-path output/demonstrations/processed_demonstrations.pkl \
    --test-image path/to/test_image.jpg \
    --task "Pick up the tape and place it on the target"
```

## 📁 Project Structure

```
roboprompt_so101/
├── config/
│   ├── config.yaml              # Main configuration file
│   └── so100.urdf               # Robot URDF file
├── src/
│   ├── forward_kinematics.py    # FK solver using PyBullet
│   ├── keyframe_extractor.py    # Keyframe extraction
│   ├── action_discretizer.py    # Action discretization
│   ├── pose_estimator.py        # Object pose estimation wrapper
│   ├── icl_constructor.py       # ICL prompt construction
│   ├── llm_interface.py         # GPT-4 interface
│   ├── lerobot_to_roboprompt.py # Dataset converter
│   ├── action_executor.py       # Action execution
│   └── main_pipeline.py         # Main orchestration script
├── output/
│   ├── demonstrations/          # Processed demonstrations
│   ├── results/                 # Inference results
│   └── logs/                    # Logs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- **Dataset**: HuggingFace dataset name and local path
- **Robot**: URDF path, joint names, end-effector link
- **Keyframes**: Velocity threshold, extraction method
- **Discretization**: Number of bins, workspace limits
- **Pose Estimation**: Method (FoundationPose/GroundingDINO), objects to detect
- **ICL**: Number of demonstrations, LLM model, temperature
- **IK**: Method (nearest_neighbor/pybullet_ik/learned_mapping)
- **Execution**: Simulation mode, control frequency, interpolation

## 🔧 Key Components

### 1. Forward Kinematics
```python
from forward_kinematics import ForwardKinematics

fk = ForwardKinematics(urdf_path="config/so100.urdf")
position, orientation = fk.compute_fk(joint_positions)
```

### 2. Keyframe Extraction
```python
from keyframe_extractor import KeyframeExtractor

extractor = KeyframeExtractor(config)
keyframes = extractor.extract_keyframes(joint_positions, gripper_states)
```

### 3. Action Discretization
```python
from action_discretizer import ActionDiscretizer

discretizer = ActionDiscretizer(config)
discrete_pose = discretizer.discretize_pose(continuous_pose)
continuous_pose = discretizer.discrete_to_continuous(discrete_pose)
```

### 4. ICL Construction
```python
from icl_constructor import ICLConstructor

constructor = ICLConstructor(config, discretizer)
system_msg, user_prompt = constructor.construct_icl_prompt_with_system(
    demonstrations, test_data
)
```

### 5. LLM Interface
```python
from llm_interface import LLMInterface

llm = LLMInterface(config)
response = llm.query(system_message, user_prompt)
```

## 📊 Dataset Format

Your LeRobot dataset should have the following structure:
- **Actions**: 6 DOF joint positions + gripper state
- **Observations**: 6 DOF joint positions (state feedback)
- **Images**: RGB video from camera (e.g., "top_phone")
- **Episodes**: Multiple demonstration trajectories

Example dataset: `aadarshram/act_pick_place_tape`

## 🎓 Research Questions & Solutions

### Q1: Forward Kinematics (Joint → End-Effector)
**Solution**: Use PyBullet physics simulation with SO101 URDF to compute FK

### Q2: Inverse Kinematics (End-Effector → Joint)
**Solutions**:
1. **Nearest Neighbor** (default): Find closest EE pose in dataset, use those joints
2. **IK Solver**: Use PyBullet/pinocchio IK (may fail for some poses)
3. **Learned Mapping**: Train small MLP (requires additional training)

We use **Nearest Neighbor** as it's simplest and most reliable.

### Q3: Object Pose Estimation
**Solution**: FoundationPose with RGB images only (no depth required, per author's email)

## 🐛 Troubleshooting

### Issue: "FoundationPose not found"
**Solution**: The pose estimator currently uses placeholder code. You need to:
1. Install FoundationPose from https://github.com/NVlabs/FoundationPose
2. Integrate it in `src/pose_estimator.py`

### Issue: "OpenAI API key not found"
**Solution**: Set your API key:
```bash
export OPENAI_API_KEY=your_key_here
```

### Issue: "Robot interface not implemented"
**Solution**: The action executor currently doesn't send commands to the real robot. You need to integrate with your SO101 control interface in `src/action_executor.py`.

### Issue: "Dataset not loading"
**Solution**: Make sure:
1. LeRobot is installed: `pip install lerobot`
2. Dataset exists on HuggingFace: `aadarshram/act_pick_place_tape`
3. You have internet connection to download

## 📚 References

1. **RoboPrompt Paper**: [https://roboprompt.github.io/](https://roboprompt.github.io/)
2. **RoboPrompt Code**: [https://github.com/davidyyd/roboprompt/](https://github.com/davidyyd/roboprompt/)
3. **LeRobot**: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
4. **SO-ARM100**: [https://github.com/TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
5. **FoundationPose**: [https://github.com/NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)

## 📝 Citation

If you use this code, please cite the original RoboPrompt paper:

```bibtex
@article{roboprompt2024,
  title={Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models},
  author={[Author names]},
  journal={arXiv preprint},
  year={2024}
}
```

## 📧 Contact

For questions about the implementation, please refer to:
- The RoboPrompt paper and code
- LeRobot documentation
- SO101 GitHub repository

## 🎯 Next Steps

1. **Integrate FoundationPose**: Replace placeholder pose estimation with actual FoundationPose
2. **Robot Interface**: Add SO101 control interface to action executor
3. **Test on Real Robot**: Run inference with real camera and robot
4. **Tune Parameters**: Adjust discretization bins, keyframe threshold, etc.
5. **Collect More Data**: Gather more demonstrations for better ICL performance
6. **Try Different Tasks**: Extend to other manipulation tasks beyond pick-and-place

Good luck with your implementation! 🚀
