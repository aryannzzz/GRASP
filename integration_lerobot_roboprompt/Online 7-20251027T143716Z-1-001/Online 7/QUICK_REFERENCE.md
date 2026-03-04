# RoboPrompt SO101 - Quick Reference

## 🚀 Quick Commands

### Setup
```bash
./setup.sh                                  # Run setup
export OPENAI_API_KEY=your_key             # Set API key
```

### Run Pipeline
```bash
# Full pipeline
python src/main_pipeline.py --mode full --num-episodes 50

# Offline only
python src/main_pipeline.py --mode offline --num-episodes 50

# Inference only
python src/main_pipeline.py --mode inference \
    --demo-path output/demonstrations/processed_demonstrations.pkl \
    --test-image test.jpg
```

## 📂 Key Files

| File | Purpose |
|------|---------|
| `config/config.yaml` | Main configuration |
| `src/main_pipeline.py` | Main orchestration |
| `src/forward_kinematics.py` | FK solver |
| `src/action_discretizer.py` | Discretization |
| `src/icl_constructor.py` | ICL prompts |
| `src/llm_interface.py` | GPT-4 interface |

## ⚙️ Key Configurations

```yaml
# config/config.yaml

keyframes:
  velocity_threshold: 0.01  # Lower = more keyframes

discretization:
  translation_bins: 100     # Position bins
  rotation_bins: 72         # Rotation bins (5° each)

icl:
  num_demonstrations: 5     # ICL shots
  llm_model: "gpt-4-turbo" # LLM model
  temperature: 0.0          # Deterministic

ik:
  method: "nearest_neighbor"  # IK method
```

## 🔧 Common Tasks

### Test Individual Components
```bash
python src/forward_kinematics.py
python src/keyframe_extractor.py
python src/action_discretizer.py
```

### Load Demonstrations
```python
from lerobot_to_roboprompt import LeRobotDatasetConverter

converter = LeRobotDatasetConverter(config, ...)
demos = converter.load_demonstrations("output/demonstrations/processed_demonstrations.pkl")
```

### Run Inference
```python
from main_pipeline import RoboPromptPipeline

pipeline = RoboPromptPipeline()
pipeline.load_demonstrations("output/demonstrations/processed_demonstrations.pkl")
result = pipeline.run_inference(test_image, task_instruction)
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Dataset not found" | Check HuggingFace dataset exists, install `lerobot` |
| "OpenAI API error" | Set `OPENAI_API_KEY` environment variable |
| "URDF not found" | Copy SO101 URDF to `config/so100.urdf` |
| "FoundationPose error" | Integrate FoundationPose (see guide) |
| "Robot not responding" | Integrate SO101 control (see guide) |

## 📊 Expected Results

### Keyframe Extraction
- Original: ~200-300 frames per episode
- Keyframes: ~5-15 frames per episode
- Compression: 93-97%

### Discretization
- Position bins: 100 per dimension
- Rotation bins: 72 per dimension (5° each)
- Max error: ~bin_size/2

### ICL Performance
- Works best with 5-10 demonstrations
- Temperature 0.0 for deterministic output
- GPT-4 Turbo recommended

## 📚 Key Concepts

### Forward Kinematics (FK)
Joint positions → End-effector pose
```python
joints [6] → FK → pose [x,y,z,r,p,y]
```

### Inverse Kinematics (IK)
End-effector pose → Joint positions
```python
pose [x,y,z,r,p,y] → IK → joints [6]
```

### Discretization
Continuous → Discrete bins
```python
x=0.15m → bin 45 (out of 100)
roll=0.5rad → bin 35 (out of 72)
```

### In-Context Learning (ICL)
Show examples, get prediction
```
Demo1: input → output
Demo2: input → output
Test: input → ???
```

## 🎯 Pipeline Flow

```
Dataset → Keyframes → FK → Discretize → Demos
                                          ↓
Test → Objects → ICL → GPT-4 → Actions → IK → Joints → Robot
```

## 📞 Quick Help

- README: General overview
- IMPLEMENTATION_GUIDE: Detailed guide
- GitHub Issues: Bug reports
- RoboPrompt Paper: Original research

---

**Pro Tip**: Start with simulation mode (`use_simulation: true` in config) before trying real robot!
