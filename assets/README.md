# Assets

This directory contains supplementary assets for the GRASP project repository.

## Contents

### Evaluation Videos

Pre-recorded evaluation videos showing robot policy behavior:

| File | Description |
|---|---|
| `act_standard_eval.mp4` | Standard ACT evaluated on shelf-place |
| `act_modified_eval.mp4` | Modified ACT evaluated on shelf-place |
| `classical_pipeline_sim.mp4` | Classical pipeline in simulation |
| `gradcam_demo.mp4` | GradCAM saliency visualization demo |

> Videos are not tracked in git due to size. They can be generated using:
> ```bash
> python experiments/act_modifications/code/record_videos.py \
>     --model modified --checkpoint <path> --num_videos 5
> ```

### Calibration Images

Camera calibration checkerboard images used for the real robot setup are stored in:
```
experiments/classical_pipeline/camera_calibration/IMG_68*.jpeg
```

### ArUco Markers

Printable ArUco marker PDFs for real-robot object tracking (RoboPrompt experiment):
- Generated via `experiments/roboprompt/code/generate_aruco_markers.py`

---

## Adding Assets

When adding new assets:
1. Keep files under 10MB when possible
2. Use descriptive filenames with underscores
3. Update this README with a description
4. For large files (videos, datasets), use Git LFS:
   ```bash
   git lfs track "assets/*.mp4"
   git lfs track "assets/*.hdf5"
   ```
