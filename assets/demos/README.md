# Demo Assets

This directory contains or describes evaluation demos and visualization outputs for the GRASP project.

## Video Demos

Evaluation videos are generated from trained models using the recording scripts.

| File | Experiment | Description | How to Generate |
|:---|:---|:---|:---|
| `act_standard_eval.mp4` | ACT Modifications | Standard ACT on shelf-place task | `python experiments/act_modifications/code/record_videos.py --model standard` |
| `act_modified_eval.mp4` | ACT Modifications | Modified ACT (image CVAE) on shelf-place | `python experiments/act_modifications/code/record_videos.py --model modified` |
| `classical_pipeline_sim.mp4` | Classical Pipeline | Open-vocab pick-place in simulation | Run the pipeline notebook |
| `gradcam_demo.mp4` | GradCAM VLA | Saliency map evolution during policy execution | `python experiments/gradcam_vla/code/scripts/vla_gradcam_demo_v2.py` |
| `multitask_act_eval.mp4` | Multi-task ACT | Multi-task policy on 4 MetaWorld tasks | `python experiments/multitask_act/code/metaworld_act_complete.py --record` |

> Videos are not tracked in git due to size. Generate them locally using the commands above.

## GradCAM Saliency Images

Generated saliency images from the VLA-GradCAM validation are in `figures/`:

| File | Description |
|:---|:---|
| `robot_scene_comparison.png` | Side-by-side saliency maps for 4 instructions in robot scene |
| `per_action_pick_up_the_red_cup.png` | Per-action-dimension saliency — "pick up the red cup" |
| `per_action_push_the_blue_block.png` | Per-action-dimension saliency — "push the blue block" |
| `per_action_grasp_the_green_ball.png` | Per-action-dimension saliency — "grasp the green ball" |
| `office_scene_overview.png` | Multi-instruction comparison — office scene |
| `action_comparison.png` | Side-by-side action prediction comparison |

## Generating GIFs

To create animated GIFs from evaluation videos:

```bash
# Install ffmpeg
sudo apt install ffmpeg

# Convert MP4 to GIF
ffmpeg -i act_modified_eval.mp4 -vf "fps=10,scale=480:-1:flags=lanczos" \
    -c:v gif assets/demos/act_modified_eval.gif

# For GradCAM demo
ffmpeg -i gradcam_demo.mp4 -vf "fps=8,scale=640:-1:flags=lanczos" \
    -c:v gif assets/demos/gradcam_demo.gif
```

## Adding New Assets

When adding new demo assets:
1. Place evaluation videos (`.mp4`) in this folder
2. Size limit: keep individual files under 50MB
3. For large files, use Git LFS: `git lfs track "assets/demos/*.mp4"`
4. Update this README with a description of each new file
