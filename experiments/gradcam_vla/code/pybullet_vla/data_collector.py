"""
Randomized Demonstration Data Collector for VLA Training

Collects expert demonstrations with:
- Random object positions per episode (via layout seeds)
- Random camera jitter
- Random object scale variation
- Optional distractor objects
- Layout seed logging for reproducibility
- Train/val/held-out split generation with layout isolation

Split strategy:
- Layout seeds are partitioned into non-overlapping pools
- Held-out layouts contain object placements NEVER seen during training
- No layout leakage between splits
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from tqdm import tqdm

from .env import TableTopEnv, setup_env


class DemonstrationCollector:
    """Collects expert demonstrations with randomized layouts and split generation."""

    # Object display names for instructions
    OBJECT_NAME_MAP = {
        "duck_vhacd": "duck",
        "sphere_small": "sphere",
        "cube_small": "cube",
        "teddy_vhacd": "teddy bear",
        "jenga/jenga": "block",
    }

    def __init__(
        self,
        output_dir: str,
        image_size: int = 224,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.image_size = image_size

    def get_natural_name(self, object_name: str) -> str:
        return self.OBJECT_NAME_MAP.get(object_name, object_name.replace("_", " "))

    def generate_pick_instruction(self, object_name: str) -> str:
        natural_name = self.get_natural_name(object_name)
        templates = [
            f"pick up the {natural_name}",
            f"grasp the {natural_name}",
            f"grab the {natural_name}",
        ]
        return random.choice(templates)

    def generate_place_instruction(self, target: str) -> str:
        if "_" in target:
            loc_name = target.replace("_", " ")
            templates = [
                f"place it at the {loc_name}",
                f"move to the {loc_name}",
                f"put it in the {loc_name}",
            ]
        else:
            natural_name = self.get_natural_name(target)
            templates = [
                f"place it on the {natural_name}",
                f"put it on the {natural_name}",
            ]
        return random.choice(templates)

    def compute_expert_action(
        self,
        current_pos: List[float],
        target_pos: List[float],
        gripper_open: bool
    ) -> np.ndarray:
        delta = np.array(target_pos) - np.array(current_pos)
        max_step = 0.05
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_step:
            delta = delta / delta_norm * max_step

        action = np.zeros(7, dtype=np.float32)
        action[0] = np.clip(delta[0] / 0.05, -1, 1)
        action[1] = np.clip(delta[1] / 0.05, -1, 1)
        action[2] = np.clip(delta[2] / 0.05, -1, 1)
        action[6] = 1.0 if gripper_open else -1.0
        return action

    def _get_object_label(self, object_name: str) -> int:
        """Get integer label for object (for contrastive loss)."""
        all_objects = TableTopEnv.AVAILABLE_OBJECTS
        if object_name in all_objects:
            return all_objects.index(object_name)
        return 0

    def execute_pick_place_episode(
        self,
        env: TableTopEnv,
        pick_object: str,
        place_target: str,
        episode_id: int,
    ) -> List[Dict]:
        episode_demos = []

        pick_result = env.get_object_pose(pick_object)
        if pick_result is None:
            return []

        pick_pos, _ = pick_result

        if place_target in env.special_pts:
            place_pos = env.special_pts[place_target]
        else:
            place_result = env.get_object_pose(place_target)
            if place_result is None:
                return []
            place_pos, _ = place_result
            place_pos = list(place_pos)
            place_pos[2] += 0.15

        # Get bounding box for pick object (for IoU evaluation later)
        pick_bbox = env.get_object_bbox_2d(pick_object, view="topdown")
        pick_label = self._get_object_label(pick_object)

        # Get all object bboxes for the scene
        scene_bboxes = {}
        for obj_name in env.object_names:
            bbox = env.get_object_bbox_2d(obj_name, view="topdown")
            if bbox is not None:
                scene_bboxes[obj_name] = bbox

        def record_step(phase, instruction, action, target_object):
            current_pos, _ = env.get_ee_pose()
            image = env.render(view="topdown")
            img_path = self.images_dir / f"ep{episode_id:04d}_step{len(episode_demos):04d}.png"
            Image.fromarray(image).save(img_path)

            demo = {
                "image_path": str(img_path.relative_to(self.output_dir)),
                "instruction": instruction,
                "action": action.tolist(),
                "episode_id": episode_id,
                "phase": phase,
                "target_object": target_object,
                "object_label": self._get_object_label(target_object),
                "target_bbox": pick_bbox if target_object == pick_object else env.get_object_bbox_2d(target_object, view="topdown"),
                "scene_bboxes": scene_bboxes,
                "objects_in_scene": list(env.object_names),
            }
            episode_demos.append(demo)
            return current_pos

        # === Pick phases ===
        pick_instruction = self.generate_pick_instruction(pick_object)

        # Phase 1: Approach
        approach_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.20]
        for step in range(30):
            current_pos, _ = env.get_ee_pose()
            action = self.compute_expert_action(current_pos, approach_pos, gripper_open=True)
            record_step("approach_pick", pick_instruction, action, pick_object)
            env.step(action)
            if np.linalg.norm(np.array(current_pos) - np.array(approach_pos)) < 0.02:
                break

        # Phase 2: Lower to grasp
        grasp_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.05]
        for step in range(20):
            current_pos, _ = env.get_ee_pose()
            action = self.compute_expert_action(current_pos, grasp_pos, gripper_open=True)
            record_step("lower_to_grasp", pick_instruction, action, pick_object)
            env.step(action)
            if np.linalg.norm(np.array(current_pos) - np.array(grasp_pos)) < 0.02:
                break

        # Phase 3: Close gripper
        for step in range(10):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0
            record_step("close_gripper", pick_instruction, action, pick_object)
            env.step(action)

        # Phase 4: Lift
        lift_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.25]
        for step in range(20):
            current_pos, _ = env.get_ee_pose()
            action = self.compute_expert_action(current_pos, lift_pos, gripper_open=False)
            record_step("lift", pick_instruction, action, pick_object)
            env.step(action)
            if np.linalg.norm(np.array(current_pos) - np.array(lift_pos)) < 0.02:
                break

        # === Place phases ===
        place_instruction = self.generate_place_instruction(place_target)
        place_target_obj = place_target if place_target in env.object_ids else pick_object

        # Phase 5: Move to place
        place_above = [place_pos[0], place_pos[1], place_pos[2] + 0.20]
        for step in range(40):
            current_pos, _ = env.get_ee_pose()
            action = self.compute_expert_action(current_pos, place_above, gripper_open=False)
            record_step("move_to_place", place_instruction, action, place_target_obj)
            env.step(action)
            if np.linalg.norm(np.array(current_pos) - np.array(place_above)) < 0.03:
                break

        # Phase 6: Lower to place
        for step in range(20):
            current_pos, _ = env.get_ee_pose()
            action = self.compute_expert_action(current_pos, place_pos, gripper_open=False)
            record_step("lower_to_place", place_instruction, action, place_target_obj)
            env.step(action)
            if np.linalg.norm(np.array(current_pos) - np.array(place_pos)) < 0.02:
                break

        # Phase 7: Release
        for step in range(10):
            action = np.zeros(7, dtype=np.float32)
            action[6] = 1.0
            record_step("release", place_instruction, action, place_target_obj)
            env.step(action)

        return episode_demos

    def collect_with_splits(
        self,
        n_episodes: int = 500,
        n_objects: int = 3,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        camera_jitter_std: float = 0.05,
        n_distractors: int = 0,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        base_seed: int = 42,
    ):
        """
        Collect dataset with train/val/held-out splits using non-overlapping layout seeds.

        Layout seed pools:
        - Train: seeds [base_seed, base_seed + n_train)
        - Val: seeds [base_seed + n_train, base_seed + n_train + n_val)
        - Held-out: seeds [base_seed + n_train + n_val, base_seed + n_episodes)

        This guarantees no layout leakage between splits.
        """
        n_train = int(n_episodes * train_frac)
        n_val = int(n_episodes * val_frac)
        n_test = n_episodes - n_train - n_val

        print(f"Dataset generation plan:")
        print(f"  Total episodes: {n_episodes}")
        print(f"  Train: {n_train} (seeds {base_seed}..{base_seed + n_train - 1})")
        print(f"  Val:   {n_val} (seeds {base_seed + n_train}..{base_seed + n_train + n_val - 1})")
        print(f"  Test:  {n_test} (seeds {base_seed + n_train + n_val}..{base_seed + n_episodes - 1})")
        print(f"  Objects per scene: {n_objects}")
        print(f"  Scale range: {scale_range}")
        print(f"  Camera jitter std: {camera_jitter_std}")
        print(f"  Distractors: {n_distractors}")

        all_objects = TableTopEnv.AVAILABLE_OBJECTS.copy()
        splits = {
            "train": list(range(base_seed, base_seed + n_train)),
            "val": list(range(base_seed + n_train, base_seed + n_train + n_val)),
            "test": list(range(base_seed + n_train + n_val, base_seed + n_episodes)),
        }

        split_demos = {"train": [], "val": [], "test": []}
        layout_metadata = {"train": [], "val": [], "test": []}

        global_ep_id = 0

        for split_name, seed_pool in splits.items():
            print(f"\n--- Collecting {split_name} split ({len(seed_pool)} episodes) ---")

            for i, layout_seed in enumerate(tqdm(seed_pool, desc=f"{split_name}")):
                # Use layout seed for reproducible randomization
                env, object_names = setup_env(
                    n_objects=n_objects,
                    gui=False,
                    image_size=self.image_size,
                    layout_seed=layout_seed,
                    scale_range=scale_range,
                    camera_jitter_std=camera_jitter_std,
                    n_distractors=n_distractors,
                )

                # Cycle through target objects
                pick_object = all_objects[i % len(all_objects)]
                if pick_object not in object_names:
                    pick_object = object_names[i % len(object_names)]

                # Place target
                if random.random() < 0.5 and len(object_names) > 1:
                    place_candidates = [obj for obj in object_names if obj != pick_object]
                    place_target = random.choice(place_candidates) if place_candidates else list(env.special_pts.keys())[0]
                else:
                    place_target = random.choice(list(env.special_pts.keys()))

                # Collect episode
                episode_demos = self.execute_pick_place_episode(
                    env, pick_object, place_target, global_ep_id
                )

                # Store layout metadata
                meta = env.get_layout_metadata()
                meta["split"] = split_name
                meta["episode_id"] = global_ep_id
                meta["pick_object"] = pick_object
                meta["place_target"] = place_target

                split_demos[split_name].extend(episode_demos)
                layout_metadata[split_name].append(meta)

                global_ep_id += 1
                env.disconnect()

        # Save everything
        for split_name in ["train", "val", "test"]:
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)

            demos_path = split_dir / "demonstrations.json"
            with open(demos_path, 'w') as f:
                json.dump(split_demos[split_name], f, indent=2)

            meta_path = split_dir / "layout_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(layout_metadata[split_name], f, indent=2)

            print(f"{split_name}: {len(split_demos[split_name])} demos, {len(layout_metadata[split_name])} layouts")

        # Save combined metadata
        combined_meta = {
            "n_episodes": n_episodes,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "train_seeds": splits["train"],
            "val_seeds": splits["val"],
            "test_seeds": splits["test"],
            "n_train_demos": len(split_demos["train"]),
            "n_val_demos": len(split_demos["val"]),
            "n_test_demos": len(split_demos["test"]),
            "scale_range": list(scale_range),
            "camera_jitter_std": camera_jitter_std,
            "n_distractors": n_distractors,
            "n_objects": n_objects,
            "image_size": self.image_size,
            "base_seed": base_seed,
        }

        # Also save combined demonstrations.json for backward compat
        all_demos = split_demos["train"] + split_demos["val"] + split_demos["test"]
        with open(self.output_dir / "demonstrations.json", 'w') as f:
            json.dump(all_demos, f, indent=2)

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(combined_meta, f, indent=2)

        with open(self.output_dir / "layout_metadata.json", 'w') as f:
            json.dump(layout_metadata, f, indent=2)

        print(f"\nDataset saved to {self.output_dir}")
        print(f"Total demonstrations: {len(all_demos)}")

        return combined_meta


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect randomized VLA demonstrations")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-episodes", type=int, default=500, help="Total episodes")
    parser.add_argument("--n-objects", type=int, default=3, help="Objects per scene")
    parser.add_argument("--scale-min", type=float, default=0.8, help="Min object scale")
    parser.add_argument("--scale-max", type=float, default=1.2, help="Max object scale")
    parser.add_argument("--camera-jitter", type=float, default=0.05, help="Camera jitter std")
    parser.add_argument("--n-distractors", type=int, default=0, help="Distractor objects")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--base-seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    collector = DemonstrationCollector(
        output_dir=args.output_dir,
        image_size=args.image_size,
    )

    collector.collect_with_splits(
        n_episodes=args.n_episodes,
        n_objects=args.n_objects,
        scale_range=(args.scale_min, args.scale_max),
        camera_jitter_std=args.camera_jitter,
        n_distractors=args.n_distractors,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
