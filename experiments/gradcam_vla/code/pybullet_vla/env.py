"""
PyBullet Tabletop Environment for VLA Training

Randomized environment supporting:
- Random object positions per episode
- Random camera jitter
- Random object scale (+/-10-20%)
- Random distractor objects
- Layout seed logging for reproducibility and split isolation
"""

import os
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pybullet as p
import pybullet_data


class TableTopEnv:
    """Randomized tabletop manipulation environment for VLA training."""

    # Available objects - using PyBullet's built-in URDFs
    AVAILABLE_OBJECTS = [
        "duck_vhacd",      # Rubber duck
        "sphere_small",    # Small sphere
        "cube_small",      # Small cube
        "teddy_vhacd",     # Teddy bear
        "jenga/jenga",     # Jenga block
    ]

    # Object display names for instructions
    OBJECT_NAMES = {
        "duck_vhacd": "duck",
        "sphere_small": "sphere",
        "cube_small": "cube",
        "teddy_vhacd": "teddy bear",
        "jenga/jenga": "block",
    }

    def __init__(self, gui: bool = False, image_size: int = 224):
        self.gui = gui
        self.image_size = image_size
        self.client_id = None

        # IDs
        self.plane_id = None
        self.table_id = None
        self.kuka_id = None
        self.gripper_id = None
        self.object_ids = {}
        self.object_names = []
        self.distractor_ids = []

        # Robot constants
        self.ee_idx = 6
        self.num_joints = None

        # Workspace bounds (meters)
        self.workspace = {
            "x": (0.55, 1.08),
            "y": (-0.83, 0.43),
            "z": (0.65, 1.0),
        }

        # Special locations
        xmin, xmax = self.workspace["x"]
        ymin, ymax = self.workspace["y"]
        z = self.workspace["z"][0]
        self.special_pts = {
            "top_left_corner": [xmin + 0.05, ymax - 0.05, z],
            "top_right_corner": [xmax - 0.05, ymax - 0.05, z],
            "bottom_left_corner": [xmin + 0.05, ymin + 0.05, z],
            "bottom_right_corner": [xmax - 0.05, ymin + 0.05, z],
            "middle": [(xmin + xmax) / 2, (ymin + ymax) / 2, z],
        }

        # Randomization state (logged per episode)
        self._layout_seed = None
        self._object_scales = {}
        self._camera_jitter = None

    def connect(self):
        """Connect to PyBullet physics server."""
        if p.isConnected():
            p.disconnect()

        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(1. / 240.)

    def disconnect(self):
        """Disconnect from PyBullet."""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def reset(
        self,
        object_names: List[str],
        layout_seed: Optional[int] = None,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        camera_jitter_std: float = 0.05,
        n_distractors: int = 0,
        gso_path: Optional[str] = None,
    ):
        """
        Reset environment with specified objects and randomization.

        Args:
            object_names: List of object names to spawn
            layout_seed: RNG seed for reproducible layouts (logged for splits)
            scale_range: (min_scale, max_scale) for object size randomization
            camera_jitter_std: Std dev for camera position jitter (meters)
            n_distractors: Number of additional distractor objects
            gso_path: Path to Google Scanned Objects dataset
        """
        if not p.isConnected():
            self.connect()

        # Set layout seed for reproducibility
        if layout_seed is not None:
            self._layout_seed = layout_seed
            random.seed(layout_seed)
            np.random.seed(layout_seed)
        else:
            self._layout_seed = random.randint(0, 2**31)
            random.seed(self._layout_seed)
            np.random.seed(self._layout_seed)

        # Clear previous objects
        for obj_id in self.object_ids.values():
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        for obj_id in self.distractor_ids:
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        self.object_ids = {}
        self.object_names = object_names
        self.distractor_ids = []
        self._object_scales = {}

        # Camera jitter for this episode
        self._camera_jitter = np.random.normal(0, camera_jitter_std, 3)

        # Load scene
        self._load_scene()

        # Load robot
        self._load_robot()

        # Load target objects with random scale
        all_positions = []
        n_total = len(object_names) + n_distractors
        positions = self._generate_object_positions(n_total)

        for i, (name, pos) in enumerate(zip(object_names, positions[:len(object_names)])):
            scale = random.uniform(*scale_range)
            self._object_scales[name] = scale
            obj_path = f"{name}.urdf"
            try:
                obj_id = p.loadURDF(obj_path, basePosition=pos, globalScaling=scale)
                self.object_ids[name] = obj_id
                all_positions.append(pos)
            except Exception as e:
                print(f"Warning: Failed to load {name}: {e}")

        # Load distractor objects
        if n_distractors > 0:
            available_distractors = [
                obj for obj in self.AVAILABLE_OBJECTS if obj not in object_names
            ]
            for i in range(n_distractors):
                if not available_distractors:
                    break
                dist_name = random.choice(available_distractors)
                pos = positions[len(object_names) + i]
                scale = random.uniform(*scale_range)
                obj_path = f"{dist_name}.urdf"
                try:
                    obj_id = p.loadURDF(obj_path, basePosition=pos, globalScaling=scale)
                    self.distractor_ids.append(obj_id)
                except Exception:
                    pass

        # Settle physics
        for _ in range(100):
            p.stepSimulation()

    def get_layout_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current layout for logging.

        Returns:
            Dict with layout_seed, object positions, scales, camera jitter
        """
        obj_positions = {}
        for name, obj_id in self.object_ids.items():
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            obj_positions[name] = {
                "position": list(pos),
                "orientation": list(orn),
                "scale": self._object_scales.get(name, 1.0),
            }

        return {
            "layout_seed": self._layout_seed,
            "object_names": list(self.object_names),
            "object_positions": obj_positions,
            "object_scales": dict(self._object_scales),
            "camera_jitter": self._camera_jitter.tolist() if self._camera_jitter is not None else [0, 0, 0],
            "n_distractors": len(self.distractor_ids),
        }

    def _load_scene(self):
        """Load table and ground plane."""
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[1.0, -0.2, 0.0],
            baseOrientation=[0, 0, 0.7071, 0.7071]
        )

    def _load_robot(self):
        """Load KUKA robot and gripper."""
        self.kuka_id = p.loadURDF(
            "kuka_iiwa/model_vr_limits.urdf",
            basePosition=[1.4, -0.2, 0.6],
            baseOrientation=[0, 0, 0, 1]
        )

        self.gripper_id = p.loadSDF(
            "gripper/wsg50_one_motor_gripper_new_free_base.sdf"
        )[0]

        p.createConstraint(
            self.kuka_id, 6, self.gripper_id, 0, p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0.05], [0, 0, 0]
        )

        cid = p.createConstraint(
            self.gripper_id, 4, self.gripper_id, 6,
            jointType=p.JOINT_GEAR, jointAxis=[1, 1, 1],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(cid, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

        self._reset_robot_pose()
        self.num_joints = p.getNumJoints(self.kuka_id)

    def _reset_robot_pose(self):
        """Reset robot to home configuration."""
        kuka_joints = [0.0, 0.75, 0.0, 1.570793, 0.0, -1.036725, 0.0]
        gripper_base_pos = [0.923103, -0.2, 1.250036]
        gripper_base_orn = [-0.0, 0.964531, -0.0, -0.263970]

        for j in range(p.getNumJoints(self.kuka_id)):
            p.resetJointState(self.kuka_id, j, kuka_joints[j])
            p.setJointMotorControl2(self.kuka_id, j, p.POSITION_CONTROL, kuka_joints[j], 0)

        p.resetBasePositionAndOrientation(self.gripper_id, gripper_base_pos, gripper_base_orn)

        gripper_joints = [0.0, -0.011130, -0.206421, 0.205143, -0.009999, 0.0, -0.010055, 0.0]
        for j in range(p.getNumJoints(self.gripper_id)):
            p.resetJointState(self.gripper_id, j, gripper_joints[j])
            p.setJointMotorControl2(self.gripper_id, j, p.POSITION_CONTROL, gripper_joints[j], 0)

    def _generate_object_positions(self, n: int, min_distance: float = 0.25) -> List[List[float]]:
        """Generate random non-overlapping positions for objects."""
        positions = []
        max_attempts = 1000

        if n <= 2:
            x_range = (0.65, 0.95)
            y_range = (-0.50, 0.25)
        elif n <= 3:
            x_range = (0.62, 0.98)
            y_range = (-0.55, 0.30)
        else:
            x_range = (0.60, 1.00)
            y_range = (-0.65, 0.35)

        z = 0.70

        for _ in range(n):
            placed = False
            for _ in range(max_attempts):
                x = random.uniform(*x_range)
                y = random.uniform(*y_range)
                pos = [x, y, z]

                valid = all(
                    math.dist(pos, existing) >= min_distance
                    for existing in positions
                )

                if valid:
                    positions.append(pos)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Could not place {n} objects with min_distance={min_distance}")

        return positions

    def render(self, view: str = "topdown") -> np.ndarray:
        """
        Render camera image with optional random jitter.

        Args:
            view: "topdown" or "side"

        Returns:
            RGB image as numpy array (H, W, 3) uint8
        """
        if view == "topdown":
            target = [0.85, -0.2, 0.65]
            distance = 1.5
            yaw = 0
            pitch = -90
            roll = 0
        elif view == "side":
            target = [0.85, -0.2, 0.65]
            distance = 1.5
            yaw = 45
            pitch = -35
            roll = 0
        else:
            raise ValueError(f"Unknown view: {view}")

        # Apply camera jitter
        if self._camera_jitter is not None:
            target = [
                target[0] + self._camera_jitter[0],
                target[1] + self._camera_jitter[1],
                target[2] + self._camera_jitter[2],
            ]

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.01,
            farVal=100
        )

        # Try hardware renderer, fall back to tiny renderer
        try:
            img = p.getCameraImage(
                self.image_size,
                self.image_size,
                view_matrix,
                proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
        except Exception:
            img = p.getCameraImage(
                self.image_size,
                self.image_size,
                view_matrix,
                proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )

        rgb = np.array(img[2])[:, :, :3]
        return rgb.astype(np.uint8)

    def get_ee_pose(self) -> Tuple[List[float], List[float]]:
        """Get end-effector position and orientation."""
        state = p.getLinkState(self.kuka_id, self.ee_idx)
        pos = list(state[4])
        orn = list(state[5])
        return pos, orn

    def get_object_pose(self, object_name: str) -> Optional[Tuple[List[float], List[float]]]:
        """Get object position and orientation."""
        if object_name not in self.object_ids:
            return None
        obj_id = self.object_ids[object_name]
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        return list(pos), list(orn)

    def move_ee(self, position: List[float], orientation: Optional[List[float]] = None):
        """Move end-effector to target position via IK."""
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, math.pi, 0])

        joint_poses = p.calculateInverseKinematics(
            self.kuka_id,
            self.ee_idx,
            position,
            orientation
        )

        for j in range(self.num_joints):
            p.setJointMotorControl2(
                self.kuka_id,
                j,
                p.POSITION_CONTROL,
                joint_poses[j]
            )

    def set_gripper(self, open: bool):
        """Set gripper state."""
        width = -0.1 if open else 0.15
        force = 300 if open else 200

        p.setJointMotorControl2(self.gripper_id, 4, p.POSITION_CONTROL, width, force=force)
        p.setJointMotorControl2(self.gripper_id, 6, p.POSITION_CONTROL, width, force=force)

        for _ in range(50):
            p.stepSimulation()

    def step(self, action: np.ndarray, steps: int = 20):
        """
        Execute action (delta end-effector pose).

        Args:
            action: [dx, dy, dz, droll, dpitch, dyaw, gripper] in [-1, 1]
            steps: Number of simulation steps
        """
        current_pos, current_orn = self.get_ee_pose()
        current_euler = p.getEulerFromQuaternion(current_orn)

        pos_scale = 0.05
        orn_scale = 0.1

        target_pos = [
            current_pos[0] + action[0] * pos_scale,
            current_pos[1] + action[1] * pos_scale,
            current_pos[2] + action[2] * pos_scale,
        ]

        target_euler = [
            current_euler[0] + action[3] * orn_scale,
            current_euler[1] + action[4] * orn_scale,
            current_euler[2] + action[5] * orn_scale,
        ]
        target_orn = p.getQuaternionFromEuler(target_euler)

        target_pos[0] = np.clip(target_pos[0], *self.workspace["x"])
        target_pos[1] = np.clip(target_pos[1], *self.workspace["y"])
        target_pos[2] = np.clip(target_pos[2], *self.workspace["z"])

        self.move_ee(target_pos, target_orn)
        self.set_gripper(action[6] > 0)

        for _ in range(steps):
            p.stepSimulation()

    def get_object_bbox_2d(self, object_name: str, view: str = "topdown") -> Optional[Dict]:
        """
        Get approximate 2D bounding box of an object in the rendered image.

        Uses the object's 3D position projected through the camera matrix.
        Returns normalized coordinates [0,1] relative to image dimensions.

        Args:
            object_name: Name of the object
            view: Camera view ("topdown" or "side")

        Returns:
            Dict with 'center_x', 'center_y', 'bbox' (x1,y1,x2,y2) in [0,1],
            or None if object not found
        """
        pose = self.get_object_pose(object_name)
        if pose is None:
            return None

        pos, _ = pose

        # Camera parameters matching render()
        if view == "topdown":
            target = [0.85, -0.2, 0.65]
            distance = 1.5
            yaw = 0
            pitch = -90
            roll = 0
        else:
            target = [0.85, -0.2, 0.65]
            distance = 1.5
            yaw = 45
            pitch = -35
            roll = 0

        if self._camera_jitter is not None:
            target = [
                target[0] + self._camera_jitter[0],
                target[1] + self._camera_jitter[1],
                target[2] + self._camera_jitter[2],
            ]

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=100
        )

        # Project 3D point to 2D
        vm = np.array(view_matrix).reshape(4, 4, order='F')
        pm = np.array(proj_matrix).reshape(4, 4, order='F')
        pos4 = np.array([pos[0], pos[1], pos[2], 1.0])
        clip = pm @ vm @ pos4

        if abs(clip[3]) < 1e-6:
            return None

        ndc = clip[:3] / clip[3]  # [-1, 1] range
        px = (ndc[0] + 1) / 2  # [0, 1]
        py = (1 - ndc[1]) / 2  # [0, 1], flip y

        # Approximate bbox size based on object scale and distance
        scale = self._object_scales.get(object_name, 1.0)
        bbox_half = 0.06 * scale  # approximate, depends on object

        return {
            "center_x": float(np.clip(px, 0, 1)),
            "center_y": float(np.clip(py, 0, 1)),
            "bbox": [
                float(np.clip(px - bbox_half, 0, 1)),
                float(np.clip(py - bbox_half, 0, 1)),
                float(np.clip(px + bbox_half, 0, 1)),
                float(np.clip(py + bbox_half, 0, 1)),
            ],
        }


def setup_env(
    n_objects: int = 3,
    gui: bool = False,
    image_size: int = 224,
    layout_seed: Optional[int] = None,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    camera_jitter_std: float = 0.05,
    n_distractors: int = 0,
    gso_path: Optional[str] = None,
) -> Tuple[TableTopEnv, List[str]]:
    """
    Set up environment with random objects and randomization.

    Args:
        n_objects: Number of target objects to spawn
        gui: Show GUI
        image_size: Camera image size
        layout_seed: Seed for reproducible layout
        scale_range: Object scale randomization range
        camera_jitter_std: Camera position jitter std
        n_distractors: Number of distractor objects
        gso_path: Path to GSO dataset

    Returns:
        (env, object_names)
    """
    env = TableTopEnv(gui=gui, image_size=image_size)
    env.connect()

    selected = random.sample(
        TableTopEnv.AVAILABLE_OBJECTS,
        min(n_objects, len(TableTopEnv.AVAILABLE_OBJECTS))
    )

    env.reset(
        selected,
        layout_seed=layout_seed,
        scale_range=scale_range,
        camera_jitter_std=camera_jitter_std,
        n_distractors=n_distractors,
        gso_path=gso_path,
    )

    return env, selected
