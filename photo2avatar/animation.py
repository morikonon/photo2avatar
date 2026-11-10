from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body import JOINT_NAMES
from .rotations import rot_x, rot_y, rot_z

# Axes as in SMPL and glTF: y up, the body faces +z, its left side is +x.
# A negative rotation about x swings a hanging limb forward.

J = {name: i for i, name in enumerate(JOINT_NAMES)}
ARMS_DOWN = 1.2  # radians from the T-pose to arms at the sides


@dataclass
class Clip:
    name: str
    times: np.ndarray         # (K,) seconds
    rotations: np.ndarray     # (K, J, 3, 3) local joint rotations
    root_offsets: np.ndarray  # (K, 3) added to the root's rest position


def _rest(n_frames: int) -> np.ndarray:
    return np.tile(np.eye(3), (n_frames, len(JOINT_NAMES), 1, 1))


def _loop_times(duration: float, fps: int) -> np.ndarray:
    return np.arange(round(duration * fps) + 1) / fps  # last key equals the first one


def photo_pose_clip(pose: np.ndarray, blend: float = 1.0, hold: float = 1.0) -> Clip:
    rotations = np.stack([_rest(1)[0], pose, pose])
    return Clip("photo_pose", np.array([0.0, blend, blend + hold]), rotations, np.zeros((3, 3)))


def walk_clip(period: float = 1.1, fps: int = 30) -> Clip:
    t = _loop_times(period, fps)
    phase = 2 * np.pi * t / period
    R = _rest(len(t))
    for side, sign, shift in (("left", 1, 0.0), ("right", -1, np.pi)):
        p = phase + shift
        R[:, J[f"{side}_hip"]] = rot_x(-0.45 * np.sin(p))
        R[:, J[f"{side}_knee"]] = rot_x(0.08 + 0.75 * np.clip(np.cos(p), 0, None) ** 1.5)  # bends in the swing phase
        R[:, J[f"{side}_shoulder"]] = rot_x(0.35 * np.sin(p)) @ rot_z(-sign * ARMS_DOWN)  # opposite to the leg
        R[:, J[f"{side}_elbow"]] = rot_y(-sign * (0.25 + 0.15 * np.clip(-np.sin(p), 0, None)))
    R[:, J["pelvis"]] = rot_y(0.08 * np.sin(phase))
    R[:, J["spine3"]] = rot_y(-0.10 * np.sin(phase))
    bob = np.zeros((len(t), 3))
    bob[:, 1] = 0.015 * np.cos(2 * phase)
    return Clip("walk", t, R, bob)


def wave_clip(duration: float = 2.0, fps: int = 30, waves_per_second: float = 1.5) -> Clip:
    t = _loop_times(duration, fps)
    R = _rest(len(t))
    R[:, J["left_shoulder"]] = rot_z(-ARMS_DOWN)
    R[:, J["left_elbow"]] = rot_y(-0.25)
    R[:, J["right_shoulder"]] = rot_z(-0.9)
    R[:, J["right_elbow"]] = rot_z(-0.7 - 0.35 * np.sin(2 * np.pi * waves_per_second * t))
    R[:, J["neck"]] = rot_z(0.06 * np.sin(2 * np.pi * t / duration))
    return Clip("wave", t, R, np.zeros((len(t), 3)))
