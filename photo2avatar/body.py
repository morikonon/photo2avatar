from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx
import torch

SMPL_MODEL_DIR = Path(os.environ.get("SMPL_MODEL_DIR", Path.home() / ".cache" / "4DHumans" / "data" / "smpl"))

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

PARTS = [
    "head", "torso", "hips",
    "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm", "left_hand", "right_hand",
    "left_thigh", "right_thigh", "left_shin", "right_shin", "left_foot", "right_foot",
]

# a joint's skinning weights move the bone that starts at it: left_hip -> thigh, left_knee -> shin
JOINT_TO_PART = {
    "pelvis": "hips", "spine1": "torso", "spine2": "torso", "spine3": "torso",
    "left_collar": "torso", "right_collar": "torso", "neck": "head", "head": "head",
    "left_hip": "left_thigh", "right_hip": "right_thigh",
    "left_knee": "left_shin", "right_knee": "right_shin",
    "left_ankle": "left_foot", "right_ankle": "right_foot", "left_foot": "left_foot", "right_foot": "right_foot",
    "left_shoulder": "left_upper_arm", "right_shoulder": "right_upper_arm",
    "left_elbow": "left_forearm", "right_elbow": "right_forearm",
    "left_wrist": "left_hand", "right_wrist": "right_hand", "left_hand": "left_hand", "right_hand": "right_hand",
}

PART_COLORS = np.array([
    (242, 201, 76), (86, 135, 200), (60, 90, 150),
    (235, 110, 90), (180, 70, 60), (245, 160, 90), (200, 120, 50), (250, 215, 150), (215, 175, 110),
    (100, 190, 130), (60, 140, 95), (150, 210, 190), (90, 165, 150), (170, 150, 220), (120, 100, 180),
], dtype=np.float64) / 255.0


def load_smpl(model_dir: Path = SMPL_MODEL_DIR) -> smplx.SMPL:
    return smplx.SMPL(model_path=str(model_dir), gender="neutral")


@dataclass
class Rig:
    vertices: np.ndarray  # (V, 3) T-pose with the person's shape, metres, feet on y = 0
    faces: np.ndarray     # (F, 3)
    joints: np.ndarray    # (J, 3) rest joint positions
    parents: np.ndarray   # (J,) parent index, -1 for the root; parents come before children
    weights: np.ndarray   # (V, J) skinning weights

    @property
    def offsets(self) -> np.ndarray:
        offsets = self.joints.copy()
        offsets[1:] -= self.joints[self.parents[1:]]
        return offsets

    def part_labels(self) -> np.ndarray:
        joint_to_part = np.array([PARTS.index(JOINT_TO_PART[name]) for name in JOINT_NAMES])
        return joint_to_part[self.weights.argmax(axis=1)]

    def part_colors(self) -> np.ndarray:
        return PART_COLORS[self.part_labels()]

    def pose(self, rotations: np.ndarray, root_offset: np.ndarray | None = None):
        """Forward kinematics + linear blend skinning, the same maths a glTF viewer runs.

        rotations: (J, 3, 3) local joint rotations. Returns posed vertices and joint positions.
        """
        offsets = self.offsets
        if root_offset is not None:
            offsets[0] += root_offset
        world = np.zeros((len(self.parents), 4, 4))
        for j, parent in enumerate(self.parents):
            local = np.eye(4)
            local[:3, :3] = rotations[j]
            local[:3, 3] = offsets[j]
            world[j] = local if parent < 0 else world[parent] @ local
        # bind pose is a pure translation, so world @ inverse(bind) only shifts the translation
        skin = world[:, :3, :].copy()
        skin[:, :, 3] -= np.einsum("jab,jb->ja", world[:, :3, :3], self.joints)
        blended = np.einsum("vj,jab->vab", self.weights, skin)
        vertices = np.einsum("vab,vb->va", blended[:, :, :3], self.vertices) + blended[:, :, 3]
        return vertices, world[:, :3, 3]


def build_rig(smpl: smplx.SMPL, betas: np.ndarray) -> Rig:
    betas_t = torch.as_tensor(np.asarray(betas), dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        vertices = smpl.v_template + smplx.lbs.blend_shapes(betas_t, smpl.shapedirs)[0]
        joints = smpl.J_regressor @ vertices
    vertices = vertices.numpy().astype(np.float64)
    joints = joints.numpy().astype(np.float64)
    ground = np.array([-joints[0, 0], -vertices[:, 1].min(), -joints[0, 2]])
    return Rig(
        vertices=vertices + ground,
        faces=np.asarray(smpl.faces, dtype=np.int64),
        joints=joints + ground,
        parents=smpl.parents.numpy().astype(np.int64),
        weights=smpl.lbs_weights.numpy().astype(np.float64),
    )


def check_rig(rig: Rig, smpl: smplx.SMPL, betas: np.ndarray, rotations: np.ndarray) -> dict:
    """Compares the rig with SMPL for one pose.

    rig_vs_smpl_lbs: our skinning vs SMPL's own LBS (should be ~0).
    pose_correctives: SMPL's pose-dependent blend shapes, which glTF skinning cannot play back.
    """
    betas_t = torch.as_tensor(np.asarray(betas), dtype=torch.float32).reshape(1, -1)
    pose_t = torch.as_tensor(np.asarray(rotations), dtype=torch.float32).reshape(1, -1, 3, 3)

    def smpl_lbs(posedirs):
        with torch.no_grad():
            v, j = smplx.lbs.lbs(betas_t, pose_t, smpl.v_template, smpl.shapedirs, posedirs,
                                 smpl.J_regressor, smpl.parents, smpl.lbs_weights, pose2rot=False)
        return (v[0] - j[0, 0]).numpy()

    full = smpl_lbs(smpl.posedirs)
    plain = smpl_lbs(torch.zeros_like(smpl.posedirs))
    ours, ours_joints = rig.pose(rotations)
    correctives = np.linalg.norm(full - plain, axis=1) * 1000
    return {
        "rig_vs_smpl_lbs_max_mm": round(float(np.linalg.norm(ours - ours_joints[0] - plain, axis=1).max() * 1000), 4),
        "pose_correctives_mean_mm": round(float(correctives.mean()), 2),
        "pose_correctives_max_mm": round(float(correctives.max()), 2),
    }
