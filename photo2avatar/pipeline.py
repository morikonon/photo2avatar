from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import animation, body, render
from .gltf_export import write_glb
from .rotations import rot_x

CAMERA_TO_WORLD = rot_x(np.pi)  # OpenCV camera (y down) -> glTF (y up)
GIF_FPS = 30
GIF_YAW = 0.6


def export_avatar(rig: body.Rig, clips: list[animation.Clip], out_dir: Path) -> dict:
    colors = rig.part_colors()
    glb = write_glb(out_dir / "avatar.glb", rig, colors, clips, body.JOINT_NAMES)

    parts = [render.render_orthographic(rig.vertices, rig.faces, colors, size=(420, 480)),
             render.render_orthographic(rig.vertices, rig.faces, colors, size=(240, 480), yaw=np.pi / 2)]
    _save_rgb(out_dir / "parts.png", np.hstack(parts))

    for clip in clips:
        frames = [rig.pose(r, t)[0] for r, t in zip(clip.rotations, clip.root_offsets)]
        if clip.name == "photo_pose":
            views = [render.render_orthographic(frames[-1], rig.faces, colors, yaw=yaw) for yaw in (0.0, np.pi / 2)]
            _save_rgb(out_dir / "avatar.png", np.hstack(views))
            continue
        bounds = render.fixed_bounds(frames, GIF_YAW)
        images = [render.render_orthographic(f, rig.faces, colors, size=(300, 400), yaw=GIF_YAW, bounds=bounds)
                  for f in frames[:-1]]  # the last frame repeats the first
        render.save_gif(images, out_dir / f"{clip.name}.gif", GIF_FPS)

    return {
        "mesh": {"vertices": len(rig.vertices), "faces": len(rig.faces), "joints": len(rig.joints),
                 "parts": len(np.unique(rig.part_labels())),
                 "rest_height_m": round(float(np.ptp(rig.vertices[:, 1])), 3)},
        "glb": glb,
        "animations_s": {clip.name: round(float(clip.times[-1]), 2) for clip in clips},
    }


def run(image_path: Path, out_dir: Path, device: str | None = None, passes: int = 2,
        box: np.ndarray | None = None) -> dict:
    from .hmr import HMR2Estimator

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    estimator = HMR2Estimator(device)
    loaded = time.perf_counter()
    estimate = estimator.estimate(image, passes=passes, box=box)
    inferred = time.perf_counter()

    smpl = body.load_smpl()
    rig = body.build_rig(smpl, estimate.betas)
    pose = np.concatenate([(CAMERA_TO_WORLD @ estimate.global_orient)[None], estimate.body_pose])
    clips = [animation.photo_pose_clip(pose), animation.walk_clip(), animation.wave_clip()]
    exported = export_avatar(rig, clips, out_dir)

    overlay = render.render_overlay(image, estimate.vertices, rig.faces, rig.part_colors(),
                                    estimate.focal_length, estimate.principal_point)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)
    np.savez(out_dir / "smpl_params.npz", betas=estimate.betas, pose=pose, vertices_cam=estimate.vertices)

    report = {
        "input": str(image_path),
        "hmr2": {"device": estimator.device.type, "passes": passes,
                 "person_box": [round(float(c), 1) for c in estimate.box],
                 "model_load_s": round(loaded - start, 1), "inference_s": round(inferred - loaded, 1)},
        **exported,
        "rig_check": body.check_rig(rig, smpl, estimate.betas, pose),
        "total_s": round(time.perf_counter() - start, 1),
    }
    _save_report(out_dir, report)
    return report


def run_mean_body(out_dir: Path) -> dict:
    """Average SMPL body without a photo: checks the rig and export without HMR2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    smpl = body.load_smpl()
    betas = np.zeros(10)
    rig = body.build_rig(smpl, betas)
    walk = animation.walk_clip()
    report = export_avatar(rig, [walk, animation.wave_clip()], out_dir)
    report["rig_check"] = body.check_rig(rig, smpl, betas, walk.rotations[len(walk.times) // 4])
    _save_report(out_dir, report)
    return report


def _save_rgb(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image[:, :, ::-1])


def _save_report(out_dir: Path, report: dict) -> None:
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
