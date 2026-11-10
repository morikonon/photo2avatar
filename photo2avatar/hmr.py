from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BodyEstimate:
    betas: np.ndarray            # (10,)
    global_orient: np.ndarray    # (3, 3) in the OpenCV camera frame: x right, y down, z forward
    body_pose: np.ndarray        # (23, 3, 3) local joint rotations
    vertices: np.ndarray         # (6890, 3) posed mesh in camera coordinates, metres
    focal_length: float
    principal_point: np.ndarray
    box: np.ndarray              # person box used in the last pass, xyxy

    def project(self) -> np.ndarray:
        return self.focal_length * self.vertices[:, :2] / self.vertices[:, 2:3] + self.principal_point


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cam_crop_to_full(cam: np.ndarray, box_center: np.ndarray, box_size: float,
                     width: int, height: int, focal: float) -> np.ndarray:
    """Weak-perspective camera (s, tx, ty) of the crop -> camera translation for the full image."""
    scale, tx, ty = cam
    size = box_size * scale + 1e-9
    return np.array([
        2 * (box_center[0] - width / 2) / size + tx,
        2 * (box_center[1] - height / 2) / size + ty,
        2 * focal / size,
    ])


def person_box(points: np.ndarray, pad: float = 1.2) -> np.ndarray:
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo) * pad / 2
    return np.concatenate([center - half, center + half])


class _NoRenderer:
    def __init__(self, *args, **kwargs):
        pass


def load_hmr2():
    if sys.platform == "darwin":
        # pyrender, imported by hmr2, defaults to EGL, which macOS doesn't have
        os.environ.setdefault("PYOPENGL_PLATFORM", "darwin")
    import hmr2.models.hmr2 as hmr2_module
    from hmr2.configs import get_config
    from hmr2.models import DEFAULT_CHECKPOINT, HMR2, check_smpl_exists

    # HMR2 builds an OpenGL renderer for training visualisations; inference doesn't need it
    hmr2_module.MeshRenderer = _NoRenderer

    cfg = get_config(str(Path(DEFAULT_CHECKPOINT).parents[1] / "model_config.yaml"), update_cachedir=True)
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.freeze()
    check_smpl_exists()
    # the official checkpoint pickles its config (omegaconf, defaultdict),
    # which the weights_only loader of torch >= 2.6 rejects
    model = HMR2.load_from_checkpoint(DEFAULT_CHECKPOINT, strict=False, cfg=cfg,
                                      map_location="cpu", weights_only=False)
    return model, cfg


class HMR2Estimator:
    def __init__(self, device: str | None = None):
        self.model, self.cfg = load_hmr2()
        self.device = torch.device(device or auto_device())
        self.model = self.model.to(self.device).eval()

    def estimate(self, image_bgr: np.ndarray, passes: int = 2, box: np.ndarray | None = None) -> BodyEstimate:
        """The first pass runs on the whole image (or `box`). Each next pass crops around the
        projected mesh of the previous one, so no separate person detector is needed."""
        height, width = image_bgr.shape[:2]
        box = np.array([0, 0, width, height], dtype=np.float64) if box is None else np.asarray(box, np.float64)
        estimate = self._predict(image_bgr, box)
        for _ in range(passes - 1):
            estimate = self._predict(image_bgr, person_box(estimate.project()))
        return estimate

    @torch.no_grad()
    def _predict(self, image_bgr: np.ndarray, box: np.ndarray) -> BodyEstimate:
        from hmr2.datasets.vitdet_dataset import ViTDetDataset

        with contextlib.redirect_stdout(io.StringIO()):  # the dataset prints debug output
            item = ViTDetDataset(self.cfg, image_bgr, box[None].astype(np.float32))[0]
        out = self._forward(torch.as_tensor(item["img"], dtype=torch.float32)[None])

        height, width = image_bgr.shape[:2]
        focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(height, width)
        cam_t = cam_crop_to_full(out["pred_cam"][0].cpu().numpy(), np.asarray(item["box_center"]),
                                 float(item["box_size"]), width, height, focal)
        params = {k: v.cpu().numpy() for k, v in out["pred_smpl_params"].items()}
        return BodyEstimate(
            betas=params["betas"].reshape(-1),
            global_orient=params["global_orient"].reshape(-1, 3, 3)[0],
            body_pose=params["body_pose"].reshape(23, 3, 3),
            vertices=out["pred_vertices"][0].cpu().numpy() + cam_t,
            focal_length=float(focal),
            principal_point=np.array([width / 2, height / 2]),
            box=box,
        )

    def _forward(self, image: torch.Tensor) -> dict:
        try:
            return self.model({"img": image.to(self.device)})
        except (RuntimeError, NotImplementedError):
            if self.device.type == "cpu":
                raise
            self.device = torch.device("cpu")  # an op the accelerator doesn't support
            self.model = self.model.to(self.device)
            return self.model({"img": image})
