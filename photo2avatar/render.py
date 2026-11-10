from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .rotations import rot_y

LIGHT = np.array([0.3, 0.6, 1.0]) / np.linalg.norm([0.3, 0.6, 1.0])
SUPERSAMPLE = 2


def render_orthographic(vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray,
                        size: tuple[int, int] = (360, 480), yaw: float = 0.0,
                        bounds: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """RGB view of a y-up mesh seen from +z."""
    width, height = size
    v = vertices @ rot_y(yaw).T
    x0, x1, y0, y1 = bounds or (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())
    scale = 0.9 * min(width / (x1 - x0), height / (y1 - y0)) * SUPERSAMPLE
    pixels = np.column_stack([
        width * SUPERSAMPLE / 2 + (v[:, 0] - (x0 + x1) / 2) * scale,
        height * SUPERSAMPLE / 2 - (v[:, 1] - (y0 + y1) / 2) * scale,
    ])
    normals = _face_normals(v, faces)
    visible = np.flatnonzero(normals[:, 2] > 0)
    order = visible[np.argsort(v[faces[visible], 2].mean(axis=1))]
    shade = 0.35 + 0.65 * np.clip(normals @ LIGHT, 0, 1)
    colors = np.clip(vertex_colors[faces].mean(axis=1) * shade[:, None] * 255, 0, 255)

    canvas = np.full((height * SUPERSAMPLE, width * SUPERSAMPLE, 3), 255, np.uint8)
    _fill(canvas, pixels, faces, order, colors)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)


def fixed_bounds(frames: list[np.ndarray], yaw: float = 0.0) -> tuple[float, float, float, float]:
    v = np.concatenate(frames) @ rot_y(yaw).T
    return v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()


def render_overlay(image_bgr: np.ndarray, vertices_cam: np.ndarray, faces: np.ndarray,
                   vertex_colors: np.ndarray, focal: float, principal_point: np.ndarray,
                   alpha: float = 0.7) -> np.ndarray:
    pixels = focal * vertices_cam[:, :2] / vertices_cam[:, 2:3] + principal_point
    centroids = vertices_cam[faces].mean(axis=1)
    normals = _face_normals(vertices_cam, faces)
    facing = -(normals * centroids).sum(axis=1) / np.linalg.norm(centroids, axis=1)
    visible = np.flatnonzero(facing > 0)
    order = visible[np.argsort(-centroids[visible, 2])]
    shade = 0.35 + 0.65 * np.clip(facing, 0, 1)
    colors = np.clip(vertex_colors[faces].mean(axis=1)[:, ::-1] * shade[:, None] * 255, 0, 255)

    layer = np.zeros_like(image_bgr)
    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    _fill(layer, pixels, faces, order, colors)
    _fill(mask, pixels, faces, order, np.full((len(faces), 1), 255.0))
    weight = (mask.astype(np.float32) / 255 * alpha)[:, :, None]
    return (image_bgr * (1 - weight) + layer * weight).astype(np.uint8)


def save_gif(frames: list[np.ndarray], path, fps: int) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _fill(canvas: np.ndarray, pixels: np.ndarray, faces: np.ndarray, order: np.ndarray, colors: np.ndarray) -> None:
    # painter's algorithm: faces are drawn back to front; shift=2 gives sub-pixel precision
    fixed = np.round(pixels * 4).astype(np.int32)
    for f in order:
        cv2.fillConvexPoly(canvas, fixed[faces[f]], colors[f].tolist(), lineType=cv2.LINE_8, shift=2)
