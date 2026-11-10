from __future__ import annotations

import numpy as np


def _axis_rotation(angle, axis: int) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.zeros(a.shape + (3, 3))
    R[..., axis, axis] = 1.0
    R[..., i, i] = c
    R[..., j, j] = c
    R[..., i, j] = -s
    R[..., j, i] = s
    return R


def rot_x(angle) -> np.ndarray:
    return _axis_rotation(angle, 0)


def rot_y(angle) -> np.ndarray:
    return _axis_rotation(angle, 1)


def rot_z(angle) -> np.ndarray:
    return _axis_rotation(angle, 2)


def matrix_to_quaternion(R) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 4) unit quaternions in glTF order (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    q = np.empty((len(flat), 4))
    for n, m in enumerate(flat):
        trace = np.trace(m)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            q[n] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[n] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[n] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[n] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(R.shape[:-2] + (4,))


def quaternion_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def make_continuous(q: np.ndarray) -> np.ndarray:
    # q and -q are the same rotation, but slerping between keys of opposite sign
    # takes the long way round and the joint spins
    q = np.array(q, dtype=np.float64)
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return q
