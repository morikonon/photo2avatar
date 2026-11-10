from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .animation import Clip
from .body import Rig
from .rotations import make_continuous, matrix_to_quaternion

FLOAT, UNSIGNED_SHORT, UNSIGNED_INT = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
GLB_MAGIC, CHUNK_JSON, CHUNK_BIN = b"glTF", b"JSON", b"BIN\x00"
DTYPES = {FLOAT: np.float32, UNSIGNED_SHORT: np.uint16, UNSIGNED_INT: np.uint32}
WIDTHS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class _Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, component_type: int, kind: str,
            target: int | None = None, bounds: bool = False) -> int:
        array = np.ascontiguousarray(array)
        self.data += b"\x00" * (-len(self.data) % 4)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": array.nbytes}
        if target is not None:
            view["target"] = target
        self.data += array.tobytes()
        self.views.append(view)
        accessor = {"bufferView": len(self.views) - 1, "componentType": component_type,
                    "count": int(array.shape[0]), "type": kind}
        if bounds:  # required for POSITION and animation times
            flat = array.reshape(array.shape[0], -1)
            accessor["min"] = flat.min(axis=0).tolist()
            accessor["max"] = flat.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def skin_attributes(weights: np.ndarray, max_sets: int = 2):
    """Dense (V, J) weights -> JOINTS_n / WEIGHTS_n sets of 4 influences each."""
    influences = int((weights > 1e-6).sum(axis=1).max())
    n_sets = min(max(1, -(-influences // 4)), max_sets)
    joints = np.argsort(-weights, axis=1)[:, :4 * n_sets]
    values = np.take_along_axis(weights, joints, axis=1)
    values[values < 1e-6] = 0.0
    joints[values == 0] = 0  # unused slots must point at joint 0
    values /= values.sum(axis=1, keepdims=True)
    sets = [(joints[:, 4 * s:4 * s + 4].astype(np.uint16), values[:, 4 * s:4 * s + 4].astype(np.float32))
            for s in range(n_sets)]
    return influences, sets


def write_glb(path: Path, rig: Rig, colors: np.ndarray, clips: list[Clip], joint_names: list[str]) -> dict:
    buf = _Buffer()
    attributes = {
        "POSITION": buf.add(rig.vertices.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER, bounds=True),
        "NORMAL": buf.add(vertex_normals(rig.vertices, rig.faces).astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER),
        "COLOR_0": buf.add(colors.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER),
    }
    influences, sets = skin_attributes(rig.weights)
    for s, (joints, weights) in enumerate(sets):
        attributes[f"JOINTS_{s}"] = buf.add(joints, UNSIGNED_SHORT, "VEC4", ARRAY_BUFFER)
        attributes[f"WEIGHTS_{s}"] = buf.add(weights, FLOAT, "VEC4", ARRAY_BUFFER)
    indices = buf.add(rig.faces.astype(np.uint32).reshape(-1), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    n_joints = len(rig.joints)
    inverse_bind = np.tile(np.eye(4), (n_joints, 1, 1))
    inverse_bind[:, :3, 3] = -rig.joints
    # glTF stores matrices column-major
    inverse_bind = buf.add(inverse_bind.transpose(0, 2, 1).reshape(n_joints, 16).astype(np.float32), FLOAT, "MAT4")

    nodes = [{"name": name, "translation": offset.tolist()} for name, offset in zip(joint_names, rig.offsets)]
    for joint, parent in enumerate(rig.parents):
        if parent >= 0:
            nodes[parent].setdefault("children", []).append(joint)
    mesh_node = len(nodes)
    nodes.append({"name": "body", "mesh": 0, "skin": 0})

    animations = []
    for clip in clips:
        times = buf.add(clip.times.astype(np.float32), FLOAT, "SCALAR", bounds=True)
        quaternions = matrix_to_quaternion(clip.rotations)
        samplers, channels = [], []
        for joint in range(n_joints):
            track = make_continuous(quaternions[:, joint]).astype(np.float32)
            samplers.append({"input": times, "output": buf.add(track, FLOAT, "VEC4"), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": joint, "path": "rotation"}})
        root = (rig.offsets[0] + clip.root_offsets).astype(np.float32)
        samplers.append({"input": times, "output": buf.add(root, FLOAT, "VEC3"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": 0, "path": "translation"}})
        animations.append({"name": clip.name, "samplers": samplers, "channels": channels})

    gltf = {
        "asset": {"version": "2.0", "generator": "photo2avatar"},
        "scene": 0,
        "scenes": [{"nodes": [mesh_node, 0]}],
        "nodes": nodes,
        "meshes": [{"name": "body", "primitives": [
            {"attributes": attributes, "indices": indices, "material": 0, "mode": 4}]}],
        "materials": [{"name": "body_parts", "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.8}}],
        "skins": [{"name": "skeleton", "inverseBindMatrices": inverse_bind,
                   "joints": list(range(n_joints)), "skeleton": 0}],
        "animations": animations,
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    _write(Path(path), gltf, bytes(buf.data))
    return {"bytes": Path(path).stat().st_size, "max_influences_per_vertex": influences,
            "skin_sets": len(sets), "animations": [clip.name for clip in clips]}


def _write(path: Path, gltf: dict, binary: bytes) -> None:
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", GLB_MAGIC, 2, 28 + len(json_chunk) + len(binary)))
        f.write(struct.pack("<I4s", len(json_chunk), CHUNK_JSON) + json_chunk)
        f.write(struct.pack("<I4s", len(binary), CHUNK_BIN) + binary)


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC or version != 2 or total != len(data):
        raise ValueError(f"{path} is not a valid GLB 2.0 file")
    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    bin_length, bin_type = struct.unpack_from("<I4s", data, 20 + json_length)
    if (json_type, bin_type) != (CHUNK_JSON, CHUNK_BIN):
        raise ValueError(f"{path}: unexpected chunk layout")
    start = 28 + json_length
    return json.loads(data[20:20 + json_length]), data[start:start + bin_length]


def read_accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    width = WIDTHS[accessor["type"]]
    array = np.frombuffer(binary, dtype=DTYPES[accessor["componentType"]], count=accessor["count"] * width,
                          offset=view["byteOffset"] + accessor.get("byteOffset", 0))
    return array.reshape(accessor["count"], width) if width > 1 else array
