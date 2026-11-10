from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh

from photo2avatar import animation, body
from photo2avatar.gltf_export import read_accessor, read_glb, write_glb
from photo2avatar.rotations import matrix_to_quaternion, quaternion_to_matrix, rot_x, rot_y, rot_z

RNG = np.random.default_rng(0)
SMPL = body.load_smpl()
BETAS = RNG.normal(0, 1, 10)
RIG = body.build_rig(SMPL, BETAS)


def random_pose(max_angle: float = 0.5) -> np.ndarray:
    axes = RNG.normal(size=(24, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = RNG.uniform(-max_angle, max_angle, 24)[:, None, None]
    K = np.zeros((24, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    K -= K.transpose(0, 2, 1)
    return np.eye(3) + np.sin(angles) * K + (1 - np.cos(angles)) * K @ K


def write_and_read(clips: list[animation.Clip]):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "avatar.glb"
        stats = write_glb(path, RIG, RIG.part_colors(), clips, body.JOINT_NAMES)
        gltf, binary = read_glb(path)
        mesh = trimesh.load(path, force="mesh")
    return stats, gltf, binary, mesh


def test_rest_pose_is_identity():
    vertices, joints = RIG.pose(np.tile(np.eye(3), (24, 1, 1)))
    assert np.allclose(vertices, RIG.vertices, atol=1e-9)
    assert np.allclose(joints, RIG.joints, atol=1e-9)


def test_rig_matches_smpl_skinning():
    check = body.check_rig(RIG, SMPL, BETAS, random_pose())
    assert check["rig_vs_smpl_lbs_max_mm"] < 0.01, check


def test_segmentation_covers_every_vertex_and_is_symmetric():
    labels = RIG.part_labels()
    assert labels.shape == (6890,)
    assert set(np.unique(labels)) == set(range(len(body.PARTS)))
    counts = np.bincount(labels, minlength=len(body.PARTS))
    for part in body.PARTS:
        if part.startswith("left_"):
            left = counts[body.PARTS.index(part)]
            right = counts[body.PARTS.index(part.replace("left_", "right_"))]
            assert abs(left - right) / left < 0.05, (part, left, right)


def test_rotations():
    R = random_pose(np.pi)
    assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(R)), R, atol=1e-9)
    assert np.allclose(rot_x(np.pi / 2) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0])


def test_looping_clips_are_seamless():
    for clip in (animation.walk_clip(), animation.wave_clip()):
        assert np.all(np.diff(clip.times) > 0)
        assert np.allclose(clip.rotations[0], clip.rotations[-1], atol=1e-9), clip.name
        assert np.allclose(clip.root_offsets[0], clip.root_offsets[-1], atol=1e-9), clip.name


def test_walk_moves_the_feet_in_antiphase():
    walk = animation.walk_clip()
    ankle_z = np.array([RIG.pose(r, t)[1][[7, 8], 2] for r, t in zip(walk.rotations, walk.root_offsets)])
    assert np.corrcoef(ankle_z[:, 0], ankle_z[:, 1])[0, 1] < -0.9


def test_glb_round_trip():
    clips = [animation.photo_pose_clip(random_pose()), animation.walk_clip(), animation.wave_clip()]
    stats, gltf, binary, mesh = write_and_read(clips)
    assert len(mesh.vertices) == 6890 and len(mesh.faces) == 13776

    for accessor in gltf["accessors"]:
        view = gltf["bufferViews"][accessor["bufferView"]]
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(binary)

    attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert np.allclose(read_accessor(gltf, binary, attributes["POSITION"]), RIG.vertices, atol=1e-6)
    normals = read_accessor(gltf, binary, attributes["NORMAL"])
    assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)

    weights = np.zeros_like(RIG.weights)
    rows = np.arange(len(weights))[:, None]
    for s in range(stats["skin_sets"]):
        joints = read_accessor(gltf, binary, attributes[f"JOINTS_{s}"]).astype(int)
        np.add.at(weights, (rows, joints), read_accessor(gltf, binary, attributes[f"WEIGHTS_{s}"]))
    assert np.allclose(weights, RIG.weights, atol=1e-5)

    ibm = read_accessor(gltf, binary, gltf["skins"][0]["inverseBindMatrices"]).reshape(24, 4, 4).transpose(0, 2, 1)
    rest = np.tile(np.eye(4), (24, 1, 1))
    rest[:, :3, 3] = RIG.joints
    assert np.allclose(ibm @ rest, np.eye(4), atol=1e-5)

    assert [a["name"] for a in gltf["animations"]] == ["photo_pose", "walk", "wave"]
    for anim in gltf["animations"]:
        for sampler in anim["samplers"]:
            assert np.all(np.diff(read_accessor(gltf, binary, sampler["input"])) > 0)
            values = read_accessor(gltf, binary, sampler["output"])
            if values.shape[1] == 4:
                assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
                assert np.all((values[1:] * values[:-1]).sum(axis=1) >= 0)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
