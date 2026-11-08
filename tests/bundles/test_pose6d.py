"""pose6d bundle: lazy imports, the primitive fast path on ray-cast renders,
symmetry folding, the support plane, the tool boundary, and (with Open3D) the
global FPFH + RANSAC path.

The bundle is out-of-process (``gap.serving``); the numeric tests import
``_impl``, ``_synth`` and ``tools`` through the synthetic package
``gap_skills.tools.pose6d`` the way the bundle itself does.
"""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import numpy as np
import pytest

SKILLS_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = SKILLS_ROOT / "tools" / "pose6d"

_LAZY_ONLY = {"open3d", "scipy"}

BRICK = {"box": [0.0318, 0.0158, 0.0096]}  # LEGO 2x4 body without studs
DOMINO = {"box": [0.048, 0.024, 0.0075]}
PEG = {"cylinder": [0.005, 0.04]}


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("filename", ["tools.py", "_impl.py"])
def test_heavy_imports_are_lazy(filename):
    imported = _module_level_imports(BUNDLE / filename)
    leaked = {
        n for n in imported if any(n == lazy or n.startswith(lazy + ".") for lazy in _LAZY_ONLY)
    }
    assert not leaked, f"{filename}: module-level import of {sorted(leaked)}"


@pytest.fixture(scope="module")
def impl(skills_registry):
    pytest.importorskip("scipy")
    assert skills_registry.get("pose6d").kind == "tool"
    return importlib.import_module("gap_skills.tools.pose6d._impl")


@pytest.fixture(scope="module")
def synth(impl):
    return importlib.import_module("gap_skills.tools.pose6d._synth")


@pytest.fixture(scope="module")
def tools(impl):
    return importlib.import_module("gap_skills.tools.pose6d.tools")


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _pose(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _wrist_cloud(synth, spec, T, eye, seed=0):
    cam = synth.look_at(eye, T[:3, 3])
    depth, mask = synth.render(spec, T, cam, K=synth.K_D405, noise=synth.D405, seed=seed)
    return synth.backproject(depth, mask, synth.K_D405) @ cam[:3, :3].T + cam[:3, 3], cam


def _sym_rot_err(impl, spec, R_est, R_gt):
    group, _ = impl.symmetry_group(impl.parse_model(spec), None)
    return min(impl.rotation_angle_deg(R_est @ S, R_gt) for S in group)


# ---------------------------------------------------------------------------
# Model specs and symmetry groups
# ---------------------------------------------------------------------------


def test_parse_model_accepts_the_documented_specs(impl):
    assert impl.parse_model({"cube": 0.03})["kind"] == "box"
    assert np.allclose(impl.parse_model(BRICK)["dims"], BRICK["box"])
    assert impl.parse_model({"cylinder": {"radius": 0.004, "height": 0.03}})["radius"] == 0.004
    assert impl.parse_model("/tmp/part.stl") == {
        "kind": "mesh",
        "path": "/tmp/part.stl",
        "scale": 1.0,
    }
    assert impl.parse_model({"path": "p.ply", "scale": 0.001})["scale"] == 0.001
    with pytest.raises(ValueError):
        impl.parse_model({"box": [0.01, 0.02]})
    with pytest.raises(ValueError):
        impl.parse_model({"sphere": 0.01})


@pytest.mark.parametrize(
    ("spec", "order", "yaw"),
    [(BRICK, 4, 180.0), ({"box": [0.016, 0.016, 0.0096]}, 8, 90.0), ({"cube": 0.03}, 24, 90.0)],
)
def test_box_symmetry_group_follows_equal_edges(impl, spec, order, yaw):
    group, cont = impl.symmetry_group(impl.parse_model(spec), None)
    assert len(group) == order and not cont
    assert impl.yaw_ambiguity_deg(group, cont) == yaw
    d = impl.parse_model(spec)["dims"]
    assert all(np.allclose(np.abs(S) @ d, d) for S in group)


def test_symmetry_overrides(impl):
    model = impl.parse_model(BRICK)
    assert len(impl.symmetry_group(model, "none")[0]) == 1
    assert len(impl.symmetry_group(model, "z4")[0]) == 4
    group, cont = impl.symmetry_group(impl.parse_model(PEG), None)
    assert cont and impl.yaw_ambiguity_deg(group, cont) == 360.0
    assert len(impl.symmetry_group(model, [np.diag([-1.0, -1.0, 1.0])])[0]) == 2


# ---------------------------------------------------------------------------
# Fast path on noise-free and noisy renders
# ---------------------------------------------------------------------------


def test_noise_free_brick_is_exact(impl, synth):
    T = _pose(_rz(0.7), [0.01, -0.005, 0.0048])
    cam = synth.look_at([0.12, 0.05, 0.14], T[:3, 3])
    depth, mask = synth.render(BRICK, T, cam, K=synth.K_D405)
    P = synth.backproject(depth, mask, synth.K_D405) @ cam[:3, :3].T + cam[:3, 3]
    r = impl.register(P, BRICK, viewpoint=cam[:3, 3], up=[0, 0, 1])
    assert r["status"] == "registered"
    assert _sym_rot_err(impl, BRICK, r["T"][:3, :3], T[:3, :3]) < 0.2
    assert np.linalg.norm(r["T"][:3, 3] - T[:3, 3]) < 2e-4
    assert r["fitness"] > 0.98 and r["coverage"] > 0.9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tilted_brick_from_the_wrist_with_d405_noise(impl, synth, seed):
    rng = np.random.default_rng(seed)
    R = _rz(rng.uniform(-3, 3)) @ _rx(rng.uniform(0.3, 1.0))
    T = _pose(R, [0.0, 0.0, 0.05])
    az = rng.uniform(0, 2 * math.pi)
    P, cam = _wrist_cloud(synth, BRICK, T, [0.12 * math.cos(az), 0.12 * math.sin(az), 0.17], seed)
    r = impl.register(P, BRICK, viewpoint=cam[:3, 3], up=[0, 0, 1])
    assert _sym_rot_err(impl, BRICK, r["T"][:3, :3], T[:3, :3]) < 2.0
    assert np.linalg.norm(r["T"][:3, 3] - T[:3, 3]) < 1e-3


def test_upright_domino_and_lying_peg(impl, synth):
    T = _pose(_rz(0.4) @ np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]), [0.0, 0.0, 0.024])
    P, cam = _wrist_cloud(synth, DOMINO, T, [0.1, -0.1, 0.12])
    r = impl.register(P, DOMINO, viewpoint=cam[:3, 3], up=[0, 0, 1])
    assert _sym_rot_err(impl, DOMINO, r["T"][:3, :3], T[:3, :3]) < 2.0
    T = _pose(_rz(-0.9) @ _rx(math.pi / 2), [0.0, 0.0, 0.005])
    P, cam = _wrist_cloud(synth, PEG, T, [0.0, -0.12, 0.14])
    r = impl.register(P, PEG, viewpoint=cam[:3, 3], up=[0, 0, 1])
    axis_err = math.degrees(math.acos(min(1.0, abs(r["T"][:3, 2] @ T[:3, 2]))))
    assert axis_err < 2.0
    assert np.linalg.norm(r["T"][:3, 3] - T[:3, 3]) < 1.5e-3
    assert r["continuous_symmetry"] and r["equivalent_T"] == []


def test_support_plane_holds_an_overhead_brick_flat(impl, synth):
    """From 0.9 m a 9.6 mm-tall brick has ~200 noisy points; the table pins its tilt."""
    errs = {}
    for support in (None, [0.0, 0.0, 1.0, 0.0]):
        tilt = []
        for seed in range(3):
            T = _pose(_rz(0.5 + seed), [0.01, 0.0, 0.0048])
            cam = synth.look_at([0.0, -0.30, 0.85], [0.0, 0.0, 0.0])
            depth, mask = synth.render(BRICK, T, cam, noise=synth.ORBBEC, seed=seed)
            P = synth.backproject(depth, mask, synth.K_ORBBEC) @ cam[:3, :3].T + cam[:3, 3]
            r = impl.register(P, BRICK, viewpoint=cam[:3, 3], up=[0, 0, 1], support_plane=support)
            tilt.append(math.degrees(math.acos(min(1.0, abs(r["T"][2, 2])))))
        errs[support is None] = max(tilt)
    assert errs[False] < 3.0
    assert errs[False] <= errs[True]


# ---------------------------------------------------------------------------
# Symmetry folding and the reported equivalents
# ---------------------------------------------------------------------------


def test_equivalents_and_canonical_choice(impl, synth):
    T = _pose(_rz(2.8), [0.0, 0.0, 0.0048])
    P, cam = _wrist_cloud(synth, BRICK, T, [0.05, -0.12, 0.15])
    r = impl.register(P, BRICK, viewpoint=cam[:3, 3], up=[0, 0, 1], forward=[1, 0, 0])
    R = r["T"][:3, :3]
    assert R[2, 2] > 0.99  # z up
    assert R[0, 0] > 0  # x along forward
    assert r["symmetry_order"] == 4 and len(r["equivalent_T"]) == 4
    assert r["yaw_ambiguity_deg"] == 180.0
    for E in r["equivalent_T"]:
        assert _sym_rot_err(impl, BRICK, E[:3, :3], T[:3, :3]) < 1.5
    flipped = _pose(R @ _rz(math.pi), r["T"][:3, 3])
    r2 = impl.register(P, BRICK, viewpoint=cam[:3, 3], init=flipped)
    assert impl.rotation_angle_deg(r2["T"][:3, :3], flipped[:3, :3]) < 2.0


def test_symmetry_none_keeps_one_pose(impl, synth):
    T = _pose(_rz(1.0), [0.0, 0.0, 0.0048])
    P, cam = _wrist_cloud(synth, BRICK, T, [0.05, -0.12, 0.15])
    r = impl.register(P, BRICK, viewpoint=cam[:3, 3], up=[0, 0, 1], symmetry="none")
    assert r["symmetry_order"] == 1 and r["yaw_ambiguity_deg"] == 0.0
    assert r["status"] == "registered"  # the geometric twin is not reported as an alternative


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_too_few_points(impl):
    r = impl.register(np.zeros((5, 3)), BRICK)
    assert r["found"] is False and r["status"] == "poor_fit"


def test_wrong_size_is_a_poor_fit(impl, synth):
    T = _pose(_rz(0.3), [0.0, 0.0, 0.015])
    P, cam = _wrist_cloud(synth, {"cube": 0.03}, T, [0.1, -0.1, 0.15])
    r = impl.register(P, {"cube": 0.015}, viewpoint=cam[:3, 3], up=[0, 0, 1])
    assert r["status"] == "poor_fit"


def test_primitive_method_rejects_a_mesh(impl):
    with pytest.raises(ValueError):
        impl.register(np.random.default_rng(0).normal(size=(50, 3)), "x.stl", method="primitive")


# ---------------------------------------------------------------------------
# Tool boundary
# ---------------------------------------------------------------------------


def test_register_mask_returns_a_world_pose(tools, synth, impl):
    T = _pose(_rz(-0.6), [0.02, 0.01, 0.0048])
    cam = synth.look_at([0.1, -0.08, 0.16], T[:3, 3])
    depth, mask = synth.render(BRICK, T, cam, K=synth.K_D405, noise=synth.D405)
    from gap_core.types import matrix_to_pose, pose_to_matrix

    out = tools.register_mask(
        mask=mask.astype(np.uint8) * 255,
        depth=depth,
        intrinsics=synth.K_D405,
        mesh=BRICK,
        camera_pose=matrix_to_pose(cam),
        up=[0.0, 0.0, 1.0],
        support_plane=[0.0, 0.0, 1.0, 0.0],
    )
    assert out["status"] == "registered"
    Tw = pose_to_matrix(out["pose"])
    assert np.allclose(Tw, np.asarray(out["matrix"]))
    assert np.linalg.norm(Tw[:3, 3] - T[:3, 3]) < 1e-3
    assert _sym_rot_err(impl, BRICK, Tw[:3, :3], T[:3, :3]) < 1.0
    assert len(out["equivalent_poses"]) == 4


def test_register_mesh_tool_on_camera_frame_points(tools, synth):
    T = _pose(_rz(0.2), [0.0, 0.0, 0.015])
    cam = synth.look_at([0.0, -0.1, 0.15], T[:3, 3])
    depth, mask = synth.render({"box": [0.03] * 3}, T, cam, K=synth.K_D405, noise=synth.D405)
    P_cam = synth.backproject(depth, mask, synth.K_D405)
    out = tools.register_mesh(points=P_cam, mesh={"cube": 0.03})
    T_cam_obj = np.linalg.inv(cam) @ T
    assert np.linalg.norm(np.asarray(out["matrix"])[:3, 3] - T_cam_obj[:3, 3]) < 1e-3
    assert out["symmetry_order"] == 24


# ---------------------------------------------------------------------------
# Global (Open3D) path
# ---------------------------------------------------------------------------


def test_global_path_on_a_mesh_file(impl, synth, tmp_path):
    o3d = pytest.importorskip("open3d")
    mesh = o3d.geometry.TriangleMesh.create_box(0.0318, 0.0158, 0.0096)
    mesh.translate(-np.array([0.0318, 0.0158, 0.0096]) / 2)
    path = tmp_path / "brick.ply"
    o3d.io.write_triangle_mesh(str(path), mesh)
    T = _pose(_rz(0.9) @ _rx(0.5), [0.0, 0.0, 0.05])
    P, cam = _wrist_cloud(synth, BRICK, T, [0.1, -0.08, 0.17])
    r = impl.register(P, str(path), viewpoint=cam[:3, 3], symmetry="box")
    assert r["method"] == "global_fpfh_icp"
    assert _sym_rot_err(impl, BRICK, r["T"][:3, :3], T[:3, :3]) < 3.0
    assert np.linalg.norm(r["T"][:3, 3] - T[:3, 3]) < 2e-3
