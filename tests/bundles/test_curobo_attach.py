"""curobo held-object planning (``plan_with_grasped_object``) without a GPU.

``_curobo_impl`` imports torch and trimesh at module level; where the test
environment lacks them they are stubbed with empty modules, and every
torch/cuRobo call of the held-object planner goes through ``_attach_ops``,
which the tests replace. What is tested is the planner's own logic: the
held mesh is found and left out of the scene, spheres are attached to the
right link and always detached, a misplaced attachment is caught, and each
failure comes back as an explicit reason with no fallback plan.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_IMPL = Path(__file__).resolve().parents[2] / "tools" / "curobo" / "_curobo_impl.py"

_START = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_TARGET = (np.array([0.35, -0.15, 0.25]), np.array([0.0, 1.0, 0.0, 0.0]))


def _mesh(name, x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(
        name=name,
        pose=[x, y, z, 1.0, 0.0, 0.0, 0.0],
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
    )


def _world(*meshes):
    return SimpleNamespace(mesh=list(meshes))


class FakePlanner:
    tool_frames = ["tcp"]
    joint_names = [f"joint{i}" for i in range(1, 7)]

    def __init__(self):
        self.calls = []

    def reset_seed(self):
        self.calls.append("reset_seed")

    def clear_scene_cache(self):
        self.calls.append("clear_scene_cache")

    def update_world(self, scene_cfg):
        self.calls.append(("update_world", scene_cfg))


_FITTED = np.array([[0.30, 0.10, 0.20, 0.010], [0.31, 0.10, 0.20, 0.010], [0.0, 0.0, 0.0, -100.0]])


class FakeOps:
    """Stands in for ``_AttachOps``; records the order of calls."""

    def __init__(
        self,
        *,
        slots=16,
        fitted=_FITTED,
        readback=None,
        attach_error=None,
        plan_result="result",
        plan_error=None,
        outcome=None,
    ):
        self.slots = slots
        self.fitted = fitted
        self.readback = fitted if readback is None else readback
        self.attach_error = attach_error
        self.plan_result = plan_result
        self.plan_error = plan_error
        self._outcome = (True, np.zeros((5, 6)), "") if outcome is None else outcome
        self.calls = []

    def start_state(self, planner, q):
        self.calls.append(("start_state", tuple(np.round(q, 6))))
        return "start"

    def goal(self, planner, tool_frame, position, quat_wxyz):
        self.calls.append(("goal", tool_frame, tuple(position), tuple(quat_wxyz)))
        return "goal"

    def sphere_slots(self, planner, link_name):
        return self.slots

    def attach(self, planner, start_state, mesh_ns, link_name, num_spheres, surface_radius):
        self.calls.append(("attach", mesh_ns.name, link_name, num_spheres, surface_radius))
        if self.attach_error is not None:
            raise self.attach_error
        return self.fitted

    def link_spheres_world(self, planner, start_state, link_name, n):
        self.calls.append(("readback", link_name, n))
        return self.readback[:n]

    def detach(self, planner, link_name):
        self.calls.append(("detach", link_name))

    def plan(self, planner, goal, start_state, max_attempts):
        self.calls.append(("plan", goal, start_state, max_attempts))
        if self.plan_error is not None:
            raise self.plan_error
        return self.plan_result

    def outcome(self, result, dof):
        return self._outcome

    def names(self):
        return [c[0] if isinstance(c, tuple) else c for c in self.calls]


@pytest.fixture
def impl(monkeypatch):
    """A private copy of ``_curobo_impl`` with cuRobo v0.8 marked available."""
    for name in ("torch", "trimesh"):
        try:
            importlib.import_module(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location("_curobo_impl_under_test", _IMPL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._V2_AVAILABLE = True
    mod._log_curobo_event = lambda *a, **k: None
    return mod


@pytest.fixture
def planner(impl):
    planner = FakePlanner()
    impl.created = []

    def fake_get(robot_file, **kwargs):
        impl.created.append((robot_file, kwargs))
        return planner

    impl._get_attached_planner = fake_get
    impl.scenes = []

    def fake_scene(world_config, device_cfg, ignore):
        impl.scenes.append([m.name for m in world_config.mesh if m.name not in ignore])
        return "scene_cfg"

    impl._v2_scene_cfg_excluding = fake_scene
    planner.config = SimpleNamespace(device_cfg="cuda")
    return planner


def _plan(impl, world, name="cube", **kwargs):
    kwargs.setdefault("robot_file", "yam.yml")
    return impl.plan_with_grasped_object(world, _START, _TARGET, name, **kwargs)


def test_success_attaches_plans_and_detaches(impl, planner):
    ops = impl._attach_ops = FakeOps()
    ok, traj, reason, info = _plan(
        impl, _world(_mesh("table"), _mesh("cube", 0.3, 0.1, 0.2), _mesh("wall"))
    )
    assert ok and reason == ""
    assert traj.shape == (5, 6)
    assert impl.scenes == [["table", "wall"]]
    assert ("update_world", "scene_cfg") in planner.calls
    assert ops.names() == ["start_state", "attach", "readback", "goal", "plan", "detach"]
    assert ops.calls[1] == ("attach", "cube", "attached_object", 8, 0.001)
    assert ops.calls[3][1] == "tcp"
    assert info["n_spheres"] == 2 and info["attach_readback_err_m"] == 0.0
    assert info["tool_frame"] == "tcp"
    assert [r for _, r in impl._last_post_attach_spheres] == [0.01, 0.01]
    robot_file, kwargs = impl.created[0]
    assert robot_file == "yam.yml"
    assert kwargs["with_collision"] is True
    assert kwargs["link_name"] == "attached_object"
    # v0.8 has one position tolerance: the larger of the xy and z allowances.
    assert kwargs["position_threshold"] == pytest.approx(0.05)


def test_effective_position_threshold_takes_the_z_allowance(impl, planner):
    impl._attach_ops = FakeOps()
    _plan(impl, _world(_mesh("cube")), position_threshold=0.01, position_threshold_z=0.03)
    assert impl.created[0][1]["position_threshold"] == pytest.approx(0.03)


@pytest.mark.parametrize("name", ["", "brick"])
def test_missing_object_fails_before_any_planner(impl, planner, name):
    ops = impl._attach_ops = FakeOps()
    ok, traj, reason, _ = _plan(impl, _world(_mesh("table"), _mesh("cube")), name=name)
    assert not ok and traj is None
    assert reason.startswith("object_name_required" if not name else "object_not_in_world")
    assert impl.created == [] and ops.calls == []


def test_robot_without_attach_link_fails(impl, planner):
    def no_link(robot_file, **kwargs):
        raise ValueError("robot 'yam.yml' has no collision link 'attached_object'")

    impl._get_attached_planner = no_link
    impl._attach_ops = FakeOps()
    ok, _, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and reason.startswith("attach_link_missing")


def test_link_without_sphere_slots_fails(impl, planner):
    ops = impl._attach_ops = FakeOps(slots=0)
    ok, _, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and reason.startswith("attach_link_missing")
    assert "attach" not in ops.names()


def test_attach_error_is_a_failure_and_still_detaches(impl, planner):
    ops = impl._attach_ops = FakeOps(attach_error=RuntimeError("Fitted 20 spheres but link has 16"))
    ok, traj, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and traj is None
    assert reason.startswith("attach_failed: RuntimeError")
    assert ops.names()[-1] == "detach" and "plan" not in ops.names()


def test_no_positive_sphere_is_a_failure(impl, planner):
    fitted = _FITTED.copy()
    fitted[:, 3] = -100.0
    ops = impl._attach_ops = FakeOps(fitted=fitted)
    ok, _, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and reason.startswith("attach_failed")
    assert ops.names()[-1] == "detach" and "plan" not in ops.names()


def test_misplaced_attachment_is_caught(impl, planner):
    readback = _FITTED.copy()
    readback[:2, 2] += 0.1347  # spheres written relative to the wrong frame
    ops = impl._attach_ops = FakeOps(readback=readback)
    ok, _, reason, info = _plan(impl, _world(_mesh("cube")))
    assert not ok and reason.startswith("attach_frame_mismatch")
    assert "134.7 mm" in reason
    assert info["attach_readback_err_m"] == pytest.approx(0.1347)
    assert ops.names()[-1] == "detach" and "plan" not in ops.names()


def test_no_ik_is_ik_failed(impl, planner):
    ops = impl._attach_ops = FakeOps(plan_result=None, outcome=(False, None, ""))
    ok, traj, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and traj is None and reason.startswith("ik_failed")
    assert ops.names()[-1] == "detach"


def test_trajopt_failure_reports_the_errors(impl, planner):
    impl._attach_ops = FakeOps(outcome=(False, None, "min position_error=0.0812"))
    ok, traj, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and traj is None
    assert reason == "trajopt_failed: min position_error=0.0812"


@pytest.mark.parametrize(
    ("plan_result", "outcome", "kept"),
    [
        ("result", (True, np.zeros((5, 6)), ""), True),
        (None, (False, None, ""), False),
        ("result", (False, None, "min position_error=0.0"), False),
    ],
)
def test_a_failed_plan_evicts_the_cached_fit(impl, planner, plan_result, outcome, kept):
    impl._attach_ops = FakeOps(plan_result=plan_result, outcome=outcome)
    cube = _mesh("cube")
    key = impl._fit_key(cube, 8, 0.001)
    impl._attach_fit_cache.clear()
    impl._attach_fit_cache[key] = _FITTED
    _plan(impl, _world(cube))
    assert (key in impl._attach_fit_cache) is kept


def test_success_without_trajectory_is_a_failure(impl, planner):
    impl._attach_ops = FakeOps(outcome=(True, None, ""))
    ok, traj, reason, _ = _plan(impl, _world(_mesh("cube")))
    assert not ok and traj is None and reason.startswith("no_trajectory")


def test_planner_exception_propagates_after_detach(impl, planner):
    ops = impl._attach_ops = FakeOps(plan_error=RuntimeError("CUDA error: illegal memory access"))
    with pytest.raises(RuntimeError, match="CUDA"):
        _plan(impl, _world(_mesh("cube")))
    assert ops.names()[-1] == "detach"


def test_without_world_collision_no_scene_is_loaded(impl, planner):
    impl._attach_ops = FakeOps()
    ok, *_ = _plan(impl, _world(_mesh("table"), _mesh("cube")), use_world_collision=False)
    assert ok
    assert impl.created[0][1]["with_collision"] is False
    assert impl.scenes == [] and "clear_scene_cache" not in planner.calls


def test_held_object_alone_needs_no_scene(impl, planner):
    impl._attach_ops = FakeOps()
    ok, *_ = _plan(impl, _world(_mesh("cube")))
    assert ok and impl.created[0][1]["with_collision"] is False


def test_requires_curobo_v08(impl):
    impl._V2_AVAILABLE = False
    with pytest.raises(RuntimeError, match="v0.8"):
        _plan(impl, _world(_mesh("cube")))


def _franka_like():
    return {
        "robot_cfg": {
            "kinematics": {
                "collision_link_names": ["panda_hand", "attached_object"],
                "extra_collision_spheres": {"attached_object": 4},
                "extra_links": {"attached_object": {"parent_link_name": "panda_hand"}},
                "collision_sphere_buffer": 0.0,
            }
        }
    }


def test_robot_dict_reserves_sphere_slots_and_buffer(impl, monkeypatch):
    monkeypatch.setattr(impl, "_load_robot_dict", lambda robot_file: _franka_like())
    kin = impl._attached_robot_dict("franka.yml", "attached_object", 16, -0.01)["robot_cfg"][
        "kinematics"
    ]
    assert kin["extra_collision_spheres"] == {"attached_object": 16}
    assert kin["collision_sphere_buffer"] == pytest.approx(-0.01)

    big = _franka_like()
    big["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = 40
    monkeypatch.setattr(impl, "_load_robot_dict", lambda robot_file: big)
    kin = impl._attached_robot_dict("franka.yml", "attached_object", 16, None)["robot_cfg"][
        "kinematics"
    ]
    assert kin["extra_collision_spheres"]["attached_object"] == 40
    assert kin["collision_sphere_buffer"] == 0.0


def test_robot_dict_rejects_unknown_link(impl, monkeypatch):
    monkeypatch.setattr(impl, "_load_robot_dict", lambda robot_file: _franka_like())
    with pytest.raises(ValueError, match="no collision link 'gripper_tip'"):
        impl._attached_robot_dict("franka.yml", "gripper_tip", 16, None)


def test_fewer_slots_than_requested_spheres_caps_the_fit(impl, planner):
    ops = impl._attach_ops = FakeOps(slots=4)
    _plan(impl, _world(_mesh("cube")))
    assert ops.calls[1][3] == 4


def test_enclosing_spheres_cover_the_surface(impl):
    g = np.linspace(-0.015, 0.015, 7)
    surface = np.array([[x, y, z] for x in g for y in g for z in g if 0.015 in np.abs([x, y, z])])
    spheres = np.array([[-0.0075, 0.0, 0.0, 0.012], [0.0075, 0.0, 0.0, 0.012], [0, 0, 0, -100.0]])
    out = impl._enclosing_spheres(spheres, surface)
    gap = np.min(
        np.linalg.norm(surface[:, None] - out[None, :2, :3], axis=-1) - out[None, :2, 3], axis=1
    )
    assert gap.max() <= 1e-12
    assert out[0, 3] == out[1, 3] > 0.012
    assert out[2, 3] == -100.0
    np.testing.assert_array_equal(out[:, :3], spheres[:, :3])
    # Spheres that already enclose the surface are left alone.
    big = np.array([[0.0, 0.0, 0.0, 0.03]])
    np.testing.assert_array_equal(impl._enclosing_spheres(big, surface), big)


def test_spheres_posed_into_the_world(impl):
    local = np.array([[0.01, 0.0, 0.0, 0.005], [0.0, 0.0, 0.0, -100.0]])
    # 90 deg about z, then translate.
    pose = [0.3, 0.1, 0.2, np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    world = impl._spheres_to_world(local, pose)
    np.testing.assert_allclose(world[0], [0.3, 0.11, 0.2, 0.005], atol=1e-12)
    np.testing.assert_allclose(world[1, :3], [0.3, 0.1, 0.2])
    assert world[1, 3] == -100.0


def test_attachment_manager_found_on_the_solver_core(impl):
    manager = object()

    class Planner:
        @property
        def attachment_manager(self):
            return self.trajopt_solver.attachment_manager

        trajopt_solver = SimpleNamespace(core=SimpleNamespace(attachment_manager=manager))

    assert impl._AttachOps.manager(Planner()) is manager
    direct = SimpleNamespace(attachment_manager="m")
    assert impl._AttachOps.manager(direct) == "m"


def test_attached_planner_cache_invalidates(impl):
    impl._attached_planner_cache[("k",)] = object()
    impl._attached_planner_cache.invalidate()
    assert len(impl._attached_planner_cache) == 0


# --- the tool boundary -------------------------------------------------------


def _tool_world():
    tri = {"vertices": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]]), "pose": None}
    return {"meshes": [{"name": "cube", **tri}, {"name": "table", **tri}]}


def _tool_pose():
    return {
        "position": {"x": 0.35, "y": -0.15, "z": 0.25},
        "rotation": {"w": 0, "x": 1, "y": 0, "z": 0},
    }


def test_tool_schema_reports_failure_reason(tool_registry):
    schema = tool_registry.get("curobo.plan_with_grasped_object").schema
    assert set(schema.outputs) == {"success", "trajectory", "failure_reason"}
    assert schema.inputs["link_name"].default == "attached_object"
    assert schema.inputs["remove_obstacles_from_world"].default is False


def test_tool_returns_the_failure_reason(tool_registry, skills_registry, monkeypatch):
    mod = skills_registry.get("curobo").tools_module
    seen = {}

    def plan(world, q, target, name, **kwargs):
        seen.update(world=[m.name for m in world.mesh], q=q, target=target, name=name, **kwargs)
        return False, None, "ik_failed: no collision-free IK solution", {}

    monkeypatch.setattr(mod, "_impl", lambda: SimpleNamespace(plan_with_grasped_object=plan))
    out = tool_registry.invoke(
        "curobo.plan_with_grasped_object",
        world_config=_tool_world(),
        start_joint_position={"positions": _START},
        target_pose=_tool_pose(),
        object_name="cube",
        robot_file="yam.yml",
    )
    assert out == {
        "success": False,
        "trajectory": None,
        "failure_reason": "ik_failed: no collision-free IK solution",
    }
    assert seen["world"] == ["cube", "table"] and seen["name"] == "cube"
    assert seen["robot_file"] == "yam.yml"
    np.testing.assert_allclose(seen["target"][1], [0, 1, 0, 0])


def test_tool_returns_the_trajectory(tool_registry, skills_registry, monkeypatch):
    mod = skills_registry.get("curobo").tools_module
    traj = np.arange(12, dtype=np.float64).reshape(2, 6)
    monkeypatch.setattr(
        mod,
        "_impl",
        lambda: SimpleNamespace(plan_with_grasped_object=lambda *a, **k: (True, traj, "", {})),
    )
    out = tool_registry.invoke(
        "curobo.plan_with_grasped_object",
        world_config=_tool_world(),
        start_joint_position={"positions": _START},
        target_pose=_tool_pose(),
        object_name="cube",
    )
    assert out["success"] and out["failure_reason"] == ""
    assert len(out["trajectory"]["waypoints"]) == 2


def test_tool_raises_planning_failed_on_infrastructure_errors(
    tool_registry, skills_registry, monkeypatch
):
    from gap_core.errors import PlanningFailed

    mod = skills_registry.get("curobo").tools_module

    def boom(*a, **k):
        raise RuntimeError("plan_with_grasped_object: curobo v0.8 (cuRoboV2) is required")

    monkeypatch.setattr(mod, "_impl", lambda: SimpleNamespace(plan_with_grasped_object=boom))
    with pytest.raises(PlanningFailed, match="v0.8"):
        tool_registry.invoke(
            "curobo.plan_with_grasped_object",
            world_config=_tool_world(),
            start_joint_position={"positions": _START},
            target_pose=_tool_pose(),
            object_name="cube",
        )
