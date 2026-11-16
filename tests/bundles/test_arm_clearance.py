"""arm-clearance: the chain walk, the local IK, the capsule-to-box distance and the step rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

BUNDLE = "arm-clearance"


@pytest.fixture(scope="module")
def t(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "tool"
    return bundle.tools_module


def _arm(base=None, capsules=True):
    """A 6-joint arm: a turntable, shoulder and elbow about y, and a roll-pitch-roll wrist."""
    chain = [
        {"name": "base", "parent": None},
        {"name": "l1", "parent": "base", "position": [0, 0, 0.10], "joint": 0, "axis": [0, 0, 1]},
        {"name": "l2", "parent": "l1", "position": [0, 0, 0.05], "joint": 1, "axis": [0, 1, 0]},
        {"name": "l3", "parent": "l2", "position": [0.25, 0, 0], "joint": 2, "axis": [0, 1, 0]},
        {"name": "l4", "parent": "l3", "position": [0.20, 0, 0], "joint": 3, "axis": [1, 0, 0]},
        {"name": "l5", "parent": "l4", "position": [0.05, 0, 0], "joint": 4, "axis": [0, 1, 0]},
        {"name": "l6", "parent": "l5", "position": [0.05, 0, 0], "joint": 5, "axis": [1, 0, 0]},
    ]
    model = {
        "chain": chain,
        "tcp": {"body": "l6", "position": [0.08, 0, 0]},
        "limits": [[-2.8, 2.8]] * 6,
    }
    if base is not None:
        model["base"] = base
    if capsules:
        model["capsules"] = {
            "l2": [[[0, 0, 0], [0.25, 0, 0], 0.03]],
            "l3": [[[0, 0, 0], [0.20, 0, 0], 0.03]],
            "l6": [[[0.04, 0, 0], [0.04, 0, 0], 0.04]],  # a sphere: the hand
        }
    return model


def _pose(frame):
    w = math.sqrt(max(0.0, 1 + np.trace(frame[:3, :3]))) / 2
    r = frame[:3, :3]
    x, y, z = (
        (r[2, 1] - r[1, 2]) / (4 * w),
        (r[0, 2] - r[2, 0]) / (4 * w),
        (r[1, 0] - r[0, 1]) / (4 * w),
    )
    p = frame[:3, 3]
    return {
        "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
        "rotation": {"w": w, "x": float(x), "y": float(y), "z": float(z)},
    }


def test_the_chain_is_walked_as_mjcf_does(t):
    arm = t.Arm(_arm())
    straight = arm.tool(np.zeros(6))
    assert np.allclose(straight[:3, 3], [0.63, 0, 0.15]) and np.allclose(
        straight[:3, :3], np.eye(3)
    )
    up = arm.tool([0, -math.pi / 2, 0, 0, 0, 0])  # the shoulder turns the arm straight up
    assert np.allclose(up[:3, 3], [0, 0, 0.15 + 0.63], atol=1e-9)
    side = arm.tool([math.pi / 2, 0, 0, 0, 0, 0])
    assert np.allclose(side[:3, 3], [0, 0.63, 0.15], atol=1e-9)
    moved = t.Arm(
        _arm(
            base={
                "position": {"x": 1.0, "y": 2.0, "z": 0.5},
                "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
            }
        )
    )
    assert np.allclose(moved.tool(np.zeros(6))[:3, 3], [1.63, 2.0, 0.65])


def test_the_jacobian_matches_finite_differences(t):
    arm = t.Arm(_arm())
    q = np.array([0.3, -0.7, 0.9, 0.4, -0.5, 0.2])
    frame, jac = arm.jacobian(q)
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = 1e-6
        moved = arm.tool(q + dq)
        assert np.allclose((moved[:3, 3] - frame[:3, 3]) / 1e-6, jac[:3, j], atol=1e-5)
        assert np.allclose(t._rotvec(moved[:3, :3] @ frame[:3, :3].T) / 1e-6, jac[3:, j], atol=1e-5)


def test_rotation_vectors_round_trip_up_to_a_half_turn(t):
    for axis, angle in (([0, 0, 1], 0.3), ([1, 1, 0], 2.5), ([0, 1, 0], math.pi), ([1, 0, 0], 0.0)):
        a = np.array(axis, float) / np.linalg.norm(axis)
        got = t._rotvec(t._axis_angle(a, angle))
        assert np.allclose(got, a * angle, atol=1e-6) or np.allclose(got, -a * angle, atol=1e-6)


def test_ik_recovers_poses_the_arm_can_take(t):
    model, arm = _arm(), t.Arm(_arm())
    rng = np.random.default_rng(4)
    solved = 0
    for _ in range(25):
        q_true = rng.uniform(-1.2, 1.2, 6)
        out = t.solve_ik(model, _pose(arm.tool(q_true)), list(q_true + rng.uniform(-0.3, 0.3, 6)))
        if out["ok"]:
            solved += 1
            got = arm.tool(out["q"])
            want = arm.tool(q_true)
            assert np.linalg.norm(got[:3, 3] - want[:3, 3]) <= 1e-3
            assert np.linalg.norm(t._rotvec(got[:3, :3] @ want[:3, :3].T)) <= 0.02
            assert np.all(np.abs(out["q"]) <= 2.8)
    assert solved == 25


def test_ik_says_no_out_of_reach_outside_the_box_and_at_a_limit(t):
    model, arm = _arm(), t.Arm(_arm())
    far = _pose(arm.tool(np.zeros(6)))
    far["position"]["x"] = 1.5
    out = t.solve_ik(model, far, [0.0] * 6)
    assert not out["ok"] and "no solution" in out["reason"] and out["position_error_m"] > 0.5
    pose = _pose(arm.tool([0.2, -0.9, 0.3, 0, 0.3, 0]))
    assert pose["position"]["z"] > 0.3
    box = {"min": [-1, -1, 0.0], "max": [1, 1, 0.2]}
    boxed = t.solve_ik(model, pose, [0.0] * 6, bounds=box)
    assert not boxed["ok"] and "workspace box" in boxed["reason"]
    assert t.solve_ik(model, pose, [0.0] * 6, bounds={"min": [-1, -1, 0], "max": [1, 1, 1]})["ok"]
    tight = dict(model, limits=[[-0.1, 0.1]] + [[-2.8, 2.8]] * 5)
    turned = _pose(arm.tool([1.2, -0.5, 0.8, 0, 0.3, 0]))
    assert not t.solve_ik(tight, turned, [0.0] * 6)["ok"]


def test_clearance_is_the_gap_between_capsule_and_box(t):
    model = _arm()
    # the arm lies straight along x at z = 0.15; a shelf's underside is at z = 0.25
    shelf = {"name": "shelf", "center": [0.2, 0.0, 0.30], "half": [0.2, 0.2, 0.05]}
    out = t.clearance(model, [0.0] * 6, [shelf])
    assert out["clearance_m"] == pytest.approx(0.25 - 0.15 - 0.03, abs=1e-6)
    assert out["link"] in ("l2", "l3") and out["obstacle"] == "shelf"
    assert out["nearest"].endswith("to shelf")
    # the hand's sphere (radius 0.04 at x = 0.59) against a wall whose face is at x = 0.66
    wall = {"name": "wall", "center": [0.76, 0, 0.15], "half": [0.10, 0.5, 0.5]}
    assert t.clearance(model, [0.0] * 6, [wall])["clearance_m"] == pytest.approx(0.03, abs=1e-6)
    assert t.clearance(model, [0.0] * 6, [wall])["link"] == "l6"
    # nothing within the limit, and a box the arm is inside
    away = {"center": [0.2, 0.0, 1.0], "half": [0.1, 0.1, 0.1]}
    none = t.clearance(model, [0.0] * 6, [away])
    assert none["clearance_m"] == 0.15 and none["link"] == "" and none["nearest"] == ""
    through = {"center": [0.3, 0.0, 0.15], "half": [0.05, 0.2, 0.2]}
    assert t.clearance(model, [0.0] * 6, [through])["clearance_m"] < -0.03


def test_a_turned_box_and_an_obb_are_read_alike(t):
    h = math.radians(45.0) / 2
    turned = {
        "center": [0.2, 0.0, 0.40],
        "half": [0.1, 0.1, 0.1],
        "quat": [math.cos(h), 0, 0, math.sin(h)],
    }
    as_obb = {
        "center": {"x": 0.2, "y": 0.0, "z": 0.40},
        "extent": {"x": 0.1, "y": 0.1, "z": 0.1},
        "orientation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)},
    }
    a = t.clearance(_arm(), [0.0] * 6, [turned])["clearance_m"]
    assert a == pytest.approx(0.30 - 0.15 - 0.03, abs=1e-6)  # a turn about z leaves the underside
    assert t.clearance(_arm(), [0.0] * 6, [as_obb])["clearance_m"] == pytest.approx(a)


def test_a_step_is_refused_towards_the_scene_and_allowed_away_from_it(t):
    model = _arm()
    shelf = [{"name": "shelf", "center": [0.2, 0.0, 0.30], "half": [0.2, 0.2, 0.05]}]
    level, raised, lowered = [0.0] * 6, [0, -0.25, 0, 0, 0, 0], [0, 0.4, 0, 0, 0, 0]
    up = t.check_step(model, level, raised, shelf, margin_m=0.10)
    assert up["blocked"] and "shelf" in up["reason"] and "margin 10 cm" in up["reason"]
    assert up["clearance_now_m"] == pytest.approx(0.07, abs=1e-6) and up["clearance_m"] < 0.07
    # already inside the margin, but moving away: allowed
    down = t.check_step(model, level, lowered, shelf, margin_m=0.10)
    assert not down["blocked"] and down["reason"] == ""
    # the same upward step with a margin it keeps
    assert not t.check_step(model, lowered, level, shelf, margin_m=0.05)["blocked"]


def test_reachable_joins_the_two(t):
    model, arm = _arm(), t.Arm(_arm())
    shelf = [{"name": "shelf", "center": [0.2, 0.0, 0.30], "half": [0.2, 0.2, 0.05]}]
    q_now = [0, 0.4, 0.2, 0, 0, 0]
    low = _pose(arm.tool([0.1, 0.5, 0.1, 0, 0.1, 0]))
    ok = t.reachable(model, low, q_now, shelf, margin_m=0.03)
    assert ok["reachable"] and len(ok["q"]) == 6 and ok["reason"] == ""
    high = _pose(arm.tool([0, -0.3, 0.1, 0, 0, 0]))
    hit = t.reachable(model, high, q_now, shelf, margin_m=0.03)
    assert not hit["reachable"] and "shelf" in hit["reason"] and len(hit["q"]) == 6
    assert t.reachable(model, high, q_now)["reachable"]  # no scene: only the reach
    far = dict(high, position={"x": 2.0, "y": 0.0, "z": 0.2})
    none = t.reachable(model, far, q_now, shelf)
    assert not none["reachable"] and none["q"] == [] and "no solution" in none["reason"]


def test_a_malformed_model_is_refused(t):
    bad = _arm()
    bad["chain"][2]["parent"] = "l6"
    with pytest.raises(ValueError, match="after its parent"):
        t.Arm(bad)
    with pytest.raises(ValueError, match="limits"):
        t.Arm(dict(_arm(), limits=[[-1, 1]] * 5))
    with pytest.raises(ValueError, match="capsules"):
        t.Arm(dict(_arm(), capsules={"nope": []}))


def test_the_tools_are_registered_with_schemas(skills_registry, tool_registry):
    declared = set(skills_registry.get(BUNDLE).meta.tools)
    assert declared == {
        f"arm-clearance.{n}" for n in ("solve_ik", "clearance", "check_step", "reachable")
    }
    registered = tool_registry.runtime_tools()
    for name in declared:
        assert name in registered
