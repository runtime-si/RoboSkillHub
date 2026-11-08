"""straightening-a-cable-toward-a-target-curve: FakeContext tests of the matching, the pick, the place and the routing."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "straightening-a-cable-toward-a-target-curve"
REACH = 0.0102
SPAN = 0.094
R = 0.00175  # a 3.5 mm cable
CAMERA = np.array(
    [[1.0, 0.0, 0.0, 0.40], [0.0, -1.0, 0.0, 0.20], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
)  # overhead, 1 m up, looking down


@pytest.fixture(scope="module")
def script(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["straighten_cable"].module


def line(x0, y0, x1, y1, n=31, z=R):
    return [[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z] for t in np.linspace(0.0, 1.0, n)]


def bent(bump=0.03, n=31):
    """A 0.30 m cable along +y at x = 0.40 with a smooth sideways bump in the middle."""
    ys = np.linspace(0.05, 0.35, n)
    xs = 0.40 + bump * np.exp(-(((ys - 0.20) / 0.04) ** 2))
    return [[float(x), float(y), R] for x, y in zip(xs, ys, strict=True)]


STRAIGHT = line(0.40, 0.05, 0.40, 0.35)


def _to_polyline(points: np.ndarray, p: np.ndarray) -> float:
    """Plan distance from p to the polyline."""
    a, b = points[:-1, :2], points[1:, :2]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p[:2] - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p[:2], axis=1)))


def down(a_deg: float) -> dict[str, float]:
    a = math.radians(a_deg)
    return {"w": 0.0, "x": math.cos(a), "y": math.sin(a), "z": 0.0}


class World:
    """A top-down hand over a table at z = 0 and a cable of nodes lying on it.

    Moves land where commanded except that the fingertips (REACH under the tool
    centre) stop on the table. A close with the fingertips at the table and a
    cable node within 4 mm of the tool centre in plan holds the cable there;
    the release moves that node by how far the hand went and its neighbours
    with a Gaussian along the cable (``drag`` its width, 0 = the node alone).
    The overhead camera's fit returns the nodes (``fit_offset`` added, as a
    perception error; ``hidden`` empties the mask)."""

    def __init__(self, cable, drag=0.04, refuse=None, ik=None, fit_offset=(0.0, 0.0), stuck=False):
        self.cable = np.asarray(cable, dtype=float)
        self.p = np.array([0.30, 0.0, 0.25])
        self.a = 0.0
        self.width = SPAN
        self.held: np.ndarray | None = None
        self.held_node = -1
        self.drag = drag
        self.refuse = refuse or (lambda tool, pose: None)
        self.ik = ik or (lambda pose: {"success": True, "within_servo_tolerance": True})
        self.fit_offset = np.array([*fit_offset, 0.0])
        self.stuck = stuck
        self.hidden = False
        self.moves: list[tuple[str, np.ndarray]] = []
        self.grips: list[object] = []
        self.fits = 0

    def _move(self, tool):
        def go(pose, **kwargs):
            target = np.array([pose["position"][k] for k in "xyz"])
            self.moves.append((tool, target.copy()))
            why = self.refuse(tool, pose)
            if why:
                if why == "raise":
                    raise RuntimeError("motion server refused the move")
                return {"reached": False, "outcome": "unreachable", "reason": why}
            stopped = target[2] < REACH - 1e-9
            self.p = np.array([target[0], target[1], max(target[2], REACH)])
            self.a = math.degrees(math.atan2(pose["rotation"]["y"], pose["rotation"]["x"]))
            if stopped and tool == "robot.move_cartesian_until_contact":
                return {"reached": False, "outcome": "contact"}
            if stopped:
                return {"reached": False, "outcome": "timeout"}
            return {"reached": True, "outcome": "reached"}

        return go

    def _close(self, **kwargs):
        self.grips.append("close")
        d = np.linalg.norm(self.cable[:, :2] - self.p[:2], axis=1)
        if self.p[2] < REACH + 0.003 and _to_polyline(self.cable, self.p) < 0.004:
            self.held = self.p[:2].copy()
            self.held_node = int(np.argmin(d))
            self.width = 0.0054
        else:
            self.width = 0.0
        return {"position": self.width / SPAN}

    def _open(self, **kwargs):
        self.grips.append("open")
        if self.held is not None and not self.stuck:
            shift = self.p[:2] - self.held
            arcs = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(self.cable[:, :2], axis=0), axis=1))]
            )
            ds = arcs - arcs[self.held_node]
            w = np.exp(-((ds / self.drag) ** 2)) if self.drag > 0 else (ds == 0).astype(float)
            self.cable[:, :2] += w[:, None] * shift
        self.held = None
        self.width = SPAN
        return {"position": 1.0}

    def _fit(self, **kwargs):
        self.fits += 1
        pts = self.cable + self.fit_offset
        return {
            "points": pts.tolist(),
            "radius_m": R,
            "arclength_m": 0.0,
            "nodes": len(pts),
            "ordered": True,
        }

    def ctx(self, **overrides):
        tools = {
            "robot.get_ee_pose": lambda **kw: {
                "pose": {
                    "position": dict(zip("xyz", map(float, self.p), strict=True)),
                    "rotation": down(self.a),
                }
            },
            "robot.go_to_pose": self._move("robot.go_to_pose"),
            "robot.go_to_pose_cartesian": self._move("robot.go_to_pose_cartesian"),
            "robot.move_cartesian_until_contact": self._move("robot.move_cartesian_until_contact"),
            "robot.solve_ik": lambda pose, **kw: self.ik(pose),
            "robot.grasp_frame": lambda close_heading_deg, **kw: {
                "rotation": down(close_heading_deg / 2.0),
                "rotation_flipped": down(close_heading_deg / 2.0 + 90.0),
            },
            "robot.describe_gripper": {
                "span_m": SPAN,
                "width_fit": {"at_open_m": SPAN, "at_closed_m": 0.0},
                "finger": {"reach_m": REACH},
            },
            "robot.describe_workspace": {"surface_z": 0.0},
            "robot.open_gripper": self._open,
            "robot.close_gripper": self._close,
            "robot.get_gripper": lambda **kw: {"position": self.width / SPAN},
            "robot.set_grip": lambda width_m, **kw: self.grips.append(width_m) or {"ok": True},
            "robot.get_observation": lambda **kw: {
                "arms": [{"joint_state": {"positions": [0.0] * 6}}],
                "cameras": [
                    {
                        "name": "overhead",
                        "rgb": "o",
                        "depth": np.zeros((2, 2)),
                        "intrinsics": np.eye(3),
                        "pose": CAMERA,
                    }
                ],
            },
            "sam3.segment_text": lambda image, **kw: {
                "masks": [] if self.hidden else [np.ones((2, 2))]
            },
            "curve.fit_centerline": self._fit,
        }
        tools.update({k.replace("__", "."): v for k, v in overrides.items()})
        return FakeContext(tools)


def run(script, world, **kwargs):
    kwargs.setdefault("target", STRAIGHT)
    kwargs.setdefault("table_z", 0.0)
    kwargs.setdefault("min_tool_height", 0.005)
    kwargs.setdefault("lift_check", False)
    ctx = world.ctx()
    return script.run(ctx, **kwargs), ctx


# --- the matching (upstream's ordering search) ---------------------------------------


def test_match_finds_the_reversed_ordering(script):
    m = script.match_curves(STRAIGHT[::-1], STRAIGHT)
    assert m["reversed"] is True
    assert m["max_error"] == pytest.approx(0.0, abs=1e-9)
    assert m["coverage"] == pytest.approx(1.0)


def test_match_resamples_at_equal_plan_arc_length(script):
    # Unevenly spaced nodes, and z jitter that would lengthen a 3D arc.
    ys = np.array([0.05, 0.06, 0.20, 0.21, 0.35])
    cable = [[0.40, y, R + (0.004 if k % 2 else 0.0)] for k, y in enumerate(ys)]
    m = script.match_curves(cable, STRAIGHT, spacing=0.01)
    assert m["cable_length"] == pytest.approx(0.30)
    assert len(m["errors"]) == 31
    assert m["max_error"] == pytest.approx(0.0, abs=1e-9)


def test_match_slides_a_short_target_along_the_cable(script):
    target = line(0.40, 0.15, 0.40, 0.25, n=2)  # a 10 cm segment over the cable's middle
    slide = script.match_curves(STRAIGHT, target, mode="slide")
    assert slide["max_error"] == pytest.approx(0.0, abs=1e-9)
    assert slide["offset_m"] == pytest.approx(0.10)
    assert slide["coverage"] == pytest.approx(1.0)
    ends = script.match_curves(STRAIGHT, target, mode="ends", along_slack=0.0)
    assert ends["max_error"] == pytest.approx(0.10, abs=1e-6)  # an end of the cable on an end of it


def test_along_slack_measures_across_a_cable_crept_along_its_line(script):
    # On the line but 2 cm further along it: by order every node is 2 cm off;
    # within the slack only the overhanging end is.
    crept = line(0.40, 0.07, 0.40, 0.37)
    strict = script.match_curves(crept, STRAIGHT, along_slack=0.0)
    assert strict["mean_error"] == pytest.approx(0.02, abs=1e-6)
    m = script.match_curves(crept, STRAIGHT)
    assert m["mean_error"] < 0.002
    assert m["max_error"] == pytest.approx(0.02, abs=1e-6)  # the end past the target's end
    assert m["worst"] in (0, len(m["errors"]) - 1)
    # A sideways bump is still measured across, and its target is square to the line.
    b = script.match_curves(bent(0.03), STRAIGHT)
    top = b["worst"]
    assert b["target_nodes"][top][1] == pytest.approx(b["cable_nodes"][top][1], abs=1e-3)


def test_match_keeps_the_least_mean_placement_and_its_worst_node(script):
    m = script.match_curves(bent(0.03), STRAIGHT)
    assert not m["reversed"]
    assert m["worst"] == 15  # the bump's top, at mid-length
    assert m["max_error"] == pytest.approx(0.03, abs=1e-3)
    assert m["mean_error"] < m["max_error"]


def test_a_shorter_cable_reports_its_coverage(script):
    m = script.match_curves(line(0.40, 0.05, 0.40, 0.20), STRAIGHT)
    assert m["coverage"] == pytest.approx(0.5)
    assert m["max_error"] == pytest.approx(0.0, abs=1e-9)


def test_place_past_follows_params_no_rots(script):
    # Upstream: new_p1 = p0 - direction * (length + overshoot), direction = p0 - p1 normalised.
    place = script.place_past(np.array([0.0, 0.0, 0.002]), np.array([0.10, 0.0, 0.0]), 0.02)
    np.testing.assert_allclose(place, [0.12, 0.0, 0.002])
    same = script.place_past(np.array([0.1, 0.1, 0.0]), np.array([0.1, 0.1, 0.0]), 0.02)
    np.testing.assert_allclose(same, [0.1, 0.1, 0.0])
    # A small move overshoots by half of itself, not the full overshoot.
    small = script.place_past(np.array([0.0, 0.0, 0.0]), np.array([0.012, 0.0, 0.0]), 0.01)
    np.testing.assert_allclose(small, [0.018, 0.0, 0.0])


def test_onto_table_slides_points_along_the_camera_rays(script):
    cam = np.eye(4)
    cam[:3, 3] = [0.0, 0.0, 1.0]
    out = script.onto_table(np.array([[0.098, 0.0, 0.02]]), cam, 0.0)
    np.testing.assert_allclose(out, [[0.1, 0.0, 0.0]], atol=1e-12)


def test_pick_candidates_are_ordered_spaced_and_off_the_ends(script):
    m = script.match_curves(bent(0.03), STRAIGHT)
    picks = script.pick_candidates(m, 0.005, 0.015, 3, 0.03)
    assert picks[0] == 15
    s = [m["cable_s"][i] for i in picks]
    assert all(abs(a - b) >= 0.03 - 1e-9 for k, a in enumerate(s) for b in s[k + 1 :])
    # A cable bent at its very end: the pick is clamped 15 mm in from it.
    hook = [list(p) for p in STRAIGHT]
    hook[-1][0] += 0.03
    mh = script.match_curves(hook, STRAIGHT)
    assert mh["worst"] == 30
    lc = mh["cable_length"]
    assert script._clamp_arc(lc, lc, 0.015) == pytest.approx(lc - 0.015)
    assert script._clamp_arc(0.0, lc, 0.015) == pytest.approx(0.015)
    assert script._clamp_arc(0.0, 0.02, 0.015) == pytest.approx(0.01)  # too short: its middle


def test_pull_not_push(script):
    pts = np.asarray(STRAIGHT)
    arcs = script.plan_arcs(pts)
    at = lambda s: script.at_arcs(pts, arcs, s)[0]  # noqa: E731
    # Across the cable: kept, with the overshoot.
    s, p, g, slides, why = script.pull_not_push(
        pts, arcs, 0.15, at(0.15), at(0.15) + [0.02, 0, 0], 0.015
    )
    assert (s, slides, why) == (0.15, False, "")
    # Outward at the far end: a slide, kept.
    s, p, g, slides, why = script.pull_not_push(
        pts, arcs, 0.285, at(0.285), at(0.285) + [0, 0.01, 0], 0.015
    )
    assert s == pytest.approx(0.285) and slides and not why
    # Into the cable at the near end: the far end is pulled the same way instead.
    s, p, g, slides, why = script.pull_not_push(
        pts, arcs, 0.015, at(0.015), at(0.015) + [0, 0.012, 0], 0.015
    )
    assert s == pytest.approx(0.285) and slides and not why
    np.testing.assert_allclose(g - p, [0, 0.012, 0], atol=1e-12)
    # Along the cable in its middle: refused.
    *_, why = script.pull_not_push(pts, arcs, 0.15, at(0.15), at(0.15) + [0, 0.02, 0], 0.015)
    assert "buckle" in why


def test_an_overhanging_end_slides_the_cable_from_its_other_end(script):
    # On the line but 13 mm short of its far end and past its near end.
    crept = line(0.40, 0.037, 0.40, 0.337)
    world = World(crept)
    out, _ = run(script, world, centerline={"points": crept})
    assert out["route"] == "corrected", out["detail"]
    assert out["pick_point"]["y"] == pytest.approx(0.337 - 0.015, abs=1e-6)  # the far end, pulled
    assert out["place_point"]["y"] == pytest.approx(
        0.337 - 0.015 + 0.013, abs=1e-3
    )  # exactly, no overshoot
    assert out["place_point"]["x"] == pytest.approx(0.40, abs=1e-6)


def test_only_along_cable_moves_left_routes_not_converged(script, monkeypatch):
    monkeypatch.setattr(
        script, "pull_not_push", lambda pts, arcs, s, p, g, m: (s, p, g, False, script.BUCKLE)
    )
    world = World(bent(0.04))
    out, _ = run(script, world, centerline={"points": bent(0.04)})
    assert out["route"] == "not_converged", out["detail"]
    assert "along the cable" in out["detail"] and world.moves == []


# --- the loop --------------------------------------------------------------------------


def test_already_on_target_routes_shaped_without_moving(script):
    world = World(STRAIGHT)
    out, _ = run(script, world, centerline={"points": STRAIGHT, "radius_m": R})
    assert out["route"] == "shaped", out["detail"]
    assert out["corrections"] == 0 and world.moves == []
    assert out["max_error"] == pytest.approx(0.0, abs=1e-9)


def test_converges_by_camera_pick_and_place(script):
    world = World(bent(0.04))
    out, ctx = run(script, world, cable_description="the white cable")
    assert out["route"] == "shaped", out["detail"]
    assert out["initial_max_error"] == pytest.approx(0.04, abs=1e-3)
    assert out["max_error"] < 0.01
    assert 1 <= out["corrections"] <= 8 and out["misses"] == 0
    assert out["errors"][0] > out["errors"][-1]
    fit = ctx.calls_to("curve.fit_centerline")[0].kwargs
    assert fit["smooth"] == pytest.approx(1e-5) and fit["nodes"] == 60
    assert world.fits == out["corrections"] + 1
    # The first correction picks the bump's top and puts it down 1 cm past the line.
    assert out["errors"][0] == pytest.approx(0.04, abs=1e-3)


def test_first_correction_carries_to_the_target_plus_overshoot(script):
    world = World(bent(0.04))
    out, _ = run(script, world, centerline={"points": bent(0.04)}, overshoot=0.01)
    assert out["route"] == "corrected", out["detail"]
    assert out["pick_point"]["x"] == pytest.approx(0.44, abs=1e-3)
    assert out["pick_point"]["y"] == pytest.approx(0.20, abs=2e-3)
    assert out["place_point"]["x"] == pytest.approx(0.39, abs=1e-3)  # 1 cm past x = 0.40
    assert out["place_point"]["y"] == pytest.approx(out["pick_point"]["y"], abs=3e-3)
    names = [t for t, _ in world.moves]
    # The grasp skill's approach / narrow / contact / press / lift, then carry, set down, rise.
    assert names[-3:] == ["robot.go_to_pose_cartesian"] * 3
    carry, low, rise = (p for _, p in world.moves[-3:])
    np.testing.assert_allclose(carry[:2], [0.39, out["place_point"]["y"]], atol=1e-3)
    assert carry[2] == pytest.approx(REACH + 0.04, abs=1e-3)  # the grasp's lifted height
    assert low[2] == pytest.approx(REACH + 0.003, abs=1e-6)
    assert rise[2] == pytest.approx(REACH + 0.003 + 0.05, abs=1e-6)
    assert world.grips[-2:] == ["close", "open"]
    assert "fit the centreline again" in out["detail"]


def test_a_pick_near_an_end_gets_no_overshoot(script):
    # The last 5 cm curl 15 mm sideways: the end is the worst node, picked 15 mm in.
    ys = np.linspace(0.05, 0.35, 31)
    xs = 0.40 + 0.015 * np.clip((ys - 0.30) / 0.05, 0.0, None) ** 2
    curl = [[float(x), float(y), R] for x, y in zip(xs, ys, strict=True)]
    world = World(curl)
    out, _ = run(script, world, centerline={"points": curl}, overshoot=0.01)
    assert out["route"] == "corrected", out["detail"]
    pick, place = out["pick_point"], out["place_point"]
    assert pick["y"] > 0.33
    worst = out["errors"][0]
    assert 0.010 < worst <= 0.015
    assert place["x"] == pytest.approx(
        pick["x"] - worst, abs=1e-3
    )  # the end's own correction, exactly
    assert place["y"] == pytest.approx(pick["y"], abs=1e-3)


def test_not_converged_after_max_iters(script):
    world = World(bent(0.06), drag=0.0)  # only the picked node moves: slow progress
    out, _ = run(script, world, cable_description="cable", max_iters=2)
    assert out["route"] == "not_converged", out["detail"]
    assert out["corrections"] == 2
    assert len(out["errors"]) == 3


def test_not_converged_when_the_error_stalls(script):
    world = World(bent(0.04), stuck=True)  # the cable does not move
    out, _ = run(script, world, cable_description="cable", max_iters=10)
    assert out["route"] == "not_converged", out["detail"]
    assert "no better for 3" in out["detail"]
    assert out["corrections"] == 3


def test_lost_cable_when_not_seen_after_a_correction(script):
    world = World(bent(0.04))
    real_open = world._open

    def open_and_hide(**kw):
        world.hidden = True
        return real_open(**kw)

    ctx = world.ctx(robot__open_gripper=open_and_hide)
    out = script.run(
        ctx,
        target=STRAIGHT,
        table_z=0.0,
        min_tool_height=0.005,
        lift_check=False,
        cable_description="cable",
    )
    assert out["route"] == "lost_cable", out["detail"]
    assert "not found" in out["detail"] and out["corrections"] == 1


def test_lost_cable_when_not_seen_at_all(script):
    world = World(bent(0.04))
    world.hidden = True
    out, _ = run(script, world, cable_description="cable")
    assert out["route"] == "lost_cable" and world.moves == []


def test_lost_cable_when_only_part_of_it_is_seen(script):
    world = World(line(0.40, 0.05, 0.40, 0.15))
    out, _ = run(script, world, cable_description="cable")
    assert out["route"] == "lost_cable", out["detail"]
    assert out["coverage"] == pytest.approx(1 / 3, abs=1e-6)


def test_grasp_failed_after_two_misses_in_a_row(script):
    world = World(bent(0.04), fit_offset=(0.0, 0.0))
    ctx = world.ctx(
        robot__close_gripper=lambda **kw: world.grips.append("close") or {"position": 0.0}
    )
    out = script.run(
        ctx,
        target=STRAIGHT,
        table_z=0.0,
        min_tool_height=0.005,
        lift_check=False,
        cable_description="cable",
    )
    assert out["route"] == "grasp_failed", out["detail"]
    assert out["misses"] == 2 and out["corrections"] == 2


def test_a_miss_without_a_camera_is_grasp_failed(script):
    world = World(bent(0.04), fit_offset=(0.02, 0.0))  # the given centreline is 2 cm off
    off = (np.asarray(bent(0.04)) + [0.02, 0.0, 0.0]).tolist()
    out, _ = run(script, world, centerline=off, target=line(0.42, 0.05, 0.42, 0.35))
    assert out["route"] == "grasp_failed", out["detail"]
    assert out["misses"] == 1


def test_an_unreachable_pick_falls_back_to_the_next_discrepancy(script):
    worst_y = 0.20

    def ik(pose):
        far = abs(pose["position"]["y"] - worst_y) < 0.005
        return {"success": not far, "within_servo_tolerance": not far, "reason": "out of reach"}

    world = World(bent(0.04), ik=ik)
    out, _ = run(script, world, centerline={"points": bent(0.04)})
    assert out["route"] == "corrected", out["detail"]
    assert abs(out["pick_point"]["y"] - worst_y) >= 0.02  # 3 cm along the bent cable


def test_every_pick_unreachable_routes_unreachable(script):
    world = World(bent(0.04), ik=lambda pose: {"success": False, "reason": "out of reach"})
    out, _ = run(script, world, centerline={"points": bent(0.04)})
    assert out["route"] == "unreachable", out["detail"]
    assert "no pick reached" in out["detail"]
    assert "close" not in world.grips


@pytest.mark.parametrize("how", ["routed", "raise"])
def test_a_refused_carry_sets_the_cable_down_and_routes_unreachable(script, how):
    lifted = REACH + 0.04

    def refuse(tool, pose):
        p = pose["position"]
        carrying = (
            tool == "robot.go_to_pose_cartesian" and abs(p["z"] - lifted) < 1e-6 and p["x"] < 0.42
        )
        return ("raise" if how == "raise" else "no IK solution") if carrying else None

    world = World(bent(0.04), refuse=refuse)
    out, _ = run(script, world, centerline={"points": bent(0.04)})
    assert out["route"] == "unreachable", out["detail"]
    assert "set down where the hand stood" in out["detail"]
    assert world.grips[-1] == "open" and world.held is None
    assert world.p[2] == pytest.approx(REACH + 0.003 + 0.05, abs=1e-6)


def test_arm_id_is_passed_only_when_nonzero(script):
    world = World(bent(0.04))
    _, ctx = run(script, world, centerline={"points": bent(0.04)})
    assert all("arm_id" not in c.kwargs for c in ctx.calls_to("robot.go_to_pose_cartesian"))
    world = World(bent(0.04))
    _, ctx = run(script, world, centerline={"points": bent(0.04)}, arm_id=1)
    assert all(c.kwargs.get("arm_id") == 1 for c in ctx.calls_to("robot.go_to_pose_cartesian"))


def test_the_grasp_gets_the_pick_and_the_cable_way(script):
    world = World(bent(0.04))
    _, ctx = run(script, world, centerline={"points": bent(0.04)})
    heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
    assert (heading + 90.0) % 180.0 - 90.0 == pytest.approx(
        0.0, abs=3.0
    )  # square to the cable's +y way at the bump


def test_bad_inputs_raise(script):
    world = World(STRAIGHT)
    with pytest.raises(ValueError, match="centerline"):
        script.run(world.ctx(), target=STRAIGHT, table_z=0.0)
    with pytest.raises(ValueError, match="match"):
        script.run(world.ctx(), target=STRAIGHT, centerline=STRAIGHT, match="nearest")
    with pytest.raises(ValueError, match="target"):
        script.run(world.ctx(), target=[[0.4, 0.2, 0.0]], centerline=STRAIGHT)


def test_example_declares_every_route(script):
    root = Path(script.__file__).resolve().parents[1]
    example = json.loads((root / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = Path(script.__file__).read_text()
    routes = set(re.findall(r'finish\(\s*"([a-z_]+)"', source))
    assert routes == {
        "shaped",
        "corrected",
        "not_converged",
        "grasp_failed",
        "lost_cable",
        "unreachable",
    }
    assert routes <= declared
