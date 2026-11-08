"""planning-a-domino-layout: the ported toppling physics against its sources' numbers, and the planner's
slots, pitch, curve checks, gates, build order and every exit with FakeContext (no motion)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

BUNDLE = "planning-a-domino-layout"
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / BUNDLE
H, W, T = 0.048, 0.024, 0.0075
MU = 0.25
# The real station's left-arm box (motion_server.BOUNDS), left base frame = world.
STATION = {
    "surface_z": -0.03,
    "bounds": {"min_m": [-0.15, 0.10, -0.02], "max_m": [0.35, 0.55, 0.35]},
}
WIDE = {"surface_z": 0.0, "bounds": {"min_m": [-0.6, -0.6, 0.0], "max_m": [0.6, 0.6, 0.3]}}


@pytest.fixture(scope="module")
def plan(skills_registry):
    bundle = skills_registry.get(BUNDLE)
    assert bundle.kind == "skill"
    return bundle.canonical_scripts["plan_layout"].module


@pytest.fixture(scope="module")
def physics(plan):
    return plan.physics


@pytest.fixture(scope="module")
def placer(skills_registry):
    bundle = skills_registry.get("placing-an-upright-domino-beside-neighbours")
    return bundle.canonical_scripts["place_domino"].module


def ctx(ws=WIDE):
    return FakeContext({"robot.describe_workspace": ws})


def line(x0, y0, x1, y1):
    return {"type": "line", "start": [x0, y0], "end": [x1, y1]}


def xy_of(slot):
    return np.array([slot["position"]["x"], slot["position"]["y"]])


def yaw_of(rot):
    w, x, y, z = (rot[k] for k in "wxyz")
    return math.degrees(math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


# ----------------------------------------------------------------------------- the physics


def test_gap_chain_reproduces_upstream_critical_gap(physics, monkeypatch):
    """Upstream's spacing_threshold.py (h = 1, d = 0.2, friction 0.8, spacing 0.5, a 10 rad/s push,
    g = 10, the gap after 20 tiles): 'Found value: s ~= 0.96'."""
    monkeypatch.setattr(physics, "G", 10.0)

    def stops_at(gap):
        chain = physics.GapChain.uniform(30, 1.0, 0.5, 0.2, 0.8, 1.0)
        chain.gaps[20] = gap
        return physics.run_until_failure(chain, 10.0)

    assert stops_at(0.95) is None
    assert stops_at(0.96) == 21


def test_a_wave_that_reaches_the_table_does_not_cross(physics):
    """Upstream's run_until_failure read a gap the top never reached as crossed once the leading
    tile hit the ground; here it stops there."""
    chain = physics.GapChain.uniform(6, H, 0.03, T, MU)
    chain.gaps[2] = 1.2 * H
    assert physics.run_until_failure(chain, 10.0) == 2
    assert chain.ground_hit


def test_stacked_line_matches_chapmans_simulation(physics):
    """zmatt.net: 48 x 24 x 7.5 mm tiles, friction 0.25; his simulation 59.7 / 37.2 / 26.2 tiles/s
    at a 16 / 24 / 32 mm pitch (experiment 64.4 / 42.3 / 28.1)."""
    for pitch, ref in ((0.016, 59.7), (0.024, 37.2), (0.032, 26.2)):
        f = physics.StackedLine(H, W, T, pitch, MU).frequency(30)
        assert f == pytest.approx(ref, rel=0.03), (pitch, f)


def test_stacked_line_stalls_and_never_hits_past_the_tile_height(physics):
    assert physics.StackedLine(H, W, T, H + T + 0.001, MU).run(5, 5.0) == []
    slow = physics.StackedLine(H, W, T, 0.016, MU).run(10, 1.0)
    assert len(slow) < 9  # turned back before the end


def test_critical_gap_of_a_standard_tile(physics):
    crit = physics.critical_gap(H, T, 0.037, MU)
    assert 0.047 < crit < 0.048
    # Barely moves with the pitch the wave arrives at, or with friction.
    assert physics.critical_gap(H, T, 0.020, MU) == pytest.approx(crit, abs=2e-4)
    assert physics.critical_gap(H, T, 0.037, 0.8) == pytest.approx(crit, abs=1.5e-3)


def test_the_placing_skills_twin_result_agrees(physics):
    """placing-an-upright-domino-beside-neighbours' twin: 1.0 h (48 mm) cascaded, 1.5 h did not."""
    assert physics.chain_crosses(H, T, 1.0 * H, MU, None, 7, 0)
    assert not physics.chain_crosses(H, T, 1.5 * H, MU, None, 5, 0)


def test_a_slow_push_stalls_only_at_a_small_pitch(physics):
    w0 = 1.2 * physics.tipping_speed(H, T)
    assert physics.tipping_speed(H, T) == pytest.approx(2.69, abs=0.01)
    assert not physics.chain_crosses(H, T, 0.016, MU, None, 11, 0, omega0=w0)
    assert physics.chain_crosses(H, T, 0.020, MU, None, 11, 0, omega0=w0)


# ----------------------------------------------------------------------------- slots and pitch


def test_auto_pitch_on_a_straight_line(plan):
    out = plan.run(ctx(STATION), line(0.30, 0.15, -0.10, 0.15))
    assert out["route"] == "planned", out["detail"]
    crit = out["critical_gap"]
    assert out["pitch"] == pytest.approx(2 * (crit + T) / 3)
    assert out["pitch"] == pytest.approx(0.0369, abs=3e-4)
    assert out["pitch_source"] == "auto"
    # The line's face gap is as far under the critical gap as a one-slot gate is over it.
    assert crit - out["face_gap"] == pytest.approx(2 * out["pitch"] - T - crit)
    assert out["topple_margin"] == pytest.approx(0.0184, abs=3e-4)
    assert out["n_tiles"] == int(0.40 / out["pitch"]) + 1
    for k, s in enumerate(out["slots"]):
        assert s["position"]["z"] == pytest.approx(-0.03)  # the workspace's surface
        assert s["position"]["x"] == pytest.approx(0.30 - k * out["pitch"], abs=1e-6)
        assert abs((s["facing_deg"] - 180.0 + 180.0) % 360.0 - 180.0) < 1e-6
        assert abs((yaw_of(s["rotation"]) - s["facing_deg"] + 180) % 360 - 180) < 1e-6
    assert out["first_slot"] == out["slot_poses"][0]
    assert out["trigger"]["push_direction"]["x"] == pytest.approx(-1.0)
    assert sorted(out["build_order"]) == list(range(out["n_tiles"]))


def test_slots_read_back_through_the_placing_skill(plan, placer):
    out = plan.run(
        ctx(),
        {"type": "arc", "center": [0.0, 0.3], "radius": 0.15, "start_deg": -60, "sweep_deg": 150},
    )
    assert out["route"] == "planned", out["detail"]
    for s, pose in zip(out["slots"], out["slot_poses"], strict=True):
        xy, facing = placer.tile_pose(pose)
        assert np.allclose(xy, xy_of(s))
        assert abs((facing - s["facing_deg"] + 180) % 360 - 180) < 1e-6


def test_every_build_step_passes_the_placing_skills_finger_check(plan, placer):
    """The plan's steps, replayed through the placing skill's own finger_conflicts with the tiles
    standing at each step, are never refused - on a curve, with gates, at the tightest pitch."""
    size = {"height": H, "width": W, "thickness": T}
    open_width = W + 2 * placer.GRIP_CLEARANCE
    for path, pitch in (
        (
            {
                "type": "arc",
                "center": [0.0, 0.0],
                "radius": 0.35,
                "start_deg": 10,
                "sweep_deg": 160,
            },
            "auto",
        ),
        (line(-0.45, 0.35, 0.45, 0.35), 0.0198),
        ([[0.3, 0.15], [0.3, 0.45], [-0.1, 0.45]], "auto"),
    ):
        out = plan.run(ctx(), path, pitch=pitch)
        assert out["route"] == "planned", out["detail"]
        poses = out["slot_poses"]
        for step in out["build_steps"]:
            k = step["slot_index"]
            assert step["neighbours"] == sorted(step["neighbours"])
            nbrs = [placer.tile_pose(poses[t]) for t in step["neighbours"]]
            xy, f = placer.tile_pose(poses[k])
            assert placer.finger_conflicts(xy, f, nbrs, size, open_width) == []


def test_given_pitches_against_the_twin(plan):
    straight = line(0.30, 0.15, -0.10, 0.15)
    at_1h = plan.run(ctx(STATION), straight, pitch=1.0 * H)
    assert at_1h["route"] == "planned", at_1h["detail"]
    assert at_1h["topple_margin"] == pytest.approx(at_1h["critical_gap"] - (H - T))
    assert at_1h["topple_margin"] == pytest.approx(0.0073, abs=3e-4)
    assert not any("over the" in p for p in at_1h["problems"])
    at_15h = plan.run(ctx(STATION), straight, pitch=1.5 * H)
    assert at_15h["route"] == "too_loose"
    assert "64.5 mm" in at_15h["detail"]
    over = plan.run(ctx(STATION), straight, pitch=0.050)
    assert over["route"] == "planned"
    assert any("over the 1.0 h" in p for p in over["problems"])
    edge = plan.run(ctx(STATION), straight, pitch=0.052)  # 44.5 mm face gap, 3.3 mm under critical
    assert edge["route"] == "too_loose"
    assert (
        plan.run(ctx(STATION), straight, pitch=0.052, topple_margin_min=0.002)["route"] == "planned"
    )


def test_too_tight_pitches(plan):
    straight = line(0.30, 0.15, -0.10, 0.15)
    fingers = plan.run(ctx(STATION), straight, pitch=0.018)
    assert fingers["route"] == "too_tight"
    assert "19.8 mm" in fingers["detail"]
    assert plan.run(ctx(STATION), straight, pitch=0.005)["route"] == "too_tight"
    # A 12 mm thick tile: the fingers fit from 22 mm, but a slow push stalls until about 30 mm.
    thick = {"thickness": 0.012}
    stall = plan.run(ctx(STATION), straight, pitch=0.025, tile_size=thick)
    assert stall["route"] == "too_tight", stall["detail"]
    assert "push stalls" in stall["detail"]
    assert plan.run(ctx(STATION), straight, pitch=0.032, tile_size=thick)["route"] == "planned"


def test_n_tiles_and_length(plan):
    straight = line(0.30, 0.15, -0.10, 0.15)
    fit = plan.run(ctx(STATION), straight, n_tiles=6)
    assert fit["route"] == "planned"
    assert fit["n_tiles"] == 6 and fit["pitch_source"] == "auto"
    assert fit["used_length"] == pytest.approx(5 * fit["pitch"])
    shrunk = plan.run(ctx(STATION), line(0.30, 0.15, 0.05, 0.15), n_tiles=9)
    assert shrunk["route"] == "planned", shrunk["detail"]
    assert shrunk["pitch"] == pytest.approx(0.25 / 8)
    assert shrunk["pitch_source"] == "auto, shrunk to fit"
    cut = plan.run(ctx(STATION), straight, length=0.1)
    assert cut["n_tiles"] == int(0.1 / cut["pitch"]) + 1
    assert cut["path_length"] == pytest.approx(0.40)


def test_too_short(plan):
    assert plan.run(ctx(STATION), line(0.0, 0.2, 0.02, 0.2))["route"] == "too_short"
    out = plan.run(ctx(STATION), line(0.30, 0.15, -0.10, 0.15), pitch=0.03, n_tiles=20)
    assert out["route"] == "too_short"
    assert "570 mm" in out["detail"]


def test_invalid_input(plan):
    assert plan.run(ctx(), {"type": "spiral"})["route"] == "invalid_input"
    assert plan.run(ctx(), {"type": "polyline", "points": [[0, 0]]})["route"] == "invalid_input"
    assert (
        plan.run(ctx(), {"type": "arc", "center": [0, 0], "radius": -1, "sweep_deg": 90})["route"]
        == "invalid_input"
    )
    assert plan.run(ctx(), line(0, 0.2, 0.3, 0.2), pitch="fast")["route"] == "invalid_input"
    assert (
        plan.run(ctx(), line(0, 0.2, 0.3, 0.2), tile_size={"thickness": 0.05})["route"]
        == "invalid_input"
    )


def test_without_a_workspace(plan):
    out = plan.run(
        FakeContext({}), {"type": "line", "start": [0, 0.2, 0.01], "heading_deg": 0, "length": 0.3}
    )
    assert out["route"] == "planned", out["detail"]
    assert out["slots"][0]["position"]["z"] == pytest.approx(0.01)  # the path's own height
    assert any("reach not checked" in p for p in out["problems"])
    pinned = plan.run(FakeContext({}), line(0, 0.2, 0.3, 0.2), table_z=0.05)
    assert pinned["slots"][0]["position"]["z"] == pytest.approx(0.05)


# ----------------------------------------------------------------------------- curves


def test_arc_radius_floor_from_where_the_top_lands(plan):
    arc = {"type": "arc", "center": [0.1, 0.3], "start_deg": 0, "sweep_deg": 120}
    sharp = plan.run(ctx(), {**arc, "radius": 0.055}, gate_every=0)
    assert sharp["route"] == "too_tight"
    assert "off its centre" in sharp["detail"]
    ok = plan.run(ctx(), {**arc, "radius": 0.06}, gate_every=0)
    assert ok["route"] == "planned", ok["detail"]
    p = ok["pitch"]
    assert ok["max_contact_offset"] < W / 2
    assert ok["max_contact_offset"] == pytest.approx(p * p / (2 * 0.06), rel=0.25)
    # On a curve the wave crosses more than p - d.
    assert ok["max_face_gap"] > p - T
    assert ok["max_turn_deg"] == pytest.approx(math.degrees(p / 0.06), abs=0.5)


def test_clockwise_arc_faces_along_the_path(plan):
    out = plan.run(
        ctx(),
        {"type": "arc", "center": [0.0, 0.0], "radius": 0.3, "start_deg": 120, "sweep_deg": -60},
    )
    assert out["route"] == "planned", out["detail"]
    for s in out["slots"]:
        a = math.atan2(s["position"]["y"], s["position"]["x"])
        tangent = math.degrees(a) - 90.0
        assert abs((s["facing_deg"] - tangent + 180) % 360 - 180) < 0.5


def test_a_closed_circle_comes_back_on_itself(plan):
    out = plan.run(
        ctx(), {"type": "arc", "center": [0.0, 0.3], "radius": 0.1, "sweep_deg": 360}, gate_every=0
    )
    assert out["route"] == "too_tight"
    assert "comes back on itself" in out["detail"]


def test_inner_corner_overlap(plan):
    """A square corner with no rounding: the tiles each side of it are turned 90 deg, and a pitch
    that puts both near the corner makes their inner corners overlap."""
    out = plan.run(
        ctx(),
        {
            "type": "polyline",
            "points": [[0.0, 0.2], [0.03, 0.2], [0.03, 0.4]],
            "corner_radius": 0.0,
        },
        pitch=0.02,
        gate_every=0,
    )
    assert out["route"] == "too_tight"
    assert "inner corner" in out["detail"]


def test_polyline_corners_are_rounded(plan):
    out = plan.run(ctx(), [[0.3, 0.15], [0.3, 0.45], [-0.1, 0.45]])
    assert out["route"] == "planned", out["detail"]
    assert out["max_turn_deg"] < 30.0
    # About 2 * r * tan(45 deg) - (pi / 2) * r shorter than the legs.
    assert out["path_length"] == pytest.approx(0.70 - (2 - math.pi / 2) * 0.08, abs=1e-3)
    sharp = plan.run(
        ctx(),
        {
            "type": "polyline",
            "points": [[0.3, 0.15], [0.3, 0.45], [-0.1, 0.45]],
            "corner_radius": 0.03,
        },
    )
    assert sharp["route"] == "too_tight"


def test_sampled_curve(plan):
    xs = np.linspace(-0.3, 0.3, 300)
    pts = [[float(x), 0.35 + 0.03 * math.sin(2 * math.pi * x / 0.3)] for x in xs]
    out = plan.run(ctx(), {"points": pts})
    assert out["route"] == "planned", out["detail"]
    for s in out["slots"]:
        x = s["position"]["x"]
        slope = 0.03 * 2 * math.pi / 0.3 * math.cos(2 * math.pi * x / 0.3)
        assert abs(s["facing_deg"] - math.degrees(math.atan(slope))) < 1.5
        assert s["position"]["y"] == pytest.approx(
            0.35 + 0.03 * math.sin(2 * math.pi * x / 0.3), abs=5e-4
        )


# ----------------------------------------------------------------------------- reach and obstacles


def test_out_of_reach(plan):
    out = plan.run(ctx(STATION), line(0.30, 0.15, 0.50, 0.15))
    assert out["route"] == "out_of_reach"
    assert "outside every arm" in out["detail"]
    tabled = plan.run(
        ctx(STATION), line(0.30, 0.15, -0.10, 0.15), table={"min": [0.0, 0.0], "max": [1.0, 1.0]}
    )
    assert tabled["route"] == "out_of_reach" and "off the table" in tabled["detail"]
    annulus = plan.run(
        ctx(), line(0.30, 0.15, -0.10, 0.15), arms=[{"base": [0, 0], "reach": [0.16, 0.6]}]
    )
    assert annulus["route"] == "out_of_reach"


def test_lead_in_must_fit_the_box(plan):
    """A row pointing at the arm whose last tile is 20 mm from the box's edge: the placing skill
    would come in along the row from outside the box."""
    out = plan.run(ctx(STATION), line(0.0, 0.12, 0.0, 0.52), gate_every=0)
    assert out["route"] == "out_of_reach"
    assert "lead-in" in out["detail"]
    ok = plan.run(ctx(STATION), line(0.0, 0.14, 0.0, 0.52), gate_every=0)
    assert ok["route"] == "planned", ok["detail"]


def test_obstacles(plan, physics):
    straight = line(-0.2, 0.35, 0.2, 0.35)
    for ob in (
        {"center": [0.0, 0.40], "radius": 0.02},
        {"center": {"x": 0.0, "y": 0.30}, "size": [0.05, 0.05], "yaw_deg": 30},
        {"center": [0.0, 0.33], "extent": {"x": 0.01, "y": 0.005, "z": 0.02}},
        {"min": [-0.01, 0.30], "max": [0.01, 0.31]},
        {"position": {"x": 0.0, "y": 0.39}, "facing_deg": 0},  # a standing tile
    ):
        out = plan.run(ctx(), straight, obstacles=[ob])
        assert out["route"] == "blocked", (ob, out["detail"])
    clear = plan.run(ctx(), straight, obstacles=[{"center": [0.0, 0.45], "radius": 0.02}])
    assert clear["route"] == "planned"


# ----------------------------------------------------------------------------- gates and order


def test_automatic_gates_along_a_row_across_the_arm(plan):
    out = plan.run(ctx(), line(-0.45, 0.35, 0.45, 0.35))
    assert out["route"] == "planned", out["detail"]
    n = out["n_tiles"]
    assert out["gate_count"] == 2 and out["gate_width_slots"] == 1
    gate_slots = [s for g in out["gates"] for s in g["slots"]]
    assert [s["index"] for s in out["slots"] if s["gate"]] == sorted(gate_slots)
    # Filled last, in closing order; one empty slot is a face gap of 2p - d.
    assert out["build_order"][-len(gate_slots) :] == gate_slots
    for g in out["gates"]:
        assert g["gap"] == pytest.approx(2 * out["pitch"] - T)
        assert g["margin"] == pytest.approx(g["gap"] - out["critical_gap"])
        assert g["margin"] >= 0.010
    assert out["gate_margin"] == pytest.approx(min(g["margin"] for g in out["gates"]))
    # The smaller merged run first; the last gate joins the whole line.
    assert out["gates"][0]["at_risk"] < out["gates"][1]["at_risk"] == n
    # Every gate has two standing tiles each side, and no gate touches the trigger.
    for s in gate_slots:
        assert 2 <= s <= n - 3


def test_gate_width_at_a_tight_pitch(plan):
    out = plan.run(ctx(), line(-0.45, 0.35, 0.45, 0.35), pitch=0.024)
    assert out["route"] == "planned", out["detail"]
    assert (
        out["gate_width_slots"] == 2
    )  # 2p - d = 40.5 mm does not stop the wave; 3p - d = 64.5 does
    assert out["gates"][0]["gap"] == pytest.approx(3 * 0.024 - T)


def test_given_gates(plan):
    across = line(-0.45, 0.35, 0.45, 0.35)
    out = plan.run(ctx(), across, gates=[12, {"x": 0.30, "y": 0.35}])
    assert out["route"] == "planned", out["detail"]
    firsts = sorted(g["slots"][0] for g in out["gates"])
    nearest = int(np.argmin([abs(s["position"]["x"] - 0.30) for s in out["slots"]]))
    assert firsts == sorted([12, nearest])
    assert plan.run(ctx(), across, gates=[1])["route"] == "blocked"
    assert plan.run(ctx(), across, gates=[10, 12])["route"] == "blocked"
    none = plan.run(ctx(), across, gates=[])
    assert none["gate_count"] == 0


def test_no_gate_on_a_row_pointing_at_the_arm(plan):
    """Closing a gate there means reaching past the tile two slots nearer the arm."""
    radial = line(0.0, 0.05, 0.0, 0.55)
    auto = plan.run(ctx(), radial)
    assert auto["route"] == "planned", auto["detail"]
    assert auto["gate_count"] == 0
    assert any("no closable gate" in p for p in auto["problems"])
    assert plan.run(ctx(), radial, gates=[8])["route"] == "blocked"


def test_build_order_works_toward_the_arm(plan):
    out = plan.run(ctx(), line(0.0, 0.05, 0.0, 0.55), gate_every=0)
    assert out["route"] == "planned"
    n = out["n_tiles"]
    assert out["build_order"] == list(range(n - 1, -1, -1))  # farthest from the base first
    assert all(s["at_risk"] == k + 1 for k, s in enumerate(out["build_steps"]))
    assert not any(s["between_standing"] for s in out["build_steps"])


def test_no_line_step_reaches_past_a_standing_tile(plan):
    for path in (
        line(0.30, 0.15, -0.10, 0.15),
        {"type": "arc", "center": [0.1, 0.3], "radius": 0.12, "start_deg": -90, "sweep_deg": 270},
        [[0.3, 0.15], [0.3, 0.45], [-0.1, 0.45]],
    ):
        out = plan.run(ctx(), path)
        assert out["route"] == "planned", out["detail"]
        xy = [xy_of(s) for s in out["slots"]]
        base = np.zeros(2)
        for step in out["build_steps"]:
            k = step["slot_index"]
            beside = {k - 1, k + 1}
            if step["gate"]:
                g = next(g["slots"] for g in out["gates"] if k in g["slots"])
                beside = {g[0] - 1, g[-1] + 1} | set(g)
            for t in set(step["neighbours"]) - beside:
                assert not plan.in_corridor(base, xy[k], xy[t]), (path, k, t)


def test_where_two_runs_meet_is_reported(plan):
    out = plan.run(ctx(STATION), line(0.30, 0.15, -0.10, 0.15), gate_every=0)
    assert out["route"] == "planned"
    between = [s["slot_index"] for s in out["build_steps"] if s["between_standing"]]
    assert between == [out["build_order"][-1]]
    # The meeting point is the slot nearest the arm's base.
    nearest = int(np.argmin([np.linalg.norm(xy_of(s)) for s in out["slots"]]))
    assert between == [nearest]
    assert any("between two standing tiles" in p for p in out["problems"])


def test_blocked_when_every_order_reaches_past(plan):
    """A row pointing at arm 0, whose near end only arm 1 reaches (arm 0's reach starts at 0.2 m):
    arm 1's tiles go first (farther from their base), and then arm 0 would reach past them."""
    arms = [
        {"arm_id": 0, "base": [0.0, 0.0], "reach": [0.2, 1.0]},
        {"arm_id": 1, "base": [0.5, 0.0], "reach": [0.0, 0.54]},
    ]
    out = plan.run(FakeContext({}), line(0.0, 0.30, 0.0, 0.15), arms=arms, table_z=0.0)
    assert out["route"] == "blocked", out["detail"]
    assert "reaching past" in out["detail"]
    # One arm alone, from the far end toward itself, never does.
    alone = plan.run(
        FakeContext({}), line(0.0, 0.30, 0.0, 0.15), arms=[{"base": [0.0, 0.0]}], table_z=0.0
    )
    assert alone["route"] == "planned"


def test_two_arms_split_the_row(plan):
    arms = [
        {
            "arm_id": 0,
            "base": [0.0, 0.0],
            "bounds": {"min_m": [-0.15, 0.10], "max_m": [0.35, 0.55]},
        },
        {
            "arm_id": 1,
            "base": [0.60, 0.0],
            "bounds": {"min_m": [0.25, 0.10], "max_m": [0.75, 0.55]},
        },
    ]
    out = plan.run(FakeContext({}), line(-0.10, 0.30, 0.70, 0.30), arms=arms, table_z=0.0)
    assert out["route"] == "planned", out["detail"]
    by = {s["slot_index"]: s["arm_id"] for s in out["build_steps"]}
    for s in out["slots"]:
        x = s["position"]["x"]
        if x < 0.25:
            assert by[s["index"]] == 0
        elif x > 0.35:
            assert by[s["index"]] == 1
    assert set(by.values()) == {0, 1}
    # Out of both boxes.
    assert (
        plan.run(FakeContext({}), line(-0.10, 0.30, 0.90, 0.30), arms=arms, table_z=0.0)["route"]
        == "out_of_reach"
    )


# ----------------------------------------------------------------------------- the bundle


def test_every_route_is_declared(plan):
    example = json.loads((SKILL_DIR / "examples" / "canonical_subgraph.json").read_text())
    declared = set(example["exit"]["success_values"]) | {example["on_error"]}
    source = (SKILL_DIR / "scripts" / "plan_layout.py").read_text()
    routes = set(re.findall(r'finish\(\s*"([a-z_]+)"', source)) | set(
        re.findall(r'return\s+\(?\s*"([a-z_]+)",\s*\(?\s*f"', source)
    )
    assert {"planned", "too_tight", "out_of_reach"} <= routes
    assert routes <= declared, routes - declared
    md = (SKILL_DIR / "SKILL.md").read_text()
    for r in declared:
        assert f"    {r}:" in md
