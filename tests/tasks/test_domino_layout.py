"""plan_straight_layout: centre pitch vs face gap, overlap and finger clearance (the placing
skill's rule), build order against the arm corridor (the layout skill's rule)."""

from __future__ import annotations

import math

import pytest

from roboskillhub.tasks.domino import geometry as geo
from roboskillhub.tasks.domino import plan_straight_layout

G = {"height_m": 0.048, "width_m": 0.024, "thickness_m": 0.0075}


def req(**kw):
    r = {
        "frame_id": "station",
        "geometry": G,
        "start_m": [0.40, 0.0],
        "heading_deg": 90.0,
        "n_tiles": 5,
        "table_z_m": 0.0,
        "spacing": {"center_pitch_m": 0.04},
    }
    r.update(kw)
    return r


def test_line_poses_spacing_and_facing():
    out = plan_straight_layout(req())
    assert out["status"] == "succeeded", out["failure"]
    ev = out["evidence"]
    assert len(ev["slots"]) == 5
    assert ev["face_gap_m"] == pytest.approx(0.04 - 0.0075)
    p = ev["slots"][3]["pose"]
    assert p["frame_id"] == "station" and p["position_m"] == pytest.approx([0.40, 0.12, 0.024])
    assert geo.facing_deg_of(p) == pytest.approx(90.0)
    same = plan_straight_layout(req(spacing={"face_gap_m": 0.0325}))
    assert same["evidence"]["center_pitch_m"] == pytest.approx(0.04)
    assert same["evidence"]["slots"][3]["pose"]["position_m"] == pytest.approx(p["position_m"])
    assert "plan only" in out["unverified"][0]


def test_spacing_must_be_one_of_centre_or_face():
    out = plan_straight_layout(req(spacing={"center_pitch_m": 0.04, "face_gap_m": 0.03}))
    assert out["status"] == "rejected" and out["failure"]["code"] == "invalid_request"


def test_overlapping_slots_are_rejected():
    out = plan_straight_layout(req(spacing={"center_pitch_m": 0.005}))
    assert out["status"] == "rejected" and out["failure"]["code"] == "infeasible_geometry"


def test_slots_too_close_for_the_open_fingers_have_no_order():
    out = plan_straight_layout(req(spacing={"center_pitch_m": 0.015}))
    assert out["status"] == "rejected"
    assert out["failure"]["code"] == "retreat_blocked"
    assert all("open fingers" in c["reason"] for c in out["evidence"]["order_candidates"].values())


def test_radial_row_orders_far_first_and_rejects_near_first():
    # The row runs straight away from the arm base: placing near tiles first makes the
    # arm reach past them for every later slot.
    r = req(
        start_m=[0.30, 0.0],
        heading_deg=0.0,
        arm_base_xy_m=[0.0, 0.0],
        order_preference=["forward", "reverse", "far_first"],
    )
    out = plan_straight_layout(r)
    cands = out["evidence"]["order_candidates"]
    assert not cands["forward"]["feasible"] and "reach past" in cands["forward"]["reason"]
    assert cands["reverse"]["feasible"]
    assert out["evidence"]["order_name"] == "reverse"
    assert out["evidence"]["build_order"] == [4, 3, 2, 1, 0]


def test_existing_observed_tiles_are_obstacles():
    blocker = {
        "id": "old",
        "pose": geo.pose("station", [0.40, 0.09, 0.0], geo.yaw_quat(math.pi / 2)),
    }
    out = plan_straight_layout(req(existing=[blocker]))
    assert out["status"] == "rejected"


def test_workspace_bounds_the_slots():
    out = plan_straight_layout(req(workspace={"min_m": [0.3, -0.1], "max_m": [0.5, 0.1]}))
    assert out["status"] == "rejected" and "outside the workspace" in out["failure"]["detail"]


def test_cascade_physics_only_enforced_on_request():
    loose = req(spacing={"center_pitch_m": 0.07}, n_tiles=3)
    assert plan_straight_layout(loose)["status"] == "succeeded"
    out = plan_straight_layout({**loose, "cascade": {"require": True}})
    assert out["status"] == "rejected"
    assert out["evidence"]["cascade"]["critical_gap_m"] < 0.07 - 0.0075


def test_emits_a_rsh_task_layout_target():
    out = plan_straight_layout(req(arm_base_xy_m=[0.0, 0.0]))
    t = out["evidence"]["layout_target"]
    assert t["kind"] == "domino_layout" and t["piece"]["size_m"] == [0.048, 0.024, 0.0075]
    assert sorted(s["order"] for s in t["slots"]) == list(range(5))
    assert t["slots"][0]["neighbours"] == ["s1"] and t["slots"][2]["neighbours"] == ["s1", "s3"]
    assert out["evidence"]["postcondition"]["passed"]
