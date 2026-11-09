# SPDX-License-Identifier: Apache-2.0
"""synthetic_contract_test: the domino evaluator on hand-built observation streams.

These streams are not physics; they pin the verdict logic (status semantics, freshness,
symmetry, neighbour and release checks). Physics negatives live in the backend runs.
"""

import copy
import math

import pytest

from roboskillhub.tasks.assembly_eval import domino_geometry as geo
from roboskillhub.tasks.assembly_eval.domino_postcondition import (
    Thresholds,
    evaluate_layout,
    evaluate_placement,
)
from roboskillhub.tasks.assembly_eval.domino_scene import DominoSceneSpec

SPEC = DominoSceneSpec(n_tiles=3)
TH = Thresholds()


def standing(k, dx=0.0, yaw=None, tilt_deg=0.0, dz=0.0):
    p = SPEC.slot_pose(k)
    yaw = p["yaw"] if yaw is None else yaw
    q = geo.quat_from_yaw(yaw)
    if tilt_deg:
        q = geo.quat_mul(q, geo.quat_from_axis_angle((0, 1, 0), math.radians(tilt_deg)))
    pos = list(p["position"])
    pos[0] += dx
    pos[2] += dz
    return {"position": pos, "quat_xyzw": list(q)}


def stream(t0=1.0, t1=2.2, hz=30, objects=None, contacts="table", per_t=None):
    out, n = [], int((t1 - t0) * hz) + 1
    for i in range(n):
        t = t0 + i / hz
        objs = copy.deepcopy(objects if per_t is None else per_t(t))
        c = (
            None
            if contacts is None
            else {k: (["table"] if contacts == "table" else list(contacts)) for k in objs}
        )
        out.append(
            {
                "t": t,
                "stamp": t,
                "seq": i,
                "source": "simulation_truth",
                "frame": "table",
                "objects": objs,
                "contacts": c,
            }
        )
    return out


EVENTS = {
    "release_t": 0.8,
    "retreat_done_t": 1.0,
    "eval_t": 2.2,
    "phases": [
        {"name": "release", "t0": 0.8, "t1": 0.9},
        {"name": "retreat", "t0": 0.9, "t1": 1.0},
        {"name": "observe", "t0": 1.0, "t1": 2.2},
    ],
}
NEIGH = {"tile_0": standing(0), "tile_1": standing(1)}


def judge(samples, neighbours=NEIGH, events=EVENTS, claim="finished", **kw):
    return evaluate_placement(
        SPEC,
        2,
        "tile_2",
        neighbours,
        samples,
        events,
        kw.pop("th", TH),
        executor_report={"status": claim},
        **kw,
    )


def objs(**over):
    base = {"tile_0": standing(0), "tile_1": standing(1), "tile_2": standing(2)}
    base.update(over)
    return base


def test_success_on_fresh_window():
    r = judge(stream(objects=objs()))
    assert r["status"] == "succeeded", r
    assert r["executor_claim_contradicted"] is False
    assert r["observation_sources"] == ["simulation_truth"]


def test_yaw_half_turn_is_equivalent():
    r = judge(stream(objects=objs(tile_2=standing(2, yaw=math.pi))))
    assert r["status"] == "succeeded"


def test_finished_claim_but_tile_absent_fails():
    far = standing(2, dx=0.2)
    r = judge(stream(objects=objs(tile_2=far)))
    assert r["status"] == "failed"
    assert "not_at_slot" in r["failure_types"]
    assert r["executor_claim_contradicted"] is True


def test_lying_at_slot_fails_even_though_xy_matches():
    lying = standing(2, tilt_deg=90.0, dz=-0.024 + 0.00375)
    r = judge(stream(objects=objs(tile_2=lying)))
    assert r["status"] == "failed"
    assert "toppled" in r["failure_types"]


def test_leaning_is_not_upright():
    r = judge(stream(objects=objs(tile_2=standing(2, tilt_deg=7.0))))
    assert r["status"] == "failed" and "not_upright" in r["failure_types"]


def test_offset_is_placement_error():
    r = judge(stream(objects=objs(tile_2=standing(2, dx=0.008))))
    assert r["status"] == "failed" and r["failure_types"] == ["placement_error"]


def test_hanging_in_gripper_fails_release():
    s = stream(
        objects=objs(tile_2=standing(2, dz=0.0015)), contacts=["gripper_pad_l", "gripper_pad_r"]
    )
    r = judge(s)
    assert r["status"] == "failed"
    assert "release_not_confirmed" in r["failure_types"]


def test_neighbour_knocked_fails_with_phase_attribution():
    def per_t(t):
        return objs(tile_0=standing(0, tilt_deg=80.0) if t >= 1.0 else standing(0))

    pre = stream(t0=0.8, t1=0.98, per_t=per_t)
    post = stream(t0=1.0, t1=2.2, per_t=per_t)
    for i, s in enumerate(post):
        s["seq"] = 100 + i
    r = judge(pre + post)
    assert r["status"] == "failed"
    assert r["neighbours_disturbed"] == ["tile_0"]
    assert r["neighbour_first_moved"]["tile_0"]["phase"] in ("retreat", "observe")
    # no contact recorded with the gripper or the target: only the phase, low confidence
    assert r["cause_hypotheses"][0]["confidence"] == "low"


def test_gripper_contact_attributed():
    def per_t(t):
        return objs(tile_0=standing(0, tilt_deg=80.0) if t >= 1.0 else standing(0))

    s = stream(t0=0.9, t1=2.2, per_t=per_t)
    for x in s:
        if x["stamp"] < 1.0:
            x["contacts"]["tile_0"] = ["table", "gripper_pad_l"]
    r = judge(s)
    assert r["cause_hypotheses"][0]["cause"] == "gripper_contact"


def test_repeated_frames_do_not_fill_the_window():
    s = stream(objects=objs())
    frozen = [dict(s[0], t=x["t"]) for x in s]  # same stamp and seq, delivered later
    r = judge(frozen)
    assert r["status"] == "unverified"
    assert any("observation_stale" in u for u in r["unverified_reasons"])


def test_old_stamps_are_stale():
    s = stream(objects=objs())
    for x in s:
        x["t"] = x["stamp"] + 0.5
    r = judge(s)
    assert r["status"] == "unverified"


def test_target_missing_is_unverified_not_success():
    s = stream(objects={"tile_0": standing(0), "tile_1": standing(1)})
    r = judge(s)
    assert r["status"] == "unverified"
    assert any("orientation_missing" in u for u in r["unverified_reasons"])


def test_position_only_is_unverified():
    o = objs()
    o["tile_2"] = {"position": o["tile_2"]["position"], "quat_xyzw": None}
    r = judge(stream(objects=o))
    assert r["status"] == "unverified"


def test_no_contact_information_is_unverified():
    r = judge(stream(objects=objs(), contacts=None))
    assert r["status"] == "unverified"
    assert any("support_unknown" in u for u in r["unverified_reasons"])


def test_short_window_is_unverified():
    ev = dict(EVENTS, eval_t=1.5)
    r = judge(stream(t1=1.5, objects=objs()), events=ev)
    assert r["status"] == "unverified"


def test_drift_in_window_is_unstable():
    def per_t(t):
        return objs(tile_2=standing(2, dx=0.002 * (t - 1.0)))

    r = judge(stream(per_t=per_t))
    assert r["status"] == "failed" and "unstable_after_release" in r["failure_types"]


def test_floating_tile_is_not_supported():
    r = judge(stream(objects=objs(tile_2=standing(2, dz=0.006)), contacts=[]))
    assert r["status"] == "failed" and "not_supported" in r["failure_types"]


def test_malformed_request_rejected():
    s = stream(objects=objs())
    s[3]["frame"] = "camera"
    assert judge(s)["status"] == "rejected"
    assert (
        judge(
            stream(objects=objs()), events={"release_t": 2.0, "retreat_done_t": 1.0, "eval_t": 2.2}
        )["status"]
        == "rejected"
    )
    bad = stream(objects=objs())
    bad[0]["objects"]["tile_2"]["quat_xyzw"] = [0, 0, 0, 0]
    assert judge(bad)["status"] == "rejected"


def test_layout_needs_every_tile():
    s = stream(objects=objs())
    placements = [{"status": "succeeded"}] * 3
    lay = evaluate_layout(SPEC, s, 1.0, 2.2, placements, TH)
    assert lay["status"] == "succeeded" and lay["standing_at_slot"] == 3
    s2 = stream(objects=objs(tile_1=standing(1, tilt_deg=60)))
    assert evaluate_layout(SPEC, s2, 1.0, 2.2, placements, TH)["status"] == "failed"
    assert evaluate_layout(SPEC, s, 1.0, 2.2, placements[:2], TH)["status"] == "unverified"


@pytest.mark.parametrize("claim", ["finished", "ok"])
def test_claim_never_decides(claim):
    r = judge(stream(objects=objs(tile_2=standing(2, tilt_deg=30))), claim=claim)
    assert r["status"] == "failed" and r["executor_claim_contradicted"]


def test_attribution_ignores_samples_before_this_placement():
    # Before this placement began, tile_0 still stood in the rack (far from its baseline).
    def per_t(t):
        if t < 0.5:
            return objs(tile_0=standing(0, dx=-0.3))
        return objs(tile_0=standing(0, tilt_deg=80.0) if t >= 0.95 else standing(0))

    ev = dict(EVENTS, phases=[{"name": "carry", "t0": 0.6, "t1": 0.8}, *EVENTS["phases"]])
    early = stream(t0=0.0, t1=0.98, per_t=per_t)
    post = stream(t0=1.0, t1=2.2, per_t=per_t)
    for i, s in enumerate(post):
        s["seq"] = 1000 + i
    r = judge(early + post, events=ev)
    assert r["status"] == "failed"
    first = r["neighbour_first_moved"]["tile_0"]
    assert first["t"] >= 0.95 and first["phase"] == "retreat"
