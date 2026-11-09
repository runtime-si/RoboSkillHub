"""domino.place_upright through the real workcell runtime (rsh-task/1): Runtime +
task lease + primitive_skill_context + run_task_skill, over FakeDominoStation, a
StationBackend double driving the same synthetic kinematic world (tier:
synthetic_contract_test). Real wall-clock time: the runtime checks observation age."""

from __future__ import annotations

import copy
import time

import pytest
from _domino_world import FRAME, SOURCE, FakeTaskContext, request

from roboskillhub.runtime.contracts import Limits
from roboskillhub.runtime.engine import Runtime
from roboskillhub.runtime.task import primitive_skill_context, run_task_skill
from roboskillhub.tasks.domino import SKILLS

PLACE = next(s for s in SKILLS if s.skill_id == "domino.place_upright")


class FakeDominoStation:
    """StationBackend over the kinematic world; not YAM, not Isaac."""

    def __init__(self, world: FakeTaskContext):
        self.world = world
        self.width = 0.03
        self.stops = 0

    def describe(self) -> dict:
        caps = set(self.world.capabilities) | {"gripper_feedback"}
        d = {
            "name": "fake-domino-station",
            "frame_id": FRAME,
            "capabilities": sorted(caps),
            "evidence_level": "kinematic_sim",
        }
        if "contact_force_feedback" in caps:
            d["contact_signal"] = {
                "source": "synthetic:contact",
                "unit": "N",
                "rate_hz": 500.0,
                "max_latency_s": 0.01,
            }
            d["controller"] = {"kind": "admittance"}
        return d

    def guarded_move(self, **kw) -> dict:
        return self.world.guarded_move(**kw)

    def get_state(self, *, timeout_s: float) -> dict:
        w, now = self.world, time.time()
        objects = {}
        for oid, rec in zip(
            [w.target, *w.neighbors],
            w._records([w.target, *w.neighbors]),
            strict=True,
        ):
            objects[oid] = {k: v for k, v in rec.items() if k not in ("object_id",)} | {
                "timestamp_s": now
            }
        return {
            "pose": copy.deepcopy(w.tcp),
            "timestamp_s": now,
            "objects": objects,
            "gripper": {
                "holding": w.held,
                "width_m": self.width,
                "timestamp_s": now,
                "source": "synthetic:jaw-gap",
            },
        }

    def move_pose(self, pose: dict, *, speed_m_s: float, timeout_s: float, linear: bool) -> dict:
        self.world._move(pose, speed_m_s)
        return {"ok": True}

    def set_gripper(self, width_m: float, *, timeout_s: float) -> dict:
        self.width = width_m
        self.world.release_to(width_m)
        return {"ok": True}

    def stop(self) -> None:
        self.stops += 1


LIMITS = Limits.from_dict(
    {
        "frame_id": FRAME,
        "workspace_min_m": [0.0, -0.5, 0.0],
        "workspace_max_m": [0.8, 0.5, 0.5],
        "max_speed_m_s": 0.5,
        "max_contact_distance_m": 0.2,
        "gripper_min_width_m": 0.0,
        "gripper_max_width_m": 0.1,
        "max_contact_force_N": 5.0,
    }
)


def run(faults=(), capabilities=None, table_z=0.0, **over):
    world = FakeTaskContext(
        faults=set(faults), real_time=True, capabilities=capabilities, table_z=table_z
    )
    station = FakeDominoStation(world)
    runtime = Runtime(station, LIMITS)
    req = request(world, speed_m_s=0.5, descent_speed_m_s=0.5, timeout_s=30.0, **over)
    with runtime.task_lease("domino-it", 60.0) as lease:
        ctx = primitive_skill_context(runtime, lease, request_id="domino-it/n1/a1")
        out = run_task_skill(PLACE, req, ctx)
    return out, station


def test_place_upright_succeeds_on_the_real_runtime():
    out, station = run()
    assert out["status"] == "succeeded", (out.get("failure"), out.get("error"))
    assert out["evidence"]["postcondition"]["passed"] and out["observations"]
    assert all(o["source"] == SOURCE for o in out["observations"])
    assert out.get("downgraded") is None and out["children"]
    assert station.stops == 0


@pytest.mark.parametrize(
    "faults, status, failure_type",
    [
        ({"drop_in_transit"}, "failed", "grasp_lost"),
        ({"retreat_knocks"}, "failed", "domino.neighbour_disturbed"),
        ({"fall_on_release"}, "failed", "domino.toppled_after_release"),
        ({"still_holding"}, "failed", "release_failed"),
        ({"missed_grasp"}, "rejected", "precondition_unmet"),
    ],
)
def test_negatives_on_the_real_runtime(faults, status, failure_type):
    out, station = run(faults)
    assert (out["status"], out["failure_type"]) == (status, failure_type), out.get("failure")


def test_capability_gate_of_the_runtime():
    world = FakeTaskContext(
        real_time=True,
        capabilities={"pose_feedback", "cartesian_motion", "gripper", "stop", "object_feedback"},
    )
    runtime = Runtime(FakeDominoStation(world), LIMITS)
    with runtime.task_lease("domino-it", 10.0) as lease:
        ctx = primitive_skill_context(runtime, lease, request_id="r")
        out = run_task_skill(PLACE, request(world, timeout_s=30.0), ctx)
    assert out["status"] == "rejected" and out["failure_type"] == "capability_missing"
    assert out["children"] == []


def test_contact_mode_through_guarded_press_on_the_real_runtime():
    from roboskillhub.tasks.domino.contract import CAPABILITIES

    kw = {
        "capabilities": CAPABILITIES["contact_feedback"],
        "mode": "contact_feedback",
        "contact": {"force_limit_N": 2.0, "speed_m_s": 0.05},
    }
    out, station = run(**kw)
    assert out["status"] == "succeeded", (out.get("failure"), out.get("error"))
    assert out["evidence"]["support_contact"]["stop_reason"] == "force_threshold"
    assert out["evidence"]["evidence_level"] == "kinematic_sim"
    out, station = run(table_z=-0.012, **kw)
    assert (out["status"], out["failure_type"]) == ("failed", "domino.support_mismatch")
