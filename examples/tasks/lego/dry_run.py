"""Offline dry run of lego.align_connection -> lego.guarded_seat -> lego.verify_connection on
scripted fakes. Evidence level: unit_or_dryrun. No robot, simulator or network.

    PYTHONPATH=src python examples/tasks/lego/dry_run.py [--case seated|position_only|lifted]
"""

from __future__ import annotations

import argparse
import json

from roboskillhub.tasks.lego import align_connection, guarded_seat, verify_connection
from roboskillhub.tasks.lego import geometry as g
from roboskillhub.tasks.lego.fakes import (
    FakeClock,
    FakeContact,
    FakeMotion,
    FakeObserver,
    press_trace,
)

W = "world"
DOWN = [1.0, 0.0, 0.0, 0.0]


def o(value, t, source, frame=W, **kw):
    return {
        "obs_id": f"{source}@{t:.2f}",
        "source": source,
        "kind": "measured",
        "frame_id": frame,
        "timestamp_s": t,
        "value": value,
        "uncertainty_m": 0.0001,
        "uncertainty_deg": 0.05,
        **kw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="seated", choices=["seated", "position_only", "lifted"])
    case = ap.parse_args().case
    clock = FakeClock()
    t = clock()
    plate = {
        "plate_id": "baseplate",
        "pose": {"frame_id": W, "position_m": [0.3, 0.0, 0.02], "quaternion_xyzw": [0, 0, 0, 1]},
        "studs": [16, 16],
        "system": "lego",
        "calibration_id": "example-cal",
    }
    t_tcp_obj = {"frame_id": "tcp", "position_m": [0, 0, 0.02], "quaternion_xyzw": DOWN}
    tcp = {"frame_id": W, "position_m": [0.25, 0.05, 0.15], "quaternion_xyzw": DOWN}
    align_req = {
        "request_id": "ex-align",
        "connection_id": "ex-conn",
        "held_brick_id": "b1",
        "reference_id": "baseplate",
        "brick": {"system": "lego", "studs": "2x4"},
        "plate": plate,
        "target_cell": {"x": 4, "y": 4, "layer": 1, "rotation": 0},
        "build": [],
        "held": {
            "object_id": "b1",
            "gripper_id": "left",
            "grasp_id": "g1",
            "close_axis": 0,
            "T_TCP_object": t_tcp_obj,
            "holding_evidence": o(True, t, "grasp-check"),
        },
        "tcp_observation": o(tcp, t, "fk"),
        "limits": {
            "xy_tolerance_m": 0.0004,
            "yaw_tolerance_deg": 1.0,
            "workspace_min_m": [0, -0.5, 0],
            "workspace_max_m": [0.8, 0.5, 0.6],
            "limits_source": "example (not a station calibration)",
        },
    }
    motion = FakeMotion(clock)
    grasp_offset = g.to_pose(g.mul(g.matrix(t_tcp_obj), g.tr(0.001, 0, 0)), "tcp")
    a = align_connection(align_req, motion, FakeObserver(clock, motion, grasp_offset), clock=clock)
    print("align ", a["status"], a["reason_code"], "corrections", a["retries"])

    contact = FakeContact(samples=press_trace(0.01, g.SYSTEMS["lego"].stud_height_m))
    if case == "position_only":
        contact = FakeContact(
            capabilities=["pose_feedback", "bounded_position_approach"], controller_kind="position"
        )
    s = guarded_seat(
        {
            "request_id": "ex-seat",
            "connection_id": "ex-conn",
            "system": "lego",
            "alignment": a,
            "axis": a["evidence"]["plan"]["engagement_axis"],
            "max_travel_m": 0.0105,
            "speed_m_s": 0.005,
            "timeout_s": 5,
            "force_limit_N": 15,
            "limits_source": "example",
        },
        contact,
    )
    print("seat  ", s["status"], s["reason_code"], "|", s["message"])
    if s["status"] != "succeeded":
        return

    released = clock()
    seat = a["evidence"]["plan"]["seat_pose"]
    dz = 0.0012 if case == "lifted" else 0.0001
    samples = []
    for _i in range(3):
        clock.advance(0.3)
        p = g.to_pose(g.mul(g.matrix(seat), g.tr(0, 0, dz)), W)
        samples.append(o(p, clock(), "overhead-depth"))
    v = verify_connection(
        {
            "request_id": "ex-verify",
            "connection_id": "ex-conn",
            "brick_id": "b1",
            "brick": align_req["brick"],
            "plate": plate,
            "target_cell": align_req["target_cell"],
            "executor_id": "fake-contact",
            "released_at_s": released,
            "release": o(
                {"gripper_open": True, "retreated": True}, released + 0.05, "gripper", frame="tcp"
            ),
            "pre_release_observation": o(seat, released - 0.1, "overhead-depth"),
            "observations": samples,
        },
        clock=clock,
    )
    print("verify", v["status"], v["reason_code"], "| claim:", v["evidence"]["claim"])
    print(json.dumps({"findings": v["evidence"]["findings"]}, indent=1))


if __name__ == "__main__":
    main()
