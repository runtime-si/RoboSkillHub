"""Request builders for the tests/tasks/test_lego_* files (unit_or_dryrun only)."""

from __future__ import annotations

import copy
from typing import Any

from roboskillhub.tasks.lego import geometry as g
from roboskillhub.tasks.lego.fakes import FakeClock

WORLD = "world"
DOWN = [1.0, 0.0, 0.0, 0.0]  # tool z down: Rx(180)
PLATE_POSE = {"frame_id": WORLD, "position_m": [0.30, 0.0, 0.02], "quaternion_xyzw": [0, 0, 0, 1]}
T_TCP_OBJECT = {"frame_id": "tcp", "position_m": [0.0, 0.0, 0.02], "quaternion_xyzw": DOWN}


def plate(system: str = "lego", studs=(16, 16), pose: dict | None = None) -> dict[str, Any]:
    return {
        "plate_id": "baseplate",
        "pose": copy.deepcopy(pose or PLATE_POSE),
        "studs": list(studs),
        "system": system,
        "calibration_id": "cal-test-1",
    }


def obs(
    value: Any,
    t: float,
    *,
    source: str = "wrist-cam",
    kind: str = "measured",
    sigma: float = 0.0001,
    frame_id: str = WORLD,
    obs_id: str | None = None,
    **kw: Any,
) -> dict[str, Any]:
    return {
        "obs_id": obs_id or f"{source}-{t:.3f}",
        "source": source,
        "kind": kind,
        "frame_id": frame_id,
        "timestamp_s": t,
        "value": value,
        "uncertainty_m": sigma,
        "uncertainty_deg": kw.pop("sigma_deg", 0.05),
        **kw,
    }


def align_request(
    clock: FakeClock,
    *,
    system: str = "lego",
    studs="2x4",
    cell: dict | None = None,
    build: list | None = None,
    tcp_yaw_deg: float = 0.0,
    **over: Any,
) -> dict[str, Any]:
    t = clock()
    tcp = g.to_pose(
        g.mul(
            g.tr(0.25, 0.05, 0.15),
            g.mul(
                g.rz(tcp_yaw_deg),
                g.matrix({"frame_id": WORLD, "position_m": [0, 0, 0], "quaternion_xyzw": DOWN}),
            ),
        ),
        WORLD,
    )
    req = {
        "request_id": "req-1",
        "connection_id": "conn-1",
        "held_brick_id": "b1",
        "reference_id": "baseplate",
        "brick": {"system": system, "studs": studs},
        "plate": plate(system),
        "target_cell": cell or {"x": 4, "y": 4, "layer": 1, "rotation": 0},
        "build": build or [],
        "held": {
            "object_id": "b1",
            "gripper_id": "left",
            "grasp_id": "g-1",
            "close_axis": 0,
            "T_TCP_object": copy.deepcopy(T_TCP_OBJECT),
            "holding_evidence": obs(True, t - 0.1, source="gripper-width-and-wrist-cam"),
        },
        "tcp_observation": obs(tcp, t - 0.05, source="joint-encoders-fk"),
        "limits": {
            "xy_tolerance_m": 0.0004 if system == "lego" else 0.0008,
            "yaw_tolerance_deg": 1.0,
            "workspace_min_m": [0.0, -0.5, 0.0],
            "workspace_max_m": [0.8, 0.5, 0.6],
            "limits_source": "test-config",
            "max_corrections": 3,
            "max_total_correction_m": 0.006,
        },
    }
    req.update(over)
    return req


def placed(
    brick_id: str,
    x: int,
    y: int,
    layer: int = 1,
    rotation: int = 0,
    studs="2x4",
    system: str = "lego",
) -> dict[str, Any]:
    return {
        "brick_id": brick_id,
        "system": system,
        "studs": studs,
        "cell": {"x": x, "y": y, "layer": layer, "rotation": rotation},
    }


def seat_of(system: str, studs: str, cell: dict, pose: dict | None = None) -> dict:
    p = g.Plate(pose or PLATE_POSE, (16, 16), g.SYSTEMS[system], "cal-test-1")
    return g.seat_pose(p, g.brick_type(system, studs), g.Cell.from_dict(cell))


def mating_target(
    target_id: str,
    x: int,
    y: int,
    layer0: int = 0,
    quarter: int = 0,
    *,
    system: str = "lego",
    studs=(2, 4),
    depends_on=(),
    **over,
) -> dict:
    """Contract LegoMatingTarget (layer 0 on the plate), mating_pose from the placement."""
    cell = {"x": x, "y": y, "layer": layer0 + 1, "rotation": 90 * quarter}
    t = {
        "kind": "lego_mating",
        "target_id": target_id,
        "frame_id": WORLD,
        "baseplate": {
            "object_id": "baseplate",
            "T_W_baseplate": copy.deepcopy(PLATE_POSE),
            "stud_pitch_m": g.SYSTEMS[system].stud_pitch_m,
            "studs": [16, 16],
            "calibration_id": "cal-test-1",
        },
        "brick": {"class": "lego_brick", "part": "3001", "studs": list(studs)},
        "placement": {"stud_x": x, "stud_y": y, "layer": layer0, "yaw_quarter_turns": quarter},
        "mating_pose": seat_of(system, f"{studs[0]}x{studs[1]}", cell),
        "depends_on": list(depends_on),
        "requires_support": layer0 > 0,
        "tolerances_ref": "tests",
    }
    t.update(over)
    return t


def held_object(t: float, object_id: str = "b1", **over) -> dict:
    h = {
        "object_id": object_id,
        "gripper_id": "left",
        "T_TCP_object": copy.deepcopy(T_TCP_OBJECT),
        "estimate": "grasp_geometry_prior",
        "timestamp_s": t,
        "valid": True,
        "invalidated_reason": None,
    }
    h.update(over)
    return h


def shifted(pose: dict, *, dx=0.0, dy=0.0, dz=0.0, yaw=0.0, tilt_x=0.0) -> dict:
    """``pose`` moved in its own axes, turned about its z and tilted about its x (deg)."""
    import math

    c, s = math.cos(math.radians(tilt_x)), math.sin(math.radians(tilt_x))
    rx = [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
    m = g.mul(g.mul(g.mul(g.matrix(pose), g.tr(dx, dy, dz)), g.rz(yaw)), rx)
    return g.to_pose(m, pose["frame_id"])
