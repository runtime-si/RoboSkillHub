"""A synthetic, kinematic table world and a fake rsh-task/1 task context for the domino
contract tests.

Not a simulator and not a robot: rigid poses moved by rules, with injectable faults.
Every observation it emits has ``source: "synthetic:kinematic-world"`` (tier
synthetic_contract_test). The faults change the *world* (where tiles end up, what the
gripper does); the skill under test must infer them from observations and primitive
results, never from the fault labels. ``FakeTaskContext`` mimics the primitives Agent 05
specified (move_waypoints, set_gripper, observe_objects) closely enough to exercise the
composition; it is not the runtime.
"""

from __future__ import annotations

import copy
import math
import time
import uuid

import numpy as np

from roboskillhub.tasks.domino import geometry as geo
from roboskillhub.tasks.domino.contract import CAPABILITIES

FRAME = "station"
H, W, T = 0.048, 0.024, 0.0075
GEOMETRY = {"size_m": [H, W, T], "yaw_symmetry_deg": 180.0}
SIZE = {"height_m": H, "width_m": W, "thickness_m": T, "origin": "geometric_center"}
T0 = 1_800_000_000.0
DOWN = [1.0, 0.0, 0.0, 0.0]  # TCP z pointing down (180 deg about x), xyzw
GRASP_HANG = 0.030  # TCP this far over the tile's bottom (place_domino.GRASP_HEIGHT)
SOURCE = "synthetic:kinematic-world"


def tile_pose(x: float, y: float, yaw_deg: float = 90.0, table_z: float = 0.0) -> dict:
    """A standing tile's geometric-centre pose."""
    return geo.pose(FRAME, [x, y, table_z + H / 2], geo.yaw_quat(math.radians(yaw_deg)))


def t_tcp_object(dx: float = 0.0) -> dict:
    """The tile's pose in the TCP frame: TCP z points down, the tile's bottom GRASP_HANG
    below the TCP, so its centre GRASP_HANG - H/2 below."""
    return geo.pose("tcp", [dx, 0.0, GRASP_HANG - H / 2], [1.0, 0.0, 0.0, 0.0])


def lying(p: dict, table_z: float = 0.0) -> dict:
    """The tile fallen on its broad face, along its facing."""
    t = geo.to_matrix(p)
    r = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # +90 deg about local y
    t2 = t.copy()
    t2[:3, :3] = t[:3, :3] @ r
    t2[:3, 3] += t[:3, :3] @ np.array([H / 2, 0, 0])
    t2[2, 3] = table_z + T / 2
    return geo.from_matrix(t2, FRAME)


def observation(oid: str, p: dict | None, t: float, **kw) -> dict:
    rec = {
        "object_id": oid,
        "class": "domino",
        "frame_id": FRAME,
        "position_m": None if p is None else list(p["position_m"]),
        "quaternion_xyzw": None if p is None else list(p["quaternion_xyzw"]),
        "timestamp_s": t,
        "source": SOURCE,
        "privileged": False,
        "confidence": 0.95,
        "uncertainty": {"position_m": 0.001, "rotation_rad": 0.01},
        "visibility": "visible",
        "state_label": None,
    }
    rec.update(kw)
    return rec


class FakeTaskContext:
    def __init__(
        self,
        *,
        faults: set[str] | None = None,
        capabilities=None,
        grasp_dx: float = 0.0,
        table_z: float = 0.0,
        cancel_after_s: float | None = None,
        real_time: bool = False,
    ):
        self.t = 0.0
        self.real_time = real_time
        self.task_id, self.request_id = "task-1", "task-1/n1/a1"
        self.privileged_allowed = False
        self.faults = set(faults or ())
        self.capabilities = frozenset(
            CAPABILITIES["geometric"] if capabilities is None else capabilities
        )
        self.table_z = table_z
        self.true_T = t_tcp_object(grasp_dx)
        self.target = "d2"
        self.neighbors = {"d0": tile_pose(0.40, 0.00), "d1": tile_pose(0.40, 0.04)}
        self.tcp = geo.pose(FRAME, [0.30, 0.10, 0.20], DOWN)
        self.held = "missed_grasp" not in self.faults
        self.tile = None if self.held else lying(tile_pose(0.25, -0.1))
        self.calls: list[str] = []
        self.requests: list[tuple[str, dict]] = []
        self.released = False
        self.frozen_t: float | None = None
        self.cancel_after_s = cancel_after_s

    # -- clock
    def now(self) -> float:
        return time.time() if self.real_time else T0 + self.t

    def sleep(self, dt: float) -> None:
        if self.real_time:
            time.sleep(max(0.0, dt))
        self.t += max(0.0, dt)

    def cancelled(self) -> bool:
        return self.cancel_after_s is not None and self.t > self.cancel_after_s

    # -- world rules
    def tile_pose(self) -> dict:
        return geo.object_from_tcp(self.tcp, self.true_T) if self.held else self.tile

    def _move(self, pose: dict, speed: float) -> None:
        target = copy.deepcopy(pose)
        leaving_start = self.tcp["position_m"][2] > 0.15
        if "tcp_offset" in self.faults and not self.released and target["position_m"][2] < 0.06:
            target["position_m"][0] += 0.012
        self.t += max(
            0.02,
            float(np.linalg.norm(np.subtract(target["position_m"], self.tcp["position_m"])))
            / speed,
        )
        if self.held:
            low = geo.lowest_point_height(
                geo.object_from_tcp(target, self.true_T), SIZE, [0, 0, self.table_z], [0, 0, 1]
            )
            if low < 0:  # the held tile stops on the real table
                target["position_m"][2] -= low
        self.tcp = target
        if "drop_in_transit" in self.faults and self.held and leaving_start:
            self.held = False
            self.tile = lying(tile_pose(0.33, 0.12), self.table_z)
        if self.released and "retreat_knocks" in self.faults:
            self.neighbors["d1"] = lying(tile_pose(0.40, 0.04), self.table_z)

    # -- rsh-task/1 primitives (as Agent 05 specified them)
    def invoke(self, capability: str, request: dict) -> dict:
        self.calls.append(capability)
        self.requests.append((capability, copy.deepcopy(request)))
        out = {
            "invocation_id": str(uuid.uuid4()),
            "capability": capability,
            "status": "succeeded",
            "failure_type": None,
            "evidence": {},
        }
        if capability == "move_waypoints":
            if request.get("linear") and "linear_cartesian_motion" not in self.capabilities:
                return {**out, "status": "rejected", "failure_type": "capability_missing"}
            for wp in request["waypoints"]:
                self._move(wp, float(request.get("speed_m_s", 0.03)))
            self.t += 0.01
            err = float(
                np.linalg.norm(
                    np.subtract(self.tcp["position_m"], request["waypoints"][-1]["position_m"])
                )
            )
            out["evidence"] = {
                "measured_pose": copy.deepcopy(self.tcp),
                "state_timestamp_s": self.now(),
                "position_error_m": err,
            }
            if err > 0.010:
                return {**out, "status": "failed", "failure_type": "motion_not_converged"}
            held = request.get("held")
            if held:
                obj = self.tile_pose()
                pred = geo.object_from_tcp(self.tcp, held["T_TCP_object"])
                if (
                    not self.held
                    or float(np.linalg.norm(np.subtract(obj["position_m"], pred["position_m"])))
                    > 0.02
                ):
                    return {**out, "status": "failed", "failure_type": "held_object_slipped"}
            return out
        if capability == "set_gripper":
            assert request["purpose"] == "release"
            reported = self.release_to(request["width_m"])
            out["evidence"] = {
                "purpose": "release",
                "gripper_feedback": {
                    "holding": reported,
                    "width_m": request["width_m"],
                    "timestamp_s": self.now(),
                    "source": "synthetic:jaw-gap",
                },
            }
            if reported:
                return {**out, "status": "failed", "failure_type": "release_failed"}
            return out
        raise AssertionError(f"unexpected primitive {capability}")

    def release_to(self, width_m: float) -> bool:
        """Open to ``width_m``; returns what the jaw sensor reports as holding."""
        self.t += 0.2
        if "still_holding" not in self.faults:
            p = self.tile_pose()
            drop = geo.lowest_point_height(p, SIZE, [0, 0, self.table_z], [0, 0, 1])
            rest = geo.translated(p, [0, 0, -drop])
            self.tile = (
                lying(rest, self.table_z)
                if (drop > 0.008 or "fall_on_release" in self.faults)
                else rest
            )
            self.released = True
            self.held = "sticky_release" in self.faults
        return self.held and "sticky_release" not in self.faults

    def observe(self, object_ids, *, require_orientation: bool = False) -> dict:
        """Like SkillContext.observe: the observe_objects result, records in evidence."""
        self.calls.append("observe_objects")
        records = self._records(list(object_ids))
        bad = [
            r
            for r in records
            if r["visibility"] != "visible"
            or (require_orientation and r["quaternion_xyzw"] is None)
        ]
        return {
            "invocation_id": str(uuid.uuid4()),
            "capability": "observe_objects",
            "status": "unverified" if bad else "succeeded",
            "failure_type": (
                "object_not_observed" if bad[0]["visibility"] != "visible" else "evidence_missing"
            )
            if bad
            else None,
            "evidence": {"observations": records},
        }

    def _records(self, object_ids) -> list[dict]:
        self.t += 0.1
        t = self.now()
        if self.released and (
            "stale_obs" in self.faults
            or ("repeated_frames" in self.faults and self.tcp["position_m"][2] > 0.06)
        ):
            self.frozen_t = self.frozen_t or t
            t = self.frozen_t
        if (
            self.released
            and "fall_in_settle" in self.faults
            and self.tcp["position_m"][2] > 0.06
            and geo.tilt_rad(self.tile, [0, 0, 1]) < 0.1
        ):
            self.tile = lying(self.tile, self.table_z)
        out = []
        for oid in object_ids:
            if oid == self.target:
                rec = observation(oid, self.tile_pose(), t)
                if self.released and "no_orientation" in self.faults:
                    rec["quaternion_xyzw"] = None
                if self.released and "wrong_identity" in self.faults:
                    rec["confidence"] = 0.2
            elif oid in self.neighbors:
                rec = observation(oid, self.neighbors[oid], t)
            elif oid == "table":
                rec = observation(
                    "table",
                    geo.pose(FRAME, [0.4, 0.08, self.table_z], [0, 0, 0, 1]),
                    t,
                    **{"class": "support"},
                )
            else:
                rec = observation(oid, None, t, visibility="not_found")
            out.append(rec)
        return out

    def state(self) -> dict:
        self.t += 0.01
        holding = (
            None
            if "gripper_unknown" in self.faults
            else self.held and not ("sticky_release" in self.faults and self.released)
        )
        return {
            "tcp_pose": copy.deepcopy(self.tcp),
            "timestamp_s": self.now(),
            "gripper": {
                "holding": holding,
                "timestamp_s": self.now(),
                "source": "synthetic:jaw-gap",
            },
        }


def request(ctx: FakeTaskContext, **over) -> dict:
    now = ctx.now()
    req = {
        "request_id": "req-1",
        "scene_version": "synthetic-row-v1",
        "frame_id": FRAME,
        "target_id": "d2",
        "geometry": dict(GEOMETRY),
        "target_object_pose": tile_pose(0.40, 0.08),
        "held": {
            "object_id": "d2",
            "gripper_id": "left",
            "T_TCP_object": t_tcp_object(),
            "estimate": "grasp_geometry_prior",
            "timestamp_s": now,
            "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
            "collision_geometry": {"type": "box", "size_m": [H, W, T]},
            "valid": True,
            "invalidated_reason": None,
        },
        "support": {
            "frame_id": FRAME,
            "point_m": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "height_std_m": 0.001,
            "calibration_version": "table-cal-synthetic-1",
            "region_min_m": [0.1, -0.3],
            "region_max_m": [0.7, 0.3],
        },
        "neighbors": [
            observation("d0", tile_pose(0.40, 0.00), now),
            observation("d1", tile_pose(0.40, 0.04), now),
        ],
        "retreat": {"distance_m": 0.06},
        "hand": {"max_open_width_m": 0.094, "tip_reach_m": 0.0102},
        "workspace": {"min_m": [0.0, -0.5, 0.0], "max_m": [0.8, 0.5, 0.5]},
        "speed_m_s": 0.1,
        "timeout_s": 60.0,
        "max_recovery_attempts": 2,
    }
    req.update(over)
    return req
