"""A synthetic, kinematic table world for the domino task's contract tests.

Not a simulator and not a robot: rigid poses moved by rules, with injectable faults.
Every observation it emits is labelled ``synthetic_contract_test``. The faults change
the *world* (where the tile and neighbours end up); the observer and classifier must
infer them from observations, never from the fault labels.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from roboskillhub.tasks.domino import geometry as geo
from roboskillhub.tasks.domino.contract import CAPABILITIES, threshold_profile

FRAME = "station"
GEOMETRY = {
    "height_m": 0.048,
    "width_m": 0.024,
    "thickness_m": 0.0075,
    "origin": "bottom_face_center",
    "yaw_symmetry_deg": 180.0,
}
T0 = 1_800_000_000.0
DOWN = [1.0, 0.0, 0.0, 0.0]  # TCP z pointing down (180 deg about x), xyzw
GRASP_HANG = 0.030  # TCP this far over the tile's bottom (place_domino.GRASP_HEIGHT)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def wall(self) -> float:
        return T0 + self.t

    def mono(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)


def tile_pose(x: float, y: float, yaw_deg: float = 90.0, z: float = 0.0) -> dict:
    return geo.pose(FRAME, [x, y, z], geo.yaw_quat(math.radians(yaw_deg)))


def t_tcp_object(dx: float = 0.0) -> dict:
    """The tile's pose in the TCP frame: TCP points down, tile bottom GRASP_HANG below it."""
    # In the TCP frame (z down), the object's up axis is -z: rotate 180 about x.
    return geo.pose("tcp", [dx, 0.0, GRASP_HANG], [1.0, 0.0, 0.0, 0.0])


def lying(p: dict) -> dict:
    """The tile fallen on its broad face, along its facing."""
    t = geo.to_matrix(p)
    r = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # +90 about local y
    t2 = t.copy()
    t2[:3, :3] = t[:3, :3] @ r
    t2[2, 3] = t[2, 3] + GEOMETRY["thickness_m"] / 2.0
    return geo.from_matrix(t2, FRAME)


class SyntheticStation:
    def __init__(
        self,
        clock: FakeClock,
        *,
        faults: set[str] | None = None,
        capabilities=None,
        grasp_dx: float = 0.0,
        table_z: float = 0.0,
        neighbors: dict | None = None,
    ):
        self.clock = clock
        self.faults = set(faults or ())
        self.capabilities = set(CAPABILITIES["geometric"] if capabilities is None else capabilities)
        self.table_z = table_z
        self.true_T = t_tcp_object(grasp_dx)
        self.target = "d2"
        self.neighbors = copy.deepcopy(
            neighbors
            if neighbors is not None
            else {"d0": tile_pose(0.40, 0.00), "d1": tile_pose(0.40, 0.04)}
        )
        self.tcp = geo.pose(FRAME, [0.30, 0.10, 0.20], DOWN)
        self.held = "missed_grasp" not in self.faults
        self.tile = None if self.held else lying(tile_pose(0.25, -0.1))
        self.calls: list[str] = []
        self.seq = 0
        self.frozen = None
        self.released = False
        self.in_lease = False

    # -- helpers
    def _tick(self, dt: float = 0.05) -> None:
        self.clock.sleep(dt)

    def tile_pose(self) -> dict:
        return geo.object_from_tcp(self.tcp, self.true_T) if self.held else self.tile

    # -- PlacementBackend
    def describe(self) -> dict:
        return {
            "name": "synthetic-domino-world",
            "mode": "synthetic_contract_test",
            "frame_id": FRAME,
            "capabilities": sorted(self.capabilities),
        }

    def begin_transaction(self, *, timeout_s: float) -> None:
        self.calls.append("begin")
        self.in_lease = True

    def end_transaction(self) -> None:
        self.calls.append("end")
        self.in_lease = False

    def move(self, pose: dict, *, linear: bool, speed_m_s: float, timeout_s: float) -> dict:
        assert self.in_lease
        self.calls.append(f"move:{'linear' if linear else 'free'}")
        target = copy.deepcopy(pose)
        if "tcp_offset" in self.faults and linear and not self.released:
            target["position_m"][0] += 0.012
        dist = float(np.linalg.norm(np.subtract(target["position_m"], self.tcp["position_m"])))
        self._tick(max(0.05, dist / speed_m_s))
        if self.held:
            obj = geo.object_from_tcp(target, self.true_T)
            low = geo.lowest_point_height(obj, GEOMETRY, [0, 0, self.table_z], [0, 0, 1])
            if low < 0:  # the tile stops on the real table
                target["position_m"][2] -= low
        self.tcp = target
        if "drop_in_transit" in self.faults and not linear and self.held:
            self.held = False
            self.tile = lying(tile_pose(0.33, 0.12, z=self.table_z))
        if self.released and linear and "retreat_knocks" in self.faults:
            self.neighbors["d1"] = lying(self.neighbors["d1"])
        if self.released and linear and "still_holding" in self.faults:
            pass
        return {"ok": True}

    def guarded_descent(self, pose, *, max_travel_m, speed_m_s, timeout_s):
        return self.move(pose, linear=True, speed_m_s=speed_m_s, timeout_s=timeout_s)

    def release(self, width_m: float, *, timeout_s: float) -> dict:
        self.calls.append("release")
        self._tick()
        if "still_holding" in self.faults:
            return {"ok": True}
        p = self.tile_pose()
        drop = geo.lowest_point_height(p, GEOMETRY, [0, 0, self.table_z], [0, 0, 1])
        rest = geo.translated(p, [0, 0, -drop])
        self.held, self.released = False, True
        self.tile = lying(rest) if (drop > 0.008 or "fall_on_release" in self.faults) else rest
        return {"ok": True}

    def get_state(self, *, timeout_s: float) -> dict:
        self._tick(0.01)
        return {"pose": copy.deepcopy(self.tcp), "timestamp_s": self.clock.wall()}

    def observe(self, *, timeout_s: float) -> dict:
        self._tick(0.1)
        if self.frozen is not None:
            return copy.deepcopy(self.frozen)
        if self.released and "fall_in_settle" in self.faults and self.tcp["position_m"][2] > 0.06:
            self.tile = lying(self.tile) if geo.tilt_rad(self.tile, [0, 0, 1]) < 0.1 else self.tile
        self.seq += 1
        tile = self.tile_pose()
        objects = {
            self.target: {"pose": copy.deepcopy(tile), "identity_confidence": 0.95, "visible": True}
        }
        if "no_orientation" in self.faults and self.released:
            objects[self.target]["pose"] = None
        if "wrong_identity" in self.faults and self.released:
            objects[self.target]["identity_confidence"] = 0.2
        for k, p in self.neighbors.items():
            objects[k] = {"pose": copy.deepcopy(p), "identity_confidence": 0.95, "visible": True}
        obs = {
            "sample_id": f"syn-{self.seq}",
            "timestamp_s": self.clock.wall(),
            "frame_id": FRAME,
            "source": "synthetic-domino-world",
            "source_kind": "synthetic_contract_test",
            "objects": objects,
            "gripper": {"holding": self.held, "source": "synthetic-jaw-gap"},
            "tcp_pose": copy.deepcopy(self.tcp),
        }
        if "observe_support" in self.faults:
            obs["support_plane"] = {
                "point_m": [0.0, 0.0, self.table_z],
                "normal": [0, 0, 1],
                "source": "synthetic-depth-plane",
            }
        if "gripper_unknown" in self.faults:
            obs["gripper"] = {"holding": None, "source": "synthetic-jaw-gap"}
        if ("stale_obs" in self.faults and self.released) or (
            "repeated_frames" in self.faults and self.released and self.tcp["position_m"][2] > 0.06
        ):
            self.frozen = copy.deepcopy(obs)
            if "stale_obs" in self.faults:
                self.frozen["sample_id"] = "frozen"
        return obs

    def stop(self) -> None:
        self.calls.append("stop")


def request(clock: FakeClock, **over) -> dict:
    now = clock.wall()
    req = {
        "request_id": "req-1",
        "scene_version": "synthetic-row-v1",
        "frame_id": FRAME,
        "target_id": "d2",
        "geometry": dict(GEOMETRY),
        "target_object_pose": tile_pose(0.40, 0.08),
        "holding": {
            "object_id": "d2",
            "gripper_id": "left",
            "T_tcp_object": t_tcp_object(),
            "position_std_m": 0.001,
            "orientation_std_rad": 0.02,
            "source": "agent01.synthetic_holding_estimate",
            "timestamp_s": now,
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
            {"id": "d0", "pose": tile_pose(0.40, 0.00), "timestamp_s": now, "source": "synthetic"},
            {"id": "d1", "pose": tile_pose(0.40, 0.04), "timestamp_s": now, "source": "synthetic"},
        ],
        "approach": {"distance_m": 0.05},
        "retreat": {"distance_m": 0.06},
        "hand": {"max_open_width_m": 0.094, "tip_reach_m": 0.0102},
        "workspace": {"min_m": [0.0, -0.5, 0.0], "max_m": [0.8, 0.5, 0.5]},
        "mode": "geometric",
        "speed_m_s": 0.1,
        "timeout_s": 60.0,
        "max_recovery_attempts": 2,
        "thresholds": threshold_profile(),
    }
    req.update(over)
    return req
