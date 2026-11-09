"""A seeded kinematic fake of the domino line scene, for graph semantics only.

Nothing here is physics: a piece "stands" by rule, faults are drawn from fixed
probabilities, and every observation is the fake's own truth (``privileged``).
Results from this world test interpreter, recovery and search plumbing; they
are not simulation or hardware results and are labelled so in ``describe``.
"""

from __future__ import annotations

import math
import random
import time

from roboskillhub.runtime.contracts import Limits, compose, invert

SIZE = (0.048, 0.024, 0.0075)
"""height, width, thickness (m)."""
UPRIGHT = [0.0, 0.0, 0.0, 1.0]
TOPPLED = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]
DESCRIBE = {
    "kind": "fake-kinematic",
    "physics": False,
    "observations": "fake world truth, privileged",
    "use": "graph semantics and search plumbing only; not simulation-validated",
}


def _pose(x, y, z, q=UPRIGHT):
    return {"frame_id": "world", "position_m": [x, y, z], "quaternion_xyzw": list(q)}


class FakeDominoStation:
    def __init__(self, seed: int, *, slots: int = 5, p_miss=0.1, p_slip=0.002, p_topple=0.1):
        self.rng = random.Random(seed)
        self.p = {"miss": p_miss, "slip": p_slip, "topple": p_topple}
        self.tcp = _pose(0.3, 0.0, 0.2)
        self.width = 0.08
        self.held: str | None = None
        self.T_TCP_held = None
        self.speed = 0.03
        self.pieces = {
            f"p{i}": _pose(0.25, -0.15 + 0.03 * i, SIZE[0] / 2) for i in range(slots + 4)
        }
        self.standing = set(self.pieces)
        self.interventions = 0
        """Toppled pieces the fake clears out of the line by itself; a person would have to."""
        self.slots = [
            {
                "slot_id": f"s{i}",
                "order": i,
                "pose": _pose(0.36 + 0.03 * i, 0.0, SIZE[0] / 2),
                "neighbours": [f"s{j}" for j in (i - 1, i + 1) if 0 <= j < slots],
            }
            for i in range(slots)
        ]

    def describe(self):
        return {
            "frame_id": "world",
            "backend": "fake-domino",
            **DESCRIBE,
            "capabilities": [
                "pose_feedback",
                "cartesian_motion",
                "gripper",
                "gripper_feedback",
                "stop",
                "object_feedback",
                "object_pose_feedback",
            ],
        }

    def get_state(self, *, timeout_s):
        now = time.time()
        if self.held is not None:
            self.pieces[self.held] = compose(self.tcp, self.T_TCP_held)
        objects = {
            name: {
                "position_m": list(p["position_m"]),
                "quaternion_xyzw": list(p["quaternion_xyzw"]),
                "frame_id": "world",
                "timestamp_s": now,
                "source": "fake_world_truth",
                "privileged": True,
                "confidence": 1.0,
                "class": "domino",
                "geometry": {"size_m": list(SIZE)},
                "state_label": "standing" if name in self.standing else "toppled",
            }
            for name, p in self.pieces.items()
        }
        gripper = {
            "holding": self.held is not None,
            "width_m": self.width,
            "source": "fake_world_truth",
            "timestamp_s": now,
        }
        return {"pose": self.tcp, "timestamp_s": now, "objects": objects, "gripper": gripper}

    def move_pose(self, target, *, speed_m_s, timeout_s, linear):
        self.speed = speed_m_s
        self.tcp = target
        if self.held is not None and self.rng.random() < self.p["slip"]:
            self._drop(self.held)
        return {"ok": True}

    def _drop(self, name):
        p = compose(self.tcp, self.T_TCP_held)["position_m"]
        self.pieces[name] = _pose(p[0], p[1], SIZE[2] / 2, TOPPLED)
        self.standing.discard(name)
        self.held = None

    def set_gripper(self, width_m, *, timeout_s):
        self.width = width_m
        if width_m > SIZE[1] + 0.002 and self.held is not None:
            self._release(self.held)
        elif width_m < SIZE[1] and self.held is None:
            for name, pose in self.pieces.items():
                close = math.dist(pose["position_m"], self.tcp["position_m"]) < 0.02
                if close and name in self.standing and self.rng.random() >= self.p["miss"]:
                    self.held = name
                    self.T_TCP_held = compose(
                        invert(self.tcp, frame_id="tcp"), pose, frame_id="tcp"
                    )
                    self.width = SIZE[1]
                    break
        return {"ok": True}

    def _release(self, name):
        pose = compose(self.tcp, self.T_TCP_held)
        self.held = None
        bottom = pose["position_m"][2] - SIZE[0] / 2
        risk = self.p["topple"] * max(1.0, self.speed / 0.02)
        crowded = [
            other
            for other in self.standing - {name}
            if math.dist(self.pieces[other]["position_m"], pose["position_m"]) < SIZE[2] * 1.5
        ]
        if abs(bottom) > 0.003 or crowded or self.rng.random() < risk:
            for fallen in [name, *crowded]:
                x, y, _ = (pose if fallen == name else self.pieces[fallen])["position_m"]
                self.pieces[fallen] = _pose(x, y - 0.08, SIZE[2] / 2, TOPPLED)
                self.standing.discard(fallen)
                self.interventions += 1
        else:
            pose["position_m"][2] = SIZE[0] / 2
            self.pieces[name] = pose

    def stop(self):
        pass

    def task_success(self) -> bool:
        """Independent check from fake truth: every slot has a standing piece on it."""
        for slot in self.slots:
            target = slot["pose"]["position_m"]
            if not any(
                math.dist(self.pieces[n]["position_m"], target) < 0.005 for n in self.standing
            ):
                return False
        return True


def limits() -> Limits:
    return Limits.from_dict(
        {
            "frame_id": "world",
            "workspace_min_m": [0.0, -0.5, 0.0],
            "workspace_max_m": [0.8, 0.5, 0.5],
            "max_speed_m_s": 0.1,
            "max_contact_distance_m": 0.3,
            "gripper_min_width_m": 0.0,
            "gripper_max_width_m": 0.08,
        }
    )


def domino_world(seed: int, *, slots: int = 5, **faults) -> dict:
    station = FakeDominoStation(seed, slots=slots, **faults)
    inputs = {
        "layout": {
            "kind": "domino_layout",
            "layout_id": f"line{slots}",
            "frame_id": "world",
            "piece": {"class": "domino", "size_m": list(SIZE)},
            "slots": station.slots,
        },
        "feeder": {"pieces": sorted(station.pieces, key=lambda n: int(n[1:]))},
        "gripper_id": "fake",
        "table_z": 0.0,
        "grasp": {
            "T_object_TCP": {
                "frame_id": "object",
                "position_m": [0.0, 0.0, 0.01],
                "quaternion_xyzw": UPRIGHT,
            },
            "pregrasp_height_m": 0.05,
            "lift_height_m": 0.06,
            "open_width_m": 0.04,
            "close_width_m": 0.02,
            "uncertainty": {"position_m": 0.001, "rotation_rad": 0.02},
            "collision_geometry": {"type": "box", "size_m": list(SIZE)},
        },
        "tolerances": {"max_drift_m": 0.005},
        "place": {
            "geometry": {"size_m": list(SIZE), "yaw_symmetry_deg": 180.0},
            "support": {
                "frame_id": "world",
                "point_m": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "height_std_m": 0.001,
                "calibration_version": "fake-world-plane",
                "region_min_m": [0.1, -0.3],
                "region_max_m": [0.7, 0.3],
            },
            "retreat": {"distance_m": 0.06},
            "hand": {"max_open_width_m": 0.08, "tip_reach_m": 0.0102},
            "workspace": {"min_m": [0.0, -0.5, 0.0], "max_m": [0.8, 0.5, 0.5]},
            "timeout_s": 60.0,
            "tolerances_ref": "tasks/domino/config/line_v1.yaml",
        },
    }
    return {
        "backend": station,
        "limits": limits(),
        "inputs": inputs,
        "describe": {**DESCRIBE, "seed": seed, "faults": station.p, "slots": slots},
        "task_success": station.task_success,
        # Every observation here is the fake's truth; tools that refuse privileged
        # input must be told so explicitly, and episodes carry privileged_inputs.
        "privileged_allowed": True,
        "interventions": lambda: station.interventions,
    }
