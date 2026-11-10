# SPDX-License-Identifier: Apache-2.0
"""Run hub skill bundles (``skills/<name>/scripts/*.py``, gap ``NodeContext`` scripts) as contract
``rsh-task/1`` TaskSkills on a workcell runtime.

``GapTools`` answers the ``robot.*`` tools those scripts call with the runtime's primitives through the
``SkillContext`` (``move_pose``, ``set_gripper``, ``observe_objects``, the state read) - nothing else:

- ``robot.get_ee_pose`` / ``robot.get_observation`` read the measured TCP and gripper; the observation
  carries a hand-mounted depth camera (optical axis along the TCP's approach, ``CAMERA_IN_TCP``) whose image
  is ray-cast (``depth.render``) from the observed objects, with D405 noise, when the station has no camera
  of its own;
- ``robot.go_to_pose`` is ``move_pose`` (a failed move routes ``timeout`` / ``unreachable``, as the scripts
  expect from a simulator);
- ``robot.solve_ik`` accepts top-down poses only (the floating hand serves nothing else); no IK runs;
- ``robot.get_gripper`` / ``describe_gripper`` / ``open_gripper`` map to the measured jaw width and
  ``set_gripper(release)``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from . import depth as D

REPO = Path(__file__).resolve().parents[4]
SKILLS_DIR = REPO / "skills"
MAX_OPENING_M = 0.05
"""The LEGO physics station's floating hand (``lego_station_isaac.HandSpec.max_opening``)."""
CAMERA_IN_TCP = np.array(
    [[1.0, 0.0, 0.0, 0.05], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.08], [0.0, 0.0, 0.0, 1.0]]
)
"""Camera in the TCP frame: looking along the TCP's approach (+z, down for a top-down hand), 5 cm to the
side of the TCP and 8 cm behind it - the YAM D405's offsets without its 30 deg tilt (the floating hand
serves top-down poses only, so a tilted camera could never look straight down)."""
TOP_DOWN_COS = math.cos(math.radians(1.0))


def _ensure_gap() -> None:
    """The scripts import ``gap.NodeContext`` for a type hint only; stub it where gap is not installed."""
    try:
        import gap  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("gap")
        stub.NodeContext = object  # type: ignore[attr-defined]
        sys.modules["gap"] = stub


def load_script(bundle: str, script: str):
    _ensure_gap()
    path = SKILLS_DIR / bundle / "scripts" / script
    name = f"rsh_hub_{bundle.replace('-', '_')}_{Path(script).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no hub skill script at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gap_pose(pose: dict) -> dict:
    p, q = pose["position_m"], pose["quaternion_xyzw"]
    return {
        "position": dict(zip("xyz", map(float, p), strict=True)),
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _contract_pose(pose: dict) -> dict:
    p, r = pose["position"], pose["rotation"]
    return {
        "frame_id": "world",
        "position_m": [float(p[k]) for k in "xyz"],
        "quaternion_xyzw": [float(r[k]) for k in ("x", "y", "z", "w")],
    }


def _matrix(pose: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = D.quat_matrix(pose["quaternion_xyzw"])
    m[:3, 3] = pose["position_m"]
    return m


def _quat_xyzw(r: np.ndarray) -> list[float]:
    t = float(np.trace(r))
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        w, x, y, z = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x, y, z = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = 2 * math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x, y, z = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = 2 * math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x, y, z = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        )
    return [float(x), float(y), float(z), float(w)]


class GapTools:
    """``tool(name, **kw)`` for hub scripts over a ``SkillContext``."""

    def __init__(
        self,
        ctx: Any,
        object_ids: list[str],
        *,
        camera: bool = True,
        seed: int = 0,
        speed_m_s: float = 0.02,
        timeout_s: float = 10.0,
    ):
        self.ctx, self.object_ids = ctx, list(object_ids)
        self.camera = camera
        self.rng = np.random.default_rng(seed)
        self.speed, self.timeout = speed_m_s, timeout_s
        self.calls: list[str] = []
        self.renders = 0
        self.observations: dict[str, dict] = {}
        self.watch: str | None = None
        self.watch_log: list[dict] = []

    # gap's NodeContext surface used by the scripts
    def tool(self, name: str, **kw: Any) -> Any:
        self.calls.append(name)
        fn = {
            "robot.get_ee_pose": self._ee_pose,
            "robot.get_observation": self._observation,
            "robot.go_to_pose": self._go_to_pose,
            "robot.solve_ik": self._solve_ik,
            "robot.get_gripper": self._gripper,
            "robot.describe_gripper": self._describe_gripper,
            "robot.open_gripper": self._open,
        }.get(name)
        if fn is None:
            raise RuntimeError(f"{name} is not served on this station")
        kw.pop("arm_id", None)
        return fn(**kw)

    def _ee_pose(self, **_kw) -> dict:
        return {"pose": _gap_pose(self.ctx.state()["tcp_pose"])}

    def _objects(self) -> dict:
        r = self.ctx.observe(self.object_ids, require_orientation=True)
        out = {}
        for o in (r.get("evidence") or {}).get("observations", []):
            oid = o.get("object_id")
            if oid is None or o.get("position_m") is None:
                continue
            out[oid] = o
            self.observations[oid] = o
        return out

    def _observation(self, **_kw) -> dict:
        state = self.ctx.state()
        tcp = state["tcp_pose"]
        obs: dict[str, Any] = {
            "arms": [{"ee_pose": _gap_pose(tcp), "joint_state": {}}],
            "cameras": [],
        }
        if self.camera:
            cam = _matrix(tcp) @ CAMERA_IN_TCP
            clean = D.render(self._objects(), cam)
            self.renders += 1
            obs["cameras"].append(
                {
                    "name": "hand_eye_in_hand",
                    "pose": _gap_pose(
                        {"position_m": list(cam[:3, 3]), "quaternion_xyzw": _quat_xyzw(cam[:3, :3])}
                    ),
                    "intrinsics": D.D405_K.tolist(),
                    "depth": D.add_noise(clean, D.D405_K, self.rng),
                    "source": "rendered from privileged object observations (D405 noise model)",
                }
            )
        return obs

    def _go_to_pose(self, pose: dict, **_kw) -> dict:
        r = self.ctx.invoke(
            "move_pose",
            {
                "target_pose": _contract_pose(pose),
                "speed_m_s": self.speed,
                "timeout_s": self.timeout,
            },
        )
        if self.watch:
            # The watched object's height after each move (pull-up: did the brick come up?).
            seen = (self.ctx.observe([self.watch]).get("evidence") or {}).get("observations") or [
                {}
            ]
            tcp = self.ctx.state()["tcp_pose"]["position_m"]
            p = seen[0].get("position_m")
            self.watch_log.append(
                {
                    "cmd_z": float(pose["position"]["z"]),
                    "tcp_z": float(tcp[2]),
                    "obj_z": None if p is None else float(p[2]),
                }
            )
        if r.get("status") == "succeeded":
            return {"outcome": "reached", "reached": True}
        ft = r.get("failure_type") or ""
        if ft in ("unreachable", "invalid_request", "capability_missing"):
            return {"outcome": "unreachable", "reached": False, "reason": ft}
        return {"outcome": "timeout", "reached": False, "reason": ft}

    def _solve_ik(self, pose: dict, **_kw) -> dict:
        r = D.quat_matrix(_contract_pose(pose)["quaternion_xyzw"])
        ok = float(r[2, 2]) <= -TOP_DOWN_COS
        return {"success": ok, "within_servo_tolerance": ok}

    def _gripper(self, **_kw) -> dict:
        width = float((self.ctx.state().get("gripper") or {}).get("width_m") or 0.0)
        return {"position": max(0.0, min(1.0, width / MAX_OPENING_M))}

    def _describe_gripper(self, **_kw) -> dict:
        return {"width_fit": {"at_open_m": MAX_OPENING_M, "at_closed_m": 0.0}}

    def _open(self, **_kw) -> dict:
        r = self.ctx.invoke(
            "set_gripper",
            {"width_m": MAX_OPENING_M, "purpose": "release", "timeout_s": self.timeout},
        )
        return {"success": r.get("status") == "succeeded"}
