# The two-arm pattern is APEX-MR's support primitive (github.com/intelligent-control-lab/APEX-MR,
# src/lego_policy.cpp LegoPolicy::support: the second arm goes to a support pose on the assembly and
# stays there while the first arm acts; MIT). No code is copied: on YAM the support arm closes its hand
# and presses down on the base (the lower brick or the plate beside the brick) through a guarded
# descent, the release itself is the hub's disassembling-a-brick-by-twist-lever run on the other arm,
# and the support's read-back pose before and after says whether the base moved.
"""Free a clutched brick from a loose base: one arm presses the base down, the other twists the brick off."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRESS_DEPTH = 0.004
"""How far below the base's top the support's closed fingertips are commanded [m]: the servo stalls on
the base, so this is the preload, not a motion (APEX-MR's support pose is at the surface; a YAM's
position servo needs a few millimetres of interference to push)."""

APPROACH = 0.06
"""Height over the support point from which the support hand descends [m]."""

STALL_TOLERANCE = 0.003
"""A press leg that times out or stops on contact within this of the target (laterally, and above it by
at most press_depth plus this) is the hand stalled on the base, as intended [m]."""

TIP_BELOW_TCP = 0.0102
"""Closed linear_4310 fingertips reach this far below the tool centre [m] (PORTING lesson 6)."""

MIN_TCP_Z = 0.014
"""A real YAM station refuses a tool centre under 14 mm over its table [m]."""

BASE_MOVED = 0.003
"""Support tool centre drift during the twist past which the base counts as moved [m]: over the
YAM servo's settle noise (~1 mm), under a DUPLO stud's 4.5 mm engagement."""

ROUTES = (
    "separated",
    "base_moved",
    "clutch_too_strong",
    "missed",
    "unverified",
    "support_unreachable",
    "unreachable",
)

TWIST_TO_ROUTE = {
    "removed": "separated",
    "unverified": "unverified",
    "stuck": "clutch_too_strong",
    "slipped": "clutch_too_strong",
    "neighbour_lifted": "base_moved",
    "missed": "missed",
    "unreachable": "unreachable",
}


class Output(TypedDict):
    route: str
    detail: str
    twist_route: str
    twist_detail: str
    support_used: bool
    support_drift_mm: float
    support_pose: dict[str, Any]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        v = v.get("position", v)
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=np.float64).reshape(3)


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached", "contact"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("outcome") is None and (
        reply.get("reached") is False or reply.get("success") is False
    ):
        return str(reply.get("reason") or "the move reported failure")
    return ""


def _twist_module() -> Any:
    """The hub's disassembling-a-brick-by-twist-lever script, a sibling bundle in the same skills root."""
    rel = Path("disassembling-a-brick-by-twist-lever") / "scripts" / "disassemble_brick.py"
    roots = [Path(__file__).resolve().parents[2]]
    roots += [
        Path(p) / "skills" for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p
    ]
    path = next((r / rel for r in roots if (r / rel).is_file()), None)
    if path is None:
        raise RuntimeError(
            "disassembling-a-brick-by-twist-lever is needed beside this bundle (skills root or GAP_SKILLS_PATH)"
        )
    spec = importlib.util.spec_from_file_location("disassemble_brick_for_hold_and_twist", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(
    ctx: NodeContext,
    seat_position: dict[str, float],
    support_point: dict[str, float] | None = None,
    support: bool = True,
    seat_yaw_deg: float = 0.0,
    twist_arm: int = 0,
    support_arm: int = 1,
    press_depth: float = PRESS_DEPTH,
    base_moved_tolerance: float = BASE_MOVED,
    twist_inputs: dict[str, Any] | None = None,
) -> Output:
    if support and support_point is None:
        raise ValueError(
            "support_point (where the support hand presses the base, its top surface) is required"
        )
    if int(twist_arm) == int(support_arm):
        raise ValueError("twist_arm and support_arm must differ")
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "twist_route": "",
        "twist_detail": "",
        "support_used": False,
        "support_drift_mm": 0.0,
        "support_pose": {},
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        logger.info("disassembling-a-stack-with-hold-and-twist: %s (%s)", route, why)
        return out

    on = {"arm_id": int(support_arm)} if int(support_arm) else {}
    held = None
    if support:
        target = _xyz(support_point)
        here = _pose(ctx.tool("robot.get_ee_pose", **on))
        rot = here["rotation"]
        try:
            gf = ctx.tool("robot.grasp_frame", **on)
            rot = gf.get("rotation", rot) if isinstance(gf, dict) else rot
        except Exception as exc:  # keep the current rotation
            logger.info("no grasp frame for the support arm (%s)", exc)
        tcp_z = max(float(target[2]) + TIP_BELOW_TCP - float(press_depth), MIN_TCP_Z)
        ctx.tool("robot.close_gripper", **on)
        for name, z in (
            ("over the support point", float(target[2]) + TIP_BELOW_TCP + APPROACH),
            ("onto the base", tcp_z),
        ):
            pose = {
                "position": dict(zip("xyz", (float(target[0]), float(target[1]), z), strict=True)),
                "rotation": rot,
            }
            try:
                why = _refusal(ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on))
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why and name == "onto the base":
                # The press leg is meant to stall on the base: a timeout or contact near the target is the preload.
                got = _xyz(_pose(ctx.tool("robot.get_ee_pose", **on)))
                if (
                    np.linalg.norm(got[:2] - target[:2]) < STALL_TOLERANCE
                    and got[2] - tcp_z < float(press_depth) + STALL_TOLERANCE
                ):
                    why = ""
            if why:
                return finish("support_unreachable", f"support arm {name}: {why}")
        held = _xyz(_pose(ctx.tool("robot.get_ee_pose", **on)))
        out["support_used"] = True
        out["support_pose"] = {
            "position": dict(zip("xyz", map(float, held), strict=True)),
            "rotation": rot,
        }

    kwargs = dict(twist_inputs or {})
    twist = _twist_module().run(
        ctx,
        seat_position=seat_position,
        seat_yaw_deg=float(seat_yaw_deg),
        arm_id=int(twist_arm),
        **kwargs,
    )
    out["twist_route"], out["twist_detail"] = str(twist["route"]), str(twist.get("detail", ""))

    drift = 0.0
    if held is not None:
        now = _xyz(_pose(ctx.tool("robot.get_ee_pose", **on)))
        drift = float(np.linalg.norm(now - held))
        out["support_drift_mm"] = round(drift * 1e3, 2)
        up = dict(out["support_pose"])
        up["position"] = {**up["position"], "z": float(held[2]) + APPROACH}
        try:
            ctx.tool("robot.go_to_pose_cartesian", pose=up, **on)
        except Exception as exc:
            logger.info("support arm could not rise (%s)", exc)
    route = TWIST_TO_ROUTE.get(out["twist_route"], "unreachable")
    if drift > float(base_moved_tolerance):
        route = "base_moved"
    msg = f"twist {out['twist_route']}" + (
        f", support drifted {drift * 1e3:.1f} mm" if held is not None else ", no support"
    )
    if not math.isfinite(drift):
        route = "unreachable"
    return finish(route, msg)
