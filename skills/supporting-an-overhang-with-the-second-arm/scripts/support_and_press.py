# The two-arm pattern is APEX-MR's support primitive (github.com/intelligent-control-lab/APEX-MR,
# src/lego_policy.cpp LegoPolicy::support, MIT): before a press, the second arm moves to a support pose at
# an offset (sup_dx, sup_dy, sup_dz studs) from the brick being pressed onto and stays there, so the press
# force goes into the support, not into a cantilevered sub-assembly. No code is copied: APEX-MR's
# pressdown (commanding 5 cm past contact and letting the Yaskawa's controller stall) is not reproduced;
# the press is the hub's pressing-a-brick-onto-studs (guarded descent), run on the other arm, and the
# support's read-back pose before and after says whether the structure moved.
"""Press a brick onto an overhanging part while the second arm holds the part up from below or beside."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

APPROACH = 0.06
"""Distance from which the support hand comes to its pose along approach_offset's direction [m]."""

PRELOAD = 0.002
"""How far past the support pose (along the approach) the support hand is commanded [m]: the servo stalls
on the part, which is the preload; small, so a support from below does not lift the overhang."""

MOVED = 0.003
"""Support tool-centre drift during the press past which the structure counts as moved [m] (over the YAM
servo's ~1 mm settle noise, under a LEGO stud's 1.7 mm engagement plus play)."""

STUD = {"lego": 0.008, "duplo": 0.016}
BRICK_H = {"lego": 0.0096, "duplo": 0.0192}

ROUTES = (
    "seated",
    "not_seated",
    "misaligned",
    "not_holding",
    "unverified",
    "structure_moved",
    "support_unreachable",
    "unreachable",
)


class Output(TypedDict):
    route: str
    detail: str
    press_route: str
    press_detail: str
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


def _press_module() -> Any:
    """The hub's pressing-a-brick-onto-studs script, a sibling bundle in the same skills root."""
    rel = Path("pressing-a-brick-onto-studs") / "scripts" / "press_brick.py"
    roots = [Path(__file__).resolve().parents[2]]
    roots += [
        Path(p) / "skills" for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p
    ]
    path = next((r / rel for r in roots if (r / rel).is_file()), None)
    if path is None:
        raise RuntimeError(
            "pressing-a-brick-onto-studs is needed beside this bundle (skills root or GAP_SKILLS_PATH)"
        )
    spec = importlib.util.spec_from_file_location("press_brick_for_support", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def support_point_from_studs(
    base_brick: dict[str, float], sup: list[float], system: str, yaw_deg: float = 0.0
) -> np.ndarray:
    """APEX-MR's support offset: (sup_dx, sup_dy) studs along the brick's axes and sup_dz brick heights."""
    p = STUD[system]
    a = np.radians(float(yaw_deg))
    bx, by = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])
    return (
        _xyz(base_brick)
        + float(sup[0]) * p * bx
        + float(sup[1]) * p * by
        + np.array([0, 0, float(sup[2]) * BRICK_H[system]])
    )


def run(
    ctx: NodeContext,
    seat_position: dict[str, float],
    support_pose: dict[str, Any] | None = None,
    support_point: dict[str, float] | None = None,
    support_studs: list[float] | None = None,
    support_from: dict[str, float] | None = None,
    approach_offset: list[float] | None = None,
    support: bool = True,
    system: str = "duplo",
    seat_yaw_deg: float = 0.0,
    press_arm: int = 0,
    support_arm: int = 1,
    preload: float = PRELOAD,
    moved_tolerance: float = MOVED,
    press_inputs: dict[str, Any] | None = None,
) -> Output:
    if system not in STUD:
        raise ValueError(f"system must be one of {sorted(STUD)}, got {system!r}")
    if int(press_arm) == int(support_arm):
        raise ValueError("press_arm and support_arm must differ")
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "press_route": "",
        "press_detail": "",
        "support_used": False,
        "support_drift_mm": 0.0,
        "support_pose": {},
    }

    def finish(route: str, why: str) -> Output:
        out["route"], out["detail"] = route, why
        logger.info("supporting-an-overhang-with-the-second-arm: %s (%s)", route, why)
        return out

    on = {"arm_id": int(support_arm)} if int(support_arm) else {}
    held = None
    if support:
        if support_pose is not None:
            target, rot = _xyz(support_pose), support_pose.get("rotation")
        elif support_point is not None or (support_studs is not None and support_from is not None):
            target = (
                _xyz(support_point)
                if support_point is not None
                else support_point_from_studs(support_from, support_studs, system, seat_yaw_deg)
            )
            rot = None
        else:
            raise ValueError("give support_pose, support_point, or support_studs with support_from")
        if rot is None:
            rot = _pose(ctx.tool("robot.get_ee_pose", **on))["rotation"]
            try:
                gf = ctx.tool("robot.grasp_frame", **on)
                rot = gf.get("rotation", rot) if isinstance(gf, dict) else rot
            except Exception as exc:
                logger.info("no grasp frame for the support arm (%s)", exc)
        off = np.asarray(
            approach_offset if approach_offset is not None else [0.0, 0.0, APPROACH], dtype=float
        )
        n = float(np.linalg.norm(off))
        if n < 1e-6:
            raise ValueError(
                "approach_offset must be a non-zero vector (from the support pose to the approach)"
            )
        ctx.tool("robot.close_gripper", **on)
        for name, p in (
            ("approach", target + off),
            ("support pose", target - off / n * float(preload)),
        ):
            pose = {"position": dict(zip("xyz", map(float, p), strict=True)), "rotation": rot}
            try:
                why = _refusal(ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on))
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why:
                return finish("support_unreachable", f"support arm {name}: {why}")
        held = _xyz(_pose(ctx.tool("robot.get_ee_pose", **on)))
        out["support_used"] = True
        out["support_pose"] = {
            "position": dict(zip("xyz", map(float, held), strict=True)),
            "rotation": rot,
        }

    kw = {"system": system, **dict(press_inputs or {})}
    press = _press_module().run(
        ctx,
        seat_position=seat_position,
        seat_yaw_deg=float(seat_yaw_deg),
        arm_id=int(press_arm),
        **kw,
    )
    out["press_route"], out["press_detail"] = str(press["route"]), str(press.get("detail", ""))
    drift = 0.0
    if held is not None:
        drift = float(np.linalg.norm(_xyz(_pose(ctx.tool("robot.get_ee_pose", **on))) - held))
        out["support_drift_mm"] = round(drift * 1e3, 2)
        back = held + np.asarray(
            approach_offset if approach_offset is not None else [0.0, 0.0, APPROACH], float
        )
        try:
            ctx.tool(
                "robot.go_to_pose_cartesian",
                pose={
                    "position": dict(zip("xyz", map(float, back), strict=True)),
                    "rotation": out["support_pose"]["rotation"],
                },
                **on,
            )
        except Exception as exc:
            logger.info("support arm could not back out (%s)", exc)
    route = out["press_route"] if out["press_route"] in ROUTES else "unreachable"
    if drift > float(moved_tolerance):
        route = "structure_moved"
    return finish(
        route,
        f"press {out['press_route']}"
        + (f", support drifted {drift * 1e3:.1f} mm" if held is not None else ", no support"),
    )
