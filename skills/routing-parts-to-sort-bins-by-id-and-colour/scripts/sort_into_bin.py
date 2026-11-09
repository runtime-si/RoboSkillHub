# Executes the bin choice of plan_sort_route (adapted from lego-sorter-public, MIT, see that script's
# header): lego-sorter routes a part to a virtual bin label and leaves the move to an industrial arm's
# taught dump poses (src/robot/dump.py); here the pick is grasping-flat-objects' script and the drop
# reuses placing-into-container-obb - its whole script for a bin given by description (found with sam3),
# or its footprint and drop-candidate helpers over a taught bin opening.
"""Pick the routed part off the staging pad and drop it into its bin."""

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

GRASP_SKILL, GRASP_SCRIPT = "grasping-flat-objects", "scripts/grasp_flat.py"
PLACE_SKILL, PLACE_SCRIPT = "placing-into-container-obb", "scripts/place_into_container.py"

LIFT_HEIGHT = 0.05
"""The grasp's lift [m]: grasping-flat-objects' LIFT_HEIGHT, inside a YAM's reach."""

OPENING_HALF = 0.035
"""Inner half extents of a taught bin with none given [m]: a 70 mm opening, under the 95 mm of the
YAM station twin's box."""

FOOTPRINT_MARGIN = 0.008
"""Room kept between the part's footprint and the bin's inner walls [m]: placing-into-container-obb's
FOOTPRINT_MARGIN (one calibrated camera error plus sway in the jaws)."""

APPROACH_CLEARANCE = 0.04
"""The part's bottom over the rim while the hand carries it to the bin [m]: 10 mm over
placing-into-container-obb's 0.03, because the hand arrives from a 50 mm lift off the table, not
from inside a perceived container's frame."""

RELEASE_OVER_RIM = 0.015
"""The part's bottom over the rim when the hand opens [m]: the part drops into the bin, so a bin's
contents never meet the hand (lego-sorter dumps from above too). 15 mm: one real station's 12 mm move
tolerance plus margin, so the part never catches the rim."""

RETREAT_HEIGHT = 0.06
"""Straight retreat after the release [m]."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a pose [m]: no tighter than a real YAM station's 12 mm per move."""

OPEN_FRACTION = 0.5
"""After open_gripper, a jaw opening (1.0 = open) under this did not open: placing-into-container-obb's."""

PLACE_ROUTES = {
    "placed": "sorted",
    "not_holding": "grasp_failed",
}
"""placing-into-container-obb's exits mapped onto this skill's; every other one is drop_failed."""


class Output(TypedDict):
    route: str
    detail: str
    bin_name: str
    release_position: dict[str, float]
    grasp_route: str
    place_route: str
    held_width: float
    bin_counts: dict[str, int]


_MODULES: dict[str, Any] = {}


def sibling(skill: str, script: str) -> Any:
    """A skill's script installed beside this one (or under a GAP_SKILLS_PATH root)."""
    if skill in _MODULES:
        return _MODULES[skill]
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(p) for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p]
    for root in roots:
        for path in (root / skill / script, root / "skills" / skill / script):
            if path.is_file():
                spec = importlib.util.spec_from_file_location(
                    f"_sort_{skill.replace('-', '_')}", path
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _MODULES[skill] = module
                return module
    raise RuntimeError(
        f"{skill} is not installed beside this skill (looked under {', '.join(str(r) for r in roots)})"
    )


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(p: dict[str, Any]) -> np.ndarray:
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def held_box(
    part_obb: dict[str, Any], grasp_position: dict[str, float], tcp: np.ndarray
) -> dict[str, Any]:
    """The part's box now: the planned box moved with the hand from the grasp pose to ``tcp``."""
    shift = tcp - _xyz(grasp_position)
    return {
        "center": _vec(_xyz(part_obb["center"]) + shift),
        "extent": dict(part_obb["extent"]),
        "orientation": dict(
            part_obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        ),
    }


def bin_opening(b: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
    """A taught bin: (opening centre xy, its two horizontal axes, inner half extents, rim height)."""
    p = b.get("drop_position") or b.get("center") or {}
    c = _xyz({k: p.get(k, 0.0) for k in "xyz"})
    rim = float(b.get("rim_z", c[2]))
    half = b.get("opening_half")
    if half is None:
        half = [OPENING_HALF, OPENING_HALF]
    elif isinstance(half, (int, float)):
        half = [float(half), float(half)]
    yaw = math.radians(float(b.get("yaw_deg", 0.0)))
    axes = [
        np.array([math.cos(yaw), math.sin(yaw), 0.0]),
        np.array([-math.sin(yaw), math.cos(yaw), 0.0]),
    ]
    return c[:2], axes, np.array([float(half[0]), float(half[1])]), rim


def run(
    ctx: NodeContext,
    part_obb: dict[str, Any],
    sort_bin: dict[str, Any],
    bin_counts: dict[str, int] | None = None,
    support_z: float | None = None,
    use_perception: bool = True,
    lift_height: float = LIFT_HEIGHT,
    approach_clearance: float = APPROACH_CLEARANCE,
    release_over_rim: float = RELEASE_OVER_RIM,
    retreat_height: float = RETREAT_HEIGHT,
    footprint_margin: float = FOOTPRINT_MARGIN,
    reach_tolerance: float = REACH_TOLERANCE,
    grasp_inputs: dict[str, Any] | None = None,
    place_inputs: dict[str, Any] | None = None,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    counts = {str(k): int(v) for k, v in (bin_counts or {}).items()}
    name = str((sort_bin or {}).get("name", ""))
    out: Output = {
        "route": "grasp_failed",
        "detail": "",
        "bin_name": name,
        "release_position": {},
        "grasp_route": "",
        "place_route": "",
        "held_width": -1.0,
        "bin_counts": counts,
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join(d for d in (detail, *notes) if d)
        if route == "sorted":
            counts[name] = counts.get(name, 0) + 1
        return out

    if not part_obb or not part_obb.get("center"):
        return finish("grasp_failed", "no part box to pick (the plan had no depth)")
    if not sort_bin or not name:
        return finish("drop_failed", "no bin given; nothing moved")

    # 1. Pick: grasping-flat-objects (short side first, jaw-gap checks, one retry on the long side).
    grasp = sibling(GRASP_SKILL, GRASP_SCRIPT).run(
        ctx,
        object_obb=part_obb,
        support_z=support_z,
        lift_height=float(lift_height),
        check_reach=bool(check_reach),
        arm_id=int(arm_id),
        **(grasp_inputs or {}),
    )
    out["grasp_route"] = str(grasp.get("route", ""))
    out["held_width"] = float(grasp.get("held_width", -1.0))
    if out["grasp_route"] != "grasped":
        return finish(
            "grasp_failed", f"{GRASP_SKILL}: {grasp.get('route')} ({grasp.get('detail', '')})"
        )

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    start = here()
    tcp0 = _xyz(start["position"])
    held = held_box(part_obb, grasp["grasp_position"], tcp0)
    place = sibling(PLACE_SKILL, PLACE_SCRIPT)

    # 2a. A bin given by description: placing-into-container-obb finds it and drops into it.
    if use_perception and sort_bin.get("description"):
        kw = dict(place_inputs or {})
        if sort_bin.get("pixel") is not None:
            kw.setdefault("container_pixel", list(sort_bin["pixel"]))
        placed = place.run(
            ctx,
            container_description=str(sort_bin["description"]),
            held_obb=held,
            check_reach=bool(check_reach),
            arm_id=int(arm_id),
            **kw,
        )
        out["place_route"] = str(placed.get("route", ""))
        out["release_position"] = dict(placed.get("release_position") or {})
        route = PLACE_ROUTES.get(out["place_route"], "drop_failed")
        return finish(route, f"{PLACE_SKILL}: {out['place_route']} ({placed.get('detail', '')})")

    # 2b. A taught bin: placing-into-container-obb's footprint and drop candidates over its opening.
    c_xy, axes, inner, rim_z = bin_opening(sort_bin)
    rotation = start["rotation"]
    offset_xy = _xyz(held["center"])[:2] - tcp0[:2]
    hang = float(tcp0[2] - place._bottom_z(held))
    reach = np.array([place.footprint_reach(held, a) for a in axes])
    cands = place.drop_candidates(inner, reach, float(footprint_margin))
    if not cands:
        return finish(
            "drop_failed",
            f"the part needs {reach[0] * 2e3:.0f} x {reach[1] * 2e3:.0f} mm plus "
            f"{footprint_margin * 1e3:.0f} mm a side; bin {name!r} opens "
            f"{inner[0] * 2e3:.0f} x {inner[1] * 2e3:.0f} mm; still holding",
        )

    def go(tool: str, position: np.ndarray) -> str:
        try:
            why = place._refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}, **on_arm)
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        off = float(np.linalg.norm(_xyz(here()["position"]) - position))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand stopped {off * 1e3:.0f} mm from it"
        return ""

    def reachable(position: np.ndarray) -> bool:
        if not check_reach:
            return True
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(position), "rotation": rotation}, **on_arm
            )
        except Exception as exc:  # no preview served: let the move decide
            logger.info("routing-parts-to-sort-bins: no IK preview (%s)", exc)
            return True
        return not (
            isinstance(reply, dict)
            and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
        )

    carry_z = rim_z + float(approach_clearance) + hang
    if tcp0[2] < carry_z - 1e-3:
        why = go("robot.go_to_pose_cartesian", np.array([tcp0[0], tcp0[1], carry_z]))
        if why:
            notes.append(f"rise to the carry height: {why}")
    tried: list[str] = []
    for cand, du, dv in cands:
        obj_xy = c_xy + du * axes[0][:2] + dv * axes[1][:2]
        hand_xy = obj_xy - offset_xy
        above = np.array([hand_xy[0], hand_xy[1], carry_z])
        release = np.array([hand_xy[0], hand_xy[1], rim_z + float(release_over_rim) + hang])
        if not (reachable(above) and reachable(release)):
            tried.append(f"{cand}: out of reach (IK preview)")
            continue
        why = go("robot.go_to_pose", above)
        if why:
            tried.append(f"{cand}: over the bin: {why}")
            continue
        why = go("robot.go_to_pose_cartesian", release)
        if why:
            back = go("robot.go_to_pose_cartesian", above)
            tried.append(f"{cand}: descent: {why}" + (f"; back up: {back}" if back else ""))
            continue
        at = _xyz(here()["position"])
        out["release_position"] = _vec(np.array([*(at[:2] + offset_xy), at[2] - hang]))
        out["place_route"] = "released"
        opened = ctx.tool("robot.open_gripper", **on_arm)
        jaw = float(opened.get("position", -1.0)) if isinstance(opened, dict) else -1.0
        if jaw < 0.0:
            try:
                jaw = float(ctx.tool("robot.get_gripper", **on_arm)["position"])
            except Exception:  # unreadable: take the open as done
                jaw = -1.0
        if 0.0 <= jaw < OPEN_FRACTION:
            out["place_route"] = "not_released"
            return finish(
                "drop_failed", f"over bin {name!r} ({cand}) the jaws read {jaw:.3f} after opening"
            )
        back = go("robot.go_to_pose_cartesian", at + np.array([0.0, 0.0, float(retreat_height)]))
        if back:
            notes.append(f"retreat: {back}")
        if tried:
            notes.append("before: " + "; ".join(tried))
        return finish(
            "sorted",
            f"dropped into bin {name!r} at its {cand} point, the part's bottom "
            f"{(at[2] - hang - rim_z) * 1e3:.0f} mm over the rim",
        )
    out["place_route"] = "unreachable"
    return finish(
        "drop_failed",
        f"no drop point over bin {name!r} was reached ({'; '.join(tried)}); still holding",
    )
