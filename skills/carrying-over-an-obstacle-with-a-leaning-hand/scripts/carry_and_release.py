"""Carry a held part over an obstacle with the hand leaning, and let go only over the container."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

LEANS = ((55.0, 15.0), (40.0, 15.0))
"""(to the left, forward) [deg] the hand leans while it carries, tried in this order. tuned_on: an
i2rt YAM arm on one station: pointing straight down it held the hand 26 cm over the pile at most,
and a part hanging from it at 28 cm caught a 154 mm tall box and dragged it (twice, 2026-11-13);
leaning left the same arm held the hand at 40-48 cm there. Other arms need their own."""

LEAN_DROP = 25.0
"""Also tried for letting go: leaning this far forward only [deg]. tuned_on: same."""

JAW_TURNS = (0.0, 180.0)
JAW_HEADINGS = (0.0, 90.0, 45.0, 135.0)
"""Jaw headings tried with each lean: as the hand is and turned half round, then these."""

CLEAR_OVER_RIM = 0.11
"""The tool centre crosses no lower than this over the highest rim [m]: a part hangs from it.
tuned_on: 5-9 cm printed parts held by an end, same station."""

CARRY_SPAN = 0.135
"""The highest carry tried is this far over the clearing height [m] (0.40 m over that station's
counter), stepping down 3 cm at a time."""
HEIGHT_STEP = 0.03

DROP_OVER_RIM = 0.12
"""The hand opens this far over the container's rim [m]. tuned_on: same."""

DROP_MARGIN = 0.05
"""Drop points keep this far inside the container's outer walls [m]. tuned_on: same; a drop point
5 cm inside the near wall, tried first for reach, put four parts in a row just outside that wall
when the hand was a few cm short of where its joints said: the middle is tried first."""

DROP_INSIDE = 0.03
"""A carry that ends short still lets go when the tool centre is this far inside the container's
walls [m], by the hand's own pose reading. tuned_on: same: four carries in one run ended 43-46 mm
short of a drop point at the edge of reach, over the box, and the parts were let go for nothing."""

ARRIVE_TOLERANCE = 0.04
"""A carry ending within this of its drop point has arrived [m]. tuned_on: same."""

SAME_SPOT = 0.025
"""A drop point within this of one in avoid_points is not tried [m]."""

EMPTY_FRACTION = 0.06
"""A jaw reading (robot.get_gripper, 1.0 = open) at or under this holds nothing. tuned_on: the YAM
linear_4310 hand; a part pinched by its edge read 0.07."""

MAX_IK_QUERIES = 40
"""robot.solve_ik previews allowed when no arm_model is given. A guard, not a measurement: the
full search is up to a few hundred poses, and sending that many plan requests in a row to the
source station's motion server stalled its control loop."""

PATH_STEP_M, PATH_STEP_DEG = 0.06, 20.0
"""With an arm_model, the straight carry is checked in steps of at most this."""


class Output(TypedDict):
    route: str
    detail: str
    release_position: dict[str, float] | None
    drop_point: dict[str, float] | None
    carried_z: float
    clear_z: float
    inside_m: float
    fell_short_at: list
    options: int
    ik_queries: int
    clearance_checked: bool


def _rot(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1e-12)) * 2
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[k] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
        q = [(r[k, j] - r[j, k]) / s, *v]
    return dict(zip("wxyz", (float(c) for c in q), strict=True))


def _axis(axis: str, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def lean_rotation(
    down: np.ndarray,
    jaw_turn_deg: float,
    forward_deg: float,
    left_deg: float,
    forward_heading_deg: float,
) -> np.ndarray:
    """A hand that points down as ``down`` does, its jaws turned ``jaw_turn_deg`` about the
    vertical, then leant so its fingertips point ``left_deg`` to the left and ``forward_deg``
    forward. "Forward" is the world heading ``forward_heading_deg`` (pointing away from the arm
    across the work area); "left" is a quarter turn anticlockwise from it, seen from above."""
    frame = _axis("z", forward_heading_deg - 90.0)
    tilt = frame @ _axis("x", forward_deg) @ _axis("y", left_deg) @ frame.T
    return tilt @ _axis("z", jaw_turn_deg) @ down


def drop_points(container_obb: dict[str, Any], margin: float = DROP_MARGIN) -> list[np.ndarray]:
    """A 4 x 4 grid of plan-view points inside the container, ``margin`` from its outer walls,
    the middle of the container first."""
    c, e = container_obb["center"], container_obb["extent"]
    r = _rot(container_obb.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0})
    yaw = math.atan2(r[1, 0], r[0, 0])
    us = np.linspace(-max(float(e["x"]) - margin, 0.0), max(float(e["x"]) - margin, 0.0), 4)
    vs = np.linspace(-max(float(e["y"]) - margin, 0.0), max(float(e["y"]) - margin, 0.0), 4)
    loc = np.array([[u, v] for u in us for v in vs])
    cs, sn = math.cos(yaw), math.sin(yaw)
    world = np.column_stack(
        [
            float(c["x"]) + cs * loc[:, 0] - sn * loc[:, 1],
            float(c["y"]) + sn * loc[:, 0] + cs * loc[:, 1],
        ]
    )
    order = np.argsort(np.linalg.norm(loc, axis=1), kind="stable")
    return [world[i] for i in order]


def inside_by(xy: Any, container_obb: dict[str, Any]) -> float:
    """How far inside the container's outer walls a plan-view point is [m] (negative: outside)."""
    c, e = container_obb["center"], container_obb["extent"]
    r = _rot(container_obb.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0})
    yaw = math.atan2(r[1, 0], r[0, 0])
    dx, dy = float(xy[0]) - float(c["x"]), float(xy[1]) - float(c["y"])
    lx = math.cos(yaw) * dx + math.sin(yaw) * dy
    ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return min(float(e["x"]) - abs(lx), float(e["y"]) - abs(ly))


def _rim(obb: dict[str, Any]) -> float:
    return float(obb["center"]["z"]) + float(obb["extent"]["z"])


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _xy(p: Any) -> np.ndarray:
    if isinstance(p, dict):
        return np.array([float(p["x"]), float(p["y"])])
    return np.asarray(p, float).reshape(-1)[:2]


def run(
    ctx: NodeContext,
    container_obb: dict,
    over_obbs: list | None = None,
    clear_z: float = 0.0,
    carry_z: float = 0.0,
    leans: list | None = None,
    forward_heading_deg: float = 90.0,
    pull_back_m: list | None = None,
    avoid_points: list | None = None,
    spread_index: int = 0,
    arm_model: dict | None = None,
    obstacles: list | None = None,
    margin_m: float = 0.02,
    workspace_bounds: dict | None = None,
    max_ik_queries: int = MAX_IK_QUERIES,
    drop_over_rim: float = DROP_OVER_RIM,
    drop_margin: float = DROP_MARGIN,
    drop_inside: float = DROP_INSIDE,
    arrive_tolerance: float = ARRIVE_TOLERANCE,
    empty_fraction: float = EMPTY_FRACTION,
    arm_id: int = 0,
) -> Output:
    """Carry what the hand holds over the rims between it and ``container_obb`` and release it there.

    Args:
        container_obb: The container to release into, in the frame the hand moves in (locate it
            with the wrist camera first). Upright; half extents of its outer size; its top the rim.
        over_obbs: Other boxes whose rims the carry must clear (the bin the part came from).
        clear_z: The lowest the tool centre may be while crossing [m, world]; 0 = 11 cm over the
            highest rim of container_obb and over_obbs.
        carry_z: The highest carry height tried [m, world]; 0 = clear_z + 13.5 cm.
        leans: [[left_deg, forward_deg], ...] to try, first choice first.
        forward_heading_deg: The world heading that points away from the arm across the work
            area [deg from x]; leans and pull-backs are relative to it.
        pull_back_m: Distances back (against forward) from where the hand is at which it may
            rise instead, tried in order after 0: out from under something overhead.
        avoid_points: Drop points not to try (those a carry fell short of before).
        spread_index: Rotates the order of the drop options, so parts do not stack on one point;
            pass the number already placed.
        arm_model, obstacles, margin_m, workspace_bounds: Given an arm_model, every candidate
            pose and the straight carry are checked with arm-clearance.reachable (locally,
            against ``obstacles`` at ``margin_m``) and robot.solve_ik is never asked.
        max_ik_queries: Without an arm_model, robot.solve_ik previews allowed in the search.
        drop_over_rim, drop_margin, drop_inside, arrive_tolerance, empty_fraction: see the
            module constants.

    Returns:
        ``route`` "placed", "returned", "blocked", "rise_short", "unreachable" or "lost".
    """
    on_arm = {"arm_id": int(arm_id)} if arm_id else {}
    rims = [_rim(container_obb)] + [_rim(o) for o in (over_obbs or [])]
    clear = float(clear_z) or max(rims) + CLEAR_OVER_RIM
    top = float(carry_z) or clear + CARRY_SPAN
    heights = [float(z) for z in np.arange(top, clear - 1e-6, -HEIGHT_STEP)] or [clear]
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "release_position": None,
        "drop_point": None,
        "carried_z": 0.0,
        "clear_z": clear,
        "inside_m": 0.0,
        "fell_short_at": [],
        "options": 0,
        "ik_queries": 0,
        "clearance_checked": bool(arm_model and obstacles),
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join(d for d in (detail, *notes) if d)
        return out

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def jaws() -> float:
        try:
            return float(ctx.tool("robot.get_gripper", **on_arm)["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning(
                "carrying-over-an-obstacle-with-a-leaning-hand: jaws unreadable (%s)", exc
            )
            return 1.0

    def go(position: Any, rotation: np.ndarray, straight: bool = False) -> None:
        tool = "robot.go_to_pose_cartesian" if straight else "robot.go_to_pose"
        try:
            ctx.tool(tool, pose={"position": _vec(position), "rotation": _quat(rotation)}, **on_arm)
        except Exception as exc:  # a refused move is judged by where the hand ends
            notes.append(f"a move was refused ({str(exc) or type(exc).__name__})")

    if jaws() <= float(empty_fraction):
        return finish("lost", "the jaws read empty before anything moved")

    start = here()
    p0, r0 = _xyz(start), _rot(start["rotation"])
    q_now: list[float] = []
    if arm_model:
        arms = (ctx.tool("robot.get_observation") or {}).get("arms") or []
        state = arms[int(arm_id)] if len(arms) > int(arm_id) else {}
        q_now = [float(v) for v in (state.get("joint_state") or {}).get("positions", [])]
        if not q_now:
            return finish("unreachable", "an arm_model was given but the observation has no joints")

    def solve(position: Any, rotation: np.ndarray, q_from: list[float]) -> list[float] | None:
        """Joints for the pose when the arm can take it (clear of the scene, with a model); an
        empty list stands for "yes" when the connector's IK was asked."""
        pose = {"position": _vec(position), "rotation": _quat(rotation)}
        if arm_model:
            reply = ctx.tool(
                "arm-clearance.reachable",
                model=arm_model,
                pose=pose,
                q_now=q_from,
                obstacles=obstacles or [],
                margin_m=float(margin_m),
                bounds=workspace_bounds,
            )
            return list(reply["q"]) if reply.get("reachable") else None
        if out["ik_queries"] >= int(max_ik_queries):
            return None
        out["ik_queries"] += 1
        try:
            reply = ctx.tool("robot.solve_ik", pose=pose, **on_arm)
        except Exception:  # noqa: BLE001 - unreachable is the answer
            return None
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            return None
        return []

    def path_clear(a: Any, ra: np.ndarray, qa: list[float], b: Any, rb: np.ndarray) -> bool:
        """With a model: the straight carry from a to b, the hand turning evenly, step by step."""
        if not arm_model:
            return True
        a, b = np.asarray(a, float), np.asarray(b, float)
        rel = ra.T @ rb
        angle = math.acos(max(-1.0, min(1.0, (float(np.trace(rel)) - 1) / 2)))
        n = int(
            max(
                1,
                math.ceil(np.linalg.norm(b - a) / PATH_STEP_M),
                math.ceil(math.degrees(angle) / PATH_STEP_DEG),
            )
        )
        if angle > 1e-6:
            k = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
            k = k / (np.linalg.norm(k) or 1.0)
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        q = qa
        for i in range(1, n + 1):
            t = i / n
            turn = np.eye(3)
            if angle > 1e-6:
                turn = np.eye(3) + math.sin(t * angle) * kx + (1 - math.cos(t * angle)) * (kx @ kx)
            got = solve(a + t * (b - a), ra @ turn, q)
            if got is None:
                return False
            q = got
        return True

    # The hand pointing down as this connector has it, and how far the jaws are turned from that.
    try:
        gf = ctx.tool("robot.grasp_frame", close_heading_deg=0.0, **on_arm)
        down = _rot(gf["rotation"])
    except Exception as exc:  # no grasp frame: lean from the hand as it is
        logger.info("carrying-over-an-obstacle-with-a-leaning-hand: no grasp frame (%s)", exc)
        down = r0
    rel = r0 @ down.T
    turn0 = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
    turns = [turn0 + t for t in JAW_TURNS] + list(JAW_HEADINGS)
    fwd = float(forward_heading_deg)
    hands = [
        lean_rotation(down, turn, float(f), float(left), fwd)
        for left, f in (leans or LEANS)
        for turn in turns
    ]
    ahead = np.array([math.cos(math.radians(fwd)), math.sin(math.radians(fwd))])
    backs = list(dict.fromkeys([0.0] + [float(d) for d in (pull_back_m or [])]))
    rise_at = [np.array([p0[0] - d * ahead[0], p0[1] - d * ahead[1], p0[2]]) for d in backs]

    # 1. Where and how the hand rises: the first spot and lean that reach up clear of everything.
    rise = None
    for spot in rise_at:
        for hand in hands:
            for z in heights:
                q = solve([spot[0], spot[1], z], hand, q_now)
                if q is not None:
                    rise = (spot, hand, z, q)
                    break
            if rise:
                break
        if rise:
            break
    # 2. Where it lets go: up to three drop points it reaches at a clearing height, with the
    # rising hand or another (the hand that clears what is over the pile need not be the one
    # that reaches the container), never one a carry fell short of before.
    options = []
    if rise:
        spot, hand, z_up, q_up = rise
        up = np.array([spot[0], spot[1], z_up])
        drop_hands = [hand] + [h for h in hands if h is not hand]
        drop_hands += [lean_rotation(down, t, LEAN_DROP, 0.0, fwd) for t in (turn0, 0.0)]
        avoid = [_xy(p) for p in (avoid_points or [])]
        for d in drop_points(container_obb, float(drop_margin)):
            if any(np.linalg.norm(d - a) < SAME_SPOT for a in avoid):
                continue
            hit = None
            for h in drop_hands:
                for z in heights:
                    q = solve([d[0], d[1], z], h, q_up)
                    if q is not None and path_clear(up, hand, q_up, [d[0], d[1], z], h):
                        hit = (d, h, z)
                        break
                if hit:
                    break
            if hit:
                options.append(hit)
            if len(options) >= 3:
                break
    out["options"] = len(options)
    if not arm_model and out["ik_queries"] >= int(max_ik_queries):
        notes.append(f"the search stopped at {int(max_ik_queries)} IK previews")
    if not rise or not options:
        what = "to rise with" if not rise else "over the container"
        return finish(
            "unreachable",
            f"no lean and height {what} at or over {clear:.3f} m; nothing moved, still holding",
        )
    k = int(spread_index) % len(options)
    options = options[k:] + options[:k]

    # 3. Rise (pulled back first, the hand as it is), and check that up means up.
    if np.linalg.norm(spot[:2] - p0[:2]) > 1e-6:
        go(spot, r0, straight=True)
    go(up, hand)
    z_now = float(_xyz(here())[2])
    if z_now < clear - 1e-3:
        return finish(
            "rise_short", f"the hand rose to {z_now:.3f} m, under the {clear:.3f} m needed to cross"
        )
    if jaws() <= float(empty_fraction):
        return finish("lost", "the jaws read empty after the rise")

    # 4. Across. Short of the point but over the container, by the hand's own measure: let go there.
    arrived = None
    for d, h, z in options:
        target = np.array([d[0], d[1], z])
        go(target, h)
        at = _xyz(here())
        if at[2] < clear - 1e-3:
            return finish(
                "blocked",
                f"the hand sank to {at[2]:.3f} m on the way across (needs {clear:.3f} m) and "
                "stopped there, still holding",
            )
        if float(np.linalg.norm(at - target)) <= float(arrive_tolerance):
            arrived = (d, target, h)
            break
        out["fell_short_at"].append({"x": float(d[0]), "y": float(d[1])})
        if inside_by(at[:2], container_obb) >= float(drop_inside):
            arrived = (d, at, _rot(here()["rotation"]))
            notes.append(
                f"the carry ended {np.linalg.norm(at - target) * 1e3:.0f} mm short, over the container"
            )
            break
    if arrived is None:
        go(up, hand)
        return finish(
            "returned",
            f"{len(options)} carr{'y' if len(options) == 1 else 'ies'} ended short and not over "
            "the container; the hand went back to where it rose, still holding",
        )
    d, at, h = arrived
    if jaws() <= float(empty_fraction):
        return finish("lost", "the jaws read empty after the carry across")

    # 5. Lower where it reaches, let go, back up.
    low = np.array([at[0], at[1], _rim(container_obb) + float(drop_over_rim)])
    if low[2] < at[2] - 1e-3 and solve(low, h, q_now) is not None:
        go(low, h, straight=True)
    release = _xyz(here())
    ctx.tool("robot.open_gripper", **on_arm)
    go(at, h, straight=True)
    out["release_position"] = _vec(release)
    out["drop_point"] = {"x": float(d[0]), "y": float(d[1]), "z": float(low[2])}
    out["carried_z"] = float(at[2])
    out["inside_m"] = inside_by(release[:2], container_obb)
    return finish(
        "placed",
        f"let go {out['inside_m'] * 1e3:.0f} mm inside the container's walls, "
        f"{release[2] - _rim(container_obb):.2f} m over its rim, after a carry at {at[2]:.2f} m",
    )
