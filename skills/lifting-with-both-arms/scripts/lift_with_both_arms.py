# Adapted from RoboTwin (https://github.com/robotwin-Platform/RoboTwin),
# envs/lift_pot.py, lift_pot.play_once / check_success, MIT, Copyright (c) 2025
# Tianxing Chen; handle points as cap-x's two-arm lift takes them
# (https://github.com/capgym/cap-x, capx/integrations/franka/two_arm_lift.py,
# get_handle0_pos / get_handle1_pos, MIT).
# Kept: each arm grasps its own contact point on the object (RoboTwin's pot
# contact points 0 and 1, cap-x's handle 0 and 1) from a short pre-grasp
# stand-off, then both arms rise by the same displacement; success is both
# tool centres still at their grasp points on the object (RoboTwin: within
# 0.03 m) with the object risen and level.
# Changed: the pot's simulator contact points replaced by two points near the
# ends of a world-frame OBB's long axis (or given handle points), each assigned
# to the arm whose tool centre is nearer; near-top-down grasps through
# robot.grasp_frame with an explicit arm_id, each wrist leaned away from the
# other hand, the jaws closing across the long axis, the free wrist roll
# finished on the last joint; RoboTwin's half-closed
# pre-shape dropped (the jaws open fully, the object is a plain bar); the one
# synchronous move_by_displacement replaced by small interleaved steps, one arm
# then the other, so the hands never differ in height by more than one step;
# both jaw openings and the hands' separation are checked after every step
# (RoboTwin checks the tool centres at the end only); refused moves are routed,
# and an arm that cannot follow has the other brought back level with it.
"""Lift an elongated object with both arms, one grasp near each end, in small level steps."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

LIFT_HEIGHT = 0.10
"""How far both hands rise [m]: RoboTwin lifts the pot to z 0.88 m, about 0.1 m
over where it rests on its 0.74 m table, and scores it lifted above 0.82 m."""

LIFT_STEP = 0.01
"""Height of one interleaved step [m]: the hands differ by at most this between
the two half-steps, 3.6 deg of tilt over a 0.16 m grasp separation."""

PRE_GRASP = 0.08
"""Each hand's hover over its grasp point before it descends [m]. RoboTwin's
pre_grasp_dis is 0.035 behind a planned approach; a joint-space go_to_pose to a
hover that low can sweep the open jaws through the object, so the descent from
here is a straight cartesian move."""

OUTWARD_TILT_DEG = 15.0
"""How far each hand's approach leans from vertical, its wrist away from the other
hand [deg]. tuned_on: YAM station twin, 200 x 40 x 40 mm bar between arm bases
0.61 m apart, grasps 0.14 m apart: straight top-down the two arms' wrist links
(link4) came within 0.1-1.6 mm of each other (a vertex measure, so likely
touching); at 15 deg 63 mm with the bar lifted level (1.0 deg); at 25-35 deg
over 100 mm, but the leaning pads let the bar pivot 4-8 deg in the jaws."""

END_INSET = 0.03
"""How far in from each end of the long axis a grasp point lies [m]: the YAM's
pads fully on the object with a few mm to spare."""

MIN_INSET_FRACTION = 0.25
"""On a short object the grasp points stay at least this fraction of the
half-length from its centre, so the two hands do not meet in the middle."""

GRASP_DEPTH = 0.02
"""How far under the object's top face the tool centre closes [m] (never past
the object's centre): the YAM's closed fingertips reach about 10 mm below the
tool centre (PORTING lesson 6), so on a 40 mm bar they stop 10 mm over the
table."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent [m]: a servo's 4 mm ball plus a
real arm's sag, no tighter than the connectors' own bounds."""

LEVEL_TOLERANCE = 0.008
"""How far a lifting hand may end from its commanded height [m] before it counts
as not following: under a step, so a stalled arm is not read as one that kept up."""

SEPARATION_TOLERANCE = 0.015
"""How far the distance between the two tool centres may drift from its value at
the grasp [m] before a hand has slid along the object (or the arms stretch it):
both servos' 4 mm arrival balls plus a real YAM's sag, under RoboTwin's 0.03 m
grasp-point bound."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (robot.get_gripper, 1.0 = open) closed on
nothing: 3.8 mm on a 94 mm YAM span, the stop plus verifying-grasps' 3 mm margin."""

ROLL_TOLERANCE_DEG = 3.0
"""How far a hand's jaw roll may end from the commanded one before the rest is
turned on its last joint [deg]: a servo's own 3 deg arrival bound."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back, stacking-objects' value."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad]: the i2rt YAM's joint 6 (+-120 deg, its MJCF)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""

DOWN = np.array([0.0, 0.0, -1.0])


class Output(TypedDict):
    route: str
    detail: str
    rise_m: float
    hand_tilt_deg: float
    separation_change_m: float
    first_grip: float
    second_grip: float
    first_grasp_pose: dict[str, Any]
    second_grasp_pose: dict[str, Any]
    object_obb: dict[str, Any]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _rotation(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _wrap180(deg: float) -> float:
    """An angle folded to [-90, 90): parallel jaws look the same turned half a turn."""
    return (float(deg) + 90.0) % 180.0 - 90.0


def _roll_residual(commanded: dict[str, float], now: dict[str, float], axis: np.ndarray) -> float:
    """How far the hand must still turn about ``axis`` to reach ``commanded`` [deg]."""
    err = _rotation(commanded) @ _rotation(now).T
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(err)) - 1.0) / 2.0)))
    if angle < 1e-9:
        return 0.0
    if math.pi - angle < 1e-6:
        sym = err + np.eye(3)
        about = sym[:, int(np.argmax(np.linalg.norm(sym, axis=0)))]
        about = about / float(np.linalg.norm(about))
    else:
        about = np.array([err[2, 1] - err[1, 2], err[0, 2] - err[2, 0], err[1, 0] - err[0, 1]])
        about = about / (2.0 * math.sin(angle))
    return _wrap180(math.degrees(angle * float(np.dot(about, axis))))


def _point(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        v = v.get("position", v)
        return np.array([float(v[k]) for k in ("x", "y", "z")])
    return np.asarray(v, dtype=np.float64).reshape(3)


def grasp_points(
    obb: dict[str, Any], end_inset: float = END_INSET, grasp_depth: float = GRASP_DEPTH
) -> tuple[np.ndarray, np.ndarray, float] | str:
    """Two tool-centre points near the ends of the box's long horizontal axis and
    the jaws' closing heading across it [deg], or why there are none."""
    centre = _point(obb["center"])
    rot = _rotation(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    i = int(np.argmax(half))
    flat = float(np.linalg.norm(rot[:2, i]))
    if abs(float(rot[2, i])) > flat:
        return "the object's long axis stands more upright than flat: no top-down end grasps"
    half_len = half[i] * flat
    if half_len < 1e-4:
        return "the box has no horizontal extent"
    along = rot[:, i].copy()
    along[2] = 0.0
    along /= float(np.linalg.norm(along))
    reach = max(half_len - float(end_inset), MIN_INSET_FRACTION * half_len)
    up = float(np.abs(rot.T @ np.array([0.0, 0.0, 1.0])) @ half)
    z = centre[2] + up - min(float(grasp_depth), up)
    a = centre + along * reach
    b = centre - along * reach
    a[2] = b[2] = z
    heading = math.degrees(math.atan2(along[1], along[0])) + 90.0
    return a, b, heading


def _joints(ctx: NodeContext, arm_id: int) -> np.ndarray:
    """One arm's joint positions from robot.get_observation, or empty."""
    try:
        arms = ctx.tool("robot.get_observation").get("arms") or []
    except Exception as exc:
        logger.info("lifting-with-both-arms: no joint reading (%s)", exc)
        return np.zeros(0)
    if int(arm_id) >= len(arms):
        return np.zeros(0)
    q = arms[int(arm_id)].get("joint_state")
    if isinstance(q, dict):
        q = q.get("positions")
    return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)


def run(
    ctx: NodeContext,
    object_obb: dict[str, Any],
    handles: list[Any] | None = None,
    first_arm: int = 0,
    second_arm: int = 1,
    close_heading_deg: float | None = None,
    outward_tilt_deg: float = OUTWARD_TILT_DEG,
    lift_height: float = LIFT_HEIGHT,
    lift_step: float = LIFT_STEP,
    pre_grasp: float = PRE_GRASP,
    end_inset: float = END_INSET,
    grasp_depth: float = GRASP_DEPTH,
    reach_tolerance: float = REACH_TOLERANCE,
    level_tolerance: float = LEVEL_TOLERANCE,
    separation_tolerance: float = SEPARATION_TOLERANCE,
    empty_fraction: float = EMPTY_FRACTION,
) -> Output:
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "rise_m": 0.0,
        "hand_tilt_deg": 0.0,
        "separation_change_m": 0.0,
        "first_grip": -1.0,
        "second_grip": -1.0,
        "first_grasp_pose": {},
        "second_grasp_pose": {},
        "object_obb": dict(object_obb),
    }
    arms = (int(first_arm), int(second_arm))
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join([d for d in (detail, *notes) if d])
        return out

    def arm(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": int(a)} if int(a) else {}

    def here(a: int) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def move(
        tool: str, a: int, position: np.ndarray, rotation: dict[str, float], level: bool = False
    ) -> str:
        """Send one arm and judge arrival from a fresh reading of that arm."""
        try:
            why = _refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}, **arm(a))
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        now = _xyz(here(a))
        off = float(np.linalg.norm(now - position))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; arm {a} stopped {off * 1e3:.0f} mm from its target"
        if level and abs(now[2] - position[2]) > float(level_tolerance):
            return f"{why or 'ended low'}; arm {a} ended {(now[2] - position[2]) * 1e3:+.0f} mm off its step height"
        return ""

    def grip(a: int) -> float:
        """The jaw opening (1.0 = open), or -1 when the hand cannot be read."""
        try:
            return float(ctx.tool("robot.get_gripper", **arm(a))["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning("lifting-with-both-arms: arm %d gripper unreadable (%s)", a, exc)
            return -1.0

    def holds(fraction: float) -> bool:
        return fraction < 0.0 or fraction > float(empty_fraction)

    if arms[0] == arms[1]:
        return finish("unreachable", f"first and second arm are both arm {arms[0]}")
    start = {a: here(a) for a in arms}
    q = start[arms[1]].get("rotation") or {}
    if sum(float(q.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_xyz(start[arms[0]]) - _xyz(start[arms[1]]))) < SAME_ARM
    ):
        # A connector that clamps or ignores arm_id reports the first arm (or zeros)
        # for the second; "both" arms would then be one arm moved twice.
        return finish("unreachable", f"the connector does not report a separate arm {arms[1]}")

    # Grasp points: given handles, else near the ends of the long axis.
    if handles:
        if len(handles) != 2:
            return finish("unreachable", f"{len(handles)} handles given, two needed")
        a_pt, b_pt = _point(handles[0]), _point(handles[1])
        span = b_pt - a_pt
        if float(np.linalg.norm(span[:2])) < 1e-4:
            return finish("unreachable", "the two handles stand one over the other")
        heading = math.degrees(math.atan2(span[1], span[0])) + 90.0
    else:
        found = grasp_points(object_obb, end_inset, grasp_depth)
        if isinstance(found, str):
            return finish("unreachable", found)
        a_pt, b_pt, heading = found
    if close_heading_deg is not None:
        heading = float(close_heading_deg)
    # Each arm takes the end nearer its own hand (the pairing with the shorter total reach).
    pa, pb = (_xyz(start[a])[:2] for a in arms)
    keep = np.linalg.norm(pa - a_pt[:2]) + np.linalg.norm(pb - b_pt[:2])
    swap = np.linalg.norm(pa - b_pt[:2]) + np.linalg.norm(pb - a_pt[:2])
    target = {arms[0]: a_pt, arms[1]: b_pt} if keep <= swap else {arms[0]: b_pt, arms[1]: a_pt}
    # Top-down, or leaning each wrist outward (away from the other hand) by outward_tilt_deg.
    tilt = math.radians(float(outward_tilt_deg))
    approach: dict[int, np.ndarray] = {}
    for a, b in (arms, arms[::-1]):
        inward = target[b] - target[a]
        inward[2] = 0.0
        inward /= max(float(np.linalg.norm(inward)), 1e-9)
        approach[a] = math.cos(tilt) * DOWN + math.sin(tilt) * inward
    hover = {a: target[a] - approach[a] * float(pre_grasp) for a in arms}

    rotation: dict[int, dict[str, float]] = {}
    for a in arms:
        ctx.tool("robot.open_gripper", **arm(a))
        try:
            frame = ctx.tool(
                "robot.grasp_frame",
                approach=_vec(approach[a]),
                close_heading_deg=heading,
                **arm(a),
            )
            candidates = [
                frame[k]
                for k in ("rotation", "rotation_flipped")
                if isinstance(frame, dict) and frame.get(k)
            ]
        except Exception as exc:
            return finish("unreachable", f"no top-down rotation for arm {a} ({exc})")
        why = "no rotation offered"
        for candidate in candidates:
            why = move("robot.go_to_pose", a, hover[a], candidate)
            if not why:
                rotation[a] = candidate
                break
        if a not in rotation:
            return finish("unreachable", f"arm {a} to its hover over its grasp point: {why}")
        # The jaws' roll about the approach, read back; a solver that keeps only the
        # approach axis leaves it free, and the rest is turned on the last joint.
        commanded, gain = rotation[a], 1.0
        now = here(a)
        residual = _roll_residual(commanded, now["rotation"], approach[a])
        for _ in range(TURN_TRIES):
            if abs(residual) <= ROLL_TOLERANCE_DEG:
                break
            jq = _joints(ctx, a)
            if jq.size == 0:
                notes.append(f"no joint reading, arm {a}'s wrist was not turned")
                break
            command = math.radians(residual) / gain
            lo, hi = ROLL_LIMITS
            if not lo + LIMIT_MARGIN <= jq[-1] + command <= hi - LIMIT_MARGIN:
                command = math.radians(residual - math.copysign(180.0, residual)) / gain
                if not lo + LIMIT_MARGIN <= jq[-1] + command <= hi - LIMIT_MARGIN:
                    notes.append(f"arm {a}'s last joint has no room to turn {residual:+.0f} deg")
                    break
            turned = jq.copy()
            turned[-1] += command
            try:
                why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints",
                        joint_config={"positions": turned.tolist()},
                        **arm(a),
                    )
                )
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why:
                notes.append(f"arm {a} wrist turn: {why}")
                break
            before = residual
            now = here(a)
            residual = _roll_residual(commanded, now["rotation"], approach[a])
            moved = math.radians(_wrap180(before - residual))
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = move("robot.go_to_pose", a, hover[a], now["rotation"])
            if why:
                return finish("unreachable", f"re-centring arm {a} after its wrist turn: {why}")
            now = here(a)
            residual = _roll_residual(commanded, now["rotation"], approach[a])
        if abs(residual) > ROLL_TOLERANCE_DEG:
            notes.append(f"arm {a}'s jaws {residual:+.0f} deg off square to the object")
        rotation[a] = now["rotation"]

    names = {arms[0]: "first", arms[1]: "second"}
    for a in arms:
        out[f"{names[a]}_grasp_pose"] = {"position": _vec(target[a]), "rotation": rotation[a]}

    def back_off(route: str, detail: str) -> Output:
        for a in arms:
            ctx.tool("robot.open_gripper", **arm(a))
        for a in arms:
            backed = move("robot.go_to_pose_cartesian", a, hover[a], rotation[a])
            if backed:
                notes.append(f"arm {a} backing off: {backed}")
        return finish(route, detail)

    for a in arms:
        why = move("robot.go_to_pose_cartesian", a, target[a], rotation[a])
        if why:
            return back_off("unreachable", f"arm {a} down onto its grasp point: {why}")
    for a in arms:
        ctx.tool("robot.close_gripper", **arm(a))
    for a in arms:
        out[f"{names[a]}_grip"] = grip(a)
    empty = [a for a in arms if not holds(out[f"{names[a]}_grip"])]
    if empty:
        return back_off(
            "slipped",
            "arm "
            + " and ".join(str(a) for a in empty)
            + " closed to the stop at its grasp point: the object was not between the jaws",
        )

    base = {a: _xyz(here(a)) for a in arms}
    horizontal = float(np.linalg.norm(base[arms[0]][:2] - base[arms[1]][:2]))
    sep0 = float(np.linalg.norm(base[arms[0]] - base[arms[1]]))

    def status() -> tuple[dict[int, np.ndarray], float]:
        now = {a: _xyz(here(a)) for a in arms}
        rise = {a: float(now[a][2] - base[a][2]) for a in arms}
        out["rise_m"] = float(np.mean(list(rise.values())))
        out["hand_tilt_deg"] = math.degrees(
            math.atan2(rise[arms[0]] - rise[arms[1]], max(horizontal, 1e-9))
        )
        drift = float(np.linalg.norm(now[arms[0]] - now[arms[1]])) - sep0
        if abs(drift) > abs(out["separation_change_m"]):
            out["separation_change_m"] = drift
        centre = _point(object_obb["center"]) + np.array([0.0, 0.0, out["rise_m"]])
        out["object_obb"] = {**object_obb, "center": _vec(centre)}
        return now, drift

    steps = max(1, math.ceil(float(lift_height) / float(lift_step) - 1e-9))
    for k in range(1, steps + 1):
        dz = float(lift_height) * k / steps
        order = arms if k % 2 else arms[::-1]
        for i, a in enumerate(order):
            why = move(
                "robot.go_to_pose_cartesian",
                a,
                base[a] + np.array([0.0, 0.0, dz]),
                rotation[a],
                level=True,
            )
            if not why:
                continue
            # Bring the hand that is ahead back level with the one that stopped.
            other = order[1 - i]
            now, _ = status()
            stuck = now[a][2] - base[a][2]
            if now[other][2] - base[other][2] > stuck + float(level_tolerance):
                levelled = move(
                    "robot.go_to_pose_cartesian",
                    other,
                    base[other] + np.array([0.0, 0.0, stuck]),
                    rotation[other],
                )
                notes.append(
                    f"arm {other} brought back level"
                    + (f" but: {levelled}" if levelled else f" ({stuck * 1e3:.0f} mm up)")
                )
            status()
            return finish(
                "unreachable",
                f"step {k}/{steps}, arm {a} up to {dz * 1e3:.0f} mm: {why}; both hands still closed on the object",
            )
        _, drift = status()
        for a in arms:
            out[f"{names[a]}_grip"] = grip(a)
        empty = [a for a in arms if not holds(out[f"{names[a]}_grip"])]
        if empty:
            return finish(
                "slipped",
                f"after step {k}/{steps} ({out['rise_m'] * 1e3:.0f} mm up) arm "
                + " and ".join(str(a) for a in empty)
                + "'s jaws ran to the stop",
            )
        if abs(drift) > float(separation_tolerance):
            return finish(
                "slipped",
                f"after step {k}/{steps} ({out['rise_m'] * 1e3:.0f} mm up) the hands' separation "
                f"changed {drift * 1e3:+.0f} mm: a hand slid along the object or an arm lags",
            )
    return finish("lifted", "")
