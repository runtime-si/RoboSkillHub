# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill),
# mani_skill/examples/motionplanning/panda/solutions/stack_cube.py (solve, the
# "Search a valid pose" and "Stack" steps), Apache-2.0, and from RoboTwin
# (https://github.com/robotwin-Platform/RoboTwin), envs/_base_task.py
# (get_place_pose, place_actor) and envs/stack_blocks_three.py, MIT,
# Copyright (c) 2025 Tianxing Chen (license: ../LICENSE.robotwin).
# Changes: the cubes' simulator poses are the held object's and the target's
# oriented boxes (world frame, measured by perception); the goal is the
# target's top face centre (ManiSkill: the target pose plus one cube height),
# reached by carrying the held object's measured offset in the hand; the hand
# is turned about the vertical so the held box's faces line up with the
# target's (RoboTwin's start2target alignment, folded to the box's 90 deg face
# symmetry), read back from the hand and finished on the last joint
# (robot.move_to_joints) when the arm's solver leaves the wrist roll free, and
# ManiSkill's dry-run yaw search over 0 / +-90 deg is a robot.solve_ik preview
# over the equivalent turns; ManiSkill's straight move
# to the goal and RoboTwin's pre_dis / dis moves along the target's up axis are
# the hub's placing-with-contact-descent descent (approach above, two
# robot.move_cartesian_until_contact moves judged by the hand's measured stop),
# because a real stack's height is known only to the camera's error; a stop far
# from the target's top, a descent that meets nothing or a stop that pushed
# the hand sideways is routed and the object kept instead of released.
"""Stack a held object centred on top of a target object's box: turn to its faces, come down until contact, release."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

APPROACH_HEIGHT = 0.08
"""Height of the approach over the expected contact [m] (placing-with-contact-descent's;
ManiSkill lifts 0.1 m before the stack move, RoboTwin approaches 0.05 m over the
target along its up axis). Clears a 30 mm block standing beside the target with a
30 mm part hanging under the hand."""

OVERTRAVEL = 0.02
"""How far below the expected contact the descent is sent [m], before the floor
over the target's top (placing-with-contact-descent's: deeper than the 12 mm a
friction-held cube slides up the jaws in a YAM station twin)."""

CONTACT_TOLERANCE = 0.012
"""How far above the expected contact a stop still counts as the target's top
[m]: about two standard deviations of a calibrated YAM station's camera error
(RMS 5.4 mm), as in placing-with-contact-descent."""

MIN_CLEARANCE = 0.012
"""Lowest tool-centre height over the target's top [m]. Closed linear_4310
fingertips reach about 10 mm under the tool centre (measured in a YAM station
twin); 2 mm more keeps the jaws off the target's top face."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool-centre height over the table, when table_z is given [m]. A real
YAM station refuses a tool centre under 14 mm over its calibrated table."""

LATERAL_TOLERANCE = 0.008
"""How far the hand may stop from its planned xy and still release [m]. On a
30 mm block the held object's centre then stays 7 mm inside the target's edge;
the jaws catching an edge pushed the hand 7 mm sideways in a YAM station twin
(placing-with-contact-descent, box corner)."""

MAX_TARGET_TILT_DEG = 10.0
"""Steepest target top that is stacked on [deg]: a block tilted more rests on
something else and its top is not a support. A choice, not tuned."""

YAW_PERIOD_DEG = 90.0
"""The held box's faces line up with the target's every 90 deg (two boxes).
180 lines up the boxes' x axes, 360 a full heading."""

YAW_TOLERANCE_DEG = 3.0
"""How far the hand's turn may end from the planned one before the rest is
turned on the last joint [deg]: a servo's own 3 deg arrival bound (a real YAM
station's is 0.04 rad, 2.3 deg)."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back and re-centred (the first one
learns which way the joint turns the hand)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad] when the caller states none: the i2rt YAM's joint 6
(+-120 deg, its MJCF; pouring-by-wrist-roll's value)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the approach or the retreat [m]: no tighter
than a real YAM station's own 12 mm per-move bound."""

MIN_DESCENT = 0.002
"""A refused contact move that descended less than this did not move at all [m]."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped by contact."""

CONTACT_MOVES = 3
"""Contact moves sent while one times out still descending (a servo's move has a
fixed step budget)."""


class Output(TypedDict):
    route: str
    detail: str
    released: bool
    yaw_turn_deg: float
    yaw_error_deg: float
    expected_contact_z: float
    contact_z: float
    contact_offset: float
    stacked_position: dict[str, float]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _wxyz(q: dict[str, float]) -> np.ndarray:
    return np.array([float(q[k]) for k in ("w", "x", "y", "z")])


def _quat(q: np.ndarray) -> dict[str, float]:
    return dict(zip(("w", "x", "y", "z"), map(float, q), strict=True))


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _matrix(q: dict[str, float] | None) -> np.ndarray:
    w, x, y, z = _wxyz(q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


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


def _wrap(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def hand_yaw(rotation: dict[str, float]) -> float:
    """The heading of the hand's x axis in the horizontal plane [deg]."""
    r = _matrix(rotation)
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


def box_frame(obb: dict[str, Any]) -> dict[str, float]:
    """An oriented box seen from above: its centre, its top and bottom heights
    (highest and lowest corner), its tilt (deg between world up and the nearest
    face axis) and its yaw (deg, heading of a horizontal face axis; any of its
    faces' headings, the others are 90 deg on)."""
    r = _matrix(obb.get("orientation"))
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    reach_z = float(np.abs(r[2, :]) @ half)
    up = int(np.argmax(np.abs(r[2, :])))
    side = r[:, (up + 1) % 3]
    return {
        "x": float(centre[0]),
        "y": float(centre[1]),
        "z": float(centre[2]),
        "top": float(centre[2]) + reach_z,
        "bottom": float(centre[2]) - reach_z,
        "tilt": math.degrees(math.acos(min(1.0, abs(float(r[2, up]))))),
        "yaw": math.degrees(math.atan2(float(side[1]), float(side[0]))),
    }


def yaw_turns(held_yaw: float, target_yaw: float, period: float) -> list[float]:
    """The hand turns [deg] that line the held box's faces up with the target's,
    smallest first: the nearest one and one period either side of it (ManiSkill
    searches its grasp at 0 and +-90 deg)."""
    p = float(period)
    base = (float(target_yaw) - float(held_yaw) + p / 2.0) % p - p / 2.0
    turns = [base, base + p, base - p] if p < 360.0 else [base]
    return sorted(turns, key=lambda t: (abs(t), t))


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    held_obb: dict[str, Any],
    table_z: float | None = None,
    align_yaw: bool = True,
    yaw_period_deg: float = YAW_PERIOD_DEG,
    yaw_tolerance_deg: float = YAW_TOLERANCE_DEG,
    roll_limits: list[float] | None = None,
    approach_height: float = APPROACH_HEIGHT,
    overtravel: float = OVERTRAVEL,
    contact_tolerance: float = CONTACT_TOLERANCE,
    min_clearance: float = MIN_CLEARANCE,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    lateral_tolerance: float = LATERAL_TOLERANCE,
    max_target_tilt_deg: float = MAX_TARGET_TILT_DEG,
    retreat_height: float | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    start_rot = start["rotation"]
    tcp = _xyz(start)
    target, held = box_frame(target_obb), box_frame(held_obb)
    top = target["top"]

    # The goal: the held box's bottom centre on the target's top centre
    # (ManiSkill's target pose plus one cube height), reached by carrying the
    # held object's offset in the hand, measured now.
    goal = np.array([target["x"], target["y"], top])
    offset_xy = np.array([held["x"], held["y"]]) - tcp[:2]
    hang = float(tcp[2] - held["bottom"])
    expected = top + hang

    out: Output = {
        "route": "blocked",
        "detail": "",
        "released": False,
        "yaw_turn_deg": 0.0,
        "yaw_error_deg": 0.0,
        "expected_contact_z": float(expected),
        "contact_z": float(tcp[2]),
        "contact_offset": 0.0,
        "stacked_position": _vec(goal),
    }

    def blocked(why: str) -> Output:
        out["route"], out["detail"] = "blocked", why
        return out

    if target["tilt"] > float(max_target_tilt_deg):
        return blocked(
            f"the target's top is tilted {target['tilt']:.0f} deg "
            f"(over {float(max_target_tilt_deg):.0f}); it is not a support"
        )

    floor = top + float(min_clearance)
    if table_z is not None:
        floor = max(floor, float(table_z) + float(min_tool_height))
    bottom_z = max(expected - float(overtravel), floor)
    lift = float(approach_height if retreat_height is None else retreat_height)

    def turned(offset_turn: float) -> np.ndarray:
        """The held object's offset in the hand after the hand turned this much
        [deg] about the vertical through the tool centre."""
        a = math.radians(offset_turn)
        c, s = math.cos(a), math.sin(a)
        return np.array([c * offset_xy[0] - s * offset_xy[1], s * offset_xy[0] + c * offset_xy[1]])

    def plan(turn: float) -> dict[str, float]:
        a = math.radians(turn)
        qz = np.array([math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)])
        return _quat(_mul(qz, _wxyz(start_rot)))

    turns = yaw_turns(held["yaw"], target["yaw"], yaw_period_deg) if align_yaw else [0.0]
    chosen: float | None = None
    refused: list[str] = []
    for turn in turns:
        if not check_reach:
            chosen = turn
            break
        contact_pose = np.array([*(goal[:2] - turned(turn)), max(expected, bottom_z)])
        try:
            reply = ctx.tool(
                "robot.solve_ik",
                pose={"position": _vec(contact_pose), "rotation": plan(turn)},
                **on_arm,
            )
        except Exception as exc:
            logger.info("stacking-objects: no IK preview (%s)", exc)
            reply = None
        # A stall at the edge of reach reads exactly like contact.
        if isinstance(reply, dict) and (
            reply.get("success") is False or reply.get("within_servo_tolerance") is False
        ):
            refused.append(f"{turn:+.0f}")
            continue
        chosen = turn
        break
    if chosen is None:
        return blocked(
            f"the hand cannot reach the contact pose over the target at any aligned turn "
            f"({', '.join(refused)} deg; IK preview)"
        )
    turn = chosen
    approach_z = expected + float(approach_height)

    def move(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        try:
            reply = ctx.tool(tool, pose={"position": _vec(position), "rotation": rot}, **on_arm)
            return _refusal(reply)
        except Exception as exc:  # a refused or failed move is routed, not raised
            return str(exc)

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def where() -> np.ndarray:
        return _xyz(here())

    def go(tool: str, position: np.ndarray, rot: dict[str, float]) -> str:
        """A free move judged by where the hand is: "" when it arrived within
        reach_tolerance (a servo's timeout that ended that close has arrived),
        otherwise why not."""
        why = move(tool, position, rot)
        off = float(np.linalg.norm(where() - position))
        if (why and not why.startswith("timeout")) or off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand is {off * 1e3:.0f} mm from it"
        return ""

    def joints() -> np.ndarray:
        try:
            arms = ctx.tool("robot.get_observation").get("arms") or []
        except Exception as exc:
            logger.info("stacking-objects: no joint reading (%s)", exc)
            return np.zeros(0)
        if int(arm_id) >= len(arms):
            return np.zeros(0)
        q = arms[int(arm_id)].get("joint_state")
        if isinstance(q, dict):
            q = q.get("positions")
        return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)

    # Straight up first if the hand is under the approach height (it holds
    # something), keeping its rotation; then over to the approach, turning.
    if tcp[2] < approach_z - 1.0e-3:
        why = go("robot.go_to_pose_cartesian", np.array([tcp[0], tcp[1], approach_z]), start_rot)
        if why:
            return blocked(f"rise: {why}")
    why = go("robot.go_to_pose", np.array([*(goal[:2] - turned(turn)), approach_z]), plan(turn))
    if why:
        return blocked(f"approach: {why}")

    # The turn, read back: an arm that holds the whole pose arrives turned; one
    # whose solver keeps only the approach axis (a free wrist roll) does not, and
    # the rest is turned on its last joint. The tool centre is off that joint's
    # axis on some arms, so the hand is re-centred after each turn.
    start_yaw = hand_yaw(start_rot)
    gain = -1.0 if _matrix(plan(turn))[2, 2] < 0 else 1.0
    lo, hi = (float(v) for v in (roll_limits or ROLL_LIMITS))
    notes: list[str] = []
    now = here()
    for _ in range(TURN_TRIES if align_yaw else 0):
        residual = _wrap(turn - _wrap(hand_yaw(now["rotation"]) - start_yaw))
        if abs(residual) <= float(yaw_tolerance_deg):
            break
        q = joints()
        if q.size == 0:
            notes.append("no joint reading, the wrist was not turned")
            break
        room = (lo + LIMIT_MARGIN - q[-1], hi - LIMIT_MARGIN - q[-1])
        options = [residual, residual - math.copysign(float(yaw_period_deg), residual)]
        fits = [r for r in options if room[0] <= math.radians(r) / gain <= room[1]]
        if not fits:
            notes.append(f"the last joint has no room to turn {residual:+.0f} deg")
            break
        if fits[0] != residual:
            turn += fits[0] - residual
        command = math.radians(fits[0]) / gain
        target_q = q.copy()
        target_q[-1] += command
        kwargs: dict[str, Any] = {"joint_config": {"positions": target_q.tolist()}, **on_arm}
        try:
            why = _refusal(ctx.tool("robot.move_to_joints", **kwargs))
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        if why:
            notes.append(f"wrist turn: {why}")
            break
        before = hand_yaw(now["rotation"])
        now = here()
        moved = math.radians(_wrap(hand_yaw(now["rotation"]) - before))
        if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
            gain = moved / command
        done = _wrap(hand_yaw(now["rotation"]) - start_yaw)
        why = go(
            "robot.go_to_pose", np.array([*(goal[:2] - turned(done)), approach_z]), now["rotation"]
        )
        if why:
            return blocked(f"re-centring after the wrist turn: {why}")
        now = here()
    rotation = now["rotation"]
    done = _wrap(hand_yaw(rotation) - start_yaw)
    held_off = turned(done)
    hand_xy = goal[:2] - held_off
    approach = np.array([*hand_xy, approach_z])
    bottom = np.array([*hand_xy, bottom_z])
    out["yaw_turn_deg"] = float(done)
    p = float(yaw_period_deg)
    out["yaw_error_deg"] = float((target["yaw"] - held["yaw"] - done + p / 2.0) % p - p / 2.0)
    if not align_yaw:
        out["yaw_error_deg"] = 0.0

    def descend(begin: np.ndarray, goal_at: np.ndarray) -> tuple[np.ndarray, str, bool]:
        """One contact move (repeated while it times out still descending): where
        the hand stopped, why, and whether it stopped short of the goal."""
        stop, why = begin, ""
        for _ in range(CONTACT_MOVES):
            before = stop
            why = move("robot.move_cartesian_until_contact", goal_at, rotation)
            stop = where()
            if not why.startswith("timeout") or before[2] - stop[2] < MIN_DESCENT:
                break
        return stop, why, bool(why) and float(stop[2] - goal_at[2]) > MIN_DESCENT

    tol = float(contact_tolerance)

    def record(stop: np.ndarray) -> float:
        d = float(stop[2] - expected)
        out["contact_z"], out["contact_offset"] = float(stop[2]), d
        out["stacked_position"] = _vec(np.array([*(stop[:2] + held_off), stop[2] - hang]))
        return d

    def back_up(stop: np.ndarray, why: str) -> Output:
        """Not released: lift the object back to the approach height."""
        out["route"] = "misaligned"
        up = go("robot.go_to_pose_cartesian", np.array([stop[0], stop[1], approach[2]]), rotation)
        out["detail"] = why + (f"; lifting back: {up}" if up else "; lifted back, still holding")
        return out

    # The descent until contact, in two moves (placing-with-contact-descent): to
    # the top of the contact window, where a stop is an early contact read before
    # the hand has pressed far, then on through it.
    window_top = np.array([*hand_xy, max(expected + tol, bottom_z)])
    stop, why, stopped = descend(approach, window_top)
    if why.startswith(REFUSALS) or (
        why and float(approach[2] - stop[2]) < MIN_DESCENT and not why.startswith("contact")
    ):
        return blocked(f"descent: {why}")
    if stopped:
        d = record(stop)
        return back_up(
            stop,
            f"stopped {d * 1e3:.0f} mm above the target's top (tolerance {tol * 1e3:.0f} mm): "
            "something higher is under the held object, or it hangs lower than measured",
        )
    stop, why, stopped = descend(stop, bottom)
    d = record(stop)
    if why.startswith(REFUSALS):
        return blocked(f"descent through the contact window: {why}; still holding")
    if not stopped:
        return back_up(
            stop,
            f"reached the bottom of the descent, {-d * 1e3:.0f} mm below the expected contact, "
            f"without stopping ({why or 'the move finished'}): the held object missed the "
            "target's top, or slid up the jaws",
        )
    lateral = float(np.linalg.norm(stop[:2] - hand_xy))
    if lateral > float(lateral_tolerance):
        return back_up(
            stop,
            f"stopped on contact but {lateral * 1e3:.0f} mm sideways of the planned point "
            f"(tolerance {float(lateral_tolerance) * 1e3:.0f} mm): caught on an edge",
        )

    ctx.tool("robot.open_gripper", **on_arm)
    out["released"] = True
    why = go("robot.go_to_pose_cartesian", np.array([stop[0], stop[1], stop[2] + lift]), rotation)
    if why:
        return blocked(f"released, but the retreat: {why}")
    out["route"] = "stacked"
    out["detail"] = (
        f"turned {done:+.1f} deg ({out['yaw_error_deg']:+.1f} deg off the target's faces), stopped on contact "
        f"{d * 1e3:+.1f} mm from the expected height, {lateral * 1e3:.1f} mm sideways; released"
    )
    if notes:
        out["detail"] += "; " + "; ".join(notes)
    return out
