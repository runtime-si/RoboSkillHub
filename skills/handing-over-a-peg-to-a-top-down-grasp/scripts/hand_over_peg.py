# The two-arm sequence (one arm moves at a time, every move read back, the
# receiver's hold read before the giver opens and after each later move) and
# the receiver's wrist-roll finish are generalised from the hub's
# handing-over-between-arms (MIT, adapted from RoboTwin envs/handover_block.py
# and ACT scripted_policy.py, both MIT). Changed for a vertical peg: the
# receiver comes down along -z onto the free top end instead of in from the
# side; its jaws close along the giver's own jaw axis so its open fingers do
# not pass over the giver's housing; how deep it bites is set by the gap left
# above the giver's pads and refused below a minimum before anything moves; the
# giver backs out horizontally the way its hand came in; and the result is the
# peg's pose in the receiver's tool frame, estimated from both arms' poses.
"""Hand a vertical peg held by a side grasp on one arm to a top-down grasp on the other."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TIP_REACH = 0.0102
"""How far the receiver's closed fingertips reach below its tool centre [m]:
the linear_4310's tip reach as the YAM station twin registers it (PORTING
lesson 6: ~10 mm)."""

PAD_HALF = 0.00775
"""Half the giver's fingertip pad along the peg [m]: the real linear_4310 tip is
15.5 mm wide, and a side grasp stands it across the shaft's length."""

MIN_FINGER_GAP = 0.005
"""Least gap kept between the receiver's fingertips and the top of the giver's
pads along the peg [m]: checking-dual-arm-clearance's too_close (5 mm)."""

BITE_MAX = 0.012
"""Deepest the receiver's tool centre goes below the peg's top [m]. tuned_on:
YAM station twin, 10 x 100 mm peg (see SKILL.md)."""

BITE_MIN = 0.003
"""Shallowest bite worth closing on [m]: under it the pads' upper edge is over
the top of the peg and the hand would close on its end face."""

PRE_GRASP = 0.08
"""Receiver's stand-off above its grasp pose before it comes down [m]
(handing-over-between-arms' 0.10 less the peg's own exposed length)."""

RETREAT_DISTANCE = 0.08
"""Giver's horizontal back-out after it opens [m]: its open fingers (47 mm
either side) and pads leave the shaft's footprint with the 15.5 mm pad and a
margin (RoboTwin handover_block backs out 0.07-0.10)."""

RECEIVER_LIFT = 0.02
"""Receiver's lift with the peg after the giver is clear [m], so the last hold
check carries the peg's weight on the receiver alone."""

REACH_TOLERANCE = 0.015
"""How far a hand may end from where it was sent [m]: a servo's 4 mm ball plus a
real YAM's 12 mm per move."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (1.0 = open) closed on nothing: 3.8 mm on the
YAM's 94 mm span (handing-over-between-arms)."""

JAW_SPAN = 0.094
"""YAM linear_4310 span at full open [m] (opening 1.0)."""

WIDTH_TOLERANCE = 0.008
"""How much wider than the peg's diameter the receiver's jaws may read and still
be across the shaft [m]. Wider means a finger, a jaw tip or the giver's pad.
tuned_on: YAM station twin (see SKILL.md)."""

OPEN_FRACTION = 0.5
"""A giver reading under this after open_gripper did not open."""

MAX_TILT_DEG = 15.0
"""How far the peg's axis may lean from vertical for a straight -z approach [deg]."""

LEFT_SIGMA = 0.002
"""Assumed 1-sigma of the giver's T_TCP_object when left_grip gives none [m]."""

REPEAT_SIGMA = 0.001
"""Per-arm repeatability folded into sigma_mm [m] (YAM, joint encoders)."""

ROLL_TOLERANCE_DEG = 3.0
"""How far the receiver's jaw roll may end from the commanded one before the
rest is turned on its last joint [deg] (handing-over-between-arms)."""

TURN_TRIES = 3
ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad]: the i2rt YAM's joint 6."""
LIMIT_MARGIN = 0.05
SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""


class Output(TypedDict):
    route: str
    detail: str
    held: dict[str, Any]
    receiver_grip: float
    bite_m: float
    finger_gap_m: float
    receiver_grasp_pose: dict[str, Any]
    peg_top: dict[str, float]


# ------------------------------------------------------------------ geometry
def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = _pose(pose)["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rotation(q: dict[str, float]) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat(r: np.ndarray) -> dict[str, float]:
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (r[j, i] + r[i, j]) / s
        v[k] = (r[k, i] + r[i, k]) / s
        q = ((r[k, j] - r[j, k]) / s, *v)
    return dict(zip("wxyz", (float(c) for c in q), strict=True))


def _matrix(pose: Any) -> np.ndarray:
    """A 4x4 from an Se3Pose dict ({position, rotation wxyz}) or a 4x4 nested list."""
    if isinstance(pose, dict):
        pose = _pose(pose)
        t = np.eye(4)
        t[:3, :3] = _rotation(pose.get("rotation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        t[:3, 3] = _xyz(pose)
        return t
    t = np.asarray(pose, dtype=float)
    if t.shape != (4, 4):
        raise ValueError(f"a transform must be an Se3Pose or a 4x4, not shape {t.shape}")
    return t


def _se3(t: np.ndarray) -> dict[str, Any]:
    return {"position": _vec(t[:3, 3]), "rotation": _quat(t[:3, :3])}


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a
    refusal instead of raising; PORTING lesson 1)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _wrap180(deg: float) -> float:
    """Folded to [-90, 90): parallel jaws look the same turned half a turn."""
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


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


# ------------------------------------------------------------------ the skill
def run(
    ctx: NodeContext,
    peg: dict[str, Any],
    left_grip: dict[str, Any],
    giver_arm: int = 0,
    receiver_arm: int = 1,
    handover_position: dict[str, float] | None = None,
    close_heading_deg: float | None = None,
    giver_retreat: dict[str, float] | None = None,
    observed_tip: dict[str, float] | None = None,
    observed_sigma_mm: float = 2.0,
    bite_max: float = BITE_MAX,
    min_finger_gap: float = MIN_FINGER_GAP,
    pre_grasp: float = PRE_GRASP,
    retreat_distance: float = RETREAT_DISTANCE,
    receiver_lift: float = RECEIVER_LIFT,
    reach_tolerance: float = REACH_TOLERANCE,
    empty_fraction: float = EMPTY_FRACTION,
    width_tolerance: float = WIDTH_TOLERANCE,
    max_tilt_deg: float = MAX_TILT_DEG,
) -> Output:
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "held": {},
        "receiver_grip": -1.0,
        "bite_m": 0.0,
        "finger_gap_m": 0.0,
        "receiver_grasp_pose": {},
        "peg_top": {},
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join([d for d in (detail, *notes) if d])
        return out

    def arm(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": int(a)} if int(a) else {}

    def here(a: int) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def move(tool: str, a: int, position: np.ndarray, rotation: dict[str, float]) -> str:
        """Send one arm and judge arrival from a fresh reading of that arm (lessons 1, 3)."""
        try:
            why = _refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}, **arm(a))
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        off = float(np.linalg.norm(_xyz(here(a)) - position))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; arm {a} stopped {off * 1e3:.0f} mm from its target"
        return ""

    def grip(a: int) -> float:
        """The jaw opening (1.0 = open), or -1 when the hand cannot be read."""
        try:
            return float(ctx.tool("robot.get_gripper", **arm(a))["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning("handing-over-a-peg: arm %d gripper unreadable (%s)", a, exc)
            return -1.0

    def holds(fraction: float) -> bool:
        return fraction < 0.0 or fraction > float(empty_fraction)

    # 1. Two separate arms, the giver holding something.
    gv, rc = int(giver_arm), int(receiver_arm)
    if gv == rc:
        return finish("unreachable", f"giver and receiver are both arm {gv}")
    left0, right0 = here(gv), here(rc)
    q0 = right0.get("rotation") or {}
    if sum(float(q0.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_xyz(left0) - _xyz(right0))) < SAME_ARM
    ):
        return finish("unreachable", f"the connector does not report a separate arm {rc}")
    left_open = grip(gv)
    if not holds(left_open):
        return finish("dropped", f"the giver's jaws are at the stop ({left_open:.3f}): no peg")

    length = float(peg["length_m"])
    diameter = float(peg["diameter_m"])
    axis_obj = _unit(np.asarray(peg.get("axis_in_object", [0.0, 0.0, 1.0]), dtype=float))
    if axis_obj is None or length <= 0 or diameter <= 0:
        return finish("unreachable", "peg needs a positive length_m, diameter_m and an axis")
    tip_sign = -1.0 if str(peg.get("tip_end", "-z")).startswith("-") else 1.0
    t_tcp_obj_left = _matrix(left_grip["T_TCP_object"])

    # 2. The giver to the handover point (optional), keeping its rotation.
    if handover_position is not None:
        t_w_obj = _matrix(left0) @ t_tcp_obj_left
        target = np.array([float(handover_position[k]) for k in "xyz"])
        tcp = _xyz(left0) + (target - t_w_obj[:3, 3])
        why = move("robot.go_to_pose", gv, tcp, left0["rotation"])
        if why:
            return finish(
                "unreachable", f"giver to the handover point: {why}; it still holds the peg"
            )
        left0 = here(gv)

    # 3. The peg in the world from the giver's hand: vertical, free end up.
    t_w_left = _matrix(left0)
    t_w_obj = t_w_left @ t_tcp_obj_left
    axis_w = t_w_obj[:3, :3] @ axis_obj
    centre = t_w_obj[:3, 3]
    up = 1.0 if axis_w[2] >= 0 else -1.0
    axis_up = axis_w * up
    tilt = _angle_between(axis_up, np.array([0.0, 0.0, 1.0]))
    if tilt > float(max_tilt_deg):
        return finish(
            "unreachable",
            f"the peg leans {tilt:.0f} deg from vertical (over {max_tilt_deg:.0f}): no straight top-down approach",
        )
    top = centre + axis_up * length / 2.0
    out["peg_top"] = _vec(top)
    if up * tip_sign > 0:
        notes.append("the peg's tip_end points up: the receiver takes the tip end")
    # The giver's pads along the peg: its tool centre projected on the axis.
    pads_top = float(np.dot(_xyz(left0) - centre, axis_up)) + PAD_HALF
    exposed = length / 2.0 - pads_top
    if left_grip.get("exposed_len_m") is not None:
        exposed = min(exposed, float(left_grip["exposed_len_m"]))

    # 4. How deep to bite, pre-checked against the giver's pads.
    bite = min(float(bite_max), exposed - TIP_REACH - float(min_finger_gap))
    out["bite_m"] = bite
    out["finger_gap_m"] = exposed - max(bite, BITE_MIN) - TIP_REACH
    if bite < BITE_MIN:
        return finish(
            "collision_risk",
            f"only {exposed * 1e3:.0f} mm of peg stands above the giver's pads: a {BITE_MIN * 1e3:.0f} mm bite"
            f" would bring the receiver's fingertips within {out['finger_gap_m'] * 1e3:.1f} mm of them"
            f" (min {min_finger_gap * 1e3:.0f}); nothing moved",
        )

    # 5. The receiver's jaw axis: along the giver's own jaw axis, so its open
    #    fingers pass beside the giver's pads and not over its housing.
    toward_receiver = _xyz(right0) - centre
    toward_receiver -= float(np.dot(toward_receiver, axis_up)) * axis_up
    giver_close, giver_approach = _giver_axes(
        ctx, gv, t_w_left[:3, :3], axis_up, toward_receiver, notes
    )
    heading = (
        math.degrees(math.atan2(giver_close[1], giver_close[0]))
        if close_heading_deg is None
        else float(close_heading_deg)
    )
    down = -axis_up
    grasp = top - axis_up * bite
    pre = grasp + axis_up * float(pre_grasp)

    ctx.tool("robot.open_gripper", **arm(rc))
    try:
        frame = ctx.tool(
            "robot.grasp_frame", approach=_vec(down), close_heading_deg=heading, **arm(rc)
        )
        rotations = [
            frame[k]
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:
        return finish("unreachable", f"no receiver rotation for a top-down approach ({exc})")
    rotation, why = None, "no rotation offered"
    for candidate in rotations:
        why = move("robot.go_to_pose", rc, pre, candidate)
        if not why:
            rotation = candidate
            break
    if rotation is None:
        return finish("unreachable", f"receiver over the peg: {why}; the giver still holds the peg")

    # 6. The receiver's jaw roll read back and finished on its last joint
    #    (PORTING lesson 10: the twin's IK keeps only the approach axis).
    commanded, gain = rotation, 1.0
    now = here(rc)
    residual = _roll_residual(commanded, now["rotation"], down)
    for _ in range(TURN_TRIES):
        if abs(residual) <= ROLL_TOLERANCE_DEG:
            break
        q = _joints(ctx, rc)
        if q.size == 0:
            notes.append("no joint reading, the receiver's wrist was not turned")
            break
        command = math.radians(residual) / gain
        lo, hi = ROLL_LIMITS
        if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
            command = math.radians(residual - math.copysign(180.0, residual)) / gain
            if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                notes.append(f"the receiver's last joint has no room to turn {residual:+.0f} deg")
                break
        target_q = q.copy()
        target_q[-1] += command
        try:
            why = _refusal(
                ctx.tool(
                    "robot.move_to_joints",
                    joint_config={"positions": target_q.tolist()},
                    **arm(rc),
                )
            )
        except Exception as exc:
            why = str(exc) or type(exc).__name__
        if why:
            notes.append(f"receiver wrist turn: {why}")
            break
        before = residual
        now = here(rc)
        residual = _roll_residual(commanded, now["rotation"], down)
        moved = math.radians(_wrap180(before - residual))
        if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
            gain = moved / command
        why = move("robot.go_to_pose", rc, pre, now["rotation"])
        if why:
            return finish("unreachable", f"re-centring the receiver after its wrist turn: {why}")
        now = here(rc)
        residual = _roll_residual(commanded, now["rotation"], down)
    if abs(residual) > 20.0:
        return finish(
            "collision_risk",
            f"the receiver's jaws stay {residual:+.0f} deg off the giver's jaw axis: its open fingers"
            " would come down over the giver's hand; it waits over the peg, the giver still holds it",
        )
    rotation = now["rotation"]
    out["receiver_grasp_pose"] = {"position": _vec(grasp), "rotation": rotation}

    def back_off(route: str, detail: str) -> Output:
        ctx.tool("robot.open_gripper", **arm(rc))
        backed = move("robot.go_to_pose_cartesian", rc, pre, rotation)
        return finish(route, detail + (f"; backing off: {backed}" if backed else ""))

    # 7. Down onto the free end, close, and confirm the hold before the giver lets go.
    why = move("robot.go_to_pose_cartesian", rc, grasp, rotation)
    if why:
        return back_off(
            "receiver_missed", f"receiver down onto the peg: {why}; the giver still holds it"
        )
    ctx.tool("robot.close_gripper", **arm(rc))
    g = out["receiver_grip"] = grip(rc)
    if not holds(g):
        return back_off(
            "receiver_missed",
            f"the receiver's jaws closed to the stop ({g:.3f}); the giver still holds the peg",
        )
    if g >= 0.0 and g * JAW_SPAN > diameter + float(width_tolerance):
        return back_off(
            "receiver_missed",
            f"the receiver's jaws stopped at {g * JAW_SPAN * 1e3:.0f} mm on a {diameter * 1e3:.0f} mm peg:"
            " not across the shaft (a jaw tip, the peg's end or the giver's pad); the giver still holds it",
        )
    t_w_right = _matrix(here(rc))
    t_w_left_close = _matrix(here(gv))
    pushed = float(np.linalg.norm(t_w_left_close[:3, 3] - t_w_left[:3, 3]))

    # 8. The giver opens and backs out horizontally the way its hand came in.
    opened = ctx.tool("robot.open_gripper", **arm(gv))
    giver_open = float(opened.get("position", -1.0)) if isinstance(opened, dict) else -1.0
    if giver_open < 0.0:
        giver_open = grip(gv)
    if 0.0 <= giver_open < OPEN_FRACTION:
        return finish(
            "unreachable",
            f"the giver's jaws read {giver_open:.3f} after opening; both hands hold the peg",
        )
    g = out["receiver_grip"] = grip(rc)
    if not holds(g):
        return finish(
            "dropped", f"the receiver's jaws ran to the stop ({g:.3f}) when the giver let go"
        )
    back = None
    if giver_retreat is not None:
        back = _unit(np.array([float(giver_retreat[k]) for k in "xyz"]))
    if back is None:
        back = -giver_approach
    retreat = t_w_left_close[:3, 3] + back * float(retreat_distance)
    why = move("robot.go_to_pose_cartesian", gv, retreat, _quat(t_w_left_close[:3, :3]))
    if why:
        return finish(
            "unreachable",
            f"the giver opened but could not back out: {why}; the receiver holds the peg",
        )
    g = out["receiver_grip"] = grip(rc)
    if not holds(g):
        return finish(
            "dropped",
            f"the receiver's jaws ran to the stop ({g:.3f}) as the giver backed out",
        )

    # 9. The receiver lifts the peg on its own and is read once more.
    if float(receiver_lift) > 0:
        lift_why = move(
            "robot.go_to_pose_cartesian",
            rc,
            t_w_right[:3, 3] + axis_up * float(receiver_lift),
            rotation,
        )
        if lift_why:
            notes.append(f"receiver lift: {lift_why}")
        g = out["receiver_grip"] = grip(rc)
        if not holds(g):
            return finish("dropped", f"the receiver's jaws ran to the stop ({g:.3f}) on the lift")

    # 10. The peg in the receiver's hand, from both arms' poses at the close.
    t_w_obj = t_w_left_close @ t_tcp_obj_left
    h = math.radians(heading)
    local_close = _rotation(commanded).T @ np.array([math.cos(h), math.sin(h), 0.0])
    right_close = _unit(t_w_right[:3, :3] @ local_close)
    if right_close is not None:
        # The receiver's jaws centre the shaft across them.
        offset = float(np.dot(t_w_obj[:3, 3] - t_w_right[:3, 3], right_close))
        t_w_obj[:3, 3] -= offset * right_close
    t_tcp_obj = np.linalg.inv(t_w_right) @ t_w_obj
    axis_tcp = t_tcp_obj[:3, :3] @ axis_obj
    approach_tcp = _rotation(commanded).T @ down
    axis_down = axis_tcp if float(np.dot(axis_tcp, approach_tcp)) >= 0 else -axis_tcp
    tip_tcp = t_tcp_obj[:3, 3] + axis_down * length / 2.0
    tip_offset = float(np.dot(tip_tcp, approach_tcp))
    s_left = float(left_grip.get("sigma_mm", LEFT_SIGMA * 1e3)) * 1e-3
    right_off = float(np.linalg.norm(t_w_right[:3, 3] - grasp))
    sigma = math.sqrt(s_left**2 + pushed**2 + right_off**2 + 2 * REPEAT_SIGMA**2)
    if observed_tip is not None:
        seen = np.array([float(observed_tip[k]) for k in "xyz"])
        now_right = _matrix(here(rc))
        seen_tcp = (np.linalg.inv(now_right) @ np.append(seen, 1.0))[:3]
        tip_offset = float(np.dot(seen_tcp, approach_tcp))
        shift = (tip_offset - float(np.dot(tip_tcp, approach_tcp))) * axis_down
        t_tcp_obj[:3, 3] += shift
        sigma = min(sigma, float(observed_sigma_mm) * 1e-3)
        notes.append(
            f"tip offset from the observed tip ({float(np.linalg.norm(shift)) * 1e3:.1f} mm shift)"
        )
    out["held"] = {
        "arm": "right" if rc == 1 else f"arm{rc}",
        "arm_id": rc,
        "T_TCP_object": _se3(t_tcp_obj),
        "tip_offset_m": tip_offset,
        "axis_tilt_deg": _angle_between(axis_down, approach_tcp),
        "sigma_mm": sigma * 1e3,
    }
    return finish(
        "handed_over",
        f"bite {bite * 1e3:.1f} mm, fingertips {out['finger_gap_m'] * 1e3:.1f} mm over the giver's pads,"
        f" giver moved {pushed * 1e3:.1f} mm while both held",
    )


# ------------------------------------------------------------------ helpers
def _joints(ctx: NodeContext, arm_id: int) -> np.ndarray:
    """One arm's joint positions from robot.get_observation, or empty."""
    try:
        arms = ctx.tool("robot.get_observation").get("arms") or []
    except Exception as exc:
        logger.info("handing-over-a-peg: no joint reading (%s)", exc)
        return np.zeros(0)
    if int(arm_id) >= len(arms):
        return np.zeros(0)
    q = arms[int(arm_id)].get("joint_state")
    if isinstance(q, dict):
        q = q.get("positions")
    return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)


def _horizontal_axes(r: np.ndarray, axis_up: np.ndarray) -> list[np.ndarray]:
    """The two tool axes most square to the peg, made square to it."""
    cols = sorted(range(3), key=lambda i: abs(float(np.dot(r[:, i], axis_up))))[:2]
    out = []
    for i in cols:
        v = r[:, i] - float(np.dot(r[:, i], axis_up)) * axis_up
        out.append(v / float(np.linalg.norm(v)))
    return out


def _giver_axes(
    ctx: NodeContext,
    arm_id: int,
    r: np.ndarray,
    axis_up: np.ndarray,
    away_from: np.ndarray,
    notes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """The giver's jaw axis and approach (wrist to fingers) in the world, square to the peg.

    A side grasp has both its approach and its jaw axis square to a vertical
    peg; the connector's own grasp_frame for each signed guess tells them
    apart. Without a match the approach is the other tool axis, signed toward
    ``away_from`` (the receiver's side), and ``notes`` says so.
    """
    a, b = _horizontal_axes(r, axis_up)
    for approach, close in ((a, b), (-a, b), (b, a), (-b, a)):
        try:
            frame = ctx.tool(
                "robot.grasp_frame",
                approach=_vec(approach),
                close_heading_deg=math.degrees(math.atan2(close[1], close[0])),
                **({"arm_id": int(arm_id)} if int(arm_id) else {}),
            )
        except Exception as exc:
            notes.append(f"giver's hand axes assumed (no grasp_frame: {exc})")
            break
        for k in ("rotation", "rotation_flipped"):
            q = frame.get(k) if isinstance(frame, dict) else None
            if q is None:
                continue
            err = _rotation(q).T @ r
            if math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(err)) - 1) / 2)))) < 25.0:
                return close, approach
    else:
        notes.append("giver's hand axes assumed: its hand matches no side-grasp frame")
    approach = a if float(np.dot(a, away_from)) >= 0 else -a
    return b, approach
