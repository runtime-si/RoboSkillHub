# Adapted from ACT (https://github.com/tonyzhaozh/act), scripted_policy.py,
# InsertionPolicy.generate_trajectory, MIT, Copyright (c) 2023 Tony Z. Zhao.
# Kept: each arm holds one part; both are brought to a shared meet point, the
# parts facing each other on one axis, each a stand-off back from its final pose
# (ACT: meet_xyz +-0.10 m, then +-0.05 m: 0.05 m of closing per arm), the hands
# keep the rotations they picked with (ACT's fixed quaternions), and the gap is
# closed along that axis while the other hand holds its part.
# Changed: simulator part poses (env_state) and ACT's hand-tuned lift_right
# (0.00715 m between the two parts' axes) replaced by in-hand offsets measured
# while held (registering-held-objects, or a perceived point with the hands
# unmoved since) and the two hands' read-back poses: the moving part's tip is
# aligned on the holder's actual entry, across the axis, before anything
# touches; ACT's open-loop interpolation replaced by short read-back steps that
# stop on contact, felt without a force sensor as the pushing hand's tracking lag
# beyond its free-motion lag, the holding hand being pushed off its pose, or a
# joint-effort residual where the connector publishes efforts; one arm moves per
# call with an explicit arm_id; refused moves and every outcome are routed.
"""Insert the part one arm holds into (or onto) the part the other arm holds, in mid-air."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

STANDOFF = 0.05
"""Gap between the moving part's tip and the entry, along the axis, at the
rendezvous [m]: ACT closes 0.05 m per arm (meet_xyz +-0.10 -> +-0.05)."""

CONTACT_MARGIN = 0.005
"""The moving tip is brought this far short of the entry in one straight move,
re-aligned, and stepped from there [m]: far enough that the servo's arrival ball
(4 mm in a simulator servo) cannot touch the rim."""

STEP = 0.0025
"""Length of each insertion step [m]: detecting-contact-from-joint-effort's
step (the real YAM threading station's SETTLE_STEP)."""

LAG_THRESHOLD = 0.003
"""How much further behind its command than in free motion the pushing hand
must stand to be held [m]: the station's SETTLE_CONTACT, as in
detecting-contact-from-joint-effort."""

HOLDER_SHIFT_THRESHOLD = 0.002
"""How far the holding hand may be pushed off the pose it held when the steps
began before that is contact [m]. A choice: the holder is not commanded during
the steps, so this is a displacement, not an arrival bound; in the YAM twin it
stood within 0.3 mm while the other arm stepped in free air (simulation check)."""

EFFORT_THRESHOLD = 0.5
"""Least joint-effort residual read as contact [N.m]: i2rt's
clog_force_threshold, as in detecting-contact-from-joint-effort; raised per joint
to EFFORT_SIGMAS of its free-motion windows."""

EFFORT_SIGMAS = 4.0
"""Per-joint effort threshold floor in standard deviations of the free windows."""

BASELINE_WINDOW = 3
"""Free-motion samples (lags, effort windows) the baseline is the median / mean of."""

CLEARANCE = 0.001
"""Diametral clearance between the parts [m] (peg 1 mm under the hole): the
lateral alignment must be within half of it before the push."""

MIN_ALIGN_TOLERANCE = 0.0002
"""Floor on the lateral alignment bound [m]: a joint servo's read-back noise."""

ALIGN_TRIES = 3
"""Lateral correction moves tried before the tip is called misaligned."""

FINE_TOLERANCE = 0.0003
"""Arrival ball asked of an alignment move and of each step [m], passed as the
connector's ``tolerance`` (a simulator servo's default ball is 4 mm, over the
whole clearance). 0 leaves the connector's default."""

ENTRY_MARGIN = 0.002
"""A stop with the tip less than this past the entry is the rim, not a jam [m]."""

DEPTH_TOLERANCE = 0.003
"""The tip within this of insert_depth counts as in [m]: the lag threshold, so a
hand short by less than a lag the test cannot see is in."""

BACKOFF = 0.003
"""How far the pushing hand goes back along the axis after contact [m], so it stops pushing."""

RETREAT = 0.05
"""How far the mover's open hand backs out after a release [m]."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from a free move's target [m]: no tighter than a
servo's 4 mm or a real YAM station's 12 mm (the alignment that follows is read
back and corrected)."""

ANGLE_TOLERANCE_DEG = 2.0
"""How far either hand's rotation may end from the one it held the part in when
the offsets were measured [deg]: 2 deg turns a 1 mm-clearance part across its
own 20 mm depth by 0.7 mm."""

ROLL_TOLERANCE_DEG = 1.0
"""A hand's roll about its approach axis is turned back on its last joint past
this [deg] (a free wrist roll swings a part held off the tool axis; 1 deg moves a
tip 50 mm out by 0.9 mm). handing-over-between-arms uses 3 deg for a grasp."""

TURN_TRIES = 3
"""Last-joint read-back turns per hand (handing-over-between-arms' value)."""

ROLL_LIMITS = (-2.0944, 2.0944)
"""Last joint's range [rad]: the i2rt YAM's joint 6 (+-120 deg, its MJCF)."""

LIMIT_MARGIN = 0.05
"""Kept clear of each roll limit [rad]."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (robot.get_gripper, 1.0 = open) holds nothing:
3.8 mm on the YAM's 94 mm span, the stop plus verifying-grasps' 3 mm margin."""

OPEN_FRACTION = 0.5
"""A mover reading under this after open_gripper did not open."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""

MIN_AXIS_GAP = 0.005
"""Tip and entry closer than this give no direction to infer the axis from [m]."""

SETTLE_QUIET = 0.0003
SETTLE_PERIOD = 0.3
SETTLE_TIMEOUT = 3.0
"""A hand has settled when two reads SETTLE_PERIOD [s] apart differ by under
SETTLE_QUIET [m], or after SETTLE_TIMEOUT [s] (detecting-contact-from-joint-effort's
values, the real station's still()). 0 period reads once."""

REFUSALS = ("unreachable", "bad_speed_scale")
"""Routed outcomes that mean a move was refused, not stopped short."""


class Output(TypedDict):
    route: str
    detail: str
    depth: float
    lateral_error: float
    lag_over_baseline: float
    holder_shift: float
    contact_signal: str
    steps: int
    roll_turns: int
    efforts_available: bool
    entry_position: dict[str, float]
    tip_position: dict[str, float]
    mover_pose: dict[str, Any]
    holder_pose: dict[str, Any]


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _arr(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        if "position" in v:
            v = v["position"]
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(-1)[:3]


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _unit(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    a = _arr(v)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


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


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    err = a @ b.T
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(err)) - 1.0) / 2.0))))


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _roll_residual(commanded: np.ndarray, now: np.ndarray, axis: np.ndarray) -> float:
    """How far the hand must still turn about ``axis`` to reach ``commanded`` [deg]
    (no half-turn folding: a held part is not symmetric under it)."""
    err = commanded @ now.T
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


def _refusal(reply: Any) -> tuple[str, str]:
    """("ok" | "short" | "refused", why) from a move's reply."""
    if not isinstance(reply, dict):
        return "ok", ""
    outcome = reply.get("outcome")
    if outcome in REFUSALS:
        return "refused", f"{outcome}: {reply.get('reason', '')}".rstrip(": ")
    if outcome not in (None, "reached") or reply.get("reached") is False or reply.get("success") is False:
        return "short", str(outcome or reply.get("reason") or reply.get("message") or "not reached")
    return "ok", ""


def _joint_state(obs: Any, arm_id: int) -> dict[str, Any] | None:
    if not isinstance(obs, dict):
        return None
    arms = obs.get("arms")
    if isinstance(arms, list) and len(arms) > arm_id and isinstance(arms[arm_id], dict):
        js = arms[arm_id].get("joint_state")
        return js if isinstance(js, dict) else None
    return None


def _field(js: dict[str, Any] | None, *names: str) -> np.ndarray | None:
    for name in names:
        value = (js or {}).get(name)
        if value is not None:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size:
                return arr
    return None


def run(
    ctx: NodeContext,
    moving_tip: dict[str, float] | None = None,
    receiving_entry: dict[str, float] | None = None,
    moving_tip_in_tcp: dict[str, Any] | None = None,
    receiving_entry_in_tcp: dict[str, Any] | None = None,
    insertion_axis: dict[str, float] | None = None,
    insert_depth: float = 0.02,
    meet_position: dict[str, float] | None = None,
    mover_arm: int = 1,
    holder_arm: int = 0,
    mode: str = "holder_still",
    clearance: float = CLEARANCE,
    standoff: float = STANDOFF,
    step: float = STEP,
    signal: str = "auto",
    lag_threshold: float = LAG_THRESHOLD,
    holder_shift_threshold: float = HOLDER_SHIFT_THRESHOLD,
    effort_threshold: float = EFFORT_THRESHOLD,
    depth_tolerance: float = DEPTH_TOLERANCE,
    fine_tolerance: float = FINE_TOLERANCE,
    step_max_steps: int = 0,
    reach_tolerance: float = REACH_TOLERANCE,
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    fix_roll: bool = True,
    release: str = "none",
    retreat: float = RETREAT,
    empty_fraction: float = EMPTY_FRACTION,
    settle_period: float = SETTLE_PERIOD,
) -> Output:
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "depth": 0.0,
        "lateral_error": 0.0,
        "lag_over_baseline": 0.0,
        "holder_shift": 0.0,
        "contact_signal": "",
        "steps": 0,
        "roll_turns": 0,
        "efforts_available": False,
        "entry_position": {},
        "tip_position": {},
        "mover_pose": {},
        "holder_pose": {},
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"] = route
        out["detail"] = "; ".join(d for d in (detail, *notes) if d)
        try:
            out["mover_pose"] = read(mover_arm)
            out["holder_pose"] = read(holder_arm)
        except Exception as exc:  # the route stands even if a last read fails
            logger.info("inserting-mid-air: final read failed (%s)", exc)
        return out

    if mode not in ("holder_still", "both_move"):
        raise ValueError(f"mode must be holder_still or both_move, got {mode!r}")
    if release not in ("none", "mover"):
        raise ValueError(f"release must be none or mover, got {release!r}")
    if signal not in ("auto", "lag", "effort"):
        raise ValueError(f"signal must be auto, lag or effort, got {signal!r}")

    def arm(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": int(a)} if int(a) else {}

    knobs: dict[str, Any] = {}
    if float(fine_tolerance) > 0:
        knobs["tolerance"] = float(fine_tolerance)
    if int(step_max_steps) > 0:
        knobs["max_steps"] = int(step_max_steps)

    def read(a: int) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **arm(a)))

    def at(a: int) -> tuple[np.ndarray, np.ndarray]:
        p = read(a)
        return _arr(p["position"]), _rotation(p["rotation"])

    def settled(a: int) -> tuple[np.ndarray, np.ndarray]:
        """Where a hand stands once it stops moving (the station's still())."""
        p, r = at(a)
        if float(settle_period) <= 0:
            return p, r
        t0 = time.monotonic()
        while time.monotonic() - t0 < SETTLE_TIMEOUT:
            time.sleep(float(settle_period))
            q, s = at(a)
            if float(np.linalg.norm(q - p)) < SETTLE_QUIET:
                return q, s
            p, r = q, s
        return p, r

    def send(tool: str, a: int, position: np.ndarray, rotation: dict[str, float], fine: bool) -> tuple[str, str]:
        kw = dict(knobs) if fine else {}
        pose = {"position": _vec(position), "rotation": rotation}
        try:
            try:
                reply = ctx.tool(tool, pose=pose, **kw, **arm(a))
            except TypeError as exc:
                if not kw or "keyword" not in str(exc):
                    raise
                knobs.clear()  # this connector takes no servo knobs
                reply = ctx.tool(tool, pose=pose, **arm(a))
        except Exception as exc:  # a station raises where a simulator routes
            why = str(exc) or type(exc).__name__
            return ("short" if "did not reach" in why else "refused"), why
        return _refusal(reply)

    def move(tool: str, a: int, position: np.ndarray, rotation: dict[str, float]) -> str:
        """A free move judged from a fresh reading of that arm ("" when it arrived)."""
        kind, why = send(tool, a, position, rotation, fine=False)
        off = float(np.linalg.norm(at(a)[0] - position))
        if off > float(reach_tolerance) or kind == "refused":
            return f"{why or 'ended off'}; arm {a} stopped {off * 1e3:.0f} mm from its target"
        return ""

    def grip(a: int) -> float:
        try:
            return float(ctx.tool("robot.get_gripper", **arm(a))["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning("inserting-mid-air: arm %d gripper unreadable (%s)", a, exc)
            return -1.0

    def holds(a: int) -> bool:
        g = grip(a)
        return g < 0.0 or g > float(empty_fraction)

    def joints(a: int) -> np.ndarray:
        try:
            js = _joint_state(ctx.tool("robot.get_observation"), a)
        except Exception as exc:
            logger.info("inserting-mid-air: no joint reading (%s)", exc)
            return np.zeros(0)
        q = _field(js, "positions")
        return q if q is not None else np.zeros(0)

    def fix(a: int, commanded: np.ndarray, rotation: dict[str, float], position: np.ndarray) -> str:
        """Turn a hand's free wrist roll back on its last joint and re-centre it
        (stacking-objects / handing-over-between-arms' procedure). "" when done."""
        if not fix_roll:
            return ""
        gain = 1.0
        for _ in range(TURN_TRIES):
            _, now = at(a)
            residual = _roll_residual(commanded, now, now[:, 2])
            if abs(residual) <= ROLL_TOLERANCE_DEG:
                return ""
            q = joints(a)
            if q.size == 0:
                notes.append(f"no joint reading: arm {a}'s wrist roll is {residual:+.1f} deg off")
                return ""
            command = math.radians(residual) / gain
            lo, hi = ROLL_LIMITS
            if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                return f"arm {a}'s last joint has no room to turn {residual:+.1f} deg back"
            target = q.copy()
            target[-1] += command
            try:
                kind, why = _refusal(
                    ctx.tool("robot.move_to_joints", joint_config={"positions": target.tolist()}, **arm(a))
                )
            except Exception as exc:
                kind, why = "refused", str(exc) or type(exc).__name__
            if kind == "refused":
                return f"arm {a}'s wrist turn: {why}"
            out["roll_turns"] += 1
            _, after = at(a)
            moved = math.radians(_wrap180(residual - _roll_residual(commanded, after, after[:, 2])))
            if abs(command) > math.radians(1.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = move("robot.go_to_pose", a, position, rotation)
            if why:
                return f"re-centring arm {a} after its wrist turn: {why}"
        _, now = at(a)
        residual = _roll_residual(commanded, now, now[:, 2])
        return "" if abs(residual) <= ROLL_TOLERANCE_DEG else f"arm {a}'s wrist roll stayed {residual:+.1f} deg off"

    # --- two separate arms, both holding --------------------------------------
    if int(mover_arm) == int(holder_arm):
        return finish("unreachable", f"mover and holder are both arm {int(mover_arm)}")
    m0, h0 = read(mover_arm), read(holder_arm)
    q = h0.get("rotation") or {}
    if sum(float(q.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_arr(m0["position"]) - _arr(h0["position"]))) < SAME_ARM
    ):
        # A connector that clamps or ignores arm_id reports one arm twice.
        return finish("unreachable", f"the connector does not report a separate arm {int(holder_arm)}")
    for a, who in ((mover_arm, "mover"), (holder_arm, "holder")):
        if not holds(a):
            return finish("slipped", f"the {who}'s jaws are at the stop: it holds nothing; nothing moved")

    # --- in-hand offsets, in each hand's tool frame -----------------------------
    pm0, rm0 = _arr(m0["position"]), _rotation(m0["rotation"])
    ph0, rh0 = _arr(h0["position"]), _rotation(h0["rotation"])
    if moving_tip_in_tcp is not None:
        tip_in = _arr(moving_tip_in_tcp)
    elif moving_tip is not None:
        tip_in = rm0.T @ (_arr(moving_tip) - pm0)
    else:
        raise ValueError("give moving_tip (world) or moving_tip_in_tcp")
    if receiving_entry_in_tcp is not None:
        entry_in = _arr(receiving_entry_in_tcp)
    elif receiving_entry is not None:
        entry_in = rh0.T @ (_arr(receiving_entry) - ph0)
    else:
        raise ValueError("give receiving_entry (world) or receiving_entry_in_tcp")

    def tip_of(p: np.ndarray, r: np.ndarray) -> np.ndarray:
        return p + r @ tip_in

    def entry_of(p: np.ndarray, r: np.ndarray) -> np.ndarray:
        return p + r @ entry_in

    tip0, entry0 = tip_of(pm0, rm0), entry_of(ph0, rh0)
    axis = _unit(insertion_axis)
    if axis is None:
        gap = entry0 - tip0
        if float(np.linalg.norm(gap)) < MIN_AXIS_GAP:
            return finish("unreachable", "the tip is at the entry and no insertion_axis was given")
        axis = gap / float(np.linalg.norm(gap))
    out["entry_position"], out["tip_position"] = _vec(entry0), _vec(tip0)

    mover_rot, holder_rot = m0["rotation"], h0["rotation"]

    # --- rendezvous ----------------------------------------------------------------
    meet = _arr(meet_position) if meet_position is not None else entry0
    half = float(standoff) / 2.0 if mode == "both_move" else 0.0
    holder_pre = meet + axis * half - rh0 @ entry_in
    if mode == "both_move" or meet_position is not None:
        why = move("robot.go_to_pose", holder_arm, holder_pre, holder_rot)
        if why:
            return finish("unreachable", f"holder to the meet point: {why}; both still hold their parts")
        why = fix(holder_arm, rh0, holder_rot, holder_pre)
        if why:
            return finish("unreachable", why)
    tip_pre = meet - axis * (float(standoff) - half)
    mover_pre = tip_pre - rm0 @ tip_in
    why = move("robot.go_to_pose", mover_arm, mover_pre, mover_rot)
    if why:
        return finish("unreachable", f"mover to its stand-off: {why}; both still hold their parts")
    why = fix(mover_arm, rm0, mover_rot, mover_pre)
    if why:
        return finish("unreachable", why)
    if half:
        # ACT closes from both sides: the holder brings its part in to the meet
        # point, then holds still as the anvil the mover pushes against.
        holder_meet = meet - rh0 @ entry_in
        why = move("robot.go_to_pose_cartesian", holder_arm, holder_meet, holder_rot)
        if why:
            return finish("unreachable", f"holder in to the meet point: {why}")

    # --- align on the holder's actual entry, across the axis -----------------------
    tol = max(float(clearance) / 2.0, MIN_ALIGN_TOLERANCE)

    def relative() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        pm, rm = settled(mover_arm)
        ph, rh = at(holder_arm)
        tip, entry = tip_of(pm, rm), entry_of(ph, rh)
        d = entry - tip
        lateral = d - axis * float(np.dot(d, axis))
        return pm, rm, ph, rh, float(np.dot(tip - entry, axis)), lateral

    def tilted(rm: np.ndarray, rh: np.ndarray) -> str:
        for a, r, r0 in ((mover_arm, rm, rm0), (holder_arm, rh, rh0)):
            off = _angle_deg(r, r0)
            if off > float(angle_tolerance_deg):
                return f"arm {a} is turned {off:.1f} deg from the rotation it held its part in"
        return ""

    def align(target_depth: float) -> tuple[str, np.ndarray, np.ndarray]:
        """Bring the tip to target_depth along the axis on the entry's line; "" when within tol."""
        pm, rm, ph, rh, depth, lateral = relative()
        for _ in range(ALIGN_TRIES + 1):
            out["lateral_error"] = float(np.linalg.norm(lateral))
            out["depth"] = depth
            err = lateral + axis * (target_depth - depth)
            if float(np.linalg.norm(lateral)) <= tol and abs(target_depth - depth) <= max(tol, 0.001):
                return "", ph, rh
            send("robot.go_to_pose_cartesian", mover_arm, pm + err, mover_rot, fine=True)
            pm, rm, ph, rh, depth, lateral = relative()
        out["lateral_error"] = float(np.linalg.norm(lateral))
        out["depth"] = depth
        if float(np.linalg.norm(lateral)) <= tol:
            return "", ph, rh
        return (
            f"the tip stays {out['lateral_error'] * 1e3:.2f} mm off the entry's line "
            f"(allowed {tol * 1e3:.2f} mm) after {ALIGN_TRIES} corrections",
            ph,
            rh,
        )

    pm, rm, ph, rh, depth, lateral = relative()
    why = tilted(rm, rh)
    if why:
        return finish("misaligned", why + "; nothing touched")
    why, _, _ = align(-float(standoff))
    if why:
        return finish("misaligned", why + f"; the mover waits {float(standoff) * 1e3:.0f} mm back")
    # One straight move to just short of the entry, then re-align there.
    pm, rm, ph, rh, depth, lateral = relative()
    approach = pm + lateral + axis * (-CONTACT_MARGIN - depth)
    kind, why = send("robot.go_to_pose_cartesian", mover_arm, approach, mover_rot, fine=False)
    if kind == "refused":
        return finish("unreachable", f"mover's approach to the entry: {why}")
    why, ph_hold, rh_hold = align(-CONTACT_MARGIN)
    if why:
        send("robot.go_to_pose_cartesian", mover_arm, mover_pre, mover_rot, fine=False)
        return finish("misaligned", why + "; the mover backed off to its stand-off")
    pm, rm, ph, rh, depth, lateral = relative()
    why = tilted(rm, rh)
    if why:
        send("robot.go_to_pose_cartesian", mover_arm, mover_pre, mover_rot, fine=False)
        return finish("misaligned", why + "; the mover backed off to its stand-off")

    # --- step in, feeling for contact on both arms ----------------------------------
    efforts_on = False
    if signal in ("auto", "effort"):
        try:
            obs = ctx.tool("robot.get_observation")
            efforts_on = all(
                _field(_joint_state(obs, a), "efforts", "effort", "torques") is not None
                for a in (mover_arm, holder_arm)
            )
        except Exception as exc:
            logger.info("inserting-mid-air: no observation (%s)", exc)
    out["efforts_available"] = efforts_on
    if signal == "effort" and not efforts_on:
        return finish("unreachable", "signal=effort, but the observation carries no joint efforts; nothing pushed")

    def efforts() -> np.ndarray | None:
        obs = ctx.tool("robot.get_observation")
        parts = [_field(_joint_state(obs, a), "efforts", "effort", "torques") for a in (mover_arm, holder_arm)]
        return None if any(p is None for p in parts) else np.concatenate(parts)

    lateral_fix = lateral
    start_depth = depth
    pm_start = pm.copy()
    # Steps are commanded from where the hand stands, so standing there is a lag of 0.
    lags: list[float] = [0.0]
    windows: list[np.ndarray] = []
    if efforts_on:
        w = efforts()
        if w is not None:
            windows.append(w)
    target = float(insert_depth)
    n_steps = max(1, int(math.ceil((target - start_depth) / float(step))))
    contact = ""
    commanded_depth = start_depth
    for k in range(1, n_steps + 1):
        commanded_depth = min(target, start_depth + k * float(step))
        tcp_cmd = pm_start + lateral_fix + axis * (commanded_depth - start_depth)
        kind, why = send("robot.go_to_pose", mover_arm, tcp_cmd, mover_rot, fine=True)
        out["steps"] = k
        if kind == "refused":
            return finish("unreachable", f"insertion step {k}: {why}; the mover still holds its part in the entry")
        pm, rm, ph, rh, depth, lateral = relative()
        lag = float(np.dot(tcp_cmd - pm, axis))
        base = float(np.median(lags[-BASELINE_WINDOW:]))
        out["lag_over_baseline"] = lag - base
        out["holder_shift"] = float(np.linalg.norm(ph - ph_hold))
        out["depth"] = depth
        out["lateral_error"] = float(np.linalg.norm(lateral))
        fired = []
        if signal != "effort" and lag - base > float(lag_threshold):
            fired.append("lag")
        if out["holder_shift"] > float(holder_shift_threshold):
            fired.append("holder")
        if efforts_on:
            w = efforts()
            if w is not None and len(windows) >= 1:
                recent = np.asarray(windows[-BASELINE_WINDOW:])
                mean = recent.mean(axis=0)
                sigma = recent.std(axis=0) if len(windows) >= 3 else np.zeros_like(mean)
                limit = np.maximum(float(effort_threshold), EFFORT_SIGMAS * sigma)
                if bool(np.any(np.abs(w - mean) > limit)):
                    fired.append("effort")
                elif not fired:
                    windows.append(w)
        if fired:
            contact = "+".join(fired)
            break
        lags.append(lag)
    out["contact_signal"] = contact

    if depth >= target - float(depth_tolerance):
        if contact:
            notes.append(f"contact ({contact}) at the bottom, {depth * 1e3:.1f} mm in")
            send("robot.go_to_pose", mover_arm, pm - axis * BACKOFF, mover_rot, fine=True)
        if not holds(holder_arm):
            return finish("slipped", "the holder's jaws ran to the stop during the insertion")
        if release == "none":
            if not holds(mover_arm):
                return finish("slipped", "the mover's jaws ran to the stop during the insertion")
            return finish("inserted", f"tip {depth * 1e3:.1f} mm past the entry, both hands still hold")
        opened = ctx.tool("robot.open_gripper", **arm(mover_arm))
        g = float(opened.get("position", -1.0)) if isinstance(opened, dict) else -1.0
        if g < 0.0:
            g = grip(mover_arm)
        if 0.0 <= g < OPEN_FRACTION:
            return finish("stalled", f"inserted {depth * 1e3:.1f} mm but the mover's jaws read {g:.2f} after opening")
        pm, rm = at(mover_arm)
        back = pm - rm[:, 2] * float(retreat)  # back along the hand's own approach
        why = move("robot.go_to_pose_cartesian", mover_arm, back, mover_rot)
        if not holds(holder_arm):
            return finish("slipped", "the holder's jaws ran to the stop after the mover let go")
        return finish(
            "inserted",
            f"tip {depth * 1e3:.1f} mm past the entry; the mover let go"
            + (f"; its back-out: {why}" if why else "")
            + "; the holder holds the assembly",
        )

    # Short of the depth: back off so the hand stops pushing, keep holding.
    pm_now, _ = at(mover_arm)
    if depth < ENTRY_MARGIN:
        send("robot.go_to_pose_cartesian", mover_arm, mover_pre, mover_rot, fine=False)
        return finish(
            "misaligned",
            f"stopped on the rim ({contact or 'no advance'}) with the tip {depth * 1e3:.1f} mm "
            f"past the entry, {out['lateral_error'] * 1e3:.2f} mm off its line; the mover backed off "
            "to its stand-off, both still hold",
        )
    send("robot.go_to_pose", mover_arm, pm_now - axis * BACKOFF, mover_rot, fine=True)
    if not holds(holder_arm) or not holds(mover_arm):
        return finish("slipped", f"a hand's jaws ran to the stop with the tip {depth * 1e3:.1f} mm in")
    return finish(
        "stalled",
        f"jammed ({contact or 'no advance'}) {depth * 1e3:.1f} mm in of {target * 1e3:.1f} mm; "
        f"the mover backed off {BACKOFF * 1e3:.0f} mm and both still hold",
    )
