"""Turn a held flat piece about the vertical by a required angle within the wrist joint's range, regrasping if needed.

Written for the hub. The regrasp idea - put the part down, turn the empty hand, take it again so the remaining
turn becomes reachable - is WRS's regrasp planning (https://github.com/wanweiwei07/wrs, MIT, 0000_examples/
x6wg2_regrasp) reduced to a flat part on a table and one wrist joint; no WRS code is used. The empty hand turns
by half a turn only, so it closes on the same two contact points with its jaws swapped: the grip chosen for the
slot (e.g. by selecting-a-grip-clear-of-placed-neighbours) is kept.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

from gap import NodeContext

WRIST_LIMIT_RAD = 2.0944
"""The YAM's last joint turns within +-2.0944 rad (+-120 deg; i2rt YAM, and the station MJCF's
joint6 range). Taken from robot.describe_arm when it reports joint limits."""

LIMIT_MARGIN_RAD = 0.05
"""Kept off each end of the range (3 deg): the joint's position loop overshoots a little at speed."""

YAW_TOL_DEG = 0.5
"""A turn is finished when the hand's yaw read back is within this of its goal."""

MAX_CORRECTIONS = 4
"""Extra joint commands to take out a turn's residual (the twin's wrist lags; a real one settles)."""

PROBE_DEG = 4.0
"""First, small step of the first held turn: the sign and gain of joint 6 against the hand's world yaw are
read from it (the joint axis points along the tool's -z on the YAM, but a connector may flip it)."""

LIFT_M = 0.03
"""Height over the grasp height the hand carries the piece at while turning, and leaves the piece from."""

REACH_TOL_M = 0.001
"""Set-down and regrasp points are re-commanded until the tool centre reads back within 1 mm (the station
twin's servo reported "reached" 2.3 mm above the set-down height; the piece then dropped 3 mm and shifted).
A real YAM station refuses moves it cannot finish within 12 mm, so the read-back matters there too."""

SETTLE_STEPS = 40
"""Simulation steps waited after a grip or release where the connector offers robot.wait_steps."""

STEP_DEG = 1.0
"""Grid of the first held turn searched by the regrasp planner."""


class Output(TypedDict, total=False):
    route: str
    plan: dict[str, Any]
    turned_deg: float
    hand_turned_deg: float
    regrasps: int
    joint6_final: float
    jaws_swapped: bool
    detail: str


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def plan_turn(
    q6_deg: float, turn_deg: float, lo_deg: float, hi_deg: float, gain: float = 1.0
) -> dict[str, Any] | None:
    """Joint-6 moves for turning the piece by ``turn_deg`` (counter-clockwise from above, any multiple of
    360 apart is the same turn) from joint position ``q6_deg``, staying within [lo_deg, hi_deg].
    ``gain`` is the hand's world yaw per unit of joint 6 (+1 or -1).

    Returns {"kind": "direct", "held": [dq]} or {"kind": "regrasp", "held": [dq1, dq2], "empty": de} in
    joint degrees, the cheapest by total joint travel, or None. In a regrasp the empty hand turns +-180 deg
    so it takes the same contacts back with its jaws swapped."""
    want = turn_deg / gain

    def ok(q: float) -> bool:
        return lo_deg - 1e-9 <= q <= hi_deg + 1e-9

    best = None
    for k in (-1, 0, 1):
        dq = want + 360.0 * k
        if ok(q6_deg + dq) and (best is None or abs(dq) < abs(best)):
            best = dq
    if best is not None:
        return {"kind": "direct", "held": [best], "travel": abs(best)}
    plans = []
    n = int(round((hi_deg - lo_deg) / STEP_DEG))
    for i in range(n + 1):
        q1 = lo_deg + i * STEP_DEG
        dq1 = q1 - q6_deg
        for de in (180.0, -180.0):
            q2 = q1 + de
            if not ok(q2):
                continue
            for k in (-1, 0, 1):
                dq2 = want - dq1 + 360.0 * k
                if ok(q2 + dq2):
                    plans.append((abs(dq1) + abs(de) + abs(dq2), dq1, de, dq2))
    if not plans:
        return None
    travel, dq1, de, dq2 = min(plans)
    held = [dq for dq in (dq1, dq2)]
    return {"kind": "regrasp", "held": held, "empty": de, "travel": travel}


def _pose(reply: Any) -> dict:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _yaw_deg(rot: dict) -> float:
    """World yaw of the tool's x axis (the hand's heading seen from above)."""
    w, x, y, z = (float(rot[k]) for k in "wxyz")
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    return math.degrees(math.atan2(r10, r00))


def _refused(reply: Any) -> str | None:
    if isinstance(reply, dict) and (reply.get("reached") is False or reply.get("success") is False):
        return str(reply.get("outcome") or reply.get("reason") or "not reached")
    return None


def _limits(ctx: NodeContext, arm_id: int) -> tuple[float, float]:
    try:
        info = ctx.tool("robot.describe_arm", **({"arm_id": arm_id} if arm_id else {}))
    except Exception:
        info = None
    for key in ("joint_limits", "joint_ranges", "limits"):
        lim = (info or {}).get(key) if isinstance(info, dict) else None
        if lim and len(lim) >= 6:
            lo, hi = lim[5][0], lim[5][1]
            return float(lo), float(hi)
    return -WRIST_LIMIT_RAD, WRIST_LIMIT_RAD


class _Arm:
    def __init__(self, ctx: NodeContext, arm_id: int):
        self.ctx, self.arm_id = ctx, arm_id
        self.kw = {"arm_id": arm_id} if arm_id else {}

    def joints(self) -> list[float]:
        obs = self.ctx.tool("robot.get_observation")
        arms = obs.get("arms") or []
        arm = arms[self.arm_id] if len(arms) > self.arm_id else arms[0]
        return [float(v) for v in arm["joint_state"]["positions"]]

    def pose(self) -> dict:
        return _pose(self.ctx.tool("robot.get_ee_pose", **self.kw))

    def to_joints(self, q: list[float]) -> str | None:
        try:
            reply = self.ctx.tool("robot.move_to_joints", joint_config={"positions": q}, **self.kw)
        except Exception as exc:
            return f"move_to_joints raised: {exc}"
        return _refused(reply)

    def to_pose(self, pose: dict, cartesian: bool = False) -> str | None:
        name = "robot.go_to_pose_cartesian" if cartesian else "robot.go_to_pose"
        try:
            reply = self.ctx.tool(name, pose=pose, **self.kw)
        except Exception as exc:
            return f"{name} raised: {exc}"
        return _refused(reply)

    def to_point(self, xy: dict, z: float, rot: dict) -> str | None:
        """Straight to (xy, z) and read back: re-commanded until within REACH_TOL_M (a servo can report
        "reached" short of the goal; set down short, the piece drops and shifts)."""
        why = self.to_pose({"position": {**xy, "z": z}, "rotation": rot}, True)
        for _ in range(MAX_CORRECTIONS):
            if why:
                return why
            here = self.pose()["position"]
            err = math.dist((here["x"], here["y"], here["z"]), (xy["x"], xy["y"], z))
            if err <= REACH_TOL_M:
                return None
            why = self.to_pose({"position": {**xy, "z": z}, "rotation": rot}, True)
        here = self.pose()["position"]
        err = math.dist((here["x"], here["y"], here["z"]), (xy["x"], xy["y"], z))
        return None if err <= 3 * REACH_TOL_M else f"stopped {err * 1e3:.1f} mm from the goal"

    def wait(self) -> None:
        try:
            self.ctx.tool("robot.wait_steps", steps=SETTLE_STEPS)
        except Exception:
            pass


def _turn_joint(
    arm: _Arm, dq_deg: float, sign: float, lo: float, hi: float
) -> tuple[str | None, float]:
    """Turn joint 6 by ``dq_deg`` and correct on the hand's yaw read back (goal: ``sign * dq_deg`` of hand
    yaw). Returns (refusal or None, hand yaw turned in degrees, read back)."""
    yaw0 = _yaw_deg(arm.pose()["rotation"])
    goal = yaw0 + sign * dq_deg
    q = arm.joints()
    q[5] = min(hi, max(lo, q[5] + math.radians(dq_deg)))
    why = arm.to_joints(q)
    for _ in range(MAX_CORRECTIONS):
        if why:
            break
        err = _wrap(goal - _yaw_deg(arm.pose()["rotation"]))
        if abs(err) <= YAW_TOL_DEG:
            break
        q = arm.joints()
        q[5] = min(hi, max(lo, q[5] + math.radians(err / sign)))
        why = arm.to_joints(q)
    return why, _wrap(_yaw_deg(arm.pose()["rotation"]) - yaw0)


def _fail(route: str, detail: str, **extra) -> Output:
    return {"route": route, "detail": detail, **extra}


def run(
    ctx: NodeContext,
    turn_deg: float,
    grasp_z: float | None = None,
    set_down_xy: Any = None,
    grip_width_m: float = 0.0,
    squeeze_m: float = 0.003,
    gain: float = 0.0,
    plan_only: bool = False,
    joint6_deg: float | None = None,
    arm_id: int = 0,
) -> Output:
    """Turn the held piece by ``turn_deg`` (counter-clockwise seen from above; e.g. -angle_deg of
    localizing-a-piece-in-a-reference-image, or matching-a-piece-to-placed-neighbours' rotation).

    ``grasp_z`` is the tool centre's world z when gripping the piece on the table (where it was picked):
    needed only when a regrasp is. ``set_down_xy`` is a free spot to set it down at (default: straight
    below). ``grip_width_m`` > 0 grips with robot.set_grip (object width, ``squeeze_m``), else
    robot.close_gripper. ``gain`` (+1 / -1) is the hand's world yaw per unit of joint 6 if known; else a small
    first held step measures it. ``plan_only`` returns the plan without moving (``joint6_deg`` may stand in
    for the current joint, e.g. to choose at pick time whether to grip with the jaws swapped)."""
    lo, hi = _limits(ctx, arm_id)
    lo_r, hi_r = lo + LIMIT_MARGIN_RAD, hi - LIMIT_MARGIN_RAD
    lo_d, hi_d = math.degrees(lo_r), math.degrees(hi_r)
    arm = _Arm(ctx, arm_id)
    try:
        q6 = math.degrees(arm.joints()[5]) if joint6_deg is None else float(joint6_deg)
    except Exception as exc:
        return _fail("unreachable", f"cannot read the arm's joints: {exc}")
    sign = gain if gain in (1.0, -1.0) else 1.0
    plan = plan_turn(q6, float(turn_deg), lo_d, hi_d, sign)
    if plan is None:
        return _fail("unreachable", f"no turn of {turn_deg:.1f} deg fits joint 6 from {q6:.1f} deg")
    if plan_only:
        return {
            "route": "planned",
            "plan": plan,
            "regrasps": 0 if plan["kind"] == "direct" else 1,
            "jaws_swapped": plan["kind"] == "regrasp",
            "detail": f"{plan['kind']}: held {[round(v, 1) for v in plan['held']]} deg"
            + (f", empty {plan['empty']:+.0f} deg" if "empty" in plan else ""),
        }
    if plan["kind"] == "regrasp" and grasp_z is None:
        return _fail("unreachable", "the turn needs a regrasp and grasp_z is not given", plan=plan)

    here = arm.pose()
    if grasp_z is not None and here["position"]["z"] < grasp_z + LIFT_M - 0.002:
        why = arm.to_pose(
            {"position": {**here["position"], "z": grasp_z + LIFT_M}, "rotation": here["rotation"]},
            True,
        )
        if why:
            return _fail("unreachable", f"lift refused: {why}", plan=plan)

    turned = 0.0
    if gain not in (1.0, -1.0) and abs(turn_deg) > YAW_TOL_DEG:
        # Probe toward the middle of the range: learn which way the hand turns for +joint 6. The piece is
        # held, so the probe is part of the turn.
        step = PROBE_DEG if q6 < 0.5 * (lo_d + hi_d) else -PROBE_DEG
        why, moved = _turn_joint(arm, step, 1.0, lo_r, hi_r)
        if why:
            return _fail("unreachable", f"probe turn refused: {why}", plan=plan)
        if abs(moved) < 0.25 * PROBE_DEG:
            return _fail(
                "unreachable", f"joint 6 moved the hand {moved:+.2f} deg for {step:+.1f}", plan=plan
            )
        sign = 1.0 if moved * step > 0 else -1.0
        turned += moved
        q6 = math.degrees(arm.joints()[5])
        plan = plan_turn(q6, float(turn_deg) - turned, lo_d, hi_d, sign)
        if plan is None:
            return _fail(
                "unreachable", "no plan for the rest of the turn after the probe", turned_deg=turned
            )
        if plan["kind"] == "regrasp" and grasp_z is None:
            return _fail(
                "unreachable", "the turn needs a regrasp and grasp_z is not given", plan=plan
            )

    steps: list[tuple[str, float]] = [("held", plan["held"][0])]
    if plan["kind"] == "regrasp":
        steps += [("regrasp", plan["empty"]), ("held", plan["held"][1])]
    regrasps = 0
    for kind, dq in steps:
        if kind == "held":
            if abs(dq) < 1e-6:
                continue
            why, moved = _turn_joint(arm, dq, sign, lo_r, hi_r)
            turned += moved
            if why:
                return _fail(
                    "unreachable",
                    f"turn refused: {why}",
                    plan=plan,
                    turned_deg=turned,
                    regrasps=regrasps,
                )
            continue
        why = _regrasp(
            ctx, arm, dq, sign, grasp_z, set_down_xy, grip_width_m, squeeze_m, lo_r, hi_r
        )
        regrasps += 1
        if why == "piece_lost":
            return _fail("piece_lost", "the jaws closed past the piece's width after the regrasp", plan=plan,
                         turned_deg=turned, regrasps=regrasps)  # fmt: skip
        if why:
            return _fail("unreachable", why, plan=plan, turned_deg=turned, regrasps=regrasps)
    q6_end = arm.joints()[5]
    return {
        "route": "turned" if regrasps == 0 else "turned_with_regrasp",
        "plan": plan,
        "turned_deg": turned,
        "hand_turned_deg": turned,
        "regrasps": regrasps,
        "joint6_final": q6_end,
        "jaws_swapped": regrasps % 2 == 1,
        "detail": (
            f"turned the piece {turned:+.1f} deg by the hand's yaw (asked {turn_deg:+.1f}, off"
            f" {_wrap(turned - turn_deg):+.2f}) with {regrasps} regrasp(s); joint 6 ends at"
            f" {math.degrees(q6_end):+.1f} deg"
        ),
    }


def _regrasp(
    ctx, arm, empty_deg, sign, grasp_z, set_down_xy, grip_width_m, squeeze_m, lo, hi
) -> str | None:
    """Set the piece down, open, rise, turn the empty hand by ``empty_deg`` of joint 6 (half a turn), come
    down, grip, lift. Returns a refusal, "piece_lost", or None."""
    here = arm.pose()
    xy = {"x": here["position"]["x"], "y": here["position"]["y"]}
    if set_down_xy is not None:
        if isinstance(set_down_xy, dict):
            xy = {"x": float(set_down_xy["x"]), "y": float(set_down_xy["y"])}
        else:
            xy = {"x": float(set_down_xy[0]), "y": float(set_down_xy[1])}
        # Straight across at carry height: a joint-space move's path swung the held piece into the
        # placed neighbours in the station twin and knocked it out of the jaws.
        why = arm.to_pose(
            {"position": {**xy, "z": grasp_z + LIFT_M}, "rotation": here["rotation"]}, True
        )
        if why:
            return f"set-down spot refused: {why}"
    rot = arm.pose()["rotation"]
    why = arm.to_point(xy, grasp_z, rot)
    if why:
        return f"set-down refused: {why}"
    ctx.tool("robot.open_gripper", **arm.kw)
    arm.wait()
    why = arm.to_pose({"position": {**xy, "z": grasp_z + LIFT_M}, "rotation": rot}, True)
    if why:
        return f"rise refused: {why}"
    why, _ = _turn_joint(arm, empty_deg, sign, lo, hi)
    if why:
        return f"empty turn refused: {why}"
    rot = arm.pose()["rotation"]
    why = arm.to_point(xy, grasp_z, rot)
    if why:
        return f"regrasp descent refused: {why}"
    if grip_width_m > 0:
        ctx.tool("robot.set_grip", object_width_m=grip_width_m, squeeze_m=squeeze_m, **arm.kw)
    else:
        ctx.tool("robot.close_gripper", **arm.kw)
    arm.wait()
    why = arm.to_pose({"position": {**xy, "z": grasp_z + LIFT_M}, "rotation": rot}, True)
    if why:
        return f"lift after the regrasp refused: {why}"
    if grip_width_m > 0:
        try:
            g = ctx.tool("robot.get_gripper", **arm.kw)
        except Exception:
            g = None
        w = g.get("width_m", g.get("width")) if isinstance(g, dict) else None
        if w is not None and float(w) < 0.5 * grip_width_m:
            return "piece_lost"
    return None
