# Adapted from RoboTwin (https://github.com/robotwin-Platform/RoboTwin),
# envs/place_dual_shoes.py and envs/pick_dual_bottles.py, MIT, Copyright (c)
# 2025 Tianxing Chen.
# Kept: one object per arm; both grasped top-down from a pre-grasp stand-off
# (pre_grasp_dis 0.10 shoes / 0.08 bottles), both lifted (0.15 shoes / 0.10
# bottles), carried to a pre-place point above the place point (pre_dis 0.07)
# and lowered to a release height (dis 0.02 shoes / 0 bottles), opened; the
# two arms move together while they are apart, and where they would meet one
# places and goes back to where it started while the other waits
# (place_dual_shoes' staggered placing and back_to_origin).
# Changed: the arm for each object is chosen from an IK preview of its whole
# plan, then the nearer side (RoboTwin hard-codes left / right by the object's
# side); simulator actor poses and functional points are replaced by
# world-frame boxes and place points; the two interpolated trajectories become
# interleaved one-arm robot.* moves with an explicit arm_id, each gated on the
# two tool centres staying min_hand_gap apart along the move; the jaw roll is
# read back and finished on the last joint (a solver may leave it free); each
# hold is read from the jaw opening after the close, the lift and the lowering
# (RoboTwin checks only at the end); refused moves are routed per arm.
"""Pick two objects with two arms, one each, and place both, moving the arms together while they are apart."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PRE_GRASP = 0.10
"""Hover over the grasp point before coming down [m] (RoboTwin place_dual_shoes
pre_grasp_dis 0.10; pick_dual_bottles 0.08)."""

LIFT = 0.10
"""Lift straight up after the close [m] (RoboTwin pick_dual_bottles 0.10;
place_dual_shoes 0.15)."""

PRE_PLACE = 0.07
"""Carry to this far above the release point, then lower straight down [m]
(RoboTwin place_dual_shoes pre_dis 0.07)."""

RELEASE_HEIGHT = 0.01
"""The object's bottom over its rest height when the hand opens [m]: between
RoboTwin's dis 0.02 (shoes) and 0 (bottles); 0 would press the object into
its support by the arm's own arrival error (servo 4 mm, real YAM 12 mm)."""

MIN_HAND_GAP = 0.15
"""Least distance between the two tool centres along any move [m]: two open
YAM hands (94 mm span, fingers about 60 mm either side of the tool centre) plus
30 mm. A move that would come closer waits for the other arm. tuned_on: YAM
station twin, the measured mesh clearance is in the SKILL.md."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent [m]: a servo's 4 mm ball plus
a real arm's sag, no tighter than the connectors' own bounds."""

IK_TOLERANCE = 0.02
"""An IK preview landing farther than this from the pose did not solve it [m]
(selecting-reachable-grasp's 20 mm)."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (robot.get_gripper, 1.0 = open) closed on
nothing: 3.8 mm on a 94 mm YAM span, the stop plus verifying-grasps' 3 mm margin."""

FINGER_REACH = 0.010
"""Closed linear_4310 fingertips reach this far below the tool centre [m]."""

FLOOR_MARGIN = 0.004
"""Fingertips kept this far over the object's bottom at the grasp [m]."""

SQUARE_TOLERANCE = 0.005
"""Horizontal half-extents this close [m]: the object is closed across either
side, both headings are previewed."""

ROLL_TOLERANCE_DEG = 3.0
"""How far the jaw roll may end from the commanded one before the rest is
turned on the last joint [deg]: a servo's own 3 deg arrival bound."""

TURN_TRIES = 3
"""Last-joint turns tried, each read back (stacking-objects' value)."""

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
    arm_a: int
    arm_b: int
    outcome_a: str
    outcome_b: str
    failed_arms: str
    min_hand_gap_m: float
    staggered: bool
    release_pose_a: dict[str, Any]
    release_pose_b: dict[str, Any]


@dataclass
class Step:
    kind: str  # "move" | "open" | "close" | "hold"
    phase: int  # 0 pick, 1 place
    label: str
    tool: str = ""
    target: np.ndarray | None = None
    rotation: dict[str, float] | None = None  # None: the hand's current grasp rotation


@dataclass
class Plan:
    heading_deg: float
    rotation: dict[str, float]
    hover: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    pre_place: np.ndarray
    release: np.ndarray


@dataclass
class Hand:
    arm: int
    obj: str
    plan: Plan
    home: dict[str, Any]
    rotation: dict[str, float]
    steps: list[Step] = field(default_factory=list)
    outcome: str = ""
    notes: list[str] = field(default_factory=list)
    waited: bool = False


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


def _segment_gap(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Least distance from point p to the segment a-b."""
    ab = b - a
    n = float(np.dot(ab, ab))
    t = 0.0 if n < 1e-12 else max(0.0, min(1.0, float(np.dot(p - a, ab)) / n))
    return float(np.linalg.norm(a + t * ab - p))


def _box(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, axes (columns) and half-extents of a world-frame box."""
    centre = np.array([float(obb["center"][k]) for k in "xyz"])
    rot = _rotation(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    half = np.array([float(obb["extent"][k]) for k in "xyz"])
    return centre, rot, half


def _headings(obb: dict[str, Any]) -> list[float]:
    """Close headings to try [deg, world]: across the narrower horizontal side,
    and across the other too when the two are about equal."""
    _, rot, half = _box(obb)
    up = int(np.argmax(np.abs(rot[2, :])))
    sides = [i for i in range(3) if i != up]
    sides.sort(key=lambda i: half[i])
    out = []
    for i in sides:
        axis = rot[:, i]
        if math.hypot(axis[0], axis[1]) < 1e-6:
            continue
        out.append(math.degrees(math.atan2(axis[1], axis[0])))
    if len(out) == 2 and abs(half[sides[1]] - half[sides[0]]) > SQUARE_TOLERANCE:
        out = out[:1]
    return out or [0.0]


def _joints(ctx: NodeContext, arm_id: int) -> np.ndarray:
    """One arm's joint positions from robot.get_observation, or empty."""
    try:
        arms = ctx.tool("robot.get_observation").get("arms") or []
    except Exception as exc:
        logger.info("picking-and-placing-two-objects: no joint reading (%s)", exc)
        return np.zeros(0)
    if int(arm_id) >= len(arms):
        return np.zeros(0)
    q = arms[int(arm_id)].get("joint_state")
    if isinstance(q, dict):
        q = q.get("positions")
    return np.asarray(q if q is not None else [], dtype=np.float64).reshape(-1)


def run(
    ctx: NodeContext,
    object_a_obb: dict[str, Any],
    object_b_obb: dict[str, Any],
    place_a: dict[str, float],
    place_b: dict[str, float],
    arm_ids: list[int] | None = None,
    pre_grasp: float = PRE_GRASP,
    lift: float = LIFT,
    pre_place: float = PRE_PLACE,
    release_height: float = RELEASE_HEIGHT,
    min_hand_gap: float = MIN_HAND_GAP,
    reach_tolerance: float = REACH_TOLERANCE,
    empty_fraction: float = EMPTY_FRACTION,
) -> Output:
    arms = [int(a) for a in (arm_ids if arm_ids is not None else (0, 1))]
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "arm_a": -1,
        "arm_b": -1,
        "outcome_a": "",
        "outcome_b": "",
        "failed_arms": "",
        "min_hand_gap_m": -1.0,
        "staggered": False,
        "release_pose_a": {},
        "release_pose_b": {},
    }

    def on(a: int) -> dict[str, int]:
        # arm_id only when nonzero: a single-arm connector takes no such keyword.
        return {"arm_id": int(a)} if int(a) else {}

    def here(a: int) -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on(a)))

    def grip(a: int) -> float:
        """The jaw opening (1.0 = open), or -1 when the hand cannot be read."""
        try:
            return float(ctx.tool("robot.get_gripper", **on(a))["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning(
                "picking-and-placing-two-objects: arm %d gripper unreadable (%s)", a, exc
            )
            return -1.0

    def holds(fraction: float) -> bool:
        return fraction < 0.0 or fraction > float(empty_fraction)

    if len(arms) != 2 or arms[0] == arms[1]:
        out["detail"] = f"two different arms are needed, got arm_ids {arms}"
        out["failed_arms"] = ",".join(str(a) for a in arms)
        return out
    start = {a: here(a) for a in arms}
    q = start[arms[1]].get("rotation") or {}
    if sum(float(q.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_xyz(start[arms[0]]) - _xyz(start[arms[1]]))) < SAME_ARM
    ):
        # A connector that clamps or ignores arm_id reports arm 0 (or zeros) for
        # the second arm; moving "the second arm" would move the first.
        out["detail"] = f"the connector does not report a separate arm {arms[1]}"
        out["failed_arms"] = str(arms[1])
        return out

    objects = {"a": object_a_obb, "b": object_b_obb}
    places = {"a": place_a, "b": place_b}

    def reachable(a: int, position: np.ndarray, rotation: dict[str, float]) -> bool:
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(position), "rotation": rotation}, **on(a)
            )
        except Exception as exc:  # a station raises on an unreachable pose
            logger.info("picking-and-placing-two-objects: IK preview refused (%s)", exc)
            return False
        if not isinstance(reply, dict):
            return reply is not None
        if reply.get("success") is False or reply.get("within_servo_tolerance") is False:
            return False
        err = reply.get("position_error_m")
        return not (isinstance(err, (int, float)) and float(err) > IK_TOLERANCE)

    def plan_for(name: str, a: int) -> tuple[Plan | None, str]:
        centre, rot, half = _box(objects[name])
        reach_down = float(np.abs(rot.T @ DOWN) @ half)
        bottom = centre[2] - reach_down
        grasp = centre.copy()
        grasp[2] = max(centre[2], bottom + FINGER_REACH + FLOOR_MARGIN)
        offset = grasp - centre
        place = np.array([float(places[name][k]) for k in "xyz"])
        release = place + offset + np.array([0.0, 0.0, float(release_height)])
        poses = {
            "hover": grasp + np.array([0.0, 0.0, float(pre_grasp)]),
            "grasp": grasp,
            "lift": grasp + np.array([0.0, 0.0, float(lift)]),
            "pre_place": release + np.array([0.0, 0.0, float(pre_place)]),
            "release": release,
        }
        why = "no grasp rotation"
        for heading in _headings(objects[name]):
            try:
                frame = ctx.tool("robot.grasp_frame", close_heading_deg=heading, **on(a))
            except Exception as exc:
                why = f"robot.grasp_frame: {exc}"
                continue
            for key in ("rotation", "rotation_flipped"):
                rotation = frame.get(key) if isinstance(frame, dict) else None
                if not rotation:
                    continue
                missed = next((k for k, p in poses.items() if not reachable(a, p, rotation)), "")
                if not missed:
                    return Plan(heading_deg=heading, rotation=rotation, **poses), ""
                why = f"its {missed} pose is out of reach"
        return None, why

    plans: dict[tuple[str, int], tuple[Plan | None, str]] = {
        (n, a): plan_for(n, a) for n in ("a", "b") for a in arms
    }

    def planar(a: int, name: str) -> float:
        d = _xyz(start[a]) - _box(objects[name])[0]
        return float(math.hypot(d[0], d[1]))

    pairings = [(arms[0], arms[1]), (arms[1], arms[0])]
    scored = sorted(
        pairings,
        key=lambda pr: (
            sum(plans[(n, a)][0] is None for n, a in zip("ab", pr, strict=True)),
            planar(pr[0], "a") + planar(pr[1], "b"),
        ),
    )
    best = scored[0]
    out["arm_a"], out["arm_b"] = best
    missing = [(n, a) for n, a in zip("ab", best, strict=True) if plans[(n, a)][0] is None]
    if missing:
        out["route"] = "unreachable"
        out["failed_arms"] = ",".join(str(a) for _, a in missing)
        for n, _ in missing:
            out[f"outcome_{n}"] = "unreachable"  # type: ignore[literal-required]
        notes = []
        for n, a in missing:
            b = arms[1] if a == arms[0] else arms[0]
            notes.append(
                f"object {n} with arm {a}: {plans[(n, a)][1]}"
                f" (with arm {b}: {plans[(n, b)][1] or 'reachable'})"
            )
        out["detail"] = "; ".join(notes) + "; nothing moved"
        return out

    hands: list[Hand] = []
    for n, a in zip("ab", best, strict=True):
        plan = plans[(n, a)][0]
        assert plan is not None
        home = {"position": dict(start[a]["position"]), "rotation": dict(start[a]["rotation"])}
        hand = Hand(arm=a, obj=n, plan=plan, home=home, rotation=plan.rotation)
        cart, joint = "robot.go_to_pose_cartesian", "robot.go_to_pose"
        hand.steps = [
            Step("open", 0, "open"),
            Step("move", 0, "hover", joint, plan.hover, plan.rotation),
            Step("move", 0, "descend", cart, plan.grasp),
            Step("close", 0, "close"),
            Step("move", 0, "lift", cart, plan.lift),
            Step("hold", 0, "after the lift"),
            Step("move", 1, "carry", joint, plan.pre_place),
            Step("move", 1, "lower", cart, plan.release),
            Step("hold", 1, "after the lowering"),
            Step("open", 1, "release"),
            Step("move", 1, "raise", cart, plan.pre_place),
            Step("move", 1, "home", joint, _xyz(home), home["rotation"]),
        ]
        hands.append(hand)

    tcp = {h.arm: _xyz(start[h.arm]) for h in hands}
    gaps = [float(np.linalg.norm(tcp[arms[0]] - tcp[arms[1]]))]

    def other(h: Hand) -> Hand:
        return hands[1] if h is hands[0] else hands[0]

    def clear(h: Hand, target: np.ndarray) -> bool:
        there = tcp[other(h).arm]
        along = _segment_gap(tcp[h.arm], target, there)
        now = float(np.linalg.norm(tcp[h.arm] - there))
        # A move that never comes closer than the hands already are (a retreat) is let through.
        return along >= float(min_hand_gap) or along >= now - 1e-3

    def move(h: Hand, tool: str, target: np.ndarray, rotation: dict[str, float]) -> str:
        """Send one arm and judge arrival from a fresh reading of that arm."""
        try:
            why = _refusal(
                ctx.tool(tool, pose={"position": _vec(target), "rotation": rotation}, **on(h.arm))
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        now = here(h.arm)
        tcp[h.arm] = _xyz(now)
        gaps.append(float(np.linalg.norm(tcp[arms[0]] - tcp[arms[1]])))
        off = float(np.linalg.norm(tcp[h.arm] - target))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; stopped {off * 1e3:.0f} mm from its target"
        return ""

    def turn_roll(h: Hand) -> str:
        """Finish the jaw roll about the approach on the last joint (a solver
        that keeps only the approach axis leaves the roll free); re-centre."""
        commanded, gain = h.plan.rotation, 1.0
        now = here(h.arm)
        residual = _roll_residual(commanded, now["rotation"], DOWN)
        for _ in range(TURN_TRIES):
            if abs(residual) <= ROLL_TOLERANCE_DEG:
                break
            q = _joints(ctx, h.arm)
            if q.size == 0:
                h.notes.append("no joint reading, the wrist was not turned")
                break
            command = math.radians(residual) / gain
            lo, hi = ROLL_LIMITS
            if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                command = math.radians(residual - math.copysign(180.0, residual)) / gain
                if not lo + LIMIT_MARGIN <= q[-1] + command <= hi - LIMIT_MARGIN:
                    h.notes.append(f"the last joint has no room to turn {residual:+.0f} deg")
                    break
            target_q = q.copy()
            target_q[-1] += command
            try:
                why = _refusal(
                    ctx.tool(
                        "robot.move_to_joints",
                        joint_config={"positions": target_q.tolist()},
                        **on(h.arm),
                    )
                )
            except Exception as exc:
                why = str(exc) or type(exc).__name__
            if why:
                h.notes.append(f"wrist turn: {why}")
                break
            before = residual
            now = here(h.arm)
            residual = _roll_residual(commanded, now["rotation"], DOWN)
            moved = math.radians(_wrap180(before - residual))
            if abs(command) > math.radians(2.0) and 0.5 < abs(moved / command) < 1.5:
                gain = moved / command
            why = move(h, "robot.go_to_pose", h.plan.hover, now["rotation"])
            if why:
                return f"re-centring after the wrist turn: {why}"
            now = here(h.arm)
            residual = _roll_residual(commanded, now["rotation"], DOWN)
        h.rotation = now["rotation"]
        if abs(residual) > ROLL_TOLERANCE_DEG:
            h.notes.append(f"jaw roll {residual:+.1f} deg off the commanded one")
        return ""

    def fail(h: Hand, outcome: str, why: str, then: list[Step]) -> None:
        h.outcome = outcome
        h.notes.insert(0, why)
        h.steps = then

    def execute(h: Hand, step: Step) -> None:
        cart = "robot.go_to_pose_cartesian"
        if step.kind == "open":
            ctx.tool("robot.open_gripper", **on(h.arm))
            if step.label == "release":
                h.outcome = "placed"
                pose = {"position": _vec(tcp[h.arm]), "rotation": h.rotation}
                out[f"release_pose_{h.obj}"] = pose  # type: ignore[literal-required]
            return
        if step.kind == "close":
            ctx.tool("robot.close_gripper", **on(h.arm))
            g = grip(h.arm)
            if not holds(g):
                fail(
                    h,
                    "grasp_missed",
                    f"the jaws closed to the stop ({g:.3f})",
                    [Step("open", 0, "open"), Step("move", 0, "back up", cart, h.plan.hover)]
                    + [
                        Step(
                            "move", 0, "home", "robot.go_to_pose", _xyz(h.home), h.home["rotation"]
                        )
                    ],
                )
            return
        if step.kind == "hold":
            g = grip(h.arm)
            if not holds(g):
                up = h.plan.pre_place if step.phase else h.plan.lift
                fail(
                    h,
                    "dropped",
                    f"the jaws ran to the stop ({g:.3f}) {step.label}",
                    [Step("open", step.phase, "open"), Step("move", step.phase, "up", cart, up)]
                    + [
                        Step(
                            "move",
                            step.phase,
                            "home",
                            "robot.go_to_pose",
                            _xyz(h.home),
                            h.home["rotation"],
                        )
                    ],
                )
            return
        assert step.target is not None
        rotation = step.rotation or h.rotation
        why = move(h, step.tool, step.target, rotation)
        if not why and step.label == "hover":
            why = turn_roll(h)
        if not why:
            return
        why = f"{step.label}: {why}"
        if h.outcome:
            # Already released, or already backing off a failure: note it and go on.
            h.notes.append(why)
            return
        home = Step(
            "move", step.phase, "home", "robot.go_to_pose", _xyz(h.home), h.home["rotation"]
        )
        if step.label == "hover":
            fail(h, "move_refused", why + "; nothing held", [home])
        elif step.label == "descend":
            back = Step("move", 0, "back up", cart, h.plan.hover)
            fail(h, "move_refused", why + "; nothing held", [back, home])
        elif step.label == "lift":
            back = Step("move", 0, "back up", cart, h.plan.hover)
            fail(
                h,
                "move_refused",
                why + "; opened where it was",
                [Step("open", 0, "open"), back, home],
            )
        elif step.label in ("carry", "lower"):
            back = Step("move", 1, "back to the lift", "robot.go_to_pose", h.plan.lift)
            fail(h, "move_refused", why + "; still holds the object", [back])

    barrier_off = False
    while any(h.steps for h in hands):
        progressed = barrier_held = False
        for h in hands:
            if not h.steps:
                continue
            step, o = h.steps[0], other(h)
            if not barrier_off and o.steps and o.steps[0].phase < step.phase:
                barrier_held = True
                continue
            if step.kind == "move" and not clear(h, step.target):
                h.waited = True
                continue
            h.steps.pop(0)
            execute(h, step)
            progressed = True
        if progressed:
            barrier_off = False
            continue
        if barrier_held and not barrier_off:
            # The other arm is waiting for clearance: let this one go on to its next phase.
            barrier_off = True
            continue
        for h in hands:
            if h.steps:
                fail(
                    h,
                    h.outcome or "clearance",
                    f"{h.steps[0].label}: would come within {float(min_hand_gap) * 1e3:.0f} mm"
                    f" of arm {other(h).arm}'s tool centre",
                    [],
                )
        break

    placed = [h for h in hands if h.outcome == "placed"]
    out["route"] = ("none_placed", "one_placed", "both_placed")[len(placed)]
    for h in hands:
        out[f"outcome_{h.obj}"] = h.outcome  # type: ignore[literal-required]
    out["failed_arms"] = ",".join(str(h.arm) for h in hands if h.outcome != "placed")
    out["min_hand_gap_m"] = float(min(gaps))
    out["staggered"] = any(h.waited for h in hands)
    out["detail"] = "; ".join(
        f"arm {h.arm} (object {h.obj}): {h.outcome}"
        + (f" - {', '.join(h.notes)}" if h.notes else "")
        for h in hands
    )
    return out
