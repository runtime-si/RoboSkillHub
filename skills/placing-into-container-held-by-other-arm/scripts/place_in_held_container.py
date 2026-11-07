# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/place_held_in_held_container/robotwin/policy.py, Apache-2.0.
# Kept: the container arm is first brought to a pose the carrying arm reaches and
# then never opened; the carrying arm goes over the container with its own
# rotation first and then RoboRSI's tilted alternatives (30 deg about +-x and
# +-y, 45 deg about +-x), takes the first one that previews as reachable, comes
# down to a drop height over the container, opens, and the container hand's
# jaws are checked unchanged afterwards.
# Changed: simulator EE poses and gripper values replaced by world-frame boxes of
# the two held things and the hands' own poses and jaw openings (robot.* with an
# explicit arm_id); the container keeps its rotation on the way to the
# presentation pose (RoboRSI sent the EE to an xy with no rotation), so it stays
# upright, and its tilt is read back from the holder's rotation; the drop point
# is over the container's opening (its box and the holder's pose), shifted away
# from the holder's hand as far as the object still fits, instead of the
# container EE's xy; the object rides at its measured offset from the carrying
# hand under every tilt, and tilts that lean the carrying wrist away from the
# holder are tried first; cuRobo's swept-sphere check between the arms is
# replaced by an IK preview (robot.solve_ik) and a keep-out around the holder's
# tool centre; the carrying arm approaches above and comes straight down
# (RoboRSI moved straight to the drop pose); after the release it backs up and
# closes once to read that the object left its jaws; refused moves are routed.
"""Put the object one arm holds into a container the other arm holds."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

TILTS = (
    ("own", None, 0.0),
    ("tilt_30_+x", (1.0, 0.0, 0.0), 30.0),
    ("tilt_30_-x", (1.0, 0.0, 0.0), -30.0),
    ("tilt_30_+y", (0.0, 1.0, 0.0), 30.0),
    ("tilt_30_-y", (0.0, 1.0, 0.0), -30.0),
    ("tilt_45_+x", (1.0, 0.0, 0.0), 45.0),
    ("tilt_45_-x", (1.0, 0.0, 0.0), -45.0),
)
"""Carrying-hand rotations, each a turn about a world axis applied to the hand's
own rotation: RoboRSI's _QUAT_CANDIDATES (top-down, then 30 deg about +-x and
+-y, then 45 deg about +-x), with the hand's current rotation in place of
RoboRSI's fixed top-down one so the object keeps its pose in the hand. The order
tried is set by LEAN_AWAY."""

LEAN_AWAY = 0.25
"""Tilts that lean the carrier's wrist away from the holder by more than this
(the horizontal part of the turned vertical, along the holder-to-container
direction) are tried before the hand's own rotation, smaller tilt first; the
rest follow in RoboRSI's order. RoboRSI tilts "when the holding arm's elbow
conflicts with the container arm"; in the YAM station twin the two forearms came
within 0.2-0.4 mm of each other with the own rotation at all six presentation
points tried, and 6-11 mm apart with the wrist leaning 30 deg away."""

APPROACH_HEIGHT = 0.06
"""The object's bottom over the container's rim before the straight descent [m]:
RoboRSI's drop_height_m."""

RELEASE_HEIGHT = 0.015
"""The object's bottom over the rim when the hand opens [m]. tuned_on: YAM station
twin, 30 mm cube into a 60 x 60 x 40 mm cup; low enough that the cube cannot
bounce out over a 40 mm wall, high enough that the carrying hand's fingertips
(10 mm under its tool centre) stay over the rim."""

RIM_MARGIN = 0.008
"""Kept between the object's footprint and the container's outer edge [m]: a
6 mm wall plus 2 mm."""

HOLDER_KEEPOUT = 0.035
"""Least horizontal distance from the holder's tool centre to the object's near
edge at the release [m]: the stand-in for RoboRSI's swept-sphere check between
the arms. A YAM linear_4310 blade is 30 mm thick along its slide, so a hand
gripping a 6 mm wall puts its inner blade 3 + 30 mm into the opening; plus 2 mm.
tuned_on: YAM station twin, where a 30 mm cube dropped 19 mm from the holder's
tool centre into a 60 mm cup landed on that blade and bounced out."""

MAX_TILT_DEG = 15.0
"""Most the container's opening may lean from vertical, before and after the
presentation [deg]; past it a drop lands on the low wall."""

RETREAT_DISTANCE = 0.10
"""The carrying hand's back-off straight up after it opens [m] (handing-over-between-arms)."""

REACH_TOLERANCE = 0.02
"""How far a hand may end from where it was sent [m]: a servo's 4 mm ball plus a
real arm's sag, no tighter than the connectors' own bounds."""

EMPTY_FRACTION = 0.04
"""A jaw opening at or under this (robot.get_gripper, 1.0 = open) closed on
nothing: 3.8 mm on a 94 mm YAM span, the stop plus verifying-grasps' 3 mm margin."""

OPEN_FRACTION = 0.5
"""A carrying hand reading under this after open_gripper did not open."""

SAME_ARM = 0.001
"""Two arm_ids whose tool centres read this close are one arm [m]."""

MIN_LEVER = 0.002
"""Under this horizontal distance from the container's centre to the holder's
tool centre there is no "away from the holder" side [m]."""


class Output(TypedDict):
    route: str
    detail: str
    chosen_rotation: str
    container_tilt_deg: float
    container_position: dict[str, float]
    release_position: dict[str, float]
    hand_gap: float
    container_pose: dict[str, Any]


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


def _quat(r: np.ndarray) -> dict[str, float]:
    """A rotation matrix as a unit quaternion (w >= 0)."""
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + r[i, i] - r[j, j] - r[k, k], 1e-12))
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    a = np.array(q) / float(np.linalg.norm(q))
    if a[0] < 0.0:
        a = -a
    return dict(zip("wxyz", map(float, a), strict=True))


def _about(axis: tuple[float, float, float] | None, deg: float) -> np.ndarray:
    if axis is None or not deg:
        return np.eye(3)
    a = np.asarray(axis, float)
    a = a / float(np.linalg.norm(a))
    h = math.radians(deg) / 2.0
    return _rotation(
        {
            "w": math.cos(h),
            "x": a[0] * math.sin(h),
            "y": a[1] * math.sin(h),
            "z": a[2] * math.sin(h),
        }
    )


def _box(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, rotation (columns are the box axes) and half-extents of an OBB."""
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    rot = _rotation(obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    return centre, rot, half


def _reach(rot: np.ndarray, half: np.ndarray, direction: np.ndarray) -> float:
    """How far a box reaches from its centre along a unit direction."""
    return float(np.abs(rot.T @ direction) @ half)


def _tilt_deg(up: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(up[2]))))))


def run(
    ctx: NodeContext,
    container_obb: dict[str, Any],
    object_obb: dict[str, Any],
    presentation_position: dict[str, float] | None = None,
    container_arm: int = 0,
    object_arm: int = 1,
    approach_height: float = APPROACH_HEIGHT,
    release_height: float = RELEASE_HEIGHT,
    rim_margin: float = RIM_MARGIN,
    holder_keepout: float = HOLDER_KEEPOUT,
    max_tilt_deg: float = MAX_TILT_DEG,
    retreat_distance: float = RETREAT_DISTANCE,
    reach_tolerance: float = REACH_TOLERANCE,
    empty_fraction: float = EMPTY_FRACTION,
    check_reach: bool = True,
    rotations: list[str] | None = None,
) -> Output:
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "chosen_rotation": "",
        "container_tilt_deg": 0.0,
        "container_position": {},
        "release_position": {},
        "hand_gap": 0.0,
        "container_pose": {},
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
        """Send one arm and judge arrival from a fresh reading of that arm."""
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

    def reachable(a: int, position: np.ndarray, rotation: dict[str, float]) -> bool:
        if not check_reach:
            return True
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(position), "rotation": rotation}, **arm(a)
            )
        except Exception as exc:  # no preview served: let the move itself decide
            logger.info("placing-into-container-held-by-other-arm: no IK preview (%s)", exc)
            return True
        # A stall at the edge of reach reads like contact with the rim.
        return not (
            isinstance(reply, dict)
            and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
        )

    def grip(a: int) -> float:
        """The jaw opening (1.0 = open), or -1 when the hand cannot be read."""
        try:
            return float(ctx.tool("robot.get_gripper", **arm(a))["position"])
        except Exception as exc:  # an unreadable hand is not an empty one
            logger.warning(
                "placing-into-container-held-by-other-arm: arm %d unreadable (%s)", a, exc
            )
            return -1.0

    def holds(fraction: float) -> bool:
        return fraction < 0.0 or fraction > float(empty_fraction)

    a_id, b_id = int(container_arm), int(object_arm)
    if a_id == b_id:
        return finish("unreachable", f"the container and the object are both on arm {a_id}")
    a0, b0 = here(a_id), here(b_id)
    q = b0.get("rotation") or {}
    if sum(float(q.get(k, 0.0)) ** 2 for k in "wxyz") < 0.25 or (
        float(np.linalg.norm(_xyz(a0) - _xyz(b0))) < SAME_ARM
    ):
        # A connector that clamps or ignores arm_id reports one arm for both.
        return finish("unreachable", f"the connector does not report a separate arm {b_id}")

    if not holds(grip(a_id)):
        return finish("unreachable", f"arm {a_id}'s jaws are at the stop: it holds no container")
    if not holds(grip(b_id)):
        return finish("missed", f"arm {b_id}'s jaws are at the stop: it holds no object")

    c0, c_rot0, c_half = _box(container_obb)
    o0, o_rot0, o_half = _box(object_obb)
    # The opening faces along the box axis nearest vertical.
    up_axis = int(np.argmax(np.abs(c_rot0[2, :])))
    up0 = c_rot0[:, up_axis] * (1.0 if c_rot0[2, up_axis] > 0 else -1.0)
    out["container_tilt_deg"] = _tilt_deg(up0)
    if out["container_tilt_deg"] > float(max_tilt_deg):
        return finish(
            "unreachable",
            f"the container leans {out['container_tilt_deg']:.0f} deg from upright (over {max_tilt_deg:.0f})",
        )

    # 1. The holder presents the container, keeping its rotation (so it stays upright).
    a_rot0 = _rotation(a0["rotation"])
    if presentation_position is None:
        mid = 0.5 * (_xyz(a0) + _xyz(b0))
        present = np.array([mid[0], mid[1], c0[2]])
    else:
        present = np.array([float(presentation_position[k]) for k in ("x", "y", "z")])
    a_target = present + (_xyz(a0) - c0)
    if float(np.linalg.norm(a_target - _xyz(a0))) > float(reach_tolerance):
        why = move("robot.go_to_pose_cartesian", a_id, a_target, a0["rotation"])
        if why:
            notes.append(f"straight-line presentation refused ({why})")
            why = move("robot.go_to_pose", a_id, a_target, a0["rotation"])
        if why:
            return finish("unreachable", f"container to the presentation pose: {why}")
    a1 = here(a_id)
    out["container_pose"] = a1
    # Where the container is now, from the holder's actual pose (a wrist that came
    # back turned took the container with it).
    turn_a = _rotation(a1["rotation"]) @ a_rot0.T
    c1 = _xyz(a1) + turn_a @ (c0 - _xyz(a0))
    c_rot = turn_a @ c_rot0
    up = turn_a @ up0
    out["container_position"] = _vec(c1)
    out["container_tilt_deg"] = _tilt_deg(up)
    if out["container_tilt_deg"] > float(max_tilt_deg):
        return finish(
            "unreachable",
            f"the holder's wrist leaves the container {out['container_tilt_deg']:.0f} deg from upright",
        )
    rim_z = c1[2] + _reach(c_rot, c_half, np.array([0.0, 0.0, 1.0]))
    sides = [i for i in range(3) if i != up_axis]
    axes = [c_rot[:, i] for i in sides]
    halves = [float(c_half[i]) for i in sides]
    # Away from the holder's hand across the opening.
    lever = np.array([float(np.dot(_xyz(a1) - c1, ax)) for ax in axes])
    away = (
        -lever / float(np.linalg.norm(lever)) if float(np.linalg.norm(lever)) > MIN_LEVER else None
    )

    # 2. The carrier's rotation: its own, then RoboRSI's tilts.
    b_rot0 = _rotation(b0["rotation"])
    hold_offset = _xyz(b0) - o0  # tool centre from the object's centre
    tried: list[str] = []
    chosen = None
    toward = c1 - _xyz(a1)
    toward[2] = 0.0
    if float(np.linalg.norm(toward)) > MIN_LEVER:
        toward = toward / float(np.linalg.norm(toward))
        lean = {
            t[0]: float(np.dot(_about(t[1], t[2]) @ np.array([0.0, 0.0, 1.0]), toward))
            for t in TILTS
        }
        first = sorted(
            (t for t in TILTS if lean[t[0]] > LEAN_AWAY), key=lambda t: (abs(t[2]), -lean[t[0]])
        )
        order = first + [t for t in TILTS if t not in first]
    else:
        order = list(TILTS)
    if rotations:
        named = {t[0]: t for t in TILTS}
        unknown = [r for r in rotations if r not in named]
        if unknown:
            return finish(
                "unreachable", f"unknown carrier rotations {unknown}; known: {sorted(named)}"
            )
        order = [named[r] for r in rotations]
    for label, axis, deg in order:
        turn = _about(axis, deg)
        o_rot = turn @ o_rot0
        room = [
            h - float(rim_margin) - _reach(o_rot, o_half, ax)
            for h, ax in zip(halves, axes, strict=True)
        ]
        if min(room) < 0.0:
            tried.append(f"{label}: the object does not fit the opening")
            continue
        shift = 0.0
        if away is not None:
            shift = min(
                r / abs(float(w)) for r, w in zip(room, away, strict=True) if abs(float(w)) > 1e-9
            )
        drop_uv = away * shift if away is not None else np.zeros(2)
        drop = c1 + axes[0] * drop_uv[0] + axes[1] * drop_uv[1]
        bottom = _reach(o_rot, o_half, np.array([0.0, 0.0, 1.0]))
        obj_release = np.array([drop[0], drop[1], rim_z + float(release_height) + bottom])
        rot = turn @ b_rot0
        b_release = obj_release + turn @ hold_offset
        b_approach = b_release + np.array(
            [0.0, 0.0, float(approach_height) - float(release_height)]
        )
        # The object's near edge from the holder's tool centre, across the opening.
        toward = _xyz(a1) - obj_release
        toward[2] = 0.0
        dist = float(np.linalg.norm(toward))
        gap = dist - (_reach(o_rot, o_half, toward / dist) if dist > 1e-9 else 0.0)
        if gap < float(holder_keepout):
            tried.append(
                f"{label}: the object's edge would be {gap * 1e3:.0f} mm from the holder's hand"
            )
            continue
        quat = _quat(rot)
        if not (reachable(b_id, b_approach, quat) and reachable(b_id, b_release, quat)):
            tried.append(f"{label}: out of reach (IK preview)")
            continue
        why = move("robot.go_to_pose", b_id, b_approach, quat)
        if why:
            tried.append(f"{label}: approach: {why}")
            continue
        chosen = (label, turn, obj_release, gap)
        break
    if chosen is None:
        return finish(
            "unreachable",
            "no carrying-hand rotation reaches over the container ("
            + "; ".join(tried)
            + f"); arm {b_id} still holds the object, arm {a_id} the container",
        )
    label, turn, obj_release, gap = chosen
    out["chosen_rotation"], out["hand_gap"] = label, gap

    # 3. Down to the release, re-aimed from the hand's actual rotation (a free
    # wrist roll turns the object about the hand).
    b1 = here(b_id)
    turn_b = _rotation(b1["rotation"]) @ b_rot0.T
    b_release = obj_release + turn_b @ hold_offset
    why = move("robot.go_to_pose_cartesian", b_id, b_release, b1["rotation"])
    if why:
        back = move("robot.go_to_pose_cartesian", b_id, _xyz(b1), b1["rotation"])
        return finish(
            "unreachable",
            f"carrying hand down to the release ({label}): {why}"
            + (f"; back up: {back}" if back else "")
            + f"; arm {b_id} still holds the object",
        )
    b2 = here(b_id)
    released = _xyz(b2) - _rotation(b2["rotation"]) @ b_rot0.T @ hold_offset
    out["release_position"] = _vec(released)

    # 4. Release; the holder's jaws are never touched.
    opened = ctx.tool("robot.open_gripper", **arm(b_id))
    b_open = float(opened.get("position", -1.0)) if isinstance(opened, dict) else -1.0
    if b_open < 0.0:
        b_open = grip(b_id)
    if 0.0 <= b_open < OPEN_FRACTION:
        return finish(
            "missed", f"arm {b_id}'s jaws read {b_open:.3f} after opening: it still holds it"
        )

    # 5. Back up, then close once: jaws that stop short of the stop still hold the object.
    up_pose = _xyz(b2) + np.array([0.0, 0.0, float(retreat_distance)])
    retreat = move("robot.go_to_pose_cartesian", b_id, up_pose, b2["rotation"])
    if retreat:
        notes.append(f"carrying hand's retreat: {retreat}")
    ctx.tool("robot.close_gripper", **arm(b_id))
    b_closed = grip(b_id)
    ctx.tool("robot.open_gripper", **arm(b_id))
    if b_closed > float(empty_fraction):
        return finish(
            "missed",
            f"arm {b_id}'s jaws closed to {b_closed:.3f} after the release: the object is still in them",
        )

    # 6. The container: still held, and the release over its opening.
    a_grip = grip(a_id)
    if not holds(a_grip):
        return finish(
            "missed", f"arm {a_id}'s jaws ran to the stop ({a_grip:.3f}): the container fell"
        )
    a2 = here(a_id)
    turn_a2 = _rotation(a2["rotation"]) @ a_rot0.T
    c2 = _xyz(a2) + turn_a2 @ (c0 - _xyz(a0))
    out["container_pose"] = a2
    out["container_position"] = _vec(c2)
    out["container_tilt_deg"] = _tilt_deg(turn_a2 @ up0)
    d = released - c2
    outside = [
        (float(np.dot(d, turn_a2 @ c_rot0[:, i])), float(c_half[i]) - float(rim_margin))
        for i in sides
    ]
    if any(abs(u) > lim for u, lim in outside):
        return finish(
            "missed",
            "released outside the container's opening ("
            + ", ".join(f"{u * 1e3:+.0f} of +-{lim * 1e3:.0f} mm" for u, lim in outside)
            + ")",
        )
    return finish(
        "placed",
        f"released {release_height * 1e3:.0f} mm over the rim with the {label} rotation; "
        "the carrying hand is empty, the holder still holds the container",
    )
