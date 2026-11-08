"""Turn a visual plan's endpoint into a checked 3D target, a goal height and a travel route.

A visual plan is what ``robobrain.trace`` and ``molmo-motion.forecast`` return
(their shared schema: ``tracks[i].query_pixel`` / ``pixels`` ...). Only the
image part is trusted: the endpoint pixel, the path's pixels and the direction
of its last segment. The 3D target comes from the measured depth and the
calibrated camera, the heights from the measured scene, and the checks (on the
table, in reach, an optional pointing cross-check) decide whether the plan is
used. Nothing moves.
"""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

PLANNER = "robobrain.trace"
"""Default planner tool. ``molmo-motion.forecast`` (and a future visual-plan
forecaster) return the same schema; tools named ``*.forecast`` are given the
start as ``query_points``, the others as ``start_point``."""

DEPTH_WINDOW = 3
"""Half-size [px] of the depth window under the endpoint (7 x 7, molmo-motion's
lifting window); its median is the endpoint's depth."""

MIN_PLAN_PX = 12.0
"""An endpoint closer than this to the start [px] is a plan that does not move.
MolmoMotion forecast 0-3.5 mm of motion from still station frames (its
Measured section): it continues motion, it does not invent it."""

ON_TABLE_TOLERANCE = 0.02
"""How far under the table plane the endpoint's surface may read [m]: a few
times a calibrated station camera's error (RMS 5.4 mm). Further down is the
floor past the table's edge."""

MAX_TARGET_HEIGHT = 0.30
"""Highest a target surface may stand over the table [m]; above it the depth
read the arm or something held, not a place to go."""

TOP_RADIUS = 0.06
"""Horizontal radius [m] around the target in which the object's top is
measured: covers a 100 mm box's walls from its centre."""

TOP_RANK = 5
"""The object's top (and an obstacle's) is the TOP_RANK-th highest point of the
area (every 2nd depth pixel): a thin rim is a few per cent of a box's area, so a
percentile misses it, and a lone noisy pixel on an edge is not the top."""

GOAL_CLEARANCE = 0.03
"""Tool-centre height over the measured top when the caller gives no goal
height [m]: a 30 mm part hanging under the hand then clears a box's rim."""

TRAVEL_HEIGHT = 0.12
"""Lowest travel height over the table [m]: approaching-above-a-target's and
executing-keypoint-motion's hover clearance (graph-as-policy's 0.12 m, MOKA's
13 cm safe_z)."""

TRAVEL_CLEARANCE = 0.08
"""Clearance over anything measured under the travel route [m]: placing-with-
contact-descent's approach height, which clears a 40 mm wall with a 30 mm part
hanging under the hand."""

MAX_TRAVEL_HEIGHT = 0.35
"""Highest travel height over the table [m]; a route that needs more is
rejected. RoboBrain's own arcs rose to 0.22-0.33 m (tools/robobrain Measured)."""

CORRIDOR_RADIUS = 0.04
"""Half-width [m] of the corridor under each travel leg searched for obstacles:
an open linear_4310 hand's half span (94 mm / 2)."""

HAND_RADIUS = 0.08
"""Points within this horizontal radius [m] of the tool centre, from 6 cm
under it upward, are the hand itself and are ignored."""

MIN_WAYPOINT_SPACING = 0.03
"""Path waypoints closer than this [m] to the previous one or to the target
are dropped."""

CROSS_CHECK_TOOL = "gemini-er.point"
CROSS_CHECK_TOLERANCE = 0.05
"""How far [m, horizontal] the cross-check's nearest point may be from the
plan's target: half a 100 mm box, about the 20-35 px RoboBrain's cable
endpoints missed a ring by at ~0.9 m (tools/robobrain Measured)."""

MIN_TOOL_HEIGHT = 0.015
"""Floor on the goal [m over the table]: a real YAM station refuses a tool
centre under 14 mm over its calibrated table."""


class Output(TypedDict):
    route: str
    detail: str
    proposal: dict[str, Any]
    target_point: dict[str, float]
    goal_pose: dict[str, Any]
    hover_pose: dict[str, Any]
    waypoints: list[dict[str, float]]
    approach_direction: dict[str, float]
    target_obb: dict[str, Any]
    endpoint_pixel: list[float]
    object_top_z: float
    table_z: float
    travel_z: float
    cross_check_distance: float
    planner_calls: int
    attempts: list[str]


def _vec(p: np.ndarray) -> dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _xyz(p: dict[str, Any]) -> np.ndarray:
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _pose_matrix(pose: Any) -> np.ndarray:
    """gap's Se3Pose (camera-to-world) or a 4x4."""
    if isinstance(pose, dict):
        p, q = pose["position"], pose["rotation"]
        w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.eye(4)
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, 3] = [float(p[k]) for k in ("x", "y", "z")]
        return m
    m = np.asarray(pose, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError("camera pose is an Se3Pose or a 4x4 camera-to-world matrix")
    return m


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    usable = [
        c
        for c in cameras
        if c.get("rgb") is not None
        and c.get("depth") is not None
        and c.get("intrinsics") is not None
        and c.get("pose") is not None
    ]
    if not usable:
        raise ValueError("no camera with rgb, depth, intrinsics and pose in the observation")
    if name:
        for c in usable:
            if c.get("name") == name:
                return c
        raise ValueError(f"no usable camera named {name!r} (has {[c.get('name') for c in usable]})")
    for c in usable:
        if any(k in str(c.get("name", "")).lower() for k in ("top", "overhead")):
            return c
    return usable[0]


class _View:
    """One calibrated RGB-D camera (OpenCV axes, camera-to-world pose)."""

    def __init__(self, camera: dict[str, Any]) -> None:
        self.name = str(camera.get("name", ""))
        self.rgb = np.asarray(camera["rgb"], dtype=np.uint8)
        self.depth = np.asarray(camera["depth"], dtype=np.float64).reshape(self.rgb.shape[:2])
        self.k = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        self.t = _pose_matrix(camera["pose"])
        self.h, self.w = self.depth.shape
        self._cloud: np.ndarray | None = None

    def inside(self, uv: np.ndarray) -> bool:
        return 0.0 <= uv[0] < self.w and 0.0 <= uv[1] < self.h

    def ray(self, uv: np.ndarray) -> np.ndarray:
        return self.t[:3, :3] @ np.array(
            [(uv[0] - self.k[0, 2]) / self.k[0, 0], (uv[1] - self.k[1, 2]) / self.k[1, 1], 1.0]
        )

    def lift(self, uv: np.ndarray) -> np.ndarray | None:
        """World point under a pixel: the median valid depth of a small window."""
        u, v = int(round(uv[0])), int(round(uv[1]))
        r = DEPTH_WINDOW
        patch = self.depth[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1]
        ok = np.isfinite(patch) & (patch > 0.05) & (patch < 5.0)
        if not ok.any():
            return None
        z = float(np.median(patch[ok]))
        cam = np.array(
            [
                (uv[0] - self.k[0, 2]) * z / self.k[0, 0],
                (uv[1] - self.k[1, 2]) * z / self.k[1, 1],
                z,
            ]
        )
        return self.t[:3, :3] @ cam + self.t[:3, 3]

    def at_height(self, uv: np.ndarray, z: float) -> np.ndarray | None:
        """Where the pixel's ray crosses the plane at world height z."""
        ray = self.ray(uv)
        if abs(ray[2]) < 1e-6:
            return None
        s = (float(z) - self.t[2, 3]) / ray[2]
        return None if s <= 0 else self.t[:3, 3] + s * ray

    def cloud(self, stride: int = 2) -> np.ndarray:
        """Every stride-th valid depth pixel as a world point."""
        if self._cloud is None:
            v, u = np.mgrid[0 : self.h : stride, 0 : self.w : stride]
            d = self.depth[v, u]
            ok = np.isfinite(d) & (d > 0.05) & (d < 5.0)
            u, v, d = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok]
            cam = np.stack(
                [(u - self.k[0, 2]) * d / self.k[0, 0], (v - self.k[1, 2]) * d / self.k[1, 1], d],
                -1,
            )
            self._cloud = cam @ self.t[:3, :3].T + self.t[:3, 3]
        return self._cloud


def _high(z: np.ndarray) -> float:
    """The TOP_RANK-th highest value, or -inf for none."""
    if not len(z):
        return -math.inf
    return float(np.sort(z)[-min(TOP_RANK, len(z))])


def _table_z(
    ctx: NodeContext, view: _View, given: float | None, arm: dict[str, Any]
) -> tuple[float, str]:
    if given is not None:
        return float(given), "given"
    try:
        return float(
            ctx.tool("robot.describe_workspace", **arm)["surface_z"]
        ), "robot.describe_workspace"
    except Exception as exc:  # not every connector serves it: fall back to the scene
        logger.info("following-a-visual-plan: no workspace surface (%s)", exc)
    z = view.cloud()[:, 2]
    bins = np.floor(z / 0.004)
    values, counts = np.unique(bins, return_counts=True)
    return float(
        (values[np.argmax(counts)] + 0.5) * 0.004
    ), "the depth image's fullest 4 mm height bin"


def _ik_ok(ctx: NodeContext, pose: dict[str, Any], arm: dict[str, Any]) -> bool | None:
    """True / False from robot.solve_ik, None when the connector cannot preview."""
    try:
        reply = ctx.tool("robot.solve_ik", pose=pose, **arm)
    except Exception as exc:
        logger.info("following-a-visual-plan: no IK preview (%s)", exc)
        return None
    if not isinstance(reply, dict):
        return True
    return not (reply.get("success") is False or reply.get("within_servo_tolerance") is False)


def _call_planner(
    ctx: NodeContext,
    view: _View,
    planner: str,
    instruction: str,
    start_pixel: list[float] | None,
    planner_args: dict[str, Any] | None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "image": view.rgb,
        "instruction": instruction,
        "depth": view.depth.astype(np.float32),
        "intrinsics": view.k,
        "camera_pose": view.t,
    }
    if start_pixel is not None:
        start = [float(start_pixel[0]), float(start_pixel[1])]
        if planner.endswith(".forecast"):
            args["query_points"] = [start]
        else:
            args["start_point"] = start
    elif planner.endswith(".forecast"):
        raise ValueError(f"{planner} forecasts given pixels: pass start_pixel")
    args.update(planner_args or {})
    return ctx.tool(planner, **args)


def run(
    ctx: NodeContext,
    plan: dict[str, Any] | None = None,
    instruction: str = "",
    observation: dict[str, Any] | None = None,
    planner: str = PLANNER,
    start_pixel: list[float] | None = None,
    planner_args: dict[str, Any] | None = None,
    replan: bool = False,
    track: int = 0,
    camera_name: str = "",
    table_z: float | None = None,
    workspace_bounds: dict[str, Any] | None = None,
    goal_height: float | None = None,
    goal_clearance: float = GOAL_CLEARANCE,
    travel_height: float = TRAVEL_HEIGHT,
    travel_clearance: float = TRAVEL_CLEARANCE,
    max_travel_height: float = MAX_TRAVEL_HEIGHT,
    follow_path: bool = False,
    approach_offset: float = 0.0,
    approach_from: str = "last_segment",
    target_phrase: str = "",
    cross_check_tool: str = CROSS_CHECK_TOOL,
    cross_check_tolerance: float = CROSS_CHECK_TOLERANCE,
    on_table_tolerance: float = ON_TABLE_TOLERANCE,
    max_target_height: float = MAX_TARGET_HEIGHT,
    top_radius: float = TOP_RADIUS,
    check_reach: bool = True,
    allow_top_down: bool = True,
    arm_id: int = 0,
) -> Output:
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "no_plan",
        "detail": "",
        "proposal": {},
        "target_point": {},
        "goal_pose": {},
        "hover_pose": {},
        "waypoints": [],
        "approach_direction": {},
        "target_obb": {},
        "endpoint_pixel": [],
        "object_top_z": float("nan"),
        "table_z": float("nan"),
        "travel_z": float("nan"),
        "cross_check_distance": -1.0,
        "planner_calls": 0,
        "attempts": [],
    }
    tries = 2 if replan and instruction else 1
    for attempt in range(tries):
        if attempt or observation is None:
            observation = ctx.tool("robot.get_observation")
        try:
            view = _View(_camera(observation, camera_name))
        except ValueError as exc:
            out.update(route="no_plan", detail=str(exc))
            return out
        if attempt or plan is None:
            if not instruction:
                out.update(
                    route="no_plan", detail="no plan given and no instruction to ask a planner with"
                )
                return out
            out["planner_calls"] += 1
            try:
                plan = _call_planner(ctx, view, planner, instruction, start_pixel, planner_args)
            except Exception as exc:  # a planner that raised gave no plan
                plan = {"found": False, "error": f"{planner} raised: {exc}"}
        route, detail = _propose(
            ctx,
            out,
            view,
            plan,
            arm,
            track=int(track),
            table_z=table_z,
            workspace_bounds=workspace_bounds,
            goal_height=goal_height,
            goal_clearance=float(goal_clearance),
            travel_height=float(travel_height),
            travel_clearance=float(travel_clearance),
            max_travel_height=float(max_travel_height),
            follow_path=bool(follow_path),
            approach_offset=float(approach_offset),
            approach_from=str(approach_from),
            target_phrase=target_phrase,
            cross_check_tool=cross_check_tool,
            cross_check_tolerance=float(cross_check_tolerance),
            on_table_tolerance=float(on_table_tolerance),
            max_target_height=float(max_target_height),
            top_radius=float(top_radius),
            check_reach=bool(check_reach),
            allow_top_down=bool(allow_top_down),
        )
        out["attempts"].append(f"{route}: {detail}" if detail else route)
        out.update(route=route, detail=detail)
        if route == "proposed":
            break
        logger.info("following-a-visual-plan: attempt %d %s: %s", attempt + 1, route, detail)
    if out["planner_calls"] > 1 or len(out["attempts"]) > 1:
        out["detail"] = " | ".join(f"plan {i + 1}: {a}" for i, a in enumerate(out["attempts"]))
    return out


def _propose(
    ctx: NodeContext,
    out: Output,
    view: _View,
    plan: dict[str, Any] | None,
    arm: dict[str, Any],
    *,
    track: int,
    table_z: float | None,
    workspace_bounds: dict[str, Any] | None,
    goal_height: float | None,
    goal_clearance: float,
    travel_height: float,
    travel_clearance: float,
    max_travel_height: float,
    follow_path: bool,
    approach_offset: float,
    approach_from: str,
    target_phrase: str,
    cross_check_tool: str,
    cross_check_tolerance: float,
    on_table_tolerance: float,
    max_target_height: float,
    top_radius: float,
    check_reach: bool,
    allow_top_down: bool,
) -> tuple[str, str]:
    # --- the plan's image part ------------------------------------------------
    plan = plan or {}
    tracks = plan.get("tracks") or []
    if not plan.get("found") or not tracks:
        return "no_plan", str(plan.get("error") or "the planner returned no track")
    if not 0 <= track < len(tracks):
        return "no_plan", f"the plan has {len(tracks)} track(s), asked for track {track}"
    tr = tracks[track]
    pixels = [np.asarray(p, dtype=np.float64) for p in tr.get("pixels") or []]
    if not pixels:
        return "no_plan", "the track has no keypoints after its start"
    start_px = np.asarray(tr.get("query_pixel") or pixels[0], dtype=np.float64)
    end_px = pixels[-1]
    out["endpoint_pixel"] = [float(end_px[0]), float(end_px[1])]
    moved = float(np.linalg.norm(end_px - start_px))
    if moved < MIN_PLAN_PX:
        return "no_plan", (
            f"the plan ends {moved:.1f} px from its start: no motion proposed "
            "(a forecaster shown a still scene forecasts stillness)"
        )
    if not view.inside(end_px):
        return (
            "plan_rejected",
            f"the endpoint ({end_px[0]:.0f}, {end_px[1]:.0f}) px is outside the image",
        )

    # --- lift the endpoint with the measured depth ------------------------------
    target = view.lift(end_px)
    if target is None:
        return "plan_rejected", "no valid depth at the endpoint pixel"
    out["target_point"] = _vec(target)
    tz, tz_source = _table_z(ctx, view, table_z, arm)
    out["table_z"] = tz
    over = float(target[2] - tz)
    if over < -on_table_tolerance:
        return "plan_rejected", (
            f"off the table: the endpoint's surface is {-over * 1e3:.0f} mm under the table plane "
            f"({tz_source})"
        )
    if over > max_target_height:
        return (
            "plan_rejected",
            f"the endpoint's surface is {over * 1e3:.0f} mm over the table: not a place on it",
        )
    if workspace_bounds:
        for axis in ("x", "y"):
            lo, hi = (float(b) for b in workspace_bounds.get(axis, (-math.inf, math.inf)))
            if not lo <= target["xyz".index(axis)] <= hi:
                return "plan_rejected", (
                    f"off the table: target {axis} = {target['xyz'.index(axis)]:.3f} m is outside "
                    f"the workspace [{lo:.3f}, {hi:.3f}]"
                )

    # --- the measured scene: the object's top, the hand ----------------------------
    hand_pose = _pose(ctx.tool("robot.get_ee_pose", **arm))
    hand = _xyz(hand_pose["position"])
    cloud = view.cloud()
    near_hand = (np.linalg.norm(cloud[:, :2] - hand[:2], axis=1) < HAND_RADIUS) & (
        cloud[:, 2] > hand[2] - 0.06
    )
    scene = cloud[~near_hand & (cloud[:, 2] > tz - on_table_tolerance)]
    around = scene[np.linalg.norm(scene[:, :2] - target[:2], axis=1) < top_radius]
    top = max(float(target[2]), _high(around[:, 2]))
    out["object_top_z"] = top
    if goal_height is not None:
        goal_z = float(goal_height)
    else:
        goal_z = top + goal_clearance
    goal_z = max(goal_z, tz + MIN_TOOL_HEIGHT)

    # --- approach side: the last segment of the path in the image ------------------
    prev = start_px
    if approach_from == "last_segment":
        for p in reversed(pixels[:-1]):
            if np.linalg.norm(p - end_px) >= 3.0:
                prev = p
                break
    a, b = view.at_height(prev, target[2]), view.at_height(end_px, target[2])
    direction = np.zeros(3)
    if a is not None and b is not None and np.linalg.norm((b - a)[:2]) > 1e-6:
        direction[:2] = (b - a)[:2] / np.linalg.norm((b - a)[:2])
    out["approach_direction"] = _vec(direction)

    # --- travel height and route --------------------------------------------------
    travel = max(tz + travel_height, top + travel_clearance, goal_z)
    ceiling = tz + max_travel_height
    route_pts: list[np.ndarray] = []
    for _ in range(3):
        route_pts = []
        last = hand[:2]
        if follow_path:
            for p in pixels[:-1]:
                w = view.at_height(p, travel)
                if w is None:
                    continue
                if (
                    np.linalg.norm(w[:2] - last) < MIN_WAYPOINT_SPACING
                    or np.linalg.norm(w[:2] - target[:2]) < MIN_WAYPOINT_SPACING
                ):
                    continue
                route_pts.append(np.array([w[0], w[1], travel]))
                last = w[:2]
        if approach_offset > 0 and np.linalg.norm(direction) > 0:
            pre = target[:2] - direction[:2] * approach_offset
            route_pts.append(np.array([pre[0], pre[1], travel]))
        legs = [hand[:2], *(w[:2] for w in route_pts), target[:2]]
        highest = -math.inf
        for p0, p1 in zip(legs[:-1], legs[1:], strict=False):
            n = max(2, int(np.linalg.norm(p1 - p0) / 0.01) + 1)
            for s in np.linspace(0.0, 1.0, n):
                q = p0 + s * (p1 - p0)
                under = scene[np.linalg.norm(scene[:, :2] - q, axis=1) < CORRIDOR_RADIUS]
                highest = max(highest, _high(under[:, 2]))
        need = max(travel, highest + travel_clearance)
        if need <= travel + 1e-3:
            break
        travel = need
    out["travel_z"] = travel
    if travel > ceiling + 1e-6:
        return "plan_rejected", (
            f"the route needs a travel height {(travel - tz) * 1e3:.0f} mm over the table "
            f"(something {(travel - travel_clearance - tz) * 1e3:.0f} mm high under it); "
            f"the ceiling is {max_travel_height * 1e3:.0f} mm"
        )

    # --- reach: preview the hover and the goal -------------------------------------
    rotations = [("current", hand_pose["rotation"])]
    if allow_top_down:
        try:
            frame = ctx.tool("robot.grasp_frame", **arm)
            rotations += [("top-down", frame["rotation"])]
            if frame.get("rotation_flipped"):
                rotations += [("top-down flipped", frame["rotation_flipped"])]
        except Exception as exc:
            logger.info("following-a-visual-plan: no grasp frame (%s)", exc)
    hover_pos = np.array([target[0], target[1], travel])
    goal_pos = np.array([target[0], target[1], goal_z])
    chosen = rotations[0]
    if check_reach:
        verdicts = []
        for label, rot in rotations:
            ok = [
                _ik_ok(ctx, {"position": _vec(p), "rotation": rot}, arm)
                for p in (hover_pos, goal_pos)
            ]
            if None in ok:  # the connector previews nothing: the moves will tell
                chosen = (label, rot)
                break
            if all(ok):
                chosen = (label, rot)
                break
            verdicts.append(label)
        else:
            return "plan_rejected", (
                f"out of reach: no IK for the hover / goal over ({target[0]:.3f}, {target[1]:.3f}) "
                f"with the {', '.join(verdicts)} rotation(s)"
            )
        kept = []
        for w in route_pts:
            if _ik_ok(ctx, {"position": _vec(w), "rotation": chosen[1]}, arm) is False:
                logger.info(
                    "following-a-visual-plan: dropped an unreachable waypoint %s", w.round(3)
                )
                continue
            kept.append(w)
        route_pts = kept

    # --- optional cross-check of the target phrase ---------------------------------
    cross_note = ""
    if target_phrase:
        try:
            reply = ctx.tool(cross_check_tool, image=view.rgb, query=target_phrase)
        except Exception as exc:
            reply = None
            cross_note = f"; cross-check unavailable ({cross_check_tool} raised: {exc})"
        if reply is not None:
            points = reply.get("points") or (
                [{"pixel_x": reply["pixel_x"], "pixel_y": reply["pixel_y"]}]
                if reply.get("found")
                else []
            )
            if not reply.get("found", bool(points)) or not points:
                return (
                    "plan_rejected",
                    f"cross-check: {cross_check_tool} found no {target_phrase!r}",
                )
            best = math.inf
            for p in points:
                w = view.lift(np.array([float(p["pixel_x"]), float(p["pixel_y"])]))
                if w is not None:
                    best = min(best, float(np.linalg.norm(w[:2] - target[:2])))
            out["cross_check_distance"] = best if math.isfinite(best) else -1.0
            if best > cross_check_tolerance:
                return "plan_rejected", (
                    f"cross-check disagrees: {cross_check_tool}'s nearest {target_phrase!r} is "
                    + (f"{best * 1e3:.0f} mm" if math.isfinite(best) else "without depth,")
                    + f" from the plan's target (bound {cross_check_tolerance * 1e3:.0f} mm)"
                )

    rotation = chosen[1]
    hover = {"position": _vec(hover_pos), "rotation": rotation}
    goal = {"position": _vec(goal_pos), "rotation": rotation}
    half_h = max(0.0, (top - tz) / 2.0)
    out.update(
        goal_pose=goal,
        hover_pose=hover,
        waypoints=[_vec(w) for w in route_pts],
        target_obb={
            "center": {"x": float(target[0]), "y": float(target[1]), "z": tz + half_h},
            "extent": {"x": 0.0, "y": 0.0, "z": half_h},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    )
    out["proposal"] = {
        "target_point": out["target_point"],
        "goal_pose": goal,
        "hover_pose": hover,
        "waypoints": out["waypoints"],
        "travel_z": travel,
        "table_z": tz,
        "object_top_z": top,
        "rotation_source": chosen[0],
        "approach_direction": out["approach_direction"],
        "endpoint_pixel": out["endpoint_pixel"],
        "camera": view.name,
        "backend": plan.get("backend", ""),
    }
    return "proposed", (
        f"target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) from endpoint "
        f"({end_px[0]:.0f}, {end_px[1]:.0f}) px, top {(top - tz) * 1e3:.0f} mm, goal "
        f"{(goal_z - tz) * 1e3:.0f} mm and travel {(travel - tz) * 1e3:.0f} mm over the table, "
        f"{len(route_pts)} waypoint(s), {chosen[0]} rotation{cross_note}"
    )
