"""Locate an open container of known size in the wrist camera's own frame, before placing into it."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

CELL = 0.004
"""Plan-view grid the outline is slid over [m] (finding-boxes-of-known-size-from-overhead-depth)."""

MIN_POINTS = 150
"""Fewer rim-band points than this in the view and nothing is fitted."""

YAW_STEP_DEG = 2.0

BAND_BELOW, BAND_ABOVE = 0.022, 0.012
"""Rim points are those from 22 mm under to 12 mm over the expected rim height [m]. tuned_on: a
66 mm tall cardboard box seen by a YAM's wrist depth camera from 0.2-0.35 m (2026-11-15). The band
is tighter than the overhead finder's because the arm's height error is smaller than its error in
the plane; not swept."""

SEARCH_RADIUS = 0.30
"""Only points within this of the expected centre are looked at [m]. tuned_on: same; chosen."""

YAW_WINDOW_DEG = 12.0
"""The outline is turned at most this far from the expected turn. tuned_on: same; chosen. It
keeps a nearly square box from coming out a quarter turn round."""

MIN_COVERAGE = 0.30
"""Share of the outline that must lie on rim points. tuned_on: same; a wrist view cuts the rim, so
it is lower than the overhead finder's 0.35. Not swept."""

EXCLUDE_MARGIN = 0.012
"""Points this far outside a neighbouring box's footprint are still that box's wall [m]."""

MAX_SHIFT = 0.10
"""A fit further than this from the expected centre is taken for something else [m]. tuned_on:
same station: the wrist camera put the box 27, 72 and 78 mm from the overhead camera's place for
it in the three runs whose notes give the number; 10 cm is a guard over that, not a measured bound."""

MIN_RANGE = 0.16
"""Depth nearer than this is the hand's own fingers and is dropped [m]. tuned_on: a YAM
linear_4310 hand and its wrist D405; on another hand pass min_range or an ignore mask's worth."""

CLEAR_OVER_RIM = 0.11
"""Before crossing to a view the empty hand rises to this far over the container's rim [m] (or
clear_z). tuned_on: same station."""

STANDOFFS_AS_GIVEN = (0.14, 0.20, 0.08)
"""With the hand turned as `rotations` say, the views are this far short of the container's
centre, towards where the hand starts [m], tried in this order. tuned_on: same station (a hand
leaning 40-55 deg sideways, as it would carry)."""

STANDOFFS_DOWN = (0.05, 0.09, 0.0)
"""With the hand pointing down, the views are this far short of the centre [m]: over the box's
near half. Added after a layout in which no leaning view was reachable (2026-11-15)."""

HEIGHTS_AS_GIVEN = (0.33, 0.30, 0.27, 0.24, 0.21, 0.18, 0.15, 0.12)
HEIGHTS_DOWN = (0.23, 0.20, 0.17, 0.14, 0.11, 0.08)
"""Hand heights over the rim tried at each standoff, highest first [m]. tuned_on: same station
(0.40 m and 0.30 m over a counter with a 66 mm rim, in 3 cm steps)."""

MAX_IK_QUERIES = 30
"""robot.solve_ik previews before the search for a view gives up. A guard, not a measurement: on
the source station an unbounded search (well over a hundred plan requests in a row) held up the
motion server's control loop until its supervisor stopped the arms."""


class Output(TypedDict):
    route: str
    detail: str
    located_obb: dict[str, Any] | None
    aim_point: dict[str, float] | None
    shift: dict[str, float] | None
    shift_m: float
    yaw_deg: float
    coverage: float
    rim_points: int
    views_tried: int
    ik_queries: int
    view_pose: dict[str, Any] | None
    camera: str


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


def _matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "pose" in pose and "position" not in pose:
        pose = pose["pose"]
    if isinstance(pose, dict):
        m = np.eye(4)
        m[:3, :3] = _rot(pose.get("rotation") or {"w": 1, "x": 0, "y": 0, "z": 0})
        p = pose["position"]
        m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
        return m
    m = np.asarray(pose, dtype=float)
    if m.shape != (4, 4):
        raise ValueError("camera pose must be an Se3Pose {position, rotation} or a 4x4 matrix")
    return m


def fit_known_rectangle(
    xy: np.ndarray,
    size: tuple[float, float],
    cell: float = CELL,
    min_coverage: float = MIN_COVERAGE,
    yaws_deg: Any = None,
) -> dict[str, Any] | None:
    """Where a ``size`` (long, short) rectangle's outline lies along the plan-view points ``xy``.

    For each turn of the rectangle its outline (2 cells thick) is slid over the points' occupancy
    grid (grown one cell), and the place and turn with the largest share of the outline on occupied
    cells wins. A rim seen in pieces, or beside something as tall as it, is still placed: the size
    is given, not measured. Returns {center (2,), yaw_deg in [-90, 90), coverage} or None when
    there are under MIN_POINTS points or the best coverage is under ``min_coverage``.
    """
    import cv2

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < MIN_POINTS:
        return None
    long_side, short_side = float(size[0]), float(size[1])
    half = int(math.hypot(long_side, short_side) / 2 / cell) + 3
    lo = xy.min(0) - (half + 1) * cell
    n = np.ceil((xy.max(0) - lo) / cell).astype(int) + half + 2
    img = np.zeros((n[1], n[0]), np.float32)
    ij = ((xy - lo) / cell).astype(int)
    img[ij[:, 1], ij[:, 0]] = 1.0
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    yaws = np.arange(-90.0, 90.0, YAW_STEP_DEG) if yaws_deg is None else np.asarray(yaws_deg, float)
    corners0 = np.array(
        [
            [long_side / 2, short_side / 2],
            [-long_side / 2, short_side / 2],
            [-long_side / 2, -short_side / 2],
            [long_side / 2, -short_side / 2],
        ]
    )
    best: tuple[float, Any] = (0.0, None)
    for yaw in np.radians(yaws):
        c, s = math.cos(yaw), math.sin(yaw)
        kernel = np.zeros((2 * half + 1, 2 * half + 1), np.uint8)
        corners = corners0 @ np.array([[c, s], [-s, c]])
        cv2.polylines(
            kernel, [np.round(corners / cell + half).astype(np.int32)], True, 1, thickness=2
        )
        score = cv2.filter2D(
            img, -1, kernel.astype(np.float32) / kernel.sum(), borderType=cv2.BORDER_CONSTANT
        )
        _, top, _, at = cv2.minMaxLoc(score)
        if top > best[0]:
            best = (float(top), (at, float(yaw)))
    if best[1] is None or best[0] < min_coverage:
        return None
    (ix, iy), yaw = best[1]
    yaw_deg = (math.degrees(yaw) + 90.0) % 180.0 - 90.0
    return {
        "center": lo + (np.array([ix, iy]) + 0.5) * cell,
        "yaw_deg": float(yaw_deg),
        "coverage": float(best[0]),
    }


def footprint(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """An upright box's (centre xy, half extents along its own x / y, yaw [rad])."""
    c = obb["center"]
    e = obb["extent"]
    r = _rot(obb.get("orientation") or {"w": 1, "x": 0, "y": 0, "z": 0})
    return (
        np.array([float(c["x"]), float(c["y"])]),
        np.array([float(e["x"]), float(e["y"])]),
        math.atan2(r[1, 0], r[0, 0]),
    )


def outside(xy: np.ndarray, obb: dict[str, Any], margin: float) -> np.ndarray:
    """Which plan-view points lie more than ``margin`` outside ``obb``'s footprint."""
    centre, half, yaw = footprint(obb)
    d = xy - centre
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return (np.abs(lx) > half[0] + margin) | (np.abs(ly) > half[1] + margin)


def upright_obb(
    center_xy: np.ndarray, size: tuple[float, float], yaw_deg: float, rim_z: float, support_z: float
) -> dict[str, Any]:
    """The hub's OrientedBoundingBox (half extents; local x the long side) for an upright box."""
    h = math.radians(yaw_deg) / 2
    return {
        "center": {
            "x": float(center_xy[0]),
            "y": float(center_xy[1]),
            "z": (rim_z + support_z) / 2,
        },
        "extent": {"x": size[0] / 2, "y": size[1] / 2, "z": max(rim_z - support_z, 0.0) / 2},
        "orientation": {"w": math.cos(h), "x": 0.0, "y": 0.0, "z": math.sin(h)},
    }


def _wrist_camera(ctx: NodeContext, frame: dict | None, name: str | None) -> dict | None:
    """The named posed depth camera; unnamed, the first on a wrist."""
    if frame is not None:
        return frame
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    if isinstance(cams, dict):
        cams = [dict(v, name=v.get("name", k)) for k, v in cams.items()]
    cams = [c for c in (cams or []) if c.get("depth") is not None and c.get("pose") is not None]
    if name:
        return next((c for c in cams if str(c.get("name")) == name), None)
    return next(
        (c for c in cams if any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))),
        None,
    )


def world_points(cam: dict, stride: int = 1, min_range: float = 0.0) -> np.ndarray:
    """The frame's valid depth pixels further than ``min_range`` as (N, 3) world points."""
    depth = np.asarray(cam["depth"], dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t = _matrix(cam["pose"])
    sub = depth[::stride, ::stride]
    vv, uu = np.nonzero(np.isfinite(sub) & (sub > max(min_range, 0.0)))
    z = sub[vv, uu]
    pc = np.column_stack(
        [(uu * stride - k[0, 2]) * z / k[0, 0], (vv * stride - k[1, 2]) * z / k[1, 1], z]
    )
    return pc @ t[:3, :3].T + t[:3, 3]


def box_in_view(
    points: np.ndarray,
    container_obb: dict[str, Any],
    other_obbs: list | None = None,
    min_coverage: float = MIN_COVERAGE,
    yaw_window_deg: float = YAW_WINDOW_DEG,
    search_radius: float = SEARCH_RADIUS,
) -> dict[str, Any] | None:
    """The container as these world points have it: {center (2,), yaw_deg, coverage, rim_points}.

    The rectangle of the container's own size is slid along the rim points near where it is
    expected, turned at most ``yaw_window_deg`` from the expected turn. A rim the view cuts is
    still placed by its size: a rectangle put round what is seen sat 3.7 cm off on a box seen at
    a slant. None when too little rim is seen.
    """
    centre, half, yaw = footprint(container_obb)
    rim_z = float(container_obb["center"]["z"]) + float(container_obb["extent"]["z"])
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    near = np.linalg.norm(pts[:, :2] - centre, axis=1) < search_radius
    band = near & (pts[:, 2] > rim_z - BAND_BELOW) & (pts[:, 2] < rim_z + BAND_ABOVE)
    xy = pts[band, :2]
    for other in other_obbs or []:
        xy = xy[outside(xy, other, EXCLUDE_MARGIN)]
    yaw0 = math.degrees(yaw)
    fit = fit_known_rectangle(
        xy,
        (2 * half[0], 2 * half[1]),
        min_coverage=min_coverage,
        yaws_deg=np.arange(yaw0 - yaw_window_deg, yaw0 + yaw_window_deg + 0.1, YAW_STEP_DEG),
    )
    if fit is None:
        return None
    fit["rim_points"] = int(len(xy))
    return fit


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v: Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def view_candidates(
    centre: np.ndarray,
    rim_z: float,
    hand_xy: np.ndarray,
    given: list[dict],
    down: list[dict],
) -> list[Any]:
    """Per view: (xy, rotations to try, heights to try, highest first).

    The views stand short of the container's centre on the side the hand starts from: first with
    the hand as ``given`` (as it will be when it lets go), then pointing ``down`` from over the
    container's near half, lower - an empty hand only has to clear the rims.
    """
    back = hand_xy - centre
    dist = float(np.linalg.norm(back))
    back = back / dist if dist > 1e-3 else np.zeros(2)
    views = []
    if given:
        views += [
            (centre + d * back, given, [rim_z + h for h in HEIGHTS_AS_GIVEN])
            for d in STANDOFFS_AS_GIVEN
        ]
    if down:
        views += [
            (centre + d * back, down, [rim_z + h for h in HEIGHTS_DOWN]) for d in STANDOFFS_DOWN
        ]
    return views


def run(
    ctx: NodeContext,
    container_obb: dict,
    other_obbs: list | None = None,
    rotations: list | None = None,
    view_poses: list | None = None,
    camera: str = "",
    camera_frame: dict | None = None,
    move: bool = True,
    clear_z: float = 0.0,
    max_shift: float = MAX_SHIFT,
    min_coverage: float = MIN_COVERAGE,
    min_range: float = MIN_RANGE,
    max_ik_queries: int = MAX_IK_QUERIES,
    stride: int = 2,
    arm_id: int = 0,
) -> Output:
    """Where the hand's own camera has the container, and how far that is from where it was expected.

    Args:
        container_obb: The container as another camera has it (upright; half extents of its outer
            size; its top the rim), e.g. from finding-boxes-of-known-size-from-overhead-depth.
        other_obbs: Boxes beside it whose walls must not be taken for its rim.
        rotations: Hand rotations (wxyz) to look with, first choice first: the hand as it will be
            when it lets go. Default: none, only the hand pointing down (robot.grasp_frame).
        view_poses: Se3Poses to look from, tried in order instead of the built-in views. Each is
            sent without an IK preview; plan them wherever a reach search is cheap.
        camera / camera_frame: The wrist camera's name in the observation (default: the first
            wrist camera), or a CameraFrame; with a camera_frame nothing moves.
        move: False: look from where the hand is.
        clear_z: The empty hand rises to this height [m, world] before crossing to a view
            (0 = 11 cm over the container's rim).
        max_shift: A fit further than this from container_obb's centre is not accepted [m].
        min_range: Depth nearer than this is dropped (the hand's own fingers) [m].
        max_ik_queries: robot.solve_ik previews allowed in the search for a view.

    Returns:
        ``route`` "located" with ``located_obb``, ``shift`` (add it to anything aimed from
        container_obb) and ``aim_point`` (the located centre on the rim plane); "not_seen";
        or "unreachable".
    """
    on_arm = {"arm_id": int(arm_id)} if arm_id else {}
    centre, half, _yaw = footprint(container_obb)
    rim_z = float(container_obb["center"]["z"]) + float(container_obb["extent"]["z"])
    out: Output = {
        "route": "not_seen",
        "detail": "",
        "located_obb": None,
        "aim_point": None,
        "shift": None,
        "shift_m": 0.0,
        "yaw_deg": 0.0,
        "coverage": 0.0,
        "rim_points": 0,
        "views_tried": 0,
        "ik_queries": 0,
        "view_pose": None,
        "camera": "",
    }
    notes: list[str] = []

    def look() -> bool:
        """Read the camera and fit; True when the container was located."""
        cam = _wrist_camera(ctx, camera_frame, camera or None)
        if cam is None:
            notes.append(f"no posed wrist depth camera {camera or '(any)'!r} in the observation")
            return False
        out["camera"] = str(cam.get("name", ""))
        pts = world_points(cam, max(1, int(stride)), float(min_range))
        fit = box_in_view(pts, container_obb, other_obbs, float(min_coverage))
        if fit is None:
            notes.append("too little of the rim in view")
            return False
        shift = fit["center"] - centre
        out["coverage"], out["rim_points"] = fit["coverage"], fit["rim_points"]
        if float(np.linalg.norm(shift)) >= float(max_shift):
            notes.append(
                f"a fit {np.linalg.norm(shift) * 1e3:.0f} mm from the expected place was not "
                f"accepted (max_shift {float(max_shift) * 1e3:.0f} mm)"
            )
            return False
        out["shift"] = {"x": float(shift[0]), "y": float(shift[1]), "z": 0.0}
        out["shift_m"] = float(np.linalg.norm(shift))
        out["yaw_deg"] = fit["yaw_deg"]
        out["aim_point"] = {"x": float(fit["center"][0]), "y": float(fit["center"][1]), "z": rim_z}
        support_z = float(container_obb["center"]["z"]) - float(container_obb["extent"]["z"])
        out["located_obb"] = upright_obb(
            fit["center"], (2 * half[0], 2 * half[1]), fit["yaw_deg"], rim_z, support_z
        )
        return True

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join(d for d in (detail, *notes) if d)
        return out

    def located() -> Output:
        s = out["shift"]
        return finish(
            "located",
            f"the wrist camera has the container {s['x'] * 1e3:+.0f}, {s['y'] * 1e3:+.0f} mm "
            f"from where it was expected ({out['coverage']:.2f} of the outline seen)",
        )

    if camera_frame is not None or not move:
        out["views_tried"] = 1
        return located() if look() else finish("not_seen", "")

    start = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
    budget = int(max_ik_queries)

    def preview(xy: np.ndarray, rots: list[dict], heights: list[float]) -> Any:
        """The first rotation and height at ``xy`` that passes the IK preview, or None."""
        for rot in rots:
            for z in heights:
                if out["ik_queries"] >= budget:
                    return None
                out["ik_queries"] += 1
                pose = {"position": _vec([xy[0], xy[1], z]), "rotation": rot}
                try:
                    reply = ctx.tool("robot.solve_ik", pose=pose, **on_arm)
                except Exception:  # noqa: BLE001 - unreachable is the answer
                    continue
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    continue
                return np.array([xy[0], xy[1], z]), rot
        return None

    if view_poses:
        candidates: list[Any] = [(_xyz(_pose(v)), _pose(v)["rotation"]) for v in view_poses]
    else:
        try:
            gf = ctx.tool("robot.grasp_frame", close_heading_deg=0.0, **on_arm)
            down = [gf[k] for k in ("rotation", "rotation_flipped") if gf.get(k)]
        except Exception as exc:  # a connector without grasp_frame: the hand as it is
            logger.info("locating-a-container-from-the-wrist-camera: no grasp frame (%s)", exc)
            down = [start["rotation"]]
        candidates = view_candidates(centre, rim_z, _xyz(start)[:2], list(rotations or []), down)

    clear = float(clear_z) or rim_z + CLEAR_OVER_RIM
    refused = passed = 0
    for candidate in candidates:
        # Each view is previewed only when its turn comes: a view never gone to costs no query.
        found = candidate if view_poses else preview(*candidate)
        if found is None:
            if out["ik_queries"] >= budget:
                notes.append(f"the view search stopped at {budget} IK previews")
                break
            continue
        passed += 1
        position, rot = found
        now = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
        here = _xyz(now)
        try:
            if here[2] < clear - 1e-3:  # up first where it stands, then across
                ctx.tool(
                    "robot.go_to_pose_cartesian",
                    pose={
                        "position": _vec([here[0], here[1], clear + 0.02]),
                        "rotation": now["rotation"],
                    },
                    **on_arm,
                )
            ctx.tool(
                "robot.go_to_pose", pose={"position": _vec(position), "rotation": rot}, **on_arm
            )
        except Exception as exc:  # a refused move: the next view
            refused += 1
            notes.append(f"a view move was refused ({str(exc) or type(exc).__name__})")
            continue
        out["views_tried"] += 1
        out["view_pose"] = _pose(ctx.tool("robot.get_ee_pose", **on_arm))
        if look():
            return located()
    if passed == 0:
        return finish("unreachable", "no view pose passed the IK preview; nothing moved")
    if out["views_tried"] == 0:
        return finish("unreachable", f"all {refused} view moves were refused")
    return finish("not_seen", f"{out['views_tried']} view(s) tried")
