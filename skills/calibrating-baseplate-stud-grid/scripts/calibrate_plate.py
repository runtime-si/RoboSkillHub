# The output follows APEX-MR's plate calibration format
# (https://github.com/intelligent-control-lab/APEX-MR,
# plate_calibration.json in its config folder for the LEGO tasks, at commit
# 6b5a17f1133c3f739caf50fa0fff6c478ef73b7a: x, y, z, roll, pitch, yaw, deg,
# width, height), MIT License. APEX-MR's values are hand-measured; the
# estimation here is new: a coarse plate pose from ArUco / AprilTag markers
# (OpenCV's aruco module and solvePnP, Apache-2.0, through cv2), a given pose,
# or a plate mask and a depth plane, refined by fitting the stud lattice in a
# wrist close-up (stud_lattice.py). The camera aiming (camera-to-tool from the
# observation, a look-down rotation, an IK preview, rise first, refusals routed,
# arrival read back, a settle wait) is adapted from the hub's
# refining-target-from-wrist-closeup (itself from RoboRSI, Apache-2.0).
"""Estimate a LEGO / DUPLO baseplate's pose: coarse from markers (or a mask and a
depth plane), refined to sub-mm by fitting the stud lattice in a wrist close-up."""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stud_lattice as lat  # noqa: E402

logger = logging.getLogger(__name__)

CAMERA_RANGE = 0.20
"""Wrist camera to plate along the optical axis [m]. At 0.20 m a 640 x 480 D405
(fx ~394 px) sees a LEGO stud ~9 px wide and ~0.33 x 0.24 m of plate; the
hub's refining-target-from-wrist-closeup found 0.15 m look-down views blocked by
its 50 mm tool clearance on a YAM."""

FALLBACK_RANGES = (0.20, 0.25, 0.15, 0.30)
"""Ranges tried after the asked one when a view is out of reach [m]."""

MIN_TOOL_CLEARANCE = 0.05
"""Lowest tool centre over the plate [m] (refining-target-from-wrist-closeup's)."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from the view pose [m]: over a real YAM station's
12 mm per-move bound; the frame is lifted with the pose actually reached."""

SETTLE_READS = 5
SETTLE_DISTANCE = 0.0005
SETTLE_ANGLE_DEG = 0.1
"""Two tool readings this close count as settled [m, deg]; ~1 px at 0.2 m."""

MAX_MOVES = 4
"""Most view moves per view point before it is given up."""

PLANE_WINDOW = 0.015
"""Close-up points within this distance of the coarse plate plane [m] feed the
plane fit. The coarse z must be better than this (markers with the depth plane:
a few mm)."""

MIN_STUDS = 12
"""Fewest lattice inliers for `calibrated`: 12 studs over a few pitches fix the
yaw to ~0.1 deg at 0.3 mm per stud."""

RESIDUAL_FRACTION = 0.125
"""Largest RMS stud-to-lattice residual, in pitches (1.0 mm LEGO, 2.0 mm DUPLO);
the renders give 0.3-0.4 mm with the depth detector."""

SNAP_TOLERANCE = 0.35
"""Largest coarse-to-lattice correction along a lattice axis, in pitches, before
the snap counts as ambiguous: at half a pitch either neighbour fits equally."""

MAX_YAW_CORRECTION_DEG = 10.0
"""Largest yaw correction trusted [deg]: a coarse pose is better than this, and
the lattice repeats every 90 deg."""

MASK_PLANE_THRESHOLD = 0.004
"""Inlier band of the plate-mask depth plane [m] (same camera noise; studs are
1.7 mm on LEGO and fall inside it)."""

ROUTES = ("calibrated", "coarse_only", "no_lattice", "no_markers")


class Output(TypedDict):
    route: str
    detail: str
    plate_pose: dict[str, Any]
    coarse_plate_pose: dict[str, Any]
    coarse_source: str
    markers_used: list[int]
    marker_reprojection_px: float
    yaw_ambiguity_deg: float
    lattice_status: str
    correction: dict[str, float]
    residual_mm: float
    max_residual_mm: float
    studs_used: int
    studs_detected: int
    snap_margin: float
    plate_studs: list[int]
    system: str
    stud_pitch: float
    apex_plate: dict[str, float]
    views: list[dict[str, Any]]
    camera_name: str


# -- poses and cameras ----------------------------------------------------------------


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cos)))))


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1.0e-9:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1.0e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or ""."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _is_wrist(cam: dict[str, Any]) -> bool:
    return any(k in str(cam.get("name", "")) for k in ("eye_in_hand", "wrist"))


def _cameras(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        c
        for c in obs.get("cameras") or []
        if c.get("pose") is not None and c.get("intrinsics") is not None
    ]


def _wrist_camera(obs: dict[str, Any], arm_id: int, name: str | None) -> dict[str, Any] | None:
    cams = _cameras(obs)
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if _is_wrist(c)]
    if not wrist:
        return None
    arms = obs.get("arms") or []
    if len(arms) > arm_id and arms[arm_id].get("ee_pose") is not None:
        tool = lat.xyz(arms[arm_id]["ee_pose"])
        return min(wrist, key=lambda c: float(np.linalg.norm(lat.xyz(c["pose"]) - tool)))
    return wrist[0]


def _settled(ctx: NodeContext, arm_id: int) -> dict[str, Any]:
    obs = ctx.tool("robot.get_observation")
    for _ in range(SETTLE_READS):
        nxt = ctx.tool("robot.get_observation")
        a = lat.matrix(obs["arms"][arm_id]["ee_pose"])
        b = lat.matrix(nxt["arms"][arm_id]["ee_pose"])
        obs = nxt
        if (
            float(np.linalg.norm(a[:3, 3] - b[:3, 3])) <= SETTLE_DISTANCE
            and _angle_deg(a[:3, :3], b[:3, :3]) <= SETTLE_ANGLE_DEG
        ):
            break
    else:
        logger.info("calibrating-baseplate-stud-grid: the hand was still moving after the reads")
    return obs


def _apex(m: np.ndarray, studs: list[int]) -> dict[str, float]:
    """APEX-MR's plate entry; roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r = m[:3, :3]
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    return {
        "x": float(m[0, 3]),
        "y": float(m[1, 3]),
        "z": float(m[2, 3]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "deg": math.degrees(yaw),
        "width": int(studs[0]),
        "height": int(studs[1]),
    }


# -- coarse pose ----------------------------------------------------------------------


def _marker_layout(markers: list[dict[str, Any]]) -> dict[int, tuple[float, np.ndarray]]:
    """id -> (size, marker-to-plate 4x4). A marker is {id, size, pose} or
    {id, size, position, yaw_deg} (face up, OpenCV's marker axes)."""
    out: dict[int, tuple[float, np.ndarray]] = {}
    for m in markers:
        try:
            mid, size = int(m["id"]), float(m["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"a marker needs id and size [m], got {m!r}") from exc
        if size <= 0.0:
            raise ValueError(f"marker {mid}: size must be positive")
        if m.get("pose") is not None:
            t = lat.matrix(m["pose"])
        else:
            t = np.eye(4)
            t[:3, 3] = lat.xyz(m.get("position", [0.0, 0.0, 0.0]))
            a = math.radians(float(m.get("yaw_deg", 0.0)))
            t[:2, :2] = lat.rz(a)
        out[mid] = (size, t)
    return out


def _marker_corners_plate(size: float, t_plate_marker: np.ndarray) -> np.ndarray:
    """OpenCV's corner order: top-left, top-right, bottom-right, bottom-left
    (marker x right, y up, z out of the face)."""
    h = size / 2.0
    c = np.array([[-h, h, 0.0, 1.0], [h, h, 0.0, 1.0], [h, -h, 0.0, 1.0], [-h, -h, 0.0, 1.0]])
    return (c @ t_plate_marker.T)[:, :3]


def _detect_markers(gray: np.ndarray, dictionary: str) -> tuple[list[np.ndarray], list[int]]:
    name = dictionary if dictionary.startswith("DICT_") else f"DICT_{dictionary}"
    if not hasattr(cv2.aruco, name):
        raise ValueError(
            f"unknown marker dictionary {dictionary!r} (e.g. DICT_4X4_50, DICT_APRILTAG_36h11)"
        )
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params
    )
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return [], []
    return [c.reshape(4, 2).astype(np.float64) for c in corners], [int(i) for i in ids.ravel()]


def _coarse_from_markers(
    cams: list[dict[str, Any]],
    layout: dict[int, tuple[float, np.ndarray]],
    dictionary: str,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for cam in cams:
        if cam.get("rgb") is None:
            continue
        corners, ids = _detect_markers(lat.as_gray(cam["rgb"]), dictionary)
        hits = [(c, i) for c, i in zip(corners, ids, strict=True) if i in layout]
        if hits and (best is None or len(hits) > len(best["hits"])):
            best = {"cam": cam, "hits": hits, "seen": ids}
    if best is None:
        return None
    cam, hits = best["cam"], best["hits"]
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    dist = np.asarray(cam.get("distortion", np.zeros(5)), dtype=np.float64)
    obj = np.concatenate([_marker_corners_plate(*layout[i]) for _, i in hits])
    img = np.concatenate([c for c, _ in hits])
    t_world_cam = lat.matrix(cam["pose"])
    # Candidates: SQPnP over every corner, and both IPPE_SQUARE poses of each
    # marker (a small marker's planar ambiguity: two poses ~36 deg apart that
    # reproject almost equally well in the renders), each refined by LM on all
    # corners. (OpenCV 5's general IPPE flipped the plate for corners off its
    # z = 0 plane, so it is not used.) Of those within twice the best RMS
    # (+0.5 px), the one whose plate z is nearest world +z: plates lie on tables.
    starts: list[tuple[np.ndarray, np.ndarray]] = []
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        starts.append((rvec, tvec))
    for c, i in hits:
        size, t_plate_marker = layout[i]
        h = size / 2.0
        square = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            square, c, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        for r_m, t_m in zip(rvecs[:n_sol], tvecs[:n_sol], strict=True):
            t_cam_marker = np.eye(4)
            t_cam_marker[:3, :3] = cv2.Rodrigues(r_m)[0]
            t_cam_marker[:3, 3] = t_m.ravel()
            t_cp = t_cam_marker @ np.linalg.inv(t_plate_marker)
            starts.append((cv2.Rodrigues(t_cp[:3, :3])[0], t_cp[:3, 3].reshape(3, 1).copy()))
    if not starts:
        return None
    sols = []
    for rvec, tvec in starts:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, k, dist, rvec.copy(), tvec.copy())
        proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        r_world_plate = t_world_cam[:3, :3] @ cv2.Rodrigues(rvec)[0]
        if float(tvec.ravel()[2]) > 0.0:
            sols.append((rms, -float(r_world_plate[2, 2]), rvec, tvec))
    if not sols:
        return None
    best_rms = min(s[0] for s in sols)
    reproj, _, rvec, tvec = min(
        (s for s in sols if s[0] <= 2.0 * best_rms + 0.5), key=lambda s: s[1]
    )
    t_cam_plate = np.eye(4)
    t_cam_plate[:3, :3] = cv2.Rodrigues(rvec)[0]
    t_cam_plate[:3, 3] = tvec.ravel()
    # PnP only: a depth plane fitted inside two 40 mm quads at 0.9 m tilted the
    # renders' plate by 8 deg (PnP alone: 2 deg); the close-up refits tilt and height.
    return {
        "pose": t_world_cam @ t_cam_plate,
        "camera": str(cam.get("name", "")),
        "ids": sorted(i for _, i in hits),
        "reprojection_px": reproj,
    }


def _coarse_from_mask(
    cam: dict[str, Any],
    mask: np.ndarray,
    studs: list[int],
    pitch: float,
    yaw_hint_deg: float,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Plate top from the masked depth (a plane), its outline's min-area
    rectangle for the centre and the yaw (sides matched to the stud counts)."""
    depth = lat.as_depth(cam["depth"])
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    t_world_cam = lat.matrix(cam["pose"])
    pts, _ = lat.backproject(depth, k, t_world_cam, mask=mask, zmin=0.05, zmax=5.0)
    fit = lat.fit_plane(pts, MASK_PLANE_THRESHOLD, rng, max_tilt_deg=30.0)
    if fit is None:
        return None
    n, d, inl = fit
    frame = lat.plane_frame(n, d, np.array([1.0, 0.0, 0.0]), pts[inl].mean(axis=0))
    local = (pts[inl] - frame[:3, 3]) @ frame[:3, :3]
    (cx, cy), (w, h), angle = cv2.minAreaRect(local[:, :2].astype(np.float32))
    a = math.radians(angle)
    lx, ly = studs[0] * pitch, studs[1] * pitch
    # Which rectangle side is the plate's x (its first stud count).
    if abs(w - lx) + abs(h - ly) <= abs(h - lx) + abs(w - ly):
        yaw, side_x, side_y = a, w, h
    else:
        yaw, side_x, side_y = a + math.pi / 2.0, h, w
    period = math.pi / 2.0 if studs[0] == studs[1] else math.pi
    hint = math.radians(yaw_hint_deg)
    yaw_world = math.atan2(*(frame[:3, :2] @ np.array([math.cos(yaw), math.sin(yaw)]))[[1, 0]])
    yaw += (hint - yaw_world) + lat.wrap(yaw_world - hint, period)
    m = np.eye(4)
    m[:2, :2] = lat.rz(yaw)
    m[:2, 3] = (cx, cy)
    return {
        "pose": frame @ m,
        "camera": str(cam.get("name", "")),
        "size": (float(side_x), float(side_y)),
        "expected": (lx, ly),
        "period_deg": math.degrees(period),
        "points": int(np.count_nonzero(inl)),
    }


# -- the close-up ---------------------------------------------------------------------


def _look(
    ctx: NodeContext,
    obs: dict[str, Any],
    cam_name: str,
    arm_id: int,
    target: np.ndarray,
    up: np.ndarray,
    camera_range: float,
    check_reach: bool,
    move: dict[str, Any],
    reach_tolerance: float,
    risen: list[bool],
) -> tuple[dict[str, Any] | None, dict[str, Any], float, list[str]]:
    """Place the wrist camera on the plate normal through ``target``, looking
    down it, and return the settled frame."""
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    tool_now = lat.matrix(obs["arms"][arm_id]["ee_pose"])
    cam = _wrist_camera(obs, arm_id, cam_name)
    tool_cam = np.linalg.inv(tool_now) @ lat.matrix(cam["pose"])
    axis_tool, offset_tool = tool_cam[:3, 2], tool_cam[:3, 3]
    bases = [tool_now[:3, :3]]
    try:
        frame = ctx.tool("robot.grasp_frame", **arm)
        bases += [
            lat.rot(frame[k])
            for k in ("rotation", "rotation_flipped")
            if isinstance(frame, dict) and frame.get(k)
        ]
    except Exception as exc:
        logger.info("calibrating-baseplate-stud-grid: no grasp frame (%s)", exc)
    rotations: list[np.ndarray] = []
    for base in bases:
        r = _align(base @ axis_tool, -up) @ base
        if all(_angle_deg(r, seen) > 2.0 for seen in rotations):
            rotations.append(r)
    ranges = list(dict.fromkeys([float(camera_range), *FALLBACK_RANGES]))
    failures: list[str] = []
    moves = 0
    for r_tool in rotations:
        for rng_m in ranges:
            if moves >= MAX_MOVES:
                break
            eye = target + rng_m * up
            position = eye - r_tool @ offset_tool
            if float((position - target) @ up) < MIN_TOOL_CLEARANCE:
                failures.append(
                    f"range {rng_m:.2f} m: the hand would come within 50 mm of the plate"
                )
                continue
            pose = {"position": lat.vec(position), "rotation": lat.quat(r_tool)}
            if check_reach:
                try:
                    reply = ctx.tool("robot.solve_ik", pose=pose, **arm)
                except Exception as exc:
                    failures.append(f"range {rng_m:.2f} m: IK preview refused ({exc})")
                    continue
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    failures.append(f"range {rng_m:.2f} m: out of reach (IK preview)")
                    continue
            moves += 1
            why = ""
            if not risen[0]:
                start = _pose(ctx.tool("robot.get_ee_pose", **arm))
                if float(start["position"]["z"]) < position[2] - 1.0e-3:
                    rise = {
                        "position": {**start["position"], "z": float(position[2])},
                        "rotation": start["rotation"],
                    }
                    try:
                        why = _refusal(ctx.tool("robot.go_to_pose", pose=rise, **move))
                    except Exception as exc:
                        why = str(exc)
                    why = f"rise: {why}" if why else ""
                risen[0] = not why
            if not why:
                try:
                    why = _refusal(ctx.tool("robot.go_to_pose", pose=pose, **move))
                except Exception as exc:
                    why = str(exc)
            if why:
                failures.append(f"range {rng_m:.2f} m: {why}")
                continue
            off = float(
                np.linalg.norm(lat.xyz(_pose(ctx.tool("robot.get_ee_pose", **arm))) - position)
            )
            if off > float(reach_tolerance):
                failures.append(
                    f"range {rng_m:.2f} m: the hand ended {off * 1e3:.0f} mm from the view pose"
                )
                continue
            seen_obs = _settled(ctx, arm_id)
            seen = _wrist_camera(seen_obs, arm_id, cam_name)
            if seen is None or seen.get("depth") is None:
                failures.append(f"range {rng_m:.2f} m: the wrist camera gave no depth frame")
                continue
            return seen, pose, rng_m, failures
    return None, {}, 0.0, failures


def run(
    ctx: NodeContext,
    plate_studs: list[int],
    system: str = "lego",
    markers: list[dict[str, Any]] | None = None,
    marker_dictionary: str = "DICT_4X4_50",
    coarse_plate_pose: dict[str, Any] | None = None,
    plate_mask: Any = None,
    plate_prompt: str | None = None,
    coarse_camera: str | None = None,
    wrist_camera: str | None = None,
    refine: bool = True,
    look: bool = True,
    view_points: list[list[float]] | None = None,
    camera_range: float = CAMERA_RANGE,
    detector: str = "both",
    stud_pitch: float | None = None,
    stud_height: float | None = None,
    stud_diameter: float | None = None,
    max_residual: float | None = None,
    min_studs: int = MIN_STUDS,
    snap_tolerance: float = SNAP_TOLERANCE,
    max_yaw_correction_deg: float = MAX_YAW_CORRECTION_DEG,
    plane_window: float = PLANE_WINDOW,
    yaw_hint_deg: float = 0.0,
    check_reach: bool = True,
    arm_id: int = 0,
    tolerance: float | None = None,
    max_steps: int | None = None,
    reach_tolerance: float = REACH_TOLERANCE,
    seed: int = 0,
) -> Output:
    if system not in lat.PRESETS:
        raise ValueError(f"system must be one of {sorted(lat.PRESETS)}, got {system!r}")
    if detector not in ("both", "depth", "rgb"):
        raise ValueError(f"detector must be 'both', 'depth' or 'rgb', got {detector!r}")
    studs = [int(v) for v in plate_studs]
    if len(studs) != 2 or min(studs) < 1:
        raise ValueError(f"plate_studs must be two positive stud counts, got {plate_studs!r}")
    preset = lat.PRESETS[system]
    pitch = float(stud_pitch if stud_pitch is not None else preset["stud_pitch"])
    stud_h = float(stud_height if stud_height is not None else preset["stud_height"])
    stud_r = float(stud_diameter if stud_diameter is not None else preset["stud_diameter"]) / 2.0
    max_res = float(max_residual if max_residual is not None else RESIDUAL_FRACTION * pitch)
    layout = _marker_layout(markers or [])
    rng = np.random.default_rng(seed)
    move: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    if tolerance is not None:
        move["tolerance"] = float(tolerance)
    if max_steps is not None:
        move["max_steps"] = int(max_steps)

    out: Output = {
        "route": "no_markers",
        "detail": "",
        "plate_pose": {},
        "coarse_plate_pose": {},
        "coarse_source": "",
        "markers_used": [],
        "marker_reprojection_px": 0.0,
        "yaw_ambiguity_deg": 0.0,
        "lattice_status": "not_run",
        "correction": {},
        "residual_mm": 0.0,
        "max_residual_mm": 0.0,
        "studs_used": 0,
        "studs_detected": 0,
        "snap_margin": 0.0,
        "plate_studs": studs,
        "system": system,
        "stud_pitch": pitch,
        "apex_plate": {},
        "views": [],
        "camera_name": "",
    }
    notes: list[str] = []

    def finish(route: str, why: str) -> Output:
        out["route"] = route
        out["detail"] = "; ".join(s for s in (why, *notes) if s)
        logger.info("calibrating-baseplate-stud-grid: %s (%s)", route, out["detail"])
        return out

    # 1. The coarse plate pose.
    obs = ctx.tool("robot.get_observation")
    cams = _cameras(obs)
    if coarse_camera:
        coarse_cams = [c for c in cams if c.get("name") == coarse_camera]
    else:
        coarse_cams = cams
    t_coarse = None
    if layout:
        found = _coarse_from_markers(coarse_cams, layout, marker_dictionary)
        if found is not None:
            t_coarse = found["pose"]
            out["coarse_source"] = "markers"
            out["markers_used"] = found["ids"]
            out["marker_reprojection_px"] = found["reprojection_px"]
            notes.append(
                f"coarse from markers {found['ids']} in {found['camera']!r} "
                f"({found['reprojection_px']:.2f} px RMS)"
            )
        else:
            notes.append(f"none of markers {sorted(layout)} seen")
    if t_coarse is None and coarse_plate_pose:
        t_coarse = lat.matrix(coarse_plate_pose)
        out["coarse_source"] = "input"
    if t_coarse is None and (plate_mask is not None or plate_prompt):
        pool = coarse_cams if coarse_camera else ([c for c in cams if not _is_wrist(c)] or cams)
        cam = next((c for c in pool if c.get("depth") is not None), None)
        if cam is not None:
            mask = None
            if plate_mask is not None:
                mask = np.asarray(plate_mask)
            else:
                try:
                    seg = ctx.tool(
                        "sam3.segment_text", image=cam["rgb"], query=plate_prompt, max_results=1
                    )
                    if isinstance(seg, dict) and seg.get("masks"):
                        mask = np.asarray(seg["masks"][0])
                except Exception as exc:
                    notes.append(f"sam3.segment_text failed: {exc}")
            if mask is not None and mask.ndim == 3:
                mask = mask[..., 0]
            found = (
                _coarse_from_mask(cam, mask > 0, studs, pitch, yaw_hint_deg, rng)
                if mask is not None and np.any(mask)
                else None
            )
            if found is not None:
                t_coarse = found["pose"]
                out["coarse_source"] = "mask_plane"
                out["yaw_ambiguity_deg"] = found["period_deg"]
                sx, sy = found["size"]
                ex, ey = found["expected"]
                notes.append(
                    f"coarse from the plate mask in {found['camera']!r}: "
                    f"{sx * 1e3:.0f} x {sy * 1e3:.0f} mm against {ex * 1e3:.0f} x {ey * 1e3:.0f}; "
                    f"yaw known modulo {found['period_deg']:.0f} deg"
                )
                if abs(sx - ex) > 0.15 * ex or abs(sy - ey) > 0.15 * ey:
                    notes.append("the mask's outline is more than 15 % off the plate's size")
            else:
                notes.append("no plate plane in the mask")
        else:
            notes.append("no camera with depth for the plate mask")
    if t_coarse is None:
        return finish(
            "no_markers",
            "no coarse plate pose: no listed marker seen, no coarse_plate_pose, no usable plate mask",
        )
    out["coarse_plate_pose"] = lat.se3(t_coarse)
    out["plate_pose"] = lat.se3(t_coarse)
    out["apex_plate"] = _apex(t_coarse, studs)
    if not refine:
        out["lattice_status"] = "not_requested"
        return finish("coarse_only", "refine is off")

    # 2. The close-up frame(s).
    frames: list[dict[str, Any]] = []
    cam0 = _wrist_camera(obs, int(arm_id), wrist_camera)
    if cam0 is None:
        names = [c.get("name") for c in cams]
        out["lattice_status"] = "no_wrist_camera"
        return finish(
            "coarse_only", f"no wrist camera with a pose in the observation (cameras: {names})"
        )
    out["camera_name"] = str(cam0.get("name", ""))
    if look:
        up = t_coarse[:3, 2]
        risen = [False]
        failures: list[str] = []
        for vp in view_points or [[0.0, 0.0]]:
            target = (t_coarse @ np.array([float(vp[0]), float(vp[1]), 0.0, 1.0]))[:3]
            seen, pose, rng_m, why = _look(
                ctx,
                obs,
                out["camera_name"],
                int(arm_id),
                target,
                up,
                camera_range,
                check_reach,
                move,
                reach_tolerance,
                risen,
            )
            failures += [f"view ({vp[0]:+.3f}, {vp[1]:+.3f}): {w}" for w in why]
            if seen is not None:
                frames.append(seen)
                out["views"].append(
                    {
                        "view_point": [float(vp[0]), float(vp[1])],
                        "camera_range": rng_m,
                        "view_pose": pose,
                    }
                )
        if not frames:
            out["lattice_status"] = "unreachable"
            return finish("coarse_only", "no close-up view reached: " + "; ".join(failures[:6]))
    else:
        if cam0.get("depth") is None:
            out["lattice_status"] = "no_wrist_camera"
            return finish("coarse_only", f"the wrist camera {out['camera_name']!r} has no depth")
        frames.append(cam0)
        out["views"].append({"view_point": [], "camera_range": 0.0, "view_pose": {}})

    # 3. The plate plane from the close-up depth, in the coarse plate frame.
    inv_coarse = np.linalg.inv(t_coarse)
    # The plane from points over the plate only: a margin round a small plate
    # lets the table (a plate's thickness under it) outvote it.
    half = np.array([studs[0] * pitch / 2.0, studs[1] * pitch / 2.0])
    pooled = []
    per_frame = []
    for f in frames:
        k = np.asarray(f["intrinsics"], dtype=np.float64)
        depth = lat.as_depth(f["depth"])
        t_c_cam = inv_coarse @ lat.matrix(f["pose"])
        pts, _ = lat.backproject(depth, k, t_c_cam, stride=2)
        near = (np.abs(pts[:, 2]) < plane_window) & np.all(np.abs(pts[:, :2]) <= half, axis=1)
        pooled.append(pts[near])
        per_frame.append((f, k, depth, t_c_cam))
    pts_all = np.concatenate(pooled) if pooled else np.zeros((0, 3))
    plane = lat.stepped_plane(
        pts_all, max(0.35 * stud_h, 0.0005), rng, drop=0.6 * preset["brick_height"]
    )
    if plane is None:
        out["lattice_status"] = "no_plane"
        return finish(
            "no_lattice",
            f"no plate plane within {plane_window * 1e3:.0f} mm of the coarse pose in the close-up "
            f"({len(pts_all)} depth points)",
        )
    n, d, inl = plane
    t_c_b = lat.plane_frame(n, d, np.array([1.0, 0.0, 0.0]), np.zeros(3))
    inv_b = np.linalg.inv(t_c_b)

    # 4. Stud candidates in each frame, pooled in the plane frame.
    cands = []
    n_dep = n_rgb = 0
    plane_pts_b = (pts_all[inl] - t_c_b[:3, 3]) @ t_c_b[:3, :3]
    for f, k, depth, t_c_cam in per_frame:
        t_b_cam = inv_b @ t_c_cam
        mine, _ = lat.backproject(depth, k, t_b_cam, stride=4)
        mine = mine[np.abs(mine[:, 2]) < 2.0 * stud_h]
        if len(mine) < 30:
            continue
        lo = np.maximum(np.percentile(mine[:, :2], 0.5, axis=0), -half - pitch / 2.0)
        hi = np.minimum(np.percentile(mine[:, :2], 99.5, axis=0), half + pitch / 2.0)
        if np.any(hi - lo < 2 * pitch):
            continue
        raster = lat.Raster(lo[0], lo[1], hi[0], hi[1], pitch / lat.RECT_PX_PER_PITCH)
        found = lat.detect_and_fit(
            f.get("rgb"), depth, k, t_b_cam, raster, stud_h, stud_r, pitch, detector
        )
        cands.append(found["xy"])
        n_dep += found["n_depth"]
        n_rgb += found["n_rgb"]
    q = np.concatenate(cands) if cands else np.zeros((0, 2))
    out["studs_detected"] = len(q)
    phase = (0.5 if studs[0] % 2 == 0 else 0.0, 0.5 if studs[1] % 2 == 0 else 0.0)
    fit = lat.fit_lattice(q, pitch, rng, phase=phase, yaw_prior=0.0) if len(q) else None
    if fit is None or fit["n_inliers"] < int(min_studs):
        got = 0 if fit is None else fit["n_inliers"]
        out["studs_used"] = got
        out["lattice_status"] = "too_few_studs"
        return finish(
            "no_lattice",
            f"{got} studs on a {pitch * 1e3:.0f} mm lattice ({len(q)} candidates: {n_dep} depth, "
            f"{n_rgb} RGB; need {min_studs})",
        )
    yaw = fit["yaw"]
    centre = q[fit["inliers"]].mean(axis=0)
    t, corr, margin = lat.minimal_shift(yaw, fit["t"], pitch, centre)
    out["studs_used"] = fit["n_inliers"]
    out["residual_mm"] = fit["rms"] * 1e3
    out["max_residual_mm"] = fit["max_residual"] * 1e3
    out["snap_margin"] = margin
    t_b_p = np.eye(4)
    t_b_p[:2, :2] = lat.rz(yaw)
    t_b_p[:2, 3] = t
    t_refined = t_coarse @ t_c_b @ t_b_p
    t_corr = inv_coarse @ t_refined
    out["correction"] = {
        "x": float(t_corr[0, 3]),
        "y": float(t_corr[1, 3]),
        "z": float(t_corr[2, 3]),
        "yaw_deg": math.degrees(math.atan2(float(t_corr[1, 0]), float(t_corr[0, 0]))),
        "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(t_corr[2, 2]))))),
        "lateral": float(math.hypot(t_corr[0, 3], t_corr[1, 3])),
        "at_view_x": float(corr[0]),
        "at_view_y": float(corr[1]),
    }
    summary = (
        f"{fit['n_inliers']} of {len(q)} stud candidates on the lattice "
        f"({fit['rms'] * 1e3:.2f} mm RMS, {len(plane_pts_b)} plate points); correction "
        f"{corr[0] * 1e3:+.2f} / {corr[1] * 1e3:+.2f} mm at the view, "
        f"{math.degrees(yaw):+.2f} deg"
    )
    if fit["rms"] > max_res:
        out["lattice_status"] = "high_residual"
        return finish("no_lattice", f"{summary}; residual over {max_res * 1e3:.2f} mm")
    if abs(math.degrees(yaw)) > float(max_yaw_correction_deg):
        out["lattice_status"] = "yaw_off"
        return finish(
            "no_lattice",
            f"{summary}; the yaw correction is over {max_yaw_correction_deg:.0f} deg (the coarse yaw is off, "
            "or the lattice is another one)",
        )
    if margin > float(snap_tolerance):
        out["lattice_status"] = "ambiguous"
        return finish(
            "no_lattice",
            f"{summary}; the coarse pose is {margin:.2f} pitch off the nearest stud row "
            f"(over {snap_tolerance:.2f}): the neighbouring row fits as well",
        )
    out["lattice_status"] = "ok"
    out["plate_pose"] = lat.se3(t_refined)
    out["apex_plate"] = _apex(t_refined, studs)
    return finish("calibrated", summary)
