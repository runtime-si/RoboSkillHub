"""Calibrate one arm's base against another's through a target both arms' cameras (or one camera) see.

Each arm in turn goes through its list of joint configurations; at each the camera frame is
read, the target (a ChArUco board, or one ArUco / AprilTag marker) is posed by PnP, and the
arm's tool pose is read back. Each arm's views are solved as robot-world / hand-eye
(A X = Z B, see handeye.py); the two "shared target in my base" results compose into arm 1's
base in arm 0's base frame. Views can also be passed in (``views``), e.g. a stored tour.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handeye as he  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TARGET = {
    "type": "charuco",
    "squares": [6, 8],
    "square_m": 0.03,
    "marker_m": 0.022,
    "dictionary": "DICT_5X5_100",
}
"""The YAM station's printed board (real/calibration: 6 x 8 squares of 30 mm, 22 mm 5x5 markers)."""

MIN_VIEWS = 6
"""Views per arm with the target posed below which the arm routes tag_not_seen. Each view
adds 6 equations for 12 unknowns per arm; the station's tours used 7-14."""

MIN_CORNERS = 8
"""ChArUco corners a view needs to be posed (the station's tours saw 12-35)."""

MAX_RMS_MM = 6.0
"""Per-arm RMS of where each view puts the shared target, above which the result routes
residual_too_high. The station's own wrist tours with the plain server kinematics: 3.6-8 mm
per tour (dual_fit_report "live calibration"); the left 13:38 tour here: SKILL.md, Measured."""

MAX_RMS_DEG = 1.5
"""The same for rotation."""

MAX_JACKKNIFE_MM = 5.0
"""Leave-one-view-out spread of the base-to-base translation above which the result routes
residual_too_high: the views do not pin it down."""


class Output(TypedDict):
    route: str
    T_base0_base1: list[list[float]]
    translation_m: list[float]
    rpy_deg: list[float]
    yaw_deg: float
    tilt_deg: float
    jackknife_mm: float
    jackknife_deg: float
    arms: list[dict[str, Any]]
    reason: str


# ---------------------------------------------------------------------------
# Target detection
# ---------------------------------------------------------------------------


def _dictionary(name: str):
    import cv2

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _detector_params():
    import cv2

    params = cv2.aruco.DetectorParameters()
    # Small cells at a wrist camera's range: the station's calibrate_board.py settings.
    params.minMarkerPerimeterRate = 0.005
    params.errorCorrectionRate = 0.8
    params.perspectiveRemovePixelPerCell = 10
    params.perspectiveRemoveIgnoredMarginPerCell = 0.2
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 4
    return params


def pose_target(
    rgb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None,
    target: dict[str, Any],
    upscale: int = 2,
) -> dict[str, Any] | None:
    """``{T_cam_target, corners, reproj_px}`` for the target in one RGB frame, or None."""
    import cv2

    grey = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    if upscale > 1:
        grey = cv2.resize(grey, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    kind = target.get("type", "charuco")
    if kind == "charuco":
        board = cv2.aruco.CharucoBoard(
            tuple(int(v) for v in target["squares"]),
            float(target["square_m"]),
            float(target["marker_m"]),
            _dictionary(target["dictionary"]),
        )
        board.setLegacyPattern(bool(target.get("legacy", False)))
        det = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), _detector_params())
        corners, ids, _, _ = det.detectBoard(grey)
        if ids is None or len(ids) < MIN_CORNERS:
            return None
        img = (corners.reshape(-1, 2) + 0.5) / upscale - 0.5  # pixel centres of the upscaled copy
        obj = board.getChessboardCorners()[ids.ravel()].astype(np.float64)
    elif kind == "aruco":
        det = cv2.aruco.ArucoDetector(_dictionary(target["dictionary"]), _detector_params())
        corners, ids, _ = det.detectMarkers(grey)
        if ids is None:
            return None
        hit = [i for i, v in enumerate(ids.ravel()) if int(v) == int(target.get("id", v))]
        if not hit:
            return None
        img = (corners[hit[0]].reshape(-1, 2) + 0.5) / upscale - 0.5
        h = float(target["marker_m"]) / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    else:
        raise ValueError(f"target type must be charuco or aruco, got {kind!r}")
    flag = cv2.SOLVEPNP_IPPE_SQUARE if kind == "aruco" else cv2.SOLVEPNP_IPPE
    img = img.astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return {"T_cam_target": T, "corners": len(obj), "reproj_px": round(err, 3)}


# ---------------------------------------------------------------------------
# Robot I/O
# ---------------------------------------------------------------------------


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    p, r = pose["position"], pose["rotation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _camera(obs: dict[str, Any], name: str) -> dict[str, Any] | None:
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    for c in cams:
        if str(c.get("name")) == name and c.get("rgb") is not None:
            return c
    return None


def collect_views(
    ctx: NodeContext,
    arm_id: int,
    joint_configs: list[list[float]],
    camera: str,
    target: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Visit each configuration on ``arm_id``; one view per pose where the target was posed."""
    on_arm = {"arm_id": int(arm_id)} if int(arm_id) else {}
    views, notes = [], []
    for k, q in enumerate(joint_configs):
        try:
            why = _refusal(
                ctx.tool(
                    "robot.move_to_joints",
                    joint_config={"positions": [float(v) for v in q]},
                    **on_arm,
                )
            )
        except Exception as exc:  # a refused move skips the pose
            why = str(exc)
        if why:
            notes.append(f"arm {arm_id} pose {k}: move refused ({why})")
            continue
        tool = _matrix(ctx.tool("robot.get_ee_pose", **on_arm))
        cam = _camera(ctx.tool("robot.get_observation"), camera)
        if cam is None or cam.get("intrinsics") is None:
            notes.append(f"arm {arm_id} pose {k}: no frame with intrinsics from {camera!r}")
            continue
        seen = pose_target(np.asarray(cam["rgb"]), cam["intrinsics"], cam.get("distortion"), target)
        if seen is None:
            notes.append(f"arm {arm_id} pose {k}: target not found")
            continue
        views.append({"T_base_tool": tool, "T_cam_target": seen["T_cam_target"], "joints": list(q)})
    return views, notes


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def solve_arm(views: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    a, b = he.arm_equations(views, mode)
    x, z = he.solve_axzb(a, b)
    mm, deg = he.per_view_errors(a, b, x, z)
    return {
        "X": x,
        "Z": z,
        "rms_mm": float(np.sqrt(np.mean(np.square(mm)))),
        "rms_deg": float(np.sqrt(np.mean(np.square(deg)))),
        "per_view_mm": [round(v, 2) for v in mm],
        "per_view_deg": [round(v, 3) for v in deg],
    }


def _jackknife(
    views0: list, views1: list, mode: str, base: np.ndarray, flat: bool
) -> tuple[float, float]:
    from scipy.spatial.transform import Rotation

    worst_mm = worst_deg = 0.0
    for which, views in ((0, views0), (1, views1)):
        if len(views) <= MIN_VIEWS - 1:
            continue
        for i in range(len(views)):
            rest = views[:i] + views[i + 1 :]
            z = solve_arm(rest, mode)["Z"]
            other = solve_arm(views1 if which == 0 else views0, mode)["Z"]
            t = he.base_to_base(z, other) if which == 0 else he.base_to_base(other, z)
            if flat:
                t = he.level(t)
            worst_mm = max(worst_mm, float(np.linalg.norm(t[:3, 3] - base[:3, 3]) * 1e3))
            rel = Rotation.from_matrix(base[:3, :3].T @ t[:3, :3]).magnitude()
            worst_deg = max(worst_deg, float(np.degrees(rel)))
    return worst_mm, worst_deg


def _as_views(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for v in raw or []:
        cam_target = v.get("T_cam_target", v.get("T_cam_board"))
        out.append(
            {
                "T_base_tool": np.asarray(v["T_base_tool"], dtype=np.float64).reshape(4, 4),
                "T_cam_target": np.asarray(cam_target, dtype=np.float64).reshape(4, 4),
            }
        )
    return out


def run(
    ctx: NodeContext,
    joint_configs: dict[str, list[list[float]]] | None = None,
    views: dict[str, list[dict[str, Any]]] | None = None,
    arm_ids: list[int] | None = None,
    cameras: dict[str, str] | str | None = None,
    mode: str = "eye_in_hand",
    target: dict[str, Any] | None = None,
    assume_level: bool = False,
    min_views: int = MIN_VIEWS,
    max_rms_mm: float = MAX_RMS_MM,
    max_rms_deg: float = MAX_RMS_DEG,
    max_jackknife_mm: float = MAX_JACKKNIFE_MM,
) -> Output:
    """Arm ``arm_ids[1]``'s base in ``arm_ids[0]``'s base frame (default arms 0 and 1).

    ``joint_configs``: per arm id (as a string key, "0" / "1") the joint configurations to
    visit, chosen so the camera sees the target from varied angles. ``mode``:
    ``eye_in_hand`` (each arm's wrist camera, ``cameras[arm]``, sees one fixed target) or
    ``eye_to_hand`` (one fixed camera, ``cameras`` a name, sees a target each gripper carries
    in turn). ``views`` (per arm ``[{T_base_tool, T_cam_target}]``) skips the motion.
    """
    if mode not in ("eye_in_hand", "eye_to_hand"):
        raise ValueError(f"mode must be eye_in_hand or eye_to_hand, got {mode!r}")
    ids = [int(v) for v in (arm_ids or [0, 1])]
    if len(ids) != 2 or ids[0] == ids[1]:
        raise ValueError(f"arm_ids must name two different arms, got {arm_ids!r}")
    target = {**DEFAULT_TARGET, **(target or {})}
    notes: list[str] = []
    per_arm: list[list[dict[str, Any]]] = []
    for arm in ids:
        if views is not None:
            per_arm.append(_as_views(views.get(str(arm)) or views.get(arm) or []))  # type: ignore[call-overload]
            continue
        if not joint_configs or not (joint_configs.get(str(arm)) or joint_configs.get(arm)):  # type: ignore[call-overload]
            raise ValueError(f"no joint_configs (or views) for arm {arm}")
        configs = joint_configs.get(str(arm)) or joint_configs.get(arm)  # type: ignore[call-overload]
        if isinstance(cameras, str):
            cam = cameras
        elif cameras:
            cam = cameras.get(str(arm)) or cameras.get(arm)  # type: ignore[call-overload]
        else:
            cam = "top" if mode == "eye_to_hand" else ("left_wrist", "right_wrist")[min(arm, 1)]
        got, why = collect_views(ctx, arm, configs, str(cam), target)
        per_arm.append(got)
        notes += why

    counts = [len(v) for v in per_arm]
    empty: Output = {
        "route": "tag_not_seen",
        "T_base0_base1": [],
        "translation_m": [],
        "rpy_deg": [],
        "yaw_deg": 0.0,
        "tilt_deg": 0.0,
        "jackknife_mm": 0.0,
        "jackknife_deg": 0.0,
        "arms": [{"arm_id": a, "views": n} for a, n in zip(ids, counts, strict=True)],
        "reason": "",
    }
    short = [a for a, n in zip(ids, counts, strict=True) if n < min_views]
    if short:
        empty["reason"] = (
            f"arm(s) {short} posed the target in {counts} views (need {min_views}); "
            + "; ".join(notes[:6])
        )
        return empty

    sols = [solve_arm(v, mode) for v in per_arm]
    base = he.base_to_base(sols[0]["Z"], sols[1]["Z"])
    if assume_level:
        base = he.level(base)
    jk_mm, jk_deg = _jackknife(per_arm[0], per_arm[1], mode, base, assume_level)
    desc = he.describe(base)
    arms_out = []
    for a, s, n in zip(ids, sols, counts, strict=True):
        arms_out.append(
            {
                "arm_id": a,
                "views": n,
                "rms_mm": round(s["rms_mm"], 2),
                "rms_deg": round(s["rms_deg"], 3),
                "per_view_mm": s["per_view_mm"],
                ("T_tool_cam" if mode == "eye_in_hand" else "T_tool_target"): np.round(
                    s["X"], 6
                ).tolist(),
                ("T_base_target" if mode == "eye_in_hand" else "T_base_cam"): np.round(
                    s["Z"], 6
                ).tolist(),
            }
        )
    bad = [
        f"arm {o['arm_id']} RMS {o['rms_mm']} mm / {o['rms_deg']} deg"
        for o in arms_out
        if o["rms_mm"] > max_rms_mm or o["rms_deg"] > max_rms_deg
    ]
    if jk_mm > max_jackknife_mm:
        bad.append(f"leave-one-out moves the result {jk_mm:.1f} mm")
    route = "residual_too_high" if bad else "calibrated"
    t = desc["translation_m"]
    reason = (
        "; ".join(bad)
        if bad
        else (
            f"base {ids[1]} at ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m, yaw {desc['yaw_deg']:.2f} deg in base {ids[0]}"
        )
    )
    logger.info("calibrate_arms: %s: %s", route, reason)
    return {
        "route": route,
        "T_base0_base1": np.round(base, 6).tolist(),
        "translation_m": t,
        "rpy_deg": desc["rpy_deg"],
        "yaw_deg": desc["yaw_deg"],
        "tilt_deg": desc["tilt_deg"],
        "jackknife_mm": round(jk_mm, 2),
        "jackknife_deg": round(jk_deg, 3),
        "arms": arms_out,
        "reason": reason,
    }
