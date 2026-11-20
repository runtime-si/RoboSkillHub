"""Calibrate a wrist camera's pose in the hand from looks at one flat part that lies still: no board, no second camera."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_depth as wd  # noqa: E402

logger = logging.getLogger(__name__)

HEADINGS_DEG = (0.0, 45.0, -45.0, 90.0)
"""The hand turned about the vertical through the part by these [deg], from the pose it starts in. Turning about the
vertical is what fixes the camera's place across the hand; four headings 45 deg apart were what both YAM arms reached
over one tile (yam_dominos, 2026-11-19: 4 headings x 4 looks + pitched looks = 18-20 views)."""

RISES = (0.0, 0.06)
"""At each heading, the start height and this much higher [m]: the same heading from two ranges."""

SHIFTS = ((0.025, 0.0), (0.0, -0.025))
"""... and moved sideways by these [m] at half the rise: moving without turning is what fixes the camera's turn."""

TILTS_DEG = (15.0, 25.0, -15.0, -25.0)
"""Looks tilted about a level axis through the part [deg], at the start heading. Without them the camera's place
ALONG its line of sight is free (every upright look sees the same point at the same range)."""

MIN_VIEWS = 8
"""Fewer looks that made the part out: not fitted. Six unknowns; a look gives three (the point) plus the plane."""

SETTLE_READS = 5
SETTLE_DISTANCE = 0.0005
"""Two observations agreeing on the tool within this [m] before a frame is used (refining-target-from-wrist-closeup's)."""

MAX_POINT_MM = 2.5
"""After the fit the part's middle, seen from every look, is one point within this [mm, rms], or the exit is
``inconsistent``. A guess that the first real fit missed: the real right YAM came from 26.4 mm to 3.3 mm (worst
look 6.5 mm) and was still far better than what it had; ``inconsistent`` carries the fit so the caller can decide."""

MAX_LEFT_OUT_MM = 4.0
"""... and a look left out is predicted by the rest within this [mm, rms]. The same real fit: 4.2 mm."""

POINT_SIGMA = 0.001
PLANE_SIGMA_RAD = 0.004
"""Residual scales: the point to 1 mm, the support's normal to 0.004 rad (1 mm over 25 cm)."""


class Output(TypedDict):
    route: str
    camera_in_tool: dict[str, Any]
    point_spread_mm: float
    point_spread_before_mm: float
    left_out_mm: float
    moved_mm: float
    moved_deg: float
    views: int
    reason: str


def to_matrix(x: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:]
    return m


def from_matrix(m: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(m[:3, :3]).as_rotvec(), m[:3, 3]]


def in_world(views: list[dict[str, Any]], x: np.ndarray):
    """Each look through its tool pose and camera-in-tool pose ``x``: the part's middle [n, 3], the support's normal
    [n, 3] and the support's height under the part [n]."""
    pts, nrm, lvl = [], [], []
    for v in views:
        t = v["T_base_tool"] @ x
        p = t[:3, :3] @ v["point_cam"] + t[:3, 3]
        n = t[:3, :3] @ v["plane_n"]
        q = t[:3, :3] @ (v["plane_n"] * v["plane_d"]) + t[:3, 3]
        pts.append(p)
        nrm.append(n)
        lvl.append(q[2] + (n[0] * (q[0] - p[0]) + n[1] * (q[1] - p[1])) / n[2])
    return np.array(pts), np.array(nrm), np.array(lvl)


def fit(views: list[dict[str, Any]], x0: np.ndarray) -> np.ndarray:
    """The camera-in-tool pose for which every look has the part's middle at one point and the support as one level
    plane (soft-L1, from ``x0``)."""

    def residuals(x):
        pts, nrm, lvl = in_world(views, to_matrix(x))
        return np.r_[
            ((pts - pts.mean(0)) / POINT_SIGMA).ravel(),
            (nrm[:, :2] / PLANE_SIGMA_RAD).ravel(),
            (lvl - lvl.mean()) / POINT_SIGMA,
        ]

    sol = least_squares(
        residuals,
        from_matrix(x0),
        loss="soft_l1",
        f_scale=2.0,
        x_scale=np.r_[[0.01] * 3, [0.002] * 3],
    )
    return to_matrix(sol.x)


def spread_mm(views: list[dict[str, Any]], x: np.ndarray) -> tuple[float, float]:
    """(rms, worst) distance of the part's middle from its mean over the looks [mm]."""
    pts, _n, _l = in_world(views, x)
    dist = np.linalg.norm(pts - pts.mean(0), axis=1)
    return 1000.0 * float(np.sqrt((dist**2).mean())), 1000.0 * float(dist.max())


def left_out_mm(views: list[dict[str, Any]], x0: np.ndarray) -> float:
    """Each look predicted by the fit made without it [mm, rms]: the honest error."""
    errs = []
    for i in range(len(views)):
        rest = views[:i] + views[i + 1 :]
        x = fit(rest, x0)
        pts, _n, _l = in_world(rest, x)
        mine, _n, _l = in_world([views[i]], x)
        errs.append(float(np.linalg.norm(mine[0] - pts.mean(0))))
    return 1000.0 * float(np.sqrt(np.mean(np.square(errs))))


def differ(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.linalg.inv(a) @ b
    return 1000.0 * float(np.linalg.norm(d[:3, 3])), float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec()))
    )


def view_poses(tool: np.ndarray, part_xy: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The tool poses to look from: the start pose turned about the vertical through the part, raised, shifted, and
    tilted about a level axis through the part. Poses of the hand, not of the camera: with the calibration wrong, a
    camera pose turns into a hand pose the arm may not reach."""
    pivot = np.array([part_xy[0], part_xy[1], 0.0])

    def about(rotation: np.ndarray, pose: np.ndarray, centre: np.ndarray) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = rotation @ pose[:3, :3]
        out[:3, 3] = centre + rotation @ (pose[:3, 3] - centre)
        return out

    out = []
    for head in HEADINGS_DEG:
        turned = about(Rotation.from_euler("z", head, degrees=True).as_matrix(), tool, pivot)
        for rise in RISES:
            p = turned.copy()
            p[2, 3] += rise
            out.append((f"turn{head:+.0f}_up{1000 * rise:.0f}", p))
        for dx, dy in SHIFTS:
            p = turned.copy()
            p[:3, 3] += [dx, dy, 0.5 * RISES[-1]]
            out.append((f"turn{head:+.0f}_shift{1000 * dx:+.0f}{1000 * dy:+.0f}", p))
    level = np.cross([0.0, 0.0, 1.0], tool[:3, 3] - pivot)
    level = (
        level / np.linalg.norm(level) if np.linalg.norm(level) > 1e-6 else np.array([1.0, 0.0, 0.0])
    )
    for tilt in TILTS_DEG:
        out.append(
            (
                f"tilt{tilt:+.0f}",
                about(Rotation.from_rotvec(math.radians(tilt) * level).as_matrix(), tool, pivot),
            )
        )
    return out


def run(
    ctx: NodeContext,
    target_obb: dict[str, Any],
    views: list[dict[str, Any]] | None = None,
    poses: list[dict[str, Any]] | None = None,
    camera: str | None = None,
    arm_id: int = 0,
    support_z: float = 0.0,
    min_views: int = MIN_VIEWS,
    max_point_mm: float = MAX_POINT_MM,
    max_left_out_mm: float = MAX_LEFT_OUT_MM,
    return_to_start: bool = True,
) -> Output:
    """``target_obb``: the part (its centre to a couple of centimetres, its extent). ``views``: looks already taken
    ([{T_base_tool, point_cam, plane_n, plane_d, camera_in_tool?}]), to fit without moving. ``poses``: tool poses
    to look from instead of the built-in tour."""
    arm = {"arm_id": int(arm_id)} if arm_id else {}
    size = sorted((2.0 * float(target_obb["extent"][k]) for k in ("x", "y", "z")), reverse=True)
    part_xy = np.array([float(target_obb["center"]["x"]), float(target_obb["center"]["y"])])
    notes: list[str] = []
    x0 = None
    taken: list[dict[str, Any]] = []
    if views is not None:
        for v in views:
            taken.append(
                {
                    "T_base_tool": np.asarray(v["T_base_tool"], float),
                    "point_cam": np.asarray(v["point_cam"], float),
                    "plane_n": np.asarray(v["plane_n"], float),
                    "plane_d": float(v["plane_d"]),
                }
            )
            if x0 is None and v.get("camera_in_tool") is not None:
                x0 = np.asarray(v["camera_in_tool"], float)
    else:
        start = wd.matrix(wd.pose_of(ctx.tool("robot.get_ee_pose", **arm)))
        todo = (
            [(f"given{i}", wd.matrix(p)) for i, p in enumerate(poses)]
            if poses
            else view_poses(start, part_xy)
        )
        for name, pose in todo:
            goal = wd.pose_dict(pose)
            try:
                reply = ctx.tool("robot.solve_ik", pose=goal, **arm)
                if isinstance(reply, dict) and (
                    reply.get("success") is False or reply.get("within_servo_tolerance") is False
                ):
                    notes.append(f"{name}: out of reach")
                    continue
                why = wd.refusal(ctx.tool("robot.go_to_pose", pose=goal, **arm))
            except Exception as exc:  # a connector that raises its refusals
                why = str(exc) or type(exc).__name__
            if why:
                notes.append(f"{name}: {why}")
                continue
            obs = ctx.tool("robot.get_observation")
            for _ in range(SETTLE_READS):
                nxt = ctx.tool("robot.get_observation")
                still = (
                    np.linalg.norm(
                        wd.xyz(obs["arms"][arm_id]["ee_pose"])
                        - wd.xyz(nxt["arms"][arm_id]["ee_pose"])
                    )
                    <= SETTLE_DISTANCE
                )
                obs = nxt
                if still:
                    break
            tool = wd.matrix(obs["arms"][arm_id]["ee_pose"])
            cam = wd.wrist_camera(obs, tool, camera)
            if cam is None:
                notes.append(f"{name}: no wrist camera with depth in the observation")
                continue
            t_cam = wd.matrix(cam["pose"])
            if x0 is None:
                x0 = (
                    np.linalg.inv(tool) @ t_cam
                )  # the calibration the connector poses its frames with
            try:
                found = wd.flat_parts(cam["depth"], cam["intrinsics"], t_cam, size, support_z)
            except ValueError as exc:
                notes.append(f"{name}: {exc}")
                continue
            near = [
                p
                for p in found["parts"]
                if np.hypot(p["centre"][0] - part_xy[0], p["centre"][1] - part_xy[1]) < 0.05
            ]
            if not near:
                notes.append(f"{name}: the part not made out whole")
                continue
            best = min(
                near,
                key=lambda p: np.hypot(p["centre"][0] - part_xy[0], p["centre"][1] - part_xy[1]),
            )
            n, d = found["plane"]
            taken.append(
                {
                    "T_base_tool": tool,
                    "point_cam": np.asarray(best["point_cam"]),
                    "plane_n": np.asarray(n),
                    "plane_d": float(d),
                }
            )
        if return_to_start:
            try:
                ctx.tool("robot.go_to_pose", pose=wd.pose_dict(start), **arm)
            except Exception as exc:
                notes.append(f"return to the start pose: {exc}")
    empty = {"position": wd.vec(np.zeros(3)), "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}
    if len(taken) < int(min_views) or x0 is None:
        return {
            "route": "too_few_views",
            "camera_in_tool": empty if x0 is None else wd.pose_dict(x0),
            "point_spread_mm": 0.0,
            "point_spread_before_mm": 0.0,
            "left_out_mm": 0.0,
            "moved_mm": 0.0,
            "moved_deg": 0.0,
            "views": len(taken),
            "reason": f"{len(taken)} looks made the part out, {min_views} needed; "
            + "; ".join(notes),
        }
    x = fit(taken, x0)
    before, _ = spread_mm(taken, x0)
    after, worst = spread_mm(taken, x)
    loo = left_out_mm(taken, x0)
    mm, deg = differ(x0, x)
    good = after <= float(max_point_mm) and loo <= float(max_left_out_mm)
    return {
        "route": "calibrated" if good else "inconsistent",
        "camera_in_tool": wd.pose_dict(x),
        "T_tool_camera": np.round(x, 6).tolist(),
        "point_spread_mm": round(after, 2),
        "point_worst_mm": round(worst, 2),
        "point_spread_before_mm": round(before, 2),
        "left_out_mm": round(loo, 2),
        "moved_mm": round(mm, 1),
        "moved_deg": round(deg, 2),
        "views": len(taken),
        "reason": (
            ""
            if good
            else f"the looks agree to {after:.1f} mm (limit {max_point_mm}), left out {loo:.1f} mm "
            f"(limit {max_left_out_mm}); "
        )
        + "; ".join(notes),
    }
