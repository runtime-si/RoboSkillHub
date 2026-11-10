"""Scan a table region with the wrist camera and build one top-down image of it at a chosen resolution.

Written for the hub. Each frame is mapped onto the region's plane by the homography its camera pose and
intrinsics induce (the plane-to-image projection K [r1 r2 t] of multiple-view geometry, Hartley and
Zisserman 2004, eq. 8.1; OpenCV's warpPerspective does the resampling) and the frames are blended with
weights falling to zero at their borders, as a rectified-mosaic stitcher such as OpenStitching
(https://github.com/OpenStitching/stitching, Apache-2.0) does after its own feature-based registration;
here the arm's poses stand in for that registration, and the overlaps are only checked by phase
correlation. No code copied.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

OVERLAP = 0.25
"""Fraction of a frame's footprint shared with each neighbouring frame: enough for the seam check and
for feathering, and for a pose error of a few mm not to leave a gap."""

MIN_HEIGHT_M = 0.10
MAX_HEIGHT_M = 0.30
"""Camera height over the surface is chosen for the asked resolution within these. The D405's depth
starts at 7 cm and is best at 10-30 cm (Intel's datasheet range for the D405); colour has no lower bound,
but under 10 cm the hand comes within a few cm of placed pieces."""

USE_FRACTION = 0.9
"""Central fraction of each frame (by width and height) counted as its footprint: lens edges are softer."""

MAX_TILES = 64
"""Refuse scans that need more frames than this (choose a lower resolution or a smaller region)."""

SETTLE_STEPS = 30
"""Simulation steps waited after each move where the connector offers robot.wait_steps (motion blur)."""

SEAM_MIN_OVERLAP_PX = 32

VIEWS = (
    (0.0, 0.0),
    (180.0, 0.0),
    (90.0, 0.0),
    (-90.0, 0.0),
    (0.0, 15.0),
    (180.0, 15.0),
    (0.0, 30.0),
    (180.0, 30.0),
)
"""(roll, tilt) degrees tried in order for each frame: straight down first; tilted views when the arm
cannot look straight down there (the station twin could not at 10 cm over the board)."""

AIM_TOL_M = 0.004
AIM_STEPS = 2
"""After a move, where the camera's axis meets the surface is compared with the frame's centre and the
hand is shifted to close the gap, up to AIM_STEPS times, until within AIM_TOL_M. A connector whose IK
leaves the wrist roll free (the station twin) turns the camera, which sits off the tool axis, away."""

FILL_PASSES = 2
"""Extra passes aiming new frames at the largest uncovered patches when coverage is under 98 %."""
"""Overlaps narrower than this (in mosaic pixels) are not used for the seam check."""


class Output(TypedDict, total=False):
    route: str
    mosaic: Any
    height_map: Any
    metres_per_pixel: float
    origin: dict[str, float]
    size_px: list[int]
    board_corners_px: list[list[float]]
    tiles: int
    tiles_seen: int
    covered_fraction: float
    camera_height_m: float
    seam_shift_px: float
    seam_shift_max_px: float
    detail: str


# --- pose helpers (same conventions as refining-target-from-wrist-closeup) -----------------------------


def _pose(reply: Any) -> dict:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _rot(q: dict) -> np.ndarray:
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


def _quat(r: np.ndarray) -> dict:
    t = float(np.trace(r))
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return dict(zip(("w", "x", "y", "z"), (float(v) for v in q), strict=True))


def _matrix(pose: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    p = pose["position"]
    m[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return m


def _to_pose(m: np.ndarray) -> dict:
    return {
        "position": dict(zip("xyz", (float(v) for v in m[:3, 3]), strict=True)),
        "rotation": _quat(m[:3, :3]),
    }


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or "the move reported failure")
    return ""


def _camera(obs: dict, name: str, tool: np.ndarray | None) -> dict | None:
    cams = [
        c
        for c in obs.get("cameras") or []
        if c.get("pose") is not None and c.get("rgb") is not None
    ]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    if not wrist:
        return None
    if tool is None:
        return wrist[0]
    return min(wrist, key=lambda c: float(np.linalg.norm(_matrix(c["pose"])[:3, 3] - tool[:3, 3])))


def _rgb(img: Any) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a[..., :3])


# --- geometry -----------------------------------------------------------------------------------------


def plane_to_mosaic(
    center: np.ndarray, yaw_deg: float, size_px: tuple[int, int], mpp: float
) -> np.ndarray:
    """3x3 map from plane coordinates (world x, y on the region's plane) to mosaic pixels: the region seen
    from above, its +x axis to the right, its +y axis up, centred."""
    a = math.radians(yaw_deg)
    xb, yb = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    w, h = size_px
    # u = W/2 + (p - c) . xb / mpp ; v = H/2 - (p - c) . yb / mpp
    m = np.array(
        [
            [xb[0] / mpp, xb[1] / mpp, w / 2 - (center[:2] @ xb) / mpp],
            [-yb[0] / mpp, -yb[1] / mpp, h / 2 + (center[:2] @ yb) / mpp],
            [0.0, 0.0, 1.0],
        ]
    )
    return m


def plane_to_image(k: np.ndarray, cam_to_world: np.ndarray, z0: float) -> np.ndarray:
    """3x3 homography from (x, y) on the plane z = z0 (world) to image pixels: K [r1 r2 r3 * z0 + t]."""
    w2c = np.linalg.inv(cam_to_world)
    r, t = w2c[:3, :3], w2c[:3, 3]
    return k @ np.column_stack([r[:, 0], r[:, 1], r[:, 2] * z0 + t])


def plan_tiles(
    center: np.ndarray, size: tuple[float, float], yaw_deg: float, footprint: tuple[float, float]
) -> list[np.ndarray]:
    """Frame centres (world x, y) covering the region with OVERLAP between neighbours, in a serpentine."""
    a = math.radians(yaw_deg)
    xb, yb = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    out = []
    counts = []
    for extent, fp in zip(size, footprint, strict=True):
        step = fp * (1.0 - OVERLAP)
        n = max(1, math.ceil(max(0.0, extent - fp) / step) + 1)
        counts.append(n)
    nx, ny = counts
    for j in range(ny):
        ys = 0.0 if ny == 1 else (j / (ny - 1) - 0.5) * max(0.0, size[1] - footprint[1])
        cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
        for i in cols:
            xs = 0.0 if nx == 1 else (i / (nx - 1) - 0.5) * max(0.0, size[0] - footprint[0])
            out.append(center[:2] + xs * xb + ys * yb)
    return out


def _downward(yaw_deg: float, roll_deg: float = 0.0, tilt_deg: float = 0.0) -> np.ndarray:
    """Camera rotation looking down, image +x along the region's +x turned by roll_deg, then tilted
    tilt_deg about the image x axis (the view leans along the image's y)."""
    a = math.radians(yaw_deg + roll_deg)
    x = np.array([math.cos(a), math.sin(a), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    r = np.column_stack([x, y, z])
    t = math.radians(tilt_deg)
    tilt = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]]
    )
    return r @ tilt


def _axis_hit(cam: np.ndarray, z0: float) -> np.ndarray | None:
    """Where the camera's optical axis meets the plane z = z0 (world x, y)."""
    d = cam[:3, 2]
    if d[2] > -1e-6:
        return None
    t = (z0 - cam[2, 3]) / d[2]
    return cam[:2, 3] + t * d[:2]


def run(
    ctx: NodeContext,
    region_center: dict,
    region_size: Any,
    region_yaw_deg: float = 0.0,
    px_per_m: float = 2000.0,
    camera_name: str = "",
    surface_z: float | None = None,
    height_m: float | None = None,
    return_height: bool = True,
    arm_id: int = 0,
) -> Output:
    """Scan ``region_center`` (world, z = the surface if ``surface_z`` is not given) by ``region_size``
    ([w, h] m along the region's x and y, x at ``region_yaw_deg``) with the wrist camera from straight
    above, and blend the frames into one top-down image at ``px_per_m``. The camera height is fx /
    px_per_m within 10-30 cm unless ``height_m`` is given."""
    kw = {"arm_id": arm_id} if arm_id else {}
    c = np.array([float(region_center[k]) for k in "xyz"])
    size = (float(region_size[0]), float(region_size[1]))
    z0 = float(surface_z) if surface_z is not None else c[2]
    try:
        tool = _matrix(_pose(ctx.tool("robot.get_ee_pose", **kw)))
        obs = ctx.tool("robot.get_observation")
    except Exception as exc:
        return {"route": "failed", "detail": f"cannot read the arm or the cameras: {exc}"}
    cam = _camera(obs, camera_name, tool)
    if cam is None:
        names = [x.get("name") for x in obs.get("cameras") or []]
        return {
            "route": "failed",
            "detail": f"no wrist camera with a pose and rgb (cameras: {names})",
        }
    name = str(cam.get("name", ""))
    k = np.asarray(cam["intrinsics"], dtype=np.float64)
    img_h, img_w = np.asarray(cam["rgb"]).shape[:2]
    tool_cam = np.linalg.inv(tool) @ _matrix(cam["pose"])
    fx = 0.5 * (k[0, 0] + k[1, 1])
    h_cam = float(height_m) if height_m else min(MAX_HEIGHT_M, max(MIN_HEIGHT_M, fx / px_per_m))
    footprint = (USE_FRACTION * img_w * h_cam / k[0, 0], USE_FRACTION * img_h * h_cam / k[1, 1])
    centres = plan_tiles(c, size, region_yaw_deg, footprint)
    if len(centres) > MAX_TILES:
        return {
            "route": "failed",
            "tiles": len(centres),
            "detail": f"{len(centres)} frames needed (over {MAX_TILES}): lower px_per_m or scan a smaller region",
        }
    mpp = 1.0 / px_per_m
    out_w, out_h = math.ceil(size[0] / mpp), math.ceil(size[1] / mpp)
    to_mosaic = plane_to_mosaic(c, region_yaw_deg, (out_w, out_h), mpp)
    acc = np.zeros((out_h, out_w, 3), np.float64)
    wsum = np.zeros((out_h, out_w), np.float64)
    hacc = np.zeros((out_h, out_w), np.float64)
    hw = np.zeros((out_h, out_w), np.float64)
    warped_tiles: list[tuple[np.ndarray, np.ndarray]] = []
    seen, skipped = 0, []

    def capture(xy: np.ndarray) -> bool:
        nonlocal acc, wsum, hacc, hw, seen
        goal = None
        for roll, tilt in VIEWS:
            cam_goal = np.eye(4)
            cam_goal[:3, :3] = _downward(region_yaw_deg, roll, tilt)
            # A tilted camera stands back so its axis still meets the surface at the frame's centre.
            cam_goal[:3, 3] = [xy[0], xy[1], z0] - cam_goal[:3, 2] * (
                h_cam / max(0.3, -cam_goal[2, 2])
            )
            cand = cam_goal @ np.linalg.inv(tool_cam)
            try:
                ik = ctx.tool("robot.solve_ik", pose=_to_pose(cand), **kw)
            except Exception:
                ik = None
            if isinstance(ik, dict) and (
                ik.get("success") is False or ik.get("within_servo_tolerance") is False
            ):
                continue
            try:
                reply = ctx.tool("robot.go_to_pose", pose=_to_pose(cand), **kw)
            except Exception:
                continue
            if _refusal(reply):
                continue
            goal = cand
            break
        if goal is None:
            return False
        frame = None
        for step in range(AIM_STEPS + 1):
            try:
                ctx.tool("robot.wait_steps", steps=SETTLE_STEPS)
            except Exception:
                pass
            frame = _camera(ctx.tool("robot.get_observation"), name, None)
            if frame is None:
                return False
            hit = _axis_hit(_matrix(frame["pose"]), z0)
            if hit is None or step == AIM_STEPS or float(np.linalg.norm(hit - xy)) <= AIM_TOL_M:
                break
            goal = goal.copy()
            goal[:2, 3] += xy - hit
            try:
                reply = ctx.tool("robot.go_to_pose", pose=_to_pose(goal), **kw)
            except Exception:
                break
            if _refusal(reply):
                break
        seen += 1
        rgb = _rgb(frame["rgb"])
        cam_pose = _matrix(frame["pose"])
        kk = np.asarray(frame.get("intrinsics", k), dtype=np.float64)
        h_img = plane_to_image(kk, cam_pose, z0)
        img_to_mosaic = to_mosaic @ np.linalg.inv(h_img)
        ih, iw = rgb.shape[:2]
        valid = np.zeros((ih, iw), np.uint8)
        my, mx = int(0.5 * (1 - USE_FRACTION) * ih), int(0.5 * (1 - USE_FRACTION) * iw)
        valid[my : ih - my, mx : iw - mx] = 255
        weight = cv2.distanceTransform(valid, cv2.DIST_L2, 3).astype(np.float32)
        warped = cv2.warpPerspective(rgb, img_to_mosaic, (out_w, out_h), flags=cv2.INTER_LINEAR)
        wmap = cv2.warpPerspective(
            weight, img_to_mosaic, (out_w, out_h), flags=cv2.INTER_LINEAR
        ).astype(np.float64)
        acc += warped * wmap[..., None]
        wsum += wmap
        warped_tiles.append((cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), wmap > 0))
        depth = frame.get("depth")
        if return_height and depth is not None:
            d = np.asarray(depth, dtype=np.float64)
            vv, uu = np.mgrid[0:ih, 0:iw]
            ok = np.isfinite(d) & (d > 0) & (valid > 0)
            pc = np.column_stack(
                [
                    (uu[ok] - kk[0, 2]) * d[ok] / kk[0, 0],
                    (vv[ok] - kk[1, 2]) * d[ok] / kk[1, 1],
                    d[ok],
                ]
            )
            pw = pc @ cam_pose[:3, :3].T + cam_pose[:3, 3]
            pm = (to_mosaic @ np.vstack([pw[:, 0], pw[:, 1], np.ones(len(pw))])).T
            ui, vi = np.round(pm[:, 0]).astype(int), np.round(pm[:, 1]).astype(int)
            inside = (ui >= 0) & (ui < out_w) & (vi >= 0) & (vi < out_h)
            np.add.at(hacc, (vi[inside], ui[inside]), pw[inside, 2] - z0)
            np.add.at(hw, (vi[inside], ui[inside]), 1.0)
        return True

    for idx, xy in enumerate(centres):
        if not capture(xy):
            skipped.append(idx)
    from_mosaic = np.linalg.inv(to_mosaic)
    extra = 0
    for _ in range(FILL_PASSES):
        gaps = (wsum <= 0).astype(np.uint8)
        if gaps.mean() < 0.02 or len(centres) + extra >= MAX_TILES:
            break
        n, lab, stats, cents = cv2.connectedComponentsWithStats(gaps, connectivity=8)
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
        for comp in order:
            if (
                stats[comp, cv2.CC_STAT_AREA] < 0.01 * gaps.size
                or len(centres) + extra >= MAX_TILES
            ):
                break
            # Aim at the patch's pixel nearest its centroid (a ring-shaped gap's centroid may be covered).
            ys, xs = np.nonzero(lab == comp)
            j = int(np.argmin((xs - cents[comp][0]) ** 2 + (ys - cents[comp][1]) ** 2))
            p = from_mosaic @ [xs[j], ys[j], 1.0]
            extra += 1
            capture(np.array([p[0] / p[2], p[1] / p[2]]))
    if seen == 0:
        return {
            "route": "failed",
            "tiles": len(centres),
            "tiles_seen": 0,
            "detail": "no frame position was reached",
        }
    covered = wsum > 0
    mosaic = np.zeros((out_h, out_w, 3), np.uint8)
    mosaic[covered] = np.clip(acc[covered] / wsum[covered, None], 0, 255).astype(np.uint8)
    shifts = []
    for a in range(len(warped_tiles)):
        for b in range(a + 1, len(warped_tiles)):
            both = warped_tiles[a][1] & warped_tiles[b][1]
            ys, xs = np.nonzero(both)
            if len(xs) == 0 or min(np.ptp(xs), np.ptp(ys)) < SEAM_MIN_OVERLAP_PX:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ga = warped_tiles[a][0][y0:y1, x0:x1].astype(np.float32)
            gb = warped_tiles[b][0][y0:y1, x0:x1].astype(np.float32)
            if ga.std() < 2 or gb.std() < 2:
                continue
            win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
            if resp > 0.1:
                shifts.append(math.hypot(dx, dy))
    out: Output = {
        "mosaic": mosaic,
        "metres_per_pixel": mpp,
        "origin": {
            "x": float((np.linalg.inv(to_mosaic) @ [0, 0, 1])[0]),
            "y": float((np.linalg.inv(to_mosaic) @ [0, 0, 1])[1]),
            "z": z0,
        },
        "size_px": [out_w, out_h],
        "board_corners_px": [
            [0.0, 0.0],
            [float(out_w), 0.0],
            [float(out_w), float(out_h)],
            [0.0, float(out_h)],
        ],
        "tiles": len(centres) + extra,
        "tiles_seen": seen,
        "covered_fraction": float(covered.mean()),
        "camera_height_m": h_cam,
        "seam_shift_px": float(np.median(shifts)) if shifts else 0.0,
        "seam_shift_max_px": float(np.max(shifts)) if shifts else 0.0,
    }
    if return_height:
        hm = np.full((out_h, out_w), np.nan, np.float32)
        nz = hw > 0
        hm[nz] = (hacc[nz] / hw[nz]).astype(np.float32)
        out["height_map"] = hm
    route = "mosaic_built" if out["covered_fraction"] >= 0.98 else "partial"
    out.update(
        route=route,
        detail=(
            f"{seen}/{len(centres)} frames at {h_cam * 100:.1f} cm, {out_w}x{out_h} px at {px_per_m:.0f} px/m,"
            f" {out['covered_fraction']:.0%} covered, seams {out['seam_shift_px']:.1f} px median"
            + (f"; frames {skipped} not reached" if skipped else "")
        ),
    )
    return out
