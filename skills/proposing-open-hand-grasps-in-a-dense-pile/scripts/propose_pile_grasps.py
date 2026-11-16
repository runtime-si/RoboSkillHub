"""Propose top-down grasps in a dense pile inside a bin: wherever the fully open hand fits round something."""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

CELL = 0.002
"""Heightmap cell [m], in the bin's own frame. tuned_on: a YAM's wrist D405 looking into a bin
from about 0.3 m over the counter (2026-11-14/15)."""

BIN_MARGIN = 0.012
"""The map starts this far inside the bin's outer walls [m]. tuned_on: a 449 x 262 mm corrugated
cardboard box (4 mm walls, flaps and fit error); chosen."""

HALF_OPEN = 0.047
"""The open fingers' inner faces from the hand's centre [m], when robot.describe_gripper states no
max_grasp_width_m. tuned_on: i2rt YAM linear_4310 hand (94 mm span)."""

PAD_WIDTH, FINGER_THICKNESS = 0.028, 0.014
"""A fingertip's footprint across the jaw and along it [m], as this method used them on that hand,
with room to spare; not measured with a caliper. Pass pad_width / finger_thickness for another."""

MIDDLE = (0.024, 0.012)
"""What the hand closes on is the highest solid thing within this box at its centre: 24 mm along
the jaw, 12 mm across [m]. tuned_on: 5-9 cm printed brackets and tube sockets; chosen."""

SPIKE_OPEN = 5
"""The part's top is read from the map opened by 5 cells (10 mm): a depth reading standing alone
over its neighbours is not a part. tuned_on: that D405; one pixel read 25 cm up, 7 cm over the
pile, and was offered as the best grasp before this."""

MIN_GRIP = 0.015
"""What stands between the fingers must be at least this far over the highest thing under either
fingertip [m]. tuned_on: those parts on the real station; chosen, not swept."""

MAX_DEPTH = 0.035
"""The fingertips go at most this far under the part's top [m]. tuned_on: same."""

TIP_CLEAR = 0.002
"""The fingertips stop this far over what is under them [m]. The source task then sent the hand
8 mm lower still ("the pile gives"): that is the grasping skill's choice, not made here."""

WALL = 0.02
"""No finger footprint within this of the map's edge [m] (so 32 mm from the bin's outer wall).
tuned_on: same; the task ran with 0.03."""

FLOOR_CLEAR = 0.012
"""Cells under this far over the bin's support are its floor, not a part [m]. tuned_on: that box
on a counter; 6 mm brackets lying flat on the floor are therefore never proposed."""

ALLOW_MIN, AVOID_MAX = 0.6, 0.15
"""A grasp centre needs over 60 % of the 7 x 7 cells round it allowed and under 15 % of the
15 x 15 cells round it to avoid. tuned_on: grey printed parts among black aluminium extrusion,
told apart by colour; chosen."""

KNOWN_MIN = 0.6
"""Over this share of the two finger footprints must have been seen; unseen cells otherwise count
as empty. tuned_on: single slanted wrist looks that saw 34-48 % of the pile."""

MIN_SEEN = 0.10
"""Under this share of the bin's inside seen, the look is not_seen. Chosen here, not on a robot."""

TIP_REACH = 0.015
"""Fingertips below the tool centre [m] when robot.describe_gripper states none. tuned_on: the
source station's grasp site on the YAM hand."""

APART = 0.03
"""Proposals are at least this far apart in the plane [m]."""

HEADINGS = (-90.0, 80.0, 10.0)
"""Jaw closing directions tried: from, to, step [deg from world x]. The source arm used
(-60, 50): its last joint stalled outside that."""


class Output(TypedDict):
    route: str
    detail: str
    grasps: list
    position: dict[str, float] | None
    close_heading_deg: float
    jaw_width_m: float
    top_z: float
    tip_z: float
    tool_z: float
    grip_depth_m: float
    score: float
    grasp_pose: dict[str, Any] | None
    n_grasps: int
    seen_fraction: float
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


def _wrist_camera(ctx: NodeContext, name: str | None) -> dict | None:
    """The named posed depth camera; unnamed, the first on a wrist, else the first of any."""
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
    if isinstance(cams, dict):
        cams = [dict(v, name=v.get("name", k)) for k, v in cams.items()]
    cams = [c for c in (cams or []) if c.get("depth") is not None and c.get("pose") is not None]
    if name:
        return next((c for c in cams if str(c.get("name")) == name), None)
    return next(
        (c for c in cams if any(k in str(c.get("name", "")) for k in ("wrist", "eye_in_hand"))),
        cams[0] if cams else None,
    )


def _as_bool(mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != shape:
        raise ValueError(f"a mask of shape {m.shape} does not match the depth image {shape}")
    return m > (127 if m.dtype == np.uint8 else 0)


class Pile:
    """The inside of a bin as a grid in the bin's frame (x along its long side): per cell the
    highest point seen ``z`` (nan where nothing was), and the shares ``allow`` / ``avoid``."""

    def __init__(self, bin_obb: dict[str, Any], margin: float = BIN_MARGIN) -> None:
        self.centre, half, self.yaw = footprint(bin_obb)
        self.support = float(bin_obb["center"]["z"]) - float(bin_obb["extent"]["z"])
        self.rim = float(bin_obb["center"]["z"]) + float(bin_obb["extent"]["z"])
        self.x0, self.y0 = -half[0] + margin, -half[1] + margin
        self.nx = max(1, int((2 * half[0] - 2 * margin) / CELL))
        self.ny = max(1, int((2 * half[1] - 2 * margin) / CELL))
        self.z = np.full((self.ny, self.nx), -np.inf, np.float32)
        self.allow = np.zeros((self.ny, self.nx), np.float32)
        self.avoid = np.zeros((self.ny, self.nx), np.float32)
        self.seen_fraction = 0.0

    def local(self, p: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = p[..., :2] - self.centre
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    def world(self, q: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack(
            [
                self.centre[0] + c * q[..., 0] - s * q[..., 1],
                self.centre[1] + s * q[..., 0] + c * q[..., 1],
            ],
            -1,
        )

    def add(self, cam: dict, allow: Any = None, avoid: Any = None, min_range: float = 0.0) -> None:
        """Write one frame's points in: the highest per cell, with its masks' values."""
        depth = np.asarray(cam["depth"], dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        k = np.asarray(cam["intrinsics"], dtype=np.float64)
        t = _matrix(cam["pose"])
        vv, uu = np.nonzero(np.isfinite(depth) & (depth > max(min_range, 0.0)))
        zc = depth[vv, uu]
        pc = np.column_stack([(uu - k[0, 2]) * zc / k[0, 0], (vv - k[1, 2]) * zc / k[1, 1], zc])
        pts = pc @ t[:3, :3].T + t[:3, 3]
        q = self.local(pts)
        ok = (
            (q[:, 0] > self.x0)
            & (q[:, 0] < self.x0 + self.nx * CELL)
            & (q[:, 1] > self.y0)
            & (q[:, 1] < self.y0 + self.ny * CELL)
            & (pts[:, 2] > self.support - 0.01)
            & (pts[:, 2] < self.rim + 0.12)
        )
        q, z, vv, uu = q[ok], pts[ok, 2], vv[ok], uu[ok]
        r = ((q[:, 1] - self.y0) / CELL).astype(int).clip(0, self.ny - 1)
        c = ((q[:, 0] - self.x0) / CELL).astype(int).clip(0, self.nx - 1)
        a = _as_bool(allow, depth.shape)
        b = _as_bool(avoid, depth.shape)
        order = np.argsort(z)  # the highest written last
        r, c, z, vv, uu = r[order], c[order], z[order], vv[order], uu[order]
        higher = z >= self.z[r, c]
        r, c, z, vv, uu = r[higher], c[higher], z[higher], vv[higher], uu[higher]
        self.z[r, c] = z
        self.allow[r, c] = 1.0 if a is None else a[vv, uu]
        self.avoid[r, c] = 0.0 if b is None else b[vv, uu]

    def close(self) -> None:
        """Small unseen gaps take their neighbours' mean (3 x 3, at least 3 seen); the rest stay
        unknown."""
        import cv2

        seen = np.isfinite(self.z)
        self.seen_fraction = float(seen.mean())
        k = np.ones((3, 3), np.float32)
        den = cv2.filter2D(seen.astype(np.float32), -1, k)
        fill = ~seen & (den >= 3)
        for name in ("z", "allow", "avoid"):
            v = getattr(self, name)
            num = cv2.filter2D(np.where(seen, v, 0).astype(np.float32), -1, k)
            blank = np.nan if name == "z" else 0.0  # nothing seen: nothing to take hold of
            setattr(self, name, np.where(seen, v, np.where(fill, num / np.maximum(den, 1), blank)))


def grasps_fit(
    pile: Pile,
    headings_deg: Any,
    half_open: float = HALF_OPEN,
    pad_width: float = PAD_WIDTH,
    finger_thickness: float = FINGER_THICKNESS,
    min_grip: float = MIN_GRIP,
    max_depth: float = MAX_DEPTH,
    wall: float = WALL,
    roi: dict | None = None,
    top_n: int = 12,
    prefer_high: float = 0.3,
    rank: str = "mixed",
) -> list[dict[str, Any]]:
    """Top-down grasps where the open hand fits, best first.

    For every cell and each jaw closing direction in ``headings_deg`` (world): the two fingertip
    footprints at full opening go down to the highest thing under either (``floor``); the highest
    solid thing in the middle box for that direction is the part's ``top``; ``top - floor`` is
    what the fingers have to close on and must reach ``min_grip``. Unseen cells count as empty,
    but a footprint must be mostly seen.

    ``rank``: "mixed" scores the grip (counted to 3 cm) plus ``prefer_high`` times the top's
    height over the rim; "depth" puts first the grasps whose fingertips go deepest under the top
    (to ``max_depth``, in 2 mm steps), then the higher part.
    """
    import cv2

    if rank not in ("mixed", "depth"):
        raise ValueError('rank must be "mixed" or "depth"')
    seen = np.isfinite(pile.z)
    z = np.where(seen, pile.z, pile.support).astype(np.float32)
    ny, nx = z.shape
    half = int((half_open + finger_thickness + 0.004) / CELL) + 2
    size = 2 * half + 1
    # What is taken hold of is a part, not a stray depth reading: the top is read from the opened
    # map. The fingers' own clearance keeps every point.
    z_solid = cv2.morphologyEx(z, cv2.MORPH_OPEN, np.ones((SPIKE_OPEN, SPIKE_OPEN), np.uint8))
    # the bin's bare floor is not something to take hold of
    stands = np.where(seen & (z >= pile.support + FLOOR_CLEAR), pile.allow, 0.0)
    allow = cv2.blur(stands.astype(np.float32), (7, 7))
    avoid = cv2.blur(pile.avoid.astype(np.float32), (15, 15))
    m = int(wall / CELL)
    inside = np.zeros_like(seen)
    inside[m : ny - m, m : nx - m] = True
    cc, rr = np.meshgrid(np.arange(nx), np.arange(ny))
    wxy = pile.world(np.stack([pile.x0 + cc * CELL, pile.y0 + rr * CELL], -1))
    if roi:
        inside &= (wxy[..., 0] >= float(roi.get("x_min", -np.inf))) & (
            wxy[..., 0] <= float(roi.get("x_max", np.inf))
        )
        inside &= (wxy[..., 1] >= float(roi.get("y_min", -np.inf))) & (
            wxy[..., 1] <= float(roi.get("y_max", np.inf))
        )
    found = []
    for heading in headings_deg:
        a = math.radians(float(heading)) - pile.yaw
        u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])

        def rect(
            c0: float, c1: float, w: float, u: np.ndarray = u, v: np.ndarray = v
        ) -> np.ndarray:
            k = np.zeros((size, size), np.uint8)
            pts = [c0 * u - w / 2 * v, c1 * u - w / 2 * v, c1 * u + w / 2 * v, c0 * u + w / 2 * v]
            cv2.fillPoly(k, [np.round(np.array(pts) / CELL + half).astype(np.int32)], 1)
            return k

        fingers = rect(half_open, half_open + finger_thickness, pad_width) | rect(
            -half_open - finger_thickness, -half_open, pad_width
        )
        middle = rect(-MIDDLE[0] / 2, MIDDLE[0] / 2, MIDDLE[1])
        floor = cv2.dilate(z, fingers)
        top = cv2.dilate(z_solid, middle)
        known = cv2.filter2D(
            seen.astype(np.float32),
            -1,
            fingers.astype(np.float32) / fingers.sum(),
            borderType=cv2.BORDER_CONSTANT,
        )
        grip = top - floor
        clear = cv2.erode(
            inside.astype(np.uint8), fingers | middle, borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)
        ok = (
            clear
            & (grip >= min_grip)
            & (allow > ALLOW_MIN)
            & (avoid < AVOID_MAX)
            & (known > KNOWN_MIN)
        )
        for r, c in zip(*np.nonzero(ok), strict=True):
            top_z, floor_z = float(top[r, c]), float(floor[r, c])
            tips = max(floor_z + TIP_CLEAR, top_z - max_depth)
            high = top_z - pile.rim
            if rank == "depth":
                score = round((top_z - tips) / 0.002) + 2.0 * prefer_high * high
                score += 0.1 * float(known[r, c])
            else:
                score = min(float(grip[r, c]), 0.03) + prefer_high * high
                score += 0.01 * float(known[r, c])
            found.append((score, int(r), int(c), float(heading), tips, top_z))
    found.sort(key=lambda t: -t[0])
    kept: list[dict[str, Any]] = []
    for score, r, c, heading, tips, top_z in found:
        p = wxy[r, c]
        if any(math.hypot(p[0] - g["x"], p[1] - g["y"]) <= APART for g in kept):
            continue
        # top_z is the top of what stands between the fingers for this jaw direction, the same
        # the depth is measured from. The cell's own height gave "depths" of -6 cm when the
        # part's top was 1 cm to the side, and a hand that set off too low.
        kept.append(
            {
                "x": float(p[0]),
                "y": float(p[1]),
                "close_heading_deg": (heading + 90.0) % 180.0 - 90.0,
                "top_z": top_z,
                "tip_z": tips,
                "grip_depth_m": top_z - tips,
                "score": float(score),
            }
        )
        if len(kept) >= top_n:
            break
    return kept


def run(
    ctx: NodeContext,
    bin_obb: dict,
    camera: str = "",
    camera_frame: dict | None = None,
    camera_frames: list | None = None,
    allow_mask: Any = None,
    avoid_mask: Any = None,
    headings_deg: list | None = None,
    heading_min: float = HEADINGS[0],
    heading_max: float = HEADINGS[1],
    heading_step: float = HEADINGS[2],
    half_open: float = 0.0,
    pad_width: float = PAD_WIDTH,
    finger_thickness: float = FINGER_THICKNESS,
    min_grip: float = MIN_GRIP,
    max_depth: float = MAX_DEPTH,
    wall: float = WALL,
    roi: dict | None = None,
    rank: str = "mixed",
    prefer_high: float = 0.3,
    top_n: int = 12,
    min_range: float = 0.16,
    frame: bool = True,
    arm_id: int = 0,
) -> Output:
    """Ranked top-down grasps in the pile inside ``bin_obb``, for a hand that comes down fully open.

    Args:
        bin_obb: The bin (upright; half extents of its outer size; its top the rim), in the frame
            of the camera read - anchor it first when that is a wrist camera.
        camera / camera_frame / camera_frames: The camera's name in the observation (default: the
            first wrist camera), one CameraFrame, or several looks merged into one map (the
            higher reading per cell). Masks apply to a single frame only.
        allow_mask, avoid_mask: Image masks of what may be grasped and of what no grasp may be
            near (another kind of part); default everything, nothing.
        headings_deg, or heading_min / heading_max / heading_step: Jaw closing directions to try
            [deg from world x]; restrict them to what the wrist can turn to.
        half_open: Half the open hand's inner width [m] (0 = half of robot.describe_gripper's
            max_grasp_width_m, else 47 mm). pad_width, finger_thickness: the fingertip footprint.
        min_grip, max_depth, wall, roi, rank, prefer_high, top_n: see grasps_fit.
        min_range: Depth nearer than this is dropped (the hand's own fingers) [m].
        frame: Ask robot.grasp_frame for the best grasp's rotation.

    Returns:
        ``route`` "proposed" (``grasps`` best first, the best one's fields flattened),
        "nothing_fits" or "not_seen".
    """
    on_arm = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "not_seen",
        "detail": "",
        "grasps": [],
        "position": None,
        "close_heading_deg": 0.0,
        "jaw_width_m": 0.0,
        "top_z": 0.0,
        "tip_z": 0.0,
        "tool_z": 0.0,
        "grip_depth_m": 0.0,
        "score": 0.0,
        "grasp_pose": None,
        "n_grasps": 0,
        "seen_fraction": 0.0,
        "camera": "",
    }
    hand: dict[str, Any] = {}
    try:
        hand = ctx.tool("robot.describe_gripper", **on_arm) or {}
    except Exception:  # noqa: BLE001 - the hand's numbers are optional
        hand = {}
    finger = hand.get("finger") or {}
    reach = float(finger.get("reach_m") or TIP_REACH) if finger.get("stated", True) else TIP_REACH
    span = float(hand.get("max_grasp_width_m") or hand.get("span_m") or 0.0)
    opening = float(half_open) or (span / 2 if span > 0 else HALF_OPEN)

    frames = list(camera_frames or ([camera_frame] if camera_frame is not None else []))
    if not frames:
        cam = _wrist_camera(ctx, camera or None)
        if cam is None:
            out["detail"] = f"no posed depth camera {camera or '(any)'!r} in the observation"
            return out
        frames = [cam]
    out["camera"] = str(frames[0].get("name", ""))
    pile = Pile(bin_obb)
    single = len(frames) == 1
    for f in frames:
        pile.add(
            f,
            allow_mask if single else None,
            avoid_mask if single else None,
            float(min_range),
        )
    pile.close()
    out["seen_fraction"] = pile.seen_fraction
    if pile.seen_fraction < MIN_SEEN:
        out["detail"] = f"{pile.seen_fraction:.0%} of the bin's inside has depth (need 10 %)"
        return out
    if headings_deg is None:
        headings_deg = list(
            np.arange(float(heading_min), float(heading_max) + 1e-6, float(heading_step))
        )
    found = grasps_fit(
        pile,
        headings_deg,
        half_open=opening,
        pad_width=float(pad_width),
        finger_thickness=float(finger_thickness),
        min_grip=float(min_grip),
        max_depth=float(max_depth),
        wall=float(wall),
        roi=roi,
        top_n=int(top_n),
        prefer_high=float(prefer_high),
        rank=rank,
    )
    if not found:
        out["route"] = "nothing_fits"
        out["detail"] = (
            f"{pile.seen_fraction:.0%} of the bin seen; nowhere does something stand "
            f"{float(min_grip) * 1e3:.0f} mm over both fingertips of a hand open "
            f"{2 * opening * 1e3:.0f} mm"
        )
        return out
    for g in found:
        g["tool_z"] = g["tip_z"] + reach
        g["jaw_width_m"] = 2 * opening
        g["position"] = {"x": g.pop("x"), "y": g.pop("y"), "z": g["tool_z"]}
    best = found[0]
    out["route"] = "proposed"
    out["grasps"], out["n_grasps"] = found, len(found)
    for key in ("position", "close_heading_deg", "jaw_width_m", "top_z", "tip_z", "tool_z"):
        out[key] = best[key]  # type: ignore[literal-required]
    out["grip_depth_m"], out["score"] = best["grip_depth_m"], best["score"]
    if frame:
        try:
            gf = ctx.tool(
                "robot.grasp_frame", close_heading_deg=best["close_heading_deg"], **on_arm
            )
            out["grasp_pose"] = {"position": best["position"], "rotation": gf["rotation"]}
        except Exception as exc:  # the proposal stands without the rotation
            logger.info("proposing-open-hand-grasps-in-a-dense-pile: no grasp frame (%s)", exc)
    out["detail"] = (
        f"{len(found)} grasp(s); the best closes along {best['close_heading_deg']:.0f} deg, "
        f"fingertips {best['grip_depth_m'] * 1e3:.0f} mm under a top at {best['top_z']:.3f} m; "
        f"{pile.seen_fraction:.0%} of the bin seen"
    )
    return out
