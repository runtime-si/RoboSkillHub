"""Turn a flat piece lying on the table to a target yaw by pushing one of its corners with the closed fingertips,
re-measuring its yaw from the wrist camera after every push.

Written for the hub; no code copied. Each look puts the wrist camera straight above the piece, lifts its depth to
the world and takes everything a millimetre over the table as the piece; the yaw is the turn of the piece's known
top-down mask that overlaps that footprint best. Each push starts just outside a body corner and moves the closed
tips tangentially (perpendicular to the corner's radius), so the piece turns about its centre by about the push
length over the radius (the standard pushing approximation, M. T. Mason, Mechanics of Robotic Manipulation, 2001,
ch. 10). Steps are capped and the loop closes on the measurement.
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np
from gap import NodeContext

TOL_DEG = 5.0
"""Stop when the measured yaw is within this of the target."""

MAX_STEP_DEG = 10.0
"""Largest turn asked of one push."""

MAX_PUSHES = 12
"""Give up after this many pushes."""

VIEW_HEIGHT_M = 0.15
"""Camera height over the table for each look (D405 depth is good at 10-30 cm)."""

ABOVE_TABLE_M = 0.001
"""A point this far over the table belongs to the piece (pieces are 2 mm thick)."""

GRID_M = 0.0005
"""Footprint raster for the yaw fit."""

TIP_CLEAR_M = 0.006
"""A push starts this far outside the corner along the push line (closed blade tips are about 3.9 mm thick
per blade, so the tool centre sits 4 mm behind the touching face; plus margin)."""

TIP_HEIGHT_M = 0.0005
"""Blade tips this far over the table during a push (they push the piece's 2 mm side)."""

TIP_REACH_M = 0.0102
"""Tool centre to the blade bottoms (station model, linear_4310)."""

GAIN = 0.8
"""A push is sized for this fraction of the remaining turn (a pushed corner also slides; asking for the whole
turn overshoots)."""


class Output(TypedDict, total=False):
    route: str
    yaw_deg: float
    error_deg: float
    pushes: int
    position: dict[str, float]
    history: list[dict[str, float]]
    detail: str


def _rot(q: dict) -> np.ndarray:
    w, x, y, z = (float(q[k]) for k in "wxyz")
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
    q = np.zeros(4)
    t = np.trace(r)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (r[j, i] + r[i, j]) / s
        v[k] = (r[k, i] + r[i, k]) / s
        q = [(r[k, j] - r[j, k]) / s, *v]
    return dict(zip("wxyz", (float(a) for a in q), strict=True))


def _pose(reply: Any) -> dict:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _matrix(pose: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = [float(pose["position"][k]) for k in "xyz"]
    return m


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _refused(reply: Any) -> bool:
    return isinstance(reply, dict) and (
        reply.get("reached") is False or reply.get("success") is False
    )


def _wrist(obs: dict) -> dict | None:
    cams = [
        c
        for c in obs.get("cameras") or []
        if c.get("depth") is not None and c.get("pose") is not None
    ]
    return next(
        (c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))),
        None,
    )


def template_points(mask: np.ndarray, mpp: float) -> np.ndarray:
    """The mask's pixels as metres round its centroid (x right, y up)."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    p = np.column_stack([xs, -ys]).astype(np.float64) * mpp
    return p - p.mean(axis=0)


def fit_yaw(footprint: np.ndarray, tmpl: np.ndarray, prior: float | None) -> tuple[float, float]:
    """Yaw (deg) of the template best covering the footprint points (both centred), and its IoU proxy."""
    fp = footprint - footprint.mean(axis=0)
    lo = fp.min(axis=0) - 0.01
    n = np.ceil((fp.max(axis=0) + 0.01 - lo) / GRID_M).astype(int) + 1
    obs = np.zeros((n[1], n[0]), np.uint8)
    ij = np.round((fp - lo) / GRID_M).astype(int)
    obs[ij[:, 1], ij[:, 0]] = 1
    obs = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    total = obs.sum()
    angles = (
        np.arange(-180.0, 180.0, 2.0)
        if prior is None
        else np.arange(prior - 30.0, prior + 30.1, 2.0)
    )
    best = (-1.0, 0.0)

    def score(a):
        r = math.radians(a)
        q = tmpl @ np.array([[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]])
        k = np.round((q - lo) / GRID_M).astype(int)
        ok = (k[:, 0] >= 0) & (k[:, 0] < n[0]) & (k[:, 1] >= 0) & (k[:, 1] < n[1])
        hit = obs[k[ok, 1], k[ok, 0]].sum()
        return hit / (len(q) + total - hit)

    for a in angles:
        s = score(a)
        if s > best[0]:
            best = (s, a)
    for a in np.arange(best[1] - 2.0, best[1] + 2.01, 0.25):
        s = score(a)
        if s > best[0]:
            best = (s, a)
    return _wrap(best[1]), best[0]


class _Look:
    def __init__(self, ctx: NodeContext, kw: dict):
        self.ctx, self.kw = ctx, kw
        tool = _matrix(_pose(ctx.tool("robot.get_ee_pose", **kw)))
        cam = _wrist(ctx.tool("robot.get_observation"))
        if cam is None:
            raise ValueError("no wrist camera with depth and a pose")
        self.tool_cam = np.linalg.inv(tool) @ _matrix(cam["pose"])
        self.guess_z = float(tool[2, 3]) - 0.2

    def footprint(self, xy: np.ndarray, table_z: float | None) -> tuple[np.ndarray, float]:
        goal = np.eye(4)
        x = np.array([1.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, -1.0])
        goal[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        goal[:3, 3] = [
            xy[0],
            xy[1],
            (table_z if table_z is not None else self.guess_z) + VIEW_HEIGHT_M,
        ]
        tool = goal @ np.linalg.inv(self.tool_cam)
        self.ctx.tool(
            "robot.go_to_pose",
            pose={
                "position": dict(zip("xyz", (float(v) for v in tool[:3, 3]), strict=True)),
                "rotation": _quat(tool[:3, :3]),
            },
            **self.kw,
        )
        try:
            self.ctx.tool("robot.wait_steps", steps=30)
        except Exception:
            pass
        cam = _wrist(self.ctx.tool("robot.get_observation"))
        d = np.asarray(cam["depth"], dtype=np.float64)
        k = np.asarray(cam["intrinsics"], dtype=np.float64)
        vv, uu = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
        ok = np.isfinite(d) & (d > 0)
        pc = np.column_stack(
            [(uu[ok] - k[0, 2]) * d[ok] / k[0, 0], (vv[ok] - k[1, 2]) * d[ok] / k[1, 1], d[ok]]
        )
        m = _matrix(cam["pose"])
        pw = pc @ m[:3, :3].T + m[:3, 3]
        near = np.hypot(pw[:, 0] - xy[0], pw[:, 1] - xy[1]) < 0.035
        pw = pw[near]
        if len(pw) < 100:
            return np.zeros((0, 2)), table_z if table_z is not None else self.guess_z
        tz = float(np.median(pw[:, 2])) if table_z is None else table_z
        if table_z is None:
            hist, edges = np.histogram(
                pw[:, 2], bins=np.arange(pw[:, 2].min(), pw[:, 2].max() + 0.0005, 0.0005)
            )
            tz = float(edges[int(np.argmax(hist))] + 0.00025)
        top = pw[(pw[:, 2] > tz + ABOVE_TABLE_M) & (pw[:, 2] < tz + 0.006)]
        return top[:, :2], tz


def run(
    ctx: NodeContext,
    piece_mask: Any,
    metres_per_pixel: float,
    position: dict,
    target_yaw_deg: float,
    yaw_deg: float | None = None,
    tol_deg: float = TOL_DEG,
    max_pushes: int = MAX_PUSHES,
    arm_id: int = 0,
) -> Output:
    """Turn the piece at ``position`` (world x, y) to ``target_yaw_deg`` (the yaw of ``piece_mask``'s frame:
    the mask as given has yaw 0) by pushing its corners. ``yaw_deg`` is its current yaw if known (else
    measured over a full turn)."""
    kw = {"arm_id": arm_id} if arm_id else {}
    tmpl = template_points(piece_mask, metres_per_pixel)
    if len(tmpl) < 50:
        return {"route": "not_seen", "detail": "empty piece mask"}
    try:
        look = _Look(ctx, kw)
    except Exception as exc:
        return {"route": "not_seen", "detail": str(exc)}
    xy = np.array([float(position["x"]), float(position["y"])])
    if position.get("z") is not None:
        look.guess_z = float(position["z"])
    half = 0.5 * np.ptp(tmpl, axis=0)
    radius = float(np.hypot(*half)) * 0.85
    table_z, prior, hist = None, yaw_deg, []
    pushes = 0
    while True:
        fp, table_z = look.footprint(xy, table_z)
        if len(fp) < 100:
            return {
                "route": "not_seen",
                "pushes": pushes,
                "history": hist,
                "detail": "the piece is not under the camera",
            }
        xy = fp.mean(axis=0)
        yaw, fit = fit_yaw(fp, tmpl, prior)
        err = _wrap(target_yaw_deg - yaw)
        hist.append({"yaw": yaw, "x": float(xy[0]), "y": float(xy[1]), "fit": fit})
        if abs(err) <= tol_deg:
            return {
                "route": "turned", "yaw_deg": yaw, "error_deg": err, "pushes": pushes,
                "position": {"x": float(xy[0]), "y": float(xy[1]), "z": table_z}, "history": hist,
                "detail": f"yaw {yaw:+.1f} deg ({err:+.1f} from the target) after {pushes} pushes",
            }  # fmt: skip
        if pushes >= max_pushes:
            return {"route": "not_turned", "yaw_deg": yaw, "error_deg": err, "pushes": pushes, "history": hist,
                    "detail": f"still {err:+.1f} deg off after {pushes} pushes"}  # fmt: skip
        step = math.copysign(min(MAX_STEP_DEG, GAIN * abs(err)), err)
        # Corner: the body corner whose tangential push turns the piece the right way; any corner works, take
        # the one nearest the arm's side is not modelled - the +x/+y corner of the current pose.
        a = math.radians(yaw)
        corner = xy + radius * np.array([math.cos(a + math.pi / 4), math.sin(a + math.pi / 4)])
        rvec = corner - xy
        tang = np.array([-rvec[1], rvec[0]]) / np.linalg.norm(rvec) * math.copysign(1.0, step)
        dist = radius * math.radians(abs(step))
        start = corner - tang * TIP_CLEAR_M
        end = corner + tang * dist
        ctx.tool("robot.close_gripper", **kw)
        heading = math.atan2(tang[1], tang[0]) + math.pi / 2  # blades' wide face across the push
        r = np.column_stack(
            [
                [math.cos(heading), math.sin(heading), 0.0],
                [math.sin(heading), -math.cos(heading), 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        rot = _quat(r)
        z_push = table_z + TIP_REACH_M + TIP_HEIGHT_M
        for p, z in ((start, z_push + 0.03), (start, z_push), (end, z_push), (end, z_push + 0.03)):
            reply = ctx.tool("robot.go_to_pose_cartesian" if z == z_push else "robot.go_to_pose",
                             pose={"position": {"x": float(p[0]), "y": float(p[1]), "z": float(z)}, "rotation": rot}, **kw)  # fmt: skip
            if _refused(reply):
                return {
                    "route": "unreachable",
                    "pushes": pushes,
                    "history": hist,
                    "detail": f"push move refused: {reply}",
                }
        pushes += 1
        prior = yaw + step
