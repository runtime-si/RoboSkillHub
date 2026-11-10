"""Insert a peg held top-down into a perceived hole: align, look, descend, search, push, confirm, release."""

from __future__ import annotations

import importlib.util
import logging
import math
import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

STANDOFF = 0.025
"""Tip height over the mouth for the alignment and the wrist-camera look [m]:
inserting-a-peg-with-pose-refinement's 30 mm less 5 mm, so the tip and the
mouth both stay inside a wrist camera's near field."""

ENTER_DEPTH = 0.004
"""How far past the mouth the tip must get, as held, to count as entered [m]:
past a 0.5-1 mm chamfer and a contact stop's 1-2 mm overshoot."""

MOUTH_BAND = 0.003
"""A first contact with the tip within this of the mouth plane [m] is a block at
the mouth (rim or chamfer); one higher than this is something above the mouth."""

DEPTH_TOLERANCE = 0.002
"""How far short of the target depth the tip may end and still count as there [m]:
the YAM station's 2 mm per-axis settle, measured by the top probe."""

SLIP_TOLERANCE = 0.002
"""How much higher than expected the peg's top may be found with the hand having
reached the depth [m] before the peg is said to have slid up in the grip."""

FINGER_CLEAR = 0.015
"""Least distance the tool centre keeps over the mouth [m]: closed linear_4310
fingertips reach 10.2 mm below it, plus 5 mm (push_insert's value)."""

FINGER_REACH = 0.0102
"""Closed linear_4310 fingertips below the tool centre [m] (the twin's tip_reach,
measured on the station's fingers): the top probe touches with them."""

MAX_LAG = 0.004
"""Lag limit of the final push [m] (pressing-in-with-guarded-push's max_lag, 6 mm,
less a third: a peg in a clearance hole needs no press force, and a lower limit
stops sooner on a jam before the grip slips)."""

SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX = 0.002, 0.006
"""Bounds of the spiral's radius [m]; inside them it is 3 sigma of the hold plus
the hole estimate (searching-insertion-spiral's 6 mm is the top)."""

MAX_PROBES = 37
"""Spiral probes at most: searching-insertion-spiral's 6 mm / 2 mm hexagon."""

MIN_PITCH = 0.0005
"""Finest spiral pitch [m]. A hexagonal grid leaves every point within pitch / sqrt(3)
of a probe, so a pitch of sqrt(3) x (radial clearance + chamfer) cannot miss; under
0.5 mm the arm's own 0.2-0.6 mm landing scatter does the rest."""

PROBE_HOVER = 0.003
"""Height over the mouth each spiral probe starts from [m]."""

TILT_DEG = 3.0
"""Tilt of the tilt-then-upright attempt [deg]; tried only with a chamfer of at
least TILT_MIN_CHAMFER, where a tilted tip can drop into the chamfer's ring."""

TILT_MIN_CHAMFER = 0.0003
"""Least chamfer [m] for the tilt-then-upright attempt."""

CAM_STEPS = 6
"""Wrist-camera corrections at most (correcting-insertion-offset-from-port-opening: 8)."""

CAM_MAX_STEP = 0.0015
"""Largest single camera correction [m]."""

CAM_MAX_TOTAL = 0.006
"""Largest summed camera correction [m] (correcting-insertion-offset-from-port-opening's
6 mm); a larger one means the hole estimate is wrong, not the arm."""

CAM_MIN_POINTS = 20
"""Least depth pixels on the peg's lowest 3 mm, and on the rim, for a camera reading."""

CAM_RIM_BAND = 0.0015
"""Depth points within this of the mouth plane are the fixture's top face [m]."""

CAM_MIN_RIM_BINS = 8
"""Of 24 angular bins round the mouth, the least with a rim point for a circle fit."""

RETREAT = 0.04
"""Back-out along the hole's axis after the release [m] (handing-over's 0.05 less 10 mm)."""

TOP_PROBE_SPAN = 0.012
"""The top probe starts this far over where the peg's top should be and goes as far
under it [m]."""

PROBE_MAX_LAG = 0.003
"""Lag limit of the top probe [m]: half the guarded push's 6 mm; the probe only has
to feel the top, not press it."""

SIBLINGS = {
    "align": ("inserting-a-peg-with-pose-refinement", "scripts/align_pre_insert.py"),
    "push": ("inserting-a-peg-with-pose-refinement", "scripts/push_insert.py"),
    "press": ("pressing-in-with-guarded-push", "scripts/guarded_push.py"),
    "spiral": ("searching-insertion-spiral", "scripts/spiral_search.py"),
    "views": ("correcting-insertion-offset-from-port-opening", "scripts/correct_offset.py"),
}
_MODULES: dict[str, Any] = {}


def sibling(key: str) -> Any:
    """A hub skill's script installed beside this one or under a GAP_SKILLS_PATH root."""
    if key in _MODULES:
        return _MODULES[key]
    skill, script = SIBLINGS[key]
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(p) for p in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if p]
    for root in roots:
        for path in (root / skill / script, root / "skills" / skill / script):
            if path.is_file():
                spec = importlib.util.spec_from_file_location(f"_held_peg_{key}", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _MODULES[key] = module
                return module
    raise RuntimeError(f"{skill} is not installed beside this skill")


class Output(TypedDict):
    route: str
    detail: str
    stage: str
    refined_by: str
    lateral_error: float
    camera_correction: float
    search: str
    probes: int
    contact_depth: float
    hand_depth: float
    tip_depth: float
    target_depth: float
    slip: float
    peak_lag: float
    released: bool
    tip_in_tcp: dict[str, Any]
    hole_position: dict[str, float]
    elapsed_s: float


def _vec3(v: Any) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    return np.asarray(v, dtype=float).reshape(3)


def _vec(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def matrix(pose: dict[str, Any]) -> np.ndarray:
    pose = _pose(pose)
    q = pose.get("rotation") or pose.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([float(q[k]) for k in ("x", "y", "z", "w")]).as_matrix()
    out[:3, 3] = _vec3(pose["position"])
    return out


def pose_of(m: np.ndarray) -> dict[str, Any]:
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    return {
        "position": _vec(m[:3, 3]),
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _refusal(reply: Any) -> str:
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached", "contact"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False and reply.get("outcome") != "contact":
        return str(reply.get("reason") or "the move reported failure")
    return ""


def tip_in_tcp_from_held(held: dict[str, Any], peg: dict[str, Any] | None) -> np.ndarray:
    """The peg's tip frame in the tool frame (z out through the tip along the axis).

    The object frame of ``held.T_TCP_object`` has its origin on the peg's axis at
    mid-length; ``peg.axis_in_object`` and ``peg.tip_end`` say which way the tip is.
    Without a peg length the tip is ``held.tip_offset_m`` along the axis from the
    foot of the tool centre on it."""
    T = matrix(held["T_TCP_object"])
    peg = peg or {}
    axis = _vec3(peg.get("axis_in_object") or [0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    sign = -1.0 if str(peg.get("tip_end", "-z")).startswith("-") else 1.0
    direction = T[:3, :3] @ (sign * axis)
    if peg.get("length_m"):
        tip = T[:3, 3] + direction * 0.5 * float(peg["length_m"])
    else:
        foot = T[:3, 3] - direction * float(T[:3, 3] @ direction)
        tip = foot + direction * float(held.get("tip_offset_m", 0.0))
    out = np.eye(4)
    out[:3, :3] = _frame_with_z(direction)
    out[:3, 3] = tip
    return out


def _frame_with_z(z: np.ndarray) -> np.ndarray:
    z = z / np.linalg.norm(z)
    hint = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = hint - z * float(hint @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Kasa least-squares circle through 2-D points: (centre, radius)."""
    if len(points) < 3:
        return None
    A = np.column_stack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))])
    b = (points**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return sol[:2], float(math.sqrt(r2))


def cloud(view: dict[str, Any]) -> np.ndarray | None:
    """World points of a view's depth image (N x 3), or None without depth."""
    depth = view.get("depth")
    if depth is None:
        return None
    d = np.asarray(depth, dtype=float)
    if d.ndim == 3:
        d = d[..., 0]
    v, u = np.nonzero(np.isfinite(d) & (d > 0.0))
    if len(u) == 0:
        return None
    z = d[v, u]
    K = view["K"]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    T = view["T"]
    return cam @ T[:3, :3].T + T[:3, 3]


def read_tip_and_mouth(
    view: dict[str, Any],
    tip_pred: np.ndarray,
    mouth_pred: np.ndarray,
    axis: np.ndarray,
    peg_radius: float,
    hole_radius: float,
    search: float,
) -> dict[str, Any] | None:
    """The peg's tip and the hole's mouth centre seen in one depth view (world).

    The tip: depth points within peg_radius + search of the predicted peg line,
    above the mouth plane; the lowest 3 mm of them, their centroid pushed away from
    the camera by 2r/pi (a camera sees only the near half of the shaft). The mouth:
    points on the fixture's top face within hole_radius + search + 4 mm of the
    estimate; per 24 angular bins the innermost is the rim; a circle fit through
    the rims (the peg hides some of them)."""
    pts = cloud(view)
    if pts is None:
        return None
    R = _frame_with_z(axis)
    plane_h = (pts - mouth_pred) @ axis  # > 0 below the mouth plane
    lat = (pts - mouth_pred) @ R[:, :2]

    s = (pts - tip_pred) @ axis
    radial = np.linalg.norm((pts - tip_pred) - np.outer(s, axis), axis=1)
    on_peg = (radial <= peg_radius + search) & (s >= -0.03) & (s <= 0.01) & (plane_h < -0.002)
    out: dict[str, Any] = {}
    if on_peg.sum() >= CAM_MIN_POINTS:
        peg_pts, peg_s = pts[on_peg], s[on_peg]
        low = peg_pts[peg_s >= peg_s.max() - 0.003]
        if len(low) >= CAM_MIN_POINTS // 2:
            c = low.mean(axis=0)
            away = c - view["T"][:3, 3]
            away -= axis * float(away @ axis)
            n = float(np.linalg.norm(away))
            if n > 1e-9:
                c = c + away / n * (2.0 * peg_radius / math.pi)
            out["tip"] = c + axis * float(np.max((low - c) @ axis))

    ring = hole_radius + search + 0.004
    face = (np.abs(plane_h) <= CAM_RIM_BAND) & (np.linalg.norm(lat, axis=1) <= ring)
    if face.sum() >= CAM_MIN_POINTS:
        f = lat[face]
        ang = np.arctan2(f[:, 1], f[:, 0])
        rad = np.linalg.norm(f, axis=1)
        bins = np.floor((ang + math.pi) / (2 * math.pi) * 24).astype(int) % 24
        rims = [f[bins == b][np.argmin(rad[bins == b])] for b in range(24) if np.any(bins == b)]
        if len(rims) >= CAM_MIN_RIM_BINS:
            fit = fit_circle(np.asarray(rims))
            if fit is not None and abs(fit[1] - hole_radius) <= max(0.002, 0.3 * hole_radius):
                out["mouth"] = mouth_pred + R[:, :2] @ fit[0]
                out["rim_radius"] = fit[1]
    return out or None


def run(
    ctx: NodeContext,
    held: dict[str, Any],
    hole: dict[str, Any],
    peg: dict[str, Any] | None = None,
    insert_depth: float | None = None,
    standoff: float = STANDOFF,
    enter_depth: float = ENTER_DEPTH,
    mouth_band: float = MOUTH_BAND,
    depth_tolerance: float = DEPTH_TOLERANCE,
    slip_tolerance: float = SLIP_TOLERANCE,
    finger_clear: float = FINGER_CLEAR,
    finger_reach: float = FINGER_REACH,
    max_lag: float = MAX_LAG,
    camera: str | None = None,
    use_camera: bool = True,
    cam_steps: int = CAM_STEPS,
    max_probes: int = MAX_PROBES,
    tilt_deg: float = TILT_DEG,
    retreat: float = RETREAT,
    top_probe: bool = True,
    release: bool = True,
    check_reach: bool = True,
    settle_period: float | None = None,
    arm_id: int | None = None,
) -> Output:
    import time

    t0 = time.monotonic()
    if arm_id is None:
        arm_id = 1 if str(held.get("arm", "right")).lower().startswith("r") else 0
    kw_arm: dict[str, Any] = {"arm_id": int(arm_id)} if int(arm_id) else {}

    axis = _vec3(matrix(hole["T_world_mouth"])[:3, 2])
    axis /= np.linalg.norm(axis)
    mouth = matrix(hole["T_world_mouth"])[:3, 3]
    diameter = float(hole["diameter_m"])
    chamfer = float(hole.get("chamfer_m") or 0.0)
    peg_d = float((peg or {}).get("diameter_m") or diameter)
    clearance = max(0.0, diameter - peg_d)
    tip_T = tip_in_tcp_from_held(held, peg)
    tip_offset = float(np.linalg.norm(tip_T[:3, 3]))
    room = max(0.0, tip_offset - float(finger_clear))
    hole_depth = float(hole.get("depth_m") or 0.02)
    target = float(insert_depth) if insert_depth is not None else hole_depth - 0.002
    target = max(float(enter_depth), min(target, room))
    sigma = float(held.get("sigma_mm", 1.0)) * 1e-3

    out: Output = {
        "route": "unverified",
        "detail": "",
        "stage": "align",
        "refined_by": "arm",
        "lateral_error": -1.0,
        "camera_correction": 0.0,
        "search": "none",
        "probes": 0,
        "contact_depth": -1.0,
        "hand_depth": 0.0,
        "tip_depth": -1.0,
        "target_depth": target,
        "slip": 0.0,
        "peak_lag": 0.0,
        "released": False,
        "tip_in_tcp": pose_of(tip_T),
        "hole_position": _vec(mouth),
        "elapsed_s": 0.0,
    }

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, detail
        out["elapsed_s"] = round(time.monotonic() - t0, 2)
        logger.info("inserting-a-held-peg-top-down: %s at %s (%s)", route, out["stage"], detail)
        return out

    if float(axis @ np.array([0.0, 0.0, -1.0])) < math.cos(math.radians(30.0)):
        return finish("unreachable", "the hole's axis is more than 30 deg off straight down")

    def hand() -> np.ndarray | None:
        try:
            return matrix(ctx.tool("robot.get_ee_pose", **kw_arm))
        except Exception as exc:
            logger.info("inserting-a-held-peg-top-down: no pose reading (%s)", exc)
            return None

    def move(T: np.ndarray, cartesian: bool = True) -> str:
        name = "robot.go_to_pose_cartesian" if cartesian else "robot.go_to_pose"
        try:
            return _refusal(ctx.tool(name, pose=pose_of(T), **kw_arm))
        except Exception as exc:
            return str(exc) or "refused"

    def depth_of(H: np.ndarray, tipT: np.ndarray) -> float:
        return float(((H @ tipT)[:3, 3] - mouth) @ axis)

    # 1. Align the tip over the mouth on the hole's axis (inserting-a-peg-with-pose-refinement).
    align = sibling("align").run(
        ctx,
        hole_centre=_vec(mouth),
        hole_axis=_vec(axis),
        peg_in_tcp=pose_of(tip_T),
        standoff=float(standoff),
        lateral_tolerance=max(0.0005, min(0.002, clearance / 2 + chamfer)),
        check_reach=check_reach,
        arm_id=int(arm_id),
    )
    out["lateral_error"] = float(align.get("lateral_error", -1.0))
    if align["route"] == "unreachable":
        return finish("unreachable", f"align: {align.get('detail', '')}")
    pre = matrix(align["pre_insert_pose"])

    # 2. Refine from the wrist camera: tip and mouth in the same depth view.
    out["stage"] = "camera"
    if use_camera and int(cam_steps) > 0:
        try:
            views_of = sibling("views")._views
        except RuntimeError as exc:
            views_of = None
            logger.info("inserting-a-held-peg-top-down: no view parser (%s)", exc)
        total = np.zeros(3)
        for step in range(int(cam_steps) + 1):
            if views_of is None:
                break
            H = hand()
            if H is None:
                return finish("unreachable", "the hand's pose could not be read")
            try:
                views = views_of(ctx.tool("robot.get_observation"), [camera] if camera else None)
            except Exception as exc:
                logger.info("inserting-a-held-peg-top-down: no observation (%s)", exc)
                break
            if camera is None:
                side = "right" if int(arm_id) else "left"
                views = {k: v for k, v in views.items() if "wrist" in k and side in k} or {
                    k: v for k, v in views.items() if "wrist" in k
                }
            tip_pred = (H @ tip_T)[:3, 3]
            seen = None
            for v in views.values():
                seen = read_tip_and_mouth(
                    v, tip_pred, mouth, axis, peg_d / 2, diameter / 2, 3 * sigma + 0.003
                )
                if seen and "tip" in seen and "mouth" in seen:
                    break
            if not seen or "tip" not in seen or "mouth" not in seen:
                logger.info("inserting-a-held-peg-top-down: camera step %d saw %s", step, seen)
                break
            out["refined_by"] = "camera"
            # The tip seen shifts the registration (a tilted or offset hold); the mouth
            # seen replaces the estimate. Both lateral to the axis.
            tip_seen, mouth_seen = seen["tip"], seen["mouth"]
            d_tip = tip_seen - tip_pred
            d_tip -= axis * float(d_tip @ axis)
            if step == 0:
                tip_T = tip_T.copy()
                tip_T[:3, 3] += H[:3, :3].T @ d_tip
                out["tip_in_tcp"] = pose_of(tip_T)
            m_shift = mouth_seen - mouth
            m_shift -= axis * float(m_shift @ axis)
            if float(np.linalg.norm(m_shift)) > CAM_MAX_TOTAL:
                logger.info("inserting-a-held-peg-top-down: camera mouth %.1f mm off, ignored",
                            np.linalg.norm(m_shift) * 1e3)  # fmt: skip
                break
            err = mouth_seen - tip_seen
            err -= axis * float(err @ axis)
            out["lateral_error"] = float(np.linalg.norm(err))
            out["hole_position"] = _vec(mouth + m_shift)
            if out["lateral_error"] < max(clearance / 2, 1e-4) or step >= int(cam_steps):
                break
            n = float(np.linalg.norm(err))
            nudge = err * min(1.0, CAM_MAX_STEP / n)
            if float(np.linalg.norm(total + nudge)) > CAM_MAX_TOTAL:
                break
            total += nudge
            out["camera_correction"] = float(np.linalg.norm(total))
            T = pre.copy()
            T[:3, 3] += total
            why = move(T, cartesian=False)
            if why:
                return finish("unreachable", f"camera correction {step + 1} refused ({why})")
        if out["refined_by"] == "camera":
            mouth = _vec3(out["hole_position"])
            pre = pre.copy()
            pre[:3, 3] += total

    # 3. Descend with stop-on-contact (push_insert: steps, contact / deflection stop).
    out["stage"] = "descend"
    push = sibling("push")

    def descend(start: np.ndarray, depth: float) -> dict[str, Any]:
        return push.run(
            ctx,
            hole_centre=_vec(mouth),
            hole_axis=_vec(axis),
            peg_in_tcp=pose_of(tip_T),
            pre_insert_pose=pose_of(start),
            insert_depth=float(depth),
            min_depth=float(enter_depth),
            finger_clear=float(finger_clear),
            back_off=False,
            check_reach=check_reach,
            arm_id=int(arm_id),
        )

    first = descend(pre, target)
    if first["route"] == "unreachable":
        return finish("unreachable", f"descend: {first.get('detail', '')}")
    H = hand()
    if H is None:
        return finish("unreachable", "the hand's pose could not be read")
    out["contact_depth"] = depth_of(H, tip_T)
    entered = out["contact_depth"] >= float(enter_depth)

    if not entered:
        if out["contact_depth"] < -float(mouth_band):
            move(pre)
            return finish(
                "missed_hole",
                f"the tip stopped {-out['contact_depth'] * 1e3:.1f} mm over the mouth plane: "
                "something stands over the hole, or the hole estimate is high",
            )
        # 4. Blocked at the mouth: bounded spiral, then tilt-then-upright.
        out["stage"] = "search"
        out["search"] = "spiral"
        radius = min(SEARCH_RADIUS_MAX, max(SEARCH_RADIUS_MIN, 3 * sigma + 0.001))
        pitch = max(MIN_PITCH, math.sqrt(3.0) * (clearance / 2 + chamfer))
        grid = sibling("spiral").hex_grid(radius, pitch)[1 : int(max_probes) + 1]
        R = _frame_with_z(axis)
        for k, (dx, dy) in enumerate(grid):
            out["probes"] = k + 1
            shift = R[:, 0] * dx + R[:, 1] * dy
            start = pre.copy()
            start[:3, 3] += shift + axis * (float(standoff) - PROBE_HOVER)
            if move(start):
                continue
            res = descend(start, target)
            if res["route"] == "unreachable":
                continue
            H = hand()
            if H is None:
                return finish("unreachable", "the hand's pose could not be read")
            out["contact_depth"] = depth_of(H, tip_T)
            if out["contact_depth"] >= float(enter_depth):
                entered = True
                pre = pre.copy()
                pre[:3, 3] += shift
                break
            move(start)
        if not entered and chamfer >= TILT_MIN_CHAMFER and tilt_deg > 0:
            out["search"] = "spiral+tilt"
            for k, ang in enumerate((0.0, 90.0, 180.0, 270.0)):
                d = R[:, 0] * math.cos(math.radians(ang)) + R[:, 1] * math.sin(math.radians(ang))
                hinge = np.cross(axis, d)
                turn = Rotation.from_rotvec(hinge * math.radians(float(tilt_deg))).as_matrix()
                Hp = pre.copy()
                Hp[:3, 3] += axis * (float(standoff) - PROBE_HOVER)
                tipw = (Hp @ tip_T)[:3, 3]
                Ht = np.eye(4)
                Ht[:3, :3] = turn @ Hp[:3, :3]
                Ht[:3, 3] = tipw + turn @ (Hp[:3, 3] - tipw) + d * (clearance / 2 + chamfer)
                if move(Ht):
                    continue
                low = Ht.copy()
                low[:3, 3] += axis * (PROBE_HOVER + float(enter_depth) / 2)
                move(low)
                up = low.copy()
                tipw = (low @ tip_T)[:3, 3]
                up[:3, :3] = pre[:3, :3]
                up[:3, 3] = tipw + turn.T @ (low[:3, 3] - tipw) + axis * 0.001
                move(up)
                H = hand()
                if H is None:
                    return finish("unreachable", "the hand's pose could not be read")
                out["probes"] = len(grid) + k + 1
                out["contact_depth"] = depth_of(H, tip_T)
                if out["contact_depth"] >= float(enter_depth) - 0.001:
                    entered = True
                    pre = H.copy()
                    pre[:3, 3] -= axis * (out["contact_depth"] + float(standoff))
                    break
                move(Hp)
        if not entered:
            move(pre)
            return finish(
                "missed_hole",
                f"blocked at the mouth; {out['probes']} probe(s) ({out['search']}) "
                f"within {radius * 1e3:.1f} mm did not enter",
            )

    # 5. Push to the target depth with a lag limit (pressing-in-with-guarded-push).
    out["stage"] = "push"
    H = hand()
    if H is None:
        return finish("unreachable", "the hand's pose could not be read")
    now_depth = depth_of(H, tip_T)
    goal = H[:3, 3] + axis * max(0.0, target - now_depth)
    press_kw: dict[str, Any] = {}
    if settle_period is not None:
        press_kw.update(
            settle_period=float(settle_period), settle_timeout=10 * float(settle_period)
        )
    press = sibling("press").run(
        ctx,
        direction=_vec(axis),
        max_translation=max(0.0, target - now_depth) + 0.001,
        target=_vec(goal),
        goal_tolerance_if_lockup=float(depth_tolerance),
        max_lag=float(max_lag),
        check_reach=check_reach,
        arm_id=int(arm_id),
        **press_kw,
    )
    out["peak_lag"] = float(press.get("peak_lag", 0.0))
    H = hand()
    if H is None:
        return finish("unreachable", "the hand's pose could not be read")
    out["hand_depth"] = depth_of(H, tip_T)
    reached = out["hand_depth"] >= target - float(depth_tolerance)
    if press["route"] == "blocked" and not reached:
        return finish("unreachable", f"push: {press.get('detail', '')}")
    if not reached:
        out["stage"] = "release"
        if release:
            _open_and_back(ctx, kw_arm, H, axis, float(retreat), move, out)
        return finish(
            "jammed",
            f"entered {out['contact_depth'] * 1e3:.1f} mm, the push stopped at "
            f"{out['hand_depth'] * 1e3:.1f} of {target * 1e3:.1f} mm "
            f"({press['route']}, lag {out['peak_lag'] * 1e3:.1f} mm)",
        )

    # 6. Release, back out, and probe the peg's top: where the tip really is.
    out["stage"] = "release"
    if not release:
        out["tip_depth"] = out["hand_depth"]
        return finish("unverified", "depth reached by the hand; not released, nothing probed")
    held_at = H.copy()
    _open_and_back(ctx, kw_arm, H, axis, float(retreat), move, out)
    if not top_probe:
        return finish("unverified", "depth reached by the hand; the top was not probed")
    out["stage"] = "verify"
    top_expected = held_at @ tip_T
    length = float((peg or {}).get("length_m") or 0.0)
    if length <= 0.0:
        return finish("unverified", "depth reached by the hand; no peg length to probe its top")
    top = top_expected[:3, 3] - axis * length
    try:
        ctx.tool("robot.close_gripper", **kw_arm)
    except Exception as exc:
        logger.info("inserting-a-held-peg-top-down: close before probe failed (%s)", exc)
    start = held_at.copy()
    start[:3, 3] = top - axis * (finger_reach + TOP_PROBE_SPAN)
    if move(start):
        return finish("unverified", "depth reached by the hand; the probe start is out of reach")
    # The probe is a guarded push with closed fingers: resistance from tracking lag,
    # so it needs no contact outcome from the connector.
    probe = sibling("press").run(
        ctx,
        direction=_vec(axis),
        max_translation=2 * TOP_PROBE_SPAN,
        max_lag=PROBE_MAX_LAG,
        check_reach=check_reach,
        arm_id=int(arm_id),
        **press_kw,
    )
    probe_travel = float(probe.get("contact_travel", -1.0))
    move(start)
    if probe["route"] == "blocked":
        return finish("unverified", f"depth reached by the hand; top probe: {probe.get('detail')}")
    if probe_travel < 0.0:
        return finish(
            "unverified",
            "the hand reached the depth, but the probe met no peg top within "
            f"{TOP_PROBE_SPAN * 1e3:.0f} mm of where it should be (fell, or pushed through)",
        )
    rise = TOP_PROBE_SPAN - probe_travel  # > 0: the top stands higher than expected
    out["slip"] = float(rise)
    out["tip_depth"] = float(out["hand_depth"] - rise)
    if rise > float(slip_tolerance):
        return finish(
            "slipped_in_grip",
            f"the hand went {out['hand_depth'] * 1e3:.1f} mm in but the peg's top stands "
            f"{rise * 1e3:.1f} mm high: the tip is at {out['tip_depth'] * 1e3:.1f} mm",
        )
    return finish(
        "inserted",
        f"tip at {out['tip_depth'] * 1e3:.1f} mm of {target * 1e3:.1f} mm (top probe "
        f"{rise * 1e3:+.1f} mm), search {out['search']}, refined by {out['refined_by']}",
    )


def _open_and_back(
    ctx: NodeContext,
    kw_arm: dict[str, Any],
    H: np.ndarray,
    axis: np.ndarray,
    retreat: float,
    move: Any,
    out: Output,
) -> None:
    try:
        ctx.tool("robot.open_gripper", **kw_arm)
        out["released"] = True
    except Exception as exc:
        logger.info("inserting-a-held-peg-top-down: open failed (%s)", exc)
    back = H.copy()
    back[:3, 3] -= axis * retreat
    move(back)
