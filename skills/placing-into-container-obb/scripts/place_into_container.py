# Adapted from RoboRSI (https://github.com/nssmd/RoboRSI),
# roborsi/embodied/skills/base/place_obb/robotwin/policy.py and
# roborsi/embodied/skills/base/_lib/robotwin/_perception.py (object_mask,
# container_opening, container_drop_target), Apache-2.0.
# Kept: the holding precondition; the straight lift before anything else;
# segmenting the named container (best mask, or the one under a given pixel; a
# mask over 40% of the frame rejected), its masked depth cloud denoised and boxed;
# the opening read from the cloud's top band (20 mm under the rim), the box as
# the fallback when the band is degenerate; the drop candidates (the opening centre,
# then half-way to each edge) tried in order, released only when the hand reached
# the release pose (no release on or over the rim), then a straight retreat.
# Changed: the lift is 0.10 m (0.20 m was out of a YAM's reach) and the rim the
# cloud's 98th-percentile height (RoboRSI's 90th fell on the floor of a thin-walled
# box), the opening's centre and half extents the middle and half of the band's
# 10th-90th percentile span along each box axis (RoboRSI: the band's mean and the
# 90th-percentile distance from it, which a tilted view of one wall's inner face
# pulled 32 mm off-centre); Grounded-SAM and the simulator's camera are sam3.segment_text and
# geometry.mask_to_world_points on robot.get_observation; DBSCAN and the OBB are
# geometry.filter_noise and geometry.compute_obb (upright); the opening's half
# extents are measured along the box's axes (RoboRSI: world x / y) and shrunk by
# the wall thickness to the inner opening; the release height is over the
# container's floor (or contents), seen from depth, instead of RoboRSI's 50 mm
# under the rim; each candidate is moved inside the opening so the held object's
# footprint (from its box, at its offset in the hand) keeps a margin from the
# inner walls, and an object that fits nowhere routes no_room; the hand keeps its
# own rotation (RoboRSI: a fixed top-down quaternion); the approach goes over the
# candidate above the rim, then straight down; each pose is previewed with
# robot.solve_ik, refused or short moves are routed, and arrival is judged from
# robot.get_ee_pose; RoboRSI's depth containment check (_verify_deposit) is not
# ported: the outputs aperture_center and rim_z feed verifying-placement.
"""Put the object the hand holds into an open container the camera sees."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

logger = logging.getLogger(__name__)

LIFT_CLEARANCE = 0.10
"""Straight lift of the held object before it is carried anywhere [m]. RoboRSI's
_LIFT_CLEARANCE_M is 0.20 (CaP-X: clear the scene first); in the YAM station twin
a 0.20 m lift from 0.10 m over the table stopped 48 mm short at the edge of reach
and the cube turned 15 deg in the jaws, so half of it."""

SCORE_MIN = 0.25
"""Least sam3 score for the container's mask. sam3's notes put the usual cut at
about 0.3; in the YAM station twin "open grey box" scored 0.39 and the table 0.34
(checking-spatial-relations' sim check), so the mask-size reject below does the
rest."""

WHOLE_SCENE_FRACTION = 0.40
"""A mask covering more of the frame than this is the table, not the container:
RoboRSI's _WHOLE_SCENE_MASK_FRAC."""

MIN_POINTS = 30
"""Fewest container depth points to fit anything: RoboRSI's 30."""

DBSCAN_EPS = 0.005
DBSCAN_MIN_SAMPLES = 10
"""geometry.filter_noise's parameters: RoboRSI's (CaP-X's) DBSCAN eps 5 mm, 10 samples."""

RIM_PERCENTILE = 98.0
"""The rim is the container cloud's 98th-percentile height (geometry.compute_obb's
top). RoboRSI's container_opening takes the 90th: in the YAM station twin the
box's 2.3 mm walls were under 10% of the points and that put the rim on the floor."""

HALF_PERCENTILE = 90.0
RIM_BAND = 0.02
"""The opening is the points within 20 mm under the rim (RoboRSI's container_opening),
spanning their 10th to 90th percentile along each box axis: its centre the middle of
that span, its half extent half of it (at most the box's). RoboRSI centres on the
band's mean and takes the 90th-percentile distance from it; in the YAM station twin
the tilted top camera sees one wall's inner face inside the band, which pulled the
mean 32 mm toward that wall and read the 95 mm opening as 149 mm along it."""

MIN_RIM_POINTS = 6
"""Fewer rim-band points than this and the box stands in for the opening: RoboRSI's 6."""

WALL_THICKNESS = 0.003
"""Subtracted from the rim's half extents for the inner opening [m]. The YAM
station twin's box has 2.3 mm walls; a bin with thicker walls needs its own value."""

FOOTPRINT_MARGIN = 0.008
"""Least room kept between the held object's footprint and the inner walls [m]:
about one calibrated YAM station camera error (RMS 5.4 mm) plus a 2-3 mm sway of
the object in the jaws."""

FLOOR_INSET = 0.005
"""Floor points are taken this far inside the inner opening [m], away from the
foot of the walls."""

MIN_FLOOR_POINTS = 10
"""Fewer visible floor points than this and the floor is the box's bottom plus a wall."""

FLOOR_PERCENTILE = 10.0
"""The floor (or the top of what lies in the container) is the 10th-percentile height
of the visible floor points: robust to a few low flying pixels."""

RELEASE_HEIGHT = 0.01
"""The held object's bottom over the floor when the hand opens [m]: low enough
that a 30 mm cube does not bounce out of a 40 mm box, high enough that a
floor estimate one camera error (5.4 mm RMS) high does not press it down."""

APPROACH_CLEARANCE = 0.03
"""The held object's bottom over the rim while the hand comes over a drop point [m]."""

MIN_TOOL_HEIGHT = 0.015
"""Lowest tool centre over the floor [m]: closed linear_4310 fingertips reach
about 10 mm under it, and a real YAM station refuses a tool centre under 14 mm
over its table."""

RETREAT_HEIGHT = 0.08
"""The hand's straight retreat after the release [m]: RoboRSI's +0.08."""

REACH_TOLERANCE = 0.015
"""How far the hand may end from a pose [m]: no tighter than a real YAM station's
12 mm per move (a servo arrives within 4 mm)."""

EMPTY_FRACTION = 0.04
"""A jaw opening (robot.get_gripper, 1.0 = open) at or under this holds nothing:
3.8 mm on a 94 mm YAM span (handing-over-between-arms)."""

OPEN_FRACTION = 0.5
"""After open_gripper, a jaw opening under this did not open."""

CANDIDATE_FRACTION = 0.5
"""Drop candidates after the centre sit this fraction of the way to each inner
edge (RoboRSI: half-way), pulled in until the object keeps its margin."""


class Output(TypedDict):
    route: str
    detail: str
    container_obb: dict[str, Any]
    aperture_center: dict[str, float]
    rim_z: float
    floor_z: float
    opening_obb: dict[str, Any]
    opening_source: str
    release_position: dict[str, float]
    candidate: str


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


def _rotation(q: dict[str, float] | None) -> np.ndarray:
    q = q or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _points(reply: Any) -> np.ndarray:
    """An (N, 3) array from a geometry tool's {"points": PointCloud} reply."""
    cloud = reply.get("points", reply) if isinstance(reply, dict) else reply
    if isinstance(cloud, dict):
        cloud = cloud.get("points", [])
    return np.asarray(cloud, dtype=np.float64).reshape(-1, 3)


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any] | None:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    cameras = [c for c in cameras if c.get("depth") is not None]
    if name:
        cameras = [c for c in cameras if c.get("name") == name]
    return cameras[0] if cameras else None


def container_opening(
    points: np.ndarray,
    obb: dict[str, Any],
    wall_thickness: float = WALL_THICKNESS,
    rim_band: float = RIM_BAND,
) -> dict[str, Any]:
    """The container's opening from its world cloud and upright box: rim height,
    opening centre (rim-band xy centroid), the inner opening's half extents along
    the box's two horizontal axes, the floor height, and where they came from."""
    centre = np.array([float(obb["center"][k]) for k in ("x", "y", "z")])
    half = np.array([float(obb["extent"][k]) for k in ("x", "y", "z")])
    rot = _rotation(obb.get("orientation"))
    axes = [rot[:, 0].copy(), rot[:, 1].copy()]
    for a in axes:
        a[2] = 0.0
        a /= max(float(np.linalg.norm(a)), 1e-9)
    source = "obb"
    rim_z = float(centre[2] + half[2])
    c_xy = centre[:2].copy()
    outer = half[:2].copy()
    if len(points) >= 12:
        top = float(np.percentile(points[:, 2], RIM_PERCENTILE))
        rim = points[points[:, 2] >= top - float(rim_band)]
        if len(rim) >= MIN_RIM_POINTS:
            source = "rim"
            rim_z = top
            m = rim[:, :2].mean(axis=0)
            d = rim[:, :2] - m
            lo, hi = (
                np.array([float(np.percentile(d @ a[:2], q)) for a in axes])
                for q in (100.0 - HALF_PERCENTILE, HALF_PERCENTILE)
            )
            c_xy = m + 0.5 * (lo[0] + hi[0]) * axes[0][:2] + 0.5 * (lo[1] + hi[1]) * axes[1][:2]
            outer = np.minimum(0.5 * (hi - lo), half[:2])
    inner = np.maximum(outer - float(wall_thickness), 0.0)
    floor_z = float(centre[2] - half[2]) + float(wall_thickness)
    floor_source = "box bottom + wall"
    if len(points):
        d = points[:, :2] - c_xy
        inside = (
            (np.abs(d @ axes[0][:2]) < inner[0] - FLOOR_INSET)
            & (np.abs(d @ axes[1][:2]) < inner[1] - FLOOR_INSET)
            & (points[:, 2] < rim_z)
        )
        if int(inside.sum()) >= MIN_FLOOR_POINTS:
            floor_z = float(np.percentile(points[inside, 2], FLOOR_PERCENTILE))
            floor_source = "seen"
    return {
        "source": source,
        "rim_z": rim_z,
        "center_xy": c_xy,
        "axes": axes,
        "inner_half": inner,
        "floor_z": floor_z,
        "floor_source": floor_source,
    }


def footprint_reach(held_obb: dict[str, Any], axis: np.ndarray) -> float:
    """How far the held object's box reaches from its centre along a horizontal axis."""
    half = np.array([float(held_obb["extent"][k]) for k in ("x", "y", "z")])
    rot = _rotation(held_obb.get("orientation"))
    return float(np.abs(rot.T @ axis) @ half)


def _bottom_z(held_obb: dict[str, Any]) -> float:
    half = np.array([float(held_obb["extent"][k]) for k in ("x", "y", "z")])
    rot = _rotation(held_obb.get("orientation"))
    return float(held_obb["center"]["z"]) - float(np.abs(rot[2, :]) @ half)


def drop_candidates(
    inner_half: np.ndarray, reach: np.ndarray, margin: float
) -> list[tuple[str, float, float]]:
    """RoboRSI's drop points (centre, then half-way to each edge) as offsets along
    the opening's two axes, each pulled in so the footprint keeps ``margin`` from
    the inner walls; empty when the object fits nowhere."""
    room = inner_half - reach - float(margin)
    if float(room.min()) < 0.0:
        return []
    out = [("centre", 0.0, 0.0)]
    for i, name in ((0, "u"), (1, "v")):
        step = min(CANDIDATE_FRACTION * float(inner_half[i]), float(room[i]))
        if step < 1e-3:
            continue
        for sign in (-1.0, 1.0):
            off = [0.0, 0.0]
            off[i] = sign * step
            out.append((f"{'+' if sign > 0 else '-'}{name}", off[0], off[1]))
    return out


def run(
    ctx: NodeContext,
    container_description: str,
    held_obb: dict[str, Any],
    camera_name: str = "",
    container_pixel: list[int] | None = None,
    lift_clearance: float = LIFT_CLEARANCE,
    release_height: float = RELEASE_HEIGHT,
    approach_clearance: float = APPROACH_CLEARANCE,
    footprint_margin: float = FOOTPRINT_MARGIN,
    wall_thickness: float = WALL_THICKNESS,
    min_tool_height: float = MIN_TOOL_HEIGHT,
    retreat_height: float = RETREAT_HEIGHT,
    reach_tolerance: float = REACH_TOLERANCE,
    score_min: float = SCORE_MIN,
    empty_fraction: float = EMPTY_FRACTION,
    check_reach: bool = True,
    arm_id: int = 0,
) -> Output:
    on_arm: dict[str, Any] = {"arm_id": int(arm_id)} if arm_id else {}
    out: Output = {
        "route": "unreachable",
        "detail": "",
        "container_obb": {},
        "aperture_center": {},
        "rim_z": 0.0,
        "floor_z": 0.0,
        "opening_obb": {},
        "opening_source": "",
        "release_position": {},
        "candidate": "",
    }
    notes: list[str] = []

    def finish(route: str, detail: str) -> Output:
        out["route"], out["detail"] = route, "; ".join(d for d in (detail, *notes) if d)
        return out

    def here() -> dict[str, Any]:
        return _pose(ctx.tool("robot.get_ee_pose", **on_arm))

    def go(tool: str, position: np.ndarray, rotation: dict[str, float]) -> str:
        """Send the hand and judge arrival from a fresh reading: "" when it ended
        within reach_tolerance, otherwise why not."""
        try:
            why = _refusal(
                ctx.tool(tool, pose={"position": _vec(position), "rotation": rotation}, **on_arm)
            )
        except Exception as exc:  # a refused move is routed, not raised
            why = str(exc) or type(exc).__name__
        off = float(np.linalg.norm(_xyz(here()) - position))
        if off > float(reach_tolerance):
            return f"{why or 'ended off'}; the hand stopped {off * 1e3:.0f} mm from it"
        return ""

    def reachable(position: np.ndarray, rotation: dict[str, float]) -> bool:
        if not check_reach:
            return True
        try:
            reply = ctx.tool(
                "robot.solve_ik", pose={"position": _vec(position), "rotation": rotation}, **on_arm
            )
        except Exception as exc:  # no preview served: let the move decide
            logger.info("placing-into-container-obb: no IK preview (%s)", exc)
            return True
        # A stall at the edge of reach reads like a stop on the rim.
        return not (
            isinstance(reply, dict)
            and (reply.get("success") is False or reply.get("within_servo_tolerance") is False)
        )

    # 1. Holding? Jaws at the stop closed on nothing.
    try:
        fraction = float(ctx.tool("robot.get_gripper", **on_arm)["position"])
    except Exception as exc:  # an unreadable hand is not an empty one
        logger.warning("placing-into-container-obb: jaws unreadable (%s)", exc)
        fraction = -1.0
    if 0.0 <= fraction <= float(empty_fraction):
        return finish("not_holding", f"the jaws read {fraction:.3f}: they hold nothing")

    start = here()
    tcp0 = _xyz(start)
    obj_c = np.array([float(held_obb["center"][k]) for k in ("x", "y", "z")])
    offset_xy = obj_c[:2] - tcp0[:2]
    hang = float(tcp0[2] - _bottom_z(held_obb))

    # 2. Lift straight up (RoboRSI goes on whether or not the lift arrived).
    if float(lift_clearance) > 0.0:
        why = go(
            "robot.go_to_pose_cartesian",
            tcp0 + np.array([0.0, 0.0, float(lift_clearance)]),
            start["rotation"],
        )
        if why:
            notes.append(f"lift: {why}")
    lifted = here()
    rotation = lifted["rotation"]
    turn = _rotation(rotation) @ _rotation(start["rotation"]).T
    offset_xy = (turn @ np.array([*offset_xy, 0.0]))[:2]

    # 3. The container: mask, cloud, box.
    try:
        camera = _camera(ctx.tool("robot.get_observation"), camera_name)
    except Exception as exc:
        return finish("container_not_found", f"no observation ({exc})")
    if camera is None:
        return finish("container_not_found", f"no RGB-D camera named {camera_name or '(any)'!r}")
    found = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=container_description, max_results=3
    )
    masks = [np.asarray(m) for m in (found.get("masks") or [])]
    scores = [float(s) for s in (found.get("scores") or [])]
    scores += [0.0] * (len(masks) - len(scores))
    mask = None
    if masks and container_pixel is not None:
        u, v = int(container_pixel[0]), int(container_pixel[1])
        under = [m for m in masks if 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and m[v, u] > 0]
        mask = under[0] if under else None
    elif masks and scores[0] >= float(score_min):
        mask = masks[0]
    if mask is None:
        return finish(
            "container_not_found",
            f"sam3 found no {container_description!r}"
            + (f" (best score {scores[0]:.2f} < {score_min:.2f})" if masks else "")
            + (" under the given pixel" if masks and container_pixel is not None else ""),
        )
    frac = float((mask > 0).mean())
    if frac > WHOLE_SCENE_FRACTION:
        return finish(
            "container_not_found",
            f"the {container_description!r} mask covers {frac:.0%} of the frame: the table, not a container",
        )
    pts = _points(
        ctx.tool(
            "geometry.mask_to_world_points",
            mask=(mask > 0).astype(np.uint8) * 255,
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )
    )
    if len(pts) < MIN_POINTS:
        return finish(
            "container_not_found",
            f"only {len(pts)} depth points on the {container_description!r} (< {MIN_POINTS})",
        )
    try:
        pts = _points(
            ctx.tool(
                "geometry.filter_noise", points=pts, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
            )
        )
    except Exception as exc:  # unfiltered is still a cloud
        notes.append(f"no noise filter ({exc})")
    obb = ctx.tool("geometry.compute_obb", points=pts)["obb"]
    out["container_obb"] = obb
    op = container_opening(pts, obb, wall_thickness=float(wall_thickness))
    rim_z, floor_z = float(op["rim_z"]), float(op["floor_z"])
    c_xy, axes, inner = op["center_xy"], op["axes"], op["inner_half"]
    out["rim_z"], out["floor_z"] = rim_z, floor_z
    out["aperture_center"] = _vec(np.array([c_xy[0], c_xy[1], rim_z]))
    out["opening_obb"] = {
        "center": _vec(np.array([c_xy[0], c_xy[1], 0.5 * (rim_z + floor_z)])),
        "extent": _vec(np.array([inner[0], inner[1], max(0.5 * (rim_z - floor_z), 0.0)])),
        "orientation": obb.get("orientation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    out["opening_source"] = op["source"]
    seen = (
        f"opening {inner[0] * 2e3:.0f} x {inner[1] * 2e3:.0f} mm from the {op['source']}, "
        f"rim {rim_z:.3f}, floor {floor_z:.3f} ({op['floor_source']})"
    )

    # 4. Where the object fits.
    reach = np.array([footprint_reach(held_obb, a) for a in axes])
    cands = drop_candidates(inner, reach, float(footprint_margin))
    if not cands:
        return finish(
            "no_room",
            f"the held object needs {reach[0] * 2e3:.0f} x {reach[1] * 2e3:.0f} mm "
            f"plus {footprint_margin * 1e3:.0f} mm a side; {seen}; still holding",
        )

    # 5. Over each candidate above the rim, straight down, release only on arrival.
    bottom_release = floor_z + float(release_height)
    tcp_z = max(bottom_release + hang, floor_z + float(min_tool_height))
    if tcp_z - hang > bottom_release + 1e-4:
        notes.append(
            f"released {(tcp_z - hang - floor_z) * 1e3:.0f} mm over the floor to keep the tool "
            f"centre {min_tool_height * 1e3:.0f} mm up"
        )
    tried: list[str] = []
    for name, du, dv in cands:
        obj_xy = c_xy + du * axes[0][:2] + dv * axes[1][:2]
        hand_xy = obj_xy - offset_xy
        release = np.array([hand_xy[0], hand_xy[1], tcp_z])
        above = np.array([hand_xy[0], hand_xy[1], rim_z + float(approach_clearance) + hang])
        if not (reachable(above, rotation) and reachable(release, rotation)):
            tried.append(f"{name}: out of reach (IK preview)")
            continue
        why = go("robot.go_to_pose", above, rotation)
        if why:
            tried.append(f"{name}: over the drop point: {why}")
            continue
        why = go("robot.go_to_pose_cartesian", release, rotation)
        if why:
            back = go("robot.go_to_pose_cartesian", above, rotation)
            tried.append(f"{name}: descent: {why}" + (f"; back up: {back}" if back else ""))
            continue
        at = _xyz(here())
        out["candidate"] = name
        out["release_position"] = _vec(np.array([*(at[:2] + offset_xy), at[2] - hang]))
        opened = ctx.tool("robot.open_gripper", **on_arm)
        jaw = float(opened.get("position", -1.0)) if isinstance(opened, dict) else -1.0
        if jaw < 0.0:
            try:
                jaw = float(ctx.tool("robot.get_gripper", **on_arm)["position"])
            except Exception:  # unreadable: take the open as done
                jaw = -1.0
        if 0.0 <= jaw < OPEN_FRACTION:
            return finish(
                "not_released",
                f"at the {name} drop point the jaws read {jaw:.3f} after opening; {seen}",
            )
        why = go(
            "robot.go_to_pose_cartesian", at + np.array([0.0, 0.0, float(retreat_height)]), rotation
        )
        if why:
            notes.append(f"retreat: {why}")
        if tried:
            notes.append("before: " + "; ".join(tried))
        return finish(
            "placed",
            f"released at the {name} drop point, the object's bottom "
            f"{(at[2] - hang - floor_z) * 1e3:.0f} mm over the floor; {seen}",
        )
    return finish(
        "unreachable",
        "no drop point admitted the descent (" + "; ".join(tried) + f"); {seen}; still holding",
    )
