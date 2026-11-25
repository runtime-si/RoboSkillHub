# Generalized from the YAM pen station's yam_gap/pens/tools/pen_shape_probe.py (look poses at 30/60/75/85
# deg tilt and 0.15-0.25 m round each target, an RGB-D frame and its camera pose per view,
# nothing touched) and the analysis of its capture pen_shape_20261109-001140 (each cap's
# points by depth under its top, p90 radius per band); the look pose is the probe's
# steps.look_pose with the camera aimed instead of a pinch, the points station.points().
"""Carry the wrist RGB-D camera to each doubted place in the twin, look from several angles
and distances without touching anything, and measure what was doubted."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doubt_geometry as geo  # noqa: E402

logger = logging.getLogger(__name__)

SKILL = "resolving-reconstruction-doubts-with-wrist-views"

TILTS_DEG = (30.0, 60.0, 75.0, 85.0)
"""Camera tilts off straight down [deg], the pen probe's: 30 sees the top round (the
top height and the knob), 60-75 the side and the top's rim (the cap's diameter), 85 the
side from level (lean, a holder's wall). tuned_on: pen-shape capture 2026-11-09, where
the views that read each band round came from 60-85 and the top from 30-75."""

DISTANCES_M = (0.15, 0.20, 0.25)
"""Camera-to-target ranges tried per tilt, nearest first [m]: the probe's. A D405 at
640 x 480 sees 1 mm as ~2.6 px at 0.15 m and ~1.6 px at 0.25 m."""

AZIMUTHS_DEG = (0.0,)
"""Azimuth offsets of the views from the arm's own side of the target [deg]. The probe
looked from one side only; give (0, 60, -60) to see a footprint or a lean round."""

MIN_TOOL_OVER_SURFACE = 0.12
"""Lowest tool centre over the surface [m]: the probe skipped any look pose under
table_z + 0.12 (steps.look_at), and none of its 11 views touched anything."""

STANDOFF_MARGIN = 0.03
"""Added to the fingers' reach (robot.describe_gripper) for the closest the tool centre
may come to any doubted point [m]. tuned_on: pen probe, closest look (mid_cap, 75 deg,
0.15 m) put the tool centre 41 mm from its target with the fingers' reach 10 mm: 40 mm."""

DEFAULT_FINGER_REACH = 0.0102
"""Fingertips below the tool centre when robot.describe_gripper does not say [m]: the
YAM linear_4310's TIP_REACH."""

TRANSIT_OVER_TARGETS = 0.10
"""Height over the highest doubted point at which the hand crosses between views [m].
Not tuned: the probe moved straight between planned poses; this keeps the open fingers
(~10 mm under the tool centre) 9 cm over anything doubted while the arm travels."""

REACH_TOLERANCE = 0.015
"""How far a hand may end from its view pose and the view still count [m]: above a real
YAM station's 12 mm early answer (runs/real-v1); the frame is posed from where the arm
actually stands, and the target must still project in the image."""

VIEW_MARGIN = 0.08
"""The target must project at least this share of the image size inside its border."""

SETTLE_READS = 5
SETTLE_DISTANCE = 0.0005
SETTLE_ANGLE_DEG = 0.1
"""A frame is taken once two observations agree on the tool within 0.5 mm / 0.1 deg (at
most 5 extra reads): the real YAM server answers a move with the tool still 3-11 mm off
and closes with a 0.26 s time constant (settle_probe 2026-11-08); a frame posed from a
moving arm reads the object a few mm off."""

ROUTE_ORDER = ("inconsistent", "no_view", "not_reachable")
"""The skill's route when a doubt is not measured: the first of these any doubt has."""


class Output(TypedDict):
    route: str
    doubts: list[dict[str, Any]]
    frames: list[dict[str, Any]]
    twin_updates: list[dict[str, Any]]
    n_doubts: int
    n_measured: int
    n_views: int
    detail: str


# ----------------------------------------------------------------------------- poses


def _pose(reply: Any) -> dict[str, Any]:
    return reply.get("pose", reply) if isinstance(reply, dict) else reply


def _xyz(pose: dict[str, Any]) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _vec(v) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _rot(q: dict[str, float]) -> np.ndarray:
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


def _quat(r: np.ndarray) -> dict[str, float]:
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


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rot(pose["rotation"])
    m[:3, 3] = _xyz(pose)
    return m


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def _refusal(reply: Any) -> str:
    """Why a move's reply says it did not arrive, or "" (connectors that route a refusal
    instead of raising)."""
    if not isinstance(reply, dict):
        return ""
    if reply.get("outcome") not in (None, "reached"):
        return f"{reply['outcome']}: {reply.get('reason', '')}".rstrip(": ")
    if reply.get("reached") is False or reply.get("success") is False:
        return str(reply.get("reason") or reply.get("message") or "the move reported failure")
    return ""


def _wrist_camera(obs: dict[str, Any], tool: np.ndarray, name: str | None) -> dict[str, Any] | None:
    """The named camera, else the eye-in-hand / wrist camera riding nearest the tool."""
    cams = [c for c in obs.get("cameras") or [] if c.get("pose") is not None]
    if name:
        return next((c for c in cams if c.get("name") == name), None)
    wrist = [c for c in cams if any(k in str(c.get("name", "")) for k in ("eye_in_hand", "wrist"))]
    if not wrist:
        return None
    return min(wrist, key=lambda c: float(np.linalg.norm(_xyz(c["pose"]) - tool[:3, 3])))


def _settled(ctx: NodeContext, arm: int) -> dict[str, Any]:
    obs = ctx.tool("robot.get_observation")
    for _ in range(SETTLE_READS):
        nxt = ctx.tool("robot.get_observation")
        a, b = _matrix(obs["arms"][arm]["ee_pose"]), _matrix(nxt["arms"][arm]["ee_pose"])
        obs = nxt
        if (
            float(np.linalg.norm(a[:3, 3] - b[:3, 3])) <= SETTLE_DISTANCE
            and _angle_deg(a[:3, :3], b[:3, :3]) <= SETTLE_ANGLE_DEG
        ):
            break
    else:
        logger.info("%s: arm %d still moving after %d reads", SKILL, arm, SETTLE_READS)
    return obs


def _on_arm(arm: int) -> dict[str, Any]:
    return {"arm_id": int(arm)} if arm else {}


def _go(ctx: NodeContext, arm: int, position: np.ndarray, rotation: np.ndarray) -> str:
    pose = {"position": _vec(position), "rotation": _quat(rotation)}
    try:
        return _refusal(ctx.tool("robot.go_to_pose", pose=pose, **_on_arm(arm)))
    except Exception as exc:  # a refused move is routed, not raised
        return str(exc) or type(exc).__name__


def _travel(
    ctx: NodeContext, arm: int, position: np.ndarray, rotation: np.ndarray, transit_z: float
) -> str:
    """Up in place to the transit height, across at it, then down onto the pose: the hand
    never sweeps through a doubted object between views. Returns why it stopped, or ""."""
    now = _pose(ctx.tool("robot.get_ee_pose", **_on_arm(arm)))
    p0, r0 = _xyz(now), _rot(now["rotation"])
    legs = []
    if p0[2] < transit_z - 1.0e-3:
        legs.append(("rise", np.array([p0[0], p0[1], transit_z]), r0))
    if position[2] < transit_z - 1.0e-3 and np.linalg.norm(position[:2] - p0[:2]) > 1.0e-3:
        legs.append(("cross", np.array([position[0], position[1], transit_z]), rotation))
    legs.append(("view", position, rotation))
    for name, p, r in legs:
        why = _go(ctx, arm, p, r)
        if why:
            return f"{name}: {why}"
    return ""


# ----------------------------------------------------------------------------- looking


def _arm_rig(ctx: NodeContext, obs: dict[str, Any], arm: int, camera: str | None):
    tool = _matrix(obs["arms"][arm]["ee_pose"])
    cam = _wrist_camera(obs, tool, camera)
    if cam is None:
        return None
    return {
        "arm": arm,
        "start": tool,
        "camera": str(cam.get("name", "")),
        "tool_cam": np.linalg.inv(tool) @ _matrix(cam["pose"]),
    }


def _standoff(ctx: NodeContext, arm: int) -> float:
    reach = DEFAULT_FINGER_REACH
    try:
        hand = ctx.tool("robot.describe_gripper", **_on_arm(arm)) or {}
        reach = float((hand.get("finger") or {}).get("reach_m") or reach)
    except Exception as exc:  # optional: the YAM's reach stands in
        logger.info("%s: no gripper description (%s)", SKILL, exc)
    return reach + STANDOFF_MARGIN


def _look(
    ctx: NodeContext,
    rig: dict[str, Any],
    doubt: dict[str, Any],
    targets: list[np.ndarray],
    surface_z: float | None,
    transit_z: float,
    plan: dict[str, Any],
    failures: list[str],
) -> list[dict[str, Any]]:
    """Every view of one doubt one arm can take: per tilt and azimuth, the nearest range
    whose pose clears the surface and every doubted point, passes the IK preview, is
    reached, and shows the target."""
    arm, tool_cam = rig["arm"], rig["tool_cam"]
    target = np.asarray(doubt["point"], dtype=float)
    standoff = _standoff(ctx, arm)
    side = rig["start"][:3, 3][:2] - target[:2]
    az0 = (
        math.degrees(math.atan2(side[1], side[0]))
        if np.linalg.norm(side) > 0.01
        else (math.degrees(math.atan2(-target[1], -target[0])))
    )
    hand = rig["start"][:3, :3]
    views: list[dict[str, Any]] = []
    for tilt in plan["tilts"]:
        for daz in plan["azimuths"]:
            if len(views) >= plan["max_views"]:
                return views
            az = az0 + float(daz)
            tag = f"{doubt['id']} arm {arm} tilt {tilt:.0f} az {daz:+.0f}"
            for dist in plan["distances"]:
                pos, rot, eye = geo.look_pose(target, tilt, az, dist, hand, tool_cam)
                if surface_z is not None and pos[2] < surface_z + MIN_TOOL_OVER_SURFACE:
                    failures.append(f"{tag} {dist:.2f} m: tool under the surface floor")
                    continue
                near = min(float(np.linalg.norm(pos - t)) for t in targets)
                if near < standoff:
                    failures.append(
                        f"{tag} {dist:.2f} m: tool {near * 1e3:.0f} mm from a doubted point"
                    )
                    continue
                pose = {"position": _vec(pos), "rotation": _quat(rot)}
                if plan["check_reach"]:
                    try:
                        reply = ctx.tool("robot.solve_ik", pose=pose, **_on_arm(arm))
                    except Exception as exc:
                        failures.append(f"{tag} {dist:.2f} m: IK preview refused ({exc})")
                        continue
                    if isinstance(reply, dict) and (
                        reply.get("success") is False
                        or reply.get("within_servo_tolerance") is False
                    ):
                        failures.append(f"{tag} {dist:.2f} m: out of reach (IK preview)")
                        continue
                rig["moved"] = True
                why = _travel(ctx, arm, pos, rot, transit_z)
                if why:
                    failures.append(f"{tag} {dist:.2f} m: {why}")
                    continue
                obs = _settled(ctx, arm)
                tool = _matrix(obs["arms"][arm]["ee_pose"])
                off = float(np.linalg.norm(tool[:3, 3] - pos))
                if off > plan["reach_tolerance"]:
                    failures.append(f"{tag} {dist:.2f} m: the hand ended {off * 1e3:.0f} mm off")
                    continue
                cam = _wrist_camera(obs, tool, rig["camera"])
                if cam is None:
                    failures.append(
                        f"{tag}: the wrist camera {rig['camera']!r} left the observation"
                    )
                    break
                K, T = np.asarray(cam["intrinsics"], dtype=float), _matrix(cam["pose"])
                h, w = np.asarray(cam["rgb"]).shape[:2]
                u, v, z = geo.project(K, T, target)
                mx, my = VIEW_MARGIN * w, VIEW_MARGIN * h
                if z <= 0.0 or not (mx <= u <= w - mx and my <= v <= h - my):
                    failures.append(
                        f"{tag} {dist:.2f} m: target outside the image ({u:.0f}, {v:.0f})"
                    )
                    continue
                views.append(
                    {
                        "id": f"{doubt['id']}_a{arm}_t{tilt:.0f}_az{daz:+.0f}_d{dist * 100:.0f}",
                        "doubt": doubt["id"],
                        "arm_id": arm,
                        "camera": rig["camera"],
                        "tilt_deg": float(tilt),
                        "azimuth_deg": float(az),
                        "distance_m": float(dist),
                        "camera_pose": cam["pose"],
                        "tool_pose": obs["arms"][arm]["ee_pose"],
                        "target_px": [u, v],
                        "rgb": np.asarray(cam["rgb"]),
                        "depth": np.asarray(cam["depth"], dtype=np.float32),
                        "K": K,
                        "T": T,
                    }
                )
                break
    return views


def _mask(ctx: NodeContext, segmenter: str, view: dict[str, Any], name: str, notes: list[str]):
    """The segmenter's mask of ``name`` holding (or nearest) the target's pixel, or None."""
    try:
        reply = ctx.tool(segmenter, image=view["rgb"], query=name, max_results=5)
    except Exception as exc:
        notes.append(f"{view['id']}: {segmenter} failed ({exc}); colour and crop only")
        return None
    masks = [np.asarray(m) > 0 for m in (reply or {}).get("masks") or []]
    if not masks:
        notes.append(f"{view['id']}: {segmenter} found no {name!r}")
        return np.zeros(np.asarray(view["rgb"]).shape[:2], dtype=bool)
    u, v = view["target_px"]

    def miss(m: np.ndarray) -> float:
        rows, cols = np.nonzero(m)
        return float(np.min(np.hypot(cols - u, rows - v))) if len(rows) else math.inf

    return min(masks, key=miss)


# ----------------------------------------------------------------------------- write-back


def _set(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        tree = tree.setdefault(k, {})
    tree[keys[-1]] = value


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _updates(result: dict[str, Any], doubt: dict[str, Any], files: dict[str, str], stamp: str):
    """One twin update per quantity read, each naming the views and frames it came from."""
    out = []
    for name, qv in result["quantities"].items():
        value = qv["value"]
        if name == "profile":
            value = [
                {"from_mm": b["from_mm"], "to_mm": b["to_mm"], "diameter_m": b["diameter"]}
                for b in value
                if b["diameter"] is not None
            ]
        views = qv.get("from_views") or qv.get("views") or []
        out.append(
            {
                "doubt": doubt["id"],
                "quantity": name,
                "value": _plain(value),
                "unit": qv.get("unit", ""),
                "prior": _plain((doubt.get("prior") or {}).get(name)),
                "write_to": (doubt.get("write") or {}).get(name),
                "consistent": bool(qv.get("consistent", True)),
                "checked": bool(qv.get("checked", False)),
                "source": {
                    "skill": SKILL,
                    "measured_at": stamp,
                    "views": list(views),
                    "agreeing_views": list(qv.get("views") or []),
                    "frames": [files[v] for v in views if v in files],
                },
            }
        )
    return out


def _write_twin(path: str, updates: list[dict[str, Any]], doubts: list[dict[str, Any]]) -> int:
    """Merge the measured doubts' updates into the twin JSON: every one under
    ``measured.<doubt>.<quantity>`` with its source, and the value also at the doubt's
    ``write`` key when it names one. Returns the values written to ``write`` keys."""
    p = Path(path)
    twin = json.loads(p.read_text()) if p.exists() else {}
    status = {d["id"]: d["status"] for d in doubts}
    written = 0
    for upd in updates:
        if status.get(upd["doubt"]) != "measured" or not upd["consistent"]:
            continue
        twin.setdefault("measured", {}).setdefault(upd["doubt"], {})[upd["quantity"]] = {
            k: upd[k] for k in ("value", "unit", "prior", "checked", "source")
        }
        if upd["write_to"]:
            _set(twin, upd["write_to"], upd["value"])
            written += 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(twin, indent=1))
    return written


# ----------------------------------------------------------------------------- run


def run(
    ctx: NodeContext,
    doubts: list[dict[str, Any]],
    arm_ids: list[int] | None = None,
    arm_id: int = 0,
    camera: str | None = None,
    surface_z: float | None = None,
    tilts_deg: list[float] | None = None,
    distances_m: list[float] | None = None,
    azimuths_deg: list[float] | None = None,
    max_views_per_doubt: int | None = None,
    segmenter: str = "sam3.segment_text",
    out_dir: str | None = None,
    twin_path: str | None = None,
    keep_images: bool | None = None,
    check_reach: bool = True,
    reach_tolerance: float = REACH_TOLERANCE,
    return_to_start: bool = True,
) -> Output:
    items = [geo.normalize_doubt(d, i) for i, d in enumerate(doubts or [])]
    out: Output = {
        "route": "no_view",
        "doubts": [],
        "frames": [],
        "twin_updates": [],
        "n_doubts": len(items),
        "n_measured": 0,
        "n_views": 0,
        "detail": "",
    }
    if not items:
        out["detail"] = "no doubts given"
        return out
    arms = [int(a) for a in (arm_ids if arm_ids else [arm_id])]
    if surface_z is None:
        try:
            ws = ctx.tool("robot.describe_workspace", **_on_arm(arms[0])) or {}
            surface_z = None if ws.get("surface_z") is None else float(ws["surface_z"])
        except Exception as exc:  # heights need it; the rest does not
            logger.info("%s: no workspace description (%s)", SKILL, exc)
    plan = {
        "tilts": [float(t) for t in (tilts_deg or TILTS_DEG)],
        "distances": [float(d) for d in (distances_m or DISTANCES_M)],
        "azimuths": [float(a) for a in (azimuths_deg or AZIMUTHS_DEG)],
        "check_reach": bool(check_reach),
        "reach_tolerance": float(reach_tolerance),
    }
    plan["max_views"] = int(max_views_per_doubt or len(plan["tilts"]) * len(plan["azimuths"]))
    targets = [np.asarray(d["point"], dtype=float) for d in items]
    transit_z = max(float(t[2]) for t in targets) + TRANSIT_OVER_TARGETS

    obs = ctx.tool("robot.get_observation")
    rigs = [r for a in arms if (r := _arm_rig(ctx, obs, a, camera)) is not None]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    folder = Path(out_dir) if out_dir else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    keep = (folder is None) if keep_images is None else bool(keep_images)

    for doubt in items:
        failures: list[str] = []
        notes: list[str] = []
        views: list[dict[str, Any]] = []
        if not rigs:
            failures.append(f"no wrist camera with a pose on arms {arms}")
        for rig in rigs:
            got = _look(ctx, rig, doubt, targets, surface_z, transit_z, plan, failures)
            views += got
            if got:
                break
        files: dict[str, str] = {}
        selected = []
        for view in views:
            mask = None
            if doubt.get("object_name") and segmenter:
                mask = _mask(ctx, segmenter, view, str(doubt["object_name"]), notes)
            frame = {
                "id": view["id"],
                "rgb": view["rgb"],
                "depth": view["depth"],
                "K": view["K"],
                "T": view["T"],
            }
            selected.append((view["id"], geo.select_points(frame, doubt, surface_z, mask)))
            if folder:
                f = folder / f"{view['id']}.npz"
                np.savez_compressed(
                    f, rgb=view["rgb"], depth_m=view["depth"], K=view["K"], T_base_camera=view["T"]
                )
                files[view["id"]] = str(f)
        if views:
            result = geo.measure(selected, doubt, surface_z)
        else:
            result = {
                "status": "not_reachable",
                "quantities": {},
                "detail": "; ".join(failures[-8:]) or "no view pose",
            }
        record = {
            "id": doubt["id"],
            "status": result["status"],
            "detail": result["detail"],
            "point": doubt["point"],
            "what": doubt["what"],
            "quantities": _plain(result["quantities"]),
            "views": [v["id"] for v in views],
            "points_per_view": {i: int(len(P)) for i, P in selected},
            "failures": failures,
            "notes": notes,
        }
        out["doubts"].append(record)
        out["twin_updates"] += _updates(result, doubt, files, stamp)
        for view in views:
            entry = {
                k: view[k]
                for k in (
                    "id",
                    "doubt",
                    "arm_id",
                    "camera",
                    "tilt_deg",
                    "azimuth_deg",
                    "distance_m",
                    "camera_pose",
                    "tool_pose",
                )
            }
            entry["intrinsics"] = view["K"]
            entry["file"] = files.get(view["id"])
            if keep:
                entry["rgb"], entry["depth"] = view["rgb"], view["depth"]
            out["frames"].append(entry)
        logger.info("%s: %s %s (%s)", SKILL, doubt["id"], record["status"], record["detail"])

    if return_to_start:
        for rig in rigs:
            if rig.get("moved"):
                why = _travel(ctx, rig["arm"], rig["start"][:3, 3], rig["start"][:3, :3], transit_z)
                if why:
                    logger.info("%s: arm %d did not return (%s)", SKILL, rig["arm"], why)

    statuses = [d["status"] for d in out["doubts"]]
    out["n_measured"] = statuses.count("measured")
    out["n_views"] = len(out["frames"])
    out["route"] = next((r for r in ROUTE_ORDER if r in statuses), "measured")
    if folder:
        (folder / "measurements.json").write_text(
            json.dumps(
                _plain(
                    {
                        "doubts": out["doubts"],
                        "twin_updates": out["twin_updates"],
                        "frames": [
                            {k: v for k, v in f.items() if k not in ("rgb", "depth")}
                            for f in out["frames"]
                        ],
                    }
                ),
                indent=1,
            )
        )
    written = _write_twin(twin_path, out["twin_updates"], out["doubts"]) if twin_path else 0
    out["detail"] = (
        f"{out['n_measured']} of {len(items)} doubts measured from {out['n_views']} views; "
        + ", ".join(f"{d['id']}: {d['status']}" for d in out["doubts"])
        + (f"; {written} twin keys written to {twin_path}" if twin_path else "")
    )
    return out
