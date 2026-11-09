"""Keep stable ids for many look-alike parts across frames: ByteTrack over a detector's boxes.

One call is one frame. The tracker state goes out as ``state`` and comes back in on the
next call (a graph loop carries it), so ids survive between calls, through occlusion by
the arm, for ``track_buffer`` frames.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytetrack import LOST, ByteTracker  # noqa: E402

logger = logging.getLogger(__name__)

TRACK_THRESH = 0.1
"""Detections above this score keep tracks and (above it + 0.1, upstream's rule) start
them; those in (low_thresh, this) only keep them (ByteTrack's second association).
Upstream's 0.5 is for YOLOX scores; OWLv2 scored the station's 18 px parts 0.1-0.3
(SKILL.md, Measured), where 0.5 tracked nothing and 0.15 covered 81 % of the visible frames instead of 96 %.
Detectors scoring near 1 (colour / mask segmenters) work with it as well."""

LOW_THRESH = 0.05
"""Floor for the low-score association (upstream 0.1, scaled with TRACK_THRESH); also
the threshold passed to owlv2.detect."""

MATCH_THRESH = 0.8
"""Upstream's first-association cost limit (1 - IoU, x score when fuse_score)."""

FUSE_SCORE = False
"""Upstream multiplies IoU by the detection score (off in its MOT20 setting). With
scores of 0.1-0.3 every cost then exceeds MATCH_THRESH and nothing matches."""

TRACK_BUFFER = 150
"""Frames a lost track is kept (scaled by frame_rate / 30, as upstream; upstream 30).
Parts on a table stay under the arm for many seconds: on the station clip at 5 fps one
part was covered for 110 frames; upstream's 30 there cost 3 id switches."""

SWITCH_RADIUS = 2.5
"""A track born within this many box sizes of a lost track's last box is reported as a
suspected id switch: with identical parts the tracker cannot tell a part that was
moved while covered from a new one. A part bumped 25 px (1.4 sizes) under the arm
reappeared 1.4 sizes from its lost box."""

DEFAULT_DETECTOR_ARGS = {
    "owlv2.detect": {"threshold": LOW_THRESH},
    "grounding-dino.detect": {"box_threshold": LOW_THRESH},
}


class Output(TypedDict):
    route: str
    state: dict[str, Any]
    tracks: list[dict[str, Any]]
    n_tracked: int
    n_lost: int
    new_ids: list[int]
    lost_ids: list[int]
    refound_ids: list[int]
    removed_ids: list[int]
    suspected_switches: list[dict[str, Any]]
    n_detections: int
    reason: str


def _camera_rgb(observation: dict[str, Any], camera: str | None) -> tuple[str, np.ndarray | None]:
    cams = (observation or {}).get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} if "name" not in v else v for k, v in cams.items()]
    frames = {str(c.get("name") or f"camera{i}"): c for i, c in enumerate(cams)}
    names = [camera] if camera else sorted(frames, key=lambda n: "wrist" in n.lower())
    for name in names:
        rgb = (frames.get(name) or {}).get("rgb")
        if rgb is not None:
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255)
                    arr = arr.astype(np.uint8)
                return name, np.ascontiguousarray(arr)
    return (camera or ""), None


def _clean(detections: list[dict[str, Any]], labels: bool) -> list[dict[str, Any]]:
    out = []
    for d in detections or []:
        box = d.get("box") if isinstance(d, dict) else None
        if isinstance(box, dict):  # BoundingBox2D
            box = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
        if box is None or len(box) != 4 or any(v is None for v in box):
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            {
                "box": [x1, y1, x2, y2],
                "score": float(d.get("score", 1.0)),
                "label": str(d.get("label", "")) if labels else "",
            }
        )
    return out


def _area(b: np.ndarray) -> float:
    return float(max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0))


def _suspected(tracker: ByteTracker, new_ids: list[int], radius: float) -> list[dict[str, Any]]:
    """New tracks that look like a lost one (same label, area within 2x) and were born
    within ``radius`` of the lost part's own size from where it was last seen."""
    lost = [t for t in tracker.tracks if t["state"] == LOST]
    born = [t for t in tracker.tracks if t["id"] in set(new_ids)]
    out = []
    for b in born:
        bb = tracker.box(b)
        cb = (bb[:2] + bb[2:]) / 2
        best = None
        for t in lost:
            if t.get("label", "") != b.get("label", ""):
                continue
            lb = tracker.box(t)
            ratio = _area(bb) / max(_area(lb), 1e-6)
            if not 0.5 <= ratio <= 2.0:
                continue
            size = max(lb[2] - lb[0], lb[3] - lb[1])
            dist = float(np.linalg.norm((lb[:2] + lb[2:]) / 2 - cb))
            if dist <= radius * size and (best is None or dist < best[1]):
                best = (t["id"], dist)
        if best is not None:
            out.append(
                {"new_id": int(b["id"]), "lost_id": int(best[0]), "distance_px": round(best[1], 1)}
            )
    return out


def run(
    ctx: NodeContext,
    state: dict[str, Any] | None = None,
    detections: list[dict[str, Any]] | None = None,
    prompts: list[str] | str | None = None,
    detector: str = "owlv2.detect",
    detector_args: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    camera: str | None = None,
    track_thresh: float = TRACK_THRESH,
    low_thresh: float = LOW_THRESH,
    match_thresh: float = MATCH_THRESH,
    fuse_score: bool = FUSE_SCORE,
    track_buffer: int = TRACK_BUFFER,
    frame_rate: float = 30.0,
    freeze_lost: bool = True,
    use_labels: bool = True,
    switch_radius: float = SWITCH_RADIUS,
) -> Output:
    """Advance the tracker ``state`` (None: start) by one frame.

    The frame's boxes are ``detections`` (``[{box: [x1, y1, x2, y2], score, label}]``,
    pixels, e.g. from owlv2 / grounding-dino / sam3 / a colour segmenter), or, without
    them, ``detector`` run with ``prompts`` on ``camera`` of ``observation`` (default a
    fresh ``robot.get_observation``). Boxes of different labels never share a track
    (``use_labels``). Pass the same camera every call: ids are pixel tracks.
    """
    reason = ""
    if detections is None:
        if not prompts:
            raise ValueError("tracking-identical-parts-by-id needs detections or prompts")
        if observation is None:
            observation = ctx.tool("robot.get_observation")
        name, rgb = _camera_rgb(observation, camera)
        if rgb is None:
            return _no_frame(state, f"no RGB on camera {camera or 'any'}")
        query = prompts if isinstance(prompts, str) else ". ".join(str(p) for p in prompts)
        kwargs = dict(DEFAULT_DETECTOR_ARGS.get(detector, {}))
        kwargs.update(detector_args or {})
        try:
            result = ctx.tool(detector, image=rgb, query=query, **kwargs)
        except Exception as exc:  # noqa: BLE001  (unserved tool, model load, CUDA)
            logger.warning("track_parts: %s failed: %s", detector, exc)
            return _no_frame(state, f"{detector} failed: {exc}")
        detections = (result or {}).get("detections") or []
        reason = f"{detector} on {name}: "
    dets = _clean(detections, use_labels)

    tracker = ByteTracker(
        state,
        track_thresh=track_thresh,
        low_thresh=low_thresh,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
        track_buffer=track_buffer,
        frame_rate=frame_rate,
        freeze_lost=freeze_lost,
    )
    events = tracker.update(dets)
    switches = _suspected(tracker, events["new"], switch_radius) if tracker.frame_id > 1 else []
    tracks = tracker.outputs()
    n_tracked = sum(t["state"] != LOST for t in tracks)
    n_lost = len(tracks) - n_tracked
    if switches:
        route = "id_switch_suspected"
        reason += ", ".join(f"new {s['new_id']} near lost {s['lost_id']}" for s in switches)
    elif events["lost"]:
        route = "track_lost"
        reason += f"lost {events['lost']}"
    else:
        route = "tracks_updated"
        reason += f"{n_tracked} tracked, {n_lost} covered"
    logger.info("track_parts: frame %d %s (%s)", tracker.frame_id, route, reason)
    return {
        "route": route,
        "state": tracker.to_state(),
        "tracks": tracks,
        "n_tracked": n_tracked,
        "n_lost": n_lost,
        "new_ids": events["new"],
        "lost_ids": events["lost"],
        "refound_ids": events["refound"],
        "removed_ids": events["removed"],
        "suspected_switches": switches,
        "n_detections": len(dets),
        "reason": reason,
    }


def _no_frame(state: dict[str, Any] | None, reason: str) -> Output:
    return {
        "route": "no_frame",
        "state": state or {},
        "tracks": [],
        "n_tracked": 0,
        "n_lost": 0,
        "new_ids": [],
        "lost_ids": [],
        "refound_ids": [],
        "removed_ids": [],
        "suspected_switches": [],
        "n_detections": 0,
        "reason": reason,
    }
