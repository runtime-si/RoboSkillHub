# Adapted from ByteTrack (https://github.com/FoundationVision/ByteTrack, commit d1bf0191adff),
# yolox/tracker/byte_tracker.py (BYTETracker.update, STrack), yolox/tracker/matching.py
# (iou_distance, fuse_score, linear_assignment) and basetrack.py, MIT License,
# Copyright (c) 2021 Yifu Zhang. See NOTICE.
# Changes: numpy only (scipy's linear_sum_assignment on lap.lapjv's extended cost matrix instead
# of lap / cython_bbox, no torch); the tracker state is a plain dict that a graph carries between
# calls (ids are per-state counters, not a process-wide class counter); detections carry a label
# and boxes of different labels never match; `freeze_lost` (default on) zeroes a lost track's
# velocity so a part that is covered stays where it was last seen (upstream zeroes only the
# height velocity); boxes come in pixels, no detector-scale division. The Kalman filter is
# written here from its textbook form with ByteTrack's state (x, y, a, h and their velocities) and
# noise weights (1/20 position, 1/160 velocity); upstream's kalman_filter.py (from deep_sort) is
# not copied.
"""ByteTrack multi-object association over per-frame boxes, with a serializable state."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

TRACKED, LOST = "tracked", "lost"

#: ByteTrack's Kalman noise weights, relative to the box height.
STD_WEIGHT_POSITION = 1.0 / 20
STD_WEIGHT_VELOCITY = 1.0 / 160


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter on (cx, cy, aspect w/h, h)
# ---------------------------------------------------------------------------

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)


def kf_initiate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = z[3]
    mean = np.r_[z, np.zeros(4)]
    std = np.array(
        [
            2 * STD_WEIGHT_POSITION * h,
            2 * STD_WEIGHT_POSITION * h,
            1e-2,
            2 * STD_WEIGHT_POSITION * h,
            10 * STD_WEIGHT_VELOCITY * h,
            10 * STD_WEIGHT_VELOCITY * h,
            1e-5,
            10 * STD_WEIGHT_VELOCITY * h,
        ]
    )
    return mean, np.diag(std**2)


def kf_predict(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = mean[3]
    q = np.array(
        [
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * h,
            1e-2,
            STD_WEIGHT_POSITION * h,
            STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * h,
            1e-5,
            STD_WEIGHT_VELOCITY * h,
        ]
    )
    return _F @ mean, _F @ cov @ _F.T + np.diag(q**2)


def kf_update(mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = mean[3]
    r = np.array([STD_WEIGHT_POSITION * h, STD_WEIGHT_POSITION * h, 1e-1, STD_WEIGHT_POSITION * h])
    s = _H @ cov @ _H.T + np.diag(r**2)
    gain = np.linalg.solve(s, _H @ cov).T  # cov H^T S^-1 (S symmetric)
    mean = mean + gain @ (z - _H @ mean)
    cov = cov - gain @ s @ gain.T
    return mean, cov


# ---------------------------------------------------------------------------
# Boxes and matching
# ---------------------------------------------------------------------------


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    h = m[3]
    w = m[2] * h
    return np.array([m[0] - w / 2, m[1] - h / 2, m[0] + w / 2, m[1] + h / 2], dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` [N, 4] with every box in ``b`` [M, 4] (xyxy)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list, list, list]:
    """``(matches, unmatched_rows, unmatched_cols)`` as ``lap.lapjv(extend_cost=True,
    cost_limit=thresh)`` gives them: leaving a row and a column unmatched costs ``thresh``,
    so only pairs cheaper than that are matched, and those with cost <= thresh kept."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    big = thresh * 1e3 + 1e3
    ext = np.full((n + m, n + m), big)
    ext[:n, :m] = np.where(cost > thresh, big, cost)
    np.fill_diagonal(ext[:n, m:], thresh / 2)
    np.fill_diagonal(ext[n:, :m], thresh / 2)
    ext[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(ext)
    matches = [
        (r, c) for r, c in zip(rows, cols, strict=True) if r < n and c < m and cost[r, c] <= thresh
    ]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return matches, [r for r in range(n) if r not in mr], [c for c in range(m) if c not in mc]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ByteTracker:
    """BYTETracker.update over a dict state; one call per frame."""

    def __init__(
        self,
        state: dict[str, Any] | None = None,
        *,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: float = 30.0,
        fuse_score: bool = True,
        freeze_lost: bool = True,
    ) -> None:
        state = state or {}
        self.frame_id = int(state.get("frame_id", 0))
        self.next_id = int(state.get("next_id", 1))
        self.tracks: list[dict[str, Any]] = [dict(t) for t in state.get("tracks", [])]
        for t in self.tracks:
            t["mean"] = np.asarray(t["mean"], dtype=np.float64)
            t["cov"] = np.asarray(t["cov"], dtype=np.float64).reshape(8, 8)
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.fuse_score = fuse_score
        self.freeze_lost = freeze_lost

    # -- state ------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        tracks = []
        for t in self.tracks:
            d = dict(t)
            d["mean"] = [round(float(v), 6) for v in t["mean"]]
            d["cov"] = [round(float(v), 9) for v in t["cov"].ravel()]
            tracks.append(d)
        return {"frame_id": self.frame_id, "next_id": self.next_id, "tracks": tracks}

    @staticmethod
    def box(track: dict[str, Any]) -> np.ndarray:
        return xyah_to_xyxy(track["mean"][:4])

    def _cost(self, tracks: list[dict], dets: list[dict], fuse: bool) -> np.ndarray:
        cost = 1.0 - iou_matrix(
            np.array([self.box(t) for t in tracks]).reshape(-1, 4),
            np.array([d["box"] for d in dets]).reshape(-1, 4),
        )
        if fuse and cost.size:
            cost = 1.0 - (1.0 - cost) * np.array([d["score"] for d in dets])[None]
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if t.get("label", "") != d.get("label", ""):
                    cost[i, j] = 1.0
        return cost

    def _update(self, t: dict, d: dict, *, refind: bool) -> None:
        t["mean"], t["cov"] = kf_update(
            t["mean"], t["cov"], xyxy_to_xyah(np.asarray(d["box"], float))
        )
        t["state"] = TRACKED
        t["activated"] = True
        t["frame_id"] = self.frame_id
        t["score"] = float(d["score"])
        t["tracklet_len"] = 0 if refind else t.get("tracklet_len", 0) + 1

    # -- one frame --------------------------------------------------------

    def update(self, detections: list[dict[str, Any]]) -> dict[str, list]:
        """Associate this frame's ``[{box: [x1, y1, x2, y2], score, label}]``; returns events."""
        self.frame_id += 1
        events: dict[str, list] = {"new": [], "lost": [], "refound": [], "removed": []}
        high = [d for d in detections if d["score"] > self.track_thresh]
        # upstream: low = (0.1, track_thresh) open, so a score equal to track_thresh is in neither
        low = [d for d in detections if self.low_thresh < d["score"] < self.track_thresh]

        unconfirmed = [t for t in self.tracks if t["state"] == TRACKED and not t["activated"]]
        tracked = [t for t in self.tracks if t["state"] == TRACKED and t["activated"]]
        lost = [t for t in self.tracks if t["state"] == LOST]

        pool = tracked + lost
        for t in pool:
            mean = t["mean"].copy()
            if t["state"] != TRACKED:
                mean[7] = 0.0
                if self.freeze_lost:
                    mean[4:8] = 0.0
            t["mean"], t["cov"] = kf_predict(mean, t["cov"])

        # Step 2: high-score boxes against tracked + lost tracks.
        matches, u_track, u_det = linear_assignment(
            self._cost(pool, high, self.fuse_score), self.match_thresh
        )
        for i, j in matches:
            t = pool[i]
            was_lost = t["state"] == LOST
            self._update(t, high[j], refind=was_lost)
            if was_lost:
                events["refound"].append(t["id"])

        # Step 3: low-score boxes against the still unmatched tracked ones.
        r_tracked = [pool[i] for i in u_track if pool[i]["state"] == TRACKED]
        matches, u_r, _ = linear_assignment(self._cost(r_tracked, low, False), 0.5)
        for i, j in matches:
            self._update(r_tracked[i], low[j], refind=False)
        for i in u_r:
            t = r_tracked[i]
            t["state"] = LOST
            t["lost_since"] = self.frame_id
            events["lost"].append(t["id"])

        # Unconfirmed (one-frame) tracks against the remaining high boxes.
        rest = [high[j] for j in u_det]
        matches, u_unconf, u_det = linear_assignment(
            self._cost(unconfirmed, rest, self.fuse_score), 0.7
        )
        for i, j in matches:
            self._update(unconfirmed[i], rest[j], refind=False)
        removed_ids = {unconfirmed[i]["id"] for i in u_unconf}

        # Step 4: new tracks.
        born = []
        for j in u_det:
            d = rest[j]
            if d["score"] < self.det_thresh:
                continue
            mean, cov = kf_initiate(xyxy_to_xyah(np.asarray(d["box"], float)))
            t = {
                "id": self.next_id,
                "label": d.get("label", ""),
                "state": TRACKED,
                "activated": self.frame_id == 1,
                "mean": mean,
                "cov": cov,
                "score": float(d["score"]),
                "frame_id": self.frame_id,
                "start_frame": self.frame_id,
                "tracklet_len": 0,
            }
            self.next_id += 1
            born.append(t)
            events["new"].append(t["id"])

        # Step 5: drop lost tracks past the buffer.
        for t in self.tracks:
            if t["state"] == LOST and self.frame_id - t["frame_id"] > self.max_time_lost:
                removed_ids.add(t["id"])
        events["removed"] = sorted(removed_ids)
        self.tracks = [t for t in self.tracks if t["id"] not in removed_ids] + born
        self._remove_duplicates(events)
        events["lost"] = [i for i in events["lost"] if i not in removed_ids]
        return events

    def _remove_duplicates(self, events: dict[str, list]) -> None:
        """upstream remove_duplicate_stracks: a tracked and a lost track over the same box
        (IoU > 0.85) with the same label: the younger one goes."""
        tracked = [t for t in self.tracks if t["state"] == TRACKED]
        lost = [t for t in self.tracks if t["state"] == LOST]
        if not tracked or not lost:
            return
        iou = iou_matrix(
            np.array([self.box(t) for t in tracked]), np.array([self.box(t) for t in lost])
        )
        drop = set()
        for p, q in zip(*np.where(1.0 - iou < 0.15), strict=True):
            if tracked[p].get("label", "") != lost[q].get("label", ""):
                continue
            tp = tracked[p]["frame_id"] - tracked[p]["start_frame"]
            tq = lost[q]["frame_id"] - lost[q]["start_frame"]
            drop.add(lost[q]["id"] if tp > tq else tracked[p]["id"])
        if drop:
            self.tracks = [t for t in self.tracks if t["id"] not in drop]
            events["removed"] = sorted(set(events["removed"]) | drop)
            for key in ("new", "lost", "refound"):
                events[key] = [i for i in events[key] if i not in drop]

    def outputs(self) -> list[dict[str, Any]]:
        """Every live track (upstream returns the activated tracked ones; lost ones are added
        here with their last box so a caller sees covered parts too)."""
        out = []
        for t in self.tracks:
            if t["state"] == TRACKED and not t["activated"]:
                continue
            out.append(
                {
                    "id": int(t["id"]),
                    "label": t.get("label", ""),
                    "box": [round(float(v), 2) for v in self.box(t)],
                    "score": round(float(t["score"]), 4),
                    "state": t["state"],
                    "frames_seen": int(t["frame_id"] - t["start_frame"] + 1),
                    "frames_lost": int(self.frame_id - t["frame_id"]),
                }
            )
        return sorted(out, key=lambda o: o["id"])
