# Adapted from CNS (https://github.com/hhcaz/CNS, commit 6b42138ea8f15d872eeca916b9e1fb1f429bcac4),
# cns/benchmark/pipeline.py (CorrespondenceBasedPipeline.set_target / get_control_rate) and
# cns/benchmark/stop_policy.py (PixelStopPolicy), MIT License, Copyright (c) 2023 hhcaz. The modules
# under cns_core/ are CNS's frontend (classic detectors), midend (correspondence graph), GraphVS model,
# GraphVS controller and stop policies, trimmed as their headers say. Changes here: sessions keyed by id
# instead of one pipeline object; images in RGB (converted to CNS's BGR); the stop policy runs on call
# counts instead of wall-clock seconds; the weights download from the pinned commit on first use.
"""cns tool bundle: CNS learned visual servoing (keypoint correspondences -> graph -> camera velocity).

``cns.start`` records the goal view (a frame from the pose to come back to) and the distance from the
camera to the scene; ``cns.step`` takes the current frame of the same camera and returns the camera
velocity (camera frame, OpenCV axes) CNS's recurrent graph network commands toward the goal view, the
pixel error and whether CNS's stop policy says converged. torch / torch_geometric load lazily.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

WEIGHTS_URL = (
    "https://raw.githubusercontent.com/hhcaz/CNS/6b42138ea8f15d872eeca916b9e1fb1f429bcac4/"
    "checkpoints/cns_state_dict.pth"
)
WEIGHTS_SHA256 = "2b91c0d5b5b6d51c42456cf1d6cb798c2b66779d5e4f2c460b1963ee897eb67f"
#: Upstream demo_real_ur5.py: PixelStopPolicy(waiting_time=2 s, conduct_thresh=0.005) at ~10 Hz.
STOP_WAIT_STEPS = 6
STOP_CONDUCT = 0.005
MAX_SESSIONS = 4

_lock = threading.Lock()
_net: dict[str, Any] = {}
_sessions: dict[str, dict[str, Any]] = {}


class StartResult(TypedDict):
    session_id: str
    keypoints: int  # detected in the goal view
    ok: bool


class StepResult(TypedDict):
    velocity: list[float]  # [vx, vy, vz, wx, wy, wz] m/s, rad/s, camera frame (x right, y down, z forward)
    matches: int  # goal-current correspondences used (0: none, velocity zero)
    error: float  # CNS's pixel error (normalized image coordinates)
    converged: bool  # CNS's PixelStopPolicy
    seconds: float


def _weights() -> str:
    path = Path(os.environ.get("GAP_CNS_WEIGHTS") or Path.home() / ".cache" / "gap" / "cns_state_dict.pth")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned CNS checkpoint")
    return str(path)


def _core():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from cns_core.controller import GraphVSController
    from cns_core.frontend.classic import Classic
    from cns_core.midend.corr2graph import Midend
    from cns_core.stop_policy import PixelStopPolicy
    from cns_core.utils.perception import CameraIntrinsic

    return GraphVSController, Classic, Midend, PixelStopPolicy, CameraIntrinsic


def weights_cached() -> bool | None:
    path = Path(os.environ.get("GAP_CNS_WEIGHTS") or Path.home() / ".cache" / "gap" / "cns_state_dict.pth")
    return path.exists()


def prefetch() -> None:
    _weights()


@tool(
    name="cns.start",
    summary="Start a CNS servo session on a goal view (RGB frame, intrinsics, camera-to-scene distance).",
    tags=("perception", "control"),
)
def start(
    goal_image: np.ndarray, intrinsics: list[list[float]], distance_m: float, detector: str = "SIFT"
) -> StartResult:
    """``detector``: CNS frontend string (SIFT, ORB, AKAZE, or e.g. 'AKAZE:02+ORB:13')."""
    import torch

    img = np.asarray(goal_image)
    if img.dtype != np.uint8 or img.ndim != 3:
        raise ValueError(f"cns.start expects a uint8 [H, W, 3] image, got {img.dtype} {img.shape}")
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    try:
        GraphVSController, Classic, Midend, PixelStopPolicy, CameraIntrinsic = _core()
        device = os.environ.get("GAP_CNS_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with _lock:
            if "ckpt" not in _net:
                _net["ckpt"] = _weights()
            intr = CameraIntrinsic(img.shape[1], img.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            frontend = Classic(intr, detector, ransac=True)
            ok = bool(frontend.update_target_frame(np.ascontiguousarray(img[..., 2::-1])))
            sid = uuid.uuid4().hex[:12]
            while len(_sessions) >= MAX_SESSIONS:
                _sessions.pop(next(iter(_sessions)))
            _sessions[sid] = {
                "frontend": frontend,
                "midend": Midend(),
                "control": GraphVSController(_net["ckpt"], device),
                "stop": PixelStopPolicy(STOP_WAIT_STEPS, STOP_CONDUCT),
                "distance": float(distance_m),
                "new": True,
                "t": 0,
            }
    except Exception as exc:
        raise ToolError("cns.start", f"CNS failed: {exc}") from exc
    n = 0 if frontend.tar_pos is None else len(frontend.tar_pos)
    return {"session_id": sid, "keypoints": int(n), "ok": ok}


@tool(name="cns.step", summary="CNS camera velocity toward the session's goal view from the current frame.", tags=("perception", "control"))
def step(session_id: str, image: np.ndarray) -> StepResult:
    t0 = time.perf_counter()
    s = _sessions.get(str(session_id))
    if s is None:
        raise ToolError("cns.step", f"no session {session_id!r} (closed or evicted)")
    img = np.ascontiguousarray(np.asarray(image)[..., 2::-1])
    try:
        corr = s["frontend"].process_current_frame(img)
        if corr is None:
            return {"velocity": [0.0] * 6, "matches": 0, "error": float("inf"), "converged": False,
                    "seconds": round(time.perf_counter() - t0, 3)}
        data = s["midend"].get_graph_data(corr)
        if s["new"]:
            data.start_new_scene()
            s["new"] = False
        data.set_distance_scale(s["distance"])
        setattr(data, "intrinsic", s["frontend"].intrinsic)
        vel = s["control"](data)
        s["t"] += 1
        conv = bool(s["stop"](data, float(s["t"])))
        n = 0 if corr.match is None else int(len(corr.match))
    except Exception as exc:
        raise ToolError("cns.step", f"CNS failed: {exc}") from exc
    return {"velocity": [float(v) for v in vel], "matches": n, "error": float(s["stop"].cur_err),
            "converged": conv, "seconds": round(time.perf_counter() - t0, 3)}


@tool(name="cns.stop", summary="Close a CNS session.", tags=("control",))
def stop(session_id: str) -> dict[str, bool]:
    return {"closed": _sessions.pop(str(session_id), None) is not None}
