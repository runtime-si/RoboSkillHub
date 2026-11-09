"""Agent API, trusted station configuration loading, and execution provenance."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

import yaml

from .contracts import (
    CAPABILITY_BUNDLES,
    CONTRACT_VERSION,
    SCHEMAS,
    STATUSES,
    Limits,
    Rejected,
    validate_request,
)
from .engine import REQUIRES, Runtime, TaskLease

CONDITIONAL_REQUIREMENTS = {
    "move_waypoints": {
        "linear=true": ["linear_cartesian_motion"],
        "held": ["object_feedback"],
    },
    "observe_objects": {"require_orientation=true": ["object_pose_feedback"]},
}

_CACHE: dict[tuple[str, str], tuple[Runtime, dict]] = {}
_CACHE_LOCK = threading.Lock()
_TRACKERS: dict[tuple, object] = {}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


_LOADED_RUNTIME_DIGEST = runtime_digest()


def _config(station_path: str | None) -> tuple[Path, dict, str] | None:
    filename = station_path or os.environ.get("ROBOSKILLHUB_STATION")
    if not filename:
        return None
    path = Path(filename).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise Rejected("station configuration requires schema_version: 1")
    if not isinstance(cfg.get("station_id"), str) or not cfg["station_id"]:
        raise Rejected("station_id is required")
    unknown = set(cfg) - {
        "schema_version",
        "station_id",
        "backend",
        "limits",
        "tracking",
        "log_dir",
    }
    if unknown:
        raise Rejected(f"unknown station config keys: {sorted(unknown)}")
    for key in ("backend", "limits", "tracking"):
        if key in cfg and not isinstance(cfg[key], dict):
            raise Rejected(f"station {key} must be an object")
    if "log_dir" in cfg and (not isinstance(cfg["log_dir"], str) or not cfg["log_dir"]):
        raise Rejected("log_dir must be a nonempty path string")
    return path, cfg, hashlib.sha256(raw).hexdigest()


def _station(config: tuple[Path, dict, str] | None) -> Runtime:
    if config is None:
        raise Rejected("no station configured; set ROBOSKILLHUB_STATION or --station")
    path, cfg, config_digest = config
    key = (str(path), config_digest)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key][0]
        limits = Limits.from_dict(cfg.get("limits"))
        backend_cfg = cfg.get("backend")
        if not isinstance(backend_cfg, dict) or set(backend_cfg) - {"factory", "kwargs"}:
            raise Rejected("backend requires a trusted local factory and optional kwargs")
        factory = backend_cfg.get("factory")
        if not isinstance(factory, str) or factory.count(":") != 1:
            raise Rejected("backend.factory must be an importable module:function")
        kwargs = backend_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise Rejected("backend.kwargs must be an object")
        if "limits" in kwargs:
            raise Rejected("backend.kwargs cannot override the station limits")
        module, name = factory.split(":")
        backend = getattr(importlib.import_module(module), name)(limits=limits, **kwargs)
        for method in ("describe", "get_state", "move_pose", "set_gripper", "stop"):
            if not callable(getattr(backend, method, None)):
                raise Rejected(f"station backend is missing {method}()")
        rt = Runtime(backend, limits)
        _CACHE[key] = (rt, cfg)
        return rt


def describe(station_path: str | None = None) -> dict:
    """Return schemas and configured capabilities, without commanding motion.

    Factory construction is allowed to connect/read, never to move. Configured
    capabilities are declarations; physical verification is a separate eval.
    """
    config = _config(station_path)
    backend = None
    error = None
    if config and config[1].get("backend"):
        try:
            backend = _station(config).backend.describe()
        except Exception as exc:
            error = str(exc)
    capabilities = set(backend.get("capabilities", [])) if backend else set()
    entries = []
    for name, schema in SCHEMAS.items():
        if name == "track_points":
            available = all(importlib.util.find_spec(m) for m in ("numpy", "cv2"))
            reason = "CPU OpenCV-LK dependency check; TAPIR requires its own configured weights"
        else:
            available = bool(backend) and REQUIRES[name] <= capabilities
            reason = error or (
                f"missing backend capabilities: {sorted(REQUIRES[name] - capabilities)}"
                if not available
                else "declared by configured backend; not a physical validation result"
            )
        entries.append(
            {
                "capability": name,
                "bundle": CAPABILITY_BUNDLES[name],
                "version": "0.1.0",
                "input_schema": schema,
                "configured": bool(available),
                "readiness_note": reason,
                "statuses": list(STATUSES),
                "kind": "primitive",
                "requires": sorted(REQUIRES.get(name, ())),
                "requires_if": CONDITIONAL_REQUIREMENTS.get(name, {}),
            }
        )
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "capabilities": entries,
        "backend": backend,
        "station_id": config[1]["station_id"] if config else None,
        "runtime_sha256": _LOADED_RUNTIME_DIGEST,
        "observation_contract": {
            "units": "metres",
            "quaternion": "xyzw",
            "timestamps": "Unix seconds",
            "pose_frame": "station limits.frame_id",
            "pixel_coordinates": "x,y; top-left origin",
        },
    }


def _tracking(request: dict, config: tuple[Path, dict, str] | None) -> dict:
    validate_request("track_points", request)
    import numpy as np

    from .tracking import LKPointTracker, TapirPointTracker

    path = Path(request["frames_path"]).expanduser().resolve(strict=True)
    if path.suffix != ".npy":
        raise Rejected("frames_path must be an uncompressed RGB uint8 .npy array [T,H,W,3]")
    max_bytes = 512 * 1024 * 1024
    if path.stat().st_size > max_bytes:
        raise Rejected("frame batch exceeds 512 MiB; split into smaller batches")
    # Immutable input snapshot: later file replacement/deletion cannot change
    # the inference data or provenance attached to an already completed result.
    with path.open("rb") as stream:
        frame_bytes = stream.read(max_bytes + 1)
    if len(frame_bytes) > max_bytes:
        raise Rejected("frame batch exceeds 512 MiB; split into smaller batches")
    frame_digest = hashlib.sha256(frame_bytes).hexdigest()
    frames = np.load(io.BytesIO(frame_bytes), allow_pickle=False)
    tracking = config[1].get("tracking", {}) if config else {}
    if not isinstance(tracking, dict):
        raise Rejected("station tracking config must be an object")
    backend = request.get("backend", tracking.get("backend", "opencv-lk"))
    if backend == "opencv-lk":
        tracker = LKPointTracker()
    elif backend == "tapir":
        checkpoint = tracking.get("checkpoint_path")
        digest = tracking.get("checkpoint_sha256")
        if not isinstance(checkpoint, str) or not isinstance(digest, str) or len(digest) != 64:
            raise Rejected("TAPIR requires station tracking.checkpoint_path and checkpoint_sha256")
        p = Path(checkpoint).expanduser()
        if not p.is_absolute() and config:
            p = config[0].parent / p
        p = p.resolve(strict=True)
        key = (str(p), digest, tracking.get("device", "cpu"))
        with _CACHE_LOCK:
            if key not in _TRACKERS:
                _TRACKERS[key] = TapirPointTracker.from_checkpoint(
                    p, device=key[2], expected_sha256=digest
                )
            tracker = _TRACKERS[key]
    else:
        raise Rejected("tracking backend must be opencv-lk or tapir")
    evidence = tracker.track(frames, request["query_points"])
    # Complete tracking is the skill postcondition. Partial tracks remain useful
    # evidence for a recovery branch, rather than being declared complete.
    status = {"tracked": "succeeded", "partial": "unverified", "lost": "failed"}[evidence["status"]]
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "kind": "primitive",
        "phase": "observe",
        "failure_type": {"succeeded": None, "unverified": "evidence_missing"}.get(
            status, "postcondition_failed"
        ),
        "recovery_hints": [] if status == "succeeded" else ["reobserve"],
        "invocation_id": str(uuid.uuid4()),
        "capability": "track_points",
        "skill": CAPABILITY_BUNDLES["track_points"],
        "version": "0.1.0",
        "status": status,
        "backend": {"name": backend, "kind": "perception"},
        "evidence": evidence,
        "trace": [],
        "input_artifact": {
            "frames_sha256": frame_digest,
            "checkpoint_sha256": tracking.get("checkpoint_sha256") if backend == "tapir" else None,
        },
    }


def task_lease(task_id: str, timeout_s: float, station_path: str | None = None):
    """Context manager holding the configured station for one task graph.

    Pass the yielded lease to :func:`invoke`; invoking without it from the
    holding thread is rejected as ``nested_lease`` instead of deadlocking.
    """
    return _station(_config(station_path)).task_lease(task_id, timeout_s)


def invoke(
    capability: str,
    request: dict,
    station_path: str | None = None,
    *,
    lease: TaskLease | None = None,
    request_id: str | None = None,
) -> dict:
    start = time.monotonic()
    config = None
    try:
        validate_request(capability, request)
        config = _config(station_path)
        if capability == "track_points":
            result = _tracking(request, config)
        else:
            runtime = _station(config)
            if lease is not None and lease.runtime is not runtime:
                raise Rejected("lease belongs to another station", failure_type="nested_lease")
            extra = {"lease": lease, "request_id": request_id}
            extra = {k: v for k, v in extra.items() if v is not None}
            result = runtime.invoke(capability, request, **extra)
    except Exception as exc:
        failure = getattr(exc, "failure_type", None) or (
            "invalid_request" if isinstance(exc, Rejected) else "precondition_unmet"
        )
        result = {
            "schema_version": 1,
            "contract_version": CONTRACT_VERSION,
            "invocation_id": str(uuid.uuid4()),
            "capability": capability,
            "skill": CAPABILITY_BUNDLES.get(capability) if isinstance(capability, str) else None,
            "version": "0.1.0",
            "kind": "primitive",
            "status": "rejected",
            "phase": "validate",
            "failure_type": failure,
            "recovery_hints": [],
            "error": {"code": type(exc).__name__, "message": str(exc)},
            "evidence": {},
            "trace": [],
        }
    result.setdefault("elapsed_s", time.monotonic() - start)
    result["manifest"] = {
        "runtime_sha256": _LOADED_RUNTIME_DIGEST,
        "station_id": config[1]["station_id"] if config else None,
        "station_config_sha256": config[2] if config else None,
        "backend_factory": config[1].get("backend", {}).get("factory") if config else None,
    }
    if config and config[1].get("log_dir"):
        directory = Path(config[1]["log_dir"]).expanduser()
        if not directory.is_absolute():
            directory = config[0].parent / directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{result['invocation_id']}.json").write_text(
                json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
        except (OSError, TypeError, ValueError) as exc:
            result["logging_error"] = str(exc)
    return result
