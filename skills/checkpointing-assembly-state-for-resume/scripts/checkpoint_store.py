# The activity stream (_Activity, activity_since) is adapted from py_trees
# (https://github.com/splintered-reality/py_trees, py_trees/blackboard.py, ActivityType,
# ActivityItem, ActivityStream, Blackboard.absolute_name, commit
# 49bcffd069afa2b7bd9ca7b0e7312f5cbb3640bd), BSD-3-Clause, Copyright (c) 2020 Daniel Stonier;
# licence in LICENSE.py_trees. Changes: the stream records the difference between two
# checkpoints of a keyed state (INITIALISED / WRITE / UNSET with previous and current value)
# instead of live blackboard accesses; items are plain dicts so they serialise; a full stream
# drops its oldest item (upstream's push pops the newest); client ids are replaced by the
# step index. Everything else here (atomic files, blob store, hashing, rotation, schema
# migration) is written for this skill.
"""Files of checkpointing-assembly-state-for-resume: atomic writes, hashes, blobs, versions.

A checkpoint is one JSON file ``<task>.<seq:08d>.ckpt.json`` holding an envelope
``{"schema", "schema_version", "sha256", "payload"}``; ``sha256`` is over the payload's
canonical JSON (sorted keys, no spaces), so any torn or flipped byte is caught. Arrays
(snapshot RGB / depth, item masks) go to a content-addressed blob store ``blobs/<sha>.npy.z``
(zlib-compressed .npy, named by the hash of the uncompressed bytes) beside the checkpoints and appear in the payload as ``{"$blob": sha, "dtype", "shape"}``;
unchanged snapshots are stored once. Every file is written to a temp file in the same
directory, flushed, fsynced and renamed over the target, then the directory is fsynced.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "roboskillhub.assembly-checkpoint"
SCHEMA_VERSION = 3
"""Current payload version. 1: a bare run state as the jigsaw / LEGO skills take it
(placements, sequence, ledger or ledger_path, queue; no envelope, no hash). 2: the first
envelope (``step``, ``queue``, ``arms`` keyed by id, one ``snapshot_path``, ``saved_at``
in unix seconds). 3: this file's format. Older versions are migrated on load."""

KEEP = 5
"""Checkpoints kept per task. A torn newest file still leaves four to fall back to; a
550-piece state with the localizer records of every piece and a 40-event ledger is 0.5 MB
(measured), so five cost little."""

MAX_ACTIVITY = 500
"""Longest activity stream carried from checkpoint to checkpoint (py_trees' default
ActivityStream maximum_size)."""

INLINE_ARRAY_MAX = 64
"""Arrays up to this many elements (joint vectors, a 4x4 pose, a 3x3 K) are written inline
as JSON lists; larger ones (frames, masks) go to the blob store."""

BLOB_ZLIB_LEVEL = 1
"""zlib level of blob files. A full-frame item mask (640 x 480 bool, 307 kB) shrinks to
~1-2 kB, a 640 x 480 RGB-D snapshot (2.1 MB) to about half; level 1 costs ~10 ms per frame."""

SEPARATOR = "/"
"""Key separator of the activity stream (py_trees' Blackboard.separator)."""

CKPT_RE = re.compile(r"^(?P<task>[A-Za-z0-9_.-]+)\.(?P<seq>\d{8})\.ckpt\.json$")
TMP_PREFIX = ".tmp-"


class CheckpointError(Exception):
    """A checkpoint file that cannot be used; the message says why."""


# --- names and hashes --------------------------------------------------------------------------


def safe_task(task_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id)).strip("._")
    if not s:
        raise ValueError(f"task_id {task_id!r} has no usable characters")
    return s[:120]


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- atomic files ------------------------------------------------------------------------------


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: str | Path, data: bytes, fsync: bool = True) -> None:
    """Temp file in the same directory, flush, fsync, rename over ``path``, fsync the
    directory: a reader sees the old file or the whole new one, never half of it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=TMP_PREFIX + p.name + ".", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        _fsync_dir(p.parent)


# --- blobs ---------------------------------------------------------------------------------------


def _blob_dir(root: Path) -> Path:
    return root / "blobs"


def _blob_path(root: Path, sha: str) -> Path:
    return _blob_dir(root) / f"{sha}.npy.z"


def _read_blob(path: Path, sha: str) -> bytes:
    try:
        data = zlib.decompress(path.read_bytes())
    except (OSError, zlib.error) as exc:
        raise CheckpointError(f"blob {sha[:12]} is corrupt ({exc})") from exc
    if sha256_bytes(data) != sha:
        raise CheckpointError(f"blob {sha[:12]} is corrupt (hash mismatch)")
    return data


def put_blob(root: Path, arr: np.ndarray, written: list[str], fsync: bool = True) -> dict:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    data = buf.getvalue()
    sha = sha256_bytes(data)
    path = _blob_path(root, sha)
    try:
        ok = path.exists() and bool(_read_blob(path, sha))
    except CheckpointError:
        ok = False
    if not ok:
        atomic_write(path, zlib.compress(data, BLOB_ZLIB_LEVEL), fsync=fsync)
        written.append(sha)
    return {"$blob": sha, "dtype": str(arr.dtype), "shape": list(arr.shape)}


def get_blob(root: Path, ref: dict) -> np.ndarray:
    sha = str(ref["$blob"])
    path = _blob_path(root, sha)
    if not path.exists():
        raise CheckpointError(f"blob {sha[:12]} is missing")
    return np.load(io.BytesIO(_read_blob(path, sha)), allow_pickle=False)


def encode(value: Any, root: Path, written: list[str], fsync: bool = True) -> Any:
    """JSON-safe copy of ``value``: numpy scalars become numbers, small arrays lists, large
    arrays blob references."""
    if isinstance(value, dict):
        return {str(k): encode(v, root, written, fsync) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, root, written, fsync) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= INLINE_ARRAY_MAX and value.dtype != bool:
            return value.tolist()
        return put_blob(root, value, written, fsync)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot store a {type(value).__name__} in a checkpoint")


def decode(value: Any, root: Path, errors: list[str]) -> Any:
    """Inverse of ``encode``; a missing or corrupt blob becomes None and is listed."""
    if isinstance(value, dict):
        if "$blob" in value:
            try:
                return get_blob(root, value)
            except CheckpointError as exc:
                errors.append(str(exc))
                return None
        return {k: decode(v, root, errors) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v, root, errors) for v in value]
    return value


def blob_refs(value: Any, out: set[str]) -> set[str]:
    if isinstance(value, dict):
        if "$blob" in value:
            out.add(str(value["$blob"]))
        else:
            for v in value.values():
                blob_refs(v, out)
    elif isinstance(value, list):
        for v in value:
            blob_refs(v, out)
    return out


# --- checkpoint files ----------------------------------------------------------------------------


def list_checkpoints(root: Path, task: str | None = None) -> list[tuple[str, int, Path]]:
    """(task, seq, path), newest first (highest seq; ties by task name)."""
    out = []
    if not root.is_dir():
        return out
    for p in root.iterdir():
        m = CKPT_RE.match(p.name)
        if m and p.is_file() and (task is None or m.group("task") == task):
            out.append((m.group("task"), int(m.group("seq")), p))
    out.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return out


def stale_temps(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.name.startswith(TMP_PREFIX))


def checkpoint_path(root: Path, task: str, seq: int) -> Path:
    return root / f"{task}.{seq:08d}.ckpt.json"


def envelope(payload: dict) -> bytes:
    return json.dumps(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "sha256": sha256_bytes(canonical_json(payload)),
            "payload": payload,
        },
        sort_keys=True,
        indent=1,
        allow_nan=False,
    ).encode()


def read_checkpoint(path: Path) -> tuple[dict, int, list[int]]:
    """(payload at SCHEMA_VERSION, version found, versions migrated through). Raises
    CheckpointError for a torn, corrupt, foreign or newer file."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"unreadable: {exc}") from exc
    if not raw.strip():
        raise CheckpointError("empty file")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"not valid JSON (torn write?): {exc.msg} at byte {exc.pos}") from exc
    except UnicodeDecodeError as exc:
        raise CheckpointError(f"not valid JSON (bad UTF-8 at byte {exc.start})") from exc
    if not isinstance(doc, dict):
        raise CheckpointError("not a JSON object")
    if "schema_version" not in doc and "payload" not in doc:
        found = 1
        data: dict = doc
    else:
        if doc.get("schema", SCHEMA) != SCHEMA:
            raise CheckpointError(f"foreign schema {doc.get('schema')!r}")
        try:
            found = int(doc.get("schema_version"))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"bad schema_version {doc.get('schema_version')!r}") from exc
        data = doc.get("payload")
        if not isinstance(data, dict):
            raise CheckpointError("no payload object")
        want = doc.get("sha256")
        if not want:
            raise CheckpointError("no sha256 in the envelope")
        if sha256_bytes(canonical_json(data)) != want:
            raise CheckpointError("payload hash mismatch (corrupt or edited)")
    if found > SCHEMA_VERSION:
        raise CheckpointError(f"schema_version {found} is newer than this skill's {SCHEMA_VERSION}")
    if found < 1:
        raise CheckpointError(f"bad schema_version {found}")
    m = CKPT_RE.match(path.name)
    seq = int(m.group("seq")) if m else 0
    migrated: list[int] = []
    v = found
    while v < SCHEMA_VERSION:
        data = MIGRATIONS[v](data, seq)
        migrated.append(v)
        v += 1
    missing = [k for k in ("task_id", "step_index") if k not in data]
    if missing:
        raise CheckpointError(f"payload lacks {missing}")
    return data, found, migrated


# --- schema migration ----------------------------------------------------------------------------


def _v1_to_v2(d: dict, seq: int) -> dict:
    """A bare run state: placements (jigsaw), or a planner sequence with ``placed`` (count
    already built) or ``bricks``, a ledger or ledger_path, a queue."""
    placements = list(d.get("placements") or [])
    bricks = list(d.get("bricks") or [])
    queue = list(d.get("queue") or d.get("pending") or [])
    seqn = d.get("sequence")
    if isinstance(seqn, dict):
        seqn = seqn.get("sequence")
    if isinstance(seqn, list) and not bricks:
        n = int(d.get("placed", d.get("n_placed", 0)) or 0)
        bricks, rest = seqn[:n], seqn[n:]
        queue = queue or rest
    step = d.get("step", d.get("step_index"))
    if step is None:
        step = len(placements) + len(bricks)
    return {
        "task_id": str(d.get("task_id") or d.get("task") or "task"),
        "step": int(step),
        "placements": placements,
        "bricks": bricks,
        "queue": queue,
        "ledger": d.get("ledger"),
        "ledger_path": d.get("ledger_path"),
        "snapshot_path": d.get("snapshot_path") or d.get("snapshot"),
        "arms": {},
        "saved_at": d.get("saved_at"),
    }


def _v2_to_v3(d: dict, seq: int) -> dict:
    from datetime import datetime, timezone

    saved = d.get("saved_at")
    arms_v2 = d.get("arms") or {}
    if isinstance(arms_v2, dict):
        arms = [{"id": str(k), **(v or {})} for k, v in sorted(arms_v2.items())]
    else:
        arms = list(arms_v2)
    ledgers: dict[str, Any] = {}
    if d.get("ledger") is not None or d.get("ledger_path"):
        ledgers["backtracking"] = {"data": d.get("ledger"), "path": d.get("ledger_path")}
    reference = None
    if d.get("snapshot_path"):
        reference = {"snapshot_path": str(d["snapshot_path"]), "snapshot_sha256": None}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "task_id": str(d.get("task_id") or "task"),
        "task_kind": d.get("task_kind") or ("lego" if d.get("bricks") else "jigsaw"),
        "seq": seq,
        "step_index": int(d.get("step", 0)),
        "created_unix": float(saved) if saved is not None else None,
        "created_at": (
            datetime.fromtimestamp(float(saved), timezone.utc).isoformat()
            if saved is not None
            else None
        ),
        "placements": list(d.get("placements") or []),
        "bricks": list(d.get("bricks") or []),
        "items": list(d.get("items") or []),
        "pieces": d.get("pieces"),
        "pending": list(d.get("queue") or []),
        "in_progress": d.get("in_progress"),
        "ledgers": ledgers,
        "reference": reference,
        "build": d.get("build"),
        "arms": arms,
        "grid": d.get("grid"),
        "extra": d.get("extra") or {},
        "activity": [],
        "previous": None,
    }


MIGRATIONS = {1: _v1_to_v2, 2: _v2_to_v3}


# --- activity stream (adapted from py_trees' blackboard) ----------------------------------------


def absolute_name(namespace: str, key: str) -> str:
    """py_trees' Blackboard.absolute_name: '/' + 'placements' = '/placements'."""
    namespace = namespace if namespace.startswith(SEPARATOR) else SEPARATOR + namespace
    namespace = namespace if namespace.endswith(SEPARATOR) else namespace + SEPARATOR
    key = key.strip(SEPARATOR)
    return namespace + key


def keyed_view(p: dict) -> dict[str, Any]:
    """The state as blackboard keys: one per placed thing, arm, ledger and counter."""
    view: dict[str, Any] = {}
    for coll, idk in (("placements", "piece"), ("bricks", "id"), ("items", "id")):
        for i, rec in enumerate(p.get(coll) or []):
            rid = rec.get(idk, rec.get("id", i)) if isinstance(rec, dict) else i
            view[absolute_name(coll, str(rid))] = rec
    for i, arm in enumerate(p.get("arms") or []):
        view[absolute_name("arms", str(arm.get("id", i)))] = arm
    for name, led in (p.get("ledgers") or {}).items():
        view[absolute_name("ledgers", name)] = (led or {}).get("sha256")
    view["/pending"] = len(p.get("pending") or [])
    view["/in_progress"] = p.get("in_progress")
    view["/step_index"] = p.get("step_index")
    ref = p.get("reference") or {}
    view["/reference/step_index"] = ref.get("step_index")
    return view


def activity_since(
    prev: dict | None, cur: dict, max_size: int = MAX_ACTIVITY
) -> tuple[list[dict], int]:
    """The previous checkpoint's stream plus what changed since, oldest dropped first; and
    how many items are new."""
    old = keyed_view(prev) if prev else {}
    new = keyed_view(cur)
    stream = list((prev or {}).get("activity") or [])
    n0 = len(stream)
    step = cur.get("step_index")
    for key in sorted(set(old) | set(new)):
        if key not in old:
            item = {"key": key, "activity_type": "INITIALISED", "current_value": new[key]}
        elif key not in new:
            item = {"key": key, "activity_type": "UNSET", "previous_value": old[key]}
        elif canonical_json(old[key]) != canonical_json(new[key]):
            item = {
                "key": key,
                "activity_type": "WRITE",
                "previous_value": old[key],
                "current_value": new[key],
            }
        else:
            continue
        item["step_index"] = step
        stream.append(item)
    n_new = len(stream) - n0
    if len(stream) > max_size:
        stream = stream[len(stream) - max_size :]
    return stream, n_new


# --- frames and sibling skills -------------------------------------------------------------------


def load_frame_file(path: str | Path) -> dict[str, Any]:
    """A snapshot file as a CameraFrame: .npz (rgb, depth_m / depth, K / intrinsics, pose,
    name) or an image (.png / .jpg)."""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            frame: dict[str, Any] = {"rgb": z["rgb"]}
            for src, dst in (
                ("depth_m", "depth"),
                ("depth", "depth"),
                ("K", "intrinsics"),
                ("intrinsics", "intrinsics"),
                ("pose", "pose"),
            ):
                if src in keys and dst not in frame:
                    frame[dst] = z[src]
            if "name" in keys:
                frame["name"] = str(z["name"])
        return frame
    import cv2

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise CheckpointError(f"cannot read image {p}")
    return {"rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB)}


_SIBLINGS: dict[str, Any] = {}


def sibling_module(skill: str, script: str) -> Any | None:
    """Another hub skill's script, from the skill installed beside this one or under a
    GAP_SKILLS_PATH root; None when it is not installed."""
    key = f"{skill}/{script}"
    if key in _SIBLINGS:
        return _SIBLINGS[key]
    roots = [Path(__file__).resolve().parents[2]]
    roots += [Path(r) for r in os.environ.get("GAP_SKILLS_PATH", "").split(os.pathsep) if r]
    for root in roots:
        for path in (root / skill / script, root / "skills" / skill / script):
            if path.is_file():
                name = "_ckpt_" + re.sub(r"\W", "_", key)
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _SIBLINGS[key] = module
                return module
    _SIBLINGS[key] = None
    return None


def deep(v: Any) -> Any:
    return copy.deepcopy(v)
