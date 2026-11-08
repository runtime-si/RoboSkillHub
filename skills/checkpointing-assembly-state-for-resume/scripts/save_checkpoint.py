"""Write one versioned, hashed checkpoint of a long assembly run, atomically, and rotate.

Written for the hub. The state is what the jigsaw and brick skills already pass around, kept
in their own formats so a resumed run can hand it straight back to them: placements
(choosing-next-piece-and-slot / backtracking-a-wrong-placement), placed bricks
(planning-brick-assembly-order's sequence entries), generic items with poses, the pending
queue, backtracking-a-wrong-placement's ledger, the reference snapshot and item list for
detecting-disturbed-pieces-by-snapshot-diff, the build-check settings for
verifying-build-against-model, the arms and timestamps. The file layout, hashing, blob
store and the activity stream are in checkpoint_store.py beside this file.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint_store as cs  # noqa: E402

BLOB_GRACE_S = 600.0
"""A blob no kept checkpoint references is deleted only when older than this [s]: another
task saving into the same directory writes its blobs before its checkpoint file, and a save
takes well under a second (35-80 ms measured for a 550-piece state, see SKILL.md)."""

ITEM_KEYS = ("mask", "polygon_px", "box", "cell")
"""An item carrying one of these can be re-found by detecting-disturbed-pieces-by-snapshot-
diff (its item forms)."""


class Output(TypedDict):
    route: str
    checkpoint_path: str
    task_id: str
    seq: int
    step_index: int
    sha256: str
    bytes: int
    n_kept: int
    removed: list[str]
    blobs_written: int
    blobs_removed: int
    n_activity: int
    previous_seq: int
    reference_step_index: int
    detail: str


def _newest_valid(root: Path, task: str) -> tuple[dict | None, int]:
    for _, seq, path in cs.list_checkpoints(root, task):
        try:
            payload, _, _ = cs.read_checkpoint(path)
            return payload, seq
        except cs.CheckpointError:
            continue
    return None, -1


def _only_camera(obs: Any, camera_name: str) -> Any:
    if not camera_name or not isinstance(obs, dict) or "cameras" not in obs:
        return obs
    cams = obs.get("cameras") or []
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    keep = [c for c in cams if c.get("name") == camera_name]
    if not keep:
        raise ValueError(
            f"observation has no camera {camera_name!r} (has {[c.get('name') for c in cams]})"
        )
    return {**obs, "cameras": keep}


def _ledger_entry(data: Any, path: str | None, notes: list[str], name: str) -> dict:
    entry: dict[str, Any] = {"path": str(Path(path).resolve()) if path else None}
    if path and Path(path).is_file():
        raw = Path(path).read_bytes()
        try:
            entry["data"] = json.loads(raw)
            entry["sha256"] = cs.sha256_bytes(raw)
        except json.JSONDecodeError:
            notes.append(f"ledger {name} at {path} is not valid JSON; stored without it")
            entry["data"] = data
            entry["sha256"] = None
        return entry
    if path:
        notes.append(f"ledger {name} file {path} does not exist yet")
    entry["data"] = data
    entry["sha256"] = cs.sha256_bytes(cs.canonical_json(data)) if data is not None else None
    return entry


def _gc_blobs(root: Path, keep: set[str]) -> int:
    bdir = root / "blobs"
    if not bdir.is_dir():
        return 0
    refs = set(keep)
    for p in root.iterdir():
        if cs.CKPT_RE.match(p.name):
            try:
                cs.blob_refs(json.loads(p.read_bytes()), refs)
            except (OSError, ValueError):
                continue
    now = time.time()
    removed = 0
    for b in bdir.glob("*.npy.z"):
        if b.name.split(".", 1)[0] in refs:
            continue
        try:
            if now - b.stat().st_mtime > BLOB_GRACE_S:
                b.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def run(
    ctx: NodeContext,
    checkpoint_dir: str,
    task_id: str,
    step_index: int,
    task_kind: str = "",
    placements: list[dict[str, Any]] | None = None,
    bricks: list[dict[str, Any]] | None = None,
    items: list[dict[str, Any]] | None = None,
    pieces: list[dict[str, Any]] | None = None,
    pending: list[Any] | None = None,
    in_progress: dict[str, Any] | None = None,
    ledger: dict[str, Any] | None = None,
    ledger_path: str | None = None,
    ledgers: dict[str, Any] | None = None,
    reference_snapshot: Any = None,
    reference_snapshot_path: str | None = None,
    reference_items: list[dict[str, Any]] | None = None,
    snapshot_now: bool = False,
    camera_name: str = "",
    grid: dict[str, Any] | None = None,
    diff_args: dict[str, Any] | None = None,
    carry_reference: bool = True,
    build: dict[str, Any] | None = None,
    arms: list[dict[str, Any]] | None = None,
    read_arms: bool = False,
    extra: dict[str, Any] | None = None,
    keep: int = cs.KEEP,
    fsync: bool = True,
) -> Output:
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")
    root = Path(checkpoint_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    task = cs.safe_task(task_id)
    notes: list[str] = []
    prev, prev_seq = _newest_valid(root, task)
    existing = cs.list_checkpoints(root, task)
    seq = (existing[0][1] + 1) if existing else 1
    if prev is not None and int(step_index) < int(prev.get("step_index", 0)):
        notes.append(f"step went back from {prev['step_index']} to {step_index}")

    obs = None
    if snapshot_now or (read_arms and arms is None):
        obs = ctx.tool("robot.get_observation")
    if snapshot_now:
        reference_snapshot = obs

    written: list[str] = []

    def enc(v: Any) -> Any:
        return cs.encode(v, root, written, fsync)

    # The reference snapshot: new (now, given, or from a file) or the previous checkpoint's.
    reference: dict[str, Any] | None = None
    if reference_snapshot is not None or reference_snapshot_path:
        src_path, src_sha = None, None
        if reference_snapshot is None:
            src_path = str(Path(str(reference_snapshot_path)).expanduser().resolve())
            src_sha = cs.file_sha256(src_path)
            reference_snapshot = cs.load_frame_file(src_path)
        if reference_items is None:
            reference_items = []
            if grid is not None:
                reference_items += [
                    {"id": str(p["piece"]), "cell": [int(p["col"]), int(p["row"])]}
                    for p in placements or []
                ]
            reference_items += [
                it for it in items or [] if isinstance(it, dict) and any(k in it for k in ITEM_KEYS)
            ]
        reference = {
            "snapshot": enc(_only_camera(reference_snapshot, camera_name)),
            "snapshot_path": src_path,
            "snapshot_sha256": src_sha,
            "items": enc(reference_items),
            "step_index": int(step_index),
            "camera_name": camera_name,
            "grid": enc(grid),
            "diff_args": enc(diff_args or {}),
            "taken_at": datetime.now(timezone.utc).isoformat(),
        }
    elif carry_reference and prev is not None and prev.get("reference"):
        reference = prev["reference"]
        if reference_items is not None or diff_args is not None:
            reference = dict(reference)
            if reference_items is not None:
                reference["items"] = enc(reference_items)
            if diff_args is not None:
                reference["diff_args"] = enc(diff_args)

    led: dict[str, Any] = {}
    if ledger is not None or ledger_path:
        led["backtracking"] = _ledger_entry(ledger, ledger_path, notes, "backtracking")
    for name, v in (ledgers or {}).items():
        if isinstance(v, dict) and set(v) <= {"path", "data"}:
            led[name] = _ledger_entry(v.get("data"), v.get("path"), notes, name)
        elif isinstance(v, str):
            led[name] = _ledger_entry(None, v, notes, name)
        else:
            led[name] = _ledger_entry(v, None, notes, name)

    if arms is None and read_arms and isinstance(obs, dict):
        arms = [
            {"id": str(a.get("id", a.get("name", i))), **a}
            for i, a in enumerate(obs.get("arms") or [])
        ]
    kind = task_kind or ("lego" if bricks and not placements else "jigsaw" if placements else "")
    now = time.time()
    payload: dict[str, Any] = {
        "schema": cs.SCHEMA,
        "schema_version": cs.SCHEMA_VERSION,
        "task_id": str(task_id),
        "task_kind": kind or "generic",
        "seq": seq,
        "step_index": int(step_index),
        "created_unix": now,
        "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "writer": {"host": socket.gethostname(), "pid": os.getpid()},
        "placements": enc(placements or []),
        "bricks": enc(bricks or []),
        "items": enc(items or []),
        "pieces": enc(pieces),
        "pending": enc(pending or []),
        "in_progress": enc(in_progress),
        "ledgers": enc(led),
        "reference": reference,
        "build": enc(build),
        "arms": enc(arms or []),
        "grid": enc(grid),
        "extra": enc(extra or {}),
        "previous": (
            {"seq": prev_seq, "step_index": prev.get("step_index")} if prev is not None else None
        ),
    }
    payload["activity"], n_new = cs.activity_since(prev, payload)
    data = cs.envelope(payload)
    path = cs.checkpoint_path(root, task, seq)
    cs.atomic_write(path, data, fsync=fsync)

    removed = []
    for _, _, old in cs.list_checkpoints(root, task)[keep:]:
        try:
            old.unlink()
            removed.append(old.name)
        except OSError as exc:
            notes.append(f"could not remove {old.name}: {exc}")
    blobs_removed = _gc_blobs(root, set(written))
    ref_step = int(reference["step_index"]) if reference else -1
    detail = (
        f"saved {task} step {step_index} as seq {seq} ({len(data) / 1e3:.1f} kB, "
        f"{len(written)} new blobs); kept {min(keep, len(existing) + 1)}"
        + (f", reference from step {ref_step}" if reference else ", no reference snapshot")
    )
    return {
        "route": "saved",
        "checkpoint_path": str(path),
        "task_id": str(task_id),
        "seq": seq,
        "step_index": int(step_index),
        "sha256": cs.sha256_bytes(cs.canonical_json(payload)),
        "bytes": len(data),
        "n_kept": len(cs.list_checkpoints(root, task)),
        "removed": removed,
        "blobs_written": len(written),
        "blobs_removed": blobs_removed,
        "n_activity": n_new,
        "previous_seq": prev_seq,
        "reference_step_index": ref_step,
        "detail": "; ".join([detail, *notes]),
    }
