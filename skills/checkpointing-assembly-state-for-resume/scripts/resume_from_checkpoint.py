"""Load the newest valid checkpoint of an assembly run, check it against a fresh view, and
say which items to re-verify before going on.

Written for the hub. Newest first, each checkpoint file is read, its payload hash checked and
its schema migrated; the first that passes is used and the rest are listed with the reason.
The believed state is then checked against now: detecting-disturbed-pieces-by-snapshot-diff
re-finds every item of the stored reference snapshot in a fresh observation, and for a brick
build verifying-build-against-model compares the placed bricks with what the camera sees.
Items those checks could not clear (moved, lost, hidden, placed after the reference, held in
a gripper or half-placed when the run stopped, touched by a ledger decision written after
the checkpoint) are returned as ``recheck``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

from gap import NodeContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoint_store as cs  # noqa: E402

DIFF_SKILL = "detecting-disturbed-pieces-by-snapshot-diff"
DIFF_SCRIPT = "scripts/diff_snapshots.py"
BUILD_SKILL = "verifying-build-against-model"
BUILD_SCRIPT = "scripts/verify_build.py"

MAX_DISTURBED_FRACTION = 0.25
"""``state_mismatch`` when at least this fraction of the checked items (and at least
MIN_MISMATCH_ITEMS) moved, vanished or are wrong: the board was bumped, cleared or swapped,
and re-verifying item by item costs more than perceiving it afresh. Design choice: a nudge
during a stop moves 1-3 neighbours (the snapshot-diff skill flags a moved piece's direct
neighbours too), a dropped tray or a person tidying moves most of them."""

MIN_MISMATCH_ITEMS = 3
"""With fewer moved items than this the route stays resumed_with_rechecks however small
the board (two of six pieces knocked is a recheck, not a lost state)."""

CHANGED_FRACTION_RECHECK = 0.4
"""An item the snapshot diff calls unchanged is still rechecked when at least this fraction
of its pixels changed. Unmoved items read at most 0.21 (99th percentile 0.05) over 3475 items
of 360 synthetic before/after pairs (seeds 0-19, jigsaw / bricks / dominoes, overhead and
wrist); two removed 2 mm jigsaw pieces of the evaluation seeds read 0.48 and 0.86 with a
match of 0.56-0.59, just over the diff skill's 0.55 lost threshold."""

REASON_RANK = {
    "lost": 0,
    "held_when_stopped": 1,
    "missing": 2,
    "misplaced": 3,
    "wrong_colour": 4,
    "moved": 5,
    "appearance_changed": 6,
    "in_progress_when_stopped": 6,
    "ledger_after_checkpoint": 7,
    "unseated": 8,
    "extra": 9,
    "unexpected_object": 10,
    "occluded": 11,
    "unverified_at_save": 12,
    "placed_after_reference": 13,
    "not_checked": 14,
}
"""Order of the recheck list: things that are surely wrong first, things merely unseen last."""


class Output(TypedDict):
    route: str
    checkpoint_path: str
    task_id: str
    task_kind: str
    step_index: int
    seq: int
    schema_version_found: int
    migrated_from: list[int]
    age_s: float
    placements: list[dict[str, Any]]
    bricks: list[dict[str, Any]]
    items: list[dict[str, Any]]
    pieces: Any
    pending: list[Any]
    in_progress: Any
    ledger: Any
    ledger_path: str
    ledgers: dict[str, Any]
    arms: list[dict[str, Any]]
    grid: Any
    extra: dict[str, Any]
    activity: list[dict[str, Any]]
    recheck: list[dict[str, Any]]
    n_recheck: int
    recheck_ids: list[str]
    lost_items: list[str]
    unexpected: list[dict[str, Any]]
    snapshot_check: dict[str, Any]
    build_check: dict[str, Any]
    ledger_check: dict[str, Any]
    skipped: list[dict[str, str]]
    n_skipped: int
    stale_temp_files: list[str]
    state_json: str
    detail: str


def _empty(detail: str, skipped: list, temps: list) -> Output:
    return {
        "route": "no_checkpoint",
        "checkpoint_path": "",
        "task_id": "",
        "task_kind": "",
        "step_index": -1,
        "seq": -1,
        "schema_version_found": 0,
        "migrated_from": [],
        "age_s": -1.0,
        "placements": [],
        "bricks": [],
        "items": [],
        "pieces": None,
        "pending": [],
        "in_progress": None,
        "ledger": None,
        "ledger_path": "",
        "ledgers": {},
        "arms": [],
        "grid": None,
        "extra": {},
        "activity": [],
        "recheck": [],
        "n_recheck": 0,
        "recheck_ids": [],
        "lost_items": [],
        "unexpected": [],
        "snapshot_check": {},
        "build_check": {},
        "ledger_check": {},
        "skipped": skipped,
        "n_skipped": len(skipped),
        "stale_temp_files": temps,
        "state_json": "",
        "detail": detail,
    }


def _ids(state: dict) -> dict[str, dict]:
    """Every placed thing by id: placements by piece, bricks and items by id (or index)."""
    out: dict[str, dict] = {}
    for p in state.get("placements") or []:
        out[str(p["piece"])] = {"kind": "piece", "col": p.get("col"), "row": p.get("row")}
    for i, b in enumerate(state.get("bricks") or []):
        out[str(b.get("id", i))] = {"kind": "brick", "cell": b.get("cell")}
    for i, it in enumerate(state.get("items") or []):
        rec = {"kind": "item"}
        for k in ("cell", "pose", "col", "row"):
            if k in it:
                rec[k] = it[k]
        out[str(it.get("id", i))] = rec
    return out


class _Rechecks:
    def __init__(self, placed: dict[str, dict]):
        self.placed = placed
        self.by_id: dict[str, dict] = {}

    def add(self, rid: Any, reason: str, **info: Any) -> None:
        rid = str(rid)
        rec = self.by_id.setdefault(rid, {"id": rid, **self.placed.get(rid, {"kind": "region"})})
        rec.setdefault("reasons", [])
        if reason not in rec["reasons"]:
            rec["reasons"].append(reason)
        for k, v in info.items():
            if v is not None:
                rec[k] = v

    def ordered(self) -> list[dict]:
        out = list(self.by_id.values())
        for r in out:
            r["reasons"].sort(key=lambda s: REASON_RANK.get(s, 99))
        out.sort(key=lambda r: (REASON_RANK.get(r["reasons"][0], 99), r["id"]))
        return out


def _pick_camera(obs: Any, name: str, need_depth: bool) -> dict | None:
    if not isinstance(obs, dict):
        return None
    cams = obs.get("cameras") if "cameras" in obs else [obs]
    if isinstance(cams, dict):
        cams = [{"name": k, **v} for k, v in cams.items()]
    cams = [c for c in cams or [] if isinstance(c, dict)]
    if need_depth:
        cams = [
            c
            for c in cams
            if c.get("depth") is not None
            and c.get("intrinsics") is not None
            and c.get("pose") is not None
        ]
    if name:
        cams = [c for c in cams if c.get("name") == name] or ([] if len(cams) != 1 else cams)
    return cams[0] if cams else None


def _ledger_check(state: dict, restore: bool, fsync: bool, rc: _Rechecks, notes: list) -> dict:
    out: dict[str, Any] = {}
    for name, led in (state.get("ledgers") or {}).items():
        led = led or {}
        path, data = led.get("path"), led.get("data")
        res: dict[str, Any] = {"path": path}
        if not path:
            res["status"] = "embedded"
            out[name] = res
            continue
        p = Path(path)
        file_data = None
        if p.is_file():
            raw = p.read_bytes()
            try:
                file_data = json.loads(raw)
            except json.JSONDecodeError:
                file_data = None
            if file_data is not None and led.get("sha256") == cs.sha256_bytes(raw):
                res["status"] = "same"
                out[name] = res
                continue
        if file_data is None:
            if data is not None and restore:
                cs.atomic_write(p, json.dumps(data, indent=1).encode(), fsync=fsync)
                res["status"] = "restored"
                notes.append(f"ledger {name} was missing or unreadable; restored the saved copy")
            else:
                res["status"] = "missing"
            out[name] = res
            continue
        if data is None:
            res["status"] = "no_saved_copy"
            state["ledgers"][name]["data"] = file_data
            out[name] = res
            continue
        old_events = data.get("events") or []
        new_events = file_data.get("events") or []
        if len(new_events) > len(old_events):
            res["status"] = "file_newer"
            res["new_events"] = len(new_events) - len(old_events)
            for e in new_events[len(old_events) :]:
                rc.add(e.get("piece"), "ledger_after_checkpoint", event=e.get("id"))
                for act in (e.get("decision") or {}).get("actions") or []:
                    if act.get("piece") is not None:
                        rc.add(act["piece"], "ledger_after_checkpoint", op=act.get("op"))
            notes.append(
                f"ledger {name} has {res['new_events']} decision(s) newer than the checkpoint; "
                "their pieces are rechecked (the actions may be half done)"
            )
        else:
            res["status"] = "differs"
        state["ledgers"][name]["data"] = file_data
        out[name] = res
    return out


def _snapshot_check(
    ctx: NodeContext,
    ref: dict,
    obs: Any,
    root: Path,
    placed: dict[str, dict],
    rc: _Rechecks,
    notes: list,
) -> tuple[dict, set[str], bool]:
    """(summary, ids judged unchanged, unregistrable)."""
    errors: list[str] = []
    snap = cs.decode(ref.get("snapshot"), root, errors)
    if snap is None and ref.get("snapshot_path"):
        sp = Path(ref["snapshot_path"])
        if not sp.is_file():
            errors.append(f"snapshot file {sp} is missing")
        elif ref.get("snapshot_sha256") and cs.file_sha256(sp) != ref["snapshot_sha256"]:
            errors.append(f"snapshot file {sp} changed since the checkpoint")
        else:
            snap = cs.load_frame_file(sp)
    items = cs.decode(ref.get("items") or [], root, errors)
    if errors or snap is None:
        return {"status": "reference_damaged", "errors": errors}, set(), False
    if not items:
        return {"status": "no_items"}, set(), False
    mod = cs.sibling_module(DIFF_SKILL, DIFF_SCRIPT)
    if mod is None:
        notes.append(f"{DIFF_SKILL} is not installed: the snapshot was not compared")
        return {"status": "not_installed"}, set(), False
    kwargs = dict(cs.decode(ref.get("diff_args") or {}, root, errors))
    grid = cs.decode(ref.get("grid"), root, errors)
    if grid is not None:
        kwargs.setdefault("grid", grid)
    try:
        out = mod.run(
            ctx,
            reference=snap,
            items=items,
            current=obs,
            camera_name=ref.get("camera_name") or "",
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a failed check means "recheck", not a crash
        notes.append(f"snapshot diff failed: {exc}")
        return {"status": "failed", "error": str(exc)}, set(), False
    by_id = {str(r["id"]): r for r in out.get("items") or []}
    ok = set()
    gone = sorted(rid for rid in by_id if rid not in placed)
    if gone:
        notes.append(f"{len(gone)} reference item(s) were removed after the snapshot, as recorded")
    for rid, r in by_id.items():
        st = r.get("status")
        if rid not in placed:
            continue
        if st == "unchanged":
            frac = float(r.get("changed_fraction") or 0.0)
            if frac >= CHANGED_FRACTION_RECHECK:
                rc.add(rid, "appearance_changed", changed_fraction=frac, match=r.get("match"))
            else:
                ok.add(rid)
        elif st in ("moved", "lost"):
            rc.add(
                rid,
                st,
                shift_mm=r.get("shift_mm"),
                rotation_deg=r.get("rotation_deg"),
                depth_change_mm=r.get("depth_change_mm"),
                translation_world=r.get("translation_world"),
            )
        elif st == "occluded":
            rc.add(rid, "occluded", visible_fraction=r.get("visible_fraction"))
    for i, blob in enumerate(out.get("unexpected") or []):
        rc.add(
            f"unexpected_{i}",
            "unexpected_object",
            centroid_px=blob.get("centroid_px"),
            kind_of_change=blob.get("kind"),
            position_world=blob.get("position_world"),
            cell=blob.get("cell"),
            area_mm2=blob.get("area_mm2"),
        )
    summary = {
        "status": out["route"],
        "n_items": len(by_id) - len(gone),
        "removed_since_reference": gone,
        "n_unchanged": len(ok),
        "disturbed": list(out.get("disturbed") or []),
        "occluded": list(out.get("occluded") or []),
        "unexpected": list(out.get("unexpected") or []),
        "max_shift_mm": out.get("max_shift_mm"),
        "camera_shift_px": out.get("camera_shift_px"),
        "rho": (out.get("registration") or {}).get("rho"),
        "detail": out.get("detail", ""),
    }
    return summary, ok, out["route"] == "unregistrable"


def _build_check(
    ctx: NodeContext,
    state: dict,
    obs: Any,
    observed: Any,
    root: Path,
    rc: _Rechecks,
    notes: list,
) -> tuple[dict, set[str]]:
    bricks = state.get("bricks") or []
    build = cs.decode(state.get("build") or {}, root, [])
    target = [dict(b, id=b.get("id", i)) for i, b in enumerate(bricks)]
    if not target:
        return {"status": "no_bricks"}, set()
    mod = cs.sibling_module(BUILD_SKILL, BUILD_SCRIPT)
    if mod is None:
        notes.append(f"{BUILD_SKILL} is not installed: the build was not compared")
        return {"status": "not_installed"}, set()
    kwargs = {k: v for k, v in build.items() if k not in ("camera_name", "args")}
    kwargs.update(build.get("args") or {})
    kwargs.setdefault("check_colour", False)
    if observed is None:
        cam = _pick_camera(obs, build.get("camera_name") or "", need_depth=True)
        if cam is None or kwargs.get("plate_pose") is None or kwargs.get("plate_studs") is None:
            notes.append(
                "no depth frame with pose, or no plate_pose / plate_studs: build not compared"
            )
            return {"status": "no_view"}, set()
        kwargs["camera"] = cam
    try:
        out = mod.run(ctx, target=target, observed=observed, **kwargs)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"build check failed: {exc}")
        return {"status": "failed", "error": str(exc)}, set()
    ok = {str(m["target"]["id"]) for m in out.get("matched") or []}
    for r in out.get("missing") or []:
        rc.add(r["id"], "missing")
    for r in out.get("misplaced") or []:
        rc.add(
            r["id"],
            "misplaced",
            shift_studs=r.get("shift_studs"),
            shift_layers=r.get("shift_layers"),
        )
    for r in out.get("wrong_colour") or []:
        rc.add(r["id"], "wrong_colour", seen_colour=r.get("seen_colour"))
    for r in out.get("unseated") or []:
        rc.add(
            r["id"], "unseated", how=r.get("how"), height_residual_mm=r.get("height_residual_mm")
        )
        ok.discard(str(r["id"]))
    for i, r in enumerate(out.get("extra") or []):
        rc.add(f"extra_{i}", "extra", cell=r.get("cell"), size=r.get("size"))
    for r in out.get("unverified") or []:
        rc.add(r["id"], "occluded")
    wrong = sum(int(out.get(k, 0)) for k in ("n_missing", "n_misplaced", "n_wrong_colour"))
    return {
        "status": out["route"],
        "score": out.get("score"),
        "n_target": out.get("n_target"),
        "n_matched": out.get("n_matched"),
        "n_wrong": wrong,
        "n_extra": out.get("n_extra"),
        "n_unseated": out.get("n_unseated"),
        "detail": out.get("detail", ""),
    }, ok


def run(
    ctx: NodeContext,
    checkpoint_dir: str,
    task_id: str = "",
    current: Any = None,
    verify: str = "auto",
    build_observed: Any = None,
    expect_grid: list[int] | None = None,
    restore_ledgers: bool = True,
    max_disturbed_fraction: float = MAX_DISTURBED_FRACTION,
    fsync: bool = True,
) -> Output:
    if verify not in ("auto", "none"):
        raise ValueError(f"verify must be 'auto' or 'none', got {verify!r}")
    root = Path(checkpoint_dir).expanduser()
    task = cs.safe_task(task_id) if task_id else None
    temps = cs.stale_temps(root)
    files = cs.list_checkpoints(root, task)
    skipped: list[dict[str, str]] = []
    state = None
    found_v, migrated, path, seq = 0, [], None, -1
    for _, s, p in files:
        try:
            state, found_v, migrated = cs.read_checkpoint(p)
            path, seq = p, s
            break
        except cs.CheckpointError as exc:
            skipped.append({"file": p.name, "reason": str(exc)})
    if state is None:
        where = f"for task {task_id!r} " if task_id else ""
        why = (
            f"no valid checkpoint {where}in {root} ({len(files)} file(s), all unusable)"
            if files
            else f"no checkpoint {where}in {root}"
        )
        return _empty(why, skipped, temps)

    notes: list[str] = []
    if skipped:
        notes.append(f"skipped {len(skipped)} newer unusable file(s): {skipped[0]['reason']}")
    if migrated:
        notes.append(f"migrated from schema {found_v} to {cs.SCHEMA_VERSION}")
    state.setdefault("ledgers", {})
    placed = _ids(state)
    rc = _Rechecks(placed)
    mismatch: list[str] = []

    grid = state.get("grid") or {}
    if expect_grid is not None and grid:
        have = [grid.get("cols"), grid.get("rows")]
        if have != [int(v) for v in expect_grid]:
            mismatch.append(f"checkpoint grid {have} is not the expected {list(expect_grid)}")

    ledger_check = _ledger_check(state, restore_ledgers, fsync, rc, notes)

    # Held or half-done when the run stopped.
    lost = []
    for arm in state.get("arms") or []:
        held = arm.get("holding")
        if held not in (None, ""):
            lost.append(str(held))
            rc.add(held, "held_when_stopped", arm=arm.get("id"))
    ip = state.get("in_progress")
    if isinstance(ip, dict):
        ip_id = ip.get("piece", ip.get("id"))
        if ip_id is not None:
            rc.add(ip_id, "in_progress_when_stopped", phase=ip.get("phase"))
    for p in state.get("placements") or []:
        if p.get("verified") is False:
            rc.add(p["piece"], "unverified_at_save")
    for i, b in enumerate(state.get("bricks") or []):
        if b.get("verified") is False:
            rc.add(b.get("id", i), "unverified_at_save")

    snapshot_check: dict[str, Any] = {"status": "skipped"}
    build_check: dict[str, Any] = {"status": "skipped"}
    judged: set[str] = set()
    ref = state.get("reference") or {}
    ref_ids: set[str] = set()
    checks_ran = False
    if verify == "auto":
        want_diff = bool(ref) and bool(ref.get("items"))
        want_build = bool(state.get("bricks")) and state.get("build") is not None
        obs = current
        needs_obs = want_diff or (want_build and build_observed is None)
        if needs_obs and obs is None:
            try:
                obs = ctx.tool("robot.get_observation")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"robot.get_observation failed: {exc}")
                obs = None
        if want_diff and obs is not None:
            ref_ids = {
                str(it.get("id"))
                for it in cs.decode(ref.get("items") or [], root, [])
                if isinstance(it, dict)
            }
            snapshot_check, ok, unreg = _snapshot_check(ctx, ref, obs, root, placed, rc, notes)
            judged |= ok
            checks_ran |= snapshot_check["status"] not in (
                "reference_damaged",
                "not_installed",
                "failed",
            )
            if unreg:
                mismatch.append(
                    "the fresh view does not register onto the reference snapshot "
                    "(camera moved or scene changed)"
                )
            n = snapshot_check.get("n_items") or 0
            moved = sum(
                1
                for r in rc.by_id.values()
                if {"moved", "lost", "appearance_changed"} & set(r.get("reasons", []))
                and r["id"] in ref_ids
            )
            if n and moved >= MIN_MISMATCH_ITEMS and moved / n >= max_disturbed_fraction:
                mismatch.append(f"{moved} of {n} reference items moved or vanished")
            if snapshot_check["status"] == "reference_damaged":
                notes.append("reference snapshot damaged: " + "; ".join(snapshot_check["errors"]))
        elif want_diff:
            snapshot_check = {"status": "no_observation"}
        if want_build and (obs is not None or build_observed is not None):
            build_check, ok = _build_check(ctx, state, obs, build_observed, root, rc, notes)
            judged |= ok
            checks_ran |= build_check["status"] not in ("no_view", "not_installed", "failed")
            nt = int(build_check.get("n_target") or 0)
            nw = int(build_check.get("n_wrong") or 0) + int(build_check.get("n_extra") or 0)
            if nt and nw >= MIN_MISMATCH_ITEMS and nw / nt >= max_disturbed_fraction:
                mismatch.append(f"{nw} of {nt} bricks are missing, misplaced, wrong or extra")
        elif want_build:
            build_check = {"status": "no_observation"}

        # Everything placed that no check cleared.
        for rid in placed:
            if rid in judged or rid in rc.by_id:
                continue
            if checks_ran and ref_ids and rid not in ref_ids and placed[rid]["kind"] != "brick":
                rc.add(rid, "placed_after_reference")
            else:
                rc.add(rid, "not_checked")
    else:
        notes.append("verify none: the state is returned as saved, nothing compared")

    recheck = rc.ordered()
    created = state.get("created_unix")
    age = round(time.time() - float(created), 1) if created is not None else -1.0
    if mismatch:
        route = "state_mismatch"
        head = "the board does not match the checkpoint: " + "; ".join(mismatch)
    elif recheck:
        route = "resumed_with_rechecks"
        head = f"resumed step {state['step_index']}; {len(recheck)} item(s) to re-verify"
    else:
        route = "resumed"
        head = f"resumed step {state['step_index']}; every placed item checked out"
    bt = (state.get("ledgers") or {}).get("backtracking") or {}
    out: Output = {
        "route": route,
        "checkpoint_path": str(path),
        "task_id": str(state.get("task_id")),
        "task_kind": str(state.get("task_kind") or ""),
        "step_index": int(state["step_index"]),
        "seq": seq,
        "schema_version_found": found_v,
        "migrated_from": migrated,
        "age_s": age,
        "placements": cs.deep(state.get("placements") or []),
        "bricks": cs.deep(state.get("bricks") or []),
        "items": cs.decode(state.get("items") or [], root, []),
        "pieces": cs.deep(state.get("pieces")),
        "pending": cs.deep(state.get("pending") or []),
        "in_progress": cs.deep(state.get("in_progress")),
        "ledger": cs.deep(bt.get("data")),
        "ledger_path": str(bt.get("path") or ""),
        "ledgers": cs.deep(state.get("ledgers") or {}),
        "arms": cs.deep(state.get("arms") or []),
        "grid": cs.deep(state.get("grid")),
        "extra": cs.deep(state.get("extra") or {}),
        "activity": cs.deep(state.get("activity") or []),
        "recheck": recheck,
        "n_recheck": len(recheck),
        "recheck_ids": [r["id"] for r in recheck],
        "lost_items": lost,
        "unexpected": snapshot_check.get("unexpected", []),
        "snapshot_check": snapshot_check,
        "build_check": build_check,
        "ledger_check": ledger_check,
        "skipped": skipped,
        "n_skipped": len(skipped),
        "stale_temp_files": temps,
        "state_json": "",
        "detail": "; ".join([head, *notes]),
    }
    out["state_json"] = json.dumps(
        {k: state.get(k) for k in state if k not in ("reference", "activity")}, default=str
    )
    return out
