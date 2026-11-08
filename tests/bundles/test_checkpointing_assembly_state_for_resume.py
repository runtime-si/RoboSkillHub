"""checkpointing-assembly-state-for-resume: atomic saves, torn and corrupt files, killed
writers, schema migration, ledgers, and resuming against a disturbed synthetic board."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

from . import _build_scene as B
from . import _snapshot_scenes as S

BUNDLE = "checkpointing-assembly-state-for-resume"


@pytest.fixture(scope="module")
def bundle(skills_registry):
    b = skills_registry.get(BUNDLE)
    assert b.kind == "skill"
    return b


@pytest.fixture(scope="module")
def save(bundle):
    return bundle.canonical_scripts["save_checkpoint"].module


@pytest.fixture(scope="module")
def resume(bundle):
    return bundle.canonical_scripts["resume_from_checkpoint"].module


@pytest.fixture(scope="module")
def cs(save):
    return save.cs


def _placements(n, cols=22):
    return [{"piece": f"p{i}", "col": i % cols, "row": i // cols} for i in range(n)]


def _ledger(events=0):
    return {
        "version": 1,
        "grid": [22, 25],
        "forbidden": [],
        "attempts": {},
        "unexplained": {},
        "p_right": {},
        "events": [
            {
                "id": f"e{i}",
                "piece": f"p{i}",
                "col": i,
                "row": 0,
                "kind": "not_seated",
                "involved": [f"p{i}@{i},0"],
                "decision": {"actions": [{"op": "reseat", "piece": f"p{i}", "col": i, "row": 0}]},
            }
            for i in range(events)
        ],
        "neighbour_removals": {},
        "deferred_cells": [],
        "backtracks": events,
    }


# --- saving and loading --------------------------------------------------------------------------


def test_save_writes_a_hashed_versioned_envelope_and_resume_returns_the_state(
    save, resume, cs, tmp_path
):
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(_ledger(2)))
    pl = _placements(30)
    queue = [{"piece": "p30", "col": 8, "row": 1, "angle_deg": 90.0}]
    arms = [{"id": "left", "joints": np.zeros(6), "gripper": "open"}, {"id": "right"}]
    out = save.run(
        FakeContext(),
        str(tmp_path / "ck"),
        "hogwarts",
        30,
        placements=pl,
        pending=queue,
        ledger_path=str(led),
        arms=arms,
        grid={"cols": 22, "rows": 25},
        extra={"note": "after the frame"},
    )
    assert out["route"] == "saved" and out["seq"] == 1 and out["step_index"] == 30
    doc = json.loads(Path(out["checkpoint_path"]).read_text())
    assert doc["schema_version"] == 3 and doc["schema"] == cs.SCHEMA
    assert doc["sha256"] == cs.sha256_bytes(cs.canonical_json(doc["payload"])) == out["sha256"]
    assert doc["payload"]["created_at"] and doc["payload"]["writer"]["pid"] == os.getpid()

    r = resume.run(FakeContext(), str(tmp_path / "ck"), "hogwarts", verify="none")
    assert r["route"] == "resumed", r["detail"]
    assert r["placements"] == pl and r["pending"] == queue and r["step_index"] == 30
    assert r["ledger"]["backtracks"] == 2 and r["ledger_path"] == str(led.resolve())
    assert r["arms"][0]["joints"] == [0.0] * 6 and r["grid"] == {"cols": 22, "rows": 25}
    assert r["extra"] == {"note": "after the frame"} and r["schema_version_found"] == 3
    assert r["ledger_check"]["backtracking"]["status"] == "same"
    assert json.loads(r["state_json"])["task_id"] == "hogwarts"


def test_nothing_to_resume(resume, save, tmp_path):
    r = resume.run(FakeContext(), str(tmp_path / "nowhere"), "t")
    assert r["route"] == "no_checkpoint" and "no checkpoint" in r["detail"]
    save.run(FakeContext(), str(tmp_path), "other_task", 3, placements=_placements(3))
    assert resume.run(FakeContext(), str(tmp_path), "t")["route"] == "no_checkpoint"
    r = resume.run(FakeContext(), str(tmp_path), "", verify="none")
    assert r["route"] == "resumed" and r["task_id"] == "other_task"


def test_rotation_keeps_the_newest_n_and_counts_on(save, resume, tmp_path):
    for step in range(1, 9):
        out = save.run(
            FakeContext(), str(tmp_path), "t", step, placements=_placements(step), keep=3
        )
    assert out["seq"] == 8 and out["n_kept"] == 3
    names = sorted(p.name for p in tmp_path.glob("*.ckpt.json"))
    assert names == ["t.00000006.ckpt.json", "t.00000007.ckpt.json", "t.00000008.ckpt.json"]
    assert out["previous_seq"] == 7
    r = resume.run(FakeContext(), str(tmp_path), "t", verify="none")
    assert r["seq"] == 8 and len(r["placements"]) == 8


def test_the_activity_stream_records_what_changed_and_drops_the_oldest(save, cs, tmp_path):
    save.run(FakeContext(), str(tmp_path), "t", 2, placements=_placements(2))
    pl = _placements(3)
    pl[0]["col"] = 9  # p0 moved to another cell
    pl = pl[1:]  # ... and then lifted
    out = save.run(FakeContext(), str(tmp_path), "t", 3, placements=pl)
    doc = json.loads(Path(out["checkpoint_path"]).read_text())["payload"]["activity"]
    new = [a for a in doc if a["step_index"] == 3]
    kinds = {a["key"]: a["activity_type"] for a in new}
    assert kinds["/placements/p2"] == "INITIALISED" and kinds["/placements/p0"] == "UNSET"
    assert kinds["/step_index"] == "WRITE" and out["n_activity"] == len(new)
    step = next(a for a in new if a["key"] == "/step_index")
    assert step["previous_value"] == 2 and step["current_value"] == 3
    stream, n_new = cs.activity_since(
        {"activity": [{"key": f"/k{i}"} for i in range(10)], "step_index": 0},
        {"step_index": 1},
        max_size=5,
    )
    assert len(stream) == 5 and stream[-1]["key"] == "/step_index" and n_new >= 1
    assert stream[0]["key"] != "/k0"  # the oldest went first


# --- crashes and corruption ----------------------------------------------------------------------


def test_a_crash_in_the_middle_of_a_write_leaves_the_previous_checkpoint(
    save, resume, cs, tmp_path, monkeypatch
):
    save.run(FakeContext(), str(tmp_path), "t", 5, placements=_placements(5))

    def die(*_a, **_k):
        raise OSError("power cut before the rename")

    monkeypatch.setattr(cs.os, "replace", die)
    with pytest.raises(OSError):
        save.run(FakeContext(), str(tmp_path), "t", 6, placements=_placements(6))
    monkeypatch.undo()
    assert cs.stale_temps(tmp_path) == []  # the temp file was cleaned up
    # A save killed outright leaves its temp file behind: it is listed and ignored.
    (tmp_path / ".tmp-t.00000002.ckpt.json.abc").write_bytes(b'{"schema": "roboskillhub.as')
    r = resume.run(FakeContext(), str(tmp_path), "t", verify="none")
    assert r["route"] == "resumed" and r["step_index"] == 5 and r["seq"] == 1
    assert r["stale_temp_files"] == [".tmp-t.00000002.ckpt.json.abc"]


def test_a_torn_or_flipped_newest_file_falls_back_to_the_one_before(save, resume, tmp_path):
    save.run(FakeContext(), str(tmp_path), "t", 5, placements=_placements(5))
    out = save.run(FakeContext(), str(tmp_path), "t", 6, placements=_placements(6))
    p = Path(out["checkpoint_path"])
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])  # torn: a copy or disk that did not finish
    r = resume.run(FakeContext(), str(tmp_path), "t", verify="none")
    assert r["route"] == "resumed" and r["step_index"] == 5
    assert r["n_skipped"] == 1 and "not valid JSON" in r["skipped"][0]["reason"]

    flipped = raw.replace(b'"col": 5', b'"col": 7', 1)
    assert flipped != raw
    p.write_bytes(flipped)  # valid JSON, wrong content
    r = resume.run(FakeContext(), str(tmp_path), "t", verify="none")
    assert r["step_index"] == 5 and "hash mismatch" in r["skipped"][0]["reason"]

    p.write_bytes(b"")
    r = resume.run(FakeContext(), str(tmp_path), "t", verify="none")
    assert r["step_index"] == 5 and r["skipped"][0]["reason"] == "empty file"


def test_every_file_corrupt_is_no_checkpoint_with_the_reasons(save, resume, tmp_path):
    out = save.run(FakeContext(), str(tmp_path), "t", 1, placements=_placements(1))
    Path(out["checkpoint_path"]).write_text("[1, 2")
    r = resume.run(FakeContext(), str(tmp_path), "t")
    assert r["route"] == "no_checkpoint" and r["n_skipped"] == 1
    assert "all unusable" in r["detail"]


_CHILD = r"""
import importlib.util, sys
from gap.testing import FakeContext
spec = importlib.util.spec_from_file_location("save_ck", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
pl = [{"piece": f"p{i}", "col": i % 22, "row": i // 22, "score": 0.9} for i in range(550)]
frame = {"rgb": np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)}
step = 0
while True:
    step += 1
    frame["rgb"][0, 0, 0] = step % 255  # a new blob every save
    out = m.run(FakeContext(), sys.argv[2], "kill", step, placements=pl[:step % 550],
                reference_snapshot=frame, keep=3)
    print(out["seq"], flush=True)
"""


def test_a_writer_killed_at_random_moments_never_loses_the_last_checkpoint(
    bundle, resume, tmp_path
):
    script = str(bundle.canonical_scripts["save_checkpoint"].path)
    rng = np.random.default_rng(0)
    for trial in range(4):
        d = tmp_path / f"k{trial}"
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD, script, str(d)],
            stdout=subprocess.PIPE,
            text=True,
        )
        seen = 0
        deadline = time.time() + 30
        while seen < 2 and time.time() < deadline:  # let it save a couple first
            line = proc.stdout.readline()
            if line:
                seen = int(line)
        time.sleep(float(rng.uniform(0.0, 0.15)))
        proc.send_signal(signal.SIGKILL)
        rest = proc.communicate()[0].split()
        last = int(rest[-1]) if rest else seen
        r = resume.run(FakeContext(), str(d), "kill", verify="none")
        assert r["route"] == "resumed", r["detail"]
        assert r["seq"] in (last, last + 1)  # acknowledged, or written just before the kill
        assert r["n_skipped"] == 0


def test_a_corrupt_snapshot_blob_means_recheck_everything(save, resume, tmp_path):
    frame = {"rgb": np.full((60, 80, 3), 120, np.uint8)}
    items = [{"id": "p0", "polygon_px": [[10, 10], [30, 10], [30, 30], [10, 30]]}]
    save.run(
        FakeContext(),
        str(tmp_path),
        "t",
        1,
        placements=[{"piece": "p0", "col": 0, "row": 0}],
        reference_snapshot=frame,
        reference_items=items,
    )
    blob = next((tmp_path / "blobs").glob("*.npy.z"))
    blob.write_bytes(b"garbage")
    r = resume.run(FakeContext(), str(tmp_path), "t", current=frame)
    assert r["route"] == "resumed_with_rechecks"
    assert r["snapshot_check"]["status"] == "reference_damaged"
    assert r["recheck"][0]["reasons"] == ["not_checked"] and "corrupt" in r["detail"]


def test_unreferenced_blobs_are_collected_after_rotation(save, tmp_path, monkeypatch):
    monkeypatch.setattr(save, "BLOB_GRACE_S", -1.0)
    for step in range(4):
        frame = {"rgb": np.full((40, 40, 3), step * 10, np.uint8)}
        out = save.run(FakeContext(), str(tmp_path), "t", step, reference_snapshot=frame, keep=2)
    assert out["blobs_removed"] >= 1
    assert len(list((tmp_path / "blobs").glob("*.npy.z"))) == 2


def test_a_snapshot_file_is_copied_in_so_it_may_disappear(save, resume, tmp_path):
    rgb = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    f = tmp_path / "snap.npz"
    np.savez(f, rgb=rgb, depth_m=np.full((48, 64), 0.9, np.float32), K=np.eye(3))
    out = save.run(
        FakeContext(), str(tmp_path / "ck"), "t", 1, reference_snapshot_path=str(f), items=[]
    )
    assert out["reference_step_index"] == 1 and out["blobs_written"] == 2
    f.unlink()
    doc = json.loads(Path(out["checkpoint_path"]).read_text())["payload"]["reference"]
    assert doc["snapshot_path"] == str(f.resolve()) and len(doc["snapshot_sha256"]) == 64
    r = resume.run(FakeContext(), str(tmp_path / "ck"), "t", verify="none")
    assert r["route"] == "resumed"


def test_unstorable_values_fail_the_save_and_keep_the_old_files(save, tmp_path):
    save.run(FakeContext(), str(tmp_path), "t", 1)
    with pytest.raises(TypeError):
        save.run(FakeContext(), str(tmp_path), "t", 2, extra={"f": object()})
    assert [p.name for p in tmp_path.glob("*.ckpt.json")] == ["t.00000001.ckpt.json"]
    with pytest.raises(ValueError):
        save.run(FakeContext(), str(tmp_path), "t", 2, keep=0)


# --- migration -----------------------------------------------------------------------------------


def test_a_bare_v1_jigsaw_state_migrates(resume, tmp_path):
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(_ledger(1)))
    v1 = {"task_id": "jig", "placements": _placements(4), "ledger_path": str(led), "queue": [1]}
    (tmp_path / "jig.00000003.ckpt.json").write_text(json.dumps(v1))
    r = resume.run(FakeContext(), str(tmp_path), "jig", verify="none")
    assert r["route"] == "resumed" and r["schema_version_found"] == 1, r["recheck"]
    assert r["migrated_from"] == [1, 2] and r["seq"] == 3
    assert r["step_index"] == 4 and r["placements"] == _placements(4) and r["pending"] == [1]
    assert r["task_kind"] == "jigsaw" and r["ledger_path"] == str(led)
    assert r["ledger_check"]["backtracking"]["status"] == "no_saved_copy"
    assert r["ledger"]["backtracks"] == 1
    assert "migrated from schema 1" in r["detail"]


def test_a_bare_v1_planner_sequence_splits_into_built_and_pending(resume, tmp_path):
    seq = [{"id": i, "cell": {"x": i, "y": 0, "layer": 1}, "size": "2x2"} for i in range(5)]
    v1 = {"task_id": "car", "sequence": {"sequence": seq}, "placed": 2}
    (tmp_path / "car.00000001.ckpt.json").write_text(json.dumps(v1))
    r = resume.run(FakeContext(), str(tmp_path), "car", verify="none")
    assert [b["id"] for b in r["bricks"]] == [0, 1] and [b["id"] for b in r["pending"]] == [2, 3, 4]
    assert r["task_kind"] == "lego" and r["step_index"] == 2


def test_a_v2_envelope_migrates_and_a_newer_version_is_skipped(resume, cs, tmp_path):
    payload = {
        "task_id": "t",
        "step": 7,
        "placements": _placements(7),
        "queue": [],
        "arms": {"right": {"holding": "p9"}, "left": {}},
        "saved_at": 1.7e9,
    }
    env = {"schema": cs.SCHEMA, "schema_version": 2, "payload": payload}
    env["sha256"] = cs.sha256_bytes(cs.canonical_json(payload))
    (tmp_path / "t.00000004.ckpt.json").write_text(json.dumps(env))
    future = {"schema": cs.SCHEMA, "schema_version": 4, "payload": {"task_id": "t"}}
    future["sha256"] = cs.sha256_bytes(cs.canonical_json(future["payload"]))
    (tmp_path / "t.00000005.ckpt.json").write_text(json.dumps(future))
    r = resume.run(FakeContext(), str(tmp_path), "t")
    assert r["seq"] == 4 and r["migrated_from"] == [2]
    assert "newer than this skill" in r["skipped"][0]["reason"]
    assert [a["id"] for a in r["arms"]] == ["left", "right"]
    assert r["age_s"] > 0  # saved_at became created_unix
    assert r["lost_items"] == ["p9"] and r["recheck"][0]["reasons"] == ["held_when_stopped"]
    assert r["route"] == "resumed_with_rechecks"
    # Edited after writing: the v2 hash catches it too.
    env["payload"]["step"] = 8
    (tmp_path / "t.00000004.ckpt.json").write_text(json.dumps(env))
    assert resume.run(FakeContext(), str(tmp_path), "t")["route"] == "no_checkpoint"


def test_resumed_state_saves_again_as_the_current_version(save, resume, cs, tmp_path):
    v1 = {"task_id": "jig", "placements": _placements(4)}
    (tmp_path / "jig.00000001.ckpt.json").write_text(json.dumps(v1))
    r = resume.run(FakeContext(), str(tmp_path), "jig", verify="none")
    out = save.run(
        FakeContext(), str(tmp_path), "jig", r["step_index"] + 1, placements=r["placements"]
    )
    assert out["seq"] == 2 and out["previous_seq"] == 1
    assert (
        json.loads(Path(out["checkpoint_path"]).read_text())["schema_version"] == cs.SCHEMA_VERSION
    )


# --- what the stop left behind -------------------------------------------------------------------


def test_held_in_progress_and_unverified_items_are_rechecked_in_order(save, resume, tmp_path):
    pl = _placements(5)
    pl[4]["verified"] = False
    save.run(
        FakeContext(),
        str(tmp_path),
        "t",
        5,
        placements=pl,
        in_progress={"piece": "p6", "col": 6, "row": 0, "phase": "descending"},
        arms=[{"id": "left", "holding": "p5"}, {"id": "right", "holding": None}],
    )
    r = resume.run(FakeContext(), str(tmp_path), "t", verify="none")
    assert r["route"] == "resumed_with_rechecks"
    assert r["recheck_ids"] == ["p5", "p6", "p4"]
    assert r["recheck"][0]["arm"] == "left" and r["recheck"][1]["phase"] == "descending"
    assert r["lost_items"] == ["p5"]


def test_no_observation_means_every_placed_item_is_unchecked(save, resume, tmp_path):
    frame = {"rgb": np.zeros((40, 40, 3), np.uint8)}
    save.run(
        FakeContext(),
        str(tmp_path),
        "t",
        2,
        placements=_placements(2),
        reference_snapshot=frame,
        reference_items=[{"id": "p0", "polygon_px": [[1, 1], [9, 1], [9, 9]]}],
    )

    def boom(**_):
        raise RuntimeError("robot offline")

    r = resume.run(FakeContext({"robot.get_observation": boom}), str(tmp_path), "t")
    assert r["route"] == "resumed_with_rechecks" and r["n_recheck"] == 2
    assert all(x["reasons"] == ["not_checked"] for x in r["recheck"])
    assert "robot offline" in r["detail"] and r["snapshot_check"]["status"] == "no_observation"


def test_a_grid_other_than_expected_is_a_mismatch(save, resume, tmp_path):
    save.run(FakeContext(), str(tmp_path), "t", 1, grid={"cols": 22, "rows": 25})
    r = resume.run(FakeContext(), str(tmp_path), "t", expect_grid=[20, 27], verify="none")
    assert r["route"] == "state_mismatch" and "[22, 25]" in r["detail"]


# --- ledgers -------------------------------------------------------------------------------------


def test_ledger_newer_than_the_checkpoint_rechecks_its_pieces(save, resume, tmp_path):
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(_ledger(1)))
    save.run(
        FakeContext(), str(tmp_path / "ck"), "t", 3, placements=_placements(3), ledger_path=str(led)
    )
    newer = _ledger(1)
    newer["events"].append(
        {
            "id": "e9",
            "piece": "p2",
            "col": 2,
            "row": 0,
            "kind": "seam_mismatch",
            "involved": ["p2@2,0", "p1@1,0"],
            "decision": {
                "actions": [
                    {"op": "remove", "piece": "p1", "col": 1, "row": 0},
                    {"op": "place", "piece": "p7", "col": 1, "row": 0},
                ]
            },
        }
    )
    led.write_text(json.dumps(newer))
    r = resume.run(FakeContext(), str(tmp_path / "ck"), "t", verify="none")
    assert r["ledger_check"]["backtracking"] == {
        "path": str(led.resolve()),
        "status": "file_newer",
        "new_events": 1,
    }
    assert set(r["recheck_ids"]) == {"p1", "p2", "p7"}
    assert len(r["ledger"]["events"]) == 2  # the newer file is what the run goes on with


def test_a_missing_ledger_is_restored_from_the_checkpoint(save, resume, tmp_path):
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps(_ledger(3)))
    save.run(FakeContext(), str(tmp_path / "ck"), "t", 3, ledger_path=str(led))
    led.unlink()
    r = resume.run(FakeContext(), str(tmp_path / "ck"), "t", verify="none")
    assert r["ledger_check"]["backtracking"]["status"] == "restored"
    assert json.loads(led.read_text())["backtracks"] == 3
    led.write_text("{not json")
    r = resume.run(FakeContext(), str(tmp_path / "ck"), "t", verify="none", restore_ledgers=False)
    assert r["ledger_check"]["backtracking"]["status"] == "missing"


# --- resuming against a disturbed board (detecting-disturbed-pieces-by-snapshot-diff) ----------


@pytest.fixture(scope="module")
def jigsaw_board():
    sc = S.build("jigsaw", 1, "overhead")
    t, k = S.camera_pose("overhead")
    ref, ids = S.render(sc, t, k, seed=1, ss=2)
    items = [{"id": i, "mask": m} for i, m in S.item_masks(sc, ids).items() if m.sum() >= 4]
    pl = [
        {
            "piece": it["id"],
            "col": int(it["id"][1:].split("_")[0]),
            "row": int(it["id"].split("_")[1]),
        }
        for it in items
    ]
    return sc, t, k, S.observation(ref, sc), items, pl


def _save_board(save, d, board, **kw):
    sc, t, k, ref, items, pl = board
    args = {"placements": pl, "reference_snapshot": ref, "reference_items": items}
    args.update(kw)
    return save.run(FakeContext(), str(d), "board", len(pl), **args)


def _now(sc, t, k, seed=7):
    return S.observation(S.render(sc, t, k, seed=seed, ss=2)[0], sc)


def test_an_untouched_board_resumes_without_rechecks(save, resume, jigsaw_board, tmp_path):
    sc, t, k, *_ = jigsaw_board
    _save_board(save, tmp_path, jigsaw_board)
    ctx = FakeContext({"robot.get_observation": _now(sc, t, k)})
    r = resume.run(ctx, str(tmp_path), "board")
    assert r["route"] == "resumed", r["detail"]
    assert r["snapshot_check"]["status"] == "unchanged" and r["n_recheck"] == 0
    assert ctx.call_count("robot.get_observation") == 1


def test_a_piece_knocked_during_the_stop_is_rechecked_with_its_shift(
    save, resume, jigsaw_board, tmp_path
):
    sc, t, k, ref, items, pl = jigsaw_board
    _save_board(save, tmp_path, jigsaw_board)
    moved = items[3]["id"]
    cur = _now(S.shift(sc, moved, 0.004, 0.002), t, k)
    r = resume.run(FakeContext(), str(tmp_path), "board", current=cur)
    assert r["route"] == "resumed_with_rechecks", r["detail"]
    assert r["recheck"][0]["id"] == moved and r["recheck"][0]["reasons"] == ["moved"]
    assert 3.0 < r["recheck"][0]["shift_mm"] < 6.0
    assert len(r["recheck"]) <= 3  # at most the moved piece's direct neighbours besides it


def test_a_removed_piece_is_lost(save, resume, jigsaw_board, tmp_path):
    sc, t, k, ref, items, pl = jigsaw_board
    _save_board(save, tmp_path, jigsaw_board)
    gone = items[5]["id"]
    r = resume.run(FakeContext(), str(tmp_path), "board", current=_now(S.remove(sc, gone), t, k))
    assert r["route"] == "resumed_with_rechecks"
    assert gone in r["recheck_ids"]
    reasons = set(next(x for x in r["recheck"] if x["id"] == gone)["reasons"])
    assert {"lost", "moved", "appearance_changed"} & reasons


def test_an_item_matched_but_mostly_changed_is_still_rechecked(
    save, resume, cs, tmp_path, monkeypatch
):
    class Diff:
        @staticmethod
        def run(ctx, reference, items, current, camera_name, **kw):
            return {
                "route": "unchanged",
                "items": [
                    {"id": "p0", "status": "unchanged", "changed_fraction": 0.86, "match": 0.59},
                    {"id": "p1", "status": "unchanged", "changed_fraction": 0.21, "match": 0.9},
                ],
            }

    monkeypatch.setitem(cs._SIBLINGS, f"{resume.DIFF_SKILL}/{resume.DIFF_SCRIPT}", Diff)
    frame = {"rgb": np.zeros((40, 40, 3), np.uint8)}
    sq = [[1, 1], [9, 1], [9, 9], [1, 9]]
    save.run(
        FakeContext(),
        str(tmp_path),
        "t",
        2,
        placements=_placements(2),
        reference_snapshot=frame,
        reference_items=[{"id": "p0", "polygon_px": sq}, {"id": "p1", "polygon_px": sq}],
    )
    r = resume.run(FakeContext(), str(tmp_path), "t", current=frame)
    assert r["recheck_ids"] == ["p0"] and r["recheck"][0]["reasons"] == ["appearance_changed"]


def test_a_swept_board_is_a_state_mismatch(save, resume, jigsaw_board, tmp_path):
    sc, t, k, ref, items, pl = jigsaw_board
    _save_board(save, tmp_path, jigsaw_board)
    sw = sc
    for it in items[:6]:
        sw = S.shift(sw, it["id"], 0.006, -0.004)
    r = resume.run(FakeContext(), str(tmp_path), "board", current=_now(sw, t, k))
    assert r["route"] == "state_mismatch", r["detail"]
    assert "reference items moved or vanished" in r["detail"]


def test_another_scene_does_not_register_and_is_a_mismatch(save, resume, jigsaw_board, tmp_path):
    _save_board(save, tmp_path, jigsaw_board)
    other = S.build("lego", 4, "wrist")
    t2, k2 = S.camera_pose("wrist")
    r = resume.run(FakeContext(), str(tmp_path), "board", current=_now(other, t2, k2))
    assert r["route"] == "state_mismatch"
    assert r["snapshot_check"]["status"] == "unregistrable"


def test_pieces_placed_or_lifted_after_the_snapshot(save, resume, jigsaw_board, tmp_path):
    sc, t, k, ref, items, pl = jigsaw_board
    _save_board(save, tmp_path, jigsaw_board)
    lifted = pl[2]["piece"]
    later = [p for p in pl if p["piece"] != lifted] + [{"piece": "new1", "col": 5, "row": 4}]
    out = save.run(FakeContext(), str(tmp_path), "board", len(pl) + 1, placements=later)
    assert out["reference_step_index"] == len(pl)  # carried over from the first save
    r = resume.run(FakeContext(), str(tmp_path), "board", current=_now(S.remove(sc, lifted), t, k))
    assert r["route"] == "resumed_with_rechecks", r["detail"]
    assert r["recheck_ids"] == ["new1"]
    assert r["recheck"][0]["reasons"] == ["placed_after_reference"]
    assert r["snapshot_check"]["removed_since_reference"] == [lifted]


def test_jigsaw_cells_become_snapshot_items_through_the_grid(save, jigsaw_board, tmp_path):
    sc, t, k, ref, items, pl = jigsaw_board
    g = sc.grid
    grid = {
        "origin": {
            "position": {"x": g["x0"], "y": g["y0"], "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        "cols": g["cols"],
        "rows": g["rows"],
        "cell_size": g["cell"],
        "item_height": g["thick"],
    }
    out = save.run(
        FakeContext(), str(tmp_path), "board", 15, placements=pl, reference_snapshot=ref, grid=grid
    )
    doc = json.loads(Path(out["checkpoint_path"]).read_text())["payload"]["reference"]
    assert doc["items"][0] == {"id": pl[0]["piece"], "cell": [pl[0]["col"], pl[0]["row"]]}
    assert doc["grid"]["cols"] == g["cols"]


# --- bricks (verifying-build-against-model) ------------------------------------------------------

PLATE = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


def _build(seed=11):
    rng = np.random.default_rng(seed)
    scene = B.random_build(rng, "lego")
    _, t_cam, k, noise = B.cameras(rng, "lego")[1]
    return scene, t_cam, k, noise


def test_bricks_are_checked_against_what_the_camera_sees(save, resume, tmp_path):
    scene, t_cam, k, noise = _build()
    bricks = [b.as_target(i) for i, b in enumerate(scene.bricks)]
    build = {
        "plate_pose": PLATE,
        "plate_studs": [12, 12],
        "system": "lego",
        "camera_name": "left_wrist",
    }
    save.run(FakeContext(), str(tmp_path), "car", len(bricks), bricks=bricks, build=build)

    def obs(sc):
        depth, rgb = B.render(sc, np.eye(4), t_cam, k, noise, seed=3)
        cam = {"name": "left_wrist", "depth": depth, "rgb": rgb, "intrinsics": k, "pose": t_cam}
        return {"cameras": [cam]}

    r = resume.run(FakeContext({"robot.get_observation": obs(scene)}), str(tmp_path), "car")
    assert r["build_check"]["status"] == "matches", r["detail"]
    assert r["route"] == "resumed" and r["task_kind"] == "lego"

    top = B.tops(scene.bricks)[0]
    knocked = B.Scene(
        scene.system, scene.plate, [b for i, b in enumerate(scene.bricks) if i != top]
    )
    r = resume.run(FakeContext(), str(tmp_path), "car", current=obs(knocked))
    assert r["route"] == "resumed_with_rechecks", r["detail"]
    assert r["recheck"][0]["id"] == str(top) and r["recheck"][0]["reasons"] == ["missing"]


def test_bricks_from_upstream_perception_and_a_wrecked_build(save, resume, tmp_path):
    bricks = [
        {
            "id": f"b{i}",
            "cell": {"x": 2 * i, "y": 0, "layer": 1, "rotation": 0},
            "size": "2x2",
            "colour": "red",
        }
        for i in range(5)
    ]
    save.run(FakeContext(), str(tmp_path), "w", 5, bricks=bricks, build={"system": "lego"})
    seen = [dict(b) for b in bricks[:4]]
    r = resume.run(FakeContext(), str(tmp_path), "w", build_observed=seen)
    assert r["route"] == "resumed_with_rechecks" and r["recheck_ids"] == ["b4"]
    r = resume.run(FakeContext(), str(tmp_path), "w", build_observed=seen[:1])
    assert r["route"] == "state_mismatch" and "4 of 5 bricks" in r["detail"]


def test_examples_declare_every_route(bundle):
    root = Path(bundle.bundle_dir)
    routes = {
        "scripts/resume_from_checkpoint.py": {
            "resumed",
            "resumed_with_rechecks",
            "no_checkpoint",
            "state_mismatch",
        },
        "scripts/save_checkpoint.py": {"saved", "save_failed"},
    }
    for ex in (root / "examples").glob("*.json"):
        g = json.loads(ex.read_text())
        script = next(iter(g["nodes"].values()))["script"]
        declared = set(g["exit"]["success_values"]) | {g["on_error"]}
        assert routes[script] <= declared, ex.name
