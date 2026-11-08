"""Benchmark for checkpointing-assembly-state-for-resume (CPU, no robot).

    python -m tests.bundles._checkpoint_eval --part all --workers 10

kill:     a writer saving a 550-piece jigsaw state (backtracking ledger with 40 events, a new
          640 x 480 RGB-D snapshot every save) is SIGKILLed at a random moment; resume must
          return the last acknowledged checkpoint or the one being written.
corrupt:  one random byte of the newest checkpoint is changed, or the file is cut at a random
          length; resume must never return a state other than one that was saved.
disturb:  synthetic RGB-D boards (tests/bundles/_snapshot_scenes.py: jigsaw, bricks, dominoes;
          overhead and wrist cameras) saved with a reference snapshot, then resumed against a
          fresh render after a disturbance: none, one item moved 3-6 mm, one removed, two moved,
          a sweep (40 % moved 4-8 mm).
build:    random LEGO builds (tests/bundles/_build_scene.py) checked from the wrist depth:
          clean, a top brick removed, a top brick moved one stud.
timing:   save and resume times for the 550-piece state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills" / "checkpointing-assembly-state-for-resume" / "scripts"
sys.path.insert(0, str(ROOT))

from tests.bundles import _build_scene as B  # noqa: E402
from tests.bundles import _snapshot_scenes as S  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAVE = _load("save_checkpoint")
RESUME = _load("resume_from_checkpoint")


def big_state(rng):
    pl = [
        {"piece": f"p{i}", "col": i % 22, "row": i // 22, "score": 0.9, "margin": 0.2}
        for i in range(550)
    ]
    pieces = [
        {
            "piece": f"p{i}",
            "candidates": [
                {"col": int(c), "row": int(r), "score": float(s), "angle_deg": 0.0}
                for c, r, s in zip(
                    rng.integers(0, 22, 5),
                    rng.integers(0, 25, 5),
                    rng.uniform(0.5, 1, 5),
                    strict=True,
                )
            ],
        }
        for i in range(550)
    ]
    ledger = {
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
                "col": i % 22,
                "row": i // 22,
                "kind": "not_seated",
                "involved": [f"p{i}@{i % 22},{i // 22}"],
                "decision": {"actions": [{"op": "reseat", "piece": f"p{i}"}]},
            }
            for i in range(40)
        ],
        "neighbour_removals": {},
        "deferred_cells": [],
        "backtracks": 40,
    }
    return pl, pieces, ledger


# --- kill ----------------------------------------------------------------------------------------

_CHILD = r"""
import importlib.util, json, sys
import numpy as np
from gap.testing import FakeContext
sys.path.insert(0, sys.argv[3])
from tests.bundles._checkpoint_eval import big_state
spec = importlib.util.spec_from_file_location("save_ck", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng = np.random.default_rng(0)
pl, pieces, ledger = big_state(rng)
led = sys.argv[2] + "/ledger.json"
open(led, "w").write(json.dumps(ledger))
frame = {"name": "overhead", "rgb": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
         "depth": rng.uniform(0.8, 1.0, (480, 640)).astype(np.float32), "intrinsics": np.eye(3)}
step = 0
while True:
    step += 1
    frame["rgb"][0, 0, 0] = step % 255
    out = m.run(FakeContext(), sys.argv[2], "kill", step, placements=pl[: 100 + step % 450],
                pieces=pieces, ledger_path=led, reference_snapshot=frame, keep=5)
    print(out["seq"], flush=True)
"""


def kill_trial(seed):
    rng = np.random.default_rng(seed)
    d = tempfile.mkdtemp(prefix="ckkill")
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(SCRIPTS / "save_checkpoint.py"), d, str(ROOT)],
        stdout=subprocess.PIPE,
        text=True,
    )
    acked = 0
    target = int(rng.integers(1, 4))
    while acked < target:
        line = proc.stdout.readline()
        if not line:
            break
        acked = int(line)
    time.sleep(float(rng.uniform(0.0, 0.6)))
    proc.send_signal(signal.SIGKILL)
    rest = proc.communicate()[0].split()
    if rest:
        acked = int(rest[-1])
    r = RESUME.run(FakeContext(), d, "kill", verify="none")
    temps = len(r["stale_temp_files"])
    ok = r["route"] == "resumed" and r["seq"] in (acked, acked + 1)
    return {
        "ok": ok,
        "which": "acked" if r["seq"] == acked else "in_flight" if ok else "bad",
        "temps": temps,
        "skipped": r["n_skipped"],
        "route": r["route"],
        "led": r["ledger_check"].get("backtracking", {}).get("status"),
    }


# --- corrupt -------------------------------------------------------------------------------------


def corrupt_trial(seed):
    rng = np.random.default_rng(seed)
    d = Path(tempfile.mkdtemp(prefix="ckcor"))
    pl, pieces, ledger = big_state(rng)
    SAVE.run(FakeContext(), str(d), "t", 10, placements=pl[:200], ledger=ledger, fsync=False)
    out = SAVE.run(FakeContext(), str(d), "t", 11, placements=pl[:201], ledger=ledger, fsync=False)
    p = Path(out["checkpoint_path"])
    raw = bytearray(p.read_bytes())
    mode = "flip" if seed % 2 == 0 else "cut"
    if mode == "flip":
        i = int(rng.integers(len(raw)))
        raw[i] = (raw[i] + int(rng.integers(1, 256))) % 256
        p.write_bytes(bytes(raw))
    else:
        p.write_bytes(bytes(raw[: int(rng.integers(len(raw)))]))
    r = RESUME.run(FakeContext(), str(d), "t", verify="none")
    if r["step_index"] == 11:
        same = len(r["placements"]) == 201 and r["placements"] == pl[:201]
        res = "harmless" if same else "WRONG"
    elif r["step_index"] == 10:
        res = "fell_back" if len(r["placements"]) == 200 else "WRONG"
    else:
        res = "WRONG"
    return {"mode": mode, "result": res}


# --- disturb -------------------------------------------------------------------------------------


def disturb_trial(args):
    kind, cam, case, seed = args
    cases = ("none", "one_moved", "one_removed", "two_moved", "sweep")
    rng = np.random.default_rng(seed * 31 + cases.index(case))
    sc = S.build(kind, seed, cam)
    t, k = S.camera_pose(cam)
    ref, ids = S.render(sc, t, k, seed=seed, ss=2)
    items = [{"id": i, "mask": m} for i, m in S.item_masks(sc, ids).items() if m.sum() >= 4]
    d = tempfile.mkdtemp(prefix="ckdis")
    rows = [{"id": it["id"], "cell": None} for it in items]
    SAVE.run(
        FakeContext(),
        d,
        "b",
        len(items),
        items=rows,
        reference_snapshot=S.observation(ref, sc),
        reference_items=items,
        fsync=False,
    )
    order = [it["id"] for it in items]
    rng.shuffle(order)
    moved: list[str] = []
    cur_sc = sc
    if case == "one_moved":
        a = rng.uniform(0, 2 * math.pi)
        r_m = rng.uniform(0.003, 0.006)
        moved = order[:1]
        cur_sc = S.shift(sc, moved[0], r_m * math.cos(a), r_m * math.sin(a))
    elif case == "two_moved":
        moved = order[:2]
        for m in moved:
            a = rng.uniform(0, 2 * math.pi)
            r_m = rng.uniform(0.003, 0.006)
            cur_sc = S.shift(cur_sc, m, r_m * math.cos(a), r_m * math.sin(a))
    elif case == "one_removed":
        moved = order[:1]
        cur_sc = S.remove(sc, moved[0])
    elif case == "sweep":
        moved = order[: max(3, int(round(0.4 * len(order))))]
        for m in moved:
            a = rng.uniform(0, 2 * math.pi)
            r_m = rng.uniform(0.004, 0.008)
            cur_sc = S.shift(cur_sc, m, r_m * math.cos(a), r_m * math.sin(a))
    cur, _ = S.render(cur_sc, t, k, seed=seed + 1000, ss=2)
    t0 = time.perf_counter()
    r = RESUME.run(FakeContext(), d, "b", current=S.observation(cur, cur_sc))
    dt = time.perf_counter() - t0
    flagged = {x["id"] for x in r["recheck"] if x.get("kind") != "region"}
    return {
        "kind": kind,
        "cam": cam,
        "case": case,
        "route": r["route"],
        "found": len(set(moved) & flagged),
        "n_moved": len(moved),
        "extra": len(flagged - set(moved)),
        "n_items": len(items),
        "regions": sum(1 for x in r["recheck"] if x.get("kind") == "region"),
        "s": dt,
    }


# --- build ---------------------------------------------------------------------------------------

PLATE = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


def build_trial(args):
    case, seed = args
    rng = np.random.default_rng(seed)
    scene = B.random_build(rng, "lego")
    _, t_cam, k, noise = B.cameras(rng, "lego")[1]
    bricks = [b.as_target(i) for i, b in enumerate(scene.bricks)]
    d = tempfile.mkdtemp(prefix="ckbld")
    build = {"plate_pose": PLATE, "plate_studs": [12, 12], "system": "lego"}
    SAVE.run(FakeContext(), d, "car", len(bricks), bricks=bricks, build=build, fsync=False)
    target = None
    cur = scene
    if case != "clean":
        tops = B.tops(scene.bricks)
        target = int(tops[int(rng.integers(len(tops)))])
        rest = [b for i, b in enumerate(scene.bricks) if i != target]
        if case == "removed":
            cur = B.Scene(scene.system, scene.plate, rest)
        else:
            b = scene.bricks[target]
            moved = None
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = B.Brick(b.x + dx, b.y + dy, b.layer, b.size, b.rotation, b.colour, b.step)
                if B._free(rest, c, scene.plate):
                    moved = c
                    break
            if moved is None:
                return {"case": case, "skip": True}
            cur = B.Scene(scene.system, scene.plate, [*rest, moved])
    depth, rgb = B.render(cur, np.eye(4), t_cam, k, noise, seed=seed + 7)
    cam = {"name": "wrist", "depth": depth, "rgb": rgb, "intrinsics": k, "pose": t_cam}
    r = RESUME.run(FakeContext(), d, "car", current={"cameras": [cam]})
    first = r["recheck"][0] if r["recheck"] else {}
    hit = (
        r["route"] == "resumed"
        if case == "clean"
        else first.get("id") == str(target)
        and first["reasons"][0] in ("missing", "misplaced", "moved")
    )
    return {
        "case": case,
        "skip": False,
        "route": r["route"],
        "hit": bool(hit),
        "reason": (first.get("reasons") or ["-"])[0],
        "n_recheck": r["n_recheck"],
    }


# --- timing --------------------------------------------------------------------------------------


def timing():
    rng = np.random.default_rng(0)
    pl, pieces, ledger = big_state(rng)
    frame = {
        "name": "overhead",
        "rgb": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        "depth": rng.uniform(0.8, 1.0, (480, 640)).astype(np.float32),
        "intrinsics": np.eye(3),
    }
    d = tempfile.mkdtemp(prefix="cktime")
    res = {}

    def med(f, n=15):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            out = f()
            ts.append(time.perf_counter() - t0)
        return round(1e3 * float(np.median(ts)), 1), out

    step = [0]

    def save_plain():
        step[0] += 1
        return SAVE.run(FakeContext(), d, "t", step[0], placements=pl, pieces=pieces, ledger=ledger)

    def save_snap():
        step[0] += 1
        frame["rgb"][0, 0, 0] = step[0] % 255
        return SAVE.run(
            FakeContext(),
            d,
            "t",
            step[0],
            placements=pl,
            pieces=pieces,
            ledger=ledger,
            reference_snapshot=frame,
        )

    res["save_550_no_snapshot_ms"], out = med(save_plain)
    res["checkpoint_kB"] = round(out["bytes"] / 1e3, 1)
    res["save_550_new_snapshot_ms"], out = med(save_snap)
    res["save_550_carried_snapshot_ms"], out = med(save_plain)
    res["resume_550_verify_none_ms"], _ = med(
        lambda: RESUME.run(FakeContext(), d, "t", verify="none")
    )
    blobs = sum(p.stat().st_size for p in (Path(d) / "blobs").glob("*.npy.z"))
    res["blob_dir_MB_after_rotation"] = round(blobs / 1e6, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kills", type=int, default=200)
    ap.add_argument("--corrupt", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=100)
    a = ap.parse_args()
    parts = {"kill", "corrupt", "disturb", "build", "timing"} if a.part == "all" else {a.part}
    report = {}
    with ProcessPoolExecutor(a.workers) as ex:
        if "kill" in parts:
            rows = list(ex.map(kill_trial, range(a.kills)))
            report["kill"] = {
                "trials": len(rows),
                "resumed_ok": sum(r["ok"] for r in rows),
                "which": dict(Counter(r["which"] for r in rows)),
                "stale_temp_files_seen": sum(r["temps"] > 0 for r in rows),
                "skipped_files": sum(r["skipped"] for r in rows),
                "ledger": dict(Counter(r["led"] for r in rows)),
            }
            print(json.dumps(report["kill"]), flush=True)
        if "corrupt" in parts:
            rows = list(ex.map(corrupt_trial, range(a.corrupt)))
            report["corrupt"] = {
                m: dict(Counter(r["result"] for r in rows if r["mode"] == m))
                for m in ("flip", "cut")
            }
            print(json.dumps(report["corrupt"]), flush=True)
        if "disturb" in parts:
            jobs = [
                (kind, cam, case, s)
                for kind in ("jigsaw", "lego", "domino")
                for cam in ("overhead", "wrist")
                for case in ("none", "one_moved", "one_removed", "two_moved", "sweep")
                for s in range(a.first_seed, a.first_seed + a.seeds)
            ]
            rows = list(ex.map(disturb_trial, jobs, chunksize=2))
            table = {}
            for r in rows:
                key = f"{r['kind']}/{r['cam']}/{r['case']}"
                t = table.setdefault(
                    key, {"routes": Counter(), "found": 0, "moved": 0, "extra": [], "s": [], "n": 0}
                )
                t["routes"][r["route"]] += 1
                t["found"] += r["found"]
                t["moved"] += r["n_moved"]
                t["extra"].append(r["extra"])
                t["s"].append(r["s"])
                t["n"] += 1
            report["disturb"] = {
                k: {
                    "routes": dict(v["routes"]),
                    "found": f"{v['found']}/{v['moved']}",
                    "extra_mean": round(float(np.mean(v["extra"])), 2),
                    "extra_max": int(np.max(v["extra"])),
                    "resume_s_median": round(float(np.median(v["s"])), 2),
                }
                for k, v in sorted(table.items())
            }
            for k, v in report["disturb"].items():
                print(k, json.dumps(v), flush=True)
        if "build" in parts:
            jobs = [
                (case, s)
                for case in ("clean", "removed", "moved")
                for s in range(a.first_seed, a.first_seed + a.seeds)
            ]
            rows = [r for r in ex.map(build_trial, jobs) if not r.get("skip")]
            report["build"] = {
                c: {
                    "n": sum(1 for r in rows if r["case"] == c),
                    "hit": sum(r["hit"] for r in rows if r["case"] == c),
                    "routes": dict(Counter(r["route"] for r in rows if r["case"] == c)),
                    "first_reason": dict(Counter(r["reason"] for r in rows if r["case"] == c)),
                }
                for c in ("clean", "removed", "moved")
            }
            print(json.dumps(report["build"]), flush=True)
    if "timing" in parts:
        report["timing"] = timing()
        print(json.dumps(report["timing"]), flush=True)
    Path("/tmp/checkpoint_eval.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
