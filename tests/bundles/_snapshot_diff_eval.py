"""detecting-disturbed-pieces-by-snapshot-diff on synthetic boards: detection rate against
shift size and rotation, and false alarms. Not collected by pytest; run from the repo root:

    .venv/bin/python tests/bundles/_snapshot_diff_eval.py --trials 20 --first-seed 1000

For each family (jigsaw pieces, bricks on a plate, standing dominoes) and camera (overhead
Orbbec at 1 m, wrist D405 at 25 cm) a board is rendered as the reference; every current frame
of the trial gets its own lighting drift (gain 0.85-1.15, tint +-4 %, gradient +-15 %, a soft
shadow half the time) and camera bump (0-0.3 deg, 0-2 mm), so the no-change frame is a pure
nuisance pair. One random item is then shifted 1-10 mm in a random direction, turned 2-15
degrees, removed or (dominoes) toppled; an object is added; a hand reaches over the board; the
robot's arm hovers over it (masked from its ee pose). Items are given as reference masks.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _snapshot_scenes as S  # noqa: E402

SCRIPT = (
    HERE.parents[1] / "skills/detecting-disturbed-pieces-by-snapshot-diff/scripts/diff_snapshots.py"
)
SHIFTS_MM = (1, 2, 3, 5, 7, 10)
REGION = {
    "center": {"x": 0.0, "y": 0.0, "z": 0.03},
    "extent": {"x": 0.11, "y": 0.11, "z": 0.03},
    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}
TURNS_DEG = (2, 5, 10, 15)


def _skill():
    spec = importlib.util.spec_from_file_location("diff_snapshots", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Ctx:
    def tool(self, name, **kw):
        raise AssertionError(name)


def _nuisance(rng, t_cam):
    lg = S.Lighting(
        gain=float(rng.uniform(0.85, 1.15)),
        tint=tuple(float(v) for v in rng.uniform(0.96, 1.04, 3)),
        gradient=tuple(float(v) for v in rng.uniform(-0.15, 0.15, 2)),
        shadow=(
            float(rng.uniform(100, 540)),
            float(rng.uniform(80, 400)),
            float(rng.uniform(60, 160)),
            float(rng.uniform(0.1, 0.3)),
        )
        if rng.random() < 0.5
        else None,
    )
    t = S.bump(t_cam, rng, rot_deg=float(rng.uniform(0, 0.3)), trans_m=float(rng.uniform(0, 0.002)))
    return lg, t


def trial(args):
    kind, cam, seed = args
    d = _skill()
    rng = np.random.default_rng(seed)
    sc = S.build(kind, seed, cam)
    t_cam, k = S.camera_pose(cam)
    ref, ids = S.render(sc, t_cam, k, seed=seed)
    masks = S.item_masks(sc, ids)
    items = [{"id": i, "mask": m} for i, m in masks.items() if m.sum() >= 4]
    names = [it["id"] for it in items]
    target = names[int(rng.integers(len(names)))]
    rows = []

    def check(label, scene, value=None, arms=False, **kw):
        lg, t = _nuisance(rng, t_cam)
        cur, _ = S.render(scene, t, k, lighting=lg, seed=int(rng.integers(1 << 30)))
        # The bump is not known to the skill: a fixed camera reports its calibrated pose, a
        # wrist camera the arm's forward kinematics (off by up to 1 mm and 0.15 deg).
        cur["pose"] = S.se3(
            t_cam if cam == "overhead" else S.bump(t, rng, rot_deg=0.15, trans_m=0.001)
        )
        current = S.observation(cur, scene) if arms else cur
        out = d.run(
            _Ctx(),
            ref,
            items,
            current=current,
            region=REGION,
            arm_base_positions=scene.arm_bases if arms else None,
            **kw,
        )
        by = {r["id"]: r for r in out["items"]}
        tr = by.get(target, {})
        others = [
            r["id"] for r in out["items"] if r["id"] != target and r["status"] in ("moved", "lost")
        ]
        rows.append(
            {
                "kind": kind,
                "cam": cam,
                "label": label,
                "value": value,
                "route": out["route"],
                "target_status": tr.get("status"),
                "shift_mm": tr.get("shift_mm"),
                "rotation_deg": tr.get("rotation_deg"),
                "false_items": len(others),
                "n_unexpected": out["n_unexpected"],
                "occluders": len(out["occluders"]),
                "rho": out["registration"]["rho"],
                "noise_mm": [
                    r["shift_mm"]
                    for r in out["items"]
                    if r["id"] != target and r.get("shift_mm") is not None
                ],
                "noise_px": [
                    math.hypot(*r["translation_px"])
                    for r in out["items"]
                    if r["id"] != target and r.get("translation_px")
                ],
            }
        )

    check("none", sc)
    for mm in SHIFTS_MM:
        a = rng.uniform(0, 2 * math.pi)
        check("shift", S.shift(sc, target, mm / 1000 * math.cos(a), mm / 1000 * math.sin(a)), mm)
    for deg in TURNS_DEG:
        check("turn", S.shift(sc, target, 0, 0, deg * (1 if rng.random() < 0.5 else -1)), deg)
    if kind == "domino":
        check("topple", S.topple(sc, target, float(rng.uniform(0, 360))))
    check("remove", S.remove(sc, target))
    # An object set down in the work area, at least 12 mm clear of every item.
    for _ in range(200):
        xy = rng.uniform(-0.09, 0.09, 2)
        if all(np.abs(xy - b.center[:2]).max() > b.half[:2].max() + 0.0195 for b in sc.items()):
            break
    check("add_object", S.add_object(sc, (float(xy[0]), float(xy[1]))))
    tb = sc.get(target)
    check(
        "hand",
        S.add_hand(
            sc,
            (float(tb.center[0]), float(tb.center[1])),
            height=float(rng.uniform(0.08, 0.15)),
            direction_deg=float(rng.choice([-90, 0, 180])),
        ),
    )
    if cam == "overhead":  # a wrist camera rides on the arm
        check(
            "arm",
            S.add_arm(sc, (float(tb.center[0]), float(tb.center[1]), 0.09), (0.0, -0.45, 0.2)),
            arms=True,
        )
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["cam"])].append(r)
    for (kind, cam), rs in sorted(groups.items()):
        print(f"\n## {kind} / {cam}")
        none = [r for r in rs if r["label"] == "none"]
        fa = sum(r["route"] != "unchanged" for r in none)
        print(
            f"no-change pairs: {len(none)}, false alarms {fa} ({', '.join(sorted({r['route'] for r in none if r['route'] != 'unchanged'}))})"
        )
        nm = [v for r in rs if r["label"] in ("none", "shift", "turn") for v in r["noise_mm"]]
        npx = [v for r in rs if r["label"] in ("none", "shift", "turn") for v in r["noise_px"]]
        if nm:
            print(
                f"unmoved items' read shift: median {np.median(nm):.2f} mm / {np.median(npx):.2f} px, 99th pct {np.percentile(nm, 99):.2f} mm / {np.percentile(npx, 99):.2f} px (n={len(nm)})"
            )
        bad = [r["rho"] for r in rs if r["route"] == "unregistrable"]
        if bad:
            print(f"unregistrable: {len(bad)} pairs, background rho {sorted(bad)}")
        for label, values in (("shift", SHIFTS_MM), ("turn", TURNS_DEG)):
            parts = []
            for v in values:
                sel = [r for r in rs if r["label"] == label and r["value"] == v]
                hit = sum(
                    r["target_status"] in ("moved", "lost") and r["route"] == "disturbed"
                    for r in sel
                )
                err = [
                    abs((r["shift_mm"] or 0) - v)
                    for r in sel
                    if label == "shift" and r["target_status"] == "moved"
                ]
                rot = [
                    abs(abs(r["rotation_deg"] or 0) - v)
                    for r in sel
                    if label == "turn" and r["target_status"] == "moved"
                ]
                e = (
                    f" err {np.median(err):.2f}mm"
                    if err
                    else (f" err {np.median(rot):.1f}deg" if rot else "")
                )
                parts.append(f"{v}: {hit}/{len(sel)}{e}")
            fi = sum(r["false_items"] for r in rs if r["label"] == label)
            print(f"{label}: " + "; ".join(parts) + f"  (other items flagged: {fi})")
        for label in ("topple", "remove", "add_object", "hand", "arm"):
            sel = [r for r in rs if r["label"] == label]
            if not sel:
                continue
            routes = defaultdict(int)
            for r in sel:
                routes[r["route"]] += 1
            extra = ""
            if label in ("topple", "remove"):
                extra = f", target lost/moved {sum(r['target_status'] in ('lost', 'moved') for r in sel)}"
            if label == "hand":
                extra = f", occluder found {sum(r['occluders'] > 0 for r in sel)}, items flagged {sum(r['false_items'] for r in sel)}"
            if label == "arm":
                extra = f", items flagged {sum(r['false_items'] for r in sel)}"
            print(f"{label}: {dict(routes)}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--kinds", default="jigsaw,lego,domino")
    ap.add_argument("--cams", default="overhead,wrist")
    a = ap.parse_args()
    jobs = [
        (kind, cam, a.first_seed + i)
        for kind in a.kinds.split(",")
        for cam in a.cams.split(",")
        for i in range(a.trials)
    ]
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.imap_unordered(trial, jobs) for r in rs]
    summarize(rows)


if __name__ == "__main__":
    main()
