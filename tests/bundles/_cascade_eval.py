"""Sweep of verifying-a-domino-cascade-from-video over synthetic clips (CPU).

    .venv/bin/python -m tests.bundles._cascade_eval [--seeds 3]

Prints per condition: route accuracy, per-tile fell/standing accuracy (unknown
counted wrong), stop-index error, onset error against the true start of the fall
and against the frame the true angle leaves the upright band, and the wave speed
error. The numbers in the skill's SKILL.md come from this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

from . import _cascade_render as R

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills/verifying-a-domino-cascade-from-video/scripts/verify_cascade.py"


def load():
    spec = importlib.util.spec_from_file_location("verify_cascade", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hands_for(kind: str, sc: R.Scene, slots):
    p0 = slots[0]["position"]
    if kind == "none":
        return []
    if kind == "pusher_over":
        # Mechanism A: the hand hovers over the first tiles, then pulls back just
        # before the push.
        c = (p0["x"] + 0.01, p0["y"], 0.075)
        back = (p0["x"] - 0.12, p0["y"], 0.075)
        return [R.Hand([(0.0, c), (sc.t_trigger - 0.04, c), (sc.t_trigger + 0.02, back)])]
    if kind == "mid_pass":
        # The arm sweeps over the line mid-cascade, 70-110 mm up.
        k = min(4, sc.n - 1)
        pk = slots[k]["position"]
        t0 = sc.t_trigger + 2 * sc.pitch / sc.speed
        return [
            R.Hand(
                [
                    (t0, (pk["x"], pk["y"] - 0.15, 0.09)),
                    (t0 + 0.12, (pk["x"], pk["y"] + 0.15, 0.09)),
                ],
                size=(0.06, 0.08, 0.04),
            )
        ]
    raise ValueError(kind)


def truth_up_frame(clip, i, up):
    a = clip["angles"][i]
    k = np.flatnonzero(a >= up)
    return int(k[0]) if len(k) else None


def evaluate(mod, sc: R.Scene, hand: str, signal: str):
    sc.hands = hands_for(hand, sc, R.slots_for(sc))
    clip = R.render(sc)
    out = mod.run(FakeContext({}), slots=clip["slots"], frames=clip["frames"], signal=signal)
    fell = clip["fell"]
    if not fell.any():
        want = "not_triggered"
    elif fell.all():
        want = "all_fell"
    else:
        want = "stopped_at"
    states = [p["state"] for p in out["tiles"]]
    tile_ok = [
        (s == "fell") == bool(f) and s in ("fell", "standing")
        for s, f in zip(states, fell, strict=True)
    ]
    est_stop = (
        out["stop_index"]
        if out["route"] == "stopped_at"
        else (-1 if out["route"] == "all_fell" else None)
    )
    stop_err = (
        None if est_stop is None or want == "not_triggered" else abs(est_stop - clip["stop_index"])
    )
    onset_err, frame_err = [], []
    for p in out["tiles"]:
        i = p["index"]
        if p["state"] == "fell" and not p["inferred"] and i < len(fell) and fell[i]:
            onset_err.append(p["onset_time"] - clip["onsets"][i])
            tf = truth_up_frame(clip, i, p["upright_deg"])
            if tf is not None:
                frame_err.append(p["onset_frame"] - tf)
    speed_err = None
    if out["wave_speed"] and fell.sum() >= 5:
        speed_err = out["wave_speed"] / sc.speed - 1.0
    return {
        "route_ok": out["route"] == want,
        "route": out["route"],
        "want": want,
        "tile_ok": tile_ok,
        "stop_err": stop_err,
        "onset_err": onset_err,
        "frame_err": frame_err,
        "speed_err": speed_err,
    }


def summarise(label, rows):
    n = len(rows)
    route = sum(r["route_ok"] for r in rows)
    tiles = [x for r in rows for x in r["tile_ok"]]
    stops = [r["stop_err"] for r in rows if r["stop_err"] is not None]
    on = np.array([x for r in rows for x in r["onset_err"]]) * 1e3
    fr = np.array([x for r in rows for x in r["frame_err"]])
    sp = np.array([r["speed_err"] for r in rows if r["speed_err"] is not None]) * 100
    wrong = sorted({f"{r['want']}->{r['route']}" for r in rows if not r["route_ok"]})
    print(
        f"{label:44s} route {route}/{n}  tiles {sum(tiles)}/{len(tiles)}  "
        f"stop err {('max ' + str(max(stops))) if stops else '-'} ({len(stops)} judged)  "
        f"onset {np.median(on) if len(on) else math.nan:+.0f} ms (p90 |dev| "
        f"{np.percentile(np.abs(on - np.median(on)), 90) if len(on) else math.nan:.0f})  "
        f"frame err {np.median(fr) if len(fr) else math.nan:+.0f} [{fr.min() if len(fr) else 0:+d},{fr.max() if len(fr) else 0:+d}]  "
        f"speed {np.median(np.abs(sp)) if len(sp) else math.nan:.1f}% (max {np.abs(sp).max() if len(sp) else math.nan:.1f})"
        + (f"  wrong: {wrong}" if wrong else "")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--extra", action="store_true", help="pitch / placement-error / speed sweep instead"
    )
    ap.add_argument(
        "--station-only", action="store_true", help="with --extra: only the station camera"
    )
    args = ap.parse_args()
    mod = load()
    outcomes = [("all", None, False), ("stop", "rand", False), ("none", None, True)]
    if args.extra:
        cases = [
            ("pitch 24 mm", {"pitch": 0.024, "speed": 0.95}),
            ("pitch 48 mm", {"pitch": 0.048, "speed": 0.68}),
            ("placement error 4 mm, yaw 4 deg", {"place_err": 0.004, "yaw_err_deg": 4.0}),
            ("depth noise 4 mm, 5 % holes", {"depth_noise": 0.004, "holes": 0.05}),
            ("arc r = 0.25 m", {"layout": "arc"}),
            ("20 tiles", {"n": 20}),
            # The real overhead camera's pose and the depth noise / holes measured on
            # its frame cable_frame_20261103-1421 (3.5 mm median local residual at
            # ~1 m, 11 % holes); the line where the left arm reaches.
            (
                "station camera, 3.5 mm noise, 11 % holes",
                {
                    "camera": "station",
                    "origin": (0.15, 0.40),
                    "depth_noise": 0.0035 * 0.81,
                    "holes": 0.11,
                },
            ),
        ]
        if args.station_only:
            cases = cases[-1:]
        for signal in ("depth", "rgb"):
            for label, kw in cases:
                rows = []
                for (_, stop, none), seed in itertools.product(outcomes, range(args.seeds)):
                    rng = np.random.default_rng(1000 + seed)
                    n = kw.get("n", 10)
                    sc = R.Scene(
                        fps=60.0,
                        seed=seed,
                        stop_at=int(rng.integers(2, n - 1)) if stop else None,
                        no_trigger=none,
                        **kw,
                    )
                    rows.append(evaluate(mod, sc, "none", signal))
                summarise(f"{signal}/overhead 60 fps {label}", rows)
        return
    for signal, camera in (("depth", "overhead"), ("depth", "wrist"), ("rgb", "overhead")):
        for fps in (30.0, 60.0, 120.0):
            for hand in ("none", "pusher_over", "mid_pass"):
                rows = []
                for (_, stop, none), seed in itertools.product(outcomes, range(args.seeds)):
                    rng = np.random.default_rng(1000 + seed)
                    sc = R.Scene(
                        n=10,
                        fps=fps,
                        camera=camera,
                        seed=seed,
                        stop_at=int(rng.integers(2, 9)) if stop else None,
                        no_trigger=none,
                    )
                    rows.append(evaluate(mod, sc, hand, signal))
                summarise(f"{signal}/{camera} {fps:.0f} fps hand={hand}", rows)


if __name__ == "__main__":
    main()
