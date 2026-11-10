"""Fit the fused judge's thresholds on training seeds, report it and each single check on held-out seeds."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("f", sys.argv[1])
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
TRAIN = {0, 1, 2}


def fused(r, h, rise, step):
    if step == "pick":
        m = (r.get("confirm_measured") or {}).get("gripper") or {}
        grip = "stalled_on_object" if m.get("holding") else ("reached" if m else "")
        return F.run(None, "pick", grip_route=grip)["route"]
    rel = (
        "stalled_jammed"
        if (r.get("release_status") == "failed" or r.get("release_holding") is True)
        else ("reached" if r.get("release_status") else "")
    )
    return F.run(
        None,
        "place",
        release_grip_route=rel,
        seat_route=r.get("seat_route") or "",
        height_error_m=r.get("height_error_m"),
        pull_route="not_connected" if r.get("brick_rise_m") is not None else "",
        brick_rise_m=r.get("brick_rise_m"),
        max_height_error_m=h,
        max_pull_rise_m=rise,
    )["route"]


def singles(r):
    out = {}
    out["depth seat (hub verify_seat_depth route)"] = (
        None if not r.get("seat_route") else r["seat_route"] == "seated"
    )
    out["pull-up route (hub pull_up_check)"] = (
        None if not r.get("pull_route") else r["pull_route"] == "connected"
    )
    out["task verify (lego.verify_connection)"] = (
        None if not r.get("verify_verdict") else r["verify_verdict"] == "satisfied"
    )
    out["press connection_verified (guarded_seat)"] = r.get("press_verified")
    out["release jaws opened (gripper at release)"] = (
        None if not r.get("release_status") else (r.get("release_status") == "succeeded")
    )
    return out


def score(pred, truth):
    """pred: done / failed / uncertain (or True/False/None)."""
    n = len(truth)
    P = sum(truth)
    N = n - P
    done = [p == "done" or p is True for p in pred]
    fd = sum(1 for d, t in zip(done, truth, strict=True) if d and not t)
    acc_succ = sum(1 for d, t in zip(done, truth, strict=True) if d and t)
    unc = sum(1 for p in pred if p in ("uncertain", None))
    caught = sum(
        1 for p, t in zip(pred, truth, strict=True) if (p == "failed" or p is False) and not t
    )
    return {
        "n": n,
        "succ": P,
        "fail": N,
        "false_done": fd,
        "false_done_rate": round(fd / N, 3) if N else None,
        "accepted_succ": acc_succ,
        "accept_rate": round(acc_succ / P, 3) if P else None,
        "uncertain": unc,
        "failed_caught": caught,
    }


place = [r for r in rows if r["truth_pick"]]
tr = [r for r in place if r["seed"] in TRAIN]
te = [r for r in place if r["seed"] not in TRAIN]
best = None
for h, rise in itertools.product(
    [0.0003, 0.0005, 0.0008, 0.001, 0.0015], [0.0006, 0.0008, 0.001, 0.0015, 0.002, 0.003]
):
    s = score([fused(r, h, rise, "place") for r in tr], [r["truth_place"] for r in tr])
    if (s["false_done_rate"] or 0) <= 0.01:
        key = (s["accepted_succ"], -h, -rise)
        if best is None or key > best[0]:
            best = (key, h, rise, s)
_, H, RISE, s_tr = best
print(
    f"chosen on train seeds {sorted(TRAIN)}: max_height_error {H * 1e3:.1f} mm, max_pull_rise {RISE * 1e3:.1f} mm; train {s_tr}"
)
res = {"chosen": {"max_height_error_m": H, "max_pull_rise_m": RISE}, "train": s_tr}
for name, sub in (
    ("held-out", te),
    ("held-out, no depth result", [r for r in te if not r.get("seat_route")]),
):
    if not sub:
        continue
    truth = [r["truth_place"] for r in sub]
    print(f"\n## place, {name}: {len(sub)} placements, {sum(truth)} succeeded")
    print(
        "| judge | false done | accepted successes | uncertain | failures caught |\n|---|---|---|---|---|"
    )
    s = score([fused(r, H, RISE, "place") for r in sub], truth)
    res[name] = {"fused": s}
    print(
        f"| **fused** | {s['false_done']}/{s['fail']} | {s['accepted_succ']}/{s['succ']} | {s['uncertain']} | {s['failed_caught']}/{s['fail']} |"
    )
    for k in singles(sub[0]):
        s = score([singles(r)[k] for r in sub], truth)
        res[name][k] = s
        print(
            f"| {k} | {s['false_done']}/{s['fail']} | {s['accepted_succ']}/{s['succ']} | {s['uncertain']} | {s['failed_caught']}/{s['fail']} |"
        )
    s = score(
        [(r["brick_rise_m"] <= RISE) if r.get("brick_rise_m") is not None else None for r in sub],
        truth,
    )
    res[name]["pull rise alone"] = s
    print(
        f"| brick rise <= {RISE * 1e3:.1f} mm alone | {s['false_done']}/{s['fail']} | {s['accepted_succ']}/{s['succ']} | {s['uncertain']} | {s['failed_caught']}/{s['fail']} |"
    )
pick = rows
truth = [r["truth_pick"] for r in pick]
s = score([fused(r, H, RISE, "pick") for r in pick], truth)
res["pick"] = s
print(f"\n## pick, all seeds: {len(pick)} grasps, {sum(truth)} held: fused {s}")
Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
