"""Depth seat and the fused judge on ray-cast depth (round 3 units) vs Isaac RTX depth (rendered units), same
scenes, faults and seeds; thresholds fixed at the fused judge's defaults (chosen on round 3's training seeds)."""

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("f", sys.argv[1])
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
ray = [json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()]
ren = [json.loads(x) for x in Path(sys.argv[3]).read_text().splitlines() if x.strip()]
ren = [dict(r, fault=r["fault"].replace("+render", "")) for r in ren]
key = lambda r: (r["scene"], r["fault"], r["seed"], r["k"])  # noqa: E731
have = {key(r) for r in ren}
ray = [r for r in ray if key(r) in have]


def fused(r):
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
    )["route"]


def depth(r):
    return (
        None
        if not r.get("seat_route") or r.get("depth_status") == "unverified"
        else r["seat_route"] == "seated"
    )


def score(rows, judge):
    fd = acc = unc = caught = 0
    for r in rows:
        p = judge(r)
        done = p == "done" or p is True
        if done and not r["truth_place"]:
            fd += 1
        if done and r["truth_place"]:
            acc += 1
        if p in ("uncertain", None):
            unc += 1
        if (p == "failed" or p is False) and not r["truth_place"]:
            caught += 1
    P = sum(r["truth_place"] for r in rows)
    return f"{fd}/{len(rows) - P} | {acc}/{P} | {unc} | {caught}/{len(rows) - P}"


print(f"{len(ren)} rendered placements matched with {len(ray)} ray-cast ones")
print(
    "| depth source | judge | false done | successes accepted | uncertain | failures caught |\n|---|---|---|---|---|---|"
)
for name, rows in (("ray-cast (round 3)", ray), ("Isaac RTX render", ren)):
    print(f"| {name} | depth seat alone | {score(rows, depth)} |")
    print(f"| {name} | fused (#162) | {score(rows, fused)} |")
hs = [round(r["height_error_m"] * 1e3, 2) for r in ren if r.get("height_error_m") is not None]
print(
    "rendered height errors (mm), succeeded:",
    sorted(
        round(r["height_error_m"] * 1e3, 2)
        for r in ren
        if r.get("height_error_m") is not None and r["truth_place"]
    ),
)
print(
    "rendered height errors (mm), failed:",
    sorted(
        round(r["height_error_m"] * 1e3, 2)
        for r in ren
        if r.get("height_error_m") is not None and not r["truth_place"]
    ),
)
print(
    "ray-cast height errors (mm), succeeded:",
    sorted(
        round(r["height_error_m"] * 1e3, 2)
        for r in ray
        if r.get("height_error_m") is not None and r["truth_place"]
    ),
)
print(
    "ray-cast height errors (mm), failed:",
    sorted(
        round(r["height_error_m"] * 1e3, 2)
        for r in ray
        if r.get("height_error_m") is not None and not r["truth_place"]
    ),
)
