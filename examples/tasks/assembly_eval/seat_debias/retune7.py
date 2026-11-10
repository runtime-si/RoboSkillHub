"""Retune the depth seat's height / tilt tolerances (stud-mask estimator, fitted noise) on seeds 0-2, report
depth seat and the #162 fused judge on seeds 3-5."""

import glob
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("f", sys.argv[1])
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
rows = []
for f in glob.glob(sys.argv[2] + "/*__r7_*/**/episode.json", recursive=True):
    e = json.loads(Path(f).read_text())
    ind = e["independent"]
    name = Path(f).relative_to(sys.argv[2]).parts[0]
    steps = {s.get("node"): s for s in e["steps"]}
    if not ind["placements"]:
        continue
    p = ind["placements"][0]
    d = (steps.get("depth") or {}).get("result") or {}
    hub = (d.get("evidence") or {}).get("hub_outputs") or {}
    rel = (steps.get("release") or {}).get("result") or {}
    relm = ((rel.get("evidence") or {}).get("postcondition") or {}).get("measured") or {}
    pull = (steps.get("pullup") or {}).get("result") or {}
    rows.append(
        {
            "name": name,
            "seed": e["seed"],
            "truth": p["status"] == "succeeded",
            "route": hub.get("route"),
            "h": hub.get("height_error"),
            "tilt": hub.get("tilt_deg"),
            "release": "stalled_jammed"
            if rel.get("status") == "failed" or relm.get("holding") is True
            else ("reached" if rel else ""),
            "rise": (pull.get("evidence") or {}).get("brick_rise_m"),
            "true_tilt_deg": __import__("math").degrees(
                abs(((p.get("measured") or {}).get("settled_error") or {}).get("tilt_rad") or 0.0)
            ),
            "true_dz_mm": 1e3
            * (((p.get("measured") or {}).get("settled_error") or {}).get("dz_m") or 0.0),
            "scene": name.split("__")[0],
        }
    )
TR = [r for r in rows if r["seed"] < 3]
TE = [r for r in rows if r["seed"] >= 3]
GEOM_FAIL = {"no_brick", "shifted_off_grid"}


def seat(r, H, T):
    if r["route"] in (None, "unverified", "unreachable") or r["h"] is None:
        return None
    if r["route"] in GEOM_FAIL:
        return False
    return abs(r["h"]) <= H and r["tilt"] <= T


def fused(r, H, T):
    s = seat(r, H, T)
    route = "" if s is None else ("seated" if s else "not_seated")
    return F.run(
        None,
        "place",
        release_grip_route=r["release"],
        seat_route=route,
        height_error_m=r["h"] if s else None,
        pull_route="not_connected" if r["rise"] is not None else "",
        brick_rise_m=r["rise"],
        max_height_error_m=10.0,
    )["route"]


def score(rows, j):
    fd = sum(1 for r in rows if (j(r) in ("done", True)) and not r["truth"])
    P = sum(r["truth"] for r in rows)
    acc = sum(1 for r in rows if (j(r) in ("done", True)) and r["truth"])
    unc = sum(1 for r in rows if j(r) in ("uncertain", None))
    cau = sum(1 for r in rows if (j(r) in ("failed", False)) and not r["truth"])
    return {
        "false_done": f"{fd}/{len(rows) - P}",
        "accepted": f"{acc}/{P}",
        "uncertain": unc,
        "caught": f"{cau}/{len(rows) - P}",
        "fd": fd,
        "acc": acc,
    }


best = None
for H, T in itertools.product(
    [0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014], [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
):
    s = score(TR, lambda r, H=H, T=T: seat(r, H, T))
    key = (s["fd"] == 0, s["acc"], -H, -T)
    if best is None or key > best[0]:
        best = (key, H, T, s)
_, H, T, s_tr = best
print(f"train seeds 0-2 ({len(TR)}): chose height tol {H * 1e3:.1f} mm, tilt tol {T} deg -> {s_tr}")
print(f"held-out seeds 3-5 ({len(TE)}):")
print(
    "  depth seat, default tolerances (0.6 mm, 1.5 deg):", score(TE, lambda r: seat(r, 0.0006, 1.5))
)
print("  depth seat, retuned:", score(TE, lambda r: seat(r, H, T)))
print("  fused #162 with the retuned seat:", score(TE, lambda r: fused(r, H, T)))
print(
    "seated heights mm:",
    sorted(round(r["h"] * 1e3, 2) for r in rows if r["truth"] and r["h"] is not None),
)
print(
    "failed heights mm:",
    sorted(round(r["h"] * 1e3, 2) for r in rows if not r["truth"] and r["h"] is not None),
)
print(
    "tilts deg (seated):",
    sorted(round(r["tilt"], 1) for r in rows if r["truth"] and r["tilt"] is not None),
)

print("seated: estimator tilt vs BrickSim tilt (deg), height vs true dz (mm), by scene:")
print(
    Counter(
        (
            r["scene"],
            round(r["tilt"], 1),
            round(r["true_tilt_deg"], 2),
            round(r["h"] * 1e3, 2),
            round(r["true_dz_mm"], 2),
        )
        for r in rows
        if r["truth"] and r["h"] is not None
    )
)
print(
    "failed:",
    Counter(
        (
            r["scene"],
            round(r["tilt"], 1),
            round(r["true_tilt_deg"], 2),
            round(r["h"] * 1e3, 2),
            round(r["true_dz_mm"], 2),
        )
        for r in rows
        if not r["truth"] and r["h"] is not None
    ),
)
