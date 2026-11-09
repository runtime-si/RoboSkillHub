# SPDX-License-Identifier: Apache-2.0
"""Summarise domino episode records (``record.json``) under one or more run roots.

    python examples/tasks/assembly_eval/domino_summarize.py RUNS_DIR [...] --out summary.json

Reports, per backend and label, episode statuses as numerator/denominator, placements
succeeded/attempted, failure types seen, and placement error/tilt ranges of the
succeeded placements. Nothing is inferred beyond what the records contain.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    groups: dict = defaultdict(
        lambda: {
            "episodes": 0,
            "status": defaultdict(int),
            "placements_ok": 0,
            "placements": 0,
            "failure_types": defaultdict(int),
            "claim_contradicted": 0,
            "pos_err_mm": [],
            "yaw_err_deg": [],
            "max_tilt_deg": [],
            "seeds": [],
            "sim_s": 0.0,
            "wall_s": 0.0,
            "physics_steps": 0,
            "unverified_reasons": defaultdict(int),
        }
    )
    meta = {}
    for root in args.roots:
        for rec_path in sorted(root.rglob("record.json")):
            r = json.loads(rec_path.read_text())
            be = r["backend"]["name"]
            g = groups[(be, r["scene_config_sha256"][:12], r["label"])]
            meta[be] = {
                k: v for k, v in r["backend"].items() if k not in ("usd_sha256", "mjcf_sha256")
            }
            g["episodes"] += 1
            g["seeds"].append(r["seed"])
            g["status"][r["status"]] += 1
            g["placements_ok"] += r.get("placements_succeeded", 0)
            g["placements"] += r.get("placements_attempted", 0)
            g["sim_s"] += r.get("sim_time_s", 0.0)
            g["wall_s"] += r.get("wall_s", 0.0)
            g["physics_steps"] += r.get("sim_physics_steps", 0)
            for p in r.get("placements", []):
                for ft in p.get("failure_types", []):
                    g["failure_types"][ft] += 1
                for u in p.get("unverified_reasons", []):
                    g["unverified_reasons"][u.split(":")[0]] += 1
                if p.get("executor_claim_contradicted"):
                    g["claim_contradicted"] += 1
                if p.get("status") == "succeeded":
                    g["pos_err_mm"].append(p["final"]["position_error_m"] * 1e3)
                    g["yaw_err_deg"].append(p["final"]["yaw_error_deg"])
                    g["max_tilt_deg"].append(p["max_tilt_deg"])
    out = {"backends": meta, "groups": []}
    for (be, scene, label), g in sorted(groups.items()):
        row = {
            "backend": be,
            "scene_sha256_12": scene,
            "label": label,
            "episodes": g["episodes"],
            "seeds": sorted(g["seeds"]),
            "status": {k: f"{v}/{g['episodes']}" for k, v in sorted(g["status"].items())},
            "placements_succeeded": f"{g['placements_ok']}/{g['placements']}",
            "failure_types": dict(g["failure_types"]),
            "unverified_reasons": dict(g["unverified_reasons"]),
            "executor_finished_but_not_succeeded": g["claim_contradicted"],
            "sim_time_s": round(g["sim_s"], 1),
            "wall_s": round(g["wall_s"], 1),
            "physics_steps": g["physics_steps"],
        }
        for key in ("pos_err_mm", "yaw_err_deg", "max_tilt_deg"):
            if g[key]:
                row[key + "_range"] = [round(min(g[key]), 4), round(max(g[key]), 4)]
        out["groups"].append(row)
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text)
    for row in out["groups"]:
        print(
            f"{row['backend']:7s} {row['label']:28s} {row['status']} placements={row['placements_succeeded']} "
            f"failures={row['failure_types']} unverified={row['unverified_reasons']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
