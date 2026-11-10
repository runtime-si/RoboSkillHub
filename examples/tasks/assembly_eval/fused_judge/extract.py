"""Per-placement evidence and truth from lego_graph_hub episodes (runs/<scene>__<fault>/.../episode.json)."""

import glob
import json
import sys
from pathlib import Path


def res(step):
    r = step.get("result") or {}
    return r, (r.get("evidence") or {}), (r.get("outputs") or {})


rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*/**/episode.json", recursive=True)):
    e = json.loads(Path(f).read_text())
    ind = e.get("independent") or json.loads((Path(f).parent / "independent.json").read_text())
    name = Path(f).relative_to(sys.argv[1]).parts[0]
    scene, fault, *grip = name.split("__")
    fault = fault + (f"+{grip[0]}" if grip else "")
    groups, cur = [], None
    for s in e.get("steps", []):
        if s.get("node") == "select":
            cur = {}
            groups.append(cur)
            continue
        if cur is not None and s.get("node") not in cur:
            cur[s.get("node")] = s
    groups = [g for g in groups if "pick" in g or "observe_piece" in g]
    for k, p in enumerate(ind["placements"]):
        g = groups[k] if k < len(groups) else {}
        post = {c["name"]: c["passed"] for c in p["postconditions"]}
        row = {
            "episode": f"{name}/seed{e['seed']}",
            "seed": e["seed"],
            "scene": scene,
            "fault": fault,
            "k": k,
            "truth_pick": bool(post.get("grasped") and post.get("held_until_place")),
            "truth_place": p["status"] == "succeeded",
            "failure_type": p["failure_type"],
        }
        pick = g.get("pick")
        conf = g.get("confirm")
        if pick:
            r, ev, o = res(pick)
            row["pick_status"], row["pick_held"] = r.get("status"), o.get("held")
        if conf:
            r, ev, o = res(conf)
            row["confirm_status"], row["confirm_held"] = r.get("status"), o.get("held")
            row["confirm_measured"] = (ev.get("postcondition") or {}).get("measured")
        if g.get("seat"):
            r, ev, o = res(g["seat"])
            row["press_verified"] = o.get("connection_verified")
            row["press_status"] = r.get("status")
        if g.get("release"):
            r, ev, o = res(g["release"])
            m = (ev.get("postcondition") or {}).get("measured") or {}
            row["release_status"], row["release_holding"] = r.get("status"), m.get("holding")
        if g.get("pullup"):
            r, ev, o = res(g["pullup"])
            row["pull_route"], row["brick_rise_m"], row["pull_status"] = (
                ev.get("hub_route") or o.get("route"),
                ev.get("brick_rise_m"),
                r.get("status"),
            )
        if g.get("verify"):
            r, ev, o = res(g["verify"])
            row["verify_verdict"], row["verify_status"] = o.get("verdict"), r.get("status")
        if g.get("depth"):
            r, ev, o = res(g["depth"])
            row["seat_route"], row["height_error_m"], row["depth_status"] = (
                ev.get("hub_route") or o.get("route"),
                o.get("height_error_m"),
                r.get("status"),
            )
        rows.append(row)
Path(sys.argv[2]).write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
print(len(rows), "placements from", len({r["episode"] for r in rows}), "episodes")
