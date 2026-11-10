"""Compare the success judges on the LEGO placement units (manifest.json + *_results.jsonl)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
units = {u["id"]: u for u in json.loads((H / "manifest.json").read_text())}


def jl(name):
    rows = []
    for p in [H / name, *sorted(H.glob(name.replace(".jsonl", "_s*.jsonl")))]:
        if p.exists():
            rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def rate(pred, ids):
    """(accuracy, success recall, failure recall, n, false 'done' count)."""
    tp = sum(1 for i in ids if pred[i] and units[i]["success"])
    tn = sum(1 for i in ids if not pred[i] and not units[i]["success"])
    p = sum(1 for i in ids if units[i]["success"])
    n = len(ids) - p
    fp = n - tn
    return (
        (tp + tn) / len(ids),
        tp / p if p else float("nan"),
        tn / n if n else float("nan"),
        len(ids),
        fp,
    )


def subset(name):
    ids = sorted(units)
    if name == "all":
        return ids
    return [i for i in ids if i.startswith(name + "/")]


def row(label, pred, ids, score=None):
    acc, sr, fr, n, fp = rate(pred, ids)
    au = ""
    if score is not None:
        a = auroc(
            [score[i] for i in ids if units[i]["success"]],
            [score[i] for i in ids if not units[i]["success"]],
        )
        au = f"{a:.2f}"
    return f"| {label} | {n} | {acc:.2f} | {sr:.2f} | {fr:.2f} | {fp} | {au} |"


judges = {}
scores = {}
rv = defaultdict(dict)
for r in jl("rynn_results.jsonl"):
    rv[(r["model"], r["clip"])][r["id"]] = r
for (m, clip), d in sorted(rv.items()):
    if set(d) < set(units):
        continue
    scores[f"RynnValue-{m} {clip}: -remaining_s"] = {i: -d[i]["final_remaining_s"] for i in units}
    judges[f"RynnValue-{m} {clip}: Success=Yes"] = {i: bool(d[i]["success_flag"]) for i in units}
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    judges[f"RynnValue-{m} {clip}: remaining <= {th:g} s"] = {
        i: d[i]["final_remaining_s"] <= th for i in units
    }
dr = {r["id"]: r for r in jl("dr_results.jsonl")}
if set(dr) >= set(units):
    scores["DenseReward last frame"] = {i: dr[i]["last"]["progress"] or 0.0 for i in units}
    judges["DenseReward last frame >= 0.85 (#124 done)"] = {
        i: (dr[i]["last"]["progress"] or 0.0) >= 0.85 for i in units
    }
for name, f in (("flash", "gemini_flash.jsonl"), ("er2", "gemini_er2.jsonl")):
    g = {r["id"]: r for r in jl(f)}
    if set(g) >= set(units):
        judges[f"detecting-subtask-success ({name})"] = {
            i: g[i]["detect_route"] == "succeeded" for i in units
        }
        judges[f"judging-step-with-before-now-vlm ({name})"] = {
            i: g[i]["judge_route"] == "complete" for i in units
        }

for sub in ("all", "p0", "p1", "p2", "v1", "v2"):
    ids = subset(sub)
    if sub in ("p0", "p1", "p2"):
        continue
    if sub == "all":
        ids_main = [i for i in ids if i.split("/")[0] in ("p0", "p1", "p2")]
        title, ids = "default view, gripper force 0.1 (main set)", ids_main
    elif sub == "v1":
        title = "same episodes, opposite camera"
    else:
        title = "gripper force 1.0 (more grasp failures)"
    p = sum(units[i]["success"] for i in ids)
    print(f"\n### {title}: {len(ids)} units, {p} succeeded\n")
    print("| judge | n | accuracy | success recall | failure recall | false done | AUROC |")
    print("|---|---|---|---|---|---|---|")
    for k, pred in judges.items():
        sk = None
        if k.startswith("RynnValue"):
            sk = scores[k.split(":")[0] + ": -remaining_s"]
        elif k.startswith("DenseReward"):
            sk = scores["DenseReward last frame"]
        print(row(k, pred, ids, sk))

print("\n### failures by type (all 75 units): judges that called them done\n")
by = defaultdict(list)
for i, u in units.items():
    if not u["success"]:
        by[u["failure_type"]].append(i)
print("| failure | n | " + " | ".join(judges) + " |")
print("|---|---|" + "---|" * len(judges))
for ft, ids in sorted(by.items()):
    print(
        f"| {ft} | {len(ids)} | "
        + " | ".join(str(sum(judges[k][i] for i in ids)) for k in judges)
        + " |"
    )
