"""Score recorded attempts with judging-attempt-success-from-frames-robojev and the real tools/robojev.

    python -m tests.bundles._robojev_twin_eval EPISODES_DIR [--task "Put the cube in the box."]

EPISODES_DIR holds one folder per attempt with ``frames.npz`` (``top`` [N, H, W, 3] and ``wrist``
frames at the attempt's waypoints, ``tags``) and ``label.json`` (``success``: the simulator's verdict).
Each attempt is judged three ways - every overhead frame + the wrist camera's last frame, only the first
and last overhead frames + the wrist's last (what a before / now judge sees), every overhead frame with
no extra view - and each successful attempt is also judged cut at its lift (``lifted``), which is not
done yet. Needs a GPU with ~56 GB free; writes EPISODES_DIR/robojev_results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--task", default="Put the cube in the box.")
    a = ap.parse_args()
    tools = _load("robojev_tools", ROOT / "tools/robojev/tools.py")
    skill = _load(
        "judge",
        ROOT / "skills/judging-attempt-success-from-frames-robojev/scripts/judge_attempt.py",
    )
    rows = []
    for ep in sorted(p for p in Path(a.episodes).iterdir() if (p / "label.json").is_file()):
        label = json.loads((ep / "label.json").read_text())
        d = np.load(ep / "frames.npz")
        top, wrist, tags = list(d["top"]), list(d["wrist"]), [str(t) for t in d["tags"]]

        def obs(i, top=top, wrist=wrist):
            return {"cameras": [{"name": "top", "rgb": top[i]}, {"name": "wrist", "rgb": wrist[i]}]}

        cases = [
            ("all frames + wrist end", label["success"], top, -1, True),
            ("start/end + wrist end", label["success"], [top[0], top[-1]], -1, True),
            ("all frames, no views", label["success"], top, -1, False),
        ]
        if label["success"] and "lifted" in tags:
            k = tags.index("lifted")
            cases.append(("cut at lift (not done)", False, top[: k + 1], k, True))
        for name, truth, clip, end, views in cases:
            ctx = FakeContext({"robot.get_observation": obs(end), "robojev.judge": tools.judge})
            t = time.time()
            out = skill.run(ctx, task=a.task, frames=clip, camera="top", use_views=views)
            rows.append(
                {
                    "episode": ep.name,
                    "variant": label["variant"],
                    "case": name,
                    "truth": bool(truth),
                    "p_done": out["p_done"],
                    "route": out["route"],
                    "secs": round(time.time() - t, 3),
                }
            )
            print(
                ep.name,
                label["variant"],
                name,
                truth,
                round(out["p_done"], 4),
                out["route"],
                flush=True,
            )
    (Path(a.episodes) / "robojev_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
