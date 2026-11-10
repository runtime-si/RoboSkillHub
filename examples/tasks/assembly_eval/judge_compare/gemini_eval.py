"""The two hub VLM judges on every unit: detecting-subtask-success (last frame) and judging-step-with-before-now-vlm
(pick_start -> last frame), through the hub's vlm.query on Gemini's OpenAI-compatible endpoint. Local only."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

WT = Path(sys.argv[1])
manifest = Path(sys.argv[2])
root = Path(sys.argv[3])
out = Path(sys.argv[4])
MODEL = sys.argv[5]
keys = os.environ.get("GEMINI_API_KEYS", "").replace(";", ",").replace(" ", ",")
os.environ.update(
    GAP_VLM_PROVIDER="openrouter",
    GAP_VLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai",
    GAP_VLM_API_KEY=next(
        k.strip().strip("'\"[]") for k in keys.split(",") if k.strip().strip("'\"[]")
    ),
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vlm = load(WT / "tools/vlm/tools.py", "vlm_tools")
detect = load(WT / "skills/detecting-subtask-success/scripts/detect_success.py", "detect")
judge = load(WT / "skills/judging-step-with-before-now-vlm/scripts/judge_step.py", "judge")


class Ctx:
    def __init__(self, obs):
        self.obs = obs

    def tool(self, name, **kw):
        if name == "robot.get_observation":
            return self.obs
        if name == "vlm.query":
            return vlm.query(**kw)
        raise KeyError(name)


def obs(path):
    return {
        "cameras": [
            {"name": "scene_camera", "rgb": np.asarray(Image.open(root / path).convert("RGB"))}
        ]
    }


units = json.loads(manifest.read_text())
done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
for u in units:
    if u["id"] in done:
        continue
    now = obs(u["frames"][-1])
    t0 = time.time()
    d = detect.run(Ctx(now), subtask=u["instruction"], now=now, model=MODEL)
    t1 = time.time()
    j = judge.run(
        Ctx(now),
        before=obs(u["frames"][0]),
        task=u["instruction"],
        subgoal=u["instruction"],
        now=now,
        model=MODEL,
    )
    t2 = time.time()
    rec = {
        "id": u["id"],
        "model": MODEL,
        "detect_route": d["route"],
        "detect_reason": str(d.get("reason", ""))[:200],
        "judge_route": j["route"],
        "judge_status": j["status"],
        "judge_reason": j["reason"][:200],
        "detect_s": round(t1 - t0, 1),
        "judge_s": round(t2 - t1, 1),
    }
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(u["id"], u["success"], d["route"], j["route"], flush=True)
