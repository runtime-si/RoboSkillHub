"""densereward.progress (the #124 tool) on the last frame (after_probe) and the settled frame of every unit."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "stub")
spec = importlib.util.spec_from_file_location("dr", "dr_tools.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
manifest, root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
units = json.loads(Path(manifest).read_text())
done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
for u in units:
    if u["id"] in done:
        continue
    rec = {"id": u["id"]}
    for name, path in (
        ("last", u["frames"][-1]),
        ("settled", u["frames"][-3] if len(u["frames"]) >= 3 else u["frames"][-1]),
    ):
        img = np.asarray(Image.open(root / path).convert("RGB"))
        r = T.progress(image=img, task=u["instruction"])
        rec[name] = {
            "progress": r["progress"],
            "found": r["found"],
            "raw": str(r.get("raw", ""))[:80],
        }
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(u["id"], u["success"], rec["last"]["progress"], rec["settled"]["progress"], flush=True)
