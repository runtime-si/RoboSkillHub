"""Guardian (paulpacaud/guardian-vanilla @ 5246b65, Apache-2.0 weights; Guardian-FailCot @ a3d3fa3 loader, MIT) on the
#155 LEGO placement units: single viewpoint, BEFORE = pick-start frame, AFTER = last frame. Needs the repo checkout with
peft (the checkpoint is an unmerged LoRA on InternVL3-8B; plain AutoModel leaves the LLM uninitialised), transformers
4.46.3 and the InternVL3-8B remote-code files (OpenGVLab/InternVL3-8B @ 853e3a7) beside the weights.

Result (75 units): passed 25/25 successes and 45/50 failures (false done) - it calls nearly every step a success."""

import json
import sys
import time
from pathlib import Path

R = Path.home() / "guardian" / "Guardian-FailCot"
sys.path[:0] = [str(R), str(R / "InternVL" / "internvl_chat")]
from InternVL.internvl_chat.guardian.guardian import Guardian  # noqa: E402

g = Guardian(sys.argv[1], thinking=False, disable_flash_attn=True)
units = json.loads(Path("manifest.json").read_text())
out = open("guardian_results.jsonl", "w")
for u in units:
    t = time.time()
    res = g.verify_subtask(
        [u["frames"][0], u["frames"][-1]],
        "build the LEGO structure on the grey baseplate",
        u["instruction"],
    )
    out.write(
        json.dumps(
            {
                "id": u["id"],
                "truth": u["success"],
                "ft": u["failure_type"],
                "res": [str(x) for x in (res if isinstance(res, tuple) else (res,))],
                "s": round(time.time() - t, 2),
            }
        )
        + "\n"
    )
    out.flush()
    print(u["id"], u["success"], res, flush=True)
