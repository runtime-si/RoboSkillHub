"""Units for the judge comparison: one per placement (frames pick_start..after_probe, the BrickSim verdict)."""

import json
import sys
from pathlib import Path

COLOR = {"1": None}
STEP_TEXT = {
    (
        "single_on_brick",
        "2",
    ): "put the blue LEGO brick on top of the red brick and press it onto the studs",
    (
        "single_on_plate",
        "1",
    ): "put the red LEGO brick on the grey baseplate and press it onto the studs",
    ("seq3_stack", "1"): "put the red LEGO brick on the grey baseplate and press it onto the studs",
    (
        "seq3_stack",
        "2",
    ): "put the green LEGO brick on the baseplate right next to the red brick and press it onto the studs",
    (
        "seq3_stack",
        "3",
    ): "put the blue LEGO brick on top of the red brick and press it onto the studs",
    (
        "seq3_spaced",
        "1",
    ): "put the red LEGO brick on the grey baseplate and press it onto the studs",
    (
        "seq3_spaced",
        "2",
    ): "put the green LEGO brick on the baseplate one stud away from the red brick and press it onto the studs",
    (
        "seq3_spaced",
        "3",
    ): "put the blue LEGO brick on top of the red brick and press it onto the studs",
    (
        "seq3_bridge",
        "1",
    ): "put the red LEGO brick on the grey baseplate and press it onto the studs",
    (
        "seq3_bridge",
        "2",
    ): "put the green LEGO brick on the baseplate right next to the red brick and press it onto the studs",
    (
        "seq3_bridge",
        "3",
    ): "put the blue LEGO brick across the red and green bricks, bridging them, and press it onto the studs",
}
ORDER = [
    "pick_start",
    "grasped",
    "before_place",
    "released",
    "retreated",
    "settled",
    "probe_end",
    "after_probe",
]

units = []
for root in sys.argv[2:]:
    root = Path(root)
    for f in sorted(root.glob("*__s0*.json")):
        d = json.loads(f.read_text())
        steps = {str(s["part"]): s for s in d["record"]["steps"]}
        for p in d["placements"]:
            st = steps.get(str(p["part"]))
            if st is None:
                continue
            snaps = st["snapshots"]
            frames = [
                str(root / snaps[k]["frame"]) for k in ORDER if k in snaps and "frame" in snaps[k]
            ]
            if len(frames) < 3:
                continue
            units.append(
                {
                    "id": f"{root.name}/{d['episode_id']}/part{p['part']}",
                    "scene": d["scene"],
                    "fault": d["fault"],
                    "part": str(p["part"]),
                    "faulted": bool(st.get("faulted")),
                    "success": p["status"] == "succeeded",
                    "failure_type": p["failure_type"],
                    "instruction": STEP_TEXT[(d["scene"], str(p["part"]))],
                    "frames": frames,
                    "settled_error": p.get("measured", {}).get("settled_error"),
                }
            )
Path(sys.argv[1]).write_text(json.dumps(units, indent=1))
print(len(units), "units;", sum(u["success"] for u in units), "succeeded")
