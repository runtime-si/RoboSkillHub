"""Twist-only test (harness): pinch a bare piece across its width, hold it 0.5 mm over the mat, turn it to
yaw 0 with rotating-a-held-piece-within-wrist-limits in 10 degree steps, read its true yaw."""

import importlib.util
import json
import os
from pathlib import Path
from typing import TypedDict

from gap import NodeContext

HERE = Path(os.environ.get("GAP_SKILLS_PATH", "."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load(Path(__file__).resolve().parent / "e2e_helpers.py", "e2e_helpers")
R = _load(
    HERE / "skills" / "rotating-a-held-piece-within-wrist-limits" / "scripts" / "rotate_piece.py",
    "rot",
)


class Output(TypedDict):
    route: str


def run(ctx: NodeContext, index: int, yaw: float) -> Output:
    slot = {k: float(os.environ[f"JIGCHECK_SLOT_{k.upper()}"]) for k in "xyz"}
    ctx.tool("robot.open_gripper")
    here = E._pose(ctx.tool("robot.get_ee_pose"))
    ctx.tool(
        "robot.go_to_pose_cartesian",
        pose={
            "position": {**here["position"], "z": slot["z"] + 0.12},
            "rotation": here["rotation"],
        },
    )
    ctx.tool("check.place_piece", dx=0.09, dy=0.0, yaw_deg=yaw)
    E._wait(ctx, 80)
    p0 = ctx.tool("check.piece_pose")
    x, y = p0["position"]["x"], p0["position"]["y"]
    grasp_z = slot["z"] + 0.0102 + 0.0003
    above = {"x": x, "y": y, "z": grasp_z + 0.05}
    frame = ctx.tool("robot.grasp_frame", close_heading_deg=0.0)
    for key in ("rotation", "rotation_flipped"):
        r = frame.get(key)
        if (
            r
            and ctx.tool(
                "robot.go_to_pose", pose={"position": above, "rotation": r}, tolerance=0.001
            ).get("reached")
            is not False
        ):
            break
    E._square_jaws(ctx, p0["yaw_deg"], above, {})
    rot = E._pose(ctx.tool("robot.get_ee_pose"))["rotation"]
    ctx.tool(
        "robot.go_to_pose",
        pose={"position": {**above, "z": grasp_z}, "rotation": rot},
        tolerance=0.0003,
        max_steps=400,
    )
    ctx.tool(
        "robot.set_grip", object_width_m=0.021, squeeze_m=float(os.environ.get("SQUEEZE", "0.003"))
    )
    E._wait(ctx)
    p1 = ctx.tool("check.piece_pose")
    turn = E._wrap(-p1["yaw_deg"])
    out = R.run(
        ctx,
        turn_deg=turn,
        grasp_z=grasp_z,
        grip_width_m=0.021,
        carry_lift_m=float(os.environ.get("TWIST_LIFT", "0.0005")),
        max_step_deg=float(os.environ.get("TWIST_STEP", "10")),
    )
    p2 = ctx.tool("check.piece_pose")
    rec = {
        "index": index,
        "start_yaw": round(p1["yaw_deg"], 2),
        "turn": round(turn, 1),
        "route": out["route"],
        "hand_turned": round(out.get("turned_deg", 0.0), 1),
        "yaw_after": round(E._wrap(p2["yaw_deg"]), 2),
        "piece_turned": round(E._wrap(p2["yaw_deg"] - p1["yaw_deg"]), 2),
        "tilt": round(p2["tilt_deg"], 2),
        "moved_mm": round(
            1e3 * ((p2["position"]["x"] - x) ** 2 + (p2["position"]["y"] - y) ** 2) ** 0.5, 2
        ),
    }
    print("SKILLCHECK twist " + json.dumps(rec), flush=True)
    ctx.tool("robot.open_gripper")
    return {"route": "ok"}
