"""Offline demo of the perception_regrasp tools. SYNTHETIC data only.

1. Render a lying domino with the hub's pose6d ray-caster (station depth-noise
   model, wrist D405), fit it with the RGB-D provider, print the v1 Observation and
   its error against the render's pose.
2. Grasp candidates for the YAM linear_4310 on that observation.
3. reorient_or_regrasp plans on the i2rt YAM kinematic model: a direct wrist turn
   (lying -> standing) and a case where the direct turn is infeasible but a
   staging put-down / regrasp is feasible.

Nothing moves; the plans are kinematic-model results, not simulated or executed.

    PYTHONPATH=src python examples/tasks/perception_regrasp/offline_demo.py --out /tmp/perception_regrasp_demo
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from roboskillhub.runtime.task import validate
from roboskillhub.tasks.perception_regrasp import (
    MaskedObject,
    RgbdFrame,
    RgbdProvider,
    SimTruthProvider,
    Workcell,
    YamIk,
    _hub,
    domino,
    plan_reorient_or_regrasp,
    pose_error,
    propose_grasps,
)
from roboskillhub.tasks.perception_regrasp.regrasp import _resting_pose

SPEC = domino()
TABLE_IN_BASE = -0.033  # arm base 33 mm over the table, as on the station twin (illustrative)


def perceive() -> dict:
    syn = _hub.load("pose6d_synth")
    T_true = _resting_pose(SPEC, 2, 1.0, 0.4, (0.30, 0.02), 0.0)
    c = T_true[:3, 3]
    Tc = syn.look_at([c[0], c[1] - 0.12, 0.17], c)
    depth, mask = syn.render(
        {"box": list(SPEC.size_m)}, T_true, Tc, K=syn.K_D405, noise=syn.D405, table_z=0.0, seed=7
    )
    frame = RgbdFrame(depth, syn.K_D405, Tc, time.time(), "world", "wrist_synthetic")
    obs = RgbdProvider(
        lambda: frame,
        lambda f: [MaskedObject("d1", mask, source="render mask")],
        {"d1": SPEC},
        table_z=0.0,
    ).observe()[0]
    v1 = obs.to_v1()
    validate("Observation", v1)
    cands = propose_grasps("d1", obs.T_frame_object, SPEC, "world", 0.0)
    feasible = [c for c in cands if c.feasible]
    return {
        "data": "synthetic (pose6d _synth.render, D405 noise model)",
        "observation_v1": v1,
        "error_vs_render": pose_error(obs.T_frame_object, T_true, SPEC),
        "grasps": {
            "total": len(cands),
            "feasible": len(feasible),
            "best_v1": feasible[0].to_v1() if feasible else None,
        },
    }


def plan(start_xy, start_yaw, goal_xy, goal_yaw, staging=True) -> dict:
    T0 = _resting_pose(SPEC, 2, 1.0, math.radians(start_yaw), start_xy, TABLE_IN_BASE)
    Tg = _resting_pose(SPEC, 0, 1.0, math.radians(goal_yaw), goal_xy, TABLE_IN_BASE)
    truth = {"timestamp_s": 100.0, "frame_id": "base", "objects": {"d1": {"T_frame_object": T0}}}
    obs = SimTruthProvider(lambda: truth, {"d1": SPEC}, table_z=TABLE_IN_BASE).observe()[0]
    wc = Workcell("base", TABLE_IN_BASE, np.eye(4), YamIk())
    r = plan_reorient_or_regrasp(
        obs,
        SPEC,
        Tg,
        wc,
        now_s=100.1,
        allow_privileged=True,
        allow_staging=staging,
        max_path_evals=150,
    )
    for s in r["outputs"].get("steps", []):
        s.pop("q_path", None)  # long; kept in the API output
    return {
        "data": "synthetic start/goal; privileged sim-truth observation (labelled)",
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "result": r,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/perception_regrasp_demo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {
        "perceive_and_grasp": perceive(),
        "direct_lying_to_standing": plan((0.30, 0.0), 0, (0.30, 0.08), 0),
        "staging_case_direct_only": plan((0.34, -0.08), 0, (0.30, 0.10), 0, staging=False),
        "staging_case": plan((0.34, -0.08), 0, (0.30, 0.10), 0),
    }
    for name, data in runs.items():
        (out / f"{name}.json").write_text(json.dumps(data, indent=1))
        r = data.get("result")
        if r:
            o = r["outputs"]
            print(f"{name}: {r['status']} {r['failure_type'] or o.get('plan_type')} "
                  f"(path evals {r['evidence'].get('path_evals')}, {r['evidence'].get('planning_s')} s)")  # fmt: skip
        else:
            e = data["error_vs_render"]
            print(f"{name}: {data['observation_v1']['state_label']}, error {e['rotation_deg']:.2f} deg / "
                  f"{e['translation_m'] * 1e3:.2f} mm, {data['grasps']['feasible']}/{data['grasps']['total']} grasps feasible")  # fmt: skip
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
