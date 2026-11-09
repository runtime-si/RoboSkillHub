"""Consumer example for Agents 2 and 3: bind an observation, through a grasp and the
held state ``T_TCP_object``, to a Domino layout slot and to a LEGO mating target.
SYNTHETIC data only (poses typed in here, not measured).

Both targets use the same held-object semantics:
``T_W_tcp = T_W_object_goal @ inverse(T_TCP_object)``, over the part's symmetric
equivalents, with the contract's object frames on the wire (DominoLayoutTarget:
+z the tile's height axis).

    PYTHONPATH=src python examples/tasks/perception_regrasp/consumer_bindings.py
"""

from __future__ import annotations

import json
import math

import numpy as np

from roboskillhub.runtime.task import validate
from roboskillhub.tasks.perception_regrasp import (
    ObjectObservation,
    bind_domino_target,
    bind_lego_mating,
    domino,
    lego_brick,
    prior_from_grasp,
    propose_grasps,
)
from roboskillhub.tasks.perception_regrasp.frames import T_to_pose, make_T, pose_to_T, rot_z

# A v1 Observation of a standing tile (synthetic numbers, contract object frame).
DOMINO_OBS = {
    "object_id": "domino-3",
    "class": "domino",
    "frame_id": "world",
    "position_m": [0.32, -0.05, 0.024],
    "quaternion_xyzw": [0.0, 0.0, math.sin(0.15), math.cos(0.15)],
    "timestamp_s": 0.0,
    "source": "rgbd_pose6d",
    "privileged": False,
    "visibility": "visible",
    "geometry": {"type": "box", "size_m": [0.048, 0.024, 0.0075]},
}
# One DominoLayoutTarget slot (contract 3a).
SLOT = {
    "slot_id": "s0",
    "pose": {
        "frame_id": "world",
        "position_m": [0.40, 0.10, 0.024],
        "quaternion_xyzw": [0, 0, 0, 1],
    },
}
# A LegoMatingTarget-like placement on a calibrated plate (contract 3b fields; layer 1 = on
# the plate's top, as in computing-brick-pose-from-stud-grid - see the integration request).
PLATE_POSE = {
    "frame_id": "world",
    "position_m": [0.35, -0.10, 0.0032],
    "quaternion_xyzw": [0.0, 0.0, math.sin(0.1), math.cos(0.1)],
}
PLACEMENT = {"stud_x": 4, "stud_y": 6, "layer": 1, "yaw_quarter_turns": 1}


def main() -> None:
    spec = domino(tuple(DOMINO_OBS["geometry"]["size_m"]))
    obs = ObjectObservation.from_v1(DOMINO_OBS)
    grasp = next(
        c
        for c in propose_grasps(obs.object_id, obs.T_frame_object, spec, "world", 0.0)
        if c.feasible
    )
    held = prior_from_grasp(grasp, spec, "yam_left", obs.timestamp_s, obs)
    validate("HeldObject", held.to_v1())
    d = bind_domino_target(
        held, pose_to_T(SLOT["pose"], "world"), "world", T_frame_tcp_now=grasp.T_frame_tcp
    )
    best = pose_to_T(d["outputs"]["tcp_targets"][0]["tcp_pose"], "world")
    print("domino: grasp", grasp.closing_axis, grasp.approach_face, "-> slot", SLOT["slot_id"])
    print("  T_TCP_object (contract frame):", json.dumps(held.to_v1()["T_TCP_object"]))
    print("  TCP target:", json.dumps(T_to_pose(best, "world")))
    placed = best @ pose_to_T(held.to_v1()["T_TCP_object"], "tcp")
    print(
        "  part lands at", np.round(placed[:3, 3], 5).tolist(), "slot", SLOT["pose"]["position_m"]
    )

    brick = lego_brick((2, 4))
    T_brick = make_T(rot_z(0.3), [0.25, 0.08, brick.size_m[2] / 2])
    bg = next(c for c in propose_grasps("b2", T_brick, brick, "world", 0.0) if c.feasible)
    bheld = prior_from_grasp(bg, brick, "yam_left", 0.0)
    cell = {
        "x": PLACEMENT["stud_x"],
        "y": PLACEMENT["stud_y"],
        "layer": PLACEMENT["layer"],
        "rotation_deg": 90 * PLACEMENT["yaw_quarter_turns"],
    }
    # Plate frame at the centre of its top surface (calibrating-baseplate-stud-grid's plate_pose).
    b = bind_lego_mating(
        bheld, pose_to_T(PLATE_POSE, "world"), (16, 16), cell, "world", hover_m=0.002
    )
    print("lego: seated brick pose", json.dumps(b["outputs"]["seated_brick_pose"]))
    print(
        "  TCP targets (2 yaw equivalents, 2 mm hover):",
        [t["tcp_pose"]["position_m"] for t in b["outputs"]["tcp_targets"]],
    )


if __name__ == "__main__":
    main()
