# Sources, reuse and limitations (Agent 02, domino)

## Reused from this hub (imported in place, not copied)

`tasks/domino/upstream.py` loads these scripts by path and calls their functions:

| hub file (sha256 at 58dcc23) | symbols used | upstream licence of the file |
|---|---|---|
| `skills/placing-an-upright-domino-beside-neighbours/scripts/place_domino.py` (`4eff369e…dea3f`) | `finger_conflicts`, `judge_tile`, `order_cameras`, `_cameras`; constants `POSITION_TOLERANCE`, `YAW_TOLERANCE_DEG`, `DESCENT_TOLERANCE`, `RELEASE_CLEARANCE`, `RELEASE_GAP`, `TRANSIT_CLEARANCE`, `TIP_REACH`, `GRIP_CLEARANCE`, `LEAD_IN`, `HEIGHT_TOLERANCE` | adapted from predicators (MIT), bundle Apache-2.0 |
| `skills/planning-a-domino-layout/scripts/plan_layout.py` (`8ff11d99…6ac6ae`) + `topple_physics.py` (`2a6c6af6…7c4c`) | `footprint`, `poly_distance`, `hand_envelope`, `in_corridor`, `critical_gap`, `OVERLAP_CLEARANCE`, `NEIGHBOUR_CLEARANCE`, `TOPPLE_MARGIN_MIN`, `DOMINO_FRICTION` | layout reimplemented in-hub; physics adapted from domino-fall-simulation (MIT) |
| `skills/detecting-disturbed-pieces-by-snapshot-diff` | its output (`disturbed` ids) accepted as neighbour evidence; not imported | Apache-2.0 bundle |

The motion sequence of `place_upright` (rise to transit height, cross, vertical descent to
`release_gap`, open to tile width + clearance, vertical retreat) is that skill's place half
expressed as rsh-task/1 primitives; its measured sim results (44/44 tiles stood, 0.1–1.9 mm)
belong to that skill on a MuJoCo station twin, **not** to this module.

Rules reused as rules (no code): far-first build order (DominoRobot FieldPlanner via
planning-a-domino-layout; GPL-3.0 code not used); staged lower/open/retreat semantics
(MoveIt Task Constructor `pick_place_task.cpp`, BSD-3-Clause, reference only).

## External sources (intake/domino.json): use

| source | pinned commit | use here |
|---|---|---|
| Franka-Domino-Project | 89cacd86 | reference only: task decomposition (place vs normalize vs topple kept separate); its force-based table contact is **not** claimed (YAM has no force feedback); its `current_domino_poses` software update is exactly what the independent observer replaces. No code copied; per-package licences (Apache-2.0 / MIT) not needed. |
| MoveIt Task Constructor | f527048b | reference only (approach/lower/open/retreat staging). |
| Geodude | 9dbc2a6b | reference only (surface place interface). |
| Dig-Grasping, ReorientBot, DominoRobot | — | not used (licence / scope); regrasp is Agent 01's. |

## Limitations (what is not established)

- **No physics run.** All tests are CPU contract tests on a synthetic kinematic world
  (`synthetic_contract_test`): faults are rules, not dynamics. Isaac evaluation is Agent
  04's; status `not_run` here.
- **No real robot**, no station, no calibration. `line_v1.yaml` values come from the
  placing skill's station twin or are labelled choices.
- Straight legs are dense waypoints (5 mm) unless a backend declares
  `linear_cartesian_motion`; between waypoints the path is backend-defined.
- No contact sensing: support before release is either a support observation or
  assumed from the table calibration (reported in `unverified`). `contact_feedback` mode is
  rejected.
- Without `ctx.state()` the pre-motion holding check only catches a tile seen resting on
  the support; other missed grasps are caught after the carry.
- Tilt needs `object_pose_feedback`; today only a privileged simulator provides it. An
  RGB-D centroid (position only) gives `unverified`; the placing skill's depth check can
  show a fall but never proves upright.
- Retreat-contact causes are hypotheses (medium confidence) from the timing of neighbour
  disturbance, not proof of contact.
- Reuse by path requires a repo checkout with `skills/`, the `gap` extra and scipy;
  otherwise the module rejects with `precondition_unmet`.
