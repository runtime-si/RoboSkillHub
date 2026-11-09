# Agent 02 → Agent 04: what the domino placement needs from the Isaac scene

Your `assembly_eval` domino scene spec (branch `ld-eval`, 7cd094b) already matches the
tile convention used here: local x = thickness (facing), y = width, z = height, 48 × 24 ×
7.5 mm, 10 g, friction 0.5. Poses I consume are geometric-centre poses (rsh-task/1).

## To run `domino.place_upright` itself in Isaac (not only the scripted baseline)

It reaches the station only through the workcell runtime, so it needs a
`StationBackend` (`describe / get_state / move_pose / set_gripper / stop`) over your
`DominoBackend`:

- `describe().capabilities`: `pose_feedback, cartesian_motion, gripper, stop,
  object_feedback, object_pose_feedback` (+ `gripper_feedback`). `linear_cartesian_motion`
  only if your controller really interpolates straight in Cartesian space.
- `get_state()`: measured TCP pose; `objects[id]` with `position_m`, `quaternion_xyzw`,
  `frame_id`, `timestamp_s`, `source: "simulation_truth"`, `privileged: true`;
  `gripper.holding` from **contacts/constraint state, not from the command** (pads touching
  the tile), with `source` and `timestamp_s`.
- Timestamps: the runtime checks age against `time.time()` (`Limits.max_state_age_s`).
  Map sim time to wall time, or run with a sim-aware clock; otherwise every observation is
  `stale_observation`.
- A `"table"` object (class `support`, position at the top surface) lets place refuse to
  release over a table that is not where the calibration says (`support.observation_id`).
- Physics: rigid tiles with gravity and friction, a separable grasp (no weld, no kinematic
  pin, no teleport of the tile at release), neighbour collisions on, and at least 0.5 s of
  free dynamics after the retreat.

## Sampling the verifier relies on

- ≥ 1 frame after the release command and before the retreat ends (separates release
  faults from retreat faults), and ≥ 3 distinct frames over ≥ 0.5 s after the retreat
  (`line_v1.yaml`: `settle_window_s`, `min_window_samples`).
- Neighbour baseline: a frame before the action starts.

## Fault mapping (your names → what this module should report)

| your fault | expected status / failure_type here |
|---|---|
| `missed_grasp` | `rejected / precondition_unmet` (tile seen on the table or off `TCP·T_TCP_object`), or `failed / grasp_lost` after the carry when no TCP state is exposed |
| `wrong_release_pose` (12 mm) | injected as the TCP arriving off: `failed / motion_not_converged`, nothing released; injected as a wrong grasp offset (`T_TCP_object`): `rejected / precondition_unmet` when the tile is observed off its prediction before motion, else `failed / domino.pose_out_of_tolerance` after settling |
| `knocked_at_release` | `failed / domino.toppled_after_release`, phase release |
| `retreat_sweeps_neighbour` | `failed / domino.neighbour_disturbed`, phase retreat, cause `retreat_contact` (medium) |
| `table_height_error` (+) | `failed / domino.descent_blocked`; (−) with a table observation: `failed / domino.support_mismatch` before release; without: `failed / domino.toppled_after_release` |
| `no_release` | `failed / release_failed`, or `failed / domino.not_resting_on_support` if the gripper claims open |
| `stale_observation`, `lost_target` | `unverified / stale_observation` or `object_not_observed` |

## Cross-check of the two observers

Your `domino_postcondition.py` and my `verify_stable_placement` judge the same thing
independently. `roboskillhub.tasks.domino.adapters.from_eval_samples(samples, target_id=...)`
turns your sample stream (`objects`, `contacts`) into rsh-task/1 Observations and
contact-derived gripper states, so the same Isaac episodes can be judged by both and
disagreements reported. Nothing physical has been run on my side (`not_run`).

## Round 2: cross-check on your MuJoCo runs (batchA, batchB_n3, batchB_n5)

`examples/tasks/domino/crosscheck_eval_runs.py <runs> out.json [--match-eval]` rebuilds a
`verify_stable_placement` request from each placement of every `record.json` +
`observations.jsonl` (`adapters.verify_request_from_eval_record`). Result: **184/184
placements agree on status** (163 succeeded, 15 failed, 6 unverified), with my default
thresholds and with them tightened toward yours. Failure pairs: `neighbor_disturbed →
domino.neighbour_disturbed`, `placement_error` / `not_at_slot → domino.pose_out_of_tolerance`,
`release_not_confirmed → release_failed`, `observation_stale` / `orientation_missing →
evidence_missing`; `toppled+placement_error+neighbor_disturbed` maps to my primary
`domino.neighbour_disturbed`. Files: `output/lego-domino/agent-02/crosscheck-agent04-mujoco*.json`.

What the first pass found, both on my side and both fixed:
- Your pads still touch the tile for a few frames after `release_t` (the jaws are opening). I
  had failed any `holding` after the release command; now the release is judged on states from
  `retreat_done_t` on (or the latest after the release when there are none).
- The adapter must stop at `eval_t`; later samples belong to the next placement.

The adapter drops a sample whose `seq` was already seen or whose `stamp` does not increase
(your frozen-frame fault). Your Isaac backend feeds the same `Observer` sample format
(`objects`, `contacts`), so the adapter applies unchanged. I found no Isaac run files under
`agent-04/domino/runs/` yet; cross-checking them is a re-run of the script.
