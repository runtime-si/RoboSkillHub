---
name: judging-assembly-steps-by-fused-checks
description: Turns the routes and numbers of the hub's per-step checks into one
  verdict for an assembly step - done, failed with the reason, or uncertain -
  trusting per step type the checks that see its failure modes. Pick - the
  gripper-opening stall (and a scene-diff / VLM check only to contradict).
  Place / press - depth seat height and the brick's rise under the pull-up must
  both pass; the jaws must have opened at the release; a camera VLM only breaks
  a tie towards failure; RynnValue's remaining time can hold back a done.
  Routes done, failed or uncertain. Use when a LEGO, domino or jigsaw graph
  must decide after a pick or a place whether to go on, retry or ask, with
  whichever of those checks ran; logic only, no motion, no camera.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [judge, verification, fusion, success-detection, failure-detection, lego, assembly]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [lego, pick-place, domino, jigsaw]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools: []
  required_inputs:
    step_type: string
  # Optional inputs (empty string / null = the check did not run): grip_route, held,
  # scene_diff_route (pick); release_grip_route, seat_route, height_error_m, pull_route,
  # brick_rise_m (place); vlm_route, rynn_remaining_s (both); max_height_error_m,
  # max_pull_rise_m, rynn_veto_s.
  produces_outputs:
    reason: string
    step_type: string
  exit_conditions:
    done: Pick - the jaws stopped on the part (tracking-gripper-opening-stall stalled_on_object, or the hand reports it held) and no camera check contradicts. Place - the jaws opened at the release, the depth seat reads seated within max_height_error_m (0.3 mm) and the brick rose at most max_pull_rise_m (0.6 mm) under the pull-up; and RynnValue, if given, expects at most rynn_veto_s (2.5 s) more.
    failed: A check that sees this failure mode failed - the jaws closed on nothing or on the wrong width (pick), did not open at the release, both geometric checks failed, the only geometric check failed, or they disagree and the camera judge says not done; `reason` names it.
    uncertain: The evidence cannot pass the step - no geometric check ran (a camera judge alone passes unseated bricks), only one of the two ran and passed, they disagree without a camera "not done", the jaws hold something the camera says is still at the source, or RynnValue vetoed a done.
  canonical_scripts:
    - fuse_checks: scripts/fuse_checks.py
  examples:
    - title: Fuse a place step's depth seat and pull-up readings
      path: examples/canonical_subgraph.json
---

# judging-assembly-steps-by-fused-checks

One verdict per step from the checks that ran, with the reason.

```text
pick  -> tracking-gripper-opening-stall [-> checking-pick-by-scene-diff] -> fuse (step_type=pick)
place -> release (gripper) -> verifying-brick-seated-by-height -> verifying-a-brick-connection-by-pull-up
         [-> detecting-subtask-success] [-> rynnvalue.remaining_time] -> fuse (step_type=place)
```

## Why these rules

Measured on the same LEGO placements (#155, Isaac + BrickSim, 75 units):
the Gemini camera judges passed nearly every success and caught every grasp
failure but called 13-19 of 20 bricks resting ~1.7 mm high "done";
RynnValue-8B at 2.0 s remaining passed 1 failure in 75 but only 3-5 of 11
successes; DenseReward scored most grasp failures done. So a place is passed
only by geometry (depth seat **and** pull-up), a camera judge may only turn a
geometric disagreement into a failure, and RynnValue may only hold a done back.
A pick is judged by the jaws (tracking-gripper-opening-stall: closing on nothing
or on the wrong width fails it); a scene-diff or VLM "not changed" only makes
a held grasp uncertain (a wrong part, or the hand still over the source).

The pull-up enters as the brick's own rise during the lift (`brick_rise_m`)
when given: the pull-up skill's route reads the hand's tracking lag, which the
LEGO station's floating hand never shows - it routed `not_connected` on all
207 placements below, seated or not, while the brick rise separated them.

## Held-out check: LEGO physics station (Isaac Sim 5.1 + BrickSim)

Episodes from the hub's LEGO graph with its verifiers in the loop
(`examples/tasks/assembly_eval/lego_graph_hub.py`, report mode so every check
runs), GCP pool, 2026-11-09: 4 scenes (single_on_plate, single_on_brick,
seq3_stack, seq3_spaced) x no fault and the six station faults (perception
offset 3 mm / yaw 9 deg, force bias, stuck jaws, retreat deviation, lost
target), plus two low grip-force variants; seeds 0-5. Seeds differ: the station
randomises each brick's feeder pose (+-1 mm, +-2 deg) per seed. 169 episodes,
207 placements; truth is the independent evaluator's per-placement verdict.
Thresholds chosen on seeds 0-2 (132 placements: the largest acceptance with
false done <= 1 %: 0.3 mm, 0.6 mm), reported on seeds 3-5:

| judge, held-out seeds 3-5 (75 placements: 42 succeeded, 33 failed) | false done | successes accepted | uncertain | failures caught |
|---|---|---|---|---|
| **this skill** | **0 / 33** | **36 / 42** | 6 | **33 / 33** |
| depth seat alone (verifying-brick-seated-by-height route) | 0 / 33 | 36 / 42 | 12 | 27 / 33 |
| task verify alone (lego.verify_connection) | 0 / 33 | 36 / 42 | 12 | 27 / 33 |
| brick rise <= 0.6 mm alone | 6 / 33 | 42 / 42 | 0 | 27 / 33 |
| pull-up route alone (hand lag) | 0 / 33 | 0 / 42 | 0 | 33 / 33 |
| jaws opened at release alone | 27 / 33 | 36 / 42 | 0 | 6 / 33 |

Failures held out: not seated 27, stuck jaws (the part left with the hand) 6.

- **No false done in 75 held-out placements (0 / 33 failures)**; with 33
  negatives that bounds the rate below ~9 % (95 %), not yet the 1 % target.
- The fusion's gain over the depth check is the stuck-jaws case: depth has no
  brick to read there (unverified), the release gripper reading fails it.
- The 6 successes it does not accept are the lost-target fault (the brick
  vanishes from the observations after its release): no check can see it, so
  uncertain is the honest answer.
- **No pick failures occurred** on this station (207 of 207 grasps held, also
  at 0.2-0.5 N grip): the pick rules are checked by the unit tests and by the
  twin runs of tracking-gripper-opening-stall (10/10) and
  checking-pick-by-scene-diff (#119), not by this data.
- The camera judges and RynnValue were not in this check: the LEGO station has
  no rendered camera yet (an RTX camera for it is in progress elsewhere).
  Their roles come from #155's frames; the depth here is the station's
  ray-cast depth model, not an RTX render.

## Inputs

`step_type` (`pick`, `place`; `press` / `seat` mean `place`) and the outputs of
whichever checks ran, wired from their nodes: routes as strings, numbers as
floats; an empty string or null means "did not run". Thresholds:
`max_height_error_m` (0.3 mm), `max_pull_rise_m` (0.6 mm), `rynn_veto_s`
(2.5 s; #155: successes ended at 0.9-3.0 s, failures 1.9-5.1 s).
