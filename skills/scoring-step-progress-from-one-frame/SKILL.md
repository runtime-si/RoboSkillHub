---
name: scoring-step-progress-from-one-frame
description: Judges whether a step is done from the current camera frame alone
  - scores the task's progress 0..1 with the DenseReward single-frame reward
  model (densereward.progress) on a few fresh frames, takes the median, and
  routes done / in_progress / regressed, or low_confidence when the frames
  disagree or nothing parses. No start frame, no video, ~0.1 s a frame. Use
  when a graph needs a cheap "is this step done?" right after a place, press
  or push, with only the current camera view, as a second opinion beside a
  geometric check.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [judge, success-detection, progress, reward-model, single-frame]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting]
    provenance:
      authors: [runtime-si]
      code: https://huggingface.co/densereward/densereward-1frame/blob/9796cb2a7be55b339670d8f95fb07841a61487a8/README.md
      upstream_license: Apache-2.0
      paper: https://arxiv.org/abs/2607.13033
gap:
  allowed_tools:
    - robot.get_observation
    - densereward.progress
    - densereward.progress_with_history
  required_inputs:
    task: string
  produces_outputs:
    progress: float
    reason: string
    camera: string
    model: string
  exit_conditions:
    done: The median progress of the sampled frames is at least done_threshold (0.85) and they agree within max_spread.
    in_progress: The median is below done_threshold and the frames agree; also "not started" (the model scores 0 for both not started and failed).
    regressed: The progress fell more than regress_drop below previous_progress, or the 3-frame model (history given) named a failure (miss / collision / fall / smooth / failure).
    low_confidence: The sampled frames disagree by more than max_spread, the camera has no RGB, densereward raised, or no float parsed; nothing was judged.
  canonical_scripts:
    - score_progress: scripts/score_progress.py
  examples:
    - title: Score the current step from fresh overhead frames
      path: examples/canonical_subgraph.json
---

# scoring-step-progress-from-one-frame

A cheap "is it done yet?" judge that needs only the frame you have now.

```text
<the step: skills / policy> -> score (this skill) -> done: next step
                                                  -> in_progress: retry / continue
                                                  -> regressed: recover
                                                  -> low_confidence: ask a geometric check / a VLM judge
```

The hub's other progress judges need more: `robobrain.progress` and
`judging-step-with-before-now-vlm` a frame from before the step,
`verifying-a-domino-cascade-from-video` a video. This one reads one frame
(sampled a few times) and the task text.

## What it does

1. Takes `observation` (default a fresh `robot.get_observation`) and picks
   `camera` (default the first external camera, else a wrist one).
2. Scores it with `densereward.progress(image, task)`, then `samples - 1`
   more fresh frames of the same camera (default 3 in all), and takes the
   median. Spread over `max_spread` (0.3): `low_confidence`.
3. With `history` (earlier observations, oldest first) it instead sends
   the last two frames of that camera plus the current one, once, to
   `densereward.progress_with_history`, whose reason word routes
   `regressed` when it names a failure. Measured to be much weaker (below).
4. Routes: a failure word or a drop of more than `regress_drop` (0.3) below
   `previous_progress` is `regressed`; else the median against
   `done_threshold` (0.85).

## Inputs

- `task` (required): the step as an instruction, e.g. "put the cube in the
  black box". The model was trained on pick-and-place phrased that way.
- `observation`, `camera`, `samples` (3), `history`, `previous_progress`.
- `done_threshold` (0.85), `max_spread` (0.3), `regress_drop` (0.3).

## Outputs

`route`, `progress` (median, 3 decimals), `scores` (each sample), `reason`
(why the route, with the model's word if any), `camera`, `model`, `raw`.

## Measured (real YAM station overhead frames)

Full table in `tools/densereward/SKILL.md`. On a real cube-into-box episode
(33 frames over 64 s, each with its two neighbouring camera frames as the
other samples), the defaults routed:

| frames | done | in_progress | low_confidence |
| --- | --- | --- | --- |
| cube in the box (7) | 5 | 1 | 1 |
| before / during the motion (24) | 0 | 19 | 5 |
| cube taken back out by a person (2) | 0 | 0 | 2 |

No false `done` (a single frame would have given 3). On 65 more not-done
frames (a missed grasp, a refused reach, five static scenes) the single
frame never exceeded 0.34. Against `robobrain.progress` with the start
frame: it ranks the episode better (Spearman 0.93 vs 0.36) but never
scored a finished frame above 0.25, so it cannot route `done` by threshold.

What to know:

- Judge after the hand has moved clear when you can. Mid-reach and
  mid-grasp frames are where DenseReward jumped to 0.94.
- `regressed` by drop is only meaningful between scores taken at rest
  (after a step): during motion successive frames of the successful
  episode fell by 0.3-0.5 three times.
- The training scenes are DROID / LIBERO-style pick-and-place; bricks,
  dominoes and jigsaw pieces are untested. Check thresholds on your frames.
- `in_progress` covers "not started" and "failed" alike (the model gives 0
  for both). Use the 3-frame variant or another judge to tell them apart.

## Licence

The skill script is the hub's own (Apache-2.0). The models it calls through
`tools/densereward` are Apache-2.0 (see that bundle and NOTICE).
