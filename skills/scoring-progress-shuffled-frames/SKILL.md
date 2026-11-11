---
name: scoring-progress-shuffled-frames
description: Generative Value Learning - shows a VLM the frames of a step in a
  shuffled order (after the start scene, rated 0 %) and asks for each frame's
  task-completion percentage; put back in time order, the ratings give a
  progress curve whose rank correlation with time (VOC) says whether the step
  progressed, and the last frame's rating whether it is done. Routes complete,
  progress_increasing, stalled, regressed or no_verdict. Use when a graph has
  a step's frames and wants a "still progressing / stalled / undone" monitor
  from the camera alone, once per step (one multi-image call, ~20 s); not as a
  success gate for millimetre seating (see the check below).
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [task-monitor, progress, vlm, gvl, success-detection]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/budzianowski/opengvl/blob/36a39f3edede1935d3c0f5c8ba4ca724d5c0408b/opengvl/clients/base.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2411.04549
gap:
  allowed_tools:
    - vlm.query
  required_inputs:
    instruction: string
  # Optional inputs (wired from a script node): frames (list of RGB arrays or observations, oldest
  # first), initial (start frame; default the first), seed, done_pct (90), progress_voc (0.5),
  # regress_drop_pct (20), model.
  produces_outputs:
    last_pct: float
    voc: float
    reason: string
  exit_conditions:
    complete: The last frame (in time) is rated at least done_pct (90 %).
    regressed: The last frame is rated more than regress_drop_pct (20 points) under the best earlier frame.
    progress_increasing: Not complete, not regressed, and the ratings rise with time (VOC >= 0.5).
    stalled: Not complete, not regressed, and the ratings do not rise with time (VOC < 0.5 or undefined).
    no_verdict: Fewer than 2 frames with RGB, vlm.query failed, or the reply did not rate every frame.
  canonical_scripts:
    - score_shuffled: scripts/score_shuffled.py
  examples:
    - title: Rate a step's frames
      path: examples/canonical_subgraph.json
---

# scoring-progress-shuffled-frames

GVL (Ma et al., arXiv 2411.04549) as reimplemented by OpenGVL: shuffling the
frames stops the VLM from simply reading time order as progress; the ratings,
put back in time order, are a value curve.

## Live check (LEGO placements, Isaac + BrickSim, Gemini through vlm.query)

The 75 placement units of #155 (8 frames each, from pick start to after the
pull-up probe; BrickSim's verdict as truth), gemini-3-flash-preview, one call
per unit (median 21 s), seed 1:

| set | AUROC (last frame %) | `complete`: successes | `complete`: failures (false done) | no_verdict |
|---|---|---|---|---|
| main view (25: 11 ok) | 0.70 | 11/11 | 6/14 | 4 |
| other view (25: 11 ok) | 0.75 | 10/11 | 6/14 | 5 |
| gripper force 1.0 (25: 3 ok) | 0.90 | 3/3 | 3/22 | 7 |

- **High recall, many false "done"**: like the hub's Gemini judges (#155:
  10-11 false done of 14), GVL rates bricks resting ~1.7 mm high as done. It
  catches lost / missed grasps (regressed / stalled on 10 of the force-1.0
  failures). Compare RynnValue-8B (#155: AUROC 0.75 / 0.92 / 0.94, 0-1 false
  done at 2 s). Use it as a progress monitor, gate success with geometry
  (judging-assembly-steps-by-fused-checks).
- The ratings track time well on most steps (median VOC 0.93-0.94 on the two
  default-force sets, 0.58 on the force-1.0 set where steps fail midway).
- **no_verdict 16 of 75**: replies that did not rate every frame; the first run
  had 31 (Gemini bolds the labels, `**Frame 1**:`; the parser now accepts it).
  The thresholds (90 %, VOC 0.5, 20 points) are upstream-style defaults, not
  tuned here.

## Changes from OpenGVL

Adapted from `opengvl/utils/prompts.py`, `configs/prompt_phrases/default.yaml`,
`clients/base.py` (event order), `mapper/regex_mapper.py`, `metrics/voc.py`
(MIT, `LICENSE.opengvl`): frames come from the graph and are shuffled here
(seeded); the hub's vlm.query puts images after the text, so the prompt lists
them by number; ratings are read per "Frame i" label (upstream read them in
order and, for fractional values, rescaled them to sum to 100); no few-shot
context episodes; VOC and the last rating become exits.
