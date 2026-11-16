---
name: cross-checking-a-placed-count-from-overhead
description: Cross-checks a count of parts said to have been placed into an open
  container (the hand's own tally of releases, an operator's number) against a
  rough estimate from one overhead RGB-D frame - what stands over the
  container's floor in the plan view, as blobs and as occupied area over one
  part's area - and routes contradicted when the camera sees under half of
  what is claimed, or anything where nothing is claimed. It is NOT a count
  verifier - on the real station it came from the estimate was off by one or
  two of four parts whenever parts were in the box (touching parts read high,
  stacked parts low, shadows as parts), so it only catches gross disagreement
  such as every part having landed outside the container. Use when a placing
  loop keeps its own count and needs a second, independent signal that the
  parts are in the container at all, with the hand out of the overhead view
  and one part's plan-view area measured on that camera; never use the
  estimate as the success criterion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verification, counting, container, depth, overhead, cross-check, caution, cpu]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.1.0
    tuned_on: [yam_i2rt]
    tasks: [sorting]
    provenance:
      authors: [RoboSkillHub contributors]
    notes: A caution more than a capability. The estimate matched the true count in 1
      of the 4 real runs where the truth is known (the one where it was 0). Routes
      only gross disagreement; needs an independent count to say anything.
gap:
  allowed_tools:
    - robot.get_observation
  required_inputs:
    container_obb: OrientedBoundingBox
    part_area: float
  # Optional inputs: claimed_count (int), camera (string), dark_parts (bool). Wired from a script
  # node (gap has no type for it): camera_frame (CameraFrame).
  produces_outputs:
    estimate: int
    blobs: int
    by_area: int
    area_m2: float
    claimed_count: int
    seen_fraction: float
  exit_conditions:
    supported: At least half of the container's inside was in view and the camera's rough estimate is not under half of claimed_count (or both are 0). The claim is not contradicted; it is not confirmed either - the estimate was wrong by 1-2 of 4 in the real runs.
    contradicted: The camera's estimate is under half of claimed_count (including 0 seen), or something is seen where 0 is claimed. On the source station this pattern meant every release had missed the container. Someone, or another sensor, must look.
    uncertain: No judgement - no claimed_count was given (the estimate alone decides nothing), under half of the container's inside is in view (a hand over it, a wrong box), or there is no posed overhead RGB-D camera; `detail` says which. Route as not verified.
  canonical_scripts:
    - cross_check_count: scripts/cross_check_count.py
  examples:
    - title: Check the hand's tally against the overhead view
      path: examples/canonical_subgraph.json
---

# cross-checking-a-placed-count-from-overhead

Read this as a warning with a small tool attached.

## What went wrong with counting from overhead

The task this comes from had to put four printed parts into a box and say
when it had. Its overhead count goes by area: the plan-view cells that stand
over the box floor (or are dark on it), opened, counted as blobs and as area
over one part's area. Against what the operator found in the box:

| real run | in the box (truth) | overhead estimate | why |
| --- | --- | --- | --- |
| 1 | 4 (operator's count) | 5 (3 blobs, 141 cm2) | area rounding; fixed by re-measuring one part's area on this very frame |
| 2 | 4 (operator confirmed) | 5 (1 blob, 167 cm2) | parts lying touching |
| 3 | 4 (hand's tally, no part lost) | 2 (61 cm2) | all four let go over one point and stacked |
| 4 | 0 (all 4 beside the box, seen in the picture) | 0 | correct |
| earlier | not checked | 2, with 1 placed by the hand's tally | - |

Before a colour test was added, the arm's shadow in the empty box counted as
two parts. So: exact once in four, and that once was the empty box. Stacking
makes it read low, touching and shadows make it read high, and no setting
fixes both.

What made it worse was the verdict built on it. Each time the rule was
loosened so that a good-looking run would pass (take the larger of the
camera's count and the hand's tally), it produced a false success: in run 4
the hand had let go four times beside the box, the camera saw 0, and "the
larger" reported the task done. The lesson recorded there: a run's verdict
needs independent evidence and must never be widened to make a run pass;
record the operator's count as ground truth.

## What this skill does instead

It keeps the one thing the estimate was good for. A camera that sees
**nothing** in the box when the hand claims four is strong evidence the
parts are not there. So the skill compares, and only flags gross
disagreement:

- `contradicted`: the estimate is under half of `claimed_count`, or
  non-zero when the claim is 0;
- `supported`: it is not;
- `uncertain`: there is no claim, or the box is not properly in view.

"Half" is the rule the source task ended with. It was written after run 4,
which it would have caught; it has not been shown to catch a second case,
and it lets through a claim of 4 with 2 seen (run 3, where 4 were really in).

## Why its stage is `judge`, not `verify`

A `verify` skill is offered by search as the condition that closes a step.
This one must not be picked for that: it is filed under `judge` and under the
`sorting` task only, so a search for "put the part in the box" does not offer
it as the verifier.

## What it does not do

- It does not verify a count. `supported` with claim 4 is consistent with
  2 to 4 or more parts in the box.
- It does not replace the second signal: without `claimed_count` it always
  routes `uncertain`.
- It does not tell parts from anything else in the box.

For a real verdict use something that does not share its failure modes: the
jaw reading at each release with the hand located over the box by its own
camera (`locating-a-container-from-the-wrist-camera`), a wrist close-up, a
VLM check of the box (`detecting-subtask-success`), a scale, or a person.

## How

1. The frame's depth is lifted to the world and binned at 2 mm in the
   container's frame, 15 mm inside its walls; the highest point per cell and
   its colour are kept. Under 50 % of cells seen: `uncertain`.
2. A cell is occupied when it stands 12 mm over the container's support, or
   (`dark_parts`) is darker than 0.6 of the floor's median brightness and not
   floor-tinted.
3. Occupied cells are opened by 1 cm; blobs of at least 12 cm2 are counted;
   `by_area = round(area / part_area)`; `estimate = max(blobs, by_area)`, or
   0 with no blob.

Preconditions: the hand and arm out of the camera's view of the container
(the source moved the hand aside and took the median of 8 frames), and
`container_obb` in this camera's frame.

## Constants, and what they were measured on

One YAM station's overhead depth camera, a 238 x 196 mm cardboard box with a
66 mm rim, grey 5-9 cm printed parts, 2026-11-15.

- `part_area`: an input. There it was 36 cm2 (2 parts covered 75 cm2, 4
  covered 141 cm2); at 30 cm2 those four counted as five.
- `FLOOR_CLEAR = 12 mm`, `OPEN = 5 cells`, `MIN_BLOB_CELLS = 300` (12 cm2),
  `MARGIN = 15 mm`: chosen there, not swept.
- `DARK_BELOW = 0.6` and the tint test: for grey parts on brown cardboard
  only; off by default here, on in the source.
- `MIN_SEEN = 0.5`, and the `claimed_count == 0` rule: chosen here, on no
  robot.

## Evidence

The table above is all of it: five real runs on one station, four with a
known truth. Nothing was run in a simulator suite. This port's own numbers
come from unit tests on synthetic boxes. Maturity is `community`, and it
should stay there until someone shows the `contradicted` exit catching
misses it was not written from.

## Required end states

- `supported`: not contradicted. Not a confirmation.
- `contradicted`: under half of the claim seen. Stop and look.
- `uncertain`: nothing can be said.
