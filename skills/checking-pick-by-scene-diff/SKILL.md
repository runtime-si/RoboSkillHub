---
name: checking-pick-by-scene-diff
description: Checks whether a pick (or a place, a push, any intended change)
  happened by showing a VLM one panel - the camera frame from before the step,
  the frame now, and a hot pixel-difference heatmap side by side - and asking
  whether anything changed and whether the expected change happened; when the
  graph knows where the object was (or should go), the pixels there must agree
  (changed, and emptied for a pick or filled for a place), else the verdict is
  uncertain. Routes changed_as_expected, changed_otherwise, unchanged,
  uncertain or no_verdict. Use when a pick (or place) has just run and the
  gripper width alone cannot tell held from empty (thin or small parts), so a
  source-side check is wanted, with a robot.get_observation captured before
  the step; camera-only, no motion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verification, grasp-quality, success-detection, before-after, pixel-diff, vlm]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, sorting, jigsaw, lego, domino]
    provenance:
      authors: [runtime-si]
      code: https://github.com/nssmd/RoboRSI/blob/90c8569705bf4004a4f79e34e650c87ab7f075d4/roborsi/embodied/skills/base/visual_diff/robotwin/policy.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - vlm.query
  required_inputs:
    before: Observation
    expected_change: string
  # Optional inputs (wired from a script node; gap has no list type here): now
  # (Observation, default a fresh robot.get_observation), camera, source_box_px
  # [x0, y0, x1, y1], source_mask_px (H x W bool), region_expect (vacated | filled |
  # changed), diff_threshold, min_source_change, model.
  produces_outputs:
    changed: bool
    matches_expectation: bool
    reason: string
    changed_fraction: float
    source_changed_fraction: float
    source_contrast_ratio: float
  exit_conditions:
    changed_as_expected: The VLM answered CHANGED yes and MATCH yes, and (when a source region is given) at least 15 % of its pixels changed and it emptied (vacated, contrast ratio <= 0.85) or filled (filled, ratio >= 1.5) as region_expect asks.
    changed_otherwise: The VLM saw a change (CHANGED yes) but not the expected one (MATCH no) - e.g. the hand moved over the object, or another object moved.
    unchanged: The VLM saw no change (CHANGED no, MATCH no) and the source region, if given, did not change either.
    uncertain: The VLM and the pixels disagree - MATCH yes but the region did not change, did not empty or did not fill as expected, MATCH yes with CHANGED no, or CHANGED no while the region changed; re-observe (move the hand clear) before trusting either.
    no_verdict: Nothing was judged - no camera has RGB in both observations, the frames differ in size, vlm.query raised, or the reply lacked a CHANGED or MATCH line.
  canonical_scripts:
    - check_scene_diff: scripts/check_scene_diff.py
  examples:
    - title: Check a pick against the observation captured before it
      path: examples/canonical_subgraph.json
---

# checking-pick-by-scene-diff

A before / after verifier: did the step change the scene the way it should?

```text
before (tool: robot.get_observation) -> <pick / place / push> -> check (this skill)
```

Capture `before` with a `robot.get_observation` tool node just before the
step; after it, run this skill with a one-line `expected_change`. It takes a
fresh observation, builds the panel and asks once.

## What it does

1. Picks one camera present in both observations (`camera`, else the first
   external one, then wrist ones) and requires the two frames to be the same
   size.
2. Builds RoboRSI's panel, `BEFORE | AFTER | DIFF (hot)`: the per-pixel sum
   of absolute RGB differences, normalised to the frame's largest difference
   and coloured with OpenCV's HOT map, labelled in the top-left corners.
3. Asks `vlm.query` with RoboRSI's prompt: report `CHANGED: YES/NO` (did
   anything visibly change) and `MATCH: YES/NO` (did the expected change
   happen), then a one-line reason.
4. If the graph gives the source region (`source_box_px` or
   `source_mask_px`: where the picked object was, or where a placed one should
   land), it measures two numbers there that must agree with the VLM:
   - `source_changed_fraction`: share of region pixels whose summed RGB
     difference exceeds 30 (upstream's threshold); at least 15 % must change.
   - `source_contrast_ratio`: how much the region stands out from a ring
     around it (mean RGB distance from the ring's median colour), after /
     before. A pick empties the region (`region_expect: vacated`, ratio
     <= 0.85); a place fills it (`filled`, ratio >= 1.5); `changed` skips
     this test.

| CHANGED | MATCH | region (if given) | exit |
|---|---|---|---|
| yes | yes | agrees | `changed_as_expected` |
| yes | yes | disagrees | `uncertain` |
| no | yes | - | `uncertain` |
| yes | no | - | `changed_otherwise` |
| no | no | unchanged | `unchanged` |
| no | no | changed as expected | `uncertain` |
| missing line / call failed | | | `no_verdict` |

The panel is returned as `panel` for the trace.

## Inputs

- `before`: observation before the step (cameras with `name` and `rgb`, a
  list or a mapping). `now`: defaults to a fresh `robot.get_observation`.
- `expected_change`: one line, e.g. "the red brick was picked up: it is no
  longer at its spot on the table" or "the red brick is inside the grey box".
- `camera` (auto), `source_box_px` / `source_mask_px` (none),
  `region_expect` (`vacated`), `diff_threshold` (30), `min_source_change`
  (0.15), `model` (the tool's configured one).

Draw the source box about 1.5x the object's image footprint, from the
perception that planned the pick (a detection box, a projected OBB).

## Live check

`run()` through the hub's `vlm.query` (OpenAI-compatible provider on Gemini's
endpoint), 11 labelled pairs x 2 repeats per model, one 640 x 480 overhead
frame per side. Sim pairs come from a YAM station twin's overhead camera in an
earlier sorting run (small 15 mm blocks); real pairs from the station's
overhead camera.

| pair (BEFORE -> AFTER), expected change | truth | src changed | contrast ratio |
|---|---|---|---|
| sim: red block on the table -> moved into the box (a blue one dropped near its spot), "red block picked up" | yes | 0.77 | 0.77 |
| same pair, "red block put inside the grey box" (region = box, `filled`) | yes | 0.65 | 3.74 |
| sim: same frame twice | no | 0.00 | 1.00 |
| sim: same scene, re-rendered | no | 0.00 | 1.00 |
| sim: hand hovering over the block, not yet closed | no | 0.92 | 1.65 |
| sim: a blue block appeared, "blue block picked up" | no | 0.72 | 2.17 |
| sim: red block in the box stays while a blue one is dropped | no | 0.00 | 1.00 |
| real: both arms moved, brick plate stays, "plate picked up" | no | 0.02 | 1.00 |
| real: plate removed (synthetic: inpainted out of the real frame) | yes | 0.43 | 0.22 |
| real: plate removed, "marker holder picked up" | no | 0.00 | 1.01 |
| real: static cable scene, no region | no | - | - |

| model | skill correct | false `changed_as_expected` | `uncertain` | VLM alone (CHANGED and MATCH) correct | VLM alone false yes |
|---|---|---|---|---|---|
| gemini-3-flash-preview | 22/22 | 0 | 2 | 20/22 | 2 (blue block appeared, both repeats) |
| gemini-robotics-er-2-preview | 22/22 | 0 | 2 | 20/22 | 2 (hand hovering 1/2, block appeared 1/2) |

2-8 s per judgment (one flash reply 20 s). What to know:

- **The VLM alone says "picked" when the source is covered or replaced.** Both
  models twice called a pick done when the hand hid the block or another block
  landed in its spot; the region's contrast ratio (1.65, 2.17: the region got
  busier, not barer) sent those to `uncertain`. Nothing true was rejected.
- **The thresholds were set on these same pairs** (in-sample): 0.85 sits
  between the vacated 0.77 and the static 1.00, 1.5 between 1.00 and the
  filled 3.74, 15 % between 2 % and 43 %. The sim "picked" pair's 0.77 is high
  because the next block was dropped on nearly the same spot; a bare table
  gives ~0.2 (the real case).
- **The only real-camera "picked" pair is synthetic** (the plate inpainted
  out of a real frame): no real pick before/after pair was available. Real
  picks were checked in the twin instead (below).
- Flash said `CHANGED: YES` for two re-renders of the same scene (pixel share
  0.0), so `changed_otherwise` can mean "the VLM imagined a change"; ER-2
  said NO. The routing never turns that into a pick.
- Judge after the hand has moved clear of the source, as for
  judging-step-with-before-now-vlm: a hand over the spot reads as "changed
  otherwise" or `uncertain`.

### Real picks and misses in the station twin

The arm really picked (or missed) a 25 mm red cube in a YAM station twin
(the tracking-gripper-opening-stall sim run on the GCP pool): overhead frames
before the approach, after an 8 cm lift (hand over the spot) and after moving
12 cm aside; truth from the cube's simulated pose. Source box 23 x 23 px
around the cube (~15 px wide). 8 pairs x 2 repeats per model:

| pair | truth | flash (2 runs) | ER-2 (2 runs) | src changed | contrast ratio |
|---|---|---|---|---|---|
| pick, hand moved aside | picked | as_expected x2 | as_expected x2 | 0.82 | 0.03 |
| second pick, hand aside | picked | as_expected x2 | as_expected x2 | 0.86 | 0.02 |
| a "miss" 40 mm off that grasped the cube anyway (lifted 91 mm), hand aside | picked | as_expected x2 | as_expected x2 | 0.90 | 0.66 |
| pick, hand still over the spot | picked | as_expected x2 | as_expected x2 | 0.77 | 0.28 |
| pick whose BEFORE frame has the cube hidden by the parked hand | picked | otherwise x2 | otherwise x2 | 1.00 | 0.13 |
| miss (cube nudged ~6 px by a finger), hand aside | not picked | otherwise x2 | otherwise x2 | 0.82 | 0.68 |
| miss, hand over the spot | not picked | otherwise x2 | otherwise x2 | 0.78 | 0.63 |
| two BEFORE frames of separate trials | not picked | otherwise x2 | otherwise x2 | 0.37 | 1.26 |

14/16 per model, **no false `changed_as_expected`**; both misses were caught
by the VLM. The two "wrong" verdicts are the pair whose BEFORE frame does not
show the cube (0 red pixels: the parked hand hides it), so neither model could
see a pick; capture BEFORE with the object in view.

**The contrast ratio is a coarse veto, not a vacancy test.** On these small
boxes the misses scored 0.63-0.68 (the nudged cube left part of the box) and
one true pick 0.66, so no threshold separates them; here the VLM did all the
work and the region check neither vetoed nor was needed. Its use is the
first set's: a region that got busier (ratio well above 1: a hand covering
the spot, another object landing there) blocks a VLM "yes".

Azure OpenAI was not checked: its variables were not set on the porting
machine.

## Relation to other skills

- judging-step-with-before-now-vlm: plain BEFORE / NOW frames and a task-level
  status with recovery actions; this skill adds the diff heatmap and a pixel
  check on one region, and answers one question (did this change happen).
- detecting-disturbed-pieces-by-snapshot-diff: registration-based, per-item
  movement in mm for items that should NOT have moved; no VLM.
- verifying-grasps: the gripper side (jaw gap, held object near the hand);
  this skill is the source side. Use both for thin parts.

## Changes from RoboRSI

Adapted from `roborsi/embodied/skills/base/visual_diff/robotwin/policy.py`
(Apache-2.0): the panel, the HOT heatmap normalised to the frame's maximum,
the prompt and the CHANGED / MATCH parsing are upstream's.

- BEFORE is an observation from the graph, not a snapshot anchor cached in the
  process; AFTER is a fresh `robot.get_observation`, not a simulator render.
- The panel goes to the hub's `vlm.query` as an array; the system and user
  text are one prompt (the tool has no system prompt).
- The two answers become exits; a missing line routes `no_verdict` (upstream
  read it as NO).
- New: the source-region pixel share and contrast ratio, and the `uncertain`
  exit when they disagree with the VLM.
