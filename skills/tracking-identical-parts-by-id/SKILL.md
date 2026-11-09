---
name: tracking-identical-parts-by-id
description: Keeps a stable id for each of many look-alike parts (dominoes,
  same-colour bricks, jigsaw pieces, cubes) across overhead frames while the
  arm covers and uncovers them - ByteTrack (Kalman box prediction, two-stage
  IoU matching that also uses low-score boxes) over a detector's boxes, one
  frame per call with the tracker state carried by the graph; reports parts
  that went out of sight and new ids born where a lost part was (a likely id
  switch). CPU only. Use when a graph must know which of several identical
  parts is which over time (checkpointing, backtracking, disturbance checks)
  and a per-frame detector alone renumbers them.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, tracking, multi-object, identity, occlusion, bytetrack]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [sorting, pick-place, domino, jigsaw, lego]
    provenance:
      authors: [Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang]
      code: https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff/yolox/tracker/byte_tracker.py
      upstream_license: MIT
      paper: https://arxiv.org/abs/2110.06864
gap:
  allowed_tools:
    - robot.get_observation
    - owlv2.detect
    - grounding-dino.detect
  required_inputs:
    prompts: string
  # Optional inputs (gap has no list / open-dict type here, so they are wired from a script
  # node): state (the previous call's `state`; none starts), detections ([{box, score,
  # label}] instead of running a detector), detector, detector_args, observation, camera,
  # track_thresh, low_thresh, match_thresh, fuse_score, track_buffer, frame_rate,
  # freeze_lost, use_labels, switch_radius.
  produces_outputs:
    n_tracked: int
    n_lost: int
    n_detections: int
    reason: string
  exit_conditions:
    tracks_updated: Every track that was in view is matched (or new parts started tracks); `tracks` lists id, box, label, state (tracked / lost) and frames lost for every live part.
    track_lost: At least one tracked part had no box this frame (covered by the arm, picked, or missed by the detector); its id stays in `tracks` as lost at its last box for track_buffer frames and `lost_ids` names it.
    id_switch_suspected: A new track started within switch_radius (2.5) part sizes of where a lost part of the same label and size was last seen - probably that part, moved while covered; `suspected_switches` pairs new_id with lost_id. Decide which (re-identify, or accept the new id).
    no_frame: The camera had no RGB or the detector raised; nothing was tracked and `state` is returned unchanged.
  canonical_scripts:
    - track_parts: scripts/track_parts.py
  examples:
    - title: Track the parts in a fresh overhead frame (loop it with state wired back in)
      path: examples/canonical_subgraph.json
---

# tracking-identical-parts-by-id

Dominoes, same-colour bricks and puzzle pieces look alike. A detector gives
boxes for each frame but no identity, and its order changes from frame to
frame. This skill keeps an id per part with ByteTrack. It predicts each box
with a constant-velocity Kalman filter, then matches tracks to this frame's
boxes by IoU twice: once with the confident boxes and once with the
low-score boxes that are left. A partly covered part scores low, so the
second pass keeps its id. A part that is not seen stays "lost" at its last
box for `track_buffer` frames and gets its id back when it reappears.

```text
detect + track (this skill) --state--> next call (same camera) --state--> ...
        |-- track_lost: a part went out of sight (arm over it, picked, missed)
        |-- id_switch_suspected: a "new" part appeared where a lost one was
```

## How a graph uses it

Call it once per frame (after each move, or in a perception loop) and feed
`state` back in. With `prompts` it takes a fresh `robot.get_observation`,
chooses the external camera (or `camera`), and runs `detector` (default
`owlv2.detect`, threshold 0.05; `grounding-dino.detect` works too). If you
already have boxes (sam3 masks, a colour segmenter, SAHI tiles), pass
`detections` instead. Boxes with different `label`s never share an id.
Keep the same camera for every call: ids are pixel tracks.

The outputs are `tracks` (`id`, `box` [x1, y1, x2, y2], `label`, `score`,
`state`, `frames_seen`, `frames_lost`), plus `new_ids`, `lost_ids`,
`refound_ids`, `removed_ids`, `suspected_switches`, `n_tracked`, `n_lost`,
`n_detections` and `reason`.

## Changes from ByteTrack

This is a numpy-only port. It was checked against upstream BYTETracker
(with `lap` and an IoU without cython_bbox's +1 pixel) on 40 random
sequences, 4800 frames. With `freeze_lost=False` and `fuse_score=True`, the
same ids and boxes came out on every frame.

- **Assignment:** scipy's `linear_sum_assignment` on the extended matrix
  that `lap.lapjv(extend_cost=True, cost_limit)` builds. It gave the same
  matches on 2500 random matrices. Boxes use continuous pixels, without the
  +1 that cython_bbox adds.
- **Kalman filter:** written from its textbook form, with upstream's state
  (centre, aspect, height and their velocities) and noise weights.
  Upstream's `kalman_filter.py` comes from deep_sort and is not copied.
- **State:** a JSON-able dict that the graph carries. Ids count per state.
  Lost tracks are also returned, at their last box.
- **New behaviour, with defaults for parts on a table and the hub's detectors**
  (upstream values in brackets):
  - `freeze_lost`: a covered part keeps its last position. Upstream zeroes
    only the height velocity, so a lost track drifts.
  - Labels gate the matching.
  - `track_thresh` 0.1 [0.5] and `low_thresh` 0.05 [0.1].
  - `fuse_score` off [on; upstream's MOT20 setting turns it off]. OWLv2
    scores of 0.1-0.3 times IoU never get under the match limit, so with
    fusion on nothing matches.
  - `track_buffer` 150 frames [30].
- **id_switch_suspected** is new. With identical parts a tracker cannot
  tell whether a part was moved under the arm or a new part appeared, so
  the skill reports the pair instead of guessing.

## Measured (real station clip with pasted identical parts)

There is no real station video with many identical parts, so the clip is
half real:
- **Real:** the overhead recording of the 2026-11-03 cube pick-and-place
  (`real/recordings/20261103_082451`), 50 s at 5 fps (250 frames), with the
  real arm moving through the scene.
- **Pasted:** 8 copies of the real cube cut from the first frame (18 x 14
  px) on the table along the arm's path. A copy is drawn only where the
  frame still matches the background, so the real arm covers it. Part 1
  slides 40 px during t = 20-24 s. Part 4 is covered for 190 frames and
  part 7 for 150.
- **clip2:** part 7 is also bumped 25 px while covered.

Boxes come from the hub's `owlv2.detect` ("small black block", threshold
0.05) on rshgpu1, about 24 per frame, including the real cube and boxes on
the arm. Detector recall on visible pasted parts is 0.97 at score >= 0.05,
0.96 at 0.1, 0.77 at 0.2 and 0.005 at 0.3.

Scoring: each pasted part is matched to an output box with IoU >= 0.3 in
every frame where it is at least half visible. "Identity acc" is the share
of those frames in which the part carries its most frequent id. "Tracks per
frame" counts output tracks; the scene has 9 real parts.

| 5 fps | id switches | ids per part | identity acc | coverage | tracks / frame |
| --- | --- | --- | --- | --- | --- |
| clip, this skill (defaults) | 0 | 1.00 | 0.965 | 0.965 | 9.6 |
| clip, same but upstream buffer 30, no freeze | 3 | 1.38 | 0.898 | 0.929 | 9.0 |
| clip, centre Hungarian >= 0.1, 12 px gate, no memory | 10 | 2.25 | 0.864 | 0.959 | 14.0 |
| clip, same, ids remembered 150 frames | 0 | 1.00 | 0.959 | 0.959 | 14.0 |
| clip, upstream defaults (0.5, fused scores) | - | - | 0.002 | 0.005 | 1.2 |
| clip2, this skill | 1 | 1.12 | 0.928 | 0.964 | 9.7 |
| clip2, centre Hungarian with memory | 1 | 1.12 | 0.921 | 0.956 | 14.1 |

What the clips show:

- **Routes.** On clip the skill routed 237 `tracks_updated`, 13
  `track_lost` and **0** `id_switch_suspected`. On clip2 it raised **one**
  `id_switch_suspected`: in the frame where the bumped part reappeared (new
  id 14 against lost id 8, 25.2 px apart). A centre matcher also gives that
  part a new id, but nothing tells the graph.
- **Against a centre matcher with memory** (the hand-rolled alternative; the
  hub has nothing that tracks many ids), ids are as stable. The skill keeps
  a third fewer spurious tracks, because a track needs a 0.2 box to start.
  Without memory, the matcher renumbers covered parts 10-14 times.
- **At 1 fps** (every 5th frame) the sliding part moved 10 px per frame,
  more than half its 18 px width. The IoU match then broke, so its id
  changed once on clip and twice on clip2. Each break was reported as
  `id_switch_suspected`, alongside 3-4 suspicions that did not end in a
  switch. The centre matcher with a 12 px gate did not break. **Track
  small parts often enough that they move less than half their size
  between calls.**
- **Scale the thresholds to your detector.** Upstream's 0.5 and fused
  scores assume YOLOX-like scores and tracked nothing here. With
  `track_thresh` 0.15 coverage fell to 0.81.

The clip builder, the detections and the evaluation scripts are in
`~/sim-wt/bytetrack-check` on the porting machine. They are not part of the
bundle.

## Licence

Adapted from ByteTrack (MIT, Copyright (c) 2021 Yifu Zhang); see the header
of `scripts/bytetrack.py` and NOTICE. No code from deep_sort's Kalman
filter (GPL-3.0) is used.
