---
name: locating-a-hole-lattice-by-wide-search
description: Finds where a holder's lattice of round holes really is from one RGB-D frame (overhead or wrist),
  searching the prior's pose over +-1.5 pitch and +-15 deg, scoring each pose by holes matched (an open disc ringed
  by the holder's face) minus nodes contradicted (seen on solid face or on the table past the holder), and saying
  when it cannot tell - a second pose about a pitch away scoring nearly as well (ambiguous), the best pose on the
  window's edge, or too few holes. Use before inserting into a holder that may have been pushed or whose lattice
  may be fitted one row or column off, and whenever a local hole refinement (a few mm) would report "found" anyway.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [holder, hole-lattice, peg-in-hole, rgb-d, localization, wrist-camera, overhead-camera]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [insertion, pick-place]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools:
    - robot.get_observation
  # Inputs (wired from a script node; gap has no type for them): prior ({"nodes": [[x, y], ...]} or
  # {"centre", "yaw_deg", "pitch_m", "rows", "cols"}, world frame; required). Optional: camera (a CameraFrame dict: depth, intrinsics, pose or T_world_cam, rgb), camera_name,
  # hole_d (18.7 mm), face_z (else the densest height over the lattice), search_pitch (1.5), search_yaw_deg (15),
  # match_score (0.25), miss_score (= match_score), min_matched (4), ambiguous_ratio (0.85).
  # Also returned: pose {x, y, yaw_deg}, shift, nodes [[x, y], ...], second.
  produces_outputs:
    matched: int
    visible: int
    contradicted: int
    residual_mm: float
  exit_conditions:
    located: The best pose matches at least min_matched holes, lies inside the search window, and no pose about a pitch (or 8 deg) away scores within ambiguous_ratio of it; nodes are the lattice's hole centres there.
    ambiguous: A second pose about a pitch away scores within ambiguous_ratio of the best (second holds it) - typically a view that shows too little of the holder's border to tell which row or column is which.
    at_search_edge: The best pose lies on the window's border - the holder may be further than search_pitch pitches or search_yaw_deg away.
    not_found: Fewer than min_matched holes match anywhere in the window (no holder face in view, or no frame).
  canonical_scripts:
    - locate_lattice: scripts/locate_lattice.py
  examples:
    - title: Locate the lattice from the current camera
      path: examples/canonical_subgraph.json
---

# locating-a-hole-lattice-by-wide-search

A holder that slides when it is pushed, or a lattice fitted one column off,
is a whole pitch from where the graph believes. A refinement that searches a
few millimetres round the prior then settles on a local optimum and calls it
found. On a real YAM (pens task, `probe-insert-1` round 2, 2026-11-10) the
believed lattice was about one pitch off, the station's +-5 mm shift stopped
at its edge on every frame and still reported "found", and 18 insertions went
down into the air beside the holder: the wrist camera saw table under the
pen's foot, and the overhead camera showed the holes a column away.

## How

1. The frame's depth is lifted to the world. The hole face's height is the
   densest 1 mm band over the lattice (or `face_z`).
2. On a grid (1 mm, coarser for a far camera: 1.2 x the pixel's footprint),
   cells with a point within 4 mm of the face are *face*; cells seen 6 mm or
   more under it, or dark in RGB near it, are *open*.
3. A hole centred on a cell scores (face fraction of a ring at 1.25 r) minus
   (face fraction of a disc of 0.75 r), each computed once for every cell by
   convolution. The holder's body is the face closed over its holes (dilated
   then eroded by a hole's radius + 3 mm); a centre outside it scores -1.
   A node matches over `match_score` (0.25); a node in view that does not
   match contradicts the pose (solid face, the table, or just past the
   holder's edge, where the ring is half face and scores 0.2-0.3 like a poor
   hole).
4. **The border.** The lattice's next ring (each node one pitch further
   along either lattice direction, where no node is) must hold no hole: a
   ring cell scoring over 0.8 of the pose's own matched holes' median counts
   against it. A lattice one row or column off puts that ring onto the real
   outer row, so it can no longer tie with the truth in a full view.
5. Every pose of the prior moved by up to 1.5 pitch (1 mm steps) and turned
   by up to 15 deg (1 deg) is scored as matched - contradicted; the best is
   refined to 0.25 mm / 0.5 deg. The best pose more than 0.7 pitch or 8 deg
   away is the second hypothesis.

## Inputs

- `prior`: `{"nodes": [[x, y], ...]}` (world, metres), or `{"centre", "yaw_deg",
  "pitch_m", "rows", "cols"}`.
- `camera`: a camera dict (`depth` [m], `intrinsics` 3x3, `pose` {position,
  rotation wxyz} or `T_world_cam`, `rgb` optional), else `camera_name` of a
  fresh `robot.get_observation` (the first camera with depth by default).

## Outputs

`pose` {x, y, yaw_deg} of the lattice's centre, `shift` from the prior,
`nodes` (every hole centre there), `matched`, `contradicted`, `visible`, `of`,
`residual_mm`, `second`, `face_z`, `detail`.

## Measured

Offline on recorded real frames of the YAM pens station (wrist D405 at the
survey and over-hole poses, overhead camera; the prior is the lattice the run
believed; 103 frames from 30 runs, 0.1-0.5 s each on CPU):

| case | result |
|---|---|
| wrist survey frames, both of a run located | 14 of 14 runs agree between the two frames within 3 mm |
| prior moved by up to +-25 mm (1.2 pitch) and +-10 deg on 33 located reference frames, 132 trials | 124 located within 2.5 mm of the reference, 2 ambiguous, 6 located 2.6-3.5 mm off, **none a pitch off** |
| `probe-insert-1` round 2 (the lattice about a pitch off) | overhead and 6 wrist frames: `not_found` (1-2 holes match at any pose in the window); **none `located` at the wrong prior** (the station's local refinement said found, +2.0 / -1.0 mm); round 1 (lattice right): 5 of 5 wrist frames `located`, 11 of 11 holes, shift 2-5 mm |
| all 103 frames | located 53, not_found 44, ambiguous 6, at_search_edge 0 |
| synthetic full top view, prior one column off / one row off and turned 5 deg | `located` on the truth within 1.5 mm (tests) |

The overhead camera (2.5 mm a pixel) mostly routes `not_found`: the 18.7 mm
holes are 7 pixels across and the holder's closed outline is too coarse to
trust; its `located` answers before the border rule disagreed with the wrist
surveys by a pitch. Take the lattice from a wrist survey.

Close wrist views over one hole see 5-6 nodes and often route `not_found` or
`ambiguous`: that is the point - they cannot tell a pitch apart; take the
survey view or the overhead frame for the lattice, the close view for the
last millimetre.

## Limits

- Round holes in a flat face, the face seen from above (within ~45 deg).
- A holder with no border in view and a periodic lattice is genuinely
  ambiguous by a pitch; the skill says so rather than guessing.
- The overhead camera at 2.5 mm a pixel resolves holes of 18.7 mm only
  partly (6 of 12 matched on `survey_top`); use a wrist survey when it can.
