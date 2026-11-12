---
name: inserting-a-held-tip-with-the-other-arm-assisting
description: Inserts the tip of an object one arm holds (a pen, a peg) into a
  close-fitting hole with the other arm's wrist camera as a side-view assistant.
  The holding arm hovers the tip over the estimated mouth, the other arm takes
  two side views 90 deg apart from 0.2 m (tool centres kept >= 0.15 m apart),
  the tip-to-hole offsets along the two image-horizontal axes give both lateral
  axes, the holding arm corrects (bounded), the camera arm goes back, the
  holding arm descends, and a hand that stops short means the tip is on the
  face. Use when a top-down estimate of the hole is off by up to about a pitch,
  and a side view can see the gap under the tip.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: manipulation
  tags: [bimanual, insertion, peg-in-hole, wrist-camera, active-perception]
  hub:
    stage: execute
    embodiments: [any]
    arms: 2
    version: 0.1.0
    tasks: [insertion, bimanual]
    provenance:
      authors: [runtime-si]
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
  required_inputs:
    hole_mouth: Vec3
    tip_offset_m: float
  produces_outputs:
    measured_offset_m: float
    correction_m: float
    min_tcp_separation_m: float
    shortfall_m: float
  exit_conditions:
    inserted: The hand reached the commanded depth (insert_depth under the mouth), within face_stop. With assist, after at most `rounds` view rounds and corrections.
    tip_on_face: The hand stopped over face_stop short of the commanded depth; the tip rests on the face, the hand holds there.
    not_in_view: A side view did not see the tip or the hole, or the views asked for more than max_correction in total; the camera arm went back and the holding arm hovers over the hole.
    arms_too_close: A view pose would put the tool centres under min_separation apart; the camera arm went back.
    unreachable: The holding arm could not reach the hover over the mouth or the correction, or the camera arm could not reach a view; also same arm / no second arm.
  canonical_scripts:
    - assist_insert: scripts/assist_insert.py
  examples:
    - title: Insert a pen held by arm 1 with arm 0's wrist camera assisting
      path: examples/canonical_subgraph.json
---

# Inserting a held tip with the other arm assisting

## Which assist (the YAM pens station)

The pens task places pens (16.5 mm) held by the right arm into the grey holder's 18.2 mm square holes. Three
ways the left arm could help were checked against the station's constraints (pens journal 2026-11-06 19:43,
SKILL_IDEAS B5):

| mode | reach | tool-centre separation (station keeps >= 0.15 m) | verdict |
|---|---|---|---|
| (1) left wrist camera, side views from 0.2 m | the camera can stand off on the left's own side (x <= 0.41); it never reaches into the grey holder | about 0.18-0.2 m by construction, refused under 0.15 m | **feasible**: this skill |
| (2) half-open fingers over the hole as a funnel | the left does not reach the grey holder (IK stops at x 0.41; the holes lie at x 0.40-0.46) | the funnel's tool centre lies under the right's held pen, 0.07-0.1 m apart | infeasible |
| (3) holding the holder steady | the grey holder's west wall (x about 0.395) is at the reach limit | the wall is 0.05-0.12 m from the right's tool centre over the hole | infeasible (B5: only far-side bracing, not reachable) |

The earlier left-arm observer (pens `put.observer`) was not recommended: 8/12 against 12/12, because placed
pens hide the back row's holes from the left. This skill keeps that limit. A view that does not see the tip
and the hole exits `not_in_view`, and the holding arm does not descend blind on the views.

## Steps

1. The holding arm moves the tip `hover_gap` over the estimated mouth, keeping its rotation.
2. Views: two horizontal directions from the mouth, +-45 deg about the camera arm's side. The camera stands
   `standoff` away at `elevation_deg` and looks at the mouth. A view whose tool centre would be under
   `min_separation` from the holding hand is refused.
3. `vision.tip_hole_offset` gives, per view, the tip's offset from the hole's axis along the view's
   image-horizontal axis. Two views give the xy offset; the holding arm moves by the opposite of it.
   It takes up to `rounds` rounds, stops under `tolerance`, and never goes past `max_correction` in total.
4. The camera arm goes back to where it started. The holding arm descends to `insert_depth` under the mouth,
   and ends `tip_on_face` when the hand stops more than `face_stop` short.

## Reuse

- The camera rotation (`looking_along`) is taken from inspecting-a-held-object-with-the-other-wrist-camera.
- Run checking-dual-arm-clearance on the view poses before this skill on a real station, because the
  tool-centre test does not see links.
- locating-a-hole-lattice-by-wide-search gives `hole_mouth` when the estimate is a pitch off. The views fix
  up to `max_correction` (15 mm, under one 20.2 mm pitch).
- holding-steady-while-other-arm-acts is for mode (3), which is infeasible here.

## Open

- The per-view measurement is a parameter, `offset_tool` (default `vision.tip_hole_offset`), and no hub tool
  provides it yet. It is not in `allowed_tools`, so a station connector has to supply it. A candidate
  implementation is `gemini-er.point` on the camera arm's frame (tip point, hole-mouth point), with the pixel
  difference scaled by range / fx.
- Twin check (sim_check peg harness, square 18.2 mm hole, 16.5 mm x 115 mm peg, offsets 4-11 mm): pending.
  The first runs ended `unreachable` at the second view: the camera arm's servo replied `timeout`. The skill
  now judges arrival by the hand's reading. The camera stood 182 mm from the holding hand's tool centre, and
  the arms' mesh clearance stayed 111-124 mm.
