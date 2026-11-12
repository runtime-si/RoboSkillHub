---
name: inserting-a-held-pen-into-a-lattice-hole
description: Put a held pen, peg or rod into one hole of a holder with a
  lattice of holes, a millimetre of play, from the cameras alone - locate
  the lattice by a wide search (rescan once when the view is ambiguous or
  the holder is past the window), hover over the chosen hole, coarse-align
  the tip by a VLM's direction answers, fine-align it from wrist depth, then
  plunge with a spiral fallback and confirm with a probe and a camera. Hands
  off to the other arm's side views when the depth servo cannot see the
  tip or the hole, and relocates the lattice when the tip turns out to be
  over the table. Use when a held part's foot keeps landing a few
  millimetres off on the face beside its hole (the YAM pens task) and no
  force sensor is available.
license: Apache-2.0
metadata:
  hub:
    version: 0.1.0
    embodiments: [any]
    tuned_on: [yam_i2rt]
    tasks: [insertion, pick-place]
    tier: camera_only
recipe:
  inputs:
    lattice_prior: string
    slot: string
    tip_offset_m: float
    object_description: string
  start: locate
  nodes:
    locate:
      skill: locating-a-hole-lattice-by-wide-search
      note: From the wrist survey pose (the overhead camera at 2.5 mm a pixel mostly answers not_found). The chosen slot's node gives hole_xy and face_z for the rest.
    rescan:
      skill: locating-a-hole-lattice-by-wide-search
      note: A second, wider look - raise the wrist so the holder's border is in view, search_pitch 2.5; ambiguous means a view that cannot tell a pitch apart, not a bad lattice.
    hover:
      skill: moving-to-pose
      note: The tool centre over the slot's hole_xy, the tip about 10 mm above face_z, approach straight down.
    coarse:
      skill: aligning-a-held-tip-by-vlm-relative-nudges
      note: provider azure (vlm.query). Optional and slow (about 60-80 s); every exit goes on to the depth servo, which reads the hole itself.
    fine:
      skill: aligning-a-held-tip-over-a-hole-by-visual-servoing
      note: hole_xy and face_z from the lattice; pass table_z so a tip over the table routes not_over_holder.
    relocate:
      skill: locating-a-hole-lattice-by-wide-search
      note: The tip stood over the table - the lattice is off (a real pens run aimed about a pitch off). Locate again from a raised wrist survey; no second relocation.
    assist:
      skill: inserting-a-held-tip-with-the-other-arm-assisting
      note: The other arm's wrist camera looks from the side (0.2 m off, tool centres kept 0.15 m apart) and the holding arm corrects and descends itself.
    probe:
      skill: verifying-insertion-by-probe-push
      note: After the assisted descent the hand still holds the part; a short push says whether it is home.
    plunge:
      skill: inserting-a-held-peg-top-down
      note: Straight down at the aligned spot, with its own spiral (and tilt-then-upright) at the mouth; releases and checks the part's top with the closed fingers.
    confirm:
      skill: verifying-placement
      note: One wrist RGB-D view after the release - the part's visible top inside the hole's footprint and below where a part standing on the face would put it.
    done: {end: success}
    abort: {end: failure, recovery: [robot.go_to_pose]}
  routes:
    locate: {located: hover, ambiguous: rescan, at_search_edge: rescan, not_found: rescan}
    rescan: {located: hover, ambiguous: abort, at_search_edge: abort, not_found: abort}
    hover: {succeeded: coarse, unverified: coarse, failed: assist, rejected: abort}
    coarse: {centred: fine, not_converging: fine, no_verdict: fine}
    fine: {aligned: plunge, not_converging: assist, tip_not_seen: assist, hole_not_seen: assist, not_over_holder: relocate}
    relocate: {located: assist, ambiguous: abort, at_search_edge: abort, not_found: abort}
    assist: {inserted: probe, tip_on_face: plunge, not_in_view: plunge, arms_too_close: plunge, unreachable: abort}
    probe: {seated: done, not_seated: plunge, blocked: confirm}
    plunge: {inserted: confirm, unverified: confirm, jammed: confirm, slipped_in_grip: confirm, missed_hole: abort, unreachable: abort}
    confirm: {verified: done, not_placed: abort}
---

# Inserting a held pen into a lattice hole, cameras only

The pens task on the real YAM kept landing the pen's foot 4-11 mm off on the
holder's face beside an 18.2 mm hole (16.5 mm pen, 0.85 mm radial play).
Three things were wrong at once, and each stage below answers one of them:
the believed lattice was sometimes a pitch off (the locator), the held
part's foot was several millimetres off the tool axis (the two aligners),
and nothing stopped a blind descent (the plunge's spiral and the checks).

## Why this order

1. **Lattice first, from a wrist survey.** Every later stage takes the
   hole's centre and face height from it. `ambiguous`, `at_search_edge` and
   `not_found` mean the view cannot settle the pose, so one wider rescan is
   tried, then the graph stops instead of aiming at a guess.
2. **Coarse by direction answers, fine by depth.** The VLM is never asked
   where the hole is (absolute pointing was 14-21 mm off), only which way to
   move; it brings a 2-10 mm offset to about 1-2 mm. The depth servo reads
   the foot and the hole in one wrist frame and moves the rest. The coarse
   step never ends the graph: whatever it answers, the servo reads again.
3. **The other arm when depth cannot see.** `tip_not_seen`, `hole_not_seen`
   and `not_converging` hand over to side views from the other arm's wrist,
   which see the tip against the hole's mouth where the top-down depth does
   not reach (a 30 mm deep hole past a 16.5 mm pen).
4. **`not_over_holder` is a lattice error, not an alignment error.** It goes
   back to the locator once, then to the assisted insertion at the relocated
   hole.
5. **No loops.** Every retry is a distinct node with its own failure exits,
   so the graph always ends.

## Expected rates (from each component's own numbers; not run end to end)

| stage | measured | where |
|---|---|---|
| locate (wrist survey) | real frames: 14/14 runs agree within 3 mm; prior moved up to 1.2 pitch: 124/132 within 2.5 mm, 2 ambiguous, none a pitch off; round with the lattice a pitch off: `not_found`, never `located` wrong | #233 |
| coarse (VLM nudges, Azure) | synthetic close-ups, 10 cases 2-10 mm: final median 0.99 mm, 9/10 `centred`, 5/10 under 1 mm, 1 walked off (now capped at 15 mm) | #240 (draft) |
| fine (depth servo) | twin: hole centre within 0.1-0.3 mm, foot about 1 mm biased; servo runs often `hole_not_seen` (round 3: 11/11), 0/6 inserted after `aligned` in round 2 | #230 (draft) |
| assist (other arm) | station feasibility checked (separation, reach); twin check pending; the older observer was 8/12 vs 12/12 | #239 (draft) |
| plunge | spiral + tilt-then-upright at the mouth, slip check by the closed fingers | inserting-a-held-peg-top-down |

Reading these together: the locator should be right or say so (at most a
few per cent of tries end at `rescan` and then `abort`). The coarse step
should hand the servo a tip within 1-2 mm in about 9 of 10 tries. The servo
then decides the outcome. As it stands it often routes `hole_not_seen`, so
expect most tries to go to the assisted insertion, whose success rate is not
measured yet. The plunge's spiral covers the last 1-2 mm. **An end-to-end
success rate is not known**; check it in the station twin before a real run.

## Before running

- `tip_offset_m` from measuring-a-held-tip-offset-by-contact-descent after
  the grasp (the pens slipped in the jaws).
- The coarse step needs the `vlm` tool with `provider: azure` (or a Gemini
  model through `vertex` / `openrouter`, which did worse: 0-2/10 under 1 mm).
- The assisted insertion needs a second arm whose wrist camera can stand
  0.2 m to the side; on one arm, route `assist` exits to `plunge` / `abort`.
