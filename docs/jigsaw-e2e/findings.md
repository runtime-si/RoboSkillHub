# First end-to-end jigsaw chain in the YAM station twin: findings (2026-11-09)

One piece, picked off the mat and put into its slot by a chain of the hub's jigsaw skills. **It did not
complete: 0 of 8 trials seated.** What follows is what each stage did, where the chain breaks, and what to
fix first.

## Setup

- **Twin:** YAM station twin (left arm, wrist D405 at the real station's calibrated pose, `SIM_GRIP.md`
  grip fix on), GCP pool (rshgpu2 and rshgpu3).
- **Scene:** the seating check's jigsaw scene. A 21 x 24 mm, 2 mm thick generated piece carries a
  10 x 4 x 10 mm harness grip block: bare 2 mm edges do not hold in the twin's jaws. Its six neighbours
  (the rows north and south of the slot) are one fixed board; the cells west and east are empty.
- **Start:** the piece lies on the mat 90 mm east of its slot, turned 30, 135, -100 or -20 degrees.
- **Orchestrator:** [`e2e_orchestrator.py`](e2e_orchestrator.py), a harness script calling the skills'
  scripts in order. Simulator truth is used only to score each stage, except where marked TRUTH.

The chain:

1. `scanning-the-board-into-a-wrist-mosaic`: 8 x 8 cm round the staging spot at 2 px per mm, with a
   height map.
2. **Segment:** the height map over 1 mm, the largest component; the grip block over 6 mm.
3. `classifying-piece-sides-for-border-first`: the side types, then the piece's yaw as the turn whose side
   order matches the slot's (south, east, north, west = tab, blank, blank, blank).
4. `selecting-a-grip-clear-of-placed-neighbours`: the piece mask turned into its slot, the board cells as
   occupied cells, the station blades (15.5 x 3.9 mm).
5. **Pick by the grip block:** jaws squared to the block's long axis on joint 6, `robot.set_grip` with a
   10 mm width and a 3 mm squeeze, lift 30 mm.
6. `rotating-a-held-piece-within-wrist-limits`: turn by minus the estimated yaw; the drop spot is chosen if
   a regrasp is needed.
7. `seating-a-piece-by-wiggle-search`: the true slot (a localizer's job, not run here) and the piece's
   offset from the tool. The x, y offset is from the mosaic (the body centre minus the grip block, turned
   with the piece); the z offset is TRUTH. Yaw error is given as 0.
8. `verifying-piece-flushness-from-wrist-depth`: the wrist frame after the seat, the piece outline put at
   the slot, and the neighbours as cell rectangles.

`registering-assembled-board-to-reference` was not run: one placed piece among plain-coloured neighbours
gives it nothing to align on (`too_little_placed`); it needs textured pieces.

## Per stage (2 runs x 4 trials)

| stage | result | notes |
|---|---|---|
| scan | 8 / 8 | 100 % covered, 1 frame |
| segment | 8 / 8 | piece and grip block found in every trial |
| classify + yaw | 7 / 8 right | yaw error 0-1 degrees; in one trial a blank was read as a tab (tab, blank, tab, blank), giving a yaw 180 degrees off |
| grip choice | `no_clear_grip` 8 / 8 | with 15.5 mm blades no tangent grip clears the north and south neighbours; the grip block was used anyway |
| pick | 8 / 8 held | piece turned 0.4-3.7 degrees in the jaws on the pick |
| turn | 6 / 8 held | the same start (30 degrees, a -30 degree direct turn) lost the piece in both runs; the skill reported `turned` |
| seat | 0 / 6 | `jammed` every time (36-39 probes); x, y offset estimate 0.4-2.1 mm off, plus 2-3 degrees of yaw slip, plus one piece 180 degrees off |
| flush check | 6 / 6 right | it called the not-seated pieces `tilted` / `raised` (level 13.9-14.2 mm: the jammed piece was lifted back) |
| **completion** | **0 / 8** | |

## Where it breaks, worst first

1. **Seating with perception-grade errors.** The seat skill was checked with 0-3 mm and 0-5 degrees of
   error, but it jammed on every real chain trial. Here the errors combine:
   - an offset from the mosaic that is 0.4-2.1 mm off: the mask centroid gave 2.1 mm; the body centre was
     not much better because the piece also shifts on the pick;
   - 2-3 degrees of yaw slip on the pick, which the chain passes as 0.

   Fix next: measure the held piece with the wrist camera after the pick and after the turn (offset and yaw
   from its outline), and pass the measured yaw error to the seat. No hub skill does this yet: the gap is a
   "measuring-a-held-piece-from-the-wrist-camera" skill.
2. **Gripping a bare 2 mm piece is not modelled.** The twin's blades do not hold a 2 mm edge, so every
   jigsaw check grips a harness block. The grip choice is therefore not executed, and it says
   `no_clear_grip` for this slot with the real blades anyway (#142). The real arm needs either narrower
   pads or a different end effector (suction) for 2-sided slots; the twin needs a thin-part grip model
   before the grip skill can be checked in it.
3. **A short direct turn drops the piece** (both runs, the -30 degree turn) while the skill reports
   `turned`: it reads the hand's yaw, not the piece. Fix: a cheap "still held" check after each turn
   (`robot.get_gripper` width against the grip width, or the wrist view).
4. **Side typing on a clean mask** misread a blank once in 8 (both knobs of this piece are deep): one wrong
   side flips the yaw 180 degrees. Fix: route `uncertain` pieces to a closer look, and in the chain reject a
   yaw whose side match is under 4 / 4.
5. **After a regrasp the grip block turns about 45 degrees in the jaws** with the grip fix (see #160). It was
   not hit here: these turns needed no regrasp.

## What worked

The chain's perception front end: scan, segment, side types and yaw (7 / 8 within 1 degree), the pick by
the block, and direct turns that keep the piece. The flush check correctly refused every unseated piece.

## Not done

- **Multi-piece run:** a 3 x 3 puzzle of free, textured pieces, with localization against a reference
  picture and registration of the finished board. The twin's pieces are plain-coloured meshes.
- **A recipe:** waits for the chain to complete.
