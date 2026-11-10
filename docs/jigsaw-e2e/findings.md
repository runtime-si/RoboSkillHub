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


## Update, rounds 4-5 (2026-11-09 / 11-10)

Three of the blockers above have fixes on main:

- **#172, measuring-a-held-piece-from-the-wrist-camera.** The piece's mosaic mask is fitted to the wrist
  depth past the blades. The chain now measures the piece after the pick and after the turn, and gives the
  seat that offset (z included, no longer TRUTH) and the measured remaining yaw error.
- **#169, `dropped` route in rotating-a-held-piece-within-wrist-limits.** A gripper reading under 0.78 of the
  held reading after a turn routes `dropped` instead of `turned`.
- **#169, `expected_sides` / `mismatch` in classifying-piece-sides-for-border-first.** The yaw is given only
  when all four sides agree; on `mismatch` the chain re-scans once at 3 px per mm.

The orchestrator (`e2e_orchestrator.py`) wires all three in, and bounds the seat to 300 s (`max_time_s`). In
round 4 the seat ran over 20 minutes on a busy host.

**Round 4 rerun: 2 of 8 trials before the time box.**

| trial | classify | measured after the pick (x / y / z, yaw) | turn | measured after the turn | seat |
|---|---|---|---|---|---|
| 0 (30 deg) | 4 / 4, yaw err 0.0 | -0.17 / -0.48 / -0.09 mm, -0.18 deg | `dropped`, true: the piece fell (round 3 called it `turned`) | - | - |
| 1 (135 deg) | 4 / 4, yaw err 2.0 | +1.59 / +0.93 / -0.11 mm, -1.1 deg | `turned` | -0.45 / -2.13 / +0.33 mm, -1.1 deg | still running after 20 min (no time bound then) |

- The drop is now caught.
- The held piece is measured to within about 2 mm and 1.5 degrees; before, the error was 0.4-2.1 mm, and
  2-3 degrees of yaw slip went unknown.
- **Completion against the 0 / 8 baseline is not yet measured.**

**Round 5:** no GPU host could be started (the GCP pool's gcloud credentials had expired, and the AutoDL
fallbacks timed out), so the 8-trial rerun is still to do.

Run it from `~/sim-wt/jigsaw-e2e` once a host is up (`gcloud auth login`, then `~/yam_gap/rsh_pool.sh up 3`):

    GPU_SKIP_RSS=1 ~/yam_gap/gpu_run.sh --src ~/sim-wt/jigsaw-e2e --detach -- \
      bash -c "MAX_STEPS=12000000 python run_check.py --graph=graph_e2e --privileged"

**Expected next breakpoint: the seat.** The measurement after the turn is 2 mm off in one axis, and the seat
search was checked for 0-3 mm (#115). A 2 mm offset with a 1 degree yaw error is within its range but at
its edge.

**Still open:**

- The bare 2 mm grip is not modelled in the twin (blocker 2).
- The grip block twists in the jaws after a regrasp (#160): pivoting about the jaw axis, which the twin's
  grip fix does not cure (`SIM_GRIP.md`, condim 4/6 pending).


## Update, round 6 (2026-11-10)

**A bare 2 mm piece can be gripped in the twin now.** The twin's grip fix (`SIM_GRIP.md`) arrived after the
harness's grip block was introduced. In a re-test with the stock linear_4310 blades:

- the jaws close across the piece's 21 mm width with the blade tips 0.3 mm over the mat, `robot.set_grip`
  21-24 mm with a 3 mm squeeze;
- the piece is lifted 29.7-30.1 mm in 4 / 4 trials (start yaws 0, 30, -60 and 90 degrees), tilt at most
  0.7 degrees, turned 1.1-1.8 degrees.

So blocker 2 above (the bare grip) is gone, and the chain can drop the grip block. Harness:
`~/sim-wt/narrow-blade`, `graph_bare`.

**Narrow-blade fingertips** (1.8 mm thick, 4-6 mm wide). In the geometry evaluation of
`selecting-a-grip-clear-of-placed-neighbours`, a 2-sided slot gets a direct grip in 264-275 of 305 cases
(station blades: 15). A twin variant (box blades replacing the tip colliders) fails to load:
robosimstudio's station embodiment requires the tip meshes as colliders (`decompose_colliders` matched no
mesh collider), so the blade swap needs a gripper-spec change there. Not done.

**8-trial rerun with all fixes:** 2 of 8 finished within the round.

- **Trial 0 (30 degree start):** the turn routed `dropped` (true: the piece fell). Round 3 reported it as
  `turned`.
- **Trial 1 (135 degree start):** the seat ended `not_seated`; the piece was lifted back 15 mm, ending
  0.8 / 2.6 mm off the slot.

Completion so far: **0 / 2** (baseline 0 / 8). The worst breakpoint is still the seat: the seat search
finds no drop with the measured 2 mm, 1 degree residual.


## Update, round 7 (2026-11-10)

**Changes to the chain** (`e2e_orchestrator.py`):

- **Bare pieces.** The grip block is gone (`ASSETS=assets_bare`). The pick goes where
  `selecting-a-grip-clear-of-placed-neighbours` puts it: its `pick_grip_pixel`, `pick_closing_yaw_deg` and
  `grip_width_m`, with the board cells grown by 0.25 cell. The blade tips sit 0.3 mm over the mat.
- **Scan** at 3 px per mm, for a finer template for the held-piece fit.
- **Three seat variants**, 8 trials each over 5 GCP hosts: A, the default wiggle search; B, a ±4 mm and
  ±8 degree search; C, `approach="offset_push"`.

**Result: no trial reached the seat. The new worst breakpoint is the turn with a bare piece.**

| stage | result (A, B, C first halves: 12 trials; second halves not finished in the round) |
|---|---|
| scan, segment | 12 / 12 |
| classify (four sides must agree) | 10 / 12; the two `mismatch` trials stopped after one re-scan (a `partial` scan in one) |
| grip choice | `no_clear_grip` with the station blades every time (the least-bad grip, closing across the 20.8 mm width, was used) |
| pick, bare piece | 10 / 10 lifted 29.5-29.8 mm |
| measured after the pick | 0.3-0.9 mm, 1.1-1.3 degrees (sigma 0.1-0.4 mm, 0.9-1.8 degrees) |
| turn | **0 / 9 kept the piece**: flung out and flipped over (final tilt 180 degrees), landing 80-150 mm away |
| seat | not reached |

- **Why.** A 2 mm piece pinched by its two edges is held by edge contacts only. The grip fix stops sliding,
  not pivoting about the jaw axis (`SIM_GRIP.md`), and the wrist turn spins the piece out.
- **The drop is not detected.** The gripper read 0.189 before and after: with an edge grip the jaws hold
  their commanded width with or without the piece. `rotating-a-held-piece-within-wrist-limits`' `dropped`
  check (#169) needs the jaws to close further when the piece goes, so it misses this case.

**Fixes to try next, in order:**

1. Turn the piece before the pick instead of while holding it (with a bare edge grip). Put the hand at the
   needed yaw over the piece and grip it there; then only the slip has to be corrected, as a small turn.
   This replaces the held turn for jigsaw pieces.
2. Hold slower: limit the wrist's speed during held turns (the skill sends one joint target per turn).
3. Add a "still held" check to the rotation skill by re-measuring with
   `measuring-a-held-piece-from-the-wrist-camera` (`not_seen` means dropped), instead of the gripper reading.
4. Re-test with condim 4/6 (torsional friction) when it lands in the templates.

**Completion so far: 0 / 12** (baseline 0 / 8). Round 6's 2 trials also ended 0 / 2.


## Update, round 8 (2026-11-10)

**A held turn cannot be avoided by gripping at another yaw.** With one grasp the piece turns between pick
and place exactly as far as the hand does. Another hand yaw at the pick only changes which two points are
pinched. A put-down and regrasp does not help either: the piece turns only while held, and empty-hand turns
do not move it. So the turn has to happen with the piece held.

Two ways of making it were added to `rotating-a-held-piece-within-wrist-limits`: `carry_lift_m` (turn at
0.5 mm, the mat still carrying the piece) and `max_step_deg` (5-10 degree steps with a settle between). Each
was tried on bare pieces (seat variant A, 4 hosts). Only 3 trials reached the turn within the round:

| turn mode | trials at the turn | piece still held after it |
|---|---|---|
| on the mat, 10 degree steps | 2 | 0 (left on the mat, flat: the mat's friction holds it while the jaws twist) |
| at 3 cm, 5 degree steps | 1 | 0 (flung out and flipped, tilt 180 degrees) |

- **The cause.** An edge grip of a 2 mm piece has almost no hold about the jaw axis in the twin (condim 3).
  The twin's grip fix does not change this (`SIM_GRIP.md`).
- **Drop detection by the wrist camera** (re-measure, `not_seen`) is wired for the place step, but no trial
  got that far.
- **Completion: 0 / 3 new, 0 / 15 overall** since the fixes (baseline 0 / 8).

**Next:**

- condim 4/6 (torsional friction) on the pads and the piece;
- or a different hold: a pad on the piece's top (suction, or a pinch across the piece's top and bottom),
  rather than its edges;
- or turning the piece on the mat by pushing it (no grip) before the pick.


## Update, round 9 (2026-11-10)

**Torsion fix.** The jigsaw harness carried its own copy of the grip fix (impratio and solimp only). It now
calls the template's `sim_grip._pad_friction`, which gives the finger pads condim 4 with 0.004 m torsional
friction. The log confirms it: `pads condim 4, torsion 0.004 m ... on 200-218 geoms`.

**Held turn of a bare, edge-gripped piece, with the torsion fix:** bare 2 mm pieces picked at the
grip-selection skill's grip, lifted 3 cm and turned by `rotating-a-held-piece-within-wrist-limits`, on 3 hosts.

| turn mode | turns | kept | flung (piece upside down, tilt 180 degrees) |
|---|---|---|---|
| 5 degree steps (`max_step_deg`) | 2 | 0 | 2 |
| one joint move per turn | 3 | 0 | 3 |
| **all** | **5** | **0** | **5** |

The piece flips end over end about the line through the two contacts, which condim 4 should resist (that
line is the contacts' normal). The flip may come from something else:

- the joint-6 move itself (an acceleration jerk);
- the piece touching the blades as it hangs.

Not diagnosed in this round.

**Pushing to the yaw on the mat before the pick** (no held turn) was not built in the round.

**Completion: 0 / 5 this round, 0 / 20 since the fixes** (baseline 0 / 8).

**Next, in order:**

1. Trace one held turn (piece pose and contact forces each step) to see what starts the flip.
2. Turn by pushing the piece on the mat before the pick.
3. A hold on the piece's top instead of its edges.


## Update, round 10 (2026-11-10)

**Two-finger twist on the mat, with the pad torsion fix.** The bare piece is pinched by its edges at the
grip-selection skill's grip and lowered back to 0.5 mm over the mat. The wrist turns in 10 degree steps
(`carry_lift_m=0.0005`, `max_step_deg=10`), so the piece would end at the slot's yaw, picked, with no turn
in the air. Truth is the piece's yaw right after the twist, against the slot's yaw. 8 trials were launched on
3 hosts; within the round:

| trial (start yaw) | turn asked | piece yaw after the twist (target 0) | tilt | held after the lift |
|---|---|---|---|---|
| 0 (30 deg) | -28 deg | +101.8 deg | 0 | no |
| 1 (135 deg) | -138 deg | -55.0 deg | 0 | no |

The other trials stopped earlier: one at the side check (`mismatch`), one at the pick.

- The piece stays flat on the mat but does not follow the jaws. It slips and spins between them as the wrist
  turns, and ends 55-102 degrees off. The yaw is not controlled, so the twist cannot replace the held turn.
- **Press and rotate** (closed tips pressing the top off-centre, dragging round) was not built in the round.
  Expect it to slide the piece more than turn it: the mat's friction acts over the piece's whole area, the
  tips' over a small patch.

**Completion: 0 / 2 reached the place step; 0 / 20 overall since the fixes** (baseline 0 / 8).

**What this leaves.** Every way of turning a 2 mm piece held only by its edges has failed in the twin:

- turned in the air: flung, 0 / 14;
- twisted on the mat: slipped, 0 / 2.

The ways forward are a different hold (suction or a top pad, which a real jigsaw robot uses) or turning by
pushing with no grip. Both are new manipulation skills, not fixes to the chain.
