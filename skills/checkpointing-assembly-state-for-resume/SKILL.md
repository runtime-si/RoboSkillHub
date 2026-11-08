---
name: checkpointing-assembly-state-for-resume
description: Saves and restores the state of a long assembly run (a 550-piece
  jigsaw, a LEGO build, a domino layout) so a run stopped after hours resumes
  where it was. save_checkpoint writes one versioned JSON checkpoint
  atomically (temp file, fsync, rename) with a SHA-256 over its payload - task,
  step, placed pieces / bricks / items with cells and poses, the pending queue,
  backtracking-a-wrong-placement's ledger, the reference snapshot and items for
  detecting-disturbed-pieces-by-snapshot-diff, arms and what they hold,
  timestamps and an activity stream of what changed - keeping the newest N.
  resume_from_checkpoint loads the newest checkpoint that passes its hash
  (migrating older schema versions), compares it with a fresh observation
  through the snapshot-diff skill and, for bricks, verifying-build-against-model,
  and lists the items to re-verify before going on. Use after every few
  verified placements of a long run (save), and when a stopped run starts up
  again (resume); files and CPU only, no motion.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: recovery
  tags: [checkpoint, resume, persistence, long-horizon, jigsaw, lego, domino, recovery]
  hub:
    stage: recover
    embodiments: [any]
    version: 0.1.0
    tasks: [jigsaw, lego, assembly]
    provenance:
      authors: [runtime-si]
      code: https://github.com/splintered-reality/py_trees/blob/49bcffd069afa2b7bd9ca7b0e7312f5cbb3640bd/py_trees/blackboard.py
      upstream_license: BSD-3-Clause
gap:
  requires: {}
  allowed_tools:
    - robot.get_observation
    - colour.classify
  required_inputs:
    checkpoint_dir: string
  # save_checkpoint also needs task_id (string) and step_index (int). The state is wired from
  # a script node (gap has no list or open-dict type here): placements, bricks, items, pieces,
  # pending, in_progress, ledger / ledger_path / ledgers, reference_snapshot /
  # reference_snapshot_path / snapshot_now, reference_items, camera_name, grid, diff_args,
  # build, arms, extra. resume_from_checkpoint takes task_id, current, verify,
  # build_observed, expect_grid.
  produces_outputs:
    checkpoint_path: string
    task_id: string
    step_index: int
    seq: int
    sha256: string
    n_recheck: int
    n_skipped: int
    age_s: float
    state_json: string
  exit_conditions:
    saved: save_checkpoint wrote the checkpoint (whole or not at all) and removed the ones past `keep`; `checkpoint_path`, `seq`, `sha256`.
    save_failed: The write raised (disk full, no permission, an unstorable value); the earlier checkpoints are untouched.
    resumed: resume_from_checkpoint loaded a valid checkpoint and every placed item was cleared by a check against the fresh view (unchanged in the snapshot diff, matched by the build check); nothing held or half done.
    resumed_with_rechecks: A valid checkpoint was loaded but some items must be re-verified first; `recheck` lists them, surest problems first, each with its reasons (lost, held_when_stopped, missing, misplaced, wrong_colour, moved with its shift, appearance_changed, in_progress_when_stopped, ledger_after_checkpoint, unseated, extra, unexpected_object, occluded, unverified_at_save, placed_after_reference, not_checked).
    no_checkpoint: The directory has no checkpoint for the task, or none passes its hash and schema checks (`skipped` says why for each); start the task fresh.
    state_mismatch: The fresh view does not register onto the reference snapshot, at least a quarter (and 3) of the checked items moved, vanished or are wrong, or the grid differs from expect_grid; re-perceive the board instead of trusting the checkpoint.
  canonical_scripts:
    - save_checkpoint: scripts/save_checkpoint.py
    - resume_from_checkpoint: scripts/resume_from_checkpoint.py
  examples:
    - title: Resume a stopped assembly run and list what to re-verify
      path: examples/canonical_subgraph.json
    - title: Checkpoint a jigsaw run with a fresh overhead snapshot after a verified placement
      path: examples/saving_subgraph.json
---

# checkpointing-assembly-state-for-resume

A 550-piece jigsaw takes a robot 5-9 hours; a LEGO set or a domino field is
similar. Over that time something stops the run - a motor fault, a crash, the
end of the day. Graph-as-Policy's graph state lives in the process; this skill
keeps the task's state on disk so a new process can pick it up, and then asks
whether the world still looks like that state before building on it.

## How it composes

- **Save** after every verified step (or every few): the forward loop's
  `placements` (choosing-next-piece-and-slot / backtracking-a-wrong-placement:
  `{piece, col, row, score?, margin?, confidence?, verified?, locked?}`), the
  bricks placed so far (planning-brick-assembly-order's `sequence` entries:
  `{id, seq, step, cell {x, y, layer, rotation}, size, colour, arm, ...}`),
  generic `items` (`{id, pose?, cell?, mask? | polygon_px? | box?}` - dominoes,
  parts), the `pending` queue (choose_next's `queue` or the planner's
  remaining sequence), `in_progress` (the job started but not finished:
  `{piece | id, col, row, arm, phase}`), the ledger (`ledger_path` of
  backtracking-a-wrong-placement, read and embedded with its hash), the arms
  (`{id, joints?, ee_pose?, gripper?, holding?}`), `grid`, and a reference
  snapshot: `snapshot_now` (one `robot.get_observation`, cut to `camera_name`),
  `reference_snapshot` (an Observation or CameraFrame) or
  `reference_snapshot_path` (.npz with rgb / depth_m / K / pose, or an image;
  its arrays are copied in, so the file may go away). Take the snapshot only
  after a step passed its check; between snapshots the last one is carried
  over, so pieces placed since are known not to be in it.
- **Resume** once at start-up, before the arms move: `resumed` - go on with
  `pending`; `resumed_with_rechecks` - park the arms (returning-to-a-recorded-
  pose), look for `lost_items`, run the jigsaw / brick checks on the `recheck`
  ids (a `moved` piece: nudge it back or re-seat it; `missing` / `misplaced` /
  `unseated` bricks: verifying-build-against-model's repairs), then go on;
  `no_checkpoint` - start fresh; `state_mismatch` - re-perceive the board
  (localizer, build check) and save a new checkpoint. The returned
  `placements`, `bricks`, `pending`, `ledger_path`, `pieces` are in the
  formats the other skills take.

## save_checkpoint

1. **One file per save**, `<task>.<seq:08d>.ckpt.json` in `checkpoint_dir`:
   `{"schema", "schema_version": 3, "sha256", "payload"}`, the hash over the
   payload's canonical JSON (sorted keys, no spaces). Written to a temp file in
   the same directory, flushed, fsynced, renamed over the name, then the
   directory is fsynced: a reader sees the old set of files or the old set
   plus the whole new one.
2. **Arrays** (snapshot RGB / depth, item masks) go to `blobs/<sha256>.npy.z`
   (zlib-compressed .npy, named by the hash of the uncompressed bytes,
   written atomically before the checkpoint that names them); the payload
   holds `{"$blob", "dtype", "shape"}`. An unchanged snapshot is stored once.
   Arrays up to 64 numbers (a pose, K, joints) stay inline.
3. **Every save is the whole state** except the reference, which is carried
   from the previous checkpoint unless a new one is given (`carry_reference`).
   Jigsaw placements become snapshot-diff items by cell when `grid` is given
   (the diff skill's grid forms); items with a mask, polygon or box are used
   as they are; or pass `reference_items`.
4. **Activity stream** (py_trees' blackboard ActivityStream, adapted): the
   state is viewed as keys (`/placements/<piece>`, `/bricks/<id>`,
   `/arms/<id>`, `/ledgers/<name>`, `/pending`, `/step_index`, ...) and every
   key INITIALISED, WRITTEN or UNSET since the previous checkpoint is appended
   with its previous and current value and the step; the newest 500 are kept.
5. **Rotate**: the newest `keep` (5) checkpoints of the task stay; blobs no
   checkpoint in the directory references are deleted once 10 minutes old.

## resume_from_checkpoint

1. **Pick**: the task's files newest first (or every task's with no
   `task_id`); the first that parses, matches its hash, is not from a newer
   schema and has a task and step is used. Each one skipped is listed in
   `skipped` with the reason; leftover temp files of a killed save are listed
   in `stale_temp_files` and ignored.
2. **Migrate** older versions step by step: v1, a bare run state as the
   jigsaw / LEGO skills take it (`placements`, or a planner `sequence` with
   `placed` - the count already built -, `ledger` / `ledger_path`, `queue`,
   `snapshot_path`; no envelope, no hash); v2, the first envelope (`step`,
   `queue`, `arms` keyed by id, one `snapshot_path`, `saved_at` in unix
   seconds). A v1 file has no hash to check.
3. **Ledgers**: the file at the saved path is compared with the embedded copy.
   Same - fine. Missing or unreadable - the copy is written back
   (`restore_ledgers`). More events than the copy - backtracking decided after
   the checkpoint, its actions may be half done: the event's piece and every
   piece its actions name are rechecked (`ledger_after_checkpoint`), and the
   newer file is returned.
4. **What the stop left behind**: a piece an arm was `holding` is lost
   (`held_when_stopped`, `lost_items`); the `in_progress` job's piece is
   rechecked; a placement saved with `verified: false` too.
5. **Snapshot check**: one fresh `robot.get_observation` (or `current`) goes to
   detecting-disturbed-pieces-by-snapshot-diff with the stored reference and
   items (and `grid`, `diff_args`): `moved` / `lost` items are rechecked with
   the shift, rotation and depth change it measured, `occluded` ones too; an
   item it calls unchanged although 40 % of its pixels changed is rechecked
   as `appearance_changed` (a removed thin piece can still match the table
   texture under it); unexplained changes become `unexpected_object` entries; items removed from
   the state since the snapshot (backtracking lifted them) are expected to be
   gone and ignored. Unregistrable views, or a quarter (and 3) of the items
   moved, give `state_mismatch`.
6. **Build check** (bricks with a stored `build` - `plate_pose`,
   `plate_studs`, `system`, `camera_name`, `args`): the placed bricks are the
   target of verifying-build-against-model, seen through the fresh frame's
   depth or `build_observed` bricks; `missing`, `misplaced`, `wrong_colour`,
   `unseated`, `extra` and unverified bricks are rechecked; a quarter (and 3)
   wrong gives `state_mismatch`. Colour is off unless `build.check_colour`.
7. **Everything else** that no check cleared is rechecked:
   `placed_after_reference` when the snapshot was compared but the piece is
   newer, `not_checked` when no check could run (no reference, no
   observation, a damaged snapshot blob, the sibling skill not installed).
   `verify: none` skips the checks and returns the state as saved.

Both sibling skills are loaded from the skills installed beside this one (or
a `GAP_SKILLS_PATH` root); when one is missing its check is skipped and said
so in `detail`.

## Measured

All on the CPU (Intel Core Ultra 7 265KF, 20 cores), no robot, no GPU and no
simulator: nothing moves, so the simulation check does not apply.
`tests/bundles/_checkpoint_eval.py` reproduces every number below.

**Killed writers** (`--part kill`, 200 trials): a separate process saves a
550-piece jigsaw state (placements, the localizer records of all 550 pieces,
a 40-event ledger file, a new 640 x 480 RGB-D snapshot every save, keep 5) in
a loop and is SIGKILLed at a random moment after 1-3 saves. Resume returned a
valid checkpoint 200 / 200 times: the last acknowledged one 148 times, the one
being written (renamed but not yet acknowledged) 52 times, never an older
one, no file skipped; 20 kills left a temp file, listed and ignored; the
ledger compared `same` every time. SIGKILL does not lose the page cache, so
this tests the rename protocol, not power loss (see Limits).

**Damaged files** (`--part corrupt`, 400 trials): one random byte of the
newest checkpoint changed (200) or the file cut at a random length (200):
400 / 400 skipped with the reason and the previous checkpoint returned; no
damaged state was ever accepted. (A flip that produced invalid UTF-8 crashed
the reader before this run; fixed.)

**Disturbed boards** (`--part disturb`, seeds 100-119): the synthetic RGB-D
boards of detecting-disturbed-pieces-by-snapshot-diff's tests (15 jigsaw
pieces, 8 bricks or 8 standing dominoes; the overhead camera at 1 m or the
wrist camera at 25 cm; sensor noise per frame), saved with a reference
snapshot and the items' masks, resumed against a new render. 20 boards per
row; "found" = disturbed items in `recheck`; "extra" = other items rechecked
per board.

| family / camera | untouched: resumed | one moved 3-6 mm: found | one removed: found | two moved: found (mismatch) | sweep 40 % moved: state_mismatch | extra per board (mean / max) |
|---|---|---|---|---|---|---|
| jigsaw / overhead | 20/20 | 20/20 | 20/20 | 40/40 (0) | 20/20 | 0-1.1 / 3 |
| jigsaw / wrist | 20/20 | 20/20 | 20/20 | 40/40 (3) | 20/20 | 0-1.15 / 3 |
| bricks / overhead | 20/20 | 20/20 | 20/20 | 40/40 (0) | 20/20 | 0 / 0 |
| bricks / wrist | 20/20 | 20/20 | 20/20 | 40/40 (0) | 20/20 | 0 / 0 |
| dominoes / overhead | 20/20 | 20/20 | 20/20 | 40/40 (0) | 20/20 | 0 / 0 |
| dominoes / wrist | 20/20 | 20/20 | 20/20 | 40/40 (2) | 20/20 | 0-0.2 / 1 |

- The extra rechecks are the moved item's direct neighbours (the diff skill
  flags a slid jigsaw piece's neighbour whose footprint it overlaps); the
  sweep rows found 599 of 600 moved items.
- "Two moved" ended `state_mismatch` on 5 of 120 small boards: two moved plus
  one flagged neighbour is 3 of 8 dominoes (or 4 of 15 pieces), over the
  quarter. On a 550-piece board the fraction decides, not the count.
- Before the `appearance_changed` guard, 2 of 20 removed jigsaw pieces
  (overhead) resumed unflagged: the diff skill matched them to the table
  under them at 0.56-0.59 with 48-86 % of their pixels changed. The 0.4
  threshold was then set on separate seeds (0-19: unmoved items read at most
  0.21 of 3475); the table is the rerun on seeds 100-119.

**Brick builds** (`--part build`, 20 random 8-brick LEGO builds on a 12 x 12
plate, wrist D405 depth with the station noise model): untouched 20 / 20
`resumed`; a top brick removed 20 / 20 rechecked first as `missing`; a top
brick moved one stud 19 / 20 first as `misplaced` (1 as an `extra` brick
plus the brick `missing`), every one `resumed_with_rechecks`.

**Recorded station frames** (overhead Orbbec, RGB only, 5 sessions, the scene
static; not committed): 24 pixel-square items on the table, saved from one
frame's PNG through `reference_snapshot_path`, resumed against the next frame
(10 consecutive pairs and first-to-last per session): 55 / 55 `resumed`, no
recheck.

**Time** (`--part timing`, median of 15): save of the 550-piece state 36 ms
(a 498 kB file), 79 ms with a new 640 x 480 RGB-D snapshot, 51 ms with the
snapshot carried over; resume with `verify: none` 11 ms. A resume with the
snapshot check is the diff skill's time: 0.7-0.8 s for 15 items at
640 x 480 alone, median 1.0-1.8 s with 12 resumes running at once.

**Unit tests** (`tests/bundles/test_checkpointing_assembly_state_for_resume.py`,
31): the envelope and round trip, rotation, the activity stream, a crash
before the rename, torn / flipped / empty files, all files corrupt, writers
SIGKILLed at random moments (4 runs), a corrupt snapshot blob, blob
collection, a snapshot file deleted after saving, unstorable values; v1 bare
jigsaw state, v1 planner sequence, v2 envelope, a newer version skipped, an
edited v2 file, a migrated state saved again; held, in-progress and
unverified items; no observation; a grid mismatch; a newer ledger file and a
missing one; an untouched, knocked, emptied, swept and foreign board; pieces
placed and lifted after the snapshot; grid cells as items; bricks from depth
and from upstream perception; the examples' routes.

## Limits

- **Power loss is not tested.** The writes follow the POSIX recipe (fsync the
  file, rename, fsync the directory), and a SIGKILL never left a broken
  state, but a kill keeps the page cache; a power cut on a disk that lies
  about fsync can still lose the newest file (then the one before is used).
  `fsync: false` is faster and only safe against process crashes.
- **One writer per task.** Two processes saving the same `task_id` into one
  directory can take the same sequence number; the rename keeps one of them
  whole. Different tasks may share a directory.
- **Only as good as the checks.** Pieces placed after the last snapshot can
  only be listed (`placed_after_reference`), not verified; take a snapshot
  after each verified placement or batch. The snapshot check inherits the
  diff skill's limits (a 1 mm bump is under the overhead camera's noise,
  textureless items cannot have their move measured, the camera must not
  move by centimetres) and the build check the verify skill's (overhead LEGO
  seating is below the noise).
- **Arms are reported, not moved.** The skill treats a held piece as lost
  and says so; parking (returning-to-a-recorded-pose) and finding the piece
  are the graph's job. The arms' saved joints are not compared with now.
- **Synthetic boards and static station frames only**; no stopped robot run
  was resumed on hardware yet. The station frames are RGB only, without
  pieces.
- **Schema versions 1 and 2 are this format's own earlier drafts** and the
  bare run state the jigsaw / LEGO skills pass; the migration chain is there
  for the versions that come next.
- `MAX_DISTURBED_FRACTION` (a quarter) and `MIN_MISMATCH_ITEMS` (3) are
  design choices checked on the synthetic boards, not on robot runs.

## The constants, and why

Each constant in `scripts/` carries its source. In short: `KEEP` 5 (a torn
newest file leaves four; 0.5 MB each for 550 pieces); `MAX_ACTIVITY` 500
(py_trees' ActivityStream default); `INLINE_ARRAY_MAX` 64 (a pose, K or a
joint vector stays readable in the JSON); `BLOB_ZLIB_LEVEL` 1 (a full-frame
mask 307 kB -> 1-2 kB); `BLOB_GRACE_S` 600 s (another task's blobs are never
collected between its blob and checkpoint writes); `MAX_DISTURBED_FRACTION`
0.25 and `MIN_MISMATCH_ITEMS` 3 (a nudge moves 1-3 items, a sweep most;
see the table); `CHANGED_FRACTION_RECHECK` 0.4 (unmoved items read at most
0.21).

## Method

The candidate (jigsaw survey, "2 state / 8 long run") pointed at py_trees'
blackboard - a keyed store with an activity log - as the pattern; no
jigsaw code persists its state. This skill keeps that pattern: the state is
viewed as blackboard keys (`absolute_name` with the `/` separator) and the
difference between checkpoints is recorded as py_trees' activity items
(INITIALISED / WRITE / UNSET with previous and current values) in a bounded
stream - adapted from `py_trees/blackboard.py` (BSD-3-Clause, commit 49bcffd,
header in `scripts/checkpoint_store.py`, licence in `LICENSE.py_trees`), with
the stream dropping its oldest item where upstream's `ActivityStream.push`
pops its newest. The file format (hashed envelope, content-addressed blobs,
atomic rename, rotation, migration chain) and the resume logic (ledger
reconciliation, held and in-progress items, the two sibling checks and the
recheck list) are this skill's design. The candidate's append-only JSON-lines
step log is not implemented: every checkpoint is a whole state, and the
activity stream inside it says what changed.
