---
name: assigning-bricks-to-two-arms
description: Assigns every brick of a LEGO or DUPLO build sequence to an arm -
  which arm places it, from which stock brick, with which grasp (jaws across
  the brick's x or y, offset along it), and which other arm holds the
  structure while it is pressed - with APEX-MR's task-assignment MILP (each
  task one arm, one brick of its type and one grasp; a support arm that is
  never the placing arm and takes the same grasp side; stock bricks used once
  unless they are reusable stations; stock precedence; the per-arm load
  balanced in every window of n consecutive tasks, weight 1000, so the arms
  alternate), solved with SciPy's HiGHS. Costs are travel distances from each
  arm's base to the stock and on to the target, and grasps off the brick's
  centre cost more. Arm reach (a radius, and optionally an IK preview) and
  the fingers' room beside the target decide what is feasible. Use after
  planning-brick-assembly-order, when two arms build together and the graph
  needs per-brick arm, grasp and support duties before computing poses.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, assembly, bimanual, task-assignment, milp, scheduling, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      code: https://github.com/intelligent-control-lab/APEX-MR/blob/46a9448d9eaac5bd5973e0cd064623c5e7f5254e/src/python/task_assignment.py
      upstream_license: MIT
      paper: arXiv:2503.15836
gap:
  allowed_tools:
    - robot.solve_ik
  required_inputs:
    sequence_json: string
    plate_origin: Vec3
  # Or tasks = [{id, studs, position (bottom centre), yaw_deg, height?, needs_support?,
  # support_position?}] instead of sequence_json + plate_origin. Required either way:
  # arms = [{arm_id, base, reach_m}]. Optional: stock = [{id, studs, position, yaw_deg,
  # reusable?}] and stock_precedence = [[first, second], ...]. These are wired from a script
  # node or given as literals: gap has no list type here.
  produces_outputs:
    assignment_json: string
    by_arm_json: string
    n_tasks: int
    n_support: int
    objective: float
    imbalance: float
    travel: float
    solve_s: float
  exit_conditions:
    assignment_ready: Every task has an arm, a stock brick (when stock is given) and a grasp, and every task that needs support has a second arm; assignment_json lists them in task order.
    infeasible: Some task has no feasible choice (out of every arm's reach, no stock brick of its size, the jaws too narrow or placed neighbours blocking the fingers on both grasp sides), more arms than tasks, or the MILP has no solution; detail names the tasks.
    needs_support_arm: The bricks can all be placed, but a task that needs support has no second arm that reaches its support point (or the support duties cannot be fitted with the rest); detail names the tasks.
  canonical_scripts:
    - assign_arms: scripts/assign_arms.py
  examples:
    - title: Split a planned LEGO build between the two station arms
      path: examples/canonical_subgraph.json
---

# assigning-bricks-to-two-arms

Two arms can build faster than one, and some bricks need two: one arm holds
an overhang while the other presses a brick onto it. This skill decides, for
a fixed placement order, which arm does what for each brick, using APEX-MR's
task-assignment model.

## How it composes

- **Before it**: planning-brick-assembly-order gives the placement sequence
  (its `sequence_json`, with a plate origin), or world-pose tasks from
  computing-brick-pose-from-stud-grid. A task's `needs_support` comes from a
  stability analysis (APEX-MR used a force-balance model with Gurobi, which
  is not ported) or by hand for overhangs.
  planning-brick-assembly-order's own two-arm split is a split by x with
  reach; this skill replaces it when stock, grasps, support or balance
  matter.
- **After it**: per task, the arm's `arm_id`, `close_along` and
  `grasp_offset_m` go to the grasp and press skills (pressing-a-brick-onto-studs
  on that arm). `support_arm_id` tells holding-steady-while-other-arm-acts
  which arm holds. checking-dual-arm-clearance should still gate
  simultaneous motions.

## The model (APEX-MR's MILP)

Binary `x[arm, task, stock brick, grasp]` and `y[arm, task, side]`, integer
window loads `z[arm, window]` with their max and min:

- each task: exactly one `x`;
- an arm does not both place and support one task;
- a task that needs support has exactly one `y`, and on the placing grasp's
  side (APEX-MR's support pose depends on the press side);
- a stock brick that is not `reusable` (a station) is used at most once;
- stock precedence `[a, b]`: a is used no later than b, using APEX-MR's
  `(m - j)` weights (which also means b is used only if a is);
- every window of n consecutive tasks (n = number of arms): each arm's load
  (placements + supports) is z; minimise travel + offsets + support travel
  + 1000 x sum(z_max - z_min). The balance term, APEX-MR's `lambda_balance`,
  makes the arms take turns, which is what lets them run asynchronously.

APEX-MR filled its cost tensors from its MoveIt planner's CSVs (planning
times, a large number when infeasible). Here they come from geometry:

- **travel**: base to the stock brick, then to the target (no stock: twice
  base to target), plus `offset_cost` (1 cm) per pitch of grasp offset;
- **feasible**: the target and the stock brick are within the arm's
  `reach_m` (horizontal), and with `check_ik` `robot.solve_ik` accepts a
  top-down hover `hover` (5 cm) over both with that grasp's yaw (`arm_id`
  passed when nonzero);
- **grasps**: jaws closing along the brick's x or y (APEX-MR's two press
  sides) if that span plus 4 mm fits the 94 mm jaws, times up to
  `num_offsets` (7, APEX-MR's) offsets along the other axis in half-pitch
  steps from the centre (over and between the stud columns) while the 20 mm
  finger stays on the brick (a DUPLO 2x4 across its short side: -16, -8, 0,
  +8, +16 mm; a LEGO 2x4: -4, 0, +4 mm), nearest the centre first. A grasp is dropped when a brick placed earlier
  at the target's height stands where a 10 x 20 mm finger (+3 mm) goes down
  beside it (planning-brick-assembly-order's finger boxes);
- **support**: distance from the base to `support_position` (default: one
  brick height under the target), if reachable.

Infeasible combinations get no variable at all (APEX-MR gives them a
1,000,000 cost and reports "Infeasible" when the optimum exceeds it). If the
full model has no solution but the model without support does,
`needs_support_arm` names the support points no second arm reaches.

## Inputs

- `arms` (required): `[{arm_id, base, reach_m}]`.
- `tasks`, or `sequence_json` + `plate_origin` (world position of cell (0, 0)'s
  corner on the plate top) + `plate_yaw_deg`.
- `stock`, `stock_precedence`; `system` (`duplo` / `lego`; pitch and
  heights).
- `num_offsets`, `balance_weight`, `offset_cost`, `jaw_opening`,
  `jaw_margin`, `finger_thickness`, `finger_width`, `finger_margin`,
  `check_ik`, `hover`, `time_limit_s` (30).

## Outputs

`route`, `detail`, `assignment_json` (per task: `task`, `arm_id`, `stock`,
`close_along` (`x` / `y` of the brick), `grasp_offset_m`, `grasp_yaw_deg`,
`support_arm_id`), `by_arm_json` (arm -> task ids), `n_tasks`, `n_support`,
`objective`, `imbalance` (sum over windows of z_max - z_min), `travel` (m),
`solve_s`.

## Checked

No arm motion (IK previews only), so no simulation check is needed. Tests
(`tests/bundles/test_assigning_bricks_to_two_arms.py`) cover alternation in
the windows, reach, a support arm on the same side and never the placing one,
`needs_support_arm`, flush neighbours forcing the other grasp side and
walling a brick in, narrow jaws, centred offsets, single-use and reusable
stock, size matching (a 4x2 brick for a 2x4 task), stock precedence, an IK
preview ruling out an arm, sequences from planning-brick-assembly-order, and
**the MILP's objective against brute force** (every arm choice enumerated
for 4 tasks, 5 random instances: equal).

**APEX-MR's own assembly tasks** (config/lego_tasks/assembly_tasks, copied
to tests/bundles/data/apex_mr). Each instance uses APEX-MR's plate
calibration (taken as the 48 x 48 plate's centre) and its two GP4 bases
0.88 m apart, a 0.55 m reach per arm, and one reusable station per brick
type in front of each arm:

| model | tasks | need support | route (YAM 20 mm fingers) | placed by arm 0 / 1 | arm changes between consecutive tasks | window imbalance | travel m | time s |
|---|---|---|---|---|---|---|---|---|
| cliff | 12 | 3 | assignment_ready | 5 / 7 | 9 of 11 | 1 | 9.2 | 0.15 |
| stairs_rotated | 14 | 4 | assignment_ready | 5 / 9 | 9 of 13 | 1 | 10.4 | < 0.01 |
| S | 13 | 6 | assignment_ready | 4 / 9 | 7 of 12 | 1 | 9.1 | < 0.01 |
| faucet | 14 | 3 | assignment_ready | 7 / 7 | 13 of 13 | 6 | 10.0 | < 0.01 |
| guitar | 24 | 0 | assignment_ready | 12 / 12 | 23 of 23 | 0 | 16.5 | < 0.01 |
| R | 21 | 0 | assignment_ready | 11 / 10 | 20 of 20 | 0 | 15.4 | < 0.01 |
| bridge | 38 | 18 | assignment_ready | 10 / 28 (arm 0 supports the 18 overhangs) | 20 of 37 | 1 | 27.8 | 0.01 |
| fish_high | 29 | 12 | infeasible: tasks 12 and 17 | - | - | - | - | < 0.01 |
| fish_high, 6 mm finger | 29 | 12 | assignment_ready | 11 / 18 | 21 of 28 | 12 | 20.9 | 0.01 |
| rss | 47 | 12 | assignment_ready | 18 / 29 | 34 of 46 | 10 | 33.2 | 0.01 |

Every supported task got the other arm, on the placing grasp's side. Where
nothing needs support, the arms alternate on every task (guitar, R: zero
imbalance). With support, the supporting arm's load counts in the windows,
so the arm that supports places less (bridge). Every grasp came out centred.
fish_high's tasks 12 and 17 are LEGO 2x4s with flush neighbours along a
whole face. A 20 mm YAM finger on the free faces overhangs the 16 mm brick
into a neighbour, so a parallel gripper cannot place them in APEX-MR's
order; APEX-MR's own end-of-arm tool needs no finger room (a 6 mm finger
solves it). planning-brick-assembly-order's finger check finds the same kind
of brick.

## Limits

- **Costs are distances, not plans**: no collision-checked motion, no timing.
  APEX-MR's planner costs encode both. With `check_ik` each pose is only
  IK-previewed; checking-dual-arm-clearance (or a planner) still gates the
  motions.
- **Support need is an input**: APEX-MR's stability analysis (Gurobi) is not
  ported. Overhangs must be flagged by the caller.
- **The order is fixed**: like APEX-MR's assignment, the task order is given
  (by planning-brick-assembly-order). The temporal plan graph that lets the
  arms run asynchronously (APEX-MR's TPG / ADG) is not part of this skill.
- **Grasp sides only on the brick's own axes**, offsets in half-pitch steps.

## The constants, and why

- **`LAMBDA_BALANCE = 1000`**, **`NUM_OFFSETS = 7`**: APEX-MR's
  `lambda_balance` and `num_offsets`.
- **`OFFSET_COST = 0.01 m` per pitch**: chosen; centred grasps first unless a
  centred one is blocked.
- **`JAW_OPENING = 0.094`, `JAW_MARGIN = 0.004`, `FINGER_* = 0.010 / 0.020 /
  0.003`**: planning-brick-assembly-order's YAM linear_4310 values.
- **`SAME_LAYER_FRACTION = 0.5`**: a brick whose bottom is within half a
  brick height of the target's is beside it, where the fingers go.

## Provenance

The MILP is APEX-MR's `TaskAssignmentSolver.solve_problem` (MIT; Huang et
al., "APEX-MR", RSS 2025, arXiv:2503.15836), solved with SciPy's HiGHS
instead of PuLP / CBC. The cost tensors are built in the skill from
geometry instead of the MoveIt planner's CSVs. The script's header lists
the changes; licence in LICENSE.apexmr.
