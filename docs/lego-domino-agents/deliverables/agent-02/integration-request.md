# Agent 02 → Agent 05: integration request (domino module)

Branch `ld-domino` (rebased on `origin/workcell-runtime` at 58dcc23). Module
`src/roboskillhub/tasks/domino/`, tests `tests/tasks/test_domino_*`, examples
`examples/tasks/domino/`. Nothing outside those paths is changed.

## 1. Register (only you edit shared files)

`roboskillhub.tasks.domino.SKILLS` (4 entries, all `version 0.1.0`, readiness `unit-tested`):

| skill_id | kind | requires |
|---|---|---|
| `domino.place_upright` | skill | `pose_feedback, cartesian_motion, gripper, stop, object_feedback, object_pose_feedback` (+ `linear_cartesian_motion` when `descent_path: "linear"`) |
| `domino.verify_stable_placement` | tool | — (needs Observation records with orientation, else `unverified`) |
| `domino.plan_straight_layout` | tool | — |
| `domino.plan_recovery` | tool | — |

`place_upright` must run under a task lease: it only uses `ctx.invoke` (`move_waypoints`,
`set_gripper`) and `ctx.observe` and never acquires anything itself.
`tests/tasks/test_domino_runtime_integration.py` runs it through `Runtime.task_lease` +
`primitive_skill_context` + `run_task_skill` over a `FakeDominoStation`.

## 2. SkillContext additions requested

1. **`ctx.state()`** (or a no-motion `read_state` primitive): `{tcp_pose, timestamp_s,
   gripper: {holding, width_m, timestamp_s, source}}`. Without it the pre-motion holding
   check cannot compare the observed tile with `TCP · T_TCP_object`; today a missed grasp
   is rejected before motion only when the tile is *seen resting on the support*, otherwise
   it is caught after the carry (`move_waypoints` held check → `grasp_lost`). The module
   already uses `ctx.state()` when present (`getattr`).
2. **`ctx.sleep(s)`** for the settle-window loop (falls back to `time.sleep`; tests use a
   fake clock).
3. **Station limits visible to tasks** (`max_timeout_s`, `position_tolerance_m`,
   `rotation_tolerance_rad`, `held_object_tolerance_m`, `max_speed_m_s`). The module caps
   each primitive at `primitive_timeout_s` (default 30 = `Limits.max_timeout_s` default)
   and cannot tell whether its own `tcp_tolerance_m` (6 mm) is tighter than the station's
   convergence tolerance (10 mm default).

## 3. Primitive gaps

1. **Straight legs.** No backend declares `linear_cartesian_motion`. The descent and the
   retreat are sent as `move_waypoints(linear=false)` every `waypoint_step_m` (5 mm,
   config); the path between waypoints is backend-defined. Results carry
   `unverified: ["descent_and_retreat_are_waypoint_sequences_not_linear_interpolation"]`.
   A true linear mode in the Isaac backend (Cartesian interpolation in the sim controller)
   would remove this.
2. **Guarded descent / contact.** `mode: "contact_feedback"` is rejected
   (`capability_missing`) until a primitive with a stop-on-contact condition, max travel
   and a declared signal exists. Not needed for stage A.
3. **`move_waypoints` failure evidence.** On `motion_not_converged` the result's
   `evidence` is empty; the module reads the last `pose_verification.measured_pose` from
   `trace`. Please keep `evidence.arrivals` (so far) on failure as the stable interface.
4. **Support observation.** If `support.observation_id` is set (e.g. `"table"`), place
   observes it before releasing and refuses to let go when it is off the calibrated plane by
   more than `support_height_tolerance_m` (`domino.support_mismatch`). Backends need to
   report such an object (class `support`, position at the top surface).

## 4. Contract notes / proposed v1.x additions

- Object frame for dominoes: origin at the geometric centre (DominoLayoutTarget), +x the
  facing (thickness), +y width, +z height (same as Agent 04's `TileSpec`). Please state
  the axis order of `geometry.size_m` = `[h, w, t]` against the local axes in the contract.
- `domino.*` failure types used: `neighbour_clearance`, `retreat_blocked`,
  `descent_blocked`, `pose_out_of_tolerance`, `support_mismatch`, `not_resting_on_support`,
  `toppled_after_release`, `neighbour_disturbed`. Core ones used: `invalid_request`,
  `capability_missing`, `precondition_unmet`, `unreachable`, `grasp_lost`,
  `held_object_slipped`, `motion_not_converged`, `release_failed`, `evidence_missing`,
  `stale_observation`, `object_not_observed`, `timeout`, `cancelled`, `backend_error`,
  plus `perception.low_confidence` for identity below threshold.
- Results carry extra keys beyond SkillResult (`failure` {code, phase, detail},
  `cause_hypotheses`, `recovery`, `unverified`, `module_contract`); SkillResult has no
  `additionalProperties: false`, and `validate("SkillResult")` passes (tested).
- Thresholds: `tasks/domino/config/line_v1.yaml` (`tolerances_ref`);
  `thresholds_override` may only tighten; `release_gap_m` / `release_clearance_m` cannot be
  overridden.

## 5. Graph wiring suggestion (G, not K)

`place_upright` already calls the observer; route on its `failure_type`:

- `succeeded` → next slot (from `plan_straight_layout` `outputs`/`layout_target.order`).
- `failed:domino.toppled_after_release` → Agent 01 `stage_and_regrasp` (reset_piece), then place.
- `failed:domino.neighbour_disturbed` → `plan_recovery` → restore affected tiles far-first.
- `failed:motion_not_converged` / `failed:domino.descent_blocked` / `failed:domino.support_mismatch`
  → back off (still holding), reobserve, retry.
- `failed:grasp_lost` / `failed:held_object_slipped` / `rejected:precondition_unmet` → regrasp.
- every `unverified:*` → reobserve then `verify_stable_placement` only (no motion).
- `rejected:capability_missing` → abort.

`domino.plan_recovery` returns the bounded plan in `outputs.recovery`
(`attempts_used`/`max_attempts` from the graph's own counters).

## 6. Packaging decision needed

`upstream.py` loads `skills/placing-an-upright-domino-beside-neighbours/scripts/place_domino.py`
and `skills/planning-a-domino-layout/scripts/plan_layout.py` by path (reuse, not copy). In
a wheel without `skills/` (or without the `gap` extra, which those scripts import at module
level for `NodeContext`, or without scipy for `plan_layout`), every domino entry returns
`rejected / precondition_unmet` naming the missing piece. Options: keep the
repo-checkout requirement (current), or move the scripts' `NodeContext` import under
`TYPE_CHECKING` and ship them as package data. `config/line_v1.yaml` must also ship as
package data (hatch's default includes it under `src/roboskillhub`; please confirm).
