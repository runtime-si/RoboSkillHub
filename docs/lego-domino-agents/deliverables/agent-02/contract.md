# Domino module contract (Agent 02, `domino-task/0.2.0` on `rsh-task/1`)

All results are rsh-task/1 `SkillResult`s (`validate("SkillResult")` passes), with extra
keys `failure {code, phase, detail}` (module fault code and module phase),
`cause_hypotheses [{cause, confidence, evidence}]`, `recovery`, `unverified [...]`.
Metres, seconds (Unix), radians, quaternion xyzw, explicit `frame_id`.

## Object convention

Tile frame: origin at the geometric centre (`geometry.origin: "bottom_face_center"` is
also accepted), +x facing (thickness), +y width, +z height (up axis).
`geometry: {"size_m": [h, w, t], "yaw_symmetry_deg": 180 | 360, "up_symmetric": true}`.
Yaw folds by 180° for a plain tile; 360 when the faces differ. Tilt = angle between the
tile's +z and the support normal (±z when `up_symmetric`).

## `domino.place_upright` (skill)

Request (key fields): `frame_id`, `target_id`, `geometry`, `target_object_pose` (centre
pose, upright on the support), `held` (rsh-task/1 HeldObject: `T_TCP_object` in `tcp`,
`valid: true`, fresh, `uncertainty.position_m` ≤ `max_holding_std_m`), `support`
(`point_m`, `normal`, `height_std_m`, `calibration_version`, optional region and
`observation_id`), `neighbors` (fresh Observation records with orientation), `workspace`,
`timeout_s`; optional `approach` / `retreat` (`direction`, `distance_m`), `release.width_m`
(default tile width + 2 × `release_clearance_m`), `hand` (`tip_reach_m`,
`max_open_width_m`), `mode` (`geometric`), `descent_path` (`dense_waypoints` | `linear`),
`speed_m_s`, `descent_speed_m_s`, `primitive_timeout_s` (default `ctx.limits.max_timeout_s`),
`contact` (`force_limit_N`, `search_m`, `max_travel_m`, `speed_m_s`; contact mode only), `tolerances_ref`,
`thresholds_override` (tighten only), `recovery_attempts_used`, `max_recovery_attempts`.

Before any command (→ `rejected`): frames/units/quaternions; target upright and on the
plane; held identity/validity/freshness/uncertainty; neighbour freshness and orientation;
footprint overlap; the open hand over the target clear of every neighbour (the placing
skill's `finger_conflicts`); the carry ending with the held tile 20 mm over the standing
tops (`TRANSIT_CLEARANCE`); swept-envelope checks for slanted approach/retreat; workspace
for every TCP pose; release width vs tile and hand; time budget vs descent + retreat +
settle window; capabilities ⊆ backend; the station's `ctx.limits` (frame, speed, workspace,
gripper range, contact force/travel).

Contact mode (`mode: contact_feedback`, rsh-task/1.1): adds `contact_force_feedback,
compliant_motion`; the descent stops `search_m` above the expected contact and `guarded_press`
moves the held tile down until the controller stops on force. Released only on
`force_threshold` within `support_height_tolerance_m` of the calibrated support.

A `rejected` verdict from the observer (its own request invalid) makes `place_upright`
`unverified / evidence_missing` with `failure.code: verification_rejected`.

Sequence: fresh observation (tile in the hand: not resting on the support; and when
`ctx.state()` exists, within tolerance of `TCP·T_TCP_object`; neighbour baseline) → carry
(`move_waypoints`, rise then cross, `held`) → observe tile still in hand → descent
(`move_waypoints`, dense straight leg, `held`) → measured TCP within `tcp_tolerance_m`
(high ⇒ `domino.descent_blocked`, lateral ⇒ `motion_not_converged`; nothing released) →
support check (optional support observation; held tile not tilted, not into the table) →
`set_gripper(purpose=release)` → observe → retreat (`move_waypoints`) → settle window →
`verify_stable_placement`. Status and failure come from the observer; an executor that
always says ok cannot make it succeed. No automatic re-open, re-press or retry.

## `domino.verify_stable_placement` (tool)

Input: `target_object_pose`, `support`, `geometry`, `events {action_start_s, release_s,
retreat_done_s}`, `observations` (Observation[]), `neighbors_baseline` (Observation[]
taken before `action_start_s`), optional `gripper_states`, `tcp_states`, `snapshot_diff`
(detecting-disturbed-pieces-by-snapshot-diff output), `require_unprivileged`, `now_s`.

Succeeds only if: ≥ `min_window_samples` distinct timestamps over ≥ `settle_window_s`
after the retreat, newest ≤ `max_observation_age_s`; target identity (confidence ≥ min, or
privileged/synthetic), visible, with orientation; tilt ≤ `tilt_tolerance_rad` at every
sample after the release; drift ≤ `drift_tolerance_m` / `drift_tilt_tolerance_rad`;
final position/yaw within tolerance; lowest corner on the support; gripper reported not
holding from the end of the retreat on (contact-sampled states may still touch the tile while
the jaws open; with no states after the retreat, the latest after the release counts), or the
measured TCP clear above the tile; every baseline
neighbour within displacement/tilt tolerance. Records without orientation can still show a
fall (`state_label` lying/toppled, e.g. the placing skill's depth check via
`depth.depth_upright_observation`). Privileged inputs are labelled (`privileged_inputs`).

## `domino.plan_straight_layout` (tool)

`start_m`, `heading_deg`, `n_tiles`, `table_z_m`, `spacing` (`center_pitch_m` xor
`face_gap_m`; centre = face gap + thickness), optional `arm_base_xy_m`, `workspace`,
`existing` (observed standing tiles), `order_preference` (`far_first`, `forward`,
`reverse`), `cascade {check, require}`. Returns slot poses, a `DominoLayoutTarget`, every
candidate order with its reason, the chosen build order, and cascade physics (reported;
enforced only with `require`). No motion.

## `domino.plan_recovery` (tool)

From a place/verify result: a bounded plan (`action`, `steps` with the capability each
needs, `affected` tiles, `attempt`/`max_attempts`, `executed: false`) and rsh-task/1
`recovery_hints`. Plans always start from a fresh observation (only `retreat_holding` may
precede it); `rejected` results never get motion steps; budget spent ⇒ `abort`.

## Status semantics

`rejected`: nothing commanded. `failed`: commanded, and observations show the task is not
done (or a primitive reported the failure). `unverified`: commanded, evidence insufficient
(stale/repeated/missing/occluded/orientation-less observations, cancel). `succeeded`:
the observer's postcondition passed on independent observations.
