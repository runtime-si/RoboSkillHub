# Five workcell capabilities: agent → Hub → YAM / Isaac

This integration adds executable primitives, a JSON agent API, five graph
bundles, a versioned bridge for the team's existing YAM station, a compatible
RGB-D camera client and an Isaac Core adapter. It was connected to YAM workcell
and Brev on 2026-11-06. The real camera and an actual YAM Isaac free-space move
have run, followed by isolated contact push and grasp-and-lift checks. The YAM
control chain has passed a model-only IK/TCP test without CAN
or motor access. No physical manipulation acceptance is claimed.

Camera calibration, motor configuration, meshes and station source remain in
the team's workstation projects; these private installation assets are not
copied into the public skill library.

The supplied primitives are independently implemented, with reviewed upstream
material as references. They do not copy the full ENPIRE/ManiSkill experts or
claim the upstream results. The grasp skill executes supplied grasp poses;
object detection and GraspGenX pose generation remain upstream graph operations.

## Install and inspect

```bash
pip install -e '.[runtime]'
roboskillhub runtime list
roboskillhub search '位姿运动' --embodiment yam_i2rt --arms 1 --json
roboskillhub search '抓取' --embodiment yam_i2rt --arms 1 --json
roboskillhub search '点跟踪' --embodiment yam_i2rt --arms 1 --json
roboskillhub search '推物' --embodiment yam_i2rt --arms 1 --json
roboskillhub search '拉抽屉' --embodiment yam_i2rt --arms 1 --json
```

`runtime list` returns JSON Schemas and backend capability declarations. With no
station, motion is unavailable and requests return `rejected`; there is no fake
robot fallback. Point tracking can run on saved RGB clips without a station.
An available declaration is not an installation, collision, sim or real eval.

## Uniform request and result

Positions are metres in the explicitly named station frame; orientations are
unit quaternions **xyzw**. No implicit camera/base/world conversion is performed.
All motion requests use named poses of this form:

```json
{
  "frame_id": "station_world",
  "position_m": [0.3, 0.0, 0.2],
  "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
}
```

These coordinates only illustrate the format; they are not a calibrated YAM
pose. A live configuration must set its measured frame, TCP, workspace and
gripper calibration. If the YAM tool frame differs from Isaac's configured Lula
end-effector frame, transform it explicitly in the station project.

Every result includes `schema_version`, `invocation_id`, `capability`, `skill`,
`version`, `status`, `evidence`, `trace`, `elapsed_s` and a provenance `manifest`.
The four statuses are:

- `succeeded`: the implemented postcondition has supporting measured evidence.
- `failed`: an attempted action or tracked sequence did not satisfy its condition.
- `unverified`: actions may have happened, but outcome evidence is missing or stale.
- `rejected`: invalid input, unavailable dependencies/backend, or missing
  prerequisites prevented execution.

Do not automatically retry `failed` or `unverified` without observing the new
state. The controller may have moved before a failure. Failure stops motion and
does not automatically release a held object.

## Capabilities and required fields

The authoritative machine-readable contracts are returned by `runtime list`.
Unknown request fields are rejected.

### `move_pose` / `moving-to-pose`

Required: `target_pose`.

Optional: `speed_m_s`, `timeout_s`.

The station executes motion, then the skill reads a fresh measured TCP pose.
Success requires position and orientation within station tolerances. The API
does not turn IK into collision planning; the station owns obstacle-aware motion.

### `grasp` / `grasping-with-feedback`

Required: `pregrasp_pose`, `grasp_pose`, `lift_pose`, `target_id`,
`open_width_m`, `close_width_m`. Optional: `speed_m_s`, `timeout_s`.

Sequence: open → pregrasp → approach grasp → close → observe object → lift →
observe object again. Every motion checks TCP feedback. The lift must be at
least 2 cm upward and remain within the contact-distance bound. Success checks
the object's independent observed displacement during the lift; an ACK, empty
gripper closure or TCP movement alone never establishes grasp success.

This is an initial lift-following criterion, not a proof of arbitrary grasp
stability. It requires a stable tracked object ID and sufficiently accurate
object observations. It does not implement grasp proposal generation or grasp
selection; supply those from the existing perception/geometry graph.

### `track_points` / `tracking-points`

Required: `frames_path`, `query_points`. Optional: `backend` (`opencv-lk` or
`tapir`). `frames_path` is an uncompressed `.npy` RGB uint8 array `[T,H,W,3]`,
at least two frames and at most 512 MiB. Queries are first-frame `[x,y]` pixels.
Files are loaded with `allow_pickle=False`.
Inference and the recorded input digest use the same immutable file snapshot;
later replacement or deletion of the clip cannot change its provenance.

OpenCV-LK is a real CPU baseline with forward/backward and photometric checks.
Lost tracks do not silently reinitialize. It is named separately from TAPIR.
The TAPIR adapter runs the official torch model's **offline clip** interface,
including normalization, `(t,y,x)` query conversion and occlusion confidence.
It is not the causal streaming TAPIR implementation. It requires explicit local
weights and a SHA-256 digest; no automatic model download occurs.

All frames/points valid → `succeeded`; partial tracks → `unverified`; no valid
final-frame tracks → `failed`. Missing coordinates are JSON null, never NaN.
Outputs remain 2-D image pixels; they are not metric object feedback for the
manipulation skills. The station must combine depth/calibration/object identity
to provide such feedback.

For an initialized camera application, call
`LKPointTracker().track(rgb_frames, points)` directly to avoid saving clips.

### `push` / `pushing-on-plane`

Required: `approach_pose`, `contact_pose`, `target_id`, `direction_world`,
`distance_m`. Optional: `step_m`, `speed_m_s`, `timeout_s`.

This initial skill pushes along the station XY plane using bounded Cartesian
segments. The full path is checked against workspace limits before any motion.
Object displacement is measured from the **post-contact** observation; movement
caused by approaching the contact point does not count as the requested push.
The object must move in the intended direction. This is a position-controlled
push; it does not infer contact force from a camera.

### `pull_drawer` / `pulling-drawer`

Required: `pregrasp_pose`, `grasp_pose`, `target_id`, `direction_world`,
`distance_m`, `open_width_m`, `close_width_m`.
Optional: `step_m`, `speed_m_s`, `timeout_s`, `release_on_success` (default false).

`target_id` denotes an independently observed handle/drawer reference point.
The supplied direction is the known prismatic opening axis. The skill approaches,
closes, records a fresh baseline, pulls in bounded segments, and verifies observed
translation along that axis. It supports neither a locked drawer nor a hinged
door by assuming the same geometry. No successful release is reported if pulling
fails; optional opening occurs only after the drawer postcondition passes.

Motion outcome thresholds are fixed code in K: at least 70% projected travel,
at most 2 cm lateral error, bounded overshoot; grasp additionally requires at most
2 cm displacement mismatch. Validate these thresholds against task scale and
measurement noise before freezing K. They are not universal benchmark criteria.

## Bind the actual station

The station YAML is trusted process configuration. An agent request cannot change
its Python factory, coordinates, bounds or model weights. It contains:

- `schema_version: 1`, `station_id`.
- `backend.factory`: an importable `module:function` from the station project.
  `backend.kwargs`: optional JSON/YAML keyword arguments to that factory.
- `limits`: required `frame_id`, `workspace_min_m`, `workspace_max_m`,
  `max_speed_m_s`, `max_contact_distance_m`, `gripper_min_width_m`,
  `gripper_max_width_m`; optional `max_state_age_s`, `max_timeout_s`,
  `position_tolerance_m`, `rotation_tolerance_rad`.
- Optional `tracking`: `backend`, and for TAPIR `checkpoint_path`,
  `checkpoint_sha256`, `device`. Relative checkpoint paths resolve beside YAML.
- Optional `log_dir`: one JSON invocation record per call; relative to YAML.

The factory is called as `factory(limits=validated_limits, **kwargs)` and returns
a `StationBackend`. Use those same limits in the adapter; kwargs cannot override
them. The factory must connect/inspect only, never move
at construction. Use a single owning runtime process per physical station;
in-process locks do not coordinate two independently launched processes.

### YAM binding

`roboskillhub.runtime.yam.YamBackend` takes explicitly bound site callbacks:

```python
backend = YamBackend(
    limits=limits,
    read_state=station.read_state,
    move_cartesian=station.move_cartesian,
    set_gripper_width=station.set_gripper_width,
    stop=station.stop,
    object_feedback=True,
)
```

Here `station` means your existing control application adapted to these callback
signatures, not a module shipped in this repository:

```python
read_state(timeout_s=...) -> state
move_cartesian(pose, speed_m_s=..., timeout_s=..., linear=...) -> {"ok": bool}
set_gripper_width(width_m, timeout_s=...) -> {"ok": bool}
stop() -> None
```

Callbacks preserve the controller's confirmation, bounded motion and watchdog.
They must honor speed and timeout and report actual measurements. The adapter
detects a timeout after a blocking call returns; it cannot interrupt a hung
native driver. `gap_pose_to_pose` / `pose_to_gap` convert the known named GAP
quaternion fields but never guess frame calibration or gripper conversion.
For the inspected YAM workcell installation, use the implemented binary TCP
transport rather than guessed HTTP endpoints:

```python
from roboskillhub.runtime.yam_tcp import YamTcpBackend, CameraClient, capture_clip

backend = YamTcpBackend(limits=limits, port=9201)  # read-only handshake
camera = CameraClient(port=9102)
capture_clip(camera, frame_count=12, output_path="clip.npy")
```

`YamTcpBackend` requires `roboskillhub-yam` protocol v1 and the explicit
`hub_describe`, `hub_state`, `hub_move`, `hub_grip`, `hub_stop` operations, plus
`hub_claim` / `hub_release` for exclusive ownership of an entire invocation.
The old `motion_server.py` on its own does not implement them. In the existing
station-owning process, after the operator's normal lifecycle setup, replace
the legacy RPC listener with:

```python
from roboskillhub.runtime.yam_bridge import Bridge, make_server

bridge = Bridge(station, allow_powered=True)  # station is already operator-owned
server = make_server(bridge, port=9201)  # localhost only
server.serve_forever()
```

Keep the existing arm lock, supervisor, operator confirmation and shutdown
flow. Never expose both the old motion listener and this bridge for the same
station. Constructing the bridge does not power the arm. `allow_powered=True`
only allows attachment to a station whose operator has already powered it.
For programmatic integration, use this attachment in the station's listener
thread so its main supervisor loop continues running.

The bridge adds command IDs, deadlines, cancellation during confirmation or IK,
and measured hold on stop. A late confirmation or completed plan cannot commit
an already cancelled command. The client does not retry actions on connection
loss; it uses an independent stop connection. `finish` and `reset` are never
used as stop substitutes.

Runtime acquires the bridge's invocation lease before observations and holds it
through every movement, gripper command, verification and failure stop. Another
client cannot insert actions between those steps. The lease is released only at
the end; a crashed owner is not automatically replaced while motion might remain
in progress. Operator recovery is required for an abandoned lease. Direct SDK
action calls must explicitly use `begin_invocation(timeout_s=...)` and
`end_invocation()`; the high-level Runtime does this for the agent. Local lock
waiting consumes the requested timeout and expired queued requests never start.

The inspected controller has a process-wide speed cap, not a per-request speed
setting. A request below that cap is rejected. Start the operator's existing
station with `--speed 1` to obtain its 0.05 m/s cap; a 0.03 m/s request still
cannot be honored by that controller. The old default `--speed 6` is 0.30 m/s.
Supply `speed_m_s: 0.05` explicitly for this site's x1 controller; the generic
skill default is 0.03 m/s and therefore deliberately rejects this YAM binding.
Native plans are discrete Cartesian waypoints with joint-space tracking; this
is not continuous straight-line certification or collision planning.

In trusted YAML, select `backend.factory:
roboskillhub.runtime.yam_tcp:create_backend`, with `kwargs: {host: 127.0.0.1,
port: 9201}`. The default requires a real `confirm` or `auto` station. Only an
explicit `expected_kind: dry-run` allows model-only operation; results retain
`kind: dry-run` and `virtual: true`. Such success is not physical success.

### Independent live object observations

`RGBDPointObserver` combines continuous LK surface-point tracking, aligned
depth and the explicit `T_base_from_camera` calibration. Seed target IDs on
the **same camera frame** used to choose their pixels, start the observer before
motion, and attach it as the backend's `object_provider`:

```python
from roboskillhub.runtime.rgbd_observer import RGBDPointObserver

seed = camera.read_frame()
# Perception/operator assigns target_points to exact seed-frame pixels.
observer = RGBDPointObserver(
    camera=camera,
    seed_frame=seed,
    target_points=target_points,
    frame_id=limits.frame_id,
    T_base_from_camera=calibration["T_base_from_camera"],
).start()
backend = YamTcpBackend(limits=limits, port=9201, object_provider=observer)
# Keep observer alive throughout Runtime invocations; close it at session end.
```

This is a metric surface landmark, not a semantic detector, object pose
estimator or force sensor. Target identity must come from explicit perception
or operator selection; a drawer target must lie on its moving part. The centroid
uses the same seed set throughout. Lost points, inconsistent depth, large frame
gaps and stale observations do not silently generate replacement evidence.
Observe continuously during arm movement, not only before and after it.
No object positions are inferred from commanded TCP positions.

To configure this composition through YAML, use
`roboskillhub.runtime.yam_rgbd:create_backend`. Its kwargs require `port`,
`camera_port`, `calibration_path`, and a trusted `target_selector` in
`module:function` form. The selector receives the exact live seed frame and
returns `{target_id: [[x,y], ...]}`; optional `selector_kwargs` configure the
existing perception function. It must be fast enough to preserve tracking
continuity. Slow selection is rejected; images are never silently retimestamped.

The calibration JSON must declare `calibrated: true` and `T_base_from_camera`.
If the file omits `frame_id`, set `calibration_frame_id` explicitly in YAML to
the same measured frame as `limits.frame_id`. The YAM workcell legacy file needs
this declaration. The factory only reads devices and starts observation; it
does not claim the control lease or issue robot commands. It accepts a real
backend only, so live camera data cannot validate a virtual arm execution.

This provides the factory for all manipulation capabilities while leaving
semantic target selection in the existing perception pipeline. The library does
not turn arbitrary image corners into named objects. Close the owned observer
when ending the session; restart/reseed after losing the target or changing tasks.

### Isaac binding

`roboskillhub.runtime.isaac.IsaacBackend` targets the **legacy Isaac Core API**
(5.1 documentation family; exercised on installed 6.2.0-rc.14),
not IsaacLab vector environments. It takes an already initialized `World`,
`SingleArticulation`, `ArticulationKinematicsSolver`, explicit stage scale,
gripper DOF indices and a calibrated width-to-joint mapping. Invoke it in the
standalone simulation thread that owns `World.step`, not from a physics callback.

It applies IK joint targets and advances physics, reads actual FK, and holds
measured joint positions on stop. It does not teleport or supply collision
planning. For contact-limited gripper closing, an optional `gripper_completion`
callback must return fresh evidence for the requested width; it does not imply
`holding=True`. Without that callback, closure requires measured joint arrival. The site still supplies its YAM USD/URDF, Lula end-effector link,
drives, table/object scene, collisions and camera calibration.

On stop, the arm holds measured joint positions while an already completed
gripper command retains its drive target, preserving contact preload. If a new
gripper command is interrupted before completion, its measured positions are
held and the old completed target is not restored. Neither behavior asserts
`holding=True`. The owner must keep advancing physics for a drive hold to work.
Recreate the adapter after `World.reset` or an external drive-target change;
clock-rewind detection invalidates saved targets but cannot detect every reset.

The skill timeout bounds wall-clock execution, including ownership waiting. Each
Isaac action additionally has a physics-step budget based on its remaining wall
budget; this is not a separate total simulated-time limit for a composite skill.

`simulation_object_provider` can read explicitly named prims for initial sim
contract experiments. Those records carry `source: simulation_truth`; they must
not be described as camera-only evaluation. For deployment-matched experiments,
bind the same camera-based observation logic used on the real station instead.

### Observation contract

```json
{
  "pose": {"frame_id":"station_world", "position_m":[0.3,0,0.2],
           "quaternion_xyzw":[0,0,0,1]},
  "timestamp_s": 1793923200.0,
  "objects": {
    "target": {"frame_id":"station_world", "position_m":[0.3,0,0.1],
               "timestamp_s":1793923200.0, "source":"calibrated_rgbd"}
  },
  "gripper": {"holding": null}
}
```

The numbers above illustrate the structure; historical timestamps are rejected.
Use original acquisition Unix timestamps on synchronized clocks, never replace
cached observations with response time. Post-action feedback must be acquired
after command completion. Unknown holding is null; a reported boolean requires
its own timestamp and measurement source in the YAM adapter. Object IDs must
refer to tracked scene entities, not the robot TCP or commanded positions.

## Call from an agent

```bash
roboskillhub runtime invoke move_pose --station station.yaml --request move.json
roboskillhub runtime invoke track_points --request tracking.json
roboskillhub runtime serve --station station.yaml
```

`serve` is local JSON-lines over stdin/stdout, with no network listener:

```json
{"id":"inspect","method":"describe"}
{"id":"move-1","method":"invoke","params":{"capability":"move_pose","request":{"target_pose":{"frame_id":"station_world","position_m":[0.3,0,0.2],"quaternion_xyzw":[0,0,0,1]}}}}
```

Each line gets `{id, result}` or `{id, error}`. Read the result status. The example
request is illustrative, not authorization to use those coordinates on a robot.
This is a small local protocol, not an MCP or HTTP server.

Python callers use `roboskillhub.runtime.api.describe()` and
`invoke(capability, request, station_path=...)`. Inside an initialized Isaac
application, use `Runtime(backend, limits).invoke(...)` directly so construction
and stepping stay on the owning simulation thread.

GaP callers use the five bundles' canonical scripts or the `workcell` tools.
Set `ROBOSKILLHUB_STATION` in the graph process and install this package there.
Every graph must route all four named exits. Runtime schemas are available via
`workcell.describe`, with the full response retained as `result_json`.

## Reproducibility and validation status

The runtime returns its implementation digest, the loaded station-config snapshot
digest and factory name. TAPIR records its checkpoint digest; clip calls record
the input file digest. Restart the process after changing runtime code or station
plugins. Keep the site plugin/robot asset/calibration revisions in your experiment
manifest as well: this version does not automatically hash external plugin trees.

Existing `skills.lock` pins bundle files but does not capture all code behind
`workcell.invoke`, station plugins or robot assets. Preserve the runtime manifest
and install the same Hub revision alongside the lock. The historical `camera_only`
tier classification is not proof that a station callback avoids simulator truth.

CPU tests cover API dispatch, recorded feedback, real OpenCV tracking, TAPIR
tensor conventions, YAM callback binding and the Isaac API contract. Fake Isaac
objects in tests are not an Isaac physics evaluation; controlled TAPIR models are
not an executed official checkpoint. All five new bundles remain `community`.
Only actual station runs with task-specific evidence should generate sim/real evals.

Observed integration checks on 2026-11-06:

- YAM workcell Orbbec top camera: live RGB/depth/intrinsics via the original wire
  protocol; the new client captured 8 distinct frames over 0.803 s. The tracking
  API separately processed 12 live frames with 24 seeded image features and
  returned `succeeded`. This is short-clip tracking, not a moving-object benchmark.
  The live RGB-D observer additionally retained 8 surface features over 5 fresh
  post-call observations using the site's calibration. It did not establish
  semantic object identity or independent metric calibration accuracy.
- YAM: actual site `Station.plan/fk` and `ArmKinematics` through
  Runtime → TCP client → bridge, with explicit virtual joint state. A 10 mm
  move passed; unsupported speed was rejected without movement. Instrumented
  CAN and hardware access counts were zero. This was not a physical arm run.
- Brev Isaac 6.2.0-rc.14: source YAM MJCF imported into PhysX, matching URDF/Lula
  TCP checked against independent rigid-body transforms. One `[10,0,10] mm`
  free-air move completed in 68 physics steps (0.567 simulated seconds), with
  independent final error 2.771 mm / 0.000735 rad. Initialization used a reset;
  execution used joint targets, without teleporting the arm. This proves that
  move only, not grasp/push/drawer performance or a calibrated workcell twin.
- Brev Isaac contact push: a 40 mm dynamic cube on a frictional support surface
  moved 55.847 mm for a requested 60 mm push, with 1.242 mm lateral error. Contact
  used the imported YAM finger meshes; objects retained gravity and were neither
  attached to the robot nor teleported. Object evidence was explicitly
  `simulation_truth`. This is one controlled scene, not a success-rate benchmark.
- Brev Isaac grasp-and-lift passed in the refined scene: the same 40 mm / 50 g
  cube rose 45.55 mm for a requested 50 mm lift, and remained 48.31 mm above its
  post-close baseline after one simulated second. The grasp example detects
  stable, bilateral, cube-filtered fingertip contact to complete closing;
  `holding` remains unknown and actual cube displacement proves the outcome.
  Mass, friction and the skill's success thresholds were not relaxed.
  A subsequent stop preserved both applied gripper targets; after another
  simulated second, the cube remained 48.15 mm above baseline. This validates
  controlled drive hold in this scene, not a physical emergency stop.
- Early grasp attempts remain failed records. The initial top-down pose grid
  exceeded the source joint4 lower limit; independent bounded IK identified
  a reachable pregrasp/grasp/lift path. A later test exposed a collision-shape
  issue: single convex hulls filled the concave fingertips, adding roughly
  20 mm per side at the cube's top plane. The common test scene now uses convex
  decomposition for only those two finger colliders. This is an explicitly
  recorded scene variant, not a claim to reproduce the original collision
  model unchanged. Source mesh vertices, frames and units were preserved.
  The refined variant also passed movement and push regression: TCP error
  remained 2.771 mm, and the cube moved 55.378 mm for a 60 mm push with 1.032 mm
  lateral error. The earlier unrefined-scene records remain separate evidence.
- Brev Isaac drawer pulling passed with a 200 g tray, 50 g handle and a world
  prismatic joint limited to 0–120 mm. The handle is physically fixed to the
  tray, never attached to the robot. The fixture uses a passive 10 N·s/m
  viscous drive capped at 2 N, with zero stiffness and target velocity; it
  does not command opening. Surface material friction is 0.6/0.5, without a
  claim of calibrated Coulomb rail friction. During the requested 60 mm pull,
  the handle moved 57.354 mm from the post-close baseline. After release,
  independent tray pose gave a joint coordinate of 57.962 mm; after another
  simulated second it remained 57.989 mm. Two seconds without robot contact
  produced only 0.002792 mm maximum drift. During execution, handle/tray
  translation disagreement stayed below 0.131 mm and maximum rotation below
  0.00520 rad. Handle-filtered bilateral contact completed closing, with
  93 detailed contacts in a 512-entry buffer and no saturation. The runtime
  kept `holding` unknown and used independent handle motion for success.
- The drawer missed-handle negative control also passed its acceptance checks:
  a valid 75 mm close command left clearance around the 40 mm handle, and the
  robot completed all six pull segments with final TCP error 2.901 mm /
  0.001185 rad. Handle contact forces stayed zero; drawer opening did not
  change from its 0.002792 mm initial offset. The runtime correctly returned
  `failed` for insufficient object motion and did not execute the optional
  success release. This is a passed failure-detection test, not a successful
  drawer skill execution. The optional locked-joint control was not run.

These are individual controlled-scene checks, not hardware manipulation,
success-rate or generalization benchmarks.

The grasp contact callback measures stability from consecutive joint positions
and simulation time, while retaining the raw velocity output. The observed
near-static positions with nonzero reported velocities are consistent with
NVIDIA's documented TGS drive limitation when
`enableExternalForcesEveryIteration` is false; the test does not change that
solver setting. See [Known Physics Limitations](https://nvidia-omniverse.github.io/PhysX/ovphysx/0.7.0/guides/limitations.html).
Normal-force measurements use the physics timestep to convert impulses to N.
They establish contact completion, not grasp success or contact friction forces.

Reproduce the independent Isaac tests with
[`examples/workcell/isaac_smoke.py`](../examples/workcell/isaac_smoke.py), using
the existing Isaac 6.2 Python environment and existing RSS YAM assets. Its CLI
requires an isolated output directory and emits the exact executed script,
source hashes, runtime hashes and `report.json`.

For the drawer fixture, run the positive and missed-handle control separately:

```bash
PYTHONPATH=src /path/to/isaac/python examples/workcell/isaac_smoke.py \
  --rss-assets /path/to/robosimstudio/assets/embodiments/yam_i2rt \
  --output /new/run/drawer-positive --task pull_drawer
PYTHONPATH=src /path/to/isaac/python examples/workcell/isaac_smoke.py \
  --rss-assets /path/to/robosimstudio/assets/embodiments/yam_i2rt \
  --output /new/run/drawer-missed-handle --task pull_drawer \
  --drawer-control miss_handle
```

Both runs use a 60 mm pull along the fixture's opening axis. The missed-handle
control closes to 75 mm rather than 30 mm around a 40 mm wide handle. Its request
is valid and the arm must complete the pull path, but the independently observed
drawer must remain closed. A passing negative test has `evaluation_role=negative`,
`control_passed=true` and `runtime_result.status=failed`; it does not mean the
skill succeeded. The optional `--drawer-control locked` variant is a separate
fixture control and requires an actual contact pull attempt before a failure can
count. A failure before that pull attempt does not satisfy the locked control;
the missed-handle control additionally requires the entire pull to finish and
the object-motion postcondition to fail.

All five contracts are callable. Grasp, push and drawer manipulation additionally
require target-specific perception, appropriate geometry, and contact validation
on the actual workcell; these are not inferred from the successful camera/move
checks. TAPIR's official checkpoint has not been executed in this integration.
