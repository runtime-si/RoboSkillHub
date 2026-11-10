---
name: picking-an-object-with-a-learned-policy
description: Implements the pick_object capability with a learned bimanual YAM
  action model behind a vla-yam/1 server (policies/molmoact2-yam, or
  policies/gr00t with the pi05 or n1.7 backend) - builds an instruction (and a
  grasp mark when the object is a point) from object_ref, runs
  roboskillhub.vla.run_policy (profile frame mapping, safe chunk executor)
  until the measured jaw closes and the policy lifts, then lets the shared
  verify_pick verifier decide the exit from the jaw opening, the scene diff at
  the object's spot and, when an object-position tool is wired, the lift.
  Routes picked, missed, slipped, unreachable, refused or timeout. Use when a
  graph wants a pick_object implementation that is a learned policy (to
  compare with program and hub grasps under the same verifier, or where no
  grasp program exists) and a policy server is running on a GPU host. Not a
  success signal on its own and not for the real arms before the profile's
  stations.real offset and gripper map are confirmed in the twin.
license: Apache-2.0
compatibility: requires gap>=0.1 and roboskillhub (roboskillhub.vla, roboskillhub.capability); a running vla-yam/1 or gr00t-wam/1 policy server on a CUDA host (molmoact2 ~12 GB, pi05 ~9 GB, n1.7 ~6 GB)
metadata:
  category: manipulation
  tags: [pick, policy, vla, molmoact2, pi05, gr00t, yam, capability, policy-server]
  hub:
    stage: execute
    embodiments: [yam_i2rt]
    arms: 2
    version: 0.1.0
    tasks: [pick-place]
    provenance:
      authors: [runtime-si]
gap:
  requires:
    connector: [robot.stream_dual]
  allowed_tools:
    - robot.get_observation
    - robot.stream_dual
    - robot.open_gripper
    - robot.close_gripper
  # The skill runs on the graph's host (CPU); the model runs in the policy server on a GPU host,
  # so gap.requires carries no gpu flag here (the policy bundles declare theirs).
  required_inputs:
    object_ref: string
  # object_ref: the object's id or text as a string, or (wired from a script node) the
  # contract's mapping {id | text | point: [x, y, z] world}. Optional inputs: arm (left | right | auto, default auto; left/right executes that arm
  # only), grasp_hint (accepted, unused: the policy chooses the grasp), server_url (default
  # $ROBOSKILLHUB_VLA_URL, then $GAP_GR00T_URL), profile (name under policies/*/profiles/,
  # path or mapping; default yam-molmoact2-native), station (a key of the profile's
  # stations, e.g. real; default the twin), instruction, max_steps (600 rows), timeout_s
  # (30), hold_checks (2), object_position_tool + lift_source (observation | sim_truth),
  # cameras (role -> camera name), check_profile (true).
  produces_outputs:
    held_object: string
    reason: string
  # Also returned (gap has no mapping type): T_TCP_object (null, not measured) and evidence
  # (the verdict, the run, the opening trace, the server's model and the profile).
  exit_conditions:
    picked: verify_pick says the object is held - at least two independent checks agree (jaw stopped on it, its spot emptied, it followed the hand) and none says empty.
    missed: verify_pick found nothing held - the jaw closed fully or never closed, the spot did not change - or too few checks to tell (uncertain in the evidence).
    slipped: The jaw stopped on the object but the lift or a later reading lost it.
    unreachable: Reported by the verifier only on a claim with no physical attempt; this implementation does not plan reach, so it does not claim it.
    refused: Nothing was attempted - the server, the profile or the observation was unavailable, the server runs a different profile, a precondition failed (jaw not open, the object's point outside the top view), or the executor refused the first chunk.
    timeout: The attempt ran past timeout_s.
  canonical_scripts:
    - pick_with_policy: scripts/pick_with_policy.py
  examples:
    - title: Pick a named object with MolmoAct2-BimanualYAM on the twin
      path: examples/canonical_subgraph.json
---

# picking-an-object-with-a-learned-policy

The `policy:*` implementations of [`capabilities/pick_object.yaml`](../../capabilities/pick_object.yaml)
in one bundle: same inputs, outputs and exits as the program and hub grasps, and the same
`verify_pick` deciding the exit. Which model runs is configuration - the server URL and the
checkpoint profile - not code:

| implementation | server | profile |
|---|---|---|
| `policy:molmoact2-yam` | `policies/molmoact2-yam/server.py` | `yam-molmoact2-native` |
| `policy:pi05-yam` | `policies/gr00t/server.py --backend pi05` | `yam-pi05-molmoact2` |
| GR00T N1.7 (no contract entry yet) | `policies/gr00t/server.py --backend n1.7` | a gr00t profile YAML |

## What it does

1. Resolves the profile (`policies/*/profiles/<name>.yaml`, merged with `stations.<station>`)
   and asks the server's `/info`: a vla-yam/1 or gr00t-wam/1 protocol, and the same profile
   name (two profiles could disagree on the joint offset or the gripper map). Otherwise
   `refused`, nothing moved.
2. Measures the preconditions on a fresh observation: the executing arm's jaw open (opening
   >= 0.65 through the profile's gripper map) and, for a `point` object, the point inside
   the top view. A false one is `refused`.
3. Instruction: `instruction`, else `pick up the <text | id>` (`with the left arm` when
   `arm` is given). A `point` also becomes a grasp mark on the top view (the N1.7 backend
   draws it; MolmoAct2 ignores marks) and the scene-diff region (48 px square).
4. `roboskillhub.vla.run_policy` with `VlaYamClient`: model-frame state on the wire,
   chunks executed through `robot.stream_dual` with the executor's limits (joint limits,
   jump, velocity, table height when FK is available, stale chunks). The loop stops
   `hold_checks` observations after the measured jaw of an executing arm drops under 0.5
   (the policy keeps lifting meanwhile), at `max_steps` rows, or at `timeout_s`.
5. `verify_pick` on: the jaw (`stalled_on_object` when the closed jaw stays over 0.03
   opening, `reached` when it closed fully), the scene diff of the top view before / after
   at the object's spot (point objects), and the lift (`object_position_tool` samples
   against the TCP from the observation's `ee_pose`). The policy's own end is passed as
   the claim and recorded, never trusted.

`evidence` carries the verdict (votes, reasons), the run (route, rows, chunks, jaw events,
per-chunk latency), the opening trace, the server's backend / model / revision and the
profile and station.

## Requirements

- A policy server on a GPU host (see the policy bundles' SKILL.md for the commands and
  memory; run it through `brev_gpu.sh` / the GPU pool, never on the workstation's GPU).
- The connector's `robot.get_observation` with three camera roles (top, left wrist,
  right wrist) and both arms' joints and jaw opening, and `robot.stream_dual` (the
  executor uses `robot.stream_joints` and `robot.forward_kinematics` too when the
  connector has them).
- `roboskillhub` importable in the graph's environment (the loop and the verifier live
  there).

## Limits

- Acceptance gates per implementation and embodiment (`roboskillhub capability select`):
  nothing here is offered on a robot until its twin eval records pass.
- The jaw vote uses the opening fraction, not millimetres: a very thin object can read as
  `reached`. Wire `object_position_tool` (sim truth in the twin) or point objects for the
  scene diff so two checks can agree.
- With text or id objects and no position tool, only the jaw votes and the verifier returns
  `missed` (uncertain): one sensor does not certify a pick.
- `T_TCP_object` is not measured (null).
