# Capabilities

A capability is one contract (`capabilities/<name>.yaml`) with several implementations
behind it: programs, hub skills, policies. Every implementation returns the same outputs and
exits; a shared verifier decides the exit from measurements, and an implementation is only
offered on a robot after an acceptance run there passed the capability's gate.

```bash
roboskillhub capability list
roboskillhub capability show pick_object
roboskillhub capability select pick_object --embodiment yam_i2rt            # sim gate
roboskillhub capability select pick_object --embodiment yam_i2rt --setting real --why
```

`select` exits 1 when nothing passed; `--why` also lists the rejected implementations with
the record and the gate line each one missed; `--json` prints every judgement.

## Definition format

| key | meaning |
|---|---|
| `capability`, `version`, `description` | the name matches the file name |
| `inputs`, `outputs` | field -> `{type, required, ...}` |
| `exits` | exit -> what it means; the verifier returns exactly one of these |
| `preconditions` | names the harness measures before the attempt; a false one exits `refused` |
| `timeout_s` | an attempt past it exits `timeout`, whatever it achieved |
| `verifier` | the shared verifier (`verify_pick` for `pick_object`) |
| `implementations` | `{id, kind: program\|skill\|policy, bundle}`; `bundle` is where its eval records are filed (`evals/<bundle>/`; default: the id after any `policy:` prefix) |
| `acceptance` | per setting (`sim`, `real`): `min_episodes`, `min_success_rate`, `max_false_picked_rate` |

`roboskillhub check` validates every file under `capabilities/`.

## Selection

For each implementation the selector takes the **newest** eval record with this
`capability`, `implementation`, `embodiment` and `setting`, and re-judges its numbers
against the gate; the record's own `passed` is not trusted. A record whose episodes are not
all truth-labelled cannot pass (its false-`picked` rate is unknown). Sim and real are
separate: a twin pass never offers an implementation for `--setting real`.

## Acceptance records

```python
from roboskillhub.capability.records import Episode, build_record, write_record
from roboskillhub.capability.spec import load_capability

spec = load_capability("pick_object", root)
eps = [Episode.from_verdict(verdict, truth_held=sim_says_held, seed=s) for ...]
rec = build_record(spec, "program-grasp", embodiment="yam_i2rt", setting="sim",
                   episodes=eps, suite="twin/pick_object", evidence="<url or path>")
write_record(root, rec)   # evals/program-grasp/yam_i2rt-sim-pick_object-<date>.json
```

The record is a SPEC.md eval record plus `capability`, `implementation`, `verifier`,
`truth`, `labelled`, `false_picked`, `claim_disagreements`, `exits` and `gate_failures`.
`successes` counts verifier `picked` exits the truth label does not contradict.

## `verify_pick`

```python
from roboskillhub.capability import PickObservation, verify_pick

v = verify_pick(PickObservation(gripper=stall_out, lift=samples, scene=diff_out), claim=impl_exit)
v.exit, v.reason, v.uncertain, v.votes, v.evidence, v.privileged, v.claim_agrees
```

Each check votes `held`, `empty`, `lost` (held, then not) or abstains:

| check | source | input |
|---|---|---|
| gripper | tracking-gripper-opening-stall | its output, or a raw closing trace (`gripper_trace_m` + `gripper_target_m`) run through its tracker; `gripper_after_lift_m` closing > 3 mm from the stall is `lost` |
| scene | checking-pick-by-scene-diff | its output, or `before_rgb`/`after_rgb`/`source_box_px` through its region numbers (no VLM) |
| lift | the harness | `lift`: `{"tcp", "object"}` positions over a lift of at least 10 mm; follows at >= 0.7 of the TCP travel within 15 mm; `lift_source: sim_truth` marks the verdict `privileged` |
| wrist | a wrist re-measure | `{"held": bool}` or a route (`measured` / `not_seen`) |

Rules, in order: a false precondition is `refused`; over `timeout_s` is `timeout`; a claim of
`unreachable` / `refused` stands only when nothing physical is visible; any `lost` vote, or
jaws stopped on the part while the lift or wrist check says empty, is `slipped`; `picked`
needs two `held` votes and no `empty`; only `empty` votes is `missed`. Otherwise the VLM
(`vlm: {"held": bool}`) may break the tie toward `picked` when nothing votes `empty`; the
rest is `missed` with `uncertain: true`. The implementation's claim of `picked`, `missed`
or `slipped` is recorded (`claim_agrees`) and never consulted.
