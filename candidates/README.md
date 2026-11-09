# Reviewed upstream skill materials

This intake catalog contains concrete implementation material to adapt into
RoboSkillHub. It records source entrypoints, inputs and outputs, robot and sensor
assumptions, simulator privileges, code and weight licenses, and adaptation work.
It is separate from the executable registry in `tools/`, `skills/` and `policies/`.

The initial 2026-11-05 snapshot contains 30 capability records from 21 repositories:
25 records with MIT, Apache-2.0 or BSD-3-Clause code, plus 5 research-only sources.
These are capability records, not 30 independent ready-to-run robot packages.
Every source repository is pinned to a full commit. Weight links are discovery
references; they are not yet the immutable weight manifests used by `skills.lock`.

## Find a source

From this checkout after the normal package setup:

```bash
roboskillhub discover "抓取"
roboskillhub discover "推物"
roboskillhub discover "YAM" --priority P1 --json
roboskillhub discover --destination policy --json
roboskillhub discover --category deformable --json
roboskillhub discover --include-research-only --json
roboskillhub check-candidates
```

Without installing the package, the same commands work from the checkout with:

```bash
PYTHONPATH=src python -m roboskillhub.cli discover "抓取" --json
```

Discovery matches case-insensitive English or Chinese keywords separated by
spaces (OR matching). It searches names, capability descriptions and interfaces.
It is not a task planner and does not prove capability coverage. `search` and
`view` retain their existing executable-registry meaning. No candidate is copied
into a runtime view, downloaded, installed or executed by `discover`.

`--include-research-only` explicitly includes the sources whose code has
noncommercial restrictions. The default code-license grouping is not a blanket
commercial-use determination: weights, datasets, robot assets and dependencies
can have different or unresolved terms, recorded on each entry.

## Start with the current YAM workcell

P1 indicates an adaptation priority, not operational readiness:

- ENPIRE measured pose movement and RGB-D grasp proposals: compare with the
  existing YAM services; preserve measured feedback and coordinate conventions.
- ASPIRE typed API: reuse observation and execution contracts where useful,
  without importing a second task orchestrator.
- GraspGenX: a candidate grasp backend with a serving client. Configure and
  validate the actual YAM gripper; the upstream YAM support is not complete.
- TAPIR: track visible features across images; metric 3D estimates still require
  calibration, valid depth and uncertainty handling.
- ManiSkill pushing and IsaacLab drawer state machines: extract local control
  segments, replace oracle state and robot-specific control, add observable exits.
- LeRobot ACT: a local-data training path, not a selected pretrained YAM skill.

The wider catalog covers tool pulling, object reorientation, peg and charger
insertion, articulated doors, handover, Factory/FORGE assembly, motion planning,
generalist policies, towel manipulation, cloth primitives and DLO shape planning.
DLO Planning 2 is a dual-arm shape planner/MPC source, not a ready-made
cable-threading skill; its Unity/Obi assets have separate rights. Dual-arm and
force-feedback entries retain those requirements; they are not presumed to fit
the currently confirmed single-arm RGB-D workcell.

## Promote material into the executable registry

1. Inspect the pinned source and license records; select only the local ability
   needed. Preserve attribution for any code actually copied.
2. Reuse an existing bundle when the ability already exists. Geometry helpers
   remain tools; task ordering remains a recipe; learned weights use policies.
3. Define input/output types, frames, units, observation freshness, physical
   preconditions, bounded parameters, observable exits and effects on the world.
4. Replace simulator truth with available real observations where required;
   bind robot control to the actual YAM/Isaac connector and controller capabilities.
5. Package the implementation, pin model snapshots, and validate its dependencies.
6. Run independent skill tests in the declared domain, then file eval evidence
   for the exact bundle, workcell and version. Source review is not a sim/real pass.
7. Freeze the accepted skill implementations and allowed edits before graph-search
   comparisons. Record the promoted bundle in the candidate's intake history.

No paper, public checkpoint or successful upstream video on its own proves that
the resulting bundle is callable on this workcell. `check-candidates` checks
metadata integrity only. It does not contact the upstream sites or validate the
implementations, license interpretation, physics or hardware.

## Candidate format

`catalog.json` uses an independent `schema_version: 1` and a `candidates` array.
Each record includes:

- Identity: `id`, `name`, `category`, `capabilities`, `verified_date`.
- Source: `repo`, `source_url`, `source_revision`, `artifact_type`, `entrypoints`.
- Interface: `inputs`, `outputs`, `robots`, `sensors`, `sim_privilege`.
- Rights: `code_license` and a separate `weights` record, including uncertainty.
- Intake: `hub_destination`, `existing_overlap`, `adaptation_steps`, `priority`,
  `limitations`, `evidence_urls`, `intake_group`.
- State: `readiness: source-reviewed`, `invocable_in_hub: false`.

The validator deliberately rejects executable claims inside the intake catalog.
Actual runnable bundles belong to the existing registry and its validation path.
Priorities: P1 is the first adaptation batch; P2 expands task coverage; P3 needs
additional hardware, license resolution, substantial adaptation or has high overlap.

## Excluded and overlapping material

The existing open-robot-skills imports are already in this repository and were
not counted as new acquisitions. FlingBot was not included because an applicable
license was not confirmed in this review. ENPIRE's `place_with_orientation_v1`
was inspected but not promoted as a recommended primitive: in the reviewed source,
release follows the motion attempts even when lowering fails. Any future reuse
needs explicit failure handling and a separately validated contract.
