---
name: aligning-parts-by-connector-constraints
description: Turns connector statements about a LEGO or DUPLO build - "part
  P's tubes on base B's studs, P's stud (a, b) over B's stud (i, j), turned r"
  - into exact plate cells, layers and world poses, solving them in dependency
  order on the stud lattice (Manual2Skill++'s connector-aware pose alignment,
  which for studs and tubes reduces to integer lattice arithmetic). A part
  named by two statements (a bridge over two bases) must come out the same
  from both. Unsupported parts (on none of the base's studs), overlaps,
  cycles, unknown bases and parts off the plate are reported. Outputs the
  steps planning-brick-assembly-order reads and per-part cells for
  computing-brick-pose-from-stud-grid. Use when a build is described at the
  connector level (by a VLM reading a manual page, or a person saying "put the 1x2
  on the left two studs of the red 2x4") rather than as an LDraw file, and
  needs exact stud-grid poses.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: planning
  tags: [lego, duplo, assembly, connectors, stud-grid, manual, cpu]
  hub:
    stage: plan
    embodiments: [any]
    version: 0.1.0
    tasks: [assembly, lego]
    provenance:
      authors: [runtime-si]
      paper: "Manual2Skill++ (Tie et al., ICRA 2026); https://github.com/crtie/Manual2SkillPP_official has no licence: reimplemented, no code copied"
gap:
  allowed_tools: []
  required_inputs:
    statements_json: string
  # Or statements = [{part, on, at: [i, j], rotation: 0/90/180/270, part_stud: [a, b],
  # size: "AxB", height_units: 3 (brick) / 1 (plate)}]; optional parts = [{id, size,
  # height_units, ...extra keys kept}], plate = {id, studs: [nx, ny], origin: Vec3 of cell
  # (0, 0)'s corner on the plate top, yaw_deg}.
  produces_outputs:
    parts_json: string
    steps_json: string
    order_json: string
    problems_json: string
    n_parts: int
  exit_conditions:
    poses_solved: Every part got cells, a layer and a rotation (and a world pose when the plate's origin is given); steps_json lists them in dependency order.
    constraints_inconsistent: A statement is malformed, names an unknown base, forms a cycle, puts a part on fewer than min_support of its base's studs, overlaps another part, hangs off the plate, or two statements for one part disagree; problems_json names the first one and the statement index.
  canonical_scripts:
    - align_connectors: scripts/align_connectors.py
  examples:
    - title: Solve a connector-level build description into stud-grid cells
      path: examples/canonical_subgraph.json
---

# aligning-parts-by-connector-constraints

A manual page or a person describes a build by its connections: this brick
on those studs of that one, turned this way. Manual2Skill++ turns such
connector statements into 6D part poses by aligning connector frames. For
studs and tubes every connector frame is a lattice node with a vertical axis,
so the alignment is exact integer arithmetic; this skill does that part.

## Statements

`{part, on, at: [i, j], rotation, part_stud: [a, b]}`: part's stud
`part_stud` (default its stud (0, 0)) goes over the base's stud `at`, in the
base's own stud coordinates (u along its first size number, v along its
second), with the part turned `rotation` degrees (0, 90, 180, 270) about +z
relative to the base. `on: "plate"` (or `plate.id`) is the baseplate,
coordinates (0, 0) at its corner. `size` "AxB" (or in `parts`),
`height_units` 3 for a brick and 1 for a plate.

## How it solves

1. Read the statements; a part's size and height come from `parts`, else
   from its first statement (a later different size is a conflict).
2. Repeatedly place every part whose bases are all placed (cycles and
   unknown bases are reported): the part's stud (0, 0) goes to
   `base.stud(at - R_r(part_stud))`, its rotation is the base's plus `r`, its
   bottom on the base's top (base z + base height).
3. Checks: at least `min_support` (1) of its studs sit over the base's
   studs; the same cells and height (rotation modulo the brick's symmetry)
   from every statement that names it; no shared cell with a part whose
   height range overlaps; inside the plate's stud area.
4. Out: per part the plate cell of its footprint's corner (`cell` x, y,
   layer, rotation), spans, `z_units`, `height_units`, kind, step; with
   `plate.origin` the footprint centre's world position and yaw (cells to
   metres by the pitch, 8 mm LEGO / 16 mm DUPLO); `steps_json` is one step
   per part in solve order, in planning-brick-assembly-order's format.

## Checked

No arm motion, no robot tools. Tests
(`tests/bundles/test_aligning_parts_by_connector_constraints.py`): a turned
brick on a brick, a part-stud anchor on a turned base, a bridge spelled from
both bases (agreeing and disagreeing), each inconsistency kind, one-stud
overhangs and `min_support`, world poses on a turned plate, plates stacking
by their own height, and **two LDraw models round-tripped**: the hub's brick
plans for pyramid.ldr (13 bricks) and the 1869 South African Flag (82
bricks) re-written as connector statements (each brick on the plate or on a
brick under it, at that base's stud of its own stud (0, 0)) and solved back -
every brick's cells and height identical to the plan - then fed to
planning-brick-assembly-order, which places all of them (finger check off).

## Limits

- Stud-to-tube connections only (no SNOT, Technic pins, hinges, jumper
  half-stud offsets).
- The statements must already be in stud coordinates; turning a manual page
  into them is a VLM's job (extracting-a-subassembly-tree-from-manual-pages
  or a prompt), and its mistakes come out as `constraints_inconsistent` only
  when they are inconsistent.
- No stability analysis (checking-design-connectivity-and-collisions).

## Provenance

The connector-alignment idea is Manual2Skill++'s (Tie et al., ICRA 2026).
Its repository has no licence, so nothing was copied; for LEGO the solver is
a few lines of lattice arithmetic written here. The lattice and cell
conventions are computing-brick-pose-from-stud-grid's and
planning-brick-assembly-order's.
