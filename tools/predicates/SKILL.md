---
name: predicates
description: Geometric yes/no judgments over poses, boxes and curves the graph
  already holds -- is a point inside an oriented box, does one box rest on
  another, are two points within a distance, is a body upright, has a rope gone
  through a loop -- each returning a boolean, a verdict string a router can
  branch on, and the margin it was decided by. Use when a graph must choose a
  branch from geometry it already has (a perceived OBB, an end-effector pose, a
  cable's centreline), on CPU, with no model and no simulator state.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verification, geometry, predicates, routing, cpu]
  hub:
    stage: judge
    embodiments: [any]
    version: 0.2.0
    tasks: [pick-place, insertion, sorting, cable-routing]
    provenance:
      authors: [Runtime.si]
gap:
  requires: {}
  tools:
    - predicates.point_in_obb: Whether a world point lies inside an oriented box, and by how much.
    - predicates.resting_on: Whether one box sits on top of another, from footprint overlap and the vertical gap.
    - predicates.within_distance: Whether two points are within a threshold, in 3D or in the table plane.
    - predicates.upright: Whether a body's local axis points up, within a tilt bound.
    - predicates.threaded_through: Whether a rope's centreline goes through a loop's opening, net of any pass back.
---

# Predicates

Five pure functions over `gap_core.types` values. They read nothing from the
robot or the simulator, so a graph that branches on them runs unchanged on the
real arm.

Every result carries three things:

- a boolean (`inside`, `on`, `near`, `upright`, `threaded`);
- `verdict`, the same answer as a word, so a `router` node can branch on it
  without a script in between;
- the number it was decided by (`depth_m`, `gap_m` and `xy_overlap`,
  `distance_m`, `tilt_deg`, `through_m` and `clearance_m`), so a trace shows
  how close the call was.

## Threading

`threaded_through` takes a `Centerline` (the rope) and a `FunctionalFeature`
of kind `loop` or `aperture` (the ring: `pose.position`, `axis`,
`radius_inner`). It counts passes of the centreline through the opening,
+1 along `axis` and -1 against it, so through-and-back nets zero; `passes`
counts both ways. It is threaded when the net is nonzero and both ends reach
at least `min_through_m` (default 10 mm) past the ring, so a tip that only
touches the plane does not count. The `verifying-threading` skill wraps it as
a graph's condition.

## Conventions

Metres, world frame, quaternions `wxyz`. An `OrientedBoundingBox` carries
**half**-extents, as everywhere in gap.

## Branching on a verdict

```json
"placed": {"type": "tool", "tool": "predicates.point_in_obb",
           "inputs": {"point": {"$ref": "observe.center"},
                      "obb": {"$ref": "in.container_obb"}, "margin_m": 0.01}},
"route":  {"type": "router", "inputs": {"verdict": {"$ref": "placed.verdict"}}}
```

with a conditional edge on `route` mapping `inside` and `outside` to the next
nodes.

## Limits

- `resting_on` compares the boxes' world-aligned footprints. For a box yawed
  far from the world axes that footprint is larger than the box, so overlap
  reads high; check yaw first when it matters.
- `point_in_obb` reports `depth_m` as the distance to the nearest face inside
  the box and the Euclidean distance to the box outside it (negative).
- `threaded_through` treats the opening as a flat disk and interpolates each
  crossing along the straight segment between two nodes, so a coarse
  centreline can put a pass that grazes the rim on the wrong side of it. A
  `clearance_m` near zero says the call was that close.
