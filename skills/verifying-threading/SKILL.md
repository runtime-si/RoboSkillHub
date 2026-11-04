---
name: verifying-threading
description: Decides whether a rope, cable or thread has gone through a loop -- a ring, eyelet or hook -- from the centreline and the loop the graph already holds. Counts passes through the opening net of any pass back, requires both ends to reach past the ring, and reports the margins. Use when a threading policy must choose, after an attempt, between moving on and trying again, on a real arm or in any simulator.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [verification, condition, cable, rope, deformable, threading, cpu]
  hub:
    stage: verify
    embodiments: [any]
    version: 0.1.0
    tasks: [cable-routing]
    provenance:
      authors: [Runtime.si]
gap:
  allowed_tools:
    - predicates.threaded_through
  required_inputs: {rope: string, loop: FunctionalFeature}
  produces_outputs: {threaded: bool, crossings: int, through_m: float, clearance_m: float, detail: string}
  exit_conditions:
    threaded: The rope passes through the loop's opening a net nonzero number of times and both ends reach at least `min_through_m` past it.
    not_threaded: It does not -- it misses the opening, went through and came back, only reaches the plane by a tip, or there was no rope or loop to judge; `detail` says which.
  canonical_scripts:
    - verify_threading: scripts/verify_threading.py
  streaming: false
---

# verifying-threading

Has the rope gone through the ring? A condition, as a skill.

## Why this is its own skill

Every node of a graph can be a skill, and a condition is no exception. A
threading policy has to ask "is it through?" before it lets go, and again
before it decides to retry; written inline in each graph, that question gets
answered a different way each time. Here it is answered once, with exits a
router branches on and margins a trace can be read by.

The answer is topological, not a distance. A rope tip resting against the
far side of a ring is closer to "through" than a rope that went through and
came back, yet only one of them is threaded. So the skill counts signed passes
through the opening (+1 along the loop's axis, -1 against it) and requires the
net to be nonzero, then requires both ends to reach `min_through_m` past the
ring so that a tip touching the plane does not count.

It reads no camera and no simulator state. The rope comes from
`perceiving-deformable-linear-objects` and the loop from
`perceiving-functional-features`, so the same verdict runs on a real arm and in
any simulator, from the same inputs.

## When to use

- After a threading motion, once the rope has been perceived again: route
  `threaded` onward and `not_threaded` back to the attempt or to a recovery.
- As a precondition before pulling a rope tight, to avoid pulling a rope that
  is not through.

## When NOT to use

- Routes judged by which side of each post the cable lies on: use
  `verifying-a-cable-route`.
- Inserting a rigid part into a hole: the part has a pose, not a centreline;
  use `verifying-placement` or `predicates.point_in_obb`.

## State flow

```text
rope (centreline JSON) ──► points; fewer than two ──► not_threaded (no rope)
loop (FunctionalFeature) ──► centre, axis, radius_inner; no radius ──► not_threaded (no loop)
predicates.threaded_through:
    every segment crossing the loop's plane inside radius_inner is a pass, ±1 by direction
    net ≠ 0 and both ends ≥ min_through_m past the ring ──► threaded
    otherwise ──► not_threaded
```

## Inputs

- `rope` -- the centreline as JSON text, as `perceiving-deformable-linear-objects`
  writes `rod`: a list of `[x, y, z]` points in order along the rope, or a
  `Centerline` object with `points` (and `radius_m`, used for clearance). A
  `Centerline` value is accepted as is.
- `loop` -- a `FunctionalFeature` of kind `loop` or `aperture`, as
  `perceiving-functional-features` returns it: `pose.position` is the centre,
  `axis` the normal of the opening, `radius_inner` its radius.
- `min_through_m` (0.01) -- how far past the ring both ends must reach.

## Outputs

- `exit` -- `threaded` | `not_threaded`. **The router field is `exit`.**
- `threaded` -- the same verdict as a bool.
- `crossings` -- the net signed number of passes; `0` after through-and-back.
- `through_m` -- the shorter of the two ends beyond the ring.
- `clearance_m` -- room between the rope's surface and the rim at the tightest
  pass, or minus the distance by which the rope misses the opening.
- `detail` -- passes, net, length through and clearance, in words.

## Required end states

| End state | Meaning |
|---|---|
| `threaded` | Through, with both ends past the ring. Move on. |
| `not_threaded` | Not through: read `detail`, `crossings` and `clearance_m` for how it missed. |

## Limits

- The opening is a flat disk of `radius_inner`. A hook that is open on one
  side is treated as a closed ring.
- The verdict is only as good as the centreline. A pass that grazes the rim
  can land on either side of it; `clearance_m` near zero says so.
