---
name: estimating-relative-rotation-and-symmetry
description: How far a part is turned in the image plane between a reference
  view and the current view (both from the wrist camera at a similar height),
  and modulo what -- from the two masks and the pixels inside them. The
  outline gives the turn (principal axis for elongated parts, minimum-area
  rectangle for square ones), known modulo the part's symmetry (180 deg for a
  2 x 4 brick, 90 for a 2 x 2). For a part whose look breaks its outline's
  symmetry (a face-up domino's pips, a printed piece), the rotated reference
  pixels pick the full turn, or the skill says the turns look alike. Routes
  rotation_found, symmetric_ambiguous or low_confidence. Use before a wrist
  turn to match a reference, to choose the shorter of equivalent wrist turns,
  or to check a placed part's yaw; camera-only.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, orientation, rotation, symmetry, lego, domino, jigsaw, camera-only]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [pick-place, reorientation, sorting]
    tuned_on: [yam-left-wrist-d405-renders]
    provenance:
      authors: [runtime-si]
      code: own code (OpenCV, numpy)
      upstream_license: Apache-2.0
      paper: "Evaluated and rejected as the backend: Orient Anything V2 (SpatialVision/Orient-Anything-V2 @ 73b11c9, CC-BY-4.0; weights Viglong/OriAnyV2_ckpt), arXiv 2412.18605 / V2 2025"
gap:
  allowed_tools: []
  required_inputs:
    reference_frame: CameraFrame
    reference_mask: Mask
    frame: CameraFrame
    mask: Mask
  # Optional: symmetry_order (int: 1 face-up domino / printed part, 2 elongated brick or face-down
  # domino, 4 square brick; default = the outline's), outline_order (2 / 4; default measured).
  produces_outputs:
    rotation_deg: float
    period_deg: float
    symmetry_order: int
  exit_conditions:
    rotation_found: The turn from the reference to the current view, in (-period/2, period/2], known modulo period_deg (360 when the part's look resolved every turn; 180 / 90 for a part that looks the same after a half / quarter turn).
    symmetric_ambiguous: The caller said the part is less symmetric than its outline, but its outline turns look alike in these pixels (within 0.05 correlation); the turn is reported modulo the outline's period only.
    low_confidence: A mask has under 60 px, or the best appearance match is under 0.3 correlation (another part, occluded, relit); rotation_deg is the outline's turn, unconfirmed.
  canonical_scripts:
    - relative_rotation: scripts/relative_rotation.py
  examples:
    - title: Turn of a domino since its reference wrist view, pips resolving the half-turn
      path: examples/canonical_subgraph.json
---

# estimating-relative-rotation-and-symmetry

```python
r = run(
    ctx, reference_frame=ref_cam, reference_mask=ref_mask, frame=cam, mask=mask, symmetry_order=1
)  # a face-up domino: the pips break the 180-degree symmetry
if r["route"] == "rotation_found":
    turn = r["rotation_deg"]  # current = reference turned by this, clockwise + in the image
    # any turn + k * r["period_deg"] is equivalent: take the one inside the wrist's limits
```

1. Outline: elongation sqrt(lambda1 / lambda2) of each mask's pixel covariance.
   Over 1.3 the outline is 2-fold: angle = principal axis, mod 180. Otherwise
   it is 4-fold: angle = the side of `cv2.minAreaRect`, mod 90. The turn is
   the difference, folded into the period.
2. `symmetry_order` (default: the outline's) says how many turns look the
   same. Equal to the outline's: done, `rotation_found` modulo 360 / order.
3. Lower (a face-up domino, a printed part): each of the outline's candidate
   turns (refined +-4 deg in 0.5 deg steps) is tried by rotating the
   reference patch about its centroid and correlating it with the current
   patch inside both masks (normalised, grey). The best wins if it beats the
   next by more than 0.05; otherwise `symmetric_ambiguous`.

The skill only uses appearance when told the part's look is asymmetric.
Under directional light a plain 2 x 4 brick turned by 180 deg looks
different in its shading, and that is not information about the part:
asked to pick among a plain brick's half-turns by appearance, the
correlation chose the wrong one in about half the pairs, with clear
margins (0.3-0.8). The symmetry of a known part comes from its catalogue
entry (domino face-up 1, face-down 2; brick 1 x N or 2 x N 2; N x N 4).

## Measured

80 pairs of the same part in two wrist views (left wrist D405 intrinsics, 0.15-0.25 m over the
table, up to 12 deg tilt, random camera yaw, random turn between the views). Parts: LEGO bricks
and plates 1 x 1 - 2 x 4, DUPLO 2 x 2 - 2 x 4, dominoes face-up (pips) and face-down. They were
rendered in MuJoCo and composited on real wrist frames, with the true turn from the projected
part axis; see tools/README.md for the renders. Error = |reported - true| modulo the reported
period. Reproduce with `tests/bundles/_relative_rotation_eval.py`.

| method | masks | within 5 deg | within 10 deg | median error | face-up dominoes: half-turn resolved | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| **this skill** (catalogue symmetry) | rendered | **80 / 80** | 80 / 80 | **0.34 deg** | **10 / 12** right, 2 ambiguous, 0 wrong | none |
| **this skill** | SAM 2.1 base-plus from the box (mask IoU 0.98) | 78 / 80 | 79 / 80 | 0.46 deg | 11 / 12 resolved: 10 right, 1 wrong; 1 ambiguous | none |
| this skill, outline only (symmetry not given) | rendered | 80 / 80 | 80 / 80 | 0.34 deg | (reports modulo 180) | none |
| Orient Anything V2, crops masked to white | rendered | 53 % | 72 % | 4.1 deg | 1 / 6 half-turns | 7.2 GB |
| Orient Anything V2, raw crops | -- | 8 % | 14 % | 40 deg | 0 / 6 | 7.2 GB |

- By outline: elongated parts 0.56 deg median, square parts 0.37 deg (the
  rectangle fit; the principal axis gave 6 deg on them).
- Orient Anything V2's symmetry order on masked crops: 17 of 19 square
  parts right, but 45 of 49 2-fold parts called 4-fold. It is trained on
  object-level renders; a 2-3 cm part seen from above is out of its domain.
  It was not ported.
- Time on the CPU: 2.4 ms per pair (median; 2.6 ms p90) from the
  outline, 10 ms (14 ms p90) with the appearance test.

## Limits

- Image-plane turn: the camera must look roughly down on the part (tilt
  up to 12 deg tested) at a similar height in both views. A big height
  change scales the patch; the outline turn is unaffected, the appearance
  test degrades.
- Rendered parts on real backgrounds, not real photographs of the parts.
- A part touching another: segment it alone first (its mask decides the
  outline).
