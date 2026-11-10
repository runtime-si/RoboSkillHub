---
name: simulating-d405-stereo-depth
description: Replaces a simulated camera's perfect depth with D405-like passive
  stereo depth computed from the camera's two rendered imagers
  (d405-stereo.depth_from_pair), keeping the true depth beside it - holes on
  untextured and glossy faces (LEGO tops, domino faces), edge dropout and
  z^2 noise appear where the real wrist D405 has them. Use when a sim check
  should exercise perception on depth as bad as the real camera's, instead
  of ground truth or an image-independent noise model.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [simulation, depth, stereo, sim-to-real, d405]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    tasks: [lego, domino, jigsaw]
    provenance:
      authors: [runtime-si]
      code: https://github.com/haosulab/SAPIEN/blob/788fd0afb009/python/py_package/sensor/stereodepth.py
      upstream_license: Apache-2.0
gap:
  allowed_tools:
    - robot.get_observation
    - d405-stereo.depth_from_pair
  required_inputs:
    camera: string
  produces_outputs:
    observation: Observation
    hole_fraction: float
    reason: string
  exit_conditions:
    depth_rendered: The camera's depth is now the stereo depth (0 = hole) and depth_true holds the renderer's depth.
    no_pair: The camera carries no rendered pair (ir_left / ir_right, or rgb / rgb_right) with intrinsics, or d405-stereo failed; the observation is returned unchanged.
  canonical_scripts:
    - stereo_depth: scripts/stereo_depth.py
  examples:
    - title: Stereo depth for the wrist camera of a sim observation
      path: examples/canonical_subgraph.json
---

# simulating-d405-stereo-depth

Sim depth is ground truth, but the real wrist D405 is passive stereo with
no projector. It loses glossy, untextured faces and object edges, and its
noise grows with z². #185 fitted an image-independent noise model to 401
real wrist frames. This skill instead computes depth the way the camera
does, from the sim's two rendered imagers, so the holes follow the texture.

The sim camera must render a rectified pair: the left imager plus one
18 mm to its right, same intrinsics, luminance as IR. Put them on the
camera as `ir_left` / `ir_right`, or as `rgb` plus `rgb_right`.

## Measured (MuJoCo renders vs the real D405 statistics of #185)

**Data:**
- 30 rendered pairs (`~/sim-wt/d405/render_pairs.py`), 10 each of LEGO
  bricks (flat colours, specular), dominoes (white faces, black pips) and
  jigsaw pieces (a printed picture), on a textured wood table.
- The camera looks down from 0.10-0.30 m, tilted up to 14 deg, with the
  real wrist D405's intrinsics.
- Both depth sets were measured with **#185's own `fit_noise.py`**, the
  script that produced the real statistics:
  - this skill's stereo depth;
  - the renderer's true depth plus #185's fitted noise model.

| statistic | real D405 (401 frames) | stereo (this skill) | #185 noise model |
| --- | --- | --- | --- |
| fine noise c in sigma = c·z² | 0.0228 /m | 0.0255 /m | 0.0255 /m |
| sigma at 15 cm | 0.51 mm | 0.57 mm | 0.57 mm |
| noise correlation | 2.39 px | **2.28 px** | 1.77 px |
| smooth error c | 0.0452 /m | 0.0166 /m | 0.0323 /m |
| dropout, all pixels | 25.8 % | **17.5 %** | 0.4 % |
| dropout at depth edges | 41 % | 24 % | 20 % |
| quantisation | 1 mm | 1 mm | 1 mm |

**By object.** Medians over frames; holes are on object tops away from
edges, and |error| is against the true depth:

| family | stereo holes | stereo median / p90 error | stereo > 5 mm outliers | model holes | model error |
| --- | --- | --- | --- | --- | --- |
| LEGO (glossy, untextured) | **9.3 %** | 1.27 / 4.04 mm | 6.3 % | 0 % | 0.81 / 2.00 mm |
| domino (white faces) | 3.2 % | 1.59 / **6.23** mm | **15.6 %** | 0 % | 0.90 / 2.16 mm |
| jigsaw (printed) | 0 % | 0.74 / 1.51 mm | 0.2 % | 0 % | 1.41 / 2.84 mm |

**What it shows:**
- **The fine noise and its correlation come out close to the real camera
  without fitting anything.** Stereo noise is 12 % above the real c, and
  its correlation is 2.28 px against the real 2.39.
- **The holes depend on the image, which the noise model cannot do.** The
  model drops almost nothing (0.4 %) and treats every object alike. Stereo
  drops 17.5 % of all pixels and 24 % at edges (real: 25.8 % and 41 %),
  and the holes and > 5 mm mismatches land on the LEGO tops and the white
  domino faces, where the real D405 struggles. Printed jigsaw faces match
  cleanly.
- About 18 % of the table is a hole in every frame: the left band the right
  imager cannot see at 7 cm range (SGBM's invalid border, about 100 px). A
  real D405 has the same left band.
- **Not matched:**
  - the smooth error (#185's 0.045 includes the wood table's relief);
  - real edge dropout is higher (41 % vs 24 %), because the renders have no
    motion blur, sensor defocus or real glare;
  - LEGO speculars in MuJoCo are not real ABS gloss.
- **Comparing the true depth values against the real frames was not
  possible:** they are different scenes, so only statistics are compared.

## Licence

The hub's own code (Apache-2.0); OpenCV does the matching. The design
follows SAPIEN's StereoDepthSensor (Apache-2.0); no code is copied.
