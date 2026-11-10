---
name: d405-stereo
description: Depth the way a RealSense D405 measures it, from a rendered
  passive IR stereo pair (18 mm baseline, no projector) - OpenCV semi-global
  matching with uniqueness and left-right checks, sub-pixel disparity, speckle
  filtering, image noise and the stream's 1 mm quantisation - so simulated
  depth gets the real camera's holes on untextured / glossy faces, fattened
  edges and z^2 noise. CPU only. Use when a sim station must feed perception
  skills depth that fails where the wrist D405 fails.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, depth, stereo, simulation, sim-to-real, d405]
  hub:
    stage: perceive
    embodiments: [any]
    version: 0.1.0
    provenance:
      authors: [runtime-si]
      code: https://github.com/haosulab/SAPIEN/blob/788fd0afb009/python/py_package/sensor/stereodepth.py
      upstream_license: Apache-2.0
      paper: https://doi.org/10.1109/TPAMI.2007.1166
gap:
  requires: {}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "d405-stereo"]
    protocol: stdio-msgpack
  tools:
    - d405-stereo.depth_from_pair: D405-like passive stereo depth (holes, edge fattening, z^2 noise) from a rendered rectified left/right pair.
---

# d405-stereo

`d405-stereo.depth_from_pair(left, right, intrinsics, baseline_m=0.018,
min_depth_m=0.07, max_depth_m=0.5, quant_m=0.001, ir_noise=True, seed=0)`
returns `{depth [H][W] m, 0 = hole; hole_fraction; disparity_range_px;
seconds}`.

The pair must be rectified: same intrinsics, with `right` offset +x by the
baseline. Render both D405 imagers in the sim (luminance as IR), with no
projector: the D405 has none.

**How it matches.** OpenCV `StereoSGBM`:
- the disparity range is set from the depth range (7-50 cm, the datasheet);
- 7 px blocks, P1 / P2 = 8 / 32 · block²;
- uniqueness 10, left-right check 1 px, speckle filter 100 px / 2;
- 16-step sub-pixel disparity.

Before matching, read plus shot-like noise is added to both images. After
matching, depth is z = f·b/d, clipped to the range and rounded to 1 mm (the
station stream's step measured in #185).

It follows the design of SAPIEN's StereoDepthSensor / SimSense (census SGM
on the GPU), but no code is copied: SGBM is OpenCV's. SimSense (MIT) would
be the GPU path for real-time rates. This bundle runs on CPU in
0.11 s per 640 x 480 pair (measured).

**Measured:** see `skills/simulating-d405-stereo-depth`.
