import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/home/yim/rsh-wt/seat-debias/src")
from roboskillhub.tasks.lego_hub import depth as D

spec = importlib.util.spec_from_file_location(
    "vs",
    "/home/yim/rsh-wt/seat-debias/skills/verifying-brick-seated-by-height/scripts/verify_seat.py",
)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
PLATE = {
    "position_m": [0.3, 0.0, 0.0032],
    "quaternion_xyzw": [0, 0, 0, 1],
    "class": "baseplate",
    "geometry": {"size_m": [0.256, 0.256, 0.0032]},
}
seat = np.array([0.308, 0.016, 0.0032])
objs = {
    "plate": PLATE,
    "b": {"position_m": list(seat), "quaternion_xyzw": [0, 0, 0, 1], "class": "lego_brick"},
}
cam = np.eye(4)
cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
cam[:3, 3] = [0.358, 0.016, 0.0032 + 0.0096 + 0.20]
clean = D.render(objs, cam)
K = D.D405_K


def pts(depth):
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    return (cam @ pc)[:3].T


def run(est, high_mm, kw, n=12):
    b = dict(objs["b"])
    b["position_m"] = [seat[0], seat[1], seat[2] + high_mm / 1e3]
    c = D.render({"plate": PLATE, "b": b}, cam)
    hs, ts = [], []
    for s in range(n):
        d = D.add_fitted_noise(c, K, np.random.default_rng(100 + s), **kw)
        m = V.measure(
            pts(d), seat, 0.0, 0.016 - 0.0002, 0.032 - 0.0002, 0.008, 0.0017, 0.0096, "auto", est
        )
        if m.get("error") or "height_error" not in m:
            continue
        hs.append(m["height_error"] * 1e3)
        ts.append(m["tilt_deg"])
    return np.mean(hs), np.std(hs), np.max(ts), len(hs)


for name, kw in {
    "clean": dict(scale=0.0, params={"quant_step_m": 1e-6}),
    "quant only": dict(scale=0.0, params={}),
    "fitted (refit)": dict(params={}),
    "fitted x2": dict(scale=2.0, params={}),
}.items():
    for est in ("two_layer", "stud_mask"):
        a = run(est, 0.0, kw)
        b = run(est, 1.7, kw)
        print(
            f"{name:15s} {est:10s} seated {a[0]:+.2f}+-{a[1]:.2f} mm (tilt max {a[2]:.2f}) | 1.7 mm high {b[0]:+.2f}+-{b[1]:.2f} mm  n={a[3]},{b[3]}"
        )
