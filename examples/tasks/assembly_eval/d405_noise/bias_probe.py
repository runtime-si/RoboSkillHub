import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/home/yim/rsh-wt/d405-refit/src")
from roboskillhub.tasks.lego_hub import depth as D

spec = importlib.util.spec_from_file_location(
    "vs",
    "/home/yim/rsh-wt/d405-refit/skills/verifying-brick-seated-by-height/scripts/verify_seat.py",
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


F = D.D405_FIT
cases = {
    "clean": dict(scale=0.0, params={"quant_step_m": 1e-6}),
    "quant 1 mm only": dict(scale=0.0, params={}),
    "fine only (no quant)": dict(params={"lowfreq_coeff_per_m": 0.0, "quant_step_m": 1e-6}),
    "smooth only (no quant)": dict(params={"sigma_coeff_per_m": 0.0, "quant_step_m": 1e-6}),
    "full refit": dict(params={}),
    "full refit, fine x0.5": dict(params={"sigma_coeff_per_m": F["sigma_coeff_per_m"] / 2}),
}
for name, kw in cases.items():
    hs, ts = [], []
    for s in range(8):
        d = D.add_fitted_noise(clean, K, np.random.default_rng(s), **kw)
        m = V.measure(pts(d), seat, 0.0, 0.016, 0.032, 0.008, 0.0017, 0.0096)
        if m.get("error"):
            continue
        hs.append(m["height_error"] * 1e3)
        ts.append(m["tilt_deg"])
    print(
        f"{name:28s} height err mm {np.mean(hs):+.2f} (sd {np.std(hs):.2f}), tilt deg {np.mean(ts):.2f}, n {len(hs)}"
    )
