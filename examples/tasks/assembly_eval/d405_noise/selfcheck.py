import json, subprocess, sys, tempfile, numpy as np
sys.path.insert(0, "/home/yim/rsh-wt/d405-noise/src")
from roboskillhub.tasks.lego_hub import depth as D
rng = np.random.default_rng(0)
tmp = tempfile.mkdtemp()
for i, z in enumerate(np.linspace(0.10, 0.20, 40)):
    yy, xx = np.mgrid[0:480, 0:640]
    clean = (z + 0.00002 * (xx - 320) + 0.00001 * (yy - 240)).astype(np.float32)
    np.savez(f"{tmp}/f{i:03d}_wrist.npz", depth_m=D.add_fitted_noise(clean, D.D405_K, rng), K=D.D405_K)
    np.savez(f"{tmp}/old{i:03d}.npz", depth_m=D.add_noise(clean, D.D405_K, rng), K=D.D405_K)
for tag, pat in (("fitted model", f"{tmp}/f*.npz"), ("old add_noise", f"{tmp}/old*.npz")):
    out = subprocess.run([sys.executable, "/home/yim/sim-wt/d405/fit_noise.py", pat, f"{tmp}/o.json"], capture_output=True, text=True)
    d = json.load(open(f"{tmp}/o.json"))
    print(tag, {k: d[k] for k in ("sigma_coeff_per_m", "corr_sigma_px", "lowfreq_coeff_per_m", "quant_step_m")})
