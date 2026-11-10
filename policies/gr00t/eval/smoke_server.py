"""Load a GR00T backend in-process, check where its backbone weights came from, and time act().

    python eval/smoke_server.py --profile yam-molmoact2 --backbone qwen3-vl-2b [--calls 30]

Prints one JSON line: load time, GPU memory, per-call latency (mean / p50 / p95), the action
chunk's shape and range, and - for the stand-in backbone - the largest difference between every
backbone tensor in the model and the checkpoint's safetensors (0 means the checkpoint's own
weights are in use). Runs in the Isaac-GR00T environment on a GPU host.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


def backbone_parity(backend: server.N17Backend) -> dict:
    import torch
    from safetensors import safe_open

    path = backend._local(backend.checkpoint, backend.revision)
    params = dict(backend.policy.model.named_parameters())
    worst, compared, missing = 0.0, 0, []
    for shard in sorted(path.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if not key.startswith("backbone."):
                    continue
                if key not in params:
                    missing.append(key)
                    continue
                a = f.get_tensor(key).to(params[key].device, torch.float32)
                worst = max(worst, float((a - params[key].float()).abs().max()))
                compared += 1
    model_backbone = [k for k in params if k.startswith("backbone.")]
    return {
        "backbone_tensors_compared": compared,
        "backbone_params_in_model": len(model_backbone),
        "checkpoint_keys_not_in_model": len(missing),
        "max_abs_diff": worst,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="yam-molmoact2")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--backbone", default="canonical")
    p.add_argument("--calls", type=int, default=30)
    a = p.parse_args()
    import torch

    t0 = time.time()
    backend = server.N17Backend(
        server.PROFILES[a.profile], checkpoint=a.checkpoint, backbone=a.backbone
    )
    svc = server.Service(backend)
    rng = np.random.default_rng(0)
    images = {r: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for r in server.ROLES}
    joints = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    lat, out = [], None
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.calls):
        t = time.perf_counter()
        out = svc.handle(
            "act",
            {
                "instruction": "Put the cube in the box.",
                "images": images,
                "joints": joints,
                "marks": [{"kind": "grasp", "camera": "top", "uv": [320, 240]}] if i % 2 else [],
                "t": i / 3.0,
            },
        )
        lat.append((time.perf_counter() - t) * 1e3)
    acts = np.asarray(out["actions"])
    rec = {
        "profile": a.profile,
        "backbone": a.backbone,
        "info": svc.info(),
        "wall_load_s": round(time.time() - t0 - sum(lat) / 1e3, 1),
        "latency_ms": {
            "first": round(lat[0], 1),
            "mean_after_first": round(float(np.mean(lat[1:])), 1),
            "p50": round(float(np.percentile(lat[1:], 50)), 1),
            "p95": round(float(np.percentile(lat[1:], 95)), 1),
        },
        "server_timing_ms_last": out["timing_ms"],
        "gpu_mem_mb": svc.gpu_mem_mb(),
        "actions_shape": list(acts.shape),
        "actions_min": acts.min(0).round(3).tolist(),
        "actions_max": acts.max(0).round(3).tolist(),
    }
    if a.backbone != "canonical":
        rec["parity"] = backbone_parity(backend)
    print("SMOKE " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
