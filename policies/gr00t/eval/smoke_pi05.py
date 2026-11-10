"""Load the pi05 backend in-process, time act() through the server's Service, and (optionally) score
it open-loop on a held-out MolmoAct2-YAM episode.

    python eval/smoke_pi05.py --profile yam-pi05-molmoact2 [--calls 30] \
        [--dataset allenai/19012026-block-13 --episode 0 --stride 30]

Prints one JSON line: load time, GPU memory, per-call latency (mean / p50 / p95) on blank 640x480
views, the chunk shape, and for --dataset the open-loop MSE / MAE of the predicted chunk against the
recorded actions (raw joint units; the card's 0.00140 is in quantile-normalised units, so compare
only the order of magnitude), plus the first rows of one chunk next to the ground truth. Runs in a
LeRobot 0.7 environment with the pi extra on a GPU host.
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

ROLE_KEYS = {"top": "observation.images.top", "left_wrist": "observation.images.left",
             "right_wrist": "observation.images.right"}


def open_loop(service: server.Service, repo: str, episode: int, stride: int, horizon: int) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo, episodes=[episode])
    n = len(ds)
    task = ds[0].get("task") or "do the task"
    errs, maes, sample = [], [], None
    for start in range(0, n - horizon, stride):
        item = ds[start]
        images = {role: (item[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                  for role, k in ROLE_KEYS.items()}
        service.backend.reset()
        out = service.handle("act", {"instruction": task, "images": images,
                                     "joints": item["observation.state"].numpy().tolist()})
        pred = np.asarray(out["actions"])[:horizon]
        gt = np.stack([ds[start + i]["action"].numpy() for i in range(len(pred))])
        errs.append(float(np.mean((pred - gt) ** 2)))
        maes.append(float(np.mean(np.abs(pred - gt))))
        if sample is None:
            sample = {"pred_rows_0_5_10": pred[[0, 5, 10]].round(3).tolist(),
                      "gt_rows_0_5_10": gt[[0, 5, 10]].round(3).tolist()}
    hold = []
    for start in range(0, n - horizon, stride):
        s = ds[start]["observation.state"].numpy()
        gt = np.stack([ds[start + i]["action"].numpy() for i in range(horizon)])
        hold.append(float(np.mean((gt - s) ** 2)))
    return {"dataset": repo, "episode": episode, "frames": n, "task": task, "chunks": len(errs),
            "mse": float(np.mean(errs)), "mae": float(np.mean(maes)),
            "hold_still_mse": float(np.mean(hold)), **(sample or {})}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="yam-pi05-molmoact2")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--revision", default=None)
    p.add_argument("--tokenizer", default="")
    p.add_argument("--calls", type=int, default=30)
    p.add_argument("--dataset", default="")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=30)
    args = p.parse_args()
    ns = argparse.Namespace(backend="pi05", profile=args.profile, checkpoint=args.checkpoint,
                            revision=args.revision, device="cuda", tokenizer=args.tokenizer,
                            wire_frame="policy")
    import torch

    service = server.Service(server.make_backend(ns))
    rng = np.random.default_rng(0)
    views = {r: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for r in server.ROLES}
    req = {"instruction": "Pick up the grey block and put it in the blue box.", "images": views,
           "joints": [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]}
    for _ in range(3):
        service.handle("act", req)
    torch.cuda.reset_peak_memory_stats()
    lat, out = [], None
    for _ in range(args.calls):
        t0 = time.perf_counter()
        out = service.handle("act", req)
        lat.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(out["actions"])
    res = {"info": service.info(), "calls": args.calls,
           "latency_ms": {"mean": round(float(np.mean(lat)), 1), "p50": round(float(np.percentile(lat, 50)), 1),
                          "p95": round(float(np.percentile(lat, 95)), 1)},
           "timing_ms_last": out["timing_ms"], "gpu_mem_mb": service.gpu_mem_mb(),
           "chunk": list(a.shape), "action_min": a.min(0).round(3).tolist(), "action_max": a.max(0).round(3).tolist()}
    print(json.dumps(res), flush=True)
    if args.dataset:
        print(json.dumps({"open_loop": open_loop(service, args.dataset, args.episode, args.stride, a.shape[0])}))


if __name__ == "__main__":
    main()
