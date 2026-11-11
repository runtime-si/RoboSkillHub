"""GPU smoke test: load time, per-chunk latency over HTTP, GPU memory, and parity with AllenAI's server.

    python eval/smoke_server.py --official /path/to/host_server_yam.py --out eval/out/smoke.json

Inputs are the model card's sample (assets/sample_{top,left,right}_rgb.png at the pinned revision, its
14-D state and task). Parity: the official ``host_server_yam.Policy`` (github.com/allenai/molmoact2
examples/yam, loaded from --official, its in-place modeling patch disabled: neither needle matches the
pinned revision) and this bundle's backend each predict with the same torch seed; the flow-matching
noise then matches and the chunks should agree up to bf16 kernel differences. Seeds 0 vs 1 give the
spread the noise alone causes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import server as srv  # noqa: E402

TASK = (
    "Place cups and plate in dishwasher rack, dispose of food waste, and organize remaining items."
)
STATE = np.array(
    [
        -0.06656748056411743,
        0.014686808921396732,
        0.016594186425209045,
        -0.08602273464202881,
        -0.014686808921396732,
        0.13904783129692078,
        0.9922363758087158,
        0.19512474536895752,
        0.010872052982449532,
        0.010872052982449532,
        -0.06771191209554672,
        -0.07305257022380829,
        -0.08945601433515549,
        0.9888537526130676,
    ],
    dtype=np.float32,
)


def sample_images(prof):
    from huggingface_hub import hf_hub_download
    from PIL import Image

    ck = prof["checkpoint"]
    return {
        role: np.asarray(
            Image.open(
                hf_hub_download(
                    ck["repo"], f"assets/sample_{name}_rgb.png", revision=ck["revision"]
                )
            ).convert("RGB")
        )
        for role, name in zip(srv.ROLES, ("top", "left", "right"), strict=True)
    }


def post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def npy(a):
    import base64

    return {
        "npy": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode(),
        "shape": list(a.shape),
        "dtype": "uint8",
    }


def stats(ms):
    ms = np.asarray(ms)
    return {
        "n": len(ms),
        "mean": round(float(ms.mean()), 1),
        "p50": round(float(np.median(ms)), 1),
        "p90": round(float(np.percentile(ms, 90)), 1),
        "max": round(float(ms.max()), 1),
    }


def diff(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_abs_joints": float(np.delete(d, [6, 13], axis=1).max()),
        "max_abs_grip": float(d[:, [6, 13]].max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official", default="")
    p.add_argument("--out", default=str(HERE / "eval" / "out" / "smoke.json"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cuda-graph", action="store_true")
    args = p.parse_args()
    import torch

    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    import transformers

    res["transformers"] = transformers.__version__
    prof = srv.load_profile()
    imgs = sample_images(prof)
    res["image_shapes"] = {k: list(v.shape) for k, v in imgs.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    backend = srv.MolmoAct2Backend(
        prof, dtype=args.dtype, cuda_graph=args.cuda_graph, verify="full"
    )
    res["load_s_incl_full_sha256"] = round(time.time() - t0, 1)
    res["sha256_verified"] = backend.verified
    res["mem_after_load_mb"] = round(torch.cuda.memory_allocated() / 2**20)
    service = srv.VlaService(backend)
    res["warmup_ms"] = [round(x) for x in srv.warmup(service, 3)]

    httpd = srv.g.make_http_server(service, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    res["info"] = json.loads(urllib.request.urlopen(url + "/info").read())
    body = {
        "instruction": TASK,
        "images": {r: npy(a) for r, a in imgs.items()},
        "joints": STATE.tolist(),
    }
    wall, server_ms, infer_ms = [], [], []
    for _ in range(args.n):
        t = time.perf_counter()
        r = post(url + "/act", body)
        wall.append((time.perf_counter() - t) * 1e3)
        server_ms.append(r["dt_ms"])
        infer_ms.append(r["timing_ms"]["inference"])
    res["http_chunk_ms"] = {
        "client_wall": stats(wall),
        "server_total": stats(server_ms),
        "inference": stats(infer_ms),
    }
    res["reply_keys"] = sorted(r)
    res["actions_shape"] = np.asarray(r["actions"]).shape
    res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    res["reserved_mem_mb"] = round(torch.cuda.memory_reserved() / 2**20)
    httpd.shutdown()

    req = srv.g.ActRequest.from_wire(
        {"instruction": TASK, "images": imgs, "joints": STATE.tolist()}
    )

    def ours(seed, autocast=True):
        backend.autocast = autocast
        torch.manual_seed(seed)
        a = backend.act(req)["actions"]
        backend.autocast = True
        return a

    a0, a0b, a1 = ours(0), ours(0), ours(1)
    a0_noamp = ours(0, autocast=False)
    res["ours_first_rows"] = np.round(a0[:3], 4).tolist()
    res["ours_seed0_repeat"] = diff(a0, a0b)
    res["ours_seed0_vs_seed1"] = diff(a0, a1)
    res["ours_autocast_vs_not"] = diff(a0, a0_noamp)
    res["ours_delta_from_state_row0"] = diff(a0[:1], STATE[None])

    if args.official:
        spec = importlib.util.spec_from_file_location("host_server_yam", args.official)
        off = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(off)
        off._patch_modeling_for_bf16 = lambda d: None
        t0 = time.time()
        pol = off.Policy(
            prof["checkpoint"]["repo"],
            "cuda",
            getattr(torch, args.dtype),
            enable_cuda_graph=args.cuda_graph,
        )
        res["official_load_s"] = round(time.time() - t0, 1)

        def theirs(seed):
            torch.manual_seed(seed)
            return pol.predict(
                imgs["top"],
                imgs["left_wrist"],
                imgs["right_wrist"],
                TASK,
                STATE,
                num_steps=10,
                enable_cuda_graph=args.cuda_graph,
            )

        theirs(0)  # warm
        o0, o1 = theirs(0), theirs(1)
        t = time.perf_counter()
        for _ in range(5):
            theirs(2)
        res["official_inference_ms_mean5"] = round((time.perf_counter() - t) / 5 * 1e3, 1)
        res["official_shape"] = list(o0.shape)
        res["official_first_rows"] = np.round(o0[:3], 4).tolist()
        res["ours_noautocast_vs_official_seed0"] = diff(a0_noamp, o0)
        res["ours_autocast_vs_official_seed0"] = diff(a0, o0)
        res["official_seed0_vs_seed1"] = diff(o0, o1)
        res["ours_seed1_vs_official_seed1"] = diff(a1, o1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(res, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    )
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
