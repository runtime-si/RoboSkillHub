"""MolmoAct2-BimanualYAM policy server for the YAM station: the ``molmoact2`` backend behind vla-yam/1.

    python server.py --profile profiles/yam-molmoact2-native.yaml --port 5560            # station twin
    python server.py --profile profiles/yam-molmoact2-native.yaml --station real --port 5560
    python server.py --backend fake --port 5561                                           # CPU, no model

Runs on a GPU host in this bundle's environment (pyproject: torch + transformers 4.57, no Isaac-GR00T).
The wire protocol, request parsing and the HTTP / ZMQ transports are ``policies/gr00t/server.py``'s,
loaded from the sibling bundle (``GR00T_BUNDLE`` overrides its path), so a gr00t-wam/1 client
(``gr00t.act`` / ``gr00t.run_skill`` with ``GAP_GR00T_URL``) talks to this server unchanged.

vla-yam/1 (VLA_SKILL_SPEC.md section 3) = gr00t-wam/1 plus:

- ``joints`` are in the **station** frame and units (``frame: "station"``, the default); the profile
  maps them to the checkpoint's (joint offset, gripper scale) and the actions back. ``frame: "model"``
  skips the mapping - a gr00t-wam/1 client that applies ``GAP_GR00T_JOINT_OFFSET`` itself sends that.
- the reply adds ``dt_ms`` (server time for the request), ``profile``, ``execute_steps``, ``frame``.
- optional ``num_steps`` (flow-matching solver steps, default the profile's).
- all three views are required (the checkpoint was trained on top / left / right); ``marks`` are
  accepted and ignored (``steering: "none"``, ``marks_ignored`` in the reply).

The ``molmoact2`` backend follows the model card and AllenAI's ``examples/yam/host_server_yam.py``
(github.com/allenai/molmoact2, Apache-2.0): ``AutoModelForImageTextToText`` + ``AutoProcessor`` with
``trust_remote_code`` at the pinned revision, ``predict_action(images=[top, left, right], task, state[14],
norm_tag, inference_action_mode="continuous", num_steps)``, bf16 weights under autocast by default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("molmoact2.server")

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE = HERE / "profiles" / "yam-molmoact2-native.yaml"
PROTOCOL = "vla-yam/1"


def _load_gr00t() -> Any:
    path = Path(os.environ.get("GR00T_BUNDLE") or HERE.parent / "gr00t") / "server.py"
    if "gr00t_wam_server" in sys.modules:
        return sys.modules["gr00t_wam_server"]
    spec = importlib.util.spec_from_file_location("gr00t_wam_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gr00t_wam_server"] = mod  # dataclasses resolve annotations through the module
    spec.loader.exec_module(mod)
    return mod


g = _load_gr00t()
ROLES, DOF = g.ROLES, g.DOF


# ---------------------------------------------------------------------------
# Profile: station <-> checkpoint
# ---------------------------------------------------------------------------


def load_profile(path: str | Path = DEFAULT_PROFILE, station: str | None = None) -> dict[str, Any]:
    """The profile YAML, with ``stations[station]`` merged over the top level, checked."""
    import yaml

    prof = yaml.safe_load(Path(path).read_text())
    if station:
        over = (prof.get("stations") or {}).get(station)
        if over is None:
            raise ValueError(f"profile {prof.get('profile')!r} has no station {station!r}")
        prof = {**prof, **over}
    prof["station"] = station or "twin"
    if list(prof.get("cameras") or []) != list(ROLES):
        raise ValueError(f"profile cameras must be {list(ROLES)} (model order), got {prof.get('cameras')}")
    if len(prof.get("state") or []) != DOF:
        raise ValueError(f"profile state must name {DOF} entries")
    for side in ("left", "right"):
        if np.asarray(prof["joint_offset_rad"][side], dtype=float).shape != (6,):
            raise ValueError(f"joint_offset_rad.{side} needs 6 values")
    gr = prof["gripper"]
    if gr["station_open"] == gr["station_closed"] or gr["model_open"] == gr["model_closed"]:
        raise ValueError("gripper open and closed must differ")
    act = prof["action"]
    if act.get("type") != "abs_joint" or int(act.get("dim", 0)) != DOF:
        raise ValueError(f"action must be abs_joint dim {DOF}, got {act}")
    if not 1 <= int(act["execute_steps"]) <= int(act["horizon"]):
        raise ValueError("action.execute_steps must be in 1..horizon")
    return prof


def _offset(prof: dict[str, Any]) -> np.ndarray:
    off = np.zeros(DOF)
    off[g.LEFT_Q] = prof["joint_offset_rad"]["left"]
    off[g.RIGHT_Q] = prof["joint_offset_rad"]["right"]
    return off


def _grip(x: np.ndarray, src: tuple[float, float], dst: tuple[float, float]) -> np.ndarray:
    """Linear map closed->closed, open->open."""
    (s_closed, s_open), (d_closed, d_open) = src, dst
    return d_closed + (x - s_closed) * (d_open - d_closed) / (s_open - s_closed)


def _grip_pairs(prof: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    gr = prof["gripper"]
    return (float(gr["station_closed"]), float(gr["station_open"])), (
        float(gr["model_closed"]),
        float(gr["model_open"]),
    )


def station_to_model(prof: dict[str, Any], joints: np.ndarray) -> np.ndarray:
    """``[..., 14]`` station joints/openings -> the checkpoint's state."""
    out = np.asarray(joints, dtype=np.float64) - _offset(prof)
    st, md = _grip_pairs(prof)
    for s in (g.LEFT_G, g.RIGHT_G):
        out[..., s] = _grip(out[..., s], st, md)
    return out


def model_to_station(prof: dict[str, Any], actions: np.ndarray) -> np.ndarray:
    """``[..., 14]`` checkpoint actions -> station joints/openings (inverse of station_to_model)."""
    out = np.asarray(actions, dtype=np.float64) + _offset(prof)
    st, md = _grip_pairs(prof)
    for s in (g.LEFT_G, g.RIGHT_G):
        out[..., s] = _grip(out[..., s], md, st)
    return out


def _local_checkpoint(repo: str, revision: str | None) -> Path:
    if os.path.isdir(repo):
        return Path(repo)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, revision=revision))


def verify_checkpoint(path: Path, sha256: dict[str, str], mode: str = "cache") -> dict[str, str]:
    """Check the weight shards against the profile's sha256.

    ``cache``: the HF cache stores an LFS file under its sha256, so the symlink target's name is
    compared (instant). ``full``: hash every byte. ``off``: skip. Returns {file: "ok" | how checked}."""
    if mode == "off":
        return {}
    if not sha256:
        raise ValueError("profile checkpoint.sha256 is empty; pass --verify-sha256 off to load anyway")
    out = {}
    for name, want in sha256.items():
        f = path / name
        if not f.exists():
            raise FileNotFoundError(f"checkpoint file {name} missing in {path}")
        if mode == "cache" and f.is_symlink():
            got = Path(os.path.realpath(f)).name
        else:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            got = h.hexdigest()
        if got != want:
            raise ValueError(f"{name}: sha256 {got} != pinned {want}")
        out[name] = "ok"
    return out


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class FakeMolmoAct2Model:
    """Stands in for ``MolmoAct2ForConditionalGeneration.predict_action``: holds the given state for the
    horizon, nudging every joint by ``step`` per row (so frame mapping is visible in tests)."""

    def __init__(self, horizon: int = 30, step: float = 0.001) -> None:
        self.horizon, self.step = horizon, step
        self.calls: list[dict[str, Any]] = []

    def predict_action(self, **kw: Any) -> Any:
        self.calls.append(kw)
        state = np.asarray(kw["state"], dtype=np.float32)
        rows = np.arange(1, self.horizon + 1, dtype=np.float32)[:, None]
        actions = np.tile(state, (self.horizon, 1))
        actions[:, g.LEFT_Q] += self.step * rows
        actions[:, g.RIGHT_Q] += self.step * rows
        return type("Out", (), {"actions": actions[None]})()


class MolmoAct2Backend:
    """allenai/MolmoAct2-BimanualYAM through transformers' ``predict_action``; frames via the profile."""

    steering = "none"

    def __init__(
        self,
        profile: dict[str, Any],
        device: str = "cuda",
        dtype: str | None = None,
        cuda_graph: bool = False,
        verify: str = "cache",
        checkpoint: str | None = None,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        self.profile = profile
        ck = profile["checkpoint"]
        self.checkpoint = checkpoint or ck["repo"]
        self.revision = "" if checkpoint else ck.get("revision", "")
        self.dtype_name = dtype or profile["extra"].get("dtype", "bfloat16")
        self.cuda_graph = cuda_graph
        self.autocast = True
        self.horizon = int(profile["action"]["horizon"])
        self.execute_steps = int(profile["action"]["execute_steps"])
        self.dt = 1.0 / float(profile["action"]["hz"])
        self.torch = None
        self.verified: dict[str, str] = {}
        t0 = time.time()
        if model is None:
            self.model, self.processor = self._load(device, verify)
        else:
            self.model, self.processor, self.device = model, processor, "fake"
        self.load_s = time.time() - t0
        self.lock = threading.Lock()

    def _load(self, device: str, verify: str) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        path = _local_checkpoint(self.checkpoint, self.revision or None)
        if not os.path.isdir(self.checkpoint):
            self.verified = verify_checkpoint(path, self.profile["checkpoint"].get("sha256") or {}, verify)
        dtype = getattr(torch, self.dtype_name)
        # tokenizer_config.json lists extra_special_tokens, which transformers >= 4.46 rejects; the model
        # code only reads them through convert_tokens_to_ids (host_server_yam.py does the same).
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, extra_special_tokens={})
        model = (
            AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype=dtype)
            .to(device)
            .eval()
        )
        self.device = device
        target = next(model.parameters()).dtype

        def move_and_cast(inputs: Any, dev: Any) -> dict[str, Any]:
            # predict_action moves the processor's fp32 pixel_values without casting them.
            out = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(dev)
                    if v.is_floating_point() and v.dtype != target:
                        v = v.to(target)
                out[k] = v
            return out

        model._move_inputs_to_device = move_and_cast
        return model, processor

    def info(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "backend": "molmoact2" if torch is not None else "molmoact2-fake",
            "model": self.checkpoint,
            "revision": self.revision,
            "profile": self.profile["profile"],
            "station": self.profile.get("station", "twin"),
            "horizon": self.horizon,
            "execute_steps": self.execute_steps,
            "norm_tag": self.profile["extra"]["norm_tag"],
            "dtype": self.dtype_name,
            "cuda_graph": self.cuda_graph,
            "sha256_verified": sorted(self.verified),
            "load_s": round(self.load_s, 1),
            "gpu": torch.cuda.get_device_name() if torch is not None and torch.cuda.is_available() else "cpu",
        }

    def reset(self) -> None:
        pass

    def predict(self, images: list[np.ndarray], task: str, state: np.ndarray, num_steps: int) -> np.ndarray:
        """Raw ``predict_action``: model-frame state in, model-frame ``[T, 14]`` out."""
        extra = self.profile["extra"]
        kw = dict(
            processor=self.processor,
            images=images,
            task=task,
            state=np.asarray(state, dtype=np.float32),
            norm_tag=extra["norm_tag"],
            inference_action_mode=extra.get("inference_action_mode", "continuous"),
            enable_depth_reasoning=False,
            num_steps=int(num_steps),
            normalize_language=True,
            enable_cuda_graph=self.cuda_graph,
        )
        torch = self.torch
        if torch is None:
            out = self.model.predict_action(**kw)
        else:
            amp = self.autocast and self.dtype_name != "float32" and str(self.device).startswith("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=getattr(torch, self.dtype_name), enabled=amp):
                out = self.model.predict_action(**kw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        raw = out.actions
        if hasattr(raw, "detach"):
            raw = raw.detach().float().cpu().numpy()
        a = np.asarray(raw, dtype=np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2 or a.shape[1] != DOF:
            raise RuntimeError(f"predict_action returned {a.shape}, want [T, {DOF}]")
        return a[: self.horizon]

    def act(self, req: Any, frame: str = "station", num_steps: int | None = None) -> dict[str, Any]:
        missing = [r for r in ROLES if r not in req.images]
        if missing:
            raise ValueError(f"MolmoAct2 needs all three views {list(ROLES)}; missing {missing}")
        t0 = time.perf_counter()
        state = station_to_model(self.profile, req.joints) if frame == "station" else req.joints
        images = [req.images[r] for r in ROLES]
        steps = int(num_steps or self.profile["extra"].get("num_steps", 10))
        t1 = time.perf_counter()
        with self.lock:
            raw = self.predict(images, req.instruction, state, steps)
        t2 = time.perf_counter()
        actions = model_to_station(self.profile, raw) if frame == "station" else raw
        return {
            "actions": actions.astype(np.float32),
            "prompt": req.instruction,
            "frame": frame,
            "num_steps": steps,
            "execute_steps": self.execute_steps,
            "marks_ignored": len(req.marks),
            "timing_ms": {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3},
        }


# ---------------------------------------------------------------------------
# Service: gr00t-wam/1's, announcing vla-yam/1
# ---------------------------------------------------------------------------


class VlaService(g.Service):
    def info(self) -> dict[str, Any]:
        return {**super().info(), "protocol": PROTOCOL, "compatible": [g.PROTOCOL],
                "layout": "left j1..j6, left gripper, right j1..j6, right gripper (station frame and units)"}

    def handle(self, endpoint: str, data: dict[str, Any] | None) -> dict[str, Any]:
        if endpoint != "act":
            return super().handle(endpoint, data)
        data = data or {}
        frame = str(data.get("frame") or "station")
        if frame not in ("station", "model"):
            raise ValueError(f"frame must be 'station' or 'model', got {frame!r}")
        num_steps = data.get("num_steps")
        if num_steps is not None and not 1 <= int(num_steps) <= 100:
            raise ValueError("num_steps must be in 1..100")
        req = g.ActRequest.from_wire(data)
        t0 = time.perf_counter()
        with self.lock:
            out = self.backend.act(req, frame=frame, num_steps=num_steps)
            self.calls += 1
        actions = np.asarray(out.pop("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != DOF or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"backend returned actions {actions.dtype} {actions.shape}")
        timing = dict(out.pop("timing_ms", {}))
        timing["total"] = (time.perf_counter() - t0) * 1e3
        info = self.backend.info()
        return {
            "actions": actions,
            "dt": float(self.backend.dt),
            "dt_ms": round(timing["total"], 2),
            "horizon": int(actions.shape[0]),
            "steering": self.backend.steering,
            "timing_ms": {k: round(float(v), 2) for k, v in timing.items()},
            "gpu_mem_mb": self.gpu_mem_mb(),
            **{k: info[k] for k in ("backend", "model", "revision", "profile", "station")},
            **out,
        }


def make_backend(args: argparse.Namespace) -> MolmoAct2Backend:
    profile = load_profile(args.profile, args.station)
    if args.backend == "fake":
        return MolmoAct2Backend(profile, model=FakeMolmoAct2Model(int(profile["action"]["horizon"])))
    return MolmoAct2Backend(
        profile,
        device=args.device,
        dtype=args.dtype,
        cuda_graph=args.cuda_graph,
        verify=args.verify_sha256,
        checkpoint=args.checkpoint or None,
    )


def warmup(service: VlaService, n: int, seed: int = 0) -> list[float]:
    """``n`` act calls on random views and a mid-range state (the model card: warm up with random calls)."""
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        service.handle(
            "act",
            {
                "instruction": "warm up",
                "images": {r: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for r in ROLES},
                "joints": [0.0, 1.0, 1.0, -0.7, 0.0, 0.0, 1.0] * 2,
                "frame": "model",
            },
        )
        ms.append((time.perf_counter() - t0) * 1e3)
    service.calls = 0
    return ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="molmoact2", choices=["molmoact2", "fake"])
    p.add_argument("--profile", default=str(DEFAULT_PROFILE))
    p.add_argument("--station", default=None, help="a key of the profile's stations: (e.g. real); default twin")
    p.add_argument("--checkpoint", default="", help="local checkpoint dir instead of the profile's pinned repo")
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"],
                   help="default: the profile's (bfloat16, < 16 GB; float32 ~24 GB)")
    p.add_argument("--cuda-graph", action="store_true", help="capture the action expert (~2 GB more)")
    p.add_argument("--verify-sha256", default="cache", choices=["cache", "full", "off"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--transport", default="http", choices=["http", "zmq"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--warmup", type=int, default=3, help="random act calls before binding")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    service = VlaService(make_backend(args))
    if args.backend != "fake" and args.warmup:
        logger.info("warmup ms: %s", [round(x) for x in warmup(service, args.warmup)])
    logger.info("ready: %s", json.dumps(g._jsonable(service.info())))
    if args.transport == "zmq":
        g.serve_zmq(service, args.host, args.port)
    else:
        server = g.make_http_server(service, args.host, args.port)
        logger.info("%s http on %s:%d", PROTOCOL, args.host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
