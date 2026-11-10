"""π0 / π0.5 (openpi models through LeRobot's ``PI05Policy``) behind the YAM policy server.

    python server.py --backend pi05 --profile yam-pi05-molmoact2 --port 5560

A profile is one YAML in ``profiles/`` (VLA_SKILL_SPEC section 2): the checkpoint and its pinned
revision, the PaliGemma tokenizer, the camera roles -> the checkpoint's LeRobot image keys, the 14-D
state layout, and the station <-> model joint offset and gripper units. The server's wire stays
#173's (images by role, ``joints[14]``, instruction, marks); the reply is ``actions [T, 14]``
absolute. ``wire_frame="policy"`` (default) passes joints through as #173's clients expect (they add
``GAP_GR00T_JOINT_OFFSET`` themselves); ``wire_frame="station"`` applies the profile's offset and
gripper map here, for clients that send the station frame.

Runs in a LeRobot >= 0.7 environment with the ``pi`` extra (torch, transformers 5.22). No model
import happens at module level: the server imports this module only for ``--backend pi05``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("gr00t.pi05")

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "yam-pi05-molmoact2"
ROLES = ("top", "left_wrist", "right_wrist")
DOF = 14
TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def profile_names() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str, station: str | None = None) -> dict[str, Any]:
    """``profiles/<name>.yaml`` with ``stations[station]`` merged over the top level (the twin)."""
    import yaml

    path = PROFILES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"no pi05 profile {name!r} (have {profile_names()})")
    prof = yaml.safe_load(path.read_text())
    if station and station != "twin":
        over = (prof.get("stations") or {}).get(station)
        if over is None:
            raise ValueError(f"profile {name!r} has no station {station!r}")
        prof = {**prof, **over}
    prof["station"] = station or "twin"
    if prof.get("backend") != "pi05":
        raise ValueError(f"profile {name!r} is for backend {prof.get('backend')!r}, not pi05")
    unknown = sorted(set(prof["image_keys"]) - set(ROLES))
    if unknown:
        raise ValueError(f"profile {name!r}: unknown roles {unknown}")
    if len(prof["state"]) != DOF or prof["action"]["dim"] != DOF:
        raise ValueError(f"profile {name!r}: state and action must be {DOF}-D")
    return prof


class FrameMap:
    """station <-> model: ``station_q = model_q + offset``; grippers mapped linearly between the
    profile's open/closed values. Identity for ``wire_frame="policy"``."""

    def __init__(self, profile: dict[str, Any], wire_frame: str) -> None:
        if wire_frame not in ("policy", "station"):
            raise ValueError(f"wire_frame is policy | station, not {wire_frame!r}")
        self.active = wire_frame == "station"
        off = profile.get("joint_offset_rad") or {}
        self.offset = np.zeros(DOF)
        self.offset[0:6] = np.asarray(off.get("left", [0.0] * 6), dtype=np.float64)
        self.offset[7:13] = np.asarray(off.get("right", [0.0] * 6), dtype=np.float64)
        g = profile.get("gripper") or {}
        self.s_open, self.s_closed = (
            float(g.get("station_open", 1.0)),
            float(g.get("station_closed", 0.0)),
        )
        self.m_open, self.m_closed = (
            float(g.get("model_open", 1.0)),
            float(g.get("model_closed", 0.0)),
        )

    @staticmethod
    def _lin(x: np.ndarray, a0: float, a1: float, b0: float, b1: float) -> np.ndarray:
        return b0 + (x - a0) * (b1 - b0) / (a1 - a0)

    def to_model(self, joints: np.ndarray) -> np.ndarray:
        if not self.active:
            return joints.astype(np.float64)
        q = joints.astype(np.float64) - self.offset
        for i in (6, 13):
            q[i] = self._lin(q[i], self.s_closed, self.s_open, self.m_closed, self.m_open)
        return q

    def to_station(self, actions: np.ndarray) -> np.ndarray:
        if not self.active:
            return actions
        a = actions.astype(np.float64) + self.offset
        a[:, [6, 13]] = self._lin(
            actions[:, [6, 13]], self.m_closed, self.m_open, self.s_closed, self.s_open
        )
        return a.astype(np.float32)


class Pi05Backend:
    """LeRobot ``PI05Policy`` with the checkpoint's own pre/post-processors (quantile normalisation,
    state tokenisation); the tokenizer comes from the profile's ungated copy. Marks are drawn into
    the views (π0.5 never saw them), as for N1.7."""

    steering = "emulated-overlay"

    def __init__(
        self,
        profile: dict[str, Any],
        checkpoint: str = "",
        revision: str | None = None,
        device: str = "cuda",
        tokenizer: str = "",
        wire_frame: str = "policy",
        draw_marks: Callable[[np.ndarray, list[dict]], np.ndarray] | None = None,
        mark_sentence: Callable[[list[dict]], str] | None = None,
    ) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.torch = torch
        self.profile = profile
        self.frame = FrameMap(profile, wire_frame)
        self.draw_marks, self.mark_sentence = draw_marks, mark_sentence
        ck = profile["checkpoint"]
        self.checkpoint = checkpoint or ck["repo"]
        self.revision = (
            revision if revision is not None else (None if checkpoint else ck.get("revision"))
        )
        t0 = time.time()
        path = self._local(self.checkpoint, self.revision)
        tok = tokenizer or os.environ.get("GAP_PI05_TOKENIZER") or ""
        if not tok:
            tk = profile["tokenizer"]
            tok = str(self._local(tk["repo"], tk.get("revision"), allow_patterns=TOKENIZER_FILES))
        self.tokenizer = tok
        from lerobot.configs.policies import PreTrainedConfig

        cfg = PreTrainedConfig.from_pretrained(str(path))
        # The checkpoint's config asks for torch.compile max-autotune: ~7 min at startup on a
        # Blackwell host. Off unless the profile turns it on.
        cfg.compile_model = bool((profile.get("extra") or {}).get("compile", False))
        self.policy = PI05Policy.from_pretrained(str(path), config=cfg)
        steps = (profile.get("extra") or {}).get("num_steps")
        if steps:
            self.policy.config.num_inference_steps = int(steps)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(path),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": tok},
                "device_processor": {"device": device},
            },
        )
        self.load_s = time.time() - t0
        self._check()
        self.horizon = int(self.policy.config.n_action_steps)
        self.dt = 1.0 / float(profile["action"].get("hz", 30))

    @staticmethod
    def _local(repo: str, revision: str | None, allow_patterns: list[str] | None = None) -> Path:
        if os.path.isdir(repo):
            return Path(repo)
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(repo_id=repo, revision=revision, allow_patterns=allow_patterns)
        )

    def _check(self) -> None:
        cfg = self.policy.config
        have = set(cfg.image_features)
        missing = sorted(set(self.profile["image_keys"].values()) - have)
        if missing:
            raise ValueError(
                f"profile {self.profile['profile']}: image keys {missing} not in the checkpoint ({sorted(have)})"
            )
        sdim = cfg.robot_state_feature.shape[0] if cfg.robot_state_feature else 0
        adim = cfg.output_features["action"].shape[0]
        if sdim != DOF or adim != DOF:
            raise ValueError(f"checkpoint state/action dims {sdim}/{adim}, profile needs {DOF}")

    def info(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "backend": "pi05",
            "model": self.checkpoint,
            "revision": self.revision or "",
            "profile": self.profile["profile"],
            "horizon": self.horizon,
            "num_inference_steps": int(self.policy.config.num_inference_steps),
            "wire_frame": "station" if self.frame.active else "policy",
            "station": self.profile.get("station", "twin"),
            "tokenizer": self.profile["tokenizer"]["repo"]
            if not os.path.isdir(self.tokenizer)
            else self.tokenizer,
            "load_s": round(self.load_s, 1),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        }

    def reset(self) -> None:
        self.policy.reset()

    def _image(self, img: np.ndarray) -> Any:
        size = self.profile.get("image_size")
        if size:
            from PIL import Image

            img = np.asarray(Image.fromarray(img).resize(tuple(size), Image.BILINEAR))
        return self.torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0

    def act(self, req: Any) -> dict[str, Any]:
        torch = self.torch
        t0 = time.perf_counter()
        batch: dict[str, Any] = {}
        for role, key in self.profile["image_keys"].items():
            img = req.images.get(role)
            if img is None:
                continue  # the model masks a missing view
            marks = [m for m in req.marks if m["camera"] == role]
            if marks and self.draw_marks is not None:
                img = self.draw_marks(img, marks)
            batch[key] = self._image(img)
        if not batch:
            raise ValueError(f"no camera image for any of {list(self.profile['image_keys'])}")
        q = self.frame.to_model(req.joints)
        batch["observation.state"] = torch.from_numpy(q.astype(np.float32))
        prompt = req.instruction
        if req.mark_text and self.mark_sentence is not None:
            prompt += self.mark_sentence(req.marks)
        batch["task"] = prompt
        batch = self.pre(batch)
        t1 = time.perf_counter()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = self.post(chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        actions = np.asarray(chunk.float().cpu().numpy(), dtype=np.float32).reshape(-1, DOF)
        return {
            "actions": self.frame.to_station(actions),
            "prompt": prompt,
            "marks_drawn": len(req.marks),
            "missing_cameras": sorted(set(self.profile["image_keys"]) - set(req.images)),
            "timing_ms": {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3},
        }
