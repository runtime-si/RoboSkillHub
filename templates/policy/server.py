"""Policy server for {{name}}: openpi's websocket protocol over a pinned checkpoint.

gap's PolicyManager starts this with ``uv run --project policies/{{name}} --
python server.py --port N`` and treats the policy as ready once the port
accepts connections, so the checkpoint is downloaded and loaded before the
server binds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import websockets
import yaml
from huggingface_hub import snapshot_download
from openpi_client import msgpack_numpy

HERE = Path(__file__).resolve().parent


def pinned_weights() -> tuple[str, str, str]:
    """``(repo_id, revision, sha256)`` from this bundle's SKILL.md."""
    text = (HERE / "SKILL.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(text.split("---", 2)[1])
    weights = meta["metadata"]["hub"]["weights"]
    repo, _, revision = weights["uri"].removeprefix("hf://").partition("@")
    return repo, revision, str(weights["sha256"])


def snapshot_sha256(root: Path) -> str:
    """The digest ``openskills weights push`` records: path and content of every file."""
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == ".gitattributes" or rel.startswith(".cache/"):
            continue
        lines.append(f"{rel}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def load(checkpoint: Path) -> Any:
    """Build the model from the downloaded checkpoint directory."""
    raise NotImplementedError("load the model from `checkpoint` and return it")


def infer(model: Any, observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """One observation in, one action chunk out: ``{"actions": array[H, D]}``."""
    raise NotImplementedError("run the model on `observation`")


async def serve(model: Any, port: int, metadata: dict[str, Any]) -> None:
    packer = msgpack_numpy.Packer()

    async def handler(ws) -> None:
        await ws.send(packer.pack(metadata))
        async for message in ws:
            await ws.send(packer.pack(infer(model, msgpack_numpy.unpackb(message))))

    async with websockets.serve(handler, "0.0.0.0", port, compression=None, max_size=None):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    repo, revision, sha256 = pinned_weights()
    checkpoint = Path(snapshot_download(repo_id=repo, revision=revision))
    got = snapshot_sha256(checkpoint)
    if got != sha256:
        raise SystemExit(f"{repo}@{revision}: snapshot digest {got} is not the pinned {sha256}")
    model = load(checkpoint)
    asyncio.run(serve(model, args.port, {"policy": "{{name}}", "weights": f"{repo}@{revision}"}))


if __name__ == "__main__":
    main()
