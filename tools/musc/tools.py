"""musc tool bundle -- zero-shot mutual scoring of nominally identical images (MuSc / MuSc-V2).

MuSc (Li et al., "MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual
Scoring of the Unlabeled Images", ICLR 2024; github.com/xrli-U/MuSc, MIT) and MuSc-V2
(github.com/HUST-SLOW/MuSc-V2 at b5e911dbc089, MIT; licence in LICENSE.musc): every image's patches
are scored by how far they are from their nearest patch in each of the other images; a patch that
finds few close matches elsewhere is anomalous. No reference image and no training.

Adapted from MuSc-V2 (MIT):

- ``_aggregate`` is ``SNAMD._embed_2d`` (models/modules/_SNAMD.py, "similar neighbourhood
  aggregation with multiple degrees"): each patch token is replaced by the similarity-weighted mean
  of its r x r neighbourhood (weights exp(-||f - f_centre||), zero padding as torch's Unfold), the
  r in ``r_list`` averaged. Changed: numpy, one layer at a time.
- ``_mutual_scores`` is ``compute_scores_single`` / ``MSM_single`` (models/modules/_MSM.py): for
  image i and each other image j, a patch's distance to its nearest patch of j; the mean of the
  smallest ``topmin`` share of those over j is the patch's score. Changed: numpy, all images at once
  per layer; at least one other image counts (upstream's int() gives 0 under 4 references); with
  ``masks`` the patches outside a part neither get a score nor serve as references.
- ``_rescore`` is ``RsCon`` / ``WMO`` (models/modules/_RsCon.py): each image's score is averaged
  with those of its k most similar images (by the feature of its highest-scoring patch), k in
  ``k_list`` averaged. Changed: numpy; k capped at the number of images; negative similarities
  weigh 0 (upstream's raw dot products of DINO features are positive in practice; unit DINOv2
  features need not be).
- The image score is the maximum of its patch map (``muscv2.py``'s ``ac_score``).

The backbone is DINOv2 ViT-B/14 through transformers (``facebook/dinov2-base``, Apache-2.0, pinned
revision) in place of MuSc-V2's DINO ViT-B/8 (torch.hub); layers 4, 8 and 12 (MuSc-V2's
``feature_layers: [3, 7, 11]``), the encoder's final layer norm applied to each as DINO's
``get_intermediate_layers`` does. The 3D (Point-MAE) branch is not ported. The model loads lazily on
first call; importing this module never pulls torch / transformers. Env knobs:
``GAP_MUSC_MODEL`` (a pinned repo or a local directory), ``GAP_MUSC_DEVICE`` (default ``cuda``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool

logger = logging.getLogger(__name__)

#: Hugging Face repo -> the commit the bundle loads (model cards: Apache-2.0).
MODELS = {
    "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
}
DEFAULT_MODEL = "facebook/dinov2-base"

#: Transformer blocks whose tokens are scored (1-based; MuSc-V2's feature_layers [3, 7, 11] + 1).
LAYERS = (4, 8, 12)
#: Neighbourhood sizes aggregated per patch (MuSc-V2's r_list).
R_LIST = (1, 3, 5)
#: Share of the other images whose nearest-patch distances are averaged (upstream topmin_max).
TOPMIN = 0.3
#: Neighbour counts of the re-scoring (upstream RsCon k_list).
K_LIST = (1, 7)
#: Input side [px]: 32 x 32 patches of 14 px. MuSc-V2 uses 224; on 16 rendered domino scenes (SKILL.md)
#: 448 put the odd part first in 15 of 16 against 13 at 224, and the identical controls' highest
#: score / median from 1.72 down to 1.37.
IMAGE_SIZE = 448
#: A patch takes part when at least this share of it is inside its image's mask.
MASK_SHARE = 0.5

_load_lock = threading.Lock()
_runtime: Any = None


class ScoreResult(TypedDict):
    scores: list[float]  # per image: the maximum of its patch map
    rescored: list[
        float
    ]  # per image: scores after RsCon (min-max normalised, then neighbour-averaged)
    patch_maps: list[
        np.ndarray
    ]  # per image: [grid, grid] float32 patch scores (0 outside the mask)
    grid: int
    layers: list[int]
    model: str


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _model_source() -> tuple[str, str | None]:
    name = os.environ.get("GAP_MUSC_MODEL") or DEFAULT_MODEL
    if name in MODELS:
        return name, MODELS[name]
    if os.path.isdir(name):
        return name, None
    raise ValueError(
        f"GAP_MUSC_MODEL must be one of {sorted(MODELS)} or a local directory, got {name!r}"
    )


class _Runtime:
    """One DINOv2 encoder on the device: images in, per-layer patch tokens out."""

    def __init__(self, source: str, revision: str | None, device: str) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.name = source
        logger.info("Loading DINOv2 %s@%s on %s ...", source, revision, device)
        self.model = AutoModel.from_pretrained(source, revision=revision).to(device).eval()
        self.patch = int(self.model.config.patch_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def tokens(
        self, images: list[np.ndarray], size: int, layers: tuple[int, ...]
    ) -> list[np.ndarray]:
        """Per layer, patch tokens [N, grid * grid, C] (float32, CLS dropped, final layer norm applied)."""
        torch = self.torch
        F = torch.nn.functional
        x = torch.stack(
            [torch.tensor(im, device=self.device).permute(2, 0, 1).float() / 255.0 for im in images]
        )
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - self.mean) / self.std
        out: list[list[np.ndarray]] = [[] for _ in layers]
        with torch.no_grad():
            for s in range(0, len(x), 32):
                hs = self.model(pixel_values=x[s : s + 32], output_hidden_states=True).hidden_states
                for i, layer in enumerate(layers):
                    t = self.model.layernorm(hs[layer])[:, 1:, :]
                    out[i].append(t.float().cpu().numpy())
        return [np.concatenate(o, axis=0) for o in out]


def _get_runtime() -> Any:
    global _runtime
    with _load_lock:
        if _runtime is None:
            source, revision = _model_source()
            _runtime = _Runtime(source, revision, os.environ.get("GAP_MUSC_DEVICE", "cuda"))
        return _runtime


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check`` (never downloads, never imports torch)."""
    try:
        source, revision = _model_source()
    except ValueError:
        return False
    if revision is None:
        return os.path.isfile(os.path.join(source, "config.json"))
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        hit = try_to_load_from_cache(source, "config.json", revision=revision)
    except Exception:
        return None
    return isinstance(hit, str)


def prefetch() -> None:
    """Snapshot-download the pinned weights into the HF cache (``gap skills check --download``)."""
    source, revision = _model_source()
    if revision is None:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=source, revision=revision, repo_type="model")


# ---------------------------------------------------------------------------
# Mutual scoring (numpy)
# ---------------------------------------------------------------------------


def _aggregate(tokens: np.ndarray, grid: int, r_list: tuple[int, ...]) -> np.ndarray:
    """SNAMD: [N, L, C] -> [N, L, C], each patch the similarity-weighted mean of its neighbourhoods."""
    n, _, c = tokens.shape
    r_max = max(r_list)
    if r_max == 1:
        return tokens.copy()
    f = tokens.reshape(n, grid, grid, c)
    pad = (r_max - 1) // 2
    fp = np.pad(f, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out = np.zeros_like(f)
    for r in r_list:
        h = (r - 1) // 2
        num = np.zeros_like(f)
        den = np.zeros((n, grid, grid, 1), dtype=f.dtype)
        for dy in range(-h, h + 1):
            for dx in range(-h, h + 1):
                nb = fp[:, pad + dy : pad + dy + grid, pad + dx : pad + dx + grid, :]
                w = np.exp(-np.linalg.norm(nb - f, axis=-1, keepdims=True))
                num += w * nb
                den += w
        out += num / den
    return (out / len(r_list)).reshape(n, grid * grid, c)


def _mutual_scores(z: np.ndarray, valid: np.ndarray, topmin: float) -> np.ndarray:
    """MSM: [N, L, C] unit tokens, [N, L] valid -> [N, L] patch scores (0 where not valid)."""
    n, length, _ = z.shape
    k = max(1, int((n - 1) * topmin))
    sq = np.sum(z * z, axis=-1)
    scores = np.zeros((n, length), dtype=np.float32)
    for i in range(n):
        d_ij = np.full((length, n - 1), np.inf, dtype=np.float32)
        col = 0
        for j in range(n):
            if j == i:
                continue
            if valid[j].any():
                d2 = sq[i][:, None] + sq[j][valid[j]][None, :] - 2.0 * z[i] @ z[j][valid[j]].T
                d_ij[:, col] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
            col += 1
        d_ij.sort(axis=1)
        scores[i] = np.where(valid[i], d_ij[:, :k].mean(axis=1), 0.0)
    return scores


def _rescore(scores: np.ndarray, feats: np.ndarray, k_list: tuple[int, ...]) -> np.ndarray:
    """RsCon: min-max normalised scores averaged over each image's k most similar images."""
    n = len(scores)
    span = float(scores.max() - scores.min())
    s = (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)
    f = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    w = np.maximum(f @ f.T, 0.0)
    out = []
    for k in k_list:
        kk = min(int(k), n)
        keep = np.argsort(-w, axis=1)[:, :kk]
        m = np.zeros_like(w)
        np.put_along_axis(m, keep, np.take_along_axis(w, keep, axis=1), axis=1)
        p = m / np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
        out.append(p @ s)
    return np.mean(out, axis=0)


def _patch_valid(mask: np.ndarray | None, grid: int) -> np.ndarray:
    if mask is None:
        return np.ones(grid * grid, bool)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.float32)
    h, w = m.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    share = np.array(
        [
            [
                m[ys[a] : max(ys[a + 1], ys[a] + 1), xs[b] : max(xs[b + 1], xs[b] + 1)].mean()
                for b in range(grid)
            ]
            for a in range(grid)
        ]
    )
    return (share >= MASK_SHARE).reshape(-1)


def score_tokens(
    layer_tokens: list[np.ndarray],
    grid: int,
    masks: list[Any] | None = None,
    r_list: tuple[int, ...] = R_LIST,
    topmin: float = TOPMIN,
    k_list: tuple[int, ...] = K_LIST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image scores [N], rescored [N], patch maps [N, grid, grid]) from per-layer tokens [N, L, C]."""
    n = layer_tokens[0].shape[0]
    valid = np.stack([_patch_valid(None if masks is None else masks[i], grid) for i in range(n)])
    if not valid.any(axis=1).all():
        bad = [i for i in range(n) if not valid[i].any()]
        raise ValueError(f"images {bad} have no patch inside their mask")
    maps, aggregated = [], []
    for t in layer_tokens:
        z = _aggregate(t.astype(np.float32), grid, r_list)
        z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        aggregated.append(z)
        maps.append(_mutual_scores(z, valid, topmin))
    pmap = np.mean(maps, axis=0)
    scores = pmap.max(axis=1)
    # The salient feature: the second-to-last scored layer's token at the image's highest patch.
    sal = aggregated[-2 if len(aggregated) > 1 else -1][np.arange(n), pmap.argmax(axis=1)]
    rescored = _rescore(scores, sal, k_list)
    return scores, rescored, pmap.reshape(n, grid, grid)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(
    name="musc.score",
    summary="Zero-shot mutual anomaly scoring (MuSc) of nominally identical images: each image's patches scored against all the others' with DINOv2; per-image scores and patch maps.",
    tags=("perception",),
)
def score(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    image_size: int = IMAGE_SIZE,
    topmin: float = TOPMIN,
    r_list: list[int] | None = None,
    layers: list[int] | None = None,
) -> ScoreResult:
    """Score N >= 3 RGB uint8 [H, W, 3] images of things that should look the same (crops of parts
    in the same orientation and scale) against each other. ``masks`` (one per image, any size; >0 =
    the part) leave the background out. Higher = more unlike the rest; scores are distances between
    unit DINOv2 features (0-2) and only comparable within one call."""
    name = "musc.score"
    if not isinstance(images, (list, tuple)) or len(images) < 3:
        raise ToolError(name, "need a list of at least 3 images")
    arrs = []
    for k, im in enumerate(images):
        a = np.ascontiguousarray(np.asarray(im))
        if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
            raise ToolError(
                name, f"image {k}: expected RGB uint8 [H, W, 3], got {a.dtype} {a.shape}"
            )
        arrs.append(a)
    if masks is not None and len(masks) != len(arrs):
        raise ToolError(name, f"{len(masks)} masks for {len(arrs)} images")
    if not 0.0 < topmin <= 1.0:
        raise ToolError(name, f"topmin must be in (0, 1], got {topmin}")
    lay = tuple(int(v) for v in (layers or LAYERS))
    rl = tuple(int(v) for v in (r_list or R_LIST))
    if any(r < 1 or r % 2 == 0 for r in rl):
        raise ToolError(name, f"r_list entries must be odd and positive, got {list(rl)}")
    try:
        runtime = _get_runtime()
        size = int(image_size) // runtime.patch * runtime.patch
        tokens = runtime.tokens(arrs, size, lay)
    except Exception as e:
        raise PerceptionFailed(f"musc feature extraction failed: {e}") from e
    grid = size // runtime.patch
    try:
        s, rs, pm = score_tokens(tokens, grid, masks, rl, float(topmin))
    except ValueError as e:
        raise ToolError(name, str(e)) from e
    return {
        "scores": [float(v) for v in s],
        "rescored": [float(v) for v in rs],
        "patch_maps": [m.astype(np.float32) for m in pm],
        "grid": int(grid),
        "layers": list(lay),
        "model": str(runtime.name),
    }
