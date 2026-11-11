"""Live check of few-shot part recognition: DINOv2 region embeddings vs OWLv2 exemplar embeddings.

References: K labelled parts per class drawn from the training renders (data/train, other scenes and
seeds); queries: every part of the held-out renders (data/render, 100 frames, overhead and wrist).
Each query is labelled by the classify rule of skills/segmenting-small-parts-from-a-few-reference-photos
(mean of the top-3 cosines per class) given its ground-truth mask (dinov2) or box (owlv2, whose
exemplar is the OWLv2 box best matching it). Two tasks: kind (lego / duplo / domino) and the 15
subtypes (lego_brick_2x4, domino_face_down, ...). Five reference draws per K.

    python tests/bundles/_few_shot_parts_eval.py DATA_DIR OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _stub_gap_core() -> None:
    try:
        import gap_core.tools  # noqa: F401
    except ImportError:
        gc = types.ModuleType("gap_core")
        err = types.ModuleType("gap_core.errors")
        err.PerceptionFailed = type("PerceptionFailed", (RuntimeError,), {})
        err.ToolError = type("ToolError", (RuntimeError,), {})
        tl = types.ModuleType("gap_core.tools")
        tl.tool = lambda **kw: lambda f: f
        sys.modules.update({"gap_core": gc, "gap_core.errors": err, "gap_core.tools": tl})
    try:
        import gap  # noqa: F401
    except ImportError:
        g = types.ModuleType("gap")
        g.NodeContext = object
        sys.modules["gap"] = g


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_skill():
    return _load(
        "fewshot_skill",
        ROOT / "skills/segmenting-small-parts-from-a-few-reference-photos/scripts/label_parts.py",
    )


def _objects(split_dir: Path):
    d = json.loads((split_dir / "labels.json").read_text())
    for si, s in enumerate(d["scenes"]):
        img = np.asarray(Image.open(split_dir / s["image"]).convert("RGB"))
        masks = np.load(split_dir / s["masks"])["masks"]
        objs = []
        for j, o in enumerate(s["objects"]):
            o = dict(o, inst=j if len(masks) == len(s["objects"]) else o["inst"])
            if o["area"] >= 20 and o["inst"] < len(masks):
                objs.append(o)
        yield si, s.get("view", ""), img, masks, objs


def embed_split(split_dir: Path, dino, owl_rt, owl, max_frames: int | None = None):
    rows = []
    for si, view, img, masks, objs in _objects(split_dir):
        if max_frames is not None and si >= max_frames:
            break
        if not objs:
            continue
        r = dino.embed_regions(img, masks=[masks[o["inst"]] for o in objs])
        emb_d = r["embeddings"]
        ce, raw, _obj = owl_rt.embed(img)
        H, W = img.shape[:2]
        boxes, inside = owl._decode_boxes(raw, W, H)
        for k, o in enumerate(objs):
            i, _ = owl._pick_exemplar(boxes, ce, np.asarray(o["box"], float), inside)
            v = np.asarray(ce[i], np.float32) if i >= 0 else np.zeros(ce.shape[1], np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            rows.append({"scene": si, "view": view, "cls": o["cls"], "sub": o["sub"],
                         "area": o["area"], "dino": emb_d[k], "owl": v})  # fmt: skip
    return rows


def evaluate(refs, queries, skill, key, task, k_shot, draws=5):
    classes = sorted({r[task] for r in queries})
    accs, small = [], []
    for d in range(draws):
        rng = np.random.default_rng(1000 * k_shot + d)
        bank = []
        for c in classes:
            pool = [r for r in refs if r[task] == c]
            pick = rng.choice(len(pool), size=min(k_shot, len(pool)), replace=False)
            bank += [{"label": c, "embedding": pool[i][key]} for i in pick]
        lab = skill.classify(np.stack([q[key] for q in queries]),
                             np.stack([b["embedding"] for b in bank]),
                             [b["label"] for b in bank], min_score=-1.0, min_margin=0.0)  # fmt: skip
        ok = np.array([lab[i]["label"] == q[task] for i, q in enumerate(queries)])
        sm = np.array([q["area"] < 150 for q in queries])
        accs.append(ok.mean())
        small.append(ok[sm].mean() if sm.any() else float("nan"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.nanmean(small))


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    _stub_gap_core()
    dino = _load("dinov2_tools", ROOT / "tools/dinov2/tools.py")
    owl = _load("owlv2_tools", ROOT / "tools/owlv2/tools.py")
    skill = _load_skill()
    owl_rt = owl._get_runtime()
    res = {}
    for model in ("facebook/dinov2-base", "facebook/dinov2-small"):
        import os

        os.environ["GAP_DINOV2_MODEL"] = model
        dino._runtime = None
        t0 = time.time()
        refs = embed_split(data / "train", dino, owl_rt, owl, max_frames=80)
        queries = embed_split(data / "render", dino, owl_rt, owl)
        print(f"{model}: {len(refs)} refs, {len(queries)} queries, {time.time() - t0:.0f}s",
              flush=True)  # fmt: skip
        for task in ("cls", "sub"):
            for k in (1, 3, 5, 10):
                for key in ("dino", "owl"):
                    if key == "owl" and model != "facebook/dinov2-base":
                        continue
                    name = model.split("/")[1] if key == "dino" else "owlv2-exemplar"
                    m, s, sm = evaluate(refs, queries, skill, key, task, k)
                    res[f"{task}/{k}shot/{name}"] = {"acc": round(m, 4), "std": round(s, 4),
                                                     "acc_small": round(sm, 4)}  # fmt: skip
                    print(f"{task:3s} K={k:2d} {name:16s} acc={m:.3f}±{s:.3f} small={sm:.3f}",
                          flush=True)  # fmt: skip
        views = sorted({q["view"] for q in queries})
        for v in views:
            qv = [q for q in queries if q["view"] == v]
            for key in ("dino", "owl"):
                m, _, _ = evaluate(refs, qv, skill, key, "sub", 5)
                res[f"sub/5shot/{v}/{model.split('/')[1]}/{key}"] = round(m, 4)
                print(f"view={v} sub K=5 {key}: {m:.3f} (n={len(qv)})", flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
