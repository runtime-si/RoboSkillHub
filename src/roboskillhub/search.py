"""Choose the slice of the catalog a coordinator should see for one task.

gap's coordinator reads every description in the registry it is handed, so the
registry is the prompt. Handing it the whole hub drowns the right skill among
hundreds; this module narrows it: drop what cannot run on the robot, rank the
rest against the instruction, keep the best per stage of a task, then add every
tool bundle the kept skills call.

Ranking is BM25 over names, tags, task families and descriptions, raised by
the bundle's eval records on the asked robot -- no model, no network, the same
answer every run.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .catalog import Catalog
from .model import CONNECTOR_PREFIXES, MATURITY, Bundle, Recipe

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has in into is it its of on or that the then this to use used when "
    "with without one two its their them they which while".split()
)
_CORE_STAGES = ("perceive", "execute", "verify")
"""Stages a task graph almost always needs; kept even when the words do not match."""

FAMILY_WORDS: dict[str, tuple[str, ...]] = {
    "pick-place": (
        "put",
        "place",
        "pick",
        "move",
        "drop",
        "transfer",
        "grab",
        "carry",
        "stack",
        "into",
        "onto",
        "bowl",
        "basket",
        "bin",
        "tray",
        "plate",
        "container",
        "放",
        "拿",
        "抓",
        "移",
        "搬",
        "装",
    ),
    "insertion": (
        "insert",
        "plug",
        "peg",
        "socket",
        "hole",
        "hang",
        "hook",
        "mount",
        "slot",
        "mate",
        "seat",
        "插",
        "挂",
        "装配",
        "套",
    ),
    "cable-routing": ("cable", "wire", "rope", "cord", "route", "thread", "clip", "线", "绳", "缆"),
    "sorting": (
        "sort",
        "arrange",
        "group",
        "organize",
        "categorize",
        "pair",
        "分类",
        "整理",
        "归类",
    ),
    "reorientation": ("flip", "tip", "reorient", "upright", "overturn", "翻", "立起"),
    "articulated": (
        "drawer",
        "door",
        "cabinet",
        "lid",
        "hinge",
        "knob",
        "handle",
        "抽屉",
        "柜门",
        "门",
        "盖",
    ),
    "jigsaw": ("jigsaw", "puzzle", "拼图"),
}
"""Instruction words that name a task family. An instruction says "put the block
in the bowl"; a skill says "move an object into a container". The family is the
vocabulary they share."""

FAMILY_BONUS = 1.0
"""Added to a bundle or recipe whose hub ``tasks`` include a family the instruction names.
Within a stage, a bundle of a named family also ranks ahead of one outside it: one rare
word an out-of-family description happens to share ("blocked" for "block") must not
outrank the family the instruction is about."""

EVIDENCE_WEIGHT = 2.0
"""Added per unit of the Wilson lower bound on a bundle's success rate in its eval
records for the asked embodiment, to a bundle the words already matched."""

REAL_EPISODE_WEIGHT = 2
"""A real episode counts as this many simulated ones in that bound."""


def wilson_lower(successes: float, episodes: float, z: float = 1.96) -> float:
    if episodes <= 0:
        return 0.0
    p = successes / episodes
    centre = p + z * z / (2 * episodes)
    margin = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes))
    return max(0.0, (centre - margin) / (1 + z * z / episodes))


@dataclass
class Evidence:
    """What a bundle's eval records say about one embodiment, passed or not."""

    sim: tuple[int, int] = (0, 0)
    real: tuple[int, int] = (0, 0)

    @property
    def lower(self) -> float:
        w = REAL_EPISODE_WEIGHT
        return wilson_lower(self.sim[0] + w * self.real[0], self.sim[1] + w * self.real[1])

    def __bool__(self) -> bool:
        return bool(self.sim[1] or self.real[1])

    def __str__(self) -> str:
        parts = [f"{k} {s}/{n}" for k, (s, n) in (("real", self.real), ("sim", self.sim)) if n]
        return ", ".join(parts)


def evidence(bundle: Bundle, embodiment: str | None) -> Evidence:
    tally = {"sim": [0, 0], "real": [0, 0]}
    for record in bundle.evals:
        if embodiment and record.get("embodiment") != embodiment:
            continue
        setting, n, s = record.get("setting"), record.get("episodes"), record.get("successes")
        if setting in tally and isinstance(n, int) and isinstance(s, int) and 0 <= s <= n:
            tally[setting][0] += s
            tally[setting][1] += n
    return Evidence(sim=tuple(tally["sim"]), real=tuple(tally["real"]))


def infer_families(query: str) -> list[str]:
    words = set(tokens(query))
    raw = query.lower()
    found = []
    for family, cues in FAMILY_WORDS.items():
        stems = {t for cue in cues if cue.isascii() for t in tokens(cue)}
        if words & stems or any(cue in raw for cue in cues if not cue.isascii()):
            found.append(family)
    return found


def _stem(word: str) -> str:
    """placing, placed, places, place -> plac. Crude, but the same on both sides."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP and len(w) >= 2]


def _document(bundle: Bundle | Recipe) -> list[str]:
    hub = bundle.hub
    name = bundle.name.replace("-", " ")
    parts = [name, name]
    if isinstance(bundle, Bundle):
        parts += [bundle.category, " ".join(bundle.tags), bundle.stage]
    parts += [" ".join(hub.get("tasks") or []), bundle.description]
    return tokens(" ".join(parts))


class Bm25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs = {k: Counter(v) for k, v in docs.items()}
        self.lengths = {k: len(v) for k, v in docs.items()}
        self.avg = (sum(self.lengths.values()) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for terms in self.docs.values():
            df.update(terms.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def score(self, key: str, query: list[str]) -> float:
        terms, length = self.docs[key], self.lengths[key]
        total = 0.0
        for term in set(query):
            tf = terms.get(term, 0)
            if not tf:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
        return total


@dataclass
class Pick:
    bundle: Bundle
    score: float
    reason: str


@dataclass
class Selection:
    query: str
    families: list[str] = field(default_factory=list)
    picks: list[Pick] = field(default_factory=list)
    recipes: list[tuple[Recipe, float]] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    """Bundles filtered out before ranking, and why."""
    evidence: dict[str, str] = field(default_factory=dict)
    """The eval tallies behind a ranked bundle's evidence bonus, e.g. ``sim 41/50``."""

    @property
    def names(self) -> list[str]:
        return [p.bundle.name for p in self.picks]


def eligible(
    bundle: Bundle, *, embodiment: str | None, tier: str, min_maturity: str, arms: int | None
) -> str | None:
    """Why *bundle* cannot serve this robot, or None when it can."""
    if embodiment and not bundle.supports(embodiment):
        return f"declared for {bundle.embodiments or ['nothing']}, not {embodiment}"
    if tier == "camera_only" and bundle.tier == "full":
        return f"reads simulator ground truth ({', '.join(bundle.privileged_tools)})"
    if MATURITY.index(bundle.maturity) < MATURITY.index(min_maturity):
        return f"maturity {bundle.maturity} is below {min_maturity}"
    needed = int(bundle.hub.get("arms") or 1)
    if arms is not None and needed > arms:
        return f"needs {needed} arms"
    return None


def select(
    catalog: Catalog,
    query: str,
    *,
    embodiment: str | None = None,
    tier: str = "camera_only",
    min_maturity: str = "community",
    arms: int | None = None,
    kinds: tuple[str, ...] = ("skill", "policy"),
    k: int = 16,
    per_stage: int = 2,
    stages: tuple[str, ...] = (),
    failure: str = "",
) -> Selection:
    """``stages`` and ``failure`` ask for what could replace or precede the skill a run
    failed at: only bundles of those stages are ranked, against the instruction and the
    failure's words together, and no recipe is offered."""
    selection = Selection(query=query)
    pool: dict[str, Bundle] = {}
    for bundle in catalog.bundles.values():
        if bundle.kind not in kinds or (stages and bundle.stage not in stages):
            continue
        why = eligible(
            bundle, embodiment=embodiment, tier=tier, min_maturity=min_maturity, arms=arms
        )
        if why:
            selection.excluded[bundle.name] = why
        else:
            pool[bundle.name] = bundle

    families = infer_families(query)
    selection.families = families
    q = tokens(query) + tokens(failure) + [t for family in families for t in tokens(family)]
    index = Bm25({name: _document(b) for name, b in pool.items()})

    def in_family(item: Bundle | Recipe) -> bool:
        return bool(set(item.hub.get("tasks") or []) & set(families))

    def score(item: Bundle | Recipe, bm25: Bm25) -> float:
        return bm25.score(item.name, q) + (FAMILY_BONUS if in_family(item) else 0.0)

    rank = MATURITY.index
    scores = {name: score(b, index) for name, b in pool.items()}
    proof = {name: evidence(b, embodiment) for name, b in pool.items()}
    for name, found in proof.items():
        if scores[name] > 0 and found:
            scores[name] += EVIDENCE_WEIGHT * found.lower
    selection.evidence = {name: str(found) for name, found in proof.items() if found}
    scored = sorted(
        pool.values(),
        key=lambda b: (not in_family(b), -scores[b.name], -rank(b.maturity), b.name),
    )

    # A recipe handed to the coordinator is only useful with its skills in view.
    recipe_index = Bm25({name: _document(r) for name, r in catalog.recipes.items()})
    for recipe in [] if stages else catalog.recipes.values():
        recipe_score = score(recipe, recipe_index)
        uses = {str(spec["skill"]) for spec in recipe.skill_nodes.values()}
        if recipe_score > 0 and uses <= set(pool):
            selection.recipes.append((recipe, recipe_score))

    chosen: dict[str, Pick] = {}
    by_stage: dict[str, list[Bundle]] = {}
    for bundle in scored:
        by_stage.setdefault(bundle.stage, []).append(bundle)
    for stage, members in by_stage.items():
        matching = [b for b in members if scores[b.name] > 0][:per_stage]
        if not matching and stage in _CORE_STAGES and not selection.recipes:
            matching = members[:1]
        for bundle in matching:
            chosen[bundle.name] = Pick(
                bundle, scores[bundle.name], f"best {stage or 'unstaged'} match"
            )
    for bundle in scored:
        if len(chosen) >= k:
            break
        if bundle.name not in chosen and scores[bundle.name] > 0:
            chosen[bundle.name] = Pick(bundle, scores[bundle.name], "ranked")
    selection.recipes.sort(key=lambda pair: (-pair[1], pair[0].name))
    for recipe, _ in selection.recipes:
        for spec in recipe.skill_nodes.values():
            name = str(spec["skill"])
            if name not in chosen:
                chosen[name] = Pick(pool[name], scores[name], f"used by recipe {recipe.name}")

    picks = sorted(
        chosen.values(), key=lambda p: (not in_family(p.bundle), -p.score, p.bundle.name)
    )
    for pick in list(picks):
        for tool_name in pick.bundle.allowed_tools:
            prefix = tool_name.split(".", 1)[0]
            owner = catalog.bundles.get(prefix)
            if prefix in CONNECTOR_PREFIXES or owner is None or owner.name in chosen:
                continue
            chosen[owner.name] = Pick(owner, 0.0, f"called by {pick.bundle.name}")
            picks.append(chosen[owner.name])
    selection.picks = picks
    return selection
