"""The library as a static website, for anyone to browse: what GitHub Pages serves.

``roboskillhub site --out _site`` writes:

- ``index.html``: what the library is, how much of it there is, and every bundle as a card, searchable and
  filterable by kind, stage, task family and robot, with no server behind it (the cards carry their own data);
- ``b/<name>.html``: one page per tool, skill and policy, saying what ``CATALOG.md`` says about it (the full
  description a planner reads, what goes in and comes out, what each exit means, what it calls, its scripts, where
  it came from) and how to take it into a task;
- ``recipes.html``, ``papers.html``: the task templates and the paper cards;
- ``start.html``, ``contribute.html``: README.md and CONTRIBUTING.md as pages;
- ``index.json``: the same index ``roboskillhub index`` writes, for anything that wants the data.

Generated from the catalog and never edited by hand; deterministic, so the same commit gives the same site.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .catalog_page import _functions, _link, summary
from .index import build_index
from .model import CONNECTOR_PREFIXES, STAGES, Bundle

DEFAULT_REPO = "runtime-si/RoboSkillHub"
KIND_ORDER = ("skill", "tool", "policy")
KIND_LABEL = {"skill": "Skill", "tool": "Tool", "policy": "Policy"}
STAGE_TEXT = {
    "perceive": "finds and measures things in the scene",
    "propose": "suggests candidates (grasps, placements, targets)",
    "select": "picks one candidate among several",
    "plan": "plans motions or orders steps",
    "execute": "moves the robot",
    "verify": "checks that a step did what it should",
    "recover": "gets back from a failure",
    "judge": "decides whether the task is done",
}


def e(text: Any) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------------------------- markdown
def markdown(text: str, repo: str) -> str:
    """The Markdown the library's own pages use (headings, paragraphs, lists, fenced code, tables, links, inline
    code, bold, raw HTML blocks), as HTML. Relative links point at the repository on GitHub."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0

    def inline(s: str) -> str:
        parts = s.split("`")
        for k in range(len(parts)):
            if k % 2:
                parts[k] = f"<code>{e(parts[k])}</code>"
            else:
                t = e(parts[k])
                t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
                t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", t)
                parts[k] = t
        s = "".join(parts)

        def link(m: re.Match) -> str:
            label, href = m.group(1), html.unescape(m.group(2))
            if not re.match(r"[a-z]+:|#", href):
                href = f"https://github.com/{repo}/blob/main/{href}"
            return f'<a href="{e(href)}">{label}</a>'

        return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, s)

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(f"<pre><code>{e(chr(10).join(code))}</code></pre>")
            i += 1
            continue
        m = re.match(r"(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            anchor = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f'<h{level} id="{anchor}">{inline(m.group(2))}</h{level}>')
            i += 1
            continue
        if line.lstrip().startswith("<"):
            block = []
            while i < len(lines) and lines[i].strip():
                block.append(lines[i])
                i += 1
            raw = "\n".join(block)
            raw = re.sub(r'src="(?!https?:)([^"]+)"', lambda mm: f'src="{mm.group(1)}"', raw)
            out.append(raw)
            continue
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"\|[\s:|-]+\|?$", lines[i + 1].strip())
        ):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{inline(c)}</th>" for c in head)
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows
                )
                + "</tbody></table>"
            )
            continue
        if re.match(r"\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"\s*\d+\.", line))
            items: list[str] = []
            while i < len(lines) and (
                re.match(r"\s*([-*]|\d+\.)\s+", lines[i]) or (lines[i].startswith("  ") and items)
            ):
                if re.match(r"\s*([-*]|\d+\.)\s+", lines[i]):
                    items.append(re.sub(r"\s*([-*]|\d+\.)\s+", "", lines[i], count=1))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(it)}</li>" for it in items) + f"</{tag}>")
            continue
        if not line.strip():
            i += 1
            continue
        para = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"(#{1,4}\s|```|\s*([-*]|\d+\.)\s|\|)", lines[i])
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- pieces
def robots(bundle: Bundle) -> list[str]:
    tags = list(bundle.embodiments) or ["any"]
    tuned = [str(r) for r in bundle.hub.get("tuned_on") or [] if r not in bundle.embodiments]
    out = [("any robot" if t == "any" else t) for t in tags]
    out += [f"tuned on {t}" for t in tuned]
    if int(bundle.hub.get("arms") or 1) > 1:
        out.append(f"{bundle.hub['arms']} arms")
    if bundle.tier == "full":
        out.append("simulator only")
    return out


def calls(bundle: Bundle, catalog: Catalog) -> tuple[list[str], list[str]]:
    named = bundle.allowed_tools + [
        str(t) for t in bundle.requires.get("connector") or [] if t not in bundle.allowed_tools
    ]
    tools = sorted(
        {t.split(".", 1)[0] for t in named if t.split(".", 1)[0] in catalog.bundles} - {bundle.name}
    )
    connector = sorted({t for t in named if t.split(".", 1)[0] in CONNECTOR_PREFIXES})
    return tools, connector


def used_by(catalog: Catalog) -> dict[str, list[str]]:
    users: dict[str, list[str]] = {}
    for b in catalog.bundles.values():
        for t in calls(b, catalog)[0]:
            users.setdefault(t, []).append(b.name)
    for r in catalog.recipes.values():
        for s in sorted({str(n["skill"]) for n in r.skill_nodes.values()}):
            users.setdefault(s, []).append(f"recipe:{r.name}")
    return {k: sorted(v) for k, v in users.items()}


def github(repo: str, bundle: Bundle, catalog: Catalog, relative: str = "") -> str:
    link = (
        _link(bundle, catalog, relative)
        if relative
        else _link(bundle, catalog, "SKILL.md").rsplit("/", 1)[0]
    )
    return (
        link
        if link.startswith("http")
        else f"https://github.com/{repo}/{'blob' if relative else 'tree'}/main/{link}"
    )


CSS = """
:root{--bg:#fafbfc;--card:#fff;--fg:#1d2330;--dim:#5b6475;--line:#e3e7ee;--acc:#2f6fe4;--acc2:#e9f0fd;
--skill:#2f6fe4;--tool:#b7791f;--policy:#8a4fd8;--ok:#1f8a4c;--warn:#b54708}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif}
a{color:var(--acc);text-decoration:none} a:hover{text-decoration:underline}
header.top{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.bar{max-width:1240px;margin:0 auto;padding:12px 24px;display:flex;align-items:center;gap:22px}
.brand{font-weight:750;font-size:18px;color:var(--fg)} .brand span{color:var(--acc)}
.bar nav{display:flex;gap:18px;margin-left:auto;font-size:14.5px} .bar nav a{color:var(--dim)} .bar nav a.on,.bar nav a:hover{color:var(--fg)}
main{max-width:1240px;margin:0 auto;padding:0 24px 60px}
.hero{padding:44px 0 18px;display:grid;grid-template-columns:1.1fr 1fr;gap:36px;align-items:center}
@media(max-width:900px){.hero{grid-template-columns:1fr}}
.hero h1{font-size:40px;line-height:1.15;margin:0 0 14px;letter-spacing:-.02em} .hero p{font-size:17px;color:var(--dim);margin:0 0 18px}
.hero img{width:100%;border-radius:14px;border:1px solid var(--line);background:#fff}
.btns a{display:inline-block;padding:9px 18px;border-radius:9px;margin:0 8px 8px 0;font-weight:600;font-size:14.5px}
.btns .p{background:var(--acc);color:#fff} .btns .s{border:1px solid var(--line);color:var(--fg);background:#fff}
.stats{display:flex;flex-wrap:wrap;gap:34px;margin:12px 0 4px} .stats div{color:var(--dim);font-size:13px} .stats b{display:block;font-size:30px;color:var(--fg);line-height:1.2}
h2{font-size:22px;margin:36px 0 12px;letter-spacing:-.01em} h3{font-size:17px;margin:24px 0 8px}
.filters{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 6px;align-items:center}
.filters input{flex:1 1 320px;padding:10px 14px;border:1px solid var(--line);border-radius:10px;font-size:15px;background:#fff}
.filters select{padding:9px 10px;border:1px solid var(--line);border-radius:10px;background:#fff;font-size:14px;color:var(--fg)}
.count{color:var(--dim);font-size:13.5px;margin:4px 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;gap:7px;color:var(--fg);transition:box-shadow .15s,border-color .15s}
.card:hover{text-decoration:none;border-color:#c9d3e3;box-shadow:0 4px 14px rgba(20,40,80,.07)}
.card .n{font-weight:650;font-size:15px;word-break:break-word} .card .d{color:var(--dim);font-size:13.5px;flex:1}
.tag{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim);margin:0 4px 3px 0;white-space:nowrap}
.k{font-weight:700;border:none;padding:1px 9px} .k-skill{background:#e8f0fe;color:var(--skill)} .k-tool{background:#fdf3e1;color:var(--tool)} .k-policy{background:#f2eafd;color:var(--policy)}
.simonly{background:#fff1e6;color:var(--warn);border-color:#f6d3b8}
.stages{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px;margin:6px 0}
.stage{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 14px;cursor:pointer} .stage b{font-size:20px;margin-right:8px} .stage small{color:var(--dim)}
.page{max-width:900px} .crumb{margin:26px 0 4px;color:var(--dim);font-size:14px}
.page h1{font-size:30px;margin:4px 0 8px;word-break:break-word} .lead{font-size:16.5px;color:#343b4a}
.facts{display:grid;grid-template-columns:150px 1fr;gap:6px 18px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:18px 0;font-size:14.5px}
.facts div:nth-child(odd){color:var(--dim)}
.exit{background:#fff;border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:8px;padding:9px 14px;margin:8px 0}
.exit code{font-weight:700}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f1f3f7;padding:1px 5px;border-radius:5px}
pre{background:#0f1726;color:#e6edf7;padding:14px 16px;border-radius:10px;overflow-x:auto} pre code{background:none;color:inherit;padding:0}
table{border-collapse:collapse;width:100%;background:#fff;font-size:14px} td,th{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top} th{background:#f4f6f9}
.prose img{max-width:100%} .prose h2{margin-top:30px} footer{border-top:1px solid var(--line);color:var(--dim);font-size:13px;padding:22px 24px;text-align:center}
.empty{color:var(--dim);padding:20px 0}
"""


def layout(title: str, body: str, root: str, active: str, repo: str, commit: str) -> str:
    nav = [
        ("index.html", "Catalog"),
        ("start.html", "Get started"),
        ("recipes.html", "Recipes"),
        ("papers.html", "Papers"),
        ("contribute.html", "Contribute"),
    ]
    links = "".join(
        f'<a href="{root}{h}" class="{"on" if h == active else ""}">{t}</a>' for h, t in nav
    )
    links += f'<a href="https://github.com/{repo}">GitHub ↗</a>'
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{e(title)}</title><meta name=description content='RoboSkillHub: an open library of robot skills that any robot policy can build on.'>"
        f"<style>{CSS}</style></head><body><header class=top><div class=bar><a class=brand href='{root}index.html'>Robo<span>Skill</span>Hub</a>"
        f"<nav>{links}</nav></div></header><main>{body}</main>"
        f"<footer>Generated from <a href='https://github.com/{repo}/tree/{e(commit)}'>{e(commit[:7])}</a> of "
        f"<a href='https://github.com/{repo}'>{e(repo)}</a> by <code>roboskillhub site</code> · Apache-2.0 · "
        f"each bundle keeps its own license</footer></body></html>\n"
    )


# ---------------------------------------------------------------------------------------------- pages
def card_data(bundle: Bundle) -> dict[str, Any]:
    return {
        "name": bundle.name,
        "kind": bundle.kind,
        "stage": bundle.stage or "",
        "summary": summary(bundle.description),
        "tasks": [str(t) for t in bundle.hub.get("tasks") or []],
        "robots": robots(bundle),
        "embodiments": list(bundle.embodiments) or ["any"],
        "arms": int(bundle.hub.get("arms") or 1),
        "sim_only": bundle.tier == "full",
        "text": (bundle.name + " " + bundle.description + " " + " ".join(bundle.tags)).lower(),
    }


def index_page(catalog: Catalog, repo: str, commit: str) -> str:
    bundles = sorted(catalog.bundles.values(), key=lambda b: (KIND_ORDER.index(b.kind), b.name))
    cards = [card_data(b) for b in bundles]
    counts = {k: sum(1 for b in bundles if b.kind == k) for k in KIND_ORDER}
    stages = [(s, sum(1 for b in bundles if b.kind == "skill" and b.stage == s)) for s in STAGES]
    tasks = sorted({t for c in cards for t in c["tasks"]})
    embodiments = sorted({x for c in cards for x in c["embodiments"]} - {"any"})
    hero = (
        "<section class=hero><div><h1>An open library of robot skills that any robot policy can build on.</h1>"
        "<p>Perception and planning tools, reusable skills with the exits a policy branches on, trained checkpoints, task "
        "recipes and the papers behind them. Every part says which robots it runs on, and everything is pinned, so a "
        "policy that passed in simulation runs on the real arm with exactly the code and weights it was tested with.</p>"
        f"<div class=btns><a class=p href='#catalog'>Browse the catalog</a><a class=s href='start.html'>Get started</a>"
        f"<a class=s href='contribute.html'>Contribute a skill</a></div>"
        f"<div class=stats><div><b>{counts['skill']}</b>skills</div><div><b>{counts['tool']}</b>tools</div>"
        f"<div><b>{counts['policy']}</b>policies</div><div><b>{len(catalog.recipes)}</b>recipes</div>"
        f"<div><b>{len(catalog.papers)}</b>papers</div><div><b>{len(tasks)}</b>task families</div></div></div>"
        "<div><img src='figures/overview.svg' alt='What RoboSkillHub holds and how to use it'></div></section>"
    )
    stage_html = (
        "<h2>Skills by what they do</h2><div class=stages>"
        + "".join(
            f"<div class=stage onclick=\"pick('stage','{s}')\"><b>{n}</b>{e(s)}<br><small>{e(STAGE_TEXT.get(s, ''))}</small></div>"
            for s, n in stages
            if n
        )
        + "</div>"
    )

    def sel(name: str, label: str, values) -> str:
        return (
            f"<select id={name}><option value=''>{label}: all</option>"
            + "".join(f"<option value='{e(v)}'>{e(v)}</option>" for v in values)
            + "</select>"
        )

    filters = (
        "<div class=filters><input id=q type=search placeholder='Search by what it does: grasp a cable, find the hole, check the grasp ...'>"
        + sel("kind", "Kind", KIND_ORDER)
        + sel("stage", "Stage", [s for s, n in stages if n])
        + sel("task", "Task", tasks)
        + sel("robot", "Robot", embodiments)
        + "<select id=real><option value=''>Real or sim: any</option>"
        "<option value='real'>runs on a real robot</option></select></div><div class=count id=count></div>"
    )
    script = """<script>
const C=__CARDS__;
const g=id=>document.getElementById(id), grid=g('grid');
function card(c){const k=`<span class="tag k k-${c.kind}">${c.kind}</span>`;
 return `<a class=card href="b/${c.name}.html"><div>${k}${c.stage?`<span class=tag>${c.stage}</span>`:''}${c.sim_only?'<span class="tag simonly">simulator only</span>':''}</div>
 <div class=n>${c.name}</div><div class=d>${esc(c.summary)}</div><div>${c.tasks.map(t=>`<span class=tag>${t}</span>`).join('')}
 <span class=tag>${c.robots.filter(r=>r!='simulator only').join(' · ')}</span></div></a>`}
function esc(s){return s.replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]))}
function apply(){const q=g('q').value.toLowerCase().split(/\\s+/).filter(Boolean),k=g('kind').value,s=g('stage').value,t=g('task').value,r=g('robot').value,real=g('real').value;
 const L=C.filter(c=>(!k||c.kind==k)&&(!s||c.stage==s)&&(!t||c.tasks.includes(t))&&(!r||c.embodiments.includes(r)||c.embodiments.includes('any'))
 &&(!real||!c.sim_only)&&q.every(w=>c.text.includes(w)));
 grid.innerHTML=L.map(card).join('')||'<div class=empty>Nothing matches. Try fewer words.</div>';
 g('count').textContent=`${L.length} of ${C.length} shown`;
 const p=new URLSearchParams();['q','kind','stage','task','robot','real'].forEach(i=>{if(g(i).value)p.set(i,g(i).value)});
 history.replaceState(null,'',p.toString()?'?'+p+'#catalog':location.pathname)}
function pick(id,v){g(id).value=v;apply();location.hash='catalog'}
const u=new URLSearchParams(location.search);['q','kind','stage','task','robot','real'].forEach(i=>{if(u.get(i))g(i).value=u.get(i);g(i).addEventListener('input',apply)});
apply();
</script>""".replace("__CARDS__", json.dumps(cards, separators=(",", ":")))
    body = (
        hero
        + stage_html
        + "<h2 id=catalog>Catalog</h2>"
        + filters
        + "<div class=grid id=grid></div>"
        + script
    )
    return layout(
        "RoboSkillHub · an open library of robot skills", body, "", "index.html", repo, commit
    )


def bundle_page(
    bundle: Bundle, catalog: Catalog, users: dict[str, list[str]], repo: str, commit: str
) -> str:
    hub = bundle.hub
    tools, connector = calls(bundle, catalog)
    exits = bundle.gap.get("exit_conditions")
    prov = hub.get("provenance") or {}
    fact_rows = [
        (
            "Kind",
            KIND_LABEL[bundle.kind]
            + (
                f" · {e(bundle.stage)}: {e(STAGE_TEXT.get(bundle.stage, ''))}"
                if bundle.stage
                else ""
            ),
        ),
        ("Robots", e(", ".join(robots(bundle)))),
    ]
    if hub.get("tasks"):
        fact_rows.append(
            (
                "Tasks",
                " ".join(
                    f"<a class=tag href='../index.html?task={e(t)}#catalog'>{e(t)}</a>"
                    for t in hub["tasks"]
                ),
            )
        )
    if bundle.required_inputs or bundle.produces_outputs or bundle.kind == "skill":
        fact_rows.append(
            (
                "Takes",
                ", ".join(f"<code>{e(n)}</code> {e(t)}" for n, t in bundle.required_inputs.items())
                or "nothing",
            )
        )
        fact_rows.append(
            (
                "Gives",
                ", ".join(f"<code>{e(n)}</code> {e(t)}" for n, t in bundle.produces_outputs.items())
                or "nothing",
            )
        )
    if tools:
        fact_rows.append(("Calls", ", ".join(f"<a href='{e(t)}.html'>{e(t)}</a>" for t in tools)))
    if connector:
        fact_rows.append(("Robot tools", ", ".join(f"<code>{e(t)}</code>" for t in connector)))
    if bundle.serving.get("protocol") and bundle.serving["protocol"] != "in-process":
        fact_rows.append(
            (
                "Runs",
                f"in its own environment over <code>{e(bundle.serving['protocol'])}</code>"
                + (", needs a GPU" if bundle.serving.get("requires_gpu") else ""),
            )
        )
    if bundle.weights:
        fact_rows.append(("Weights", f"<code>{e(bundle.weights.get('uri', ''))}</code>"))
    if users.get(bundle.name):
        fact_rows.append(
            (
                "Used by",
                ", ".join(
                    f"<a href='../recipes.html#{e(u[7:])}'>{e(u[7:])}</a> (recipe)"
                    if u.startswith("recipe:")
                    else f"<a href='{e(u)}.html'>{e(u)}</a>"
                    for u in users[bundle.name]
                ),
            )
        )
    fact_rows.append(("License", e(bundle.frontmatter.get("license") or "-")))
    src = [
        f"<a href='{e(github(repo, bundle, catalog))}'>source on GitHub</a>",
        f"<a href='{e(github(repo, bundle, catalog, 'SKILL.md'))}'>SKILL.md</a>",
    ]
    if prov.get("code"):
        src.append(f"<a href='{e(prov['code'])}'>upstream code</a>")
    if prov.get("paper"):
        p = str(prov["paper"])
        src.append(
            f"<a href='https://arxiv.org/abs/{e(p.split(':', 1)[1])}'>{e(p)}</a>"
            if p.startswith("arXiv:")
            else e(p)
        )
    fact_rows.append(
        ("Source", " · ".join(src) + (f" · v{e(bundle.version)}" if bundle.version else ""))
    )
    facts = (
        "<div class=facts>"
        + "".join(f"<div>{k}</div><div>{v}</div>" for k, v in fact_rows)
        + "</div>"
    )
    parts = [
        f"<div class=crumb><a href='../index.html#catalog'>Catalog</a> / {KIND_LABEL[bundle.kind]}s</div>",
        f"<span class='tag k k-{bundle.kind}'>{bundle.kind}</span>"
        + ("<span class='tag simonly'>simulator only</span>" if bundle.tier == "full" else ""),
        f"<h1>{e(bundle.name)}</h1>",
        f"<p class=lead>{e(bundle.description)}</p>",
        facts,
    ]
    if hub.get("notes"):
        parts.append(f"<h3>Note</h3><p>{e(' '.join(str(hub['notes']).split()))}</p>")
    if isinstance(exits, dict) and exits:
        parts.append(
            "<h2>Exits</h2><p>What the skill reports when it ends; a policy branches on these.</p>"
            + "".join(
                f"<div class=exit><code>{e(n)}</code> {e(' '.join(str(w).split()))}</div>"
                for n, w in exits.items()
            )
        )
    functions = _functions(bundle)
    if functions:
        parts.append(
            "<h2>Functions</h2>"
            + "".join(f"<div class=exit><code>{e(n)}</code> {e(t)}</div>" for n, t in functions)
        )
    scripts = []
    for entry in bundle.gap.get("canonical_scripts") or []:
        if isinstance(entry, dict):
            scripts += [
                f"<a href='{e(github(repo, bundle, catalog, p))}'><code>{e(n)}</code></a>"
                for n, p in entry.items()
            ]
    if scripts:
        parts.append("<h2>Scripts</h2><p>" + ", ".join(scripts) + "</p>")
    if bundle.evals:
        rows = "".join(
            f"<tr><td>{e(r.get('setting', ''))}</td><td>{e(r.get('embodiment', ''))}</td><td>{e(r.get('suite', ''))}</td>"
            f"<td>{e(r.get('successes', ''))}/{e(r.get('episodes', ''))}</td><td>{e(r.get('evidence', ''))}</td></tr>"
            for r in bundle.evals
        )
        parts.append(
            "<h2>Evaluations</h2><table><tr><th>setting</th><th>robot</th><th>suite</th><th>success</th><th>evidence</th></tr>"
            + rows
            + "</table>"
        )
    parts.append(
        "<h2>Use it</h2><p>Take this bundle (and the tools it calls) into a registry for your task, then point "
        "your policy generator at it:</p><pre><code>git clone https://github.com/"
        + e(repo)
        + "\ncd RoboSkillHub && uv sync\n"
        f'uv run --no-sync roboskillhub search "{e(summary(bundle.description, 60).rstrip(".").lower())}" --why\n'
        f"uv run --no-sync roboskillhub view --bundles {e(bundle.name)} --out my_registry</code></pre>"
        "<p>See <a href='../start.html'>Get started</a> for searching by task and robot, locking what a policy "
        "uses, and fetching it on the robot.</p>"
    )
    return layout(
        f"{bundle.name} · RoboSkillHub",
        "<div class=page>" + "".join(parts) + "</div>",
        "../",
        "",
        repo,
        commit,
    )


def recipes_page(catalog: Catalog, repo: str, commit: str) -> str:
    items = []
    for r in sorted(catalog.recipes.values(), key=lambda r: r.name):
        skills = sorted({str(s["skill"]) for s in r.skill_nodes.values()})
        items.append(
            f"<h2 id='{e(r.name)}'>{e(r.name)}</h2><p>{e(r.description)}</p><p><b>Skills</b>: "
            + ", ".join(
                f"<a href='b/{e(s)}.html'>{e(s)}</a>" if s in catalog.bundles else e(s)
                for s in skills
            )
            + f"</p><p><a href='https://github.com/{e(repo)}/blob/main/recipes/{e(r.name)}/RECIPE.md'>RECIPE.md on GitHub</a></p>"
        )
    body = (
        "<div class=page><h1 style='margin-top:30px'>Recipes</h1><p class=lead>Task-level templates: which skills a task "
        "takes, in what order, and how their exits connect.</p>"
        + ("".join(items) or "<p class=empty>None yet.</p>")
        + "</div>"
    )
    return layout("Recipes · RoboSkillHub", body, "", "recipes.html", repo, commit)


def papers_page(catalog: Catalog, repo: str, commit: str) -> str:
    items = []
    for p in sorted(catalog.papers.values(), key=lambda p: p.id):
        fm = p.frontmatter
        derived = getattr(p, "derived", []) or []
        items.append(
            f"<h2 id='{e(p.id)}'>{e(fm.get('title') or p.id)}</h2><p>{e(', '.join(map(str, fm.get('authors') or [])))}"
            + (f" ({e(fm.get('year'))})" if fm.get("year") else "")
            + "</p>"
            + (
                "<p><b>Bundles from it</b>: "
                + ", ".join(
                    f"<a href='b/{e(str(d).rsplit('/', 1)[-1])}.html'>{e(str(d).rsplit('/', 1)[-1])}</a>"
                    for d in derived
                )
                + "</p>"
                if derived
                else ""
            )
            + f"<p><a href='https://github.com/{e(repo)}/blob/main/papers/{e(p.id)}/PAPER.md'>PAPER.md on GitHub</a></p>"
        )
    body = (
        "<div class=page><h1 style='margin-top:30px'>Papers</h1><p class=lead>The papers behind the library's bundles.</p>"
        + ("".join(items) or "<p class=empty>None yet.</p>")
        + "</div>"
    )
    return layout("Papers · RoboSkillHub", body, "", "papers.html", repo, commit)


def prose_page(path: Path, title: str, active: str, repo: str, commit: str) -> str:
    text = path.read_text(encoding="utf-8") if path.is_file() else f"# {title}\n"
    text = text.replace("docs/figures/", "figures/")
    return layout(
        f"{title} · RoboSkillHub",
        f"<div class='page prose'>{markdown(text, repo)}</div>",
        "",
        active,
        repo,
        commit,
    )


def render(catalog: Catalog, repo: str = DEFAULT_REPO) -> dict[str, str]:
    """Every page as {relative path: text}."""
    index = build_index(catalog)
    commit = str(index["hub"]["commit"] or "main")
    users = used_by(catalog)
    pages = {
        "index.html": index_page(catalog, repo, commit),
        "recipes.html": recipes_page(catalog, repo, commit),
        "papers.html": papers_page(catalog, repo, commit),
        "start.html": prose_page(
            catalog.root / "README.md", "Get started", "start.html", repo, commit
        ),
        "contribute.html": prose_page(
            catalog.root / "CONTRIBUTING.md", "Contribute", "contribute.html", repo, commit
        ),
        "index.json": json.dumps(index, indent=2) + "\n",
        ".nojekyll": "",
    }
    for b in catalog.bundles.values():
        pages[f"b/{b.name}.html"] = bundle_page(b, catalog, users, repo, commit)
    return pages


def write(catalog: Catalog, out: Path, repo: str = DEFAULT_REPO) -> int:
    """The site into *out* (emptied first); the figures under docs/figures copied beside it. Returns pages written."""
    if out.exists():
        shutil.rmtree(out)
    pages = render(catalog, repo)
    for rel, text in pages.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    figures = catalog.root / "docs" / "figures"
    if figures.is_dir():
        shutil.copytree(figures, out / "figures")
    return len(pages)
