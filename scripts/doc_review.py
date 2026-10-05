"""The measuring and reading steps of a documentation pass (docs/WRITING.md §6).

Usage:

    uv run python scripts/doc_review.py measure [--rev REV] [--long N] FILE.md [FILE.md ...]
    uv run python scripts/doc_review.py counts FILE.md
    uv run python scripts/doc_review.py account OLD.md NEW.md [--also DOC.md ...]
    uv run python scripts/doc_review.py run FILE.md
    uv run python scripts/doc_review.py render FILE.md OUT.html [--title TEXT] [--label TEXT]

``scripts/doc_structure.py`` is step 5, the structure checks. This script is
the rest. Each command is one of WRITING's steps:

- ``measure``, steps 1 and 6: the prose's words and sentences, the mean
  sentence, the sentences of 35 words or more and of 45 or more, the cost
  words, the *X, not Y* aphorisms, and the tables and code blocks. Headings,
  tables, code and HTML are dropped first, and each paragraph is split on its
  own, with a bold lead ending in a full stop split from the sentence after
  it. ``--long N`` lists every sentence of N words or more, to read: the
  splitter still merges a sentence across a bold lead that holds a ``*``.
- ``counts``, §6's table of counts: ten counts, each hit listed with its
  line. The tenth lists the paragraphs that name the library's own names
  (a parameter, a call, an output field or a quoted value) that no python
  block under the same heading shows, read from the package's signatures and
  ``docs/OUTPUTS.md``, so it needs the package built.
- ``account``, step 4: every number, backticked name and link target of OLD
  is in NEW, or in a document named with ``--also``, where moved text may
  have landed. A name that moved into a code block counts as kept.
- ``run``: every python block of a draft, run the way the README test runs
  the README's: each alone, in a fresh directory, in the namespace
  ``tests/test_production_hardening.py::_readme_namespace`` builds (§3,
  "Every code block runs").
- ``render``, step 7: GitHub's markdown API in ``mode=markdown``, through
  ``gh``, wrapped in a standalone page with GitHub-like light and dark
  styles. Ids lose GitHub's ``user-content-`` prefix, so in-page links land,
  and links to repository files point at the repository on GitHub.

A count is a list to read, not a verdict: read every hit before counting it
(§6, "A flattering count is easy to produce").
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import tempfile
import traceback
from functools import cache
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

REPO = Path(__file__).resolve().parent.parent
PARSER = MarkdownIt("commonmark", {"html": True}).enable("table")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*(\[])")

# --- measure ------------------------------------------------------------------

BOLD_LEAD = re.compile(r"^(\*\*[^*]+?[.!?]\*\*)\s+(.*)$", re.S)
COST = re.compile(r"\b(costs?|pays?|buys?|for free|the price|the point)\b", re.I)
APHORISM = re.compile(r"\w[\w`*)]*, not (?:a |an |the )?\w", re.I)


def _inline_text(tok: Any) -> str:
    out = []
    for c in tok.children or []:
        if c.type == "text":
            out.append(c.content)
        elif c.type == "code_inline":
            out.append(f"`{c.content}`")
        elif c.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif c.type in ("strong_open", "strong_close"):
            out.append("**")
    return "".join(out)


def measure(text: str) -> tuple[dict[str, float], list[tuple[int, int, str]]]:
    """The counts of steps 1 and 6, and every sentence as (line, words, text)."""
    paras: list[tuple[int, str]] = []
    tables = code = python = bullets = sections = 0
    in_table = 0
    in_heading = False
    for tok in PARSER.parse(text):
        if tok.type == "table_open":
            tables += 1
            in_table += 1
        elif tok.type == "table_close":
            in_table -= 1
        elif tok.type in ("fence", "code_block"):
            code += 1
            if tok.type == "fence" and tok.info.strip().startswith("python"):
                python += 1
        elif tok.type == "list_item_open" and not in_table:
            bullets += 1
        elif tok.type == "heading_open":
            in_heading = True
            sections += tok.tag == "h2"
        elif tok.type == "heading_close":
            in_heading = False
        elif tok.type == "inline" and not in_table and not in_heading:
            paras.append(((tok.map[0] + 1) if tok.map else 0, _inline_text(tok)))
    sentences: list[tuple[int, str]] = []
    for line, p in paras:
        p = " ".join(p.split())
        m = BOLD_LEAD.match(p)
        if m:
            sentences.append((line, m.group(1)))
            p = m.group(2)
        sentences.extend((line, s) for s in SENTENCE.split(p) if s.strip())
    words = [len(s.split()) for _, s in sentences]
    prose = " ".join(s for _, s in sentences)
    counts: dict[str, float] = {
        "prose words": sum(words),
        "sentences": len(sentences),
        "mean sentence": round(sum(words) / len(words), 1) if words else 0.0,
        "35+": sum(w >= 35 for w in words),
        "45+": sum(w >= 45 for w in words),
        "cost words": len(COST.findall(prose)),
        "X, not Y": len(APHORISM.findall(prose)),
        "tables": tables,
        "code blocks": code,
        "python blocks": python,
        "bullet items": bullets,
        "## sections": sections,
    }
    return counts, [(line, w, s) for (line, s), w in zip(sentences, words, strict=True)]


# --- counts -------------------------------------------------------------------

OPENER_MAPS = re.compile(
    r"\*[A-Z][^*]{2,60}\*\s+(below\s+)?(says|shows|lists|defines|gives|holds|reads|then)"
    r"|[Bb]elow (come|are|say)\b|a part of its own|Each subsection|then get it running"
    r"|\]\(#[^)]+\)\s+(lists|shows|says|show|is for|for anyone|gives|holds)"
    r"|[Ee]ach kind of check follows|\bfollows?, led by"
    r"|paragraphs? (below|that follow|after (it|the table|the example))\s+(say|give|show|take)"
    r"|\bfollow the example\b|\bthe paragraphs below\b|\bThe paragraphs under\b"
    r"|[Tt]he rest of (this|the) section"
)
BACK = re.compile(
    r"\b(also|as well|the same way)\b|\btoo\b(?!\s+(large|little|few|many|much|small|long|short"
    r"|liberal|slow|fast|wide|narrow|coarse|fine|big|high|low))",
    re.I,
)
INVERTED = re.compile(
    r"which (its|their|the [a-z_]+'s) ([a-z_`]+ ){1,3}(takes|holds|carries|gets|uses|reads|bears)\b"
)
APHORISM_LEAD = re.compile(
    r"^\*\*(?:An?|The)\s[^*]{2,120}?\s(?:is|are)\s(?:an?|the|one|its)\s[^*]*\*\*"
)
NEGATION = re.compile(
    r"\b(?:an?|the)\s+\w+\s+(?:that|which)\s+(?:needs|has|holds|takes|reads|carries)\s+no\b"
    r"|\b\w+\s+that\s+needs\s+no\b"
)
RELATIVE = re.compile(
    r"\b(?:from here on|from now on|hereafter)\b"
    r"|\bthe (?:examples?|sections?|paragraphs?) (?:below|above)\b"
    r"|\b(?:upstream|downstream|beforehand|afterwards)\b"
    r"|\b(?:made|make|makes|comes?|computed?|built|build|added|add|sort(?:ed)?|done|do)\s+first\b"
    r"|\bfirst,|\blater\b(?!\s+(?:row|rows|than))|\bearlier\b(?!\s+than)"
)
ABOUT_THE_DOCUMENT = re.compile(
    r"\b(from here on|the examples (?:below|that follow)|the table below|the tables below"
    r"|this section"
    r"|the paragraphs|below come|Each subsection|the subsection below|as follows)\b",
    re.I,
)
PYTHON_WORDS = re.compile(r"\bexceptions?\b|\braise[sd]?\b|\braising\b", re.I)
#: An error in the sentence: a class such as `ValueError` (case matters, so
#: the everyday *exception* is not one), or *raises an error*.
AN_ERROR = re.compile(
    r"(Error|Exception|Warning)\b|\b(?i:raise[sd]?|raising)\b.*\b(?i:error|exception)s?\b"
)

#: The models' builders, which a paragraph names as models rather than as
#: calls it walks through; and words too common to count as the library's.
MODELS = {
    "ewridge", "rls", "lasso", "kalman", "huber", "quantile", "sgd", "pa", "ftrl", "holt",
    "ew_cov", "marginal", "deco", "rcov", "kmeans", "micro", "ew_class", "seqtest",
    "corrchange", "bocpd", "hmm",
}  # fmt: skip
COMMON = {
    "name", "df", "spec", "specs", "bank", "path", "frame", "value", "values", "data",
    "x", "y", "n", "k", "t", "s",
}  # fmt: skip


@cache
def library_names() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """The package's parameter names, its callables' names, and the stems of
    the output fields ``docs/OUTPUTS.md`` lists."""
    import polars as pl

    import polars_online as po

    params: set[str] = set()
    calls: set[str] = set()

    def add(fn: Any, name: str) -> None:
        calls.add(name)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return
        skip = ("self", "args", "kwargs", "common", "named", "exprs")
        params.update(p.name for p in sig.parameters.values() if p.name not in skip)

    for module in (po.spec, po.stream, po.eval, po.gram, po.corr, po.sim, po.ops):
        names = getattr(module, "__all__", None) or [n for n in dir(module) if n[0] != "_"]
        for n in names:
            obj = getattr(module, n, None)
            if callable(obj):
                add(obj, n)
    for n in ("target", "fit_predict", "predict", "increment"):
        add(getattr(po, n), n)
    for n in ("ewm_mean", "ewm_sum", "ewm_rate", "rewm_mean", "rewm_sum", "rewm_rate"):
        add(getattr(po, n), n)
    for n, obj in inspect.getmembers(po.ModelBank):
        if n[0] != "_" and callable(obj):
            add(obj, n)
    for n in ("fit_predict", "predict", "unnest", "with_windows"):
        add(getattr(pl.LazyFrame().online, n), n)
    outputs = (REPO / "docs" / "OUTPUTS.md").read_text(encoding="utf-8")
    stems = {
        re.split(r"<|__|@", f)[0].rstrip("_") for f in re.findall(r"^\| `([^`]+)`", outputs, re.M)
    }
    return frozenset(params), frozenset(calls), frozenset(s for s in stems if len(s) >= 3)


def _classify(tok: str) -> tuple[str, str] | None:
    """A backticked span as (name, kind): a parameter, a call, an output
    field or a quoted value; None for anything that is not the library's."""
    params, calls, stems = library_names()
    t = re.sub(r"^(po|pl|bank|lf|df|self|polars_online)\.", "", tok.strip())
    t = re.sub(r"^(spec|stream|eval|gram|corr|sim|ops|online)\.", "", t)
    if re.fullmatch(r'"[A-Za-z_][\w./-]*"', t):
        return t, "value"
    m = re.match(r"^([A-Za-z_]\w*)\s*(=.*|\(.*\))?$", t)
    if not m:
        return None
    w = m.group(1)
    if w in COMMON or w in MODELS or len(w) < 3:
        return None
    if w in params and ((m.group(2) or "").startswith("=") or w not in calls):
        return w, "param"
    if w in calls:
        return w, "call"
    if w in params:
        return w, "param"
    if any(w == s or w.startswith(s + "_") for s in stems):
        return w, "field"
    return None


def _shown(name: str, kind: str, code: str) -> bool:
    """Whether a section's code shows a name in its code form: a parameter
    as ``name=``, a call as ``name(``, a field anywhere, comments included,
    and a quoted value verbatim."""
    pattern = {
        "param": rf"\b{re.escape(name)}\s*=",
        "call": rf"\b{re.escape(name)}\(",
        "field": rf"\b{re.escape(name)}\b",
        "value": re.escape(name),
    }[kind]
    return re.search(pattern, code) is not None


def described_not_shown(lines: list[str]) -> list[tuple[int, str]]:
    """Count 10: paragraphs that name three or more of the library's names
    that no python block under the same heading shows, highest score first.
    A name written as code in the prose (``x=...``, ``f(...)``) scores twice.
    It ranked the `po.target` paragraph third of 25 on the README of I7
    (docs/README-ITERATIONS.md, V3)."""
    sections: list[dict[str, Any]] = [{"paras": [], "code": []}]
    buf: list[str] = []
    start, fence, code = 0, None, []

    def flush() -> None:
        if buf:
            sections[-1]["paras"].append((start, " ".join(buf)))
            buf.clear()

    for i, line in enumerate(lines, 1):
        if fence is not None:
            if line.startswith("```"):
                if fence == "python":
                    sections[-1]["code"].append("\n".join(code))
                fence, code = None, []
            else:
                code.append(line)
        elif line.startswith("```"):
            flush()
            fence = line[3:].strip()
        elif re.match(r"^#{1,6} ", line):
            flush()
            sections.append({"paras": [], "code": []})
        elif line.startswith("|") or not line.strip():
            flush()
        else:
            if not buf:
                start = i
            buf.append(line)
    flush()
    rows = []
    for sec in sections:
        code_text = "\n".join(sec["code"])
        for line, text in sec["paras"]:
            named: dict[str, str] = {}
            as_code: set[str] = set()
            for tok in re.findall(r"`([^`]+)`", text):
                got = _classify(tok)
                if got:
                    named.setdefault(*got)
                    if got[1] != "value" and re.search(r"=|\(.*\)", tok):
                        as_code.add(got[0])
            if all(kind == "value" for kind in named.values()):
                continue
            unshown = [w for w, kind in named.items() if not _shown(w, kind, code_text)]
            if len(unshown) >= 3:
                score = len(unshown) + len(as_code & set(unshown))
                marked = ", ".join(w + ("*" if w in as_code else "") for w in unshown)
                rows.append((score, line, f"[score {score}: {marked}] {text}"))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [(line, text) for _, line, text in rows]


def counts(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """Each of §6's counts with its hits, as (title, [(line, text)])."""
    lines = text.splitlines()
    fence, heads = False, []
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
        elif not fence and re.match(r"^#{2,4} ", line):
            heads.append(i)
    no_opener, maps = [], []
    for h in heads:
        j = h + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        while j < len(lines) and lines[j].startswith("*API:*"):
            j += 1
            while j < len(lines) and not lines[j].strip():
                j += 1
        if j >= len(lines) or re.match(r"^(#|```|[-*+] |\d+\. )", lines[j]):
            no_opener.append((h + 1, f"{lines[h]}  [a heading, a list or code]"))
        elif lines[j].startswith("|"):
            no_opener.append(
                (h + 1, f"{lines[h]}  [a table: keep it only if it explains the subsections]")
            )
        para = []
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(("```", "|", "#")):
            para.append(lines[j])
            j += 1
        m = OPENER_MAPS.search(" ".join(para))
        if m:
            maps.append((h + 1, f"{lines[h]}  [{m.group(0)}]"))
    tokens = PARSER.parse(text)
    paras, cells, in_table, row = [], [], 0, 0
    for i, tok in enumerate(tokens):
        in_table += (tok.type == "table_open") - (tok.type == "table_close")
        if tok.type == "tr_open":
            row = (tok.map or [0])[0] + 1
        if tok.type == "inline" and not in_table and tokens[i - 1].type == "paragraph_open":
            paras.append(((tokens[i - 1].map or [0])[0] + 1, tok.content.replace("\n", " ")))
        elif tok.type == "inline" and in_table:
            cells.append((row, tok.content))
    hits: dict[str, list[tuple[int, str]]] = {
        k: [] for k in ("semi", "back", "inv", "doc", "aph", "neg", "rel", "py")
    }
    for line, p in paras:
        m = BOLD_LEAD.match(" ".join(p.split()))
        pieces = [m.group(1), *SENTENCE.split(m.group(2))] if m else SENTENCE.split(p)
        for s in pieces:
            plain = re.sub(r"`[^`]*`", "", s)
            if ";" in plain:
                hits["semi"].append((line, s))
            if BACK.search(plain):
                hits["back"].append((line, s))
            if INVERTED.search(s):
                hits["inv"].append((line, s))
            if ABOUT_THE_DOCUMENT.search(plain):
                hits["doc"].append((line, s))
            if APHORISM_LEAD.search(s):
                hits["aph"].append((line, s))
            if NEGATION.search(plain):
                hits["neg"].append((line, s))
            hits["rel"].extend((line, f"[{m.group(0)}] {s}") for m in RELATIVE.finditer(plain))
            if PYTHON_WORDS.search(plain) and not AN_ERROR.search(s):
                hits["py"].append((line, s))
    for line, cell in cells:
        if PYTHON_WORDS.search(re.sub(r"`[^`]*`", "", cell)) and not AN_ERROR.search(cell):
            hits["py"].append((line, f"[a table cell] {cell}"))
    return [
        ("1. headings with no prose opener", no_opener),
        ("2. openers that list what follows", maps),
        (
            "3. prose sentences with a semicolon: one idea in two halves, or two ideas?",
            hits["semi"],
        ),
        ("4. back-references: within the paragraph before?", hits["back"]),
        ("5. inverted clauses", hits["inv"]),
        ("6. sentences about the document", hits["doc"]),
        ("7. bold leads of the form 'An X ... is a Z'", hits["aph"]),
        ("8. nouns defined by what they lack", hits["neg"]),
        ("9. relative words: does each say relative to what?", hits["rel"]),
        (
            "10. the library's names described where no code under the heading shows them",
            described_not_shown(lines),
        ),
        ("11. exception or raise outside their Python sense", hits["py"]),
    ]


# --- account ------------------------------------------------------------------

TICKED = re.compile(r"`([^`\n]+)`")
NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])")
LINK = re.compile(r"\]\(([^)\s]+)\)|<(https?://[^>\s]+)>")


def account(old: str, new: str) -> list[tuple[str, int, int, list[str], int]]:
    """What OLD holds that NEW lost: per kind, (kind, before, after, lost, added)."""

    def facts(text: str) -> tuple[set[str], set[str], set[str]]:
        numbers = {n.replace(",", "") for n in NUMBER.findall(text)}
        return set(TICKED.findall(text)), numbers, {a or b for a, b in LINK.findall(text)}

    new_code = "\n".join(re.findall(r"```[a-z]*\n(.*?)```", new, re.S))
    out = []
    for kind, x, y in zip(("names", "numbers", "links"), facts(old), facts(new), strict=True):
        lost = sorted(v for v in x - y if not (kind == "names" and v in new_code))
        out.append((kind, len(x), len(y), lost, len(y - x)))
    return out


# --- run ----------------------------------------------------------------------


def python_blocks(text: str) -> list[tuple[int, str]]:
    """Every ```python block, with the line its fence opens on."""
    out, buf, start, inside = [], [], 0, False
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```python"):
            inside, buf, start = True, [], i
        elif line.strip() == "```" and inside:
            inside = False
            out.append((start, "\n".join(buf)))
        elif inside:
            buf.append(line)
    return out


def run(path: Path) -> list[tuple[int, str | None]]:
    """Each python block of a file, run as the README test runs the README's:
    (line, None) for a block that ran, (line, the error) for one that did not."""
    sys.path.insert(0, str(REPO / "tests"))
    spec = importlib.util.spec_from_file_location(
        "test_production_hardening", REPO / "tests" / "test_production_hardening.py"
    )
    assert spec and spec.loader
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    here, out = os.getcwd(), []
    for line, code in python_blocks(path.read_text(encoding="utf-8")):
        env = dict(os.environ)
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                ns = tests._readme_namespace(Path(tmp))
                exec(compile(code, f"{path.name}:{line}", "exec"), ns)
                out.append((line, None))
            except Exception:
                out.append((line, traceback.format_exc().strip().splitlines()[-1]))
            finally:
                os.chdir(here)
                os.environ.clear()
                os.environ.update(env)
    return out


# --- render -------------------------------------------------------------------

REPO_URL = "https://github.com/hgilde/polars-online/blob/main/"
_DARK = """
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d; --border-muted: #3d444db3;
    --code-bg: #151b23; --inline-code-bg: #656c7633; --link: #4493f8; --row-alt: #151b23;
    --banner-bg: #121d2f; --banner-fg: #a5d6ff; --banner-border: #1f6feb66;
    --pl-c: #9198a1; --pl-c1: #79c0ff; --pl-s: #a5d6ff; --pl-k: #ff7b72; --pl-en: #d2a8ff;
    --pl-v: #ffa657; --pl-e: #7ee787; --pl-smi: #e6edf3; --pl-ent: #79c0ff; --pl-sr: #7ee787;
    color-scheme: dark;"""
CSS = f"""
:root {{
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0; --border-muted: #d1d9e0b3;
  --code-bg: #f6f8fa; --inline-code-bg: #818b981f; --link: #0969da; --row-alt: #f6f8fa;
  --banner-bg: #ddf4ff; --banner-fg: #0a3069; --banner-border: #54aeff66;
  --pl-c: #59636e; --pl-c1: #0550ae; --pl-s: #0a3069; --pl-k: #cf222e; --pl-en: #6639ba;
  --pl-v: #953800; --pl-e: #116329; --pl-smi: #1f2328; --pl-ent: #0550ae; --pl-sr: #116329;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{{_DARK} }} }}
:root[data-theme="dark"] {{{_DARK} }}
html, body {{ background: var(--bg); color: var(--fg); }}
body {{ margin: 0; padding-inline: 16px; padding-block: 24px 64px;
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial,
  sans-serif; }}
.wrap {{ max-width: 980px; margin: 0 auto; }}
.banner {{ background: var(--banner-bg); color: var(--banner-fg);
  border: 1px solid var(--banner-border);
  border-radius: 6px; padding: 10px 14px; font-size: 14px; margin-bottom: 24px; }}
.markdown-body {{ word-wrap: break-word; }}
.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {{
  margin: 24px 0 16px; font-weight: 600; line-height: 1.25; text-wrap: balance; }}
.markdown-body h1, .markdown-body h2 {{ padding-bottom: .3em;
  border-bottom: 1px solid var(--border-muted); }}
.markdown-body h1 {{ font-size: 2em; }} .markdown-body h2 {{ font-size: 1.5em; }}
.markdown-body h3 {{ font-size: 1.25em; }} .markdown-body h4 {{ font-size: 1em; }}
.markdown-body p, .markdown-body ul, .markdown-body ol, .markdown-body table, .markdown-body pre,
.markdown-body blockquote, .markdown-body .highlight {{ margin: 0 0 16px; }}
.markdown-body a {{ color: var(--link); text-decoration: none; }}
.markdown-body a:hover {{ text-decoration: underline; }}
.markdown-body a.anchor {{ display: none; }}
.markdown-body code {{
  font: 85%/1.45 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  background: var(--inline-code-bg); border-radius: 6px; padding: .2em .4em; }}
.markdown-body pre {{ background: var(--code-bg); border-radius: 6px; padding: 16px;
  overflow-x: auto;
  font: 85%/1.45 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; }}
.markdown-body pre code {{ background: none; padding: 0; font-size: 100%; }}
.markdown-body .highlight pre {{ margin: 0; }}
.markdown-body blockquote {{ color: var(--muted); border-left: .25em solid var(--border);
  padding: 0 1em;
  margin-left: 0; }}
.markdown-body table {{ display: block; width: max-content; max-width: 100%; overflow-x: auto;
  border-collapse: collapse; font-variant-numeric: tabular-nums; }}
.markdown-body th, .markdown-body td {{ border: 1px solid var(--border); padding: 6px 13px;
  vertical-align: top; }}
.markdown-body th {{ font-weight: 600; }}
.markdown-body tr:nth-child(2n) {{ background: var(--row-alt); }}
.markdown-body hr {{ border: 0; border-top: 1px solid var(--border); margin: 24px 0; }}
.markdown-body img {{ max-width: 100%; }}
.pl-c {{ color: var(--pl-c); }} .pl-c1, .pl-s .pl-v {{ color: var(--pl-c1); }}
.pl-s, .pl-pds, .pl-s .pl-pse .pl-s1 {{ color: var(--pl-s); }}
.pl-k, .pl-bu, .pl-ii {{ color: var(--pl-k); }} .pl-en {{ color: var(--pl-en); }}
.pl-v, .pl-smw {{ color: var(--pl-v); }} .pl-e {{ color: var(--pl-e); }}
.pl-smi, .pl-s .pl-s1 {{ color: var(--pl-smi); }} .pl-ent {{ color: var(--pl-ent); }}
.pl-sr {{ color: var(--pl-sr); }}
"""


def fix_links(body: str) -> str:
    """GitHub's rendering made to work off github.com: ids and names lose the
    ``user-content-`` prefix, and repository-relative links point at GitHub."""
    body = re.sub(r'(id|name)="user-content-', r'\1="', body)

    def absolute(m: re.Match[str]) -> str:
        href = m.group(1)
        if re.match(r"^(https?:|mailto:|#|data:)", href):
            return m.group(0)
        return f'href="{REPO_URL}{href.lstrip("./")}"'

    return re.sub(r'href="([^"]+)"', absolute, body)


def render(path: Path, title: str, label: str) -> str:
    """The file as GitHub renders it, in a standalone page."""
    body = subprocess.run(
        ["gh", "api", "markdown", "-f", "mode=markdown", "-f", "context=hgilde/polars-online",
         "-F", f"text=@{path}"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    banner = f'<div class="banner">{html.escape(label)}</div>' if label else ""
    return (
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n"
        f'<div class="wrap">{banner}<article class="markdown-body">\n'
        f"{fix_links(body)}\n</article></div>\n"
    )


# --- the command line -----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="steps 1 and 6: words, sentences and the rest")
    m.add_argument("--rev", help="measure each file as git holds it at this revision")
    m.add_argument("--long", type=int, help="list every sentence of this many words or more")
    m.add_argument("files", nargs="+")
    c = sub.add_parser("counts", help="§6's counts, each hit listed")
    c.add_argument("file")
    a = sub.add_parser("account", help="step 4: what the new text lost")
    a.add_argument("old")
    a.add_argument("new")
    a.add_argument("--also", nargs="*", default=[], help="documents moved text may have landed in")
    r = sub.add_parser("run", help="every python block, as the README test runs it")
    r.add_argument("file")
    d = sub.add_parser("render", help="step 7: GitHub's rendering, as a standalone page")
    d.add_argument("file")
    d.add_argument("out")
    d.add_argument("--title", default="polars-online README")
    d.add_argument("--label", default="", help="a banner above the page")
    args = ap.parse_args()

    if args.command == "measure":
        for f in args.files:
            if args.rev:
                git = ["git", "show", f"{args.rev}:{f}"]
                text = subprocess.run(git, capture_output=True, text=True, check=True).stdout
            else:
                text = Path(f).read_text(encoding="utf-8")
            numbers, sentences = measure(text)
            for line, words, s in sentences:
                if args.long and words >= args.long:
                    print(f"  {f}:{line}: {words} words: {s[:160]}{'...' if len(s) > 160 else ''}")
            print(f"{f}: " + ", ".join(f"{k} {v}" for k, v in numbers.items()))
    elif args.command == "counts":
        for title, rows in counts(Path(args.file).read_text(encoding="utf-8")):
            print(f"{title}: {len(rows)}")
            for line, s in rows:
                print(f"   {line}: {s[:200]}")
    elif args.command == "account":
        new = Path(args.new).read_text(encoding="utf-8")
        new += "".join("\n" + Path(f).read_text(encoding="utf-8") for f in args.also)
        for kind, before, after, lost, added in account(
            Path(args.old).read_text(encoding="utf-8"), new
        ):
            print(f"{kind}: {before} before, {after} after; lost {len(lost)}, added {added}")
            for v in lost:
                print(f"  LOST  {v}")
    elif args.command == "run":
        results = run(Path(args.file).resolve())
        for line, error in results:
            print(f"  L{line}: {'OK' if error is None else 'FAILED -- ' + error}")
        print(f"{sum(e is None for _, e in results)} of {len(results)} blocks ran")
        return 0 if all(e is None for _, e in results) else 1
    elif args.command == "render":
        page = render(Path(args.file), args.title, args.label)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"{out}: {len(page):,} bytes, {page.count('<h2')} h2, {page.count('<table')} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
