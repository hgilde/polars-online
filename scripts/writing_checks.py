"""The checks that hold the writing rules (docs/WRITING.md §13), in report mode.

Usage:

    uv run python scripts/writing_checks.py report [--check NAME ...] [--list] [--path TEXT]
    uv run python scripts/writing_checks.py check
    uv run python scripts/writing_checks.py apply

Five checks, each over the text its rules govern:

========== ================================================================
messages   every error and warning the package raises or prints: its shape
           (WRITING §11) -- where it happened first, a remedy last, in the
           imperative, at most three sentences, no word the glossary keeps
           out, no formula, no number past three significant figures
docstrings every public function's, class's and model's docstring: a summary
           line of one sentence with no aside, then §10's sections in §10's
           order, none missing that the object needs
vocabulary every variant the glossary retires and every British spelling of
           its spelling table, in the README, the guides, the public
           docstrings and the messages (§12)
sentences  ``scripts/doc_review.py``'s counts on every guide (§5, §6): no
           sentence of 45 words or more, a mean near 20, and the numbered
           counts
kinds      each guide declares its kind, one of the seven §9 accepts, and a
           how-to has its prerequisites and numbered steps (§9)
========== ================================================================

**Report mode.** A finding is *held* when it breaks a rule outright, and *to
read* when it is a hit of one of §6's counts, which list sentences to read
and are not verdicts. ``report`` prints both. ``check``, and the test that
runs it, compares the held findings with ``writing_ratchet.toml``, a
committed count for each check, file and rule:

- a count over its ratchet fails: the file got worse;
- a count under its ratchet fails too, naming the count to tighten it to,
  so the change that made the gain is the one that keeps it: ``apply``
  writes it, as ``scripts/budgets.py apply`` does for a budget;
- a file the check reads that the ratchet has never seen fails until
  ``apply`` records it, so a new file's counts show in the diff that adds
  it;
- a file listed under ``[finished]`` for a check fails on any held finding,
  apart from those listed under ``[[accepted]]`` with the reason a reader
  accepted them. A file moves there when the stage that rewrites it has
  finished (``docs/records/DOCS-REWRITE-MAP.md``, *When each check fails the
  gate*).

``apply`` lowers each count to what is measured, records the files the
ratchet has not seen, and drops the files no check reads any more. It never
increases a count. The vocabulary counts a source file's docstrings and its
messages apart (``x.py#docstrings``, ``x.py#messages``), since different
stages rewrite them.

The reports are printed rather than written to a file, so every checkout
reads the current one and nothing generated is committed; only the counts
are.

**What a static reading cannot see**, check by check:

- messages: a message built in another function (a helper that returns the
  text or the exception), which ``report`` counts as unread; what a bare
  ``{}`` prints, such as a float's full repr; whether a Rust string is a
  whole message or a body its caller wraps (each is read as a whole, so a
  wrapped body of ``online-polars`` reports a missing *where*: accept it
  when its file finishes); whether an ``online-core`` error reaches a
  reader (every ``Err`` string is read as one that does, since the bank
  passes them on behind ``spec "<name>": ``, which is why ``online-core``'s
  are not held to the *where* rule); a remedy is told by its first word,
  and a kept-out word by its spelling, not its sense (*task*);
- docstrings: whether a function can raise (only a model must have
  *Raises*), whether a section holds what its heading says, and whether
  *when to use it* names the alternative;
- vocabulary: a word's sense, so a variant the glossary keeps in one sense
  (*source*, *label*, *batch*) is a hit to read; a quotation is seen only in
  double quotes or a block quote;
- sentences: §6's numbered counts are lists to read, and the splitter merges
  two bullets with no blank line between them (§6, the traps);
- kinds: whether the page holds what its declared kind holds.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import pkgutil
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
RATCHET = "writing_ratchet.toml"
CHECKS = ("messages", "docstrings", "vocabulary", "sentences", "kinds")


@dataclass(frozen=True)
class Finding:
    """One finding: the check, the file and line, the rule it breaks, the
    text it is about, and whether the ratchet holds it (else: to read)."""

    check: str
    path: str
    line: int
    rule: str
    text: str
    held: bool = True

    def show(self) -> str:
        mark = "" if self.held else " (to read)"
        return f"{self.path}:{self.line}: [{self.rule}{mark}] {self.text}"


def _doc_review() -> ModuleType:
    spec = importlib.util.spec_from_file_location("doc_review", REPO / "scripts" / "doc_review.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("doc_review", module)
    spec.loader.exec_module(module)
    return module


doc_review = _doc_review()


# --- the files ----------------------------------------------------------------

#: Files that hold a guide and then a record: the guide is read up to the
#: heading the record starts at (as ``tests/test_no_plan_references.py``'s
#: ``GUIDE_PARTS`` reads it).
GUIDE_PARTS: dict[str, str] = {
    "docs/STATE-WORKFLOW.md": "\n## The research behind it, 2026-09-03\n",
}

#: The directories the rewrite writes new pages into, by kind.
KIND_DIRS = ("docs/tutorials", "docs/how-to", "docs/explanation", "docs/contributing")


def guides(repo: Path = REPO) -> list[str]:
    """Every guide: the README, ``CONTRIBUTING.md``, the index of documents,
    each page of ``docs/README.md``'s *Guides* table, and every page under
    the directories new pages go in. The records are not guides."""
    index = (repo / "docs" / "README.md").read_text(encoding="utf-8")
    table = index.split("\n## Guides\n", 1)[1].split("\n## ", 1)[0]
    listed = [f"docs/{n}" for n in re.findall(r"^\| \[[^\]]+\]\(([^)#]+)\)", table, re.M)]
    new = [
        p.relative_to(repo).as_posix() for d in KIND_DIRS for p in sorted((repo / d).rglob("*.md"))
    ]
    out = ["README.md", "CONTRIBUTING.md", "docs/README.md", *listed, *new]
    return sorted(dict.fromkeys(p for p in out if (repo / p).exists()))


def guide_text(path: str, repo: Path = REPO) -> str:
    """A guide's text, cut where a record that follows it in its file starts."""
    text = (repo / path).read_text(encoding="utf-8")
    cut = GUIDE_PARTS.get(path)
    return text[: text.index(cut)] if cut and cut in text else text


# --- the glossary ---------------------------------------------------------------


def _tables(text: str) -> Iterator[tuple[list[str], list[list[str]], int]]:
    """Each Markdown table in ``text``: its header cells, its rows' cells,
    and the line its header is on."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if (
            lines[i].startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[-|: ]+\|$", lines[i + 1])
        ):
            head = [c.strip() for c in lines[i].strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            yield head, rows, i + 1
            i = j
        else:
            i += 1


def glossary_retired(repo: Path = REPO) -> list[str]:
    """Every italic phrase in a *replaces* cell of the glossary's topic
    tables, in order: what the glossary retires, with the examples and uses
    its rows keep."""
    text = (repo / "docs" / "GLOSSARY.md").read_text(encoding="utf-8")
    out: list[str] = []
    for head, rows, _ in _tables(text):
        if not any(h.startswith("replaces") for h in head):
            continue
        at = next(i for i, h in enumerate(head) if h.startswith("replaces"))
        for row in rows:
            if at < len(row):
                out.extend(re.findall(r"(?<![*\w])\*([^*]+)\*(?![*\w])", row[at]))
    return list(dict.fromkeys(out))


def glossary_kept_out(repo: Path = REPO) -> list[str]:
    """Each word of the glossary's *Words kept out of user-facing text*
    table, as its first column names it. The heading is found by its first
    words, so a rename of what the text is called does not lose it."""
    text = (repo / "docs" / "GLOSSARY.md").read_text(encoding="utf-8")
    section = re.split(r"\n## Words kept out of [^\n]*\n", text, maxsplit=1)[1].split("\n## ", 1)[0]
    out: list[str] = []
    for head, rows, _ in _tables(section):
        if head and head[0] == "word":
            for row in rows:
                out.extend(re.findall(r"\*([^*]+)\*|`([^`]+)`", row[0]))
    return list(dict.fromkeys(a or b for a, b in out))  # type: ignore[misc]


@cache
def glossary_british(repo: Path = REPO) -> dict[str, str]:
    """Each British spelling of the glossary's spelling table, with its
    American form."""
    text = (repo / "docs" / "GLOSSARY.md").read_text(encoding="utf-8")
    section = text.split("\n## Spelling\n", 1)[1]
    out: dict[str, str] = {}
    for head, rows, _ in _tables(section):
        if head[:1] != ["British"]:
            continue
        for row in rows:
            brit = re.findall(r"\*([^*]+)\*", row[0])
            amer = re.findall(r"\*([^*]+)\*", row[-1])
            out.update(zip(brit, amer, strict=True))
    return out


#: British forms beyond the spelling table: the ones WRITING §12 names, and
#: the ones the glossary says the contributor guides and records held.
EXTRA_BRITISH: dict[str, str] = {
    "summarises": "summarizes",
    "behaviours": "behaviors",
    "colour": "color",
    "colours": "colors",
    "modelling": "modeling",
    "modelled": "modeled",
    "initialise": "initialize",
    "initialised": "initialized",
    "initialises": "initializes",
    "catalogue": "catalog",
    "catalogues": "catalogs",
    "labelling": "labeling",
}


@dataclass(frozen=True)
class Variant:
    """A word the glossary retires, as the checker scans for it: the pattern,
    the glossary's phrase it stands for, whether a hit breaks the rule
    outright (held) or may be a use the row keeps (to read), and whether it
    is a name, read inside code spans too."""

    phrase: str
    pattern: str
    held: bool
    in_code: bool = False


#: Every variant the glossary retires that a scan can find. A *held* one has
#: no sense the glossary keeps; a *to read* one has, such as *source* for
#: `po.eval.grid_search`'s parameter, so its hits are listed, not counted.
VARIANTS: list[Variant] = [
    Variant("model spec", r"\bmodel specs?\b", True),
    Variant("specification", r"\bspecifications?\b", False),
    Variant("configuration", r"\bconfigurations?\b", False),
    Variant("fresh bank", r"\bfresh banks?\b", True),
    Variant("source", r"\bsources?\b", False),
    Variant("plan", r"\bplans?\b", False),
    Variant("LazyFrame", r"\bLazyFrames?\b", False),
    Variant("DataFrame", r"\bDataFrames?\b", False),
    Variant("batch", r"\bbatch(?:es)?\b", False),
    Variant("outcome", r"\boutcomes?\b", False),
    Variant("response", r"\bresponses?\b", False),
    Variant("label", r"\blabels?\b", False),
    Variant("signal", r"\bsignals?\b", False),
    Variant("regressor", r"\bregressors?\b", True),
    Variant("predictor", r"\bpredictors?\b", True),
    Variant("covariate", r"\bcovariates?\b", True),
    Variant("partition", r"\bpartitions?\b", False),
    Variant("missing value", r"\bmissing values?\b", True),
    Variant("time column", r"\btime columns?\b", True),
    Variant("forgetting factor", r"\bforgetting factors?\b", True),
    Variant("halflife", r"\bhalflifes?\b|\bhalflives\b", True),
    Variant("λ", r"λ", False),
    Variant("lambda", r"\blambda\b", False),
    Variant(
        "goes backwards", r"\b(?:go|goes|going|went|gone|run|runs|ran|step|steps) backwards\b", True
    ),
    Variant("out of order", r"\bout of order\b", False),
    Variant("restart", r"\brestart(?:s|ed|ing)?\b", True),
    Variant("label delay", r"\blabel delays?\b", True),
    Variant("update", r"\bupdate[sd]?\b", False),
    Variant("train", r"\btrain(?:s|ed|ing)?\b", False),
    Variant("teach", r"\bteach(?:es|ing)?\b|\btaught\b", False),
    Variant("out of sample", r"\bout of sample\b", True),
    Variant("honest", r"\bhonest(?:ly)?\b", True),
    Variant("prequential", r"\bprequential\b", False),
    Variant("forecast", r"\bforecast(?:s|ed|ing)?\b", False),
    Variant("scored", r"\bscored\b", False),
    Variant("beta", r"\bbetas?\b", False),
    Variant("accumulator", r"\baccumulators?\b", True),
    Variant("sufficient statistics", r"\bsufficient statistics?\b", False),
    Variant("re-solve", r"\bre-solve[sd]?\b", True),
    Variant("windowed", r"\bwindowed\b", False),
    Variant("rolling", r"\brolling\b", False),
    Variant("warmup", r"\bwarmups?\b", True),
    Variant("burn-in", r"\bburn-in\b", True),
    Variant("gated", r"\bgated\b", True),
    Variant("suppressed", r"\bsuppressed\b", True),
    Variant("settled", r"\bsettled\b", True),
    Variant("settles", r"\bsettles\b", True),
    Variant("converged", r"\bconverged\b", False),
    Variant("notice", r"\bnotices?\b", False),
    Variant("n_eff", r"\bn_eff\b", True, in_code=True),
    Variant("effective weight", r"\beffective weights?\b", True),
    Variant("effective sample size", r"\beffective sample sizes?\b", False),
    Variant("sweep", r"\bsweeps?\b", False),
    Variant("grid combination", r"\bgrid combinations?\b", True),
    Variant("grid slot", r"\bgrid slots?\b", True),
    Variant("candidate", r"\bcandidates?\b", False),
    Variant("decay instance", r"\bdecay instances?\b", True),
    Variant("prediction slot", r"\bprediction slots?\b", True),
    Variant("struct column", r"\bstruct columns?\b", True),
    Variant("checkpoint", r"\bcheckpoints?\b", False),
    Variant("snapshot", r"\bsnapshots?\b", False),
    Variant("score rows", r"\bscores? rows\b", True),
    Variant("finished group", r"\bfinished groups?\b", True),
    Variant("CLI", r"\bCLI\b", True),
    Variant("runner", r"\brunners?\b", False),
]

#: The glossary's italic phrases in a *replaces* cell that no scan stands
#: for, each with the reason. A phrase the glossary adds is in neither list
#: until it is placed in one, and ``test_every_retired_word_is_placed``
#: fails until then.
NOT_SCANNED: dict[str, str] = {
    "key": "a group's value is *key* alone only in context; *group key* and Polars' keys stay",
    "teaches the model nothing": "an example of *teach*, which is scanned",
    "timestamp column": "kept as a description of a clock that holds times",
    "error": "retired only where it means one row's residual, a sense a scan cannot tell",
    "score": "retired only as a noun for a prediction, a sense a scan cannot tell",
    "seed": "retired only as a noun for the seeding rows; a random seed keeps it",
    "window": "retired only alone for a hard window; a window operator's span keeps it",
    "running sums": "retired only where it means a least-squares model's Gram",
    "setting": "retired only for a whole grid point, a sense a scan cannot tell",
    "lam": "allowed in backticks where the reader set it, which a scan cannot see",
    "labels that arrive late": "an example of *label* where it means a target, which is read",
    "Timestamp column": "kept as a description of a clock that holds times",
    "a clock (timestamp column `ts`)": "an example of the use the row keeps",
    "Forget": "kept as an ordinary verb in an explanation",
    "fresh": "the word the row gives as the reason; *fresh bank* is scanned",
    "stream": "retired only for one model's rows; the glossary's word for a bank's rows",
    "grid": "retired alone only for `refresh_time`'s times; the glossary's word for a spec's grid",
    "the model's rows": "the wording the row gives in place of *stream* for one model's rows",
    "break": "retired alone only for a clock break, a sense a scan cannot tell from a break test",
    "warm-up": "the chosen spelling, named as the reason *warmup* is retired",
    "slot": "retired only for a coefficient's position, a sense a scan cannot tell",
    "grid point": "retired only for one of `refresh_time`'s times, a sense a scan cannot tell",
    "the refresh grid": "the wording the row gives in place of *grid* for `refresh_time`'s times",
    "refresh time": "the wording the row gives in place of *grid point* for those times",
    "Out of order": "kept for a query whose row order Polars does not promise",
    "update equation": "the docstring section's name, which keeps *update*",
    "under a hard window": "the wording the row gives in place of *windowed*",
    "the model seeds its centers": "an example of the verb, which the row keeps",
}


def _vocabulary(text: str) -> str:
    """Text as the vocabulary is read from it: code spans, links' targets,
    HTML comments and quotations blanked (a quotation keeps its source's
    words), lengths kept so a line stays a line."""

    def blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    text = re.sub(r"<!--.*?-->", blank, text, flags=re.S)
    text = re.sub(r"\]\([^)\s]*\)", blank, text)
    text = re.sub(r"https?://\S+", blank, text)
    text = re.sub(r"“[^”\n]{0,300}”|\"[^\"\n]{0,300}\"", blank, text)
    return text


def _markdown_prose(text: str) -> str:
    """A Markdown page's user-facing words: its prose, and the comments of
    its code examples; code, code spans, quotations and block quotes
    blanked, lines kept."""
    out: list[str] = []
    fence = ""
    for line in text.splitlines():
        if fence:
            if line.strip().startswith(fence):
                fence = ""
                out.append("")
            else:
                hash_at = line.find("#")
                out.append(" " * hash_at + line[hash_at:] if hash_at >= 0 else "")
        elif m := re.match(r"^\s*(```+|~~~+)", line):
            fence = m.group(1)
            out.append("")
        elif line.lstrip().startswith(">"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _without_code_spans(text: str) -> str:
    return re.sub(r"``[^`]+``|`[^`\n]+`", lambda m: " " * len(m.group(0)), text)


@cache
def _british() -> tuple[dict[str, str], re.Pattern[str]]:
    british = {**glossary_british(), **EXTRA_BRITISH}
    words = sorted(british, key=len, reverse=True)
    return british, re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)


def vocabulary_findings(text: str, path: str, *, where: str = "", line0: int = 1) -> list[Finding]:
    """The retired variants and British spellings in user-facing ``text``."""
    prose = _vocabulary(text)
    plain = _without_code_spans(prose)
    found: list[Finding] = []
    british, brit = _british()
    prefix = f"{where}: " if where else ""

    def line_of(at: int) -> int:
        return line0 + plain.count("\n", 0, at)

    def context(src: str, m: re.Match[str]) -> str:
        start = src.rfind("\n", 0, m.start()) + 1
        end = src.find("\n", m.end())
        return " ".join(src[start : None if end < 0 else end].split())[:160]

    for v in VARIANTS:
        src = prose if v.in_code else plain
        for m in re.finditer(v.pattern, src, flags=re.I if v.phrase.islower() else 0):
            found.append(
                Finding(
                    "vocabulary",
                    path,
                    line_of(m.start()),
                    f"retired: {v.phrase}",
                    prefix + context(src, m),
                    v.held,
                )
            )
    for m in brit.finditer(plain):
        word = m.group(1).lower()
        found.append(
            Finding(
                "vocabulary",
                path,
                line_of(m.start()),
                f"british: {word} -> {british[word]}",
                prefix + context(plain, m),
            )
        )
    return found


# --- the messages ---------------------------------------------------------------


@dataclass
class Message:
    """One message as its source writes it: its parts, each a literal or a
    placeholder (the expression it formats, and its format spec)."""

    path: str
    line: int
    kind: str  # error | warning | printed
    parts: list[tuple[str, str]] = field(default_factory=list)  # ("lit", text) | ("ph", expr:spec)

    def display(self) -> str:
        return "".join(t if k == "lit" else "{" + t + "}" for k, t in self.parts)

    def plain(self) -> str:
        """The literal text, each placeholder as ``{}``."""
        return "".join(t if k == "lit" else "{}" for k, t in self.parts)

    def literal(self) -> str:
        return "".join(t if k == "lit" else " " for k, t in self.parts)


#: A message's *where*: a placeholder that names the spec or call, a spec by
#: name, or a call or a module function, before its first colon.
WHERE_PH = re.compile(
    r"^\{[^}]*\b(?:who|_who|where|spec|name|prefix|ctx|context|what|label|fn|call)\b[^}]*\}", re.I
)
WHERE_WORDS = re.compile(
    r"^(?:spec|target|group|column|feature|clock|session|weight|model|config|po\.[\w.]+|pl\.[\w.]+"
    r"|ModelBank(?:\.\w+)?|bank\.\w+|lf\.online\.\w+|df\.online\.\w+|[\w.]+\([^)]*\)|[a-z_][\w.]*)"
    r"(?:\s+(?:\{\}|\"[^\"]*\"|'[^']*'|\[[^\]]*\]|[\w.()\[\]-]+)){0,3}$"
)
#: Words that make a run of words before a colon a sentence, not a where.
NOT_WHERE = re.compile(
    r"\b(?:must|is|are|was|were|has|have|had|takes|took|got|cannot|can|needs|need|does|do|"
    r"did|should|would|could|will|reads|holds|gives|expects?|expected|requires|found|"
    r"not|no|only|either)\b",
    re.I,
)


def where_of(m: Message) -> str | None:
    """The message's *where* prefix, with its colon, or None."""
    plain = m.plain().lstrip()
    display = m.display().lstrip()
    if WHERE_PH.match(display) and plain.startswith("{}") and plain[2:3] in (":", " "):
        colon = plain.find(":")
        return plain[: colon + 1] if 0 <= colon <= 60 else None
    colon = plain.find(": ")
    if colon < 0:
        colon = plain.find(":\n")
    if colon <= 0 or colon > 60:
        return None
    head = plain[:colon]
    if NOT_WHERE.search(re.sub(r"\"[^\"]*\"|'[^']*'", "", head)):
        return None
    return head + ":" if WHERE_WORDS.match(head) else None


#: The verbs a remedy starts with: the imperative of what a reader does.
IMPERATIVE = frozenset({
    "add", "adjust", "align", "alias", "append", "apply", "ask", "avoid", "build", "call",
    "cast", "change", "check", "choose", "clear", "close", "collect", "combine", "compute",
    "convert", "copy", "correct", "cut", "declare", "decrease", "define", "delete",
    "difference", "divide", "do", "drop", "either", "enable", "ensure", "filter", "fit", "fix",
    "follow", "give", "group", "increase", "install", "join", "keep", "leave", "let", "limit",
    "list", "load", "look", "lower", "make", "map", "merge", "move", "name", "omit", "open",
    "pass", "pick", "pin", "place", "point", "predict", "provide", "put", "raise", "read",
    "rebuild", "reduce", "refit", "remove", "rename", "reorder", "replace", "rerun", "reset",
    "resume", "retry", "return", "rewrite", "run", "save", "scale", "see", "select", "send",
    "set", "shorten", "shift", "skip", "sort", "specify", "split", "start", "state", "stop",
    "supply", "swap", "take", "try", "turn", "type", "unnest", "upgrade", "use", "wait",
    "widen", "write",
})  # fmt: skip
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z{\"'(`])")


def sentences_of(text: str) -> list[str]:
    return [s for s in SENTENCE.split(" ".join(text.split())) if s.strip()]


def is_imperative(sentence: str) -> bool:
    """Whether a sentence is an instruction: a sentence of its own, whose
    first word, past a leading condition (*If the clock steps back,*),
    *Either* or *Otherwise*, is a verb a reader does, as a word rather than
    a name (``skip_learned``, ``predict =``, ``merge()``), and not the
    subject of a verb (*State has*)."""
    s = sentence.strip().lstrip("({\"'`")
    led = re.match(r"(?:If|When|Where|Unless|To|For|Before|After|Once)\b[^,]*,\s*", s)
    if led:
        s = s[led.end() :]
    s = re.sub(r"^(?:Either|Or|So|Then|Instead|Otherwise|Please),?\s+", "", s)
    if led or s[:1].islower() and s != sentence.strip():
        s = s[:1].upper() + s[1:]
    m = re.match(r"([A-Z][a-z]+)(?!\w)(?!\s*[(=\[])\s*(\w*)", s)
    if not m or m.group(1).lower() not in IMPERATIVE:
        return False
    return m.group(2).lower() not in SUBJECT_OF


#: Words after which a first word is a subject, not an instruction.
SUBJECT_OF = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did", "must", "can",
    "cannot", "will", "would", "should", "needs", "takes", "holds", "reads", "gives",
})  # fmt: skip


#: Words of the code, the tests and Polars' internals that the glossary
#: keeps out of messages, by the phrase its table gives them.
KEPT_OUT: dict[str, str] = {
    "settled": r"\bsettled\b",
    "cfg": r"\bcfgs?\b",
    "ModelKind": r"\bModelKind\b",
    "ModelState": r"\bModelState\b",
    "AnyModel": r"\bAnyModel\b",
    "Span": r"\bSpans?\b",
    "O(state)": r"\bO\(state\)",
    "O(data)": r"\bO\(data\)",
    "O(window)": r"\bO\(window\)",
    "sink": r"\bsinks?\b",
    "sink_parquet": r"\bsink_parquet\b",
    "collect": r"\bcollect(?:s|ed|ing)?\b",
    "collect_batches": r"\bcollect_batches\b",
    "scan": r"\bscan(?:s|ned|ning)?\b",
    "scan_parquet": r"\bscan_parquet\b",
    "morsel": r"\bmorsels?\b",
    "streaming engine": r"\bstreaming engine\b",
    "row group": r"\brow groups?\b",
    "the pool": r"\bthe pool\b",
    "golden": r"\bgoldens?\b",
    "oracle": r"\boracles?\b",
    "the gate": r"\bthe gate\b",
    "mutants": r"\bmutants\b",
    "λ": r"λ",
}
#: The kept-out words no scan stands for, with the reason.
KEPT_OUT_NOT_SCANNED: dict[str, str] = {
    "task": "kept out only for the work the bank gives one thread, a sense a scan cannot tell",
}

#: A format spec that prints a float past three significant figures:
#: Python's ``g`` with no precision (six), ``.Ng`` or ``.Ne`` past three
#: figures, ``.Nf`` with four decimals or more; Rust's ``.N`` (decimals) or
#: ``e`` with no precision.
_PY_DIGITS = re.compile(r"(?:^|:)[^:]*?(?:\.(\d+)([gGeEf%])|(?<![.\d])([gGeE]))$")
_RUST_DIGITS = re.compile(r":[^:]*?(?:\.(\d+)([eE]?)|(?<![.\d])([eE]))$")
_LITERAL_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?:e[-+]?\d+)?(?![\w.])")
#: A formula: a power or a product with a name on one side (``2^16``, a
#: number written as a power, is not one), a reciprocal, a norm.
FORMULA = re.compile(
    r"[A-Za-z_)\]]\s*(?:\^|\*\*)\s*[\w(]|[\w)\]]\s*(?:\^|\*\*)\s*[A-Za-z_(]"
    r"|[\w)]\s\*\s[\w(]|\b1\s*/\s*\(|\|\w+\|\^"
)


def _too_precise(spec: str, rust: bool) -> bool:
    if not spec:
        return False
    m = (_RUST_DIGITS if rust else _PY_DIGITS).search(":" + spec if rust else spec)
    if not m:
        return False
    digits, kind, bare = m.groups()
    if bare:
        return True
    n = int(digits)
    if rust:
        return n >= 3 if kind else n >= 4
    if kind in "gG":
        return n > 3
    if kind in "eE":
        return n >= 3
    return n >= 4


def _significant(number: str) -> int:
    mantissa = number.lower().split("e")[0].replace(".", "").lstrip("0")
    return len(mantissa)


def is_wrapper(m: Message) -> bool:
    """Whether a message is a wrapper, such as ``spec {:?}: {e}``: a where
    and placeholders, with no words of its own past the where."""
    plain = m.plain()
    where = where_of(m)
    body = plain[len(where) :] if where and plain.lstrip().startswith(where) else plain
    return not re.search(r"[A-Za-z]{2,}", body.replace("{}", " "))


def message_findings(m: Message, *, core: bool = False) -> list[Finding]:
    """The shape findings of one message (WRITING §11). ``core``: a string of
    ``online-core``, whose caller names where it happened, so it is not held
    to the *where* rule."""
    if is_wrapper(m):
        return []  # its text is its parts', each read where it is written
    plain = m.plain()
    where = where_of(m)
    body = plain[len(where) :] if where and plain.lstrip().startswith(where) else plain
    shown = m.display()
    found: list[Finding] = []

    def add(rule: str, held: bool = True) -> None:
        found.append(Finding("messages", m.path, m.line, rule, shown, held))

    if where is None and not core:
        add("where: no where-it-happened prefix")
    sents = sentences_of(body)
    if not sents or not is_imperative(sents[-1]):
        add("remedy: no closing sentence in the imperative")
    if len(sents) > 3:
        add(f"sentences: {len(sents)}, more than three")
    literal = m.literal()
    for word, pattern in KEPT_OUT.items():
        if re.search(pattern, literal, flags=0 if word[0].isupper() or word == "λ" else re.I):
            add(f"internal-word: {word}")
    rust = m.path.endswith(".rs")
    for kind, text in m.parts:
        if kind == "ph" and ":" in text and _too_precise(text.rsplit(":", 1)[1], rust):
            add(f"digits: {{{text}}} prints more than three significant figures")
    for num in _LITERAL_DECIMAL.findall(literal):
        if _significant(num) > 3:
            add(f"digits: {num} has more than three significant figures")
    if FORMULA.search(literal):
        add("formula: a formula in the message")
    return found


# ... Python ...


def _is_docstring(node: ast.AST, parent: ast.AST) -> bool:
    body = getattr(parent, "body", None)
    return (
        bool(body)
        and isinstance(parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and body[0] is node
    )  # type: ignore[index]


def _parts(node: ast.expr, scope: dict[str, ast.expr]) -> list[tuple[str, str]] | None:
    """A message expression as literal parts and placeholders, or None when
    a static reading cannot tell what it says."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [("lit", node.value)]
    if isinstance(node, ast.JoinedStr):
        out: list[tuple[str, str]] = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(("lit", str(v.value)))
            elif isinstance(v, ast.FormattedValue):
                spec = ""
                if isinstance(v.format_spec, ast.JoinedStr):
                    spec = "".join(
                        str(p.value) if isinstance(p, ast.Constant) else "?"
                        for p in v.format_spec.values
                    )
                out.append(("ph", ast.unparse(v.value) + (f":{spec}" if spec else "")))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _parts(node.left, scope), _parts(node.right, scope)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.IfExp):
        return _parts(node.body, scope)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
    ):
        pieces = re.split(r"(\{[^{}]*\})", node.func.value.value)
        return [("ph", p[1:-1]) if p.startswith("{") else ("lit", p) for p in pieces if p]
    if isinstance(node, ast.Name) and node.id in scope:
        return _parts(scope[node.id], {k: v for k, v in scope.items() if k != node.id})
    if isinstance(node, ast.Name | ast.Call | ast.Attribute | ast.Subscript):
        return [("ph", ast.unparse(node))]
    return None


@dataclass
class Unread:
    """A raise or warning whose message a static reading cannot see: a
    variable built elsewhere, or an exception a helper returns."""

    path: str
    line: int
    text: str


def _own(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of a module or function body, not those of a function
    nested in it, which is a scope of its own."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            stack.extend(ast.iter_child_nodes(node))


def python_messages(src: str, path: str) -> tuple[list[Message], list[Unread]]:
    """Every ``raise X(...)`` and ``warnings.warn(...)`` message in Python
    source. A message held in a name is read from the assignment to that
    name before it in the same function."""
    tree = ast.parse(src)
    found: list[Message] = []
    unread: list[Unread] = []

    # An exception built as an argument, such as a problem a check that
    # reports every problem at once appends to its list.
    kept = {
        id(a)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for a in n.args
        if isinstance(a, ast.Call)
    }

    def visit(fn: ast.AST) -> None:
        scope: dict[str, ast.expr] = {}
        nodes = sorted(
            (n for n in _own(fn) if isinstance(n, ast.Assign | ast.Raise | ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for node in nodes:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                scope[node.targets[0].id] = node.value
            elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                name = ast.unparse(node.exc.func)
                if not re.search(r"(Error|Exception|Warning|Exit)$", name):
                    unread.append(Unread(path, node.lineno, f"raise {name}(...)"))
                    continue
                if node.exc.args:
                    record(node.exc.args[0], node.lineno, "error", scope)
            elif isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name in ("warnings.warn", "warn", "_warn") and node.args:
                    record(node.args[0], node.lineno, "warning", scope)
                elif id(node) in kept and re.search(r"(Error|Exception)$", name) and node.args:
                    record(node.args[0], node.lineno, "error", scope)

    def record(arg: ast.expr, line: int, kind: str, scope: dict[str, ast.expr]) -> None:
        parts = _parts(arg, scope)
        if parts is None or all(k == "ph" for k, _ in parts):
            unread.append(Unread(path, line, ast.unparse(arg)))
        else:
            found.append(Message(path, line, kind, parts))

    visit(tree)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            visit(fn)
    found.sort(key=lambda m: m.line)
    return found, unread


# ... Rust ...

_TEST_MODULE = re.compile(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{")
_CHAR = re.compile(r"'(\\.|[^\\'])'")
_RAW = re.compile(r'b?r(#*)"')


def rust_strings(src: str) -> Iterator[tuple[int, int, str]]:
    """Each string literal in shipped Rust source, as (offset, line, text),
    a ``\\`` line continuation folded as the compiler folds it; comments,
    char literals and every ``#[cfg(test)] mod x { .. }`` skipped. The
    same walk as ``tests/test_no_plan_references.py``'s, with offsets."""
    i, n = 0, len(src)
    depth, skip_to = 0, -1
    while i < n:
        if src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end < 0 else end
        elif src.startswith("/*", i):
            i = src.find("*/", i) + 2
        elif src[i] == "'":
            m = _CHAR.match(src, i)
            i += len(m.group(0)) if m else 1
        elif skip_to < 0 and (m := _TEST_MODULE.match(src, i)):
            skip_to, depth = depth, depth + 1
            i = m.end()
        elif src[i] == "{":
            depth += 1
            i += 1
        elif src[i] == "}":
            depth -= 1
            if depth == skip_to:
                skip_to = -1
            i += 1
        elif (m := _RAW.match(src, i)) and not (i and (src[i - 1].isalnum() or src[i - 1] == "_")):
            close = '"' + m.group(1)
            end = src.find(close, m.end())
            if skip_to < 0:
                yield i, src.count("\n", 0, i) + 1, src[m.end() : end]
            i = end + len(close)
        elif src[i] == '"':
            j, text = i + 1, []
            while src[j] != '"':
                if src[j] == "\\" and src[j + 1] == "\n":
                    j += 2
                    while src[j] in " \t\n":
                        j += 1
                elif src[j] == "\\":
                    text.append(src[j : j + 2])
                    j += 2
                else:
                    text.append(src[j])
                    j += 1
            if skip_to < 0:
                yield i, src.count("\n", 0, i) + 1, "".join(text)
            i = j + 1
        else:
            i += 1


#: What precedes a Rust string that is a message, and of which kind: an
#: error returned or raised to Python, a warning (a notice the bank
#: collects for Python to issue), or text the command line prints.
RUST_OPENERS: list[tuple[str, re.Pattern[str]]] = [
    ("error", re.compile(r"\bErr\(\s*(?:format!\(\s*)?$")),
    (
        "error",
        re.compile(r"\b(?:polars_bail|polars_err|polars_ensure)!\((?:[^;]*,)?\s*\w+\s*:\s*$"),
    ),
    ("error", re.compile(r"\brefuse\(\s*(?:format!\(\s*)?$")),
    ("error", re.compile(r"\bComputeError\(\s*(?:format!\(\s*)?$")),
    ("error", re.compile(r"\bnew_err\(\s*(?:format!\(\s*)?$")),
    ("error", re.compile(r"\b(?:bail|anyhow|format_err)!\(\s*$")),
    ("error", re.compile(r"\bok_or(?:_else)?\(\s*(?:\|\|\s*)?(?:format!\(\s*)?$")),
    ("error", re.compile(r"\bmap_err\(\s*\|[^|]*\|\s*(?:format!\(\s*)?$")),
    ("warning", re.compile(r"\bnotices?\.push\(\s*(?:format!\(\s*)?$")),
    ("printed", re.compile(r"\beprintln!\(\s*$")),
]


def _rust_parts(text: str) -> list[tuple[str, str]]:
    text = text.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\").replace("\\t", " ")
    out: list[tuple[str, str]] = []
    for piece in re.split(r"(\{\{|\}\}|\{[^{}]*\})", text):
        if piece in ("{{", "}}"):
            out.append(("lit", piece[0]))
        elif piece.startswith("{") and piece.endswith("}"):
            out.append(("ph", piece[1:-1]))
        elif piece:
            out.append(("lit", piece))
    return out


def rust_messages(src: str, path: str) -> list[Message]:
    """Every message in shipped Rust source: a string that an error, a
    notice or a printed line opens with (``RUST_OPENERS``). A panic's or an
    ``expect``'s text is not one: it reports a defect, not a reader's
    mistake."""
    out = []
    for at, line, text in rust_strings(src):
        before = " ".join(src[max(0, at - 160) : at].split())
        for kind, opener in RUST_OPENERS:
            if opener.search(before):
                out.append(Message(path, line, kind, _rust_parts(text)))
                break
    return out


#: The Rust sources whose messages reach a reader, and the crate whose
#: strings are bodies its callers name.
RUST_CRATES = ("online-core", "online-polars", "online-cli", "online-py")


@cache
def all_messages(repo: Path = REPO) -> tuple[list[Message], list[Unread]]:
    msgs: list[Message] = []
    unread: list[Unread] = []
    for path in sorted((repo / "python" / "polars_online").rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        got, lost = python_messages(path.read_text(encoding="utf-8"), rel)
        msgs += got
        unread += lost
    for crate in RUST_CRATES:
        for path in sorted((repo / "crates" / crate / "src").rglob("*.rs")):
            rel_parts = path.relative_to(repo / "crates" / crate / "src").parts
            if (
                path.name in ("oracle.rs", "tests.rs")
                or path.stem.endswith("_tests")
                or "tests" in rel_parts[:-1]
            ):
                continue  # compiled for tests only
            rel = path.relative_to(repo).as_posix()
            msgs += rust_messages(path.read_text(encoding="utf-8"), rel)
    return msgs, unread


def check_messages(repo: Path = REPO, read: bool = True) -> list[Finding]:
    msgs, _ = all_messages(repo)
    found: list[Finding] = []
    for m in msgs:
        found += message_findings(m, core=m.path.startswith("crates/online-core/"))
    return found


# --- the docstrings ---------------------------------------------------------------

#: §10's sections, in §10's order, each with the headings that name it.
SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("Parameters", ("parameters",)),
    ("Returns", ("returns", "yields", "output", "outputs", "output fields", "the output fields")),
    ("Raises", ("raises", "warns")),
    (
        "Update equation",
        ("update equation", "update equations", "the update", "the update equation"),
    ),
    ("Examples", ("examples", "example")),
    ("See also", ("see also",)),
    ("Notes", ("notes",)),
]
_SECTION_OF = {alias: name for name, aliases in SECTIONS for alias in aliases}
_ORDER = {name: i for i, (name, _) in enumerate(SECTIONS)}
HEADING = re.compile(r"^\.\. rubric:: (.+)$|^([^\s].*)\n(?:-{3,}|={3,}|~{3,})$", re.M)


@dataclass(frozen=True)
class Documented:
    """A public object whose docstring the check reads: its dotted name, its
    kind (model, function, class, method or property), whether it takes
    parameters, whether it returns a value, and its source file."""

    where: str
    kind: str
    doc: str | None
    path: str
    line: int = 0
    params: bool = False
    returns: bool = False


def docstring_findings(d: Documented) -> list[Finding]:
    """The section findings of one docstring (WRITING §10)."""
    found: list[Finding] = []

    def add(rule: str, held: bool = True) -> None:
        found.append(Finding("docstrings", d.path, d.line, rule, d.where, held))

    if not d.doc or not d.doc.strip():
        add("missing: no docstring")
        return found
    doc = d.doc.strip()
    summary = doc.split("\n\n", 1)[0]
    plain = re.sub(r"``[^`]*``|:\w+:`[^`]*`|`[^`]*`", "code", " ".join(summary.split()))
    if len(sentences_of(plain)) > 1:
        add("summary: more than one sentence")
    if re.search(r" -- | — |\(|;", plain):
        add("summary: an aside (a dash, a parenthesis or a semicolon)")
    heads = [(m.start(), (m.group(1) or m.group(2)).strip()) for m in HEADING.finditer(doc)]
    if d.kind in ("property",):
        return found
    names: list[str] = []
    notes_seen = False
    for _, title in heads:
        name = _SECTION_OF.get(title.lower().rstrip(":"))
        if name is None:
            if not notes_seen:
                add(f"unknown-section: {title!r}, not one of §10's, before Notes")
            continue
        if name in names:
            add(f"duplicate: {name}")
        names.append(name)
        notes_seen = notes_seen or name == "Notes"
    order = [_ORDER[n] for n in names]
    for i, n in enumerate(names):
        if any(order[j] > order[i] for j in range(i)):
            add(f"order: {n} after {names[max(range(i), key=lambda j: order[j])]}")
    need: list[str] = []
    if d.params:
        need.append("Parameters")
    if d.returns or d.kind == "model":
        need.append("Returns")
    if d.kind == "model":
        need += ["Raises", "Update equation"]
    if d.kind in ("model", "function", "class", "method"):
        need.append("Examples")
    for n in need:
        if n not in names:
            add(f"missing: {n}")
    if d.kind == "model":
        first = heads[0][0] if heads else len(doc)
        intro = doc[len(summary) : first].strip()
        n = len(sentences_of(re.sub(r"``[^`]*``", "code", intro)))
        if not intro:
            add("missing: when to use it")
        elif n > 3:
            add(f"when: {n} sentences before the first section, more than three")
    return found


#: Where the extension's objects are written: their docstrings are the
#: bindings' doc comments.
NATIVE_SOURCE = "crates/online-py/src/lib.rs"


def _source(obj: object) -> tuple[str, int]:
    """The file and line an object's docstring is written at: its Python
    source, or the bindings' for an object of the extension."""
    if (getattr(obj, "__module__", "") or "").endswith("._polars_online"):
        return NATIVE_SOURCE, 0
    try:
        target = inspect.unwrap(obj)  # type: ignore[arg-type]
        src = inspect.getsourcefile(target)
        line = inspect.getsourcelines(target)[1]
    except (TypeError, OSError):
        return NATIVE_SOURCE, 0
    if src:
        try:
            return Path(src).resolve().relative_to(REPO.resolve()).as_posix(), line
        except ValueError:
            pass
    return NATIVE_SOURCE, 0


def _signature(obj: object) -> tuple[bool, bool]:
    """Whether a callable takes parameters (``self`` aside) and returns a
    value, read from its signature; neither when it has none to read."""
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False, False
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    ret = sig.return_annotation
    returns = ret is not inspect.Signature.empty and ret not in (None, "None")
    return bool(params), returns


@cache
def documented() -> list[Documented]:
    """Every public object of the API surface (``tests/api_surface.txt``):
    the package's ``__all__``, each public module's, each class's public
    methods and properties, and the frame namespaces' methods."""
    import polars as pl

    import polars_online as po

    seen: set[int] = set()
    out: list[Documented] = []

    def add(where: str, obj: object, kind: str | None = None) -> None:
        if id(obj) in seen or isinstance(
            obj, (str, int, float, bool, tuple, list, dict, frozenset)
        ):
            return
        if inspect.ismodule(obj):
            return
        seen.add(id(obj))
        if isinstance(obj, property):
            out.append(Documented(where, "property", obj.__doc__, *_source(obj.fget)))
            return
        if inspect.isclass(obj):
            out.append(Documented(where, "class", inspect.getdoc(obj), *_source(obj)))
            if (obj.__module__ or "").startswith("polars_online"):
                for name, member in vars(obj).items():
                    if not name.startswith("_"):
                        target = (
                            member.__func__
                            if isinstance(member, (staticmethod, classmethod))
                            else member
                        )
                        add(f"{where}.{name}", target, "method")
            return
        if not callable(obj):
            return
        params, returns = _signature(obj)
        if kind is None:
            kind = "function"
        out.append(Documented(where, kind, inspect.getdoc(obj), *_source(obj), params, returns))

    for name in po.__all__:
        add(f"polars_online.{name}", getattr(po, name))
    modules = sorted(
        m.name for m in pkgutil.iter_modules(po.__path__) if not m.name.startswith("_")
    )
    for mod in modules:
        module = importlib.import_module(f"polars_online.{mod}")
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            kind = None
            if mod == "spec" and callable(obj):
                try:
                    first = next(iter(inspect.signature(obj).parameters), "")
                except (TypeError, ValueError):
                    first = ""
                kind = "model" if first == "name" else None
            add(f"polars_online.{mod}.{name}", obj, kind)
    for frame in (pl.LazyFrame(), pl.DataFrame()):
        ns = type(frame.online)  # type: ignore[attr-defined]
        for name, member in vars(ns).items():
            if not name.startswith("_") and callable(member):
                add(f"{type(frame).__name__}.online.{name}", member, "method")
    return out


def check_docstrings(repo: Path = REPO, read: bool = True) -> list[Finding]:
    found: list[Finding] = []
    for d in documented():
        found += docstring_findings(d)
    return found


# --- the vocabulary -----------------------------------------------------------------

#: Pages that name retired words and British spellings as examples of what
#: to replace, so the scan cannot read them.
VOCABULARY_EXEMPT = {
    "docs/WRITING.md": "names the words it retires",
    "docs/GLOSSARY.md": "the same",
}


#: A source file's docstrings and its messages are rewritten in different
#: stages, so the vocabulary counts each apart, as the file with one of
#: these after it, and each can finish on its own.
DOCSTRINGS_PART = "#docstrings"
MESSAGES_PART = "#messages"


def check_vocabulary(repo: Path = REPO, read: bool = True) -> list[Finding]:
    found: list[Finding] = []
    for path in [*guides(repo), "llms.txt"]:
        if path in VOCABULARY_EXEMPT:
            continue
        found += vocabulary_findings(_markdown_prose(guide_text(path, repo)), path)
    for d in documented():
        if d.doc:
            part = d.path + DOCSTRINGS_PART
            found += vocabulary_findings(d.doc, part, where=d.where, line0=d.line)
    msgs, _ = all_messages(repo)
    for m in msgs:
        for f in vocabulary_findings(m.literal(), m.path, line0=m.line):
            # A word the glossary keeps out of messages is the message
            # check's finding, counted there once.
            if f.rule.split(": ", 1)[1] not in KEPT_OUT:
                text = m.display()[:160]
                found.append(Finding(f.check, m.path + MESSAGES_PART, m.line, f.rule, text, f.held))
    return [f for f in found if read or f.held]


# --- the sentences --------------------------------------------------------------

#: The mean sentence, in words, past which a guide has a finding: WRITING §5
#: aims near 20, and the README's rewrite reached 19.9.
MEAN_LIMIT = 22.0
LONG = 45

#: §6's numbered counts, by their title's number: whether each is held (a
#: verdict) or to read (a list of sentences to read), and its rule's name.
NUMBERED: dict[str, tuple[str, bool]] = {
    "1": ("no-opener", True),
    "2": ("contents-opener", True),
    "3": ("semicolon", False),
    "4": ("back-reference", False),
    "5": ("inverted", True),
    "6": ("about-the-document", True),
    "7": ("is-a-lead", True),
    "8": ("negation", False),
    "9": ("relative", False),
    "10": ("names-not-shown", False),
    "11": ("python-word", True),
}


def sentence_findings(text: str, path: str, *, names: bool = False) -> list[Finding]:
    """§5's and §6's findings on one guide. ``names``: the tenth count too,
    which reads the package's signatures."""
    counts, sentences = doc_review.measure(text)
    found: list[Finding] = []
    for line, words, s in sentences:
        if words >= LONG:
            found.append(Finding("sentences", path, line, f"{LONG}+: {words} words", s))
        elif words >= 35:
            found.append(Finding("sentences", path, line, f"35+: {words} words", s, False))
    if counts["mean sentence"] > MEAN_LIMIT:
        found.append(
            Finding(
                "sentences",
                path,
                0,
                f"mean: {counts['mean sentence']} words, past {MEAN_LIMIT}",
                f"{counts['sentences']} sentences",
            )
        )
    prose = " ".join(s for _, _, s in sentences)
    for m in doc_review.COST.finditer(prose):
        found.append(Finding("sentences", path, 0, "cost-word", m.group(0), False))
    for m in doc_review.APHORISM.finditer(prose):
        found.append(Finding("sentences", path, 0, "x-not-y", m.group(0), False))
    original = doc_review.described_not_shown
    if not names:
        doc_review.described_not_shown = lambda lines: []
    try:
        numbered = doc_review.counts(text)
    finally:
        doc_review.described_not_shown = original
    for title, hits in numbered:
        rule, held = NUMBERED[title.split(".", 1)[0]]
        for line, hit in hits:
            if rule == "no-opener" and hit.endswith("explains the subsections]"):
                found.append(Finding("sentences", path, line, "table-opener", hit, False))
            else:
                found.append(Finding("sentences", path, line, rule, hit, held))
    return found


def check_sentences(repo: Path = REPO, read: bool = True) -> list[Finding]:
    found: list[Finding] = []
    for path in guides(repo):
        found += sentence_findings(guide_text(path, repo), path, names=read)
    return [f for f in found if read or f.held]


# --- the page kinds ------------------------------------------------------------------

#: The kinds a page may declare (WRITING §9): the four, and the three
#: outside them.
KINDS = ("tutorial", "how-to", "reference", "explanation", "landing", "map", "record")
#: A page declares its kind in an HTML comment on its first line, which
#: GitHub does not show, so the declaration adds no sentence about the
#: page (P1) and leaves its first sentence to say what it is about (P12).
DECLARATION = re.compile(r"\A\s*<!--\s*kind:\s*([\w-]+)\s*-->")


def kind_findings(text: str, path: str) -> list[Finding]:
    """§9's findings on one guide: its declaration, and a how-to's
    prerequisites and numbered steps."""
    found: list[Finding] = []
    m = DECLARATION.match(text)
    if not m:
        return [
            Finding("kinds", path, 1, "undeclared: no <!-- kind: ... --> on the first line", path)
        ]
    kind = m.group(1)
    if kind not in KINDS:
        found.append(Finding("kinds", path, 1, f"unknown-kind: {kind!r}", path))
    if kind == "how-to":
        tokens = doc_review.PARSER.parse(text)
        ordered = next(
            (t.map[0] + 1 for t in tokens if t.type == "ordered_list_open" and t.map), None
        )
        bullet = next(
            (t.map[0] + 1 for t in tokens if t.type == "bullet_list_open" and t.map), None
        )
        if ordered is None:
            found.append(
                Finding("kinds", path, 1, "no-steps: a how-to with no numbered steps", path)
            )
        if bullet is None or (ordered is not None and bullet > ordered):
            found.append(
                Finding("kinds", path, 1, "no-prerequisites: no list before the first step", path)
            )
    return found


def check_kinds(repo: Path = REPO, read: bool = True) -> list[Finding]:
    found: list[Finding] = []
    for path in guides(repo):
        found += kind_findings((repo / path).read_text(encoding="utf-8"), path)
    return found


RUNNERS: dict[str, Callable[[Path, bool], list[Finding]]] = {
    "messages": check_messages,
    "docstrings": check_docstrings,
    "vocabulary": check_vocabulary,
    "sentences": check_sentences,
    "kinds": check_kinds,
}


@cache
def run(check: str, read: bool = False) -> tuple[Finding, ...]:
    """Every finding of one check over the repository: the held ones, and
    with ``read`` the ones to read too."""
    return tuple(RUNNERS[check](REPO, read))


def files_read(check: str, repo: Path = REPO) -> list[str]:
    """Every file a check reads, so the ratchet can tell a file it has seen
    clean from one it has never seen."""
    if check in ("sentences", "kinds"):
        return guides(repo)
    if check == "vocabulary":
        out = [p for p in [*guides(repo), "llms.txt"] if p not in VOCABULARY_EXEMPT]
        out += [d.path + DOCSTRINGS_PART for d in documented()]
        out += [m.path + MESSAGES_PART for m in all_messages(repo)[0]]
        return sorted(set(out))
    if check == "docstrings":
        return sorted({d.path for d in documented()})
    return sorted({m.path for m in all_messages(repo)[0]})


# --- the ratchet ----------------------------------------------------------------------

Key = tuple[str, str, str]  # (check, path, rule)


def rule_key(rule: str) -> str:
    """A rule as the ratchet counts it: its name, before its detail."""
    return rule.split(":", 1)[0].strip()


@dataclass
class Ratchet:
    """``writing_ratchet.toml``: each check's counts by file and rule, the
    files each check holds to no finding, and the findings accepted there."""

    counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    finished: dict[str, list[str]] = field(default_factory=dict)
    accepted: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Ratchet:
        data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        finished = {c: list(data.get("finished", {}).get(c, [])) for c in CHECKS}
        counts = {c: {p: dict(v) for p, v in data.get(c, {}).items()} for c in CHECKS}
        return cls(counts, finished, list(data.get("accepted", [])))

    def dump(self) -> str:
        lines = [HEADER.rstrip(), "", "[finished]"]
        for c in CHECKS:
            items = ", ".join(json.dumps(p) for p in sorted(self.finished.get(c, [])))
            lines.append(f"{c} = [{items}]")
        for entry in self.accepted:
            lines += ["", "[[accepted]]"]
            lines += [
                f"{k} = {json.dumps(entry[k])}" for k in ("check", "path", "rule", "text", "reason")
            ]
        for c in CHECKS:
            for p in sorted(self.counts.get(c, {})):
                lines += ["", f"[{c}.{json.dumps(p)}]"]
                rules = self.counts[c][p]
                lines += [f"{json.dumps(r)} = {n}" for r, n in sorted(rules.items()) if n]
        return "\n".join(lines) + "\n"


HEADER = """\
# The writing checks' ratchet (docs/WRITING.md §13; scripts/writing_checks.py).
#
# Each check's held findings, counted by file and rule. A count may only
# fall: a file over its count fails, and a file under it fails too, naming
# the count to tighten it to, so the change that made the gain keeps it
# (`uv run python scripts/writing_checks.py apply` writes it). A file a check
# reads has a table here even when it is clean, so a new file shows in the
# diff that adds it; a source file's docstrings and messages are counted
# apart in the vocabulary, as `x.py#docstrings` and `x.py#messages`, since
# different stages rewrite them. A file under [finished] fails on any held
# finding but those [[accepted]] with a reader's reason: it moves there when
# the stage that rewrites it has finished (docs/records/DOCS-REWRITE-MAP.md,
# "When each check fails the gate"). `scripts/writing_checks.py report`
# prints every finding, the ones to read among them.
"""


def measured(
    check: str, findings: Iterable[Finding], paths: Iterable[str]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {p: {} for p in paths}
    for f in findings:
        if f.held and f.check == check:
            rules = out.setdefault(f.path, {})
            rules[rule_key(f.rule)] = rules.get(rule_key(f.rule), 0) + 1
    return out


def _accepted(f: Finding, accepted: list[dict[str, str]]) -> bool:
    return any(
        a["check"] == f.check
        and a["path"] == f.path
        and rule_key(a["rule"]) == rule_key(f.rule)
        and a["text"] in f.text
        for a in accepted
    )


def compare(
    check: str, findings: Iterable[Finding], paths: Iterable[str], ratchet: Ratchet
) -> list[str]:
    """What breaks the ratchet: a count over its ratchet or under it, a file
    it has not seen, a finding in a finished file. Empty when it holds."""
    findings = list(findings)
    problems: list[str] = []
    finished = set(ratchet.finished.get(check, []))
    for f in findings:
        if f.held and f.path in finished and not _accepted(f, ratchet.accepted):
            problems.append(f"{f.show()}: {f.path} is finished for {check}, so any finding fails")
    now = measured(check, (f for f in findings if f.path not in finished), paths)
    held = ratchet.counts.get(check, {})
    for path in sorted(set(now) | set(held)):
        if path in finished:
            continue
        if path not in held:
            problems.append(
                f"{check}: {path} is not in {RATCHET}: "
                f"`apply` records it at {sum(now[path].values())}"
            )
            continue
        if path not in now:
            problems.append(f"{check}: {path} is no longer read: `apply` drops it")
            continue
        for rule in sorted(set(now[path]) | set(held[path])):
            got, cap = now[path].get(rule, 0), held[path].get(rule, 0)
            if got > cap:
                problems.append(
                    f"{check}: {path}: {rule}: {got} findings, over its ratchet of {cap}"
                )
            elif got < cap:
                problems.append(
                    f"{check}: {path}: {rule}: {got} findings, under its ratchet of {cap}: "
                    f"tighten it to {got} (`apply`)"
                )
    return problems


def applied(
    check: str, findings: Iterable[Finding], paths: Iterable[str], ratchet: Ratchet
) -> dict[str, dict[str, int]]:
    """The check's counts after ``apply``: each lowered to what is measured,
    a file never seen recorded at its count, a file no longer read dropped.
    A count over its ratchet stays where it was."""
    finished = set(ratchet.finished.get(check, []))
    now = measured(check, findings, paths)
    held = ratchet.counts.get(check, {})
    out: dict[str, dict[str, int]] = {}
    for path, rules in now.items():
        if path in finished:
            continue
        if path not in held:
            out[path] = dict(rules)
            continue
        out[path] = {
            r: min(rules.get(r, 0), held[path].get(r, 0)) for r in set(rules) | set(held[path])
        }
        for r, cap in held[path].items():
            if rules.get(r, 0) > cap:
                out[path][r] = cap
    return out


# --- the command line ----------------------------------------------------------------


def report(checks: list[str], *, listing: bool, where: str) -> str:
    out: list[str] = []
    for check in checks:
        findings = [f for f in run(check, True) if where in f.path]
        held = [f for f in findings if f.held]
        out.append(f"== {check}: {len(held)} held, {len(findings) - len(held)} to read")
        # The vocabulary's rules by the word, the others' by their name.
        by_rule = Counter(
            (f.rule if check == "vocabulary" else rule_key(f.rule), f.held) for f in findings
        )
        for (rule, is_held), n in sorted(
            by_rule.items(), key=lambda kv: (-kv[0][1], -kv[1], kv[0][0])
        ):
            out.append(f"   {n:6d}  {rule}{'' if is_held else ' (to read)'}")
        by_file = Counter(f.path for f in held)
        out.append("   held, by file:")
        out += [
            f"   {n:6d}  {p}" for p, n in sorted(by_file.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        if check == "messages":
            msgs, unread = all_messages()
            wrappers = sum(not message_findings(m) and is_wrapper(m) for m in msgs)
            out.append(
                f"   messages read: {len(msgs)}, {wrappers} of them wrappers whose text is their "
                f"parts'; raises and warnings a static reading cannot see: {len(unread)}"
            )
        if listing:
            out += [
                f"   {f.show()}" for f in sorted(findings, key=lambda f: (f.path, f.line, f.rule))
            ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="print every finding's totals, or with --list every finding")
    rp.add_argument("--check", action="append", choices=CHECKS)
    rp.add_argument("--list", action="store_true", help="list each finding")
    rp.add_argument("--path", default="", help="only files whose path contains this")
    sub.add_parser("check", help="compare the held findings with the ratchet")
    sub.add_parser("apply", help="lower the ratchet to what is measured")
    args = ap.parse_args()
    if args.cmd == "report":
        print(report(args.check or list(CHECKS), listing=args.list, where=args.path))
        return 0
    ratchet = Ratchet.load(REPO / RATCHET)
    if args.cmd == "check":
        problems = [p for c in CHECKS for p in compare(c, run(c), files_read(c), ratchet)]
        print("\n".join(problems) or "the ratchet holds")
        return 1 if problems else 0
    for c in CHECKS:
        ratchet.counts[c] = applied(c, run(c), files_read(c), ratchet)
    (REPO / RATCHET).write_text(ratchet.dump(), encoding="utf-8")
    print(f"wrote {RATCHET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
