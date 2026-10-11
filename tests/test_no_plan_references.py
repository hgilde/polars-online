"""No reader-facing text cites the plan, a task, a review or a record.

The user, 2026-10-10: "No doc should reference the plan, everything should
be explained from first principals." A reader of the API reference, or of
``help()``, has none of ``docs/PLAN.md``, a task's number, a review's
finding or a record such as ``docs/WARMUP-AND-CONVERGENCE.md``; a citation
of one tells them only that a reason exists somewhere else. So the text
states the reason itself: the mechanism, the measurement and how it was
made, the trade-off. A link to a guide (the *Guides* table of
``docs/README.md``) is fine, since a guide is written to be read.

Two lists hold it. ``PATTERNS`` is what may not be cited. ``FILES`` is the
reader-facing files held to it beside every public docstring; the README,
the guides, ``CONTRIBUTING.md`` and ``llms.txt`` join it as each is
rewritten to the rule. Source comments may keep their citations: they are
read by whoever changes the code, beside the plan.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import polars_online as po

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "docs" / "reference"

#: What a reader-facing text may not cite, each with what it names. The
#: records are the documents ``docs/README.md`` lists under *Records*.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("the plan", re.compile(r"\bPLAN\b")),
    ("a task by its number", re.compile(r"\b[Tt]asks? \d+")),
    (
        "a review",
        re.compile(
            r"\b[Rr]eview (?:round |of )?(?:\d|R\d)"  # review 2026-10-05, review round 4, review R2
            r"|\b(?:first|second|third|fourth|fifth) review\b"
            r"|\bREVIEW-[\w.-]*"
        ),
    ),
    (
        "a record",
        re.compile(
            r"\b(?:ENHANCEMENTS|IMPROVEMENTS|SIMPLIFICATION|README-ITERATIONS|PHRASING"
            r"|BEYOND-O-STATE|CLUSTERING|BOOSTED-TREES|ARROW-SOURCES|WARMUP-AND-CONVERGENCE"
            r"|MARGINAL-[A-Z-]+|GBDT-ADAPTER|ANSWERS-E54-E64)\b"
            r"|\brecords/"
        ),
    ),
    ("an entry of ENHANCEMENTS by its number", re.compile(r"\bE\d{1,3}\b")),
]

#: The reader-facing files held to ``PATTERNS`` beside the docstrings: the
#: reference's own pages, and the files already clean of citations.
FILES: list[str] = [
    "llms.txt",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "docs/OUTPUTS.md",
    "docs/MODELS.md",
    "docs/WINDOWS.md",
    "docs/RUNNER.md",
    "docs/GBDT.md",
    *sorted(str(p.relative_to(REPO)) for p in REFERENCE.glob("*.rst")),
]


def citations(text: str) -> list[str]:
    """Every citation in ``text``, each as ``what: the words around it``.
    Whitespace is folded first, so a citation broken across lines is one."""
    flat = " ".join(text.split())
    found = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(flat):
            found.append(f"{label}: ...{flat[max(0, m.start() - 60) : m.end() + 40]}...")
    return found


# --------------------------------------------------------------- the patterns


@pytest.mark.parametrize(
    "text",
    [
        "(docs/PLAN.md task 143, *Numerics*)",
        "docs/PLAN.md §3",
        "since task 211",
        "(review 2026-10-05, YB2)",
        "(review round 4, PD3)",
        "the second review of 2026-09-15",
        "docs/REVIEW-2026-09-18.md",
        "docs/WARMUP-AND-CONVERGENCE.md",
        "records/PHRASING.md",
        "(E37)",
    ],
)
def test_each_kind_of_citation_is_caught(text):
    assert citations(text), text


@pytest.mark.parametrize(
    "text",
    [
        "tests/check_streams.py's AUDIT_PLANTED",
        "a query plan, which runs only when you ask for its result",
        "the plan form -- never by predict",
        "the reader's task, in one pass",
        "1E5 rows, as 1e5",
        "docs/PERFORMANCE.md §37 and docs/DIAGNOSTICS.md#is-it-ready",
        "reviewed by statsmodels' OLS",
    ],
)
def test_what_is_not_a_citation_passes(text):
    assert not citations(text), citations(text)


# ------------------------------------------------------------- the docstrings

_DIRECTIVE = re.compile(r"^\.\. auto(module|class|function):: (\S+)\n((?:[ \t]+:.*\n)*)", re.M)


def _resolve(dotted: str) -> object:
    """``polars_online.ModelBank`` or ``polars_online._frame.X`` as an object."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise LookupError(dotted)


def _members(cls: type) -> Iterator[tuple[str, object]]:
    """A class's public members, with ``__init__``, as autodoc documents them."""
    for name, member in vars(cls).items():
        if not name.startswith("_") or name == "__init__":
            yield name, member


def documented() -> Iterator[tuple[str, object]]:
    """Every object whose docstring the API reference renders, read from its
    pages' directives, beside every public name ``tests/api_surface.txt``
    lists: the package's ``__all__`` and each helper module's."""
    seen: set[int] = set()

    def visit(where: str, obj: object) -> Iterator[tuple[str, object]]:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        yield where, obj
        if inspect.isclass(obj) and (obj.__module__ or "").startswith("polars_online"):
            for name, member in _members(obj):
                yield from visit(f"{where}.{name}", member)

    for page in sorted(REFERENCE.glob("*.rst")):
        for kind, target, options in _DIRECTIVE.findall(page.read_text(encoding="utf-8")):
            obj = _resolve(target)
            yield from visit(target, obj)
            if kind != "module" or ":no-members:" in options:
                continue
            listed = re.search(r":members:(.*)", options)
            names = [n.strip() for n in (listed.group(1) if listed else "").split(",")]
            names = [n for n in names if n] or list(getattr(obj, "__all__", ()))
            for name in names:
                yield from visit(f"{target}.{name}", getattr(obj, name))
    modules = [po] + [importlib.import_module(f"polars_online.{m}") for m in _helper_modules()]
    for module in modules:
        yield from visit(module.__name__, module)
        for name in getattr(module, "__all__", ()):
            yield from visit(f"{module.__name__}.{name}", getattr(module, name))


def _helper_modules() -> list[str]:
    from test_api_surface import helper_modules

    return [*helper_modules(), "spec"]


DOCSTRINGS = [
    (where, doc)
    for where, obj in documented()
    if not isinstance(obj, (str, int, float, bool)) and (doc := inspect.getdoc(obj))
]


def test_the_docstrings_were_found():
    """A check that reads nothing passes: 157 docstrings carried text when
    this was written, every page of the reference among them."""
    assert len(DOCSTRINGS) > 150, len(DOCSTRINGS)
    pages = {where.split(".")[1] if where.count(".") else where for where, _ in DOCSTRINGS}
    assert {"spec", "ops", "stream", "gram", "ModelBank", "ArrowStruct"} <= pages, pages


def test_no_public_docstring_cites_the_plan_a_task_a_review_or_a_record():
    found = {where: hits for where, doc in DOCSTRINGS if (hits := citations(doc))}
    assert not found, "\n".join(f"{where}: {hit}" for where, hits in found.items() for hit in hits)


# ------------------------------------------------------------------ the files


@pytest.mark.parametrize("path", FILES)
def test_no_reader_facing_file_cites_the_plan_a_task_a_review_or_a_record(path):
    found = citations((REPO / path).read_text(encoding="utf-8"))
    assert not found, "\n".join(found)
