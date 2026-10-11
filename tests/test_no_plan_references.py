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

import ast
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
    # Not T-E9: TESTING.md's own entry IDs, which the code and tests cite.
    ("an entry of ENHANCEMENTS by its number", re.compile(r"(?<!T-)\bE\d{1,3}\b")),
]

#: The reader-facing files held to ``PATTERNS`` beside the docstrings: the
#: reference's own pages, and the files already clean of citations.
FILES: list[str] = [
    "llms.txt",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "docs/OUTPUTS.md",
    "docs/GRAM.md",
    "docs/DIAGNOSTICS.md",
    "docs/DATA-QUALITY.md",
    "docs/MODELS.md",
    "docs/WINDOWS.md",
    "docs/RUNNER.md",
    "docs/GBDT.md",
    "docs/REGIMES.md",
    "docs/PERFORMANCE.md",
    "docs/RELEASE-READINESS.md",
    "docs/TESTING.md",
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
        "the TESTING.md entry T-E9",
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


# --------------------------------------------------------------- the messages
#
# An error or warning message is read by whoever hit it, often from an
# installed wheel, so it is held like a docstring. Every string literal in the
# shipped source is a candidate message: the Rust crates' (outside their test
# modules, which are never shipped) and the package's (docstrings apart, which
# the check above reads). A comment may keep its citation.

_TEST_MODULE = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*mod \w+", re.M)
_CHAR = re.compile(r"'(\\.|[^\\'])'")
_RAW = re.compile(r'b?r(#*)"')


def rust_strings(src: str) -> Iterator[tuple[int, str]]:
    """Each string literal in Rust source, as (line, text), with a ``\\``
    line continuation folded away as the compiler folds it. Comments, char
    literals and everything from a trailing ``#[cfg(test)] mod`` on are
    skipped."""
    if m := _TEST_MODULE.search(src):
        src = src[: m.start()]
    i, n = 0, len(src)
    while i < n:
        if src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end < 0 else end
        elif src.startswith("/*", i):
            i = src.find("*/", i) + 2
        elif src[i] == "'":
            m = _CHAR.match(src, i)
            i += len(m.group(0)) if m else 1
        elif (m := _RAW.match(src, i)) and not (i and (src[i - 1].isalnum() or src[i - 1] == "_")):
            close = '"' + m.group(1)
            end = src.find(close, m.end())
            yield src.count("\n", 0, i) + 1, src[m.end() : end]
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
            yield src.count("\n", 0, i) + 1, "".join(text)
            i = j + 1
        else:
            i += 1


def python_strings(src: str) -> Iterator[tuple[int, str]]:
    """Each string constant in Python source but the docstrings, f-strings'
    literal parts included."""
    tree = ast.parse(src)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def shipped_strings() -> Iterator[tuple[str, int, str]]:
    for path in sorted((REPO / "crates").glob("online-*/src/**/*.rs")):
        if path.name == "oracle.rs":  # compiled for tests only
            continue
        for line, text in rust_strings(path.read_text(encoding="utf-8")):
            yield str(path.relative_to(REPO)), line, text
    for path in sorted((REPO / "python" / "polars_online").rglob("*.py")):
        for line, text in python_strings(path.read_text(encoding="utf-8")):
            yield str(path.relative_to(REPO)), line, text


def test_the_string_scanner_reads_what_the_compiler_reads():
    src = (
        'let a = \'"\'; // "task 1" in a comment\n'
        'format!("first \\\n          (docs/PLAN.md task 71)");\n'
        'let r = r#"raw "quoted" text"#;\n'
        '#[cfg(test)]\nmod tests { "task 2" }\n'
    )
    assert list(rust_strings(src)) == [
        (2, "first (docs/PLAN.md task 71)"),
        (4, 'raw "quoted" text'),
    ]
    assert list(python_strings('"""doc, task 1"""\nx = f"a {y} task 3"\n')) == [
        (2, "a "),
        (2, " task 3"),
    ]


def test_the_messages_were_found():
    """A check that reads nothing passes: well over a thousand literals,
    the CLI's and the bindings' among them, when this was written."""
    found = list(shipped_strings())
    assert len(found) > 1000, len(found)
    assert {"online-cli", "online-py", "online-core", "online-polars", "polars_online"} <= {
        part for where, _, _ in found for part in Path(where).parts
    }


def test_no_message_cites_the_plan_a_task_a_review_or_a_record():
    found = [
        f"{where}:{line}: {hit}"
        for where, line, text in shipped_strings()
        for hit in citations(text)
    ]
    assert not found, "\n".join(found)


#: Files that hold a guide and then a record: the guide is held, up to the
#: heading the record starts at, and the record stays as written.
GUIDE_PARTS: dict[str, str] = {
    "docs/STATE-WORKFLOW.md": "\n## The research behind it, 2026-09-03\n",
}


@pytest.mark.parametrize("path", sorted(GUIDE_PARTS))
def test_no_guide_part_cites_the_plan_a_task_a_review_or_a_record(path):
    text = (REPO / path).read_text(encoding="utf-8")
    assert GUIDE_PARTS[path] in text, f"{path}: the record's heading moved"
    found = citations(text[: text.index(GUIDE_PARTS[path])])
    assert not found, "\n".join(found)
