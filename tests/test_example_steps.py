"""A long example names its steps (docs/WRITING.md §3).

The user, 2026-10-11: "Some of the example code is missing the few one-line
comments to explain large code blocks in sections." WRITING §3 has the rule:
a long example names its steps in brief comments, and leaves the detail to
the prose, so each step opens with a one-line comment that says what it
does. The documentation rewrites had lost them: of 144 python blocks of ten
code lines or more in the README, CONTRIBUTING and the guides, 137 had a
stretch of ten lines with no full-line comment, the longest 62.

The check counts **statements**, not lines (``tokenize``'s logical lines). A
multi-line call or literal is one statement, so a spec written one parameter
to a line, each with its comment, is one step, and the check never asks for a
comment inside it. A statement in a loop body or a function counts as one,
since a loop of ten statements is as long to read as ten statements outside
one. A comment at the end of a line says what that line gives, and is no step
comment: only a comment on a line of its own ends a run.

``N = 10`` and ``M = 8``, read off the blocks as they stood on 2026-10-11:
a block of fewer than ten statements reads whole, as most of the guides'
reference blocks do (a spec, a fit, a print or two), and WRITING's *few*
comments came to one every three to eight statements in the blocks that
already had them, so a run of nine statements with no step named is the
first that reads as a wall. A block of 10 statements or more may have no run
of more than 8 statements without a full-line comment.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

from test_no_plan_references import FILES, GUIDE_PARTS

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

#: A block of at least this many statements is held to the rule.
N = 10
#: The longest run of statements a held block may have with no full-line comment.
M = 8

#: The Markdown files held: the README, CONTRIBUTING and every guide, as
#: ``test_no_plan_references`` lists them, a guide that a record follows in
#: its file read up to the record's heading.
HELD = sorted(f for f in FILES if f.endswith(".md")) + sorted(GUIDE_PARTS)

_FENCE = re.compile(r"^\s*```python\s*$")


def python_blocks(text: str) -> list[tuple[int, str]]:
    """Each ```python block of ``text``, as (the line of its fence, its code)."""
    blocks, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        if _FENCE.match(lines[i]):
            j = i + 1
            while not lines[j].strip().startswith("```"):
                j += 1
            blocks.append((i + 1, "\n".join(lines[i + 1 : j]) + "\n"))
            i = j
        i += 1
    return blocks


def steps(code: str) -> tuple[int, int]:
    """(the block's statements, its longest run of statements with no
    full-line comment between them). A statement is a logical line, so a
    call or literal over several lines is one."""
    statements = run = longest = 0
    in_statement = False
    code_lines: set[int] = set()
    skip = (tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type in skip:
            continue
        if tok.type == tokenize.COMMENT:
            if tok.start[0] not in code_lines:  # a comment on a line of its own
                run = 0
            continue
        if tok.type == tokenize.NEWLINE:
            in_statement = False
            continue
        code_lines.add(tok.start[0])
        if not in_statement:
            in_statement = True
            statements += 1
            run += 1
            longest = max(longest, run)
    return statements, longest


def _text(path: str) -> str:
    text = (REPO / path).read_text(encoding="utf-8")
    if path in GUIDE_PARTS:
        text = text[: text.index(GUIDE_PARTS[path])]
    return text


# -------------------------------------------------------------- the counting


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("a = 1\nb = 2\n", (2, 2)),
        # A call over several lines, a comment on each argument, is one statement.
        ("spec = f(\n    x=1,   # one\n    y=2,   # two\n)\nfit(spec)\n", (2, 2)),
        # A comment on a line of its own ends a run; one at a line's end does not.
        ("a = 1  # gives a\nb = 2\n# Then c.\nc = 3\n", (3, 2)),
        # A comment line inside a literal ends a run too: it sits on its own line.
        ("d = {\n    # the keys\n    'k': 1,\n}\ne = 2\n", (2, 1)),
        # A loop's body counts statement by statement.
        ("for i in r:\n    a = i\n    b = i\n", (3, 3)),
        # A triple-quoted string is one statement, its lines none.
        ('"""one\n# not a comment\ntwo"""\nx = 1\n', (2, 2)),
    ],
)
def test_steps_counts_statements_and_the_comments_between_them(code, expected):
    assert steps(code) == expected


# ----------------------------------------------------------------- the rule


def test_the_files_were_read():
    """A check that reads nothing passes: 107 blocks were held when this was
    written, in ten files."""
    held = {
        path: n
        for path in HELD
        if (n := sum(steps(code)[0] >= N for _, code in python_blocks(_text(path))))
    }
    assert sum(held.values()) > 80, held
    assert {"README.md", "docs/MODELS.md", "docs/GRAM.md"} <= set(held), held


@pytest.mark.parametrize("path", HELD)
def test_a_long_example_names_its_steps(path):
    walls = []
    for line, code in python_blocks(_text(path)):
        statements, longest = steps(code)
        if statements >= N and longest > M:
            walls.append(f"{path}:{line}: {longest} statements with no step comment")
    assert not walls, (
        "a long example names its steps (docs/WRITING.md §3): open each step with "
        "a one-line comment that says what it does\n" + "\n".join(walls)
    )
