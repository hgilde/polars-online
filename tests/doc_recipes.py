"""The recipes of a docs page that shows each program's output under it:
``docs/DATA-QUALITY.md`` (``tests/test_data_quality.py``) and
``docs/DIAGNOSTICS.md`` (``tests/test_diagnostics_page.py``), and those of
the README and ``docs/RUNNER.md`` (``tests/test_production_hardening.py``).

Each python block of such a page is followed by a ```text block of what it
prints. A test runs the block and holds what it printed to the text block:
the same lines, the same words, and each number within two units of its last
printed digit (docs/WRITING.md §3, "Every code block runs"). The README and
the runner guide mix recipes with blocks whose output is not shown, so there
only a block followed by a ```text block is held (``shown_outputs``).
"""

from __future__ import annotations

import contextlib
import io
import itertools
import re
from decimal import Decimal
from pathlib import Path

#: A number as a recipe prints it, not a digit inside a name such as ``x1``.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w.])")

#: A number this large is a model that diverged: its digits are the platform's
#: rounding, so only its sign and its size are held (the pages say so).
BLOWN_UP = 1e6


def fenced_blocks(page: Path) -> list[tuple[str, int, str]]:
    """Every fenced block of ``page``: its info string (``python``, ``text``,
    ``sh`` ...), the line its fence opens on, and its body."""
    blocks: list[tuple[str, int, str]] = []
    fence, start, buf = None, 0, []
    for i, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if fence is None and stripped.startswith("```") and stripped != "```":
            fence, start, buf = stripped[3:], i, []
        elif fence is not None and stripped == "```":
            blocks.append((fence, start, "\n".join(buf)))
            fence = None
        elif fence is not None:
            buf.append(line)
    return blocks


def shown_outputs(page: Path, kind: str = "python") -> list[tuple[int, str, str]]:
    """Each ``kind`` block of ``page`` whose output is shown under it: a
    ```text block that opens after its closing fence with only blank lines
    between, as every recipe's is. A ```text block after prose is an
    equation or a message quoted by hand, not a block's output. The README
    and the runner guide mix recipes with blocks whose output is not shown,
    so only the blocks followed by one are held to what they print."""
    lines = page.read_text(encoding="utf-8").splitlines()
    blocks = fenced_blocks(page)
    out = []
    for (fence, line, code), (after, start, shown) in itertools.pairwise(blocks):
        # The closing fence: the first bare ``` after the opening one (1-based).
        closing = next(i for i in range(line + 1, start) if lines[i - 1].strip() == "```")
        between = lines[closing : start - 1]
        if fence == kind and after == "text" and not any(s.strip() for s in between):
            out.append((line, code, shown))
    return out


def recipes(page: Path) -> list[tuple[int, str, str]]:
    """Each python block of ``page``, with the line its fence opens on and the
    text block that must follow it: what the recipe prints."""
    out: list[tuple[int, str, str]] = []
    blocks = fenced_blocks(page)
    for j, (kind, line, code) in enumerate(blocks):
        if kind != "python":
            continue
        after = blocks[j + 1] if j + 1 < len(blocks) else None
        assert after is not None and after[0] == "text", (
            f"{page.name}:{line}: a recipe is followed by a ```text block of what it prints"
        )
        out.append((line, code, after[2]))
    return out


def differences(shown: str, printed: str) -> list[str]:
    """Where ``printed`` is not what the page shows: the same lines, the same
    words, and each number within two units of its last printed digit."""
    want, got = shown.splitlines(), printed.rstrip("\n").splitlines()
    if len(want) != len(got):
        return [f"{len(got)} lines printed, {len(want)} shown:\n{printed}"]
    problems = []
    for a, b in zip(want, got, strict=True):
        if NUMBER.sub("#", a) != NUMBER.sub("#", b):
            problems.append(f"shown  {a!r}\nprinted {b!r}")
            continue
        for x, y in zip(NUMBER.findall(a), NUMBER.findall(b), strict=True):
            e, g = float(x), float(y)
            if abs(e) >= BLOWN_UP:
                ok = abs(g) >= BLOWN_UP and (e > 0) == (g > 0)
            else:
                unit = 10.0 ** Decimal(x.split("e")[0]).as_tuple().exponent
                scale = 10.0 ** int(x.split("e")[1]) if "e" in x else 1.0
                ok = abs(e - g) <= 2 * unit * scale + 1e-12
            if not ok:
                problems.append(f"shown  {a!r}\nprinted {b!r}")
                break
    return problems


def run(code: str, page: Path, namespace: dict[str, object] | None = None) -> str:
    """What a recipe prints, run in a namespace of its own: an empty one, or
    a copy of ``namespace`` (the README's recipes run in what its *Example
    data* block builds)."""
    ns: dict[str, object] = {"__name__": "__recipe__", **(namespace or {})}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(page), "exec"), ns)
    return buf.getvalue()
