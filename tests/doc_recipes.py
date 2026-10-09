"""The recipes of a docs page that shows each program's output under it:
``docs/DATA-ISSUES.md`` (``tests/test_data_issues.py``) and
``docs/DIAGNOSTICS.md`` (``tests/test_diagnostics_page.py``).

Each python block of such a page is followed by a ```text block of what it
prints. A test runs the block and holds what it printed to the text block:
the same lines, the same words, and each number within two units of its last
printed digit (docs/WRITING.md §3, "Every code block runs").
"""

from __future__ import annotations

import contextlib
import io
import re
from decimal import Decimal
from pathlib import Path

#: A number as a recipe prints it, not a digit inside a name such as ``x1``.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w.])")

#: A number this large is a model that diverged: its digits are the platform's
#: rounding, so only its sign and its size are held (the pages say so).
BLOWN_UP = 1e6


def recipes(page: Path) -> list[tuple[int, str, str]]:
    """Each python block of ``page``, with the line its fence opens on and the
    text block that must follow it: what the recipe prints."""
    out: list[tuple[int, str, str]] = []
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


def run(code: str, page: Path) -> str:
    """What a recipe prints, run in a namespace of its own."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(page), "exec"), {"__name__": "__recipe__"})
    return buf.getvalue()
