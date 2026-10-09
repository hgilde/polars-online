"""``docs/DATA-ISSUES.md`` (docs/PLAN.md task 224): the page each finding of
``ModelBank.check`` links to, held to the code.

Every recipe on the page runs here, and prints what the page shows under it,
so a recipe and the prose that reads its output cannot drift from the library
(docs/WRITING.md §3, "Every code block runs"). Every code ``check`` can give
has a section anchored by its name, every finding's message ends with the
link to it, and the docstring lists the same codes.
"""

from __future__ import annotations

import contextlib
import io
import re
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

import check_streams as cs
import polars_online as po
from polars_online import _check

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "DATA-ISSUES.md"


def recipes() -> list[tuple[int, str, str]]:
    """Each python block of the page, with the line its fence opens on and the
    text block that must follow it: what the recipe prints."""
    out: list[tuple[int, str, str]] = []
    blocks: list[tuple[str, int, str]] = []
    fence, start, buf = None, 0, []
    for i, line in enumerate(PAGE.read_text(encoding="utf-8").splitlines(), 1):
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
            f"DATA-ISSUES.md:{line}: a recipe is followed by a ```text block of what it prints"
        )
        out.append((line, code, after[2]))
    return out


RECIPES = recipes()

#: A number as a recipe prints it, not a digit inside a name such as ``x1``.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w.])")

#: A number this large is a model that diverged: its digits are the platform's
#: rounding, so only its sign and its size are held (the page says so).
BLOWN_UP = 1e6


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


def run(code: str) -> str:
    """What a recipe prints, run in a namespace of its own."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, str(PAGE), "exec"), {"__name__": "__recipe__"})
    return buf.getvalue()


def test_the_page_has_its_recipes():
    """One recipe for the opening example and one for each problem."""
    assert len(RECIPES) >= 12, len(RECIPES)


@pytest.mark.parametrize(
    ("line", "code", "shown"), RECIPES, ids=[f"DATA-ISSUES.md:L{line}" for line, _, _ in RECIPES]
)
def test_a_recipe_prints_what_the_page_shows(line, code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a recipe writes its parquet file where it runs
    problems = differences(shown, run(code))
    assert not problems, f"DATA-ISSUES.md:{line}:\n" + "\n".join(problems)


def test_the_comparison_holds_words_digits_and_blow_ups():
    assert not differences("R² 0.578", "R² 0.579")
    assert differences("R² 0.578", "R² 0.581")
    assert differences("R² 0.578", "R2 0.578")
    assert not differences("x1 0.0464", "x1 0.0465")
    assert not differences("slope 5.4e+13", "slope 9.1e+12")
    assert differences("slope 5.4e+13", "slope -5.4e+13")
    assert differences("slope 5.4e+13", "slope 0.481")
    assert differences("rows 217", "rows 220")
    assert differences("a\nb", "a")


def _anchors() -> set[str]:
    text = PAGE.read_text(encoding="utf-8")
    return set(re.findall(r'<a id="([^"]+)"></a>', text)) | {
        re.sub(r"[^\w\- ]", "", h).strip().lower().replace(" ", "-")
        for h in re.findall(r"^#+ (.+)$", text, re.M)
    }


def test_every_code_has_a_section_on_the_page():
    """A finding's message links to ``DATA-ISSUES.md#<code>``, so each code
    is an anchor there: an explicit one, or a heading whose text is the code."""
    missing = sorted(_check.CODES - _anchors())
    assert not missing, f"codes with no anchor on docs/DATA-ISSUES.md: {missing}"


def test_the_docstring_lists_the_same_codes():
    doc = po.ModelBank.check.__doc__ or ""
    listed = set(re.findall(r"\* - ``(\w+)``\n\s+- (?:error|warning|info)", doc))
    assert listed == _check.CODES, listed ^ _check.CODES


def _every_planted_finding() -> pl.DataFrame:
    frames = []
    for make, _, _, _ in cs.PLANTED.values():
        frames.append(cs.findings(*make(0)))
    for make, _, _ in cs.AUDIT_PLANTED.values():
        frames.append(cs.audit_findings(make(0)))
    return pl.concat(frames)


def test_every_message_ends_with_the_link_to_its_code():
    found = _every_planted_finding()
    assert len(set(found["code"])) >= 20, set(found["code"])
    assert set(found["code"]) <= _check.CODES, set(found["code"]) - _check.CODES
    for code, message in found.select("code", "message").iter_rows():
        assert message.endswith(f". See {_check.DATA_ISSUES}#{code}"), message


def test_the_link_is_the_page_on_main():
    assert _check.DATA_ISSUES.endswith("/blob/main/docs/DATA-ISSUES.md")
    assert PAGE.is_file()


def test_the_readme_links_the_page_beside_check_and_audit():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for anchor in ("**`check()` reads these tables", "#### `audit` — what a stream's columns hold"):
        start = readme.index(anchor)
        section = readme[start : readme.index("\n#", start + len(anchor))]
        assert "docs/DATA-ISSUES.md" in section, anchor
