"""``docs/PERFORMANCE.md`` and the timed recipe that left it, held to the code.

The measured costs' reference holds figures, each with its date and how it
was made, and shell commands that make them again; it shows no output of
its own, so no recipe of it is held here. The recipe that times each part
of a run moved to ``docs/explanation/where-the-time-and-memory-go.md``
(task 338), where ``tests/test_kind_pages.py`` runs it in order with the
page's other blocks and holds what it prints to its words, its numbers
left free (``untimed``). What stays here is the check a run cannot make:
every line that recipe prints is a timed rate, so no result escapes the
comparison by riding on a timed line.
"""

from __future__ import annotations

from pathlib import Path

from doc_recipes import TIMED, shown_outputs

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "PERFORMANCE.md"
EXPLANATION = REPO / "docs" / "explanation" / "where-the-time-and-memory-go.md"


def test_the_reference_shows_no_output_of_its_own():
    """A block followed by what it prints belongs on a page whose blocks a
    test runs; the reference's blocks are commands to measure with."""
    assert shown_outputs(PAGE) == []


def test_the_timed_recipe_prints_only_rates():
    """The recipe that times the read, learning, learning and scoring, and
    scoring from a state prints only timed lines, each a rate: a result
    printed on a timed line would escape the comparison."""
    (shown,) = [
        shown for _, code, shown in shown_outputs(EXPLANATION) if "time to read the file" in shown
    ]
    for text in shown.splitlines():
        assert text.startswith(TIMED) and text.endswith("rows a second"), text
