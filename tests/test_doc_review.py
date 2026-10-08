"""`scripts/doc_review.py` holds the steps of a documentation pass other than
the structure checks (docs/WRITING.md §6). A count is a list a reviewer reads,
so each is held here to the fault it was written for, on a few lines of
Markdown; `account` and `render`'s link fixing are held to what they promise,
and `run` to telling a block that runs from one that does not."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("doc_review", REPO / "scripts" / "doc_review.py")
assert _spec and _spec.loader
review = importlib.util.module_from_spec(_spec)
sys.modules["doc_review"] = review
_spec.loader.exec_module(review)


def hits(text: str, number: int) -> list[str]:
    """The hits of one numbered count."""
    for title, rows in review.counts(text):
        if title.startswith(f"{number}."):
            return [s for _, s in rows]
    raise AssertionError(f"no count {number}")


def test_measure_splits_a_bold_lead_from_the_sentence_after_it():
    numbers, sentences = review.measure(
        "## Rows\n\n**Sort the rows first.** Then feed the bank. It learns.\n"
    )
    assert [words for _, words, _ in sentences] == [4, 4, 2]
    assert numbers["sentences"] == 3
    assert numbers["## sections"] == 1


def test_a_bold_lead_holding_a_code_span_with_a_star_is_split_too():
    """The README's "**A window holds the rows Polars' `rolling_*_by` would
    give it.**" was counted as one 36-word sentence with the one after it:
    the lead's pattern stopped at the `*` inside the code span (task 160,
    SC7)."""
    _, sentences = review.measure("## Rows\n\n**Use `a_*_b` first.** Then feed the bank.\n")
    assert [words for _, words, _ in sentences] == [3, 4]


def test_the_counts_find_the_faults_they_were_written_for():
    """Each phrase is one the user reported in a README pass
    (docs/README-ITERATIONS.md: W1, W11, W12, W13)."""
    text = (
        "## Targets\n\n"
        "The examples from here on read two frames.\n\n"
        "**A target that needs no window is a column made first.**\n\n"
        "Compute the returns upstream; then fit them.\n"
    )
    assert hits(text, 3) == ["Compute the returns upstream; then fit them."]
    assert hits(text, 6) == ["The examples from here on read two frames."]
    assert len(hits(text, 7)) == 1
    assert len(hits(text, 8)) == 1
    found = " ".join(hits(text, 9))
    assert "[from here on]" in found
    assert "[made first]" in found
    assert "[upstream]" in found
    assert hits(text, 1) == []


def test_count_11_finds_exception_and_raise_in_their_everyday_sense_only():
    """W8: in a Python library's docs, *exception* and *raise* mean an error
    raised, so the everyday senses are listed, and the Python ones are not."""
    text = (
        "## Models\n\n"
        "Most models take all of them, and each exception is named where its "
        "parameter is introduced.\n\n"
        "**Raise `dead_frac` when regimes change faster.** A call that finds it "
        "busy raises `RuntimeError`, and a bad file raises an error.\n\n"
        "| setting | what to do |\n|---|---|\n| too small | raise `eps` |\n"
    )
    found = hits(text, 11)
    assert len(found) == 3
    assert "each exception is named" in found[0]
    assert found[1].startswith("**Raise `dead_frac`")
    assert found[2] == "[a table cell] raise `eps`"


def test_count_10_finds_a_call_walked_through_in_prose_and_not_one_its_code_shows():
    """The shape of the `po.target` paragraph the user asked to be an example
    (docs/README-ITERATIONS.md, C12), in small: a call and its keyword
    values in prose, under a heading whose code shows none of them. The
    call is a window operator's since task 201 removed `po.target`'s
    `relative=`, whose words the paragraph walked through."""
    prose = (
        '`po.rewm_mean("p", window_size="5m")` is the mean of `p` over the next five minutes, '
        'under the default `closed="right"`, and `"left"` or `"both"` move the edges.\n'
    )
    plain = '```python\nspec = po.spec.ewridge("s", targets=["y"], features=["x0"])\n```\n\n'
    rows = hits("## Targets\n\n" + plain + prose, 10)
    assert len(rows) == 1
    assert rows[0].startswith('[score 6: rewm_mean*, closed*, "left", "both"]')
    shows = (
        '```python\none = po.rewm_mean("p", window_size="5m", closed="left")   # or "both"\n```\n\n'
    )
    assert hits("## Targets\n\n" + shows + prose, 10) == []


def test_account_reports_a_name_a_number_and_a_link_that_were_lost():
    old = "Give `half_life=600.0` ([Time and decay](#time-and-decay)) to 1,000 rows."
    lost = {kind: names for kind, _, _, names, _ in review.account(old, "Give a half-life.")}
    assert lost == {
        "names": ["half_life=600.0"],
        "numbers": ["1000", "600.0"],
        "links": ["#time-and-decay"],
    }
    moved = "Give a half-life:\n\n```python\nhalf_life=600.0\n```\n"
    kept = {kind: names for kind, _, _, names, _ in review.account(old, moved)}
    assert kept["names"] == []  # a name moved into a code block is kept


def test_render_points_relative_links_at_github_and_drops_the_id_prefix():
    body = (
        '<h2 id="user-content-time">Time</h2><a href="docs/RUNNER.md">a</a>'
        '<a href="#time">b</a><a href="https://pola.rs">c</a>'
    )
    out = review.fix_links(body)
    assert '<h2 id="time">' in out
    assert 'href="https://github.com/hgilde/polars-online/blob/main/docs/RUNNER.md"' in out
    assert 'href="#time"' in out
    assert 'href="https://pola.rs"' in out


def test_run_tells_a_block_that_runs_from_one_that_does_not(tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text(
        "```python\nrows = df.height\n```\n\n```python\nrows = never_built\n```\n",
        encoding="utf-8",
    )
    (first, ok), (second, error) = review.run(draft)
    assert (first, ok) == (1, None)
    assert second == 5
    assert error is not None and "NameError" in error
