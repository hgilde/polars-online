"""The checks that hold the writing rules (docs/WRITING.md §13), in report
mode (``scripts/writing_checks.py``, ``writing_ratchet.toml``).

Each check has an activation test: it finds a violation planted in a small
example, and passes a clean one, so a green ratchet is not a check that
reads nothing. Then each check's held findings over the repository are held
to the ratchet: a count over its ratchet, under it, a file it has not seen,
or a finding in a finished file fails, each naming what to do.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "writing_checks", REPO / "scripts" / "writing_checks.py"
)
assert _spec and _spec.loader
wc = importlib.util.module_from_spec(_spec)
# Registered first: a dataclass looks its module up while it is built.
sys.modules["writing_checks"] = wc
_spec.loader.exec_module(wc)


def rules(findings, held=True):
    return {wc.rule_key(f.rule) for f in findings if f.held or not held}


# ------------------------------------------------------------- the messages


def test_a_planted_python_message_breaks_each_rule():
    src = (
        "def f(x, lam):\n"
        "    raise ValueError(\n"
        '        f"ridge was {x:.6g}, settled near 1/(1 - lam^d) and 72.6359. '
        'It is wrong. It is very wrong. It is still wrong."\n'
        "    )\n"
    )
    (msg,), unread = wc.python_messages(src, "p.py")
    assert not unread
    assert rules(wc.message_findings(msg)) == {
        "where",
        "remedy",
        "sentences",
        "internal-word",
        "digits",
        "formula",
    }


def test_a_message_in_the_shape_passes():
    src = (
        "def f(name, r):\n"
        "    msg = (\n"
        '        f"spec {name!r}: ridge must be finite and at least 0, but it is {r:.3g}. "\n'
        '        "Set ridge to 0 for no shrinkage, or to a positive value."\n'
        "    )\n"
        "    raise ValueError(msg)\n"
        "def g(who):\n"
        '    warnings.warn(f"{who}: predictions stay withheld. If the stream is short, '
        'lower min_weight.")\n'
    )
    msgs, unread = wc.python_messages(src, "p.py")
    assert len(msgs) == 2 and not unread
    assert [wc.message_findings(m) for m in msgs] == [[], []]


def test_a_message_a_static_reading_cannot_see_is_counted_not_judged():
    src = "def f(e):\n    raise ValueError(e)\ndef g():\n    raise make_error()\n"
    msgs, unread = wc.python_messages(src, "p.py")
    assert not msgs and len(unread) == 2


def test_rust_messages_are_read_where_an_error_opens():
    src = (
        "fn a(r: f64) -> Result<(), String> {\n"
        '    if r < 0.0 { return Err(format!("ridge was {r:.5}, settled")); }\n'
        '    let x = v.expect("never shown: a defect");\n'
        '    panic!("never shown either");\n'
        '    notices.push(format!("spec {:?}: the ridge sets a coefficient. Lower ridge.", n));\n'
        '    Err(format!("spec {:?}: {e}", name))\n'
        "}\n"
        '#[cfg(test)]\nmod tests { fn t() { Err("a test\'s string".into()) } }\n'
    )
    msgs = wc.rust_messages(src, "crates/online-polars/src/x.rs")
    assert [m.kind for m in msgs] == ["error", "warning", "error"]
    assert rules(wc.message_findings(msgs[0])) == {"where", "remedy", "internal-word", "digits"}
    assert wc.message_findings(msgs[1]) == []
    assert wc.is_wrapper(msgs[2]) and wc.message_findings(msgs[2]) == []
    # online-core's strings are bodies its callers name: no where of their own.
    assert "where" not in rules(wc.message_findings(msgs[0], core=True))


@pytest.mark.parametrize(
    ("sentence", "imperative"),
    [
        ("Set ridge to 0.", True),
        ("If the clock steps back, set restart_after_step_back.", True),
        ("Either sort each group by the clock, or skip the learned rows.", True),
        ("Otherwise, give a half-life.", True),
        ("state has the wrong shape", False),
        ("skip_learned = true needs load_state", False),
        ("Predict = true does nothing here.", False),
        ("Merge() needs at least one Gram.", False),
        ("Set has no order.", False),
        ("The fit is undetermined.", False),
    ],
)
def test_the_remedy_is_told_by_its_first_word(sentence, imperative):
    assert wc.is_imperative(sentence) is imperative


@pytest.mark.parametrize(
    ("spec", "rust", "too_precise"),
    [(".3g", False, False), (".4g", False, True), ("g", False, True), (".2e", False, False),
     (".3e", False, True), (".3f", False, False), (".4f", False, True), ("", False, False),
     (".3", True, False), (".4", True, True), ("e", True, True), (".2e", True, False),
     (".3e", True, True), ("?", True, False)],
)  # fmt: skip
def test_a_visible_format_past_three_figures_is_found(spec, rust, too_precise):
    assert wc._too_precise(spec, rust) is too_precise


# ----------------------------------------------------------- the docstrings

CLEAN_MODEL = """Exponentially weighted ridge regression on running sums.

Use it first, and use ``kalman`` where the coefficients drift.

.. rubric:: Parameters

half_life : float

.. rubric:: Returns

The output fields.

.. rubric:: Raises

ValueError for a ridge below 0.

.. rubric:: Update equation

.. code-block:: text

    W' = lam * W + w

.. rubric:: Examples

.. code-block:: python

    po.spec.ewridge("m", targets=["y"], features=["x"])

.. rubric:: See also

``kalman``.

.. rubric:: Notes

The solve is a Cholesky factorization.

.. rubric:: The condition estimate

A section of the Notes.
"""


def model(doc):
    return wc.Documented("po.spec.m", "model", doc, "p.py", 1, params=True, returns=True)


def test_a_docstring_in_the_order_passes():
    assert wc.docstring_findings(model(CLEAN_MODEL)) == []


def test_a_planted_docstring_breaks_each_rule():
    doc = (
        "Ridge on running sums -- the workhorse. It is fast.\n\n"
        "One. Two. Three. Four.\n\n"
        ".. rubric:: The fit\n\nx\n\n"
        ".. rubric:: Examples\n\nx\n\n"
        ".. rubric:: Parameters\n\nx\n"
    )
    found = {f.rule for f in wc.docstring_findings(model(doc))}
    assert {wc.rule_key(r) for r in found} == {
        "summary",
        "unknown-section",
        "order",
        "missing",
        "when",
    }
    assert {r for r in found if r.startswith("missing")} == {
        "missing: Returns",
        "missing: Raises",
        "missing: Update equation",
    }
    assert wc.docstring_findings(model(None)) != []


# ------------------------------------------------------------- the vocabulary


def test_planted_vocabulary_is_found():
    text = (
        "The halflife is centred, and a CLI run reads `n_eff`.\n"
        "Each source is read, which a reader decides.\n"
    )
    found = wc.vocabulary_findings(wc._markdown_prose(text), "x.md")
    assert {f.rule for f in found if f.held} == {
        "retired: halflife",
        "british: centred -> centered",
        "retired: CLI",
        "retired: n_eff",
    }
    assert {f.rule for f in found if not f.held} == {"retired: source"}


def test_clean_vocabulary_passes():
    text = (
        "The half-life is centered, and `restart_after_step_back` is a name.\n"
        'The user said "the centre holds", a quotation.\n'
        "> A block quote keeps its centre.\n"
        "```python\nx = centre_of(y)  # the center\n```\n"
        "[a link](GLOSSARY.md#centre)\n"
    )
    found = wc.vocabulary_findings(wc._markdown_prose(text), "x.md")
    assert [f for f in found if f.held] == []
    in_comment = wc.vocabulary_findings(
        wc._markdown_prose("```python\nx = 1  # centred\n```\n"), "x.md"
    )
    assert [f.rule for f in in_comment] == ["british: centred -> centered"]


def test_the_spelling_table_is_read():
    british = wc.glossary_british()
    assert len(british) >= 25, british
    assert british["centre"] == "center" and british["towards"] == "toward"


def test_every_retired_word_is_placed():
    """Every italic phrase of a *replaces* cell in docs/GLOSSARY.md is a
    variant the check scans for, or is listed with the reason no scan stands
    for it. A phrase the glossary adds fails here until it is placed."""
    retired = wc.glossary_retired()
    assert len(retired) > 60, retired
    placed = {v.phrase for v in wc.VARIANTS} | set(wc.NOT_SCANNED)
    assert [p for p in retired if p not in placed] == []
    assert [v.phrase for v in wc.VARIANTS if v.phrase not in retired] == []


def test_every_kept_out_word_is_placed():
    kept_out = wc.glossary_kept_out()
    assert len(kept_out) > 15, kept_out
    placed = set(wc.KEPT_OUT) | set(wc.KEPT_OUT_NOT_SCANNED)
    assert [w for w in kept_out if w not in placed] == []


# -------------------------------------------------------------- the sentences


def test_planted_sentences_are_found():
    long = " ".join(["word"] * 46) + "."
    text = (
        "# Title\n\n## A heading\n\n### Straight to another\n\n"
        f"Opening. {long}\n\nThis section is about the document. " + "It is short. " * 20 + "\n"
    )
    assert rules(wc.sentence_findings(text, "x.md")) == {"45+", "no-opener", "about-the-document"}
    wordy = "\n\n".join(" ".join(["word"] * 30) + "." for _ in range(3))
    assert "mean" in rules(wc.sentence_findings(f"# T\n\n{wordy}\n", "x.md"))


def test_clean_sentences_pass():
    text = "# Title\n\nA bank fits every spec. It learns each row after predicting it.\n"
    assert rules(wc.sentence_findings(text, "x.md")) == set()


# ------------------------------------------------------------------ the kinds


def test_planted_kinds_are_found():
    assert rules(wc.kind_findings("# A page\n", "x.md")) == {"undeclared"}
    assert rules(wc.kind_findings("<!-- kind: guide -->\n# A page\n", "x.md")) == {"unknown-kind"}
    howto = "<!-- kind: how-to -->\n# Resume a bank\n\nLoad it and go on.\n"
    assert rules(wc.kind_findings(howto, "x.md")) == {"no-steps", "no-prerequisites"}
    steps_first = howto + "\n1. Load.\n2. Fit.\n\n- a saved state\n"
    assert rules(wc.kind_findings(steps_first, "x.md")) == {"no-prerequisites"}


def test_a_declared_how_to_with_its_steps_passes():
    text = (
        "<!-- kind: how-to -->\n# Resume a bank\n\nLoad a saved bank and go on learning.\n\n"
        "You need:\n\n- a saved state\n\n1. Load the state.\n2. Fit the next rows.\n"
    )
    assert wc.kind_findings(text, "x.md") == []
    for kind in wc.KINDS:
        if kind != "how-to":
            assert wc.kind_findings(f"<!-- kind: {kind} -->\n# A page\n", "x.md") == []


# ----------------------------------------------------------------- the ratchet


def finding(path, rule="where", check="messages", text="a message"):
    return wc.Finding(check, path, 1, rule, text)


def test_the_ratchet_fails_over_under_unseen_and_finished():
    ratchet = wc.Ratchet(
        counts={"messages": {"a.py": {"where": 1}, "b.py": {"where": 2}, "gone.py": {}}},
        finished={"messages": ["done.py"]},
        accepted=[
            {"check": "messages", "path": "done.py", "rule": "where", "text": "wrapped",
             "reason": "its caller names the spec"}
        ],
    )  # fmt: skip
    found = [
        finding("a.py"), finding("a.py"),  # over
        finding("b.py"),  # under
        finding("new.py"),  # unseen
        finding("done.py"),  # finished
        finding("done.py", text="a body wrapped by its caller"),  # accepted
    ]  # fmt: skip
    paths = ["a.py", "b.py", "new.py", "done.py"]
    problems = wc.compare("messages", found, paths, ratchet)
    assert len(problems) == 5, problems
    assert any("a.py: where: 2 findings, over its ratchet of 1" in p for p in problems)
    assert any(
        "b.py: where: 1 findings, under its ratchet of 2: tighten it to 1" in p for p in problems
    )
    assert any("new.py is not in" in p for p in problems)
    assert any("gone.py is no longer read" in p for p in problems)
    assert sum("is finished for messages" in p for p in problems) == 1
    counts = wc.applied("messages", found, paths, ratchet)
    assert counts == {"a.py": {"where": 1}, "b.py": {"where": 1}, "new.py": {"where": 1}}
    ratchet.counts["messages"] = counts
    assert wc.compare("messages", found[2:4], ["b.py", "new.py"], ratchet) == [
        "messages: a.py is no longer read: `apply` drops it"
    ]


def test_a_held_count_passes_and_a_finding_to_read_is_not_counted():
    ratchet = wc.Ratchet(counts={"messages": {"a.py": {"where": 1}}})
    found = [finding("a.py"), wc.Finding("messages", "a.py", 2, "where", "x", held=False)]
    assert wc.compare("messages", found, ["a.py"], ratchet) == []


def test_the_ratchet_file_is_what_apply_writes():
    """The committed file round-trips through its loader and writer, so an
    `apply` changes only the counts it moves."""
    path = REPO / wc.RATCHET
    assert wc.Ratchet.load(path).dump() == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("check", wc.CHECKS)
def test_writing_ratchet_toml_holds(check):
    """The check's held findings over the repository against
    writing_ratchet.toml. `uv run python scripts/writing_checks.py report
    --check NAME --list` prints each finding; `apply` records a gain."""
    ratchet = wc.Ratchet.load(REPO / wc.RATCHET)
    problems = wc.compare(check, wc.run(check), wc.files_read(check), ratchet)
    assert not problems, "\n".join(problems)


def test_each_check_reads_what_it_governs():
    """A check that reads nothing passes: each reads at least what it read
    when it was written, with its findings."""
    msgs, unread = wc.all_messages()
    assert len(msgs) > 1000 and len(unread) < 60, (len(msgs), len(unread))
    crates = {m.path.split("/")[1] for m in msgs if m.path.startswith("crates/")}
    assert crates == set(wc.RUST_CRATES), crates
    assert {m.kind for m in msgs} == {"error", "warning", "printed"}
    documented = wc.documented()
    assert len(documented) > 120, len(documented)
    assert sum(d.kind == "model" for d in documented) >= 20
    guides = wc.guides()
    assert {"README.md", "docs/MODELS.md", "docs/GLOSSARY.md"} <= set(guides), guides
    for check in wc.CHECKS:
        assert wc.run(check), check
