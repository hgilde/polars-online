"""`llms.txt` is a map for coding agents, and a wrong map is worse than none.

The file (llmstxt.org) sits at the repo root and is copied into the docs
build, so it is served from GitHub and from the API reference alike. Nothing
in it is generated -- it is written prose -- so these tests hold the parts
that can silently go stale: the model names, the links, and the claim that
one name is spelled two ways.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars_online as po
from polars_online import _polars_online as _native

ROOT = Path(__file__).resolve().parent.parent
LLMS = ROOT / "llms.txt"
CONF = ROOT / "docs" / "reference" / "conf.py"
TEXT = LLMS.read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

# The builders a caller writes, and the `type` strings a spec dict or TOML uses.
BUILDERS = {n for n in dir(po.spec) if not n.startswith("_")} - {
    "output_fields",
    "coef_fields",
    "coef_index",
    "output_index",
}
KINDS = set(_native.model_kinds())


def test_it_names_no_model_that_does_not_exist():
    """A hallucinated builder in the file that exists to prevent hallucination
    would be the worst of both."""
    named = set(re.findall(r"po\.spec\.([a-z_]+)", TEXT))
    assert named <= BUILDERS, f"llms.txt names {sorted(named - BUILDERS)}"
    quoted_types = set(re.findall(r'type = "([a-z_]+)"', TEXT))
    assert quoted_types <= KINDS, f"llms.txt quotes {sorted(quoted_types - KINDS)}"


def test_it_says_how_many_models_there_are():
    """A count written into the prose ages the moment a model is added."""
    counts = set(re.findall(r"all (\d+),", TEXT)) | set(re.findall(r"\ball (\d+)\b", TEXT))
    assert counts, "llms.txt no longer states how many models there are"
    assert counts == {str(len(KINDS))}, f"llms.txt says {sorted(counts)}, registry has {len(KINDS)}"


def test_it_warns_about_the_one_name_spelled_two_ways():
    """`po.spec.ewridge` builds what a spec dict calls `ew_ridge`. It is the
    only such pair, and an agent that gets it wrong writes a TOML that is
    refused."""
    differ = {b for b in BUILDERS if b not in KINDS}
    assert differ == {"ewridge"}, f"a second name now differs: {sorted(differ)}"
    assert "po.spec.ewridge" in TEXT
    assert 'type = "ew_ridge"' in TEXT


def test_every_repository_link_resolves():
    """Links into the repo are absolute (the file is served from two places),
    so nothing here can follow them -- but the path they end in is local."""
    for path in re.findall(r"https://github\.com/hgilde/polars-online/blob/main/(\S+?)\)", TEXT):
        assert (ROOT / path).exists(), f"llms.txt links to a missing file: {path}"


def test_every_readme_anchor_exists():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = {
        re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", h.strip().lower()))
        for h in re.findall(r"^#{1,6}\s+(.*?)\s*$", readme, flags=re.MULTILINE)
    }
    for anchor in re.findall(r"https://github\.com/hgilde/polars-online#([\w-]+)\)", TEXT):
        assert anchor == "readme" or anchor in headings, f"llms.txt links to #{anchor}"


def test_every_reference_page_is_built():
    """A link to `<name>.html` needs `docs/reference/<name>.rst` to exist."""
    for page in re.findall(r"https://hgilde\.github\.io/polars-online/([\w-]+)\.html", TEXT):
        assert (ROOT / "docs" / "reference" / f"{page}.rst").exists(), f"no rst for {page}.html"


def test_the_docs_build_carries_it():
    """`html_extra_path` is what puts the file at the site root; without it the
    reference and the repo would serve different maps."""
    assert 'html_extra_path = ["../../llms.txt"]' in CONF.read_text(encoding="utf-8")


def test_it_stays_an_index():
    """An index an agent reads in one gulp, not a second copy of the README."""
    assert len(TEXT.encode("utf-8")) < 12_000, "llms.txt is growing into a document"


def test_it_names_every_field_whose_schedule_follows_the_chunking():
    """Review 2026-10-05 (CI6): the chunk-invariance rule named ``coef`` as
    the one exception, and ``support_coef`` sits on ``coef``'s rows, so it
    follows the chunking too. The bug report template and the Pathway
    example say the same, and each names both. Review 2026-10-06 (SF4, DA9):
    so does the command line's guide, whose *Chunk size* named ``coef``
    alone, and it says that under a cadence nothing moves."""
    rule = re.search(r"\*\*Chunk invariance is a guarantee\.\*\*(.*?)\n- \*\*", TEXT, re.S)
    assert rule, "llms.txt no longer states the chunk-invariance rule"
    runner = (ROOT / "docs" / "RUNNER.md").read_text(encoding="utf-8")
    chunk_size = runner.split("\n### Chunk size\n", 1)[1].split("\n## ", 1)[0]
    for text in (
        rule.group(1),
        (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8"),
        (ROOT / "examples" / "pathway_integration.py").read_text(encoding="utf-8"),
        # And every sentence of the README's that says which rows carry them.
        *re.findall(r"[^.]*which rows carry[^.]*\.", README),
        chunk_size,
    ):
        assert "`coef`" in text and "`support_coef`" in text, text[:300]
    assert "`coef_every`" in chunk_size and "`max_rows_between_coefs`" in chunk_size


def test_it_says_a_clock_needs_a_gap_cap():
    """Review 2026-10-05 (CI7): the file showed a clock on a feature and never
    said that a ``clock`` needs a ``gap_cap``, which the bank refuses without:
    a spec built from the map alone was refused."""
    rules = TEXT.split("**Rules that are easy to get wrong.**", 1)[1].split("\n\n", 2)[1]
    assert "gap_cap is required when clock is given" in rules
    try:
        po.spec.ewridge("m", targets=["y"], features=["x"], clock="t", half_life=10.0)
    except ValueError as e:
        assert "gap_cap is required when clock is given" in str(e)
    else:
        raise AssertionError("a clock without a gap_cap is taken; the rule is stale")
