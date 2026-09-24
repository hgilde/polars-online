"""`scripts/pypi_readme.py`: the README's links to other files, made absolute
for PyPI at the release's tag, and the release workflow that runs it."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pypi_readme", REPO / "scripts/pypi_readme.py")
assert _spec and _spec.loader
pypi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pypi)

BASE = "https://github.com/o/r"


def test_a_relative_link_is_absolute_and_nothing_else_moves(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/A.md").write_text("a")
    text = (
        "See [A](docs/A.md) and [B](docs/A.md#part), the [folder](docs),\n"
        "[away](https://x.org/y), [here](#local) and [mail](mailto:a@b.c).\n"
        "In code: `[A](docs/A.md)`, but [`code`](docs/A.md) is a link.\n"
        "```text\n[A](docs/A.md)\n```\n"
    )
    got, changed, missing = pypi.rewrite(text, BASE, "v1.2.3", tmp_path)
    assert missing == []
    assert f"[A]({BASE}/blob/v1.2.3/docs/A.md)" in got
    assert f"[B]({BASE}/blob/v1.2.3/docs/A.md#part)" in got
    assert f"[folder]({BASE}/tree/v1.2.3/docs)" in got
    assert "[away](https://x.org/y)" in got and "[here](#local)" in got
    assert "[mail](mailto:a@b.c)" in got
    assert "`[A](docs/A.md)`" in got, "an inline code span is left alone"
    assert f"[`code`]({BASE}/blob/v1.2.3/docs/A.md)" in got, "a link whose text is code is not"
    assert "```text\n[A](docs/A.md)\n```" in got, "a fenced block is left alone"
    assert len(changed) == 4
    again, changed_again, _ = pypi.rewrite(got, BASE, "v1.2.3", tmp_path)
    assert again == got and changed_again == [], "a second pass changes nothing"


def test_a_link_to_a_missing_file_is_an_error_not_a_404(tmp_path):
    _, _, missing = pypi.rewrite("[gone](docs/GONE.md) [up](../x.md)", BASE, "v1", tmp_path)
    assert missing == ["docs/GONE.md", "../x.md"]


def test_the_readme_leaves_no_relative_link_and_links_no_missing_file():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    base = pypi.repository()
    got, changed, missing = pypi.rewrite(text, base, "vTEST", REPO)
    assert missing == [], missing
    assert changed, "the README links to other files"
    outside_code = "".join(
        piece for line in got.split("```")[::2] for piece in line.split("`")[::2]
    )
    left = [
        t
        for t in re.findall(r"\]\(([^)\s]+)\)", outside_code)
        if not t.startswith(("http://", "https://", "mailto:", "#"))
    ]
    assert left == [], left
    assert base == "https://github.com/hgilde/polars-online"


def test_every_release_build_rewrites_the_readme_before_maturin_reads_it():
    """Both jobs that write package metadata -- the sdist and every wheel --
    run the rewrite first, pinned to the ref the release runs at."""
    release = yaml.safe_load((REPO / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    for job in ("sdist", "build"):
        steps = release["jobs"][job]["steps"]
        labels = [f"{s.get('uses', '')} {s.get('run', '')}" for s in steps]
        rewrite = next(i for i, x in enumerate(labels) if "pypi_readme.py" in x)
        maturin = next(i for i, x in enumerate(labels) if "maturin-action" in x)
        assert rewrite < maturin, job
        assert '--ref "${{ github.ref_name }}"' in labels[rewrite], job
