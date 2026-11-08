"""The API reference keeps every address a released README links.

The reference gives each public function and class a page of its own,
grouped by family on its module's page (``docs/reference/families.py``).
Before that, each module page documented its objects in place, so the README
of every release on PyPI links ids such as
``spec.html#polars_online.spec.ewridge``, and keeps linking them after this
repository has moved on. Each module page therefore carries an anchor with
every id it once gave, beside a link to the object's own page. This builds
the reference and holds the README's links, and every placed object's
anchor, to the HTML.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

TIER = "essential"

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "docs" / "reference"
LINK = re.compile(r"https://hgilde\.github\.io/polars-online/([\w-]+)\.html(?:#([\w.\-]+))?")

sys.path.insert(0, str(REFERENCE))
import families  # noqa: E402 -- the reference's own list, beside conf.py


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("html")
    run = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", str(REFERENCE), str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return out


def _ids(page: Path) -> set[str]:
    return set(re.findall(r' id="([^"]+)"', page.read_text(encoding="utf-8")))


def _readme_links() -> list[tuple[str, str]]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return [(page, anchor) for page, anchor in LINK.findall(text)]


def test_the_readme_links_the_reference():
    """The README's links into the reference are found: 36 when this was
    written, 34 once the how-tos took the README's task sections (task 338
    stage 6a), every one to a module page."""
    links = _readme_links()
    assert len(links) >= 34, links
    assert sum(1 for _, anchor in links if anchor) >= 30, links


def test_every_link_the_readme_gives_lands_in_the_built_reference(built):
    missing = []
    for page, anchor in _readme_links():
        html = built / f"{page}.html"
        if not html.exists() or (anchor and anchor not in _ids(html)):
            missing.append(f"{page}.html#{anchor}")
    assert not missing, missing


def test_each_module_page_keeps_an_anchor_for_every_object_it_lists(built):
    """Every placed object, and every method and property of a class, keeps
    its old id on its module page, and that page links the object's own."""
    sys.path.insert(0, str(REFERENCE))
    import conf  # noqa: PLC0415 -- the build's own reading of the objects

    problems = []
    for fam in families.FAMILIES:
        page = built / f"{fam.page}.html"
        text = page.read_text(encoding="utf-8")
        ids = _ids(page)
        for name in fam.entries:
            obj = conf._object(name)
            kind = "class" if isinstance(obj, type) else "function"
            wanted = [name] + ([m for m, _ in conf._members(name)] if kind == "class" else [])
            problems += [f"{fam.page}.html#{i}" for i in wanted if i not in ids]
            target = f'href="api/{families.page_name(name, kind)}.html'
            if target not in text:
                problems.append(f"{fam.page}.html links no page for {name}")
            own = built / "api" / f"{families.page_name(name, kind)}.html"
            if not own.exists() or name not in _ids(own):
                problems.append(f"{name} has no page of its own")
    assert not problems, problems


def test_twins_of_another_case_get_two_files():
    """``Target`` and ``target`` would be one file on macOS and Windows."""
    assert families.page_name("polars_online.Target", "class") == "polars_online.Target-class"
    assert families.page_name("polars_online.target", "function") == "polars_online.target"
    assert families.page_name("polars_online.ModelBank", "class") == "polars_online.ModelBank"
