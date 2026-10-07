"""Links into the API reference resolve to something that exists.

The README and `llms.txt` point at https://hgilde.github.io/polars-online/,
which is Sphinx over the docstrings. A link there can rot two ways: the page
can stop being built, or the anchor can name an object that has been renamed
or removed. Both are silent -- the reader gets a page that scrolls to
nothing -- so every link is resolved here against the Python objects the
anchors are made from, which needs no build.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from polars_online import _polars_online as _native

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "docs" / "reference"
SOURCES = [ROOT / "README.md", ROOT / "llms.txt"]
LINK = re.compile(r"https://hgilde\.github\.io/polars-online/([\w-]+)\.html(?:#([\w.\-]+))?")


def _links():
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for page, anchor in LINK.findall(text):
            yield path.name, page, anchor


def _resolve(dotted: str):
    """`polars_online.ModelBank.load` -> the object, or None if the path breaks."""
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for part in parts[1:]:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def test_every_linked_page_is_built():
    for where, page, _ in _links():
        assert (REFERENCE / f"{page}.rst").exists(), (
            f"{where} links to {page}.html, which has no rst"
        )


def test_every_anchor_names_something_that_exists():
    for where, page, anchor in _links():
        if not anchor:
            continue
        if anchor.startswith("module-"):
            importlib.import_module(anchor[len("module-") :])
            continue
        assert _resolve(anchor) is not None, (
            f"{where} links to {page}.html#{anchor}, which resolves to nothing"
        )


def test_the_model_table_links_every_builder_to_its_page():
    """A new model is added to the table with a link, or this fails. The table
    is the index a reader picks a model from; a name with no link there is a
    model whose keywords cannot be found."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("| model | learns by | what it is |", 1)[1].split("\n\n", 1)[0]
    linked = set(
        re.findall(
            r"\[`([a-z_]+)`\]\(https://hgilde\.github\.io/[^)]*#polars_online\.spec\.([a-z_]+)\)",
            table,
        )
    )
    assert {name for name, _ in linked} == {target for _, target in linked}, (
        "a table link points at another model"
    )
    # Every builder is its spec `type` (docs/PLAN.md task 196, N1).
    assert {name for name, _ in linked} == set(_native.model_kinds())


def test_every_model_section_links_its_builder_and_its_source():
    """Each `#### \\`model\\`` section, under its family's `###` heading, opens
    with the API line task 58 added."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    models = readme.split("\n## Models\n", 1)[1].split("\n## ", 1)[0]
    sections = re.split(r"\n#### ", models)[1:]
    for section in sections:
        head, body = section.split("\n", 1)
        api = re.search(
            r"\*API:\* \[`po\.spec\.([a-z_]+)`\]\(https://hgilde\.github\.io/[^)]+\)", body
        )
        assert api, f"the section '#### {head}' has no *API:* link"
        assert re.search(r"\*Rust:\* \[`[\w/.]+`\]\(crates/online-core/src/[\w/]+\.rs\)", body), (
            f"the section '#### {head}' has no *Rust:* source link"
        )
        assert (
            ROOT / re.search(r"\((crates/online-core/src/[\w/]+\.rs)\)", body).group(1)
        ).exists()


def _documented() -> set[str]:
    """Every object a reference page documents, by its dotted path: each
    ``auto*`` directive's target, and every name in the ``__all__`` of a
    module documented with ``:members:``."""
    found: set[str] = set()
    directive = re.compile(r"^\.\. auto(\w+)::\s+([\w.]+)\n((?:[ \t]+:[^\n]*\n)*)", re.M)
    for page in REFERENCE.glob("*.rst"):
        text = page.read_text(encoding="utf-8") + "\n"
        for kind, target, options in directive.findall(text):
            if kind != "module":
                found.add(target)
            elif ":members:" in options:
                found |= {f"{target}.{n}" for n in importlib.import_module(target).__all__}
    return found


def test_every_name_the_package_exports_is_in_the_reference():
    """Review 2026-10-05 (CI5): ``po.ReadinessWarning`` and ``po.FormulaTarget``
    were in ``po.__all__`` and on no page. Each name the package exports, but
    its modules (each a page of its own) and ``__version__``, is documented
    under the package or under the module that defines it."""
    import types

    import polars_online as po

    documented = _documented()
    assert "polars_online.ModelBank" in documented, "the reader found nothing"
    missing = []
    for name in po.__all__:
        obj = getattr(po, name)
        if isinstance(obj, types.ModuleType) or name.startswith("__"):
            continue
        where = {f"polars_online.{name}", f"{getattr(obj, '__module__', '')}.{name}"}
        if not where & documented:
            missing.append(name)
    assert not missing, missing
