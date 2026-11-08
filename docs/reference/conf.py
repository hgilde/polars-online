"""Sphinx configuration for the API reference.

Build: ``uv run --group docs sphinx-build -W docs/reference docs/_build/html``.
``-W`` turns every warning into a failure, so a docstring that is not valid
reStructuredText, or a cross-reference to a name that has moved, fails the
build (and the gate, and CI) rather than rendering wrong. The docstrings
are the source: nothing here is written twice.

Each public object has a page of its own, generated at build time under
``api/`` (git ignores it) from ``families.py``. The module pages keep their
addresses and list their objects by family (the ``entries`` directive).
"""

import inspect
import shutil
import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.errors import ExtensionError

import polars_online

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import families  # noqa: E402 -- beside this file, found through the path above

project = "polars-online"
author = "Hans Gilde"
release = polars_online.__version__
copyright = "Hans Gilde"  # noqa: A001 -- Sphinx's name for it

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
# The family tables are written by the ``entries`` directive and the pages
# by ``_generate``; autosummary only draws a class page's table of members.
autosummary_generate = False

# Members in source order, so a class reads the way its file does; only
# documented members, so an undocumented helper is absent rather than a
# bare signature. `__init__`'s docstring joins the class's.
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
autoclass_content = "both"
# Types stay in the signature (the docstrings describe parameters in prose,
# not ``:param:`` fields, so moving hints to the description would drop
# them), written short: ``DataFrame``, not ``polars.dataframe.frame.DataFrame``.
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

html_theme = "furo"
html_title = f"polars-online {release}"
html_static_path: list[str] = []
# `llms.txt` (llmstxt.org) is written for coding agents and lives at the repo
# root, where GitHub serves it; copying it verbatim into the build root puts it
# at https://hgilde.github.io/polars-online/llms.txt as well, so one file serves
# both and cannot drift from itself. tests/test_llms_txt.py holds its contents
# to the registry.
html_extra_path = ["../../llms.txt"]
# The design notes live beside this directory as Markdown; Sphinx must not
# try to read them.
exclude_patterns = ["_build"]

API = "api"


def _object(name: str):
    """The object a dotted name names, imported."""
    import importlib

    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(name)


def _members(name: str) -> list[tuple[str, str]]:
    """A paged class's public methods and properties, in source order, as
    (dotted name, ``method`` or ``property``); a name kept only under its
    old spelling is left out, as the reference leaves it out everywhere."""
    from polars_online._renamed import is_renamed

    out = []
    for attr, value in vars(_object(name)).items():
        if attr.startswith("_"):
            continue
        target = value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
        if is_renamed(target):
            continue
        if isinstance(value, property) or inspect.isgetsetdescriptor(value):
            out.append((f"{name}.{attr}", "property"))
        elif callable(target):
            out.append((f"{name}.{attr}", "method"))
    return out


def _summary(obj) -> str:
    """The first paragraph of an object's docstring, on one line."""
    doc = inspect.getdoc(obj) or ""
    return " ".join(doc.split("\n\n", 1)[0].split())


def _directive(name: str, obj) -> str:
    if inspect.isclass(obj):
        return "autoclass"
    return "autofunction"


def _page(name: str, body: list[str]) -> str:
    title = families.title(name)
    return "\n".join([title, "=" * len(title), "", *body, ""])


def _generate(app) -> None:
    """Write a page for every entry, and one for each member of a paged
    class, under ``api/``; stale pages from an earlier build go first."""
    out = HERE / API
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for fam in families.FAMILIES:
        for name in fam.entries:
            obj = _object(name)
            if name in fam.paged:
                members = _members(name)
                body = [
                    f".. autoclass:: {name}",
                    "   :no-members:",
                    "",
                    ".. rubric:: Methods and properties",
                    "",
                    ".. autosummary::",
                    "",
                    *(f"   ~{m}" for m, _ in members),
                    "",
                    ".. toctree::",
                    "   :hidden:",
                    "",
                    *(f"   {m}" for m, _ in members),
                ]
                for m, kind in members:
                    directive = "autoproperty" if kind == "property" else "automethod"
                    (out / f"{m}.rst").write_text(
                        _page(m, [f".. {directive}:: {m}"]), encoding="utf-8"
                    )
            else:
                body = [f".. {_directive(name, obj)}:: {name}"]
            kind = "class" if inspect.isclass(obj) else "function"
            (out / f"{families.page_name(name, kind)}.rst").write_text(
                _page(name, body), encoding="utf-8"
            )


def _public() -> set[str]:
    """Every object the reference must place: the package's exports, each
    public module's, but modules, constants and names kept only under their
    old spelling (constants stay on their module's page)."""
    import importlib
    import pkgutil
    import types

    from polars_online._renamed import is_renamed

    found: set[str] = set()

    def consider(where: str, obj) -> None:
        if isinstance(obj, types.ModuleType) or not callable(obj) or is_renamed(obj):
            return
        module = getattr(obj, "__module__", "") or ""
        if module.startswith("polars_online.") and not module.split(".")[1].startswith("_"):
            where = f"{module}.{obj.__name__}"
        found.add(where)

    for name in polars_online.__all__:
        if not name.startswith("__"):
            consider(f"polars_online.{name}", getattr(polars_online, name))
    for m in pkgutil.iter_modules(polars_online.__path__):
        if m.name.startswith("_"):
            continue
        module = importlib.import_module(f"polars_online.{m.name}")
        for name in getattr(module, "__all__", ()):
            consider(f"polars_online.{m.name}.{name}", getattr(module, name))
    found |= {
        "polars_online._frame.LazyFrameOnlineNamespace",
        "polars_online._frame.DataFrameOnlineNamespace",
    }
    return found


def _check_placed(app) -> None:
    """Fail the build when a public object is in no family, in two, or a
    family names something that does not exist."""
    placed = [n for f in families.FAMILIES for n in f.entries]
    twice = sorted({n for n in placed if placed.count(n) > 1})
    public = _public()
    missing = sorted(public - set(placed))
    extra = sorted(set(placed) - public)
    if twice or missing or extra:
        raise ExtensionError(
            f"families.py: placed twice {twice}, placed nowhere {missing}, not public {extra}"
        )


class Entries(Directive):
    """``.. entries:: <family>``: a table of the family's objects, each
    linked to its page, beside an anchor for every id the module page once
    gave it, and a hidden table of contents of those pages."""

    required_arguments = 1

    def run(self):
        fam = families.family(self.arguments[0])
        rows: list[str] = [".. list-table::", "   :widths: 35 65", ""]
        docs: list[str] = []
        for name in fam.entries:
            obj = _object(name)
            ids = [name]
            members = _members(name) if name in fam.paged or inspect.isclass(obj) else []
            ids += [m for m, _ in members]
            role = "class" if inspect.isclass(obj) else "func"
            anchors = "".join(f'<span id="{i}"></span>' for i in ids)
            rows += [
                f"   * - :raw-html:`{anchors}` :{role}:`{families.title(name)} <{name}>`",
                f"     - {_summary(obj)}",
            ]
            docs.append(
                f"{API}/{families.page_name(name, 'class' if role == 'class' else 'function')}"
            )
        text = [".. role:: raw-html(raw)", "   :format: html", "", *rows, ""]
        text += [".. toctree::", "   :hidden:", "", *(f"   {d}" for d in docs), ""]
        node = nodes.container()
        self.state.nested_parse(StringList(text, source=f"entries:{fam.slug}"), 0, node)
        return node.children


def _skip_renamed(app, what, name, obj, skip, options):
    """Leave out a name kept only under its old spelling -- a refusal stub
    or a forwarding with a deprecation warning -- so the reference shows the
    new name alone, as the API snapshot does (``_renamed.is_renamed``)."""
    from polars_online._renamed import is_renamed

    return True if is_renamed(obj) else None


def setup(app):
    app.connect("builder-inited", _check_placed)
    app.connect("builder-inited", _generate)
    app.connect("autodoc-skip-member", _skip_renamed)
    app.add_directive("entries", Entries)
