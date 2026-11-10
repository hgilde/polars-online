"""A docstring example that shows its output prints that output.

WRITING §10 gives every entry one short example that runs, with its
output. ``tests/test_production_hardening.py`` runs every docstring's python
block; here a block that prints, followed directly by a ``.. code-block:: text``, is
a recipe, and what it prints is held to that text with
``tests/doc_recipes.py``'s ``differences``, as a page's recipe is. It runs
in the namespace *Example data* (docs/EXAMPLE-DATA.md) builds, which the docstring
names where it reads ``df``.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import textwrap

import pytest

import polars_online as po
from doc_recipes import differences
from polars_version import block_marks
from test_production_hardening import _readme_namespace

TIER = "essential"


def _blocks(doc: str) -> list[tuple[str, str]]:
    """Each ``.. code-block::`` of a dedented docstring: its language and
    its body, dedented."""
    out: list[tuple[str, str]] = []
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(".. code-block::"):
            lang = line.strip().split("::", 1)[1].strip()
            indent = len(line) - len(line.lstrip())
            j, body = i + 1, []
            while j < len(lines) and (
                not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent
            ):
                body.append(lines[j])
                j += 1
            out.append((lang, textwrap.dedent("\n".join(body)).strip("\n")))
            i = j
        else:
            i += 1
    return out


def _documented() -> list[tuple[str, object]]:
    """The public objects whose docstrings the API reference is built from."""
    modules = [po, po.spec, po._bank, po._frame]
    modules += [po.gram, po.eval, po.corr, po.stream, po.sim, po.gbdt]
    seen: set[int] = set()
    out: list[tuple[str, object]] = []
    for mod in modules:
        for name in getattr(mod, "__all__", ()):
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            members = [(f"{mod.__name__}.{name}", obj)]
            if inspect.isclass(obj):
                members += [
                    (f"{mod.__name__}.{name}.{m}", v)
                    for m, v in vars(obj).items()
                    if not m.startswith("_") and callable(v)
                ]
            for where, o in members:
                if id(o) not in seen:
                    seen.add(id(o))
                    out.append((where, o))
    return out


RECIPES = [
    pytest.param(code, shown, id=where, marks=block_marks(code, output=True))
    for where, obj in _documented()
    for (lang, code), (after, shown) in zip(
        _blocks(inspect.getdoc(obj) or ""), _blocks(inspect.getdoc(obj) or "")[1:], strict=False
    )
    # A text block after a block that prints is its output; one after a
    # block that prints nothing shows something else, such as a layout.
    if lang == "python" and after == "text" and "print(" in code
]


def test_the_recipes_are_found():
    """``ModelBank.fit_predict`` shows its output, at least."""
    assert any("fit_predict" in p.id for p in RECIPES), [p.id for p in RECIPES]


@pytest.mark.parametrize(("code", "shown"), RECIPES)
def test_a_docstring_recipe_prints_what_it_shows(code, shown, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns: dict[str, object] = {"__name__": "__recipe__", **_readme_namespace(tmp_path)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<docstring>", "exec"), ns)
    problems = differences(shown, buf.getvalue())
    assert not problems, "\n".join(problems)
