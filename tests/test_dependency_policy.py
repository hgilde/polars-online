"""Libraries the package does not depend on, in the tests that need them.

docs/TESTING.md, "Libraries the package does not depend on", has the rule, as
the user put it on 2026-09-24: "packages that we do not want to depend on are
fine if needed to test with other libraries". These check its mechanical
parts. The package depends on polars alone. Every library a test imports is
declared in the dev group by name, not reached through another library's
dependencies. And no test skips for want of one, since the dev group is
installed wherever the suite runs.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import child

REPO = Path(__file__).resolve().parents[1]
META = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
TESTS = sorted((REPO / "tests").glob("*.py"))
#: This repository's own modules: the tests' helpers, the examples and
#: scripts some tests import by path, and the package.
LOCAL = {
    p.stem for d in ("tests", "tests/_site", "examples", "scripts") for p in (REPO / d).glob("*.py")
} | {"polars_online"}


def _name(requirement: str) -> str:
    """A requirement's distribution name, normalised as PEP 503 does."""
    bare = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", bare).lower()


def _imports() -> dict[str, set[str]]:
    """Every top-level module a test file imports, with the files that do."""
    found: dict[str, set[str]] = {}
    for path in TESTS:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in LOCAL or top == "__future__":
                    continue
                found.setdefault(top, set()).add(path.name)
    return found


def test_the_package_depends_on_polars_alone() -> None:
    assert [_name(r) for r in META["project"]["dependencies"]] == ["polars"]
    # The extras too: numpy for the few accessors that need it, and nothing
    # else rides in on an extra (review 2026-09-25).
    extras = {k: [_name(r) for r in v] for k, v in META["project"]["optional-dependencies"].items()}
    assert extras == {"numpy": ["numpy"]}, extras


def test_every_library_a_test_imports_is_declared_in_the_dev_group() -> None:
    """By name: pandas and scipy used to arrive through statsmodels, so
    dropping statsmodels would have broken tests that never mention it."""
    declared = {_name(r) for r in META["dependency-groups"]["dev"]}
    declared |= {_name(r) for r in META["project"]["dependencies"]}
    dists = importlib.metadata.packages_distributions()
    imports = _imports()
    assert {"numpy", "pandas", "pyarrow", "river"} <= set(imports), sorted(imports)
    missing = {
        module: ({_name(d) for d in dists.get(module, [])} or "not installed", sorted(files))
        for module, files in sorted(imports.items())
        if not {_name(d) for d in dists.get(module, [])} & declared
    }
    assert not missing, f"imported by tests, not declared in the dev group: {missing}"


def _skips_on_import_error(node: ast.AST) -> bool:
    """A ``try`` whose ``except ImportError`` (or ``ModuleNotFoundError``)
    handler calls ``pytest.skip``: the same false skip as ``importorskip``,
    by the commonest idiom (review 2026-09-25)."""
    if not isinstance(node, ast.Try):
        return False
    for handler in node.handlers:
        names = {
            n.id
            for n in ast.walk(handler.type)
            if handler.type is not None
            if isinstance(n, ast.Name)
        }
        if not names & {"ImportError", "ModuleNotFoundError"}:
            continue
        for inner in ast.walk(handler):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "skip"
            ):
                return True
    return False


def test_no_test_skips_for_want_of_a_library() -> None:
    """The dev group is installed wherever the suite runs, so an
    ``importorskip`` -- or a ``pytest.skip`` behind ``except ImportError``,
    or a ``skipif`` on ``find_spec`` -- could only turn a broken environment
    into a skip."""
    offenders = []
    for path in TESTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "importorskip":
                offenders.append(f"{path.name}:{node.lineno}: importorskip")
            elif _skips_on_import_error(node):
                offenders.append(f"{path.name}:{node.lineno}: skip behind except ImportError")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "skipif"
                and any(
                    isinstance(n, ast.Attribute) and n.attr == "find_spec"
                    for arg in node.args
                    for n in ast.walk(arg)
                )
            ):
                offenders.append(f"{path.name}:{node.lineno}: skipif on find_spec")
    assert not offenders, offenders


def test_child_interpreters_run_without_pyarrow_too() -> None:
    """The finder in tests/conftest.py lives in this process; the children
    the examples and the leak checks spawn get it from
    tests/_site/sitecustomize.py, which tests/child.py puts on their path.
    Without that path a child sees pyarrow, since the dev group has it."""
    probe = (
        "try:\n    import pyarrow\nexcept ModuleNotFoundError as e:\n"
        "    print('absent', 'without_pyarrow' in str(e))\nelse:\n    print('present')"
    )
    run = lambda env: subprocess.run(  # noqa: E731
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert run(child.env()) == ["absent", "True"]
    assert run(None) == ["present"]
