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
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
META = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
TESTS = sorted((REPO / "tests").glob("*.py"))
#: This repository's own modules: the tests' helpers, the examples and
#: scripts some tests import by path, and the package.
LOCAL = {p.stem for d in ("tests", "examples", "scripts") for p in (REPO / d).glob("*.py")} | {
    "polars_online"
}


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


def test_no_test_skips_for_want_of_a_library() -> None:
    """The dev group is installed wherever the suite runs, so an
    ``importorskip`` could only turn a broken environment into a skip."""
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in TESTS
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == "importorskip"
    ]
    assert not offenders, offenders
