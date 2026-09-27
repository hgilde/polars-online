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
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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
        # A bare `except:` has no type to walk (review 2026-09-26, F5).
        if handler.type is None:
            continue
        names = {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
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


#: Licences a library in the dev or docs group may carry: those the Open
#: Source Initiative approves, as SPDX identifiers, and CC0, a public-domain
#: dedication that asks for nothing (numpy carries it for part of its code).
#: docs/TESTING.md, "Libraries the package does not depend on"; the user,
#: 2026-09-25: "enable every unlicensed library in tests and park using
#: licensed libraries". A classifier "OSI Approved" counts the same.
OPEN = {
    "0BSD",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "Apache-2.0",
    "Artistic-2.0",
    "BSD-1-Clause",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC0-1.0",
    "EPL-2.0",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "ISC",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MIT",
    "MIT-0",
    "MPL-2.0",
    "PSF-2.0",
    "Python-2.0",
    "Unlicense",
    "UPL-1.0",
    "Zlib",
}

#: Words that mark a source-available or commercial licence, wherever the
#: metadata puts them: those libraries are parked, not tested with.
PARKED = (
    "busl",
    "business source",
    "sspl",
    "server side public",
    "commercial",
    "proprietary",
    "elastic license",
)

#: Libraries whose installed metadata names no licence at all, with the
#: licence read at the source and where. An entry goes once the metadata
#: names one (`test_the_licence_table_holds_only_what_the_metadata_lacks`).
READ_AT_THE_SOURCE = {
    "bayesian-changepoint-detection": (
        "MIT",
        "LICENSE in github.com/hildensia/bayesian_changepoint_detection at 5b7edb6, "
        "read 2026-09-25; the 0.2.dev1 wheel declares none",
    ),
}


def _spdx_ids(expression: str) -> set[str]:
    """The licence identifiers in an SPDX expression, operators dropped."""
    words = re.split(r"[\s()]+", expression)
    return {w for w in words if w and w not in {"AND", "OR", "WITH"}}


def licence_of(metadata: importlib.metadata.PackageMetadata) -> str | None:
    """What the metadata says the licence is, from the most exact field it
    fills: `License-Expression` (PEP 639), then the trove classifiers, then
    the free-text `License`. `None` when it says nothing."""
    expression = metadata.get("License-Expression")
    if expression:
        return expression.strip()
    classifiers = [
        c.removeprefix("License :: ")
        for c in metadata.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)
    text = (metadata.get("License") or "").strip()
    return text or None


def is_open(licence: str) -> bool:
    """Whether a licence as `licence_of` gives it is open: nothing in it is
    `PARKED`, and it is an SPDX expression of `OPEN` identifiers or an
    "OSI Approved" classifier. Free text is not read for a name, since
    "permitted" contains "MIT"; a library that gives only text is in
    `READ_AT_THE_SOURCE` or refused."""
    if any(word in licence.lower() for word in PARKED):
        return False
    if "OSI Approved ::" in licence:
        return True
    ids = _spdx_ids(licence)
    return bool(ids) and ids <= OPEN


def _groups() -> list[str]:
    return [_name(r) for group in ("dev", "docs") for r in META["dependency-groups"][group]]


def test_every_test_library_carries_an_open_licence() -> None:
    """Each library of the dev and docs groups names an open-source licence
    in its metadata, or is in the table of those read at the source: a
    licensed one is refused here rather than remembered."""
    bad = {}
    for name in _groups():
        declared = licence_of(importlib.metadata.metadata(name))
        if declared is None:
            if name not in READ_AT_THE_SOURCE:
                bad[name] = "no licence in its metadata, and none read at the source"
            continue
        if not is_open(declared):
            bad[name] = declared
    assert not bad, f"libraries without an open licence: {bad}"


def test_the_licence_table_holds_only_what_the_metadata_lacks() -> None:
    for name, (licence, where) in READ_AT_THE_SOURCE.items():
        assert name in _groups(), f"{name} is in no group any more"
        assert licence_of(importlib.metadata.metadata(name)) is None, (
            f"{name}'s metadata names a licence now; drop it from the table"
        )
        assert is_open(licence) and where


def test_every_rust_test_library_carries_an_open_licence() -> None:
    """The same rule for the crates' dev-dependencies, which link into the
    test binaries alone (hard rule 12 covers what ships, not tests: the
    user, 2026-09-25). Their licences are read from ``cargo metadata``,
    offline, from the lock."""
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--offline"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    meta = json.loads(out)
    by_id = {p["id"]: p for p in meta["packages"]}
    licence = {p["name"]: p.get("license") for p in meta["packages"]}
    dev = {
        dep["name"]
        for member in meta["workspace_members"]
        for dep in by_id[member]["dependencies"]
        if dep["kind"] == "dev"
    }
    assert "proptest" in dev, "the property tests' library is where it was put"
    bad = {name: licence[name] for name in sorted(dev) if not is_open(licence[name] or "")}
    assert not bad, f"Rust test libraries without an open licence: {bad}"


@pytest.mark.parametrize(
    ("licence", "open_"),
    [
        ("MIT", True),
        ("MIT OR Apache-2.0", True),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", True),
        ("OSI Approved :: BSD License", True),
        ("GPL-3.0-only", True),
        ("BSL-1.0", True),  # Boost, not the Business Source License
        ("BSD 3-Clause License", False),  # text, not an identifier
        ("Use is permitted within the licensed organization", False),
        ("BUSL-1.1", False),
        ("Business Source License 1.1", False),
        ("SSPL-1.0", False),
        ("MIT OR LicenseRef-Commercial", False),
        ("Other/Proprietary License", False),
        ("Elastic License 2.0", False),
    ],
)
def test_the_licence_rule_itself(licence: str, open_: bool) -> None:
    """The rule's own cases, so a check that passed everything would fail
    here: the forms the installed libraries use, and the parked ones."""
    assert is_open(licence) is open_
