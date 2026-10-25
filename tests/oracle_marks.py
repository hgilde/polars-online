"""Which tests the ``oracle`` mark selects (docs/TESTING.md, "The machines and
the CPU paths").

A test is an *oracle* test when it holds this library's numbers to another
computation of them whose last bits depend on the CPU's paths: scipy,
statsmodels or scikit-learn, NumPy's linear algebra (OpenBLAS's kernels, one
set a CPU type), or the NumPy references in ``tests/reference*.py``, which
call both. ``tests/conftest.py`` marks every test in a module that imports
one of those or calls ``numpy.linalg``, so the mark cannot fall behind the
tests, and ``.github/workflows/cpu-paths.yml`` runs ``-m oracle`` with
OpenBLAS's kernels and NumPy's loops pinned one way and another.

A test that runs the command line (the ``online_cli`` fixture) is left
unmarked: it compares the binary with the bank, not with an oracle, and
building the binary would take a CPU-paths leg most of its time.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

#: The modules whose import makes a test module an oracle module: the
#: third-party libraries, the tests' own NumPy references, and the
#: convergence harness, whose oracles are statsmodels, scikit-learn and scipy.
ORACLE_MODULES = frozenset(
    {
        "scipy",
        "statsmodels",
        "sklearn",
        "reference",
        "reference_paths",
        "reference_cluster",
        "convergence",
    }
)

#: The fixture of the tests that run the command line.
CLI_FIXTURE = "online_cli"


def _numpy_linalg(node: ast.AST) -> bool:
    """``np.linalg`` or ``numpy.linalg``, as an attribute or an import."""
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "linalg"
            and isinstance(node.value, ast.Name)
            and node.value.id in ("np", "numpy")
        )
    if isinstance(node, ast.Import):
        return any(a.name == "numpy.linalg" for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module == "numpy.linalg" or (
            node.module == "numpy" and any(a.name == "linalg" for a in node.names)
        )
    return False


def _imports_an_oracle(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] in ORACLE_MODULES for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return node.module.split(".")[0] in ORACLE_MODULES
    return False


@functools.cache
def is_oracle_module(path: Path) -> bool:
    """Whether the test module at ``path`` compares with an oracle."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(_imports_an_oracle(n) or _numpy_linalg(n) for n in ast.walk(tree))


def is_oracle_test(path: Path, fixtures: tuple[str, ...] | list[str]) -> bool:
    """Whether a test, from its module's path and its fixtures, is one."""
    return CLI_FIXTURE not in fixtures and is_oracle_module(path)
