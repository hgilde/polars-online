"""`import rheon` needs no Polars; `rheon.polars` does, and says how to get it.

Polars is the `rheon[polars]` extra. The neutral names -- the spec builders,
`target`, the Gram and correlation tools, the errors, warnings and versions --
work in an environment without Polars, and `from rheon import polars` there
raises an `ImportError` naming the extra.

Each case runs in a child interpreter of its own (`python -I`, so nothing on
`PYTHONPATH` or the current directory shadows Polars with `rheon/polars`),
since what a process has imported cannot be undone. Polars is made
unimportable by a `sys.meta_path` finder, the technique
`tests/_site/without_pyarrow.py` uses for pyarrow; `scripts/wheel_smoke.py
--bare` runs the same check on an installed wheel.

One thing the Python side cannot promise: with Polars *installed*, the
compiled extension imports it on its own. Its global allocator,
pyo3-polars' `PolarsAllocator`, imports the allocator capsule Polars
publishes at the extension's first allocation, so that the extension and
Polars share one heap (docs/records/LINKING.md: without it the extension ran
on a second heap, 43% slower). So the case with Polars installed holds the
Python modules to importing nothing beyond what the extension imports
alone, and the `ast` walk holds each neutral module to no top-level Polars
import, by file and line.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "python" / "rheon"

#: The finder that makes Polars unimportable, as if it were not installed:
#: `polars`, its submodules and its runtime packages.
BLOCK = textwrap.dedent(
    """
    import sys

    class WithoutPolars:
        @staticmethod
        def find_spec(name, path=None, target=None):
            top = name.split(".")[0]
            if top == "polars" or top.startswith(("polars_runtime", "_polars_runtime")):
                raise ModuleNotFoundError(f"No module named {name!r} (blocked)", name=name)
            return None

    assert "polars" not in sys.modules
    sys.meta_path.insert(0, WithoutPolars)
    """
)


def _run(*parts: str) -> str:
    """Run the code ``parts``, each dedented, in a fresh isolated
    interpreter; its stdout, or a failure carrying its stderr."""
    res = subprocess.run(
        [sys.executable, "-I", "-c", "\n".join(textwrap.dedent(p) for p in parts)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
        cwd=REPO,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


def test_import_rheon_works_without_polars() -> None:
    out = _run(
        BLOCK,
        """
        import rheon

        assert "polars" not in sys.modules
        names = {
            "spec", "target", "Target", "FormulaTarget", "corr", "gram", "ArrowStruct",
            "ColumnNotFoundError", "OutputColumnNotFoundError", "RefusedValueError",
            "WrongTypeError", "PolarsOnlineWarning", "PolarsOnlineDeprecationWarning",
            "UnstableWarning", "ReadinessWarning", "native_version", "schema_version",
            "thread_pool_size", "__version__",
        }
        assert names == set(rheon.__all__), set(rheon.__all__) ^ names

        spec = rheon.spec.ewridge("m", targets=["y", rheon.target("z", name="w")],
                                  features=["x"], half_life=50.0)
        assert spec["targets"][1] == {"column": "z", "name": "w"}, spec
        assert rheon.spec.output_fields(spec)[0].startswith("pred_")
        from datetime import timedelta
        assert rheon.spec.ewridge("c", targets=["y"], features=["x"], clock="t",
                                  half_life=timedelta(minutes=10), gap_cap="1h",
                                  )["half_life"] == "10m"
        assert rheon.native_version() and rheon.schema_version() > 0
        assert rheon.thread_pool_size(threads=2) == 2

        import numpy as np

        try:
            rheon.corr.spectral(np.eye(2), 1.5)
        except rheon.WrongTypeError as e:
            assert "k must be a whole number" in str(e), e
        else:
            raise AssertionError("k=1.5 was taken")
        g = {"columns": ["a", "b"], "comoments": np.array([[2.0, 1.0], [1.0, 2.0]])}
        assert abs(rheon.gram.correlation(g)[0, 1] - 0.5) < 1e-12
        r, _, _ = rheon.corr.nearest(np.array([[1.0, 1.2], [1.2, 1.0]]))
        assert r.shape == (2, 2) and abs(r[0, 1] - 1.0) < 1e-6
        assert "polars" not in sys.modules
        print("ok")
        """,
    )
    assert out.strip() == "ok"


@pytest.mark.parametrize(
    "call",
    [
        "from rheon import polars",
        "import rheon.polars",
        'rheon.spec.output_index(rheon.spec.ewridge("m", targets=["y"], features=["x"], '
        "half_life=5.0))",
        'rheon.spec.coef_index(rheon.spec.ewridge("m", targets=["y"], features=["x"], '
        "half_life=5.0))",
    ],
)
def test_the_polars_parts_without_polars_name_the_extra(call: str) -> None:
    out = _run(
        BLOCK,
        f"""
        import rheon
        try:
            {call}
        except ImportError as e:
            print(e)
        else:
            raise AssertionError("no ImportError")
        """,
    )
    assert 'pip install "rheon[polars]"' in out, out


def test_the_errors_raised_without_polars_are_the_neutral_classes() -> None:
    """With no `rheon.polars`, the extension raises `rheon`'s own classes:
    `flavoured` finds no Polars-parented one."""
    out = _run(
        BLOCK,
        """
        import rheon
        from rheon import _errors
        assert _errors.flavoured("ColumnNotFoundError") is rheon.ColumnNotFoundError
        assert _errors.flavoured("WrongTypeError") is rheon.WrongTypeError
        assert rheon.ColumnNotFoundError.__bases__ == (ValueError,)
        assert rheon.OutputColumnNotFoundError.__bases__ == (rheon.ColumnNotFoundError,)
        assert rheon.UnstableWarning.__bases__ == (rheon.PolarsOnlineWarning, UserWarning)
        print("ok")
        """,
    )
    assert out.strip() == "ok"


def _polars_modules(code: str) -> list[str]:
    out = _run(
        code,
        """
        import sys
        print("\\n".join(sorted(m for m in sys.modules if "polars" in m.split(".")[0])))
        """,
    )
    return out.split()


def test_with_polars_installed_the_python_side_imports_none_of_it() -> None:
    """`import rheon` imports exactly the Polars modules the compiled
    extension imports when loaded on its own (its allocator's capsule): the
    Python modules add none. And `rheon.polars` is the full package."""
    import rheon

    ext = Path(rheon._rheon.__file__)
    alone = _polars_modules(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_rheon", {str(ext)!r})
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        """
    )
    assert "polars" in alone, "the extension's allocator no longer imports Polars: tighten this"
    assert _polars_modules("import rheon") == alone
    out = _run(
        """
        import sys
        import rheon
        from rheon import polars as rp
        assert rp.ModelBank is sys.modules["rheon.polars._bank"].ModelBank
        assert rp.spec is rheon.spec and rp.WrongTypeError is rheon.WrongTypeError
        assert issubclass(rp.ColumnNotFoundError, rheon.ColumnNotFoundError)
        print("ok")
        """
    )
    assert out.strip() == "ok"


def test_no_neutral_module_imports_polars_at_the_top_level() -> None:
    """Every module of `rheon` outside `rheon/polars/` imports Polars only
    under `TYPE_CHECKING` or inside a function; a regression is named by file
    and line."""
    found = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # the top level only: not a function, not an `if`
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(n.split(".")[0] == "polars" for n in names):
                found.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not found, found


def test_the_spec_builders_write_the_same_bytes_with_and_without_polars() -> None:
    code = """
        import json
        from datetime import timedelta
        import rheon
        from rheon import _spec
        specs = [
            rheon.spec.ewridge("a", targets=["y", rheon.target("z", name="w")],
                               features=["x0", "x1"], half_life=[50.0, "inf"]),
            rheon.spec.ewridge("b", targets=["y"], features=["x"], clock="t",
                               half_life=timedelta(minutes=5), gap_cap="1h"),
            rheon.spec.ew_cov("c", features=["x0", "x1"], stats=["mean"], half_life=10.0),
        ]
        print(_spec._json(specs))
        """
    assert _run(BLOCK, code) == _run("import polars", code)


def test_wheel_smoke_bare_passes_on_the_checkout() -> None:
    """`scripts/wheel_smoke.py --bare`, the check the release makes of the
    installed wheel, run against this checkout."""
    out = _run(
        f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wheel_smoke", {str(REPO / "scripts" / "wheel_smoke.py")!r}
        )
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        print(smoke.check_bare(installed=False))
        """
    )
    assert "without Polars: ok" in out, out
