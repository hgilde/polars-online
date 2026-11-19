"""Check a release as a user gets it: installed from PyPI, then run.

Usage: python -I scripts/verify_release.py --version X.Y.Z --schema N
           [--distribution polars-online-compat]

The release workflow's ``verify`` job runs this once the upload is done, on
Linux, macOS and Windows, in a fresh environment into which it installed
``polars-online==X.Y.Z`` from PyPI, or ``polars-online-compat==X.Y.Z``, the
same source built for every x86-64 CPU, which the same job checks in a leg
of its own (``--distribution``). It checks that the package came from
that environment and not a checkout, that the three versions it reports
agree with the release -- ``__version__``, the native module's
``native_version()`` and its state schema, ``schema_version()``, against
the schema the release was built with -- and that one fit runs through the
streaming plan: an ``ewridge`` over a parquet file written here, read back
with ``pl.scan_parquet``, whose last coefficient must be the line's slope.
The package's own dependencies only, as a fresh ``pip install`` gives them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The line the fit is fed: y = 1 + SLOPE * x, and a small noise that
#: repeats every seven rows.
SLOPE = 2.5


#: The two PyPI projects built from this source: one import name.
DISTRIBUTIONS = ("polars-online", "polars-online-compat")


def check(
    version: str, schema: int, installed: bool = True, distribution: str | None = None
) -> str:
    """Raise ``SystemExit`` on the first failed check; return what ran.

    With a ``distribution`` and an installed package, the check also holds
    it to that project's name, installed alone, and to this CPU having every
    feature the build needs: the warning of the import-time CPU check would
    be the first thing a user on this OS saw."""
    import polars as pl

    import rheon as po
    from rheon import _cpu_check

    where = Path(po.__file__).resolve()
    if installed and where.is_relative_to(REPO / "python"):
        raise SystemExit(f"rheon came from the checkout ({where}), not from PyPI")
    if installed and distribution is not None:
        for name in DISTRIBUTIONS:
            try:
                importlib.metadata.version(name)
                found = True
            except importlib.metadata.PackageNotFoundError:
                found = False
            if found != (name == distribution):
                raise SystemExit(
                    f"{name} is {'installed' if found else 'not installed'}: "
                    f"the package under test is {distribution} alone"
                )
        missing = _cpu_check.missing_features()
        if missing:
            raise SystemExit(f"this CPU lacks {', '.join(missing)}, which the wheel needs")
    for name, got in (
        ("__version__", po.__version__),
        ("native_version()", po.native_version()),
    ):
        if got != version:
            raise SystemExit(f"rheon.{name} is {got}, not {version}")
    if po.schema_version() != schema:
        raise SystemExit(f"rheon.schema_version() is {po.schema_version()}, not {schema}")

    n = 2000
    xs = [((i * 37) % 101) / 10.0 - 5.0 for i in range(n)]
    ys = [1.0 + SLOPE * x + 0.01 * ((i * 13) % 7 - 3) for i, x in enumerate(xs)]
    s = po.spec.ewridge("fit", targets=["y"], features=["x"], half_life=500.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "line.parquet"
        pl.DataFrame({"x": xs, "y": ys}).write_parquet(path)
        out = pl.scan_parquet(path).online.fit_predict([s]).online.unnest([s]).collect()
    if out.height != n:
        raise SystemExit(f"the fit returned {out.height} rows of {n}")
    coefs = out["coef_y_x"].drop_nulls()
    if coefs.is_empty():
        raise SystemExit("the fit reported no coefficient")
    slope = float(coefs[-1])
    if abs(slope - SLOPE) > 1e-3:
        raise SystemExit(f"the fit's slope is {slope}, not {SLOPE}")
    return (
        f"polars-online {po.__version__} (native {po.native_version()}, schema "
        f"{po.schema_version()}) on polars {pl.__version__}, from {where.parent}: "
        f"slope {slope:.6f}: ok"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="the version released")
    parser.add_argument("--schema", required=True, type=int, help="the schema it was built with")
    parser.add_argument(
        "--distribution",
        choices=DISTRIBUTIONS,
        default=DISTRIBUTIONS[0],
        help="the project installed (default: polars-online)",
    )
    args = parser.parse_args()
    print(check(args.version, args.schema, True, args.distribution), flush=True)


if __name__ == "__main__":
    sys.exit(main())
