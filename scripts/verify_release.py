"""Check a release as a user gets it: installed from PyPI, then run.

Usage: python -I scripts/verify_release.py --version X.Y.Z --schema N

The release workflow's ``verify`` job runs this once the upload is done, on
Linux, macOS and Windows, in a fresh environment into which it installed
``polars-online==X.Y.Z`` from PyPI. It checks that the package came from
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
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The line the fit is fed: y = 1 + SLOPE * x, and a small noise that
#: repeats every seven rows.
SLOPE = 2.5


def check(version: str, schema: int, installed: bool = True) -> str:
    """Raise ``SystemExit`` on the first failed check; return what ran."""
    import polars as pl

    import polars_online as po

    where = Path(po.__file__).resolve()
    if installed and where.is_relative_to(REPO / "python"):
        raise SystemExit(f"polars_online came from the checkout ({where}), not from PyPI")
    for name, got in (
        ("__version__", po.__version__),
        ("native_version()", po.native_version()),
    ):
        if got != version:
            raise SystemExit(f"polars_online.{name} is {got}, not {version}")
    if po.schema_version() != schema:
        raise SystemExit(f"polars_online.schema_version() is {po.schema_version()}, not {schema}")

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
    args = parser.parse_args()
    print(check(args.version, args.schema), flush=True)


if __name__ == "__main__":
    sys.exit(main())
