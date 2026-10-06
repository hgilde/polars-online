"""Run an installed wheel: the check release.yml makes of each wheel it builds.

Three of the release's six wheels (x86_64 macOS, aarch64 Linux, musl) were
built and uploaded without ever being installed or imported, and none of the
six was tested as the file that ships (task 160, CI3). The build job now
installs the wheel it built into a fresh environment, with its dependencies
from PyPI as ``pip install`` gives them to a user, and runs this:

    python scripts/wheel_smoke.py --version 0.14.0

It checks that the package comes from that environment rather than a
checkout, that it is the version being released, and that the native module
works: one spec through ``ModelBank.fit_predict`` against the line it was fed,
a state saved and resumed against the unbroken run, and the same spec as a
streaming ``LazyFrame`` plan. The package's own dependencies only: polars,
and not even the optional NumPy, which a fresh install leaves out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def check(version: str | None = None, installed: bool = True) -> str:
    """Raise on the first failed check; return a line naming what ran."""
    import polars as pl

    import polars_online as po

    where = Path(po.__file__).resolve()
    if installed and where.is_relative_to(REPO / "python"):
        raise SystemExit(f"polars_online came from the checkout ({where}), not the wheel")
    if version is not None and po.__version__ != version:
        raise SystemExit(f"the wheel is polars-online {po.__version__}, not {version}")

    # A line, fed without noise: by row 100 the fit has it to well inside
    # 1e-4 (the default ridge leaves about 1e-6).
    n = 200
    xs = [((i * 37) % 101) / 10.0 - 5.0 for i in range(n)]
    df = pl.DataFrame({"x": xs, "y": [1.0 + 2.0 * x for x in xs]})
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=float("inf"))

    whole = po.ModelBank([spec]).fit_predict(df)
    out = whole.unnest("m")
    if out["pred_y"][0] is not None:
        raise SystemExit("the first row has a prediction: nothing came before it")
    worst = (out["pred_y"].tail(100) - (1.0 + 2.0 * df["x"].tail(100))).abs().max()
    if not isinstance(worst, float) or worst > 1e-4:
        raise SystemExit(f"the fit misses y = 1 + 2x by {worst}")

    first = po.ModelBank([spec])
    first.fit_predict(df.head(120))
    resumed = po.ModelBank.load_bytes(first.save_bytes(), specs=[spec])
    if not resumed.fit_predict(df.tail(80)).equals(whole.tail(80)):
        raise SystemExit("a saved and resumed state parts from the unbroken run")

    streamed = df.lazy().online.fit_predict([spec]).collect()
    if not streamed.equals(whole):
        raise SystemExit("the streaming plan parts from ModelBank.fit_predict")

    return f"polars-online {po.__version__} on polars {pl.__version__}, from {where.parent}: ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="the version the wheel must be")
    args = parser.parse_args()
    print(check(args.version), flush=True)


if __name__ == "__main__":
    sys.exit(main())
