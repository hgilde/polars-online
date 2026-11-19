"""Run an installed wheel: the check release.yml makes of each wheel it builds.

Three of the release's six wheels (x86_64 macOS, aarch64 Linux, musl) were
built and uploaded without ever being installed or imported, and none of the
six was tested as the file that ships (task 160, CI3). The build job now
installs the wheel it built into a fresh environment, with its dependencies
from PyPI as ``pip install`` gives them to a user, and runs this:

    python scripts/wheel_smoke.py --version 0.14.0 [--distribution polars-online-compat]
                                  [--features +avx2,+fma,...]

It checks that the package comes from that environment rather than a
checkout, that it is the version being released, and that the native module
works: one spec through ``ModelBank.fit_predict`` against the line it was fed,
a state saved and resumed against the unbroken run, and the same spec as a
streaming ``LazyFrame`` plan. The package's own dependencies only: polars,
and not even the optional NumPy, which a fresh install leaves out.

The package is one of two distributions of the one source, ``polars-online``
and ``polars-online-compat``, imported alike, and the x86-64 wheels of the
first are built for AVX2, FMA and more. So the check also holds the wheel to
its name (``--distribution``: installed under it, and not under the other),
to the CPU features it was built to need (``--features``: the list the build
recorded for the import-time CPU check, empty for the compat build, the ARM
wheels and the sdist, and the release's list for an x86-64 ``polars-online``
wheel) and, where it needs any, to the machine it runs on having them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


#: The two PyPI projects built from this source: one import name.
DISTRIBUTIONS = ("polars-online", "polars-online-compat")


def check_build(distribution: str, features: str) -> None:
    """Raise unless the installed distribution is ``distribution`` alone, the
    features the build recorded are ``features``, and this CPU has them."""
    from rheon import _cpu_check

    if distribution not in DISTRIBUTIONS:
        raise SystemExit(f"unknown distribution {distribution!r}: one of {DISTRIBUTIONS}")
    for name in DISTRIBUTIONS:
        try:
            importlib.metadata.version(name)
            found = True
        except importlib.metadata.PackageNotFoundError:
            found = False
        if found != (name == distribution):
            raise SystemExit(
                f"{name} is {'installed' if found else 'not installed'}: "
                f"the wheel under test is {distribution} alone"
            )
    recorded = _cpu_check.baked_flags()
    if recorded != features:
        raise SystemExit(
            f"the build recorded the CPU features {recorded!r}, not the expected {features!r}"
        )
    missing = _cpu_check.missing_features()
    if missing:
        raise SystemExit(f"this CPU lacks {', '.join(missing)}, which the wheel was built for")


def check(
    version: str | None = None,
    installed: bool = True,
    distribution: str | None = None,
    features: str = "",
) -> str:
    """Raise on the first failed check; return a line naming what ran.

    ``distribution`` and ``features`` are checked when the wheel is installed
    and a distribution is named (a checkout has no distribution of its own)."""
    import polars as pl

    from rheon import polars as rp

    where = Path(rp.__file__).resolve()
    if installed and where.is_relative_to(REPO / "python"):
        raise SystemExit(f"rheon came from the checkout ({where}), not the wheel")
    if version is not None and rp.__version__ != version:
        raise SystemExit(f"the wheel is polars-online {rp.__version__}, not {version}")

    if installed and distribution is not None:
        check_build(distribution, features)

    # A line, fed without noise: by row 100 the fit has it to well inside
    # 1e-4 (the default ridge leaves about 1e-6).
    n = 200
    xs = [((i * 37) % 101) / 10.0 - 5.0 for i in range(n)]
    df = pl.DataFrame({"x": xs, "y": [1.0 + 2.0 * x for x in xs]})
    spec = rp.spec.ewridge("m", targets=["y"], features=["x"], half_life=float("inf"))

    whole = rp.ModelBank([spec]).fit_predict(df)
    out = whole.unnest("m")
    if out["pred_y"][0] is not None:
        raise SystemExit("the first row has a prediction: nothing came before it")
    worst = (out["pred_y"].tail(100) - (1.0 + 2.0 * df["x"].tail(100))).abs().max()
    if not isinstance(worst, float) or worst > 1e-4:
        raise SystemExit(f"the fit misses y = 1 + 2x by {worst}")

    first = rp.ModelBank([spec])
    first.fit_predict(df.head(120))
    resumed = rp.ModelBank.load_bytes(first.save_bytes(), specs=[spec])
    if not resumed.fit_predict(df.tail(80)).equals(whole.tail(80)):
        raise SystemExit("a saved and resumed state parts from the unbroken run")

    streamed = df.lazy().online.fit_predict([spec]).collect()
    if not streamed.equals(whole):
        raise SystemExit("the streaming plan parts from ModelBank.fit_predict")

    return f"polars-online {rp.__version__} on polars {pl.__version__}, from {where.parent}: ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="the version the wheel must be")
    parser.add_argument(
        "--distribution",
        choices=DISTRIBUTIONS,
        default=DISTRIBUTIONS[0],
        help="the project the wheel belongs to (default: polars-online)",
    )
    parser.add_argument(
        "--features",
        default="",
        help="the CPU features the build must have recorded, as +a,+b; empty for none",
    )
    args = parser.parse_args()
    print(check(args.version, True, args.distribution, args.features), flush=True)


if __name__ == "__main__":
    sys.exit(main())
