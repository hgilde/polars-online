"""What inspecting a plan costs `fit(lf)` (docs/PERFORMANCE.md §21): the
deprecated JSON path -- serialize, `json.loads` and the walk that looks for
an order hazard -- and `explain`, the filter that now skips the JSON on a
plan with no hazard, for four plans.

    uv run python scripts/plan_inspection_bench.py

Each figure is the minimum over `REPS` calls, so the parts and the total are
the same kind of number, and the parts' sum can be held against the total.
The three scans read small parquet files, whose schema stays out of the
JSON; the wide plan is a one-row frame of 200 columns, whose schema is in
it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import polars as pl

from polars_online import _frame

REPS = 2000


def best(fn: Callable[[], object]) -> float:
    """The fastest of `REPS` calls, in milliseconds."""
    b = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e3


def row(name: str, lf: pl.LazyFrame) -> str:
    """One plan's table row."""
    text = lf.serialize(format="json")
    tree = json.loads(text)
    s = best(lambda: lf.serialize(format="json"))
    lo = best(lambda: json.loads(text))
    w = best(lambda: _frame._walk(tree, []))
    tot = best(lambda: _frame._walk(json.loads(lf.serialize(format="json")), []))
    ex = best(lf.explain)
    return (
        f"| {name} | {len(text):,} | {s:.3f} ms | {lo:.3f} | {w:.3f} | {s + lo + w:.3f} |"
        f" {tot:.3f} ms | {ex:.3f} ms |"
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    d = Path(tempfile.mkdtemp())
    pl.DataFrame({"k": [1, 2], "x": [0.5, 1.5], "y": [1.0, 2.0]}).write_parquet(d / "a.parquet")
    pl.DataFrame({"k": [1, 2], "z": [3.0, 4.0]}).write_parquet(d / "b.parquet")
    scan = pl.scan_parquet(d / "a.parquet")
    plans = {
        "trivial scan": scan,
        "with a join": scan.join(pl.scan_parquet(d / "b.parquet"), on="k"),
        "50 `with_columns`": scan.with_columns(
            [(pl.col("x") * i).alias(f"w{i}") for i in range(50)]
        ),
        "200-column schema": pl.DataFrame({f"c{i}": [0.0] for i in range(200)}).lazy(),
    }
    print("| plan | JSON bytes | serialize | loads | walk | sum of parts | total | explain |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, lf in plans.items():
        print(row(name, lf))
    print(
        f"\nminimum of {REPS} calls each; polars {pl.__version__}; Python {sys.version.split()[0]}"
    )


if __name__ == "__main__":
    main()
