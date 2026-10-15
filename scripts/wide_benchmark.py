"""Throughput of wide specs, the ones that reach the matrix products (docs/PLAN.md task 265).

    uv run python scripts/wide_benchmark.py

`scripts/benchmark.py` stops at 50 features, below the width where a solve,
a factorization or the blocked Gram's product dominates a row, so a change
there does not move its numbers: a once-per-run solve against the identity
cost a 1,000-feature `ewridge` 44% of its speed and left every
`benchmark.py` row where it was. Rows per second, best of two fits, for
`ewridge` at 200, 500 and 1,000 features on three solve schedules (and every
row at 200), with and without `gram_block_rows`, and `lasso` at 200 and 500.
Run it under the last release and under this build, in turn, before a
release (docs/RELEASE-READINESS.md, step 2).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import polars_online as po


def make(rows: int, k: int, m: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, k))
    beta = rng.standard_normal((k, m)) / np.sqrt(k)
    y = x @ beta + 0.5 * rng.standard_normal((rows, m))
    data = {f"x{j}": x[:, j] for j in range(k)}
    data |= {f"y{j}": y[:, j] for j in range(m)}
    data["t"] = np.arange(rows, dtype=np.float64)
    return pl.DataFrame(data)


def rate(df: pl.DataFrame, spec: dict, repeats: int = 2) -> float:
    best = float("inf")
    for _ in range(repeats):
        bank = po.ModelBank([spec])
        t0 = time.perf_counter()
        bank.fit_predict(df)
        best = min(best, time.perf_counter() - t0)
    return df.height / best


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header

    print(header(), flush=True)
    for k, rows in {200: 20_000, 500: 5_000, 1000: 2_000}.items():
        df = make(rows, k, 1)
        feats = [f"x{j}" for j in range(k)]
        common = dict(targets=["y0"], features=feats, clock="t", gap_cap=100.0, min_weight=float(k))
        configs = {
            "ewridge, default schedule (solve ~every 20 rows)": po.spec.ewridge(
                "m", half_life=1000.0, **common
            ),
            "ewridge, solve every 256 rows": po.spec.ewridge(
                "m", half_life=1000.0, solve_every=256.0, **common
            ),
            "ewridge, solve every 256 rows, gram_block_rows 256": po.spec.ewridge(
                "m", half_life=1000.0, solve_every=256.0, gram_block_rows=256, **common
            ),
        }
        if k == 200:
            configs["ewridge, solve every row"] = po.spec.ewridge(
                "m", half_life=1000.0, solve_every=0.0, **common
            )
        for name, spec in configs.items():
            print(f"k={k:5d} | {name} | {rate(df, spec):,.0f} rows/s", flush=True)
        if k in (200, 500):
            # A shorter stream: lasso at these widths is a few hundred rows a second.
            lrows = 4 * k
            spec = po.spec.lasso("m", half_life=1000.0, penalties=[0.1, 0.01, 0.0], **common)
            print(
                f"k={k:5d} | lasso, 3 penalties ({lrows} rows) | "
                f"{rate(df.head(lrows), spec):,.0f} rows/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
