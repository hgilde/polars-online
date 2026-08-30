"""Throughput benchmark (docs/PLAN.md section 9, item 8 - not a test).

    uv run python scripts/benchmark.py [--rows N] [--markdown]

Reports rows/sec for k in {5, 20, 50}, 1 vs 10 targets, and 1 vs 5 halflives,
plus a model comparison at a fixed size.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import polars_online as po  # noqa: E402


def make(rows: int, k: int, m: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"x{j}": rng.standard_normal(rows) for j in range(k)}
    beta = rng.standard_normal((k, m))
    x = np.column_stack(list(data.values()))
    for j in range(m):
        data[f"y{j}"] = x @ beta[:, j] + 0.5 * rng.standard_normal(rows)
    data["t"] = np.cumsum(rng.exponential(1.0, rows))
    return pl.DataFrame(data)


def time_spec(df: pl.DataFrame, spec: dict, repeats: int = 3) -> float:
    """Best-of-N rows/sec (best, not mean: it is the least noisy estimate)."""
    best = float("inf")
    for _ in range(repeats):
        bank = po.ModelBank([spec])
        t0 = time.perf_counter()
        bank.fit_predict(df)
        best = min(best, time.perf_counter() - t0)
    return df.height / best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    rows = args.rows

    results: list[tuple[str, str, float]] = []

    # --- k sweep, single target, single halflife ---
    for k in (5, 20, 50):
        df = make(rows, k, 1)
        feats = [f"x{j}" for j in range(k)]
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=feats,
            clock="t",
            halflife=1000.0,
            max_dclock=100.0,
            min_periods=float(k),
        )
        results.append((f"ew_ridge k={k}", "1 target, 1 halflife", time_spec(df, spec)))

    # --- targets sweep at k=20 ---
    for m in (1, 10):
        df = make(rows, 20, m)
        feats = [f"x{j}" for j in range(20)]
        spec = po.spec.ewridge(
            "m",
            targets=[f"y{j}" for j in range(m)],
            features=feats,
            clock="t",
            halflife=1000.0,
            max_dclock=100.0,
            min_periods=20.0,
        )
        results.append((f"ew_ridge k=20, {m} target(s)", "1 halflife", time_spec(df, spec)))

    # --- halflife grid at k=20 (one accumulator per halflife) ---
    for n_hl in (1, 5):
        df = make(rows, 20, 1)
        feats = [f"x{j}" for j in range(20)]
        hl = [500.0 * (i + 1) for i in range(n_hl)]
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=feats,
            clock="t",
            halflife=hl[0] if n_hl == 1 else hl,
            max_dclock=100.0,
            min_periods=20.0,
        )
        results.append((f"ew_ridge k=20, {n_hl} halflife(s)", "1 target", time_spec(df, spec)))

    # --- model comparison at k=20 ---
    df = make(rows, 20, 1)
    feats = [f"x{j}" for j in range(20)]
    common = dict(
        targets=["y0"],
        features=feats,
        clock="t",
        halflife=1000.0,
        max_dclock=100.0,
        min_periods=20.0,
    )
    models = {
        "ew_ridge": po.spec.ewridge("m", **common),
        "rls": po.spec.rls("m", ridge=1.0, **common),
        "kalman": po.spec.kalman("m", coef_halflife=2000.0, **common),
        "lasso": po.spec.lasso("m", lasso_path=[0.1, 0.01, 0.0], **common),
        "huber": po.spec.huber("m", **common),
        "ftrl": po.spec.ftrl("m", **common),
    }
    for name, spec in models.items():
        results.append((name, "k=20, 1 target", time_spec(df, spec)))

    if args.markdown:
        print(f"Rows per run: {rows:,}. Best of 3, single process.")
        print()
        print("| configuration | notes | rows/sec |")
        print("|---|---|---|")
        for name, note, rps in results:
            print(f"| {name} | {note} | {rps:,.0f} |")
    else:
        for name, note, rps in results:
            print(f"{name:32s} {note:22s} {rps:12,.0f} rows/sec")


if __name__ == "__main__":
    main()
