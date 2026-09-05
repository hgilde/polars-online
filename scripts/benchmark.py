"""Throughput benchmark (docs/PLAN.md section 9, item 8 - not a test).

    uv run python scripts/benchmark.py [--rows N] [--markdown]

Reports rows/sec for k in {5, 20, 50}, 1 vs 10 targets, and 1 vs 5 halflives,
plus a model comparison at a fixed size, then the same for the families that
are not regressions (moments, classification, clustering, sequential tests)
and the options that add a second pass (conformal, constraints, reversion).
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


def make_labelled(rows: int, k: int, classes: int = 3, seed: int = 0) -> pl.DataFrame:
    """`make` plus a class label and four blob features `b0..b3` (each `x_j`
    shifted by 6 for the rows of class `j`): what `ew_class`, `kmeans` and
    `micro` are for."""
    rng = np.random.default_rng(seed)
    df = make(rows, k, 1, seed)
    lab = rng.integers(0, classes, rows)
    cols = [pl.Series("label", lab.astype(str))]
    for j in range(min(k, 4)):
        cols.append(pl.Series(f"b{j}", df[f"x{j}"].to_numpy() + 6.0 * (lab == j % classes)))
    return df.with_columns(cols)


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

    # --- the other families, and the second-pass options, at k=20 ---
    df = make_labelled(rows, 20)
    no_target = {k: v for k, v in common.items() if k != "targets"}
    four = dict(no_target, features=[f"b{j}" for j in range(4)])
    classes = ["0", "1", "2"]
    others = [
        (
            "ew_ridge + conformal",
            "k=20, 90% interval",
            po.spec.ewridge("m", conformal=0.9, **common),
        ),
        ("sgd", "k=20, squared loss", po.spec.sgd("m", learning_rate=0.01, **common)),
        (
            "sgd, simplex",
            "k=20, coef >= 0, sum 1",
            po.spec.sgd("m", learning_rate=0.01, coef_min=0.0, coef_sum=1.0, **common),
        ),
        ("pa", "k=20", po.spec.pa("m", **common)),
        (
            "kalman + revert",
            "k=20, revert_halflife",
            po.spec.kalman("m", coef_halflife=2000.0, revert_halflife=5000.0, **common),
        ),
        ("ew_cov", "k=20: mean, std, corr (230 statistics)", po.spec.ew_cov("m", **no_target)),
        (
            "ew_cov, mahal",
            "k=20: mean, mahal, mahal_q0.99",
            po.spec.ew_cov(
                "m",
                stats=["mean", "mahal"],
                precision_prior=1.0,
                mahal_quantiles=[0.99],
                **no_target,
            ),
        ),
        (
            "ew_class, full",
            "k=20, 3 classes (QDA)",
            po.spec.ew_class("m", label="label", classes=classes, precision_prior=1.0, **no_target),
        ),
        (
            "ew_class, shared",
            "k=20, 3 classes (LDA)",
            po.spec.ew_class(
                "m",
                label="label",
                classes=classes,
                covariance="shared",
                precision_prior=1.0,
                **no_target,
            ),
        ),
        (
            "ew_class, diagonal",
            "k=20, 3 classes (naive Bayes)",
            po.spec.ew_class(
                "m",
                label="label",
                classes=classes,
                covariance="diagonal",
                precision_prior=1.0,
                **no_target,
            ),
        ),
        ("kmeans", "4 features, K=8", po.spec.kmeans("m", k=8, **four)),
        ("kmeans", "k=20, K=8", po.spec.kmeans("m", k=8, **no_target)),
        ("micro", "4 features, eps=1", po.spec.micro("m", eps=1.0, **four)),
        (
            "seqtest",
            "sign of one column",
            po.spec.seqtest("m", targets=["y0"], clock="t", max_dclock=100.0),
        ),
    ]
    other_results: list[tuple[str, str, float]] = []
    for name, note, spec in others:
        other_results.append((name, note, time_spec(df, spec)))

    if args.markdown:
        print(f"Rows per run: {rows:,}. Best of 3, single process.")
        for table in (results, other_results):
            print()
            print("| configuration | notes | rows/sec |")
            print("|---|---|---|")
            for name, note, rps in table:
                print(f"| {name} | {note} | {rps:,.0f} |")
    else:
        for name, note, rps in results + other_results:
            print(f"{name:32s} {note:40s} {rps:12,.0f} rows/sec")


if __name__ == "__main__":
    main()
