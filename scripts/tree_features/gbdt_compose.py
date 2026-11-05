"""Does a bank's output query feed po.gbdt directly? A target-statistic stand-in
(an ewridge per group on a constant feature) as a LazyFrame, into
xgboost_matrix; then the same rows written to IPC first."""

import os
import tempfile
import time

import numpy as np
import polars as pl

import polars_online as po

rng = np.random.default_rng(0)
n = 50_000
d = pl.DataFrame(
    {
        "t": np.arange(n, dtype=float),
        "g": rng.integers(0, 50, n).astype(str),
        "x": rng.standard_normal(n),
    }
).with_columns(one=pl.lit(1.0))
d = d.with_columns(
    y=pl.col("x") + pl.col("g").cast(pl.Int64) / 10 + pl.Series(rng.standard_normal(n))
)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "in.parquet")
    d.write_parquet(p)
    s = po.spec.ewridge(
        "te",
        targets=["y"],
        features=["one"],
        fit_intercept=False,
        clock="t",
        gap_cap=10.0,
        half_life=5000.0,
        group="g",
        ridge=1e-6,
    )
    lf = (
        pl.scan_parquet(p)
        .online.fit_predict([s])
        .with_columns(enc=pl.col("te").struct.field("pred_y").fill_null(0.0))
        .select("x", "enc", "y")
    )
    t0 = time.perf_counter()
    try:
        m = po.gbdt.xgboost_matrix(lf, label="y", features=["x", "enc"])
        print(
            "po.gbdt over the bank's query: rows", m.num_row(), f"{time.perf_counter() - t0:.2f}s"
        )
    except Exception as e:
        print(
            "po.gbdt over the bank's query refused:", type(e).__name__, str(e).splitlines()[0][:300]
        )
