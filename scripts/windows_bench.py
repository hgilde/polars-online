"""A forward VWAP over a time window -- two decayed sums, one formula --
alone and as an embargoed target, against polars' `rolling` recipe
(docs/PERFORMANCE.md §32); and operators sharing one kernel against as many
kernels (docs/PLAN.md task 143).

    uv run python scripts/windows_bench.py [OUTDIR]

The input is about two rows a second (exponential gaps), three in ten of them
trades, one group: the stream docs/PLAN.md task 78 measured `rolling` on. It
is generated, seeded, into OUTDIR (default `.cache/windows_bench`, which git
ignores) and reused by later runs. Each case runs in a fresh process, parquet
in and `sink_parquet` out, and prints its wall time and peak RSS:

- `with_windows`: `lf.online.with_windows(fwd_vwap=rewm_sum(p q) / rewm_sum(q))`;
- `rolling`: `rolling(period=H, offset="0s", closed="none")`, with each
  window's weights taken from its own anchor;
- `embargoed`: the window as an `ewridge` target learned `H` after its row
  (`embargo=H`, `like=spec`), in one query;
- `target`: the same, with the expression in the spec's `targets` and the
  bank's own window core resolving it (docs/PLAN.md task 104);
- `model alone`: the same spec over the window's column, already written;
- `scan and sink`: the file read and written, the floor under all of them;
- `shared k`: `k` operators on one kernel (one queue, `k` values per row),
  against `separate k`, `k` operators on `k` kernels (`k` queues).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CASES = ("with_windows", "rolling", "embargoed", "target", "model alone", "scan and sink")
SHARED = ("shared 1", "shared 4", "shared 16", "separate 4", "separate 16")
RUNS = [(1_000_000, "1m"), (4_000_000, "1m"), (16_000_000, "1m"), (4_000_000, "4m")]


def make(path: Path, n: int) -> None:
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    t = np.cumsum(rng.exponential(0.5, n))
    trade = rng.random(n) < 0.3
    mid = 100 + np.cumsum(rng.normal(size=n)) * 0.01
    pl.DataFrame(
        {
            "ts": (t * 1e9).astype(np.int64),
            "mid": mid,
            "price": np.where(trade, mid + rng.normal(size=n) * 0.02, np.nan),
            "quantity": np.where(trade, rng.integers(1, 10, n), 0).astype(float),
            "signal_a": rng.normal(size=n),
            "signal_b": rng.normal(size=n),
        }
    ).with_columns(
        pl.col("ts").cast(pl.Datetime("ns")),
        pl.col("price").fill_nan(None),
    ).write_parquet(path)


def one(out: Path, case: str, n: int, h: str) -> None:
    """Run one case in this process and print its line."""
    import polars as pl

    import polars_online as po

    lf = pl.scan_parquet(out / f"ticks_{n}.parquet")
    spec = po.spec.ewridge(
        "fwd",
        targets=["fwd_vwap"],
        features=["signal_a", "signal_b"],
        clock="ts",
        gap_cap="5m",
        half_life="30m",
        embargo=h,
    )
    notional = pl.col("price") * pl.col("quantity")
    fwd_vwap = po.rewm_sum(notional, half_life="10s", window_size=h) / po.rewm_sum(
        "quantity", half_life="10s", window_size=h
    )
    columns = out / f"windows_{n}_{h}.parquet"
    start = time.perf_counter()
    if case == "with_windows":
        plan = lf.online.with_windows(fwd_vwap=fwd_vwap, clock="ts", gap_cap="5m")
        target = columns
    elif case.startswith(("shared", "separate")):
        kind, k = case.split()
        ops = {
            f"m{i}": po.ewm_mean(
                pl.col("mid") * (i + 1),
                half_life="10s" if kind == "shared" else f"{10 + i}s",
                window_size=h,
            )
            for i in range(int(k))
        }
        plan = lf.online.with_windows(**ops, clock="ts", gap_cap="5m")
        target = out / "ops.parquet"
    elif case == "embargoed":
        plan = lf.online.with_windows(fwd_vwap=fwd_vwap, like=spec).online.fit_predict([spec])
        target = out / "embargoed.parquet"
    elif case == "target":
        native = po.spec.ewridge(
            "fwd",
            targets=[fwd_vwap.alias("fwd_vwap")],
            features=["signal_a", "signal_b"],
            clock="ts",
            gap_cap="5m",
            half_life="30m",
            embargo=h,
        )
        plan = lf.online.fit_predict([native])
        target = out / "target.parquet"
    elif case == "model alone":
        plan = pl.scan_parquet(columns).online.fit_predict([spec])
        target = out / "model.parquet"
    elif case == "rolling":
        lam = 2.0 ** (-1.0 / 10.0)
        w = pl.col("quantity") * pl.col("price").is_not_null().cast(pl.Float64)
        age = (pl.col("ts") - pl.col("ts").min()).dt.total_nanoseconds().cast(pl.Float64) / 1e9
        f = pl.lit(lam).pow(age)
        plan = lf.rolling(index_column="ts", period=h, offset="0s", closed="none").agg(
            ((w * f * pl.col("price").fill_null(0.0)).sum() / (w * f).sum()).alias("fwd_vwap")
        )
        target = out / "rolling.parquet"
    else:
        plan = lf
        target = out / "copy.parquet"
    plan.sink_parquet(target)
    secs = time.perf_counter() - start
    # `resource` is POSIX-only; Windows reports no peak here (task 160, SC4).
    try:
        import resource
    except ImportError:
        gb = float("nan")
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gb = peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9
    print(f"{case:14} rows {n:>11,}  H {h:>3}  {secs:7.2f} s  peak RSS {gb:5.2f} GB", flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
        return
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".cache/windows_bench")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_header import header

    print(header(), flush=True)
    for n in sorted({n for n, _ in RUNS}):
        path = out / f"ticks_{n}.parquet"
        if not path.exists():
            make(path, n)
    for n, h in RUNS:
        for case in CASES:
            if case == "scan and sink" and h != "1m":
                continue
            subprocess.run(
                [sys.executable, __file__, "--one", str(out), case, str(n), h], check=True
            )
    for case in SHARED:
        subprocess.run(
            [sys.executable, __file__, "--one", str(out), case, str(16_000_000), "1m"], check=True
        )
    from bench_header import load

    print(f"load at the end: {load()}", flush=True)


if __name__ == "__main__":
    main()
