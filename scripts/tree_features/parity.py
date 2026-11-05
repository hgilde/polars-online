"""v3, review findings 1 and 14d: train/serve parity of a feature through a
saved state, as a recipe. Flights' keyless nowcast, the decayed mean of
every arrival delay known before the row (labels at their arrival: the
doubled stream), computed over January-September with save_state=, then
over October-December with load_state=, against one run over the year."""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import polars_online as po  # noqa: E402

df = C.load_flights().with_row_index("rid")
rows = pl.concat(
    [
        df.select(
            "rid",
            c=pl.col("t"),
            y=pl.lit(None, pl.Float64),
            one=pl.lit(None, pl.Float64),
            role=pl.lit(0, pl.Int8),
        ),
        df.select(
            "rid", c=pl.col("avail"), y=pl.col("y"), one=pl.lit(1.0), role=pl.lit(1, pl.Int8)
        ),
    ]
).sort("c", "role", "rid")
nowcast = dict(
    S=po.ewm_sum("y", half_life=60.0, closed="left"),
    W=po.ewm_sum("one", half_life=60.0, closed="left"),
)
cut = 273 * 1440.0
with tempfile.TemporaryDirectory() as d:
    a, b, full = Path(d) / "train.parquet", Path(d) / "serve.parquet", Path(d) / "all.parquet"
    rows.filter(pl.col("c") < cut).write_parquet(a)
    rows.filter(pl.col("c") >= cut).write_parquet(b)
    rows.write_parquet(full)
    state = str(Path(d) / "nowcast.state")
    t0 = time.perf_counter()
    first = (
        pl.scan_parquet(a)
        .online.with_windows(**nowcast, clock="c", gap_cap=1e12, save_state=state)
        .collect()
    )
    second = (
        pl.scan_parquet(b)
        .online.with_windows(**nowcast, clock="c", gap_cap=1e12, load_state=state)
        .collect()
    )
    el = time.perf_counter() - t0
    one = pl.scan_parquet(full).online.with_windows(**nowcast, clock="c", gap_cap=1e12).collect()
two = pl.concat([first, second])


def f(fr):
    return (
        fr.filter(pl.col("role") == 0)
        .sort("rid")
        .select(n=pl.col("S") / pl.col("W"))["n"]
        .to_numpy()
    )


x, z = f(two), f(one)
same = np.array_equal(np.isnan(x), np.isnan(z)) and np.array_equal(x[~np.isnan(x)], z[~np.isnan(z)])
print(f"two runs through a saved state: {two.height} rows in {el:.2f}s; equal to one run: {same}")
