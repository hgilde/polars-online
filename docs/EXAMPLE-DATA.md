<!-- kind: reference -->
# Example data

This page builds the made-up frames that the examples on the how-to,
explanation and reference pages read, and the docstrings of the API
reference too. A page whose example reads one of them says so in the line
above the example, with a link here. Each frame exists for one kind of
example. `df` holds a planted answer, so a fit's coefficients, a break or a
change of regime can be read against the truth. `today` is the 400 rows
that follow `df` on its clock, so an example can save a bank on `df` and
then serve or resume it on rows it has not learned. `trades` is a stream of
quotes with trades between them on a datetime clock, which the examples
of the window operators read. Each is also saved to a parquet file,
so an example that runs a query scans it from disk. The README's first fit
and the tutorials build their own data and need none of this.

Most examples on the how-to, explanation and reference pages read one of
two made-up frames: `df`, 400 rows of numbers and text keys a minute apart,
and `trades`, a stream of quotes with trades between them. The examples
that serve or resume a bank also read `today`, the 400 rows that follow
`df`'s on the clock `t`. Before running such an example, build all three
with this code. It saves each to a parquet file, and an
example that runs a query reads the file back with `pl.scan_parquet`, as a
stream too large to hold would be read:

```python
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars_online as po

# df: 400 rows of numbers and text keys, a minute apart.
rng = np.random.default_rng(0)
x0, x1, x2, signal_a, signal_b, noise_y, noise_ret = rng.standard_normal((7, 400))
row = np.arange(400)
df = pl.DataFrame({
    "t": np.arange(400.0),                                       # a numeric clock: the row's number
    "ts": pl.datetime_range(datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 16, 9), "1m", eager=True),   # a minute apart
    "x0": x0, "x1": x1, "x2": x2, "signal_a": signal_a, "signal_b": signal_b,
    "y": 1.0 + 2.0 * x0 - 1.0 * x1 + 0.5 * noise_y,              # the truth: intercept 1, x0 2, x1 -1; x2 plays no part
    "ret": np.where(row < 200, 0.8, -0.8) * signal_a             # a break: the beta on signal_a flips at row 200
    + np.where(row // 100 % 2 == 1, 2.0, 0.5) * noise_ret,       # two regimes: calm (0.5), volatile (2.0) on rows 100-199 and 300-399
    "stock_id": [f"b{i % 4}" for i in range(400)],               # four stocks, taking turns row by row
    "group": [f"g{i % 3}" for i in range(400)],
    "session": ["m"] * 200 + ["a"] * 200,                        # a morning session, then an afternoon one
    "venue": ["X", "Y"] * 200,
})
# Save df, and the rows that follow it, as the files the examples read as a query.
df.write_parquet("ticks.parquet")                                # df on disk, as one file
lf = pl.scan_parquet("ticks.parquet")                            # the same rows, read back from the file as a query
today = df.with_columns(pl.col("t") + 400.0)                     # the 400 rows that follow df's, on the clock t
today.write_parquet("today.parquet")
later = pl.scan_parquet("today.parquet")                         # today, read back as a query
Path("ticks").mkdir(exist_ok=True)                               # and df as two files, for the examples that read ticks/*.parquet
df.head(200).write_parquet("ticks/part-0.parquet")
df.tail(200).write_parquet("ticks/part-1.parquet")

# trades: quotes, with trades between them, about two rows a second.
rng = np.random.default_rng(7)
is_trade = rng.random(3000) < 0.3                                # three rows in ten are trades, the rest quotes
micros = np.cumsum(rng.exponential(5e5, 3000)).astype(int)       # about two rows a second
mid = 100 + np.cumsum(rng.normal(0.0, 0.01, 3000))               # the quotes' mid, on every row
trades = pl.DataFrame({
    "ts": [datetime(2024, 1, 2, 9, 30) + timedelta(microseconds=int(u)) for u in micros],
    "symbol": rng.choice(["AAA", "BBB"], 3000),
    "mid": mid,
    "side": rng.choice(["buy", "sell"], 3000),
    "quantity": rng.integers(1, 10, 3000).astype(float),
    "price": mid + rng.normal(0.0, 0.02, 3000),
}).with_columns(
    pl.when(pl.Series(is_trade)).then(pl.col("side", "quantity", "price"))   # null on a quote
)
trades.write_parquet("trades.parquet")                           # on disk, for the examples that read it
```

`df` holds a known answer, so an example's output can be read against it.
`y` is `1 + 2·x0 − x1` plus noise of spread 0.5, and `x2`, `signal_a` and
`signal_b` play no part in it. `ret`'s beta on `signal_a` is 0.8 for the
morning session's 200 rows and −0.8 for the afternoon's: a break. Its noise
is calm, of spread 0.5, on rows 0 to 99 and 200 to 299, and volatile, of
spread 2, on the other two hundred: two regimes. `trades` holds no planted
relation: its mid is a random walk.
