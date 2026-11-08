<!-- kind: how-to -->
# Detect a break

This how-to is for anyone who fits a model on a stream and wants to know
whether the relationship it fits has broken, how, and when. It tests a
regression's residuals and coefficients for a
[break](../GLOSSARY.md#fitting-a-model) as the stream runs, and dates a
break with a test run once over the whole stream. Watching the features
themselves is a separate task: `corrchange` for their correlations, and
`bocpd` for the age of their regime.

## Prerequisites

These steps need the library and a model to watch:

- polars-online installed, and numpy to make the examples' data.
- For a break in a regression: a
  [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of one of the ten
  linear models, such as `ewridge`, which take `emit_breaks` and
  `emit_drift`. [What each costs, and which models take
  it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it) lists
  them.
- For a break in the features alone: the columns to watch, for a
  `corrchange` or a `bocpd` spec.

## Test a fit that forgets for a break

A relationship can break in three ways: its intercept shifts, its noise
grows, or a slope moves. Each break test sees some of them, so turn them
on together and read which ones fire. The CUSUMs and `drift` read the
model's [out-of-sample](../explanation/out-of-sample.md) residuals, and
`break_wald` compares two fits of the target on the features. Each runs at
a memory of its own, the fit's half-life by default ([One memory per diagnostic: run
once or
windowed](../DIAGNOSTICS.md#one-memory-per-diagnostic-run-once-or-windowed)).

1. **Turn on `emit_breaks=True` and `emit_drift=True` in the spec.** Each
   row gains `studentized_<t>`, `cusum_<t>`, `cusum_sq_<t>`,
   `break_wald_<t>` and `drift_<t>` for each target `<t>`. [Has it
   broken?](../DIAGNOSTICS.md#has-it-broken) defines each.
2. **Run the query.**
3. **Flag each row against its threshold:** `|cusum| > 3`,
   `|cusum_sq| > 3`, `break_wald` past `chi2(k + 1)`'s 0.01% value
   for `k` features (21.1 for the recipe's two), and `drift` true. On a stream with no break
   the first three hold on about 0.28%, 0.28% and 0.011% of rows beside a
   fit at a half-life of 200.
4. **Read the first row each flag fires, and which flags fire together,**
   by the table in [Which flags fire on each
   break](#which-flags-fire-on-each-break).

This code makes three streams that break at row 3,000, each in one way,
and prints each flag's first row after the break and its count on the
stable rows before it:

```python
import numpy as np
import polars as pl

import polars_online as po

# Three streams that break at row 3,000 of 6,000, each in one way.
rng = np.random.default_rng(3)
n, at = 6_000, 3_000
x0, x1, e = rng.standard_normal((3, n))
after = np.arange(n) >= at
streams = {
    "mean": 1.0 * x0 + 0.5 * x1 + e + 0.5 * after,  # the intercept rises by half a noise sd
    "variance": 1.0 * x0 + 0.5 * x1 + e * np.where(after, 2.0, 1.0),  # the noise doubles
    "slope": 1.0 * x0 + np.where(after, -0.5, 0.5) * x1 + e,  # x1's slope flips sign
}
pl.DataFrame({"x0": x0, "x1": x1} | {f"y_{k}": v for k, v in streams.items()}).write_parquet("breaks.parquet")
broken = pl.scan_parquet("breaks.parquet").with_row_index("row")

# 1 and 2. One ridge per stream, with the windowed break tests and drift.
specs = [
    po.spec.ewridge(
        kind, targets=[f"y_{kind}"], features=["x0", "x1"], half_life=200.0,
        emit_breaks=True,  # memory: the model's 200 rows
        emit_drift=True,  # at drift_delta=0.5 and drift_threshold=20
    )
    for kind in streams
]
out = broken.online.fit_predict(specs).collect()

# 3. The four flags, each a test on one stream's fields.
flags = {
    "|cusum| > 3": lambda y: pl.col(f"cusum_{y}").abs() > 3,
    "|cusum_sq| > 3": lambda y: pl.col(f"cusum_sq_{y}").abs() > 3,
    "break_wald > 21.1": lambda y: pl.col(f"break_wald_{y}") > 21.1,
    "drift": lambda y: pl.col(f"drift_{y}"),
}
print(f"{'break':9}" + "".join(f"{name:>19}" for name in flags))
flagged_before = 0
for kind in streams:
    rows = out.select("row", pl.col(kind).struct.unnest())
    cells = []
    # 4. Each flag's first row after the break, and its count on the stable rows before it.
    for flag in flags.values():
        first = rows.filter((pl.col("row") >= at) & flag(f"y_{kind}"))["row"].min()
        cells.append("-" if first is None else str(first))
        flagged_before += rows.filter(pl.col("row").is_between(1_000, at - 1) & flag(f"y_{kind}")).height
    print(f"{kind:9}" + "".join(f"{c:>19}" for c in cells))
print(f"flags on rows 1,000-2,999, before any break: {flagged_before}")
```

```text
break            |cusum| > 3     |cusum_sq| > 3  break_wald > 21.1              drift
mean                    3063                  -               3135                  -
variance                   -               3024               3313                  -
slope                      -               3087               3072                  -
flags on rows 1,000-2,999, before any break: 15
```

Each cell is the first row the flag fired after the break. Before row
3,000 the flags fired on 15 of the 24,000 flag-rows (four flags on 2,000
rows of each of the three streams), at or below the rates step 3 gives
for a stream with no break. `drift` fired on none of the breaks.

### Which flags fire on each break

Leaving `drift` aside, the three flags tell the breaks apart by which fire,
and by which fires first:

| break | `cusum` | `cusum_sq` | `break_wald` |
|---|---|---|---|
| the intercept shifted | fires | quiet, or late | fires, since the intercept is a coefficient |
| the noise grew | quiet, or late | fires first, within 20 rows on the median | quiet, or late |
| a slope moved | quiet, or late | fires, while the fit catches up | fires |

How late *late* is for the noise and slope breaks, measured over 200
streams of each, is in [Has it broken?](../DIAGNOSTICS.md#has-it-broken). Beside a gradient fit (`sgd`,
`pa`, `ftrl`) or `holt`, the CUSUM finds fewer breaks, and `break_wald`,
which reads the target and the features alone, finds them beside every
model with features alike. Beside `holt`, which has only a level to
compare, it found 20% of them at a memory of 50.

## Date a break with a test run once

A fit that does not forget (`half_life=float("inf")`) runs each break test
once over the whole stream. There the CUSUM is Brown, Durbin and Evans'
test, whose path crosses a boundary where the residuals' mean has moved.

1. **Fit the spec with `half_life=float("inf")`, `emit_breaks=True` and a
   `min_weight`.** The `min_weight` keeps the fit's first, wildest
   predictions out of a test that sums every row.
2. **Sum `studentized_<t>` over the rows into the CUSUM path,** and count
   the rows it sums, `r`.
3. **Compare the path with the 5% boundary
   `±0.948 (sqrt(T) + 2r / sqrt(T))`,** where `T` is the length of the run
   being tested. A monitor fixes `T` in advance, as the horizon it will
   watch.
4. **Take the first row past the boundary as the detection, and refit from
   it.** A crossing comes after the break, by a delay that shrinks as the
   break grows, so every row from the crossing on belongs to the new
   regime.

This code runs the CUSUM once over a stream whose intercept rises at row
2,000, and refits from where it crossed:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 0.2 + 1.0 x0 + 0.5 x1 + noise until row 2,000 of 4,000, where the intercept rises to 0.7.
rng = np.random.default_rng(2)
n, at = 4_000, 2_000
x0, x1 = rng.standard_normal((2, n))
t = np.arange(n, dtype=float)
y = 0.2 + 0.5 * (t >= at) + 1.0 * x0 + 0.5 * x1 + rng.standard_normal(n)
pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y}).write_parquet("shift.parquet")
shift = pl.scan_parquet("shift.parquet")

# 1. Run once: half_life=inf weighs every row alike, and the CUSUM runs over the whole stream.
model = po.spec.ewridge(
    "ols", targets=["y"], features=["x0", "x1"], half_life=float("inf"), min_weight=30.0,
    emit_breaks=True,
)
out = shift.online.fit_predict([model]).online.unnest([model]).collect()

# 2 and 3. Brown, Durbin and Evans' path, the running sum of the studentized residuals, against
# the 5% boundary for a run of T of them.
path = out.select(
    "t",
    path=pl.col("studentized_y").fill_null(0.0).cum_sum(),
    r=pl.col("studentized_y").is_not_null().cum_sum(),
)
T = path["r"].max()
path = path.with_columns(bound=0.948 * (T**0.5 + 2 * pl.col("r") / T**0.5))
# The first row past the boundary, and the path at a few rows around the break.
crossed = path.filter(pl.col("path").abs() > pl.col("bound"))["t"].min()
print(f"the path crossed its boundary at row {crossed:.0f}")
for row, value, bound in path.filter(pl.col("t").is_in([1_000.0, 2_000.0, 2_200.0, crossed])).select("t", "path", "bound").iter_rows():
    print(f"  row {row:>5.0f}: path {value:>6.1f}, boundary {bound:.1f}")


# 4. Refit over all of a query's rows: the whole stream, or from the crossing on.
def coef(query):
    """The coefficients of a fit over every row of the query."""
    bank = po.ModelBank([po.spec.ewridge("ols", targets=["y"], features=["x0", "x1"], half_life=float("inf"))])
    bank.fit(query)
    return ", ".join(f"{term} {c:.2f}" for term, c in bank.coef().select("term", "coef").iter_rows())


print(f"every row:              {coef(shift)}")
print(f"from the crossing on:   {coef(shift.filter(pl.col('t') >= crossed))}")
```

```text
the path crossed its boundary at row 2310
  row  1000: path   30.3, boundary 88.6
  row  2000: path   -9.1, boundary 118.7
  row  2200: path   69.8, boundary 124.8
  row  2310: path  128.4, boundary 128.1
every row:              intercept 0.44, x0 1.02, x1 0.50
from the crossing on:   intercept 0.69, x0 0.98, x1 0.48
```

The path reads 30 at row 1,000 and −9 at row 2,000, the wandering of a
sum of independent residuals. Then it climbs, to 70 by row 2,200, because
each new row's residual carries part of the half-unit shift the fit has
not yet absorbed. It crosses the boundary at row 2,310, 310 rows after the
break. The fit over every row mixes the two regimes and puts the
intercept at 0.44, between 0.2 and 0.7. Refitted from the crossing, the
fit reads 0.69, the new regime's intercept: refitting there loses rows
but mixes nothing.

A slope that moves on a feature centered at zero leaves the residuals'
mean at zero, so the CUSUM cannot see it. Read `break_wald` for it on a
fit that forgets, as in [Test a fit that forgets for a
break](#test-a-fit-that-forgets-for-a-break): run once, it is null.

## Act on a break

Choose what to do by which flags fire:

| what fires | what it means | what to do |
|---|---|---|
| `cusum`, alone or with `break_wald` | the level moved | refit from the crossing, as step 4 of [Date a break with a test run once](#date-a-break-with-a-test-run-once) does, or let a fit that forgets catch up |
| `cusum_sq` alone | the noise changed and the coefficients did not | keep the fit. The residuals' spread (`sigma`, which `emit_sigma` writes), and every interval built on it, is wrong until it adapts |
| `break_wald` with `cusum_sq`, without `cusum` | a slope moved, where `break_wald` fired first, or the noise grew, where `cusum_sq` fired first and `break_wald` many rows after (3,024 and 3,313 on the recipe's variance break) | for a slope, let the fit catch up, or refit from the flag. For the noise, keep the fit, as the row before says |
| `drift` | the residuals' level rose | `drift_action="reset"` resets the model at a `drift` flag, which suits a level shift only |

`drift`'s false alarms depend on the residuals' tails: set
`drift_threshold` from them as [Measure heavy tails](measure-heavy-tails.md)
does.

## Watch the features' correlations with `corrchange`

`corrchange` tests whether the correlations among the columns you name
have changed, with no target. Its `kind="sequential"` tests each row
against a stable history as it arrives. [`corrchange` in
MODELS.md](../MODELS.md#corrchange--has-the-correlation-structure-changed)
defines its three kinds and its statistic.

1. **Set `kind="sequential"`, `window_rows=` to the rows of history to
   hold stable, and `monitor_rows=` to the rows to test after it.** A
   cycle is the history, then up to `monitor_rows` rows, each tested as it
   arrives. A flag, or the cycle's last row, ends the cycle, and the row
   after it starts a new history. `kind="monitor"` over the history's rows
   checks that they are stable.
2. **Set `boundary_gamma` to catch an early change sooner.** Above 0 the
   boundary starts lower, so a change soon after the history is caught
   sooner, and a late one later. Near the top of the range, up to 0.49,
   it also flags more stable streams. Wied and Galeano measured these
   rates on GARCH pairs (their Table 2):

   | `boundary_gamma` | a change soon after the history | stable streams flagged, at a nominal 0.05 |
   |---|---|---|
   | 0, the default | caught latest | 0.050 to 0.077 |
   | 0.25 | caught sooner | 0.047 to 0.087 |
   | 0.45 | caught soonest | 0.106 to 0.174 |

3. **Run the query, and read the rows where `flag` is true.** Each reports
   `stat` and the `crit` it passed.

This code makes two columns that start to correlate at 0.7 at row 560, 60
rows into the first cycle's monitored rows, and prints the rows each
`boundary_gamma` flags:

```python
import numpy as np
import polars as pl

import polars_online as po

# Two columns, uncorrelated until row 560, then correlated at 0.7.
rng = np.random.default_rng(10)
n, at = 1_500, 560
rho = np.where(np.arange(n) < at, 0.0, 0.7)
a, e = rng.standard_normal((2, n))
pl.DataFrame({"a": a, "b": rho * a + np.sqrt(1 - rho**2) * e}).write_parquet("corr.parquet")
pairs = pl.scan_parquet("corr.parquet").with_row_index("row")

# 1 to 3. The sequential test at three boundaries, and the rows each flags.
for gamma in [0.0, 0.25, 0.45]:
    watch = po.spec.corrchange(
        "watch", features=["a", "b"],
        kind="sequential",
        window_rows=500,  # 500 rows of history, assumed stable ...
        monitor_rows=1000,  # ... then each of up to 1,000 rows tested as it arrives
        boundary_gamma=gamma,
    )
    out = pairs.online.fit_predict([watch]).online.unnest([watch]).collect()
    print(f"boundary_gamma {gamma}: flags at rows {out.filter(pl.col('flag'))['row'].to_list()}")
```

```text
boundary_gamma 0.0: flags at rows [620]
boundary_gamma 0.25: flags at rows [595]
boundary_gamma 0.45: flags at rows [593, 1095]
```

The change at row 560 is flagged 60 rows after it at the default, and 35
and 33 rows after it at 0.25 and 0.45. At 0.45 the next cycle, whose
history starts at row 594 and holds no change, flags again at row 1,095: a
false alarm, of the kind the table's last row counts.

**Count a run of `kind="window"`'s flags as one change.** That kind
compares the correlations of two adjacent windows of `window_rows` rows
on every row, and flags against a permutation quantile at the level
`alpha` (0.05 by default). Two windows that slide by one row are almost
the same windows, so a statistic above the quantile stays above it for a
run of rows, and no rate of `alpha` per row applies. Set
`change_action="reset"`, which empties both windows at a flag, so one
change gives one flag.

## Date a regime with `bocpd`

`bocpd` keeps a probability distribution over the run length `r`, the
rows since the last break, so it says how long the current regime has lasted.
[`bocpd` in
MODELS.md](../MODELS.md#bocpd--how-long-has-this-regime-lasted) defines
its update, and [Choose a changepoint
hazard](choose-a-changepoint-hazard.md) sets how often it expects a
break.

1. **Build a `bocpd` spec on the columns, with a `hazard`.** Give
   `group=` for one model per series.
2. **Run the query.** Each row gains `run_mode` and `p_change`.
3. **Do not date a break from `p_change`.** Under a number `hazard` it is
   `P(r ≤ 1)` given the row, and under a duration the chance that the row
   began a run. Either way it moves on one row's evidence, so a break that
   takes several rows to show barely lifts it: 0.02 at most in the 50 rows
   after the spread quadruples in the example in [`bocpd` in
   MODELS.md](../MODELS.md#bocpd--how-long-has-this-regime-lasted).
   [`po.spec.bocpd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.bocpd)'s
   Notes, under *Reading p_change and run_mode*, measure three kinds of
   break.
4. **Read the regime's age from `run_mode`, the most likely run length
   before the row, and date its start as the row number less `run_mode`.**
   With
   a `group=`, count the group's own rows, with Polars'
   `pl.int_range(pl.len()).over(<group>)`, since `run_mode` counts the
   group's rows and not the stream's.

This code makes two stocks whose rows take turns, each with a spread that
quadruples at a row of its own. At each stock's last row it dates the
current regime twice: in the stock's own rows, and wrongly, in the
stream's:

```python
import numpy as np
import polars as pl

import polars_online as po

# Two stocks whose rows take turns; each one's spread quadruples at a row of its own.
rng = np.random.default_rng(11)
n = 2_000
breaks = {"A": 600, "B": 1_300}  # the truth: each stock's break, in its own rows
frames = []
for stock, at in breaks.items():
    scale = np.where(np.arange(n) < at, 1.0, 4.0)
    frames.append(pl.DataFrame({"stock": stock, "i": np.arange(n), "ret": rng.standard_normal(n) * scale}))
pl.concat(frames).sort("i", "stock").write_parquet("regimes.parquet")

# 1 and 2. One bocpd per stock, a break expected every 500 of its rows, and the query run.
runs = po.spec.bocpd("regime", features=["ret"], group="stock", hazard=500.0)
out = pl.scan_parquet("regimes.parquet").online.fit_predict([runs]).online.unnest([runs]).collect()
# 4. The row each current run began on, in the stock's own rows and in the stream's.
out = out.with_columns(
    began=pl.int_range(pl.len()).over("stock") - pl.col("run_mode"),
    began_in_stream=pl.int_range(pl.len()) - pl.col("run_mode"),
)
for stock, at in breaks.items():
    last = out.filter(pl.col("stock") == stock).tail(1).row(0, named=True)
    print(f"stock {stock}: break at row {at}, dated to row {last['began']} (stream count: {last['began_in_stream']})")
```

```text
stock A: break at row 600, dated to row 600 (stream count: 2599)
stock B: break at row 1300, dated to row 1301 (stream count: 3301)
```

Counted in each stock's own rows, each break is dated within a row of the
truth. Counted in the stream's rows, which interleave the two stocks, the
same `run_mode` dates neither.

## Check that it worked

Run the same tests on a stretch of the stream known to be stable, and
count the flags there. They should be rare, at the rates [Has it
broken?](../DIAGNOSTICS.md#has-it-broken) gives for a stream with no
break. Where they are not:

| what you see | what it means | what to do |
|---|---|---|
| `drift` on many stable rows | the residuals have heavy tails | measure them and set `drift_threshold` from them ([Measure heavy tails](measure-heavy-tails.md)) |
| a break you planted, and no CUSUM flag beside a gradient fit | the fit absorbs a level at its learning rate, so its residuals carry little of the move | read `break_wald`, which reads the target and the features alone |
| a slope break, and no CUSUM flag | a slope on a feature centered at zero leaves the residuals' mean at zero | read `break_wald` |
| flags on a target that looks ahead, with nothing broken | the residuals overlap by construction | give the spec `horizon_rows=` (with a clock) or `embargo=` (without one), so each test allows for it ([Can its t be trusted?](../DIAGNOSTICS.md#can-its-t-be-trusted)) |
| `corrchange` flags on stable streams at a high `boundary_gamma` | the boundary starts too low | lower `boundary_gamma` toward 0 |
