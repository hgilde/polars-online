<!-- kind: how-to -->
# Put an interval on a prediction

This how-to is for anyone who serves a model's predictions and wants an
interval beside each one that covers the target as often as asked. It
builds a Gaussian interval from the residuals' spread, an interval from a
quantile of their size, and a conformal interval that holds its coverage whatever
the residuals do, including when targets arrive late.

## Prerequisites

These steps need the library and a model whose error to measure:

- polars-online installed, and numpy to make the examples' data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of one of the
  ten linear models, such as `ewridge`, which take `emit_sigma`,
  `resid_quantiles` and `conformal`. [What each costs, and which models
  take it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it)
  lists them.

## Put an interval on each prediction

Each interval is read before its row, from the model's
[out-of-sample](../explanation/out-of-sample.md) residuals, so it can be
served with the row's prediction. [How wide is its
error?](../DIAGNOSTICS.md#how-wide-is-its-error) defines each field and
what it assumes.

1. **Turn on the intervals in the spec.** `emit_sigma=True` adds
   `sigma_<t>`, the residuals' spread. `resid_quantiles=[0.9]` adds
   `abs_resid_q0.9_<t>`, a quantile of `|resid|`. `conformal=0.9` adds
   `lo_<t>` and `hi_<t>`, an interval tracked to cover 90% of the rows,
   and `coverage_<t>`, the coverage it has delivered.
2. **Run the query.**
3. **Drop its first several half-lives of rows with `.slice(n)`,** while
   `sigma`, the quantile and the conformal interval's radius (its
   half-width) approach their steady values. For example, drop ten
   half-lives with `.slice(5_000)` at a half-life of 500 rows.
4. **Build each interval:** `pred ± 1.645 · sigma` for a Gaussian 90%
   interval, `pred ± abs_resid_q0.9` from the quantile, or `lo` to `hi`.
5. **Read each interval's share of rows covered and its mean
   half-width.**

This code builds three 90% intervals on two streams of the same relation,
one with Gaussian noise and one with Student's t at 3 degrees of freedom,
scaled to the same standard deviation:

```python
import numpy as np
import polars as pl

import polars_online as po

# The same relation twice, once with Gaussian noise and once with Student's t at 3 degrees of
# freedom scaled to the same standard deviation.
rng = np.random.default_rng(11)
n = 50_000
x0, x1 = rng.standard_normal((2, n))
noise = {"gauss": rng.standard_normal(n), "t3": rng.standard_t(3, n) / np.sqrt(3.0)}
pl.DataFrame({"x0": x0, "x1": x1} | {f"y_{k}": 1.0 * x0 + 0.5 * x1 + v for k, v in noise.items()}).write_parquet("width.parquet")
streams = pl.scan_parquet("width.parquet")

# 1. One ridge per stream, with each of the three intervals.
specs = [
    po.spec.ewridge(
        k, targets=[f"y_{k}"], features=["x0", "x1"], half_life=500.0,
        emit_sigma=True,  # sigma: for the Gaussian interval pred ± 1.645 sigma
        resid_quantiles=[0.9],  # abs_resid_q0.9: the exponentially weighted 90% quantile of |resid|
        conformal=0.9,  # lo, hi: an interval tracked to cover 90% of the rows
    )
    for k in noise
]
# 2 and 3. Run the query, past its first ten half-lives.
out = streams.online.fit_predict(specs).slice(5_000).collect()

# 4 and 5. Each interval's share of rows covered, and its mean half-width.
print(f"{'noise':6} {'interval':>20} {'covered':>8} {'half-width':>11}")
for k in noise:
    rows = out.select(pl.col(k).struct.unnest(), y=pl.col(f"y_{k}"))
    miss = (pl.col("y") - pl.col(f"pred_y_{k}")).abs()
    intervals = {
        "pred ± 1.645 sigma": 1.645 * pl.col(f"sigma_y_{k}"),
        "pred ± q0.9": pl.col(f"abs_resid_q0.9_y_{k}"),
        "conformal": (pl.col(f"hi_y_{k}") - pl.col(f"lo_y_{k}")) / 2,
    }
    for name, half in intervals.items():
        covered, width = rows.select(covered=(miss <= half).mean(), width=half.mean()).row(0)
        print(f"{k:6} {name:>20} {covered:>8.3f} {width:>11.3f}")
```

```text
noise              interval  covered  half-width
gauss    pred ± 1.645 sigma    0.899       1.642
gauss           pred ± q0.9    0.900       1.647
gauss             conformal    0.900       1.650
t3       pred ± 1.645 sigma    0.928       1.585
t3              pred ± q0.9    0.900       1.359
t3                conformal    0.900       1.364
```

On Gaussian noise the three intervals are the same: each covers 90% of the
rows at the same width. On Student's t, the Gaussian interval covers 92.8%,
more than asked, because `sigma` is inflated by the rare huge residuals,
and it is wider than it needs to be for most rows. The quantile and the
conformal interval cover 90.0% at a half-width of 1.36, against the
Gaussian interval's 1.59. Three more streams of 200,000 rows, with a
half-life of 2,000 rows, were measured on 2026-11-08. On Student's t
noise at 2.5 degrees of freedom the Gaussian interval covered 95.1%, and
on a scale mixture 94.2%. The third stream's slope flips halfway and its
noise triples at three quarters. On it the Gaussian interval covered
90.0% overall, since its 91.5% while the slope is relearned and its
88.3% after the noise triples cancel out. `conformal` covered 90.0% on
all three.

**Serve `lo` and `hi` from `conformal` when the coverage must hold, and
`sigma` when a scale is enough.** `conformal` tracks a quantile of
`|resid|` directly, so its long-run coverage is the number asked for,
whatever the residuals do.

## Keep the coverage when targets arrive late

Under `embargo` or `label_clock`, a row's target arrives after the row was
shown its interval. So the conformal radius learns of a miss only when
the target arrives. Once more than about 50 rows wait for their targets,
for a 90% interval, the default `conformal_rate` of 0.05 no longer holds
the radius steady. Past about 150, the radius overshoots and undershoots,
and the interval is wider on average.
`"auto"` takes smaller steps the more rows wait for their targets,
counted at each target's arrival and averaged over the model's
half-life. [Conformal intervals](../DIAGNOSTICS.md#conformal-intervals)
derives the step and measures it on New York's 2013 flights.

1. **Set `conformal_rate="auto"` in the spec,** beside `conformal` and the
   `embargo` or `label_clock` that delays the targets. With no rows
   waiting, `"auto"` is 0.05.
2. **Run the query.**
3. **Read the covered share and the mean half-width,** as step 5 of [Put an interval on each
   prediction](#put-an-interval-on-each-prediction) does.

This code learns each row 300 clock units after predicting it, at a row a
clock unit, and compares the two rates with the same stream's targets
learned at once:

```python
# y = 1.0 x0 + 0.5 x1 + noise, Student's t at 3 degrees of freedom, a row each clock unit.
rng = np.random.default_rng(12)
n = 60_000
x0, x1 = rng.standard_normal((2, n))
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": 1.0 * x0 + 0.5 * x1 + rng.standard_t(3, n) / np.sqrt(3.0)}).write_parquet("late.parquet")
late = pl.scan_parquet("late.parquet")

# The two rates, with the targets at once and 300 clock units later.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=2_000.0, conformal=0.9)
print(f"{'targets':16} {'rate':>5} {'covered':>8} {'half-width':>11}")
for delay, embargo in (("at once", {}), ("300 units later", {"embargo": 300.0})):
    for rate in (0.05, "auto"):
        # 1 and 2. The same ridge, with the targets at once or 300 units later, at each rate.
        spec = po.spec.ewridge("late", conformal_rate=rate, **embargo, **common)
        rows = late.online.fit_predict([spec]).slice(10_000).collect().select("y", pl.col("late").struct.unnest())
        # 3. The share of rows covered, and the mean half-width.
        covered, width = rows.select(
            covered=pl.col("y").is_between(pl.col("lo_y"), pl.col("hi_y")).mean(),
            width=((pl.col("hi_y") - pl.col("lo_y")) / 2).mean(),
        ).row(0)
        print(f"{delay:16} {rate!s:>5} {covered:>8.3f} {width:>11.3f}")
```

```text
targets           rate  covered  half-width
at once           0.05    0.900       1.375
at once           auto    0.900       1.375
300 units later   0.05    0.899       2.400
300 units later   auto    0.900       1.362
```

With the targets at once, the two rates are the same. With each target 300
rows late, both cover 90% of the rows, but the default rate's swings make
its half-width 2.40 on average, against 1.36 for `"auto"`, about the
half-width with no delay.

## Check that it worked

Read the covered share and the half-width over many half-lives:

| what you see | what it means | what to do |
|---|---|---|
| a Gaussian interval that covers more than asked, and is wider than the quantile's | the residuals have heavy tails, which inflate `sigma` | serve `conformal`'s `lo` and `hi`, or the `resid_quantiles` interval |
| a conformal interval at the coverage asked, but much wider under a delay than without one | the radius overshoots, since it learns of each miss only when the target arrives | set `conformal_rate="auto"` |
| a conformal interval slow to recover after the noise jumps | a small rate recovers slowly, and no rate corrects the radius faster than the targets arrive | choose the rate on an earlier stretch of the stream, as [Choose a model's settings from the stream's own errors](../CALIBRATION.md) chooses any setting |
| a conformal interval whose share of rows covered sits off the coverage asked, under a `weight` column | each step is scaled by the row's weight over the mean weight, so a heavy row's miss counts for more, and the share of rows covered can sit on either side of the coverage asked | nothing, where the heavy rows are the ones whose coverage matters; otherwise, fit a second spec without `weight` for the interval |
