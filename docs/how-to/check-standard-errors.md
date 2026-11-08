<!-- kind: how-to -->
# Check a coefficient's standard error

This how-to is for anyone who reads a regression coefficient's t on a
stream and wants to know whether to believe it. It turns on the plain and
the robust standard errors, compares them, and chooses the one whose
assumptions the residuals meet.

## Prerequisites

These steps need the library and a least-squares fit:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of `ewridge`
  or `rls`, the two models that take `emit_robust_se`. `kalman` takes
  `emit_se_coef` alone ([What each costs, and which models take
  it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it)).
- For a [target](../GLOSSARY.md#the-model-bank-and-what-it-reads) that
  looks ahead: its horizon in rows. Without a clock, the spec's `embargo=`
  gives it. With a clock, give `horizon_rows=`, the rows the target looks
  ahead.

## Compare the standard errors

A coefficient's plain standard error assumes residuals that are
independent and equally spread. A target that sums the next `h` rows'
changes shares `h − 1` of them with its neighbor, so its residuals are
correlated. Against a feature that moves slowly, a t that ignores the
overlap is then about `sqrt(h)` times too large. [Can its t be trusted?](../DIAGNOSTICS.md#can-its-t-be-trusted)
defines the three standard errors and how each diagnostic reads the
horizon. [A target that looks
ahead](../explanation/out-of-sample.md#a-target-that-looks-ahead) explains
why its rows are learned late.

1. **Give the spec the target's horizon:** `embargo=h` on a spec with no
   clock, or `horizon_rows=h` on one with a clock. `se_coef_hac`'s lags
   default to twice it.
2. **Turn on `emit_se_coef=True` and `emit_robust_se=True`.** Each row
   that carries `coef` gains `se_coef`, `se_coef_hc0` and `se_coef_hac`,
   laid out like `coef`, the intercept first.
3. **Run the query, and read each standard error beside its coefficient.**
   A t is the coefficient over its standard error.
4. **Compare the three.** Where `se_coef_hac` is much larger than
   `se_coef`, the residuals are correlated, and the plain t is too large.

This code fits 100 streams in one pass, one group each. Each target sums
the next 10 shocks, which the feature knows nothing about, so the true
slope is 0, and a test at 5% should reject on about 5% of the streams:

```python
import numpy as np
import polars as pl

import polars_online as po

# 100 independent streams of 3,000 rows. x is persistent (AR(1) at 0.95); the target sums the next
# h = 10 shocks, which x knows nothing about, so its true slope is 0.
rng = np.random.default_rng(4)
streams, n, h = 100, 3_000, 10
frames = []
for s in range(streams):
    x = np.zeros(n)
    u = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + u[i]
    # The target sums the next h shocks, so neighboring targets share h - 1 of them.
    e = rng.standard_normal(n + h)
    fwd = np.convolve(e, np.ones(h), "valid")[1 : n + 1]  # e[i+1] + ... + e[i+h]
    frames.append(pl.DataFrame({"stream": s, "x": x, "fwd": fwd}))
pl.concat(frames).write_parquet("overlap.parquet")
overlap = pl.scan_parquet("overlap.parquet")

# 1 and 2. One slope per stream, with its plain, HC0 and HAC standard errors.
model = po.spec.ewridge(
    "m", targets=["fwd"], features=["x"], half_life=float("inf"), group="stream",
    embargo=h,  # a row's target is known h rows later, and is learned only then
    emit_se_coef=True,  # se_coef: assumes uncorrelated residuals
    emit_robust_se=True,  # se_coef_hc0, and se_coef_hac to 2 * embargo = 20 lags
)
# 3. Each stream's slope and standard errors at its last row.
out = overlap.online.fit_predict([model]).collect()
last = out.group_by("stream", maintain_order=True).last().select(pl.col("m").struct.unnest())
slope = last.select(
    coef=pl.col("coef").list.get(1),  # [intercept, x]
    se=pl.col("se_coef").list.get(1),
    hc0=pl.col("se_coef_hc0").list.get(1),
    hac=pl.col("se_coef_hac").list.get(1),
)
# 4. The slope's real spread, and how often each standard error rejects the true 0.
print(f"the slope's spread over the {streams} streams: {slope['coef'].std():.4f}")
for se in ["se", "hc0", "hac"]:
    rejected = (slope["coef"] / slope[se]).abs().gt(1.96).mean()
    print(f"{se:4} median {slope[se].median():.4f}, |t| > 1.96 on {rejected:.0%} of the streams")
```

```text
the slope's spread over the 100 streams: 0.0546
se   median 0.0188, |t| > 1.96 on 44% of the streams
hc0  median 0.0188, |t| > 1.96 on 42% of the streams
hac  median 0.0504, |t| > 1.96 on 8% of the streams
```

The slope estimates spread with a standard deviation of 0.0546 over the
100 streams. `se_coef` puts it at 0.0188, 2.9 times too small, close to
`sqrt(10) = 3.2`, so a t that reads it finds a slope on 44% of the streams
where there is none. `se_coef_hc0` is no better, because the residuals'
spread does not move with `x`: their fault is correlation. `se_coef_hac`
reads 0.0504, near the true spread, and rejects on 8%.

## Choose the standard error to read

Read the standard error whose assumptions the residuals meet:

| the residuals | read | set |
|---|---|---|
| from a target that looks ahead | `se_coef_hac` | `robust_se_lags` at least the horizon in rows; it defaults to twice it |
| with a spread that moves with the features | `se_coef_hc0` | |
| from a one-step target, independent and of one spread | `se_coef` | |

On a one-step target, `breusch_pagan` and `ljung_box` test those two
assumptions: turn on
`emit_specification=True`, as [Find what a model
misses](find-what-the-model-misses.md) does, and read `se_coef` only where
both are quiet.

`po.gram.coef_stats`, which fits from a Gram, and `marginal`'s `t_stat`
read Kish's size (the effective number of rows of the weighted fit), and are silent
about the overlap. Read `po.gram.coef_stats`' t as a scale for comparing
coefficients. For `marginal`, read `t_stat_serial`, which its `lags=` and
`serial_rule=` add ([`po.spec.marginal`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.marginal)
gives their values), and which corrects each pair's count for rows that
resemble their neighbors ([Can its t be
trusted?](../DIAGNOSTICS.md#can-its-t-be-trusted)).

## Check that it worked

Fit the same spec on a stream where the true slope is known, as the
recipe does, and count how often `|t| > 1.96` rejects it: near 5% means
the standard error is right. On your own stream, compare `se_coef_hac`
with `se_coef`:

| what you see | what it means | what to do |
|---|---|---|
| `se_coef_hac` near `se_coef` | the residuals are not correlated at the lags tested | read either |
| `se_coef_hac` several times `se_coef` | the residuals are correlated, as a target that looks ahead makes them | read `se_coef_hac` |
| a `ReadinessWarning` that the horizon is 0 | the spec has a clock and an `embargo`, but no `horizon_rows` | give `horizon_rows=`, the rows the target looks ahead |
| a `ReadinessWarning` that the lags stop at 64 | the horizon read from `embargo` is past 32 rows | give `horizon_rows=`, which takes twice its value in lags, whatever they cost, or set `se_coef_hac`'s lags alone with `robust_se_lags` |
