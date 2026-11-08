<!-- kind: how-to -->
# Scale a feature by its spread

This how-to is for anyone who wants a model's coefficient to be a
response per standard deviation of a feature, at the feature's spread as
it stands on each row. It fits the model on each feature's z-score
against its own recent mean and spread.

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  `ticks.parquet`, which holds the README's `df`: 400 rows on the clock
  `t`, which steps by 1 a row. Its `y` is `1 + 2·x0 − x1` plus noise.
- A [clock](../GLOSSARY.md#time) column, and the
  [`gap_cap`](../GLOSSARY.md#time) a clock requires: the largest clock
  step a window or a model measures. A longer step ends every window open
  across it, and a model decays it as a step of `gap_cap`.

## Fit the model on z-scores

**When a target responds to a feature relative to the feature's current
volatility, fit the model on the feature's z-score.** A coefficient on a
raw feature is a response per unit of the feature, the same in a calm hour
and a volatile one. A coefficient on its z-score is a response per standard
deviation, at the feature's spread as it stands on that row.

1. **Make each z-score a column with `po.ewm_mean` and `po.ewm_std` in
   the query's `.online.with_windows`
   ([`po.stream.with_windows`](https://hgilde.github.io/polars-online/stream.html#polars_online.stream.with_windows)),
   chained in the query before the
   [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads).** The
   z-score is `(x − po.ewm_mean(x)) / po.ewm_std(x)`. Give both
   [window operators](../GLOSSARY.md#fitting-a-model) the same
   `half_life`, how fast the mean and the spread forget, in the clock's
   units. [Make features that look back over the
   stream](make-window-features.md) shows the call.
2. **Name the z-score columns in the spec's `features`,** with the same
   clock keywords as the `with_windows` call.
3. **Run the query.** The model's coefficients are per standard deviation
   of each feature.

The truth is `y = 1 + 2·x0 − x1`, so per standard deviation of `x0` the
response is 2 times `x0`'s spread on that row. The
recipe prints the spread `po.ewm_std` gives against Polars' own `ewm_std`,
and the fitted coefficients against the truth. This code uses
`ticks.parquet` from [Example data](../../README.md#example-data):

```python
# Each feature in units of its current spread, and a fit on those.
row_clock = dict(clock="t", gap_cap=10.0)                # one clock for the z-scores and the model; t steps by 1 a row
spread = dict(half_life=100.0)                           # how fast the mean and the spread forget, in the clock's units
zscored = pl.scan_parquet("ticks.parquet").online.with_windows(
    z_x0=(pl.col("x0") - po.ewm_mean("x0", **spread)) / po.ewm_std("x0", **spread),   # x0 in units of its current spread
    z_x1=(pl.col("x1") - po.ewm_mean("x1", **spread)) / po.ewm_std("x1", **spread),
    sd_x0=po.ewm_std("x0", **spread), sd_x1=po.ewm_std("x1", **spread),
    **row_clock,
)
per_sd = po.spec.ewridge("per_sd", targets=["y"], features=["z_x0", "z_x1"], half_life=100.0, **row_clock)
fitted = zscored.online.fit_predict([per_sd]).collect()   # the rows, the z-scores, the spreads and the column per_sd

# The spread against Polars', and the fit against the truth.
polars_sd = pl.read_parquet("ticks.parquet")["x0"].ewm_std(half_life=100.0, adjust=False)
print(f"po.ewm_std against Polars' ewm_std: largest difference {(fitted['sd_x0'] - polars_sd).abs().max():.12f}")
last = fitted.row(-1, named=True)
intercept, b0, b1 = last["per_sd"]["coef"]
print(f"per standard deviation: z_x0 {b0:.2f} (truth 2 x {last['sd_x0']:.2f} = {2 * last['sd_x0']:.2f}), "
      f"z_x1 {b1:.2f} (truth -1 x {last['sd_x1']:.2f} = {-last['sd_x1']:.2f})")
```

```text
po.ewm_std against Polars' ewm_std: largest difference 0.000000000000
per standard deviation: z_x0 1.81 (truth 2 x 0.95 = 1.90), z_x1 -0.92 (truth -1 x 0.98 = -0.98)
```

## Check the result

**Compare each coefficient with the truth times the feature's spread on
the same row.** In the recipe, the fitted responses sit within about a
tenth of the truth, each read at the feature's spread on the last row.

**The operators run over a stream in one pass, where Polars' own forms
need the column in memory.** On a clock that steps by 1 a row, with
`half_life` in rows, `po.ewm_std` is Polars' `ewm_std` with
`adjust=False`, as the recipe's first line shows. On any other clock, the
operators weigh each row by the time it covers ([docs/WINDOWS.md](../WINDOWS.md)
lists the operators).

## Use z-scores with any model

**A z-score adds one column per feature, and it works with every model.**
Each z-score is one more column, made by two operators that share their
window's rows in the `with_windows` pass. A model reads it as any other column, so `ewridge`,
`rls`, `kalman` and the rest take it alike. A model's coefficients are per
unit of its features, so z-scored features give coefficients per standard
deviation.

**For `kalman`, give the z-score columns with `standardize=False`,** so
the filter does not standardize them a second time against its own
moments.

**For `ftrl`, give it features near 0, z-scored or standardized in the
query before the bank.** `ftrl` does not center a feature. The docstring
of [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl)
measures, in its Notes under *Features at a level*, how far a feature's
level carries a coefficient's error into the prediction, and says why the
model has no `standardize`.

**On minute returns, every model lost less with z-scored features.**
Section 6 of [docs/VALIDATION.md](../VALIDATION.md) fits four specs
(`ewridge` at two solve cadences, `rls` and `kalman`) to BTCUSDT's next
one and next five minutes of log returns, each at a half-life of 500
minutes and with the features raw, then z-scored. Every R² is below 0,
since no model predicts these returns better than their mean, but each is
closer to 0 on the z-scores:

| model | next minute, raw | next minute, z-scores | next five minutes, raw | next five minutes, z-scores |
|---|---|---|---|---|
| `ewridge`, at its default solve cadence | -0.0148 | -0.0096 | -0.0521 | -0.0262 |
| `ewridge`, solved every row | -0.0647 | -0.0412 | -0.1113 | -0.0474 |
| `rls` | -0.0644 | -0.0412 | -0.1108 | -0.0475 |
| `kalman` | -0.0699 | -0.0455 | -0.1005 | -0.0427 |
