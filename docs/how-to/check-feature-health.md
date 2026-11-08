<!-- kind: how-to -->
# Check the health of a model's features

This how-to is for anyone who fits a linear model on a stream whose
feature columns can go stale, as a feed that stops and is carried forward
does. It watches each feature's spread and level as the stream runs, and
flags a stale or moved feature long before the coefficient on it runs away.

## Prerequisites

These steps need the library and a linear model with features:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of one of the
  nine linear models with features, such as `rls` or `ewridge`, which take
  `emit_feature_health`. [What each costs, and which models take
  it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it) lists
  them.
- A spec whose fit forgets (a finite `half_life` or a `window_size`), or
  a finite `feature_health_half_life`: the switch is refused where its
  memory is the whole stream.

## Watch each feature's spread and level

A feed that stops and is carried forward is still a number on every row,
so no residual shows it. A fit that keeps sums, such as `rls`, then sees a
column that no longer varies, and its coefficient on that column winds up.
`emit_feature_health` compares each feature's spread and mean at the fit's
memory with those at four times it. [Are its features
healthy?](../DIAGNOSTICS.md#are-its-features-healthy) defines both fields
and their thresholds.

1. **Turn on `emit_feature_health=True` in the spec.** Each row gains
   `spread_ratio_<feature>` and `mean_shift_<feature>` for each feature,
   read before the row. A steady feature reads about 1 and 0.
2. **Run the query.**
3. **Flag the rows where `spread_ratio_<feature>` falls outside 0.5 to
   2,** and those where `|mean_shift_<feature>|` passes 0.3. No row of a
   clean feature passed the first at a half-life of 200, and every move of
   one spread passed the second within 190 rows.
4. **Read the first flagged row of each feature.** Under `embargo`, the
   fields lag the feed by the embargo, since each row's features are
   folded when the row is released.

This code holds `x1` at its last value for 3,000 rows, as a feed that
stopped would, fits `rls`, and prints when `x1`'s spread collapsed beside
when `rls`'s coefficient on it ran away:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = x0 + 0.5 x1 + noise. x1's feed stops at row 300 and comes back at row 3,300; the join that
# built the table carried its last value forward.
rng = np.random.default_rng(3)
n = 3_500
t = np.arange(n, dtype=float)
x0, x1 = rng.standard_normal((2, n))
y = x0 + 0.5 * x1 + 0.2 * rng.standard_normal(n)
# The fault: x1 held at its row-299 value until row 3,300.
held = (t >= 300) & (t < 3_300)
pl.DataFrame({"t": t, "x0": x0, "x1": np.where(held, x1[299], x1), "y": y}).write_parquet("feed.parquet")
feed = pl.scan_parquet("feed.parquet")

# 1 and 2. rls, with its coefficients and each feature's health on every row.
model = po.spec.rls(
    "m", targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=20.0,
    coef_every=0,  # the coefficients on every row
    emit_feature_health=True,  # spread_ratio_x0, spread_ratio_x1, mean_shift_x0, mean_shift_x1
)
out = feed.online.fit_predict([model]).online.unnest([model]).collect()

# 3 and 4. When x1's spread collapsed, and when rls's coefficient on it ran away.
quiet = out.filter((pl.col("t") >= 300) & (pl.col("spread_ratio_x1") < 0.5))["t"].min()
wound = out.filter((pl.col("t") >= 300) & (pl.col("coef_y_x1").abs() > 1.0))["t"].min()
print(f"x1 held from row 300; its spread_ratio fell below 0.5 at row {quiet:.0f}")
print(f"rls's coefficient on x1 (0.5 in the data) passed 1 at row {wound:.0f}")
for row, ratio, coef in out.filter(pl.col("t").is_in([299.0, quiet, 1_000.0, wound, 3_299.0])).select("t", "spread_ratio_x1", "coef_y_x1").iter_rows():
    print(f"  row {row:>5.0f}: spread_ratio_x1 {ratio:.3f}, coefficient on x1 {coef:.3g}")
```

```text
x1 held from row 300; its spread_ratio fell below 0.5 at row 358
rls's coefficient on x1 (0.5 in the data) passed 1 at row 1352
  row   299: spread_ratio_x1 0.891, coefficient on x1 0.492
  row   358: spread_ratio_x1 0.499, coefficient on x1 0.332
  row  1000: spread_ratio_x1 0.000, coefficient on x1 0.333
  row  1352: spread_ratio_x1 0.000, coefficient on x1 1.23
  row  3299: spread_ratio_x1 0.000, coefficient on x1 -4.61e+13
```

The feed stops at row 300. `spread_ratio_x1` falls below 0.5 at row 358,
2.9 half-lives in. By then the coefficient has moved to 0.33, wrong but
plausible, and gives no sign of the fault. `rls`'s
coefficient on `x1` passes 1 at row 1,352, 52.6 half-lives in, and by the
time the feed returns it has blown up. The feature-health flag came about
a thousand rows before the coefficient ran away.

## Fix a feature that went wrong

Choose the fix by which field flagged:

| what flagged | what it means | what to do |
|---|---|---|
| `spread_ratio_<feature>` below 0.5 | the feature stopped varying, as a stale feed carried forward does | in the query before the bank, null each value older than its own sampling interval, so the model skips the rows it did not observe ([Data quality](../DATA-QUALITY.md#constant-and-frozen-features) has the recipe). Or fit a model whose ridge does not fade, such as `ewridge`, which does not wind up |
| `\|mean_shift_<feature>\|` that stays large | the feature moved to a new level | refit the model from the row `mean_shift` flagged, as [Detect a break](detect-a-break.md#date-a-break-with-a-test-run-once) refits from a crossing: a coefficient fitted at the old level may not hold at the new one |

## Check that it worked

Run the query again on the fixed stream, and read the fields over the
stretch that flagged. A healthy feature's `spread_ratio_<feature>` stays
within 0.5 to 2, and its coefficient stays near its value before the
stretch.

| what you see | what it means | what to do |
|---|---|---|
| `\|mean_shift\| > 0.3` on a few percent of rows of a slow-moving feature | a persistent feature wanders: an AR(1) at 0.95 passed 0.3 on 6% of rows | flag such a feature at 0.5, which it passed on 0.03% of rows |
| the flag comes later than the feed's fault | the fields are read at `feature_health_half_life`, the fit's memory by default, and under `embargo` they lag the feed by the embargo: on a feed frozen at row 1,500 at a half-life of 20, the ratio fell below 0.5 at row 1,547, and under an embargo of 100 rows at row 1,646 | expect the lag, and date the fault from the flag less the embargo |
| a refusal of `emit_feature_health` | the fit's memory is the whole stream (`half_life=inf` with no window), so both memories would be too | set `feature_health_half_life` to a finite value, or give the fit a finite `half_life` or a `window_size` |
