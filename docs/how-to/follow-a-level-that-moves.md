<!-- kind: how-to -->
# Follow a level that moves

This how-to is for anyone who fits `pa` or `sgd` on a target whose level,
the value it moves about, drifts or jumps, and wants the fit to keep up.

`pa` and `sgd` learn by moving their coefficients a step at each row, so
the fit reaches a new level only as fast as its steps carry it. To keep
`pa` up, give it a larger cap on its step, or a target with the level
taken out. To keep `sgd` up under its Huber loss, give the scale that
clips its steps a shorter memory (`huber_scale_half_life`).

## Prerequisites

These steps need the library and a spec of one of the two models:

- polars-online installed. Each example builds its own rows, so none
  needs the README's example data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of `pa`, or of
  `sgd` with `loss="huber"`, whose
  [target](../GLOSSARY.md#the-model-bank-and-what-it-reads) can change
  level.

## Give `pa` a larger `c`

`pa`, passive-aggressive regression, moves its fit at each row by the
smallest step that brings the row within a margin of its target. Under its
default `mode="pa1"`, `c` caps that step. The fit starts from zero
coefficients, and a `pa1` cap of 1, the default, moves the intercept about
one unit a row. So on a target far from 0, or one that moves faster than that,
the default fit lags behind.

1. **Give the spec `c=10`, or larger.**
2. **Fit the spec as you would at the default `c`,** with `fit_predict`, or
   a query's `online.fit_predict`.

On a target falling from 1,000 to −1,000 at 0.5 a row, at a
[half-life](../GLOSSARY.md#time) of 200 rows and noise 0.3, three runs
show the cap's effect. With the intercept alone fitting the level, the
default's intercept met the level near row 670. With the level as a
feature too, the default cap's steps swung the fit far past the level
instead. Each step is taken on the standardized features
(`standardize=True`, the default) and mapped back into the caller's
units. With `c=10` and the level as a feature, every row stayed within 3
of its target from row 467. The recipe in [Difference the target for
`pa`](#difference-the-target-for-pa) runs the last two.

**A drift the intercept carries alone keeps a lag of about 2 at any `c`.**
Half of that lag comes from `pa`'s margin, `eps`, which is measured in
`s_y`, the target's own spread, and a drift widens that spread. To take
the lag away, difference the target, as in [Difference the target for
`pa`](#difference-the-target-for-pa).

## Difference the target for `pa`

**Differencing the target and the moving features takes the level out of
the fit.** The model then predicts each row's change, and the level is the
last row's target plus the predicted change.

1. **Make the differences with Polars' `diff` in the query, before the
   bank,** as `dy=pl.col("y").diff()`, which is `y − y.shift(1)`. Difference
   each feature that moves with the level too.
2. **Name the differences as the spec's target and features,** as
   `targets=["dy"]`.
3. **Read the level back with `with_columns` after the bank,** as the
   last row's `y` plus the predicted change:
   `pl.col("y").shift(1) + pl.col("d").struct.field("pred_dy")`, where `d`
   is the spec's name, and so its output column.

This code builds the falling target of [Give `pa` a larger
`c`](#give-pa-a-larger-c), with the level itself as the feature `level`.
It fits `pa` at the default `c`, at `c=10`, and on the differences. For
each run it prints `pred − y` at rows 800 and 1,600, and the row from
which every row stays within 3 of its target:

```python
import numpy as np
import polars as pl
import polars_online as po

# y falls from 1,000 to -1,000 at 0.5 a row, with noise 0.3; the column level holds the level itself.
rng = np.random.default_rng(17)
level = np.linspace(1000.0, -1000.0, 4000)
x = rng.normal(size=(4000, 2))
y = level + 2.0 * x[:, 0] - x[:, 1] + 0.3 * rng.normal(size=4000)
pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "level": level, "y": y}).write_parquet("falling.parquet")


# The first row from which every row's miss is within 3 (a withheld row counts as a miss).
def within_from(miss):
    bad = np.flatnonzero(~(np.abs(miss) < 3.0))
    return int(bad[-1]) + 1 if len(bad) else 0


# pa at a half-life of 200 rows.
common = dict(half_life=200.0, min_weight=10.0)   # predictions withheld for the first rows' weight of 10
runs = {}
# The level as a feature, at the default c and at c = 10.
for name, c in [("default", {}), ("c=10", {"c": 10.0})]:
    spec = po.spec.pa("m", targets=["y"], features=["x0", "x1", "level"], **common, **c)
    out = pl.scan_parquet("falling.parquet").online.fit_predict([spec]).collect()
    runs[name] = out["m"].struct.field("pred_y") - out["y"]
# The differences, and the level read back as the last row's y plus the predicted change.
diffs = po.spec.pa("d", targets=["dy"], features=["dx0", "dx1"], **common)
out = (pl.scan_parquet("falling.parquet")
         .with_columns(dy=pl.col("y").diff(), dx0=pl.col("x0").diff(), dx1=pl.col("x1").diff())   # 1. the differences
         .online.fit_predict([diffs])                                                              # 2. fitted
         .with_columns(read_back=pl.col("y").shift(1) + pl.col("d").struct.field("pred_dy"))      # 3. the level read back
         .collect())
runs["differenced"] = out["read_back"] - out["y"]
print("the run       row 800  row 1,600  every row within 3 from row")
for name, miss in runs.items():
    miss = miss.to_numpy().astype(float)
    print(f"{name:<12} {miss[800]:8.2f} {miss[1600]:10.2f}  {within_from(miss)}")
```

```text
the run       row 800  row 1,600  every row within 3 from row
default       1788.61      -3.19  2795
c=10             0.67       1.12  467
differenced     -0.73      -0.05  12
```

At the default `c` the fit is still 1,789 past its target at row 800,
and stays within 3 only from row 2,795. The differenced run follows the
level from row 12. The [`pa`
docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.pa)'s
Notes, under *On a level that moves, give a larger c or difference the
target*, give these runs and more.

## Give `sgd`'s Huber scale a shorter memory

**On a stream whose level can change, give `huber_scale_half_life` about
a fifth of `half_life`, or shorter.** Under `loss="huber"`, `sgd` clips
each row's gradient at `huber_delta·s`, where `huber_delta` is 1.345 by
default and `s` is the scale of its residuals
(the target less the prediction). Under the default `huber_scale="mad"`,
`s` is a median, so `sgd`'s Huber loss follows a change of level far more
slowly than under `"rms"`, a root mean square. While `s` is the noise's
scale, the intercept moves at most `learning_rate·huber_delta·s` a row. So a
shift many times the noise waits until the median itself has crossed to
the new level. [A level shift: the trade-off a robust scale
makes](../explanation/robust-scale.md#a-level-shift-the-trade-off-a-robust-scale-makes)
explains why a robust scale holds a fit back after a shift.

1. **Give `huber_scale_half_life` a fifth of the spec's `half_life`, or
   less,** as `huber_scale_half_life=100.0` beside `half_life=500.0`. It
   is in [clock units](../GLOSSARY.md#time), as `half_life` is, and is the
   spec's `half_life` when not given.
2. **Keep `huber_scale="mad"`, the default,** which still resists a burst
   of outliers. A shorter memory follows a shift sooner, but it also
   widens the cut sooner in a long burst of outliers.

The recipe measures `sgd` on `y = 2x + 0.5ε`, with a fit half-life of 500
rows, a shift of +20 at row 10,000 and no outliers. It runs five fits: the
squared loss, the Huber loss under `"rms"`, and the Huber loss under
`"mad"` with the scale's half-life at the fit's 500 rows, at 100 and at
20. For each it prints the mean absolute error over rows 0-200 and
200-400 after the shift, and the rows after the shift until every 200-row
average of the error stays under 1.0. Seeds 2 and 3 agree with seed 1,
the one shown, to within 0.06. This code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 2x plus noise, with a shift of +20 at row 10,000; one row a unit of t.
rng = np.random.default_rng(1)
x = rng.standard_normal(14_000)
y = 2.0 * x + 0.5 * rng.standard_normal(14_000)
y[10_000:] += 20.0
pl.DataFrame({"t": np.arange(14_000.0), "x": x, "y": y}).write_parquet("shift.parquet")

# The five fits of the table, at a fit half-life of 500 rows.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=500.0)
specs = [po.spec.sgd("squared", loss="squared", **common),
         po.spec.sgd("rms", loss="huber", huber_scale="rms", **common),
         po.spec.sgd("mad", loss="huber", **common),                                  # the scale's half-life: the fit's 500
         po.spec.sgd("mad_100", loss="huber", huber_scale_half_life=100.0, **common),  # a fifth of the fit's
         po.spec.sgd("mad_20", loss="huber", huber_scale_half_life=20.0, **common)]
out = pl.scan_parquet("shift.parquet").online.fit_predict(specs).collect()
# Each fit's error after the shift, and the rows until every 200-row average stays under 1.0.
print("fit      rows 0-200  rows 200-400  rows until under 1.0")
for spec in specs:
    miss = (out[spec["name"]].struct.field("pred_y") - out["y"]).abs().to_numpy()[10_000:]
    averages = np.convolve(miss, np.ones(200) / 200, mode="valid")      # each 200-row average after the shift
    until = int(np.flatnonzero(averages >= 1.0)[-1]) + 1 + 200
    print(f"{spec['name']:<8} {miss[:200].mean():10.2f} {miss[200:400].mean():13.2f}  {until}")
```

```text
fit      rows 0-200  rows 200-400  rows until under 1.0
squared        8.67          1.17  418
rms           13.55          2.45  493
mad           19.26         17.33  885
mad_100       15.81          2.99  510
mad_20        10.34          1.41  440
```

Under `"mad"` at the fit's half-life, the error over rows 200-400 is
seven times the error under `"rms"`. A scale's half-life of 100, a fifth
of the fit's, brings it near the `"rms"` fit's.

## Check the result

**Measure the error after a known shift, at each setting, on rows where
you know the level.** Plant a shift or a drift in rows of your own, or
pick a stretch of your stream where the level is known to have moved. Then
compare two measures, as the second recipe does: the mean absolute error
over the rows after the move, and the row from which the error stays
small.
Both come from each spec's `pred_<target>` field less the target, row by
row.
