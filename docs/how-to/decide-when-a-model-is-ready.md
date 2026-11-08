<!-- kind: how-to -->
# Decide when a model is ready

This how-to is for anyone who serves a model's predictions from its first
rows and wants it to report only once it is informed enough. It reads why
a prediction was withheld, sets the gates that hold predictions back
through the [warm-up](../GLOSSARY.md#warm-up-and-readiness), and acts on
a readiness warning.

## Prerequisites

These steps need the library and a spec to gate:

- polars-online installed, and numpy to make the examples' data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) that takes the
  gates. `min_weight` applies to every model that predicts,
  `min_settled_frac` to a spec with a finite `half_life` or a `lam` below 1,
  and `max_error_inflation` to `ewridge`, `rls`, `kalman` and `lasso`.
  [Warm-up gates](../MODELS.md#warm-up-gates), in MODELS.md, states each gate, its
  default and the field it reads.

## Read why a prediction is withheld

A [gate](../GLOSSARY.md#warm-up-and-readiness) holds back predictions
only: the model learns from every row either way. On a
[withheld](../GLOSSARY.md#warm-up-and-readiness) row the prediction is
null, and the fields that report readiness are filled. [Warm-up and
readiness](../explanation/warm-up-and-readiness.md) explains what each
gate measures.

1. **Run the query, and read `withheld_reason` on each row.** It names
   the gate that held the row, or a reason the fit cannot be trusted there,
   such as `ill_conditioned`. [Is it
   ready?](../DIAGNOSTICS.md#is-it-ready) lists every reason in its order
   of precedence.
2. **Read `weight_sum` and `settled_frac` beside it,** the two numbers the
   `min_weight` and `min_settled_frac` gates compare.
3. **For `max_error_inflation` on `ewridge`, `rls` or `kalman`, turn on
   `emit_error_inflation=True`.** Each row
   gains `error_inflation_<t>`, the factor by which estimation error
   would inflate that row's prediction error over the noise floor, read
   against the row's own features.

## Choose a gate

Choose a gate by what matters more: a prediction on every row, or an
early prediction as good as a late one. `max_error_inflation` reads the
fit's own size, so it moves with the number of features.
`min_settled_frac` and `min_weight` ask for a fixed amount of history.

| gate | what it waits for | choose it when |
|---|---|---|
| the default, `max_error_inflation=sqrt(2)` (on `ewridge` alone) | estimation error to inflate a prediction's error over the noise floor by at most `sqrt(2)` | a prediction on almost every row matters most |
| `max_error_inflation` lowered toward 1, such as 1.05 | estimation error to inflate it by at most 5% | early predictions should be nearly as good as late ones |
| `min_settled_frac`, such as 0.5 | that share of the steady state: 0.5 is one half-life of the clock | the half-life was chosen to average over regimes or seasons that a shorter history would not represent |
| `min_weight`, such as 100.0 | that much `weight_sum` | a fixed amount of weight should stand behind each prediction |

1. **Set the gate you chose in the spec.**
2. **Run the query, and read each gate's first prediction and early
   error.**

This code fits ten features under each gate, and prints when each first
predicts, what held it back, and the error of its predictions before row
600:

```python
import numpy as np
import polars as pl

import polars_online as po

# Ten features, a row a second, y on all of them.
rng = np.random.default_rng(14)
n, k = 3_000, 10
x = rng.standard_normal((n, k))
y = x @ np.linspace(1.0, 0.1, k) + rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
pl.DataFrame({"t": np.arange(n, dtype=float), "y": y} | {f: x[:, j] for j, f in enumerate(features)}).write_parquet("ready.parquet")
ready = pl.scan_parquet("ready.parquet")

# 1. The same ridge under each readiness gate.
common = dict(targets=["y"], features=features, clock="t", gap_cap=5.0, half_life=300.0)
specs = [
    po.spec.ewridge("default", **common),  # max_error_inflation=sqrt(2), the one gate on by default
    po.spec.ewridge("strict", max_error_inflation=1.05, **common),  # 5% over the noise floor
    po.spec.ewridge("settled", min_settled_frac=0.5, **common),  # one half-life of history
    po.spec.ewridge("weighted", min_weight=100.0, **common),  # 100 of weight_sum
]
out = ready.online.fit_predict(specs).collect()

# 2. Each gate's first prediction, what held it back, and its early error.
for spec in specs:
    rows = out.select("t", pl.col(spec["name"]).struct.unnest())
    first = rows.filter(pl.col("pred_y").is_not_null())["t"].min()
    reasons = rows["withheld_reason"].drop_nulls().value_counts().sort("withheld_reason")
    held = ", ".join(f"{reason} on {count} rows" for reason, count in reasons.iter_rows())
    mse = rows.filter(pl.col("t") < 600).select(pl.col("resid_y").pow(2).mean()).item()
    print(f"{spec['name']:9} predicts from row {first:>3.0f} ({held}); its mse before row 600: {mse:.2f}")
```

```text
default   predicts from row  12 (above_max_error_inflation on 12 rows); its mse before row 600: 1.12
strict    predicts from row 108 (above_max_error_inflation on 108 rows); its mse before row 600: 1.01
settled   predicts from row 301 (below_min_settled_frac on 301 rows); its mse before row 600: 0.99
weighted  predicts from row 114 (below_min_weight on 114 rows); its mse before row 600: 1.02
```

The default gate holds back the first 12 rows, where 11 coefficients would
otherwise be fitted on fewer rows than they number. Its predictions before
row 600 have an error of 1.12 against a noise floor of 1, because the
earliest ones still carry much estimation error. The strict gate, at 5%
over the floor, waits for 108 rows and its early error is 1.01. The other
two gates wait for one half-life of history or for 100 of weight, and
reach the same early error.

## Act on a readiness warning

A `min_weight` above the weight a stream can reach is never met. Under a
half-life, a target's weight stops growing at a ceiling, so the gate
withholds every row. The bank then issues a
[readiness warning](../GLOSSARY.md#warm-up-and-readiness),
`ReadinessWarning`, that names the gate and gives a figure to change.

1. **Read the warning's last sentence.** It says what to change: lower
   `min_weight` below the ceiling it gives, or increase `half_life` above
   the value it gives.
2. **Change one of the two in the spec.** A lower `min_weight` predicts
   sooner on less weight. A longer `half_life` keeps the gate and
   remembers more rows.
3. **Run the query again, and check that the warning is gone and the
   model predicts.**

[Is it ready?](../DIAGNOSTICS.md#is-it-ready) gives each form of the
warning, such as one for a target the rows stopped carrying, and what to
do about each. This code fits a row every five clock units under a
`min_weight` of 30, catches the warning, and fits again with each remedy:

```python
import warnings

# y = 2x plus noise, a row every five clock units.
rng = np.random.default_rng(2)
t = np.arange(0.0, 1000.0, 5.0)
x = rng.standard_normal(len(t))
pl.DataFrame({"t": t, "x": x, "y": 2 * x + rng.standard_normal(len(t))}).write_parquet("sparse.parquet")
sparse = pl.scan_parquet("sparse.parquet")

# 1. A min_weight of 30 at a half-life of 50: the warning's remedy, its last sentence.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sparse.online.fit_predict([po.spec.ewridge("weighted", half_life=50.0, min_weight=30.0, **common)]).collect()
for w in caught:
    if issubclass(w.category, po.ReadinessWarning):
        print("warning: " + str(w.message).split(". ")[-1])

# 2 and 3. Each remedy: the first row predicted, the weight behind it, and the warnings left.
for name, gates in (("lower", dict(half_life=50.0, min_weight=14.0)), ("longer", dict(half_life=110.0, min_weight=30.0))):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = sparse.online.fit_predict([po.spec.ewridge(name, **gates, **common)]).collect()
    first = out.select("t", pl.col(name).struct.unnest()).filter(pl.col("pred_y").is_not_null()).row(0, named=True)
    left = sum(issubclass(w.category, po.ReadinessWarning) for w in caught)
    print(f"{name}: first predicts at t = {first['t']:g}, weight_sum {first['weight_sum']:.1f}, {left} warnings")
```

```text
warning: Lower min_weight below 14.9, or increase half_life above 103, as https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready explains.
lower: first predicts at t = 205, weight_sum 14.1, 0 warnings
longer: first predicts at t = 425, weight_sum 30.0, 0 warnings
```

At a row every five clock units and a half-life of 50, `weight_sum` tops
out near 14.9, so a `min_weight` of 30 withholds every row. A
`min_weight` of 14 predicts from `t = 205`. A half-life of 110 reaches
a weight of 30 at `t = 425`, 220 clock units after the lower `min_weight`'s
first prediction, on twice the weight.

## Check that it worked

Read `withheld_reason` and the first predicted row again:

| what you see | what it means | what to do |
|---|---|---|
| no prediction on any row, and a readiness warning | the gate's floor is above what the stream can reach | follow the warning's remedy, as [Act on a readiness warning](#act-on-a-readiness-warning) does |
| no prediction on any row, and no warning yet | the stream is young: the warning waits until the model is within 5% of its steady state and the gate has withheld every row for a further half-life | feed more rows, or read `weight_sum` and `settled_frac` against the gate |
| `above_max_error_inflation` on rows long after the warm-up | on `ewridge`, `rls` or `lasso`, Kish's effective sample size behind the fit (its weight, discounted for unevenness) has fallen against its degrees of freedom, as after a gap or under uneven weights; on `kalman`, the row leans on a direction of the features the data rarely showed, or follows a long clock gap | read the gate's own ratio, `bank.summary()`'s `error_inflation`; on `ewridge`, `rls` or `kalman`, also `error_inflation_<t>` (`emit_error_inflation=True`) on those rows |
| early predictions whose error is well above the later ones' | the gate opens before the estimation error has fallen | lower `max_error_inflation`, or increase `min_settled_frac` or `min_weight` |
| `ill_conditioned` or `unstable_rate` | the fit itself cannot be trusted on the row: an `ewridge` solve too near singular, or an `sgd` learning rate past its stability bound | [Is it ready?](../DIAGNOSTICS.md#is-it-ready) says what each reads |
