<!-- kind: how-to -->
# Compare two models

This how-to is for anyone with two models fitted on the same stream who
wants to know whether one predicts better than the other, or only seemed
to. It tests whether their errors differ with Diebold and Mariano's test, or
with Clark and West's when one model nests the other. A test by betting
gives evidence that can be read at any row.

## Prerequisites

These steps need the library and two models' predictions on the same rows:

- polars-online installed, and numpy to make the examples' data.
- Two [specs](../GLOSSARY.md#the-model-bank-and-what-it-reads) fitted in
  the same [bank](../GLOSSARY.md#the-model-bank-and-what-it-reads), so
  their output columns sit side by side on each row. The tests read each
  row's [out-of-sample](../explanation/out-of-sample.md) residuals, so a
  model is judged only on rows neither has learned.

## Test whether one model's error is lower

Use Diebold and Mariano's test when neither model holds the other's
features, and Clark and West's when the larger model holds every feature
of the smaller one and more. Between two such nested models, Diebold and
Mariano's test leans toward the smaller one: the larger model still
estimates its extra coefficients, and their noise adds to its error. Clark
and West's test adds that noise back. [Comparing two
models](../DIAGNOSTICS.md#comparing-two-models) defines both statistics.

1. **Fit both specs in one bank.**
2. **Run the query.**
3. **For two models that are not nested, call
   `po.eval.diebold_mariano(out, a=, b=)`.** It returns `dm`, positive
   when `a` predicts closer, and its two-sided `p_value`. For a one-sided
   test at 5% that `a` predicts closer, read `dm > 0` with
   `p_value < 0.10`. `lags=`, the lags of its Newey-West variance, is 0
   by default, for one row ahead; for a target that looks `h` rows ahead,
   give at least `h − 1`.
4. **For two nested models, call `po.eval.clark_west(out, big=, small=)`.**
   It returns a one-sided `p_value`: below 0.05, the larger model predicts
   better at the 5% level.
5. **Pass `group=` to either call** for one test per
   [group](../GLOSSARY.md#the-model-bank-and-what-it-reads).

This code compares two nested models on 200 streams, one group each. On
half of them the larger model's extra coefficient is 0.05, and on the
other half it is 0. It prints the share of streams on which each test
finds the larger model better:

```python
import numpy as np
import polars as pl

import polars_online as po

# 200 streams of 2,000 rows: y = x0 + c x1 + noise, with c = 0.05 on half of them and 0 on the rest.
# "small" fits x0 alone; "big" fits x0 and x1, so it nests "small".
rng = np.random.default_rng(10)
streams, n = 200, 2_000
c = np.repeat([0.05, 0.0], streams // 2)
x0, x1, e = rng.standard_normal((3, streams, n))
pl.DataFrame(
    {
        "stream": np.repeat(np.arange(streams), n),
        "c": np.repeat(c, n),
        "x0": x0.ravel(),
        "x1": x1.ravel(),
        "y": (x0 + c[:, None] * x1 + e).ravel(),
    }
).write_parquet("nested.parquet")
nested = pl.scan_parquet("nested.parquet")

# 1 and 2. The two nested fits in one bank, one group per stream, every row weighed alike.
common = dict(targets=["y"], half_life=float("inf"), group="stream", min_weight=20.0)
small = po.spec.ewridge("small", features=["x0"], **common)
big = po.spec.ewridge("big", features=["x0", "x1"], **common)
out = nested.online.fit_predict([small, big]).collect()

# 3 to 5. One test of each kind per stream: does big predict better than small?
dm = po.eval.diebold_mariano(out, a="big", b="small", group="stream")
cw = po.eval.clark_west(out, big="big", small="small", group="stream")
tests = (
    out.group_by("stream").agg(pl.col("c").first())
    .join(dm.select("stream", "dm", dm_p="p_value"), on="stream")
    .join(cw.select("stream", cw_p="p_value"), on="stream")
)
# The share of streams on which each test finds big better, by the true c.
found = tests.group_by("c").agg(
    diebold_mariano=((pl.col("dm") > 0) & (pl.col("dm_p") < 0.10)).mean(),  # one-sided at 5%
    clark_west=(pl.col("cw_p") < 0.05).mean(),
)
for c_, dm_share, cw_share in found.sort("c", descending=True).iter_rows():
    print(f"extra coefficient {c_:.2f}: big found better on {dm_share:.0%} of streams by "
          f"Diebold-Mariano, {cw_share:.0%} by Clark-West")
```

```text
extra coefficient 0.05: big found better on 1% of streams by Diebold-Mariano, 44% by Clark-West
extra coefficient 0.00: big found better on 0% of streams by Diebold-Mariano, 2% by Clark-West
```

Where the extra coefficient is 0.05, Diebold and Mariano's test finds the
larger model better on 1% of the streams, below its own 5% size. The lean
toward the smaller model swamps a small real gain. Clark and West's test
finds it on 44%. Where the extra coefficient is 0, Clark and West's test
rejects on 2%, within its 5% size.

Two hundred streams made the same way from five more seeds (11 to 15)
give
the same picture. Where the extra coefficient is 0.05, Diebold and
Mariano's test found the larger model better on 0% to 3% of the streams,
and Clark and West's on 36% to 47%. Where it is 0, Clark and West's test
rejected on 2% to 6%.

## Read the comparison on every row

A test computed once over the stream answers at its last row. To act the
first time the evidence is enough, read a test whose evidence can be read
at any row, as often as you like. A p-value read repeatedly inflates its
error rate. The evidence a `seqtest` keeps does not.

1. **Name the two specs in a `seqtest` spec's `a` and `b`.** The sign it
   tests is
   `|resid_b| − |resid_a|`, positive when `a` came closer. A row where
   either residual is null, as in a warm-up, sits out the test.
2. **Put the two specs and the `seqtest` spec in one bank.**
3. **When a side is a grid, pick its residual field with `a_suffix` or
   `b_suffix`:** the half-life's suffix, such as `"@h500"`, or a grid
   point's, such as `"__r0.5@h500"` ([Find an output field by what it
   holds](find-an-output-field.md)). Without the suffix, the
   bank refuses the spec and lists the grid's residual fields.
4. **Read `log_e_a_<t>` on each row,** and reject the first time it
   reaches `ln 20`: that is the 5% level, however often you read it.
   `log_e_b_<t>` is the evidence that `b` came closer.

For two specs' output already in a frame, `po.eval.seqtest(out, a=, b=)`
computes the same evidence in Polars expressions. This code makes two
stocks: on `moves` the coefficient on `x0` wanders, and on `still` it
stays at 1. A Kalman filter whose coefficients drift should come closer on
`moves`, and a ridge on `still`. Both specs set `min_weight`, so the test
reads no row of either model's warm-up:

```python
import math

# Two stocks, a row a second each: y = b x0 + 0.5 x1 + noise, where b wanders on "moves" and stays at 1 on "still".
rng = np.random.default_rng(13)
n = 20_000
x0, x1, e = rng.standard_normal((3, 2, n))
b = np.stack([1.0 + np.cumsum(rng.normal(0, 0.01, n)), np.ones(n)])
pl.DataFrame(
    {
        "t": np.tile(np.arange(n, dtype=float), 2),
        "stock": np.repeat(["moves", "still"], n),
        "x0": x0.ravel(),
        "x1": x1.ravel(),
        "y": (b * x0 + 0.5 * x1 + e).ravel(),
    }
).sort("t", "stock").write_parquet("drift.parquet")
drift = pl.scan_parquet("drift.parquet")

# 1 and 2. A ridge, a Kalman filter whose coefficients drift, and a seqtest of the two, one model each per stock.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, group="stock", min_weight=50.0)
ridge = po.spec.ewridge("ridge", half_life=500.0, **common)
kalman = po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **common)
closer = po.spec.seqtest("closer", targets=["y"], a="kalman", b="ridge", group="stock")  # does kalman predict closer?
out = drift.online.fit_predict([ridge, kalman, closer]).collect()

# 4. The first row on which each side's evidence reaches ln 20, the 5% level, on each stock.
evidence = out.select("t", "stock", pl.col("closer").struct.unnest())  # log_e_a_y, log_e_b_y, wins_a_y, wins_b_y
for stock in ("moves", "still"):
    rows = evidence.filter(stock=stock)
    for side, name in (("a", "kalman"), ("b", "ridge")):
        first = rows.filter(pl.col(f"log_e_{side}_y") >= math.log(20))["t"].min()
        print(f"{stock}: {name} closer at the 5% level " + (f"from t = {first:.0f}" if first is not None else "on no row"))

# The same test with each model's default min_weight: on "still", the ridge's evidence early on.
ungated = {k: v for k, v in common.items() if k != "min_weight"}
early = (
    drift.online.fit_predict([po.spec.ewridge("ridge", half_life=500.0, **ungated),
                              po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **ungated), closer])
    .collect().select("t", "stock", pl.col("closer").struct.unnest()).filter(stock="still")
)
first = early.filter(pl.col("log_e_b_y") >= math.log(20))["t"].min()
later = early.filter(t=100.0)["log_e_b_y"].item()
print(f"still, default min_weight: ridge closer at the 5% level from t = {first:.0f}; its log_e_b_y at t = 100: {later:.2f}")
```

```text
moves: kalman closer at the 5% level from t = 15310
moves: ridge closer at the 5% level on no row
still: kalman closer at the 5% level on no row
still: ridge closer at the 5% level from t = 14248
still, default min_weight: ridge closer at the 5% level from t = 25; its log_e_b_y at t = 100: 0.28
```

Each test finds the model the stock was made for, and neither finds the
other. At each model's default `min_weight`, the test on `still` passes the 5%
level for the ridge at `t = 25`, in the first rows the two models
predict. By `t = 100` its evidence has fallen back to
0.28, far below `ln 20`.

[`seqtest`](../MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting)
states how its evidence grows, and why `ln 20` is the 5% level at any row.

## Check that it worked

A test of two models is only as good as the rows it reads:

| what you see | what it means | what to do |
|---|---|---|
| Diebold and Mariano's test never finds a larger, nested model better | the test leans toward the smaller model | use `po.eval.clark_west` for nested models |
| a significant `dm` on a target that looks ahead, which goes away at a larger `lags` | the overlapping targets correlate the residuals, and too few lags understate the standard error | set `lags` to at least `h − 1` for a target `h` rows ahead |
| a test that finds a difference on every one of many streams, or on none | the difference is large against the noise, or the streams are too short to see it | read the size of the difference too, as the two models' `mse` from `po.eval.compare_specs` |
| a `seqtest` that passes the 5% level in the first rows and falls back | one model is still in its warm-up, and its first predictions are far off | set `min_weight` on both specs, so the test reads only rows both models are ready on |
| `log_e_a_<t>` stays null or flat | one side's residual is null on the rows, as in a warm-up | check that both specs predict the rows |
