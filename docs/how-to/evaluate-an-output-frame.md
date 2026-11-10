<!-- kind: how-to -->
# Evaluate an output frame

This how-to is for anyone who has run a bank over a stream and wants to
score its predictions afterward: per spec, per group or over windows of
the clock. It also scores an output too large to hold in memory, by
passing the query in place of its result.

## Prerequisites

These steps need the library and a bank's output:

- polars-online installed, and numpy to make the examples' data.
- The output of a [bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  run: a frame with each spec's
  [output column](../GLOSSARY.md#what-a-bank-writes) and each target's
  column, or for an output too large to hold, the bank and the query that
  feeds it.

## Score the output by spec, group and window

`po.eval` reads each [slot](../GLOSSARY.md#grids)'s
[out-of-sample](../explanation/out-of-sample.md) predictions beside its
target, and gives R², IC (the correlation of prediction with target), hit
rate (the share of rows whose sign the prediction got right) and MSE.
[Is another model better?](../DIAGNOSTICS.md#is-another-model-better)
defines each, and what they mean on a logistic fit.

1. **Run the query with `.collect()`, which returns its result as a frame
   in memory, or pass the query itself.** The calls in the steps after
   this one take either, and read a query a chunk at a time ([Score an
   output too large to hold](#score-an-output-too-large-to-hold)).
2. **Compare several specs in one table with
   `po.eval.compare_specs(out, [...])`.** It gives one row per spec and
   slot.
3. **Score one spec per group with `po.eval.metrics(out, spec,
   group=...)`.**
4. **Score one spec over windows of the clock with
   `po.eval.window_metrics(out, spec, clock=..., every=...)`.** Each window
   is a tumbling span of `every` clock units, from `window_start`.
5. **Test whether the difference between two specs is more than noise,**
   with `po.eval.diebold_mariano` or `po.eval.seqtest` ([Compare two
   models](compare-two-models.md)).

For a `group_by` of your own, `po.eval.unpack(out, "ridge")` gives one row
per row and slot, with its target, `pred` and `y`. This code fits a ridge
and a Kalman filter over three stocks whose coefficients wander, and
scores them each way:

```python
import numpy as np
import polars as pl

import polars_online as po

# Three stocks, a row a second each, y = b x0 + 0.5 x1 + noise, where each stock's b wanders.
rng = np.random.default_rng(13)
n = 20_000
x0, x1, e = rng.standard_normal((3, 3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.01, (3, n)), axis=1)
pl.DataFrame(
    {
        "t": np.tile(np.arange(n, dtype=float), 3),
        "stock": np.repeat(["A", "B", "C"], n),
        "x0": x0.ravel(),
        "x1": x1.ravel(),
        "y": (b * x0 + 0.5 * x1 + e).ravel(),
    }
).sort("t", "stock").write_parquet("stocks.parquet")
stocks = pl.scan_parquet("stocks.parquet")

# 1. A ridge and a Kalman filter over the same three stocks.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, group="stock")
ridge = po.spec.ewridge("ridge", half_life=500.0, **common)
kalman = po.spec.kalman("kalman", half_life=500.0, coef_half_life=100.0, **common)  # coefficients that drift
out = stocks.online.fit_predict([ridge, kalman]).collect()

# 2. One table, many specs: which had the lower error?
for spec, n_, r2, ic, hit, mse in po.eval.compare_specs(out, [ridge, kalman]).select("spec", "n", "r2", "ic", "hit_rate", "mse").iter_rows():
    print(f"{spec:7} {n_} rows: R² {r2:.4f}, IC {ic:.4f}, hit rate {hit:.4f}, mse {mse:.4f}")
# 3 and 4. The same metrics per group, and per tumbling window of the clock.
for stock, r2 in po.eval.metrics(out, "ridge", group="stock").select("stock", "r2").iter_rows():
    print(f"ridge on {stock}: R² {r2:.4f}")
for start, r2 in po.eval.window_metrics(out, "ridge", clock="t", every=5_000.0).select("window_start", "r2").iter_rows():
    print(f"ridge from t = {start:>6.0f}: R² {r2:.4f}")
# 5. Is kalman closer? Diebold and Mariano's t, and the evidence by betting at the last row.
dm, p = po.eval.diebold_mariano(out, a="kalman", b="ridge").select("dm", "p_value").row(0)
log_e = po.eval.seqtest(out, a="kalman", b="ridge").select(pl.col("seqtest").struct.field("log_e_a_y")).max().item()
print(f"kalman against ridge: dm {dm:.2f}; the betting evidence peaked at log_e_a {log_e:.2f} (ln 20 = 3.00)")
```

```text
ridge   59988 rows: R² 0.7899, IC 0.8889, hit rate 0.8255, mse 1.0341
kalman  59988 rows: R² 0.7927, IC 0.8903, hit rate 0.8264, mse 1.0200
ridge on A: R² 0.8089
ridge on B: R² 0.8525
ridge on C: R² 0.5585
ridge from t =      0: R² 0.6802
ridge from t =   5000: R² 0.7853
ridge from t =  10000: R² 0.8200
ridge from t =  15000: R² 0.8240
kalman against ridge: dm 11.58; the betting evidence peaked at log_e_a 8.54 (ln 20 = 3.00)
```

`kalman`, whose coefficients drift, predicts a little better than `ridge`
here, with an error of 1.0200 against 1.0341. The difference is small, but
it holds on enough rows that Diebold and Mariano's t reads 11.6. The
betting evidence peaks at 8.54, past `ln 20`, the 5% level however often
it is read. R² differs between stocks because each stock's coefficient
wandered its own way, and an R² is larger where the coefficient is larger in size
against the same noise. The first window's R² is the lowest because it
holds the fit's warm-up.

## Score an output too large to hold

When the output is too large to hold, say fifty slots over a billion rows,
give `po.eval.metrics` the query in place of its result. A query runs only
when its result is asked for, and `metrics` runs it once and reads the
result a chunk at a time, keeping ten numbers per slot and group.

What the process holds is that chunk and the sums, and on top of them
what Polars has read ahead of the call: a scan reads blocks of the file
ahead of a slower consumer, sized from its thread count. Scoring a scan of
a 20-slot output held 494 MiB at a million rows and 1,012 MiB at four
million under Polars' defaults. With four threads and the read-ahead cut
to one block, it held 205 and 208 MiB:
[Cut what Polars reads ahead](bound-memory.md#cut-what-polars-reads-ahead)
says how.

1. **Write the bank's query, `query.online.fit_predict([spec])`,** or
   scan a file a run wrote its output to, with `pl.scan_parquet`.
2. **Pass the query to `po.eval.metrics(query, spec, group=...)`.** It
   returns the same table as for the output held in memory, within the
   rounding of adding the chunks' sums. Every other `po.eval` call that
   reads a bank's output takes a query the same way: `window_metrics`,
   `compare_specs`, `diebold_mariano`, `clark_west`, `seqtest`, `unpack`,
   and `sums`, which takes `weight=` to weight the rows by a column.
3. **To add up the scores of several runs, such as one output file a day,
   reduce each with `po.eval.sums`, add them with `po.eval.merge_sums`,
   and read the metrics with `po.eval.from_sums`.** The total is the same,
   to rounding, however the rows were split.

The sums are centered, so a target far from zero keeps its variance,
where raw `Σy` and `Σy²` would lose all of it. This code scores a target
near 1e8 with unit spread over the bank's query, and compares the result
with the metrics of the whole output held in memory:

```python
# A target near 1e8 with unit spread, three stocks, 30,000 rows.
rng = np.random.default_rng(15)
n = 30_000
x0, x1, e = rng.standard_normal((3, n))
pl.DataFrame(
    {"stock": rng.choice(["A", "B", "C"], n), "x0": x0, "x1": x1, "y": 1e8 + 1.0 * x0 + 0.5 * x1 + e}
).write_parquet("far.parquet")
far = pl.scan_parquet("far.parquet")

ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=500.0, group="stock")

# 1 and 2. The bank's query, scored a chunk at a time: one chunk and a few sums per stock are held.
streamed = po.eval.metrics(far.online.fit_predict([ridge]), ridge, group="stock")

# The same metrics from the whole output held in memory, for comparison.
held = po.eval.metrics(far.online.fit_predict([ridge]).collect(), "ridge", group="stock")
for (stock, n_, r2, mse), (r2_held, mse_held) in zip(
    streamed.select("stock", "n", "r2", "mse").iter_rows(), held.select("r2", "mse").iter_rows()
):
    print(f"{stock}: {n_} rows, R² {r2:.6f} streamed and {r2_held:.6f} held, mse {mse:.6f} and {mse_held:.6f}")

# 3. The output written to a file, its two halves reduced apart, as two runs' files would be, and added up.
far.online.fit_predict([ridge]).sink_parquet("far_out.parquet")
written = pl.scan_parquet("far_out.parquet")
halves = [po.eval.sums(written.head(n // 2), ridge, group="stock"), po.eval.sums(written.tail(n // 2), ridge, group="stock")]
added = po.eval.from_sums(po.eval.merge_sums(*halves))
print(", ".join(f"{stock} R² {r2:.6f}" for stock, r2 in added.select("stock", "r2").iter_rows()), "from the two halves")
```

```text
A: 10160 rows, R² 0.545017 streamed and 0.545017 held, mse 1.017160 and 1.017160
B: 9801 rows, R² 0.551971 streamed and 0.551971 held, mse 1.006657 and 1.006657
C: 10027 rows, R² 0.551899 streamed and 0.551899 held, mse 1.017914 and 1.017914
A R² 0.545017, B R² 0.551971, C R² 0.551899 from the two halves
```

The metrics of the query agree with the ones computed from the whole
output to every digit printed, and so do the two halves' sums added up.
Only one chunk of the output is in memory at a time, beside what Polars
reads ahead of it.

## Score a target that is renamed or looks ahead

The output frame does not record how a target was formed, so a call
given a spec's name scores each slot against the column named after the
slot's target. A target renamed with `po.target(column, name=...)` names
no column, so give the call the spec itself.

1. **Pass the spec dict in place of its name** to `metrics`,
   `window_metrics`, `sums` or `unpack`, and a list of spec dicts to
   `compare_specs`. Each slot is then scored against the column its target
   reads.
2. **For a target that looks ahead, a
   [formula target](../GLOSSARY.md#fitting-a-model), add its column with
   `po.stream.with_windows(out, expr, like=spec)` before scoring.** Here
   `expr` is the expression you gave in `targets=`, aliased to the
   target's name (the slot `pred_<name>` less its `pred_`). The bank computes the target
   as it learns and does not write it out, so the call scores the slot
   against a column of the target's name, and refuses a frame without
   one.

This code fits a target renamed `ret`, whose column is still `y`. It
scores it by the spec's name, which fails, and then by the spec itself. It
uses `stocks` from the code in [Score the output by spec, group and
window](#score-the-output-by-spec-group-and-window):

```python
# 1. A target renamed with po.target: its fields say "ret", but its column is y.
renamed = po.spec.ewridge(
    "renamed", targets=[po.target("y", name="ret")], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=500.0,
    group="stock",
)
out = stocks.online.fit_predict([renamed]).collect()
try:
    po.eval.metrics(out, "renamed")  # by name: no column is named ret
except ValueError as e:
    print(f"{type(e).__name__}: {e}")
# The spec itself: each slot scored against the column its target reads.
slot, n_, r2 = po.eval.metrics(out, renamed).select("slot", "n", "r2").row(0)
print(f"{slot}: {n_} rows, R² {r2:.4f}")
```

```text
ValueError: po.eval.metrics: no column of the frame is named after slot 'pred_ret', so its target column is not known. Add the target column to the frame, or, for a target named with po.target(name=), pass the spec itself as spec.
pred_ret: 59988 rows, R² 0.7899
```

Given the spec, the call scores `pred_ret` against `y`, and reads the
same R² as the ridge of the first recipe, which fits `y` under its own
name.

## Check that it worked

Check the scores against what the run should give:

| what you see | what it means | what to do |
|---|---|---|
| a call says no column of the frame is named after a slot, or that the frame has no column for a formula target | a target was renamed with `po.target(..., name=)`, or the target is a formula target, which looks ahead, so the bank does not write it out | for a renamed target, pass the spec dict in place of its name, or a list of spec dicts to `compare_specs`; for a formula target, add its column with `po.stream.with_windows` before scoring, as step 2 of [Score a target that is renamed or looks ahead](#score-a-target-that-is-renamed-or-looks-ahead) does |
| two specs scored on different numbers of rows | each spec is scored on its own rows, and their warm-ups differ in length | compare them with `po.eval.diebold_mariano`, which reads only the rows where both have a residual |
| the first window's R² well below the rest | the window holds the fit's warm-up | read the windows after it, or set a gate ([Decide when a model is ready](decide-when-a-model-is-ready.md)) |
| a `hit_rate` of 1.0 whatever the fit | the target sits about 1, as a plain ratio does, where two positive numbers always agree | write the target to sit about zero, as the ratio less 1 or its `.log()` |
