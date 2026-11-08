<!-- kind: how-to -->
# Fit on factor residuals

This how-to is for anyone who fits factor ridges and wants each factor's
residual (its actual less its estimate) as a feature of a second ridge. The
second ridge's target looks ahead, and both run in one query over a
stream.

A factor ridge regresses a factor's dependent on its inputs, and its
residual is the part of the dependent the inputs do not explain. The bank
writes it as `resid_<target>`, from a prediction made before it learns the
row, so the residual is [out-of-sample](../GLOSSARY.md#fitting-a-model):
it never reflects its own row's target ([Why every prediction is
out-of-sample](../explanation/out-of-sample.md)). The second ridge reads
those residuals as features. Its target is a forward exponentially weighted
mean of a series less the series itself, which is known only once its
window has passed, so the bank learns each row then.

## Prerequisites

The recipe builds its own data, so these steps need only:

- polars-online installed, and numpy, which the recipe uses to make up its
  rows.
- A stream in a file, read as a
  [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) (a chain of
  Polars steps that runs only when you ask for its result). It holds a
  [clock](../GLOSSARY.md#time), the dependent and inputs of each factor,
  and the series the target looks ahead on. Here a
  [group](../GLOSSARY.md#the-model-bank-and-what-it-reads) column `g`
  names each instrument, so every spec fits one model per instrument.

## Build the pipeline

Both stages run in one query: the factor ridges first, then the second
ridge on their residuals.

1. **Fit one `ewridge` for each factor,** with its dependent as the target
   and its inputs as the features, on the stream's clock and group.
2. **Take each factor's residual out of its output column,** with
   `pl.col("fac0").struct.field("resid_d0")` in a `with_columns` after the
   factor ridges' `fit_predict`. The field becomes a column named
   `resid_d0`.
3. **Make the second ridge's target with a
   [window operator](../GLOSSARY.md#fitting-a-model) that looks
   ahead,** as `po.ewm_mean("y", half_life=5.0, window_size=10.0,
   reverse=True) - pl.col("y")`: the forward mean of `y` over the next ten
   clock units, less `y`. Under the default `closed="right"`, the window
   holds the rows after the row, up to `window_size` later, and not the row
   itself.
4. **Give the second ridge a [hard window](../GLOSSARY.md#fitting-a-model)
   with `window_size`, and `half_life=float("inf")`** to weigh every row in
   the hard window alike. A spec with `window_size` and no `half_life` is
   refused, with a message that names both choices. Set `half_life` to
   `float("inf")` to weigh every row alike, or to a finite value to decay
   the older rows within the hard window.
5. **Give it an [`embargo`](../GLOSSARY.md#time) equal to the target's
   `window_size`.** The bank holds each row until the target's window has
   closed and the embargo has passed, then learns it. `fit_predict` refuses
   an `embargo` shorter than the window ([A target that looks
   ahead](../explanation/out-of-sample.md#a-target-that-looks-ahead)).
6. **Give it `emit_robust_se=True`, and `horizon_rows` equal to the
   target's window in rows.** Two rows less than a window apart share the
   moves their targets look ahead on, so their residuals overlap.
   `se_coef_hac`, Newey and West's standard error, allows for that over
   twice `horizon_rows` lags by default. On a spec with a clock column,
   `horizon_rows` is 0 unless you give it, since `embargo` is in clock
   units there.
7. **Chain the two `fit_predict` calls in one query, and unnest the output
   with `separator="."`.** `.online.unnest` turns each spec's output column
   into one column per field. The specs share field names such as
   `weight_sum`, so `separator="."` names each column after its spec too,
   as `stage2.coef_fwd_resid_d0`.
8. **Run the query with `sink_parquet`,** which runs it and writes the
   result to a file without holding it all in memory.

This code makes up two instruments of 3,000 rows each, with a planted
answer. Each factor's dependent is the sum of its inputs plus a move of
its own, `e`. The next step of `y` is `0.5 e0 − 0.3 e1` plus noise. Every
row in the target's window comes after the row, so each holds that step,
and the coefficients on the residuals are 0.5 and −0.3. It builds its own
rows:

```python
import numpy as np
import polars as pl

import polars_online as po

# The stream: per instrument g, a row each unit of the clock t.
rng = np.random.default_rng(3)
n = 3_000
parts = []
for g in ["a", "b"]:
    x = rng.standard_normal((2, 3, n))                   # each factor's three inputs
    e = rng.standard_normal((2, n))                      # each factor's move, which its inputs do not explain
    d = x.sum(axis=1) + e                                # each factor's dependent
    # The planted answer: y's next step is 0.5 e0 - 0.3 e1, plus noise.
    step = np.r_[0.0, 0.5 * e[0, :-1] - 0.3 * e[1, :-1]] + 0.1 * rng.standard_normal(n)
    parts.append(pl.DataFrame({"t": np.arange(n, dtype=float), "g": g, "y": np.cumsum(step)}
                 | {f"d{f}": d[f] for f in range(2)}
                 | {f"x{f}_{j}": x[f, j] for f in range(2) for j in range(3)}))
# The instruments' rows interleaved in clock order, saved to a file.
pl.concat(parts).sort("t", maintain_order=True).write_parquet("factors.parquet")

# 1. One ridge per factor, and 2. each one's residual as a column.
clock = dict(clock="t", gap_cap=10.0, group="g")
factors = [po.spec.ewridge(f"fac{f}", targets=[f"d{f}"], features=[f"x{f}_{j}" for j in range(3)],
                           half_life=500.0, **clock) for f in range(2)]
residuals = [pl.col(f"fac{f}").struct.field(f"resid_d{f}") for f in range(2)]
# 3-6. The second ridge: a target that looks ahead, under a hard window, learned once it is known.
forward = po.ewm_mean("y", half_life=5.0, window_size=10.0, reverse=True) - pl.col("y")
stage2 = po.spec.ewridge(
    "stage2", targets=[forward.alias("fwd")], features=["resid_d0", "resid_d1"],
    window_size=1_000.0, half_life=float("inf"),   # 4. every row of the last 1,000 clock units alike
    embargo=10.0,                                  # 5. the target's window_size
    emit_robust_se=True, horizon_rows=10,          # 6. the target's window, in rows
    min_weight=500.0, **clock,
)
# 7 and 8. Both stages in one query, unnested and written to a file.
(pl.scan_parquet("factors.parquet")
   .online.fit_predict(factors)
   .with_columns(residuals)
   .online.fit_predict([stage2])
   .online.unnest([*factors, stage2], separator=".")
   .sink_parquet("fitted.parquet"))
```

## Check that it worked

**Read each instrument's last coefficients on the residuals, beside the
planted ones and their `se_coef_hac`.** The coefficients sit on each
group's last row in each chunk, so take each instrument's last row that
holds them. This code reads `fitted.parquet`, which [Build the
pipeline](#build-the-pipeline) wrote:

```python
fitted = pl.read_parquet("fitted.parquet").filter(pl.col("stage2.coef_fwd_resid_d0").is_not_null())
for row in fitted.group_by("g", maintain_order=True).last().iter_rows(named=True):
    for f, planted in [(0, 0.5), (1, -0.3)]:
        print(f"{row['g']}, resid_d{f}: planted {planted:+.3f}, estimated {row[f'stage2.coef_fwd_resid_d{f}']:+.3f}"
              f" (se_coef_hac {row[f'stage2.se_coef_hac_fwd_resid_d{f}']:.3f})")
```

```text
a, resid_d0: planted +0.500, estimated +0.499 (se_coef_hac 0.032)
a, resid_d1: planted -0.300, estimated -0.296 (se_coef_hac 0.023)
b, resid_d0: planted +0.500, estimated +0.501 (se_coef_hac 0.032)
b, resid_d1: planted -0.300, estimated -0.282 (se_coef_hac 0.020)
```

**Expect the standard errors to be null at first.** A diagnostic reads a
row only once the model is [ready](../GLOSSARY.md#warm-up-and-readiness)
there, and once the stream has come 95%
of the way to its steady state on the model's memory. Under a hard window
with `half_life=float("inf")`, that memory is a half-life of about
`window_size / 2.885` clock units, so 95% takes about 1.5 hard windows. In
this recipe the second ridge has `se_coef_hac` from about `t` 1,500 on.

To check each stage with the diagnostics it asks for (calibration, and
Newey and West's standard error), see [Check a two-stage
fit](check-a-two-stage-fit.md).

## If `online.unnest` refuses the output

**`.online.unnest` raises Polars' `DuplicateError` while the query is built
when two columns would get the same name.** That happens where specs write
a field of the same name, as almost any two do, or where a field is named
like a column of the frame. The message names each shared field and the
specs, and gives the two remedies: pass `separator="."`, or take the
fields you need with `pl.col("<spec>").struct.field("<field>")`. This code
unnests the factor ridges' output without a separator. It reads
`factors` and `factors.parquet` from [Build the
pipeline](#build-the-pipeline):

```python
import textwrap

try:
    pl.scan_parquet("factors.parquet").online.fit_predict(factors).online.unnest(factors)
except pl.exceptions.DuplicateError as err:
    print(textwrap.fill(str(err), 100))
```

```text
online.unnest: weight_sum, settled_frac and withheld_reason are fields of specs "fac0" and "fac1",
so the columns unnested would share a name. Pass separator="." to name each column after its spec,
or take the fields you need with pl.col("fac0").struct.field("weight_sum").
```

## Keep the memory bounded

**Two `fit_predict` calls in series in one query hold a bounded amount of
memory,** since each hands Polars its output in slices of at most 1 MiB
([Why memory stays
bounded](../explanation/chunk-invariance-and-streaming.md#why-memory-stays-bounded)).

**A `with_columns` between `pl.scan_parquet` and a slower `fit_predict` is
not bounded the same way:** the query that reads the file runs ahead of
the bank. With
the memory allocator returning freed pages at once and Polars reading one
block of the file ahead, the second ridge alone over
`pl.scan_parquet(...).with_columns(...)` held 848 MB at 4,000,000 rows. The
same bank in a loop over the file's chunks, computing the columns in each
chunk, held
279 MB, measured on 2026-11-06 as the most memory the process held, by
`scripts/factor_residuals_bench.py`.
Add a computed column that the bank does not need after `fit_predict`.
Compute one that it does need in a loop over the file's chunks, where
`collect_batches` gives the file one chunk at a time. This code runs both
stages that way, computing the residuals in each chunk, and compares the
predictions with the query's. It reads `factors`, `residuals`, `stage2`
and the two files from [Build the pipeline](#build-the-pipeline):

```python
# The same two stages in a loop over the file's chunks.
factor_bank, second = po.ModelBank(factors), po.ModelBank([stage2])
outs = []
for chunk in pl.scan_parquet("factors.parquet").collect_batches(chunk_size=1_000):
    outs.append(second.fit_predict(factor_bank.fit_predict(chunk).with_columns(residuals)))
looped = pl.concat(outs)["stage2"].struct.field("pred_fwd")
print(f"the loop's predictions are the query's: {looped.equals(pl.read_parquet('fitted.parquet')['stage2.pred_fwd'])}")
```

```text
the loop's predictions are the query's: True
```

**The hard window keeps a snapshot of the running sums for every row it
covers,** so its memory grows with the rows in the window, in each group.
To keep fewer, give the second ridge `snapshot_every`: a snapshot every 10
clock units keeps a tenth of them, and the hard window then covers between
`window_size` less 10 and `window_size`. Measured on an Apple M4 Pro,
each snapshot added about 1.3 KB to the run's peak memory, so a two-hour
hard window at one row a second, 7,200 snapshots, held about 9 MB a group.
