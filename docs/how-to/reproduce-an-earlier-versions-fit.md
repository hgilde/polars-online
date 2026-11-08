<!-- kind: how-to -->
# Reproduce an earlier version's fit

This how-to is for anyone who needs the numbers an earlier release of
polars-online gave for a spec, such as to compare a fit made before an
upgrade with one made after it.

Most changes to a model's numbers between releases have no setting that
undoes them. A few have one: the [CHANGELOG](../../CHANGELOG.md) entry
that changed a default names the setting that gives back the numbers of
the release before. To go on from a fit an earlier release saved, load
its [state](../GLOSSARY.md#saving-and-serving) instead, as [Resume from a
saved state](resume-from-a-saved-state.md) shows. A state saved by 1.0.0
or later loads and continues as it was fitted. One saved by 0.13.0 or
earlier is refused, so refit its rows under the settings on this page.

## Prerequisites

These steps need the earlier release's number and the spec it ran:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  `ticks.parquet`, 400 rows that Example data writes.
- The release whose numbers you want, such as 1.3.0, and the
  [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) you ran under it.

## Give the setting that undoes a change

A setting gives back an earlier release's numbers only where the CHANGELOG
names it as the way back, so start from the CHANGELOG.

1. **Read every CHANGELOG entry for your spec's model between the earlier
   release and this one.** Each entry that moves the model's numbers says
   by how much, and an entry with a way back names its setting.
2. **Give the spec each setting the entries name.** These are the
   settings the CHANGELOG names as a way back:

   | to get the numbers of | for | give | what it gives back |
   |---|---|---|---|
   | 1.4.0 | `huber`, `quantile` and `sgd(loss="huber")` | `refit_budget=256 * (features + 26) / (features + 20)`, in MiB; `refit_budget=256` under `huber_scale="rms"` | 1.4.0's groups that refit, which its default of 256 MiB held (see *The refit's budget*) |
   | 1.4.0 | `corrchange(kind="window")` | `permute_every_rows=50` | 1.4.0's schedule of draws for the critical value: one every 50 rows, where the default is one every `window_rows`. The statistic itself still moves in its last bits, since it is now merged from blocks of rows |
   | 1.3.0 and earlier, before the refit existed | `huber` | `huber_refit_rows=0` | numbers close to those releases', with the median absolute deviation (MAD) read from fixed buckets, as those releases read it ([How a stream keeps a median](../explanation/robust-scale.md#how-a-stream-keeps-a-median)) |
   | 1.3.0 and earlier, before the refit existed | `quantile` | `quantile_refit_rows=0` | numbers close to those releases', from the fit made in one pass |
   | 1.2.0 and earlier, before `huber_scale` existed | `huber` | `huber_scale="rms"`, and `huber_refit_rows=0` | the scale those releases read: the residuals' root mean square, with no median to keep |
   | 0.13.0 | `ew_cov` | `bias=True` | the variances without the correction `bias=False`, the default, applies: Kish's `n/(n-1)` |
   | 0.13.0 | a spec under `window_size` | `closed="both"` | the [hard window](../GLOSSARY.md#fitting-a-model)'s old edge, to the bit: a row exactly `window_size` old stays in |
   | 0.13.0 | a spec with `emit_drift` and a clock | `drift_threshold=20.0` on a numeric clock, `"20s"` on a temporal one | the drift test's old threshold, which every temporal clock read in seconds |

3. **Fit the spec as before.** Nothing else in the run changes.

**Each setting undoes one change, at the release that made it.** A later
entry can move the same model's numbers again, with no setting to undo it,
so step 1 reads every entry between the two releases. Under
`huber_scale="rms"`, with the default `min_weight` on unit weights,
`huber_refit_rows=0` gave 1.3.0's numbers to the bit on the stream it was
tested on. Away from the default `min_weight` on unit weights, 1.4.0
moved `huber`'s first rows under `min_weight`, with no setting to undo
it.

**The refit's budget gives back 1.4.0's groups, not every number.**
`refit_budget` bounds the rows that the refits of all of a spec's groups
hold, and 1.4.0's default was 256 MiB. Each group reserves the rows its
refit can hold when it first appears. The group that would pass the bound
keeps its one-pass fit, as does every new group after it. Under the MAD,
each held row now also keeps its one-pass residual and what the exact
median is read from. So a group reserves `(features + 26) · 8` bytes a
target for each row of `huber_refit_rows` or `quantile_refit_rows`, where
1.4.0 reserved `(features + 20) · 8`. At 256 MiB that covers about a fifth
fewer groups than 1.4.0 did: at 5 features and 500 rows, 2,164 groups of
one target against 2,684. A budget increased in the same ratio holds the same number
of groups, and so the same groups, since they reserve in the order they
first appear: at 5 features, `refit_budget=317.44` refits the first 2,684
such groups. For a budget `B` given to 1.4.0, give
`B · (features + 26) / (features + 20)`.
Under `huber_scale="rms"` the reservation did not grow, so
`refit_budget=256` is 1.4.0's. Either way the rows before each refit read
the scale this release reads, with no setting to undo it.

For `marginal`'s Huber bins, whose reservation grew by a different share,
scale 256 by the ratio of what a group reserves in each release. Each
release's readiness warning names it in units of 1,024 bytes: "KB a group"
under 1.4.0, "each group reserving … KiB" now. Both are rounded, so this
comes close to 1.4.0's groups but is not exact. Each warning also names
the first group that gave its refit up, so compare that group across the
two runs.

**`sgd(loss="huber")` has no way back past 1.3.0.** `sgd`'s steps have changed
since (its clip, AdaGrad's epsilon and its intercept read the target's own
spread and level). So for `sgd(loss="huber")`, `huber_refit_rows=0` gives
the one-pass fit of its current steps, which no earlier release gave.

This code fits `huber` with both settings of 1.2.0's row, on
`ticks.parquet` from [Example data](../../README.md#example-data), and
writes its output to `huber-now.parquet`:

```python
# huber as 1.2.0 fitted it: the residuals' root mean square as its scale, and no refit.
as_in_1_2_0 = po.spec.huber("robust", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
                        huber_scale="rms",          # the scale before huber_scale existed
                        huber_refit_rows=0)         # no refit of the first rows
pl.scan_parquet("ticks.parquet").online.fit_predict([as_in_1_2_0]).sink_parquet("huber-now.parquet")
```

## Check the result

**Compare the predictions with the earlier release's, row by row, on the
same input.**

1. **Install the earlier release in an environment of its own,** so that
   it does not replace this one:

   ```sh
   python -m venv old-env
   old-env/bin/python -m pip install "polars-online==1.2.0"
   ```

   On Windows the environment's Python is `old-env\Scripts\python`.
2. **Run the same spec there, without the settings it did not have yet.**
   Save this code as `fit_old.py`. It is the spec of the example, with no
   `huber_scale` and no `huber_refit_rows`, and it writes its output to
   `huber-1.2.0.parquet`:

   ```python
   import polars as pl
   import polars_online as po

   then_spec = po.spec.huber("robust", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
   pl.scan_parquet("ticks.parquet").online.fit_predict([then_spec]).sink_parquet("huber-1.2.0.parquet")
   ```

   Run it with the earlier release's Python, in the directory that holds
   `ticks.parquet`:

   ```sh
   old-env/bin/python fit_old.py
   ```
3. **Compare each `pred_<target>` field of the two outputs.** This code
   prints the largest difference:

   ```python
   then = pl.read_parquet("huber-1.2.0.parquet")["robust"].struct.field("pred_y")   # the earlier release's output
   now = pl.read_parquet("huber-now.parquet")["robust"].struct.field("pred_y")
   print(f"largest difference: {(now - then).abs().max()}")
   ```

A difference of 0 is the earlier release's fit to the bit. A difference
beyond rounding comes from a change that one of the CHANGELOG entries
names, the ones step 1 of [Give the setting that undoes a
change](#give-the-setting-that-undoes-a-change) reads. A different Polars
or another platform can move the last bits.
