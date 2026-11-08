<!-- kind: how-to -->
# Reproduce an earlier version's fit

This how-to is for anyone who needs the numbers an earlier release of
polars-online gave for a spec, such as to compare a fit made before an
upgrade with one made after it.

Most changes to a model's numbers between releases have no setting that
undoes them. A few have one: the [CHANGELOG](../../CHANGELOG.md) entry
that changed a default names the setting that gives back the numbers of
the release before. To go on from a fit an earlier release saved, you need
none of this. Load its state instead, as [Resume from a saved
state](resume-from-a-saved-state.md) shows, and it continues as it was
fitted.

## Prerequisites

These steps need the earlier release's number and the spec it ran:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The example reads
  `ticks.parquet`, 400 rows that Example data writes.
- The release whose numbers you want, such as 1.3.0, and the
  [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) you ran under it.
- For [Check the result](#check-the-result): that release installed in an
  environment of its own, and its output on the same rows.

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
   | 1.3.0 and earlier, before the refit existed | `huber` | `huber_refit_rows=0` | numbers close to those releases', with the MAD's sketch on the fixed buckets those releases used |
   | 1.2.0 and earlier, before `huber_scale` existed | `huber` | `huber_scale="rms"`, and `huber_refit_rows=0` | a scale that keeps no sketch |
   | 0.13.0 | `ew_cov` | `bias=True` | the variances without the correction `bias=False`, the default, applies: Kish's `n/(n-1)` |
   | 0.13.0 | a spec under `window_size` | `closed="both"` | the hard window's old edge, to the bit: a row exactly `window_size` old stays in |
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

**`sgd(loss="huber")` has no way back past 1.3.0.** `sgd`'s steps have changed
since (its clip, AdaGrad's epsilon and its intercept read the target's own
spread and level). So for `sgd(loss="huber")`, `huber_refit_rows=0` gives
the one-pass fit of its current steps, which no earlier release gave.

This code fits `huber` with both settings of 1.2.0's row, on
`ticks.parquet` from [Example data](../../README.md#example-data), and
writes its output to `huber-now.parquet`:

```python
# huber as 1.2.0 fitted it: a scale that keeps no sketch, and no refit.
earlier = po.spec.huber("robust", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
                        huber_scale="rms",          # the scale before huber_scale existed
                        huber_refit_rows=0)         # no refit of the first rows
pl.scan_parquet("ticks.parquet").online.fit_predict([earlier]).sink_parquet("huber-now.parquet")
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
2. **Run the same spec there, without the settings it did not have yet,**
   and write its output to a file of its own, such as
   `huber-1.2.0.parquet`.
3. **Compare each `pred_<target>` field of the two outputs.** This code
   prints the largest difference once both files exist:

   ```python
   from pathlib import Path

   if Path("huber-1.2.0.parquet").exists():               # the earlier release's output, written in step 2
       then = pl.read_parquet("huber-1.2.0.parquet")["robust"].struct.field("pred_y")
       now = pl.read_parquet("huber-now.parquet")["robust"].struct.field("pred_y")
       print(f"largest difference: {(now - then).abs().max()}")
   ```

A difference of 0 is the earlier release's fit to the bit. Any other
difference comes from a change that one of step 1's entries names.
