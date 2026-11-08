<!-- kind: how-to -->
# Measure heavy tails

This how-to is for anyone who sets a threshold on a model's residuals on a
stream, such as `drift_threshold`, and wants it to hold its false-alarm
rate. It measures the residuals' tails over a stretch known to be stable,
sets `drift_threshold` from them, and counts each detector's flags.

## Prerequisites

These steps need the library, a linear model and a stable stretch:

- polars-online installed, and numpy to make the example's data.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) of one of the
  ten linear models, such as `ewridge`, which take `emit_tails`,
  `emit_sigma` and `emit_drift`. [What each costs, and which models take
  it](../DIAGNOSTICS.md#what-each-costs-and-which-models-take-it) lists
  them.
- A stretch of the stream known to be stable, long enough to average a
  noisy kurtosis over: the recipe uses 19,000 rows.

## Set a threshold from the tails

A detector tuned on Gaussian residuals fires more often on residuals with
heavy tails, because large residuals are more common there, and nothing
has broken. So measure the tails before setting a threshold. [Are its
tails heavy?](../DIAGNOSTICS.md#are-its-tails-heavy) defines the fields
and gives the table of `drift` flags by kurtosis and threshold that step 3
reads.

1. **Turn on `emit_tails=True` and `emit_sigma=True` in the spec.** Each
   row gains `skew_<t>`, `kurtosis_<t>` (excess: 0 for a Gaussian),
   `jarque_bera_<t>` and `sigma_<t>` for each target `<t>`.
2. **Fit the stable stretch, and take the mean of `kurtosis_<t>` over it,
   from where the diagnostics start.** They start about 4.3 half-lives in,
   so the recipe reads from row 1,000 at a half-life of 200. A windowed
   kurtosis is itself noisy under heavy
   tails, so read its mean over a long stretch, not one row.
3. **Choose `drift_threshold` from that mean by the table:** 20 below 4,
   30 below 10, 50 below 30, and 80 at 30 or more.
4. **Take `sigma_<t>`'s median over the stretch as its level.** A rise of
   `sigma` past twice that level reads a change of spread on heavy tails.
5. **Fit the stream with `emit_drift=True` and the threshold chosen,** and
   count each detector's flags.

This code makes 200,000 stable rows whose noise is Student's t with 3
degrees of freedom, then 20,000 rows where the noise is three times as
large. It measures the tails over the first 20,000 rows, sets the
threshold, and counts each detector's flags before the break and its
first flag after it:

```python
import numpy as np
import polars as pl

import polars_online as po

# 200,000 rows of a stable relation whose noise is Student's t with 3 degrees of freedom, then
# 20,000 rows where the noise is three times as large.
rng = np.random.default_rng(6)
n, at = 220_000, 200_000
x0, x1 = rng.standard_normal((2, n))
noise = rng.standard_t(3, n) * np.where(np.arange(n) >= at, 3.0, 1.0)
pl.DataFrame({"x0": x0, "x1": x1, "y": 1.0 * x0 + 0.5 * x1 + noise}).write_parquet("tails.parquet")
tails_stream = pl.scan_parquet("tails.parquet").with_row_index("row")

# 1 and 2. Read the tails over a stretch known to be stable: the mean of kurtosis over rows 1,000-19,999.
stable = po.spec.ewridge(
    "m", targets=["y"], features=["x0", "x1"], half_life=200.0, emit_tails=True, emit_sigma=True,
)
tails = tails_stream.head(20_000).online.fit_predict([stable]).online.unnest([stable]).collect()
kurtosis = tails.filter(pl.col("row") >= 1_000)["kurtosis_y"].mean()
print(f"excess kurtosis of the residuals: {kurtosis:.1f}")

# 3 and 4. The threshold that kurtosis calls for, and sigma's level over the stretch.
threshold = 20.0 if kurtosis < 4 else 30.0 if kurtosis < 10 else 50.0 if kurtosis < 30 else 80.0
level = tails.filter(pl.col("row") >= 1_000)["sigma_y"].median()
print(f"drift_threshold {threshold:.0f}; sigma over the stretch {level:.2f}")


# 5. Count each detector's flags over the 200,000 stable rows, and find the break.
def report(name, out, flag):
    false = out.filter(pl.col("row") < at).select(flag.sum()).item()
    found = out.filter((pl.col("row") >= at) & flag)["row"].min()
    print(f"{name:18} {false:>6} flags before the break, first after it {'never' if found is None else f'{found - at} rows in'}")


# Drift at the default threshold, then at the one the kurtosis called for.
for value in [20.0, threshold]:
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], half_life=200.0,
        emit_drift=True, drift_threshold=value, emit_breaks=True, emit_sigma=True,
    )
    out = tails_stream.online.fit_predict([spec]).online.unnest([spec]).collect()
    report(f"drift at {value:.0f}", out, pl.col("drift_y"))
# Two other detectors, read off the last fit.
report("|cusum_sq| > 3", out, pl.col("cusum_sq_y").abs() > 3)
report("sigma past 2x", out, pl.col("sigma_y") > 2.0 * level)
```

```text
excess kurtosis of the residuals: 16.6
drift_threshold 50; sigma over the stretch 1.63
drift at 20            11 flags before the break, first after it 15011 rows in
drift at 50             1 flags before the break, first after it never
|cusum_sq| > 3       2044 flags before the break, first after it 4717 rows in
sigma past 2x        1270 flags before the break, first after it 126 rows in
```

The stretch reads an excess kurtosis of 16.6, which calls for a
`drift_threshold` of 50. At the default 20, `drift` flagged 11 of the
200,000 stable rows, the table's rate of 5.6 per 100,000, and at 50 it
flagged one. Neither found the break: when the noise triples, `sigma`
triples with it, and `drift` scores each residual against `sigma`. Its
first flag after the break at 20 is a false alarm, 15,011 rows in. The
CUSUM of squares flagged 2,044 of the stable rows, 1.0% of them, and found
the break only 4,717 rows in. With an infinite fourth moment, the squares'
own spread swamps a tripling for a long time. Divided by 2, a Gaussian's
variance of `z²`, in place of the measured one, it held on 31% of the
rows of 200 stable streams of the same noise ([Are its tails
heavy?](../DIAGNOSTICS.md#are-its-tails-heavy)), so a flag of it would say
nothing. `sigma` itself, past
twice its level over the stretch, flagged 0.6% of the stable rows and
found the break 126 rows in.

## Choose the detector for heavy tails

Choose by the excess kurtosis measured over the stable stretch:

| the stretch's mean `kurtosis` | to flag a level shift | to flag a change of spread |
|---|---|---|
| under 4 | `drift` at its default `drift_threshold` of 20 | `\|cusum_sq\| > 3` |
| 4 or more | `drift` at the threshold the table gives | `sigma` against its level over the stretch. `cusum_sq` holds its false-alarm rate there, but is slow to fire |

Where single large residuals move the fit itself, fit a robust model,
`huber` or `quantile`, which limits how far one row moves it.

## Check that it worked

Count each detector's flags over a stretch known to be stable, as step 5
does. They should be rare. At the threshold the recipe chose, 50,
`drift` flagged one of its 200,000 stable rows, 0.5 per 100,000, near the
table's 0.3 for Student's t with 3 degrees of freedom at 50.

| what you see | what it means | what to do |
|---|---|---|
| `drift` flags on stable rows | the threshold is below what the tails call for | measure the kurtosis again over a longer stretch, and increase `drift_threshold` by the table |
| `kurtosis_<t>` jumps from row to row | a windowed kurtosis is noisy under heavy tails, and never steadies where the fourth moment is infinite | read its mean over the stretch, not a row's value |
| `jarque_bera_<t>` past 5.99 on most rows | the residuals are not Gaussian | expect heavy tails, and set thresholds as this page does |
| no detector finds a break you planted in the spread | `drift` scores against a `sigma` that grows with the noise | read `sigma` against its level |
