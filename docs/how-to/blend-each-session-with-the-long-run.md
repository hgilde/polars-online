<!-- kind: how-to -->
# Blend each session with the long run

This how-to is for anyone who fits `ewridge` with a short half-life over a
stream cut into sessions, such as trading days, and wants each session to
start from a fit nearer the long run than the last session's end. You
give `session_shrink` and `long_half_life`, and the model blends its
[Gram](../GLOSSARY.md#fitting-a-model) with a slower twin's at each session change.

## Prerequisites

These steps need the library, a stream with sessions and an `ewridge`
spec:

- polars-online and numpy installed. The recipe generates its own data.
- A [session](../GLOSSARY.md#time) column, whose change of value starts a
  session, and a `session_gap` that is a number: the clock step the model
  takes at the change. `session_gap="reset"` is refused beside a blend,
  since it resets the model at the change, and so is
  `group_close="session"`, which closes the group there, and
  `window_size`.
- An `ewridge` spec with a [half-life](../GLOSSARY.md#time) short enough
  to follow a session, and a clock.

`session_shrink` and `long_half_life` are defined in [`po.spec.ewridge`'s
docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge),
whose Notes, under *The session blend*, give the blend's equations.

## Start each session from a blend

Use this where each session's relationship differs from the last one's by
chance, so that the long run is a better start than the last session's
end.

1. **Name the session column in `session=`, and give `session_gap` in
   clock units.**
2. **Give `long_half_life`, the half-life of the long run.** A second
   Gram, the slow twin, decays at it. `float("inf")` keeps the
   whole history.
3. **Give `session_shrink=f`, the long run's share, from 0 to 1.** At each
   session change the fit's means and co-moments become `1 − f` of this
   session's and `f` of the twin's. `0` is the same as no
   `session_shrink`, and `1` takes the long run's moments.
4. **Give `max_rows_between_solves=1` for the new session's first rows to
   read the blend at once.** The coefficients stay as last
   [solved](../GLOSSARY.md#fitting-a-model) until the next solve the schedule brings due, which reads the blended moments.
   With a solve on every row, the solve on the session's first row reads
   the blend, and the next row is predicted from it.

**The blend changes the moments and nothing else.** The blended moments
keep this session's weight, [Kish's effective sample size](../GLOSSARY.md#the-model-bank-and-what-it-reads) and the ridge's
decay factor (`p` in [`ewridge`'s update](../MODELS.md#the-update)), so `weight_sum`,
the warm-up gates and the solve schedule do not move. So a small `f` moves
the output by a small amount.

This code fits 40 sessions of 300 rows, each with a slope drawn around a
long-run slope of 2, at a half-life of 50 rows. It reads the slope after
each session's first row, and the error over each session's first 20 rows,
from the eleventh session on:

```python
import numpy as np

# 40 sessions of 300 rows. Each session's slope is drawn around a long-run slope of 2.
rng = np.random.default_rng(5)
days, per_day = 40, 300
day_slope = 2.0 + 0.6 * rng.standard_normal(days)
day = np.repeat(np.arange(days), per_day)
x = rng.standard_normal(days * per_day)
y = day_slope[day] * x + 0.5 * rng.standard_normal(days * per_day)
pl.DataFrame({"t": np.arange(days * per_day, dtype=float), "day": day, "x": x, "y": y}).write_parquet("sessions.parquet")

# Three blends, from none to the long run's moments alone.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=50.0,
              session="day", session_gap=1.0, max_rows_between_solves=1, coef_every=0)
blends = {f: po.spec.ewridge(f"blend_{i}", session_shrink=f, long_half_life=3000.0, **common) for i, f in enumerate([0.0, 0.5, 1.0])}
out = pl.scan_parquet("sessions.parquet").online.fit_predict(list(blends.values())).collect()

# Each session's first row, and its first 20 rows, after ten sessions of history.
starts = np.arange(10, days) * per_day
early = np.concatenate([np.arange(s, s + 20) for s in starts])
print(f"the long-run slope 2.0; the sessions' slopes spread by {day_slope.std():.2f}")
for f, spec in blends.items():
    fields = out[spec["name"]].struct.unnest()
    first_slope = fields["coef"].list.get(1).to_numpy()[starts]
    err = float(((fields["pred_y"] - out["y"]) ** 2).to_numpy()[early].mean())
    print(f"session_shrink={f}: slope off the long run after the first row {np.abs(first_slope - 2.0).mean():.2f}; error over the first 20 rows {err:.3f}")
```

```text
the long-run slope 2.0; the sessions' slopes spread by 0.49
session_shrink=0.0: slope off the long run after the first row 0.42; error over the first 20 rows 0.691
session_shrink=0.5: slope off the long run after the first row 0.26; error over the first 20 rows 0.560
session_shrink=1.0: slope off the long run after the first row 0.14; error over the first 20 rows 0.511
```

The noise's variance is 0.25, and the rest of each error is the slope the
fit has not yet found. Here the sessions' slopes are independent draws
around 2, so the long run is the best start, and the full blend predicts
the first rows best. Where a session's slope carries over from the last
one, a smaller `f` keeps more of it.

## Check the result

**Compare the error over each session's first rows at two or three values
of `session_shrink`, `0` among them, as the recipe does,** in one bank. A
blend that helps lowers the error at a session's start, and leaves it
alone once the session's own rows outweigh the blend.

| what you see | what it means | what to do |
|---|---|---|
| no change in the predictions after a session's first row | the coefficients stay as last solved until a solve comes due | give `max_rows_between_solves=1` |
| a larger error at a session's start as `f` grows | the sessions carry their relationship over, so the last session's end is the better start | give a smaller `session_shrink`, or none |
| `ValueError` naming `session_shrink` or `long_half_life` | one is given without the other, `session_shrink` is outside `[0, 1]`, or it is given without a `session` column, beside `session_gap="reset"`, `group_close="session"` or `window_size` | give both, with a `session` column and a `session_gap` that is a number |
