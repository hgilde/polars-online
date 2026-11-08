<!-- kind: how-to -->
# Pool Grams

This how-to is for anyone with a [Gram](../GLOSSARY.md#fitting-a-model)
per block of rows, such as one per week from one pass, who wants to fit or score on any set of the blocks.
You pool them into one Gram, aged to a common row, with or without the
blocks' own levels.

## Prerequisites

These steps need the library and a Gram per block:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. Each recipe generates its
  own data.
- A Gram per block, from one pass of a
  [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) over the stream: an `ewridge` spec with
  `group=` the block's column, such as a week made in the query with
  `pl.col("ts").dt.truncate("1w")`. Then `bank.gram(name)` gives one dict
  per block in the order of their keys, oldest first for a week column. [Keep weekly Grams of
  hundreds of
  features](fit-weekly-grams-in-one-pass.md#keep-weekly-grams-of-hundreds-of-features)
  shows what each block holds.
- The blocks' rows must be disjoint. A row in two blocks counts twice, and
  `po.gram.merge` cannot tell.

[Pooling blocks](../GRAM.md#pooling-blocks) explains how a pooled Gram is
formed, and what pooling within blocks leaves out.

## Pool blocks, and score a fit on blocks held out

Use this to fit on some blocks and score the fit on others, without
reading their rows again. Blocks run under `half_life=float("inf")` pool
as they are.

1. **Pool the blocks to fit on with `po.gram.merge(blocks[:30])`.** It
   returns the Gram of all their rows.
2. **Fit with `po.gram.solve`.** It returns the intercept first, then a
   coefficient per feature.
3. **Pool the blocks held out the same way.**
4. **Score the fit on the held-out Gram with a function of the Gram's
   arrays, as `mean_squared_error` in the recipe.** The fit's mean squared
   error over the held-out rows is a function of the Gram's arrays alone ([Every
   least-squares fit is a function of the
   Gram](../GRAM.md#every-least-squares-fit-is-a-function-of-the-gram)).

This code fits weeks 1 to 30, scores the fit on weeks 31 to 40, and checks
the score against those weeks' rows:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 40 weeks, a row every 20 minutes. 200 features in 40 families of five that share a factor, so
# two features of one family correlate at 0.5. y depends on six of them, and x25's coefficient is
# drawn again each week, 0.15 on average.
rng = np.random.default_rng(7)
n_weeks, per_week, k = 40, 504, 200
n = n_weeks * per_week
x = 0.7 * (rng.standard_normal((n, k)) + np.repeat(rng.standard_normal((n, k // 5)), 5, axis=1))
b25 = 0.15 + 0.3 * rng.standard_normal(n_weeks)
y = x[:, [0, 5, 10, 15, 20]] @ [0.5, 0.4, 0.3, 0.2, 0.1] + b25.repeat(per_week) * x[:, 25]
y += rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
# Stamp each row, from a Monday, and save the table for `pl.scan_parquet` to read.
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=float("inf"),
    gram_block_rows=256,      # rows added to the Gram 256 at a time, for a fit over hundreds of features
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first


# The score of step 4: a fit's error on any set of weeks, without reading their rows.
def mean_squared_error(g, coef):
    """A fit's mean squared error over the rows behind g, from g's arrays alone, for a target on every row."""
    b0, b = coef[0], coef[1:]
    gap = g["cross_moments"][0][0] - b0 - g["means"][1:] @ b  # the target's mean less the fit's
    spread = g["target_vars"][0] - 2 * b @ g["cross_centered"][0][1:] + b @ g["comoments"][1:, 1:] @ b
    return spread + gap**2


# 1 to 3. Pool the early weeks and fit; pool the weeks held out.
train = po.gram.merge(blocks[:30])  # weeks 1 to 30
held_out = po.gram.merge(blocks[30:])  # weeks 31 to 40
coef = po.gram.solve(train)
print(f"weeks 1-30: {train['weight_sum']:,.0f} rows, error {mean_squared_error(train, coef):.4f} in sample")
print(f"weeks 31-40: {held_out['weight_sum']:,.0f} rows, error {mean_squared_error(held_out, coef):.4f} held out")

# The held-out error from the rows themselves, read into memory to check.
rows = pl.read_parquet("weeks.parquet").filter(pl.col("ts") >= start + timedelta(weeks=30))
pred = coef[0] + rows.select(features).to_numpy() @ coef[1:]
print(f"from the rows of weeks 31-40: {np.mean((rows['y'].to_numpy() - pred) ** 2):.4f}")
```

```text
weeks 1-30: 15,120 rows, error 1.0912 in sample
weeks 31-40: 5,040 rows, error 1.1337 held out
from the rows of weeks 31-40: 1.1337
```

The held-out Gram gives the error the held-out rows give, 1.1337, to
every digit printed. It is above the in-sample error, 1.0912, as a fit's
error on rows it never saw usually is. The spread in
`mean_squared_error` reads `cross_centered`, the form the fit itself is
solved from, so the score keeps its digits however far the columns'
means are from 0.

## Age decayed blocks before you pool them

**Under a finite half-life, age each block to the last row of the pooled
stream before pooling it.** A block's weights are relative to its own
last row. A week that ended a month before the stream did is therefore
weighted as if it ended with the stream: too heavily, by exactly the
decay between its last row and the stream's.

1. **Read each block's last clock value from `bank.groups(name)["clock_last"]`.**
   It gives the groups in the order `gram()` gives them.
2. **Compute each block's decay, `0.5 ** (dt / half_life)`,** with `dt`
   the clock from the block's last row to the stream's last row. If no
   step of the stream's clock exceeds `gap_cap`, `dt` is the difference of
   the two clock values; otherwise, sum `min(step, gap_cap)` over the
   stream's steps after the block's last row, as the model bank caps them.
3. **Multiply the block's `weight_sum` and `target_weights` by its
   decay.** Leave the means and co-moments as they are, since they are
   weighted means. Leave `n_kish` too, since it does not change when every
   weight is scaled by one factor.
4. **Pool the aged blocks with `po.gram.merge`.**

This code runs two specs over the same stream in one pass: a Gram per
week at a half-life of four weeks, and one Gram of the whole stream at
the same half-life. Both decay on the clock `ts`, a timestamp, so their
`half_life` and `gap_cap` are durations. Steps of 20
minutes are all below the `gap_cap` of an hour, so `dt` is the plain
difference of the clocks:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 40 weeks, a row every 20 minutes. 200 features in 40 families of five that share a factor, so
# two features of one family correlate at 0.5. y depends on six of them, and x25's coefficient is
# drawn again each week, 0.15 on average.
rng = np.random.default_rng(7)
n_weeks, per_week, k = 40, 504, 200
n = n_weeks * per_week
x = 0.7 * (rng.standard_normal((n, k)) + np.repeat(rng.standard_normal((n, k // 5)), 5, axis=1))
b25 = 0.15 + 0.3 * rng.standard_normal(n_weeks)
y = x[:, [0, 5, 10, 15, 20]] @ [0.5, 0.4, 0.3, 0.2, 0.1] + b25.repeat(per_week) * x[:, 25]
y += rng.standard_normal(n)
features = [f"x{j}" for j in range(k)]
# Stamp each row, from a Monday, and save the table for `pl.scan_parquet` to read.
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# One pass, two specs: a decayed Gram per week, and one decayed Gram of the whole stream.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
decayed = dict(
    targets=["y"], features=features, clock="ts", gap_cap="1h", half_life="4w",
    solve_every=float("inf"),
)
bank = po.ModelBank(
    [po.spec.ewridge("by_week", group="week", **decayed), po.spec.ewridge("whole", **decayed)]
)
bank.fit(lf)
blocks, whole = bank.gram("by_week"), bank.gram("whole")[0]

# 1 to 3. Age each week to the stream's last row: its weights are relative to its own last row.
last = bank.groups("by_week")["clock_last"]  # each week's last row, in the order gram() gives them
aged = []
for block, end in zip(blocks, last):
    decay = 0.5 ** ((last.max() - end) / timedelta(weeks=4))  # 0.5 ** (dt / half_life)
    aged.append(
        dict(block, weight_sum=block["weight_sum"] * decay, target_weights=block["target_weights"] * decay)
    )

# 4. Pool the weeks as they are and aged, each against the whole stream's fit.
for how, parts in [("as they are", blocks), ("aged", aged)]:
    gap = np.abs(po.gram.solve(po.gram.merge(parts)) - po.gram.solve(whole)).max()
    shown = "below 1e-12" if gap < 1e-12 else f"{gap:.3f}"
    print(f"40 weeks pooled {how}: the largest gap from the whole stream's coefficients is {shown}")
```

```text
40 weeks pooled as they are: the largest gap from the whole stream's coefficients is 0.085
40 weeks pooled aged: the largest gap from the whole stream's coefficients is below 1e-12
```

Pooled as they are, the decayed weeks weigh the first week as much as the
last, and a coefficient lands 0.085 from the whole stream's. Aged before
they are pooled, they give the whole stream's own coefficients.

## Pool within blocks to leave their levels out

**Pool within weeks to ask how a feature moves the target inside a week,
and keep the union to ask how their levels move together.** Pooled as one
union, a chance correlation of the weekly means reads as a correlation of
the rows, and its standard errors divide by every row, as if each row had
confirmed it. The second question has forty observations, one a week: regress each
week's mean target (`target_means`) on its features' means (`means`).

1. **Pool with `po.gram.merge(blocks, within=True)`.** The pooled Gram is
   that of the rows less their week's means, so a fit on it has an
   intercept for each week: the within, or fixed-effects, estimator.
2. **Fit with `po.gram.solve`.** It returns `nan` for the intercept, since
   each week had one and the pooled Gram holds none of them. So does every other
   `po.gram` fit on a Gram pooled within blocks.
3. **Read the standard errors with `po.gram.coef_stats`.** It divides by
   `n − p − B`, with `p` the features and `B` the weeks the target has rows
   in, counted in the pooled Gram's `within_parts` and, per target, in
   `target_within_parts`.
4. **To fit several sets of blocks, pass `within=True` to
   `po.gram.solve_subsets`,** which pools each subset the same way.

This code makes `slow`, whose weekly level wanders, and `quick`, which
moves `y` inside each week. `y`'s weekly level wanders too, apart from
both:

```python
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import polars_online as po

# 40 weeks, a row every 20 minutes. slow is an exponentially weighted mean of noise at a half-life
# of a week, so its weekly level wanders. y's weekly level wanders apart from it, drawn for each
# week. Only quick moves y inside a week, by 0.05.
rng = np.random.default_rng(7)
n_weeks, per_week = 40, 504
n = n_weeks * per_week
slow = pl.Series(rng.standard_normal(n)).ewm_mean(half_life=per_week).to_numpy()
slow = slow / slow.std()
quick = rng.standard_normal(n)
level = 0.3 * rng.standard_normal(n_weeks)  # y's level in each week
y = 0.05 * quick + level.repeat(per_week) + rng.standard_normal(n)
# Stamp each row, from a Monday, and save the table for `pl.scan_parquet` to read.
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, "slow": slow, "quick": quick, "y": y}).write_parquet("levels.parquet")

# One pass over the file leaves a Gram per week.
lf = pl.scan_parquet("levels.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=["slow", "quick"], group="week",
    half_life=float("inf"), solve_every=float("inf"),
)
bank = po.ModelBank([spec])
bank.fit(lf)
blocks = bank.gram("blocks")

# The chance correlation of the forty weekly means.
weekly = np.array([[g["means"][1], g["target_means"][0]] for g in blocks])  # each week's mean of slow and of y
print(f"the weekly means of slow and y correlate at {np.corrcoef(weekly.T)[0, 1]:.2f} over {n_weeks} weeks")
# 1 to 3. Slow and quick, pooled as one union and pooled within weeks.
print(f"{'pooled':>12} {'slow':>7} {'t':>5} {'quick':>7} {'t':>5}")
for how, pooled in [("as a union", po.gram.merge(blocks)), ("within weeks", po.gram.merge(blocks, within=True))]:
    coef = po.gram.solve(pooled)  # within weeks, coef[0], the intercept, is nan
    t = po.gram.coef_stats(pooled, coef)["t"]
    print(f"{how:>12} {coef[1]:>7.3f} {t[1]:>5.1f} {coef[2]:>7.3f} {t[2]:>5.1f}")

# Each week alone, and a t from the spread of the weekly coefficients.
each = po.gram.solve_subsets(blocks, [[i] for i in range(n_weeks)])  # each week alone
weekly_slow = np.array([fits[0]["coef"][1] for fits in each])
t_weeks = weekly_slow.mean() / (weekly_slow.std(ddof=1) / np.sqrt(n_weeks))  # Fama and MacBeth's t
print(f"each week alone: slow's mean coefficient {weekly_slow.mean():.3f}, its t from the weeks {t_weeks:.1f}")
```

```text
the weekly means of slow and y correlate at -0.10 over 40 weeks
      pooled    slow     t   quick     t
  as a union  -0.025  -3.4   0.047   6.4
within weeks  -0.004  -0.3   0.046   6.5
each week alone: slow's mean coefficient 0.006, its t from the weeks 0.2
```

The weekly means of `slow` and `y` correlate at −0.10, within one
standard error, 1/√40 ≈ 0.16, of 0. Pooled as a union, that correlation gives
`slow` a coefficient of −0.025 and a t of −3.4. Pooled within weeks,
`slow`'s t is −0.3. Each week alone agrees: the forty weekly coefficients
average 0.006, with a t of 0.2 from their spread. `quick` moves `y` inside
each week, and both forms find it, with a t of 6.4 and of 6.5.

## Check that it worked

A pooled Gram worked when a fit on it matches what the rows would give,
as the first recipe checks against the held-out rows. What else you may
see, and what to do:

| what you see | what it means | what to do |
|---|---|---|
| pooled decayed blocks give coefficients apart from one Gram of the whole stream | the blocks were pooled with weights relative to their own last rows | age each block before you pool it ([Age decayed blocks before you pool them](#age-decayed-blocks-before-you-pool-them)) |
| `po.gram.solve_subsets` refuses a subset | the subset names one Gram twice, so its rows would count twice | name each block once |
| a union of parts refuses a Gram | one part was pooled with `within=True`: its means are 0, and the union would add the gap from 0 to the others' means | pool the blocks themselves as a union, not a Gram already pooled within |
| the intercept is `nan` | the Gram was pooled within blocks, which holds no intercept | read the slopes, or pool as a union for an intercept |
| a coefficient's t falls far when you pool within blocks | it rested on a correlation of the blocks' levels | ask the levels' question with one observation a block: regress the blocks' `target_means` on their `means` |
