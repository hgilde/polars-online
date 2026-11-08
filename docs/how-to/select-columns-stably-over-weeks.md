<!-- kind: how-to -->
# Select columns that hold across the weeks

This how-to is for anyone with a [Gram](../GLOSSARY.md#fitting-a-model)
per week from one pass who wants the columns a lasso path chooses on most
halves of the weeks. You run stability selection over a hundred halves of
the weeks, and set its threshold from null runs with each week's target
negated at random. If you keep only the columns most correlated with the
target before each path, you screen again for each half and each null
run.

## Prerequisites

These steps need the library and a Gram per week:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipes generate their
  own data: 40 weeks of 200 features in families of five, of which six
  matter.
- A Gram per week from one pass, as [Keep a Gram per block in one
  pass](fit-weekly-grams-in-one-pass.md) makes them. A half is 20 of the
  40 weeks, drawn at random, and its pooled Gram.
- [Stability selection over halves of the
  weeks](../GRAM.md#stability-selection-over-halves-of-the-weeks) explains
  the selection, the bound its threshold gives, why the halves are drawn
  from whole weeks, and why a null run negates each week's target.
- [Choose columns on the lasso
  path](choose-columns-on-the-lasso-path.md#choose-columns-on-a-grams-lasso-path)
  walks one path on a Gram. Each half here walks one.

## Select the columns most halves choose

Use this to choose columns from wide Grams, where a lasso path on one
sample can choose one set of columns and on another sample another set.

1. **Keep a Gram per week in one pass,** with an `ewridge` spec whose
   `group` is the week and whose `half_life` is `float("inf")`.
2. **Draw 50 random splits of the weeks into two halves of 20,** each half
   beside its complement: 100 halves.
3. **Walk each half's lasso path to the first knot with eight columns
   active, with `po.gram.solve_subsets(blocks, halves, path={"max_active":
   8})`.** It reads the Grams once, then merges each half and walks its
   path on the bank's thread pool, with no Python copy of a matrix per
   half. A loop of `po.gram.merge` and `po.gram.lars_paths` gives the same
   paths.
4. **Count, for each column, the share of the halves that chose it:**
   active at any knot of the half's path.
5. **Keep the columns chosen on at least a threshold of the halves,** such
   as 75%. Take `q`, the columns a half chooses on average, and `p`, the
   columns. Where the conditions of [the
   bound](../GRAM.md#stability-selection-over-halves-of-the-weeks) hold,
   the stable set holds at most `q² / ((2 π_thr − 1) p)` columns with no
   effect on average, for a threshold `π_thr` above 1/2.

This code makes the stream and runs step 1:

```python
import time
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
# Stamp each row, from a Monday, and save the table for the scan to read.
start = datetime(2025, 1, 6)  # a Monday
ts = pl.datetime_range(start, start + timedelta(minutes=20 * (n - 1)), "20m", eager=True)
pl.DataFrame({"ts": ts, **dict(zip(features, x.T)), "y": y}).write_parquet("weeks.parquet")

# 1. One pass over the file leaves a Gram per week. The week is a column made in the query.
lf = pl.scan_parquet("weeks.parquet").with_columns(week=pl.col("ts").dt.truncate("1w"))
spec = po.spec.ewridge(
    "blocks", targets=["y"], features=features, group="week",
    half_life=float("inf"), solve_every=float("inf"), target_gaps="pairwise", gram_block_rows=256,
)
bank = po.ModelBank([spec])
bank.fit(lf)  # learns from every row, and keeps only the state
blocks = bank.gram("blocks")  # one dict per week, oldest first
```

The next code runs steps 2 to 5 by a loop of merges and `lars_paths`, then
gets the same paths from `po.gram.solve_subsets` in one call, timed beside
the loop:

```python
# 2. The halves the selection runs on.
halves = []  # 50 random splits of the 40 weeks: 100 halves of 20, each beside its complement
splitter = np.random.default_rng(1)
for _ in range(50):
    order = splitter.permutation(n_weeks)
    halves += [sorted(order[:20]), sorted(order[20:])]

# 3. Merge each half and walk its lasso path, timed.
t0 = time.perf_counter()
merged = [po.gram.merge([blocks[i] for i in half]) for half in halves]
paths = po.gram.lars_paths(merged, max_active=8)  # a list per half, a path per target
seconds = time.perf_counter() - t0

# 4. Count how often the halves chose each column.
chosen, sizes = np.zeros(k), []  # how many halves chose each column, and how many each half chose
for (path,) in paths:
    ever = set().union(*path["active"])  # active at any knot, up to the stop
    sizes.append(len(ever))
    chosen[[features.index(c) for c in ever]] += 1
freq = dict(zip(features, chosen / len(halves)))

# 5. The planted columns, the most chosen of the rest, and the stable set's bound.
planted = ["x0", "x5", "x10", "x15", "x20", "x25"]
print("planted: " + ", ".join(f"{c} {freq[c]:.2f}" for c in planted))
rest = sorted((f for f in features if f not in planted), key=lambda f: -freq[f])
print("the most chosen of the rest: " + ", ".join(f"{c} {freq[c]:.2f}" for c in rest[:4]))
q, threshold = np.mean(sizes), 0.75
print(f"chosen on at least {threshold:.0%} of the halves: {' '.join(f for f in features if freq[f] >= threshold)}")
print(f"{q:.1f} columns a half, so at most {q**2 / ((2 * threshold - 1) * k):.2f} false selections expected")

# 3 again: the same paths from solve_subsets in one call, timed beside the loop.
t0 = time.perf_counter()
same = po.gram.solve_subsets(blocks, halves, path={"max_active": 8})  # the same paths, in one call
agree = all(a[0]["active"] == b[0]["active"] for a, b in zip(same, paths))
print(f"solve_subsets chooses the same: {agree}")
print(f"time: {seconds:.2f} s by merge and lars_paths, {time.perf_counter() - t0:.2f} s by solve_subsets")
```

```text
planted: x0 1.00, x5 1.00, x10 1.00, x15 1.00, x20 1.00, x25 1.00
the most chosen of the rest: x4 0.45, x11 0.25, x86 0.20, x14 0.17
chosen on at least 75% of the halves: x0 x5 x10 x15 x20 x25
8.0 columns a half, so at most 0.64 false selections expected
solve_subsets chooses the same: True
time: 0.19 s by merge and lars_paths, 0.01 s by solve_subsets
```

Every planted column is chosen on every half, and the stable set at 75%
is exactly those six. The most chosen of the rest is `x4`, on 45% of the
halves, a member of `x0`'s family. It shares `x0`'s factor, and so takes
one of the two places the six planted columns leave in a path of eight.
Each half chose 8.0 columns, so where the bound's conditions hold, over
draws of the halves the stable set holds at most 0.64 columns with no
effect on average. This draw's holds none. `solve_subsets` chooses the
same columns on every half. It took 0.01 s, where the merges and
`lars_paths` took 0.19 s, at a load average near 14 on 14 cores (the
load average is the number of processes running or waiting to run,
averaged over a minute).

`x25` is chosen on every half, though its coefficient is a different
number each week. To see whether a chosen coefficient holds from week to
week, [fit per-block coefficients](fit-per-block-coefficients.md).

## Set the threshold from null runs, by sign flips

Use this when the bound of step 5 may not hold: a path that always stops
at eight columns, among columns that share families, can choose some
columns with no effect more often than the bound allows. A null run draws
a sign for each week and negates that week's target, so no effect keeps
its sign across the weeks, and runs every half again.

1. **Draw a sign for each week, for each of 100 null runs,** as an array
   of weeks by draws.
2. **Run the real selection and the null runs pooled within the weeks,**
   with `po.gram.solve_subsets(..., within=True)` and, for the null runs,
   `target_signs=signs`. It returns one run of every half per draw.
   `target_signs` is refused without `within=True`, since pooled as a
   union a negated week's target mean would move the gaps between the
   weeks' means.
3. **Read each null run's most chosen column,** and take the 95th
   percentile of those shares as the threshold. Only 5% of the null runs
   choose any column on more halves than that.
4. **Keep the columns the real selection chose on more halves than the
   threshold.**

This code continues from the recipe of [Select the columns most halves
choose](#select-the-columns-most-halves-choose), whose `blocks`, `halves`
and `planted` it reuses:

```python
# The selection frequencies of step 3, read alike from the real run and from each null run.
def frequencies(run):
    """The share of the halves that chose each column: active at any knot of its path."""
    chosen = np.zeros(k)
    for (path,) in run:
        chosen[[features.index(c) for c in set().union(*path["active"])]] += 1
    return chosen / len(run)


# 1 and 2. The selection on the real target, and on 100 draws of a sign for each week.
signs = np.random.default_rng(2).choice([-1.0, 1.0], size=(n_weeks, 100))  # a sign per week, 100 draws
real = po.gram.solve_subsets(blocks, halves, path={"max_active": 8}, within=True)
null = po.gram.solve_subsets(blocks, halves, path={"max_active": 8}, within=True, target_signs=signs)

# 3. Each column's frequency, and a threshold from the null runs' most chosen column.
freq = dict(zip(features, frequencies(real)))
null_max = np.array([frequencies(run).max() for run in null])  # each null run's most chosen column
threshold = np.quantile(null_max, 0.95)
# 4. The planted columns, the most chosen of the rest, and what clears the threshold.
print("planted: " + ", ".join(f"{c} {freq[c]:.2f}" for c in planted))
rest = sorted((f for f in features if f not in planted), key=lambda f: -freq[f])
print("the most chosen of the rest: " + ", ".join(f"{c} {freq[c]:.2f}" for c in rest[:4]))
print(f"the most chosen column of a null run: {np.median(null_max):.2f} of the halves at the median draw, {threshold:.2f} at the 95th percentile")
print(f"null runs in which a column is chosen on at least 75% of the halves: {(null_max >= 0.75).sum()} of {len(null)}")
print(f"chosen on more halves than {threshold:.2f}: {' '.join(f for f in features if freq[f] > threshold)}")
```

```text
planted: x0 1.00, x5 1.00, x10 1.00, x15 1.00, x20 1.00, x25 1.00
the most chosen of the rest: x4 0.49, x11 0.25, x86 0.20, x14 0.16
the most chosen column of a null run: 0.81 of the halves at the median draw, 0.95 at the 95th percentile
null runs in which a column is chosen on at least 75% of the halves: 76 of 100
chosen on more halves than 0.95: x0 x5 x10 x15 x20 x25
```

With each week's sign drawn at random, the most chosen column of a null
run is still chosen on 0.81 of the halves at the median draw. Each half
chooses eight columns whatever the signs, and a column with a large
effect in some weeks keeps being chosen when its sign changes from week
to week. The threshold of 0.75 that the selection without null runs used
passes such a column in 76 of the 100 null runs. The null runs' 95th
percentile, 0.95, is the threshold this selection earns on these weeks.
The six planted columns pass it, and the most chosen of the rest, `x4`,
is at 0.49.

In one call, the real run and the hundred null runs of the hundred
halves, at 200 columns, took 0.08 s together. A Python loop of `merge`
and `lars_paths` took about 0.08 s for each draw, about 8 s for the
hundred, at a load average near 4 on 14 cores.

## Screen the columns again for each half and each null run

Use this when you keep only the columns most correlated with the target
before a path chooses among them. The screen is part of the selection,
so each half screens on its own weeks, and each null run under its own
signs.

1. **For each half and each draw of signs, compute the screen,** such as
   the 40 columns most correlated with the target, pooled within the
   half's weeks from their own moments. It reads a vector per week, and
   merges no matrix.
2. **Make a job for each half and draw:** a mapping of `"subset"`,
   `"target"`, `"draw"` and `"features"`, the screen's columns.
3. **Run every job in one call, with `po.gram.solve_subsets(...,
   jobs=jobs)`.** It returns a path per job, in order. Each half is merged
   once, on every column its jobs read, and each draw re-pools that
   half's cross-moments once. A job's path is the one `po.gram.lars_path`
   gives on `po.gram.subset(po.gram.merge(half_grams, within=True),
   ["intercept", *features])`, with each week's cross-moments times its
   sign in the draw, to the bit.
4. **Set the threshold from the null draws, as in [Set the threshold from
   null runs, by sign flips](#set-the-threshold-from-null-runs-by-sign-flips),**
   and keep the columns the real target's draw chose on more halves than
   it.

This code continues from the recipe of [Select the columns most halves
choose](#select-the-columns-most-halves-choose), whose `blocks`, `halves`
and `planted` it reuses. It runs the real target as draw 0, and 20 null draws
after it:

```python
# 20 draws of a sign per week after the real target's.
signs = np.random.default_rng(2).choice([-1.0, 1.0], size=(n_weeks, 21))
signs[:, 0] = 1.0  # draw 0 is the real target: every sign +1

# 1. Each half's screen under each draw: its 40 columns most correlated with the target, read from
# the weeks' own moments as the merge within the weeks pools them, so no matrix is merged for it.
w = np.array([b["target_weights"][0] for b in blocks])  # each week's weight
c = np.stack([b["cross_centered"][0][1:] for b in blocks])  # its cross-moments, past the intercept
v = np.stack([np.diag(b["comoments"])[1:] for b in blocks])  # and its columns' variances
# 2. A job per half and draw: the half's correlations under the draw's signs, and the top 40.
jobs = []
for s, half in enumerate(halves):
    for d in range(signs.shape[1]):
        cov = (w[half] * signs[half, d]) @ c[half] / w[half].sum()
        var = w[half] @ v[half] / w[half].sum()
        screen = np.argsort(-np.abs(cov) / np.sqrt(var))[:40]
        jobs.append({"subset": s, "draw": d, "target": "y", "features": [features[j] for j in screen]})

# 3. Every half's path under every draw on its own screen, in one call: a path per job, in order.
runs = po.gram.solve_subsets(
    blocks, halves, path={"max_active": 8}, within=True, target_signs=signs, jobs=jobs
)

# 4. How often each draw's halves chose each column, and a threshold from the null draws.
chosen = np.zeros((signs.shape[1], k))
for job, path in zip(jobs, runs):
    chosen[job["draw"], [features.index(c) for c in set().union(*path["active"])]] += 1
freq = chosen / len(halves)
null_max = freq[1:].max(axis=1)  # each null draw's most chosen column
threshold = np.quantile(null_max, 0.95)
real = dict(zip(features, freq[0]))
# How many screens the jobs ran on, the planted columns, and what clears the threshold.
screens = {frozenset(job["features"]) for job in jobs}
print(f"{len(jobs)} paths, each on 40 of the {k} columns, {len(screens)} screens among them")
print("planted: " + ", ".join(f"{c} {real[c]:.2f}" for c in planted))
print(f"the most chosen column of a null run: {np.median(null_max):.2f} of the halves at the median draw, {threshold:.2f} at the 95th percentile")
print(f"chosen on more halves than {threshold:.2f}: {' '.join(f for f in features if real[f] > threshold)}")
```

```text
2100 paths, each on 40 of the 200 columns, 2100 screens among them
planted: x0 1.00, x5 1.00, x10 1.00, x15 1.00, x20 1.00, x25 1.00
the most chosen column of a null run: 0.81 of the halves at the median draw, 0.95 at the 95th percentile
chosen on more halves than 0.95: x0 x5 x10 x15 x20 x25
```

Each of the 2,100 paths ran on a screen of its own. The planted columns
survive every half's screen and are chosen on every half. The null runs,
each screened under its own signs, put the threshold at 0.95, where the
selection without a screen put it. Only the six planted columns pass it.

Measured on an M4 Pro at a load near 7, 100 halves of 19 of 38 Grams
under 11 draws and 10 targets, each of the 11,000 paths on its own 400 of
1,500 columns, took 11 ms a half. The loop of `merge`, `subset` and
`lars_path` it replaces took 115 ms a half.

## Check the result

**The selection worked when the columns it keeps clear a threshold set by
null runs, and the null runs were screened as the real run was.** What
else you may see, and what to do:

| what you see | what it means | what to do |
|---|---|---|
| a column with no effect is chosen on most halves | it shares a family with a column that has one, and the path takes it beside that column | set the threshold from null runs ([Set the threshold from null runs, by sign flips](#set-the-threshold-from-null-runs-by-sign-flips)) |
| `target_signs` is refused | the selection was pooled as a union | pass `within=True` |
| a chosen column's coefficient changes sign from week to week | stability selection bounds false selections, and says nothing about whether a chosen coefficient holds | [fit per-block coefficients](fit-per-block-coefficients.md) |
