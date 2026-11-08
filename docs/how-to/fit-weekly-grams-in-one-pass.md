<!-- kind: how-to -->
# Keep a Gram per block in one pass, and fit from it

This how-to is for anyone who wants to fit many least-squares models, on
any run of weeks or hours and any of the columns, from one pass over a
stream too large to hold.

Three models, `ewridge`, `lasso` and `ew_cov`, keep a
[Gram](../GLOSSARY.md#fitting-a-model): the weight of the rows, the
columns' means and co-moments, and each target's moments. Every
least-squares fit on those columns reads its rows only through the Gram,
whose size depends on the number of columns alone. So one pass can leave
a Gram per week, or per hour, and every fit after the pass reads those
Grams and never the rows. [How a Gram stands in for the
rows](../GRAM.md#how-a-gram-stands-in-for-the-rows) explains why.

## Prerequisites

These steps need the library, numpy and a column that names each block:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. `bank.gram(spec)` returns
  numpy arrays, and `po.gram` fits from them.
- The README's [Example data](../../README.md#example-data) built. The
  example reads `lf`, a
  [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) over
  `ticks.parquet`: 400 rows a minute apart, with the timestamp `ts`, five
  features and the target `y`, which is `1 + 2·x0 − x1` plus noise.
- A column whose value names each row's block, or a timestamp to make one
  from in the query, as `pl.col("ts").dt.truncate("1h")`.

## Keep a Gram per block, and fit from them

One pass over the rows keeps the Grams, and each fit afterwards reads only
them.

1. **Make the block column in the query,** with Polars' `with_columns`
   before the bank.
2. **Name it as the spec's `group`,** and the bank keeps a Gram for each
   block. Give `half_life=float("inf")`, so the Grams are least squares'
   sums, and `solve_every=float("inf")`, so the pass solves nothing.
3. **Run the pass with `bank.fit`,** which learns from every row and keeps
   no output.
4. **Read the Grams with `bank.gram(name)`,** one dict of numpy arrays per
   block, oldest first.
5. **Pool the blocks you want with `po.gram.merge`,** and fit with
   `po.gram.solve`. `po.gram.coef_stats` gives each coefficient's t.

Here one pass leaves eight hourly Grams of `df`'s five features. Pooled,
they give least squares on all 400 rows, with `df`'s truth and the t of
each coefficient. Any run of hours, on any of the columns, is one more
call. This code uses `lf` from [Example data](../../README.md#example-data):

```python
# 1 to 3. One Gram per hour, from one pass.
features = ["x0", "x1", "x2", "signal_a", "signal_b"]
hours = po.spec.ewridge(
    "hours", targets=["y"], features=features, group="hour",   # a Gram per hour
    half_life=float("inf"), solve_every=float("inf"),          # least squares' sums; the pass solves nothing
)
bank = po.ModelBank([hours])
bank.fit(lf.with_columns(hour=pl.col("ts").dt.truncate("1h")))   # the hour, made in the query
hourly = bank.gram("hours")                 # 4. one dict of numpy arrays per hour, oldest first

# 5. Pool every hour into one fit.
pooled = po.gram.merge(hourly)              # the Gram of all the hours' rows
coef = po.gram.solve(pooled)                # least squares: the intercept, then the features
t = po.gram.coef_stats(pooled, coef)["t"]   # and each coefficient's t
print(f"{len(hourly)} hourly Grams, pooled over {pooled['weight_sum']:.0f} rows")
for name, b, tb in zip(pooled["columns"], coef, t):
    print(f"{name:>9} {b:+.2f}  t {tb:+6.1f}")
morning = po.gram.solve(po.gram.merge(hourly[:3]), features=["x0", "x1"])   # any hours, any columns, from the same pass
print("x0 and x1 before 12:00:", " ".join(f"{b:+.2f}" for b in morning[1:3]))
```

```text
8 hourly Grams, pooled over 400 rows
intercept +0.99  t  +39.1
       x0 +1.99  t  +78.3
       x1 -1.01  t  -39.9
       x2 -0.03  t   -1.0
 signal_a +0.02  t   +0.6
 signal_b +0.03  t   +1.3
x0 and x1 before 12:00: +2.00 -1.05
```

**The same code serves thousands of columns.** A Gram of 200 features
and an intercept is 323 KB, and one of 10,000 columns 800 MB, whatever
its rows. For weeks, make the column with `pl.col("ts").dt.truncate("1w")`.
[Weekly blocks from one pass](../GRAM.md#weekly-blocks-from-one-pass)
gives the settings for a wide pass, such as `target_gaps` and
`gram_block_rows`, with a recipe of 40 weeks and 200 features.

## Check that it worked

**When every row weighs 1, the pooled Gram's `weight_sum` counts the rows
the pass read,** 400 in the example. Its fit is least squares on those
rows, so it matches a least-squares fit on the rows read into memory. The
recipe in [Weekly blocks from one
pass](../GRAM.md#weekly-blocks-from-one-pass) makes that comparison.

## Go on from the Grams

[GRAM.md](../GRAM.md) is the guide to fitting from Grams. It gives the
theory, the keys of a Gram and what a Gram cannot answer, with a recipe
that runs for each task:

| task | its recipe |
|---|---|
| a Gram per week from one pass | [Weekly blocks from one pass](../GRAM.md#weekly-blocks-from-one-pass) |
| pool any weeks, and score a fit on others | [Pool blocks, and score a fit on blocks held out](pool-grams.md#pool-blocks-and-score-a-fit-on-blocks-held-out) |
| pool decayed blocks, each aged first | [Age decayed blocks before you pool them](pool-grams.md#age-decayed-blocks-before-you-pool-them) |
| pool weeks and leave their levels out | [Pool within blocks to leave their levels out](pool-grams.md#pool-within-blocks-to-leave-their-levels-out) |
| a ridge and its t | [A ridge and its t](../GRAM.md#a-ridge-and-its-t) |
| the lasso path | [Choosing columns on the lasso path](../GRAM.md#choosing-columns-on-the-lasso-path) |
| columns chosen on most halves of the weeks, and a null for them | [Stability selection over halves of the weeks](../GRAM.md#stability-selection-over-halves-of-the-weeks) |
| a quarter of the bytes, and one wide Gram on several threads | [A compact Gram](../GRAM.md#a-compact-gram), [`gram_threads`](../GRAM.md#one-wide-gram-on-several-threads) |
| a Gram per block written to a file as the run goes, each block's memory freed | [Close groups and read each one's row](close-groups.md) |
