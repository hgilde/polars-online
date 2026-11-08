<!-- kind: how-to -->
# Fit hundreds of features

This how-to is for anyone whose `ewridge` or `ew_cov` reads hundreds or
thousands of features, where updating the
[Gram](../GLOSSARY.md#fitting-a-model) on each row takes most of the run.
You will add the rows to the Gram in blocks with `gram_block_rows`, and
run that work on the bank's idle threads with `gram_threads`.

## Prerequisites

These steps need the library, numpy and a wide stream:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe builds its own
  data.
- An `ewridge` spec over 256 columns or more (the features and the
  intercept). `gram_threads` applies to
  `ew_cov` too, and `gram_block_rows` to `ewridge` alone.
- A machine with more than one core, for `gram_threads`.

Learning a row takes `O(k²)` time in the Gram of `k` features, and at a
thousand features that work takes most of the run. [One wide Gram on
several threads](../GRAM.md#one-wide-gram-on-several-threads) explains why
splitting it moves no bit, and how `"auto"` chooses a thread count.

## Add rows to the Gram in blocks

Use this for a fit over hundreds of features. `gram_block_rows=256` holds
256 rows back and adds them to the Gram with one matrix product, in place
of adding each row as it is learned.

1. **Give the spec `gram_block_rows=256`.** Fewer rows leave the product
   short, and more gain little and hold more rows in the state.
2. **Give a schedule that solves rarely.** A solve brings the Gram up to
   date before it runs, so blocks speed up only the rows between solves. Under
   `half_life=float("inf")`, give `solve_every=float("inf")`, which never
   comes due, and a row cap such as `max_rows_between_solves=1000`, or no
   cap where the fits are made from the Grams after the pass. The spec
   refuses `gram_block_rows` where no schedule spaces the solves, and
   beside `window_size`.

Each speed-up in the table is of the rows learned a second, against one row at
a time, on one thread with no solve. It was measured on an Apple M4 Pro of
14 cores, timing `ewridge`'s whole step (its prediction, its
cross-moments and its Gram) with one target, an intercept and a half-life
of 5,000 rows:

| features | `gram_block_rows=256`, against one row at a time |
|---:|---:|
| 256 | 5.1× |
| 1,000 | 6.6× |
| 2,000 | 5.9× |

With a solve every 512 rows, each speed-up falls to about 4×, because a
solve takes the same time either way. Blocking leaves `weight_sum` and the
timing of every prediction unchanged, and the output stays the same
however the stream is cut into chunks ([chunk
invariance](../explanation/chunk-invariance-and-streaming.md)).
The coefficients agree with the row-by-row fit to rounding, not to the
bit, since the product sums each entry in another order.

## Run the Gram's work on idle threads

Use this when a bank's few groups leave its threads idle, as a bank of
weekly blocks often does: fed a stream in time order, it reads a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) that holds one
week's rows, and so has one group to run.

1. **Leave `gram_threads` out (`"auto"`), or give a whole number for a
   fixed cap.** `"auto"` splits a group's work on its Gram across the
   bank's threads only when the work the bank runs at once, one piece per
   spec and group, leaves at least half its threads idle. A bank whose
   groups already keep every core busy gains nothing from a split. A whole
   number such as `gram_threads=8` lets one group's Gram use up to eight
   of the bank's threads whatever the groups. The threads are the bank's
   own, so it never adds one.
2. **Size the bank's threads with `threads=`,** as [Use more
   cores](use-more-cores.md#set-the-banks-thread-count) shows, or leave
   them at one per core.

The split helps past 256 columns with `gram_block_rows`, and past 725
without: up to 256 columns a block's product is one piece, and below 725
columns a row's work on the Gram stays on one thread. The output is the same to the bit at
every thread count.

The gain was measured on the same machine, at a load average of 7 to 16
(the load average is the number of processes running or waiting to run,
averaged over a minute). Each row of the table names what was timed:

| on eight threads, against one | as fast |
|---|---:|
| one Gram's 256-row block product, not a bank, at 2,000 columns | 6.1× |
| the whole bank at 2,000 columns: one target, float32 columns, `half_life=float("inf")`, no solve, `target_gaps="pairwise"`, `gram_block_rows=256` | 3.9× |
| one Gram's update a row at a time, without `gram_block_rows`, not a bank, at 1,000 to 10,000 columns | 2.2× to 3.1× |

The whole bank gains less than the product, because the rest of a row's
work, `O(k)`, stays on one thread. For a pass over 10,000 features and
9.5 million rows, the product alone falls from about 7.1 hours on one
thread to about 1.1 hours on eight.

## Time a wide stream by row, blocked, and on eight threads

This code learns 4,096 rows of 1,000 features three ways: a row at a
time, in blocks of 256 on one thread, and in blocks of 256 on up to eight.
It prints each time, whether the two blocked Grams are the same to the
bit, how far the blocked fit is from the row-by-row one, and the planted
coefficients beside the fit:

```python
import time

import numpy as np
import polars as pl

import polars_online as po

# 4,096 rows of 1,000 features in float32; y is the sum of the first ten, plus noise.
rng = np.random.default_rng(3)
n, k = 4_096, 1_000
x = rng.standard_normal((n, k), dtype=np.float32)
y = x[:, :10].sum(axis=1) + rng.standard_normal(n, dtype=np.float32)
features = [f"x{j}" for j in range(k)]
pl.DataFrame({**dict(zip(features, x.T)), "y": y}).write_parquet("wide.parquet")

# Learn the same stream three ways, keeping each Gram and time. The pass solves nothing.
grams, seconds = {}, {}
for way, block_rows, threads in [("by row", None, 1), ("blocked", 256, 1), ("blocked, 8", 256, 8)]:
    spec = po.spec.ewridge(
        "wide", targets=["y"], features=features, half_life=float("inf"), solve_every=float("inf"),
        gram_block_rows=block_rows, gram_threads=threads,
    )
    bank = po.ModelBank([spec])
    t0 = time.perf_counter()
    bank.fit(pl.scan_parquet("wide.parquet"))
    seconds[way] = time.perf_counter() - t0
    grams[way] = bank.gram("wide")[0]

# The times, the two blocked Grams bit for bit, and the fits against the truth.
print("time: " + ", ".join(f"{s:.2f} s {way}" for way, s in seconds.items()))
same = grams["blocked"]["comoments"].tobytes() == grams["blocked, 8"]["comoments"].tobytes()
print(f"the blocked Gram on 1 thread and on 8, bit for bit: {same}")
by_row, blocked = po.gram.solve(grams["by row"]), po.gram.solve(grams["blocked"])
print(f"the blocked fit and the row-by-row fit agree within 1e-12: {np.abs(by_row - blocked).max() < 1e-12}")
print("x0 to x2, planted 1, found " + ", ".join(f"{b:.2f}" for b in blocked[1:4]))
print(f"the 990 features with no effect: the largest |coefficient| {np.abs(blocked[11:]).max():.2f}")
```

```text
time: 0.39 s by row, 0.13 s blocked, 0.06 s blocked, 8
the blocked Gram on 1 thread and on 8, bit for bit: True
the blocked fit and the row-by-row fit agree within 1e-12: True
x0 to x2, planted 1, found 1.00, 0.99, 0.98
the 990 features with no effect: the largest |coefficient| 0.05
```

The blocks learned the stream about three times as fast as a row at a
time, and eight threads about twice as fast again. The times include
reading the file and each row's `O(k)` work, which stay on one thread, and
they are your machine's: the ones shown were printed on a 14-core machine
whose other work kept its load average near 5. The two blocked Grams are
the same to the bit, and the blocked fit agrees with the row-by-row one
within 1e-12. Each planted coefficient is found within about 0.02.

## Check the result

**Compare a Gram learned on one thread with one learned on several,**
bit for bit, as the recipe does, and time the two runs on the same file.

| what you see | what it means | what to do |
|---|---|---|
| a refusal of `gram_block_rows` | the spec solves on every row, the default under `half_life=float("inf")` and under `lam`, or has a `window_size` | give `solve_every` a spacing, such as `float("inf")` with a row cap, and no `window_size` |
| no speed-up from blocks | the spec solves often, and a solve takes the same time either way | solve more rarely, or leave the fits to `po.gram` after the pass |
| no speed-up from `gram_threads` | the bank's groups already keep every thread busy, so `"auto"` does not split, or the spec has fewer than 256 columns with blocks, or 725 without | nothing to gain from it; [Use more cores](use-more-cores.md) lists the other ways to use idle cores |
| the blocked fit differs from the row-by-row one in the last digits | the product sums each entry in another order | expected: the two agree to rounding |
