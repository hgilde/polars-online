<!-- kind: explanation -->
# Where the time and memory go

This page explains, for anyone who wants to know what sets a model bank's
speed and its memory, where a run spends each, and which settings change
them.

**A bank spends its time on the models' own arithmetic and its own work
around each row, and its memory on their state, a few chunks in flight,
the rows a delay holds, and what Polars has read ahead of the bank.** A [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits one model for
each [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the
description of a model) and each group, and takes its rows a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) at a time. The
thread counts and `chunk_size` change only the speed and the memory, never
the numbers a bank returns, as [Chunk invariance and
streaming](chunk-invariance-and-streaming.md) explains.
[Measured costs](../PERFORMANCE.md) has the measurement behind each
figure here that a recipe on this page does not print, with its date and
how it was made. To find the setting behind
a symptom, see [What to change when a number
surprises](#what-to-change-when-a-number-surprises). Every figure was measured on one Apple M4 Pro with 14 cores (10
performance, 4 efficiency), so your own times will differ.

## Where the time goes

**The bank's own work is about 60 to 160 ns a row at up to 50 features and
one target.** Around each row the bank clocks it, keeps its schedule,
copies its features in and writes its fields out. So its work is half or
more of a five-feature regression's row, and a sixth of a 50-feature
ridge's. A model that writes many fields a row spends 2 to 4 ns on each.
A five-feature `ewridge` row takes 107 ns in the model and 217 through the
bank, and a 50-feature one 795 and 955 ([timed against the model's step
alone](../PERFORMANCE.md#the-banks-work-around-a-row)). So a regression's row is the bank's bookkeeping as much as the model's at
five features, and the model's from about 50. A model that writes a field
per pair, as `ew_cov` does, spends most of a wide row writing it.

**Reading the file takes a small share of a run, and learning sets its
pace.** This recipe times each part of one run of `ewridge` with 20
features over a million generated rows: the read alone, learning with no
output, learning and predicting, and serving from a saved state without
learning. The times are your machine's:

```python
import time

import numpy as np
import polars as pl
import polars_online as po

# A million rows of 20 features, and a ridge over them.
rng = np.random.default_rng(9)
n = 1_000_000
features = [f"x{j}" for j in range(20)]
pl.DataFrame(
    {"y": rng.standard_normal(n)} | {f: rng.standard_normal(n) for f in features}
).write_parquet("rows.parquet")
spec = po.spec.ewridge("ridge", targets=["y"], features=features, half_life=1000.0)


# Millions of rows a second through one run.
def rows_a_second(run):
    start = time.perf_counter()
    run()
    return n / (time.perf_counter() - start) / 1e6


# Each part of a run, timed on its own.
query = pl.scan_parquet("rows.parquet")
bank = po.ModelBank([spec])
print(f"time to read the file:      {rows_a_second(lambda: query.collect()):.1f} million rows a second")
print(f"time to learn, no output:   {rows_a_second(lambda: bank.fit(query)):.1f} million rows a second")
print(f"time to learn and predict:  {rows_a_second(lambda: query.online.fit_predict([spec]).collect()):.1f} million rows a second")
print(f"time to serve from a state: {rows_a_second(lambda: query.online.predict(bank).collect()):.1f} million rows a second")
```

```text
time to read the file:      39.4 million rows a second
time to learn, no output:   2.9 million rows a second
time to learn and predict:  2.9 million rows a second
time to serve from a state: 6.5 million rows a second
```

Serving from a state skips learning, so it runs faster than learning, and
more so for a wider model: 2.2 times at 20 features in this recipe's run.
On a frame in memory, [`ewridge`'s
`predict`](../PERFORMANCE.md#serving-against-learning)
ran 1.85 times as fast as learning at 5 features and 2.82 times at 20.

**A model's speed is set by the work it does on each row.** `ewridge`
updates its Gram of `k` features on every row, which is `O(k²)` work, and
solves it only on its schedule, so ten times the features take several
times as long. This code writes 200,000
rows of 50 features to a file, then times one `ewridge` over it at 5
features and at 50, best of three runs. `ModelBank.fit` learns from every
row and keeps no output, so the time is the reading and the model:

```python
import time

import numpy as np
import polars as pl
import polars_online as po

# 200,000 rows of 50 features, on disk.
rng = np.random.default_rng(1)
n = 200_000
X = rng.standard_normal((n, 50))
pl.DataFrame(
    {"y": X[:, 0] + rng.standard_normal(n)} | {f"x{j}": X[:, j] for j in range(50)}
).write_parquet("wide.parquet")

# Time one fit at 5 features and at 50, best of three.
for k in (5, 50):
    spec = po.spec.ewridge("ridge", targets=["y"], features=[f"x{j}" for j in range(k)],
                           half_life=1000.0)
    seconds = []
    for _ in range(3):
        start = time.perf_counter()
        po.ModelBank([spec]).fit(pl.scan_parquet("wide.parquet"))   # learn every row, keep no output
        seconds.append(time.perf_counter() - start)
    print(f"time at {k} features: {n / min(seconds) / 1e6:.1f} million rows a second")
```

```text
time at 5 features: 2.8 million rows a second
time at 50 features: 0.9 million rows a second
```

These rates sit below those of [Rows a second for each
model](../PERFORMANCE.md#rows-a-second-for-each-model), most of all at
five features, because they include reading the file. That table times
`fit_predict` on a frame already in memory: the model and the bank's work
around it, without the read.

**A bank uses more cores when it has more pieces of work.** It learns each
group's rows one at a time, and runs its pieces of work on its own
threads. There is one piece per spec and group, more for a grid of
half-lives, and more again for a wide spec split across threads. [Use more cores](../how-to/use-more-cores.md) shows how
to give a bank more work it can run at once, set its thread count, and
search over factor sets in one pass. [Rows a second at each thread
count](../PERFORMANCE.md#rows-a-second-at-each-thread-count) measures the
speed at each thread count.

**Each chunk carries a fixed overhead, so a wide frame runs faster in large
chunks.** [Set the chunk size](../how-to/set-the-chunk-size.md) gives the
figures, and how to set it.

## Where the memory goes

**A bank's memory grows with the models' state and the rows a delay holds,
and never with the number of rows that have passed.** Every way of
running a bank works a chunk at a time. A run's memory has four
parts: the state, the chunks in flight, the rows an embargo, a
`label_clock` or a look-ahead target holds until they can be learned, and
what Polars' reader has read ahead. A few chunks are in flight
at once: three on the command line, and a few more in Python. [Why memory
stays
bounded](chunk-invariance-and-streaming.md#why-memory-stays-bounded) says
what bounds each part. [Bound a run's
memory](../how-to/bound-memory.md) cuts each.

**A run's peak grows a little with its length, but the bank does not.**
The growth is the memory allocator keeping pages it has freed, and nearly
all the rest of a run's peak is Polars reading ahead in the parquet file.
[Peak memory of each way of running a
bank](../PERFORMANCE.md#peak-memory-of-each-way-of-running-a-bank) measures
both, at 3M and 12M rows.

**Measure the state by saving it, since a saved state's size is what the
models keep.** This code fits one `ewridge` of 20 features on the first
10,000 rows of a generated file and on all 1,000,000, once with no groups
and once over 64. The state stays about the same size as the rows grow,
within a few counters' bytes, and it grows with the groups:

```python
import numpy as np
import polars as pl
import polars_online as po

# A million rows of 20 features and 64 groups, on disk.
rng = np.random.default_rng(3)
n = 1_000_000
features = [f"x{j}" for j in range(20)]
pl.DataFrame(
    {"g": rng.integers(0, 64, n), "y": rng.standard_normal(n)}
    | {f: rng.standard_normal(n) for f in features}
).write_parquet("long.parquet")

# The state's size after 10,000 rows and after all of them, with and without groups.
for group in (None, "g"):
    spec = po.spec.ewridge("ridge", targets=["y"], features=features, half_life=1000.0, group=group)
    for rows in (10_000, 1_000_000):
        bank = po.ModelBank([spec])
        bank.fit(pl.scan_parquet("long.parquet").head(rows))     # learn, keep no output
        kb = len(bank.save_bytes()) / 1024                        # the state, as saved
        print(f"{'64 groups' if group else 'one group'}, after {rows:>9,} rows: state {kb:,.0f} KB")
```

```text
one group, after    10,000 rows: state 10 KB
one group, after 1,000,000 rows: state 11 KB
64 groups, after    10,000 rows: state 586 KB
64 groups, after 1,000,000 rows: state 588 KB
```

**For a statistic per group over a long stream, use a spec's `group=`,
which keeps one set of running sums per group.** Polars' `rolling` over
groups, with `.over("group")` or `group_by=`, holds every row of its input:
6.5 GB on 12M rows with `.over("group")`, as [Peak memory of each way of
running a bank](../PERFORMANCE.md#peak-memory-of-each-way-of-running-a-bank)
measures.

**A window operator, such as `po.ewm_mean`, holds the rows of its window,
and does the same work on a row whatever the window's length.** [What a window operator
costs](../PERFORMANCE.md#what-a-window-operator-costs) measures its time
and memory against Polars' `rolling`.

## What to change when a number surprises

**Each symptom has a cause in the bank or in Polars, and a setting that
moves it.** Where a number is not what you expected, find what you see
here:

| what you see | where the time or memory goes | what moves it |
|---|---|---|
| a row slower than [its model's rows a second](../PERFORMANCE.md#rows-a-second-for-each-model) | the model's own learning step, `O(k²)` a row for a model that solves, and its solves on their schedule; below about 50 features, as much again in the bank's work around the row; and for a model with many fields, writing them | `solve_every` and `max_rows_between_solves`; `gram_block_rows` and `gram_threads` for a wide `ewridge`; only the `stats` you read for `ew_cov` |
| a grid slower than one model | a ridge or feature-set grid shares one set of sums; each half-life of a grid keeps its own, run alongside the others | the number of half-lives |
| small chunks slower than large ones | a fixed overhead per call: handing the frame across alone takes about 8 ms at 10,000 columns ([Set the chunk size](../how-to/set-the-chunk-size.md)), and gathering the columns and assembling the output add to it | `chunk_size`, 100,000 by default on the query and the command line; the eager call takes the frame whole unless given one |
| memory that grows with the file | Polars reading ahead in the parquet file, sized from its thread count, and the allocator keeping pages it has freed | `POLARS_ROW_GROUP_PREFETCH_SIZE`, `POLARS_MAX_THREADS` |
| an eager `fit_predict` above its output | a spec of many groups keeps buffers of about its output again until its groups are done; a spec of one group or none holds its output and two sets of buffers more | a smaller frame, or the streaming path, whose peak is a chunk's |
| memory that jumps with a filter | a filter before the bank makes Polars hold several blocks of each file per thread | a filter after the bank, or weight 0 on the rows to skip |
| one core busy | one piece of work per spec and group per chunk; within one, the rows go one at a time | more groups or specs, `POLARS_ONLINE_MAX_THREADS`, `shards` for a wide `marginal`, and `gram_threads` for a wide `ewridge` or `ew_cov` |
| a query slower than the eager bank at many groups | the query reads its input as the frames Polars hands it, and the same bank code then takes longer to group, run and assemble them: 236 ns a row against 140 at 10,000 groups ([the query against the eager bank](../PERFORMANCE.md#a-query-against-the-eager-bank)) | `ModelBank.fit_predict` on a frame that fits in memory; the query where it does not |
| a window's memory | its ring of snapshots, capped per ring | `snapshot_every`, `max_rows_between_snapshots`, `window_budget` |
| a wide `marginal` | every pair of every target | `cross_lags`, `bins`, `feature_moments="shared"`, `shards` |
| `bocpd` slowing as a stream runs | the run lengths it keeps | `max_run`, `prune_below` |
| a window operator | one queue of rows for each set of operators that share it, the same work a row whatever the window's length; memory about one window of rows | operators with the same direction, half-life, window and `closed` share one queue of rows |
| a model whose target is a window expression | the bank's own window core, at about 1.6 times the column form's time and less memory | the column form, `with_windows(..., like=spec)` |
| learning slower than serving | `predict` skips learning | serve with `predict` where the model need not learn from the rows |
