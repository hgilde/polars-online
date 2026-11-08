<!-- kind: explanation -->
# Where the time and memory go

This page explains, for anyone who wants to know what sets a model bank's
speed and its memory, where a run spends each, and which settings change
them.

**A bank spends its time on the models' own arithmetic and its own work
around each row, and its memory on their state, a few chunks in flight,
and what Polars has read ahead of the bank.** A [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits one model for
each [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the
description of a model) and each group, and takes its rows a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) at a time. The
thread counts and `chunk_size` change only the speed and the memory, never
the numbers a bank returns, as [Chunk invariance and
streaming](chunk-invariance-and-streaming.md) explains.
[PERFORMANCE.md](../PERFORMANCE.md) has the measurement behind every figure
here, and a table of what to change when a number is not what you
expected. Every figure was measured on one Apple M4 Pro with 14 cores (10
performance, 4 efficiency), so your own times will differ.

## Where the time goes

**The bank's own work is about 60 to 170 ns a row at up to 50 features and
one target.** Around each row the bank clocks it, keeps its schedule,
copies its features in and writes its fields out. So its work is half or
more of a five-feature regression's row, and a sixth of a 50-feature
ridge's. A model that writes many fields a row spends 2 to 4 ns on each
([the headline](../PERFORMANCE.md#the-headline) has the share by width).

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
