<!-- kind: how-to -->
# Use more cores

This how-to is for anyone whose model bank runs on fewer cores than the
machine has, and who wants it to finish sooner by running on more.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) learns
each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads)'s rows one at
a time, because what it learns from each row depends on the row before. On every
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) it splits its
work into pieces, one per [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
and group, or one per spec without a `group`. It runs the pieces on its own
threads, separate from Polars', largest first, so a few big groups do not
leave cores idle at the end. So a bank with one spec, one group and one
[half-life](../GLOSSARY.md#time) is one thread's work per chunk, while Polars reads and writes in
parallel around it. All of it runs in one process, by design ([What this
is not](../../README.md#what-this-is-not)). The thread count never changes
the numbers, as [Why the thread count cannot change the
numbers](../explanation/chunk-invariance-and-streaming.md#why-the-thread-count-cannot-change-the-numbers)
explains.

## Prerequisites

These steps need the library and a machine with several cores:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `ticks.parquet`, 400 rows of four stocks in the column `stock_id`. One
  example builds its own file.
- A run of a bank on a machine with more than one core.

## Give the bank more pieces of work

**To use more cores, give the bank more pieces of work: more groups, more
specs, more half-lives, or a wide spec split across threads.**

1. **Set `group=` on each spec** whose rows fall into groups, such as
   `group="stock_id"`. Each group is a piece of its own.
2. **Pass every spec you need to one bank, or to one query's
   `online.fit_predict`,** in place of one bank per spec. Each spec is a
   piece of its own, and the query reads its file once.
3. **Give a spec several half-lives as a list,** as
   `half_life=[100.0, 1000.0]`, in place of a spec per half-life.
4. **For one wide spec, split its work too,** as [Split one wide spec
   across threads](#split-one-wide-spec-across-threads) shows.

| what to add | how it runs in parallel | measured |
|---|---|---|
| groups, with a spec's `group=` | one piece per group | k=20 over 64 groups: 7.3× from 1 to 14 threads on a 14-core machine |
| specs, in the list the bank takes | one piece per spec | eight single-group specs, k=20 over 300k rows, run in 155 ms in one bank, against 641 ms one at a time |
| half-lives, as a list given to `half_life=` | each half-life in a grid is its own set of running sums, and a piece's half-lives run in parallel on the bank's threads. Ridge and feature-set grids share one set of sums and are expanded at solve time, so they need no thread | |
| `shards`, on a wide `marginal` | a piece that runs on several threads: `shards` splits its pairs across the bank's threads within a group ([`marginal`](../MODELS.md#marginal--every-pairs-moments-kept-in-the-state)) | |
| `gram_threads`, on a wide `ewridge` or `ew_cov` | a piece that runs on several threads: the work on each row's `k × k` [Gram](../GLOSSARY.md#fitting-a-model), or a block's merge, is split across the bank's threads within a group, from 725 columns, or 256 under `gram_block_rows`. A count caps the share of the threads. The default, `"auto"`, splits only when the groups running at once leave half the threads idle, since a split speeds a run up only on idle threads | one group at 2,000 features: 3.0× under `"auto"` on 14 threads; four groups at 800 features: 1.6×; 28 groups: even, where taking every thread was 9-11% slower ([`gram_threads`](../GRAM.md#one-wide-gram-on-several-threads)) |
| a Python thread that reads ahead | Python's global lock is released while a chunk is in the bank, so a Python reader thread can run ahead of `ModelBank.fit_predict` | |

**Specs run side by side, and each gives the numbers it gives alone.**
This code runs the comparison of the table's second row on a generated
parquet file: eight `ewridge` specs of 20 features, one per half-life, in
one bank and then one at a time. It prints the two times, which are your
machine's; the ones shown were printed on the machine the table was
measured on. They include reading the file, where the table times the bank
alone, so they run longer than the table's. It then prints whether the
outputs are equal to the bit, which they are:

```python
import time

import numpy as np
import polars as pl
import polars_online as po

# A file of 20 features, and eight specs that differ only in half-life.
rng = np.random.default_rng(2)
n = 300_000
features = [f"x{j}" for j in range(20)]
pl.DataFrame(
    {"y": rng.standard_normal(n)} | {f: rng.standard_normal(n) for f in features}
).write_parquet("specs.parquet")
specs = [po.spec.ewridge(f"h{h}", targets=["y"], features=features, half_life=float(h))
         for h in (100, 200, 400, 800, 1600, 3200, 6400, 12800)]   # eight specs, one piece each

# The eight in one bank, then one at a time.
start = time.perf_counter()
together = pl.scan_parquet("specs.parquet").online.fit_predict(specs).collect()
in_one_bank = time.perf_counter() - start

start = time.perf_counter()
alone = [pl.scan_parquet("specs.parquet").online.fit_predict([s]).collect() for s in specs]
one_at_a_time = time.perf_counter() - start

# The times, and whether the outputs agree.
print(f"time: {in_one_bank:.2f} s in one bank, {one_at_a_time:.2f} s one at a time")
same = all(together[s["name"]].equals(a[s["name"]]) for s, a in zip(specs, alone))
print("the same output either way:", same)
```

```text
time: 0.42 s in one bank, 1.56 s one at a time
the same output either way: True
```

## Set the bank's thread count

**Set the bank's thread count with `threads=`, on the bank or the call
that makes one.** `ModelBank(specs, threads=8)`, `ModelBank.load(path,
threads=8)`, `po.fit_predict(..., threads=8)`, `po.predict`, and the frame
namespaces' `fit_predict` and `predict` all take it. So do `po.gram`'s
`lars_path`, `lars_paths` and `solve_subsets`, and the command line, as
`--threads 8` or a `threads` key in the TOML.

1. **Give `threads=` an integer,** a set of that many threads. Every bank
   given the same count shares one set, which goes away with the last of
   them, and asking for the size the process's default set already has
   uses that set.
2. **Or leave `threads` as `None`, the default,** for the process's
   default set, one thread per core unless `POLARS_ONLINE_MAX_THREADS`
   sizes it.

The count is not saved with a state, and the output and the saved state
are the same at any count. A count below 1 or above 1024 (or the
machine's core count, where that is larger), a float or a bool is
refused, naming `threads`: 0 is not "one per core" here, `None` is.

**Two environment variables size the default sets of threads, one for
each:**

| variable | sizes | unset or `0` | read at | what took effect |
|---|---|---|---|---|
| `POLARS_ONLINE_MAX_THREADS` | the bank's default threads, for a bank or call given no `threads` | one thread per core | the first chunk a bank without `threads` fits or scores, or the first `po.thread_pool_size()` | [`po.thread_pool_size()`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.thread_pool_size) |
| `POLARS_MAX_THREADS` | Polars' readers and writers, and their read-ahead ([Bound a run's memory](bound-memory.md#give-polars-fewer-threads-and-the-bank-every-core)) | one thread per core | `import polars`, which `import polars_online` runs | `pl.thread_pool_size()` |

Polars has no per-call thread count, so its threads are set by its
variable alone, before `import polars`: in the shell, as
`POLARS_MAX_THREADS=8 python fit.py`, or with `os.environ` at the top of the
script. `POLARS_ONLINE_MAX_THREADS` is read once, when the default threads
are first used, and ignored after. `threads=` never reads it. A
`POLARS_ONLINE_MAX_THREADS` that is not a whole number, or is past the
same bound as `threads`, makes the first call that uses the default
threads raise `ValueError`, naming the variable.
Giving both sets every core slows neither, because the bank's work never
calls back into Polars' threads.

## Split one wide spec across threads

**Give `shards="auto"` to spread one wide `marginal` over the threads.**
The bank runs groups and specs in parallel, so a wide `marginal`, a screen
of every feature against every target, on one group is otherwise one
thread's work. `shards=10` splits its pairs into ten ranges of features,
each run on a thread of its own, a block of rows at a time. `"auto"` sizes
the split to the width and the bank's threads. It leaves whole a narrow
spec, and a spec under a [hard window](../GLOSSARY.md#fitting-a-model)
that snapshots every row, the default, whose block is a single row. The
numbers are the same to the bit at any count, so a saved bank resumes
under any count. On 14 threads, through `ModelBank.fit_predict` with one
spec and one group, the best of three runs, measured on 2026-09-25:

| the spec | `"auto"` ran the bank |
|---|---|
| 10,000 features, nine targets, lags and bins | 4.9 times as fast |
| the moments of one target alone | 1.2 times as fast, since the bank's own work on each row does not split |

**A wide `ewridge` or `ew_cov` splits by `gram_threads`,** whose default,
`"auto"`, already splits from 725 columns, or 256 under `gram_block_rows`, when the groups running at once
leave half the threads idle. [One wide Gram on several
threads](../GRAM.md#one-wide-gram-on-several-threads) measures it.

## Search over factor sets in one pass

**To search over factor sets, build one spec per set and pass the whole
list to one `lf.online.fit_predict` call.** Each spec runs as its own
piece of work on each group, and the query reads the file once. Its output
has one column per spec, which is what `po.eval.compare_specs` reads, and
one state file, `grid.state`, holds them all. `y` in `ticks.parquet`
(Example data's `df`) is built from `x0` and `x1`, so the set of those two scores best, and adding `x2`, which
plays no part, scores a little lower. Each spec's [grid
point](../GLOSSARY.md#grids) is chosen on the first half of the rows and
scored on the second. The best point's R² over the rows that chose it
overstates what choosing it earns ([Calibrating a
model](../CALIBRATION.md)). Here the second half scores
higher all the same, since the first holds each fit's
[warm-up](../GLOSSARY.md#warm-up-and-readiness).

This code is a script of its own, run in a new process, since Polars reads
`POLARS_MAX_THREADS` when it is first imported. It uses the file
`ticks.parquet` from [Example data](../../README.md#example-data):

```python
import os
os.environ["POLARS_MAX_THREADS"] = "8"          # Polars' readers and writers: read at import polars

import polars as pl
import polars_online as po
from itertools import product

# Three factor sets, each with and without standardizing: six specs.
factors = {"mkt": ["x0"], "mkt-sz": ["x0", "x1"], "mkt-sz-val": ["x0", "x1", "x2"]}

def factor_spec(name, features, standardize):
    return po.spec.ewridge(f"{name}-std{standardize:d}",                  # "mkt-std0", "mkt-std1", ...
                           targets=["y"], features=features, clock="t", gap_cap=300.0,
                           group="stock_id", session="session", session_gap=60.0,
                           half_life=[100.0, 1000.0], ridge=[1e-3, 0.1],   # gridded inside the spec
                           standardize=standardize)

specs = [factor_spec(n, f, s) for (n, f), s in product(factors.items(), [False, True])]

# One pass over the file fits all six.
(pl.scan_parquet("ticks.parquet")
   .online.fit_predict(specs, chunk_size=200_000, save_state="grid.state", threads=8)
   .sink_parquet("grid.parquet"))

# Each spec's grid point chosen on the first half of the rows, then scored on the second.
out = pl.read_parquet("grid.parquet")
half = out.height // 2
first = po.eval.compare_specs(out.head(half), specs)                 # one row per spec and grid point
second = po.eval.compare_specs(out.tail(out.height - half), specs)
picks = first.sort("r2", descending=True).group_by("spec", maintain_order=True).first()
best = picks.join(second, on=["spec", "slot", "target"], suffix="_after").sort(pl.col("r2_after").round(3), "spec", descending=[True, False])
for spec_name, r2, after in best.select("spec", "r2", "r2_after").iter_rows():
    print(f"{spec_name:<15} R² {r2:.3f} where chosen, {after:.3f} after")
```

```text
mkt-sz-std0     R² 0.930 where chosen, 0.944 after
mkt-sz-std1     R² 0.930 where chosen, 0.944 after
mkt-sz-val-std0 R² 0.915 where chosen, 0.935 after
mkt-sz-val-std1 R² 0.913 where chosen, 0.935 after
mkt-std0        R² 0.672 where chosen, 0.738 after
mkt-std1        R² 0.672 where chosen, 0.738 after
```

On 2.56M rows over 64 groups, the six specs make 6 × 64 pieces of work
every chunk. There the same query takes 29.0 s at one thread and 4.3 s at
fourteen, measured on 2026-09-29 by `scripts/parallel_bench.py`.

## Check the result

**Check how many threads the bank runs on with `po.thread_pool_size`, and
that the output is the same at every count.** `po.thread_pool_size()`
gives the default threads' number, and `po.thread_pool_size(threads=n)`
the number a bank given `threads=n` runs on, without starting them. This
code fits one spec on one thread and on four, and checks the two agree. It
uses the file `ticks.parquet` from [Example
data](../../README.md#example-data):

```python
import polars as pl
import polars_online as po

spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t",
                       half_life=600.0, gap_cap=300.0, group="stock_id")
one = pl.scan_parquet("ticks.parquet").online.fit_predict([spec], threads=1).collect()
four = pl.scan_parquet("ticks.parquet").online.fit_predict([spec], threads=4).collect()
print("the same output on 1 and 4 threads:", one.equals(four))
print("threads=4 runs on", po.thread_pool_size(threads=4), "threads")   # reports, starts nothing
```

```text
the same output on 1 and 4 threads: True
threads=4 runs on 4 threads
```

**Then time the run at `threads=1` and at the count you chose, on the same
file.** For `ewridge` with 20 features over 64 groups, the rows a second
rose 7.3 times from 1 to 14 threads on a 14-core machine. [Rows a second
at each thread count](../PERFORMANCE.md#rows-a-second-at-each-thread-count)
gives the rows a second at each count.
