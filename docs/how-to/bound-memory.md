<!-- kind: how-to -->
# Bound a run's memory

This how-to is for anyone whose run of a model bank holds more memory than
the machine can spare, and who wants it to hold less.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) works a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads) at a time, so its
memory never grows with the number of rows that have passed. [Why memory
stays
bounded](../explanation/chunk-invariance-and-streaming.md#why-memory-stays-bounded)
explains why, and [Peak memory of each way of running a
bank](../PERFORMANCE.md#peak-memory-of-each-way-of-running-a-bank)
measures a whole run. What a run holds has four parts: the
[state](../GLOSSARY.md#saving-and-serving) the models keep, the chunks in
flight, the rows a delay or a window holds, and what Polars reads ahead of
the bank. Polars' own settings cut the last part, and `refit_budget` cuts
the rows a robust regression holds in its state for its refit.

## Prerequisites

These steps need the library and a run that reads a file:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `ticks.parquet`, 400 rows that Example data writes.
- A run that reads a parquet file with `pl.scan_parquet`, which reads the
  file as a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads): a
  query's `online.fit_predict`, `bank.fit_predict_batches`, or a loop of
  your own over its chunks.

## Cut what Polars reads ahead

Polars, when it runs a query, reads blocks of a parquet file ahead of
whatever consumes them, sized from its thread count. While a bank is the
slowest step, a local disk needs none of that read-ahead, so it is the
part of memory you can cut. Three of Polars' environment variables set it,
in the shell or with `os.environ` in Python.

1. **Set `POLARS_ROW_GROUP_PREFETCH_SIZE` to `1`, the number of blocks
   Polars reads ahead.** It saves the most. Polars reads it each time a
   query that reads the file starts to run. So set it at any point before
   the call that runs the query, such as the one that writes its result to
   a file or `bank.fit_predict_batches(lf)`.
2. **Run the query as before.** Nothing else in it changes.

```python
import os

os.environ["POLARS_ROW_GROUP_PREFETCH_SIZE"] = "1"   # blocks of the parquet file read ahead: read when a query starts to run
```

The byte cap, `POLARS_ROW_GROUP_PREFETCH_KBYTES_BUDGET`, rarely binds,
because it counts *compressed* bytes, which take almost no memory for a
memory-mapped local file. The third variable, `POLARS_MAX_THREADS`, is the
next section's.

Measured as peak resident memory on 8M rows of 12 columns in 80 blocks,
with the allocator's page retention off so that the figure is live data:

| what runs | default prefetch | `POLARS_ROW_GROUP_PREFETCH_SIZE=1` |
|---|---:|---:|
| `lf.online.fit_predict(...).sink_parquet(...)` | 1.63 GB | 1.12 GB |
| `bank.fit_predict_batches(lf)` | 1.41 GB | 1.07 GB |

**The saving depends on how much data a block holds, so compare a run at
the default with one at `1` on your own files.** On a file whose blocks
hold 262,000 rows, the setting took a query from 1.86 GB to 0.51 GB,
measured on 2026-09-02.

## Give Polars fewer threads and the bank every core

**To fit a run in less memory, give Polars fewer threads and the bank
every core.** Polars sizes its read-ahead from `POLARS_MAX_THREADS`, while
the bank's `threads` changes only the bank's speed.

1. **Set `POLARS_MAX_THREADS` before `import polars`,** since Polars reads
   it when it is first imported: in the shell, as
   `POLARS_MAX_THREADS=4 python fit.py`, or with `os.environ` at the top of
   the script.
2. **Leave the bank's `threads` unset,** which gives the bank one thread
   per core whatever `POLARS_MAX_THREADS` says, unless
   `POLARS_ONLINE_MAX_THREADS` sets another count. [Use more
   cores](use-more-cores.md) lists the calls that take `threads=`, to give
   a count of your own.

This code is a script of its own, run in a new process, since Polars reads
`POLARS_MAX_THREADS` when it is first imported. It uses the file
`ticks.parquet` from [Example data](../../README.md#example-data):

```python
import os
os.environ["POLARS_MAX_THREADS"] = "4"           # before import polars: sizes the reader's read-ahead

import polars as pl
import polars_online as po

# The run itself does not change.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1", "x2"], clock="t",
                       half_life=600.0, gap_cap=300.0, group="stock_id")
(pl.scan_parquet("ticks.parquet")                     # written in Example data
   .online.fit_predict([spec], chunk_size=200_000)   # threads unset: the bank keeps every core
   .sink_parquet("fit.parquet"))
```

On 12M rows over 64 groups, with one spec of four features and two
half-lives, measured on 2026-09-29 on 0.13.0's build at a load average of
1.9 to 2.1 by `scripts/parallel_bench.py`:

| Polars threads | bank threads | time | peak memory |
|---:|---:|---:|---:|
| 14 | 14 | 1.63 s | 0.86 GB |
| 4 | 14 | 1.87 s | 0.58 GB |

Keeping Polars at four threads holds 33% less memory, for 15% more time.

## Bound the rows a refit holds

`huber`, a robust regression, refits each target's first rows to
convergence once it has learned them: 500 rows by default, 10 a feature
past 50 features, and off above 100, set by `huber_refit_rows`. Until
then each target holds those rows, in each
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads), so a spec over
many groups can hold many rows at once. `sgd(loss="huber")`, `quantile`
and `marginal`'s Huber bins refit the same way and take the same bound,
each with a count of its own.
[The refit of the first
rows](../explanation/robust-scale.md#the-refit-of-the-first-rows) explains
what the refit is for.

1. **Give `refit_budget` the most memory, in MiB, the refit buffers of
   all the spec's groups may reserve together.** It is 320 MiB by default,
   or `inf` for no bound.
2. **To hold no refit rows at all, give the model's count `0`,** which
   turns the refit off and leaves the fit made in one pass:
   `huber_refit_rows=0` for `huber` and `sgd(loss="huber")`,
   `quantile_refit_rows=0` for `quantile`, and `bin_refit_rows=0` for
   `marginal`'s Huber bins.
3. **Read `bank.summary()`'s `refit_given_up`** after the run, which counts
   each group's targets that gave the refit up.

**A learned row takes `(features + 10) · 8` bytes under the MAD scale, the
default, and `(features + 4) · 8` under `huber_scale="rms"`.** Under the
MAD, those bytes hold its features, target, weight and two decay factors,
and what the scale reads. So one feature and 500 rows take about 44 KB a target under the
MAD, and 20 KB under `"rms"`. A group shorter than `huber_refit_rows`
never refits, and holds its rows to its end.

**Each group reserves the most its buffers can hold when it first appears,
and the group that would pass the bound, with every new group after it,
gives its refit up.** Under the MAD, a group reserves `(features + 26) · 8`
bytes for each target and each row of the count. Under `"rms"` it reserves
`(features + 20) · 8`. The [`huber`
docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber)'s
Notes give both. A group that gives its refit up keeps its one-pass fit, and a
[readiness warning](../GLOSSARY.md#warm-up-and-readiness) says so once.

Each group's reservation is saved with the bank. So which groups refit
depends neither on the chunking, nor on the order the groups run in, nor
on a save and load between them. In exchange, once the bound is reached no
later group refits, though the first groups' buffers have long been freed.
These figures are for `huber` over groups of 150 rows, with two targets
and 20 features, fed from parquet. They give the peak resident memory
after the fit, and the saved state's size, each the median of three runs,
which spread by up to 120 MiB:

| groups | `huber_refit_rows=0`: peak; state | the default, 320 MiB: peak; state; targets given up | `refit_budget=inf`: peak; state |
|---|---|---|---|
| 5,000 | 572 MiB; 128 MiB | 691 MiB; 177 MiB; 8,178 of 10,000 | 1,199 MiB; 391 MiB |
| 20,000 | 1,195 MiB; 512 MiB | 1,390 MiB; 563 MiB; 38,178 of 40,000 | 3,915 MiB; 1,565 MiB |

A reservation is the most a buffer can hold, and these groups' buffers
held less.

This code fits `huber` on one feature over 20 groups of 150 rows, under a
bound of 1 MiB. Each group reserves `(1 + 26) · 8` bytes for each of the
500 rows of the count, so the bound holds 9 groups' refits. It builds its
own rows:

```python
import textwrap
import warnings

import numpy as np
import polars as pl
import polars_online as po

# 20 groups of 150 rows, y = 2x plus noise, saved.
rng = np.random.default_rng(4)
x = rng.standard_normal(3_000)
pl.DataFrame({"g": np.tile(np.arange(20), 150), "x": x, "y": 2 * x + rng.standard_normal(3_000)}).write_parquet("many.parquet")

# Step 1: a bound of 1 MiB on the refit buffers of all 20 groups together.
robust = po.spec.huber("robust", targets=["y"], features=["x"], half_life=float("inf"), group="g",
                       refit_budget=1.0)
bank = po.ModelBank([robust])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bank.fit(pl.scan_parquet("many.parquet"))                      # learn, and keep no output
print(textwrap.fill(str(caught[0].message).split(". ")[0] + ".", 100))   # the readiness warning's first sentence
# Step 3: the groups that refit, against the bound's share of each group's reservation.
reserved = (1 + 26) * 8 * 500                                      # bytes a group reserves
given_up = bank.summary()["refit_given_up"].sum()
print(f"groups that refit: {20 - given_up} of 20; the bound holds {2**20 // reserved}")
```

```text
spec "robust": at group "9", its groups' refits would hold more than refit_budget = 1 MiB, each
group reserving 106 KiB, so that group and every new group after it give up the refit and keep the
fit made in one pass.
groups that refit: 9 of 20; the bound holds 9
```

## Bound the other parts

**The groups, a window's snapshots, the rows a delay holds and the chunks
in flight have bounds of their own.** Each is
set where its topic lives:

| what grows | the setting that bounds it | where it is set |
|---|---|---|
| the groups a bank keeps | `group_close`, or `bank.drop_groups(...)` | [Bound the memory of many groups](fit-one-model-per-group.md#bound-the-memory-of-many-groups) |
| a hard window's snapshots | `window_budget` | [Cap the snapshots' memory](forget-rows-past-a-window.md#cap-the-snapshots-memory) |
| the rows a delay holds | the `embargo` or the `label_clock` delay, times the rows' rate; `max_held_rows` caps the rows `label_clock` holds in each group | [Learn targets known late](learn-targets-known-late.md) |
| the chunks in flight | `chunk_size` | [Set the chunk size](set-the-chunk-size.md) |

## Check the result

**Compare the run's peak memory at the default and at your settings, each
in a new process,** since `POLARS_MAX_THREADS` is read when Polars is
first imported. Measure the peak resident memory with the operating
system's own tool, such as `/usr/bin/time -l python fit.py` on macOS or
`/usr/bin/time -v python fit.py` on Linux. Polars' settings and the
bank's `threads` leave the output as it was, so write both runs' output
to files and check that `pl.read_parquet(a).equals(pl.read_parquet(b))`
is `True`.
`refit_budget` and `huber_refit_rows=0` change the fit of each group that
gives the refit up, so check `refit_given_up` instead.

**Measure the state's share by saving it.** `len(bank.save_bytes())` is
the size of what the models keep, which grows with the models' settings
and the groups, never with the rows that have passed.
