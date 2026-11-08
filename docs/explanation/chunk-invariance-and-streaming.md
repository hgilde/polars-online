<!-- kind: explanation -->
# Chunk invariance and streaming

This page is for anyone who runs a model bank over a stream too large to
hold in memory, and wants to know which of its numbers depend on how the
stream is cut up, read and ordered.

**A model bank reads its rows one at a time, in the order it is given
them, and keeps only what its models have learned.** A [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set of
models over the same rows, one for each
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the description
of a model) and each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads)
(the rows that share one value of the column `group=` names). It takes the
[stream](../GLOSSARY.md#the-model-bank-and-what-it-reads) in
[chunks](../GLOSSARY.md#the-model-bank-and-what-it-reads), the rows it
takes in at once, and carries each model's
[state](../GLOSSARY.md#saving-and-serving) (what it has learned) from one
chunk to the next. Three guarantees follow, each from its own part of that
design:

| guarantee | it follows from |
|---|---|
| any chunking gives the same numbers, to the last bit | each row is learned in turn, and a chunk's end is not a step any model takes |
| memory never grows with the rows that have passed | most models keep [running sums](../GLOSSARY.md#fitting-a-model) in place of their rows, and the rest keep a bounded set of values, such as coefficients or cluster centers. A delay, such as an [`embargo`](../GLOSSARY.md#time), holds only the rows it has not yet released |
| every prediction is out-of-sample | each row is predicted before it is learned ([Why every prediction is out-of-sample](out-of-sample.md)) |

Each holds for whatever order the rows arrive in. But that order decides
the numbers, so the bank checks it and never changes it.

## Why the chunking cannot change the numbers

**A chunk's end is not an event in the stream.** A model learns each row
of a group from the state the row before it left, whether that row came
in the same chunk or the one before. Everything a model does on a
schedule is measured on the [clock](../GLOSSARY.md#time), in weight or in
rows, never in chunks. That covers its [decay](../GLOSSARY.md#time), its
[solves](../GLOSSARY.md#fitting-a-model) and the snapshots
a [hard window](../GLOSSARY.md#fitting-a-model) takes. It covers the
release of a row an [`embargo`](../GLOSSARY.md#time) (a delay before a row
is learned) holds, and the close of the window a [formula
target](../GLOSSARY.md#fitting-a-model) reads. So one chunk or a thousand
gives bit-identical numbers. So does a save and resume between two
chunks, since a saved state is what the bank carries across a chunk's end
([A bank's state and resuming it](state-and-resuming.md)).

**Only which rows carry the coefficients can differ, and only by
default.** A model writes its `coef`, and `support_coef` and the standard
errors (`se_coef`, `se_coef_hc0`, `se_coef_hac`) beside it, on each group's last row in every
chunk, not counting [skipped](../GLOSSARY.md#the-model-bank-and-what-it-reads)
rows.
So smaller chunks report the coefficients on more rows, each the fit as it
stood after that row. Under `coef_every` or
`max_rows_between_coefs`, which place them on the clock or by a count of
rows, they do not move with the chunking
([Read the coefficients](../how-to/read-the-coefficients.md#choose-which-rows-carry-coef)).

This code runs the same spec over `df` as one chunk and as 400 chunks of
one row, through `fit_predict_batches`, and compares the predictions and
the rows that carry `coef`. The first two rows carry none in either run:
the fit needs a row per coefficient, three here, so the model solves
nothing on its first two rows. It reads `df` from the README's [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], clock="t", half_life=100.0, gap_cap=300.0)
one_chunk = po.ModelBank([spec]).fit_predict(df)["m"]                                       # all 400 rows at once
by_row = pl.concat(po.ModelBank([spec]).fit_predict_batches(df, chunk_size=1))["m"]       # 400 chunks of one row
print(f"largest difference in pred_y: {(one_chunk.struct.field('pred_y') - by_row.struct.field('pred_y')).abs().max()}")
print(f"rows with coef: {one_chunk.struct.field('coef').is_not_null().sum()} as one chunk, "
      f"{by_row.struct.field('coef').is_not_null().sum()} as 400")
```

```text
largest difference in pred_y: 0.0
rows with coef: 1 as one chunk, 398 as 400
```

## Why the thread count cannot change the numbers

**No model reads another's state, so the threads that run them cannot
change what any of them learns.** The bank shares the work of its specs
and groups among its threads, set by a bank's `threads=`. Each group's
rows still go through its model in order. A model with many features can
also split the learning of one group's rows across threads, up to its
spec's `gram_threads` (`ewridge`, `ew_cov`), and a `marginal` across its
`shards`. Either gives the same bits at any thread count. So any thread
count gives the same numbers ([Use more cores](../how-to/use-more-cores.md)).

## Why memory stays bounded

**Memory never grows with the number of rows that have passed.** A model that keeps
running sums holds the same few numbers after a million rows as after ten
thousand. Every way of running a bank works a chunk at a time, so the rows
themselves are held only while the bank works on their chunk, apart from
the rows a delay holds until they can be learned. What a bank
holds has four parts:

| part | grows with | what bounds it |
|---|---|---|
| the state | the models, their settings and the number of groups | not the stream's length. A hard window's snapshots and the bins of `marginal` grow with their settings, and each group's are capped, at 256 MiB by default |
| the chunks the bank is working on | `chunk_size` | a few chunks at once |
| the rows an embargo, a `label_clock` (each row's own time its target is known) or a formula target's window holds until they can be learned | the delay or the window, times the rows' rate | not the stream's length |
| what Polars has read ahead of the bank | Polars' thread count | Polars' own settings ([Bound a run's memory](../how-to/bound-memory.md#cut-what-polars-reads-ahead)) |

[Peak memory of each way of running a
bank](../PERFORMANCE.md#peak-memory-of-each-way-of-running-a-bank)
measures a whole run's peak and says how the figures on this page were
measured. [Where the memory goes](where-the-time-and-memory-go.md#where-the-memory-goes)
shows the state's size staying flat from 10,000 rows to 1,000,000.

**Filter after the bank, not before it.** A `filter` before the bank makes
Polars read several blocks of each parquet file ahead per thread. A run
over 12M rows peaked at 2.53 GB with the filter before the bank, against
0.78 GB with the same filter after it, measured on 2026-09-02 as the
[peak footprint](../PERFORMANCE.md#how-the-numbers-are-made). The two
also mean different things: a filter after the bank changes only which
rows come out, and one before it changes which rows the model learns
from. To keep rows from being learned, give them weight 0 in place of a
filter, as [Hold a row
back](../how-to/hold-a-row-back.md#keep-a-model-from-learning-rows-in-a-query)
shows. They are
still predicted, and the clock still moves through them.

**A column computed in the
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads) just before a
slow bank also lets Polars read ahead.** Polars passes rows from one step
to the next in blocks. Between the step that reads a file and a step that
runs on every thread, such as `with_columns`, it holds up to about ten
blocks a thread. So a query that reads a parquet file and then runs
`with_columns` reads ahead of the bank. One bank over such a query held
848 MB at 4M rows, against 279 MB for the same bank over the file's query
alone, with the column computed on each chunk `collect_batches()` gave.
So add a column the bank does not need with `with_columns` after the
bank. Compute one it needs on each chunk `lf.collect_batches()` gives,
before you pass the chunk to `bank.fit_predict`.

## What streams, and what does not

**A query streams when every step in it works a chunk at a time.** A
query is a Polars `LazyFrame`, a chain of steps that runs only when you
ask for its result. Three ways of running a bank stream from end to end:

| run by | it reads | it writes |
|---|---|---|
| a query | files, with `pl.scan_parquet`, which reads them as a query | a file, with `sink_parquet`, which runs the query and writes the result without holding it all in memory |
| your own loop | each chunk `lf.collect_batches()` gives (one chunk of the result at a time), passed to `bank.fit_predict` | whatever the loop does with each chunk's output |
| the `polars-online` command line | a file | a file |

**A step that needs every row before it can give its first row of output
holds the stream in memory, wherever it sits in the query.** A `sort`
before the bank holds every row of its input while it sorts. `collect()`,
which runs the query and returns its whole result, holds the whole
output. Polars'
`rolling` over groups, with `.over("group")` or `group_by=`, holds every
row of its input. On 12M rows it peaks at 6.5 GB with `.over("group")`
and 1.7 GB with `group_by=`, against 0.25–0.28 GB without groups. A spec's `group=` keeps one set of running
sums per group instead.

## Why you set the row order

**You set the order before the bank, because the bank never sorts the
rows it is given.** Reading, joining and ordering the rows is the job of Polars, or of
another tool, in the query before the bank. The bank holds a row back only
until its target is known, under an embargo, a `label_clock` or a formula
target's window. It still learns the rows in their order, except under
`label_clock`, which learns each row at its own label time ([Learn targets
known late](../how-to/learn-targets-known-late.md#give-each-row-its-own-target-time)). The README's [What
this is not](../../README.md#what-this-is-not) says what the library
leaves to those tools.

**A model learns in row order, so the same rows in another order give
other predictions and, for most models, another fit.** Each row is
predicted from the rows before it, and most models also forget along the
clock or take a step at each row.
A query run a chunk at a time, as the bank runs it, can also deliver rows
in another order than the same query run whole. A `join` or a `group_by`
keeps its input's order only when told to. A `unique` keeps no order when
Polars runs the query a chunk at a time, so a `sort` must follow it. [Put
rows in clock order](../how-to/put-rows-in-clock-order.md) gives each step
the setting that keeps its order, and shows `OrderNotGuaranteedWarning`
and the step-back error.

**A bank checks the order it is given in three places, and changes none
of it.** It warns about the query, and refuses a chunk that runs
backwards before learning any of it:

| check | what it reads | when it finds disorder |
|---|---|---|
| the query | the query handed to the bank: a `join`, `group_by` or `unique` whose order Polars does not guarantee, unless a `sort` follows it, and a sort by several keys without `maintain_order=True` | warns with `OrderNotGuaranteedWarning`, naming the step and its fix. The run goes on |
| the clock | each group's clock within each [session](../GLOSSARY.md#time), row by row, for every spec with a `clock` | refuses the chunk on a [step back](../GLOSSARY.md#time) that `restart_after_step_back` does not make a [reset](../GLOSSARY.md#time) |
| the group keys | with `group_close="monotone"`, the keys in the column's own order, an integer column as numbers, and a String or Categorical column as text | refuses the chunk on a key below the one before it, since a [closed group](../GLOSSARY.md#saving-and-serving) cannot reopen |

**The clock is checked group by group, within each session,** so groups
may take turns row by row: only the rows of one group, within one
session, need to be in clock order. Two rows with the same clock value are
zero apart, which is not a step back. A row with a null feature is
skipped, but its clock is still checked. A row that `predict` serves
(predicts without learning) is never refused, since it changes no state.

**The query check is best-effort, and never fails a run.** It reads
Polars' printed description of the query (`explain()`), and walks the
query's steps only when that names a join, an aggregation, a `unique` or
a `sort`. A step followed by a `sort` counts as ordered, unless that sort
is by several keys without `maintain_order=True`. Such a sort leaves rows
with equal keys in no particular order. Sorted by two keys on Polars
1.44.2 and again on 2.0.0, 2,109 of 10,000 rows came out in another order
than a stable sort's. Sorted by one key, none did. For a join with
`maintain_order="left"`, the check reads only its left input. A query
Polars cannot write out in its serialized form passes without a warning.
The check reads the query that `fit`, `fit_predict_batches` and a query's
`online` calls are given, so the frames your own loop passes are not
checked. For a query whose order you know, silence the warning with
`warnings.simplefilter("ignore", po.OrderNotGuaranteedWarning)`.

**`ModelBank.fit` does not warn when its state cannot depend on the row
order.** That is when every spec is an `ewridge` or `rls` with decay off
and none of the settings that read the order. Those settings include a
hard window, a session, a weight, an embargo or `label_clock`, a drift
reset, a `coef` cadence, a formula target, and any
[diagnostic](../GLOSSARY.md#fitting-a-model) but `emit_drift`,
`emit_clocks` and `emit_error_inflation`. `fit`'s docstring lists the
rest. The running sums of such a spec reach the same state in any order,
to rounding ([Time and decay](time-and-decay.md#without-a-decay)), and
`fit` keeps only the state.
