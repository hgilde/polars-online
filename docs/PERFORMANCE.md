<!-- kind: reference -->
# Measured costs

This page is the reference for anyone who found a model bank's run slower
or larger than they expected. It gives what each model and each way of
running a bank was measured to cost in time and memory, and how each figure
was made. Its last section, how cost grows with size, is for a contributor:
it gives the growth each surface declares and the tests that hold it.

[Where the time and memory go](explanation/where-the-time-and-memory-go.md)
explains what sets each number, and what to change when one is not what you
expected.

| section | read it when | subsections |
|---|---|---|
| [How the numbers are made](#how-the-numbers-are-made) | you want how every figure was measured, and the commands that measure it again | |
| [The current figures](#the-current-figures) | you want a figure the guides quote: each model's rows a second, the bank's work around a row, a query against the eager bank, the speed at each thread count, a run's peak memory, or what a window operator costs | [each model](#rows-a-second-for-each-model) · [the bank's work](#the-banks-work-around-a-row) · [the query](#a-query-against-the-eager-bank) · [`predict`](#serving-against-learning) · [threads](#rows-a-second-at-each-thread-count) · [peak memory](#peak-memory-of-each-way-of-running-a-bank) · [the command line's memory](#peak-memory-of-the-command-line) · [Polars' rolling](#polars-rolling-windows-against-a-specs-groups) · [window operators](#what-a-window-operator-costs) |
| [How cost grows with size](#how-cost-grows-with-size) | you want how a row's work and a state grow with each size a surface has -- features, targets, groups, specs, a window, the rows already seen -- and the test that holds each | [how a cell is measured](#how-a-cell-is-measured) · [the table](#the-table) · [specs on one stream](#specs-on-one-stream) · [what the table holds](#what-the-table-holds) |

## How the numbers are made

**Each figure here states the date it was measured on and how it was made,
so read it as of its date.** Every release measures [Rows a second for each
model](#rows-a-second-for-each-model) again. The other figures were
measured once, on the date each gives. One has no command to measure it
again: [The bank's work around a row](#the-banks-work-around-a-row) was
timed by a Rust program the repository does not keep.

**A figure was measured on one machine unless it says otherwise:** an
Apple M4 Pro with 10 performance and 4 efficiency cores, a release build
(thin LTO, `codegen-units = 1`) and a single process. Other jobs often
shared the machine, so each figure logs the load average beside its run.
The figures that depart from this say so where they stand: the scaling
ratios come from a debug build, the instruction counts from CI's Linux
machines, and the threads of several specs from a build without LTO.

**Memory is the most memory the process held at once, not the resident
set size (RSS), unless a figure says otherwise.** It is the number
`/usr/bin/time -l` prints as "peak memory footprint", which
`scripts/memory_bench.py` reads, so that script runs on macOS only. RSS is the wrong ruler
for a bank's memory. Polars has memory-mapped a local parquet file, in
py-polars 1.x and in the command line's build of 2026-09-02, and the
file's clean pages count in RSS, though the system drops them under
pressure at no cost. On 2026-09-02 the command line's RSS grew from 1.37
to 2.26 GB from 3M to 6M rows, while the most memory it held stayed at
0.75 GB. On 2026-11-09 a scan of a 481 MiB file, read as fast as it came
on four threads, peaked at 540 MiB of RSS on py-polars 1.37.0 and 523 on
1.44.2. The most memory it held was 59 and 40 MiB. py-polars 2.0.0 does
not map the file, and the same scan peaked at 59 MiB of RSS. The window
operators' table reads RSS, so compare its rows with one another, not
with the figures read the other way.

The raw numbers regenerate with these commands, each run from the
repository root:

```sh
# The pure core, no Polars anywhere: EwRidge::step in a loop
cargo run --release -p online-core --example core_bench

# Rls::step alone, and EwRidge::step with the blocked Gram update
cargo run --release -p online-core --example rls_bench
cargo run --release -p online-core --example gram_block_bench

# The bank through Python, with the time of each part of a chunk on stderr
POLARS_ONLINE_TIMING=1 uv run python scripts/benchmark.py
```

Other commands make the rest:

| command | what it measures |
|---|---|
| `uv run python scripts/scaling_bench.py` | rows a second at each thread count |
| `uv run python scripts/memory_bench.py` | a run's peak memory |
| `uv run python scripts/windows_bench.py` | the window operators |
| `cargo run --release -p online-core --example marg_bench` | `marginal` |
| `uv run python scripts/sklearn_comparison.py` | this library against scikit-learn's `SGDRegressor` |
| `cargo run --release -p online-core --example sgd_bench`, and the same with `sgd_signature` | `sgd`'s time for each feature |

**For the time inside a run, set `POLARS_ONLINE_TIMING=1`, which prints two kinds
of line to stderr:**

| printed by | once per | fields |
|---|---|---|
| the bank | chunk | in wall milliseconds, `extract` (reading the columns), `group` (finding each group's rows), `process` (running the models) and `assemble` (building the output); and `total`, the whole call, of which the four are consecutive parts (`crates/online-polars/src/bank.rs`) |
| the command line | run | `read_wait` and `write_wait`, the bank thread's waits on the reader and the writer; `bank`, its time in the bank; `writer_busy`, the writer's own time; and `total`, the whole run (`crates/online-polars/src/runner.rs`) |

**The lines are written for a person, so their words and fields are not
promised: a minor release may change them.** The variable turns them on at
`1` and at no other value, as `POLARS_ONLINE_WARN_UNSTABLE` does. It is
read once, at a process's first chunk, so set it before the process starts:
set or cleared later, it changes nothing until the next process. A read of
the environment costs in proportion to its size, so reading it once keeps a
call's work independent of the size of the environment.

**A profile says where inside `process` the time goes, and the wall clock
says how much.** A sampling profiler counts threads: `assemble` runs on
every thread and `process` on one for a single group, so on 14 threads a
20 ms `assemble` reads as 280. The amount is `POLARS_ONLINE_TIMING=1`'s,
and a profile shows where inside `process` it goes, by function. Its
attribution to a line inside an inlined recursion is not reliable. On
macOS these commands build the extension with symbols, sample a Python
process that runs the workload in a loop by its pid, and list the binary's
functions and instructions. In `out.txt`, read "Sort by top of stack":

```sh
CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_DEBUG=1 uv run maturin develop --release -m crates/online-py/Cargo.toml
sample <pid> 8 -mayDie -file out.txt
nm python/polars_online/_polars_online.abi3.so
objdump -d python/polars_online/_polars_online.abi3.so
```

[CONTRIBUTING.md](../CONTRIBUTING.md) says how to measure a change to the
library against these numbers.

## The current figures

**A figure is what one run cost on one machine on its date, under the
conditions it states.**
[Where the time and memory
go](explanation/where-the-time-and-memory-go.md) explains what sets each
number.

### Rows a second for each model

Here are rows a second for every model at its usual settings, all under
these conditions:

| | |
|---|---|
| machine | an Apple M4 Pro, one process |
| runs | three runs of `uv run python scripts/benchmark.py --markdown`, each the best of 3 over 200k rows, and the best of the three for each row; measured on 2026-11-10 at commit `ebfde144`, 1.5.0's build, beside a load average of 3.0 to 4.6 |
| noise | the three runs of a row lay within 3.7% of one another (`kalman`'s within 5.0%), and on busier days a run has moved by up to 11% from the last on this machine, so read a gap smaller than that as noise |
| half-life | 1,000, in every row that gives none of its own |
| `k`, `K` | the number of features; the number of clusters or hidden states |

The [linear models](MODELS.md#linear-models):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ewridge` | k=5, 1 target, 1 half-life | 5,703,144 | |
| `ewridge` | k=20, 1 target, 1 half-life | 3,391,220 | |
| `ewridge` | k=50, 1 target, 1 half-life | 1,220,635 | |
| `ewridge` | k=20, 10 targets | 1,563,506 | one set of feature sums for every target |
| `ewridge` | k=20, 5 half-lives, 500 to 2,500 | 1,880,475 | one set of sums per half-life, run in parallel |
| `ewridge` + `conformal` | k=20, 90% interval | 3,312,947 | nothing measurable: the interval reads the residual the model already has |
| `rls` | k=20, 1 target | 1,740,172 | its square-root form, at 51% of `ewridge`'s speed: the form is what keeps one extreme row from destroying it by cancellation |
| `kalman` | k=20, 1 target | 1,832,892 | |
| `kalman` | k=20, `revert_half_life` | 1,573,464 | |
| `lasso` | k=20, 1 target (3-point path) | 1,932,145 | |
| `huber` | k=20, 1 target | 3,585,209 | |
| `quantile` | k=20, the median | 1,733,947 | |
| `sgd` | k=20, squared loss, `standardize=False` | 7,574,120 | the raw step; the default, `standardize=True`, scales each feature on the row first |
| `sgd` | k=20, `coef_min=0`, `coef_sum=1`, `standardize=False` | 2,377,249 | a simplex constraint sorts `2k` breakpoints per row, which makes `sgd` 3.2 times slower |
| `pa` | k=20, `standardize=False` | 8,335,504 | the raw step, as for `sgd` |
| `ftrl` | k=20, 1 target | 5,103,136 | |
| `holt` | level and trend of one column | 12,181,752 | |

**Several targets on the same features in one spec's `targets` share one
set of feature sums,** so 10 targets take 2.17 times as long as one.

**A grid of half-lives keeps one set of sums per half-life, and runs them
in parallel.** So the grid of five takes 1.42 times as long as its shortest
half-life, 500, which runs alone at 2,671,331 rows a second. That is slower
than at the half-life of 1,000 that every other row here uses, because a
shorter half-life solves more often.

The models of [moments and correlation](MODELS.md#moments-and-correlation):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ew_cov` | k=20: mean, std, corr (230 statistics) | 2,169,178 | |
| `ew_cov` | k=20: mean, mahal, `mahal_q0.99` | 835,422 | the Mahalanobis distance factors a `k × k` matrix by Cholesky once per row |
| `ew_cov` | k=20: mean, cov, lags 1–5 | 1,645,781 | |
| `marginal` | 20 features against one target | 10,372,525 | `O(k)` pairs a row, and each row writes only `weight_sum`: the pairs are read from the state after the run |
| `deco` | k=20, one equicorrelation | 5,034,639 | one number for the whole matrix of its `m` series, computed in `O(m)` a row with no `m × m` matrix to factor |
| `deco` | k=20 in 4 blocks | 2,828,546 | |
| `rcov` | 4 features, kernel, blocks of 1000 | 4,489,560 | it accumulates per row and computes its kernel only when the block closes |
| `audit` | 20 columns, 256 counters each | 964,870 | each row updates every column's counts and measures, and writes only `weight_sum` |

The models of [clustering and classification](MODELS.md#clustering-and-classification):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `ew_class` | k=20, 3 classes, full covariance | 576,752 | a `k × k` matrix factored by Cholesky once per *learned* row |
| `ew_class` | k=20, 3 classes, tied covariance | 635,448 | |
| `ew_class` | k=20, 3 classes, diagonal | 3,105,881 | |
| `kmeans` | 4 features, K=8 | 5,276,494 | a distance to each center |
| `kmeans` | k=20, K=8 | 3,208,597 | a distance to each center |
| `micro` | 4 features, `eps=1` | 7,778,936 | a distance to each center |

The [sequential tests and regimes](MODELS.md#sequential-tests-and-regimes):

| model | settings | rows/sec | what sets its speed |
|---|---|---:|---|
| `seqtest` | sign of one column | 14,610,360 | a handful of operations |
| `corrchange` | 4 features, monitor, `window_rows=500` | 6,042,228 | |
| `corrchange` | 4 features, window 100, permute every 500 | 1,313,716 | its permutation null |
| `bocpd` | 4 features, `covariance_type="diag"`, `degrees_of_freedom_prior` 3 (its default there), `max_run=200` | 712,704 | the number of runs it keeps |
| `bocpd` | 4 features, `covariance_type="full"`, `degrees_of_freedom_prior=6.0` (its default there, `d + 2`), `max_run=200` | 1,144,714 | the number of runs it keeps |
| `hmm` | 4 features, K=2 | 2,546,042 | a `k × k` covariance factorized per state per row, which is `ew_class`'s work with the classes hidden |
| `hmm` | k=20, K=2 | 430,745 | |

**`corrchange`'s window kind runs at about a fifth of its monitor kind's
speed, because its permutation null redraws `n_perm` statistics every
`permute_every_rows` rows.** A number given as `crit` fixes the threshold,
and then no redraw runs.

**`bocpd` slows with the number of run lengths it keeps, so a larger
`prune_below` makes it faster.** It does `O(runs · d)` work a row for `d`
features under `covariance_type="diag"`, and `O(runs · d³)` under `"full"`,
which factors each run's `d × d` scale every row. `prune_below` drops
every run holding less than that share of the probability mass, and
`max_run`, 10,000 by default, caps how many runs it keeps. With neither bound the runs grow by one every row, and a stream
takes `O(rows²)`. A changepoint collapses them to a few dozen, so `bocpd`
runs faster on data that breaks. A stream that does not break spreads them
over thousands of run lengths, and there `max_run` is the bound. On 20,000
i.i.d. Gaussian rows with no `max_run`, measured on 2026-09-06 in one
process, the best of two or three runs:

| `prune_below` | rows/sec |
|---|---:|
| `1e-8` | 204,000 |
| `1e-6`, the default | 324,000 |
| `1e-4` | 687,000 |

### The bank's work around a row

**Around each row the bank clocks it, keeps its schedule, copies its
features in and writes its fields out, and below about 50 features that
work is as much as the model's own.** A Rust program written for the
measurement, which the repository does not keep, timed each model's step
alone in a loop against the bank's `fit_predict` of the same rows as one
frame. It ran one spec over one group, on one thread, after a warm-up, on the
system allocator, the best of eight passes. It was measured on 2026-11-03
at a load average of 5.0, on the build of [Rows a second for each
model](#rows-a-second-for-each-model). In ns a row:

| model | features | the model's step | through the bank | bank over step |
|---|---:|---:|---:|---:|
| `seqtest` | 0 | 13 | 73 | 5.6× |
| `sgd` | 5 | 46 | 139 | 3.0× |
| `ewridge` | 5 | 107 | 217 | 2.0× |
| `ew_cov`, 23 fields | 5 | 46 | 148 | 3.2× |
| `sgd` | 50 | 158 | 300 | 1.9× |
| `ewridge` | 50 | 795 | 955 | 1.2× |
| `ew_cov`, 1,328 fields | 50 | 1,431 | 4,228 | 3.0× |
| `sgd` | 500 | 1,345 | 2,360 | 1.8× |
| `ewridge` | 500 | 94,226 | 111,804 | 1.2× |
| `ew_cov`, 125,753 fields | 500 | 120,315 | 604,538 | 5.0× |

### A query against the eager bank

**A query runs slower than the eager bank on the same rows at many groups,
because it reads its input as the frames Polars hands it.** On 1M rows of a
five-feature `ewridge` on all threads,
`pl.scan_parquet(...).online.fit_predict([spec]).collect()` against
`ModelBank([spec]).fit_predict(df)`, the best of five runs, two rounds, on
1.3.0's build on 2026-10-24 at a load average of 1.9 to 3.4, in ns a row:

| groups | the query | the eager bank |
|---|---:|---:|
| 10,000 | 236 | 140 |
| 1,000 | 85 | 66 |
| one group | 205 | 196 |

### Serving against learning

**`predict` skips learning, so it runs faster than `fit_predict`, and more
so for a wider model.** On one frame of a million rows of one group,
`ModelBank([spec]).predict(df)` against `ModelBank([spec]).fit_predict(df)`
for an `ewridge` spec, `predict` ran 1.85 times as fast at k=5 and 2.82
times at k=20. It was measured on 2026-09-29 on a build of the day
before 0.13.0's release, whose run recorded neither its repetitions nor its load.

### Rows a second at each thread count

**A bank runs its pieces of work on several threads, and its speed grows
with the threads while it has pieces enough to share among them.** Rows a
second for k=20 over 64 groups, at each thread count, measured on
2026-09-29 on 0.13.0's build at a load average of 1.9 to 2.1.
`scripts/scaling_bench.py --markdown` ran each thread count in a process of
its own, and each figure is the best of three runs:

| threads | 1 | 2 | 4 | 8 | 14 |
|---|---:|---:|---:|---:|---:|
| rows/s | 0.98M | 1.89M | 3.44M | 5.98M | 7.08M |

[Use more cores](how-to/use-more-cores.md) shows how to give a bank more
work it can run at once, set its thread count, and search over factor sets
in one pass.

### Peak memory of each way of running a bank

**As the rows grow, a run's peak memory grows only by the pages the
allocator keeps, while the bank's own memory stays flat.**
Measured on 2026-10-24 as the most memory the process ever held, with
Polars 2.0.0 on 14 cores. The file has 20 features over
32 groups, parquet in and parquet out, and `ewridge` reads it with a clock,
a weight and a group per row. The 12M rows are the 3M rows four times
over, so the state is the same at both sizes. `scripts/memory_bench.py`
makes the file and runs each case:

| what you write | 3M rows | 12M rows | what it is |
|---|---:|---:|---|
| `lf.online.fit_predict([spec])` | 1.2 GB | 1.8 GB | the bank inside a query |
| `for chunk in lf.collect_batches(): bank.fit_predict(chunk)` | 1.1 GB | 1.5 GB | your own loop |

The bank does not grow from 3M to 12M rows: the growth is the memory
allocator keeping pages it has freed. Told to return them at once, with
`_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0`, the query holds
0.86 GB at 3M rows and 0.95 GB at 12M. Of the rest, nearly all is Polars
reading ahead in the parquet file. The command line holds less: see
[Peak memory of the command line](#peak-memory-of-the-command-line).

### Peak memory of the command line

**The command line's peak memory stays bounded as the rows grow, 0.75 GB
at 3M rows and 0.86 GB at 24M, because its allocator reuses the pages it
frees across all its threads.** Measured on
2026-10-24 as the most memory the process held, on one file, `ewridge`
with 20 features, parquet in and parquet out. It is the run of [Peak
memory of each way of running a
bank](#peak-memory-of-each-way-of-running-a-bank), on macOS with Polars
2.0.0 on 14 cores. The 24M rows are the 3M rows eight times over, so the
state is the same at every size. `scripts/memory_bench.py` makes these
measurements:

| what you run | 3M rows | 12M rows | 24M rows |
|---|---:|---:|---:|
| `polars-online --config` | 0.75 GB | 0.85 GB | 0.86 GB |
| `polars-online --config`, with `POLARS_ROW_GROUP_PREFETCH_SIZE=1` | 0.42 GB | 0.43 GB | 0.44 GB |

The bank's memory does not grow with the rows, and the allocator reuses
the pages it frees rather than hold more as the run goes on. Most of what
is left is Polars reading ahead in the parquet file, and
`POLARS_ROW_GROUP_PREFETCH_SIZE=1` halves it.

**The bound comes from how the binary sets up its allocator.** Off
Windows it is jemalloc, set to one arena for every thread (`narenas:1`).
jemalloc's default is four arenas a core. Each arena keeps the pages freed
in it for ten seconds, for the threads on that arena alone. It gives them
back only when one of those threads allocates again. The reader's, the
bank's and the writer's buffers then strand pages in every arena, and the
peak grows with the rows: 0.94, 1.36 and 1.48 GB at 3M, 12M and 24M rows.
One arena adds no time in these runs. Threads contending for it would show
as CPU time, and the 12M-row run took 24.5 s of CPU either way.

**`_RJEM_MALLOC_CONF` overrides the binary's setting.** With
`_RJEM_MALLOC_CONF=dirty_decay_ms:0` freed pages go back to the system at
once: 0.74, 0.76 and 0.78 GB, for about 18% more CPU time. With
`POLARS_ROW_GROUP_PREFETCH_SIZE=1`, the system allocator holds less still,
0.21-0.23 GB, but runs slower. On a million rows of a five-feature `ewridge`, best of
five runs, it took 1.6 times as long with a thousand one-row groups a
chunk and 1.2 times as long with one group. On Windows the binary's
allocator is mimalloc, and these figures were not measured there.

### Polars' rolling windows against a spec's groups

**Polars' `rolling` over groups holds every row of its input, where a
spec's `group=` keeps one set of running sums per group.** Peak memory of
a rolling window, with `.over("group")` or `group_by=`, on 12M rows,
measured on 2026-09-02 as the most memory the process held:

| a rolling window | peak memory |
|---|---:|
| with `.over("group")` | 6.5 GB |
| with `group_by=` | 1.7 GB |
| without groups | 0.25–0.28 GB |

### What a window operator costs

**A window operator, such as `po.ewm_mean`, does the same work on a row
however long its window, and holds about one window of rows.** So a window
adds little memory: through `with_windows`, within 0.04 GB of reading and
writing the file alone.
Measured on 16M rows
arriving about two a second on the clock, parquet in and parquet out, one
group, on 2026-10-03 at a load average of 5.0 to 6.0. `scripts/windows_bench.py`
runs each case in a process of its own, and reads its peak memory as the
peak RSS from `getrusage`, which counts the file's mapped pages too:

| what runs | time | peak memory |
|---|---:|---:|
| reading and writing the file, and nothing else | 0.40 s | 1.03 GB |
| a forward VWAP through `with_windows` | 2.43 s | 0.99 GB |
| the same VWAP through Polars' `rolling` recipe | 19.52 s | 5.24 GB |
| a model with a formula target, which the bank resolves itself: the native form | 9.37 s | 1.09 GB |
| the window written by `with_windows(..., like=spec)` and fed back as a plain target: the column form | 5.87 s | 1.23 GB |

**An operator whose input is null on most rows costs about what a dense
one does.** Every row joins its operator's queue of rows, valued or not. A
mean's read looks for the operator's oldest valued row from where its last
search stopped. So each read passes only the null rows that arrived since
the read before it, and each null row is passed once. On 300k rows with a window of 2,000 rows, one value in 5,000 takes
184 ns a row backward and 195 ahead, the best of two runs at a load
average of 3.3, measured on 2026-10-24.

**Operators with the same direction, back or ahead, and the same
`half_life`, `window_size` and `closed` share one queue of rows.** Sixteen
operators on one queue ran in 57% of the time of sixteen with queues of
their own, and four in 71%, measured on 2026-10-03 by
`scripts/windows_bench.py` on the same 16M rows.

**A look-ahead target written as a column runs faster, and given as an
expression in `targets` holds less memory,** as the last two rows of the
table show. In the column form, `with_windows(..., like=spec)` in the query
before the bank writes the column the spec names in its `targets`, under
the same `embargo`, and Polars runs the window and the bank at once, as
two stages of one query. In the native form the bank resolves the windows
itself, one after the other in the same pass as the model. With many
groups, that adds a Polars query per group per chunk on the calling
thread. [Make a target that looks
ahead](how-to/make-a-target-that-looks-ahead.md#check-the-targets-the-model-learned) runs both
forms.

## How cost grows with size

**Every surface states how a row's work and its kept state grow with each
size it has, and a test holds each statement by measuring two sizes.** A
budget fixes what one workload costs at one size, so it catches a change
that made the workload slower than before. It cannot catch a cost whose
growth is wrong at every size. A walk over every group on every row,
`O(groups)` a row, costs nothing measurable over a handful of groups, and
one large budget case absorbs it on the day its number is set. Holding the *ratio* of the costs at two sizes to the growth the algorithm
declares needs no number from an earlier run. The exponent comes from the
algorithm, so the check catches a defect that was there from the start.

### How a cell is measured

A *cell* is one surface and one size it has: `ewridge`'s features, the
bank's specs, the groups a window has seen. Each cell runs one workload at
two sizes, `n` and `step·n` (the step is 8 unless the table says otherwise),
over the same number of rows after a warm-up that brings the state to where
a long stream keeps it. It is held to

```text
cost(step·n) / cost(n)  <=  g(step·n) / g(n)
```

where `g` is the declared growth: 1 for a cost the size must not move,
`log n`, or `n` to a power. A cost with a fixed part below the growing one
reads under its bound, so only a growth past the declared one fails.

Four measurements are taken for each cell, at both sizes:

| measurement | where | what it reads | allowed past the declared growth, and why |
|---|---|---|---|
| a row's time | `crates/online-bench/tests/scaling.rs`, every run of `cargo test` (the essentials) | the process's CPU time over a block of rows, the least of three, the small and large sizes taken in turn; the wall clock's where CPU time is not read (Windows). CPU time, because a thread waiting for a core adds none: on a machine running more threads than it has cores, a block longer than a scheduler's slice is stretched by about the load over the cores and a shorter one may not be, so the wall clock's ratio grows with the load, 25-33x for a cell of declared O(1) growth at a load of 50 on 14 cores | a factor of `step^0.5`, 2.8 at a step of 8: time moves with the load and the caches, and a wrong growth is a whole power of the step, 8x |
| a row's allocations | the same | the allocator's count of calls over the blocks, on every thread but the test harness's (libtest's main thread, and each test's thread outside its blocks), whose calls the other tests of the binary make while one measures | 25%: what a vector's doubling or a hash table's growth adds at one size and not the other; or 4 allocations a block, whatever the ratio, for what reaches the count beside the work: a growth the check is for adds one a row, at least 200 a block |
| the saved state | the same | the bytes of the state as a state file holds it (msgpack) | 25%, as above |
| the state's heap | the same | the bytes freed when the workload is dropped, its rows dropped first; for a run of the command line, the most it held above the live heap during the run | 25%, as above |
| a row's instructions | `.github/workflows/budgets.yml`, Callgrind, on every push (the `scale_*` cases) | the instructions of the block at each size, in one run | 10% (`[scaling]` in `budgets.toml`): a count is exact, and a fixed part only lowers the ratio, so what 10% absorbs is a held structure a little deeper at the large size |

The instruction count is the fine measurement and time the coarse one. A
count does not see a cache miss, which makes a size that no longer fits in
cache read slower without its work growing, so the timed check allows a
factor of 2.8 that the counted one does not need. Callgrind runs on Linux
alone, so it counts only the cells where a count sees what a timing cannot
and that stay cheap under Valgrind, the list `BENCH_CELLS` in
`crates/online-bench/src/scaling.rs` holds. They are three models' rows
already seen, the widths of `ewridge`, `sgd`, `ew_cov` and `marginal`, `ew_cov.window`, the spec counts, the
bank's, the window operators' and `refresh_time`'s cells whose growth must
be 1, `windows.kernels`, and `bank.label_clock`. `scripts/budgets.py
check` holds each to `[scaling.cells]` within the run, a measuring run
included, since a wrong growth is a defect and not a number to re-measure.

The rows a workload is fed are copies of their own, as a stream's chunks
arrive. A slice of a frame shares its parent's buffer. So a state that
keeps a row of a slice, as the window core keeps the last row it read,
would keep every row of the parent alive. The heap would then read as
growing with the stream when it is the test's frame that grows.

The Python surfaces add their own per-call work on top of the bank's, so
`tests/test_scaling.py`, in the extended tier, times them in the same way.
It times the chunked `ModelBank.fit_predict` and the Polars source behind
`LazyFrame.online.fit_predict` by their specs, at large calls and at small
ones, and the Polars source by its rows.

A cell whose measurement exceeds its declaration is a *finding*. It is held
to its declaration in the extended tier, where it fails until it is fixed.
A declaration is corrected
only where the algorithm's own growth is higher by design, with the reason
in the table.

### The table

**113 cells: 76 over the 22 model kinds' rows, 19 over the
bank, 3 over the command line, 9 over the window operators, 4 over
`refresh_time` and 2 over `po.gbdt`'s sample.** Each row is one cell: the
size it varies, its two sizes, the growth declared for a row's work and for
the state in that size (written in `n`, the size), and the reason. The last
column has the four native ratios from large to small, as measured on
2026-11-05 by a debug build: `SCALING_REPORT=1 cargo test -p online-bench
--test scaling -- --nocapture` prints them. `bank.label_clock`'s were measured on
2026-11-08, the same way, at a load of 5. The two `.window` cells were
measured again on 2026-11-06, once a window's snapshots were laid out
flat. A row's time and allocations are per
row; the saved state and the heap are the state's. `(CI)` marks the 24 cells
Callgrind also counts. A cell whose ratio reads below 1 got cheaper at the
larger size: a fixed cost shared by more rows, or a block that held its
largest buffer at the smaller.

| cell | the size | n → step·n | a row's work | state | why | time · allocations · saved · heap, ratio |
|---|---|---|---|---|---|---|
| `ewridge.k` (CI) | features k, no solve due | 4 → 32 | O(n^2) | O(n^2) | a row is the Gram's rank-one update, k² entries; the state is the Gram | 4.97 · 1.00 · 13.43 · 4.70 |
| `ewridge.solve_k` | features k, a solve every row | 4 → 32 | O(n^3) | O(n^2) | a solve factorizes the k×k Gram, k³/3 | 36.01 · 1.00 · 13.41 · 5.65 |
| `ewridge.targets` | targets T | 1 → 8 | O(n) | O(n) | each target keeps its cross-moments with the features (k a target) and its own solve | 1.67 · 1.37 · 3.01 · 1.55 |
| `ewridge.ridge_grid` | ridge values G, a solve every row | 1 → 8 | O(n) | O(n) | one solve and one coefficient vector per value, from one set of sums | 7.97 · 6.72 · 1.72 · 2.24 |
| `ewridge.feature_sets` | feature sets S, a solve every row | 1 → 8 | O(n) | O(n) | one solve and one coefficient vector per set, from one set of sums | 7.70 · 6.77 · 1.73 · 2.32 |
| `ewridge.window` | window W clock units | 500 → 4,000 | O(1) | O(n) | a row adds a snapshot and subtracts the one leaving; the window keeps a snapshot a row | 0.71 · 0.83 · 8.23 · 7.95 |
| `ewridge.gram_block` | Gram block B rows | 8 → 64 | O(1) | O(n) | the block's rows are merged once a block, the same work a row; the block holds B rows | 0.51 · 0.18 · 2.53 · 1.60 |
| `ewridge.history` (CI) | rows already seen | 1,000 → 8,000 | O(1) | O(1) | running sums: nothing grows with the stream | 0.99 · 0.85 · 1.00 · 1.00 |
| `rls.k` | features k | 4 → 32 | O(n^2) | O(n^2) | the inverse's rank-one update, k² | 9.33 · 1.00 · 26.27 · 6.39 |
| `rls.targets` | targets T | 1 → 8 | O(n) | O(n) | one inverse shared, a coefficient vector a target | 3.68 · 1.00 · 2.24 · 1.66 |
| `rls.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | the inverse and the coefficients only | 1.00 · 1.00 · 1.00 · 1.00 |
| `lasso.k` | features k, no solve due | 4 → 32 | O(n^2) | O(n^2) | the Gram's update, k² | 4.41 · 1.00 · 12.72 · 4.06 |
| `lasso.path` | penalties P | 2 → 16 | O(n) | O(n) | each penalty's fit is read and predicts every row, k a penalty | 3.32 · 2.16 · 1.78 · 1.38 |
| `lasso.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | running sums | 0.97 · 0.85 · 1.00 · 1.00 |
| `kalman.k` | features k | 4 → 32 | O(n^2) | O(n^2) | the covariance P's update, k² | 10.04 · 1.00 · 17.62 · 5.12 |
| `kalman.targets` | targets T | 1 → 8 | O(n) | O(n) | a filter a target | 5.72 · 1.00 · 5.15 · 2.63 |
| `kalman.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | the filter's state only | 1.00 · 1.00 · 1.00 · 1.00 |
| `huber.k` | features k, no solve due | 4 → 32 | O(n^2) | O(n^2) | the weighted Gram's update, k² | 4.95 · 1.12 · 2.52 · 4.53 |
| `huber.targets` | targets T | 1 → 8 | O(n) | O(n) | a weighted Gram and scale a target | 7.57 · 1.33 · 6.53 · 3.21 |
| `huber.feature_sets` | feature sets S, a solve every row | 1 → 8 | O(n) | O(n) | a solve a set | 7.74 · 1.00 · 1.13 · 1.31 |
| `huber.refit_rows` | rows held for the first refit, R | 1,000 → 8,000 | O(log n) | O(n) | a held residual enters an ordered tree, log R; the buffer holds R rows | 0.83 · 1.00 · 8.69 · 7.66 |
| `huber.history` (CI) | rows already seen | 1,000 → 8,000 | O(1) | O(1) | after its refit the model keeps its sums and the scale's sketch | 0.97 · 0.88 · 1.00 · 1.00 |
| `quantile.k` | features k, no solve due | 4 → 32 | O(n^2) | O(n^2) | the weighted Gram's update, k² | 24.22 · 1.07 · 2.66 · 4.34 |
| `quantile.refit_rows` | rows held for the first refit, R | 1,000 → 8,000 | O(log n) | O(n) | as huber's | 0.91 · 0.78 · 8.67 · 7.65 |
| `quantile.history` (CI) | rows already seen | 1,000 → 8,000 | O(1) | O(1) | as huber's | 1.00 · 0.91 · 1.06 · 1.02 |
| `sgd.k` (CI) | features k | 4 → 32 | O(n) | O(n) | one gradient step, k | 2.68 · 1.00 · 3.36 · 1.89 |
| `sgd.targets` | targets T | 1 → 8 | O(n) | O(n) | a step a target | 4.09 · 1.00 · 3.23 · 1.67 |
| `sgd.refit_rows` | rows held for the first refit, R | 1,000 → 8,000 | O(log n) | O(n) | as huber's | 1.17 · 1.00 · 8.72 · 7.82 |
| `sgd.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | coefficients and scales only | 1.00 · 1.00 · 1.00 · 1.00 |
| `pa.k` | features k | 4 → 32 | O(n) | O(n) | one projection, k | 3.33 · 1.00 · 4.01 · 1.94 |
| `pa.targets` | targets T | 1 → 8 | O(n) | O(n) | a projection a target | 3.22 · 1.00 · 2.92 · 1.64 |
| `pa.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | coefficients only | 1.00 · 1.00 · 1.00 · 1.00 |
| `ftrl.k` | features k | 4 → 32 | O(n) | O(n) | per-coordinate running sums, k | 4.74 · 1.00 · 3.99 · 1.69 |
| `ftrl.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | running sums only | 0.99 · 1.00 · 1.00 · 1.00 |
| `holt.targets` | targets T | 1 → 8 | O(n) | O(n) | a level and trend a target | 4.95 · 1.00 · 4.99 · 1.35 |
| `holt.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | level and trend only | 0.99 · 1.00 · 1.00 · 1.00 |
| `ew_cov.k` (CI) | columns k | 4 → 32 | O(n^2) | O(n^2) | the co-moments' rank-one update, k² | 14.98 · 1.00 · 27.25 · 6.03 |
| `ew_cov.lags` | lags L | 1 → 8 | O(n) | O(n) | a lagged co-moment block a lag | 2.51 · 1.00 · 3.33 · 1.49 |
| `ew_cov.window` (CI) | window W clock units | 500 → 4,000 | O(1) | O(n) | a snapshot added and one subtracted a row; a snapshot a row held | 1.00 · 1.00 · 8.23 · 7.86 |
| `ew_cov.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | moments only | 1.00 · 1.00 · 1.00 · 1.00 |
| `marginal.k` (CI) | features p | 4 → 32 | O(n) | O(n) | each feature's pair with the target, p | 2.63 · 1.00 · 4.59 · 1.40 |
| `marginal.targets` | targets T | 1 → 8 | O(n) | O(n) | p pairs a target | 6.14 · 1.00 · 6.67 · 1.92 |
| `marginal.lags` | lags L | 1 → 8 | O(n) | O(n) | each lag's pair moments, p·L | 3.49 · 1.00 · 3.56 · 1.74 |
| `marginal.bins` | bins B a feature | 4 → 32 | O(log n) | O(n) | one search of B−1 edges a feature, log B; B cells a pair | 1.15 · 1.00 · 6.26 · 2.49 |
| `marginal.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | moments only | 1.00 · 1.00 · 1.00 · 1.00 |
| `deco.k` | columns m | 4 → 32 | O(n) | O(n) | standardized sums, m a row | 1.24 · 1.00 · 4.75 · 1.42 |
| `deco.blocks` | blocks K | 1 → 8 | O(n^2) | O(n^2) | a level a block and a pair of blocks, K² | 11.26 · 1.00 · 10.72 · 2.64 |
| `deco.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | sums only | 1.01 · 1.00 · 1.00 · 1.00 |
| `rcov.k` | columns k | 2 → 16 | O(n^2) | O(n^2) | each lag of the kernel adds a k×k outer product (H·k²); the ring holds H rows of k | 11.38 · 1.00 · 42.15 · 25.63 |
| `rcov.block_rows` | block length n | 500 → 4,000 | O(n^(3/5)) | O(n^(3/5)) | the automatic bandwidth's ring depth is c·n^(3/5) lags, read a row | 3.34 · 1.00 · 3.22 · 3.13 |
| `hmm.states` | states K | 2 → 16 | O(n^2) | O(n^2) | the forward filter's K×K transition | 8.16 · 1.00 · 10.01 · 1.52 |
| `hmm.k` | columns d, diagonal | 2 → 16 | O(n) | O(n) | a diagonal state keeps d variances | 1.71 · 1.00 · 3.09 · 1.02 |
| `hmm.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | states and counts only | 1.00 · 0.50 · 1.00 · 1.00 |
| `kmeans.clusters` | clusters K | 2 → 16 | O(n) | O(n) | the nearest of K centers | 3.80 · 0.28 · 3.90 · 2.80 |
| `kmeans.k` | columns p | 4 → 32 | O(n) | O(n) | a distance is p | 4.46 · 1.00 · 6.51 · 3.43 |
| `kmeans.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | centers only | 1.00 · 1.00 · 1.00 · 1.00 |
| `micro.max_clusters` | micro-clusters m | 8 → 64 | O(n) | O(n) | the nearest of m summaries | 5.07 · 0.51 · 4.57 · 3.51 |
| `micro.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | at most m summaries | 1.00 · 1.00 · 1.00 · 1.00 |
| `ew_class.classes` | classes C | 2 → 16 | O(n) | O(n) | a learned row updates its class; a prediction reads each class's density | 5.20 · 1.00 · 7.70 · 4.36 |
| `ew_class.k` | features k, full covariance | 4 → 16 | O(n^3) | O(n^2) | the learned class is refactorized, k³; a class keeps k² | 8.43 · 1.00 · 9.00 · 3.60 |
| `ew_class.k_diag` | features k, diagonal | 4 → 32 | O(n) | O(n) | a diagonal class keeps k variances | 3.14 · 1.00 · 5.05 · 1.72 |
| `ew_class.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | class moments only | 1.00 · 1.00 · 1.00 · 1.00 |
| `seqtest.targets` | targets T | 1 → 8 | O(n) | O(n) | a test a target | 4.85 · 1.00 · 4.60 · 1.37 |
| `seqtest.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | running evidence only | 0.98 · 1.00 · 1.00 · 1.00 |
| `corrchange.k` | columns k, monitor | 2 → 16 | O(n^2) | O(n^2) | every pair's correlation, k² | 38.98 · 1.24 · 3.21 · 1.02 |
| `corrchange.window` | window W rows | 100 → 800 | O(1) | O(n) | each window merged from block summaries: the rows at its two ends and a few merges, about k³ a row; 2W rows held, and the summaries of at most 2W/b blocks | 0.94 · 0.98 · 7.78 · 7.27 |
| `corrchange.window_crit` | window W rows, a fixed threshold | 100 → 800 | O(1) | O(n) | as above, with no permutation drawn | 0.96 · 0.98 · 7.78 · 7.27 |
| `corrchange.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | the two windows only | 1.00 · 1.00 · 1.00 · 1.00 |
| `bocpd.max_run` | runs R | 25 → 200 | O(n) | O(n) | each run's running sums, R·d | 9.14 · 1.00 · 7.45 · 4.55 |
| `bocpd.k` | columns d, diagonal | 2 → 16 | O(n) | O(n) | R·d | 3.19 · 1.00 · 5.89 · 2.19 |
| `bocpd.k_full` | columns d, full | 2 → 8 | O(n^3) | O(n^2) | each run's scale factorized, R·d³; R·d² held | 6.40 · 1.00 · 7.21 · 2.59 |
| `bocpd.history` | rows already seen | 1,000 → 8,000 | O(1) | O(1) | at most R runs | 0.99 · 1.00 · 1.00 · 1.00 |
| `audit.k` | columns c | 4 → 32 | O(n) | O(n) | each column's counters | 7.57 · 7.68 · 7.89 · 7.27 |
| `audit.pairs` | columns c, pairs | 4 → 32 | O(n^2) | O(n^2) | every pair's moments, c² | 12.44 · 7.68 · 9.35 · 7.66 |
| `audit.distinct_cap` | counter cap D | 32 → 256 | O(1) | O(n) | a value is a hash lookup; a table full empties in O(D) once per D new values; D counters held | 1.01 · 1.00 · 1.27 · 2.64 |
| `audit.history` | rows already seen | 1,000 → 8,000 | O(1) | O(log n) | D counters at most; their counts' encodings lengthen as the counts grow, log in the rows | 0.86 · 0.93 · 1.26 · 1.24 |
| `bank.groups_seen` (CI) | groups seen G (a block over 50 of them) | 250 → 2,000 | O(1) | O(n) | a row touches its group's state alone; a state a group | 1.03 · 1.00 · 7.99 · 7.14 |
| `bank.groups_seen_per_call` (CI) | groups seen G, calls of four rows to one group each | 1,000 → 8,000 | O(1) | O(n) | a call finds each of its groups' state by key and takes only their notices; a state a group | 1.00 · 1.00 · 7.98 · 7.33 |
| `bank.specs` (CI) | specs S, one set of columns | 1 → 8 | O(n) | O(n) | each spec steps its model, S | 8.28 · 7.67 · 7.80 · 7.23 |
| `bank.specs_wide` | specs S from 8 to 64 | 8 → 64 | O(n) | O(n) | S | 7.88 · 7.95 · 7.98 · 7.93 |
| `bank.specs_distinct` (CI) | specs S, each its own columns | 1 → 8 | O(n) | O(n) | S, the columns looked up a spec | 8.29 · 7.77 · 7.81 · 7.23 |
| `bank.specs_distinct_wide` | specs S from 8 to 64, each its own columns | 8 → 64 | O(n) | O(n) | S | 7.98 · 7.95 · 7.98 · 7.94 |
| `bank.specs_mixed` (CI) | copies of a mix of five kinds | 1 → 8 | O(n) | O(n) | S | 7.87 · 7.75 · 7.97 · 7.92 |
| `bank.specs_per_call` (CI) | specs S, calls of four rows | 1 → 8 | O(n) | O(n) | what a call costs a spec (its columns, its output) is linear in S | 4.32 · 6.04 · 7.80 · 7.27 |
| `bank.specs_per_call_wide` | specs S from 8 to 64, calls of four rows | 8 → 64 | O(n) | O(n) | S | 7.24 · 7.64 · 7.98 · 7.62 |
| `bank.k` | features k (ewridge) | 4 → 32 | O(n^2) | O(n^2) | the model's k², and k columns extracted | 14.85 · 1.08 · 4.39 · 1.64 |
| `bank.targets` | targets T (ewridge) | 1 → 8 | O(n) | O(n) | T | 2.37 · 1.41 · 1.88 · 1.52 |
| `bank.chunk_rows` | rows a call | 64 → 512 | O(1) | O(1) | a call's fixed cost is shared by its rows; a row's own cost does not grow | 0.72 · 0.34 · 1.00 · 0.65 |
| `bank.history` (CI) | rows already seen | 1,000 → 8,000 | O(1) | O(1) | the state only | 0.96 · 0.85 · 1.00 · 1.00 |
| `bank.embargo` (CI) | embargo E clock units | 125 → 1,000 | O(1) | O(n) | a row is held once and released once; E/step rows held | 1.02 · 1.02 · 6.79 · 5.11 |
| `bank.label_clock` (CI) | the delay D from a row's clock to its `label_clock` | 125 → 1,000 | O(n) | O(n) | each row's `label_clock` is its clock plus a delay of up to 2D, so a row is placed among the held rows in `label_clock` order by moving the shorter side of the buffer, linear in the rows held; D/step rows held | 1.09 · 1.02 · 6.81 · 5.18 |
| `bank.formula_window` (CI) | a target's window W ahead | 125 → 1,000 | O(1) | O(n) | a row waits for its window once; W/step rows held | 0.99 · 0.99 · 7.44 · 3.31 |
| `bank.ljung_box_lags` | Ljung-Box lags L | 2 → 16 | O(n) | O(n) | the residual's autocorrelations at L lags | 1.07 · 1.00 · 1.12 · 1.03 |
| `bank.robust_se_lags` | Newey-West lags L | 1 → 8 | O(n) | O(n) | L lagged scores | 1.17 · 1.01 · 1.37 · 1.08 |
| `bank.resid_quantiles` | residual quantiles Q | 1 → 8 | O(n) | O(n) | a sketch a quantile | 1.14 · 1.03 · 1.02 · 1.09 |
| `runner.rows` | rows in the run | 4,096 → 32,768 | O(1) | O(1) | the run streams: a chunk at a time, whatever the rows behind or ahead | 1.01 · 0.99 · 1.01 · 1.00 |
| `runner.specs` | specs S, calls of 1,024 rows | 1 → 8 | O(n) | O(n) | S | 7.88 · 7.72 · 7.80 · 4.99 |
| `runner.specs_per_call` | specs S from 16 to 128, calls of 16 rows | 16 → 128 | O(n) | O(n) | a call's output a spec, S: the chunk's columns and the specs' outputs are joined in one pass over their names | 7.75 · 7.79 · 7.99 · 7.81 |
| `windows.groups` | groups G, rows in turn over all | 1,000 → 8,000 | O(1) | O(n) | a row touches its group's sums; a state a group | 0.82 · 1.06 · 7.48 · 7.89 |
| `windows.groups_seen` (CI) | groups seen G (a block over 50) | 1,000 → 8,000 | O(1) | O(n) | a row resolves its own group; a state a group | 0.98 · 0.99 · 7.96 · 7.86 |
| `windows.kernels` (CI) | kernels K | 1 → 8 | O(n) | O(n) | each kernel's queue a row | 2.98 · 2.25 · 2.42 · 2.58 |
| `windows.operators` | operators O on one kernel | 1 → 8 | O(n) | O(n) | each operator's sums a row | 2.61 · 3.06 · 2.42 · 4.04 |
| `windows.window_mean` (CI) | window W clock units | 1,250 → 10,000 | O(1) | O(n) | a row enters and leaves the queue once, O(1) amortized; W/step rows held | 0.90 · 1.00 · 7.74 · 6.92 |
| `windows.window_var` | window W, ewm_var | 1,250 → 10,000 | O(1) | O(n) | as the mean | 0.89 · 1.00 · 7.78 · 7.17 |
| `windows.window_quantile` | window W, ewm_quantile | 1,250 → 10,000 | O(log n) | O(n) | the window's sketch, log W | 1.08 · 0.93 · 6.33 · 6.68 |
| `windows.forward` | window W ahead, `ewm_mean(..., reverse=True)` | 1,250 → 10,000 | O(1) | O(n) | as the backward mean | 0.94 · 1.03 · 7.81 · 7.41 |
| `windows.history` (CI) | rows already seen | 1,000 → 8,000 | O(1) | O(1) | running sums only | 1.01 · 1.00 · 1.00 · 1.00 |
| `refresh.series` | series m | 2 → 16 | O(n) | O(n) | a tick updates its series; a point checks one counter | 0.83 · 0.40 · 3.22 · 1.42 |
| `refresh.pairs` | series m, pairs | 8 → 64 | O(n) | O(n^2) | a tick updates its m−1 pairs; m² pairs held | 10.45 · 8.96 · 67.43 · 43.77 |
| `refresh.groups` (CI) | groups G | 100 → 800 | O(1) | O(n) | a tick touches its group | 1.01 · 1.00 · 8.05 · 7.70 |
| `refresh.history` (CI) | ticks already seen | 1,000 → 8,000 | O(1) | O(1) | the grid's last values only | 1.00 · 1.00 · 1.00 · 1.00 |
| `gbdt.history` | rows offered | 10,000 → 80,000 | O(1) | O(1) | a priority sample: a row is kept or not by its hash, the heap holds the sample | 0.53 · 0.83 · 1.00 · 1.00 |
| `gbdt.sample` | sample size N | 1,000 → 8,000 | O(log n) | O(n) | a heap of N, log N; N rows kept | 1.90 · 1.01 · 8.00 · 7.30 |

The surfaces and sizes the table's cells do not measure, each with where it is
held instead or why it needs no cell:

| surface | the size | a row's work | state | held by, or why |
|---|---|---|---|---|
| `lasso`, `marginal`, `ew_class` under `window_size` | the window | O(1) | grows with the snapshots the window keeps | the same snapshot machinery as `ewridge.window` and `ew_cov.window` |
| `ewridge` with `session_shrink` | rows seen | O(1) | O(1) | a second set of sums beside the first, the same update twice |
| `micro`'s pruning | micro-clusters `m` | O(m²) once a `prune_every`, O(m) a row amortized | O(m) | `micro.max_clusters` reads the amortized row |
| `kmeans`'s seeding | the warm-up's rows | O(rows·k) once, at seeding | O(warm_rows·p), freed after | once for each spec and group, before the measured block of every cell |
| `rcov` at a block's close | the block's rows | O(n·H·k²) once a close | the block's rows | the close is the bank's, once a block; `rcov.block_rows` reads the rows that build it |
| the clock | rows seen | O(1) | O(1) | the `clock_float` and `clock_temporal` budget cases count a fixed block |
| the bank's `predict` | as `fit_predict`'s | as `fit_predict`'s | unchanged by a call | the same extraction and models, without the update |
| `Bank::fit_predict` with groups that close | groups open | O(1) a row | O(groups open) | closing a group drops its state; `bank.groups_seen` holds the open ones |
| the Polars source behind `LazyFrame.online.fit_predict` | specs; rows | O(specs); O(1) | the bank's | `tests/test_scaling.py`, by time |
| the chunked `ModelBank.fit_predict` | specs, at large and small calls | O(specs) | the bank's | `tests/test_scaling.py`, by time |
| `po.gbdt`'s build | rows | O(1) a row in the sample | O(sample) in the sample; O(rows) in XGBoost's and LightGBM's own matrices | the libraries bin every row by design; the sample is `gbdt.history` and `gbdt.sample` |

### Specs on one stream

**A bank's cost is linear in the specs it holds, a row at a time and a call
at a time, whether the specs share their columns or read their own.** The
spec cells hold it from 1 spec to 8 and from 8 to 64:

| the specs | cells |
|---|---|
| over one set of columns | `bank.specs`, `bank.specs_wide` |
| over columns of each spec's own | `bank.specs_distinct`, `bank.specs_distinct_wide` |
| a mix of `ewridge`, `huber`, `ew_cov`, `kmeans` and `bocpd`, five specs then forty | `bank.specs_mixed` |
| in calls of four rows, where what a call costs a spec -- its columns looked up, its output assembled -- is most of the cost | `bank.specs_per_call`, `bank.specs_per_call_wide` |

For 8x the specs they read 4.32x to 8.29x in time and 6.04x to 7.95x in
allocations. The lowest, `bank.specs_per_call`, reads under 8x because a
call's fixed cost, shared by its specs, is a large share at four rows. Specs
that share their columns cost what specs over columns of their own cost:
`bank.specs` and `bank.specs_distinct` read within 3% of each other. At
four features a spec, its model's step is most of its row, so a column read
once or once a spec does not show. The Python surfaces read the same
(`tests/test_scaling.py`). They read 5.4x to 7.2x for 8x the specs through
the Polars source and the chunked `ModelBank`, and 4.9x at calls of 4 and
16 rows, where their fixed cost per call is a larger share.

**The bank's threads share the specs of one call, one spec to each piece
of work, so its speed-up is bounded by the costliest spec and by the
call's own serial part.** Here is the wall time of 16 calls of 4,096 rows
after a warm-up, the least of three runs of `cargo run --profile dev-wheel
-p online-bench --example spec_threads`, a build without LTO. It was
measured on 2026-11-05 at a
load of 9.8 to 11.3 on the 14 cores, so other work held some of the
threads:

| specs | kind | 1 thread (ms) | 4 threads | 8 threads | speed-up at 4 | at 8 |
|---|---|---|---|---|---|---|
| 1 | `ewridge` | 15.9 | 16.0 | 17.4 | 0.99x | 0.92x |
| 8 | `ewridge` | 122.5 | 39.5 | 34.4 | 3.11x | 3.56x |
| 64 | `ewridge` | 1,030.6 | 310.4 | 220.5 | 3.32x | 4.67x |
| 5 | the mix of five | 133.0 | 97.3 | 90.9 | 1.37x | 1.46x |
| 40 | the mix of five | 1,091.3 | 318.5 | 243.1 | 3.43x | 4.49x |

Three readings, none held by a test. One spec gains nothing from more threads, since
its rows are sequential. Five specs of different kinds gain 1.4x where four
threads could give 4x, because the call waits for its slowest spec (here `bocpd` and `huber`, each
several times an `ewridge` row): the work is not split within a spec. Eight
or more specs reach 3.1x to 3.4x at 4 threads and 3.6x to 4.7x at 8, short of
linear by the call's serial part -- extracting the columns, assembling the
output -- and by the load the machine carried.

### What the table holds

**Six cells hold four mechanisms that keep a cost at its declared growth,
so a change that removes one fails its cells.** Each cell's ratios in
[The table](#the-table) read at its declaration:

| cell | declared | what keeps it there |
|---|---|---|
| `runner.specs_per_call` | O(specs) a call | the command line joins the specs' output columns to the chunk in one pass over their names (`with_columns` in `crates/online-polars/src/runner.rs`), so no spec's column rebuilds the schema of the columns before it |
| `hmm.k`, `ew_class.k_diag` | O(k) under `covariance_type="diag"` | a diagonal state or class keeps its k variances alone (`EwCovDiag`, `crates/online-core/src/ewcov/diag.rs`), and so does a window's snapshot of it; a variance's update reads no other entry of the matrix |
| `corrchange.window`, `corrchange.window_crit` | O(1) a row in `window_rows` | each window is merged from summaries of blocks of `⌈k/2⌉ + 3` rows: a count, an anchor row, the mean's offset from it and the centered co-moments, combined by Chan, Golub and LeVeque's pairwise rule, which needs no subtraction of a row that leaves (`crates/online-core/src/window_moments.rs`). A chunk of blocks keeps its suffix sums and a running prefix sum, so a window's full blocks are one suffix and one prefix, and only the rows at its two ends are read |
| `bank.groups_seen_per_call` | O(1) a call in the groups seen | a spec keeps its groups' states in an index map, and a call borrows its chunk's states by their places in it and takes the readiness notices of those groups alone |

A `corrchange` window of fewer than four blocks is read whole from its
rows, which costs no more than merging its blocks. Its permutation draws are a separate
`O(n_perm · window_rows · k²)` at each redraw. Unset, `permute_every_rows`
is `window_rows`, so the draws also cost `O(n_perm · k²)` a row.

Four declarations are higher than the obvious guess, because the
algorithm's own growth is:

| cell | from | to | why |
|---|---|---|---|
| `rcov.k` | O(k) | O(k²) | each lag of the kernel adds a k×k outer product, as `rcov`'s documentation says |
| `rcov.block_rows` | O(1) | O(n^(3/5)) | the automatic bandwidth's ring depth is `c·block_rows^(3/5)` lags, all read on a row |
| `audit.history`, its saved state | O(1) | O(log rows) | a counter's msgpack encoding lengthens as its count grows, while its heap stays flat |
| `bank.label_clock` | O(1) | O(n) in the delay D | each row is placed in `label_clock` order among the D/step held rows, by moving the shorter side of the buffer |
