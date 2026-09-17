# Running a bank as a job: the `online` command line

A *model bank* is a set of models fitted together over the same rows
([the README's introduction](../README.md#introduction) defines
the words this document uses). The README's [Running a
bank](../README.md#running-a-bank) assumes a live Python process driving
the rows through it — a Polars query, or a loop over chunks. A scheduled
job, or a deployment with no Python at all, runs the same bank from an input
file to an output file instead, with the standalone `online` program and a
configuration file. The piece of the library that does this is called the
*runner*. Same specs, same state file, same numbers.

## The `online` command line

The same pipeline as one binary and one TOML
([examples/bank.toml](../examples/bank.toml)), for a deployment with no
Python. The binaries are attached to each GitHub release.

```sh
online --config bank.toml
online --config bank.toml --resume bank.state --save-state bank.state
online --config bank.toml --resume bank.state --predict --input today.parquet
online --config bank.toml --input ticks.csv --output scored.ndjson
online --config bank.toml --input feed.dat --input-format ipc
online --config bank.toml --no-output --save-state gram.state   # the state is the product
online --config bank.toml --dry-run          # validate and print the output schema
```

`--predict` scores against the resumed state and learns nothing; it drops
the config's `save_state`, so one TOML serves both runs. `--no-output`
suppresses the per-row output; a run needs one or the other.

A TOML spec for a model that learns from no target — `ew_cov`, `kmeans`,
`micro` — may leave `targets` out, and it is filled with `features[0]` the
way the Python builders fill it. The two surfaces then write byte-identical
specs, so a state saved from one resumes under the other. The command line
reads with polars' own scanners, which on a stable toolchain lack the SIMD
CSV parser py-polars' wheels have, so for a large CSV `po.run` is the faster
of the two. In TOML, a Windows path needs single quotes or forward slashes
(`input = 'C:\data\in.parquet'`), since a backslash in a double-quoted string
starts an escape sequence.

From Rust, the same pipeline is `online_polars::run_config` for a
`RunConfig`, `run_config_on` for a `LazyFrame` or batches the caller already
has, and `run` with a callback instead of an output file.

## The state vocabulary is the same everywhere

`load_state` and `save_state` on `po.run`, `--resume` / `--save-state` /
`--predict` on the command line, and the same two words on
`lf.online.fit_predict` all read and write the same file, and the bytes are
the same whichever wrote them ([Saving, loading and
serving](../README.md#saving-loading-and-serving)).

## Closed groups to a sidecar file

`group_close` emits a finished group's accumulators as one row and frees its
state ([One row per finished group](../README.md#one-row-per-finished-group)).
A run writes those rows to a sidecar file as it goes, which with no output at
all is the whole shape of an accumulate-only pass over a stream that does not
fit in memory:

```sh
online --config bank.toml --no-output --closed-groups blocks.parquet
```

`ModelBank.fit_predict_batches(closed_groups=path)` and
`lf.online.fit_predict(closed_groups=path)` write the same file from Python.
What has closed and not been read is saved with the state, so a driver that
saves between chunks does not lose rows silently.

## Memory

Peak footprint on one file, `ewridge` with 20 features, parquet in and out —
the same measurement as the README's [Memory](../README.md#memory-which-calls-stream)
table, with and without `--no-output`:

| what you run | 3M rows | 12M rows |
|---|---:|---:|
| `online --config` | 0.95 / 0.73 GB | 1.41 / 0.75 GB |

Flat, like the two streaming surfaces in the README; what growth there is
comes from polars' parquet read-ahead, and
`POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes a run to 0.15 GB at the same
speed. [docs/PERFORMANCE.md](PERFORMANCE.md) §11 has every measurement.

## The runner, and its parallelism

The command line is a three-stage pipeline — a reader thread, the bank on
the calling thread, a writer thread — with one chunk in flight per stage; `ONLINE_TIMING=1` prints how long the bank waited on each side.
Reading and writing are polars' work on polars' pool: parquet pages are
encoded a column at a time there, NDJSON a slice per thread.

The GIL is released while a chunk is in the bank, so independent `po.run`
calls in threads of one process share the bank's one thread pool
(`POLARS_ONLINE_MAX_THREADS`; the README's
[Parallelism](../README.md#parallelism) has the full account, including
polars' own pool and how the two interact).

`chunk_rows` is a keyword on `po.run` and on the command line
(`--chunk-rows`, or `chunk_rows` in the TOML), with the same meaning and the
same default (100,000) as on `lf.online.fit_predict`. It never changes the
numbers: one chunk or a thousand gives the same output, and the only thing
that moves is where `coef` lands, since each stream reports its coefficients
on its last row of every chunk.

## Versioning

The floor is `LazyFrame.collect_batches`, which `po.run` and
`lf.online.fit_predict` read with and py-polars added in 1.34.0; the whole
suite passes on 1.34.0, 1.38.1 and 1.44.1 with identical numbers. The
README's [Versioning and the Polars pin](../README.md#versioning-and-the-polars-pin)
has the full matrix and which interfaces carry a promise.
