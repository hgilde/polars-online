# Running a bank as a job: `po.run` and the `online` command line

A *model bank* is a set of models fitted together over the same rows, in
time order ([the README's introduction](../README.md#introduction) defines
the words this document uses). The README's [Running a
bank](../README.md#running-a-bank) assumes a live Python process driving
the rows through it — a Polars query, or a loop over chunks. A scheduled
job, or a deployment with no Python at all, runs the same bank from an input
file to an output file instead: `po.run` from a script, or the standalone
`online` program from a configuration file. The piece of the library that
does this is called the *runner*. Same specs, same state file, same numbers.

## `po.run`

The runner reads a chunk, fits it, and writes it, with one chunk in each of
those three stages at a time. The output is written to a temporary file and
renamed into place at the end, so a run that fails leaves the previous
output where it was.

```python
po.run(input="ticks.parquet", output="fitted.parquet",
       specs=[spec], chunk_rows=100_000, save_state="bank.state")   # -> {"rows": ..., "chunks": ...}

po.run("bank.toml", input="today.csv")                              # keywords override the TOML
po.run(input="today.parquet", output="scored.parquet", specs=[spec],
       load_state="bank.state", predict=True)                       # serve: learn nothing
```

`input` is anything py-polars can stream:

- a **path** in parquet, ipc, csv or ndjson — told from the extension, or
  named with `input_format=`; globs and cloud URLs as `pl.scan_*` takes them;
- a **`LazyFrame`**, with whatever scan options its query needs;
- a **`DataFrame`**;
- **any iterable of frames** in stream order — a database cursor, a socket,
  a generator.

`output` is a path in any of the four formats. `keep_columns=[...]` selects
input columns before the bank sees them. `progress(rows, chunks)` is called
after each chunk; raising in it stops the run. CSV cannot hold struct
columns, so there each spec's struct is flattened to `<spec>.<field>` columns
and the `coef` list becomes a JSON string that
`pl.col("ridge.coef").str.json_decode(pl.List(pl.Float64))` reads back
bit-exact.

**When the product of a run is its state, leave `output` out.** An
accumulator-only spec emits `n_eff` a row and nothing else; over a billion
rows that is 8 GB of file written so it can be deleted. Without an `output`
the run writes nothing and `save_state` is required — a run that writes
nothing and saves nothing has done nothing:

```python
wide = po.spec.ew_cov("gram", features=[f"x{i}" for i in range(3)], stats=[], halflife=1000.0)
po.run(input="ticks.parquet", specs=[wide], save_state="gram.state")   # no output at all
```

`no_output=True` says the same thing over a config that names an output, and
is what the command line's `--no-output` sets.

`save_state` is written only after the output is committed, so a run that
fails leaves neither a half-written output nor a state that has moved past
it. That ordering is `po.run`'s own; a query's `save_state` writes when the
bank reaches the last row whether or not what comes after the bank succeeds
([docs/STATE-WORKFLOW.md](STATE-WORKFLOW.md)).

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
A run writes those rows to a sidecar file as it goes, which with no `output`
at all is the whole shape of an accumulate-only pass over a stream that does
not fit in memory:

```python
po.run(input=by_block.lazy(), specs=[blocks], closed_groups="blocks.parquet")
```

`online --closed-groups path` writes it too. What has closed and not been
read is saved with the state, so a driver that saves between chunks does not
lose rows silently.

## Memory

Peak footprint on one file, `ewridge` with 20 features, parquet in and out —
the same measurement as the README's [Memory](../README.md#memory-which-calls-stream)
table, with and without `--no-output`:

| what you run | 3M rows | 12M rows |
|---|---:|---:|
| `po.run(...)`, `online --config` | 0.95 / 0.73 GB | 1.41 / 0.75 GB |

Flat, like the two streaming surfaces in the README; what growth there is
comes from polars' parquet read-ahead, and
`POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes a run to 0.15 GB at the same
speed. [docs/PERFORMANCE.md](PERFORMANCE.md) §11 has every measurement.

## The runner, and its parallelism

`po.run` and the command line are a three-stage pipeline — a reader thread,
the bank on the calling thread, a writer thread — with one chunk in flight
per stage; `ONLINE_TIMING=1` prints how long the bank waited on each side.
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
