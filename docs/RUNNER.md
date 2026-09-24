# Running a bank as a job: the `online` command line

A *model bank* is a set of models fitted together over the same rows
([the README's introduction](../README.md#introduction) defines the words
this document uses). The README's
[Running a bank](../README.md#running-a-bank) assumes a live Python process
driving the rows through it: a Polars query, or a loop over chunks. A
scheduled job, or a deployment with no Python at all, runs the same bank
from an input file to an output file instead. It uses the standalone
`online` program and a configuration file. The piece of the library that
does this is called the *runner*. Same specs, same state file, same
numbers.

| section | what it covers |
|---|---|
| [Running it](#running-it) | [a first run](#a-first-run) · [saving, resuming and scoring](#saving-resuming-and-scoring) · [a run whose product is its state](#a-run-whose-product-is-its-state) · [closed groups to a sidecar file](#closed-groups-to-a-sidecar-file) |
| [The configuration file](#the-configuration-file) | [flags and the TOML keys they override](#flags-and-the-toml-keys-they-override) · [input and output files](#input-and-output-files) · [specs in TOML](#specs-in-toml) · [clocks that are times](#clocks-that-are-times) |
| [Memory, threads and chunk size](#memory-threads-and-chunk-size) | [memory](#memory) · [the pipeline and its threads](#the-pipeline-and-its-threads) · [chunk size](#chunk-size) |
| [From Rust, and the Polars it needs](#from-rust-and-the-polars-it-needs) | [the same pipeline from Rust](#the-same-pipeline-from-rust) · [versioning](#versioning) |

## Running it

Every run reads a configuration file, and flags on the command line
override most of what it says
([Flags and the TOML keys they override](#flags-and-the-toml-keys-they-override)).

### A first run

The runner is the same pipeline, packaged as one binary and one TOML
file ([examples/bank.toml](../examples/bank.toml)), for a deployment with
no Python. The binaries are attached to each GitHub release. The
configuration names the input, the output and the specs. The examples
below run against this configuration, saved as `bank.toml`:

```toml
input = "ticks.parquet"      # read a chunk at a time; the extension names the format
output = "fitted.parquet"    # the input's columns, plus one column per spec

[[specs]]                    # one table per spec
name = "ridge"               # the name of the spec's output column
targets = ["y"]
features = ["x0"]
halflife = 500.0             # in rows, since the spec names no clock column
min_periods = 5.0            # the floor, in n_eff units, below which no prediction is made
[specs.model]
type = "ew_ridge"
```

A dry run validates the configuration and prints the output schema without
reading a row. A plain run then fits the specs over the input and writes
the output:

```sh
online --config bank.toml --dry-run
online --config bank.toml
```

### Saving, resuming and scoring

**`--save-state` saves the bank when the run ends, and `--resume` starts a
run from a saved one.** One file can serve as both, so a run resumes from
the state a previous run left and saves over it:

```sh
online --config bank.toml --resume run.state --save-state run.state
```

The state is saved last, after the output is committed, so a run that
fails leaves the previous output in place and the state unwritten. The
file is the one a query's `load_state` and `save_state` read and write on
`lf.online.fit_predict`, and the one `ModelBank.save` writes. The bytes are
the same whichever wrote them
([Saving, loading and serving](../README.md#saving-loading-and-serving)).

**`--predict` scores against the resumed state and learns nothing:**

```sh
online --config bank.toml --resume run.state --predict --input today.parquet
```

It drops the configuration's `save_state`, so one TOML file serves both
the learning run and the scoring run. A `--save-state` given with it is
refused, since the run has nothing new to save.

### A run whose product is its state

**`--no-output` writes no per-row output.** A run whose product is its
state has no use for it:

```sh
online --config bank.toml --no-output --save-state gram.state
```

A run needs somewhere to put its work: an output file, a `save_state`, or
a `closed_groups` file. Asked for none of the three, it is refused before
it reads a row.

### Closed groups to a sidecar file

`group_close` emits a finished group's accumulators as one row and frees
its state ([One row per finished group](../README.md#one-row-per-finished-group)).
A run writes those rows to a sidecar file as it goes. With no per-row
output at all, that file is the whole product of an accumulate-only pass
over a stream that does not fit in memory:

```sh
online --config blocks.toml --no-output --closed-groups blocks.parquet
```

The configuration needs a spec with `group_close`. Without one, the run is
refused, and the message says to add `group_close = "monotone"` or
`"session"` to the spec whose groups should be emitted.

`ModelBank.fit_predict_batches(closed_groups=path)` and
`lf.online.fit_predict(closed_groups=path)` write the same file from
Python. What has closed and not been read is saved with the state, so a
driver that saves between chunks does not lose rows silently.

## The configuration file

The TOML file names the input, the output, the run's settings and the
specs. Every key but the specs and `keep_columns` has a flag that
overrides it.

### Flags and the TOML keys they override

| flag | TOML key | what it does |
|---|---|---|
| `--config` | | the TOML file to read; required |
| `--input` | `input` | the file to read |
| `--output` | `output` | the file to write; refused together with `--no-output` |
| `--no-output` | `output` left out | write no per-row output |
| `--input-format` | `input_format` | how to read the input: `parquet`, `ipc`, `csv` or `ndjson`. The extension decides when it is not given |
| `--output-format` | `output_format` | how to write the output, from the same four |
| `--chunk-rows` | `chunk_rows` | rows per chunk; 100,000 by default |
| `--resume` | `load_state` | start from a saved state |
| `--save-state` | `save_state` | save the state when the run ends, after the output is committed |
| `--closed-groups` | `closed_groups` | write the groups that closed to a sidecar file; needs a spec with `group_close` |
| `--predict` | `predict` | score against the resumed state and learn nothing; needs `--resume` or `load_state`, drops the TOML's `save_state`, and refuses `--save-state` |
| `--dry-run` | | validate the configuration and print the output schema, reading no row |
| `-q`, `--quiet` | | suppress the per-chunk progress |
| | `keep_columns` | the input columns to keep; all of them when it is empty |
| | `[[specs]]` | the specs, one table each |

### Input and output files

Read and write any of the four formats. The extension decides, unless
`--input-format` or `--output-format` names the format, and a file whose
extension does not say needs one:

```sh
online --config bank.toml --input today.csv --output scored.ndjson
online --config bank.toml --input feed.dat --input-format ipc --output out.parquet
```

The command line reads with polars' own scanners, which on a stable
toolchain lack the SIMD CSV parser py-polars' wheels have. A large CSV
therefore reads faster through py-polars and
`ModelBank.fit_predict_batches`.

In TOML, a Windows path needs single quotes or forward slashes
(`input = 'C:\data\in.parquet'`), since a backslash in a double-quoted
string starts an escape sequence.

### Specs in TOML

A TOML spec for a model that learns from no target, such as `ew_cov`,
`kmeans` or `micro`, may leave `targets` out. It is filled the way the
Python builders fill it: with `features[0]`, or, for a `bocpd` spec with a
`hazard_col` and an `hmm` spec with an `exog_tvtp`, with that column. The
two surfaces then write byte-identical specs, so a state saved from one
resumes under the other.

### Clocks that are times

A spec whose clock is a `Datetime`, `Date` or `Duration` column gives its
clock parameters as durations, in the text polars writes them in:

```toml
[[specs]]
name = "ridge"
targets = ["y"]
features = ["x0", "x1"]
clock = "ts"           # a Datetime column
halflife = "10m"       # a row's weight halves every ten minutes
max_dclock = "5m"
label_delay = "30s"
[specs.model]
type = "ew_ridge"
```

TOML has no duration type, so the text is the whole spelling. It is the
same text a Python spec keeps for `pl.duration(minutes=10)` or
`timedelta(minutes=10)`. A numeric clock takes plain numbers of its own
units. Either mixture is refused, naming the column and the parameter
([Time and decay](../README.md#time-and-decay)).

## Memory, threads and chunk size

A run holds the bank's state, one chunk in each stage of its pipeline, and
what Polars has read ahead of it.

### Memory

The figures are the peak footprint on one file, `ewridge` with 20
features, parquet in and out: the same measurement as the README's
[Memory](../README.md#memory-which-calls-stream) table.

| what you run | 3M rows | 12M rows |
|---|---:|---:|
| `online --config` | 0.73 GB | 0.75 GB |
| `online --config`, with `POLARS_ROW_GROUP_PREFETCH_SIZE=1` | | 0.15 GB |

It is flat, like the two streaming surfaces in the README. Nearly all of
it is Polars reading ahead in the parquet file, and
`POLARS_ROW_GROUP_PREFETCH_SIZE=1` takes a run to 0.15 GB at the same
speed. [docs/PERFORMANCE.md](PERFORMANCE.md) §11 has every measurement.

### The pipeline and its threads

The command line is a three-stage pipeline: a reader thread, the bank on
the calling thread, and a writer thread, with one chunk in flight per
stage. `ONLINE_TIMING=1` prints how long the bank waited on each side.
Reading and writing are polars' work, on polars' pool: parquet pages are
encoded there a column per task, in parallel, and NDJSON a slice per
thread.

The bank's own work runs on the bank's one thread pool, sized by
`POLARS_ONLINE_MAX_THREADS`. In Python, the GIL is released while a chunk
is in the bank, so independent `ModelBank` calls in threads of one process
share that pool. The README's [Parallelism](../README.md#parallelism) has
the full account, including polars' own pool and how the two interact.

### Chunk size

`chunk_rows` is a keyword on the command line (`--chunk-rows`, or
`chunk_rows` in the TOML). It has the same meaning and the same default,
100,000, as on `lf.online.fit_predict` and `ModelBank.fit_predict_batches`.
It never changes the numbers: one chunk or a thousand gives the same
output, and it only trades memory for overhead. The one thing that moves
is where `coef` lands, since each stream reports its coefficients on its
last row of every chunk.

## From Rust, and the Polars it needs

### The same pipeline from Rust

From Rust, the same pipeline is `online_polars::run_config` for a
`RunConfig`. `run_config_on` runs it for a `LazyFrame` or batches the
caller already has, and `run` with a callback instead of an output file.

### Versioning

The command line needs no Python, so the Polars floor applies only to the
Python calls this guide names. The floor is `LazyFrame.collect_batches`,
which `lf.online.fit_predict` and `ModelBank.fit_predict_batches` read
with, and which py-polars added in 1.34.0. The whole suite passes on
1.34.0, 1.38.1, 1.44.1 and the 2.0 release candidate with identical
numbers. The README's
[Versioning and the Polars pin](../README.md#versioning-and-the-polars-pin)
has the full matrix and which interfaces carry a promise.
