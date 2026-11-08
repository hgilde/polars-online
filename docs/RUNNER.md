<!-- kind: reference -->
# The `polars-online` command line

This reference page is for anyone who runs a model bank as a job with the
`polars-online` program: it gives every flag and TOML key, the files a run
reads and writes, its exit status and its Rust entry points. A [model
bank](GLOSSARY.md#the-model-bank-and-what-it-reads) is a set of models
fitted together over the same rows. The [command
line](GLOSSARY.md#saving-and-serving) runs one from an input file to an
output file with no Python, for a scheduled job or a deployment. It takes
the same specs as Python, reads and writes the same state file, and gives
the same numbers. [Run a bank in a query](how-to/run-a-bank-in-a-query.md)
and [Run a bank in a Python loop](how-to/run-a-bank-in-a-python-loop.md)
drive the rows through a bank from a live Python process instead.

**The shell examples run in a directory that holds the files they read.**
Under each command, a block shows what it prints, standard output and
standard error together, apart from the progress line, which is drawn only
on a terminal. The files are these:

| file | what it is |
|---|---|
| `job.toml` | the configuration [The configuration file](#the-configuration-file) shows: one spec with one feature and no groups, so that its output is short |
| `blocks.toml` | the configuration [Closed groups to a sidecar file](#closed-groups-to-a-sidecar-file) shows: one `ew_cov` spec whose groups close |
| `ticks.parquet`, `today.parquet` | the files the README's [Example data](../README.md#example-data) writes: 400 rows, and the 400 rows that follow them on the clock `t` |
| `today.csv` | `today.parquet`'s rows as CSV, written with Polars' `pl.read_parquet("today.parquet").write_csv("today.csv")` |
| `feed.dat` | `ticks.parquet`'s rows as Arrow IPC, written with Polars' `pl.read_parquet("ticks.parquet").write_ipc("feed.dat")`, under a name whose extension does not say its format |

| section | what it holds |
|---|---|
| [Running it](#running-it) | where the steps of a run are, the [closed-groups file](#closed-groups-to-a-sidecar-file) and the [exit status](#exit-status) |
| [The configuration file](#the-configuration-file) | every [flag and TOML key](#flags-and-the-toml-keys-they-override), the [input and output files](#input-and-output-files), [specs in TOML](#specs-in-toml) and [clocks that are times](#clocks-that-are-times) |
| [Memory, threads and chunk size](#memory-threads-and-chunk-size) | what a run holds, its [threads](#threads) and its [chunk size](#chunk-size) |
| [From Rust, and the Polars it needs](#from-rust-and-the-polars-it-needs) | the [Rust entry points](#the-same-pipeline-from-rust) and what [`--version`](#versioning) prints |

## Running it

[Run a bank from the command line](how-to/run-a-bank-from-the-command-line.md)
gives the steps of a run: check the TOML file with `--dry-run`, run it,
save the state for the next run with `--save-state`, and keep only the
state with `--no-output`. [Resume from a saved
state](how-to/resume-from-a-saved-state.md#resume-from-the-command-line)
gives `--load-state`, and [Resume on input that overlaps the
state](how-to/resume-from-a-saved-state.md#resume-on-input-that-overlaps-the-state)
gives `--skip-learned`. [Serve from a saved
state](how-to/serve-from-a-saved-state.md#serve-from-the-command-line)
gives `--predict`.

### Closed groups to a sidecar file

**`--closed-groups` (the TOML key `closed_groups`) writes each closed
group's row to a file of its own.** `group_close` emits a [closed
group](GLOSSARY.md#saving-and-serving)'s running sums as one row and frees
its state ([Close groups and read each one's row](how-to/close-groups.md)
gives the steps). With no per-row output at all, that file is the whole
product of a pass that only learns, over a stream that does not fit in
memory. Here the run reads `blocks.toml`, which holds one `ew_cov` spec of
`x0` and `x1`, grouped by `stock_id`, whose groups close at each change of
the `session` column:

```toml
input = "ticks.parquet"
output = "fitted.parquet"    # not written: the command gives --no-output

[[specs]]
name = "cov"
features = ["x0", "x1"]
half_life = 500.0
group = "stock_id"           # one model for each stock
group_close = "session"      # a stock's group closes at each change of its session
session = "session"
[specs.model]
type = "ew_cov"
```

```sh
polars-online --config blocks.toml --no-output --closed-groups blocks.parquet
```

```text
processed 400 rows (1 chunk), no output
wrote closed groups to blocks.parquet
```

The configuration needs a spec with `group_close`. Without one, the run is
refused, and the message says to add `group_close = "monotone"` or
`"session"` to the spec whose groups should be emitted.

**The command line writes the closed rows as it goes, and publishes the
file when the run ends.** It hands each chunk's closed rows to a writer of
their own. A run that finishes publishes the file, and so does a run that
fails after any group has closed, since a closed row has left the bank and
the file is the only place it is. The output file is published only by a
run that finishes.

**From Python, `closed_groups=` writes the same rows once, when the pass
ends.** `ModelBank.fit` and `ModelBank.fit_predict_batches` take each
chunk's closed rows out of the bank and hold them in the process. They
write the file when the chunks stop, even when the run stops early or
fails. `lf.online.fit_predict` writes it only after the bank has been fed
its last row, so a run that fails writes no file. Either way, memory grows
with the number of groups that close, never with the rows. Without
`closed_groups=`, a bank keeps its closed rows until `bank.closed_groups()`
reads them, and saves them with the state, so a loop that saves between
chunks loses none.

### Exit status

A script can tell how a run ended from its exit status:

| status | the run |
|---|---|
| 0 | finished, and wrote what it was asked for; or `--dry-run`, `--help` or `--version` printed its answer |
| 1 | was refused or failed: a configuration, a state file or a value the bank refuses, or a file it cannot read or write. Standard error carries the message, led by `polars-online:` |
| 2 | could not parse its command line, such as a flag it does not know or a flag with its value missing. Standard error says what is wrong |

**The summary goes to standard output, and everything else to standard
error.** The summary is the lines a run prints when it ends, such as `wrote
400 rows (1 chunk) to fitted.parquet` and `saved state to run.state`.
After `wrote`, a spec whose groups ended the run not ready has a line of
its own. It counts the groups whose last row was withheld, by
`withheld_reason`, and those with a coefficient more ridge than data
(`support_coef_min < 0.5`), as
`spec "ridge": 2 groups whose last row was withheld (below_min_weight 2); 1 group with support_coef_min < 0.5`.
A spec with neither has no line, as in every run on this page.
Standard error carries the progress line on a terminal, the timings
`POLARS_ONLINE_TIMING=1` asks for, and the error of a run that fails. It also
carries the [readiness warnings](GLOSSARY.md#warm-up-and-readiness) about a
model's warm-up, led by `polars-online:` as an error is. So read a failure from the exit status: a line on standard error does
not mean the run failed.

**Read a run's result from its exit status and the files it writes, never
from what it prints.** The exit status and the files are the command
line's contract. The summary, the
warnings, the errors and the timing lines are text for a person, which a
minor release may reword.

## The configuration file

The TOML file names the input, the output, the run's settings and the
specs. Every key but the specs and `keep_columns` has a flag that
overrides it. `--predict` and `--skip-learned` can only turn their keys
on, and `--no-output` can only clear `output`. Every example
on this page but the closed-groups one, which reads `blocks.toml`, runs
against this configuration, saved as `job.toml`
([examples/bank.toml](../examples/bank.toml) is a fuller one):

```toml
input = "ticks.parquet"      # read a chunk at a time; the extension names the format
output = "fitted.parquet"    # the input's columns, plus one column per spec

[[specs]]                    # one table per spec
name = "ridge"               # the name of the spec's output column
targets = ["y"]              # or a table: { column = "p", name = "price" }, as po.target
features = ["x0"]
clock = "t"                  # the clock: how far apart two rows are, which half_life and gap_cap are measured in
half_life = 500.0            # in units of t
gap_cap = 300.0              # the longest step of t that decays the fit
min_weight = 5.0             # the floor, in weight_sum units, below which no prediction is made
[specs.model]
type = "ewridge"
```

### Flags and the TOML keys they override

A flag given on the command line wins over its key in the TOML file:

| flag | TOML key | what it does |
|---|---|---|
| `--config` | | the TOML file to read; required. A relative path inside it is read from its directory |
| `--input` | `input` | the file to read |
| `--output` | `output` | the file to write; refused together with `--no-output` |
| `--no-output` | `output` left out | write no per-row output; needs `save_state` or `closed_groups`, and is refused with `--predict` |
| `--input-format` | `input_format` | how to read the input: `parquet`, `ipc`, `csv` or `ndjson`. The extension decides when it is not given |
| `--output-format` | `output_format` | how to write the output, from the same four |
| `--output-layout` | `output_layout` | `struct` (the default) writes each spec as one output column; `flat` writes each of its fields as a `<spec>.<field>` column, as a CSV always is. Under `flat`, `--dry-run` names the output as `(parquet, flat)` |
| `--chunk-size` | `chunk_size` | rows per chunk; 100,000 by default |
| `--threads` | `threads` | the threads the bank runs on, a whole number from 1 to 1,024, or to the machine's core count where that is larger; left out, the bank's default number of threads, which `POLARS_ONLINE_MAX_THREADS` sets. The output is the same at any count |
| `--load-state` | `load_state` | start from a saved state |
| `--skip-learned` | `skip_learned` | drop the input's rows the loaded state has learned, as `ModelBank.skip_learned` does; needs `--load-state` or `load_state`, and a spec that reads a clock. A run that keeps no row writes an output with no rows, and saves the state it loaded, byte for byte |
| `--save-state` | `save_state` | save the state when the run ends, after the output is committed |
| `--closed-groups` | `closed_groups` | write the groups that closed to a sidecar file; needs a spec with `group_close` |
| `--predict` | `predict` | predict from the loaded state and learn nothing; needs `--load-state` or `load_state`, drops the TOML's `save_state` and `closed_groups`, and refuses `--save-state` and `--closed-groups` |
| `--dry-run` | | check the configuration, the input's schema and the bank as the run would, and print the output schema, feeding the bank no row |
| `-q`, `--quiet` | | suppress the progress line, which is drawn only when standard error is a terminal |
| `-h`, `--help` | | print every flag, and run nothing |
| `-V`, `--version` | | print the version, and run nothing: `-V` the package's version alone, `--version` three lines ([Versioning](#versioning)) |
| | `keep_columns` | the input columns to keep; all of them when it is empty |
| | `[[specs]]` | the specs, one table each |

### Input and output files

**A relative path in the TOML file is read from the file's own directory,
and a relative path given as a flag from the working directory.** So a
configuration runs the same from wherever a scheduler starts it, and
`polars-online --config jobs/job.toml` reads `jobs/ticks.parquet`. A flag
such as `--input today.parquet` names the file the shell sees, in the
directory the command runs in. The input, the output, `load_state`,
`save_state` and `closed_groups` all follow the rule.

**Every path is a local file.** A URL in any of them, such as
`https://host/x.parquet` or `s3://bucket/x.parquet`, is refused by name
before anything is read, under `--dry-run` too. Polars' own reader would
otherwise fetch it with its built-in cloud client, which retries a host that
does not answer for up to three minutes. Copy the file down first, or read it
from Python, where py-polars' reader takes the URL. A `file://` URL is read as
the local path it names: `file:///data/x.parquet` is `/data/x.parquet`. A
Windows path such as `C:\data\x.parquet` is a path, not a URL.

Read and write any of the four formats. The extension decides, unless
`--input-format` or `--output-format` names the format, and a file whose
extension does not say needs one:

```sh
polars-online --config job.toml --input today.csv --output scored.ndjson
polars-online --config job.toml --input feed.dat --input-format ipc --output out.parquet
```

```text
wrote 400 rows (1 chunk) to scored.ndjson
wrote 400 rows (1 chunk) to out.parquet
```

**A glob or a directory is read file by file, in the order of the paths as
text.** `input = "parts/*.parquet"` reads every file the pattern matches,
and a directory reads every file in it, given `input_format`, since a
directory has no extension to name its format. As text, `part-10` comes
before `part-2`, so pad the numbers in the names (`part-02`). With a clock
column, a file out of order is refused as a step back; without one, its
rows are learned out of order, and nothing says so.

The command line reads with Polars' own Rust readers, built without the
faster CSV parser py-polars' wheels carry, so a large CSV reads faster
through py-polars and `ModelBank.fit_predict_batches`. CSV holds no
records, so each spec's output column is written as `<spec>.<field>`
columns. A numeric list such as `coef` becomes JSON text, which
`str.json_decode(pl.List(pl.Float64))` reads back. So does a list of
names, such as a closed Gram's `columns`, read back with
`str.json_decode(pl.List(pl.String))`. Any other nested field is refused,
naming the column and the formats that carry it.

**A wide spec writes several times faster to parquet flat, and no faster
to IPC.** Parquet, IPC and NDJSON keep each spec's output as one output
column, a Polars struct, by default. Polars' parquet writer encodes a
struct's fields at several times the cost of the same values as plain
columns. `--output-layout flat` writes the `<spec>.<field>` columns a CSV
gets, so the writer no longer sets the run's pace. A list field such as
`coef` stays one list column. IPC writes the struct at about the cost of
the flat columns. An `ew_cov` over 20 series has 233 fields. Over 300,000
rows on 14 cores, the command line took these times a row, the first line
the best of three runs:

| output | as the struct | flat | the machine's load from other work |
|---|---:|---:|---|
| parquet | 7,697 ns | 1,274 ns | 22 to 24 |
| parquet, repeated | 7,104 ns | 1,181 ns | 3 |
| IPC | 991 ns | 1,026 ns | 2.4 to 3.1 |

The same run with no output at all took 676 ns a row, at the load of 22
to 24. The two IPC figures differ by less than the runs' noise.

In TOML, a Windows path needs single quotes or forward slashes
(`input = 'C:\data\in.parquet'`), since a backslash in a double-quoted
string starts an escape sequence.

### Specs in TOML

A TOML spec takes the names a Python builder takes, but for the columns
of two models: a table names its columns `targets` and `features` for every
model, where `ew_class`'s builder calls its target `label` and `audit`'s
calls its columns `columns`. A table that says `label` or `columns` is
refused, naming the key to use. A key under an old name is refused, naming
the new one: `halflife` is `half_life`, for one. A key renamed when the
parameters took the names they keep at 1.0 is still read, with a warning on
standard error naming the new one, until the next major version: `kmeans`' `k` is
`n_clusters`, for one. A spec for a model that learns from no target, such as `ew_cov`,
`kmeans` or `micro`, may leave `targets` out. It is filled the way the
Python builders fill it: with `features[0]`, or, for a `bocpd` spec with a
`hazard_col` and an `hmm` spec with an `exog_tvtp`, with that column. The
two surfaces then write byte-identical specs, so a state saved from one
resumes under the other.

**The same file builds the same bank in Python.** `tomllib` reads each
`[[specs]]` table as the dict a builder returns, and `po.ModelBank` takes
the list as it comes. Here the bank is built from `job.toml`, in the
directory where [a plain run](how-to/run-a-bank-from-the-command-line.md#run-a-bank-over-a-file) wrote `fitted.parquet`:

```python
import tomllib

import polars as pl
import polars_online as po

with open("job.toml", "rb") as f:
    bank = po.ModelBank(tomllib.load(f)["specs"])
# The spec a Python builder makes from the same keywords
spec = po.spec.ewridge(
    "ridge", targets=["y"], features=["x0"], clock="t",
    half_life=500.0, gap_cap=300.0, min_weight=5.0,
)
print(f"the same spec: {bank.specs[0] == spec}")
# The predictions the command line wrote, against the bank's in Python
ours = bank.fit_predict(pl.read_parquet("ticks.parquet"))["ridge"].struct.field("pred_y")
theirs = pl.read_parquet("fitted.parquet")["ridge"].struct.field("pred_y")
print(f"the same predictions, bit for bit: {ours.equals(theirs)}")
```

```text
the same spec: True
the same predictions, bit for bit: True
```

A [formula target](GLOSSARY.md#fitting-a-model), an expression that
holds at least one window operator looking ahead, is a table with its name and its formula. The formula takes the form a spec and a saved state
carry ([Make a target that looks
ahead](how-to/make-a-target-that-looks-ahead.md)):

```toml
[[specs]]
name = "edge"
targets = [{ name = "fwd", formula = ["-", ["rewm_mean", ["col", "mid"], { half_life = 10.0, window_size = 60.0 }], ["col", "mid"]] }]
features = ["x0"]
clock = "t"
gap_cap = 300.0
half_life = 600.0
embargo = 60.0           # at least the target's window_size, for a run that writes output
[specs.model]
type = "ewridge"
```

The formula is Polars' expression written as a tree: an operator, its
input and its keywords. TOML has no null, so a null literal is written
`["lit"]`. **This written form is labeled unstable**: it may change in any
release without that counting as a breaking change, in the TOML file and
in the state it is saved into. With `POLARS_ONLINE_WARN_UNSTABLE=1`, the
command line says so on standard error. The formula target itself, what it
computes and when its rows are released, is promised. `group_close` is
refused beside a formula target, and so is `group` without `clock`.

### Clocks that are times

A spec whose clock is a `Datetime`, `Date` or `Duration` column gives its
clock parameters as durations, in the text polars writes them in:

```toml
[[specs]]
name = "ridge"
targets = ["y"]
features = ["x0", "x1"]
clock = "ts"           # a Datetime column
half_life = "10m"      # a row's weight halves every ten minutes
gap_cap = "5m"
embargo = "30s"
[specs.model]
type = "ewridge"
```

TOML has no duration type, so the text is the whole spelling. It is the
same text a Python spec keeps for `pl.duration(minutes=10)` or
`timedelta(minutes=10)`. A numeric clock takes plain numbers of its own
units. Either mixture is refused, naming the column and the parameter
([Clock types and units](MODELS.md#the-clock-and-its-units)).

## Memory, threads and chunk size

A run holds the bank's state, one chunk in each of its three stages
(reading, learning and writing), and what Polars has read ahead of it.
[Peak memory of the command
line](PERFORMANCE.md#peak-memory-of-the-command-line) measures it at 3M,
12M and 24M rows. [Where the time and memory
go](explanation/where-the-time-and-memory-go.md#where-the-time-goes)
explains how the three stages share a run's time.

### Threads

The bank's own work runs on threads of its own. `--threads N`, or a
`threads` key in the TOML, which the flag overrides, gives it N threads.
Without either, it runs on as many threads as `POLARS_ONLINE_MAX_THREADS`
gives, one per core when that is unset. The count is not saved with the
state, so a resumed run takes the one it is given, and the output is the
same at any count. [Set the bank's thread
count](how-to/use-more-cores.md#set-the-banks-thread-count) has the full
account, including Polars' own threads and how the two interact.

### Chunk size

`chunk_size` is a keyword on the command line (`--chunk-size`, or
`chunk_size` in the TOML). It has the same meaning and the same default,
100,000, as on `lf.online.fit_predict` and `ModelBank.fit_predict_batches`.
It never changes the numbers: one chunk or a thousand gives the same
output, and it only trades memory for overhead. Only which rows carry
`coef`, and `support_coef` beside it, can differ, and only by default. The
bank then writes them on each group's last accepted row in every chunk, so
smaller chunks report them more often. Under `coef_every` or
`max_rows_between_coefs` they do not move:

```sh
polars-online --config job.toml --chunk-size 7 --output small-chunks.parquet
```

```text
wrote 400 rows (58 chunks) to small-chunks.parquet
```

## From Rust, and the Polars it needs

The command line is a thin wrapper over a Rust crate of this workspace,
`online-polars`, which a Rust program can call itself. The binary carries
its own Rust Polars, so it needs no Python at all.

### The same pipeline from Rust

From Rust, the same pipeline is `online_polars::run_config` for a
`RunConfig`. `run_config_on` runs it for a `LazyFrame` or batches the
caller already has. `run` takes a `Bank` the caller holds, an `Input` (a
`LazyFrame`, or frames the caller produces), an `Output` (`File`,
`Batches` for a callback of the caller's own, or `Discard`), `RunOptions`
and a progress callback. `RunOptions`' `learn_only` is the `fit` mode a run
with no output uses.

### Versioning

The Rust Polars the command line carries is 0.55.2.
`polars-online --version` prints three lines: the package's version, the
state schema its saved states are written in, and the Rust Polars it
carries. `-V` prints the first line alone.

The command line carries its own Rust Polars, so py-polars' version
matters only to the Python calls that read a query, such as
`lf.online.fit_predict` and `ModelBank.fit`. For those, the floor is
py-polars 1.37.0. They read their input with `LazyFrame.collect_batches`,
which gives a query's result one chunk at a time.
py-polars added it in 1.34.0, but before 1.37.0 it ran ahead of a slower
reader without limit, holding the input whole. [Versions and the Polars
pin](VERSIONS.md) has the versions the suite passes on, and [Polars
interfaces](explanation/polars-interfaces.md) which interfaces carry a promise.
