# Running a bank as a job: the `polars-online` command line

A *model bank* is a set of models fitted together over the same rows
([the README's introduction](../README.md#introduction) defines the words
this guide uses). The README's
[Running a bank](../README.md#running-a-bank) drives the rows through a
bank from a live Python process: a Polars query, or a loop over chunks. A
scheduled job, or a deployment with no Python at all, runs the same bank
from an input file to an output file instead, with the standalone `polars-online`
program and a configuration file. This part of the library is the
*runner*. It takes the same specs as Python, reads and writes the same
state file, and gives the same numbers.

**Every shell example here is a recipe: the commands, then what they
print.** Each runs against the configuration below, saved as
`job.toml`: a smaller file than the README's `bank.toml`, with one feature
and no groups, so that its output is short. Its directory holds the files the README's [Example
data](../README.md#example-data) writes (`ticks.parquet` and
`today.parquet`), and `run.state`, which the configuration's spec saved
from `ticks.parquet`. It also holds `today.csv` and `feed.dat`, the same
rows in two other formats. `tests/test_production_hardening.py` runs each command
through the built `polars-online` and holds what it prints, standard output and
standard error together, to what is shown. The progress line is drawn only
on a terminal, so the outputs shown leave it out.

| section | what it covers |
|---|---|
| [Running it](#running-it) | [a first run](#a-first-run) · [saving, resuming and scoring](#saving-resuming-and-scoring) · [a run whose product is its state](#a-run-whose-product-is-its-state) · [closed groups to a sidecar file](#closed-groups-to-a-sidecar-file) · [exit status](#exit-status) |
| [The configuration file](#the-configuration-file) | [flags and the TOML keys they override](#flags-and-the-toml-keys-they-override) · [input and output files](#input-and-output-files) · [specs in TOML](#specs-in-toml) · [clocks that are times](#clocks-that-are-times) |
| [Memory, threads and chunk size](#memory-threads-and-chunk-size) | [memory](#memory) · [the pipeline and its threads](#the-pipeline-and-its-threads) · [chunk size](#chunk-size) |
| [From Rust, and the Polars it needs](#from-rust-and-the-polars-it-needs) | [the same pipeline from Rust](#the-same-pipeline-from-rust) · [versioning](#versioning) |

## Running it

Every run reads a configuration file, and flags on the command line
override most of what it says
([Flags and the TOML keys they override](#flags-and-the-toml-keys-they-override)).

### A first run

The runner is a bank packaged as one binary and one TOML file
([examples/bank.toml](../examples/bank.toml) is a fuller one). The
binaries are attached to each GitHub release. The configuration names the
input, the output and the specs. Every example on this page runs against
this one, saved as `job.toml`:

```toml
input = "ticks.parquet"      # read a chunk at a time; the extension names the format
output = "fitted.parquet"    # the input's columns, plus one column per spec

[[specs]]                    # one table per spec
name = "ridge"               # the name of the spec's output column
targets = ["y"]              # or a table: { column = "p", name = "price" }, as po.target
features = ["x0"]
clock = "t"                  # the clock column, which lets a resumed run tell the rows it has learned
half_life = 500.0            # in units of t
gap_cap = 300.0              # the longest step of t that decays the fit
min_weight = 5.0             # the floor, in weight_sum units, below which no prediction is made
[specs.model]
type = "ewridge"
```

**A target computed from its row's columns, such as a return against the
mid, must be a column of the input file:** make it with Polars'
`with_columns` before writing the file the run reads. A formula target in
the configuration must look ahead, as in Python. The table form
`{ column = "p", relative_to = "mid" }` is refused by name.

**`--dry-run` checks the configuration and prints the output's columns
without feeding the bank a row.** It reads the input's schema as the run's
scan would, from a parquet footer or a CSV's first rows. It opens the bank
as the run would, loading `--load-state`'s state, and runs it on no rows of
that schema. So it refuses what the run would refuse at its first step. That
is a state that is not there or was saved from other specs, a column a spec
reads that the input or `keep_columns` lacks, and a window target's
embargo short of its window. It names each spec's output fields as
`<spec>.<field>`:

```sh
polars-online --config job.toml --dry-run
```

```text
config OK: 1 spec(s)
  ridge (ewridge):
    ridge.pred_y
    ridge.resid_y
    ridge.weight_sum
    ridge.settled_frac
    ridge.withheld_reason
    ridge.coef
    ridge.support_coef
input:  ticks.parquet (parquet)
output: fitted.parquet (parquet)
chunk_size: 100000
```

**A plain run fits the specs over the input and writes the output.** What
depends on the rows themselves, such as a clock that steps back, only the
run can find. On a terminal, a progress line above the summary counts the
rows as each chunk is read, rewritten in place. It is drawn only when
standard error is a terminal, so a log or a pipe never holds it, and
`--quiet` turns it off:

```sh
polars-online --config job.toml
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
```

### Saving, resuming and scoring

**`--save-state` saves the bank when the run ends, and `--load-state` starts a
run from a saved one.** One file can serve as both, so a run resumes from
the state a previous run left and saves over it. Here `run.state` holds
what the spec learned from `ticks.parquet`, and the run learns on through
`today.parquet`, the 400 rows that follow:

```sh
polars-online --config job.toml --input today.parquet --load-state run.state --save-state run.state
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
saved state to run.state
```

The state is saved last, after the output is committed, so a run that
fails leaves the previous output in place and the state unwritten. The
file is the one a query's `load_state` and `save_state` read and write on
`lf.online.fit_predict`, and the one `ModelBank.save` writes. The bytes are
the same whichever wrote them
([Saving, loading and serving](../README.md#saving-loading-and-serving)).

**Input that overlaps the state is refused unless `--skip-learned` drops
the overlap.** A rerun steps the clock back, and the run stops at the first
row it would learn twice. `--skip-learned` (the TOML key `skip_learned`)
drops the rows the state has learned, as Python's
`ModelBank.skip_learned` does. In every spec that reads a clock, it keeps
a row whose clock is after its group's last learned one, a row of a group
the state has not seen, and a row with a null clock. A row at its group's
last clock counts as learned. It needs `--load-state`, and a spec that
reads a clock, since with no clock there is nothing to compare, and the
overlap would be learned twice. Here the second run feeds `today.parquet`
again, which the first has just learned, and learns none of it twice:

```sh
polars-online --config job.toml --input today.parquet --load-state run.state --save-state run.state
polars-online --config job.toml --input today.parquet --load-state run.state --save-state run.state --skip-learned
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
saved state to run.state
wrote 0 rows (0 chunks) to fitted.parquet
saved state to run.state
```

The second run keeps no row: it writes an output with no rows, and it
saves the state it loaded, byte for byte. Without `--skip-learned` it would stop
at its first row, whose clock is 399 behind the state's last.

**`--predict` scores against the resumed state and learns nothing:**

```sh
polars-online --config job.toml --load-state run.state --predict --input today.parquet
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
```

It drops the configuration's `save_state` and `closed_groups`, so one TOML
file serves both the learning run and the scoring run. A `--save-state` or
a `--closed-groups` given with it is refused, since the run has nothing
new to save.

### A run whose product is its state

**`--no-output` writes no per-row output.** A run whose product is its state, such as a pass that leaves the
running sums of a fit, has no use for it:

```sh
polars-online --config job.toml --no-output --save-state gram.state
```

```text
processed 400 rows (1 chunk), no output
saved state to gram.state
```

A run needs somewhere to put its work: an output file, a `save_state`, or
a `closed_groups` file. Asked for none of the three, it is refused before
it reads a row. A run with no output keeps no prediction, so it is the
command line's `ModelBank.fit`, and a window target there takes any
embargo, as `fit` does.

### Closed groups to a sidecar file

`group_close` emits a finished group's accumulators as one row and frees
its state ([One row per finished group](../README.md#one-row-per-finished-group)).
A run writes those rows to a sidecar file as it goes. With no per-row
output at all, that file is the whole product of an accumulate-only pass
over a stream that does not fit in memory. Here `blocks.toml` holds one
`ew_cov` spec of `x0` and `x1`, grouped by `stock_id`, whose groups close
at each change of the `session` column (`group_close = "session"`):

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

`closed_groups=` on `lf.online.fit_predict`, `ModelBank.fit_predict_batches`
and `ModelBank.fit` writes the same file from Python. What has closed and
not been read is saved with the state, so a loop that saves between chunks
does not lose rows silently. The sidecar is published by a run that
drained any row into it, even one that then failed, since a drained row
has left the bank. The output is published only by a run that finishes.

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
carries the notices about a model's warm-up, led by `polars-online:` as an error
is. So read a failure from the exit status: a line on standard error does
not mean the run failed.

**The exit status and the files a run writes are the command line's
contract, and what it prints is not.** The summary, the notices, the
errors and the timing lines are text for a person, which a minor release
may reword. So a script reads the exit status and the files, never the
words.

## The configuration file

The TOML file names the input, the output, the run's settings and the
specs. Every key but the specs and `keep_columns` has a flag that
overrides it, though `--predict` can only switch scoring on.

### Flags and the TOML keys they override

A flag given on the command line wins over its key in the TOML file:

| flag | TOML key | what it does |
|---|---|---|
| `--config` | | the TOML file to read; required. A relative path inside it is read from its directory |
| `--input` | `input` | the file to read |
| `--output` | `output` | the file to write; refused together with `--no-output` |
| `--no-output` | `output` left out | write no per-row output |
| `--input-format` | `input_format` | how to read the input: `parquet`, `ipc`, `csv` or `ndjson`. The extension decides when it is not given |
| `--output-format` | `output_format` | how to write the output, from the same four |
| `--chunk-size` | `chunk_size` | rows per chunk; 100,000 by default |
| `--load-state` | `load_state` | start from a saved state |
| `--skip-learned` | `skip_learned` | drop the input's rows the loaded state has learned; needs `--load-state` or `load_state`, and a spec that reads a clock |
| `--save-state` | `save_state` | save the state when the run ends, after the output is committed |
| `--closed-groups` | `closed_groups` | write the groups that closed to a sidecar file; needs a spec with `group_close` |
| `--predict` | `predict` | score against the resumed state and learn nothing; needs `--load-state` or `load_state`, drops the TOML's `save_state` and `closed_groups`, and refuses `--save-state` and `--closed-groups` |
| `--dry-run` | | check the configuration, the input's schema and the bank as the run would, and print the output schema, feeding the bank no row |
| `-q`, `--quiet` | | suppress the progress line, which is drawn only when standard error is a terminal |
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
`str.json_decode(pl.List(pl.Float64))` reads back, and any other nested
field is refused, naming the column and the formats that carry it.

In TOML, a Windows path needs single quotes or forward slashes
(`input = 'C:\data\in.parquet'`), since a backslash in a double-quoted
string starts an escape sequence.

### Specs in TOML

A TOML spec takes the names a Python builder takes, and a key under an
old name is refused, naming the new one: `halflife` is `half_life`, for
one. A spec for a model that learns from no target, such as `ew_cov`,
`kmeans` or `micro`, may leave `targets` out. It is filled the way the
Python builders fill it: with `features[0]`, or, for a `bocpd` spec with a
`hazard_col` and an `hmm` spec with an `exog_tvtp`, with that column. The
two surfaces then write byte-identical specs, so a state saved from one
resumes under the other.

**The same file builds the same bank in Python.** `tomllib` reads each
`[[specs]]` table as the dict a builder returns, and `po.ModelBank` takes
the list as it comes. Here the bank is built from `job.toml`, in the
directory where [A first run](#a-first-run) wrote `fitted.parquet`:

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

A target that is a window expression looking ahead is a table with its
name and its formula. The formula takes the form a spec and a saved state
carry ([Windows as a model's inputs and
target](WINDOWS.md#windows-as-a-models-inputs-and-target)):

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
`["lit"]`. **This written form is labelled unstable**: it may change in any
release without that counting as a breaking change, in the TOML file and
in the state it is saved into. With `POLARS_ONLINE_WARN_UNSTABLE=1`, the
command line says so on standard error. The window target itself, what it
computes and when its rows are released, is promised. `group_close` is
refused beside a window target, and so is `group` without `clock`.

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
([Time and decay](../README.md#time-and-decay)).

## Memory, threads and chunk size

A run holds the bank's state, one chunk in each stage of its pipeline, and
what Polars has read ahead of it.

### Memory

The figures are the peak footprint on one file, `ewridge` with 20
features, parquet in and out: the same measurement as the README's
[Memory](../README.md#memory) table.

| what you run | 3M rows | 12M rows |
|---|---:|---:|
| `polars-online --config` | 0.74 GB | 0.77 GB |
| `polars-online --config`, with `POLARS_ROW_GROUP_PREFETCH_SIZE=1` | 0.24 GB | 0.24 GB |

It is flat: the bank's memory does not grow with the rows, and this
process keeps none of the freed pages the Python surfaces in the README's
table hold on to as they run longer. Nearly all of it is Polars reading
ahead in the parquet file, and with
`POLARS_ROW_GROUP_PREFETCH_SIZE=1` a run takes 0.24 GB.
`scripts/memory_bench.py` makes these measurements, on Polars 2.0.0 and 14
cores, and [docs/PERFORMANCE.md](PERFORMANCE.md) §26 re-measures the command
line, CSV and NDJSON included.

### The pipeline and its threads

The command line is a three-stage pipeline: a reader thread, the bank on
the calling thread, and a writer thread, with one chunk in flight per
stage. A `closed_groups` sidecar has a writer thread of its own.
`POLARS_ONLINE_TIMING=1` prints how long the bank waited on each side. Parquet
pages are encoded on Polars' pool, a column per task, in parallel, and
NDJSON on the writer's own thread.

The bank's own work runs on the bank's one thread pool, sized by
`POLARS_ONLINE_MAX_THREADS`. The README's
[Parallelism](../README.md#parallelism) has the full account, including
Polars' own pool and how the two interact.

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
caller already has. `run` takes a `Bank` the caller holds, an `Output`, a
file, a callback or `Discard`, and `RunOptions`, whose `learn_only` is the
`fit` mode a run with no output uses.

### Versioning

The command line needs no Python: it carries its own Rust Polars, 0.55.2.
`polars-online --version` prints three lines: the package's version, the
state schema its saved states are written in, and the Rust Polars it
carries. `-V` prints the first line alone.
So the py-polars floor applies only to the Python calls this guide names.
That floor is `LazyFrame.collect_batches`, which `lf.online.fit_predict`
and `ModelBank.fit_predict_batches` read with, and which py-polars added in
1.34.0. The README's
[Versioning and the Polars pin](../README.md#versioning-and-the-polars-pin)
has the versions the suite passes on and which interfaces carry a promise.
