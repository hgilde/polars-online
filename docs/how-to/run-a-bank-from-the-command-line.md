<!-- kind: how-to -->
# Run a bank from the command line

This how-to is for anyone who wants to run a model bank as a scheduled
job, or in a deployment with no Python, from an input file to an output
file.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set
of models over the same rows, one for each
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the description
of a model) and each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads).
The [command line](../GLOSSARY.md#saving-and-serving), the standalone
`polars-online` program, runs a bank from a TOML file. It takes the same
specs as Python, reads and writes the same state file, and gives the same
numbers. Every run reads its TOML file, and flags on the command line
override most of what it says. [docs/RUNNER.md](../RUNNER.md) is the
reference for every flag, every TOML key and the exit status.

## Prerequisites

A run needs the program, a TOML file and an input file:

- The `polars-online` program, whose binaries are attached to each GitHub
  release.
- [Example data](../EXAMPLE-DATA.md) built, in the
  directory the commands run in. The examples read `ticks.parquet`, a
  stream of 400 rows with a clock `t`.
- An input file in Parquet, Arrow IPC, CSV or NDJSON, holding every column
  the specs read, the targets included.

## Run a bank over a file

Write the TOML file once, check it, and then run it as often as the job
needs.

1. **If a target is computed from its row's columns, such as a return
   against the mid, make it a column of the input file.** Make it with
   Polars' `with_columns` before you write the file the run reads. A
   [formula target](../GLOSSARY.md#fitting-a-model) (a target written as
   a window expression) in the TOML file must look ahead, as in Python, and [Specs in TOML](../RUNNER.md#specs-in-toml)
   shows how to write one.
2. **Write the TOML file,** with the input, the output and one `[[specs]]`
   table for each spec. Each `[[specs]]` table takes the keywords every
   spec shares, and its `[specs.model]` table names the model and takes
   the model's own. Save this file as `job.toml` beside `ticks.parquet`:

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

   A relative path in the TOML file is read from the file's own directory,
   and a relative path given as a flag from the working directory
   ([Input and output files](../RUNNER.md#input-and-output-files)). So a
   TOML file runs the same from wherever a scheduler starts it.
   [examples/bank.toml](../../examples/bank.toml) is a fuller file, and
   [Specs in TOML](../RUNNER.md#specs-in-toml) gives the keys a spec takes.

3. **Check the TOML file with `--dry-run`.** It prints each spec's output
   fields, as `<spec>.<field>`, without feeding the bank a row.

4. **Run it with `--config` alone.** A plain run fits the specs over the
   input and writes the output. What depends on the rows themselves, such
   as a clock that steps back, only the run can find. On a terminal, a
   progress line above the summary counts the rows as each chunk is read,
   rewritten in place. It is drawn only when standard error is a terminal,
   so a log or a pipe never holds it, and `--quiet` turns it off.

Here the first command checks `job.toml` (step 3) and the second runs it
(step 4):

```sh
polars-online --config job.toml --dry-run
polars-online --config job.toml
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
wrote 400 rows (1 chunk) to fitted.parquet
```

**`--dry-run` refuses what the run would refuse at its first step.** It
reads the input's schema as the run would, from a parquet footer or a
CSV's first rows. It opens the bank as the run would, loading
`--load-state`'s state, and runs it on no rows of that schema. So it
refuses a state that is not there or was saved from other specs. It
refuses a column a spec reads that the input lacks, or that the TOML
file's `keep_columns` (the input columns to keep) leaves out. And it
refuses a formula target's embargo shorter than its window.

## Save the state for the next run

**`--save-state` saves the bank's
[state](../GLOSSARY.md#saving-and-serving), everything it has learned,
when the run ends.** The state is saved last, after the output is
committed, so a run that fails leaves the previous output in place and the
state unwritten. The file is the one a query's `load_state` and
`save_state` read and write on `lf.online.fit_predict`, and the one
`ModelBank.save` writes. The bytes are the same whichever wrote them
([One state file, however the bank is
run](../explanation/state-and-resuming.md#one-state-file-however-the-bank-is-run)).

1. **Give `--save-state` the file to save the state to,** or set
   `save_state` in the TOML file.
2. **To learn on from the state in the next run, give it to
   `--load-state`.** [Resume from a saved
   state](resume-from-a-saved-state.md#resume-from-the-command-line) gives
   the steps, with `--skip-learned` for input that overlaps what the state
   has learned.
3. **To predict new rows from the state without learning from them, give
   `--predict` beside `--load-state`.** [Serve from a saved
   state](serve-from-a-saved-state.md#serve-from-the-command-line) gives
   the steps.

Here the run fits `job.toml`'s spec over `ticks.parquet` and saves the
state to `run.state`:

```sh
polars-online --config job.toml --save-state run.state
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
saved state to run.state
```

## Keep only the state

**`--no-output` writes no per-row output.** A run whose product is its
state, such as a pass that leaves the running sums of a fit, has no use for
it. A run needs somewhere to put its work: an output file, a `save_state`,
or a `closed_groups` file ([Closed groups to a sidecar
file](../RUNNER.md#closed-groups-to-a-sidecar-file)). Asked for none of the
three, it is refused before it reads a row. A run with no output keeps no
prediction, so it is the command line's `ModelBank.fit`, and a formula
target there takes any embargo, as `fit` does. Here the run saves the
state and writes nothing else:

```sh
polars-online --config job.toml --no-output --save-state gram.state
```

```text
processed 400 rows (1 chunk), no output
saved state to gram.state
```

## Check that it worked

**Read the exit status: 0 means the run finished and wrote what it was
asked for.** 1 means it was refused or failed, and 2 that it could not
parse its command line. Standard error carries the message. A line on
standard error does not mean the run failed, since the [readiness
warnings](../GLOSSARY.md#warm-up-and-readiness) about a model's warm-up go
there too. [Exit status](../RUNNER.md#exit-status)
gives each status.

**Read the summary on standard output.** A run prints `wrote` with the
rows and chunks it wrote, and `saved state to` with the file, as the
runs in [Run a bank over a file](#run-a-bank-over-a-file) and [Save the
state for the next run](#save-the-state-for-the-next-run) print. After
`wrote`, a run prints a line for each spec
that ended with groups not ready, and nothing for the others. Such a line
reads `spec "ridge": 2 groups whose last row was withheld (below_min_weight 2)`.
A spec with groups that have a coefficient set more by the `ridge`
penalty than by the data adds `; 3 groups with support_coef_min < 0.5` to
its line. A script reads the
exit status and the files, never the words, which a minor release may
reword.

**To check the spec, build the same bank in Python from the TOML file.**
`tomllib` reads each `[[specs]]` table as the dict a spec builder returns,
and `po.ModelBank` takes the list as it comes. This code builds the bank
from `job.toml` and compares its spec with the one `po.spec.ewridge` makes
from the same keywords. It uses `job.toml` from [Run a bank over a
file](#run-a-bank-over-a-file):

```python
import tomllib

import polars_online as po

with open("job.toml", "rb") as f:                      # tomllib reads a file opened in binary mode
    bank = po.ModelBank(tomllib.load(f)["specs"])
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0"], clock="t",
                       half_life=500.0, gap_cap=300.0, min_weight=5.0)
print(f"the same spec: {bank.specs[0] == spec}")
```

```text
the same spec: True
```

The predictions the run writes are, bit for bit, what the same bank
writes in Python. [Specs in
TOML](../RUNNER.md#specs-in-toml) compares the command line's predictions
with the bank's.
