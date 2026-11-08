<!-- kind: how-to -->
# Resume from a saved state

This how-to is for anyone with a saved model bank state who wants the bank
to go on learning from the rows after the last one it learned, from Python
or from the command line.

## Prerequisites

Each way to resume needs a saved state and the rows that follow it:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The Python examples
  read `df`, the stream so far, and `today`, the 400 rows that follow it on
  the clock `t`. `lf` and `later` are the same rows, saved as
  `ticks.parquet` and `today.parquet` and read back as queries.
- A [saved state](../GLOSSARY.md#saving-and-serving): a file written by `bank.save`, by a query's
  `save_state=`, or by the command line's `--save-state`. [One state file,
  however the bank is
  run](../explanation/state-and-resuming.md#one-state-file-however-the-bank-is-run)
  shows them.
- The rows that follow the state's last row: for each
  [group](../GLOSSARY.md#the-model-bank-and-what-it-reads), rows whose
  [clock](../GLOSSARY.md#time) is after the last clock value the state
  learned in that group. Input that starts at or before that value has its
  own section, [Resume on input that overlaps the
  state](#resume-on-input-that-overlaps-the-state).
- For the command line: the `polars-online` program, whose binaries are
  attached to each GitHub release, and the configuration file `job.toml`
  that [docs/RUNNER.md](../RUNNER.md#a-first-run) runs its examples
  against.

## Resume a bank object

Use a bank object when your own Python code holds the rows, a frame at a
time.

1. **Load the state with `po.ModelBank.load`.** To check that the file
   holds the [specs](../GLOSSARY.md#the-model-bank-and-what-it-reads) you
   expect, pass them as `specs=`, and the load refuses a file that holds
   other specs.
2. **Feed the next rows to the loaded bank,** with `fit_predict`, `fit` or
   `fit_predict_batches`, as you would feed a bank that starts with nothing
   learned.
3. **Save the bank again with `bank.save`,** to resume from these rows
   next time.

This code fits `df`, saves the bank, loads it, and resumes on `today`. It
compares the resumed bank's predictions with one run over `df` and `today`
together. It uses `df` and `today` from [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
one_run = po.ModelBank([spec]).fit_predict(pl.concat([df, today]))   # df and today as one stream, for comparison

# A bank saved after df.
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("bank.state")

# 1. Load the state, checking that it holds this spec.
loaded = po.ModelBank.load("bank.state", specs=[spec])
# 2. Feed the next rows.
resumed = loaded.fit_predict(today)
# 3. Save again, for the next resume.
loaded.save("bank.state")


def pred(out):
    return out["ridge"].struct.field("pred_y")


print(f"largest difference from one run: {(pred(resumed) - pred(one_run.tail(400))).abs().max()}")
```

```text
largest difference from one run: 0.0
```

## Resume in a query

Use a query, a chain of Polars steps that runs only when you ask for its
result, when the next rows are in a file or a stream too large to hold.

1. **Name the state file in `load_state=` on the query's
   `online.fit_predict`,** as `later.online.fit_predict(load_state=...)`.
   The specs come from the file, so the call needs none. To have the load
   check them, pass them too.
2. **Name a file in `save_state=` to save the state when the run reaches
   its last row.** It may be the file `load_state=` names.
3. **Run the query,** here with `sink_parquet`, which runs it and writes
   the output to a file without holding it all in memory.

**A query that fails after the bank can still write the state, although
its output is missing.** A run you abandon partway, such as one whose
chunks you stop reading, or one that fails inside the bank, leaves the
`save_state` file as it was. But a failure in a step after the bank, such
as a full disk under `sink_parquet`, can write the state ([The one
gap](../explanation/state-and-resuming.md#the-one-gap-a-failure-after-the-bank)). One
shared file is safe only where nothing after the bank can fail. Otherwise,
give each run a dated state file of its own, named in `save_state=`, such
as `bank-2024-01-03.state`. To rerun a failed run, pass the state file of
the run before it, such as `bank-2024-01-02.state`, to `load_state=`. The
rerun learns the lost rows again and writes their output. Or run from
the command line, which saves the state only after the output is written.

This code fits `lf` and saves its state, then resumes on `later`, the
query over `today`'s rows. It uses `lf` and `later` from [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")   # the stream so far
(later.online.fit_predict(load_state="bank.state", save_state="bank.state")              # 1 and 2: load, and save at the end
      .sink_parquet("more.parquet"))                                                     # 3: run the query
print(f"rows fed in all: {po.ModelBank.load('bank.state').rows_fed}")
```

```text
rows fed in all: 800
```

## Resume from the command line

Use the command line to run a bank as a scheduled job, with no Python.
[docs/RUNNER.md](../RUNNER.md#flags-and-the-toml-keys-they-override) lists
every flag.

1. **Give `--config` the TOML file whose specs the state was saved with.**
2. **Give `--load-state` the state file.** The run refuses a state saved
   from other specs.
3. **Give `--input` the next rows' file,** in place of the TOML file's
   input.
4. **Give `--output` a file of its own,** or the run replaces the output
   the TOML file names.
5. **Give `--save-state` a file** to save the state to. It may be the file
   the run loads. The command line saves the state only after it has
   written the output, so a run that fails before its output is written
   leaves the previous output and state in place.

Here the first command fits `job.toml`'s spec over its input,
`ticks.parquet`, and saves `run.state`. The second resumes from it through
`today.parquet`:

```sh
polars-online --config job.toml --save-state run.state
polars-online --config job.toml --load-state run.state --input today.parquet --output today-fitted.parquet --save-state run.state
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
saved state to run.state
wrote 400 rows (1 chunk) to today-fitted.parquet
saved state to run.state
```

## Resume on input that overlaps the state

**Pass input that starts at or before a group's last learned clock value
through `bank.skip_learned` before you feed it to the bank.** A rerun of
the day, or a file that overlaps the last one, steps each group's clock
back to rows the state has learned. The bank refuses that
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads), unless
`restart_after_step_back` is set below the step back: then the model
resets and learns the overlap again from nothing. `skip_learned` drops
the rows the state has learned, so each row is learned once:

| a row of the input | `skip_learned` |
|---|---|
| after its group's last clock value, in every spec that reads a clock | keeps it |
| of a group the bank has not seen | keeps it |
| with a null clock | keeps it, for the bank to refuse by name |
| at or before its group's last clock value | drops it: the state has learned it |

**A bank whose specs all count rows, with no clock, has no position to
compare, and `skip_learned` raises `ValueError` there.** If the input
starts where the saved input did, drop the learned rows with
`frame.slice(bank.rows_fed)`. Otherwise, keep a column that numbers the
rows, such as a row index written with the input, and drop the rows the
bank learned with Polars' `filter` on it in the query before the bank. Or
give the spec a clock, so that `skip_learned` can find them.

1. **Load the state with `po.ModelBank.load`.**
2. **Pass the input through `bank.skip_learned`.** It takes a frame or a
   query and keeps the row order, so a query stays a query.
3. **Feed the kept rows to the bank,** or, for a query, run it with
   `load_state=` and `save_state=` as in [Resume in a
   query](#resume-in-a-query).

The comparison is exact: a temporal clock is compared in the integer
nanoseconds the state keeps, whatever the column's unit or time zone. A row
at a group's last clock value counts as learned, so a stream saved between
two rows with the same clock value loses every later row with that value.

This code saves a bank after 1,500 rows, then resumes it on a file that
starts 100 rows before the save. It builds its own rows:

```python
import numpy as np

# The stream, and a rerun file that overlaps the save.
rng = np.random.default_rng(3)
x0, x1, noise = rng.standard_normal((3, 2_000))
stream = pl.DataFrame({"t": np.arange(2_000.0), "x0": x0, "x1": x1,
                       "y": 0.5 + 1.5 * x0 - 2.0 * x1 + 0.1 * noise})
stream.slice(1_400).write_parquet("rerun.parquet")        # starts 100 rows before the save
# A bank saved after the first 1,500 rows.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
saved = po.ModelBank([spec])
saved.fit_predict(stream.head(1_500))
saved.save("ridge.state")

# The rerun as it stands is refused, and the bank is untouched.
bank = po.ModelBank.load("ridge.state")                    # 1. load the state
try:
    bank.fit_predict(pl.read_parquet("rerun.parquet"))
except ValueError as e:
    print(str(e).split(". ")[0])                           # the error's first sentence
kept = bank.skip_learned(pl.scan_parquet("rerun.parquet"))   # 2. a query of the rows not learned
print(f"skip_learned keeps {kept.collect().height} of {pl.scan_parquet('rerun.parquet').collect().height} rows")

# 3. Learn on from the kept rows, and compare with one run over the whole stream.
(kept.online.fit_predict(load_state="ridge.state", save_state="ridge.state")
     .sink_parquet("rerun-fitted.parquet"))
one_run = po.ModelBank([spec])
one_run.fit_predict(stream)
print(f"the state is one run's, byte for byte: {po.ModelBank.load('ridge.state').save_bytes() == one_run.save_bytes()}")
```

```text
spec "ridge": clock column "t" steps back by 99 at row 0 (counting from 0), and restart_after_step_back is not set, so the bank refused the chunk and learned none of it
skip_learned keeps 500 of 600 rows
the state is one run's, byte for byte: True
```

**On the command line, give `--skip-learned` beside `--load-state`** (the
TOML key `skip_learned`). It filters each chunk it reads as
`skip_learned` does. Here the second run feeds `job.toml`'s input,
`ticks.parquet`, again, which the first has just learned, and learns none
of it twice. The rows it drops get no output row, so its output file is
empty:

```sh
polars-online --config job.toml --save-state run.state
polars-online --config job.toml --load-state run.state --skip-learned --output rerun-fitted.parquet --save-state run.state
```

```text
wrote 400 rows (1 chunk) to fitted.parquet
saved state to run.state
wrote 0 rows (0 chunks) to rerun-fitted.parquet
saved state to run.state
```

## Check that it worked

A [saved state](../GLOSSARY.md#saving-and-serving) is everything a
[model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) has learned,
written to a file. A bank that resumes from a state this version of
polars-online saved learns each new row as one unbroken run over the old
rows and the new would have learned it. So it gives the same predictions,
to the bit, as that run. A state saved by an earlier 1.x release loads
too. It continues to the bit, apart from the kinds of state the
[CHANGELOG](../../CHANGELOG.md) declares, each with the tolerance it
continues within or the row from which it differs.
[docs/STATE-WORKFLOW.md](../STATE-WORKFLOW.md) says what a saved state
carries and what each step guarantees.

**Compare the resumed run with one unbroken run over the old rows and the
new, fitted by a bank that starts with nothing learned.** Compare the new
rows' `pred_<target>` fields, as [Resume a bank
object](#resume-a-bank-object) does, or the two states' bytes from
`save_bytes()`, as [the overlap
recipe](#resume-on-input-that-overlaps-the-state) does. Other fields can
differ where the two runs' chunks end at different rows: without
`coef_every` or `max_rows_between_coefs`, `coef` is written only on each
group's last accepted row in each chunk.

**If the state does not load, `po.ModelBank.load` raises an error that
names what is wrong with the file:**

| the file | error | what to do |
|---|---|---|
| does not exist yet | `FileNotFoundError` | on a first run, build the bank from its specs with `po.ModelBank(specs)`; otherwise, check the path |
| is not a bank, or holds different specs from the `specs=` given | `ValueError` | check the path, and pass the specs the state was saved with |
| holds a state that contradicts its own spec | `ValueError` | refit the bank from its input |
| was written under a newer state schema or file format than this version reads | `ValueError` | upgrade polars-online to the version that wrote it, or a later one |
| was saved by 0.13.0 or an earlier release | `ValueError`, naming the range of schemas this version reads | refit the bank from its input. `po.schema_version()` gives this version's schema |

This code makes three loads that fail. It uses `df` and `ticks.parquet`
from [Example data](../../README.md#example-data):

```python
# A saved bank, and a spec that does not match it.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("bank.state")
other = po.spec.ewridge("ridge", targets=["y"], features=["x0"], clock="t", half_life=600.0, gap_cap=300.0)
# Three loads that fail: no file, the wrong spec, not a state file.
for path, specs in [("nowhere.state", [spec]), ("bank.state", [other]), ("ticks.parquet", None)]:
    try:
        po.ModelBank.load(path, specs=specs)
    except (FileNotFoundError, ValueError) as e:
        print(f"{type(e).__name__}: {e}")
```

```text
FileNotFoundError: [Errno 2] No such file or directory: 'nowhere.state'
ValueError: state file: its specs do not match the bank's specs: spec "ridge" (1 of 1) differs at features: ["x0","x1"] in the file, ["x0"] in the bank. Load the state with the specs it was saved with.
ValueError: state file: these bytes are not a polars-online bank state file, or the file is cut short. Load a whole file that ModelBank.save or the command line's save_state wrote.
```
