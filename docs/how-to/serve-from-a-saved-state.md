<!-- kind: how-to -->
# Serve from a saved state

This how-to is for anyone with a saved model bank state who wants
predictions for new rows without the bank learning from them, from Python
or from the command line.

To [serve](../GLOSSARY.md#saving-and-serving) is to predict rows from a
[state](../GLOSSARY.md#saving-and-serving) without learning from them.
Every row is predicted from the same state, which `predict` never moves.
Since it skips learning each row, `ewridge` serves 1.8 times as fast as it
learns at 5 features, and 2.8 times at 20
([PERFORMANCE §30](../PERFORMANCE.md#30-where-every-row-solves-2026-09-29)).
To go on learning from the new rows instead, follow [Resume from a saved
state](resume-from-a-saved-state.md). [A bank's state and resuming
it](../explanation/state-and-resuming.md) explains why a state carries
everything a bank needs to go on.

## Prerequisites

Each way to serve needs a saved state and the rows to predict:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The Python examples
  read `df`, the stream so far, and `today`, the 400 rows that follow it on
  the clock `t`. `later` is `today`'s rows, saved as `today.parquet` and
  read back as a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  (a chain of Polars steps that runs only when you ask for its result).
- A [saved state](../GLOSSARY.md#saving-and-serving): a file written by
  `bank.save`, by a query's `save_state=`, or by the command line's
  `--save-state`.
- The rows to predict, with every column the state's
  [specs](../GLOSSARY.md#the-model-bank-and-what-it-reads) read as a
  feature, a clock, a group or a session. The target columns may be
  absent. Serve only rows the state has not learned: on a row it has
  learned, the prediction has seen its own target ([What predicting before
  learning cannot
  guarantee](../explanation/out-of-sample.md#what-predicting-before-learning-cannot-guarantee)).
- For the command line: the `polars-online` program, whose binaries are
  attached to each GitHub release, the configuration file `job.toml`, and
  its state `run.state`. [Run a bank over a
  file](run-a-bank-from-the-command-line.md#run-a-bank-over-a-file) writes
  `job.toml`, and [Save the state for the next
  run](run-a-bank-from-the-command-line.md#save-the-state-for-the-next-run)
  writes `run.state`.

## Serve in Python

Use a bank object for rows your own code holds in a frame, and a query for
rows in a file or a stream too large to hold. To serve a frame:

1. **Load the state with `po.ModelBank.load`.** To check that the file
   holds the specs you expect, pass them as `specs=`, and the load refuses
   a file that holds other specs.
2. **Predict the frame with `bank.predict(frame)`.** It returns the frame
   with one output column for each spec, and the bank does not move.

To serve a query:

1. **Call the query's `online.predict` with the loaded bank or the path of
   the state file.** It reads a bank as it stands when the query runs, and
   a path when the query is built. To pick up a newer state file, build the
   query again.
2. **Run the query,** here with `collect`, which runs it and returns its
   result, or with `sink_parquet`, which writes the result to a file
   without holding it all in memory.

This code fits `df`, saves the bank and loads it back, then serves the
rows after `df`'s three ways. It uses `df`, `today` and `later` from
[Example data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("bank.state")

# Load the state, checking that it holds this spec.
loaded = po.ModelBank.load("bank.state", specs=[spec])
# Three ways to serve the rows after df's: a frame, then a query given the bank or the path.
served = loaded.predict(today)                             # every row predicted from the same state: nothing moves
in_query = later.online.predict(loaded).collect()          # a bank, as it stands when the query runs
from_file = later.online.predict("bank.state").collect()   # a path, read when the query is built
# Every row saw the same state, and the three agree.
weights = served["ridge"].struct.field("weight_sum")
print(f"weight_sum on every row: {weights.min():.4f} to {weights.max():.4f}; the three agree: "
      f"{served['ridge'].equals(in_query['ridge']) and in_query['ridge'].equals(from_file['ridge'])}")
```

```text
weight_sum on every row: 320.4975 to 320.4975; the three agree: True
```

## Serve from the command line

Use the command line to serve as a scheduled job, with no Python.
[docs/RUNNER.md](../RUNNER.md#flags-and-the-toml-keys-they-override) lists
every flag.

1. **Give `--config` the TOML file whose specs the state was saved with.**
2. **Give `--load-state` the state file.** The run refuses a state saved
   from other specs.
3. **Give `--predict`.** The run predicts every row against the loaded
   state and learns nothing. It drops the TOML file's `save_state` and
   `closed_groups`, so one TOML file serves both the learning run and the
   serving run. It refuses a `--save-state` or a `--closed-groups` given
   beside it, since the run has nothing new to save.
4. **Give `--input` the rows to predict,** in place of the TOML file's
   input.
5. **Give `--output` a file of its own,** or the run replaces the output
   the TOML file names.

Here `run.state` holds what `job.toml`'s spec learned from
`ticks.parquet`, and the run serves `today.parquet`:

```sh
polars-online --config job.toml --load-state run.state --predict --input today.parquet --output today-served.parquet
```

```text
wrote 400 rows (1 chunk) to today-served.parquet
```

## Check that it worked

**Row *i* of the output carries what `fit_predict` would have reported had
it been the next row of its group's stream.** Its fields match
`fit_predict`'s, field for field: `pred`, `weight_sum`, `sigma`, `zscore`,
the slot `emit_selected` chooses and the metrics
([docs/DIAGNOSTICS.md](../DIAGNOSTICS.md)).
[OUTPUTS.md](../OUTPUTS.md) lists every field. Two fields differ. The
drift detector `emit_drift` adds never fires. And `coef` is filled only on
each group's last row in each chunk that is not
[skipped](../GLOSSARY.md#the-model-bank-and-what-it-reads), since the same
coefficients predict every row. The state never moves, so
the same rows are predicted the same way twice, on any thread count.

**Compare the first row served with one unbroken run that learned the old
rows and then predicted the new.** The first new row is the one row both
predict from the same state. The rows after it differ, since one run has
learned the rows before them and the served rows have not. This code
compares the two. It uses `df` and `today` from [Example
data](../../README.md#example-data), and `served` from [Serve in
Python](#serve-in-python):

```python
one_run = po.ModelBank([spec]).fit_predict(pl.concat([df, today]))   # df and today as one stream
first = served["ridge"].struct.field("pred_y")[0]
print(f"today's first row, served: {first:.6f}; one run: {one_run['ridge'].struct.field('pred_y')[400]:.6f}")
```

```text
today's first row, served: 1.603543; one run: 1.603543
```

## How each kind of row is served

**`predict` reads every row against the state as it stands, and some
kinds of row in a way of its own:**

| the row | `predict` |
|---|---|
| its target column | may be absent; `resid` is then null |
| its weight column | is not read |
| of a group the bank has never seen | is predicted as null |
| before the last clock its group learned | is predicted against the state as it stands, with a clock step of 0: never refused, never a reset |
| where `session_gap="reset"` or `group_close="session"` would reset its group | is predicted as null, with `weight_sum` 0 |
| after rows an `embargo` still holds | sees the state as it stands, since `predict` releases nothing. A held row whose delay has passed by the served row's clock stays held until the next `fit_predict` learns it. So `weight_sum` and `learned_clock` are the state's on every row |
