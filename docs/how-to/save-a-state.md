<!-- kind: how-to -->
# Save a state

This how-to is for anyone who wants to keep what a model bank has learned
in a file, to go on learning from it or serve new rows, with a bank run as
a Python object, in a query or from the command line.

## Prerequisites

These steps need the library and rows to learn from:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `lf` and `later`, [queries](../GLOSSARY.md#the-model-bank-and-what-it-reads)
  over the files `ticks.parquet` and `today.parquet`, whose rows follow
  one another on the clock `t`.
- For the command line: the `polars-online` program, whose binaries are
  attached to each GitHub release, and a TOML file of its settings.
  [Run a bank from the command line](run-a-bank-from-the-command-line.md)
  shows both.

## Pick the calls for the way the bank runs

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a
set of models over the same rows, and its
[state](../GLOSSARY.md#saving-and-serving) is everything it has learned,
for every [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) and
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads). Each way of
running a bank saves and loads the same state file, written whole or not
at all, and the same to the byte whichever way wrote it:

| run by | saves with | loads with |
|---|---|---|
| a bank object | `bank.save(path)`, or `bank.save_bytes()` for a store that is not a file | `po.ModelBank.load(path)`, or `po.ModelBank.load_bytes(blob)` |
| a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) (a Polars `LazyFrame`, which runs only when you ask for its result) | `save_state=` on `online.fit_predict` | `load_state=` on `online.fit_predict`, or the path given to `online.predict` |
| the command line | `--save-state` | `--load-state` |

[One state file, however the bank is
run](../explanation/state-and-resuming.md#one-state-file-however-the-bank-is-run)
says when each writes the file, and shows a bank and a query write the
same bytes.

## Save from a query, and load the file back

On a query, `load_state=` and `save_state=` may name one file, because the
state is read when the query is built and written whole or not at all.

1. **Name the file in `save_state=` on `online.fit_predict`.** The state
   is written once the bank has been fed its last row.
2. **Run the query,** here with `sink_parquet`, which runs it and writes
   the result to a file without holding it all in memory.
3. **Serve from the file with `online.predict(path)`,** or continue from
   it with `load_state=`, naming the same file in `save_state=` to save
   again.
4. **Load the file into a bank object with `po.ModelBank.load`,** or its
   bytes with `po.ModelBank.load_bytes`, which reads them as it reads a
   file.

This code saves a state from a query and uses it in each of those ways. It
uses `lf` and `later` from [Example data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
lf.online.fit_predict([spec], save_state="bank.state").sink_parquet("fitted.parquet")             # 1 and 2: fit, then save at the last row
served = later.online.predict("bank.state").collect()                                             # 3: serve from the file
later.online.fit_predict(load_state="bank.state", save_state="bank.state").sink_parquet("more.parquet")  # 3: the next rows: continue, save again
bank = po.ModelBank.load_bytes(po.ModelBank.load("bank.state").save_bytes(), specs=[spec])         # 4: bytes load as a file does
```

To load a saved state and go on learning, from Python or the command line,
follow [Resume from a saved state](resume-from-a-saved-state.md). It
covers input that overlaps what the state has learned, and the errors a
load raises. To predict rows without learning from them, follow [Serve
from a saved state](serve-from-a-saved-state.md).

## Give each day's input its own state file

**A query abandoned before its last row, or ended by an error inside the
bank, leaves the `save_state` file as it was.** An error in a step after
the bank, such as a full disk under `sink_parquet`, does not stop the bank
on Polars 1.x. The state is then written although the output is missing
([The one
gap](../explanation/state-and-resuming.md#the-one-gap-a-failure-after-the-bank)
has the measurements).

To write the state only with the output, use the command line, which saves
it after committing the output. In a query, follow these steps:

1. **Name a dated file in `save_state=` for each day's input,** such as
   `bank-2024-01-02.state`.
2. **Load the previous day's file with `load_state=`,** and save the next
   day to a file of its own. A rerun of a failed day then loads the state
   of the day before it, which the failure did not overwrite.

This code runs two days, each with its own state file. It uses `lf` and
`later` from [Example data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
(lf.online.fit_predict([spec], save_state="bank-2024-01-02.state")   # the first day, and a state file of its own
   .sink_parquet("fitted-2024-01-02.parquet"))
(later.online.fit_predict(load_state="bank-2024-01-02.state",        # the next day loads the one before ...
                          save_state="bank-2024-01-03.state")        # ... and writes a file of its own,
   .sink_parquet("fitted-2024-01-03.parquet"))                       # so a rerun of it loads bank-2024-01-02 again
```

## Save from the command line

The command line saves the state only after it has finished writing its
output file, so a failed run never leaves a state without its output.

1. **Give `--save-state` the file to save the state to.**
2. **Give `--load-state` that file on the next run** to go on from it.

[Run a bank from the command line](run-a-bank-from-the-command-line.md)
gives the steps and their output.

## Check that it worked

**Load the file with `po.ModelBank.load`, and read what the bank says it
holds.** The file alone is enough, since it carries its specs. This code
reads `bank-2024-01-02.state` and `bank-2024-01-03.state`, which [Give
each day's input its own state
file](#give-each-days-input-its-own-state-file) saved from `lf` and
`later` of [Example data](../../README.md#example-data):

```python
for day in ["2024-01-02", "2024-01-03"]:
    print(day, po.ModelBank.load(f"bank-{day}.state"))   # the specs, the groups, and the rows fed so far
```

```text
2024-01-02 ModelBank(['ridge'], groups=1, rows_fed=400)
2024-01-03 ModelBank(['ridge'], groups=1, rows_fed=800)
```

The second file has learned both days' 800 rows, and the first file is
unchanged by the second day's run. [Inspect a bank](inspect-a-bank.md) reads the
rest of what a loaded bank holds.

## Which builds load the file

**From 1.0, a state a 1.x build saved loads in every later 1.x
build.** A state saved by 0.13.0 or an earlier release does not load in
the releases after it: refit the bank from its input. A [window
run](../GLOSSARY.md#fitting-a-model), started with `with_windows`, keeps a
state of its own ([Resume a window run](resume-a-window-run.md)). [Why a state loads
in later
versions](../explanation/state-and-resuming.md#why-a-state-loads-in-later-versions)
gives the two parts of a state outside that promise, and the reasons.

**To read a state from a program without this library, export it as
JSON** with `bank.write_json()`, which holds everything `save` writes.
[Read a state without this library](read-a-state-without-this-library.md)
shows how.
