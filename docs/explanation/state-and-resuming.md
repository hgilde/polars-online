<!-- kind: explanation -->
# A bank's state and resuming it

This page is for anyone who saves a [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) and loads it again
to go on, and wants to know what they can rely on when they do. A model
bank fits a set of models over the same rows, one for each
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the description
of a model) and each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads)
(the rows that share one value of the column `group=` names).

**A model bank's [state](../GLOSSARY.md#saving-and-serving) is everything
it has learned, and everything it still holds: the rows it cannot learn
yet, and the rows of [closed groups](../GLOSSARY.md#saving-and-serving)
not yet read.** A resumed bank goes on exactly as one run would have, on
the build and operating system that saved it. The state can be saved to a
state file and loaded back to resume or serve, or exported as JSON to read
elsewhere:

| use | the call | what the bank guarantees |
|---|---|---|
| [resume](../GLOSSARY.md#saving-and-serving) | `po.ModelBank.load(path)`, or `load_state=` on a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) (a Polars `LazyFrame`, a chain of steps that runs only when you ask for its result) | the next row is learned as one run would learn it |
| [serve](../GLOSSARY.md#saving-and-serving) | `bank.predict(frame)`, or a query's `online.predict(path)` | each row is predicted from the state as it stands, and the state never moves ([How each kind of row is served](../how-to/serve-from-a-saved-state.md#how-each-kind-of-row-is-served) gives how that compares with `fit_predict`) |
| read elsewhere | `bank.write_json()` | everything `save` writes, as JSON |

[Resume from a saved state](../how-to/resume-from-a-saved-state.md) gives
the steps, from Python and from the [command
line](../GLOSSARY.md#saving-and-serving) (the `polars-online` program), and
[STATE-WORKFLOW.md](../STATE-WORKFLOW.md) says what each step guarantees.

## A save is a chunk's end

**Saving and loading a state is the same, to the models, as reaching the
end of a chunk, because the state holds exactly what the bank carries from
one chunk to the next.** A bank
takes its stream in [chunks](../GLOSSARY.md#the-model-bank-and-what-it-reads),
the rows it takes in at once, and no model can tell where one chunk ends
([Chunk invariance and streaming](chunk-invariance-and-streaming.md)). The
state is everything the bank carries from one chunk to the next, and
nothing it can rebuild:

| it holds | so the next run |
|---|---|
| each model's state (its [running sums](../GLOSSARY.md#fitting-a-model) in most models, with a hard window's snapshots and the rows a refit holds) and each group's clock, for every spec and group, and the memory of each [diagnostic](../GLOSSARY.md#fitting-a-model) | goes on at the next row as one run would |
| the rows still waiting out an [`embargo`](../GLOSSARY.md#time) (a delay before a row is learned) or a `label_clock` (each row's own time its target is known) | learns each one when its target is known |
| a [formula target](../GLOSSARY.md#fitting-a-model)'s open windows, and the rows they hold | learns each row once its window closes and its embargo has passed |
| the rows of [closed groups](../GLOSSARY.md#saving-and-serving) not yet read | still returns them from `closed_groups()` |

So a run resumed from a saved state gives the predictions of one run over
the whole stream, to the bit: every number is the same 64-bit value.
This code feeds `df` to a bank in four
chunks, and saves the bank after each chunk and loads it back before the
next. An `embargo` holds the last five rows of each chunk across the
save. It compares the predictions, and the state at the end,
with one run over every row. It reads `df` from the README's [Example
data](../../README.md#example-data):

```python
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0,
                       embargo=5.0)   # each row learned 5 units of t after it is predicted
whole = po.ModelBank([spec])
one_run = whole.fit_predict(df)["ridge"].struct.field("pred_y")

# The same rows in four chunks, with the bank saved and loaded back after each.
state, parts = None, []
for chunk in df.iter_slices(100):
    bank = po.ModelBank([spec]) if state is None else po.ModelBank.load_bytes(state, specs=[spec])
    parts.append(bank.fit_predict(chunk)["ridge"].struct.field("pred_y"))
    state = bank.save_bytes()   # the state as bytes, the same bytes bank.save would write to a file
print(f"largest difference from one run: {(one_run - pl.concat(parts)).abs().max()}")
print(f"the same state as one run's: {state == whole.save_bytes()}")
```

```text
largest difference from one run: 0.0
the same state as one run's: True
```

## One state file, however the bank is run

**Each way of running a bank saves and loads the same state file, written
whole or not at all.** A bank object, a query and the command line write the
same bytes for the same rows. Each writes the file to a temporary file
beside its destination and then renames it over the destination, so an
interrupted save leaves the old file whole. Each writes at its own moment:

| run by | saves with | loads with | the file is written |
|---|---|---|---|
| a bank object | `bank.save(path)` | `po.ModelBank.load` | when `bank.save` is called |
| a query | `save_state=` | `load_state=` | when the bank has been fed its last row: the input's end, or the last of the rows a `head(n)` asked for |
| the command line | `--save-state` | `--load-state` | only after it has finished writing its output file |

This code saves the same state from a bank object and from a query, and
compares their bytes. It uses `df` and `ticks.parquet` from the README's
[Example data](../../README.md#example-data):

```python
from pathlib import Path

# The same state, saved from a bank and from a query.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)
bank = po.ModelBank([spec])
bank.fit_predict(df)
bank.save("from_bank.state")                             # from a bank object
(pl.scan_parquet("ticks.parquet")                        # the same rows, from a query
   .online.fit_predict([spec], save_state="from_query.state")
   .sink_parquet("fitted.parquet"))
print(f"the same bytes: {Path('from_bank.state').read_bytes() == Path('from_query.state').read_bytes()}")
blob = bank.save_bytes()                                 # the same file as bytes, for a store that is not a file
print(f"as bytes too: {blob == Path('from_bank.state').read_bytes()}")
```

```text
the same bytes: True
as bytes too: True
```

## How large a state grows

**A state keeps running sums in place of the rows it has learned, so its
size does not grow with the stream's length, apart from closed groups'
rows that are never read.** It grows with the specs, their settings and
the number of groups, and with the rows it holds until they can be
learned. It also holds the rows of each closed group until
`closed_groups()` reads them, so read them as the stream runs to keep the
state bounded. A few models keep some rows by design, such as the
first rows a [refit](../GLOSSARY.md#fitting-a-model) holds until it runs.
The README's
[Memory](../../README.md#memory) measures it at 10,000 rows and at
1,000,000.

## Why a state loads on every operating system

**The file stores each floating-point number as the 64-bit float the
model held, bit for bit, in MessagePack, a binary format with one byte order on every
machine.** So a state saved on macOS reads back on Windows or Linux as the
same numbers, and a state loaded and saved again is the same bytes on any
of them.

**A state goes on to the bit on the operating system that saved it, and
on another within `1e-12 × (1 + |value|)`.** A model computes its
decay, and in some models a likelihood or a link function, with the
platform's math library, whose last bit can differ between macOS, Windows
and Linux. The state carries the same numbers across, so a run saved on
macOS and continued on Linux gives rows macOS computed before the save and
rows Linux computed after it. Each release checks the hand-off: it saves a
bank holding every kind of model on macOS, and loads it on Windows and on
Linux. Each feeds the rest of the stream, and its output is held to a run
of the whole stream on that machine that was never saved. Each number
may differ by at most `1e-12 × (1 + |value|)`.

## Why a state loads in later versions

**A state file names the version of its format, and a build loads every
version from 1.0's to its own, apart from two parts whose formats may
change in a minor release.** So from 1.0, a state a 1.x build saved loads
in every later 1.x build. The two parts outside that promise are the
written form of a formula target (its expression) inside a bank's state,
and a
[window run](../GLOSSARY.md#fitting-a-model)'s state ([Saving and resuming
a window run](../WINDOWS.md#saving-and-resuming-a-window-run)). A state
saved by 0.13.0 or an earlier release does not load: refit the bank from
its input. A build refuses a state whose version it cannot read, such as
one saved in a newer format, before it reads a row.

**A state of an earlier version loads, and goes on under the rules the
[CHANGELOG](../../CHANGELOG.md) gives for it.** Where a release corrected
how a model computes, a state saved before it goes on as the CHANGELOG's
entry for that change says. Most keep the old rule and go on to the bit,
and others take the new rule or change behavior from a stated row. Each
entry says which, and by how much the numbers may move.

## The one gap: a failure after the bank

**A query writes its state only once the bank has been fed its last row,
and a failure after the bank may not stop that write.** A run you abandon
partway, such as one whose chunks you stop reading, leaves the
`save_state` file as it was, and so does a run that fails inside the
bank. Polars does not end a run you abandon: it leaves it suspended, so
only the bank's last row (the input's end, or the last row a `head(n)`
asked for) can trigger the write. But Polars 1.x runs the bank to the
end of its input before it reports an error from a step after it, so the
bank reaches its last row and writes. A full disk while `sink_parquet`
writes the query's result to a file, or a bad cast, then leaves
`save_state` written with the whole stream's state, while the query's own
output is incomplete or missing.

**A rerun from that state cannot recover the lost output.** With
`load_state=p, save_state=p`, a rerun of the same input begins with rows
the state has already learned. For a spec with a clock column, every
group with rows before its last clock [steps
back](../GLOSSARY.md#time), and the bank refuses the
rerun. Only a `restart_after_step_back` set below the step makes it a
reset, which loses what each group's model learned. `bank.skip_learned`,
which drops the rows a state has learned, keeps such a rerun from
learning them twice, but it also drops the rows whose output was lost, so
that output is never written. A spec without a clock learns every row of
the rerun a second time.

**On Polars 2.0 the gap was measured narrower, on its release candidate,
and neither outcome is promised.** Polars 2.0.0rc1 stopped a long enough
stream before its end, and the state was not written. It was measured
with chunks of 500 rows and a cast that fails after the bank: Polars
1.44.1 wrote the state at 4,000 rows and at 40,000, and 2.0.0rc1 at 4,000
only. How Polars stops a failing query is Polars' to change.

**Where the state must land only with the output, use the command line.**
It saves the state only after it has finished writing its output file.

**Otherwise, write a dated `save_state` file for each day's input**, such
as `ridge-2026-01-02.state`. Then a failed run does not overwrite the state
its rerun must load, and the files keep an audit trail. [Resume from a saved
state](../how-to/resume-from-a-saved-state.md#resume-in-a-query) shows the
steps.
