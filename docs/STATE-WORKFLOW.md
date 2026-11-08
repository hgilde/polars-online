<!-- kind: reference -->
# Saved states: what each step guarantees

This reference page is for anyone who relies on a saved state across
runs: it gives what each step of saving, serving and resuming a model
bank's state guarantees, and the fact about Polars that each rule rests
on. A [model bank](GLOSSARY.md#the-model-bank-and-what-it-reads) fits a
set of models over the same rows, and its
[state](GLOSSARY.md#saving-and-serving) is everything it has learned. The
state can be written to a file and loaded back to inspect it, to
[serve](GLOSSARY.md#saving-and-serving) new rows from it, or to learn on
from it. A [window run](GLOSSARY.md#fitting-a-model), a run of
`po.stream.with_windows`, keeps a state of its own, which resumes by rules
of its own.

The steps themselves, and the reasons behind them, have pages of their
own:

| to | read |
|---|---|
| fit a bank, save it, and serve new rows from it in a new process | [Fit, save and serve a model bank](tutorials/fit-save-and-serve.md) |
| save a state from a query, a bank object or the command line | [Save a state](how-to/save-a-state.md) |
| learn on from a state, on input that overlaps it too | [Resume from a saved state](how-to/resume-from-a-saved-state.md#resume-on-input-that-overlaps-the-state) |
| know what a state holds, why a resumed run matches one that never stopped, and the one gap | [A bank's state and resuming it](explanation/state-and-resuming.md) |

## The workflow in four steps

A state file is the same file, byte for byte, whichever wrote it: a query,
the `polars-online` command line and a `ModelBank` fed the same chunks
write identical bytes.
A query and an eager frame take it as `load_state` and `save_state`, and
the TOML under the same two keys. The command line spells them `--load-state`
and `--save-state`, and a `ModelBank` has `load` and `save`.

### Each step, and what it guarantees

Each step has a call in Python and a flag or key on the command line.
Each guarantee rests on a measured fact about how Polars runs a query
([The rules, and the evidence for each](#the-rules-and-the-evidence-for-each)):

| step | in Python | on the command line | what it guarantees |
|---|---|---|---|
| **1. Fit, and keep the state** | `lf.online.fit_predict([spec], save_state="ridge.state")` | `save_state` in the TOML, or `--save-state` | the state is written whole, once the bank has been fed its last row, and two runs of the query write the same bytes |
| **2. Inspect it** | Loading the file gives the `ModelBank` back: `po.ModelBank.load(path)`, with `coef()` (the coefficients), `last_row()`, `summary()` and `describe()` (what it was fed), `gram()` and the rest. `save_bytes()` / `load_bytes()` are the same state as bytes, for a store that is not a file | | none of them needs a row of data |
| **3. Serve from it, learning nothing** | `lf.online.predict("ridge.state")` | `polars-online --load-state p --predict`, which refuses `--save-state` | every row is predicted from the state as it stands, and the state never moves, so the same rows get the same predictions twice, on any thread count |
| **4. Learn on from it** | `lf.online.fit_predict(load_state="ridge.state", save_state="ridge.state")` | `polars-online --load-state p --save-state p` | it resumes at the next row and replaces the file. The specs come from the file, and a `load_state` that is not a bank this build can read, or whose specs disagree with the ones passed, is a `ValueError` before any row is read. Input that overlaps the state is refused by default; [`skip_learned`](how-to/resume-from-a-saved-state.md#resume-on-input-that-overlaps-the-state) drops the overlap |

### What a saved bank carries

**Everything the next row needs, and nothing it can rebuild.** A state
file holds every spec, and for every spec and group:

| it holds | so the next run |
|---|---|
| the models' state, and the stream's clock and diagnostics | goes on at the next row as one run would |
| the rows still waiting out an `embargo` or their `label_clock` time | learns each one when its target is known |
| a [formula target](GLOSSARY.md#fitting-a-model)'s open windows, and the rows they hold | resumes with the windows open, and learns each row once its window closes and its embargo has passed |
| the rows of closed groups not yet read | still returns them from `closed_groups()` |

The recipe saves two specs part-way through a stream. One learns each row
50 clock units late, under an `embargo`, so its state holds the last 50
rows unlearned. The other learns a target that looks 20 units ahead, so
its state holds open windows. Each resumed run gives one run's
predictions:

```python
import numpy as np
import polars as pl

import polars_online as po

# The planted stream, cut into two files at row 1,500.
rng = np.random.default_rng(3)
x0, x1, noise = rng.standard_normal((3, 2_000))
stream = pl.DataFrame({"t": np.arange(2_000.0), "x0": x0, "x1": x1,
                       "y": 0.5 + 1.5 * x0 - 2.0 * x1 + 0.1 * noise})
stream.head(1_500).write_parquet("2025.parquet")
stream.tail(500).write_parquet("2026.parquet")
# Two specs whose state holds rows: one learns late, one looks ahead.
clock = dict(clock="t", gap_cap=300.0, half_life=600.0)
late = po.spec.ewridge("late", targets=["y"], features=["x0", "x1"], embargo=50.0, **clock)     # each row learned 50 units later
y_ahead = po.ewm_mean("y", half_life=10.0, window_size=20.0, reverse=True).alias("y_ahead")                   # the next 20 units' mean of y
window = po.spec.ewridge("window", targets=[y_ahead], features=["x0", "x1"], embargo=20.0, **clock)
specs = [late, window]

# Save part-way, resume on the second file, and compare with one run.
pl.scan_parquet("2025.parquet").online.fit_predict(specs, save_state="carry.state").sink_parquet("2025_scored.parquet")
resumed = pl.scan_parquet("2026.parquet").online.fit_predict(load_state="carry.state").collect()
one_run = po.ModelBank(specs).fit_predict(stream).tail(500)
for name, target in [("late", "y"), ("window", "y_ahead")]:
    difference = (resumed[name].struct.field(f"pred_{target}") - one_run[name].struct.field(f"pred_{target}")).abs().max()
    print(f"{name}: largest difference from one run {difference}")
```

```text
late: largest difference from one run 0.0
window: largest difference from one run 0.0
```

**The ways of running a bank differ in one case: a fit that keeps only the
state, under a formula target.** `lf.online.fit_predict(save_state=...)` keeps
predictions, so it refuses an embargo shorter than the target's window.
`ModelBank.fit` and `polars-online --no-output` keep none, so they take any
embargo.

**A build refuses a bank file whose schema it does not read.** From 1.0,
every later 1.x build loads a 1.x file, apart from a formula target's
written form and a window run's state, whose formats may change in a minor
release. A 0.x file is refused: refit the
bank from its input ([Why a state loads in later
versions](explanation/state-and-resuming.md#why-a-state-loads-in-later-versions)).
A file this build saved goes on as one run to the bit, as the recipes
show. A state of a model under a hard window, saved by 1.2.0 or earlier,
loads, and its first read is the one it saved. Once a snapshot taken after
the load is the hard window's edge, the outputs it reads from the hard
window continue within 1e-12, not to the bit, because such a state never
recorded the per-row decay factors. Other earlier states continue as the
[CHANGELOG](../CHANGELOG.md)'s entry for each change says, some within a
stated tolerance rather than to the bit.

## A window run's state

**A window run holds rows a bank never does, so it resumes by rules of its
own.** `po.stream.with_windows(..., save_state=)` writes the windows' sums
and the rows still waiting for their windows to close, and a run given
`load_state=` returns those rows first. Under a slice (`head(n)`) the run
reads past the last row it returns, so the state also records how many rows
of the input it consumed, and the input's first row and last rows read.
[Resume a window run](how-to/resume-a-window-run.md) gives the steps, with
a recipe for resuming on the next file, resuming after a slice, and a
refusal. Here is what each rule guarantees, and the test that holds it:

| the input a resumed run is given | the run | held by |
|---|---|---|
| the next file, after an unsliced run | goes on, and first returns the rows the state held | `test_a_save_and_load_at_every_row_is_one_run`, `test_a_state_saved_under_a_slice_skips_nothing_of_another_input` |
| an input whose first row is at the last stamp read, after an unsliced run, and starts no new session | refused by name, as after a sliced run: it may repeat rows the state read, which would come out twice | `test_an_unsliced_state_refuses_an_input_that_repeats_its_last_row`, `an_unsliced_state_refuses_an_input_at_the_last_stamp_it_read` |
| the same input, after a sliced run | skips the rows consumed, so any chain of sliced runs gives one run's output, at chunks of 1, 7 and 100,000 rows | `test_a_state_saved_under_a_slice_resumes_on_the_same_input`, `test_a_state_saved_under_a_slice_resumes_alike_at_any_chunk_size` |
| the next file, after a sliced run, its first row a step forward or a new start by the clock policy | skips nothing, the rows a previous file left held included | `test_a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held`, `test_a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file` |
| an input whose first row is at the last stamp read, or steps back where the policy refuses it | refused by name: `another input` | `test_a_next_file_starts_after_the_last_stamp_the_state_read`, `test_a_state_saved_under_a_slice_refuses_another_input` |
| another input that starts as the saved one did, or ends before the rows consumed | refused by name, by the rows the state read where it was cut, with and without a clock column | `test_a_state_saved_under_a_slice_refuses_another_input`, `test_a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row` |
| without a clock column, an input that does not begin with a new session | refused: a row-count clock steps forward at every row | `test_without_a_clock_the_next_file_begins_with_a_new_session` |
| a state of another version, or a damaged file | refused by its version, or reported as damaged | `test_a_windows_state_of_another_version_is_refused_by_its_version`, `test_a_damaged_windows_state_says_so` |

The windows state is version 10, and this build reads 9 too, whose layout
is 10's without a quantile operator. A bank file carries a windows state
for each formula target, and a Rust test pairs the two version numbers, so
a change to the windows state cannot pass the bank unseen. A state
resumes only the call that saved it (`test_a_state_resumes_only_its_own_call`).
`with_windows`' docstring states two limits. A clock that starts over at
the same stamp each day needs a session column. And an input whose rows
match the state's where it was cut is taken as the same input.

## A `refresh_time` run's state

**`po.stream.refresh_time` resumes as a bank does: on the input after the
last tick it read.** Under a slice it reads up to the tick that completed
the last point returned, so its state is the same whatever the chunk size.
With `pairs=True`, one tick can complete several pairs' points. A slice
that ends among them keeps the rest in its state, and the run resumed after
that tick returns them first, so the two runs give the one run's grid. A
row before a group's saved clock is refused as any step back is.

## The rules, and the evidence for each

Each guarantee of a saved state rests on a fact about how Polars runs a
Python source: the function a query reads its rows from, which is how
`lf.online.fit_predict` puts the bank in a query. Nothing in Polars'
documentation states these facts. `scripts/io_source_semantics.py`
measures each one with a source that logs its own runs, on py-polars
1.34.0, 1.38.1 and 1.44.1. Each guarantee was then checked on a bank of 3
groups of 1,000 rows, read in chunks of 250, and `tests/test_frame.py`
holds the checks:

| what is guaranteed | the fact about Polars it rests on | how it was checked |
|---|---|---|
| `save_state` is written when the bank has been fed its last row: the input's end, or the rows a `head(n)` asked for. Never on a run the caller abandoned, or one the bank ended with an error. The write is atomic | a run the caller abandons, by stopping the loop that reads the result one chunk at a time (`collect_batches`), is left suspended rather than ended, so only the source's own end can trigger the write | stopping that loop wrote no file, nor did a bank error at row 700 |
| every run of a query ends in the same state, so two runs write the same bytes. A query that uses the bank twice runs it twice when it writes a state, and with nothing to write Polars may run it once | a query that reads another query twice, as `lf.join(lf)`, `pl.concat([lf, lf])` or `pl.collect_all` over two `sink_parquet` calls (each runs the query and writes its result to a file) do, runs its source twice, at the same time on two threads | each of those three wrote the file twice, on two threads, and the file was byte-identical to the one a `ModelBank` fed the same rows saves |
| `load_state` and `predict(path)` are read when the query is built, so a query run twice gives the same result | the two runs above overlap, so a state read at run time could be read while the other run writes it | a query built against a state file, run again after the file was overwritten, gave the same frame |
| the bank is fed exactly the rows the query pulled | a `head(n)` reaches the source as a row count, and the source stops there; a `sort().head(n)` does not, so the bank reads every row | `head(613)` wrote the state of a bank fed the first 613 rows, byte for byte, and `sort().head(5)` the whole stream's |
| `predict` refuses `save_state`, as `polars-online --predict` does, since it learns nothing | | `predict(save_state=...)` is refused |
| a failure after the bank still writes the state on py-polars 1.x, and on 2.0 only for a stream short enough to finish first: [the one gap](explanation/state-and-resuming.md#the-one-gap-a-failure-after-the-bank) | Polars runs a source to the end of its input before it reports the error of a step after it | a cast that failed after the bank left the file, byte-identical to the whole stream's state |
| building a query writes nothing | building or inspecting a query, by `collect_schema()` or `explain()`, runs no source | a source's run count stayed at 0 |

**Building a query writes nothing, and every run of it ends in the same
state.** The file is written when the query runs, and `load_state` is read
when the query is built. So a query run twice writes the same bytes twice,
and a run under `head(n)` writes the state after the first `n` rows:

```python
from pathlib import Path

import numpy as np
import polars as pl

import polars_online as po

# The planted stream, saved to one file.
rng = np.random.default_rng(3)
x0, x1, noise = rng.standard_normal((3, 2_000))
pl.DataFrame({"t": np.arange(2_000.0), "x0": x0, "x1": x1,
              "y": 0.5 + 1.5 * x0 - 2.0 * x1 + 0.1 * noise}).write_parquet("stream.parquet")
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=600.0, gap_cap=300.0)

# A query that saves its state, built but not yet run.
query = pl.scan_parquet("stream.parquet").online.fit_predict([spec], save_state="run.state")
print(f"after building the query, the file exists: {Path('run.state').exists()}")
# Run it twice, then under head(100), reading the file after each.
query.collect()
first = Path("run.state").read_bytes()
query.collect()
print(f"two runs, the same bytes: {Path('run.state').read_bytes() == first}")
query.head(100).collect()
print(f"under head(100), the state has been fed {po.ModelBank.load('run.state').rows_fed} rows")
```

```text
after building the query, the file exists: False
two runs, the same bytes: True
under head(100), the state has been fed 100 rows
```
