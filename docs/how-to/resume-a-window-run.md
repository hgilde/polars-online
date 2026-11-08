<!-- kind: how-to -->
# Resume a window run

This how-to is for anyone who adds window columns with
`po.stream.with_windows` to a stream that arrives in parts, such as a file
a day, and wants the parts to give what one run over the whole stream
gives. You save the run's state at the end of one part, and load it at the
start of the next.

## Prerequisites

These steps need the library, a stream with a clock, and the call that
makes the windows:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `trades`, a stream of quotes with trades between them, saved as
  `trades.parquet`. Its `mid` is on every row, and `side`, `quantity` and
  `price` are null on a quote.
- A `with_windows` call with its formulas and clock keywords, as in [Make
  features that look back over the stream](make-window-features.md). The
  run that loads a state must have the same formulas, clock keywords and
  kind of [clock](../GLOSSARY.md#time) as the run that saved it, or
  `load_state=` raises `ValueError`.
- The stream cut into parts between stamps (clock values): a part that
  starts at the last stamp of the part before it is refused, as
  [Check the result](#check-the-result) says.
- The same polars-online release for every part. The format of a window
  run's state is not promised, so a state one release saved need not load
  in the next, and a build refuses a state it cannot read, naming its
  version.

A window target, which a spec holds in its `targets`, is saved with its
bank's state instead ([A bank's state and resuming
it](../explanation/state-and-resuming.md)). [How a window
operator weighs the rows near each row](../WINDOWS.md#saving-and-resuming-a-window-run)
says what a window run's state holds.

## Resume on the next file

Use this when each part of the stream is a file of its own.

1. **Give the first part's run `save_state=` a file name.** The run returns
   every row whose windows have closed, and saves the rest in the state,
   with the windows' sums.
2. **Give the next part's run `load_state=` the same file.** It returns the
   rows the state held before its own.
3. **To go on to a third part, give the second run `save_state=` too.**

This code cuts `trades` into two files at row 2,000, and compares the two
runs with one run over the whole stream. It uses `trades` from [Example
data](../../README.md#example-data):

```python
# The forward VWAP in one run over the whole stream, for comparison.
next_minute = dict(half_life="10s", window_size="1m", reverse=True)
vwap = dict(fwd_vwap=po.ewm_sum(pl.col("price") * pl.col("quantity"), **next_minute) / po.ewm_sum("quantity", **next_minute))
clock = dict(clock="ts", gap_cap="5m", group="symbol")
one_run = po.stream.with_windows(trades, **vwap, **clock)

# The same stream as two files.
trades.head(2000).write_parquet("day1.parquet")
trades.slice(2000).write_parquet("day2.parquet")
first = pl.scan_parquet("day1.parquet").online.with_windows(**vwap, **clock, save_state="w.state").collect()   # 1. its last rows wait
second = pl.scan_parquet("day2.parquet").online.with_windows(**vwap, **clock, load_state="w.state").collect()  # 2. they come out first
print(f"two files: {first.height} rows, then {second.height}; one run's output: {pl.concat([first, second]).equals(one_run)}")
```

```text
two files: 1884 rows, then 1116; one run's output: True
```

The first run returns 1,884 of its 2,000 rows. The other 116 wait in the
state for the next minute of trades, and the second run returns them first.

## Resume a run stopped by a slice

Use this when a run reads only the first rows of its input, such as under
`head(500)`, and a later run goes on through the same input.

1. **Run the query with `save_state=` and the slice.** It returns the rows
   asked for, and the state records how far into the input it read.
2. **Run the same input, unsliced, with `load_state=`.** It skips the rows
   the state consumed and goes on.

The sliced run reads the input only until the windows of the last row
asked for have closed. Its state records how many input rows it consumed,
the input's first row (clock and session) and the last rows it read, which
is how a resumed run tells the same input from another.

This code stops a run at 500 rows and resumes it on the same file. It uses
`trades.parquet` from [Example data](../../README.md#example-data), and
`vwap`, `clock` and `one_run` from the recipe before it:

```python
head = pl.scan_parquet("trades.parquet").online.with_windows(**vwap, **clock, save_state="h.state").head(500).collect()  # 1
rest = pl.scan_parquet("trades.parquet").online.with_windows(**vwap, **clock, load_state="h.state").collect()            # 2. it skips the rows read
print(f"head(500), then the rest: {head.height} rows, then {rest.height}; one run's output: {pl.concat([head, rest]).equals(one_run)}")
```

```text
head(500), then the rest: 500 rows, then 2500; one run's output: True
```

**If the clock starts over at the same stamp each day, add a session
column, such as the date, and name it in `session=`.** Otherwise the next
day's file starts as the saved input did, and the resumed run takes it
for the saved input. **If the
stream's rows can repeat, as a daily grid's can, resume only on the same
input:** another input that matches the state's rows where it was cut is
taken for the same input.

## Check the result

**Compare the runs' output, joined with `pl.concat`, with one run over the
whole stream, as each recipe prints.** Two runs give exactly what one run
gives.

**Any resumed run refuses an input whose first row is at the last stamp
the state read, unless that row starts a new session.** A file boundary
inside a stamp that several rows share looks the same as an input that
repeats rows the state read, which would come out twice. So cut the
stream into files between stamps. Without a clock column there is no stamp
to compare, so a repeated row is not caught.

**A run resumed from a state saved under a slice decides from its input's
first row whether to skip rows, go on or refuse, so it refuses an input it
can tell is wrong.** A refusal names `another input`. It arrives as
`polars.exceptions.ComputeError` under py-polars 1.x and as `ValueError`
under 2.0:

| the input a run resumed from a state saved under a slice is given | the run | what to do |
|---|---|---|
| the input the state was saved from, unsliced | skips the rows the state consumed, then goes on | |
| the next file, whose first row is a step forward on the clock or a new start (a [step back](../GLOSSARY.md#time) larger than `restart_after_step_back`, or a new session) | skips nothing, and returns the rows the state held before its own | |
| an input whose first row is a step back that `restart_after_step_back` does not take as a reset, such as the same input sliced by hand, or an overlapping file | refused | start the file after the last row the state read |
| without a clock column, an input that does not begin with a new session | refused: a row-count clock steps forward at every row | give the call a clock column, or start each file with a new session |
| an input that starts as the saved one did but differs where the state was cut, or ends before the rows consumed | refused | resume on the input the state was saved from |

This code resumes the sliced run's state on a file that starts
before the last row it read. It uses `trades` from [Example
data](../../README.md#example-data), and `vwap` and `clock` from the first
recipe:

```python
trades.slice(300).write_parquet("overlap.parquet")                   # starts before the last row the state read
try:
    pl.scan_parquet("overlap.parquet").online.with_windows(**vwap, **clock, load_state="h.state").collect()
except (ValueError, pl.exceptions.ComputeError) as e:                # ComputeError under py-polars 1.x
    message = str(e)[str(e).index("with_windows"):]
    print(message.split(". ")[0])
```

```text
with_windows: the state was saved from a slice of its input, and this is another input: it starts before the last row the state read
```

[A window run's state](../STATE-WORKFLOW.md#a-window-runs-state) lists what
each rule guarantees, and the test that holds it.
