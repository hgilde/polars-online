<!-- kind: how-to -->
# Put rows in clock order

This how-to is for anyone who fits a model bank on a stream with a clock.
It shows how to give the bank each group's rows in clock order, and what to
do when the bank warns about a query's row order or refuses a chunk.

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, and
  [Example data](../EXAMPLE-DATA.md) built. The examples read
  `df`, a frame of 400 rows with a clock `t`, and `lf`, a query over the
  same rows saved as `ticks.parquet`.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) with a
  [clock](../GLOSSARY.md#time): a `clock=` column, and the `gap_cap=` a clock
  requires.

## Sort the rows before the bank

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits a set
of models over the same rows, one for each
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) and
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads). A model predicts
each row from the rows before it, so rows in another order give different
predictions and, for most models, a different fit. The bank checks the
order it is given in two ways. It warns of a step in a
[query](../GLOSSARY.md#the-model-bank-and-what-it-reads) whose row order
Polars does not promise: a join or a `group_by` without `maintain_order`,
a `unique`, or a sort by several keys without `maintain_order=True`. And
it refuses a
[chunk](../GLOSSARY.md#the-model-bank-and-what-it-reads), the rows it takes
in at once, where a group's [clock](../GLOSSARY.md#time) [steps
back](../GLOSSARY.md#time) within a [session](../GLOSSARY.md#time), unless
`restart_after_step_back` is set below the step, which makes the model
[reset](../GLOSSARY.md#time) there instead. [Why you set the row
order](../explanation/chunk-invariance-and-streaming.md#why-you-set-the-row-order)
explains why the order matters.

Put the rows in clock order before the bank, in the query where there is
one, so every chunk the bank takes is in order.

1. **Sort the rows by the clock in the query, before the bank, with
   `maintain_order=True`.** A sort holds every row in memory while it
   sorts, so the stream must fit in memory; otherwise, write the rows in
   clock order where they are made, and check them as in [Check the
   result](#check-the-result). The bank checks each group's rows on their
   own, so groups may take turns row by row, and one sort by the clock
   orders every group. Two rows with the same clock value are zero apart,
   which is not a step back. Their order still changes the predictions,
   since the second is predicted after the first is learned, and Polars
   keeps rows with equal keys in their order only under
   `maintain_order=True`. This code uses `lf` from [Example
   data](../EXAMPLE-DATA.md):

   ```python
   ordered = lf.sort("t", maintain_order=True)              # a query: nothing runs yet
   ```

   For a frame in memory, sort it with `df.sort("t", maintain_order=True)`.
   For a file the command line reads, sort it with Polars before the
   command line reads it, as
   `pl.scan_parquet("in.parquet").sort("t", maintain_order=True).sink_parquet("sorted.parquet")`.
   For chunks fed by a loop of your own, sort the whole stream before you
   cut it into chunks: a sort within each chunk leaves a row that is late
   across two chunks.

2. **Give each query step that can reorder rows its guarantee.** Polars
   runs a query a block of rows at a time, so a join, a `group_by` or a
   `unique` can deliver its rows in a different order from the query's
   whole result, and in a different order on each run. A sort by the clock
   after such a step fixes the order of rows at different clock values,
   but rows that share a clock value keep the order the step gave them.
   Each step, and its guarantee:

   | a query step that may reorder rows | give it |
   |---|---|
   | `join` | `maintain_order="left"` |
   | `group_by` | `maintain_order=True` |
   | `unique` | a sort after it, by the clock and a column that tells rows at one clock value apart, with `maintain_order=True` |
   | a sort | `maintain_order=True` |
   | anything else | a sort by the clock and such a column, with `maintain_order=True`, before the bank |

3. **Run the query, and read any `OrderNotGuaranteedWarning`.** The bank
   reads the query before it runs, and warns of each step whose order
   Polars does not promise, naming the step and its fix. This code
   joins each row's venue to a fee, once as Polars' `join` does by default
   and once with `maintain_order="left"`, and prints what the bank warns of
   each. It uses `lf` from [Example data](../EXAMPLE-DATA.md):

   ```python
   import textwrap
   import warnings

   # A fee per venue, and a spec that reads it.
   pl.DataFrame({"venue": ["X", "Y"], "fee": [0.1, 0.2]}).write_parquet("fees.parquet")
   priced = po.spec.ewridge("priced", targets=["y"], features=["x0", "fee"], half_life=100.0)
   # Join both ways, and print the bank's warning, if any.
   for order in [None, "left"]:
       joined = lf.join(pl.scan_parquet("fees.parquet"), on="venue", **({"maintain_order": order} if order else {}))
       with warnings.catch_warnings(record=True) as caught:
           warnings.simplefilter("always")
           joined.online.fit_predict([priced]).collect()
       said = [str(w.message) for w in caught if issubclass(w.category, po.OrderNotGuaranteedWarning)]
       print(f"maintain_order={order!r}:", textwrap.fill(said[0], 100) if said else "no warning")
   ```

   ```text
   maintain_order=None: lf.online.fit_predict: the query's row order is not guaranteed, and an online model learns in row
   order. Its steps that leave the order open (a join without maintain_order) let Polars give the rows
   in another order when it runs the query, so the output can change from run to run. Fix the order
   with maintain_order="left" on a join, maintain_order=True on a group_by, or a sort after a unique()
   or before the bank, or, where the order is fixed by other means, silence this warning with
   `warnings.simplefilter("ignore", polars_online.OrderNotGuaranteedWarning)`.
   maintain_order='left': no warning
   ```

   The warning never stops a run. It can miss a step: a query that Polars
   cannot write out for the check to read passes in silence. For a query
   whose order you know, silence the warning with
   `warnings.simplefilter("ignore", po.OrderNotGuaranteedWarning)`.

## Check the result

Check that the run issued no `OrderNotGuaranteedWarning` and raised no
error. To check the rows themselves before the bank, count the steps back
within each group and session, with Polars' `diff` over the spec's
`group=` and `session=` columns. This code counts them in `ordered`, the
sorted query of step 1, with `stock_id` as the group and `session` as the
session:

```python
# Steps back within each group and session, summed.
steps_back = ordered.select((pl.col("t").diff().over("stock_id", "session") < 0).sum()).collect().item()
print(f"steps back: {steps_back}")
```

```text
steps back: 0
```

## Fix a chunk the bank refused

**The bank refuses a chunk in which a group's clock steps back within a
session, and learns none of it, unless `restart_after_step_back` is set
below the step.** It checks every group's clock in every
spec over the whole chunk before it learns a row. So a refused chunk
leaves the bank as it was. `predict` learns nothing, so it never refuses a
row for its clock.

**The error names the spec, the clock column, the size of the step back
and the row, counted from 0.** On a temporal clock the step is a duration,
such as `1d`. The row is the row's position in the frame given to one
`fit_predict` call, or in the whole input under `fit_predict_batches`,
`fit`, a query or the command line.

**Feed the corrected rows again in the form you fed them.** For a frame you
feed to `fit_predict` yourself, correct it and feed it again to the same
bank. Under `fit_predict_batches`, `fit`, a query or the command line, the
chunks before the refused one have been learned, so correct the input and
run it again from the start. If the bank was loaded from a [saved
state](../GLOSSARY.md#saving-and-serving), start again from that state:
load it with `po.ModelBank.load` and pass the corrected input through
`bank.skip_learned` (for a query, then run it with `load_state=`), or, on
the command line, give `--load-state` and `--skip-learned`. Otherwise, run
it on a bank that starts with nothing learned. Choose the fix by what caused the step back:

| the step back comes from | do this |
|---|---|
| rows not in clock order | sort the rows by the clock ([step 1](#sort-the-rows-before-the-bank)) |
| input that overlaps what a saved state has learned, such as a rerun of the day | drop the rows the state has learned with `bank.skip_learned(frame)`, or `--skip-learned` on the command line ([Resume from a saved state](resume-from-a-saved-state.md#resume-on-input-that-overlaps-the-state)) |
| the start of a new session, such as the next trading day | set the spec's `session=` to the column whose value changes there, and `session_gap=` to the clock step to apply at the change, or `"reset"`: the clock may step back where a session starts ([Handle sessions, gaps and steps back](handle-sessions-gaps-and-steps-back.md)) |
| a stream that starts over, such as a replayed day or a feed that started again | set `restart_after_step_back` below the step: the model resets at a step back larger than the setting, and the bank still refuses one no larger |

This code feeds ten rows of `df` with the row at `t = 6` two rows late,
then the same rows sorted, then a day replayed from its start under a
`restart_after_step_back` below the replay's step. It uses `df` from
[Example data](../EXAMPLE-DATA.md):

```python
import textwrap

late = df.head(10)[[0, 1, 2, 3, 4, 5, 7, 8, 6, 9]]           # t runs 0 ... 5, 7, 8, 6, 9
strict = po.spec.ewridge("strict", targets=["y"], features=["x0"], clock="t", half_life=100.0, gap_cap=300.0)
# Rows not in clock order: refused.
try:
    po.ModelBank([strict]).fit_predict(late)
except ValueError as err:
    print(textwrap.fill(str(err), 100, break_long_words=False, break_on_hyphens=False))   # the link kept whole

# The same rows sorted: learned.
bank = po.ModelBank([strict])
bank.fit_predict(late.sort("t"))
print(f"sorted: {bank.summary()['rows_learned'][0]} rows learned")

# A day replayed from its start: the model resets at the step back, and summary() counts both.
replayed = pl.concat([df.head(6), df.head(4)])               # t runs 0 ... 5, then 0 ... 3 again
lenient = po.spec.ewridge("lenient", targets=["y"], features=["x0"], clock="t", half_life=100.0, gap_cap=300.0,
                          restart_after_step_back=1.0)       # a step back larger than 1 resets the model
bank = po.ModelBank([lenient])
bank.fit_predict(replayed)
counts = bank.summary().row(0, named=True)
print(f"replayed: clock_backwards {counts['clock_backwards']}, resets {counts['resets']}")
```

```text
spec "strict": clock column "t" steps back by 2 at row 8 (counting from 0), and
restart_after_step_back is not set, so the bank refused the chunk and learned none of it. Sort the
rows by the clock, skip the rows a saved state has learned with ModelBank.skip_learned (or
--skip-learned), or set restart_after_step_back below 2 to reset the model at such a step back (0
resets it at every one). See
https://github.com/hgilde/polars-online/blob/main/docs/how-to/put-rows-in-clock-order.md#fix-a-chunk-the-bank-refused
for which to choose.
sorted: 10 rows learned
replayed: clock_backwards 1, resets 1
```
