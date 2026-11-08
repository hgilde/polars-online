<!-- kind: how-to -->
# Inspect a bank

This how-to is for anyone who wants to know what a model bank holds and
what it was fed, between chunks or after a run, from a live bank or from a
state file with no data at hand.

A [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) fits one
model for each [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)
and each [group](../GLOSSARY.md#the-model-bank-and-what-it-reads). It
describes itself without its data, live or loaded from its file, and
reports each group as it stood after the last row it learned from.

## Prerequisites

These steps need the library and a bank that has learned from rows:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `lf`, a [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) over
  `ticks.parquet`: 400 rows in which four stocks, `b0` to `b3` in the
  column `stock_id`, take turns, and whose target `y` is
  `1 + 2·x0 − x1` plus noise.
- A bank, live or loaded with `po.ModelBank.load` from a state file
  ([Save a state](save-a-state.md)).

## Read what a bank holds

A bank's own attributes and `groups()` say what it holds, with no table to
reshape.

1. **Print `repr(bank)`** for its specs, its groups and the rows it was
   fed.
2. **Read `bank.rows_fed` and `bank.solve_failures()`.** The second counts,
   per group, the solves that needed jitter or kept the previous fit.
3. **Read `bank.output_fields[name]`** for the fields of a spec's output
   column.
4. **List the groups with `bank.groups()`,** one row per (spec, group),
   with the rows each was fed and its last clock value.

Here a spec with a group per stock learns from `lf`, and the bank it saves
is loaded back with nothing but the file. This code uses `lf` from
[Example data](../../README.md#example-data):

```python
# A bank fitted per stock, and saved.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0,
                       half_life=600.0, group="stock_id")
fitted = po.ModelBank([spec])
fitted.fit(lf)                           # learn from every row of the file, keep only the state
fitted.save("bank.state")

# Loaded back from the file alone, it describes itself.
bank = po.ModelBank.load("bank.state")   # no specs=: the file is enough
print(repr(bank))                        # 1. its specs, its groups, the rows it was fed
print(bank.rows_fed, bank.solve_failures()["solve_failures"].to_list())   # 2. rows fed; solves that needed help, per group
print(bank.output_fields["ridge"])       # 3. the fields of the spec's column
for spec_name, group, rows, last in bank.groups().iter_rows():   # 4. one row per (spec, group)
    print(spec_name, group, rows, last)
```

```text
ModelBank(['ridge'], groups=4, rows_fed=400)
400 [0, 0, 0, 0]
['pred_y', 'resid_y', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']
ridge b0 100 396.0
ridge b1 100 397.0
ridge b2 100 398.0
ridge b3 100 399.0
```

No solve needed help here. `bank.specs` gives every spec back, as the dict
its builder made.

**`bank.specs` returns a read-only copy.** The bank runs the state it was
built from, so an editable list could drift from it and mislabel what
`coef()` reports. To change a spec, build and feed a new bank.

## Read how each fit is doing and what it was fed

Four methods each return a table, with a row per group, or per group and
coefficient or column:

| method | one row per | what it holds |
|---|---|---|
| `bank.coef()` | coefficient | spec, group, instance, ..., term, coef ([Read the coefficients](read-the-coefficients.md)) |
| `bank.summary(name)` | group | what it was fed, and its warm-up readings |
| `bank.describe(name)` | input column per group | its role, count, unusable count and moments |
| `bank.last_row(name)` | group | the record `fit_predict` wrote on its last row |

1. **Call each method you need.**
2. **Reshape the result with Polars,** such as `pivot` to set each
   group's coefficients side by side.

Here each stock's coefficients come back near the truth the example data
plants: intercept 1,
`x0` 2, `x1` −1 and `x2` 0. This code uses `lf` from [Example
data](../../README.md#example-data):

```python
# A ridge per stock, learned from every row of lf.
spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1", "x2"], clock="t", gap_cap=300.0,
                       half_life=600.0, group="stock_id")
bank = po.ModelBank([spec])
bank.fit(lf)

# 1. The four tables, each a frame.
betas = bank.coef()                    # one row per coefficient: spec, group, instance, ..., term, coef
fed = bank.summary("ridge")            # one row per group: what it was fed, and its warm-up readings
cols = bank.describe("ridge")          # one row per input column per group: role, count, nulls, moments
last = bank.last_row("ridge")          # one row per group: the record fit_predict wrote on its last row

# 2. Read three of them: the coefficients, what was fed, and b0's columns.
wide = betas.pivot("term", index="group", values="coef")   # a row per group, a column per term
for row in wide.iter_rows(named=True):
    print(row["group"], " ".join(f"{t} {row[t]:.2f}" for t in ["intercept", "x0", "x1", "x2"]))
for g, n_fed, n_learned, w in fed.select("group", "rows_fed", "rows_learned", "weight_sum").iter_rows():
    print(f"{g}: fed {n_fed}, learned {n_learned}, weight_sum {w:.1f}")
for c, role, n, mean, std in cols.filter(group="b0").select("column", "role", "count", "mean", "std").iter_rows():
    print(f"b0 {c} ({role}): {n} values, mean {mean:.2f}, std {std:.2f}")
```

```text
b0 intercept 1.00 x0 2.02 x1 -1.02 x2 -0.01
b1 intercept 1.01 x0 2.02 x1 -1.06 x2 0.05
b2 intercept 0.93 x0 1.94 x1 -0.99 x2 -0.04
b3 intercept 1.04 x0 1.99 x1 -0.97 x2 -0.13
b0: fed 100, learned 100, weight_sum 100.0
b1: fed 100, learned 100, weight_sum 100.0
b2: fed 100, learned 100, weight_sum 100.0
b3: fed 100, learned 100, weight_sum 100.0
b0 x0 (feature): 100 values, mean -0.04, std 0.95
b0 x1 (feature): 100 values, mean 0.03, std 1.01
b0 x2 (feature): 100 values, mean -0.09, std 0.99
b0 y (target): 100 values, mean 0.88, std 2.12
```

**`last_row()` matches `fit_predict` field for field:** `pred_<target>`, `sigma_<target>`,
the metrics and the interval when the spec asks for them
([DIAGNOSTICS.md](../DIAGNOSTICS.md)), `weight_sum`, and `coef` when that
row carried it. Until a group has a row that is not
[skipped](../GLOSSARY.md#the-model-bank-and-what-it-reads), its record is
all nulls.

**`summary()`'s counts and `describe()` cover every row fed to the group,
without decay.** They say what the model learned from, and `predict`
moves neither. So `summary()`'s `weight_sum` is 100 in the example, each stock's
100 rows of weight 1, where the fit's own `weight_sum` has decayed below
it. `describe()` counts a value as the models do: a null, a NaN, an
infinity or a magnitude beyond 1e100 counts in `unusable_count`, not in
`count`.

The columns of `summary()`, each count an `Int64`:

| columns | what they hold |
|---|---|
| `rows_fed` | rows routed to the group |
| `rows_processed` | rows the model accepted: every feature and the weight usable |
| `rows_skipped` | rows the model skipped, because a feature or the weight was null, NaN, infinite or beyond 1e100 |
| `rows_learned` | rows that moved the fit: a weight above zero and a target present. Under an embargo a row is counted as it arrives; a formula target's row, when it is released |
| `rows_zero_weight` | rows that advanced the clock and nothing else |
| `weight_sum` | the sum of the accepted rows' weights, undecayed: 1 a row without a weight column |
| `clock_min`, `clock_max`, `clock_last` | the clock's range and last value, in the clock column's own dtype |
| `session_changes`, `clock_backwards`, `resets` | what the clock rules met |
| `solve_failures` | solves that needed jitter or kept the previous fit, which `bank.solve_failures()` lists alone |
| `refit_given_up` | the targets whose refit of their first rows was given up, past the spec's `refit_budget` or past the cap on how long its buffer of rows ages. 0 on a spec that does not refit |
| `labels_missing`, `labels_dropped` | under `label_clock`, the rows whose targets were never learned: those whose `label_clock` is null, and those held and then dropped. 0 on every other spec |
| `settled_frac`, `weight_sum_settled`, `error_inflation`, `support_coef_min` and the feature it belongs to, `n_coef` | the warm-up readings after the last row. `weight_sum_settled` is the weight `weight_sum` tops out at, which says whether a `min_weight` can be met |

## Compare banks from different runs in one table

**All four tables cover every spec by default, with `spec` as the first
column, so the tables of banks from different runs stack.**

1. **Load each bank's state file** with `po.ModelBank.load`.
2. **Stack `coef()`, `summary()` or `describe()` with a plain
   `pl.concat`,** since they have the same columns for every spec.
3. **Stack `last_row()` with `how="diagonal_relaxed"`,** since its fields
   differ between specs. It fills a missing field with nulls.

Here two fits at different half-lives are saved to files of their own and
compared in one table. This code uses `lf` from [Example
data](../../README.md#example-data):

```python
from pathlib import Path

Path("fits").mkdir(exist_ok=True)
for name, half_life in [("fast", 50.0), ("slow", 1000.0)]:    # two half-lives, a bank and a file each
    one = po.ModelBank([po.spec.ewridge(name, targets=["y"], features=["x0", "x1"], clock="t",
                                        gap_cap=300.0, half_life=half_life)])
    one.fit(lf)
    one.save(f"fits/{name}.state")

# 1. Load the files back, and stack their tables in one.
banks = [po.ModelBank.load(f) for f in sorted(Path("fits").glob("*.state"))]
betas = pl.concat([b.coef() for b in banks])                          # 2. the same columns for every spec
last = pl.concat([b.last_row() for b in banks], how="diagonal_relaxed")   # 3. fields that differ by spec
for row in betas.pivot("term", index="spec", values="coef").iter_rows(named=True):
    print(row["spec"], " ".join(f"{t} {row[t]:.3f}" for t in ["intercept", "x0", "x1"]))
print(last.columns)
```

```text
fast intercept 0.994 x0 1.936 x1 -1.004
slow intercept 0.993 x0 1.992 x1 -1.007
['spec', 'group', 'pred_y', 'resid_y', 'weight_sum', 'settled_frac', 'withheld_reason', 'coef', 'support_coef']
```

## Check that it worked

**Summed over one spec's groups, `summary()`'s `rows_fed` equals
`bank.rows_fed`,** 400 in the first example, while no group has been
dropped or closed. Each group's count matches `bank.groups()`. A group with `rows_learned`
below its `rows_fed` has rows the model skipped or held, which
`rows_skipped`, `rows_zero_weight` and `describe()`'s `unusable_count`
account for. [Check data quality](check-data-quality.md) finds what is
wrong with such rows.
