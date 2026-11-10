<!-- kind: how-to -->
# Check a stream's data quality

This how-to is for anyone who has fitted a model bank on a stream and
wants to know whether the data can be learned from at all. It runs
`bank.check()` after a pass, with an `audit` spec beside the models, and
acts on what it finds.

## Prerequisites

These steps need the library and a stream to check:

- polars-online installed. The first recipe builds its own data, and the
  recipe in [Check that it worked](#check-that-it-worked) reuses it. The
  recipes in [Find a duplicated column](#find-a-duplicated-column) and
  [Read every group as one stream](#read-every-group-as-one-stream) use
  `df` from
  [Example data](../EXAMPLE-DATA.md).
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) for each
  model you mean to fit, and the
  [features](../GLOSSARY.md#the-model-bank-and-what-it-reads) and
  [target](../GLOSSARY.md#the-model-bank-and-what-it-reads) columns it
  reads.
- numpy, which the recipes use to make their data, and which the two
  checks that read a [Gram](../GLOSSARY.md#fitting-a-model), `collinear`
  and `leakage`, need. Install it with
  `pip install polars-online[numpy]`. Without it, `check()` gives a
  [`not_checked`](../DATA-QUALITY.md#not_checked) finding in their place.

## Run the check after a pass

A model fitted on a stream fits whatever it is given. It cannot tell you
that a column is `-999` where it was not measured, or that a feed stopped
and its last value was carried forward. Nor can it tell you that a feature
is a price level where a change was meant. `bank.check()` reads what a
[model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads) kept during
a pass and lists what is wrong with the data. The
[diagnostics](../DIAGNOSTICS.md) ask whether a model is working, and
`check()` asks whether the data can be learned from at all. It reads only
what the bank kept, adding no work to the run. Call it after a first pass
and before trusting a fit. An `audit` spec
counts what no model keeps, such as sentinel values, frozen feeds, heavy
tails and the clock's spacing, in about 14 KiB of state a column.

1. **Put an `audit` spec in the bank beside the models,** with
   `po.spec.audit(name, columns=[...])`. Give it every feature and the
   target, and the models' [clock](../GLOSSARY.md#time) keywords. An audit
   fits nothing and predicts nothing.
2. **Fit the bank over the stream,** with `bank.fit(query)` to learn from
   every row and keep only the [state](../GLOSSARY.md#saving-and-serving),
   or with `bank.fit_predict_batches(query)` to keep each row's
   prediction too.
3. **Call `bank.check()`.** It returns one row per problem, the worst
   first.
4. **Fix each finding with Polars, in the query before the bank.** Each
   finding's `message` says what is wrong and, for most codes, what to do.
   It ends with a link to the
   section of [docs/DATA-QUALITY.md](../DATA-QUALITY.md) that its `code`
   names. That section shows what the problem does to each model and how to
   fix it.

Here is a stream with three faults planted, fitted by a ridge regression
with an audit beside it. `x1` is `-999` where it was not measured, `x2` is
the same value on every row, and `x3` is null on 30% of the rows. This
code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# Quotes, a row a second, with the three faults planted.
rng = np.random.default_rng(1)
n = 20_000
x0, x1, x3 = rng.standard_normal((3, n))
y = 1.0 * x0 + 0.5 * x1 + 0.2 * x3 + rng.standard_normal(n)
pl.DataFrame(
    {
        "t": np.arange(n, dtype=float),
        "x0": x0,
        "x1": np.where(rng.random(n) < 0.05, -999.0, x1),  # -999 where it was not measured
        "x2": np.full(n, 1.0),  # one value on every row
        "x3": np.where(rng.random(n) < 0.3, np.nan, x3),  # null on 30% of the rows
        "y": y,
    }
).fill_nan(None).write_parquet("quotes.parquet")
quotes = pl.scan_parquet("quotes.parquet")

# 1. An audit beside a ridge regression, both fitted over the stream.
features = ["x0", "x1", "x2", "x3"]
audit = po.spec.audit("audit", columns=[*features, "y"], clock="t", gap_cap=5.0)
model = po.spec.ewridge(
    "ridge", targets=["y"], features=features, clock="t", gap_cap=5.0, half_life=600.0
)
bank = po.ModelBank([audit, model])
bank.fit(quotes)  # 2. learn from every row; keep only the state

# 3. The findings, and the advice the worst one gives.
findings = bank.check()  # one row per problem, the worst first
shown = findings.select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3))
for row in shown.iter_rows():
    print(*row)
print(findings.row(0, named=True)["message"])  # what is wrong, and the section of DATA-QUALITY.md on it
```

```text
warning constant audit x2 0.0
warning sentinel audit x1 0.0464
warning constant ridge x2 0.0
warning missing ridge x3 0.3
info heavy_tails audit x1 951.0
info missing audit x3 0.3
column 'x2' took one value (1) on every usable row of the stream: as a feature it is collinear with the intercept, and as a target there is nothing to learn. See https://github.com/hgilde/polars-online/blob/main/docs/DATA-QUALITY.md#constant
```

## Read the findings

**Each finding is a row, the worst first.** Its `severity` is `error` (the
model cannot learn), `warning` (the fit is unreliable) or `info` (worth
knowing, and often fine). Its `code` names the check, such as `missing`,
`constant`, `collinear` or `leakage`. The `spec`, `group` and `column` say
where it was found, the measured `value` sits beside its `threshold`, and
the `message` says what is wrong and, for most codes, what to do. The guide
[docs/DATA-QUALITY.md](../DATA-QUALITY.md) lists every code under the problem it belongs to, and the docstring of
[`ModelBank.check`](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.check)
lists every check with its threshold.

**A finding's `value` gives the number measured,** here the share
missing, the share of the commonest value, a standard deviation of 0, or
the robust z of the farthest value. Each check reads the values a model can learn
from. Like a model, it counts NaN, ±inf and any magnitude above `1e100` as
[null](../GLOSSARY.md#the-model-bank-and-what-it-reads).

**Each finding names the spec that found it.** In the recipe, the audit
sees every fault: `x2` is constant, `-999` is 4.6% of `x1`'s values, and
`x3` is missing on 30% of the rows. The ridge regression sees the two
faults it can measure, the constant feature and the missing one. It cannot
see the sentinel, because to a model `-999` is a number like any other.
The sentinel also gives `x1` heavy tails: one value 951 robust standard
deviations from the median (a robust standard deviation is the median
absolute deviation times 1.4826).

**The audit and the model can give one fault different severities.** The
audit reports `x3`'s gaps as `info`, and the model as a `warning`. An
audit knows no roles, and a gap in a target can be a design, where a gap
in a feature skips the row.

**`check()` reads only the groups still open.** A group closed under
`group_close` has left the bank with its sums, so no finding comes from
it. An audit refuses `group_close`, so an audit's findings cover every
group.

## Find a duplicated column

An audit given `pairs=True` keeps each pair of columns' correlation and
the rows on which the two are equal. `bank.check()` reads them for its
[`duplicate`](../DATA-QUALITY.md#duplicate) finding.

1. **Give the audit `pairs=True`,** with the columns you suspect among
   its `columns`.
2. **Fit the bank over the stream.**
3. **Call `bank.check()`.** A duplicated column gives a `duplicate` row
   for each group in which it was found.

This code adds to
`df` a copy of `x0` in cents, recorded with a little noise of its own,
audits it per stock, and prints what `check()` finds. It uses `df` from
[Example data](../EXAMPLE-DATA.md), and `np` and `po` from the
code in [Run the check after a pass](#run-the-check-after-a-pass):

```python
# x0 again, in cents and with a little noise: a duplicate planted in each of the four stocks.
rng = np.random.default_rng(2)
copied = df.with_columns(x0_cents=pl.col("x0") * 100 + pl.Series(rng.normal(0.0, 0.01, df.height)))
audit = po.spec.audit("audit", columns=["x0", "x1", "x0_cents", "y"], clock="t", gap_cap=300.0,
                      pairs=True, group="stock_id")
bank = po.ModelBank([audit])
bank.fit_predict(copied)                          # the output record holds weight_sum alone
problems = bank.check()                           # sentinel, frozen, few_values, random_walk, heavy_tails, duplicate, ...
for row in problems.select("severity", "code", "group", "column").iter_rows():
    print(*row)
```

```text
warning duplicate b0 x0_cents
warning duplicate b1 x0_cents
warning duplicate b2 x0_cents
warning duplicate b3 x0_cents
```

## Read every group as one stream

`bank.audit()` reads the audit's counts as a table, and its
[docstring](https://hgilde.github.io/polars-online/polars_online.html#polars_online.ModelBank.audit)
defines each column.

1. **Call `bank.audit()`.** It returns one row per
   [group](../GLOSSARY.md#the-model-bank-and-what-it-reads) and column.
   `table="pairs"` gives one row per group and pair of columns, and
   `table="clock"` one row per group for the clock.
2. **Call `bank.audit(pooled=True)` to read every group as one stream.**
   Groups merge into one audit: the counts exactly, the moments to rounding,
   and the repeats past `distinct_cap` and the median within the bounds the
   docstring of
   [`po.spec.audit`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.audit)
   gives.

This code reads the audit of [Find a
duplicated column](#find-a-duplicated-column) per stock and pooled, and
holds the pooled mean of `x0` to numpy's over all of `df`'s rows. It uses
`bank` from that code and `df` from
[Example data](../EXAMPLE-DATA.md):

```python
columns = bank.audit()                            # one row per (group, column)
pairs = bank.audit(table="pairs")                 # one row per (group, pair of columns)
clock = bank.audit(table="clock")                 # one row per group: duplicate stamps, gaps, the steps' spread
everything = bank.audit(pooled=True)              # the four groups as one stream
x0 = everything.filter(pl.col("column") == "x0").row(0, named=True)
print("rows per stock:", columns.filter(pl.col("column") == "x0")["rows_fed"].to_list())
print(f"pooled: {x0['rows_fed']} rows, mean {x0['mean']:.6f}; numpy over df: {np.mean(df['x0'].to_numpy()):.6f}")
```

```text
rows per stock: [100, 100, 100, 100]
pooled: 400 rows, mean -0.036620; numpy over df: -0.036620
```

## Check that it worked

**Fix the query, run the pass again on a bank that starts with nothing
learned, and call `check()` again.** A fault that is fixed gives no
finding. Here the sentinel is made null, so the model skips those rows,
and the constant feature is left out. This code uses `quotes` from the
code in [Run the check after a
pass](#run-the-check-after-a-pass):

```python
fixed = quotes.with_columns(x1=pl.when(pl.col("x1") == -999.0).then(None).otherwise(pl.col("x1")))   # the sentinel made null
kept = ["x0", "x1", "x3"]                                                                         # the constant x2 left out
bank = po.ModelBank([
    po.spec.audit("audit", columns=[*kept, "y"], clock="t", gap_cap=5.0),
    po.spec.ewridge("ridge", targets=["y"], features=kept, clock="t", gap_cap=5.0, half_life=600.0),
])
bank.fit(fixed)
for row in bank.check().select("severity", "code", "spec", "column", pl.col("value").round_sig_figs(3)).iter_rows():
    print(*row)
```

```text
warning missing ridge x3 0.3
info missing audit x3 0.3
```

The sentinel, its heavy tail and the constant feature give no finding.
`x3`'s gaps remain, since the fix leaves them: [Null
values](../DATA-QUALITY.md#null-values) compares the ways to handle a
feature missing on many rows.
