<!-- kind: how-to -->
# Make a target that looks ahead

This how-to is for anyone who wants a model to predict a value computed
from the rows after each row, such as the next five minutes' mean trade
price less the mid at the row. It gives the target to the model bank as a
formula it holds until the value is known, or as a column made before the
bank.

## Prerequisites

These steps need the library, a stream with a clock, and a model that
regresses its targets:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `trades.parquet`, a stream of quotes with trades between them. Its `mid`
  is on every row, and `side`, `quantity` and `price` are null on a quote.
- A [clock](../GLOSSARY.md#time) column, here the timestamp column `ts`,
  with the rows in clock order ([Put rows in clock
  order](put-rows-in-clock-order.md)).
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) whose model
  regresses its targets. Only such a model takes a target expression.
  Four kinds of model refuse one, saying why: a model that reads only its
  features, `ew_class`, `seqtest`, and an `ftrl` or `sgd` fitting a
  probability or a count.

## Give the target as a formula in the spec's targets

A [formula target](../GLOSSARY.md#fitting-a-model) is an expression (a
column computation, such as `pl.col("x") + 1`) that holds a [window
operator](../GLOSSARY.md#fitting-a-model) looking ahead. The bank computes
its windows itself, one set per
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads) on the spec's
clock, so the groups may interleave in time. It predicts each row where
the row sits, and learns from the row once the row's window has closed and
its [`embargo`](../GLOSSARY.md#time) has passed: the delay on the clock
between predicting a row and learning from it. [A target that looks
ahead](../explanation/out-of-sample.md#a-target-that-looks-ahead) explains
why the bank waits.

1. **Write the target as an expression around an operator that looks
   ahead, and name it with `.alias()`.** The operators that look ahead are
   `po.ewm_mean`, `po.ewm_sum`, `po.ewm_rate` and `po.ewm_quantile`, each
   with `reverse=True`. The alias names the output fields: `.alias("fwd_edge")`
   gives `pred_fwd_edge`. An expression with no operator that looks ahead
   is known at its own row, and the spec builder raises `ValueError`
   ([Make a target known at its own row](#make-a-target-known-at-its-own-row)).
2. **Put the expression in the spec's `targets`.**
3. **Give the spec an `embargo` of at least the window's `window_size`.**
   `fit_predict` refuses a shorter one before it reads a row. For a shorter
   `embargo`, or none, use `bank.fit` or the command line's `--no-output`,
   which keep no predictions.
4. **Give the spec the same `clock`, `gap_cap` and `group` as the
   features' windows.** The builder refuses `group` without `clock`, as
   `with_windows` does. Leave out `group_close` (the setting that closes a
   group whose rows have ended), which a formula target refuses.
5. **Run the query.** `pred_<name>` is filled on each row the model
   predicts, and `resid_<name>` is null on every row, because a row's
   target is not known when the row is predicted. The diagnostics take each
   row in when it is learned, as under any `embargo` ([Learn targets known
   late](learn-targets-known-late.md)). `pred` and `resid` are on the
   target's scale.

This code fits the next minute's VWAP (volume-weighted average price) of
the trades, less the mid now, on two trends of the mid that look back. It
uses `trades.parquet` from [Example data](../../README.md#example-data):

```python
# Features that look back, on one clock shared with the model.
clock = dict(clock="ts", gap_cap="5m", group="symbol")             # one clock for the windows and the model
trends = pl.scan_parquet("trades.parquet").online.with_windows(    # the features: windows that look back
    trend_5s=pl.col("mid") - po.ewm_mean("mid", half_life="5s", window_size="2m"),
    trend_30s=pl.col("mid") - po.ewm_mean("mid", half_life="30s", window_size="2m"),
    **clock,
)
# 1. A target that looks ahead: the next minute's VWAP, less the mid.
notional = pl.col("price") * pl.col("quantity")
next_minute = dict(half_life="10s", window_size="1m", reverse=True)
fwd_edge = (po.ewm_sum(notional, **next_minute) / po.ewm_sum("quantity", **next_minute) - pl.col("mid")).alias("fwd_edge")
# 2 to 4. The target in targets, learned two minutes late, once its minute is known.
edge = po.spec.ewridge("edge", targets=[fwd_edge], features=["trend_5s", "trend_30s"],
                       half_life="30m", embargo="2m", **clock)
fitted = trends.online.fit_predict([edge]).collect()      # 5. trades, the two trends, and the column edge
expression = fitted["edge"].struct
print(f"rows {fitted.height}, of which predicted {fitted.height - expression.field('pred_fwd_edge').null_count()}")
print(f"resid_fwd_edge null on {expression.field('resid_fwd_edge').null_count()} rows")

# An embargo shorter than the target's window is refused.
short = po.spec.ewridge("short", targets=[fwd_edge], features=["trend_5s"], half_life="30m", embargo="30s", **clock)
try:
    trends.online.fit_predict([short])                    # refused when the query is built
except ValueError as e:
    print(str(e).split(". ")[0])
```

```text
rows 3000, of which predicted 2761
resid_fwd_edge null on 3000 rows
spec "short": fit_predict needs an embargo of at least 1m, the longest window_size among the look-ahead window operators of its formula targets, but it is 30s
```

[Make features that look back over the stream](make-window-features.md)
shows how the trends are made.

## Check the targets the model learned

**To see the targets the model learned, write the target as a column with
`with_windows(fwd_edge, like=edge)`, on a frame that holds the spec's
features.** `like=edge` takes the clock keywords from `edge`, and refuses
any given beside it. It nulls the target on every row the spec would not
learn from.

**Named in `targets` under the same `embargo`, that column gives the same
predictions row for row, except when the embargo equals the window under
`closed="right"`.** Then the column form learns each row at the row exactly
one window after it, and the expression in `targets` at the next row at a
later clock value. The recipe gives the embargo two minutes, past the
window's one, and the two forms agree on every row. This code uses
`trends`, `clock`, `fwd_edge`, `edge` and `fitted` from the code in [Give the
target as a formula in the spec's
targets](#give-the-target-as-a-formula-in-the-specs-targets):

```python
# The same target as a column, and the same model on it.
as_column = trends.online.with_windows(fwd_edge, like=edge).collect()   # the target the model learned, as a column
by_column = po.spec.ewridge("by_column", targets=["fwd_edge"], features=["trend_5s", "trend_30s"],
                            half_life="30m", embargo="2m", **clock)     # the same model, its target named as that column
column_form = as_column.online.fit_predict([by_column])

# The two forms compared row for row.
expression, column = fitted["edge"].struct, column_form["by_column"].struct
print(f"largest difference between the two forms: {(expression.field('pred_fwd_edge') - column.field('pred_fwd_edge')).abs().max()}")
```

```text
largest difference between the two forms: 0.0
```

## Make the target a column before the bank

**To have a residual on each row, make the target a column with `po.stream.with_windows` in the query before the
bank, and give the spec an `embargo` of at least the window's length.**
The expression is the column `po.stream.with_windows` makes from it, fed
back to the bank, so both forms give the same predictions. They differ only
where the `embargo` equals the window under the default `closed="right"`
and a row falls exactly one window after another ([Check the targets the
model learned](#check-the-targets-the-model-learned)). Only the
residual tells them apart: a column holds the target's value at the row,
so the bank can compare it there, while the expression's value is not
known until its window closes. A column made from a window that looks ahead
needs the same `embargo` as the expression, or the model learns each row's
value at the row, before the rows that value was computed from have
arrived. So does any column that holds a value from after its row, such as
the price five minutes on: give the spec an `embargo` of that horizon
([Learn targets known late](learn-targets-known-late.md)).

1. **Make the column with `po.stream.with_windows(query, target, like=spec)`.**
   `like=` takes the spec's clock settings, and nulls the target on every
   row the spec would not learn from.
2. **Name the column in the spec's `targets`,** under the same `embargo`.
3. **Run the query.** Each row whose window has closed gets a residual.

This code fits the next five minutes' mean trade price less the mid, on a
trade's signed size, in both forms. `pred` is on the target's scale, so
`pred + mid` predicts the mean trade price itself. No two of the rows'
clock values are exactly five minutes apart, so the forms agree under an
`embargo` equal to the window. It runs on `trades.parquet`, the
quotes and trades saved in [Example data](../../README.md#example-data):

```python
# A feature, a target that looks ahead, and a fit on the trades.
flows = pl.scan_parquet("trades.parquet").with_columns(                   # a query, and a feature in it: a trade's signed size
    flow=pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
)
fwd_move = (
    po.ewm_mean("price", half_life=float("inf"), window_size="5m", reverse=True)       # the mean trade price over the next five minutes
    - pl.col("mid")                                                        # less the mid at the row
).alias("fwd_move")                                                        # the alias names the target
ahead_5m = po.spec.ewridge("ahead_5m", targets=[fwd_move], features=["flow"],
                           clock="ts", gap_cap="5m", half_life="30m", embargo="5m", group="symbol")
moves = flows.online.fit_predict([ahead_5m]).online.unnest([ahead_5m]).collect()   # does buying predict the price rising?

# The same target as a column, made in the query by with_windows on the spec's clock, then named in targets.
as_column = {**ahead_5m, "targets": ["fwd_move"]}                      # a spec is a dict: a copy with the column as target
made = po.stream.with_windows(flows, fwd_move, like=ahead_5m).online.fit_predict([as_column])   # like= takes the spec's clock settings, and nulls the target where the spec would not learn
column = made.online.unnest([as_column]).collect()

# The two forms side by side.
same = (moves["pred_fwd_move"] == column["pred_fwd_move"]).sum()
print(f"rows predicted: {moves['pred_fwd_move'].count()} by the expression, {column['pred_fwd_move'].count()} by the column, {same} equal")
# The rows of the last five minutes have no target value yet in either form.
print(f"residuals: {moves['resid_fwd_move'].count()} from the expression, {column['resid_fwd_move'].count()} from the column")
```

```text
rows predicted: 733 by the expression, 733 by the column, 733 equal
residuals: 0 from the expression, 556 from the column
```

**Write a look-ahead target as a column for speed, or give its expression
in `targets` for less memory.** In the column form, Polars runs the window
and the bank at once, as two stages of one query. In the expression form the bank resolves the windows itself, one after
the other in the same pass as the model. On 16M rows, the column form
took 5.87 s at a peak of 1.23 GB, and the expression form 9.37 s at
1.09 GB ([Window operators](../../README.md#window-operators)).

## Write a ratio target to sit about zero

**Write a ratio target as the quotient less 1 or as its `.log()`.**
`hit_rate` asks whether the prediction and the target fall on the same
side of zero ([Is another model
better?](../DIAGNOSTICS.md#is-another-model-better)). A plain ratio sits
about 1, where two positive numbers always agree, so its `hit_rate` reads
1.0 whatever the fit. A difference sits about zero too. A target that
looks ahead takes the same forms, such as the log of the next five
minutes' mean trade price over the mid,
`(po.ewm_mean("price", half_life=float("inf"), window_size="5m", reverse=True) / pl.col("mid")).log().alias("fwd_log")`.
`mid * exp(pred)` turns its prediction back into a price. This code regresses four forms of a
trade's price against the mid at its row on the same signed size, `flow`.
It reads `trades.parquet` from [Example data](../../README.md#example-data):

```python
flows = pl.scan_parquet("trades.parquet").with_columns(
    flow=pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
)
# Four forms of a trade's price against the mid.
returns = flows.with_columns(
    ratio=pl.col("price") / pl.col("mid"),                              # price / mid, about 1
    fwd_return=pl.col("price") / pl.col("mid") - 1,                     # the quotient less 1, about zero
    log_ret=(pl.col("price") / pl.col("mid")).log(),                    # ln(price / mid), about zero
    diff=pl.col("price") - pl.col("mid"),                               # price - mid, about zero
)
vs_mid = po.spec.ewridge("vs_mid", targets=["ratio", "fwd_return", "log_ret", "diff"], features=["flow"],
                         clock="ts", gap_cap="5m", half_life="30m", group="symbol", emit_metrics=True)
against_mid = returns.online.fit_predict([vs_mid]).online.unnest([vs_mid]).collect()
# How often each form's prediction had the sign right.
for target in ["ratio", "fwd_return", "log_ret", "diff"]:
    print(f"hit_rate of {target}: {against_mid[f'hit_rate_{target}'].drop_nulls()[-1]:.2f}")
```

```text
hit_rate of ratio: 1.00
hit_rate of fwd_return: 0.51
hit_rate of log_ret: 0.51
hit_rate of diff: 0.51
```

## Make a target known at its own row

**A target expression requires an operator that looks ahead; otherwise,
make the target a column with Polars' `with_columns`, chained in the query
before the bank, and name it in `targets`.** For the command line, which
has no `with_columns`, make the column in the file it reads. The refusal names the
call that makes the column. Make the column with `with_windows` instead if
it reads a window that looks back ([Make features that look back over the
stream](make-window-features.md)). Here the target, how far from the mid
each trade printed, is known at the trade's own row, so the spec needs no
embargo. The quotes have no `quantity`, so the bank skips them, as it
skips any row with a null feature. This code uses `trades.parquet` from
[Example data](../../README.md#example-data):

```python
edges = pl.scan_parquet("trades.parquet").with_columns(edge=(pl.col("price") - pl.col("mid")).abs())   # the target as a column, null on the quotes
by_size = po.spec.ewridge("by_size", targets=["edge"], features=["quantity"],
                          clock="ts", gap_cap="5m", half_life="30m", group="symbol")
printed = edges.online.fit_predict([by_size]).online.unnest([by_size]).collect()   # does a larger trade print farther from the mid?
is_trade = printed["quantity"].is_not_null()
print(f"trades predicted: {printed.filter(is_trade)['pred_edge'].count()} of {is_trade.sum()}")
print(f"quotes predicted: {printed.filter(~is_trade)['pred_edge'].count()} of {(~is_trade).sum()}")
```

```text
trades predicted: 893 of 899
quotes predicted: 0 of 2101
```
