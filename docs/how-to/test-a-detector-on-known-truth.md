<!-- kind: how-to -->
# Test a detector on known truth

This how-to is for anyone who fits a detector of changing correlation,
such as `corrchange`, `hmm`, `bocpd` or `deco`, and wants to know how well
it finds what it claims to. It simulates a stream with a planted change
with `po.sim.regimes`, fits the detector, and prints each change it found
beside the truth.

## Prerequisites

The recipes build their own data, so these steps need only:

- polars-online installed, and numpy, which `po.sim.regimes` draws with.
  Install it with `pip install polars-online[numpy]`.
- A [spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) for the
  detector you mean to test, and the question it answers: a change in
  the correlations (`corrchange`), the age of the current regime
  (`bocpd`), the regime a row belongs to (`hmm`), or an equicorrelation
  that moves (`deco`). [Choose a model](choose-a-model.md#detect-changes-and-regimes)
  sets out the four.

## Simulate a stream whose truth is known

Use `po.sim.regimes` for a stream of several series whose correlation
changes by regime. It returns the rows a detector reads and, beside them,
the truth it should find.

1. **Call `po.sim.regimes` with the number of series, each regime's
   correlation in `states`, how the regimes follow one another in
   `transition`, and the blocks the stream is made of.** Each block of
   `rows_per_block` rows holds one regime. A `seed` makes the stream again
   byte for byte, under one numpy version.
2. **Add the parts of a real stream your detector must survive,** each a
   keyword: `phi` for returns correlated with their own past, `noise` for
   noise on the observed levels, and `async_rates` for series observed
   less often than every row.
3. **Read the three frames it returns:** `rows`, `truth_rows` and
   `truth_blocks`.

This code simulates four series in two regimes and prints the columns of
each frame:

```python
import polars_online as po

sim = po.sim.regimes(                                  # 1.
    4, states=[0.2, 0.7],                     # four series; two regimes, at these equicorrelations
    transition=[[0.98, 0.02], [0.02, 0.98]],  # how the regimes switch
    n_blocks=8, rows_per_block=500,
    durations=None,                           # or a block count per state: each state lasts exactly as long as it says
    design="step", smooth_rows=0,             # or "smooth": interpolate the matrix over smooth_rows rows at a boundary
    phi=0.3, noise=0.01,                      # 2. returns correlated with their own past; observation noise
    async_rates=[1.0, 1.0, 0.4, 0.4],         # 2. two series observed less often; a series with no observation is null
    seed=0,                                   # the same seed twice is byte-identical, under one numpy version
)
rows, truth_rows, truth_blocks = sim["rows"], sim["truth_rows"], sim["truth_blocks"]   # 3.
for name, frame in [("rows", rows), ("truth_rows", truth_rows), ("truth_blocks", truth_blocks)]:
    print(f"{name}: {frame.height} rows; {', '.join(frame.columns)}")
```

```text
rows: 4000 rows; entity, t, clock, session, x_1, x_2, x_3, x_4, activity
truth_rows: 4000 rows; t, block, state, scale_mult, mix
truth_blocks: 8 rows; block, state, n_rows, corr
```

| frame | what it holds |
|---|---|
| `rows` | what a detector reads: an entity, the row's index `t`, a clock and a session, levels `x_1` .. `x_m`, and `activity`, null unless `activity=(mean, shape)` is given. For returns, apply `.diff()` to the levels. Under `async_rates`, `unpivot` the levels into the long input of [`refresh_time`](align-series-that-tick-at-their-own-times.md), and apply `.diff()` to the refresh grid it builds |
| `truth_rows` | per row `t`: the block, state, volatility multiplier and interpolation fraction |
| `truth_blocks` | each block's true correlation matrix, the mean over its rows, as its upper triangle with the diagonal, row by row: for two series `[1, rho, 1]` |

Every keyword, and what it refuses, is in `po.sim.regimes`'
[docstring](https://hgilde.github.io/polars-online/sim.html#polars_online.sim.regimes).

## Score the detector against the truth

Use these steps once the stream is simulated, to read each change the
detector found beside each change that was planted.

1. **Join `truth_rows` to `rows` on `t`,** so the truth travels with each
   row.
2. **Save the joined stream to a parquet file.**
3. **Read it back as a
   [query](../GLOSSARY.md#the-model-bank-and-what-it-reads) that takes the
   levels' changes with `diff()`.** A correlation detector reads returns,
   not levels.
4. **Fit the detector.**
5. **Take the first row of each run of flags.** A detector that flags on
   every row of a change, as `corrchange`'s window kind does, gives one
   change per run.
6. **Match each planted change to the first flag after it.** The rows
   from the change to that flag are the detector's delay. A flag that
   follows no planted change in the block where it falls is a false
   alarm.

This code plants a correlation that switches between 0 and 0.7 every
1,000 rows, and tests `corrchange`'s sequential kind on it:

```python
import polars as pl

# Two series whose correlation takes turns, 1,000 rows at a time, between 0 and 0.7.
sim = po.sim.regimes(2, states=[0.0, 0.7], transition=[[0.0, 1.0], [1.0, 0.0]],
                     n_blocks=6, rows_per_block=1_000, seed=4)
truth = sim["truth_rows"].select("t", "block", "state")
sim["rows"].join(truth, on="t").write_parquet("known.parquet")                           # 1. and 2.
returns = pl.scan_parquet("known.parquet").with_columns(pl.col("x_1", "x_2").diff())   # 3.

# 4 and 5. The detector, and the first row of each run of flags.
watch = po.spec.corrchange("watch", features=["x_1", "x_2"], kind="sequential",
                           window_rows=300, monitor_rows=1_000)
out = returns.online.fit_predict([watch]).collect().unnest("watch")
found = out.filter(pl.col("flag") & ~pl.col("flag").shift(1, fill_value=False))["t"].to_list()
# 6. Each planted change beside the first flag after it, and the flags no change explains.
planted = out.filter(pl.col("state") != pl.col("state").shift(1))["t"].to_list()
for change in planted:
    after = [f for f in found if change <= f < change + 1_000]
    print(f"planted at row {change}: " + (f"found at row {after[0]}, {after[0] - change} rows later" if after else "missed"))
false = [f for f in found if not any(c <= f < c + 1_000 for c in planted)]
print(f"false alarms: {len(false)}")
```

```text
planted at row 1000: found at row 1115, 115 rows later
planted at row 2000: found at row 2156, 156 rows later
planted at row 3000: found at row 3223, 223 rows later
planted at row 4000: found at row 4129, 129 rows later
planted at row 5000: found at row 5222, 222 rows later
false alarms: 0
```

## Check that it worked

**Run the detector on a stream with no change, over several seeds, to
measure its false alarms.** False alarms are rare and fall at random, so
one stream's count says little. This code fits the same spec on ten streams that hold one
correlation, 0.35, throughout. It uses `watch` and `pl` from the
code in [Score the detector against the
truth](#score-the-detector-against-the-truth), and `po` from the code in
[Simulate a stream whose truth is
known](#simulate-a-stream-whose-truth-is-known):

```python
# Ten streams of 6,000 rows with no change, and the flags on all of them.
flags = 0
for seed in range(10):
    quiet = po.sim.regimes(2, states=[0.35], transition=[[1.0]], n_blocks=6, rows_per_block=1_000, seed=seed)
    quiet["rows"].write_parquet("quiet.parquet")
    out = pl.scan_parquet("quiet.parquet").with_columns(pl.col("x_1", "x_2").diff()).online.fit_predict([watch]).collect()
    flags += out["watch"].struct.field("flag").sum()
print(f"flags on 60,000 rows with no change: {flags}")
```

```text
flags on 60,000 rows with no change: 6
```

| what you see | what it means | what to do |
|---|---|---|
| a planted change missed | the detector's history (`window_rows`) or monitoring period (`monitor_rows`) does not cover the change, or the change is too small for them | for `corrchange`'s sequential kind, make `monitor_rows` long enough that a cycle crosses each change; give a larger change, or more rows a block, and read where it starts to find it |
| a delay longer than you can act on | the history is long: a change is found once enough rows after it differ | shorten `window_rows`, and measure the false alarms again, since a shorter history flags more often on a quiet stream |
| many flags on the stream with no change | the level `alpha` is too loose for this stream, or the stream has a part the detector assumes away | add `phi`, `noise` or `async_rates` one at a time to find which part moves the count; [Detect a break](detect-a-break.md#watch-the-features-correlations-with-corrchange) sets `corrchange`'s boundary and its runs of flags |

[REGIMES.md](../REGIMES.md) gives what `hmm`, `corrchange`, `bocpd`,
`deco` and `rcov` find on such streams, measured over many seeds, and what
they miss.
