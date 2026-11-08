<!-- kind: how-to -->
# Keep `hmm` states on their regimes

This how-to is for anyone who fits `hmm` to name the regime a stream is
in, and wants each hidden state to stay one regime of the data. It tells
the filter (the model's row-by-row estimate of which state the stream
is in) that regimes last, with a sticky transition prior, and keeps a
single extreme row from capturing a state.

## Prerequisites

These steps need the library, a stream and a spec:

- polars-online installed, and numpy to make the second example's data.
- A stream of numeric
  [feature](../GLOSSARY.md#the-model-bank-and-what-it-reads) columns,
  saved as a file in clock order, to read back with `pl.scan_parquet`.
- An [`hmm`](../MODELS.md#hmm--which-regime-are-we-in) spec: the number
  of regimes in `n_components`, and a small `precision_prior`, a ridge on
  each state's covariance in the features' squared units.
- A rough length for a regime, in rows.

## Tell the filter that regimes last

Use a sticky prior when one row says little about its regime, such as
returns whose spread changes. The filter carries what the rows before have
shown into each row's answer through a transition matrix, the chance of
each state given the last. A prior that makes staying likely lets it weigh
each row against those before it.

1. **Give `transition=` a matrix whose diagonal is near 1,** flattened row
   by row. For regimes of about 500 rows, a chance of 0.002 a row to leave
   gives `[0.998, 0.002, 0.002, 0.998]`.
2. **Give `transition_prior=` the prior's weight,** such as `500.0`. It
   is a pseudo-count per cell: with `transition` given, each row of the
   prior counts as `transition_prior × n_components` transitions, so
   `500.0` at two states counts as 1,000. The matrix learned from the rows
   starts at the given one and moves from it as the counts grow.
3. **Add the specs to the query with `online.fit_predict(specs)`.**
4. **Run the query with `collect()`.**
5. **Read each row's `state`.** The states are numbered as they were
   seeded, so match them to your regimes on rows you know.

This code makes the stream of [Detect changes and
regimes](choose-a-model.md#detect-changes-and-regimes): two series whose
returns take turns, 500 rows at a time, between a calm regime and one 2.7
times as wide, made with `po.sim.regimes`. It fits the filter under the
default prior and under a sticky one, and scores the rows after seeding:

```python
import polars as pl
import polars_online as po

# Two series whose returns take turns, 500 rows a regime, and the regime each row was drawn in.
sim = po.sim.regimes(
    2, states=[0.3, 0.3], transition=[[0.0, 1.0], [1.0, 0.0]],
    n_blocks=6, rows_per_block=500,
    scale_state=[0.0, 1.0],                                   # regime 1's returns are e = 2.7 times as wide
    seed=3,
)
truth = sim["truth_rows"].select("t", regime="state")
sim["rows"].join(truth, on="t").write_parquet("regimes.parquet")
returns = pl.scan_parquet("regimes.parquet").with_columns(pl.col("x_1", "x_2").diff())   # from the levels

# 1 and 2. The same filter under the default prior and under a sticky one.
common = dict(features=["x_1", "x_2"], n_components=2, half_life=float("inf"), precision_prior=1e-2, warm_rows=500)
specs = [
    po.spec.hmm("default", **common),
    po.spec.hmm("sticky", **common, transition=[0.998, 0.002, 0.002, 0.998], transition_prior=500.0),
]
# 3 to 5. Add both specs, run, and read the share of the rows after seeding each puts in its true regime.
out = returns.online.fit_predict(specs).collect().filter(pl.col("t") >= 500)
for name in ("default", "sticky"):
    agree = (out[name].struct.field("state") == out["regime"]).mean()
    print(f"{name}: {max(agree, 1 - agree):.0%} of the rows in their true regime")   # read either numbering
```

```text
default: 58% of the rows in their true regime
sticky: 93% of the rows in their true regime
```

## Keep an extreme row from capturing a state

Use these steps when the stream can hold a row far outside the rest, such
as a bad tick. Such a row can be captured by one state, moving that
state's mean far from the data. A state that then stops winning rows never
forgets, so the model is left one state short.

1. **Choose a guard:**

   | guard | when | what to watch |
   |---|---|---|
   | clip the features with Polars' `clip()` in the query before the bank | in general | set the bounds where the clean rows do not reach |
   | give the states with `means` and `covs`, and freeze them with `learn=False` | you know the regimes in advance | the states no longer follow the stream |
   | give a larger `precision_prior` | you cannot clip the rows | keep it well below the data's variance: with the states held, a ridge of 1.0 on data of variance 1.0 halves every correlation |

2. **Add the spec to the query with `online.fit_predict([hidden])`.**
3. **Run the query with `collect()`.**
4. **Count the rows each state holds** over a stretch after the extreme
   row.

This code makes two overlapping blobs and a sticky chain that switches
between them about once in 200 rows. It plants one row at (1000, 1000),
and fits the filter on the raw rows and on rows clipped to ±5:

```python
import numpy as np
import polars as pl
import polars_online as po

# A sticky two-state chain, two overlapping blobs it switches between, and one extreme row.
rng = np.random.default_rng(12)
n = 6_000
state = np.zeros(n, dtype=int)
for i in range(1, n):                                         # each row has a 0.5% chance of a switch
    state[i] = 1 - state[i - 1] if rng.random() < 0.005 else state[i - 1]
xy = np.array([[-0.75, -0.75], [0.75, 0.75]])[state] + rng.standard_normal((n, 2))
xy[2_000] = [1_000.0, 1_000.0]                                # the extreme row
pl.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "truth": state}).write_parquet("states.parquet")

# 1. The same filter on the raw rows and on rows clipped before the bank.
hidden = po.spec.hmm("hidden", features=["x", "y"], n_components=2, precision_prior=1e-2,
                     warm_rows=500, half_life=float("inf"))
raw = pl.scan_parquet("states.parquet")
clipped = raw.with_columns(pl.col("x", "y").clip(-5.0, 5.0))
# 2 to 4. Add the spec, run, and read the last 3,000 rows: the share in their true state, and the rows each state holds.
for name, query in (("raw", raw), ("clipped", clipped)):
    rows = query.online.fit_predict([hidden]).collect().unnest("hidden").tail(3_000)
    agree = (rows["state"] == rows["truth"]).mean()
    held = [(rows["state"] == s).sum() for s in (0, 1)]
    print(f"{name}: {max(agree, 1 - agree):.0%} in their true state; state 0 holds {held[0]} rows, state 1 {held[1]}")
```

```text
raw: 53% in their true state; state 0 holds 3000 rows, state 1 0
clipped: 93% in their true state; state 0 holds 1383 rows, state 1 1617
```

On the raw rows one state took the extreme row and never won another, so
every later row went to the other state. Clipped, both states keep their
blobs.

## Check that it worked

Count the rows each state holds over a recent stretch, and, on rows whose
regime you know, the share in their true state, as the recipes do.

| what you see | what it means | what to do |
|---|---|---|
| few rows in their true regime, where a regime shows only over many rows, such as a spread that widens | one row says little about its regime, and the default prior does not say that regimes last | give a sticky `transition` and a `transition_prior` |
| one state holds no rows over a long stretch | an extreme row captured it | clip the features before the bank, give the states with `learn=False`, or increase `precision_prior` |
| every field but `weight_sum`, `settled_frac`, `withheld_reason` and `coef` null | the group has not yet had `warm_rows` rows to seed its states from | read the rows after seeding, or give the states with `means` and `covs` |
| the states split the rows by direction, and never by regime, under the default prior | the regimes differ in their covariance alone, which the default seeding by k-means cannot see | give a sticky `transition` and a `transition_prior` first, as the first recipe does; give `means` and `covs` where that does not suffice |

The fields are defined in [OUTPUTS.md](../OUTPUTS.md#hmm), and every
parameter in `hmm`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.hmm),
whose Notes, under *Limits* and *The transition prior*, give the rules.
[Seeding: a model that places its starting
values](../explanation/warm-up-and-readiness.md#seeding-a-model-that-places-its-starting-values)
says how the states are seeded.
