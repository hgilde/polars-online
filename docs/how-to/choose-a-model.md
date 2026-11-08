<!-- kind: how-to -->
# Choose a model

This how-to is for anyone about to write a
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the description
of a model) who wants to know which of the twenty-two models to name in
it, including which model that watches for change answers which
question.

## Prerequisites

These steps need the library and a question to put to a stream:

- polars-online installed, and the README's
  [Example data](../../README.md#example-data) built. The examples read
  `df`, 400 rows a minute apart. In `df`, `y` is `1 + 2·x0 − x1` plus
  noise, and `ret`'s slope on `signal_a` is 0.8 on rows 0 to 199 and −0.8
  from row 200. Two examples make their own data.
- numpy, for the clustering example's points.
- The question you want the stream to answer, such as *what will the
  target be?* or *has something changed?*

## Choose the family, then the model

**Choose the family by the question you are asking, then the model by what
your data need.** polars-online's models fall into four families, and each
family answers one kind of question about a stream. Each model has an
entry in [MODELS.md](../MODELS.md): when to reach for it, its update equations,
its parameters, a recipe on data whose answer is known, and what it is
checked against. That page opens with the notation the update equations share.

1. **Find your question in the table, and read its family.**
2. **Start with the model the table names.**
3. **Change to another model of the family when your data match its
   case** in the last column.
4. **Look the model up in [MODELS.md](../MODELS.md)** for its parameters,
   and in its builder's reference page, such as the one for
   [`po.spec.ewridge`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ewridge), for its keywords and their defaults.

| your question | the family | start with | change to another when |
|---|---|---|---|
| what will the target be? | [Linear models](../MODELS.md#linear-models) | `ewridge` | the coefficients drift (`kalman`); some targets are wild (`huber`, `quantile`); few features matter (`lasso`, `ftrl`); learning a row must take O(k) time for `k` features (`sgd`, `pa`); the target is a count (`sgd`) or 0/1 (`ftrl`); you want a baseline with no features (`holt`) |
| how do these columns vary and move together? | [Moments and correlation](../MODELS.md#moments-and-correlation) | `ew_cov` | there are thousands of columns (`marginal`); one correlation must stand for a whole matrix (`deco`); the prices are noisy ticks (`rcov`); you want to know what the columns hold (`audit`) |
| which group does this row belong to? | [Clustering and classification](../MODELS.md#clustering-and-classification) | `kmeans` | the clusters have any shape (`micro`); you have the labels to learn from (`ew_class`) |
| has something held, or changed? | [Sequential tests and regimes](../MODELS.md#sequential-tests-and-regimes) | `seqtest` | the correlations may have changed (`corrchange`); you want the regime's age (`bocpd`) or which regime it is (`hmm`) |

A spec's clock, grouping and warm-up mean the same whichever model it
names. Where a model takes a half-life, it is a half-life on the clock, as
[Time and decay](../explanation/time-and-decay.md) explains.
[OUTPUTS.md](../OUTPUTS.md) lists every field a model writes, and
[TESTING.md](../TESTING.md) what each is checked against.

## Predict a number with a linear model

**A linear model predicts a number, and a model whose coefficients drift
follows a relationship that breaks.** `ewridge`, with every row counted
the same, averages the two slopes of `ret` on `signal_a`. `kalman` follows
the new one, as far as the afternoon's noisier rows let it.

This code uses `df` from [Example data](../../README.md#example-data):

```python
common = dict(targets=["ret"], features=["signal_a"], clock="t", gap_cap=10.0)
specs = [
    po.spec.ewridge("ridge", half_life=float("inf"), **common),                # every row counts the same
    po.spec.kalman("kalman", half_life=50.0, coef_half_life=20.0, **common),   # a coefficient that may drift
]
bank = po.ModelBank(specs)
bank.fit(df)                                         # learn every row, keep no output
for r in bank.coef().filter(pl.col("term") == "signal_a").iter_rows(named=True):
    print(f"{r['spec']}: slope at the last row {r['coef']:.2f}")
```

```text
ridge: slope at the last row 0.10
kalman: slope at the last row -0.66
```

## Track how columns move together

**A moments model tracks how columns vary and move together, and computes
each row's statistics from the rows before it.** So a row's correlation
can serve as a feature for that row without leaking the row's own values.
Here the correlation of `signal_a` and `ret` changes sign after the change
of slope at row 200.

This code uses `df` from [Example data](../../README.md#example-data):

```python
moves = po.spec.ew_cov("moves", features=["signal_a", "ret"], clock="t", gap_cap=10.0,
                       half_life=50.0, stats=["corr"])
corr = df.online.fit_predict([moves])["moves"].struct.field("corr_signal_a_ret")
print(f"corr(signal_a, ret) at row 199: {corr[199]:.2f}, at row 399: {corr[399]:.2f}")
```

```text
corr(signal_a, ret) at row 199: 0.41, at row 399: -0.33
```

## Label rows by their cluster

**A clustering model labels each row with its nearest cluster before
learning from the row, and moves the clusters as the stream runs.** Here
`kmeans` finds three clusters planted at (0, 0), (6, 0) and (0, 6). The
code makes its own points, since `df` has no clusters:

```python
import numpy as np

# 3,000 points around three planted centers, saved.
rng = np.random.default_rng(3)
centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])             # three clusters, planted
points = centers[rng.integers(0, 3, 3000)] + rng.standard_normal((3000, 2))
pl.DataFrame(points, schema=["a", "b"]).write_parquet("points.parquet")

# k-means over the file, and the centers it found.
clusters = po.spec.kmeans("clusters", features=["a", "b"], n_clusters=3, half_life=float("inf"))
bank = po.ModelBank([clusters])
bank.fit(pl.scan_parquet("points.parquet"))
found = bank.coef()["coef"].to_numpy().reshape(3, 2).round(1)       # one row of (a, b) per center
print("centers found:", sorted(found.tolist()))
```

```text
centers found: [[0.0, -0.1], [0.0, 6.0], [6.0, 0.0]]
```

## Detect changes and regimes

**Four models watch a stream for change, and each answers its own
question.** Each answer is read before its row is learned from, as every
prediction is, so it can drive a decision on that row.
[REGIMES.md](../REGIMES.md) measures what each finds on streams whose
answer is known, and what it misses. Their update equations and parameters are
under [Sequential tests and
regimes](../MODELS.md#sequential-tests-and-regimes).

1. **Find your question in the table, and name its model in a spec.**
2. **Fit the spec, and read the field the table names on each row.**

| the question | the model | what it writes on each row |
|---|---|---|
| does a column tend to be positive, or does one spec predict closer than another? | [`seqtest`](../MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting) | the log wealth of a bet on each side; past `ln 20` is the 5% level, however often you look |
| have the correlations among these columns changed? | [`corrchange`](../MODELS.md#corrchange--has-the-correlation-structure-changed) | a test statistic, its critical value, and a flag where one passes the other |
| how long has the current regime lasted? | [`bocpd`](../MODELS.md#bocpd--how-long-has-this-regime-lasted) | `run_mode`, the most likely number of rows since the regime began |
| which regime is the stream in? | [`hmm`](../MODELS.md#hmm--which-regime-are-we-in) | the probability of each hidden state, and the most probable one |

**Read a regime's age from `bocpd`'s `run_mode`.** Its alarm, `p_change`,
scores each row on its own, so it barely moves when the spread grows 2.7
times: no single row of the wider regime is surprising enough on its own.

**Tell `hmm` that regimes last.** One return says little about which regime
it came from. A sticky `transition` matrix, counted by `transition_prior`
as 500 rows, lets the filter weigh each row against what the rows before
it said. [Keep `hmm` states on their
regimes](keep-hmm-states-on-their-regimes.md) sets the two.

This code makes two series whose returns take turns, 500 rows at a time,
between a calm regime and one 2.7 times as wide. It uses `po.sim.regimes`,
which returns the regime each row was drawn in beside the rows. `bocpd`
dates each switch, and `hmm` says which regime each row is in:

```python
import polars as pl
import polars_online as po

# Make the data, and save it with its truth.
sim = po.sim.regimes(                                         # two series, and the regime each row was drawn in
    2, states=[0.3, 0.3], transition=[[0.0, 1.0], [1.0, 0.0]],   # the two regimes take turns, a block each
    n_blocks=6, rows_per_block=500,
    scale_state=[0.0, 1.0],                                   # regime 1's returns are e = 2.7 times as wide
    seed=3,
)
truth = sim["truth_rows"].select("t", regime="state")
sim["rows"].join(truth, on="t").write_parquet("regimes.parquet")
returns = pl.scan_parquet("regimes.parquet").with_columns(pl.col("x_1", "x_2").diff())   # from the levels

# One model dates each switch, the other names the regime.
age = po.spec.bocpd("age", features=["x_1", "x_2"], hazard=500.0)   # how long has this regime lasted?
which = po.spec.hmm(                                          # which regime is this row in?
    "which", features=["x_1", "x_2"], n_components=2, half_life=float("inf"),
    precision_prior=1e-2, warm_rows=500,
    transition=[0.998, 0.002, 0.002, 0.998], transition_prior=500.0,   # a regime lasts about 500 rows
)
out = returns.online.fit_predict([age, which]).collect().with_columns(
    began=pl.col("t") - pl.col("age").struct.field("run_mode"),   # the row the current regime began on
    guess=pl.col("which").struct.field("state"),
)
# Score both against the planted switches and regimes.
for switch in (500, 1000, 1500, 2000, 2500):
    began = out.filter(pl.col("t") == switch + 50)["began"][0]
    print(f"a switch at row {switch}: 50 rows later, bocpd dates the regime to row {began}")
predicted = out.filter(pl.col("t") >= 500)                    # hmm seeds its states from the first 500 rows
agree = (predicted["guess"] == predicted["regime"]).mean()    # its states are numbered as seeded, so read either way
print(f"hmm puts {max(agree, 1 - agree):.0%} of the rows after seeding in their true regime")
```

```text
a switch at row 500: 50 rows later, bocpd dates the regime to row 497
a switch at row 1000: 50 rows later, bocpd dates the regime to row 1000
a switch at row 1500: 50 rows later, bocpd dates the regime to row 1501
a switch at row 2000: 50 rows later, bocpd dates the regime to row 1998
a switch at row 2500: 50 rows later, bocpd dates the regime to row 2501
hmm puts 93% of the rows after seeding in their true regime
```

Fifty rows after each switch, `bocpd` dates the new regime to within three
rows of the truth. Without `transition` and `transition_prior`, the filter
puts 58% of the rows in their true regime.

## Check the choice against a baseline

**Test whether the model you chose predicts closer than a simpler one,
with a sequential test you may read at any row.** Here `seqtest` asks
whether a regression on `x0` and `x1` predicts `y` closer than `holt`,
which reads only `y`'s own past. Its evidence passes the 5% level a few
dozen rows into `df`. [Compare two models](compare-two-models.md) gives
the other tests between two models.

1. **Fit the model you chose and a baseline in one bank.**
2. **Add a `seqtest` spec whose `a` and `b` name the two specs.**
3. **Read the first row where `log_e_a_<target>` reaches `ln 20`,** the
   5% level however often you look.

This code uses `df` from [Example data](../../README.md#example-data):

```python
import math

# Two predictions of y, and a test between them.
common = dict(targets=["y"], clock="t", gap_cap=10.0, half_life=100.0)
specs = [
    po.spec.ewridge("ridge", features=["x0", "x1"], **common),
    po.spec.holt("holt", **common),                                    # y's own level and trend, no feature
    po.spec.seqtest("closer", targets=["y"], a="ridge", b="holt"),     # does ridge predict closer?
]
out = df.online.fit_predict(specs)
passed = out["closer"].struct.field("log_e_a_y") >= math.log(20)    # the 5% level, however often you look
print(f"ridge predicts closer than holt at the 5% level from row {passed.arg_true()[0]}")
```

```text
ridge predicts closer than holt at the 5% level from row 48
```
