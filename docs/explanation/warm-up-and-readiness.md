<!-- kind: explanation -->
# Warm-up and readiness

This page explains why a model withholds its first predictions, for
anyone who sets a spec's readiness gates or wonders why a prediction is
null.

**A model learns from its rows from the first on, but it reports a
prediction only once it has learned enough to give one.** It learns every row it does not
[skip](../GLOSSARY.md#the-model-bank-and-what-it-reads) (a row with a
null feature or weight). A row with a null target is still predicted,
but the model learns nothing about that target from it, and `rls` learns
nothing from a row unless every target has a value. Its first fits rest on a few rows, or on a short stretch of time,
so their predictions carry more error than the predictions of a fit that
has learned more. A [gate](../GLOSSARY.md#warm-up-and-readiness) is a parameter of a
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads) (the description
of a model), such as `min_weight`. It withholds the prediction until its
condition holds. On a
[withheld](../GLOSSARY.md#warm-up-and-readiness) row the prediction is
null, and the field `withheld_reason` says why. It names the gate, or a
reason the fit cannot be trusted on that row: `ill_conditioned` for an
`ewridge` solve too near singular to keep a digit, and `unstable_rate`
for an `sgd` learning rate past its stability bound. [Is it
ready?](../DIAGNOSTICS.md#is-it-ready) lists every reason in their order of precedence, and [Decide
when a model is ready](../how-to/decide-when-a-model-is-ready.md) runs the
three main gates on one stream.

**A gate holds back output only.** The rows before a model is
[ready](../GLOSSARY.md#warm-up-and-readiness) are its
[warm-up](../GLOSSARY.md#warm-up-and-readiness), and the model learns
each of them as it learns every other row. Three statistics read from the residuals take the residual of each row
a gate withholds. They are the running spread `sigma` (and so `zscore`),
the scale of the drift detector (`emit_drift`, which flags a shift in
the residuals), and the ranking of grid points that `emit_selected` and
`emit_averaged` read. `emit_selected` picks the [grid point](../GLOSSARY.md#grids) (one
combination of the values a grid lists) with the lowest recent error. The gates'
settings, defaults and fields are listed in MODELS.md's [Warm-up
gates](../MODELS.md#warm-up-gates).

## Three questions, one gate each

"Ready" rests on up to three questions, each answered by a different
number. Every model with a decay answers the first two, and the four
fits that track their estimation variance (`ewridge`, `rls`, `kalman`
and `lasso`) answer the third too:

| the question | the number | the gate |
|---|---|---|
| has the model learned enough weight of rows? | `weight_sum`, the weight behind the prediction | `min_weight` |
| has it come far enough toward its steady state under decay? | `settled_frac`, the share of the steady-state weight reached so far, by the clock | `min_settled_frac` |
| would estimation error swamp the prediction? | `error_inflation`, how much the fit's own uncertainty widens a prediction's error beyond the noise | `max_error_inflation` |

A fourth gate, `warm_rows`, belongs to the models that place their
starting values from their first rows
([Seeding](#seeding-a-model-that-places-its-starting-values)). A fifth
question, how much of each coefficient the data determined, has a field
and no gate ([Identification: a report with no
gate](#identification-a-report-with-no-gate)).

**The questions open at very different speeds, so each gets a number of
its own.** Take four features with a half-life of 50 rows, a tiny ridge
and a row each clock unit. The noise gate (`max_error_inflation`), at about one effective row per
coefficient, opens after about 5 rows, when the fit determines all five
of its coefficients (four slopes and the intercept) and `settled_frac`
is 0.07. `settled_frac` reaches 0.5 on the row at `t = 51`, and 0.99 near the row
at `t = 333`. At an even row rate, `weight_sum` is `settled_frac` times
its ceiling. When the row rate changes, the two move apart, as [Why
`min_weight` is hard to set](#why-min_weight-is-hard-to-set) shows.

## `weight_sum`: the weight behind a prediction

**`weight_sum` is the weight of the rows a model had learned when it made
the row's prediction.** It is read before the row's own [decay](../GLOSSARY.md#time) and before
the model learns the row, so it is 0 on a model's first
row and on its first row after a [reset](../GLOSSARY.md#time). For rows
of weight 1, it runs one behind the row count while nothing is
forgotten. Under a decay, each learned row's weight shrinks as the
[clock](../GLOSSARY.md#time) moves, by half every
[half-life](../GLOSSARY.md#time). So `weight_sum` stops growing at a
ceiling, where each new row only replaces what the decay removes. For
rows of weight 1, `d` clock units apart, under a half-life `h`, each
step multiplies the weight by `q = 2^(−d/h)`, and the ceiling is
`1 / (1 − q)`. At a half-life of 50 that is 72.6 for a row every clock
unit, and 14.9 for a row every five. At a half-life of 600 with rows 0.1
apart it is near 8,657.

**`weight_sum` means the same in every model, so one `min_weight` means
the same for every spec in a [model
bank](../GLOSSARY.md#the-model-bank-and-what-it-reads).** Every model
reads it before the row and before the row's decay, and under a hard
window (a `window_size`) every model counts only the weight inside the
window. A regression model checks `min_weight` against each target's own
weight, the weight of the rows on which that target had a value, so a
target that is often null is ready later than the others. For `rls`,
which learns a row only when every target has a value, it is the weight
of the rows it learned from.

**`weight_sum` is a weight and not a sample size.** Rows of weight 10
count ten times as much as rows of weight 1. And under a decay, the
weight is spread unevenly: the most recent rows carry most of it. Kish's
effective sample size, `n_kish = (Σw)² / Σw²`, counts how many equally
weighted rows the same estimate is worth. Under a steady decay, on rows of equal weight, `n_kish` comes to rest
near twice `weight_sum`, and one row that carries most of the weight
brings `n_kish` close to 1. For a model that keeps a [Gram](../GLOSSARY.md#fitting-a-model) (the
weighted means and co-moments of its columns), such as `ewridge`,
`bank.gram()` reports `n_kish` beside `weight_sum`.

**Every linear model but `ewridge` and `holt` defaults `min_weight` to one
row per unknown: one for each feature, and one for the intercept when it
has one.** `ew_cov`, `ew_class`, `kmeans` and `micro` default to the feature count
plus one, the fewest rows a covariance of that many features needs, and
`holt`, which reads no features, to 1. `ewridge` defaults to 0, because its noise gate, on by default,
withholds those same early rows by measuring how much estimation error
widens a prediction ([The noise
gate](#the-noise-gate-estimation-error-against-the-noise)). The default
of every model is in MODELS.md's [Warm-up
gates](../MODELS.md#warm-up-gates).

## Why `min_weight` is hard to set

**A `min_weight` that is easy to reach at one row rate can be out of
reach at another, and the spec cannot know the rate.** The ceiling of
`weight_sum` depends on how far apart the rows come on the clock. A clock
column says that only as the rows arrive, so no check can refuse an
unreachable `min_weight` when the spec is built. `min_settled_frac` reads
the clock instead of the weight: a model is halfway to its steady state
after one half-life of clock, whatever the row rate. This code fits the
same ridge, under each of the two gates, on a row every clock unit and
on a row every five, and catches the warning the second gate issues:

```python
import warnings

import numpy as np
import polars as pl
import polars_online as po

# The same relation, y = 2x plus noise, with a row every clock unit and with a row every five.
rng = np.random.default_rng(2)
for step in (1.0, 5.0):
    t = np.arange(0.0, 1000.0, step)
    x = rng.standard_normal(len(t))
    frame = pl.DataFrame({"t": t, "x": x, "y": 2 * x + rng.standard_normal(len(t))})
    frame.write_parquet(f"every_{step:g}.parquet")

# One ridge under each gate, at a half-life of 50 clock units.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=50.0)  # a clock needs a gap_cap
specs = [
    po.spec.ewridge("halfway", min_settled_frac=0.5, **common),  # withhold until halfway to the steady state
    po.spec.ewridge("weighted", min_weight=30.0, **common),      # withhold until weight_sum reaches 30
]
for step in (1.0, 5.0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pl.scan_parquet(f"every_{step:g}.parquet").online.fit_predict(specs).collect()
    # Each gate's first predicted row, with the weight behind it.
    for spec in ("halfway", "weighted"):
        rows = out.select("t", pl.col(spec).struct.unnest()).filter(pl.col("pred_y").is_not_null())
        if rows.height == 0:
            print(f"a row every {step:g}, {spec}: never predicts")
            continue
        first = rows.row(0, named=True)
        print(f"a row every {step:g}, {spec}: first predicts at t = {first['t']:g}, weight_sum {first['weight_sum']:.1f}")
    # The remedy each readiness warning gives, its last sentence.
    for w in caught:
        if issubclass(w.category, po.ReadinessWarning):
            print("  warning: " + str(w.message).split(". ")[-1])
```

```text
a row every 1, halfway: first predicts at t = 51, weight_sum 36.8
a row every 1, weighted: first predicts at t = 39, weight_sum 30.3
a row every 5, halfway: first predicts at t = 55, weight_sum 8.0
a row every 5, weighted: never predicts
  warning: Lower min_weight below 14.9, or increase half_life above 103, as https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready explains.
```

At a row every five clock units, `weight_sum` tops out near 14.9, so a
`min_weight` of 30 withholds every row of the stream. `min_settled_frac`
opens on the first row after 50 clock units at either rate. Like `weight_sum`, `settled_frac` is read
before the row's own step, so at a row every clock unit it reaches 0.5
on the row at `t = 51`. The [readiness
warning](../GLOSSARY.md#warm-up-and-readiness), `ReadinessWarning`, gives
the ceiling at the row rate seen so far, and the `half_life` that would
reach 30. That half-life comes from inverting the ceiling: for rows of weight 1, `d` clock units apart, a `min_weight` of `N` is
reachable when `h > −d / log2(1 − 1/N)`. The warning rounds that bound
up.

**The warning waits until the model has had time to reach its ceiling.**
It comes once the model is within 5% of its steady state and the gate has
withheld every row for a further half-life. Before that, a low
`weight_sum` says only that the stream is young. A warning names the
gate, the spec and the target, and it projects the ceiling from the rows
seen so far. Rows that come faster or weigh more lift the ceiling, so it
promises nothing of them. [Is it ready?](../DIAGNOSTICS.md#is-it-ready)
gives each form of the warning, and what to do about each.

## `settled_frac`: how near the steady state

**`settled_frac` is the share of its steady-state weight that a model has
reached, read from the clock alone.** With `T` the clock time the model
has decayed over since its first row or its last reset (each step capped
at `gap_cap`), it is `1 − 2^(−T / half_life)`. It is 0.5 after one
half-life, 0.75 after two and 0.875 after three, whatever the row rate.
At a half-life of 50 clock units, rows 0.5, 1 and 5 units apart give
ceilings of 144.8, 72.6 and 14.9, and the same `settled_frac` at every
time. Where nothing decays, there is no steady state to approach, and
`settled_frac` is null.

**`min_settled_frac` is off by default, because the risk it guards
against depends on the process, and no statistic inside a model can size
it.** That risk is that the rows so far do not represent the stream.
Forty rows of one regime, under a half-life chosen to average across
several regimes or seasons, give a fair fit of the wrong mixture. So set
`min_settled_frac` when you know the process has regimes or seasons that
the half-life was chosen to average over. A value of `0.5` withholds
predictions until the model has one half-life of history.

**For `ewridge` and the other models that keep a Gram in mean form, a
young fit is only noisier.** They keep the means and co-moments of their
columns, divided by the weight behind them, so they do not lean toward a
starting value. On a stationary stream a fit of 40 rows is as unbiased as
a fit of 4,000, apart from the ridge's own shrinkage, and for `ewridge` and `lasso` the noise gate measures how much noisier
it is ([The noise
gate](#the-noise-gate-estimation-error-against-the-noise)).

**The gradient models, `sgd`, `pa` and `ftrl`, lean toward their start
while they are young.** They start their coefficients at 0 (or at its
projection onto a constraint) and move them a step a row. How far a fit
has come from that start depends on the steps it has taken and its
learning rate, which neither the clock nor `settled_frac` measures. Their
default `min_weight` withholds their first rows, one row per unknown.

## The noise gate: estimation error against the noise

**`error_inflation` is the factor by which the fit's own estimation error
widens a prediction's error beyond the noise, and `max_error_inflation`
withholds a prediction while it is too large.** For noise of constant
variance, independent across rows, and a fit linear in the past targets
(as a ridge regression is), the variance of a prediction at features `x`
has an exact form. With `q` each row's
decay to the present and `w` its weight:

```
Var(ŷ(x)) = σ² · h(x),    h(x) = x' G⁻¹ G₂ G⁻¹ x
G  = Σ qᵢ wᵢ xᵢxᵢ' (+ the ridge)     the Gram the model keeps, written here as sums
G₂ = Σ qᵢ² wᵢ² xᵢxᵢ'                 the same sum, with decay and weight squared
error_inflation = √(1 + h(x))
```

The noise variance `σ²` scales both the noise and the estimation error,
so the ratio needs no estimate of the noise. If the spread of the features stays the same across the rows the decay
keeps, `G₂` is `G` times a number. The statistic then becomes
`h(x) = x' Σ̂⁻¹ x / n_kish`, with `Σ̂` the ridged Gram in mean form that
the model already keeps.
Averaged over new rows whose features are drawn like the learned rows',
`h` is `edf / n_kish`:
the effective degrees of freedom the fit uses, over Kish's effective
sample size behind it. With a tiny ridge `edf` is the number of
coefficients, the intercept included, so the gate asks for enough
effective rows per coefficient.

**The gate reads the average, which adds no solve to a row.** `edf` is
read from the factor each solve computes, once a solve and never a row,
and a prediction passes the gate while `√(1 + edf / n_kish)` is below
`max_error_inflation`.

**`emit_error_inflation` writes each row's own factor, when asked.** A
row's own `h(x)` takes a triangular solve, about `k²/2` multiply-adds for
`k` coefficients, so `emit_error_inflation` writes it only when asked.
That per-row factor is large for a row that leans on a direction the
data never showed, such as a row that breaks a collinearity the fit
relied on.

**The average tracks the error it predicts, and the per-row form errs on
the side of withholding.** On a stationary stream at `n_kish` near 2.3
times the coefficients, the observed out-of-sample error was 1.214 times
the noise, and the gate's average read 1.197, within 2%. The mean of the
per-row factor read 1.315 there. The per-row `h(x)` overstates the exact
estimation variance on short memories, by 10% at 12 effective rows a
coefficient and by 54% at 2.3, because the recent rows that dominate
`G₂` are the ones the fit has already bent toward.

**The gate reads Kish's count because the noise in a fit follows how
evenly its weight is spread.** One row that carries most of the weight
makes a fit of barely one row, however large `weight_sum` is. Adding a
feature increases `edf`, so the gate holds predictions back longer for a
larger fit with no setting to change.

**`ewridge`'s default, `√2`, withholds a prediction while estimation error is at least the noise.** That is the rule of one row per unknown that
the regressions' default `min_weight` follows. With `weight_sum` equal to the number of coefficients (the features
plus the intercept), on rows of weight 1 before the decay has removed
much, `edf / n_kish` is near 1, and the inflation near `√2`.
`rls`, `kalman` and `lasso` compute the statistic too, and their gate is
off unless set. `kalman` reads each row's exact prior variance,
`z' P z / R` with `P` its coefficients' covariance and `R` the noise, so
its gate reads each row's own factor. `lasso` counts its active
coefficients as its degrees of freedom. The other models refuse the
setting. A Huber or quantile fit is not linear in its targets, so its
variance takes another form, and a gradient model such as `sgd` keeps no
second moments to read one from. `min_weight` and `min_settled_frac` gate
them instead.

## Identification: a report with no gate

**`support_coef` reports how much of each coefficient the data determined,
and how much the ridge did.** `ewridge`, `huber` and `quantile` write it
beside `coef`. With the ridge in mean form, the solve's system is
`S + ridge·I`, with `S` the Gram in mean form without the ridge, and the
diagonal of `S (S + ridge·I)⁻¹` is each coefficient's data share, between
0 and 1. A duplicated pair of features reads 0.5 each, and a feature
that stops moving slides toward 0 as the decay forgets its spread. The
intercept has no share. The shares, plus 1 for the intercept where there
is one, sum to `edf`, so they split the noise gate's statistic
coefficient by coefficient.

**No gate reads it, because a fit that cannot tell two features apart
still predicts well.** On a stream whose noise has a spread of 0.100, a
fit with one feature duplicated predicted with an error of 0.104, the
same as the clean fit. It split the coefficients 2 and 2, where the
stream was made with 1 and 3. Only the rows on which the duplicate breaks
away from its twin were predicted badly: 1.022. That is a coefficient
problem and a risk for the rows to come, which the per-row
`error_inflation` catches on the row itself. So a readiness warning
names a coefficient that the ridge determines more than the data, once,
and withholds nothing. It waits for a row whose prediction the gates let
through, and for the stream to be 95% of the way to its steady state.
Before then, the first rows' noisy spreads read some shares below 0.5
that the fit at its steady state reads above it.

## Seeding: a model that places its starting values

**A model that seeds itself holds its first `warm_rows` rows, places its
starting values from them, and then learns them.** `kmeans` places its centers, and `hmm` its hidden states and `bocpd` its
prior where you do not give them. Until seeding, the model has nothing to predict from, so its
rows are withheld: where `min_settled_frac` does not withhold them, the
first rows as
`below_min_weight` until its weight reaches `min_weight`, and the rest as
`below_warm_rows`. Once its
starting values are placed, it replays the held rows into them and frees
them, so its memory use grows with its state alone from the row that
completes seeding. `kalman` takes no `warm_rows`, but withholds a
target's predictions as `below_warm_rows` until a row has sized that
target's prior.

**`kmeans` seeds from the best of ten starts, because one start can leave
two clusters under one center for good.** A center that owns two
clusters holds all their rows inside its own radius. The split-merge move
(`split_merge`), which frees a center for a new cluster, acts on rows far
outside every center's radius, so it never sees them. Plain k-means++
places each center at a row drawn by its squared distance from the
centers so far. `init="k-means++"`, scikit-learn's greedy variant, draws
several candidates for each center and keeps the one that leaves the
rows closest to their centers. On five clusters in four dimensions, one
greedy start, run to its fixed point, still ended in a partition 1.87
times as costly as the right one in 9 of 100 seeds (28 of 100 for plain
k-means++). So the default, `init="lloyd"`, runs Lloyd's passes from
each of ten greedy k-means++ starts until no row changes center, and
keeps the start whose rows sit closest to their centers.

**On the public clustering benchmarks of Fränti and Sieranoja, the ten
starts miss a cluster in fewer orders than plain starts, and in none on
S1 and S2.** Each of the sets
S1, S2, A1, A2 and A3 was fed in 50 orders. S1 and S2 hold 15 clusters
each. A1, A2 and A3 hold 20, 35 and 50, so the 500 held rows give each
cluster only 10 to 25 rows, and many orders still missed a cluster and
split another. The plain starts are ten Lloyd's passes from each of ten
plain k-means++ starts, under which the A-sets' median order ended 1.26
to 1.35 times the optimum:

| set | orders missing a cluster, the default | orders missing a cluster, plain starts | the median order's cost over the optimum, the default |
|---|---:|---:|---:|
| S1 | 0 of 50 | 10 of 50 | 1.001 |
| S2 | 0 of 50 | 13 of 50 | 1.001 |
| A1 | 10 of 50 | 39 of 50 | 1.009 |
| A2 | 30 of 50 | 50 of 50 | 1.101 |
| A3 | 44 of 50 | 50 of 50 | 1.104 |

**Each group pays for its seeding once, at its `warm_rows`-th row, so a
bank of many groups pays it many times.** At the slowest time measured
for the default, 91 ms, 10,000 groups at 400 features spend up to a
quarter of an hour seeding. The times are for the default `warm_rows` of
500 and `n_clusters=8`, on an Apple M4 Pro. The ten starts were timed on
one thread under a heavier load than the single starts, whose times are
medians over 200 and 40 runs in one native process, so the two compare
only roughly:

| seeding | features | clusters | time | the machine's load average |
|---|---:|---|---:|---:|
| `"lloyd"`, the default | 400 | separated, a start reaching its fixed point in a few passes | 56 ms | about 110 |
| `"lloyd"`, the default | 400 | overlapping | 91 ms | about 110 |
| `"lloyd"`, the default | 20 | | 1.9 ms | about 110 |
| `"k-means++"`, four candidates a center, placing the centers | 400 | | 2.8 ms | under 15 |
| `"k-means++"`, placing the centers and replaying the 500 rows | 400 | | 3.9 ms | under 15 |
| plain k-means++, one candidate a center, placing the centers | 400 | | 0.8 ms | under 15 |
