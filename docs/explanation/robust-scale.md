<!-- kind: explanation -->
# The robust scale and the refit of the first rows

This page explains, for anyone who fits `huber`, `quantile` or
`sgd(loss="huber")`, how these models measure the scale of their residuals,
why they refit each target's first rows, and what a robust scale gives up
when the target's level shifts.

**A robust fit needs two parts: a loss whose pull stops growing at a cut,
and a scale for the cut that rows far off the fit cannot widen.** Least squares lets one row that misses by 50 pull on the fit as hard as
100 rows that miss by 0.5, and count in its loss as much as 10,000 of
them. Huber's loss caps a row's pull at a cut of `huber_delta` times the
residuals' scale. That cut catches the outliers (the rows whose target
misses far, which a robust fit is for) only if they cannot widen the
scale it is drawn in. So `huber` and `sgd(loss="huber")` draw their cut
in a robust scale, and `quantile` draws its band there (the width around
its fit inside which it treats a row as least squares). By default, and
always for `quantile`, that scale is
the MAD (median absolute deviation): the exponentially weighted median of
the absolute [residuals](../GLOSSARY.md#fitting-a-model), over
`Φ⁻¹(3/4) = 0.6745` so that it is the standard deviation when the errors
are normal.

**In a stream, the scale also has to survive the first rows, whose fit is
poor.** So once a target has learned a set number of rows, the model
fits those rows again to convergence and rebuilds the scale from that
fit's residuals: a [refit](../GLOSSARY.md#fitting-a-model).

`huber_delta`, `huber_scale`, `huber_scale_half_life`, `huber_refit_rows`,
`quantile_refit_rows` and `refit_budget` are defined in the docstrings of
[`po.spec.huber`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.huber),
[`po.spec.quantile`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.quantile)
and [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd).
Each model's update is in its entry in MODELS.md:
[`huber`](../MODELS.md#huber--robust-regression),
[`quantile`](../MODELS.md#quantile--robust-regression) and
[`sgd`](../MODELS.md#sgd--stochastic-gradient-descent).

## Huber's loss, and the weights it implies

**Huber's loss is quadratic for a small miss and linear for a large
one.** With `u = e/σ` the residual `e` in units of a scale `σ`, and `δ`
the cut `huber_delta`:

```
ρ(u) = u²/2           |u| ≤ δ
     = δ|u| − δ²/2    |u| > δ
ψ(u) = ρ'(u) = u inside the cut, ±δ beyond it
```

`ψ` is a row's pull on the fit, and it stops growing at the cut. Writing
`ψ(u) = w(u)·u` with `w(u) = min(1, δ/|u|)` turns the fit into weighted
least squares, each row at the weight `w` of its own residual. A batch
fit iterates: fit, compute the residuals, reweight, and fit again, until
the coefficients stop moving. The default cut, `δ = 1.345`, keeps 95% of
least squares' efficiency when the errors are normal: with `Z` standard
normal, the efficiency is `(2Φ(δ) − 1)² / E[ψ(Z)²]`.

**In a stream, each row is weighted once, by its residual against the fit
before the row.** That keeps the weight
[out-of-sample](../GLOSSARY.md#fitting-a-model), and it lets `huber` and `quantile`
keep [running sums](../GLOSSARY.md#fitting-a-model) in place of their
rows, apart from the first rows the refit holds (`sgd` keeps no sums of
the rows: it moves its coefficients a step a row). So the online fit is
one pass of the batch iteration, from a fit that improves as it goes: the first
rows get weights from a poor fit, and the later rows from a good one.

## Why the scale must be robust

**A root mean square of the residuals is no robust scale: the outliers
widen it, and the cut drawn in it lets them through.** Each cell of the
table is the median slope over five seeds of `y = 1000 + 2x + 0.5ε`,
20,000 rows with no [decay](../GLOSSARY.md#time), fitted in one pass (`huber_refit_rows=0`). A
share of the rows with `x` above 0 is 50 too high, so they tilt the
slope. statsmodels' `RLM`, its robust linear model, is the batch Huber
fit iterated to convergence:

| outliers | least squares | `huber_scale="mad"` (the default) | `huber_scale="rms"` | statsmodels' `RLM`, MAD scale |
|---|---|---|---|---|
| none | 1.999 | 1.999 | 2.000 | 2.000 |
| 1% | 2.415 | 2.008 | 2.057 | 2.008 |
| 5% | 4.050 | 2.040 | 2.675 | 2.039 |
| 12.5% | 6.975 | 2.124 | 4.882 | 2.124 |

The root mean square fails twice in one pass. A fit's first residuals are
the target's whole level, about 1,000 while the intercept is still near
0, and without decay they stay in the scale for good. And the outliers
inflate the scale that is meant to catch them. The median leaves the
first residuals behind once the fit has the level, and an outlier moves a
median by its weight, whatever its size. So the MAD holds until half the
weight is on outliers.

**The small bias left at 1% and 5% is Huber's own.** An outlier is still
learned, at about `1.345 · 0.5 / 50 ≈ 0.013` of an ordinary row's
weight, and hundreds of them add up. A fit of the median, which takes
only the sign of a row's miss beyond its band, is biased the same way: in
the example of [`huber`'s entry](../MODELS.md#huber--robust-regression),
it lands at 2.03 against a true slope of 2.

## How a stream keeps a median

**While a target's refit waits, its MAD is read exactly, from the
residuals the refit holds.** Each held row's weight relative to the
others is fixed once the row is held, since every decay after it applies
to all of them alike. So the residuals sit in a tree ordered by size, at
those weights, and reading the median is a walk down it, whose time grows
with the logarithm of the rows held.

**From the refit on, and where the refit is off or a target gives it up,
the median is read
from a quantile sketch kept per target, within 0.78% of the exact
weighted median.** The sketch splits each octave, a factor of two, into
64 buckets of equal width. Its buckets span at most 16 octaves around the
median, a factor of 65,536. A residual beyond them is folded into the
outermost bucket on its side, its weight kept, so a wild residual of
`1e200` takes no more memory than any other. The sketch holds at most
about 9 KB per target.

**The 0.78% holds until the median moves past a bucket a residual was
folded into.** That happens when a fit's first residuals, at the target's
level, are more than 65,536 times the noise's. The median then reads too
high until the later residuals outweigh them.

**The sketch adds 14 to 32 ns per target per row.** Against `"rms"`, with 20
features and 10 targets over 100,000 to 200,000 rows on an Apple M4 Pro
under other load, it added 14 to 17 ns per target per row to `huber`. It added 20 to 32 ns to `sgd(loss="huber")`, whose row takes less time
than `huber`'s: at 20 features the 32 ns were about half of that row.

**With the refit on, a target in another unit gives the same scale and
fit, to rounding.** Only a power of two maps the sketch's bucket edges
onto themselves. So a target in grams where it was in kilograms regroups
the residuals across bucket edges, and the median can move by up to a
bucket, 1.6%, where it should move by the factor. The refit gives the
sketch a unit that scales with the residuals: the median of its
converged residuals, an order statistic of them. Before the refit, the
exact median needs no unit. For `huber` and `quantile`, this was measured at levels up to `1e9`, with a first residual `1e-6` of
the noise. With `u` the unit roundoff, the worst gap in the scale and the slope, in
units of the noise's spread, was 10 `u·|y|max` under a factor of 3, and
178 `u·|y|max` under a shift of the level.

**No unit is taken from the residuals as they arrive, because the first
residuals may all be small beside the rest, and they carry the rounding
of the target's level.** A sketch anchored at its first
residual would carry that residual's rounding at the target's level into
every read, up to a bucket apart in two units for the life of the sketch. So
with the refit off, the sketch keeps fixed buckets, and the scale in two
units differs by up to a bucket. Over 50 streams of 1,500 rows with
heavy-tailed noise and a [half-life](../GLOSSARY.md#time) of 300 rows, that moved `huber`'s
predictions by up to 0.4% of the target's spread, with the target
multiplied by 1e±3 and 1e±6. A quantile fit, whose band is drawn in the scale, differs by more: on
the same streams, by up to 4.4% of the target's spread. A power of two moves nothing.

**Until a target has a scale above 0, no row is down-weighted.** That is before its first residual, or, under the MAD, while half the
weight or more is on residuals of exactly 0 (under `"rms"`, while every
residual so far is 0). There is then nothing to judge an outlier against. A scale taken as 1
would be a cut in the target's own units, which would down-weight the
first rows of a target in millions and none of one in millionths.

## The streams one pass leaves captured

**A median over a few streams hides the streams whose first rows the
outliers capture.** A stream is *captured* here when its final slope
ends more than 0.05 from `RLM`'s on the same rows. At 12.5% outliers, the median stream of 20,000 rows
ended within 0.002 of statsmodels' `RLM` in one pass. Over 40 streams, 13
ended more than 0.05 from it in slope, the worst by 0.43.

**The scale is what lets the outliers capture a stream.** Over the first rows the fit is close to least squares, so the outliers
pull it.
The MAD then learns the residuals of a fit the outliers have pulled. So
the cut stays wide, the outliers keep entering at nearly full weight, and
without decay the fit stays where they pulled it.

This code fits one such stream of 2,000 rows, at about
12.5% outliers, with the refit off and with its default:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 1000 + 2x plus noise, with about 12.5% of the rows 50 too high, all where x is above 0.
rng = np.random.default_rng(6)
n = 2_000
x = rng.standard_normal(n)
y = 1000.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
wild = (rng.random(n) < 0.25) & (x > 0)
y[wild] += 50.0
pl.DataFrame({"x": x, "y": y}).write_parquet("captured.parquet")

# Least squares, and huber in one pass and with its refit, every row weighed the same.
common = dict(targets=["y"], features=["x"], half_life=float("inf"))
specs = [
    po.spec.ewridge("least_squares", **common),
    po.spec.huber("one_pass", huber_refit_rows=0, **common),  # no refit
    po.spec.huber("refit", **common),                         # the first 500 rows refit, the default
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("captured.parquet"))
# Each fit's slope, beside the truth.
print(f"outliers: {wild.mean():.1%}; the true slope: 2")
for r in bank.coef().filter(pl.col("term") == "x").iter_rows(named=True):
    print(f"{r['spec']}: slope {r['coef']:.2f}")
```

```text
outliers: 12.6%; the true slope: 2
least_squares: slope 6.87
one_pass: slope 3.01
refit: slope 2.12
```

The one pass holds most of the outliers' pull off, and still ends a full
point from the truth. The refit lands on 2.12. Run on the same rows,
statsmodels' `RLM` gives 2.13, so the 0.12 left is Huber's own bias at
this share of outliers.

## The refit of the first rows

**On the row a target's count of learned rows reaches
`huber_refit_rows`, `huber` fits those rows again by the batch
iteration, to convergence.** Then it rebuilds the target's running sums
from the rows at their converged weights, and its scale from the
converged fit's residuals, and the stream goes on online from that fit.
The refit runs after the row's own prediction, moves its own target's
fit alone, and leaves the schedule of its
[solves](../GLOSSARY.md#fitting-a-model) (`solve_every`) as it was. `quantile` refits at
`quantile_refit_rows` by Newton steps on its smoothed check loss at its
band, and rebuilds its sums as the stream would take each row predicted
by the converged fit. `sgd(loss="huber")` keeps no sums: the converged
fit replaces its coefficients, and its scale is rebuilt.

**The default count is `max(500, 10·k)` rows for `k` features, so that the refit has rows enough for its
coefficients.** Above 100 features the refit is off by default, since a
refit of `10·k` rows would run for seconds a target there. A count you
give is taken at any number of features, up to 2²⁰ rows. `quantile`
refuses a count above 0 but below three rows a coefficient, the first rows it fits as ordinary least squares.

**Rebuilding the scale is what makes the refit hold.** A refit of the
coefficients alone, over the first 200 rows of a captured stream, landed
on `RLM`'s slope, and was pulled back within 300 rows. Its scale still
held the pulled fit's residuals, so the cut stayed wide. Refitting the
scale with the coefficients held the fit within 0.003 of `RLM` to the end
of the stream.

**The refit is of each target's first rows, because a refit of trailing
rows would free no captured fit.** Passes over the most recent rows leave
the first rows' weights, and the scale they set, as they were. Measured
in a simulation of the recursion, three extra passes over a trailing
buffer of 256 rows closed less than half of the gap to the batch fit at
12.5% outliers.

**The refit takes `huber` from the first poor fit to the batch fit.**
Each cell counts, over 40 seeds of the stream of [Why the scale must be
robust](#why-the-scale-must-be-robust), the streams whose final slope is
more than 0.05 from `RLM`'s, then more than 0.25, and gives the worst
distance:

| outliers | 20,000 rows, `huber_refit_rows=0` | 20,000 rows, 500 (the default) | 2,000 rows, 0 | 2,000 rows, 200 | 2,000 rows, 500 | 2,000 rows, 1,000 |
|---|---|---|---|---|---|---|
| 5% | 0 and 0 of 40; worst 0.014 | 0 and 0; 0.001 | 4 and 0; 0.17 | 0 and 0; 0.004 | 0 and 0; 0.003 | 0 and 0; 0.003 |
| 10% | 6 and 0; 0.19 | 0 and 0; 0.004 | 12 and 6; 0.49 | 0 and 0; 0.008 | 0 and 0; 0.006 | 0 and 0; 0.005 |
| 12.5% | 13 and 2; 0.43 | 0 and 0; 0.007 | 15 and 13; 0.97 | 0 and 0; 0.013 | 0 and 0; 0.009 | 0 and 0; 0.007 |
| 15% | 15 and 12; 1.08 | 0 and 0; 0.012 | 20 and 18; 2.33 | 1 and 0; 0.054 | 0 and 0; 0.016 | 0 and 0; 0.019 |
| 20% | 25 and 23; 4.30 | 4 and 0; 0.14 | 29 and 24; 6.68 | 18 and 5; 5.76 | 10 and 1; 0.32 | 3 and 0; 0.15 |

With a half-life of 5,000 rows over 20,000 rows, the refit took the
captured streams at 15% and 20% from 15 and 26 of 40 to 0 and 2. Over
20,000 rows, a refit of 200 rows left the same 4 streams at 20% as 500
do, the worst by 0.18. Over 2,000 rows at 20% outliers, the default of 500 rows left 10 streams
more than 0.05 off, where 200 rows left 18 and 1,000 rows left 3.

**At 20% outliers the refit meets the edge of Huber's breakdown.** At 20% they are 40% of the rows on their side of `x`. The refit
weighs each row first by its residual from the one-pass fit they have
already pulled, so the refit can itself be captured. Over 20,000 rows, a refit of
1,000 rows left 2 of 40 streams more than 0.05 from `RLM`, none more
than 0.25.

**The refit brings `sgd(loss="huber")` near `RLM` too.** Its one pass
starts from zero slopes, and over 2,000 rows it left 28 of 40 streams
captured at 20% outliers (the worst by 3.6) and 4 at 10% (the worst by
0.069). Over 20,000 rows at 20% it left 31, the worst by 1.6. With the
refit, on the same stream, each cell counts over 40 seeds the streams
more than 0.05 and more than 0.25 in slope from `RLM`. It gives the
worst distance where it was recorded:

| outliers | 20,000 rows, 500 (the default here) | 2,000 rows, 200 | 2,000 rows, 500 | 2,000 rows, 1,000 |
|---|---|---|---|---|
| 5% | 9 and 0 of 40; worst 0.091 | 5 and 0; 0.065 | 5 and 0; 0.065 | 5 and 0; 0.065 |
| 10% | 9 and 0; 0.11 | 2 and 0; 0.075 | 2 and 0; 0.075 | 2 and 0; 0.076 |
| 12.5% | 11 and 0; 0.11 | 7 and 0; 0.098 | 7 and 0; 0.097 | 7 and 0; 0.098 |
| 15% | 12 and 0; 0.13 | 10 and 0; 0.076 | 9 and 0; 0.078 | 10 and 0; 0.079 |
| 20% | 23 and 0 | 16 and 4; 2.98 | 8 and 0; 0.14 | 8 and 0; 0.13 |

Apart from a 200-row refit at 20% outliers, which the outliers still
capture, the streams more than 0.05 away after the refit are the
constant learning rate's own wander. Its median is about 0.03 at every
share of outliers.

**`quantile`'s refit brings its worst stream close to a fit of every
row.** Measured over 50 streams of 8,000 rows of
`y = 1 + 2x₀ − x₁ + 0.5·Laplace`, with no decay, as the worst stream's
largest coefficient gap to statsmodels' `QuantReg` on the same rows:

| the first 500 rows | `quantile_refit_rows=0` | 500 (the default) |
|---|---|---|
| clean | 0.0057 | 0.0052 |
| a fifth wild, +50, where `x₀ > 0` | 0.038 | 0.0085 |
| the first three 1e4 off | 0.22 | 0.0053 |
| t(2) noise | 0.0078 | 0.0058 |
| Cauchy noise | 0.0175 | 0.0058 |
| a shift of 20 | 0.093 | 0.061 |

A shift of every one of the first rows is no outlier among them, so the refit keeps them. With the refit or without it, they leave only
as the later rows outweigh them. That is why the shift is the refit's
worst case in the table. A state loaded at another level has already refit its
first rows, so the rows of the old level go only as the later rows
outweigh them, as they would with no refit.

**A target's rows are held until its refit, and a
[group](../GLOSSARY.md#the-model-bank-and-what-it-reads) too short to
reach the count holds them to its end.** Rows that only age a target
(its value null or its weight 0) are held as runs, one run for
consecutive such rows at the same [clock](../GLOSSARY.md#time) step. A target with more than 4
runs for each row of its count gives its refit up and keeps its one-pass
fit. Under a regular clock that never happens, since a gap of any length
is one run.

`refit_budget` bounds the memory all of a
[spec](../GLOSSARY.md#the-model-bank-and-what-it-reads)'s
groups together may
reserve for their refits (320 MiB by default), and [`huber`'s
entry](../MODELS.md#huber--robust-regression) gives what that memory is
and what the bound does at many groups.

## A level shift: the trade-off a robust scale makes

**A robust scale slows the fit after a change of level.** In a series, a
run of large residuals is either a burst of outliers or a new level, and
no rule on one row can tell them apart. The median treats every row after
a level shift as an outlier until the median itself has moved, which
holds the fit back. `huber_scale_half_life` gives the scale a memory of
its own, the spec's `half_life` by default. Shorter than the fit's, the
scale follows a shift sooner, but a longer burst of outliers then widens
the cut. Measured on `y = 2x + 0.5ε`, with `x` and `ε` standard normal
(seed 1, 14,000 rows), a fit half-life of 500 rows, a shift of +20 at
row 10,000 and no outliers, as the mean absolute error after the shift:

| the fit | rows 0-200 after the shift | rows 200-400 after the shift |
|---|---|---|
| least squares (`huber_delta=inf`) | 17.6 | 13.3 |
| `"rms"` | 19.0 | 15.1 |
| `"mad"`, the scale's half-life the fit's 500 rows | 19.9 | 19.5 |
| `"mad"`, `huber_scale_half_life=100` | 19.2 | 14.6 |

The fit's own half-life sets how long the fit takes to reach the new
level, about 2,000 rows in every row of the table. Inside that time, the
MAD at the fit's half-life holds the fit back through the 400 rows
measured: 19.5 against least squares' 13.3 at rows 200-400. Each row after
the shift enters at a weight of about `1.345 · 0.5 / 20 ≈ 0.03`, until
the median of the absolute residuals has crossed to the new level.
The MAD costs `sgd(loss="huber")` more against its own `"rms"`, since
its gradient is clipped at
the cut and it follows a shift by at most `learning_rate · huber_delta`
times the scale a row. With the same fit half-life and shift, its mean
absolute error 200-400 rows after the shift was 17.33 under `"mad"` at
the fit's half-life, 2.45 under `"rms"` and 2.99 at a
`huber_scale_half_life` of 100. The break tests of [DIAGNOSTICS.md](../DIAGNOSTICS.md#has-it-broken) flag
such a shift as it happens.

## What a robust scale cannot protect against

**Huber's weights read the residual alone, so they do not protect against
a wild feature.** A row whose features are far from the rest pulls the
fit toward itself, so its residual is small and it keeps its full
weight. Outliers concentrated in one region of `x` act the same way once
they are the majority there. At 30% of the rows (60% of those with `x`
above 0) the median breaks too, and so does every estimator that reads
only residuals. Estimators that also weigh a row by its features need each row's
leverage, which a solve such as `ewridge`'s can give. Or they need
several passes over every row, which a stream has only over the first
rows it holds for the refit.
