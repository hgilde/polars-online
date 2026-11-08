<!-- kind: how-to -->
# Choose a Kalman filter's drift speed

This how-to is for anyone who fits `kalman`, the regression whose
coefficients drift, and needs to set how fast each coefficient may move.
It chooses `coef_half_life` per coefficient from the stream's own errors,
makes a coefficient forget between bursts of its feature with
`revert_half_life`, and sets how far a prediction past the data may move.

## Prerequisites

These steps need the library and a stream with a clock:

- polars-online installed, with numpy for the recipes. Each recipe builds
  its own data and saves it to a parquet file.
- A [clock](../GLOSSARY.md#time) column, and the `gap_cap` a clock
  requires. The recipes' clock `t` steps by 1 a row.
- The parameters of
  [`po.spec.kalman`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.kalman),
  which its docstring defines, and its entry in
  [MODELS.md](../MODELS.md#kalman--random-walk-β-dynamic-linear-model).

## Give each coefficient its own drift speed

`coef_half_life` is how fast a [coefficient](../GLOSSARY.md#fitting-a-model)
may drift, as a [half-life](../GLOSSARY.md#time) in clock units on the
features standardized by `standardize`, the default. A short one follows
a coefficient that moves, and a long one averages more rows for a
coefficient that holds still.

**For a drift rate per coefficient, give `coef_half_life` a list with the
intercept's entry leading.** An entry of `inf` gives its coefficient no
drift: the filter takes it as constant and goes on learning it from every
row.

1. **List one value per coefficient, the intercept first.** For a stream
   where only `x0`'s slope moves, give
   `coef_half_life=[float("inf"), 30.0, float("inf")]`: the intercept,
   then `x0`, then `x1`.
2. **Give the spec its `half_life` too.** Here `half_life` decays the
   noise estimate and the features' standardization, not the
   coefficients' drift.

## Choose the speeds from the stream's own errors

Every prediction a [model bank](../GLOSSARY.md#the-model-bank-and-what-it-reads)
makes is [out-of-sample](../GLOSSARY.md#fitting-a-model): made before the
model learns the row. So the error each setting made on rows it had not
yet seen ranks the settings fairly, in one pass over the stream.

1. **Build one spec per value to try.** A list given to `coef_half_life`
   is one value per coefficient, not a [grid](../GLOSSARY.md#grids), so
   each speed takes a spec of its own.
2. **Score every spec with `po.eval.grid_search`.** It runs one bank of
   every spec over the query, scores each spec on the same rows, and
   chooses from earlier rows before each span it reports on. [Choose a
   half-life with
   `po.eval.grid_search`](../CALIBRATION.md#choose-a-half-life-with-poevalgrid_search)
   reads the frame it returns.
3. **Run the spec ranked 1 on the rows to come.** Read its name with
   `table.filter(kind="candidate", rank=1)["spec"]`. The `choice` row's
   error is the one to report for it, since the rows that made the choice
   did not score it.

The recipe plants a slope on `x0` that wanders, a fixed slope on `x1` and
noise of variance 1. No fit can predict closer than that noise on average,
so 1.00 is the floor, printed beside the error of the choice. On a finite
sample a spec can land a hair under it. This code builds its own
data:

```python
import numpy as np
import polars as pl
import polars_online as po

# y = 2 + b·x0 + 0.5·x1 + noise of variance 1, where b wanders as a random walk.
rng = np.random.default_rng(5)
n = 20_000
x0, x1, e = rng.standard_normal((3, n))
b = 1.0 + np.cumsum(rng.normal(0, 0.02, n))
y = 2.0 + b * x0 + 0.5 * x1 + e
pl.DataFrame({"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": y}).write_parquet("drift.parquet")

# One spec per drift speed for every coefficient, and two that let x0's slope alone drift.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=500.0)
inf = float("inf")
specs = [po.spec.kalman(f"all_{h:g}", coef_half_life=h, **common) for h in (10.0, 30.0, 100.0, 300.0)]
specs += [po.spec.kalman(f"x0_only_{h:g}", coef_half_life=[inf, h, inf], **common) for h in (30.0, 100.0)]
table = po.eval.grid_search(pl.scan_parquet("drift.parquet"), specs)

# Each spec's error on rows it had not seen, beside the noise's variance.
for spec, mse, rank in table.filter(kind="candidate").select("spec", "mse", "rank").iter_rows():
    print(f"{spec:>11}: mse {mse:.4f}, rank {rank}")
print(f"chosen from earlier rows: mse {table.filter(kind='choice')['mse'].item():.4f} (the noise alone: 1.00)")
```

```text
     all_10: mse 1.0848, rank 6
     all_30: mse 1.0195, rank 4
    all_100: mse 1.0130, rank 3
    all_300: mse 1.0590, rank 5
 x0_only_30: mse 0.9966, rank 1
x0_only_100: mse 1.0060, rank 2
chosen from earlier rows: mse 0.9966 (the noise alone: 1.00)
```

Letting only `x0`'s slope drift predicts closest: the intercept and `x1`'s
slope never move, so a drift on them adds noise to their estimates. The
error of the spec ranked 1 sits at the noise's variance. A single speed
for every coefficient can be searched with a builder and `param_grid`
instead, as [Choose a setting that takes no
list](../CALIBRATION.md#choose-a-setting-that-takes-no-list) shows.

## Forget a feature that is active only now and then

**For a feature that is active only now and then, give its coefficient a
finite `revert_half_life`.** The coefficient is then pulled toward zero on
every row, while the rows that support it pull it back (an AR(1) prior).
So the effect is forgotten between its bursts, and cannot persist through
a run of null targets. For a feature that is always active, keep the
default `inf`. The pull would hold a persistent effect below its true
size, the more so the shorter the reversion half-life.

**Under `standardize`, the default, the reversion pulls toward zero in the
standardized coordinates:** "no effect" for a slope, and "the target
averages zero" for the intercept. So unless the target averages zero, keep
the intercept's entry at `inf`, as in `[float("inf"), 50.0, 50.0]`.

1. **Give `revert_half_life` a list, the intercept first.** Keep the
   intercept's entry at `inf`, and give the occasional feature's slope a
   reversion half-life in clock units, such as `[float("inf"), 200.0]`.
2. **Run the specs in a query,** as
   `pl.scan_parquet(...).online.fit_predict(specs)` does in the recipe.
   Between bursts the slope decays toward zero, so the next burst starts
   from no effect, not from the last burst's slope.

In the recipe, `x` is active for 100 rows in every 1,000, and each burst
has its own slope, 2 or −2, drawn at random. So the last burst's slope
says nothing about the next one's. The recipe prints the size of the slope
each spec carries into a burst, and the size it reaches by a burst's end,
beside the planted 2. This code builds its own data:

```python
# x is active for 100 rows in every 1,000, each burst with a slope of its own, 2 or -2.
rng = np.random.default_rng(11)
n = 19_100                                         # the stream ends at the end of a burst
t = np.arange(n)
burst = (t % 1_000) < 100
slopes = rng.choice([-2.0, 2.0], n // 1_000 + 1)
x = np.where(burst, rng.standard_normal(n), 0.0)
y = 1.0 + np.where(burst, slopes[t // 1_000], 0.0) * x + 0.5 * rng.standard_normal(n)
pl.DataFrame({"t": t.astype(float), "x": x, "y": y}).write_parquet("bursts.parquet")

# A slope that walks, and one that reverts to zero between bursts.
common = dict(targets=["y"], features=["x"], clock="t", gap_cap=10.0, half_life=500.0,
              coef_half_life=[inf, 20.0], coef_every=0)
specs = [
    po.spec.kalman("walk", **common),
    po.spec.kalman("revert", revert_half_life=[inf, 200.0], **common),   # the intercept's entry stays inf
]
out = pl.scan_parquet("bursts.parquet").online.fit_predict(specs).collect()

# The slope's size on the row before each burst, and on each burst's last row.
for name in ("walk", "revert"):
    slope = out[name].struct.field("coef").list.get(1)
    carried = np.mean([abs(slope[b * 1_000 - 1]) for b in range(1, 20)])
    reached = np.mean([abs(slope[b * 1_000 + 99]) for b in range(19)])
    print(f"{name:>6}: carried into a burst {carried:.2f}, at a burst's end {reached:.2f} (planted 2)")
```

```text
  walk: carried into a burst 1.93, at a burst's end 1.99 (planted 2)
revert: carried into a burst 0.12, at a burst's end 1.92 (planted 2)
```

The walking slope carries the last burst's effect into the next, where it
is as likely to be wrong in sign as right. The reverting slope starts each
burst near zero, and reaches 1.92 of the planted 2 by the burst's end: the
pull holds it a little below its true size.

## Let a prediction far past the data return to the target's level

**To let a far prediction return to the target's level, give a `gap_cap`
that spans several of the slopes' reversion half-lives, and keep the
intercept's `revert_half_life` at `inf`.** `predict` shrinks each slope by
`2^(−d/κ)`, where `d` is the clock since the last learned row, capped by
`gap_cap`, and `κ` is the slope's `revert_half_life`. So a slope keeps at
least `2^(−gap_cap/κ)` of its value.

Under `standardize` the reversion shrinks the standardized slopes. So a far
prediction approaches the target's level at the features' means. That
level equals `bank.coef()`'s intercept, which is reported in the
unstandardized coordinates, only where the features average 0.

1. **Give the slopes a finite `revert_half_life`, and the intercept
   `inf`,** as in [Forget a feature that is active only now and
   then](#forget-a-feature-that-is-active-only-now-and-then).
2. **Give `gap_cap` several of the slopes' reversion half-lives.** With a
   reversion half-life of 200 clock units, a `gap_cap` of 1,000 leaves a
   slope `2^(−5)`, about 3%, of its value. `gap_cap` also caps every clock
   step the fit learns over, for the decay, the drift and the reversion
   alike. So on a stream with steps longer than the old `gap_cap`, a larger
   one changes the fit itself, not only the far predictions.
3. **Predict the rows past the data with `bank.predict`.** It learns
   nothing, and moves each coefficient over the clock since the last
   learned row, capped by `gap_cap`.

The recipe fits the burst stream twice, with a `gap_cap` of 10 and of
1,000, and predicts two rows 1,000 clock units past the last one, at
`x = 0` and `x = 1`. Their difference is the slope the prediction reads,
printed beside `2^(−gap_cap/κ)` times the slope at the last learned row.
The stream's clock steps by 1 a row, so `gap_cap` changes no step the fit
learns over. This code reads `bursts.parquet`, which the burst recipe
wrote:

```python
# The reverting spec under a short and a long gap_cap, fitted on the same rows.
banks = {}
for cap in (10.0, 1_000.0):
    spec = po.spec.kalman("k", targets=["y"], features=["x"], clock="t", gap_cap=cap, half_life=500.0,
                          coef_half_life=[inf, 20.0], revert_half_life=[inf, 200.0])
    banks[cap] = po.ModelBank([spec])
    banks[cap].fit(pl.scan_parquet("bursts.parquet"))

# Two rows 1,000 clock units past the last learned row, at x = 0 and x = 1.
ahead = pl.DataFrame({"t": [n - 1 + 1_000.0] * 2, "x": [0.0, 1.0]})
for cap, bank in banks.items():
    pred = bank.predict(ahead)["k"].struct.field("pred_y")
    intercept, slope = bank.coef()["coef"]
    print(f"gap_cap {cap:>6g}: slope read {pred[1] - pred[0]:.4f}, "
          f"2^(-gap_cap/200) x {slope:.4f} = {2 ** (-cap / 200) * slope:.4f}; "
          f"prediction at x = 1 {pred[1]:.3f}, intercept {intercept:.3f}")
```

```text
gap_cap     10: slope read 1.8423, 2^(-gap_cap/200) x 1.9073 = 1.8423; prediction at x = 1 2.856, intercept 1.012
gap_cap   1000: slope read 0.0596, 2^(-gap_cap/200) x 1.9073 = 0.0596; prediction at x = 1 1.123, intercept 1.012
```

Under the short `gap_cap`, a prediction 1,000 clock units out still reads
almost the whole slope. Under the long one, it reads 3% of it, and the
prediction stands near the target's level. `x` is 0 between its bursts and
averages about 0 within them, so here that level is near `bank.coef()`'s
intercept, which the recipe prints.

## Set the process noise outright

**To set the process noise outright, give `q=` in place of
`coef_half_life`.** `q_i` is added to coefficient `i`'s variance as
`q_i·D²`, where `D` is the clock since the row that last observed the
target. Its units are the coefficient's units squared per clock unit
squared, and under `standardize` a slope is in the target's units per
standard deviation of its feature. `q` is measured against the noise
estimate, the one place the estimate reaches the fit.

1. **Give `q` one value per coefficient, the intercept first,** each
   finite and at least 0. A 0 holds its coefficient still.
2. **Leave out `coef_half_life`.** The spec takes exactly one of `q` and
   `coef_half_life`.

The recipe gives the first recipe's stream, whose `x0` slope walks by
steps of standard deviation 0.02 a row, three values of `q` for that slope.
Its variance per row is 0.0004, against noise of variance 1. This code
reads `drift.parquet`, which the first recipe wrote:

```python
# x0's slope drifts and the rest holds still; one spec per value of x0's q.
common = dict(targets=["y"], features=["x0", "x1"], clock="t", gap_cap=5.0, half_life=500.0)
specs = [po.spec.kalman(f"q_{q:g}", q=[0.0, q, 0.0], **common) for q in (4e-5, 4e-4, 4e-3)]
table = po.eval.grid_search(pl.scan_parquet("drift.parquet"), specs)
for spec, mse, rank in table.filter(kind="candidate").select("spec", "mse", "rank").iter_rows():
    print(f"{spec:>8}: mse {mse:.4f}, rank {rank}")
```

```text
 q_4e-05: mse 1.0085, rank 2
q_0.0004: mse 0.9961, rank 1
 q_0.004: mse 1.0122, rank 3
```

The planted variance predicts closest. It is close to the process noise
that `coef_half_life=30` gives, `(ln 2 / 30)²` or about 0.0005 in units of
the noise's variance, the speed that ranked 1 in the first recipe.

## Check the result

**Read each coefficient's path against what you know of it.** With
`coef_every=0`, the output's `coef` field holds the coefficients after
every row, as the burst recipe reads them. A coefficient you expect to
hold still should not wander, and one you expect to move should follow
its moves.

**Compare the chosen spec's error with the others' on the same rows.**
`po.eval.grid_search` scores every spec on the rows where each has a
prediction. A spec ranked 1 by a wide margin is a clear choice. Specs that
nearly tie, as `x0_only_30` and `x0_only_100` do, predict about as well,
and the `choice` row says what choosing between them earned. [Why the best
error over a grid is
optimistic](../explanation/grids-and-instances.md#why-the-best-error-over-a-grid-is-optimistic)
explains why to report that row.
