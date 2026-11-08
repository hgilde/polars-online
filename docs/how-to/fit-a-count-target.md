<!-- kind: how-to -->
# Fit a count target

This how-to is for anyone whose target is a count, such as arrivals a
minute or trades an hour, and who wants a rate to predict it. It fits
`sgd` under `loss="poisson"`, the one model that takes a count target, and
keeps the gradient's cap that holds the fit steady.

## Prerequisites

These steps need the library:

- polars-online installed, with numpy for the recipe, which builds its own
  data and saves it to a parquet file.
- A target that holds counts: 0 or more on every row where it is not
  [null](../GLOSSARY.md#the-model-bank-and-what-it-reads).
- The parameters of
  [`po.spec.sgd`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.sgd),
  which its docstring defines. Its entry in
  [MODELS.md](../MODELS.md#sgd--stochastic-gradient-descent) gives the
  Poisson loss's link and gradient.

## Fit the counts under the Poisson loss

Under `loss="poisson"`, `sgd` predicts `p = exp(zᵀβ)`, where `z` is the
row's features with a 1 for the intercept, and `β` the coefficients. So the
[prediction](../GLOSSARY.md#fitting-a-model) is a rate and the
[coefficients](../GLOSSARY.md#fitting-a-model) are on the log of it. It
takes one gradient step a row.

**Under `loss="poisson"`, keep `clip_gradient`, `1e3` by default,** because
through the log link one large count would make the next gradient
exponentially bigger. At ordinary scales it does not bind for the other
losses.

1. **Give the spec `loss="poisson"`.** Each row's `pred_<target>` is then
   the rate, and `bank.coef()` reads the coefficients on the log-rate.
2. **Keep the default `clip_gradient=1e3`.** It caps each coordinate of
   the gradient at ±1,000.
3. **Choose the learning rate and its schedule.** [Choose a learning
   rate](choose-a-learning-rate.md#choose-sgds-learning-rate) shows how to
   set `learning_rate` and `schedule`.
4. **Set a null target on a row the model should predict without
   learning,** in place of a negative value, which the bank refuses.

The recipe plants counts with log-rate `1 + 0.4·busy`, and one row whose
count is 2,000, where the others average about 3. It fits them twice at a
constant rate of 0.003: with the default `clip_gradient`, and with
`clip_gradient=inf`, no cap. A third fit, with the default cap, reads the
same counts without the large one. It prints each fit beside the planted
coefficients. This code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# Counts with log-rate 1 + 0.4·busy, and one row's count of 2,000.
rng = np.random.default_rng(21)
n = 40_000
busy = rng.standard_normal(n)
arrivals = rng.poisson(np.exp(1.0 + 0.4 * busy)).astype(float)
clean = arrivals.copy()                            # the same counts, without the large one
arrivals[20_000] = 2_000.0
pl.DataFrame({"busy": busy, "arrivals": arrivals, "clean": clean}).write_parquet("arrivals.parquet")

# The Poisson loss at a constant rate, with the gradient's default cap and with none.
common = dict(targets=["arrivals"], features=["busy"], half_life=float("inf"), loss="poisson", learning_rate=0.003)
specs = [po.spec.sgd("capped", **common), po.spec.sgd("uncapped", clip_gradient=float("inf"), **common),
         po.spec.sgd("no_large", **{**common, "targets": ["clean"]})]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("arrivals.parquet"))

# Each fit's coefficients on the log-rate, beside the planted ones.
print("planted:  intercept 1.000, busy 0.400")
for spec in ("capped", "uncapped", "no_large"):
    intercept, slope = bank.coef().filter(spec=spec)["coef"]
    if abs(intercept) < 1e6:
        print(f"{spec + ':':<9} intercept {intercept:.3f}, busy {slope:.3f}")
    else:
        print(f"{spec}: diverged, its intercept past -1e6: {intercept < -1e6}")
```

```text
planted:  intercept 1.000, busy 0.400
capped:   intercept 1.022, busy 0.448
uncapped: diverged, its intercept past -1e6: True
no_large: intercept 1.022, busy 0.448
```

The capped fit lands near the planted log-rate, and to three decimals
where it lands on the same counts with no large one. Without the cap, the one count of 2,000 drove
the fit off by many orders of magnitude, and it never came back.

## Set a null target in place of a negative count

**A call that reads the target refuses a chunk with a negative count,
naming the row, before any model learns from it,** as scikit-learn's
`PoissonRegressor` refuses one. Such a call is one that learns, or
`predict` on a chunk that holds the target. A negative target taken as it
stood would make `p − y` drive the prediction down to the link's floor,
`e^−30`, and hold it there. So set the target to null on a row the model
should predict without learning. This code tries a chunk whose second row
holds −1:

```python
counts = po.spec.sgd("counts", targets=["arrivals"], features=["busy"], half_life=float("inf"), loss="poisson")
bad = pl.DataFrame({"busy": [0.1, 0.2, 0.3], "arrivals": [2.0, -1.0, 3.0]})
try:
    po.ModelBank([counts]).fit(bad)
except ValueError as refused:
    print(refused)
```

```text
spec "counts": target column "arrivals" has the value -1 at row 1 (counting from 0), but loss="poisson" takes a count, which is never negative. Set the target to null on rows the model should predict without learning, or choose another loss.
```

## Check the result

**Compare the predicted rate with the counts it predicts.** Each row's
`pred_<target>` is a rate made before the row is learned, so over many
rows its mean should stand near the counts' mean.

**Read the coefficients with `bank.coef()`.** Each is the change in the
log of the rate per unit of its feature. A coefficient far larger in size
than any log-rate the counts could have, as the uncapped fit's, marks a fit
that diverged: keep `clip_gradient`, and lower `learning_rate`.
