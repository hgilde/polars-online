<!-- kind: how-to -->
# Choose `ftrl`'s penalties

This how-to is for anyone who fits `ftrl`, the logistic or sparse linear
regression that takes one gradient step a row, and finds its fit shrunk
toward zero. It works out how hard the penalties `beta`, `l1` and `l2`
shrink the fit under a half-life, and how to shrink it less.

## Prerequisites

These steps need the library:

- polars-online installed, with numpy for the recipe, which builds its own
  data and saves it to a parquet file.
- Optionally a [clock](../GLOSSARY.md#time) column, with the `gap_cap` a
  clock requires. Without one, each row is one clock unit. The recipe's
  clock `t` steps by 1 a row, the same as no clock.
- The parameters of
  [`po.spec.ftrl`](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.ftrl),
  which its docstring defines: `alpha`, the learning-rate scale (0.1 by
  default), `beta`, its smoothing (1.0), `l1`, the L1 penalty (0.0), and
  `l2`, the L2 penalty (1.0). Its entry in
  [MODELS.md](../MODELS.md#ftrl--online-logistic-regression) gives its
  update equation.

## Work out how hard the penalties shrink the fit

**`ftrl`'s penalties keep a fixed mass, while the evidence grows with the
weights and the rows' density.** So a heavier or denser stream overcomes
them sooner, and a shorter half-life shrinks the fit more. With unit
weights and rows one clock unit apart, the penalties act in steady state as
a ridge of `(1 − f)(beta/alpha + l2)` on the mean scale. Here `f` is the
decay factor over one row's clock, `0.5 ** (1 / half_life)` for rows one
clock unit apart.

1. **Compute the ridge `(1 − f)(beta/alpha + l2)`.** At the defaults,
   `beta/alpha + l2` is 11, so at `half_life=100` the ridge is 0.076.
2. **Read what it does to the intercept.** A ridge `r` on the mean scale
   shrinks the intercept, whose feature is the constant 1, by a factor
   `1 / (1 + r)`. Under `loss="squared"`, a constant target of 5 is then
   fitted at 4.65 at `half_life=100`, and at 4.96 at 1000.
3. **To shrink less, lower `beta` and `l2`, or lengthen the half-life.**
   `beta=0.1, l2=0` makes `beta/alpha + l2` 1, an eleventh of the
   default's. A longer half-life makes `1 − f` smaller. Rows of a larger
   weight, or more rows a clock unit, outweigh the same penalties sooner.
4. **To keep `l1` zeroing the same coefficients whatever the weights and
   the rows' density, use
   [`lasso`](../MODELS.md#lasso--lasso-path-its-penalty-chosen-as-it-runs)
   instead.** Its penalty is on the mean scale.

**The ridge weighs most on a 0/1 target,** whose log-odds the evidence
pins only weakly: a probability's curvature is at most a quarter. On two
features at `half_life=200`, with one row a clock unit, the defaults left
the logistic fit at its steady state up to 0.41 from the decayed maximum
likelihood in some coefficient, the worst of 50 seeds. `beta=0.1, l2=0`
left it 0.17 from it.

The recipe plants a constant target of 5 beside a feature `x` that is
noise. It prints each spec's fitted intercept beside `5 / (1 + r)`, the
ridge's prediction, at two half-lives, at the default penalties and at
`beta=0.1, l2=0`. This code builds its own data:

```python
import numpy as np
import polars as pl
import polars_online as po

# A constant target of 5, and a feature x that is noise.
n = 20_000
x = np.random.default_rng(1).standard_normal(n)
pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": np.full(n, 5.0)}).write_parquet("five.parquet")

# ftrl under the squared loss, at two half-lives, with the default penalties and lighter ones.
settings = {"default": dict(beta=1.0, l2=1.0), "light": dict(beta=0.1, l2=0.0)}
specs = [
    po.spec.ftrl(f"{name}_{hl:g}", targets=["y"], features=["x"], loss="squared", clock="t", gap_cap=10.0,
                 half_life=hl, alpha=0.1, **kw)
    for hl in (100.0, 1_000.0) for name, kw in settings.items()
]
bank = po.ModelBank(specs)
bank.fit(pl.scan_parquet("five.parquet"))

# Each fitted intercept, beside the ridge's prediction and the planted 5.
intercepts = dict(bank.coef().filter(term="intercept").select("spec", "coef").iter_rows())
for hl in (100.0, 1_000.0):
    for name, kw in settings.items():
        r = (1 - 0.5 ** (1 / hl)) * (kw["beta"] / 0.1 + kw["l2"])     # the ridge on the mean scale
        print(f"half_life {hl:>4g}, {name:>7}: fitted {intercepts[f'{name}_{hl:g}']:.3f}, "
              f"5 / (1 + {r:.4f}) = {5 / (1 + r):.3f} (planted 5)")
```

```text
half_life  100, default: fitted 4.648, 5 / (1 + 0.0760) = 4.647 (planted 5)
half_life  100,   light: fitted 4.963, 5 / (1 + 0.0069) = 4.966 (planted 5)
half_life 1000, default: fitted 4.962, 5 / (1 + 0.0076) = 4.962 (planted 5)
half_life 1000,   light: fitted 4.995, 5 / (1 + 0.0007) = 4.997 (planted 5)
```

At the default penalties and a half-life of 100 rows, the fit stands 7%
below the planted 5. Lighter penalties, or a half-life ten times longer,
bring it within 1%.

## Check the result

**Compare the fit with an unpenalized one on the same rows.** On a target
whose level you know, as in the recipe, the fitted intercept should
stand near `1 / (1 + r)` of it. On your own stream under
`loss="squared"`, fit the same features with
[`ewridge`](../MODELS.md#ewridge--exponentially-weighted-ridge-regression)
at `ridge=0` and the same half-life, and read how far `ftrl`'s
coefficients stand from its. [Choose a learning rate](choose-a-learning-rate.md#choose-ftrls-alpha)
sets `alpha`, which moves the same ridge.
