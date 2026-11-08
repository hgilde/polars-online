<!-- kind: how-to -->
# Fit a Huber regression on a Gram

This how-to is for anyone who wants Huber's robust fit of a stream on
disk, converged rather than learned in one pass, while factoring its
[Gram](../GLOSSARY.md#fitting-a-model) once. You will run Huber's
algorithm H: each pass is two Polars queries over the rows and a solve
against that one factor.

## Prerequisites

These steps need the library, numpy and the stream as a file:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The recipe builds its own
  data.
- The rows in a file a Polars query can read in each pass, such as
  parquet. Each pass reads them all once.
- An `ewridge` spec over the features, with `half_life=float("inf")`, to
  leave the Gram of the rows at their own weights.

[Huber regression on a Gram](../GRAM.md#huber-regression-on-a-gram)
explains why algorithm H can reuse one Gram in every pass, why it starts
from a fit with the target's level, and why it reads the scale again in
each pass.

## Run algorithm H over one Gram

Use this for a fit at Huber's loss whose matrix is factored once, where
weighted least squares would need a Gram for each pass and each target.

1. **Run one pass that leaves the Gram,** with `bank.fit` over the file,
   and read it with `bank.gram(name)[0]`.
2. **Factor its features' co-moments once,** with
   `np.linalg.cholesky(g["comoments"][1:, 1:])`. No pass changes them.
3. **Start from least squares, `po.gram.solve(g)`,** or from the target's
   median alone, an intercept and no slopes. Do not start from all zeros:
   from there every residual is the target's level, and the first pass is
   lost.
4. **In each pass, build the pseudo-response in a Polars expression:**
   the fit, plus each residual clipped at `c` scales, `y* = x'b +
   s·clip(r/s, −c, c)`, with `c = 1.345` and the scale `s` the residuals'
   MAD, `median(|r|) / 0.6745`, read again each pass.
5. **Read `y*`'s mean and its centered cross-moments with the features in
   a second query,** centered at the Gram's `means`. The scale's median
   takes a query of its own, so a pass is two queries over the file.
6. **Solve the cross-moments against the factor for the slopes,** and take
   the intercept from the means. That is the next fit.
7. **Repeat until the slopes stop moving.** In the recipe ten passes
   settle the fourth decimal.

This code makes 20,000 rows of `y = 1000 + 2x` plus noise, with 50 added
to a tenth of the rows with `x` above 0, and leaves their Gram:

```python
import numpy as np
import polars as pl

import polars_online as po

# y = 1000 + 2x plus noise, and 50 added to a tenth of the rows with x above 0: 5% of all rows.
rng = np.random.default_rng(3)
n = 20_000
x = rng.standard_normal(n)
y = 1000.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
y[(rng.random(n) < 0.10) & (x > 0)] += 50.0
pl.DataFrame({"x": x, "y": y}).write_parquet("level.parquet")

# 1. One pass leaves the Gram.
features = ["x"]
spec = po.spec.ewridge(
    "g", targets=["y"], features=features, half_life=float("inf"), solve_every=float("inf")
)
bank = po.ModelBank([spec])
bank.fit(pl.scan_parquet("level.parquet"))
g = bank.gram("g")[0]
# 2. Its matrix is the same in every pass below, so it is factored once.
means = np.asarray(g["means"])[1:]  # the features' means; slot 0 is the intercept's
factor = np.linalg.cholesky(np.asarray(g["comoments"])[1:, 1:])  # of the centered co-moments
```

This code defines one pass, steps 4 to 6: the scale in one query over the
file, then the pseudo-response's mean and centered cross-moments in a
second, and their least squares against the one factor. It continues from
the code before:

```python
def refit(coef: np.ndarray, c: float = 1.345) -> np.ndarray:
    """One pass of algorithm H: this fit's scale, then its pseudo-response's mean and
    centered cross-moments, each read in a query over the file, then its least squares."""
    # 4. This fit's residuals, this pass's scale, and the pseudo-response built from them.
    fit = coef[0] + sum(b * pl.col(f) for b, f in zip(coef[1:], features))
    r = pl.col("y") - fit
    s = pl.scan_parquet("level.parquet").select(r.abs().median()).collect().item() / 0.6745
    target = fit + s * (r / s).clip(-c, c)  # each residual clipped at c scales
    # 5. The pseudo-response's mean and centered cross-moments, in a second query.
    moments = (
        pl.scan_parquet("level.parquet")
        .select(
            mean=target.mean(),
            **{f: ((pl.col(f) - m) * target).mean() for f, m in zip(features, means)},
        )
        .collect()
    )
    # 6. Its least squares against the one factor; the intercept from the means.
    cross = np.asarray(moments.select(features).row(0))
    slopes = np.linalg.solve(factor.T, np.linalg.solve(factor, cross))
    return np.r_[moments["mean"][0] - means @ slopes, slopes]
```

This code runs ten passes from each start, step 3 and step 7. It
continues from the code before:

```python
# 3. Least squares from the Gram, and the target's median alone, as the two starts.
least_squares = po.gram.solve(g)
median = pl.scan_parquet("level.parquet").select(pl.col("y").median()).collect().item()
print(f"least squares: slope {least_squares[1]:.4f}, where the planted slope is 2")
# 7. Ten passes of algorithm H from each start.
for name, coef in [("least squares", least_squares), ("the median", np.r_[median, 0.0])]:
    slopes = []
    for k in range(1, 11):
        coef = refit(coef)
        if k in (1, 2, 5, 10):
            slopes.append(f"{coef[1]:.4f}")
    print(f"from {name}: slope after 1, 2, 5, 10 passes {', '.join(slopes)}; intercept {coef[0]:.3f}")
```

```text
least squares: slope 4.0626, where the planted slope is 2
from least squares: slope after 1, 2, 5, 10 passes 2.6203, 2.1988, 2.0415, 2.0393; intercept 1000.049
from the median: slope after 1, 2, 5, 10 passes 1.7377, 1.9865, 2.0388, 2.0393; intercept 1000.049
```

Least squares is pulled to a slope of 4.06 by the outliers. From least
squares algorithm H's slope is 2.620 after one pass, 2.199 after two,
2.042 after five and 2.0393 after ten; from the median, 1.738, 1.987,
2.039 and 2.0393. Both reach the same intercept, 1000.049.

## Check the result

**Compare the converged fit with a reference implementation of Huber's
fit.** statsmodels, which this library does not depend on, has one:
`sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT(1.345)).fit(scale_est="mad", tol=1e-12)`,
with `import statsmodels.api as sm`, uses the recipe's scale (the median
of `|r|` about 0, over the normal quartile). On the same rows it gives a
slope of 2.0392520 and the same intercept, 1000.049. The passes are within
1.2e-6 of it after ten and 3e-7 from twenty on. The 3e-7 is the constant:
with Φ⁻¹(3/4) to full precision in place of 0.6745 they reach statsmodels'
fit to 1e-12. A `huber` spec, which weighs each row once in one pass over
the stream, gives 2.0394 here ([Huber regression on a
Gram](../GRAM.md#huber-regression-on-a-gram) compares the two).

| what you see | what it means | what to do |
|---|---|---|
| the slope moves by at most `c·s` a pass, and needs hundreds of passes | the scale was read elsewhere (the noise's, or `huber_scale_<t>`), so every row is clipped | read the scale from each pass's residuals, and start from least squares or the target's median |
| the slope runs away from the truth | the Huber weights went on the cross-moments alone, which shrinks a down-weighted row's target toward 0 ([Weighting only the targets' cross-moments is not weighted least squares](../GRAM.md#huber-regression-on-a-gram)) | build the pseudo-response, as `refit` does |
| the fit breaks down with many outliers | the MAD holds until half the rows are outliers; a scale built from sums, such as Huber's Proposal 2, breaks sooner | keep the MAD, as the recipe does ([Huber regression on a Gram](../GRAM.md#huber-regression-on-a-gram) compares the two) |
| the fit stops 3e-7 short of statsmodels' | the constant 0.6745 is rounded | use Φ⁻¹(3/4) to full precision |
