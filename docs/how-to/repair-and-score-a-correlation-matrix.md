<!-- kind: how-to -->
# Repair and score a correlation matrix

This how-to is for anyone with a correlation matrix, estimated by a model
bank or pieced together pair by pair, who wants to make it a valid
correlation matrix, shrink it, tell its signal from noise, or score a
predicted correlation matrix.

`po.corr` repairs, shrinks, summarizes and scores a correlation matrix.
Each function needs only numpy, and follows the paper it names. `po.corr`
is unstable: its form may change in a minor release ([What is
stable](../VERSIONS.md#what-is-stable)).

## Prerequisites

These steps need the library, numpy and a matrix:

- polars-online installed with numpy, an optional extra, which
  `pip install "polars-online[numpy]"` adds. The example builds its own
  matrix.
- A correlation matrix, given as a numpy array, a dict that
  `bank.gram()` returns, or a [closed
  group](../GLOSSARY.md#saving-and-serving)'s row ([Close groups and read
  each one's row](close-groups.md)). `po.gram.correlation` turns a
  [Gram](../GLOSSARY.md#fitting-a-model) into one.

## Repair a matrix, then shrink it and tell it from noise

A matrix pieced together pair by pair, each pair on its own rows, need not
be a correlation matrix at all. Its smallest eigenvalue can be negative,
which no single set of rows gives.

1. **Check the smallest eigenvalue** with numpy's `np.linalg.eigvalsh`.
2. **Repair the matrix with `po.corr.nearest`,** which returns the nearest
   correlation matrix, how far it moved, and the iterations it took.
3. **Shrink it with `po.corr.shrink`,** toward a constant-correlation
   target.
4. **Find the noise edges with `po.corr.mp_edge`,** for your matrix's
   rows and series; the code shows 50 series over 2,000 rows. An eigenvalue outside them is more than pure noise gives.
5. **Read a correlation's standard error with `po.corr.fisher_se`,** for
   the rows it was estimated on.

This code repairs a matrix of three correlations whose smallest
eigenvalue is negative:

```python
import numpy as np

r = np.array([[1.0, 0.9, 0.7],          # three correlations estimated pair by pair, each on its own rows:
              [0.9, 1.0, 0.1],          # x0 follows both x1 and x2 closely, yet x1 and x2 barely move together,
              [0.7, 0.1, 1.0]])         # which no single set of rows can give
print(f"smallest eigenvalue {np.linalg.eigvalsh(r).min():+.3f}")   # 1. below zero: not a correlation matrix
# 2. The repair, and how far it moved the matrix.
fixed, dist, iters = po.corr.nearest(r)    # Higham (2002): the nearest correlation matrix, by alternating projections
print(f"repaired: smallest eigenvalue within 1e-8 of 0: {abs(np.linalg.eigvalsh(fixed).min()) < 1e-8}, moved by {dist:.3f}")
print(np.round(fixed, 3).tolist())
# 3. Shrinkage.
shrunk, alpha = po.corr.shrink(fixed, alpha=0.2)   # Ledoit-Wolf shrinkage toward a constant-correlation target
print(np.round(shrunk, 3).tolist())
# 4 and 5. The noise edges, and a correlation's standard error.
lo, hi = po.corr.mp_edge(n=2000, m=50)     # the Marchenko-Pastur edges: eigenvalues inside them are what pure noise gives
print(f"noise edges for 50 series over 2,000 rows: {lo:.3f} to {hi:.3f}")
print(f"standard error of a correlation of 0.3 over 2,000 rows: {po.corr.fisher_se(n=2000, rho=0.3):.4f}")
```

```text
smallest eigenvalue -0.093
repaired: smallest eigenvalue within 1e-8 of 0: True, moved by 0.120
[[1.0, 0.837, 0.655], [0.837, 1.0, 0.135], [0.655, 0.135, 1.0]]
[[1.0, 0.778, 0.632], [0.778, 1.0, 0.216], [0.632, 0.216, 1.0]]
noise edges for 50 series over 2,000 rows: 0.709 to 1.341
standard error of a correlation of 0.3 over 2,000 rows: 0.0204
```

## Score a predicted correlation matrix

Use this to compare predictions of a correlation matrix, such as two specs'
`ew_cov` matrices, with the matrix the next period realized.

1. **Score each predicted matrix against the realized matrix with
   `po.corr.loss`.** Its default, `kind="qlike"`, is 0 for a prediction
   equal to the realized matrix and above 0 elsewhere, so the lower loss
   is the better prediction.

This code scores a prediction near the realized matrix and one of no
correlation:

```python
import numpy as np

real = np.array([[1.0, 0.5], [0.5, 1.0]])    # the matrix the period realized
good = np.array([[1.0, 0.4], [0.4, 1.0]])    # a prediction near it
flat = np.eye(2)                              # a prediction of no correlation
print(round(po.corr.loss(good, real), 4), round(po.corr.loss(flat, real), 4))   # the lower, the better
```

```text
0.0181 0.2877
```

## Check that it worked

**The repaired matrix's smallest eigenvalue is 0 to the iteration's
tolerance,** as the example prints. The repair lands on the edge of the
valid matrices, so it is not above 0. It moves each correlation a little,
0.120 in all as a Frobenius distance.

**`po.corr.loss(real, real)` is 0,** since a prediction equal to the
realized matrix scores 0. A loss above 0 on your own prediction is
expected, and of two predictions, the one with the lower loss is better.

## Choose another function

The rest, by kind, each with its arguments in the [API
reference](https://hgilde.github.io/polars-online/corr.html):

| kind | function | what it gives |
|---|---|---|
| shrinkage | `shrink` | Ledoit-Wolf shrinkage toward a constant-correlation (or identity) target, and its weight |
| transform | `to_z`, `from_z` | Fisher's transform and its inverse, clipped so a degenerate ±1 is finite |
| spectrum | `spectral`, `from_spectral` | the top eigenpairs, and the completion back to a correlation matrix |
| spectrum | `absorption` | the absorption ratio: how much of the variance the top `k` eigenpairs carry |
| spectrum | `mp_edge`, `mp_density` | the Marchenko–Pastur edges and density: what pure noise gives |
| spectrum | `absorption_shift` | the standardized absorption shift, `(fast − slow) / scale` |
| one number | `equicorr` | [`deco`](../MODELS.md#deco--one-correlation-for-the-whole-matrix)'s one number for the whole matrix, offline |
| one number | `equicorr_row` | the equicorrelation estimate of one standardized row, the `u` `deco` computes per row |
| one number | `equicorr_loglik` | the Gaussian log-density of a standardized row under an equicorrelation matrix |
| sampling error | `fisher_se` | the standard error of a correlation; `phi_a=` and `phi_b=` add the AR(1) inflation, with its caveats |
| sampling error | `signal_share` | how much of a correlation's movement between blocks is not sampling noise, at Kish's sample size |
| blocks | `block_means`, `from_blocks` | the mean correlation within and between labeled blocks, and the block matrix back from those means |
| scale | `epps_invert` | the correlation at a coarser scale, from `ew_cov`'s lagged co-moments |
| scoring | `loss` | how wrong a predicted correlation matrix was: `qlike`, `z_mse` or Engle–Colacito `minvar` |
