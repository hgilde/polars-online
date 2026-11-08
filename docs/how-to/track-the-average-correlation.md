<!-- kind: how-to -->
# Track the average correlation

This how-to is for anyone who watches many series and wants one number
that moves with how correlated they are on average, such as a market's
correlation, row by row. It uses `deco`, Engle and Kelly's dynamic
equicorrelation, which takes O(m) time a row for `m` series, where keeping
every pair's correlation moving takes O(m²).

## Prerequisites

These steps need the library and the series:

- polars-online installed, and numpy to make the example's data.
- A stream with two or more numeric columns, read as the
  [features](../GLOSSARY.md#the-model-bank-and-what-it-reads) of a
  [`deco`](../MODELS.md#deco--one-correlation-for-the-whole-matrix) spec.
- A [half-life](../GLOSSARY.md#time), which `deco` requires: the means and
  variances it standardizes each row by, and the level it reports, both
  decay on it. [Time and decay](../explanation/time-and-decay.md) says how
  a model forgets along its clock.

## Read the level, and one per block

`deco` reads each row's own estimate of the average correlation, `u`, and
reports `rho`, the level that follows it. Use both as a number that moves
with the correlation, and read an
[`ew_cov`](../MODELS.md#ew_cov--exponentially-weighted-moments)'s `corr`
for the correlation itself. Both read below the truth, since `u` is
biased downward and `rho` averages it: about two-thirds of the truth at
the higher correlations of the recipe.

1. **List the series in `features`, and leave out a column that is
   constant from its first row.** It has no standardized value, so it is
   left out of its block, and the block reads the correlation among its
   other columns. A block left with fewer than two has no `u`. `loglik`, the
   row's log-density under the level, needs every column, so a constant
   column nulls `loglik` on every row.
2. **Give `blocks` for one number per group of series and one per pair of
   groups,** such as sectors. Each feature goes in exactly one block, and a
   block needs at least two. `rho` becomes `rho_<A>` per block and
   `rho_<A>_<B>` per pair, and `u` likewise.
3. **Add both specs to the query with `online.fit_predict([eq, pairs])`,
   run it with `collect()`, and read `rho`,** the level as it stood before the
   row, so it is safe as a feature for that same row.
4. **For the correlation's size, run an `ew_cov` with `stats=["corr"]`
   over the series of one block, such as the banks,** and average its `corr_*` fields
   with Polars' `pl.mean_horizontal`, as the recipe does.

This code makes two sectors of five series. Within the banks every pair
correlates at 0.2 and then, from row 2,000, at 0.6. Within the miners
every pair correlates at 0.4 throughout, and no bank correlates with a
miner. It prints each block's `rho` and the mean of an `ew_cov`'s `corr`
over the banks, beside the truth:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two sectors: the banks' common correlation rises at row 2,000, the miners' holds at 0.4.
rng = np.random.default_rng(4)
n = 4_000
banks_truth = np.where(np.arange(n) < 2_000, 0.2, 0.6)
fa, fb, own = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal((10, n))
# Five banks on the factor fa, and five miners on fb.
banks = [f"bank{i}" for i in range(5)]
miners = [f"miner{i}" for i in range(5)]
cols = {s: np.sqrt(banks_truth) * fa + np.sqrt(1 - banks_truth) * own[i] for i, s in enumerate(banks)}
cols |= {s: np.sqrt(0.4) * fb + np.sqrt(0.6) * own[5 + i] for i, s in enumerate(miners)}
pl.DataFrame(cols).write_parquet("sectors.parquet")

# deco over every series, beside an ew_cov over the banks.
eq = po.spec.deco("eq", features=banks + miners, half_life=100.0,       # 1. every series
                  blocks={"banks": banks, "miners": miners})             # 2. one level per sector and per pair
pairs = po.spec.ew_cov("pairs", features=banks, half_life=100.0, stats=["corr"])   # 4. the banks' every pair
out = pl.scan_parquet("sectors.parquet").online.fit_predict([eq, pairs]).collect()  # 3. run both, rho read before each row
rho = out["eq"].struct.unnest()
mean_corr = out["pairs"].struct.unnest().select(pl.mean_horizontal(pl.selectors.starts_with("corr_"))).to_series()
for row in (1_999, 3_999):
    print(f"row {row}: banks truth {banks_truth[row]:.1f}, rho {rho['rho_banks'][row]:.2f}, mean corr {mean_corr[row]:.2f}; "
          f"miners truth 0.4, rho {rho['rho_miners'][row]:.2f}; between truth 0.0, rho {rho['rho_banks_miners'][row]:+.2f}")
```

```text
row 1999: banks truth 0.2, rho 0.17, mean corr 0.25; miners truth 0.4, rho 0.24; between truth 0.0, rho -0.01
row 3999: banks truth 0.6, rho 0.39, mean corr 0.60; miners truth 0.4, rho 0.26; between truth 0.0, rho -0.00
```

The banks' `rho` rises with their correlation, and the miners' holds.
Each reads below the truth, where the mean of `ew_cov`'s `corr` reads
near it.

## Check that it worked

**A level that moves when the series' correlation moves, and holds when
it holds, is working, whatever its size.** Compare it with an `ew_cov`'s
`corr` over the same series, as the recipe does, on a stretch you know.
[Regimes §7](../REGIMES.md#7-deco-an-equicorrelation-that-moves)
measures how far `rho` reads below the truth.

| what you see | what it means | what to do |
|---|---|---|
| `u` and `rho` null on a group's first rows | the model is in its [warm-up](../GLOSSARY.md#warm-up-and-readiness): `u` needs a variance for each column, from the first two rows, and `rho` starts on the row after the first `u` | read the rows after it |
| `loglik` null on every row | a column has had no positive variance, such as a constant one | leave the constant column out of `features` |
| `rho` about two-thirds of an `ew_cov`'s mean `corr` | the estimate's downward bias, which Engle and Kelly note: `E[u]` is about 0.20 for a true 0.30 at six series | expected: read `rho` for how the correlation moves, and `corr` for its size |

Every parameter is in `deco`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.deco),
including `dynamics="linear"`, which steps the level once a row with
its own `alpha` and `beta`, and the fields in [OUTPUTS.md](../OUTPUTS.md#deco).
