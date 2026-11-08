<!-- kind: how-to -->
# Estimate a block's covariance from ticks

This how-to is for anyone with tick prices of several series who wants the
covariance and correlation of their returns over each block of the
stream, such as a trading day. Noise in each tick's price inflates a
plain realized covariance's variances, and so pulls its correlation toward
zero. `rcov` estimates the covariance robust to
that noise, and to series not observed on every row.

## Prerequisites

These steps need the library and the prices:

- polars-online installed, and numpy to make the example's data.
- A stream of prices, one column per series, in clock order, and a column
  that names each row's block, such as the trading day. The block column
  is the spec's [group](../GLOSSARY.md#the-model-bank-and-what-it-reads),
  and each block's estimate is written when the block closes.
- For series that tick at their own times, either the prices carried
  forward, as step 2 does, or the series put on one grid first, as [Align
  series that tick at their own
  times](align-series-that-tick-at-their-own-times.md) shows.

## Estimate each block's covariance

`rcov` keeps no decay: a block's close ends its memory, so it refuses a
`half_life` or `lam`. Each row writes only `weight_sum`, and each block's
estimate is that block's row of `bank.closed_groups()`.

1. **Choose the estimator, `kind`, for the returns you have.**
   [Regimes §8](../REGIMES.md#8-rcov-three-estimators-against-each-blocks-truth)
   measures the three against each block's truth:

   | `kind` | the estimator | use it on |
   |---|---|---|
   | `"plain"` | `sum(x x')`, the plain realized covariance | clean returns, where it has the smallest error |
   | `"kernel"` | the multivariate realized kernel (Barndorff-Nielsen, Hansen, Lunde & Shephard 2011), `sum_h k(h/(H+1)) Gamma_h` with Parzen weights and jittered end points | returns with measurement noise, which barely moves its error |
   | `"preavg"` | the modulated realized covariance (Christensen, Kinnebrock & Podolskij 2010): returns pre-averaged over `k_n = ceil(theta * block_rows^0.6)` rows | series not observed on every row, where it is the least biased |

   On series put on one refresh grid, use `kind="kernel"`, the default.
   The kernel's bandwidth `H` is set from the data. Give `bandwidth=` to
   fix it.

2. **Compute the returns from the prices with Polars' `diff()` in the
   query before the [model
   bank](../GLOSSARY.md#the-model-bank-and-what-it-reads), over each block with `.over("block")`,** so that
   no return spans two blocks. For a series not observed on every row,
   carry its last price forward with `forward_fill().over("block")` before
   the diff. The bank
   [skips](../GLOSSARY.md#the-model-bank-and-what-it-reads) each row with a
   null return: each block's first row, and, for a series carried forward,
   every row up to and including its first price in the block.
3. **Build the spec with `group` set to the block column and
   `group_close="monotone"`,** which closes a block when the next one
   starts.
4. **Give `block_rows`, the block's expected length.** It sizes the record
   of recent returns, a hint and not a limit, and sets `"preavg"`'s window.
   The kernel with a bandwidth set from the data refuses a spec without
   it, and so does `"preavg"` without `preavg_rows`.
5. **Make a model bank of the specs with `po.ModelBank(specs)`, fit it
   with `bank.fit(returns)`, and read each block's row of
   `bank.closed_groups(name)`.** `rcov` and `rcorr` hold the upper
   triangle of the matrix, row by row, as one flat list: for two series,
   `[1, r, 1]` in `rcorr`, so `r` is its second entry. Beside them sit
   `rcov_n`, `rcov_kind`, `rcov_bandwidth_used`, `rcov_omega2` and
   `rcov_iv_sparse` (the noise variance and sparse integrated variance
   behind the kernel's bandwidth), and `rcov_iq`, a realized-quarticity
   proxy.
6. **For a correlation read on its own under `"preavg"`, give
   `psd=False`.** It selects the balanced form, a shorter window of
   `k_n = floor(theta * sqrt(block_rows))` rows with the bias term
   subtracted, which was the more accurate on the streams Regimes §8
   measures. It is not
   always positive semi-definite, so keep the default, `psd=True`, where
   the matrix must be.

This code makes 20 blocks of two prices whose returns correlate at 0.6,
each tick measured with noise. The second series is observed on only
about 30% of the rows. It carries that series forward, and prints each
estimator's mean correlation over the blocks:

```python
import numpy as np
import polars as pl
import polars_online as po

# Two noisy prices whose returns correlate at 0.6; p2 is observed on about 30% of the rows.
rng = np.random.default_rng(4)
blocks, rows = 20, 1_000
z = rng.standard_normal((2, blocks * rows))
latent = np.cumsum([z[0], 0.6 * z[0] + 0.8 * z[1]], axis=1)  # two efficient prices whose returns correlate at 0.6
observed = latent + rng.normal(0.0, 0.7, latent.shape)        # each tick measured with noise
seen = rng.random(blocks * rows) < 0.3
pl.DataFrame({
    "block": np.repeat(np.arange(blocks), rows),              # a block id: a trading day, say
    "p1": observed[0], "p2": observed[1], "seen": seen,
}).with_columns(
    p2=pl.when("seen").then("p2"),                            # null where p2 was not observed
).drop("seen").write_parquet("blocks.parquet")

# 2. Each price carried forward and differenced within its block.
returns = pl.scan_parquet("blocks.parquet").with_columns(pl.col("p1", "p2").forward_fill().diff().over("block"))
specs = [
    po.spec.rcov(name, features=["p1", "p2"],
                 group="block", group_close="monotone",       # 3. a block closes when the next starts
                 kind=kind,                                   # 1. the estimator
                 block_rows=rows,                             # 4. the block's expected length
                 psd=psd)                                     # 6. False: the balanced pre-averaged form
    for name, kind, psd in [("plain", "plain", True), ("kernel", "kernel", True),
                            ("preavg", "preavg", True), ("balanced", "preavg", False)]
]
bank = po.ModelBank(specs)
bank.fit(returns)
# 5. Each estimator's mean correlation over the blocks that closed.
for name in ("plain", "kernel", "preavg", "balanced"):
    closed = bank.closed_groups(name)                          # one row per block that closed
    print(f"{name}: mean correlation {closed['rcorr'].list.get(1).mean():.2f} over {closed.height} blocks (truth 0.60)")
```

```text
plain: mean correlation 0.11 over 19 blocks (truth 0.60)
kernel: mean correlation 0.47 over 19 blocks (truth 0.60)
preavg: mean correlation 0.64 over 19 blocks (truth 0.60)
balanced: mean correlation 0.58 over 19 blocks (truth 0.60)
```

The plain sum loses most of the correlation to the noise and the stale
prices. The kernel recovers more of it, and the two pre-averaged forms
come closest.

## Check that it worked

**Plant a known correlation in data shaped like yours, as the recipe
does, and read which kind finds it.** Fit the kinds side by side in one
bank: each is its own spec over the same returns.

| what you see | what it means | what to do |
|---|---|---|
| one block fewer than the data holds | under `group_close="monotone"` the last block closes only when a later one arrives | expected; a block that ends the input has no estimate |
| a block's `rcov` and `rcorr` null | the block was too short to estimate from, which gives nulls, not an error | check that the block column groups the rows you mean |
| `rcov_psd_repaired` true | the estimate had to be made positive semi-definite, by clipping a negative eigenvalue under `psd=True` | read the repaired estimate; null there means the repair could not run, as on an entry that is not finite or a failed eigendecomposition |
| a `ValueError` naming `half_life` or `lam` | a block's close ends its memory, so `rcov` refuses a decay | drop `half_life` and `lam` |

Every parameter and every field of the closed block is in `rcov`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.rcov).
