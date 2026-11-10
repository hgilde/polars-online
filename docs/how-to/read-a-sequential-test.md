<!-- kind: how-to -->
# Read a sequential test

This how-to is for anyone who runs `seqtest` to learn whether a column
tends to be positive, and wants to act the first time the evidence is
enough. It reads the test's evidence on every row, at a level that holds
however often you look.

## Prerequisites

These steps need the library and a stream:

- polars-online installed, and numpy to make the example's data. The
  second recipe reads `lf`, the query over `df`, from
  [Example data](../EXAMPLE-DATA.md).
- A stream saved as a file, or a frame in memory, with the column whose
  sign you test, named as a
  [target](../GLOSSARY.md#the-model-bank-and-what-it-reads) of a
  [`seqtest`](../MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting)
  spec. Each row counts as one trial, so to skip a row, null its target.

## Reject the first time the evidence is enough

A p-value read again on every row inflates its error rate. `seqtest`
keeps an *e-process* instead: the wealth of a gambler who bets on the
next sign. The null is that, given everything so far, the next sign is
no more likely positive than negative. Under it, the wealth `E⁺` is a
nonnegative supermartingale. Ville's inequality then gives
`P(E⁺ ever reaches 1/α) ≤ α`, read at every row, not only the last. The
null is about every row, not only the average: signs that run, positive
for a while and then negative, are even on average but break it, and the
bet that follows the run grows.

1. **Build a `seqtest` spec with the column in `targets=`.**
2. **Add the spec to the query with `online.fit_predict([sign])`.**
3. **Run the query with `collect()`.** Each row gains `log_e_pos_<t>`,
   the log of the evidence that the sign tends positive, and
   `log_e_neg_<t>`, that it tends negative.
4. **Reject on the first row where `log_e_pos_<t> ≥ ln 20`.** That is the
   5% level, however often you look. Read `log_e_neg_<t>` the same way for
   the other side.
5. **For the two-sided question, average the two sides' wealth,
   `(exp(log_e_pos_<t>) + exp(log_e_neg_<t>)) / 2`, and reject on the
   first row where it reaches 20.** The average is an e-value. Average the
   wealth, not its log.

The test assumes no distribution, and it does not see the size of the
values. A column of 60% small gains and 40% large losses counts as
positive, though its mean is below zero. This code makes such a column
and a column with no edge, and prints the first row at which each side
passes the 5% level:

```python
import math

import numpy as np
import polars as pl
import polars_online as po

# Gains of 0.1 on 60% of the rows and losses of 1.0 on the rest; and a column with no edge.
rng = np.random.default_rng(9)
n = 3_000
pl.DataFrame({
    "skewed": np.where(rng.random(n) < 0.6, 0.1, -1.0),   # positive 60% of the time, its mean -0.34
    "fair": rng.normal(0.0, 1.0, n),                      # positive half the time
}).write_parquet("signs.parquet")

# 1 to 3. Test both columns' signs.
sign = po.spec.seqtest("sign", targets=["skewed", "fair"])
out = pl.scan_parquet("signs.parquet").online.fit_predict([sign]).collect().unnest("sign").with_row_index("row")


def first(rows):
    return f"at row {rows[0]}" if len(rows) else "never"


for col in ("skewed", "fair"):
    # 4. The first row at the 5% level, one-sided; 5. and two-sided, from the mean of the wealth.
    two_sided = (pl.col(f"log_e_pos_{col}").exp() + pl.col(f"log_e_neg_{col}").exp()) / 2
    one = out.filter(pl.col(f"log_e_pos_{col}") >= math.log(20))["row"]
    two = out.filter(two_sided >= 20)["row"]
    print(f"{col}: mean {out[col].mean():+.2f}; positive at the 5% level {first(one)}, two-sided {first(two)}")
```

```text
skewed: mean -0.34; positive at the 5% level at row 275, two-sided at row 288
fair: mean -0.01; positive at the 5% level never, two-sided never
```

The two-sided e-value passes 20 a few rows after the one-sided, since
the losing side's wealth stays near 1 or below, so the average reaches 20
only once the winning side's wealth nears 40.

## Test each group

Use a group when the stream holds several series, such as one stock per
`stock_id`, each with its own question.

1. **Give the spec a `group=`.** Each group then keeps a test of its own.
2. **Filter the rows where `log_e_pos_<t> ≥ ln 20`.** Each is a row whose evidence
   has passed the 5% level, which holds however often you look.
3. **Keep each group's first such row** with Polars' `group_by(...,
   maintain_order=True).first()`.

Each group is tested at 5% on its own. For 5% over `k` groups together,
reject at `ln(20 k)`.

This code uses `lf`, the query over `df`, from [Example
data](../EXAMPLE-DATA.md):

```python
import math

signs = lf.online.fit_predict([po.spec.seqtest("signs", targets=["y"], group="stock_id")]).collect().unnest("signs")   # 1
rejected = (signs.filter(pl.col("log_e_pos_y") >= math.log(20))         # 2. the rows at the 5% level ...
            .group_by("stock_id", maintain_order=True).first())         # 3. ... and each stock's first
print(f"stocks whose y tends positive at the 5% level: {rejected.height} of {signs['stock_id'].n_unique()}")
```

```text
stocks whose y tends positive at the 5% level: 2 of 4
```

## Check that it worked

The evidence on a column with no edge should seldom reach `ln 20`: on at
most 5% of such columns, whenever you look. To see it hold, run the test
on a column you know has no edge, as the recipe's `fair`.

| what you see | what it means | what to do |
|---|---|---|
| the evidence passes on a column whose mean is negative | the test reads signs alone, so a column positive more often than not counts as positive | test the quantity whose sign is the question |
| the evidence stays at 0 on the first rows | each gambler stakes nothing until the counts favor its side | read on: the stakes begin once one side leads |
| a `ValueError` naming `weight`, `half_life` or `lam` | the test counts each row as one trial, and refuses them | leave them out, and null a target to skip its row |

The fields are defined in [OUTPUTS.md](../OUTPUTS.md#seqtest), and every
parameter in `seqtest`'s
[docstring](https://hgilde.github.io/polars-online/spec.html#polars_online.spec.seqtest).
For the same test over a frame in memory, use
[`po.eval.seqtest`](https://hgilde.github.io/polars-online/eval.html#polars_online.eval.seqtest).
To test which of two specs predicts closer, [Compare two
models](compare-two-models.md#read-the-comparison-on-every-row) names them
in `seqtest`'s `a` and `b`, and its evidence is read the same way.
