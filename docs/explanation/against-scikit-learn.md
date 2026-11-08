<!-- kind: explanation -->
# polars-online against scikit-learn

This page explains, for anyone who would otherwise fit a streaming
regression with scikit-learn, how this library's regressions differ from
scikit-learn's `SGDRegressor.partial_fit`, and where each one wins.

**This library's main regressions solve exactly, where scikit-learn's
streaming estimator takes gradient steps.**
[`SGDRegressor.partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html),
the scikit-learn estimator most people compare with, is a first-order
stochastic optimizer: its answer depends on the learning rate, the
schedule, the feature scaling and the row order. `ewridge`, `rls`,
`lasso`, `huber` and `quantile` keep running sums and solve from
them, so they have no learning rate. With decay off
(`half_life=float("inf")`) and `ridge=0`, `ewridge` is ordinary least
squares to 2e-13 of `numpy.linalg.lstsq`, in any row order. The like for
like is [`sgd`](../MODELS.md#sgd--stochastic-gradient-descent), the cheap
`O(k)` baseline: at the same constant step it gives `SGDRegressor`'s R² to
four places, because the two run the same recursion.

## What a prediction saw

**Every contender reaches the noise ceiling on a stationary stream, and
the gaps come from what a prediction saw.** The comparison ran on one
generated stream of 100,000 rows with `k = 20`, on 2026-09-08. Each
contender ran at its best over a sweep of its own settings, by
`scripts/sklearn_comparison.py` on scikit-learn 1.9.0. `ewridge` solves on
every row here, and its rate was measured again on 2026-09-29, on the same
stream.
The noise ceiling is the R² of the generating signal itself:

| contender | R² stationary | R² drifting | rows/sec | what a prediction saw |
|---|---:|---:|---:|---|
| noise ceiling | 0.9831 | 0.9923 | | |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 | 2,200,000 | every row before its *batch* |
| `po.spec.sgd` | 0.9826 | 0.9906 | 6,000,000 | every row before it |
| `po.spec.ewridge` | **0.9831** | **0.9907** | 480,000 | every row before it |

scikit-learn's fast form updates once per 1,000-row batch, so a prediction can
come up to 999 rows before its update, and on the drifting stream it
scores 0.9820. Called on every row, `partial_fit` runs at 3,400 rows a
second, which is Python's overhead on each row.

## What one fit can hold

**The two differ in what one fit can hold:**

| | scikit-learn | here |
|---|---|---|
| a grid of six penalties | 3.8× the work: six estimators | 1.6× the work: one [Gram](../GLOSSARY.md#fitting-a-model), six solves |
| several targets | one estimator each | one shared `X'X` |
| one fit per group | a dict of estimators | `group=`, one state per group |
| standardization | a `StandardScaler` fitted on the whole frame can leak: its means and spreads include rows after the one predicted | streaming, so it cannot leak |
| chunking | | chunk invariance is a test |
| saved model | a pickle | a versioned file that loads on every OS |

## Where each one wins

**The exact solve wins in the first hundred rows of every group.** An
exact solve is right as soon as its Gram, `X'X`, is full rank, about `k`
rows in, where a first-order method needs about `1/eta` rows per
direction, `eta` being its learning rate. On 500 groups of
200 rows at `k = 20`, `ewridge` scores an R² of 0.9693 on rows 25 to 50
against `SGDRegressor`'s 0.7277 at its best setting.

**On a very wide row a gradient step wins: batched `SGDRegressor` beats an
exact solve, and `sgd` beats both.** At
`k = 10,000`, `ewridge` keeps a `(k+1)²` matrix, 800 MB of a state of 860
MB, and updates its lower triangle on every row, or once a block under
`gram_block_rows`. Measured on 2026-11-08 over 2,000 rows, at a load
of 9 to 18, the update alone ran 185 rows a second, or 344 with
`gram_block_rows=1024`. It was timed with a `min_weight` above the row
count, so that no solve ran. A solve factors the whole matrix. With
`solve_every=inf` and `max_rows_between_solves=1000`, which solve every
1,000 rows and no more often, the same rows ran at 171 rows a second, or
584 with `gram_block_rows=256` (2026-11-08, at a load of 16 to 21). A
finite `solve_every` costs more here: until its first period passes, the
model also solves each time its weight grows by about 1.4%. With fewer rows
than coefficients, no fit was ready to predict any row. In the same run,
`sgd` at a learning rate of `0.2 / k` ran 25,678 rows a second from
0.89 MB, faster than `SGDRegressor`'s 19,180 in batches.
`scripts/sklearn_comparison.py` prints the full tables, with the grid
each contender was swept over.

**scikit-learn wins on what surrounds a model:** pipelines,
`GridSearchCV`, calibration, and a far larger body of users.
