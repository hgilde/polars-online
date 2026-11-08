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
every row here, and its rate was re-measured on 2026-09-29
([PERFORMANCE §30](../PERFORMANCE.md#30-where-every-row-solves-2026-09-29)).
The noise ceiling is the R² of the generating signal itself:

| contender | R² stationary | R² drifting | rows/sec | what a prediction saw |
|---|---:|---:|---:|---|
| noise ceiling | 0.9831 | 0.9923 | | |
| `SGDRegressor`, row by row | 0.9829 | 0.9899 | 3,400 | every row before it |
| `SGDRegressor`, batches of 1,000 | 0.9830 | 0.9820 | 2,200,000 | every row before its *batch* |
| `po.spec.sgd` | 0.9826 | 0.9906 | 6,000,000 | every row before it |
| `po.spec.ewridge` | **0.9831** | **0.9907** | 480,000 | every row before it |

sklearn's fast form updates once per 1,000-row batch, so a prediction can
come up to 999 rows before its update, and on the drifting stream it
scores 0.9820. Called on every row, `partial_fit` runs at 3,400 rows a
second, which is Python's overhead on each row. sklearn has the ecosystem:
pipelines, `GridSearchCV`, calibration, and far more use.

## What one fit can hold

**The two differ in what one fit can hold:**

| | sklearn | here |
|---|---|---|
| a grid of six penalties | 3.8× the work: six estimators | 1.6× the work: one [Gram](../GLOSSARY.md#fitting-a-model), six solves |
| several targets | one estimator each | one shared `X'X` |
| one fit per group | a dict of estimators | `group=`, one state per group |
| standardization | a `StandardScaler` fitted on the whole frame can leak | streaming, so it cannot leak |
| chunking | | chunk invariance is a test |
| saved model | a pickle | a versioned file that loads on every OS |

## Where each one wins

**The exact solve wins in the first hundred rows of every group.** An
exact solve is right as soon as its Gram, `X'X`, is full rank, about `k`
rows in, where a first-order method needs about `1/eta` rows per
direction, `eta` being its learning rate. On 500 groups of
200 rows, `ewridge` scores an R² of 0.9693 on rows 25 to 50 against
`SGDRegressor`'s 0.7277.

**On a very wide row a gradient step wins: batched `SGDRegressor` beats an
exact solve, and `sgd` beats both.** At
`k = 10,000`, `ewridge` keeps a `(k+1)²` matrix, 800 MB of a state of 860
MB, and moves all of it on every row, so it runs 53 rows a second, or 379 with
`gram_block_rows=1024`. There, `sgd` at a learning rate of `0.2 / k` runs
33,271 rows a second from 0.89 MB, faster than `SGDRegressor`'s 20,007 in
batches.
[PERFORMANCE §19](../PERFORMANCE.md#19-against-sklearnlinear_modelsgdregressor-2026-09-08)
has the full tables, the sweeps, and why a half-life gives no edge on
evenly spaced rows.
