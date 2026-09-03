"""Row order only reaches the fit through decay.

A bank keeps sufficient statistics, so with decay off (``halflife=inf`` or
``lam=1.0``) an ``ewridge`` with ``ridge=0`` *is* ordinary least squares over
every row it has seen, in whatever order the rows came -- one stream per
group, the groups interleaved however the file has them. A finite halflife
without a clock discounts by position in the stream instead; and a huge
*finite* halflife is not ``inf``, because the solve cadence it inherits
(``halflife/50``) never comes due. The README's "Any row order" section
states all three; this file keeps it honest.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

K = 4
FEATURES = [f"x{j}" for j in range(K)]


def _frame(n: int = 2000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, K))
    g = rng.integers(0, 2, n)
    # Two groups with different coefficients, interleaved at random.
    beta = np.where(g[:, None] == 0, np.arange(1, K + 1) / K, -np.arange(1, K + 1) / K)
    y = 0.7 + (x * beta).sum(axis=1) + 0.1 * rng.standard_normal(n)
    return pl.DataFrame({f"x{j}": x[:, j] for j in range(K)}).with_columns(
        y=pl.Series(y), g=pl.Series(g).cast(pl.Int32)
    )


def _ols(df: pl.DataFrame) -> np.ndarray:
    a = np.column_stack([np.ones(df.height), df.select(FEATURES).to_numpy()])
    return np.linalg.lstsq(a, df["y"].to_numpy(), rcond=None)[0]


def _orders(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        "as given": df,
        "reversed": df.reverse(),
        "shuffled": df.sample(fraction=1.0, shuffle=True, seed=7),
        "sorted by group": df.sort("g", maintain_order=True),
    }


@pytest.mark.parametrize(
    "decay", [{"halflife": math.inf}, {"lam": 1.0}], ids=["halflife=inf", "lam=1"]
)
def test_no_decay_is_least_squares_in_any_row_order(decay):
    df = _frame()
    spec = po.spec.ewridge(
        "ols", targets=["y"], features=FEATURES, ridge=0.0, group="g", min_periods=K + 1, **decay
    )
    expected = {g: _ols(df.filter(pl.col("g") == g)) for g in (0, 1)}
    for order, frame in _orders(df).items():
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        coef = bank.coef("ols")
        for g in (0, 1):
            got = coef.filter(pl.col("group") == str(g)).sort("position")["coef"].to_numpy()
            np.testing.assert_allclose(got, expected[g], atol=1e-10, err_msg=f"{order}, group {g}")
        # The plan, chunked, is the same bank.
        streamed = (
            frame.lazy().online.fit_predict([spec], chunk_rows=97).online.unnest([spec]).collect()
        )
        last = streamed.filter(pl.col("g") == 0).tail(1)
        np.testing.assert_allclose(
            last.select("^coef_.*$").row(0), expected[0], atol=1e-10, err_msg=f"{order}, plan"
        )


def test_a_row_halflife_without_a_clock_discounts_by_position():
    """No clock: the row count is the clock, so a finite halflife weights row
    i by ``0.5 ** ((n-1-i) / halflife)`` in whatever order the rows are fed --
    reversing the frame reverses the weights, and both are the weighted least
    squares of that order."""
    df = _frame(n=600).drop("g")
    h = 150.0
    spec = po.spec.ewridge(
        "ew", targets=["y"], features=FEATURES, ridge=0.0, halflife=h, solve_every=0
    )
    x = np.column_stack([np.ones(df.height), df.select(FEATURES).to_numpy()])
    for frame, xs, ys in (
        (df, x, df["y"].to_numpy()),
        (df.reverse(), x[::-1], df["y"].to_numpy()[::-1]),
    ):
        bank = po.ModelBank([spec])
        bank.fit_predict(frame)
        w = np.sqrt(0.5 ** ((frame.height - 1 - np.arange(frame.height)) / h))
        wls = np.linalg.lstsq(xs * w[:, None], ys * w, rcond=None)[0]
        np.testing.assert_allclose(
            bank.coef("ew").sort("position")["coef"].to_numpy(), wls, atol=1e-9
        )


def test_a_huge_finite_halflife_is_not_inf():
    """The trap the README warns about: the default solve cadence is
    ``halflife/50`` clock units, so ``halflife=1e12`` on a 2000-row stream
    solves once (at ``min_periods``) and never again -- the prediction is the
    stale fit -- while ``halflife=inf`` re-solves every row and ``solve_every``
    makes the finite case do the same. If this test starts failing because
    the schedule changed, update the README's "Any row order" section and the
    solve-schedule paragraph in docs/PLAN.md (2026-09-03) as well."""
    df = _frame(n=2000).drop("g")
    ols = _ols(df)

    def coef(**kw):
        bank = po.ModelBank(
            [
                po.spec.ewridge(
                    "m", targets=["y"], features=FEATURES, ridge=0.0, min_periods=30, **kw
                )
            ]
        )
        bank.fit_predict(df)
        return bank.coef("m").sort("position")["coef"].to_numpy()

    stale = coef(halflife=1e12)
    assert np.abs(stale - ols).max() > 1e-3, "the default cadence now re-solves; update the docs"
    np.testing.assert_allclose(coef(halflife=1e12, solve_every=1.0), ols, atol=1e-6)
    np.testing.assert_allclose(coef(halflife=1e12, max_rows_between_solves=1), ols, atol=1e-6)
    np.testing.assert_allclose(coef(halflife=math.inf), ols, atol=1e-10)
