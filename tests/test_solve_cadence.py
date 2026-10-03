"""The default solve cadence of the models that solve on a schedule --
`ewridge`, `lasso`, `huber`, `quantile` -- is by weight (docs/PLAN.md task
115 (b)).

The default was `half-life / 50` of clock, so a half-life much longer than the
stream solved once, at warm-up, and never again: under `half_life=1e6` the fit
after 4,000 rows was the first 20 rows' slope, 0.99, where the stream said
2.95. The default is now a solve once the weight learned since the last
reaches `ln 2 / 50` of the weight the fit holds, which in steady state is the
same `half-life / 50` of clock at any row spacing. An explicit `solve_every`
keeps its clock, and `half_life=inf` or `lam` still solve every row.
"""

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

MODELS = [
    ("ewridge", {}),
    ("lasso", {"lasso_path": [0.0]}),
    ("huber", {}),
    ("quantile", {"quantile": 0.5}),
]


def stream(n=4000, seed=0):
    """The first 100 rows say the slope is 1, the rest say 3: the whole
    stream's least-squares slope is about 2.95."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    slope = np.where(np.arange(n) < 100, 1.0, 3.0)
    return pl.DataFrame({"x0": x, "y": slope * x + 0.1 * rng.standard_normal(n)})


def slope(model, extra, df, **kw):
    spec = getattr(po.spec, model)(
        "m", targets=["y"], features=["x0"], min_weight=20.0, **extra, **kw
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.coef("m")["coef"][1]


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=[m for m, _ in MODELS])
@pytest.mark.parametrize("half_life", [1e6, 1e12])
def test_a_halflife_far_longer_than_the_stream_keeps_solving(model, extra, half_life):
    df = stream()
    every_row = slope(model, extra, df, half_life=float("inf"))
    assert every_row > 2.5  # the stream's slope, fitted every row
    got = slope(model, extra, df, half_life=half_life)
    assert abs(got - every_row) < 0.05, (got, every_row)


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=[m for m, _ in MODELS])
def test_an_explicit_solve_every_keeps_its_clock(model, extra):
    """The rule replaces only the default: a caller who names `solve_every`
    gets that clock, the stale fit included."""
    got = slope(model, extra, stream(), half_life=1e12, solve_every=1e9)
    assert abs(got - 1.0) < 0.05, got


def test_in_steady_state_the_rule_is_the_clocks_cadence():
    """Evenly spaced rows under a half-life of 500: the fit holds
    `1 / (1 - 2**(-1/500))`, about 721.8, and `ln 2 / 50` of that is 10.007
    rows -- the clock's `500 / 50 = 10`, one row later at most. So after
    warm-up the two cadences give fits a row apart, and the outputs agree
    closely."""
    df = stream(n=6000, seed=3)
    by_weight = po.ModelBank(
        [po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=500.0, min_weight=20.0)]
    ).fit_predict(df)
    by_clock = po.ModelBank(
        [
            po.spec.ewridge(
                "m",
                targets=["y"],
                features=["x0"],
                half_life=500.0,
                min_weight=20.0,
                solve_every=10.0,
            )
        ]
    ).fit_predict(df)
    a = by_weight["m"].struct.field("pred_y").to_numpy()[3000:]
    b = by_clock["m"].struct.field("pred_y").to_numpy()[3000:]
    assert np.max(np.abs(a - b)) < 1e-2
    assert math.isclose(math.log(2) / 50 / (1 - 2 ** (-1 / 500)), 10.007, rel_tol=1e-3)
