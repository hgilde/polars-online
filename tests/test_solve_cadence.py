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

import json
import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

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
    # The cadence itself, read from the model, where this line was the
    # docstring's arithmetic (review 2026-10-05, TC9): the model's share is
    # `ln 2 / 50`, and in steady state its solves fall the clock's ten rows
    # apart, one more at most (ten, measured).
    bank = po.ModelBank(
        [po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=500.0, min_weight=20.0)]
    )
    bank.fit_predict(df.slice(0, 3000))
    solved = []
    for i in range(3000, 3200):
        bank.fit_predict(df.slice(i, 1))
        model = json.loads(bank.to_json())["states"][0][0][1]["models"][0]["model"]["EwRidge"]
        if model["rows_since_solve"] == 0:
            solved.append(i)
    assert model["cfg"]["solve_share"] == pytest.approx(math.log(2) / 50, rel=1e-12)
    gaps = set(np.diff(solved).tolist())
    assert len(solved) >= 18 and gaps <= {10, 11}, gaps


#: Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
T0_NS = 1_704_067_200_000_000_000


def millisecond_rows(n: int, unit: str) -> tuple[pl.DataFrame, np.ndarray]:
    """``n`` rows 1 ms apart on a ``Datetime`` clock in ``unit``, with a
    feature and a target, and the same instants in integer nanoseconds."""
    ns = T0_NS + np.arange(n, dtype=np.int64) * 1_000_000
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    clock = pl.Series("t", ns).cast(pl.Datetime("ns")).cast(pl.Datetime(unit))
    return pl.DataFrame({"t": clock, "x0": x, "y": 2.0 * x + rng.standard_normal(n)}), ns


def solves(out: pl.DataFrame) -> list[int]:
    """The rows a solve happened at: where ``coef``, written on every row,
    changes (a row's ``coef`` is the fit after it)."""
    coef = out["m"].struct.field("coef").to_list()
    return [i for i, c in enumerate(coef) if c is not None and (i == 0 or c != coef[i - 1])]


def on_the_clock(first: int, ns: np.ndarray, span_ns: int) -> list[int]:
    """The definition over the raw nanoseconds: after the first solve, each
    at the first row whose instant is at least ``span_ns`` past the last
    solve's."""
    rows, last = [first], first
    for i in range(first + 1, len(ns)):
        if int(ns[i]) - int(ns[last]) >= span_ns:
            rows.append(i)
            last = i
    return rows


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=[m for m, _ in MODELS])
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_a_clock_cadence_is_decided_on_the_exact_clock(model, extra, unit):
    """Task 180: two thousand steps of 1 ms summed in doubles are
    1.9999999999998905 s, so ``solve_every="2s"`` solved a row late, and
    every solve after it a row later again. The cadence is now decided on
    the decayed clock held exactly, so the solves fall every 2,000th row, as
    the raw nanoseconds say, in every unit."""
    df, ns = millisecond_rows(6_100, unit)
    spec = getattr(po.spec, model)(
        "m",
        targets=["y"],
        features=["x0"],
        clock="t",
        gap_cap="1h",
        half_life="inf",
        min_weight=0.0,
        solve_every="2s",
        coef_every=0,
        **extra,
    )
    got = solves(po.ModelBank([spec]).fit_predict(df))
    assert got == on_the_clock(got[0], ns, 2_000_000_000)
    assert np.diff(got).tolist() == [2_000] * 3, got
