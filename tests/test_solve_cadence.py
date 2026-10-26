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
    ("lasso", {"penalties": [0.0]}),
    # The refit moves a target's fit on its own row, off the cadence held
    # here (task 277).
    ("huber", {"huber_refit_rows": 0}),
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
    gets that clock, the stale fit included. `ewridge` first solves where
    the weight reaches `min_weight` (20 rows of slope 1); `huber`,
    `quantile` and `lasso` where their target has a row per coefficient, two
    here, whatever `min_weight` says (task 278, D10): the fit of the first
    two rows, held."""
    got = slope(model, extra, stream(), half_life=1e12, solve_every=1e9)
    if model in ("huber", "quantile", "lasso"):
        first = slope(model, extra, stream().head(2), half_life=1e12, solve_every=1e9)
        assert got == first, (got, first)
        assert abs(got - 2.95) > 1.0, got
    else:
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
        model = json.loads(bank.write_json())["states"][0][0][1]["models"][0]["model"]["EwRidge"]
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


# ---- the cadence relation (task 278, batch 27) ----
#
# A schedule decides only *when* a model solves, never what a solve reads: the
# sums are updated on every row either way. So on each row a scheduled
# `ewridge` solves, its `coef` is the every-row model's, to the bit, and the
# prediction on the row after reads that fit too.
# Measured 2026-10-26 on `relation_frame`: 40, 277 and 564 solve rows (of
# 2,000) for `solve_every=50`, a row cap of 7 and the default, every one
# bit-identical. `lasso`'s descent warm-starts from its last solve, so a
# scheduled path reaches the every-row optimum from elsewhere: equal within
# the descent's stopping rule, not to the bit (below).


def relation_frame(seed: int = 278, n: int = 2000, gaps: bool = True) -> pl.DataFrame:
    """Three features of unlike scales and means, a slope that changes half
    way, two targets -- ``z`` null where ``x0`` is high, so its rows are its
    own -- uneven weights with a zero on the first row and every 37th, and a
    clock column that is the row number."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = rng.standard_normal((n, 3)) * [1.0, 2.0, 0.5] + [0.5, -1.0, 3.0]
    beta = np.array([2.0, -1.0, 0.5])
    y = 1.0 + x @ beta + np.where(t < n / 2, 0.0, 1.5) * x[:, 0] + rng.standard_normal(n)
    z = 0.5 - x @ beta[::-1] + rng.standard_normal(n)
    if gaps:
        z[x[:, 0] > 1.2] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[::37] = 0.0
    return pl.DataFrame(
        {"t": t, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y, "w": w}
        | {"z": pl.Series(z).fill_nan(None)}
    )


RELATION = dict(
    targets=["y", "z"],
    features=["x0", "x1", "x2"],
    clock="t",
    gap_cap=1e9,
    weight="w",
    half_life=200.0,
    min_weight=10.0,
    coef_every=0,
)

#: The three schedules: the clock, a row cap under a clock that never falls
#: due, and the default weight cadence.
SCHEDULES = {
    "solve_every": {"solve_every": 50.0},
    "row cap": {"max_rows_between_solves": 7, "solve_every": 1e9},
    "default": {},
}


def relation_run(model: str, df: pl.DataFrame, **kw) -> pl.DataFrame:
    spec = getattr(po.spec, model)("m", **{**RELATION, **kw})
    return po.ModelBank([spec], threads=1).fit_predict(df).unnest("m")


def coef_rows(out: pl.DataFrame) -> list[np.ndarray | None]:
    return [
        None if c is None else np.array([np.nan if e is None else e for e in c], dtype=float)
        for c in out["coef"].to_list()
    ]


def solve_rows(coef: list[np.ndarray | None]) -> list[int]:
    """The rows a solve moved the fit on (``coef`` is written on every row
    under ``coef_every=0`` and is the fit after the row)."""
    return [
        i
        for i in range(1, len(coef))
        if coef[i] is not None
        and (coef[i - 1] is None or not np.array_equal(coef[i], coef[i - 1], equal_nan=True))
    ]


@pytest.mark.parametrize("schedule", SCHEDULES.values(), ids=SCHEDULES.keys())
def test_on_each_solve_row_the_fit_is_the_every_row_models_to_the_bit(schedule):
    df = relation_frame()
    every = relation_run("ewridge", df, max_rows_between_solves=1)
    sched = relation_run("ewridge", df, **schedule)
    ce, cs = coef_rows(every), coef_rows(sched)
    solved = solve_rows(cs)
    print(f"{schedule}: {len(solved)} solve rows of {df.height}")
    # The evidence the schedule acts: far fewer solves than rows, and the
    # every-row model's fit moves on rows the schedule holds.
    assert 30 <= len(solved) < df.height // 3
    assert len(solve_rows(ce)) > 1.5 * len(solved)
    for i in solved:
        assert np.array_equal(cs[i], ce[i], equal_nan=True), i
    support_e, support_s = every["support_coef"].to_list(), sched["support_coef"].to_list()
    assert all(support_s[i] == support_e[i] for i in solved)
    after = [i + 1 for i in solved if i + 1 < df.height]
    for c in ("pred_y", "pred_z"):
        np.testing.assert_array_equal(
            sched[c].to_numpy().astype(float)[after], every[c].to_numpy().astype(float)[after]
        )


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=[m for m, _ in MODELS])
def test_solve_every_zero_is_a_row_cap_of_one(model, extra):
    """``solve_every=0`` solves on every row, as a row cap of 1 does: the
    same output, every field, to the bit, on two targets with gaps and
    zero-weight rows."""
    df = relation_frame(n=600)
    zero = relation_run(model, df, solve_every=0.0, **extra)
    cap = relation_run(model, df, max_rows_between_solves=1, **extra)
    assert zero["pred_y"].is_not_null().sum() > 500
    assert zero.equals(cap, null_equal=True)


def test_a_row_cap_longer_than_the_cadence_changes_nothing():
    """The default cadence solves at least every few rows at this half-life,
    so a cap of 200 rows never falls due first: the output is the same."""
    df = relation_frame()
    default = relation_run("ewridge", df)
    capped = relation_run("ewridge", df, max_rows_between_solves=200)
    spacing = int(np.diff(solve_rows(coef_rows(default))).max())
    print(f"the default cadence's longest spacing: {spacing} rows")
    assert spacing < 200
    assert default.equals(capped, null_equal=True)
    # And a cap shorter than the cadence does act.
    assert not default.equals(relation_run("ewridge", df, max_rows_between_solves=2))


@pytest.mark.parametrize("schedule", SCHEDULES.values(), ids=SCHEDULES.keys())
def test_on_each_solve_row_lasso_is_the_every_row_path_within_its_rule(schedule):
    """``lasso`` warm-starts each descent from its last solve's path, so a
    scheduled solve starts further from the optimum than the every-row
    model's and stops at another point inside the rule. At ``tol=1e-14`` the
    two agree on every solve row to 3.1e-14, 3.6e-15 and 1.8e-15 (the three
    schedules, measured 2026-10-26; 32 of 40, 230 of 277 and 507 of 564
    rows to the bit), held to 1e-12."""
    df = relation_frame()
    path = dict(penalties=[0.5, 0.05], tol=1e-14, max_iter=100_000)
    every = relation_run("lasso", df, max_rows_between_solves=1, **path)
    sched = relation_run("lasso", df, **schedule, **path)
    ce, cs = coef_rows(every), coef_rows(sched)
    solved = solve_rows(cs)
    assert 30 <= len(solved) < df.height // 3
    worst = max(float(np.nanmax(np.abs(cs[i] - ce[i]))) for i in solved)
    print(f"{schedule}: {len(solved)} solve rows, worst |d coef| {worst:.3g}")
    assert worst < 1e-12
