"""Parameters that used to build a spec, run, and produce garbage without a
word (docs/IMPROVEMENTS.md C4). Each now fails at the builder with a message
that names the spec and the parameter, and the legal neighbour of each still
builds.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import polars_online as po

BASE = dict(targets=["y"], features=["x0"])
NAN = float("nan")
INF = float("inf")


def _df(n: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": [float(i % 7) for i in range(n)],
            "y": [float((i % 7) * 2 + 1) for i in range(n)],
            "s": ["a"] * (n // 2) + ["b"] * (n - n // 2),
        }
    )


# (builder, kwargs beyond BASE, expected message fragment)
REJECTED = [
    (po.spec.ewridge, dict(half_life=10.0, ridge=-1.0), "ridge must be finite and >= 0"),
    # The builder's own check since S27: `ridge` left the table of what may be
    # infinite, where only Rust refused it (review 2026-09-12).
    (po.spec.ewridge, dict(half_life=10.0, ridge=INF), "ridge must be finite, got float inf"),
    (po.spec.ewridge, dict(half_life=10.0, ridge=NAN), "ridge must not be NaN"),
    (po.spec.ewridge, dict(half_life=10.0, ridge=[1e-6, -1.0]), "ridge must be finite and >= 0"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, ridge=[1e-6, 1e-6]),
        "ridge lists 0.000001 more than once",
    ),
    (
        po.spec.ewridge,
        dict(half_life=10.0, clock="t", gap_cap=-5.0),
        "gap_cap must be a finite number > 0",
    ),
    (po.spec.ewridge, dict(half_life=10.0, clock="t", gap_cap=NAN), "gap_cap must not be NaN"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, clock="t", gap_cap=10.0, session="s", session_gap=-1.0),
        "session_gap must be >= 0",
    ),
    (
        po.spec.ewridge,
        dict(half_life=10.0, clock="t", gap_cap=10.0, session="s", session_gap=NAN),
        "session_gap must not be NaN",
    ),
    (
        po.spec.ewridge,
        dict(half_life=10.0, solve_every=-1.0),
        "solve_every must be finite and >= 0",
    ),
    (po.spec.ewridge, dict(half_life=10.0, solve_every=NAN), "solve_every must not be NaN"),
    (po.spec.ewridge, dict(half_life=[10.0, 10.0]), "half_life lists 10 more than once"),
    (po.spec.ewridge, dict(half_life=NAN), "half_life must not be NaN"),
    (po.spec.ewridge, dict(half_life=-1.0), "half_life must be > 0"),
    (po.spec.ewridge, dict(lam=NAN), "lam must not be NaN"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, session="s", session_gap=0.0, session_shrink=0.5, long_half_life=-1.0),
        "long_half_life must be > 0",
    ),
    (po.spec.huber, dict(half_life=10.0, ridge=-1.0), "ridge must be finite and >= 0"),
    (po.spec.huber, dict(half_life=10.0, solve_every=-1.0), "solve_every must be finite and >= 0"),
    (
        po.spec.quantile,
        dict(half_life=10.0, quantile=0.5, ridge=-1e-3),
        "ridge must be finite and >= 0",
    ),
    # A plain f64 on the Rust side: refused in Python by name (IMPROVEMENTS U2).
    (po.spec.quantile, dict(half_life=10.0, quantile=0.5, ridge=INF), "ridge must be finite"),
    (po.spec.rls, dict(half_life=10.0, ridge=INF), "ridge must be finite"),
    (
        po.spec.kalman,
        dict(half_life=10.0, coef_half_life=10.0, obs_var=INF),
        "obs_var must be finite",
    ),
    (po.spec.ewridge, dict(half_life=10.0, solve_every=INF), "solve_every must be finite"),
    (
        po.spec.quantile,
        dict(half_life=10.0, quantile=0.5, quantile_eps=NAN),
        "quantile_eps must not be NaN",
    ),
    (po.spec.rls, dict(half_life=10.0, ridge=0.0), "rls ridge must be finite and > 0"),
    (po.spec.rls, dict(half_life=10.0, ridge=-1.0), "rls ridge must be finite and > 0"),
    (
        po.spec.lasso,
        dict(half_life=10.0, lasso_path=[0.1, 0.1]),
        "lasso_path must be strictly decreasing",
    ),
    (
        po.spec.lasso,
        dict(half_life=10.0, lasso_path=[0.1, -0.1]),
        "lasso_path values must be finite and >= 0",
    ),
    (
        po.spec.lasso,
        dict(half_life=10.0, lasso_path=[0.1], select_half_life=0.0),
        "select_half_life must be > 0",
    ),
    (
        po.spec.lasso,
        dict(half_life=10.0, lasso_path=[0.1], tol=0.0),
        "tol must be finite and > 0",
    ),
    (po.spec.kalman, dict(half_life=10.0, coef_half_life=NAN), "coef_half_life must not be NaN"),
    (
        po.spec.kalman,
        dict(half_life=10.0, coef_half_life=10.0, q=[-1.0, 1.0]),
        "q values must be finite and >= 0",
    ),
    (
        po.spec.kalman,
        dict(half_life=10.0, coef_half_life=10.0, obs_var=-1.0),
        "obs_var must be finite and > 0",
    ),
    (
        po.spec.kalman,
        dict(half_life=10.0, coef_half_life=10.0, p0=0.0),
        "p0 must be finite and > 0",
    ),
    # A feature set's own rules, by name (review 2026-09-12, S7): a repeated
    # column split its coefficient across identical slots, and an empty set
    # was refused as "out-of-range indices".
    (
        po.spec.ewridge,
        dict(half_life=10.0, feature_sets={"a": ["x0", "x0"]}),
        'feature set "a" lists "x0" more than once',
    ),
    (po.spec.ewridge, dict(half_life=10.0, feature_sets={"a": []}), 'feature set "a" is empty'),
    # A knob whose switch is off did nothing, without a word (S22).
    (po.spec.ewridge, dict(half_life=10.0, drift_action="reset"), "needs emit_drift"),
    (po.spec.ewridge, dict(half_life=10.0, drift_delta=0.1), "drift_delta needs emit_drift"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, drift_threshold=5.0),
        "drift_threshold needs emit_drift",
    ),
    (po.spec.ewridge, dict(half_life=10.0, average_eta=2.0), "average_eta needs emit_averaged"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, resid_autocorr_lag=2),
        "resid_autocorr_lag needs emit_autocorr",
    ),
    (
        po.spec.ewridge,
        dict(half_life=10.0, long_half_life=100.0),
        "long_half_life needs session_shrink",
    ),
    (
        po.spec.ewridge,
        dict(
            half_life=10.0,
            clock="t",
            gap_cap=10.0,
            session="s",
            session_gap="reset",
            session_shrink=0.5,
            long_half_life=100.0,
        ),
        'session_shrink does not apply with session_gap = "reset"',
    ),
    (
        po.spec.ewridge,
        dict(
            half_life=10.0,
            group="g",
            session="s",
            group_close="session",
            session_shrink=0.5,
            long_half_life=100.0,
        ),
        'session_shrink does not apply with group_close = "session"',
    ),
    (po.spec.ewridge, dict(half_life=10.0, session_gap=5.0), "session_gap needs session"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, restart_after_step_back=1.0),
        "restart_after_step_back needs clock",
    ),
    # The policies that absorbed a step back are gone (task 120); the parser
    # refuses them before a spec has a name, in test_clock_order.py.
    # The cap is finite and above 0; a session gap finite or "reset".
    (po.spec.ewridge, dict(half_life=10.0, clock="t", gap_cap=INF), "gap_cap must be finite"),
    (po.spec.ewridge, dict(half_life=10.0, clock="t", gap_cap=0.0), "gap_cap must be > 0"),
    (
        po.spec.ewridge,
        dict(half_life=10.0, clock="t", gap_cap=10.0, session="s", session_gap=INF),
        "session_gap must be finite",
    ),
    # The window's cadence (docs/PLAN.md task 162): the clock spacing is
    # clock units, finite and >= 0, and the row cap a count of rows; `0` of
    # either is every row, below.
    (
        po.spec.ewridge,
        dict(half_life=10.0, window_size=5.0, window_every=-1.0),
        "window_every must be finite and >= 0 clock units",
    ),
    (
        po.spec.ewridge,
        dict(half_life=10.0, window_size=5.0, max_rows_between_snapshots=-1),
        "max_rows_between_snapshots must be >= 0, got -1",
    ),
]

ACCEPTED = [
    (po.spec.ewridge, dict(half_life=10.0, ridge=0.0)),
    (po.spec.ewridge, dict(half_life=10.0, ridge=[1e-6, 1e-3])),
    (
        po.spec.ewridge,
        dict(half_life=10.0, clock="t", gap_cap=10.0, session="s", session_gap=0.0),
    ),
    (po.spec.ewridge, dict(half_life=10.0, solve_every=0.0)),
    # Every row, as `solve_every = 0` solves on every row (task 162).
    (po.spec.ewridge, dict(half_life=10.0, window_size=5.0, window_every=0)),
    (po.spec.ewridge, dict(half_life=10.0, window_size=5.0, max_rows_between_snapshots=0)),
    (po.spec.ewridge, dict(half_life=INF)),
    (po.spec.ewridge, dict(half_life=[10.0, 20.0])),
    (po.spec.huber, dict(half_life=10.0, ridge=0.0)),
    (po.spec.lasso, dict(half_life=10.0, lasso_path=[0.1, 0.0])),
    (po.spec.kalman, dict(half_life=10.0, coef_half_life=INF, q=[0.0, 0.0])),
    (po.spec.ewridge, dict(half_life=10.0, feature_sets={"a": ["x0"]})),
    (po.spec.ewridge, dict(half_life=10.0, emit_drift=True, drift_action="reset")),
    (po.spec.ewridge, dict(half_life=10.0, ridge=[1e-6, 1.0], emit_averaged=True, average_eta=2.0)),
    (po.spec.ewridge, dict(half_life=10.0, emit_autocorr=True, resid_autocorr_lag=2)),
    (
        po.spec.ewridge,
        dict(
            half_life=10.0,
            clock="t",
            gap_cap=10.0,
            session="s",
            session_gap=0.0,
            session_shrink=0.5,
            long_half_life=100.0,
        ),
    ),
    (
        po.spec.ewridge,
        dict(
            half_life=10.0,
            clock="t",
            gap_cap=10.0,
            restart_after_step_back=0.0,
        ),
    ),
]


def _label(case):
    builder, kw = case[0], case[1]
    return builder.__name__ + ":" + ",".join(f"{k}={v}" for k, v in kw.items())


@pytest.mark.parametrize("builder,kw,msg", REJECTED, ids=[_label(c) for c in REJECTED])
def test_bad_parameters_are_refused_by_name(builder, kw, msg):
    with pytest.raises(ValueError) as exc:
        builder("m", **BASE, **kw)
    text = str(exc.value)
    assert msg in text, text
    assert 'spec "m"' in text, text


@pytest.mark.parametrize("builder,kw", ACCEPTED, ids=[_label(c) for c in ACCEPTED])
def test_the_legal_neighbours_still_run(builder, kw):
    spec = builder("m", **BASE, **kw)
    out = po.ModelBank([spec]).fit_predict(_df()).unnest("m")
    for col in [c for c in out.columns if c.startswith("weight_sum")]:
        assert all(math.isfinite(v) for v in out[col].to_list()), col


def test_a_nan_deep_in_a_list_names_the_parameter():
    with pytest.raises(ValueError, match='spec "m": coef_prior must not be NaN'):
        po.spec.ewridge("m", half_life=10.0, coef_prior=[[0.0, NAN]], **BASE)


def test_a_feature_set_named_twice_is_refused_by_name():
    """A hand-written list can name one set twice. The bank's field-name
    tripwire caught it, under a comment saying the case could not arise
    (review 2026-09-12, S7)."""
    spec = po.spec.ewridge("m", half_life=10.0, **BASE)
    spec["model"]["feature_sets"] = [["a", ["x0"]], ["a", ["x0"]]]
    with pytest.raises(ValueError, match='spec "m": feature_sets names "a" more than once'):
        po.ModelBank([spec])


@pytest.mark.parametrize("name", ["", "spec", "group"])
def test_a_spec_name_the_bank_cannot_carry_is_refused(name):
    """``last_row`` puts ``spec`` and ``group`` columns beside a struct named
    after the spec, so a spec of either name collided there, and an empty
    name names no struct at all (S7)."""
    with pytest.raises(ValueError, match="name"):
        po.spec.ewridge(name, half_life=10.0, **BASE)


def test_holt_takes_its_level_halflife_once():
    """``half_life`` and ``level_half_life`` are one knob under two names.
    Given both, the level and ``weight_sum`` followed one and ``sigma`` the other
    (S22)."""
    with pytest.raises(ValueError, match="half_life and level_half_life"):
        po.spec.holt("m", targets=["y"], half_life=10.0, level_half_life=20.0)
    po.spec.holt("m", targets=["y"], half_life=20.0)
    po.spec.holt("m", targets=["y"], level_half_life=20.0)


def test_holt_takes_a_trend_halflife_only_with_a_trend():
    """``trend=False`` holds the trend at zero (docs/PLAN.md task 115, S30),
    so a ``trend_half_life`` beside it would be read by nothing."""
    with pytest.raises(ValueError, match="trend_half_life applies only with a trend"):
        po.spec.holt("m", targets=["y"], half_life=20.0, trend_half_life=80.0, trend=False)
    po.spec.holt("m", targets=["y"], half_life=20.0, trend=False)
    po.spec.holt("m", targets=["y"], half_life=20.0, trend_half_life=80.0)


@pytest.mark.parametrize(
    "builder,kw",
    [
        (po.spec.ew_cov, dict(features=["x0", "y"], half_life=10.0)),
        (po.spec.marginal, dict(**BASE, half_life=10.0)),
        (po.spec.bocpd, dict(features=["x0"], prior_scale=[1.0])),
    ],
    ids=["ew_cov", "marginal", "bocpd"],
)
def test_coef_every_is_refused_where_there_is_no_coef(builder, kw):
    """A model with no coefficients has nothing for ``coef_every`` to emit
    (S22)."""
    with pytest.raises(ValueError, match="coef_every"):
        builder("m", coef_every=10, **kw)
    builder("m", **kw)


def test_add_intercept_moves_no_warm_up_where_there_is_no_intercept():
    """``ew_cov`` has no intercept, yet its default ``min_weight`` was
    ``k + fit_intercept``, so the flag moved its first reported row (S22)."""
    first = []
    for flag in (True, False):
        spec = po.spec.ew_cov(
            "m", features=["x0"], stats=["mean"], half_life=10.0, fit_intercept=flag
        )
        out = po.ModelBank([spec]).fit_predict(_df(20)).unnest("m")
        mean = next(c for c in out.columns if c.startswith("mean"))
        first.append(out[mean].is_not_null().arg_max())
    assert first[0] == first[1], first


def test_marginal_takes_a_window_with_lags_only_by_name():
    """Lags under a window cost each snapshot the lag moments (docs/PLAN.md
    task 137), and the spec says so before it pays: without
    ``window_lags=True`` the pair is refused with this spec's own numbers;
    with it, it is taken; ``window_lags`` without both is refused. It was
    refused outright (C18), since the lag ring kept no snapshot."""
    with pytest.raises(ValueError, match="need window_lags = true") as exc:
        po.spec.marginal("m", half_life=10.0, window_size=50.0, lags=[1], **BASE)
    # BASE's one feature and one target at one lag, the cross lag at it:
    # L·T + (L + 2C)·p·T = 1 + 3 = 4 doubles beside (3p + 5)·T = 8.
    assert "4 doubles beside the 8" in str(exc.value), exc.value
    assert "1.5 times the size" in str(exc.value), exc.value
    po.spec.marginal("m", half_life=10.0, window_size=50.0, lags=[1], window_lags=True, **BASE)
    for kw in ({"window_size": 50.0}, {"lags": [1]}, {}):
        with pytest.raises(ValueError, match="window_lags applies only with both"):
            po.spec.marginal("m", half_life=10.0, window_lags=True, **kw, **BASE)
    po.spec.marginal("m", half_life=10.0, window_size=50.0, **BASE)
    po.spec.marginal("m", half_life=10.0, lags=[1], **BASE)


def test_every_door_fills_validates_and_builds_a_spec_alike():
    """The bank filled a spec's defaults, validated it and built its models;
    ``output_fields``, ``output_index`` and ``coef_fields`` only validated.
    So a dict for a model with no target and no ``targets`` -- which the bank
    fills from ``features[0]`` -- was refused there, and a spec the core
    refuses was given field names there (S25)."""
    e53 = {"name": "c", "model": {"type": "ew_cov"}, "features": ["x0", "y"], "half_life": 10.0}
    po.ModelBank([e53])
    assert po.spec.output_fields(e53)
    assert po.spec.output_index(e53).height > 0
    assert po.spec.coef_fields(e53).height == 0
    refused = {
        "name": "r",
        "model": {"type": "ew_ridge", "window_size": 10.0, "ridge_scale": "sum"},
        "targets": ["y"],
        "features": ["x0"],
        "half_life": 10.0,
    }
    doors = [po.spec.output_fields, po.spec.output_index, po.spec.coef_fields, po.ModelBank]
    for door in doors:
        with pytest.raises(ValueError, match="ridge_scale"):
            door([refused] if door is po.ModelBank else refused)
