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
    (po.spec.ewridge, dict(halflife=10.0, ridge=-1.0), "ridge must be finite and >= 0"),
    (po.spec.ewridge, dict(halflife=10.0, ridge=INF), "ridge must be finite and >= 0"),
    (po.spec.ewridge, dict(halflife=10.0, ridge=NAN), "ridge must not be NaN"),
    (po.spec.ewridge, dict(halflife=10.0, ridge=[1e-6, -1.0]), "ridge must be finite and >= 0"),
    (
        po.spec.ewridge,
        dict(halflife=10.0, ridge=[1e-6, 1e-6]),
        "ridge lists 0.000001 more than once",
    ),
    (po.spec.ewridge, dict(halflife=10.0, clock="t", max_dclock=-5.0), "max_dclock must be >= 0"),
    (po.spec.ewridge, dict(halflife=10.0, clock="t", max_dclock=NAN), "max_dclock must not be NaN"),
    (
        po.spec.ewridge,
        dict(halflife=10.0, clock="t", max_dclock=10.0, session="s", session_gap=-1.0),
        "session_gap must be >= 0",
    ),
    (
        po.spec.ewridge,
        dict(halflife=10.0, clock="t", max_dclock=10.0, session="s", session_gap=NAN),
        "session_gap must not be NaN",
    ),
    (po.spec.ewridge, dict(halflife=10.0, solve_every=-1.0), "solve_every must be finite and >= 0"),
    (po.spec.ewridge, dict(halflife=10.0, solve_every=NAN), "solve_every must not be NaN"),
    (po.spec.ewridge, dict(halflife=[10.0, 10.0]), "halflife lists 10 more than once"),
    (po.spec.ewridge, dict(halflife=NAN), "halflife must not be NaN"),
    (po.spec.ewridge, dict(halflife=-1.0), "halflife must be > 0"),
    (po.spec.ewridge, dict(lam=NAN), "lam must not be NaN"),
    (
        po.spec.ewridge,
        dict(halflife=10.0, session="s", session_gap=0.0, session_shrink=0.5, long_halflife=-1.0),
        "long_halflife must be > 0",
    ),
    (po.spec.huber, dict(halflife=10.0, ridge=-1.0), "ridge must be finite and >= 0"),
    (po.spec.huber, dict(halflife=10.0, solve_every=-1.0), "solve_every must be finite and >= 0"),
    (
        po.spec.quantile,
        dict(halflife=10.0, quantile=0.5, ridge=-1e-3),
        "ridge must be finite and >= 0",
    ),
    # A plain f64 on the Rust side: refused in Python by name (IMPROVEMENTS U2).
    (po.spec.quantile, dict(halflife=10.0, quantile=0.5, ridge=INF), "ridge must be finite"),
    (po.spec.rls, dict(halflife=10.0, ridge=INF), "ridge must be finite"),
    (
        po.spec.kalman,
        dict(halflife=10.0, coef_halflife=10.0, obs_var=INF),
        "obs_var must be finite",
    ),
    (po.spec.ewridge, dict(halflife=10.0, solve_every=INF), "solve_every must be finite"),
    (
        po.spec.quantile,
        dict(halflife=10.0, quantile=0.5, quantile_eps=NAN),
        "quantile_eps must not be NaN",
    ),
    (po.spec.rls, dict(halflife=10.0, ridge=0.0), "rls ridge must be finite and > 0"),
    (po.spec.rls, dict(halflife=10.0, ridge=-1.0), "rls ridge must be finite and > 0"),
    (
        po.spec.lasso,
        dict(halflife=10.0, lasso_path=[0.1, 0.1]),
        "lasso_path must be strictly decreasing",
    ),
    (
        po.spec.lasso,
        dict(halflife=10.0, lasso_path=[0.1, -0.1]),
        "lasso_path values must be finite and >= 0",
    ),
    (
        po.spec.lasso,
        dict(halflife=10.0, lasso_path=[0.1], select_halflife=0.0),
        "select_halflife must be > 0",
    ),
    (
        po.spec.lasso,
        dict(halflife=10.0, lasso_path=[0.1], cd_tol=0.0),
        "cd_tol must be finite and > 0",
    ),
    (po.spec.kalman, dict(halflife=10.0, coef_halflife=NAN), "coef_halflife must not be NaN"),
    (
        po.spec.kalman,
        dict(halflife=10.0, coef_halflife=10.0, q=[-1.0, 1.0]),
        "q values must be finite and >= 0",
    ),
    (
        po.spec.kalman,
        dict(halflife=10.0, coef_halflife=10.0, obs_var=-1.0),
        "obs_var must be finite and > 0",
    ),
    (po.spec.kalman, dict(halflife=10.0, coef_halflife=10.0, p0=0.0), "p0 must be finite and > 0"),
    # A feature set's own rules, by name (review 2026-09-12, S7): a repeated
    # column split its coefficient across identical slots, and an empty set
    # was refused as "out-of-range indices".
    (
        po.spec.ewridge,
        dict(halflife=10.0, feature_sets={"a": ["x0", "x0"]}),
        'feature set "a" lists "x0" more than once',
    ),
    (po.spec.ewridge, dict(halflife=10.0, feature_sets={"a": []}), 'feature set "a" is empty'),
    # A knob whose switch is off did nothing, without a word (S22).
    (po.spec.ewridge, dict(halflife=10.0, drift_action="reset"), "needs emit_drift"),
    (po.spec.ewridge, dict(halflife=10.0, drift_delta=0.1), "drift_delta needs emit_drift"),
    (
        po.spec.ewridge,
        dict(halflife=10.0, drift_threshold=5.0),
        "drift_threshold needs emit_drift",
    ),
    (po.spec.ewridge, dict(halflife=10.0, average_eta=2.0), "average_eta needs emit_averaged"),
    (
        po.spec.ewridge,
        dict(halflife=10.0, resid_autocorr_lag=2),
        "resid_autocorr_lag needs emit_autocorr",
    ),
    (
        po.spec.ewridge,
        dict(halflife=10.0, long_halflife=100.0),
        "long_halflife needs session_shrink",
    ),
    (
        po.spec.ewridge,
        dict(
            halflife=10.0,
            clock="t",
            max_dclock=10.0,
            session="s",
            session_gap="reset",
            session_shrink=0.5,
            long_halflife=100.0,
        ),
        'session_shrink does not apply with session_gap = "reset"',
    ),
    (
        po.spec.ewridge,
        dict(
            halflife=10.0,
            group="g",
            session="s",
            group_close="session",
            session_shrink=0.5,
            long_halflife=100.0,
        ),
        'session_shrink does not apply with group_close = "session"',
    ),
    (po.spec.ewridge, dict(halflife=10.0, session_gap=5.0), "session_gap needs session"),
    (po.spec.ewridge, dict(halflife=10.0, on_clock_reset="zero"), "on_clock_reset needs clock"),
    # The eight counts whose floor is 1 say so in the builder (D8).
    (
        po.spec.ewridge,
        dict(halflife=10.0, window=5.0, window_every=0),
        "window_every must be >= 1, got 0",
    ),
]

ACCEPTED = [
    (po.spec.ewridge, dict(halflife=10.0, ridge=0.0)),
    (po.spec.ewridge, dict(halflife=10.0, ridge=[1e-6, 1e-3])),
    (po.spec.ewridge, dict(halflife=10.0, clock="t", max_dclock=INF)),
    (po.spec.ewridge, dict(halflife=10.0, clock="t", max_dclock=0.0)),
    (
        po.spec.ewridge,
        dict(halflife=10.0, clock="t", max_dclock=10.0, session="s", session_gap=0.0),
    ),
    (po.spec.ewridge, dict(halflife=10.0, solve_every=0.0)),
    (po.spec.ewridge, dict(halflife=INF)),
    (po.spec.ewridge, dict(halflife=[10.0, 20.0])),
    (po.spec.huber, dict(halflife=10.0, ridge=0.0)),
    (po.spec.lasso, dict(halflife=10.0, lasso_path=[0.1, 0.0])),
    (po.spec.kalman, dict(halflife=10.0, coef_halflife=INF, q=[0.0, 0.0])),
    (po.spec.ewridge, dict(halflife=10.0, feature_sets={"a": ["x0"]})),
    (po.spec.ewridge, dict(halflife=10.0, emit_drift=True, drift_action="reset")),
    (po.spec.ewridge, dict(halflife=10.0, ridge=[1e-6, 1.0], emit_averaged=True, average_eta=2.0)),
    (po.spec.ewridge, dict(halflife=10.0, emit_autocorr=True, resid_autocorr_lag=2)),
    (
        po.spec.ewridge,
        dict(
            halflife=10.0,
            clock="t",
            max_dclock=10.0,
            session="s",
            session_gap=0.0,
            session_shrink=0.5,
            long_halflife=100.0,
        ),
    ),
    (po.spec.ewridge, dict(halflife=10.0, clock="t", max_dclock=10.0, on_clock_reset="zero")),
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
    for col in [c for c in out.columns if c.startswith("n_eff")]:
        assert all(math.isfinite(v) for v in out[col].to_list()), col


def test_a_nan_deep_in_a_list_names_the_parameter():
    with pytest.raises(ValueError, match='spec "m": coef_prior must not be NaN'):
        po.spec.ewridge("m", halflife=10.0, coef_prior=[[0.0, NAN]], **BASE)


def test_a_feature_set_named_twice_is_refused_by_name():
    """A hand-written list can name one set twice. The bank's field-name
    tripwire caught it, under a comment saying the case could not arise
    (review 2026-09-12, S7)."""
    spec = po.spec.ewridge("m", halflife=10.0, **BASE)
    spec["model"]["feature_sets"] = [["a", ["x0"]], ["a", ["x0"]]]
    with pytest.raises(ValueError, match='spec "m": feature_sets names "a" more than once'):
        po.ModelBank([spec])


@pytest.mark.parametrize("name", ["", "spec", "group"])
def test_a_spec_name_the_bank_cannot_carry_is_refused(name):
    """``last_row`` puts ``spec`` and ``group`` columns beside a struct named
    after the spec, so a spec of either name collided there, and an empty
    name names no struct at all (S7)."""
    with pytest.raises(ValueError, match="name"):
        po.spec.ewridge(name, halflife=10.0, **BASE)


def test_holt_takes_its_level_halflife_once():
    """``halflife`` and ``level_halflife`` are one knob under two names.
    Given both, the level and ``n_eff`` followed one and ``sigma`` the other
    (S22)."""
    with pytest.raises(ValueError, match="halflife and level_halflife"):
        po.spec.holt("m", targets=["y"], halflife=10.0, level_halflife=20.0)
    po.spec.holt("m", targets=["y"], halflife=20.0)
    po.spec.holt("m", targets=["y"], level_halflife=20.0)


@pytest.mark.parametrize(
    "builder,kw",
    [
        (po.spec.ew_cov, dict(features=["x0", "y"], halflife=10.0)),
        (po.spec.marginal, dict(**BASE, halflife=10.0)),
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
    """``ew_cov`` has no intercept, yet its default ``min_periods`` was
    ``k + add_intercept``, so the flag moved its first reported row (S22)."""
    first = []
    for flag in (True, False):
        spec = po.spec.ew_cov(
            "m", features=["x0"], stats=["mean"], halflife=10.0, add_intercept=flag
        )
        out = po.ModelBank([spec]).fit_predict(_df(20)).unnest("m")
        mean = next(c for c in out.columns if c.startswith("mean"))
        first.append(out[mean].is_not_null().arg_max())
    assert first[0] == first[1], first


def test_marginal_refuses_a_window_with_lags():
    """The lag ring has no snapshot, so under a window every ``lagcorr`` was
    the whole history's co-moment over the window's variance. ``ew_cov``
    refuses the pair, and now so does ``marginal`` (C18)."""
    with pytest.raises(ValueError, match="window and lags"):
        po.spec.marginal("m", halflife=10.0, window=50.0, lags=[1], **BASE)
    po.spec.marginal("m", halflife=10.0, window=50.0, **BASE)
    po.spec.marginal("m", halflife=10.0, lags=[1], **BASE)


def test_every_door_fills_validates_and_builds_a_spec_alike():
    """The bank filled a spec's defaults, validated it and built its models;
    ``output_fields``, ``output_index`` and ``coef_fields`` only validated.
    So a dict for a model with no target and no ``targets`` -- which the bank
    fills from ``features[0]`` -- was refused there, and a spec the core
    refuses was given field names there (S25)."""
    e53 = {"name": "c", "model": {"type": "ew_cov"}, "features": ["x0", "y"], "halflife": 10.0}
    po.ModelBank([e53])
    assert po.spec.output_fields(e53)
    assert po.spec.output_index(e53).height > 0
    assert po.spec.coef_fields(e53).height == 0
    refused = {
        "name": "r",
        "model": {"type": "ew_ridge", "window": 10.0, "ridge_decay": True},
        "targets": ["y"],
        "features": ["x0"],
        "halflife": 10.0,
    }
    doors = [po.spec.output_fields, po.spec.output_index, po.spec.coef_fields, po.ModelBank]
    for door in doors:
        with pytest.raises(ValueError, match="ridge_decay"):
            door([refused] if door is po.ModelBank else refused)
