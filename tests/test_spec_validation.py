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
    with pytest.raises(ValueError, match='spec "m": coef0 must not be NaN'):
        po.spec.ewridge("m", halflife=10.0, coef0=[[0.0, NAN]], **BASE)
