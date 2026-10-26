"""The command line writes what the bank writes, for every kind at its fullest
(task 278, b32; PARAMS A8).

Each case is one kind with every option its TOML can carry switched on: the
stream keys (a clock, a half-life or ``lam``, a gap cap, a weight, a session
and its gap, groups, the readiness gates, the coefficient cadence, an
embargo), every diagnostic the kind takes with its memory, and every one of
the kind's model keys. Options that cannot sit together are split across a
kind's cases. The spec goes to the command line as TOML (``spec_toml``) over
a parquet file this module writes, and to ``ModelBank.fit_predict`` over the
same rows; every field of every spec is held to the bit, and so are the saved
state's bytes and, where groups close, the closed-groups sidecar. Both run
the same arithmetic, so any difference is a defect, not a tolerance.

The run-level keys vary from case to case: the chunk size (the bank takes
the whole frame in one call, so this is chunk invariance through the command
line too), the thread count, the output layout and the output format.

``test_every_kind_and_key_is_covered`` keeps the cases honest: every kind
the TOML takes has a case or is waiting, every model key is used by a case
of its kind, and every ``[[specs]]`` key is used by a case of each kind or
refused by that kind.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import _toml_value, spec_toml

TIER = "mixed"

REPO = Path(__file__).resolve().parent.parent
INF = float("inf")
F3 = ["x0", "x1", "x2"]
XY: dict[str, Any] = dict(targets=["y", "z"], features=F3)


def stream(n: int = 1500, seed: int = 7) -> pl.DataFrame:
    """Three interleaved groups on an irregular clock with two gaps past the
    cap, a session change and a level shift halfway, null features and
    targets, zero weights, a 0/1 target, a class label, a per-row hazard, an
    exogenous driver, 0/1 weights, monotone blocks of 500 rows, and the
    clock again as a Datetime."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0.5, 1.0, 1.5, 2.0], size=n))
    t[n // 4 :] += 40.0
    t[3 * n // 4 :] += 40.0
    x = rng.standard_normal((n, 3))
    y = 0.5 + x @ np.array([1.0, -0.5, 0.0]) + 0.3 * rng.standard_normal(n)
    y[n // 2 :] += 2.0
    z = 0.2 * y + x[:, 2] + 0.3 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.choice(n, n // 25, replace=False)] = 0.0
    x0 = x[:, 0].copy()
    x0[rng.choice(n, n // 100, replace=False)] = np.nan
    yy = y.copy()
    yy[rng.choice(n, n // 100, replace=False)] = np.nan
    return pl.DataFrame(
        {
            "t": t,
            "g": np.array(["a", "b", "c"])[np.arange(n) % 3],
            "s": np.where(np.arange(n) < n // 2, "am", "pm"),
            "block": np.arange(n) // 500,
            "x0": x0,
            "x1": x[:, 1],
            "x2": x[:, 2],
            "y": yy,
            "z": z,
            "w": w,
            "hit": (y > 0.5).astype(float),
            "lab": np.where(y > 0.5, "up", "down"),
            "h": rng.uniform(50.0, 200.0, n),
            "e": rng.standard_normal(n),
            "w01": (rng.uniform(size=n) > 0.05).astype(float),
        }
    ).with_columns(
        pl.col("x0", "y").fill_nan(None),
        ts=pl.datetime(2024, 1, 2) + pl.duration(microseconds=(pl.col("t") * 1e6).cast(pl.Int64)),
    )


# ------------------------------------------------------------- the key sets

#: The stream keys a kind with a decay takes together.
STREAM: dict[str, Any] = dict(
    clock="t",
    half_life=25.0,
    gap_cap=6.0,
    weight="w",
    session="s",
    session_gap=30.0,
    group="g",
    min_weight=3.0,
    min_settled_frac=0.05,
    max_rows_between_coefs=7,
    emit_clocks=True,
)
#: The other stream keys, which cannot sit beside the first set: ``lam`` for
#: ``half_life``, a session reset for a session gap, a clock cadence for the
#: row cap, and an embargo, which no group close can sit beside.
STREAM2: dict[str, Any] = dict(
    clock="t",
    lam=0.97,
    gap_cap=6.0,
    weight="w",
    restart_after_step_back=10.0,
    session="s",
    session_gap="reset",
    coef_every=5.0,
    embargo=3.0,
    fit_intercept=False,
    emit_clocks=True,
)
#: One case of each kind runs its groups to a close: monotone blocks of 500
#: rows, each closed group drained to the sidecar. (Blocks of 100 rows were
#: too short for the diagnostics to warm up in.)
CLOSE: dict[str, Any] = dict(group="block", group_close="monotone")
#: The kinds that report no coefficients take no cadence for them.
NOCOEF = {k: v for k, v in STREAM.items() if k != "max_rows_between_coefs"}
NOCOEF2 = {k: v for k, v in STREAM2.items() if k != "coef_every"}
#: The kinds that decay nothing by a half-life take neither it nor a gate on
#: settling toward it.
NODECAY = {k: v for k, v in NOCOEF.items() if k not in ("half_life", "min_settled_frac")}
NODECAY2 = {k: v for k, v in NOCOEF2.items() if k != "lam"}

#: Every stream diagnostic, with its memory.
OUTPUTS: dict[str, Any] = dict(
    emit_sigma=True,
    emit_zscore=True,
    emit_metrics=True,
    conformal=0.9,
    conformal_rate=0.01,
    resid_quantiles=[0.1, 0.9],
    emit_autocorr=True,
    resid_autocorr_lag=2,
    emit_drift=True,
    drift_delta=0.01,
    drift_threshold=20.0,
    emit_calibration=True,
    calibration_half_life=40.0,
    emit_breaks=True,
    breaks_half_life=40.0,
    emit_specification=True,
    specification_half_life=40.0,
    ljung_box_lags=3,
    horizon_rows=2,
    emit_tails=True,
    tails_half_life=40.0,
    emit_feature_health=True,
    feature_health_half_life=40.0,
)
#: What a least-squares fit adds.
LS: dict[str, Any] = dict(
    emit_error_inflation=True,
    max_error_inflation=3.0,
    emit_se_coef=True,
    emit_robust_se=True,
    robust_se_half_life=40.0,
    robust_se_lags=2,
    emit_influence=True,
    influence_half_life=40.0,
)
#: What a grid of slots adds.
GRID: dict[str, Any] = dict(emit_selected=True, emit_averaged=True, average_eta=0.5)
WINDOW: dict[str, Any] = dict(
    window_size=60.0,
    closed="both",
    snapshot_every=5.0,
    max_rows_between_snapshots=10,
    window_budget={"thin": 64.0},
)


def full(**kw: Any) -> dict[str, Any]:
    """A regression's first set: the stream keys and every diagnostic."""
    return {**XY, **STREAM, **OUTPUTS, **kw}


def full2(**kw: Any) -> dict[str, Any]:
    """A regression's second set, with drift resetting the fit: at threshold
    6 it fires a handful of times on this stream and leaves the diagnostics
    time to warm up between (at 2 it fired 35-50 times and they never did;
    measured 2026-10-25)."""
    return {
        **XY,
        **STREAM2,
        **OUTPUTS,
        "drift_action": "reset",
        "drift_threshold": 6.0,
        "drift_delta": 0.005,
        **kw,
    }


def no_features(kw: dict[str, Any]) -> dict[str, Any]:
    """``holt`` reads the target alone, so it has no features to watch."""
    drop = ("features", "emit_feature_health", "feature_health_half_life")
    return {k: v for k, v in kw.items() if k not in drop}


class Case(NamedTuple):
    label: str
    builder: str
    kw: dict[str, Any]
    #: Specs the case reads (a ``seqtest``'s two sides): (name, builder, kw).
    reads: list[tuple[str, str, dict[str, Any]]] = []


FWD = (
    po.ewm_mean("y", half_life=dt.timedelta(seconds=10), window_size="8s", reverse=True)
    - pl.col("y")
).alias("fwd")
SIDE: dict[str, Any] = dict(
    targets=["y"], features=F3, clock="t", gap_cap=6.0, half_life=25.0, max_rows_between_coefs=7
)

CASES: list[Case] = [
    Case(
        "ewridge",
        "ewridge",
        full(
            **LS,
            **GRID,
            ridge=[1e-3, 0.1],
            feature_sets={"a": ["x0"], "b": ["x1", "x2"]},
            standardize=True,
            solve_every=3.0,
            gram_block_rows=8,
            gram_threads=2,
            target_gaps="pairwise",
        ),
    ),
    Case(
        "ewridge_long",
        "ewridge",
        full2(
            **LS,
            ridge=0.01,
            ridge_scale="sum",
            coef_prior=[[0.0, 1.0, -0.5], [0.0, 0.0, 0.0]],
            long_half_life=50.0,
            session_shrink=0.5,
            session_gap=30.0,
            max_rows_between_solves=4,
        ),
    ),
    Case("ewridge_window", "ewridge", full(**CLOSE, **LS, **WINDOW)),
    Case(
        "ewridge_temporal",
        "ewridge",
        dict(
            targets=["y", FWD],
            features=F3,
            clock="ts",
            half_life=dt.timedelta(seconds=25),
            gap_cap="6s",
            embargo=dt.timedelta(seconds=10),
            restart_after_step_back="1m",
            session="s",
            session_gap="30s",
            coef_every="5s",
            group="g",
            min_weight=3.0,
            emit_clocks=True,
            emit_metrics=True,
            emit_calibration=True,
            calibration_half_life="40s",
            solve_every="3s",
        ),
    ),
    Case(
        "quantile",
        "quantile",
        full(
            **CLOSE, quantile=0.8, ridge=0.01, standardize=True, solve_every=3.0, quantile_eps=1e-4
        ),
    ),
    Case("quantile2", "quantile", full2(quantile=0.3, max_rows_between_solves=4)),
    Case(
        "rls",
        "rls",
        full(
            **CLOSE,
            **LS,
            delta=0.5,
            coef_prior=[[0.0, 1.0, -0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ),
    ),
    Case("rls2", "rls", full2(**LS)),
    Case(
        "kalman",
        "kalman",
        full(
            **CLOSE,
            **{k: v for k, v in LS.items() if "robust_se" not in k},
            coef_half_life=50.0,
            obs_var=0.2,
            p0=4.0,
            share_p=True,
            revert_half_life=[INF, 200.0, 200.0, 200.0],
            standardize=False,
        ),
    ),
    Case("kalman2", "kalman", full2(q=[0.0, 1e-4, 1e-4])),
    Case(
        "lasso",
        "lasso",
        full(
            **GRID,
            penalties=[0.2, 0.05, 0.0],
            l1_ratio=0.7,
            select_half_life=50.0,
            solve_every=3.0,
            max_iter=200,
            tol=1e-6,
            target_gaps="pairwise",
        ),
    ),
    Case(
        "lasso2",
        "lasso",
        full2(penalties=[0.1], max_rows_between_solves=4, max_error_inflation=3.0),
    ),
    Case("lasso_window", "lasso", full(**CLOSE, penalties=[0.1, 0.0], **WINDOW)),
    Case(
        "holt",
        "holt",
        {**no_features(full(**CLOSE)), "targets": ["y"], "trend_half_life": 80.0},
    ),
    Case("holt2", "holt", {**no_features(full2()), "targets": ["y"], "trend": False}),
    Case(
        "sgd",
        "sgd",
        full(
            loss="squared",
            learning_rate=0.01,
            schedule="inv_scaling",
            power=0.3,
            l2=1e-3,
            clip_gradient=5.0,
            standardize=False,
            coef_min=-3.0,
            coef_max=3.0,
            coef_sum=2.0,
        ),
    ),
    Case(
        "sgd_quantile",
        "sgd",
        full2(loss="quantile", quantile=0.7, learning_rate=0.05, schedule="adagrad"),
    ),
    Case(
        "sgd_eps",
        "sgd",
        full(**CLOSE, loss="epsilon_insensitive", eps=0.1, learning_rate=0.01),
    ),
    Case(
        "sgd_logistic",
        "sgd",
        {**full(loss="logistic", learning_rate=0.05, strict_binary=True), "targets": ["hit"]},
    ),
    Case("sgd_poisson", "sgd", {**full(loss="poisson", learning_rate=0.01), "targets": ["hit"]}),
    Case(
        "pa",
        "pa",
        full(
            **CLOSE,
            mode="pa2",
            c=0.5,
            eps=0.1,
            coef_min=-3.0,
            coef_max=3.0,
            coef_sum=2.0,
            standardize=False,
        ),
    ),
    Case("pa2", "pa", full2(mode="pa")),
    Case(
        "ftrl",
        "ftrl",
        {
            **full(**CLOSE, alpha=0.2, beta=2.0, l1=0.01, l2=0.1, strict_binary=True),
            "targets": ["hit"],
        },
    ),
    Case("ftrl2", "ftrl", full2(loss="squared")),
    Case(
        "seqtest",
        "seqtest",
        dict(
            targets=["y"],
            clock="t",
            gap_cap=6.0,
            **CLOSE,
            emit_clocks=True,
            a="ra",
            b="rb",
            a_suffix="__r0.001",
        ),
        [
            ("ra", "ewridge", dict(SIDE, **CLOSE, ridge=[0.001, 0.1])),
            ("rb", "kalman", dict(SIDE, **CLOSE, coef_half_life=50.0)),
        ],
    ),
    Case(
        "seqtest2",
        "seqtest",
        dict(
            targets=["y"],
            clock="t",
            gap_cap=6.0,
            session="s",
            session_gap="reset",
            embargo=2.0,
            fit_intercept=False,
            min_weight=2.0,
            restart_after_step_back=10.0,
            a="ra",
            b="rb",
            a_suffix="",
            b_suffix="__r0.1",
        ),
        [
            ("ra", "kalman", dict(SIDE, coef_half_life=50.0)),
            ("rb", "ewridge", dict(SIDE, ridge=[0.001, 0.1])),
        ],
    ),
    Case(
        "marginal",
        "marginal",
        {
            **XY,
            **NOCOEF,
            "lags": [1, 3],
            "cross_lags": [3],
            "serial_rule": "bartlett",
            "bins": 4,
            "bin_rule": "quantile",
            "bin_warm_rows": 50,
            "bin_budget": 64,
            "bin_stat": "huber",
            "bin_huber_delta": 2.0,
            "shards": 2,
            "feature_moments": "shared",
        },
    ),
    Case(
        "marginal2",
        "marginal",
        {**XY, **NOCOEF2, "bin_edges": [[-0.5, 0.5]] * 3, "shards": "auto"},
    ),
    Case(
        "marginal_window",
        "marginal",
        {
            **XY,
            **NOCOEF,
            **CLOSE,
            **WINDOW,
            "lags": [1],
            "window_lags": True,
            "serial_rule": "geometric",
        },
    ),
    Case(
        "ew_cov",
        "ew_cov",
        {
            **NOCOEF,
            "features": F3,
            "stats": ["mean", "var", "std", "cov", "corr", "partial_corr", "mahal"],
            "bias": True,
            "precision_prior": 1e-6,
            "mahal_quantiles": [0.5, 0.99],
            "pca": 2,
            "pca_every": 5.0,
            "gram_threads": 2,
        },
    ),
    Case(
        "ew_cov2",
        "ew_cov",
        {
            **NOCOEF2,
            "features": ["x1", "x2"],
            "stats": ["corr", "lag_corr"],
            "lags": [1, 3],
            "pca": 1,
            "max_rows_between_pcas": 6,
        },
    ),
    Case(
        "ew_cov_window",
        "ew_cov",
        {
            **NOCOEF,
            **CLOSE,
            **WINDOW,
            "features": F3,
            "stats": ["mean", "cov", "mahal"],
            "precision_prior": 1e-6,
        },
    ),
    Case(
        "ew_class",
        "ew_class",
        {
            **STREAM,
            "features": F3,
            "label": "lab",
            "classes": ["down", "up"],
            "covariance_type": "tied",
            "precision_prior": 0.5,
        },
    ),
    Case(
        "ew_class2",
        "ew_class",
        {
            **STREAM2,
            "features": F3,
            "label": "lab",
            "classes": ["down", "up"],
            "covariance_type": "diag",
            "precision_prior": 0.5,
        },
    ),
    Case(
        "ew_class_window",
        "ew_class",
        {
            **STREAM,
            **CLOSE,
            **WINDOW,
            "features": F3,
            "label": "lab",
            "classes": ["down", "up"],
            "precision_prior": 1.0,
        },
    ),
    Case(
        "deco",
        "deco",
        {**STREAM, **CLOSE, "features": F3, "dynamics": "linear", "alpha": 0.05, "beta": 0.9},
    ),
    Case(
        "deco2",
        "deco",
        {**STREAM2, "features": [*F3, "z"], "blocks": {"p": ["x0", "x1"], "q": ["x2", "z"]}},
    ),
    Case(
        "rcov",
        "rcov",
        {
            "features": F3,
            **CLOSE,
            "kind": "kernel",
            "kernel": "parzen",
            "jitter": 3,
            "psd": False,
            "block_rows": 40,
            "max_bandwidth": 20,
            "noise_stride": 2,
            "iv_stride": 10,
            "weight": "w01",
            "min_weight": 2.0,
            "restart_after_step_back": 10.0,
            "fit_intercept": False,
            "session": "s",
            "session_gap": 30.0,
            "emit_clocks": True,
            "clock": "t",
            "gap_cap": 6.0,
        },
    ),
    Case("rcov_bandwidth", "rcov", {"features": F3, **CLOSE, "bandwidth": 5}),
    Case(
        "rcov_preavg",
        "rcov",
        {"features": F3, **CLOSE, "kind": "preavg", "theta": 0.8, "psd": False, "block_rows": 40},
    ),
    Case("rcov_preavg_rows", "rcov", {"features": F3, **CLOSE, "kind": "preavg", "preavg_rows": 5}),
    Case(
        "rcov_plain",
        "rcov",
        {"features": F3, "group": "g", "group_close": "session", "session": "s", "kind": "plain"},
    ),
    Case(
        "audit",
        "audit",
        {
            "columns": ["x0", "x1", "y", "hit"],
            "pairs": True,
            "distinct_cap": 16,
            "group": "g",
            "clock": "t",
            "gap_cap": 6.0,
            "restart_after_step_back": 10.0,
            "session": "s",
            "session_gap": 30.0,
            "emit_clocks": True,
        },
    ),
    Case("audit2", "audit", {"columns": ["x0", "y"], "group": "block", "fit_intercept": False}),
    Case(
        "kmeans",
        "kmeans",
        {
            **STREAM,
            **CLOSE,
            "features": ["x1", "x2"],
            "n_clusters": 3,
            "warm_rows": 30,
            "init": "lloyd",
            "seed": 5,
            "update_every_rows": 3,
            "split_merge": 0.5,
            "split_merge_every_rows": 50,
            "dead_frac": 0.1,
            "standardize": False,
        },
    ),
    Case(
        "kmeans2",
        "kmeans",
        {
            **STREAM2,
            "features": ["x1", "x2"],
            "n_clusters": 2,
            "init": "farthest",
            "scale_floor": 1e-3,
        },
    ),
    Case(
        "micro",
        "micro",
        {
            **STREAM,
            **CLOSE,
            "features": ["x1", "x2"],
            "eps": 0.5,
            "beta_mu": 0.3,
            "max_clusters": 20,
            "prune_every": 5.0,
            "macro_link": 2.0,
            "standardize": False,
        },
    ),
    Case(
        "micro2",
        "micro",
        {
            **STREAM2,
            "features": ["x1", "x2"],
            "eps": 0.5,
            "max_rows_between_prunes": 20,
            "scale_floor": 1e-3,
        },
    ),
    Case(
        "hmm",
        "hmm",
        {
            **STREAM,
            **CLOSE,
            "features": ["x1", "x2"],
            "n_components": 2,
            "precision_prior": 0.1,
            "covariance_type": "diag",
            "transition_prior": 0.5,
            "warm_rows": 40,
            "init": "first",
            "seed": 3,
        },
    ),
    Case(
        "hmm_given",
        "hmm",
        {
            **STREAM2,
            "features": ["x1", "x2"],
            "n_components": 2,
            "precision_prior": 0.1,
            "learn": False,
            "transition": [0.9, 0.1, 0.2, 0.8],
            "means": [-1.0, 0.0, 1.0, 0.0],
            "covs": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        },
    ),
    Case(
        "hmm_tvtp",
        "hmm",
        {
            **STREAM,
            "features": ["x1", "x2"],
            "n_components": 2,
            "precision_prior": 0.1,
            "covariance_type": "tied",
            # Each group weighs about 9 rows at this half-life: the default
            # warm-up never ended, so nothing was scored.
            "warm_rows": 5,
            "exog_tvtp": "e",
            "tvtp_coef": [[2.0, -2.0, -2.0, 2.0], [0.5, -0.5, -0.5, 0.5]],
        },
    ),
    Case(
        "bocpd",
        "bocpd",
        {
            **NODECAY,
            "features": ["x1", "x2"],
            "hazard": 100.0,
            "covariance_type": "full",
            "mean_prior": [0.0, 0.0],
            "mean_precision_prior": 0.5,
            "degrees_of_freedom_prior": 4.0,
            "covariance_prior": [1.0, 0.0, 0.0, 1.0],
            "prune_below": 1e-6,
            "max_run": 300,
        },
    ),
    Case(
        "bocpd_diag",
        "bocpd",
        {
            **NODECAY,
            **CLOSE,
            "features": ["x1", "x2"],
            "covariance_type": "diag",
            "robust_beta": 0.1,
            "covariance_prior": [1.0],
            "warm_rows": 20,
        },
    ),
    Case("bocpd2", "bocpd", {**NODECAY2, "features": ["x1", "x2"], "hazard_col": "h"}),
    Case(
        "corrchange",
        "corrchange",
        {
            **NODECAY,
            "features": F3,
            "kind": "monitor",
            "window_rows": 30,
            "alpha": 0.01,
            "alpha_adjust": "none",
            "bandwidth": 3,
        },
    ),
    Case(
        "corrchange_seq",
        "corrchange",
        {
            **NOCOEF2,
            "features": F3,
            "kind": "sequential",
            "window_rows": 30,
            "scalar": True,
            "crit": 0.5,
            "monitor_rows": 40,
            "boundary_gamma": 0.2,
        },
    ),
    Case(
        "corrchange_window",
        "corrchange",
        {
            **NODECAY,
            **CLOSE,
            "features": F3,
            "kind": "window",
            "window_rows": 30,
            "norm": "linf",
            "change_action": "reset",
            "n_perm": 99,
            "permute_every_rows": 10,
            "perm_block": 3,
            "seed": 7,
        },
    ),
]

#: Task 277 changes ``huber`` and ``sgd(loss="huber")`` (a converged
#: warm-up refit, schema 64); until it merges they are not asserted here.
#: Registering them is the one line ``CASES += AFTER_TASK_277``. Measured on
#: main at 7bfc72b9 (2026-10-25): both already agree with the bank to the bit.
AFTER_TASK_277: list[Case] = [
    Case(
        "huber",
        "huber",
        full(
            **CLOSE,
            **GRID,
            huber_delta=1.5,
            huber_scale="rms",
            huber_scale_half_life=40.0,
            ridge=0.01,
            standardize=True,
            solve_every=3.0,
            feature_sets={"a": ["x0"], "b": ["x1", "x2"]},
            emit_huber_scale=True,
        ),
    ),
    Case("huber2", "huber", full2(max_rows_between_solves=4, ridge_scale="sum")),
    Case(
        "sgd_huber",
        "sgd",
        full(
            loss="huber",
            huber_delta=1.5,
            huber_scale="rms",
            huber_scale_half_life=40.0,
            learning_rate=0.01,
        ),
    ),
]
#: What waits for task 277: a kind by name, or a (kind, model key) pair.
WAITING = {
    "huber",
    ("sgd", "huber_delta"),
    ("sgd", "huber_scale"),
    ("sgd", "huber_scale_half_life"),
}


# ---------------------------------------------------------------- comparing


def bits(s: pl.Series) -> tuple:
    """A series as its exact contents: floats by their bits (so -0.0 and
    0.0 differ, and so do two NaNs of different payloads), nested types by
    their parts, anything else by value."""
    dtype = s.dtype
    if dtype == pl.Float64:
        a = s.to_numpy(allow_copy=True)
        mask = s.is_null().to_numpy()
        return ("f", mask.tobytes(), np.where(mask, 0.0, a).view(np.uint64).tobytes())
    if isinstance(dtype, pl.List):
        return ("l", bits(s.is_null()), bits(s.list.len()), bits(s.explode()))
    if isinstance(dtype, pl.Array):
        return ("a", bits(s.is_null()), bits(s.arr.to_list()))
    if isinstance(dtype, pl.Struct):
        fields = tuple((f, bits(s.struct.field(f))) for f in dtype.to_schema())
        return ("s", bits(s.is_null()), fields)
    return ("v", str(dtype), tuple(s.to_list()))


def first_difference(a: pl.Series, b: pl.Series) -> str:
    for i, (p, q) in enumerate(zip(a.to_list(), b.to_list(), strict=True)):
        if p != q or repr(p) != repr(q):
            return f"row {i}: {p!r} vs {q!r}"
    return "the values compare equal but their bits differ"


def differences(
    want: pl.DataFrame, got: pl.DataFrame, field: Callable[[str, str], pl.Series]
) -> list[str]:
    """Every column of ``want`` that ``got`` does not hold to the bit; a
    struct column field by field, each read from ``got`` by ``field``."""
    bad = []
    for c in want.columns:
        if not isinstance(want.schema[c], pl.Struct):
            if bits(want[c]) != bits(got[c]):
                bad.append(f"{c}: {first_difference(want[c], got[c])}")
            continue
        for f in want.schema[c].to_schema():
            w, g = want[c].struct.field(f), field(c, f)
            if w.dtype != g.dtype:
                bad.append(f"{c}.{f}: {w.dtype} vs {g.dtype}")
            elif bits(w) != bits(g):
                bad.append(f"{c}.{f}: {first_difference(w, g)}")
    return bad


# ------------------------------------------------------------------ the runs


def build(case: Case) -> list[dict[str, Any]]:
    specs = [getattr(po.spec, b)(n, **kw) for n, b, kw in case.reads]
    return [*specs, getattr(po.spec, case.builder)(case.label, **case.kw)]


#: The run-level keys, one row per case in turn: chunk size, threads,
#: output layout and output format.
RUNS = [(97, 1, "struct", "parquet"), (13, 3, "flat", "ipc"), (500, 2, "struct", "ipc")]


def check_the_cli_writes_what_the_bank_does(case: Case, index: int, exe: Path, tmp: Path) -> None:
    df = stream()
    specs = build(case)
    closes = any(s.get("group_close") for s in specs)
    bank = po.ModelBank(specs, threads=1)
    # The warm-up and readiness warnings the fullest specs raise say the
    # same thing to both runs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = bank.fit_predict(df)
    closed_want = bank.closed_groups() if closes else None
    chunk, threads, layout, fmt = RUNS[index % len(RUNS)]
    src, dst, state = tmp / "in.parquet", tmp / f"out.{fmt}", tmp / "bank.state"
    df.write_parquet(src)
    top = {
        "input": src.as_posix(),
        "output": dst.as_posix(),
        "save_state": state.as_posix(),
        "chunk_size": chunk,
        "threads": threads,
        "output_layout": layout,
    }
    if closes:
        top["closed_groups"] = (tmp / "closed.parquet").as_posix()
    lines = [f"{k} = {_toml_value(v)}" for k, v in top.items()]
    cfg = tmp / "bank.toml"
    cfg.write_text("\n".join(lines + [spec_toml(s) for s in specs]) + "\n", encoding="utf-8")
    res = subprocess.run(
        [str(exe), "--config", str(cfg), "--quiet"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    got = pl.read_parquet(dst) if fmt == "parquet" else pl.read_ipc(dst)
    if layout == "flat":
        flat = [
            f"{c}.{f}"
            for c in want.columns
            if isinstance(want.schema[c], pl.Struct)
            for f in want.schema[c].to_schema()
        ]
        plain = [c for c in want.columns if not isinstance(want.schema[c], pl.Struct)]
        assert got.columns == plain + flat
        bad = differences(want, got, lambda c, f: got[f"{c}.{f}"])
    else:
        assert got.columns == want.columns
        bad = differences(want, got, lambda c, f: got[c].struct.field(f))
    assert not bad, (
        f"{case.label} ({chunk} rows a chunk, {threads} threads, {layout}, {fmt}):\n"
        + ("\n".join(bad))
    )
    saved = state.read_bytes()
    assert saved == bank.save_bytes(), (
        f"{case.label}: the saved state is {len(saved)} bytes, the bank's {len(bank.save_bytes())}"
    )
    out = want[case.label].struct.unnest()
    # Evidence the paths ran: something in the struct moves, a reset fired
    # where drift resets, and groups closed where they close.
    moving = [
        c for c in out.columns if out[c].dtype == pl.Float64 and out[c].drop_nulls().n_unique() > 1
    ]
    assert moving, f"{case.label}: no field moves, so the comparison held nothing"
    if case.kw.get("drift_action") == "reset":
        drift = [c for c in out.columns if c.startswith("drift_")]
        assert any(out[c].any() for c in drift), f"{case.label}: drift never fired"
    if closes:
        assert closed_want is not None and closed_want.height > 0, f"{case.label}: no group closed"
        closed_got = pl.read_parquet(tmp / "closed.parquet")
        bad = differences(closed_want, closed_got, lambda c, f: closed_got[c].struct.field(f))
        assert closed_got.columns == closed_want.columns and not bad, "\n".join(bad)


#: The essential case: the workhorse at its fullest.
ESSENTIAL = {"ewridge"}


@pytest.mark.parametrize(
    "index",
    [
        i
        if c.label in ESSENTIAL
        else pytest.param(
            i, marks=pytest.mark.extended(reason="every kind through the command line")
        )
        for i, c in enumerate(CASES)
    ],
    ids=[c.label for c in CASES],
)
def test_the_cli_writes_what_the_bank_does(index, online_cli, tmp_path):
    check_the_cli_writes_what_the_bank_does(CASES[index], index, online_cli, tmp_path)


# ----------------------------------------------------------------- coverage


def toml_keys() -> tuple[list[str], dict[str, list[str]]]:
    """The ``[[specs]]`` keys and each kind's ``[specs.model]`` keys, from the
    API surface's ``[toml keys]`` section."""
    text = (REPO / "tests" / "api_surface.txt").read_text(encoding="utf-8")
    section = text[text.index("[toml keys]") : text.index("[cli flags]")]
    common = section[section.index("  [[specs]]:") : section.index("  [specs.model], by type:")]
    model = section[section.index("  [specs.model], by type:") :]
    keys = re.findall(r"^    (\w+)$", common, re.M)
    kinds = {m[1]: m[2].split(", ") for m in re.finditer(r"^    (\w+): (.*)$", model, re.M)}
    return keys, kinds


#: The companions a key needs before a kind can be asked whether it takes
#: the key at all: a memory needs its diagnostic, and so on.
NEEDS: dict[str, list[str]] = {
    "average_eta": ["emit_averaged"],
    "breaks_half_life": ["emit_breaks"],
    "calibration_half_life": ["emit_calibration"],
    "conformal_rate": ["conformal"],
    "drift_action": ["emit_drift"],
    "drift_delta": ["emit_drift"],
    "drift_threshold": ["emit_drift"],
    "feature_health_half_life": ["emit_feature_health"],
    "horizon_rows": ["emit_calibration"],
    "influence_half_life": ["emit_influence"],
    "ljung_box_lags": ["emit_specification"],
    "resid_autocorr_lag": ["emit_autocorr"],
    "robust_se_half_life": ["emit_robust_se"],
    "robust_se_lags": ["emit_robust_se"],
    "session_gap": ["session"],
    "specification_half_life": ["emit_specification"],
    "tails_half_life": ["emit_tails"],
}
SAMPLE: dict[str, Any] = {
    **STREAM,
    **OUTPUTS,
    **LS,
    **GRID,
    "drift_action": "reset",
    "embargo": 3.0,
    "lam": 0.97,
    "restart_after_step_back": 10.0,
    "fit_intercept": False,
    "coef_every": 5.0,
    "targets": ["y"],
    "features": ["x1", "x2"],
    **CLOSE,
}


def test_every_kind_and_key_is_covered():
    keys, kinds = toml_keys()
    constructors = {
        n for n in dir(po.spec) if not n.startswith("_") and callable(getattr(po.spec, n))
    }
    helpers = {"coef_fields", "coef_index", "output_fields", "output_index"}
    # Every model the spec builders make is a kind the TOML takes.
    assert set(kinds) <= constructors, set(kinds) - constructors
    assert {c for c in constructors - helpers if c in kinds} == set(kinds)
    used: dict[str, set[str]] = {}
    for c in CASES + AFTER_TASK_277:
        used.setdefault(c.builder, set()).update(c.kw)
    missing = []
    for kind, model_keys in sorted(kinds.items()):
        if kind in WAITING:
            continue
        if kind not in used:
            missing.append(f"{kind}: no case")
            continue
        missing += [
            f"{kind}.{k}" for k in model_keys if k not in used[kind] and (kind, k) not in WAITING
        ]
        first = next(c for c in CASES if c.builder == kind)
        for k in keys:
            if k in ("name", "model") or k in used[kind]:
                continue
            # A key no case of the kind uses: the kind must refuse it.
            kw = {**first.kw, k: SAMPLE.get(k, True)}
            for need in NEEDS.get(k, []):
                kw[need] = SAMPLE.get(need, True)
            if k == "lam":
                kw.pop("half_life", None)
            try:
                getattr(po.spec, kind)("m", **kw)
            except (ValueError, TypeError):
                continue
            missing.append(f"{kind}: takes `{k}`, which no case uses")
    assert not missing, "\n".join(missing)
