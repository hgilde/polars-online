"""Seeded streams for ``ModelBank.check`` (docs/PLAN.md task 223 (a)): clean
ones, on which no check may fire, and one per check with its problem planted.

``tests/test_check.py`` runs them, and the measurement behind each threshold in
the ``check`` docstring runs the same generators over ten seeds. Each function
returns ``(df, specs)``; a clean stream's specs are every regression at its
defaults and with ``standardize=False`` where it has one, a ``marginal`` and an
``ew_cov``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

import polars_online as po

N = 2000
FEATURES = ["x0", "x1", "x2"]


def _frame(x: np.ndarray, y: np.ndarray, **extra: Any) -> pl.DataFrame:
    cols: dict[str, Any] = {"t": np.arange(len(y), dtype=float)}
    cols.update({f"x{j}": x[:, j] for j in range(x.shape[1])})
    cols["y"] = y
    cols.update(extra)
    return pl.DataFrame(cols)


def every_model(features: list[str], half_life: float = 200.0, **kw: Any) -> list[dict[str, Any]]:
    """Every kind ``check`` reads with features and a target, at its defaults,
    plus the three that centre only under ``standardize``, without it."""
    c = dict(targets=["y"], features=features, half_life=half_life, **kw)
    return [
        po.spec.ewridge("ewridge", **c),
        po.spec.ewridge("ewridge_std", standardize=True, **c),
        po.spec.lasso("lasso", lasso_path=[1e-3], **c),
        po.spec.huber("huber", **c),
        po.spec.quantile("quantile", quantile=0.5, **c),
        po.spec.kalman("kalman", coef_half_life=1e6, **c),
        po.spec.kalman("kalman_raw", coef_half_life=1e6, standardize=False, **c),
        po.spec.rls("rls", **c),
        po.spec.sgd("sgd", **c),
        po.spec.sgd("sgd_raw", standardize=False, **c),
        po.spec.pa("pa", **c),
        po.spec.pa("pa_raw", standardize=False, **c),
        po.spec.ftrl("ftrl", **c),
        po.spec.marginal("marginal", **c),
        po.spec.ew_cov(
            "ew_cov", features=features, half_life=half_life, **{k: v for k, v in kw.items()}
        ),
    ]


def _linear(rng: np.random.Generator, k: int, rho: float, r2: float, n: int = N):
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    x = rng.multivariate_normal(np.zeros(k), cov, size=n)
    b = np.r_[1.0, 0.3 * np.ones(k - 1)]
    signal = x @ b
    y = signal + np.sqrt(np.var(signal) * (1 - r2) / r2) * rng.normal(size=n)
    return x, y


# -- clean shapes: no check may fire ------------------------------------------


def clean_independent(seed: int):
    """Three independent unit features, R^2 0.9, one stream."""
    x, y = _linear(np.random.default_rng(seed), 3, 0.0, 0.9)
    return _frame(x, y), every_model(FEATURES)


def clean_correlated(seed: int):
    """Ten features equicorrelated at 0.8, R^2 0.99: correlated, not collinear."""
    x, y = _linear(np.random.default_rng(seed), 10, 0.8, 0.99)
    feats = [f"x{j}" for j in range(10)]
    return _frame(x, y), every_model(feats)


def clean_grouped(seed: int):
    """Five groups of 400 rows, R^2 0.5, half-life 50."""
    rng = np.random.default_rng(seed)
    x, y = _linear(rng, 3, 0.3, 0.5)
    g = np.repeat([f"g{j}" for j in range(5)], N // 5)
    return _frame(x, y, g=g), every_model(FEATURES, half_life=50.0, group="g")


def clean_small_groups(seed: int):
    """Forty groups of 50 rows, half-life 5: the chance guards' case."""
    rng = np.random.default_rng(seed)
    x, y = _linear(rng, 3, 0.0, 0.7)
    g = np.repeat([f"g{j:02d}" for j in range(40)], N // 40)
    return _frame(x, y, g=g), every_model(FEATURES, half_life=5.0, group="g")


def clean_sparse_target(seed: int):
    """A target present on 40% of the rows, as sparse labels are."""
    rng = np.random.default_rng(seed)
    x, y = _linear(rng, 3, 0.0, 0.9)
    y = np.where(rng.random(N) < 0.4, y, np.nan)
    return _frame(x, y), every_model(FEATURES)


CLEAN: dict[str, Callable[[int], tuple[pl.DataFrame, list[dict[str, Any]]]]] = {
    "independent": clean_independent,
    "correlated": clean_correlated,
    "grouped": clean_grouped,
    "small_groups": clean_small_groups,
    "sparse_target": clean_sparse_target,
}


# -- planted: each check's problem, and the specs it fires for ----------------


def _base(seed: int, n: int = N):
    return _linear(np.random.default_rng(seed), 3, 0.0, 0.9, n)


def planted_missing(seed: int):
    rng = np.random.default_rng(seed + 1000)
    x, y = _base(seed)
    x[rng.random(N) < 0.2, 1] = np.nan
    return _frame(x, y), every_model(FEATURES)


def planted_few_learned(seed: int):
    rng = np.random.default_rng(seed + 1000)
    x, y = _base(seed)
    x[rng.random(N) < 0.6, 1] = np.nan
    return _frame(x, y), every_model(FEATURES)


def planted_nothing_learned(seed: int):
    x, _ = _base(seed)
    return _frame(x, np.full(N, np.nan)), every_model(FEATURES)


def planted_constant_feature(seed: int):
    x, y = _base(seed)
    x[:, 2] = 3.0
    return _frame(x, y), every_model(FEATURES)


def planted_constant_target(seed: int):
    x, _ = _base(seed)
    return _frame(x, np.full(N, 2.5)), every_model(FEATURES)


def planted_level(seed: int):
    """x0 at 100 standard deviations from zero: past every uncentred model's
    limit, and below the centred models' info line."""
    x, y = _base(seed)
    x[:, 0] += 100.0
    return _frame(x, y), every_model(FEATURES)


def planted_level_far(seed: int):
    """x0 at 1e7 standard deviations: the centred models' info line."""
    x, y = _base(seed)
    x[:, 0] += 1e7
    return _frame(x, y), every_model(FEATURES)


def planted_scales(seed: int):
    """x0 spread 10 times as wide as the others, its coefficient scaled to
    match."""
    x, y = _base(seed)
    x[:, 0] *= 10.0
    return _frame(x, y), every_model(FEATURES)


def planted_ridge(seed: int):
    """x0 spread 1e-3 wide: the default ridge of 1e-6 is its variance."""
    x, y = _base(seed)
    x[:, 0] *= 1e-3
    return _frame(x, y), every_model(FEATURES)


def planted_collinear(seed: int):
    """x2 = x0 + 0.01 N(0, 1): a near-duplicate (VIF about 1e4)."""
    rng = np.random.default_rng(seed + 1000)
    x, y = _base(seed)
    x[:, 2] = x[:, 0] + 0.01 * rng.normal(size=N)
    return _frame(x, y), every_model(FEATURES)


def planted_duplicate(seed: int):
    """x2 = x0 exactly."""
    x, y = _base(seed)
    x[:, 2] = x[:, 0]
    return _frame(x, y), every_model(FEATURES)


def planted_leakage(seed: int):
    """A feature that is the target plus 1e-3 of its spread."""
    rng = np.random.default_rng(seed + 1000)
    x, y = _base(seed)
    leak = y + 1e-3 * np.std(y) * rng.normal(size=N)
    return _frame(x, y, leak=leak), every_model([*FEATURES, "leak"])


def planted_step_back(seed: int):
    """Five blocks of the clock fed out of order, restarted at each step back."""
    x, y = _base(seed)
    df = _frame(x, y)
    order = np.random.default_rng(seed + 1000).permutation(5)
    df = pl.concat([df.slice(int(b) * (N // 5), N // 5) for b in order])
    return df, every_model(FEATURES, clock="t", gap_cap=10.0, restart_after_step_back=0.0)


def planted_few_rows(seed: int):
    """A group of two rows beside one of the rest."""
    x, y = _base(seed)
    g = np.where(np.arange(N) < 2, "tiny", "big")
    return _frame(x, y, g=g), every_model(FEATURES, group="g")


def planted_group_sizes(seed: int):
    """A group of 15 rows beside one of 1,985: past 100 times apart, and past
    the coefficient count."""
    x, y = _base(seed)
    g = np.where(np.arange(N) < 15, "small", "big")
    return _frame(x, y, g=g), every_model(FEATURES, group="g")


def planted_short(seed: int):
    """Fifty rows at a half-life of 200: less than one half-life of clock."""
    x, y = _base(seed, 50)
    return _frame(x, y), every_model(FEATURES)


def planted_min_weight(seed: int):
    """A min_weight of 1e4 at a half-life of 200, where the weight settles
    near 289."""
    x, y = _base(seed)
    return _frame(x, y), every_model(FEATURES, min_weight=1e4)


def planted_withheld(seed: int):
    """min_settled_frac 0.99 over a stream of three half-lives."""
    x, y = _base(seed, 600)
    return _frame(x, y), every_model(FEATURES, min_settled_frac=0.99)


def planted_resets(seed: int):
    """A session column whose change restarts the stream."""
    x, y = _base(seed)
    s = np.arange(N) // 500
    return _frame(x, y, s=s), every_model(
        FEATURES, clock="t", gap_cap=10.0, session="s", session_gap="reset"
    )


#: Each planted stream, the codes it must raise, and the specs it must raise
#: them for (``None``: every spec but those named in the third entry, which
#: must stay silent on that code).
PLANTED: dict[str, tuple[Callable[[int], Any], str, frozenset[str] | None, frozenset[str]]] = {
    "missing": (planted_missing, "missing", None, frozenset()),
    "few_learned": (planted_few_learned, "few_learned", None, frozenset()),
    "nothing_learned": (
        planted_nothing_learned,
        "nothing_learned",
        frozenset(
            [
                "ewridge",
                "ewridge_std",
                "lasso",
                "huber",
                "quantile",
                "kalman",
                "kalman_raw",
                "rls",
                "sgd",
                "sgd_raw",
                "pa",
                "pa_raw",
                "ftrl",
                "marginal",
            ]
        ),
        frozenset(["ew_cov"]),
    ),
    "constant_feature": (planted_constant_feature, "constant", None, frozenset()),
    "constant_target": (
        planted_constant_target,
        "constant",
        frozenset(
            [
                "ewridge",
                "ewridge_std",
                "lasso",
                "huber",
                "quantile",
                "kalman",
                "kalman_raw",
                "rls",
                "sgd",
                "sgd_raw",
                "pa",
                "pa_raw",
                "ftrl",
                "marginal",
            ]
        ),
        frozenset(["ew_cov"]),
    ),
    "level": (
        planted_level,
        "level_over_spread",
        frozenset(["kalman_raw", "rls", "sgd_raw", "pa_raw", "ftrl"]),
        frozenset(
            [
                "ewridge",
                "ewridge_std",
                "lasso",
                "huber",
                "quantile",
                "kalman",
                "sgd",
                "pa",
                "marginal",
                "ew_cov",
            ]
        ),
    ),
    "level_far": (planted_level_far, "level_over_spread", None, frozenset()),
    "scales": (
        planted_scales,
        "scales_apart",
        frozenset(["kalman_raw", "sgd_raw", "pa_raw", "ftrl"]),
        frozenset(
            [
                "ewridge",
                "ewridge_std",
                "lasso",
                "huber",
                "quantile",
                "kalman",
                "sgd",
                "pa",
                "rls",
                "marginal",
                "ew_cov",
            ]
        ),
    ),
    "ridge": (
        planted_ridge,
        "ridge_shrinks",
        frozenset(["ewridge", "huber", "quantile"]),
        frozenset(["ewridge_std", "lasso"]),
    ),
    "collinear": (
        planted_collinear,
        "collinear",
        frozenset(["ewridge", "ewridge_std", "lasso", "ew_cov"]),
        frozenset(),
    ),
    "duplicate": (
        planted_duplicate,
        "collinear",
        frozenset(["ewridge", "ewridge_std", "lasso", "ew_cov"]),
        frozenset(),
    ),
    "leakage": (
        planted_leakage,
        "leakage",
        frozenset(["ewridge", "ewridge_std", "lasso", "marginal"]),
        frozenset(),
    ),
    "step_back": (planted_step_back, "step_back", None, frozenset()),
    "few_rows": (
        planted_few_rows,
        "few_rows",
        frozenset(
            [
                "ewridge",
                "ewridge_std",
                "lasso",
                "huber",
                "quantile",
                "kalman",
                "kalman_raw",
                "rls",
                "sgd",
                "sgd_raw",
                "pa",
                "pa_raw",
                "ftrl",
            ]
        ),
        frozenset(["marginal", "ew_cov"]),
    ),
    "group_sizes": (planted_group_sizes, "group_sizes", None, frozenset()),
    "short": (planted_short, "never_settled", None, frozenset()),
    "min_weight": (planted_min_weight, "below_min_weight", None, frozenset()),
    # A marginal withholds nothing: it has no prediction to withhold.
    "withheld": (planted_withheld, "withheld", None, frozenset(["marginal"])),
    "resets": (planted_resets, "resets", None, frozenset()),
}


def findings(df: pl.DataFrame, specs: list[dict[str, Any]], chunks: int = 1) -> pl.DataFrame:
    """``check()`` after feeding ``df`` in ``chunks`` slices."""
    import warnings

    bank = po.ModelBank(specs)
    size = max(1, -(-df.height // chunks))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for part in df.iter_slices(size):
            bank.fit_predict(part)
    return bank.check()
