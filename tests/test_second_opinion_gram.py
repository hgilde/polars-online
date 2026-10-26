"""``po.gram``'s fits against scikit-learn where it computes the same
quantity (task 278, batch 12): ``solve`` with a ridge above 0, plain and
standardized, against ``Ridge`` on the same weighted rows, and
``lasso_path``'s ``penalty_weights`` against ``enet_path`` on columns
rescaled by the weights. Until now ``solve`` met a third-party oracle only
at ridge 0, where ``standardize`` changes nothing, and ``lasso_path``'s
weights only this library's own LARS path.

scikit-learn is the oracle, taken as given; what is tested is the Gram's
algebra: the co-moments' weighting, the intercept's elimination, the
correlation form under ``standardize`` and the weights' meaning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge, enet_path

import polars_online as po

TIER = "mixed"

U = 2.0**-53


def gamma(m: int) -> float:
    """Higham's ``gamma_m = m u / (1 - m u)``."""
    return m * U / (1.0 - m * U)


def ridge_rows(seed: int, n: int = 400, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Columns of unlike scales (1e-2 to 1e2) and levels (to 100), sharing a
    common factor, so the standardized ridge and the plain one differ."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-2.0, 2.0, k)
    level = rng.uniform(-100.0, 100.0, k)
    x = (rng.standard_normal((n, k)) + 0.5 * rng.standard_normal((n, 1))) * scale + level
    y = x @ (rng.standard_normal(k) / scale) + 3.0 + rng.standard_normal(n)
    return x, y


def gram_of(x: np.ndarray, y: np.ndarray, half_life: float, fit_intercept: bool) -> dict[str, Any]:
    features = [f"x{i}" for i in range(x.shape[1])]
    frame = pl.DataFrame({f: x[:, i] for i, f in enumerate(features)}).with_columns(y=y)
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=features,
        half_life=half_life,
        fit_intercept=fit_intercept,
        min_weight=5.0,
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(frame)
    return bank.gram("m")[0]


CASES = [
    (half_life, fit_intercept, ridge, standardize)
    for half_life in (float("inf"), 150.0)
    for fit_intercept in (True, False)
    for ridge in (1e-3, 0.3, 5.0)
    for standardize in (False, True)
]


def check_solve(seed: int, half_life: float, fit_intercept: bool, ridge: float, std: bool) -> float:
    """``solve(g, ridge, standardize)`` against ``Ridge`` on the rows, and
    the error over its bound: at most 1 passes.

    The Gram weighs row ``i`` of ``n`` by ``v_i = 2^-((n-1-i)/h)`` (one
    clock step a row) and keeps weighted means, so its system is ``(C +
    ridge I) b = c`` with ``C`` the weighted (centred) co-moments: that is
    ``Ridge(alpha = ridge * sum(v))`` with ``sample_weight=v``, which
    minimizes ``sum v (y - X b)^2 + alpha |b|^2``. Under ``standardize``
    the columns are scaled by the roots of ``C``'s diagonal first and the
    slopes unscaled after.

    The bound is Higham's Theorem 7.2, ``2 kappa eta / (1 - kappa eta)``,
    in the coordinates both solves are accurate in -- each column scaled
    by the root of its diagonal in ``C + ridge I`` -- with ``kappa`` that
    system's condition and ``eta = gamma(4 n)`` the rounding of moments
    accumulated over ``n`` rows. The intercept ``ybar - m . b`` carries the
    slopes' error times the means, plus its own rounding."""
    x, y = ridge_rows(seed)
    n, k = x.shape
    g = gram_of(x, y, half_life, fit_intercept)
    ours = np.asarray(po.gram.solve(g, ridge=ridge, standardize=std))
    v = np.ones(n) if np.isinf(half_life) else 0.5 ** ((n - 1 - np.arange(n)) / half_life)
    mu = np.average(x, axis=0, weights=v) if fit_intercept else np.zeros(k)
    xc = x - mu
    c = (xc * v[:, None]).T @ xc / v.sum()
    sd = np.sqrt(np.diag(c)) if std else np.ones(k)
    fit = Ridge(alpha=ridge * v.sum(), fit_intercept=fit_intercept, solver="cholesky")
    fit.fit(x / sd, y, sample_weight=v)
    want = fit.coef_ / sd
    got = ours[1:] if fit_intercept else ours
    h = c / np.outer(sd, sd) + ridge * np.eye(k)
    e = np.sqrt(np.diag(h))
    kappa = float(np.linalg.cond(h / np.outer(e, e)))
    ke = kappa * gamma(4 * n)
    bound = 2.0 * ke / (1.0 - ke)
    # In the equilibrated coordinates: slope i times e_i * sd_i.
    scale = e * sd
    err = float(np.max(np.abs(got - want) * scale) / np.max(np.abs(want) * scale))
    ratio = err / bound
    if fit_intercept:
        ybar = float(np.average(y, weights=v))
        explained = np.abs(mu) @ np.abs(got - want)
        rounding = 4 * gamma(4 * n) * (abs(ybar) + np.abs(mu) @ np.abs(want))
        ratio = max(ratio, abs(ours[0] - fit.intercept_) / (explained + rounding))
    # The ridge took effect: the fit is not least squares.
    assert np.abs(want - np.asarray(po.gram.solve(g, standardize=std))[-k:]).max() > 0
    return ratio


class TestSolveIsSklearnsRidge:
    """Measured 2026-10-25 over seeds 0-49 and every case: the error at most
    315 ``u`` of the slopes' size (the worst under ``standardize``, where
    the Gram's moments and numpy's from the rows differ by their
    accumulation), and at most 0.19 of the bound ``check_solve`` derives."""

    @pytest.mark.parametrize(("half_life", "fit_intercept", "ridge", "standardize"), CASES)
    def test_one_seed(self, half_life, fit_intercept, ridge, standardize):
        ratio = check_solve(0, half_life, fit_intercept, ridge, standardize)
        assert ratio <= 1.0, ratio

    @pytest.mark.oracle_grid(reason="50 seeds x 24 cases, each a bank and a Ridge fit")
    def test_fifty_seeds(self):
        worst = max((check_solve(seed, *case), seed, case) for seed in range(50) for case in CASES)
        print(f"worst error over its bound {worst[0]:.3f} (seed {worst[1]}, case {worst[2]})")
        assert worst[0] <= 1.0, worst


def lasso_rows(seed: int, n: int = 200, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Correlated columns at a level of 5. The mixing is mild (0.3) so that
    scikit-learn's descent meets its own tolerance on every seed: at a
    mixing of 1 its ``enet_path`` stopped short on some (its
    ``ConvergenceWarning``) and the comparison measured it, not us."""
    rng = np.random.default_rng(seed)
    mix = np.eye(k) + 0.3 * rng.standard_normal((k, k))
    x = rng.standard_normal((n, k)) @ mix + 5.0
    y = x @ rng.standard_normal(k) + 1.0 + 2.0 * rng.standard_normal(n)
    return x, y


PENALTIES = [0.8, 0.4, 0.2, 0.1, 0.05, 0.01]


def weighted_path_error(seed: int) -> tuple[float, float]:
    """``lasso_path`` with ``penalty_weights`` against ``enet_path`` on the
    standardized columns divided by the weights, and how far the weights
    moved the path.

    With ``u_i = w_i b_i`` the penalty ``l sum w_i |b_i|`` is ``l |u|_1``
    and ``x_i b_i`` is ``(x_i / w_i) u_i``, so the weighted lasso is the
    plain one on the rescaled columns, and ``b = u / w``. That holds for
    the L1 part alone, so the comparison is at ``l1_ratio = 1``. The
    columns are standardized by their population spread and the target
    centred, the correlation form the ``lam=1`` Gram reaches by its own
    arithmetic."""
    x, y = lasso_rows(seed)
    k = x.shape[1]
    w = np.random.default_rng(1000 + seed).uniform(0.25, 4.0, k)
    g = gram_of(x, y, float("inf"), True)
    sd = x.std(axis=0)
    xs = (x - x.mean(axis=0)) / sd
    _, coefs, _ = enet_path(
        xs / w, y - y.mean(), l1_ratio=1.0, alphas=PENALTIES, tol=1e-14, max_iter=100_000
    )
    kw = dict(tol=1e-14, max_iter=100_000)
    ours = po.gram.lasso_path(g, PENALTIES, penalty_weights=list(w), **kw)
    want = (coefs / w[:, None]).T
    plain = po.gram.lasso_path(g, PENALTIES, **kw)
    return float(np.abs(ours[:, 1:] * sd - want).max()), float(np.abs(ours - plain).max())


class TestLassoPathsWeightsAreEnetPathOnRescaledColumns:
    """Measured 2026-10-25 over seeds 0-49: within 1.8e-11 of scikit-learn
    in the standardized coefficients, which are of order 1 to 10; held at
    1e-10. The weights move the path by 1.6 to 25 there."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_one_seed(self, seed):
        err, moved = weighted_path_error(seed)
        print(f"seed {seed}: {err:.1e} from enet_path, the weights moved it {moved:.2f}")
        assert moved > 1.0
        assert err <= 1e-10

    @pytest.mark.oracle_grid(reason="50 seeds of a weighted path against enet_path")
    def test_fifty_seeds(self):
        errs = [weighted_path_error(seed) for seed in range(50)]
        worst = max(range(50), key=lambda s: errs[s][0])
        print(f"worst {errs[worst][0]:.1e} at seed {worst}")
        assert all(moved > 1.0 for _, moved in errs)
        assert errs[worst][0] <= 1e-10
