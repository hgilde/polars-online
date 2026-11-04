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

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge, enet_path

import ledger
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


def ridge_solve(
    seed: int, half_life: float, fit_intercept: bool, ridge: float, std: bool
) -> tuple[dict[str, Any], np.ndarray]:
    """One seed's Gram and ``solve``'s coefficients on it."""
    x, y = ridge_rows(seed)
    g = gram_of(x, y, half_life, fit_intercept)
    return g, np.asarray(po.gram.solve(g, ridge=ridge, standardize=std))


#: The ledger's inputs (tests/ledger.py): ``solve`` on ten seeds at every
#: case, where the grid reads fifty. The oracle agrees to round-off, so the
#: hash keeps the doubles themselves.
ledger.guard(
    "ridge",
    lambda: [ridge_solve(s, *case)[1] for s in range(10) for case in CASES],
    oracles=["scikit-learn", "numpy"],
    bits=52,
)


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
    g, ours = ridge_solve(seed, half_life, fit_intercept, ridge, std)
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
        rows = [(check_solve(seed, *case), seed, case) for seed in range(50) for case in CASES]
        worst = max(rows)
        print(f"worst error over its bound {worst[0]:.3f} (seed {worst[1]}, case {worst[2]})")
        # Round-off, which another platform's arithmetic moves: the ledger
        # holds it at twice the measured 0.19, under the bound's 1.
        ledger.record("ridge", "error_over_bound", [r[0] for r in rows], floor=0.4)
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


def path_weights(seed: int, k: int = 10) -> np.ndarray:
    return np.random.default_rng(1000 + seed).uniform(0.25, 4.0, k)


def weighted_paths(seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """One seed's Gram, ``lasso_path`` on it with the seed's
    ``penalty_weights``, and without them."""
    x, y = lasso_rows(seed)
    g = gram_of(x, y, float("inf"), True)
    kw = dict(tol=1e-14, max_iter=100_000)
    ours = po.gram.lasso_path(g, PENALTIES, penalty_weights=list(path_weights(seed)), **kw)
    return g, np.asarray(ours), np.asarray(po.gram.lasso_path(g, PENALTIES, **kw))


ledger.guard(
    "lasso_weights",
    lambda: [weighted_paths(s)[1:] for s in range(50)],
    oracles=["scikit-learn"],
    bits=52,
)


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
    w = path_weights(seed, k)
    g, ours, plain = weighted_paths(seed)
    sd = x.std(axis=0)
    xs = (x - x.mean(axis=0)) / sd
    _, coefs, _ = enet_path(
        xs / w, y - y.mean(), l1_ratio=1.0, alphas=PENALTIES, tol=1e-14, max_iter=100_000
    )
    want = (coefs / w[:, None]).T
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
        # Round-off: about four times the largest of the measured 1.8e-11
        # (macOS arm64) and 2.5e-11 (two Linux x86-64 runs).
        ledger.record("lasso_weights", "error", [e for e, _ in errs], floor=9.9e-11)
        assert all(moved > 1.0 for _, moved in errs)
        assert errs[worst][0] <= 1e-10


# ------------------------------------------------- the condition estimate
#
# The solve's condition estimate, `‖H⁻¹‖₁` of the equilibrated matrix `H =
# D⁻¹ A D⁻¹` by Hager's method as Higham refined it, is the one LAPACK's
# `dpocon` runs (`dlacn2`). It is not exposed to Python, so the Rust test
# `crates/online-core/src/solve/condition_tests.rs::the_estimate_is_lapacks_dpocon`
# holds it to a table of `dpocon`'s numbers, and this test holds the table
# to scipy: each matrix is rebuilt from the same formula (exact entries,
# the same bits in both languages), and scipy's `dpocon` on its factor must
# read the number the table says, to the rounding of a solve with `H`.

CONDITION_TESTS = (
    Path(__file__).resolve().parents[1] / "crates/online-core/src/solve/condition_tests.rs"
)


def dpocon_case(name: str) -> np.ndarray:
    """The Rust test's `dpocon_case`, in numpy."""
    if name in ("hilbert5", "hilbert7"):
        k, lcm = (5, 2520.0) if name == "hilbert5" else (7, 360360.0)
        return np.array([[lcm / (i + j + 1) for j in range(k)] for i in range(k)])
    if name == "pascal6":
        return np.array([[float(math.comb(i + j, i)) for j in range(6)] for i in range(6)])
    if name == "gram8":
        b = np.array(
            [
                [float((r * r * 3 + j * (r + 2) * 5 + j * j) % 17 - 8) for j in range(8)]
                for r in range(24)
            ]
        )
        return b.T @ b
    if name == "collinear5":
        b = np.array([[float((r * (2 * j + 5) + j) % 13 - 6) for j in range(5)] for r in range(30)])
        b[:, 4] = 64.0 * b[:, 0] + np.array([float(r % 3 - 1) for r in range(30)])
        return b.T @ b
    if name == "alternating3":
        c = 1.0 - 2.0**-20
        return np.array([[1.0, c, 0.0], [c, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if name == "kahan6":
        r = np.zeros((6, 6))
        for i in range(6):
            r[i, i:] = -0.75 * 0.5**i
            r[i, i] = 0.5**i
        return r.T @ r
    raise ValueError(name)


def test_the_rust_tables_dpocon_numbers_are_scipys():
    from scipy.linalg import cholesky, lapack

    text = CONDITION_TESTS.read_text(encoding="utf-8")
    table = re.search(r"const DPOCON: \[\(&str, f64\); (\d+)\] = \[(.*?)\];", text, re.S)
    assert table is not None, "the table moved"
    rows = re.findall(r'\("(\w+)", ([-+.e\d]+)\)', table.group(2))
    assert len(rows) == int(table.group(1)) == 7, rows
    for name, number in rows:
        a = dpocon_case(name)
        k = len(a)
        d = np.sqrt(np.diag(a))
        h = a / np.outer(d, d)
        h_norm = np.abs(h).sum(axis=0).max()
        rcond, info = lapack.dpocon(cholesky(h, lower=False), h_norm)
        assert info == 0, name
        est = 1.0 / (rcond * h_norm)
        # The rounding of LAPACK's solves with `H`, which another build of
        # LAPACK may take in another order: `k κ₁ u` of it, `κ₁ = ‖H‖₁ ×`
        # the estimate. The table is this build's to the bit (scipy 1.18.1,
        # 2026-11-04).
        tol = k * h_norm * est * U
        assert abs(float(number) / est - 1.0) <= tol, (name, float(number), est)
        # The estimate is at most the norm it estimates, numpy's exact
        # `cond(H, 1) / ‖H‖₁`.
        assert est <= np.linalg.cond(h, 1) / h_norm * (1.0 + tol), name


# ---------------------------------------------------- `po.gram.condition`
#
# Belsley's collinearity diagnostics, which `po.gram.condition` computes
# from the accumulated moments: the eigendecomposition of the unit-length
# column-scaled second-moment matrix. numpy computes the same quantities
# from the rows another way, the SVD of the weighted design scaled to unit
# column length (Belsley, Kuh and Welsch 1980, §3.2): its singular values,
# `np.linalg.cond` of it for `kappa`, and the proportions, by Belsley's
# definition, `φ_kj = v_jk² / μ_k²` normalised over the components `k` for
# each column `j`. The moments carry the design's square, so a singular
# value `μ_k` comes back to about `(μ_max / μ_k)² u` of itself from them
# (an eigenvalue of the moments to `u` of the largest), where the SVD of the
# rows reads it to `u`. The tolerance is that, eight times: measured
# 2026-11-04 at 0.06 to 0.43 of `κ² u` on the designs below, `κ` to 1.1e5.


def condition_rows(
    seed: int, eps: float, n: int = 600, k: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Columns of unlike scales (1e-2 to 1e2) and levels (to 50), the third
    three times the first plus ``eps`` of its spread."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k)) * 10.0 ** rng.uniform(-2.0, 2.0, k) + rng.uniform(-50.0, 50.0, k)
    x[:, 2] = 3.0 * x[:, 0] + eps * rng.normal(size=n) * x[:, 0].std()
    return x, x @ np.ones(k) + rng.normal(size=n)


@pytest.mark.parametrize(
    ("seed", "eps", "half_life", "fit_intercept"),
    [
        (1, 1.0, float("inf"), True),
        (2, 1e-3, float("inf"), True),
        (3, 1e-3, 150.0, True),
        (4, 1e-2, 150.0, False),
    ],
)
def test_condition_is_numpys_svd_of_the_scaled_rows(seed, eps, half_life, fit_intercept):
    x, y = condition_rows(seed, eps)
    g = gram_of(x, y, half_life, fit_intercept)
    got = po.gram.condition(g)
    n = len(y)
    w = 0.5 ** (np.arange(n)[::-1] / half_life)
    design = np.column_stack([np.ones(n), x]) if fit_intercept else x
    assert got["columns"] == g["columns"] and len(got["columns"]) == design.shape[1]
    scaled = design * np.sqrt(w)[:, None]
    scaled = scaled / np.linalg.norm(scaled, axis=0)
    _, mu, vt = np.linalg.svd(scaled, full_matrices=False)
    kappa = np.linalg.cond(scaled)
    tol = 8.0 * kappa**2 * U
    assert kappa > 2000.0, "the case: a near-dependency"
    np.testing.assert_allclose(got["singular_values"], mu, rtol=tol, atol=0.0)
    np.testing.assert_allclose(got["condition_indexes"], mu[0] / mu, rtol=tol, atol=0.0)
    assert got["kappa"] == pytest.approx(kappa, rel=tol)
    phi = vt.T**2 / mu**2
    proportions = (phi / phi.sum(axis=1, keepdims=True)).T
    np.testing.assert_allclose(got["proportions"], proportions, rtol=0.0, atol=tol)
    # The dependency is where it was built: the last component carries most
    # of the variance of the first and third features.
    j0 = got["columns"].index("x0")
    assert proportions[-1, j0] > 0.5 and proportions[-1, j0 + 2] > 0.5
