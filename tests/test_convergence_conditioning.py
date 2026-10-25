"""Convergence class 7: conditioning (docs/PLAN.md task 278).

A solve is only as accurate as its system's condition allows, and the
library must reach that accuracy, and say so where no double-precision solve
can. The class feeds the solves near-collinear, rank-deficient and badly
scaled designs, and NIST's Statistical Reference Datasets, whose certified
coefficients are an oracle independent of every library.

**What each solve can reach.** ``ewridge`` (and ``lasso`` and the Gram
solves) keep the centred moments and factorize the normal equations, so
their condition is the correlation matrix's, ``kappa_c``, the square of the
centred, column-scaled design's. A Cholesky is invariant to a diagonal
scaling (van der Sluis), so the features' units do not enter, and the
forward error of the standardized slopes is at most ``2 kappa_c eta / (1 -
kappa_c eta)`` (Higham, Theorem 7.2; ``crate::oracle::forward_bound`` in
the Rust tests), where ``eta`` is the backward error of forming and
factorizing the system. ``rls`` rotates each row into a triangular factor
of the uncentred design (Givens), a QR method, so its condition is the
design's own, ``kappa_x``, with Wedin's residual term: ``(kappa_x +
kappa_x^2 tan(theta)) eta``, ``theta`` the angle between the target and
its fit. Where ``kappa_c eta`` reaches 1 the normal equations carry no
digit, and no solve of them can: NIST's ``Filip`` is that case.

**The bounds' ``eta``.** A weighted Welford update rounds about four times
a row (the deviation, the weighted step, the mean's move, the co-moment's),
each a relative ``u`` of the running magnitude, and NIST's polynomial
models' powers ``x**j`` are computed in ``j`` roundings; the factorization
adds Higham's ``gamma_{3k+1}`` (Theorem 10.4). So ``eta =
gamma_{4n + 3k + 1 + degree}``, worst-case, growing with the rows.

The Rust side holds the same solves to task 274's componentwise bounds
(``crates/online-core/src/ewridge/tests.rs``,
``a_conditioned_design_is_solved_within_its_closed_form_bound``), where the
mutation run sees them. The gradient models and every standardized model are
free of the features' units to the bit (class 6,
``tests/test_convergence_equivariance.py``), so a badly scaled design is no
harder for them than a well-scaled one.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
import polars_online.gram as gram
from data import STRD_LINEAR, nist_strd_linear_or_skip

TIER = "mixed"

U = 2.0**-53


def gamma(m: int) -> float:
    """Higham's ``gamma_m = m u / (1 - m u)``."""
    return m * U / (1.0 - m * U)


def forward_bound(kappa: float, eta: float) -> float:
    """``2 kappa eta / (1 - kappa eta)``, Higham's Theorem 7.2, or ``inf``
    where ``kappa eta`` reaches 1 and the bound says nothing."""
    ke = kappa * eta
    return 2.0 * ke / (1.0 - ke) if ke < 1.0 else float("inf")


def fit(spec: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[pl.DataFrame, po.ModelBank]:
    """One chunk through a bank of `spec`, its struct unnested, and the bank."""
    cols: dict[str, Any] = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    cols["y"] = y
    bank = po.ModelBank([spec])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", po.ReadinessWarning)
        out = bank.fit_predict(pl.DataFrame(cols))["m"].struct.unnest()
    return out, bank


def served(bank: po.ModelBank, x: np.ndarray) -> np.ndarray:
    """The bank's predictions for the rows `x`, scored against its state
    as it stands (`predict`, which learns nothing)."""
    cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    return bank.predict(pl.DataFrame(cols))["m"].struct.field("pred_y").to_numpy()


def solve_failures(bank: po.ModelBank) -> int:
    return int(bank.solve_failures()["solve_failures"].sum())


def last_coef(out: pl.DataFrame) -> np.ndarray:
    return np.array(out["coef"].to_list()[-1], float)


def design(x: np.ndarray, intercept: bool) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The column scales the slopes are standardized by, the centred (or,
    without an intercept, raw) correlation matrix's condition ``kappa_c``,
    the column-scaled uncentred design's condition ``kappa_x``, and the
    design ``z`` itself."""
    n = len(x)
    xc = x - x.mean(0) if intercept else x
    s = np.sqrt((xc**2).mean(0))
    corr = (xc / s).T @ (xc / s) / n
    z = np.column_stack([np.ones(n), x]) if intercept else x
    zs = z / np.sqrt((z**2).sum(0))
    return s, z, float(np.linalg.cond(corr)), float(np.linalg.cond(zs))


def standardized_error(coef: np.ndarray, want: np.ndarray, s: np.ndarray, intercept: bool) -> float:
    """``|(b - B) * s| / |B * s|`` over the slopes: the normwise relative
    error in the coordinates the conditioning is stated in."""
    b, w = (coef[1:], want[1:]) if intercept else (coef, want)
    return float(np.linalg.norm((b - w) * s) / np.linalg.norm(w * s))


def lre(coef: np.ndarray, want: np.ndarray) -> float:
    """The log relative error, NIST's measure: the fewest correct digits of
    any coefficient, capped at 15."""
    rel = np.where(
        want != 0, np.abs(coef - want) / np.abs(np.where(want != 0, want, 1)), np.abs(coef)
    )
    worst = float(np.max(rel))
    return 15.0 if worst == 0 else min(15.0, -np.log10(worst))


# ---------------------------------------------------------------------------
# NIST StRD


def _strd_specs(k: int, intercept: bool) -> dict[str, dict[str, Any]]:
    common: dict[str, Any] = dict(
        targets=["y"],
        features=[f"x{i}" for i in range(k)],
        half_life=float("inf"),
        coef_every=0,
        fit_intercept=intercept,
    )
    return {
        "ewridge": po.spec.ewridge("m", ridge=0.0, **common),
        "ewridge_standardized": po.spec.ewridge("m", ridge=0.0, standardize=True, **common),
        # A prior of 1e-30 next to sums of order 1 to 1e20: no digit of it
        # reaches a coefficient.
        "rls": po.spec.rls("m", delta=1e-30, **common),
    }


#: The datasets whose normal equations keep digits, ``kappa_c eta < 1``:
#: every one but ``Filip``, whose correlation matrix's condition is 1.5e16.
STRD_SOLVABLE = tuple(n for n in STRD_LINEAR if n != "Filip")


@pytest.mark.extended(reason="the network: NIST's datasets, downloaded once and cached")
class TestNistLinearRegression:
    """Each solve against NIST's certified coefficients, within the bound
    its method derives (the module docstring). What each reached, measured
    2026-10-25 as NIST's log relative error (the fewest correct digits of
    any coefficient, the intercept's included):

    ========  =======  =======  ====  ===========================
    dataset   kappa_c  ewridge  rls   ewridge's standardized error
                                      and its bound
    ========  =======  =======  ====  ===========================
    Norris    1.0      12.3     11.8  4.7e-15 of 3.3e-14
    Pontius   6.9e1    11.3     12.2  1.2e-14 of 2.6e-12
    NoInt1    1.0      14.7     14.9  1.9e-15 of 1.1e-14
    NoInt2    1.0      15.0     15.0  4.6e-16 of 3.8e-15
    Longley   1.2e4    11.5     11.1  4.8e-13 of 2.3e-10
    Wampler1  2.7e6     6.3      9.6  9.6e-11 of 6.2e-08
    Wampler2  2.7e6     9.6     12.8  9.5e-11 of 6.2e-08
    Wampler3  2.7e6     6.2      9.5  1.2e-10 of 6.2e-08
    Wampler4  2.7e6     6.3      7.5  9.6e-11 of 6.2e-08
    Wampler5  2.7e6     6.4      5.5  8.5e-11 of 6.2e-08
    Filip     1.5e16    0.0      7.3  no bound: ``kappa_c u`` is 1.7
    ========  =======  =======  ====  ===========================

    Wampler5's residual is 18 times its fit, so ``rls``'s Wedin term,
    ``kappa_x^2 tan(theta)``, costs it digits the normal equations keep.
    The normal equations lose ``log10(kappa_c)`` digits and the QR method
    about half as many, which is the theory: ``rls`` is the library's solve
    for a design whose correlation matrix is past ``1 / u``.
    """

    @pytest.mark.parametrize("name", STRD_SOLVABLE)
    def test_the_normal_equations_reach_their_bound(self, name):
        d = nist_strd_linear_or_skip(name)
        x, y, want, icpt = d["x"], d["y"], d["coef"], d["intercept"]
        n, k = x.shape
        s, _, kappa_c, _ = design(x, icpt)
        eta = gamma(4 * n + 3 * k + 1 + d["degree"])
        bound = forward_bound(kappa_c, eta)
        assert bound < 1.0, f"{name}: the bound says nothing at kappa {kappa_c:.1e}"
        for model in ("ewridge", "ewridge_standardized"):
            out, _ = fit(_strd_specs(k, icpt)[model], x, y)
            coef = last_coef(out)
            err = standardized_error(coef, want, s, icpt)
            print(f"{name} {model}: LRE {lre(coef, want):.1f}, error {err:.1e}, bound {bound:.1e}")
            assert err <= bound, (name, model, err, bound)

    @pytest.mark.parametrize("name", STRD_SOLVABLE)
    def test_the_gram_solve_reaches_the_bound_standardized(self, name):
        # `gram.solve` diagonalizes the Gram (numpy's `eigh`); standardized,
        # the correlation matrix's eigenvalues are what it inverts, and the
        # normal equations' bound holds. Unstandardized it does not
        # (`TestKnownDefects`).
        d = nist_strd_linear_or_skip(name)
        x, y, want, icpt = d["x"], d["y"], d["coef"], d["intercept"]
        n, k = x.shape
        s, _, kappa_c, _ = design(x, icpt)
        _, bank = fit(_strd_specs(k, icpt)["ewridge"], x, y)
        coef = np.asarray(gram.solve(bank.gram("m")[0], ridge=0.0, standardize=True)).ravel()
        err = standardized_error(coef, want, s, icpt)
        # The eigendecomposition's backward error is a few `k u` more than
        # the Cholesky's (Higham, Section 8 and LAPACK's `dsyevd`), well
        # inside the rows' `4 n` here.
        bound = forward_bound(kappa_c, gamma(4 * n + 3 * k + 1 + d["degree"] + 4 * k))
        assert err <= bound, (name, err, bound)

    @pytest.mark.parametrize("name", STRD_SOLVABLE)
    def test_the_rotations_reach_their_bound(self, name):
        d = nist_strd_linear_or_skip(name)
        x, y, want, icpt = d["x"], d["y"], d["coef"], d["intercept"]
        n, k = x.shape
        s, z, _, kappa_x = design(x, icpt)
        out, bank = fit(_strd_specs(k, icpt)["rls"], x, y)
        coef = last_coef(out)
        err = standardized_error(coef, want, s, icpt)
        fitted = z @ want
        tan = float(np.linalg.norm(y - fitted) / np.linalg.norm(fitted))
        # Wedin's bound for a backward-stable QR, with a rotation's backward
        # error `gamma_{6k}` a row (Higham, Theorem 19.10) beside the inputs'.
        eta = gamma(6 * (k + 1) * n + d["degree"])
        bound = 4.0 * (kappa_x + kappa_x**2 * tan) * eta
        print(f"{name} rls: LRE {lre(coef, want):.1f}, error {err:.1e}, bound {bound:.1e}")
        assert err <= bound, (name, err, bound)
        assert solve_failures(bank) == 0

    def test_filip_is_solved_by_rotations_where_the_normal_equations_carry_nothing(self):
        # Wedin's worst case is vacuous for Filip (`kappa_x` 5.2e9 squared,
        # times `tan(theta)` 3.7e-3), but the rotations reach 5.0e-8 in
        # the standardized slopes, NIST's 7.3 digits (measured 2026-10-25);
        # held at 1e-6, twenty times that.
        d = nist_strd_linear_or_skip("Filip")
        x, y, want = d["x"], d["y"], d["coef"]
        s, _, kappa_c, _ = design(x, True)
        assert kappa_c * U > 1.0, "Filip's normal equations were expected past 1 / u"
        out, _ = fit(_strd_specs(x.shape[1], True)["rls"], x, y)
        err = standardized_error(last_coef(out), want, s, True)
        print(f"Filip rls: LRE {lre(last_coef(out), want):.1f}, error {err:.1e}")
        assert err <= 1e-6


# ---------------------------------------------------------------------------
# Synthetic designs


def _rows(n: int, seed: int, shape: str, eps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=(3, n))
    if shape == "collinear":
        x = np.column_stack([2.0 + a, 2.0 + a + eps * b, -1.0 + c])
        beta = np.array([1.0, -2.0, 0.5])
    elif shape == "scales":
        x = np.column_stack([1e-9 + 1e-10 * a, 10.0 + b, 1e11 + 1e10 * c])
        beta = np.array([3e9, -0.5, 2e-10])
    elif shape == "duplicate":
        x = np.column_stack([a, a, c])
        beta = np.array([1.0, 1.0, -1.0])
    elif shape == "constant":
        x = np.column_stack([a, np.full(n, 3.0), c])
        beta = np.array([2.0, 0.0, -1.0])
    else:
        raise AssertionError(shape)
    return x, 0.7 + x @ beta + 0.1 * rng.normal(size=n)


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """numpy's least squares (an SVD) on the column-scaled design, scaled
    back: backward stable on the design itself, so accurate to ``kappa_x u``
    where the normal equations reach ``kappa_c u = kappa_x^2 u``."""
    z = np.column_stack([np.ones(len(y)), x])
    scale = np.sqrt((z**2).mean(0))
    return np.linalg.lstsq(z / scale, y, rcond=None)[0] / scale


_SYNTH: dict[str, Any] = dict(
    targets=["y"], features=["x0", "x1", "x2"], half_life=float("inf"), coef_every=0
)


class TestSyntheticDesigns:
    """The normal equations' bound on designs built to stress it, against
    numpy's SVD least squares: near-collinear to a correlation matrix's
    condition of 1e12, and features from 1e-10 to 1e10, where the raw Gram
    spans 40 orders of magnitude and the correlation matrix is near the
    identity."""

    @pytest.mark.parametrize("standardize", [False, True])
    @pytest.mark.parametrize(
        # To a condition of 3e8: past it the worst-case `eta`, which grows
        # with the rows, makes the bound vacuous at 800 rows, and the Rust
        # test carries the sweep to 4e12 under a componentwise bound.
        "shape, eps",
        [("collinear", 1e-2), ("collinear", 1e-4), ("scales", 0)],
    )
    def test_the_slopes_reach_the_normal_equations_bound(self, shape, eps, standardize):
        n = 800
        x, y = _rows(n, 7, shape, eps)
        s, z, kappa_c, kappa_x = design(x, True)
        out, bank = fit(po.spec.ewridge("m", ridge=0.0, standardize=standardize, **_SYNTH), x, y)
        want = _lstsq(x, y)
        err = standardized_error(last_coef(out), want, s, True)
        # The reference's own error, `kappa_x * gamma` of an SVD, beside ours.
        bound = forward_bound(kappa_c, gamma(4 * n + 10)) + 4 * kappa_x * gamma(10 * n)
        print(f"{shape} {eps} std {standardize}: kappa {kappa_c:.1e}, {err:.1e} of {bound:.1e}")
        assert bound < 1e-3 and err <= bound
        # The final fit's predictions on the design's own rows, scored by
        # `predict` (the serving path), are determined far better than the
        # slopes: within the bound in units of the fit's spread.
        pred = served(bank, x[-200:])
        ref = z[-200:] @ want
        assert float(np.max(np.abs(pred - ref))) / float(np.std(ref)) <= bound

    @pytest.mark.parametrize("ridge", [0.0, 1e-6])
    @pytest.mark.parametrize("standardize", [False, True])
    def test_a_duplicated_column_predicts_as_least_squares_and_says_it_is_shared(
        self, standardize, ridge
    ):
        # A rank-deficient design: the two copies' slopes are not determined,
        # their sum is. A ridge splits it evenly and `support_coef` reads 0.5
        # each (the data determine half of each); at ridge 0 the solve is
        # singular, the solver's jitter splits it, and every jittered solve
        # is counted in `solve_failures`. Either way the predictions are the
        # least-squares predictions, which the column space determines.
        n = 800
        x, y = _rows(n, 8, "duplicate")
        out, bank = fit(po.spec.ewridge("m", ridge=ridge, standardize=standardize, **_SYNTH), x, y)
        coef, support = last_coef(out), out["support_coef"].to_list()[-1]
        z = np.column_stack([np.ones(n), x])
        ref = z @ np.linalg.lstsq(z, y, rcond=None)[0]
        # The final fit, served, against least squares' fitted values: the
        # same column space, so the same predictions, to the normal
        # equations' rounding at the ridge's bias (1e-6 of a unit variance).
        pred = served(bank, x)
        assert float(np.max(np.abs(pred - ref))) < 1e-5
        assert coef[1] + coef[2] == pytest.approx(2.0, abs=0.01)
        assert support[1] == pytest.approx(0.5, abs=1e-3)
        assert support[2] == pytest.approx(0.5, abs=1e-3)
        if ridge == 0.0:
            assert solve_failures(bank) > n // 2, "the singular solves were not counted"
        else:
            assert solve_failures(bank) == 0

    def test_a_constant_column_is_dropped_and_says_so(self):
        n = 800
        x, y = _rows(n, 9, "constant")
        for standardize in (False, True):
            out, _ = fit(po.spec.ewridge("m", standardize=standardize, **_SYNTH), x, y)
            coef, support = last_coef(out), out["support_coef"].to_list()[-1]
            assert coef[2] == 0.0
            assert support[2] == 0.0
            np.testing.assert_allclose(coef[[1, 3]], [2.0, -1.0], atol=0.02)

    def test_rls_reaches_its_bound_on_a_collinear_design(self):
        # The rotations at their own bound, near-collinear at a correlation
        # condition of 1e8 (`lasso`'s descent on the same design is class
        # 8's, `tests/test_convergence_iterations.py`).
        n = 800
        x, y = _rows(n, 10, "collinear", 1e-4)
        s, z, kappa_c, kappa_x = design(x, True)
        want = _lstsq(x, y)
        out, bank = fit(po.spec.rls("m", delta=1e-30, **_SYNTH), x, y)
        err = standardized_error(last_coef(out), want, s, True)
        tan = float(np.linalg.norm(y - z @ want) / np.linalg.norm(z @ want))
        bound = 4.0 * (kappa_x + kappa_x**2 * tan) * gamma(6 * 4 * n) + 4 * kappa_x * gamma(10 * n)
        print(f"rls collinear: kappa_x {kappa_x:.1e}, error {err:.1e}, bound {bound:.1e}")
        assert err <= bound


class TestKnownDefects:
    """What this class found (task 278 (b), 2026-10-25), pinned as it
    stands so that the fix flips the test rather than passing unnoticed."""

    def test_a_numerically_singular_solve_is_reported_nowhere(self):
        # Past `kappa_c u = 1` the normal equations carry no digit. The
        # Cholesky does not always fail there -- rounding can leave the
        # matrix numerically positive definite -- and where it succeeds the
        # coefficients are off by far more than their own size with nothing
        # said: the last solve took no jitter (`solve_failures` does not
        # move on it), `support_coef` reads 1 for every slope, the
        # error-inflation gate passes and no warning is raised. Measured
        # 2026-10-25 at `eps = 1e-11` (a correlation condition near 1e22 by
        # construction; numpy's estimate saturates near `1 / u`): silent on
        # 19 of 40 seeds, 16 of 40 at 1e-12 and 19 of 40 at 1e-13; seed 3
        # has slopes -364.1 and 363.1 for a truth of 1 and -2. The rest
        # jittered their last solve, which is counted, and read support
        # 0.5. NIST's Filip is the same (`test_filip_is_reported_nowhere`).
        n = 800
        x, y = _rows(n, 3, "collinear", 1e-11)
        _, _, kappa_c, _ = design(x, True)
        assert kappa_c * U > 0.5, "the design was meant to be numerically singular"
        spec = po.spec.ewridge("m", ridge=0.0, **_SYNTH)
        cols = {f"x{i}": x[:, i] for i in range(3)}
        frame = pl.DataFrame({**cols, "y": y})
        bank = po.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            head = bank.fit_predict(frame[:-1])["m"].struct.unnest()
            before = solve_failures(bank)
            tail = bank.fit_predict(frame[-1:])["m"].struct.unnest()
        out = pl.concat([head, tail])
        coef = last_coef(out)
        print("slopes", coef[1:], "support", out["support_coef"].to_list()[-1])
        assert np.max(np.abs(coef[1:3] - [1.0, -2.0])) > 100.0
        assert solve_failures(bank) == before
        assert np.all(np.array(out["support_coef"].to_list()[-1][1:]) > 0.99)
        assert out["withheld_reason"][-1] is None
        assert not [w for w in caught if issubclass(w.category, po.ReadinessWarning)]

    @pytest.mark.extended(reason="the network: NIST's Filip, downloaded once and cached")
    def test_filip_is_reported_nowhere(self):
        # Filip by the normal equations: every coefficient off by its own
        # size (NIST's 0.0 digits), and the last solve took no jitter,
        # `support_coef` reads 1, no gate withholds the prediction and no
        # warning is raised. The fitted values' residual spread is 0.00389
        # against the certified 0.00335 (measured 2026-10-25). `rls` reaches
        # 7.3 digits (`test_filip_is_solved_by_rotations...`).
        d = nist_strd_linear_or_skip("Filip")
        x, y, want = d["x"], d["y"], d["coef"]
        k = x.shape[1]
        spec = _strd_specs(k, True)["ewridge"]
        cols = {f"x{i}": x[:, i] for i in range(k)}
        frame = pl.DataFrame({**cols, "y": y})
        bank = po.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            head = bank.fit_predict(frame[:-1])["m"].struct.unnest()
            before = solve_failures(bank)
            tail = bank.fit_predict(frame[-1:])["m"].struct.unnest()
        out = pl.concat([head, tail])
        coef = last_coef(out)
        assert lre(coef, want) < 1.0
        assert solve_failures(bank) == before
        assert np.all(np.array(out["support_coef"].to_list()[-1][1:]) > 0.99)
        assert out["withheld_reason"][-1] is None
        assert not [w for w in caught if issubclass(w.category, po.ReadinessWarning)]

    def test_the_gram_solve_unstandardized_is_not_free_of_the_features_units(self):
        # `gram.solve` diagonalizes the raw Gram with numpy's `eigh`, whose
        # eigenvalues are accurate to `u` times the largest: with features
        # from 1e-10 to 1e10 the small feature's eigenvalue (1e-20) is lost
        # beside 1e20, and the solve is garbage where the model's own
        # Cholesky on the same Gram, and `gram.solve(standardize=True)`,
        # are accurate to 1e-14. Measured 2026-10-25: relative errors 8.9,
        # 2.4e-2, 1.0 and 1.1e-3 against least squares. On NIST's Longley
        # (features from 1e2 to 4e5) it keeps 6.4 digits where the model
        # keeps 11.5, 3.8e-7 against the normal equations' bound of 3.4e-10.
        n = 800
        x, y = _rows(n, 12, "scales")
        out, bank = fit(po.spec.ewridge("m", ridge=0.0, **_SYNTH), x, y)
        want = _lstsq(x, y)
        g = bank.gram("m")[0]
        raw = np.asarray(gram.solve(g, ridge=0.0)).ravel()
        std = np.asarray(gram.solve(g, ridge=0.0, standardize=True)).ravel()
        model = last_coef(out)
        np.testing.assert_allclose(model, want, rtol=1e-12)
        np.testing.assert_allclose(std, want, rtol=1e-12)
        assert float(np.max(np.abs(raw - want) / np.abs(want))) > 1e-3
