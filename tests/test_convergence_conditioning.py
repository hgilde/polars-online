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

import ledger
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


# The oracle ledger (tests/ledger.py): each solve's bound is the theory's,
# up to six hundred times the error it reaches, so a change could lose two
# digits without failing. The ledger holds each error as measured; they are
# rounding, so each has a floor, twice the larger of the errors measured on
# macOS arm64 and on CI's Linux x86-64 (2026-10-26), and the group's hash
# keeps every bit.

#: Each case's floor.
NIST_FLOORS: dict[str, float] = {
    "filip_rls": 1.1e-07,
    "longley_ewridge": 9.7e-13,
    "longley_ewridge_standardized": 9.1e-13,
    "longley_gram_standardized": 1.3e-12,
    "longley_gram_unstandardized": 1.5e-12,
    "longley_rls": 1.3e-12,
    "noint1_ewridge": 3.9e-15,
    "noint1_ewridge_standardized": 4.3e-15,
    "noint1_gram_standardized": 4.3e-15,
    "noint1_gram_unstandardized": 3.9e-15,
    "noint1_rls": 2.6e-15,
    "noint2_ewridge": 9.2e-16,
    "noint2_ewridge_standardized": 3.1e-16,
    "noint2_gram_standardized": 6.2e-16,
    "noint2_gram_unstandardized": 6.2e-16,
    "noint2_rls": 6.2e-16,
    "norris_ewridge": 9.4e-15,
    "norris_ewridge_standardized": 8.9e-15,
    "norris_gram_standardized": 9.4e-15,
    "norris_gram_unstandardized": 8.9e-15,
    "norris_rls": 9.4e-15,
    "pontius_ewridge": 2.5e-14,
    "pontius_ewridge_standardized": 4.2e-14,
    "pontius_gram_standardized": 3.4e-14,
    "pontius_gram_unstandardized": 3.2e-14,
    "pontius_rls": 1.9e-15,
    "wampler1_ewridge": 2e-10,
    "wampler1_ewridge_standardized": 3.2e-10,
    "wampler1_gram_standardized": 2.4e-10,
    "wampler1_gram_unstandardized": 2.4e-10,
    "wampler1_rls": 1.2e-14,
    "wampler2_ewridge": 1.9e-10,
    "wampler2_ewridge_standardized": 3.1e-10,
    "wampler2_gram_standardized": 2.1e-10,
    "wampler2_gram_unstandardized": 5.9e-10,
    "wampler2_rls": 1.3e-13,
    "wampler3_ewridge": 2.4e-10,
    "wampler3_ewridge_standardized": 3.6e-10,
    "wampler3_gram_standardized": 4.3e-10,
    "wampler3_gram_unstandardized": 3.9e-10,
    "wampler3_rls": 1.8e-13,
    "wampler4_ewridge": 2e-10,
    "wampler4_ewridge_standardized": 3.2e-10,
    "wampler4_gram_standardized": 2.4e-10,
    "wampler4_gram_unstandardized": 4.3e-10,
    "wampler4_rls": 1.5e-11,
    "wampler5_ewridge": 1.7e-10,
    "wampler5_ewridge_standardized": 2.7e-11,
    "wampler5_gram_standardized": 6e-11,
    "wampler5_gram_unstandardized": 2e-10,
    "wampler5_rls": 1.5e-09,
}


def _record_nist(case: str, err: float) -> None:
    ledger.record("nist_ewridge_rls_gram", case, err, floor=NIST_FLOORS.get(case, 0.0))


def nist_outputs(name: str) -> dict:
    """Every field of each spec on one NIST set, and the Gram's solves."""
    d = nist_strd_linear_or_skip(name)
    x, y, icpt = d["x"], d["y"], d["intercept"]
    out = {}
    for model, spec in _strd_specs(x.shape[1], icpt).items():
        out[model], bank = fit(spec, x, y)
        if model == "ewridge":
            g = bank.gram("m")[0]
            out["gram"] = [gram.solve(g, ridge=0.0, standardize=z) for z in (True, False)]
    return out


ledger.guard(
    "nist_ewridge_rls_gram",
    lambda: [nist_outputs(name) for name in STRD_LINEAR],
    dataset="nist:" + ",".join(STRD_LINEAR),
    bits=52,
)


@pytest.mark.oracle_grid(
    reason="the network: NIST's datasets, each solve's error held to the ledger"
)
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
            _record_nist(f"{name.lower()}_{model}", err)
            assert err <= bound, (name, model, err, bound)

    @pytest.mark.parametrize("name", STRD_SOLVABLE)
    def test_the_gram_solve_reaches_the_bound_standardized(self, name):
        # `gram.solve` diagonalizes the Gram (numpy's `eigh`); standardized,
        # the correlation matrix's eigenvalues are what it inverts, and the
        # normal equations' bound holds. Unstandardized the Gram is scaled
        # by its diagonal first, and holds it too (`TestGramSolveScales`).
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
        _record_nist(f"{name.lower()}_gram_standardized", err)
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
        _record_nist(f"{name.lower()}_rls", err)
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
        _record_nist("filip_rls", err)
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


def _last_row(
    spec: dict[str, Any], x: np.ndarray, y: np.ndarray
) -> tuple[pl.DataFrame, po.ModelBank, int]:
    """Every row but the last in one chunk, then the last alone, so the
    last row's own solve can be told from the rest: the fit's rows, the
    bank, and the failures counted before the last row."""
    cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    frame = pl.DataFrame({**cols, "y": y})
    bank = po.ModelBank([spec])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", po.ReadinessWarning)
        head = bank.fit_predict(frame[:-1])["m"].struct.unnest()
        before = solve_failures(bank)
        tail = bank.fit_predict(frame[-1:])["m"].struct.unnest()
    return pl.concat([head, tail]), bank, before


def _served_reasons(bank: po.ModelBank, x: np.ndarray) -> tuple[int, list[Any]]:
    """`predict` on the rows `x` against the fit as it stands -- the rows
    after the stream, the serving path -- its null predictions counted, and
    each row's `withheld_reason`."""
    out = bank.predict(pl.DataFrame({f"x{i}": x[:, i] for i in range(x.shape[1])}))
    out = out["m"].struct.unnest()
    return out["pred_y"].null_count(), out["withheld_reason"].cast(pl.String).to_list()


class TestIllConditioned:
    """A solve past the condition at which it keeps a digit is withheld and
    counted (task 278 (d4); defect D4 of 278 (b)). Past ``kappa_c u = 1`` the
    normal equations carry no digit, and the Cholesky does not always fail
    there: rounding can leave the matrix numerically positive definite, and
    where it succeeded the coefficients were off by far more than their own
    size with nothing said -- no jitter, ``support_coef`` 1, every gate
    passed, no warning. Measured 2026-10-25 at ``eps = 1e-11`` (a correlation
    condition near ``1e22`` by construction): silent on 19 of 40 seeds, 16 of
    40 at ``1e-12`` and 19 of 40 at ``1e-13``; seed 3 had slopes -364.1 and
    363.1 for a truth of 1 and -2.

    The solve now estimates the equilibrated system's ``||H^-1||_1`` from its
    factor (Hager and Higham's estimator) and withholds the fit where the
    forward-error bound of the solve reaches 1, ``3 k gamma_{3k+1}
    ||H^-1||_1 >= 1`` (``crates/online-core/src/solve/condition.rs`` derives
    it): ``coef``, ``pred`` and ``support_coef`` null, ``withheld_reason``
    ``ill_conditioned``, and the solve counted in ``solve_failures``. A row's
    prediction comes from the fit before it, so the last row's own solve is
    read on its ``coef`` and ``solve_failures``, and its withheld fit on the
    rows predicted after it."""

    def test_a_numerically_singular_solve_is_withheld_and_counted(self):
        n = 800
        x, y = _rows(n, 3, "collinear", 1e-11)
        _, _, kappa_c, _ = design(x, True)
        assert kappa_c * U > 0.5, "the design was meant to be numerically singular"
        out, bank, before = _last_row(po.spec.ewridge("m", ridge=0.0, **_SYNTH), x, y)
        last = out.row(-1, named=True)
        print("last row", {k: last[k] for k in ("coef", "support_coef")})
        assert solve_failures(bank) == before + 1, "the last solve was not counted"
        assert all(c is None for c in last["coef"])
        assert all(v is None for v in last["support_coef"])
        # The rows predicted from it -- `predict`, the serving path, and the
        # next row a stream learns from -- are withheld and say why.
        assert _served_reasons(bank, x[-5:]) == (5, ["ill_conditioned"] * 5)
        cols = {f"x{i}": x[:1, i] for i in range(3)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            nxt = bank.fit_predict(pl.DataFrame({**cols, "y": y[:1]}))["m"].struct.unnest()
        assert nxt["pred_y"][0] is None
        assert nxt["withheld_reason"].cast(pl.String)[0] == "ill_conditioned"

    @pytest.mark.extended(reason="120 fits of 800 rows, the seed grid of the pinned defect")
    @pytest.mark.parametrize("eps", [1e-11, 1e-12, 1e-13])
    def test_no_seed_is_silent_past_one_over_u(self, eps):
        # Every seed's last fit either took jitter -- counted, the duplicated
        # pair's shares 0.5, the predictions least squares' -- or was
        # withheld as ill-conditioned; none reports coefficients off by more
        # than their size with nothing said. Measured 2026-10-25: withheld
        # 19, 16 and 19 of 40 at 1e-11, 1e-12 and 1e-13 -- the seeds that
        # were silent -- and the rest jittered.
        counts = {"withheld": 0, "jittered": 0}
        for seed in range(40):
            x, y = _rows(800, seed, "collinear", eps)
            out, bank, before = _last_row(po.spec.ewridge("m", ridge=0.0, **_SYNTH), x, y)
            last = out.row(-1, named=True)
            assert solve_failures(bank) > before, (seed, "the last solve was not counted")
            if all(c is None for c in last["coef"]):
                assert _served_reasons(bank, x[-3:]) == (3, ["ill_conditioned"] * 3), seed
                counts["withheld"] += 1
            else:
                coef = np.array(last["coef"], float)
                assert coef[1] + coef[2] == pytest.approx(-1.0, abs=0.05), (seed, coef)
                assert np.max(np.abs(np.array(last["support_coef"][1:3]) - 0.5)) < 0.01, seed
                counts["jittered"] += 1
        print(eps, counts)
        assert counts["withheld"] > 0, "the detector never fired"

    @pytest.mark.extended(reason="the network: NIST's Filip, downloaded once and cached")
    def test_filip_by_the_normal_equations_is_withheld(self):
        # Filip by the normal equations kept 0.0 digits of every coefficient
        # (NIST's measure) with nothing said; `rls` reaches 7.3
        # (`test_filip_is_solved_by_rotations...`). Its correlation matrix's
        # condition is 1.5e16, past `1 / u`, so the estimate is too.
        d = nist_strd_linear_or_skip("Filip")
        x, y = d["x"], d["y"]
        out, bank, before = _last_row(_strd_specs(x.shape[1], True)["ewridge"], x, y)
        last = out.row(-1, named=True)
        assert solve_failures(bank) == before + 1
        assert all(c is None for c in last["coef"])
        assert _served_reasons(bank, x[-3:]) == (3, ["ill_conditioned"] * 3)
        # Measured 2026-10-25: of Filip's 82 rows, 29 predicted from a fit
        # withheld for its condition; the rest from a fit that took jitter
        # (counted) or under the noise gate.
        assert (out["withheld_reason"].cast(pl.String) == "ill_conditioned").sum() > 10

    @pytest.mark.extended(reason="the network: NIST's datasets, downloaded once and cached")
    @pytest.mark.parametrize("name", STRD_SOLVABLE)
    def test_the_solvable_datasets_are_not_withheld(self, name):
        # The detector does not fire where the normal equations keep digits:
        # the last fit of each of NIST's solvable designs (Wampler's
        # correlation matrices at 2.7e6, seven orders below the limit) is
        # reported, and its solve counted nowhere.
        d = nist_strd_linear_or_skip(name)
        x, y, icpt = d["x"], d["y"], d["intercept"]
        out, bank, before = _last_row(_strd_specs(x.shape[1], icpt)["ewridge"], x, y)
        assert "ill_conditioned" not in out["withheld_reason"].cast(pl.String).to_list()
        assert solve_failures(bank) == before
        assert np.all(np.isfinite(last_coef(out)))
        n_null, reasons = _served_reasons(bank, x[-3:])
        assert n_null == 0 and reasons == [None] * 3


class TestGramSolveScales:
    """``gram.solve(standardize=False)`` scales the Gram by its diagonal
    before ``eigh`` (task 278 (d4); defect D5 of 278 (b)). ``eigh``'s
    eigenvalues are accurate to ``u`` times the largest, so on the raw Gram
    with features from 1e-10 to 1e10 the small feature's eigenvalue (1e-20)
    was lost beside 1e20: relative errors of 8.9, 2.4e-2, 1.0 and 1.1e-3
    against least squares (measured 2026-10-25), and NIST's Longley kept 6.4
    digits where the model keeps 11.5."""

    def test_the_gram_solve_unstandardized_is_free_of_the_features_units(self):
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
        np.testing.assert_allclose(raw, want, rtol=1e-12)
        # Measured 2026-10-25: 9.4e-14 at most, against 8.9 before.
        print("raw", float(np.max(np.abs(raw - want) / np.abs(want))))

    def test_a_ridge_grid_on_unequal_scales_solves_each_ridge_as_alone(self):
        # The uniform ridge is not uniform once the columns are scaled, so
        # on columns of unequal scale each ridge takes a decomposition of its
        # own: a grid gives each ridge's row as a call with that ridge alone
        # gives it, and each row solves its own system, against scipy's
        # Cholesky solve of the column-scaled system (the oracle). On the
        # standardized system, whose diagonal is even, the grid rides one
        # decomposition, as before.
        import scipy.linalg

        n = 800
        x, y = _rows(n, 12, "scales")
        _, bank = fit(po.spec.ewridge("m", ridge=0.0, **_SYNTH), x, y)
        g = bank.gram("m")[0]
        ridges = [0.0, 1e-24, 1e-18, 1e-6, 1.0]
        grid = np.asarray(gram.solve(g, ridge=ridges))
        assert g["columns"][0] == "intercept"
        c = np.asarray(g["comoments"], float)[1:, 1:]
        b = np.asarray(g["cross_centered"], float)[0][1:]
        for i, r in enumerate(ridges):
            alone = np.asarray(gram.solve(g, ridge=r)).ravel()
            # The slopes to the bit; the intercept, `ybar - m . b`, is a
            # product numpy takes over the grid's rows at once, which may
            # round apart from one row's.
            np.testing.assert_array_equal(grid[i][1:], alone[1:])
            np.testing.assert_allclose(grid[i][0], alone[0], rtol=1e-14)
            m = c + r * np.eye(3)
            e = np.sqrt(np.diag(m))
            want = scipy.linalg.solve(m / np.outer(e, e), b / e, assume_a="pos") / e
            np.testing.assert_allclose(grid[i][1:], want, rtol=1e-11, err_msg=str(r))

    @pytest.mark.oracle_grid(
        reason="the network: NIST's datasets, each solve's error held to the ledger"
    )
    @pytest.mark.parametrize("name", STRD_SOLVABLE)
    def test_the_gram_solve_unstandardized_reaches_the_bound(self, name):
        # As `test_the_gram_solve_reaches_the_bound_standardized`, on the
        # raw Gram: Longley's features run from 1e2 to 4e5, and it kept 6.4
        # digits, 3.8e-7 against the bound of 2.3e-10, before the Gram was
        # scaled; 12.0 digits, 3.2e-13, after (measured 2026-10-25).
        d = nist_strd_linear_or_skip(name)
        x, y, want, icpt = d["x"], d["y"], d["coef"], d["intercept"]
        n, k = x.shape
        s, _, kappa_c, _ = design(x, icpt)
        _, bank = fit(_strd_specs(k, icpt)["ewridge"], x, y)
        coef = np.asarray(gram.solve(bank.gram("m")[0], ridge=0.0)).ravel()
        err = standardized_error(coef, want, s, icpt)
        bound = forward_bound(kappa_c, gamma(4 * n + 3 * k + 1 + d["degree"] + 4 * k))
        _record_nist(f"{name.lower()}_gram_unstandardized", err)
        assert err <= bound, (name, err, bound)
