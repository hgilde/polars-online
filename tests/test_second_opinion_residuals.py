"""The residual diagnostics on ``huber``, ``quantile``, ``lasso``, ``pa``,
``ftrl`` and ``holt``, each recomputed by statsmodels, scipy or numpy from
that kind's own emitted predictions and residuals.

The statistics are one stream's code whatever the kind, and their
oracles already run on ``ewridge`` (``test_second_opinion_diagnostics.py``)
and on ``rls``, ``sgd`` and ``kalman`` for some. What differs by kind is
what feeds them: the predictions and residuals, the leverage, which rows
fold, and for ``quantile`` the calibration and the CUSUM's terms. So each is
checked here on each kind, run once (``half_life = inf``) with unit
weights, against the same libraries:

- ``calibration_slope``, ``calibration_intercept`` and ``calibration_wald``:
  Mincer and Zarnowitz's regression of ``y`` on the scored ``pred``,
  statsmodels' ``OLS`` and twice its ``f_test``'s F. On ``quantile`` the
  calibration is the coverage, the share of rows below the prediction, and
  its binomial test ``(sum d)^2 / (q (1 - q) n)``, ``d = 1{y < pred} - q``.
- ``ljung_box``, ``breusch_pagan`` and ``ramsey_reset``: statsmodels'
  ``acorr_ljungbox``, ``het_breuschpagan`` on the slot's features (``holt``
  has none, and reports none) and ``compare_lm_test`` of the residual on
  ``pred``, ``pred^2`` and ``pred^3`` against ``pred``.
- ``ic`` (``emit_metrics``): scipy's ``pearsonr`` of prediction and target.
- ``studentized``: the residual over its error inflation, over the spread
  of those folded before it, once that spread holds ten rows. No kind here
  keeps a per-row estimation variance, so the inflation is 1, and
  ``studentized`` is the residual over the root mean square of the earlier
  ones; ``cusum`` is the Brown, Durbin and Evans path of them over the
  square root of their count, on ``quantile`` of the standardized
  indicators ``d / sqrt(q (1 - q))``.
- ``break_wald`` at ``breaks_half_life = 60``: the Wald distance between
  statsmodels' ``WLS`` fits of the target on the slot's features at that
  memory and at four times it, over the exact variance of their
  difference. The twin fits read no prediction, so they fold every row.

Each fold starts at the instance's readiness: the first row with a
prediction (``min_weight``), and on ``pa``, which standardizes its features
by default, the first row after its scaler's warm-up of 22 rows by Kish's
count. ``ic`` folds from the first prediction on every kind.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from scipy.stats import chi2, pearsonr
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

import polars_online as po

TIER = "essential"

#: Each kind's required options beyond targets and features.
KINDS: dict[str, dict[str, Any]] = {
    "huber": {},
    "quantile": {"quantile": 0.5},
    "lasso": {"penalties": [0.01]},
    "pa": {},
    "ftrl": {"loss": "squared"},
    "holt": {},
}

#: The rows a standardizing ``pa`` warms its scaler over before its
#: diagnostics fold, by Kish's count (unit weights here: its row count).
SCALER_ROWS = 22

#: Relative error. Measured 2026-11-04 over every kind and row checked
#: (aarch64, macOS): the worst 1.3e-12, ``holt``'s RESET at row 699, whose
#: ``pred^3`` columns are badly conditioned; every other statistic within
#: 3.4e-13. The two sides solve the same small least-squares problems by
#: different factorizations; the bound leaves a hundred times the worst.
TOL = 1e-10

#: Rows each statistic is read at: early, middle and last.
ROWS = (150, 400, 699)


def spec(kind: str, **kw: Any) -> dict[str, Any]:
    options = {"targets": ["y"], "half_life": float("inf"), "min_weight": 10.0}
    if kind != "holt":
        options["features"] = ["x0", "x1"]
    return getattr(po.spec, kind)("m", **options, **KINDS[kind], **kw)


def stream(n: int, seed: int, shift: float = 0.0) -> pl.DataFrame:
    """Two features; noise whose spread grows with the first; a square of it
    no fit here models; and an intercept that moves by ``shift`` at row
    ``2n/3``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    e = rng.normal(size=n) * np.maximum(0.2, 1.0 + 0.6 * x[:, 0])
    level = 0.5 + shift * (np.arange(n) >= 2 * n // 3)
    y = level + x @ np.array([1.0, -1.0]) + 0.2 * x[:, 0] ** 2 + e
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


def folded(kind: str, resid: np.ndarray) -> np.ndarray:
    """The rows the diagnostics fold: scored, and past ``pa``'s scaler."""
    start = SCALER_ROWS if kind == "pa" else 0
    return np.isfinite(resid) & (np.arange(len(resid)) >= start)


def rel(got: Any, want: float) -> float:
    assert got is not None
    return abs(got - want) / abs(want)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_calibration_specification_and_ic_are_statsmodels_and_scipy(kind):
    df = stream(700, 61)
    out = (
        po.ModelBank(
            [
                spec(
                    kind,
                    emit_calibration=True,
                    emit_specification=True,
                    ljung_box_lags=6,
                    emit_metrics=True,
                )
            ]
        )
        .fit_predict(df)
        .unnest("m")
    )
    y, x = df["y"].to_numpy(), df.select("x0", "x1").to_numpy()
    r = out["resid_y"].to_numpy().astype(float)
    p = out["pred_y"].to_numpy().astype(float)
    fold = folded(kind, r)
    errors: dict[str, float] = {}

    def check(field: str, t: int, want: float) -> None:
        errors[field] = max(errors.get(field, 0.0), rel(out[field][t], want))

    for t in ROWS:
        ok = fold[:t]
        rt, pt, yt = r[:t][ok], p[:t][ok], y[:t][ok]
        if kind == "quantile":
            below = (yt < pt).astype(float)
            check("calibration_coverage_y", t, below.mean())
            check("calibration_wald_y", t, (below - 0.5).sum() ** 2 / (0.25 * len(below)))
        else:
            fit = sm.OLS(yt, sm.add_constant(pt)).fit()
            f = float(fit.f_test((np.eye(2), np.array([0.0, 1.0]))).fvalue)
            check("calibration_intercept_y", t, fit.params[0])
            check("calibration_slope_y", t, fit.params[1])
            check("calibration_wald_y", t, 2.0 * f)
        check("ljung_box_y", t, acorr_ljungbox(rt, lags=[6])["lb_stat"].iloc[0])
        if kind == "holt":
            assert out["breusch_pagan_y"][t] is None
        else:
            check("breusch_pagan_y", t, het_breuschpagan(rt, sm.add_constant(x[:t][ok]))[0])
        q = pt - pt[0]
        full = sm.OLS(rt, sm.add_constant(np.column_stack([q, q**2, q**3]))).fit()
        base = sm.OLS(rt, sm.add_constant(q)).fit()
        check("ramsey_reset_y", t, full.compare_lm_test(base)[0])
        scored = np.isfinite(p[:t])
        check("ic_y", t, pearsonr(p[:t][scored], y[:t][scored])[0])
    print(kind, {k: f"{v:.2g}" for k, v in errors.items()})
    # The heteroskedastic noise and the square are there to be found.
    assert out["breusch_pagan_y"][699] is None or out["breusch_pagan_y"][699] > 5.99
    assert max(errors.values()) <= TOL, errors


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_studentized_and_cusum_are_brown_durbin_and_evans_with_no_leverage(kind):
    df = stream(900, 41, shift=1.0)
    out = po.ModelBank([spec(kind, emit_breaks=True)]).fit_predict(df).unnest("m")
    y = df["y"].to_numpy()
    r = out["resid_y"].to_numpy().astype(float)
    p = out["pred_y"].to_numpy().astype(float)
    z = out["studentized_y"].to_numpy().astype(float)
    cusum = out["cusum_y"].to_numpy().astype(float)
    fold = folded(kind, r)
    worst, checked = 0.0, 0
    for t in np.flatnonzero(fold):
        before = np.flatnonzero(fold[:t])
        if len(before) < 10:
            assert np.isnan(z[t])
            continue
        want = r[t] / np.sqrt(np.mean(r[before] ** 2))
        worst = max(worst, abs(z[t] - want) / abs(want))
        checked += 1
    # The CUSUM on row t sums the terms folded before it.
    if kind == "quantile":
        terms = np.where(fold, ((y < p).astype(float) - 0.5) / 0.5, np.nan)
    else:
        terms = np.where(fold & np.isfinite(z), z, np.nan)
    ok = np.isfinite(terms)
    path, count = np.cumsum(np.where(ok, terms, 0.0)), np.cumsum(ok)
    read = np.flatnonzero(np.isfinite(cusum))
    want_cusum = path[read - 1] / np.sqrt(count[read - 1])
    worst_cusum = float((np.abs(cusum[read] - want_cusum) / (1.0 + np.abs(want_cusum))).max())
    print(f"{kind}: {checked} studentized rows, worst {worst:.2g}; cusum worst {worst_cusum:.2g}")
    assert checked > 800 and len(read) > 800
    assert worst <= TOL and worst_cusum <= TOL


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_break_wald_is_two_wls_fits(kind):
    h = 60.0
    df = stream(900, 41, shift=1.0)
    out = po.ModelBank([spec(kind, emit_breaks=True, breaks_half_life=h)]).fit_predict(df)
    out = out.unnest("m")
    y = df["y"].to_numpy()
    X = sm.add_constant(df.select("x0", "x1").to_numpy()) if kind != "holt" else np.ones((900, 1))
    worst = 0.0
    for t in (300, 650, 899):
        age = (t - 1) - np.arange(t)
        wf, ws = 0.5 ** (age / h), 0.5 ** (age / (4 * h))
        Xt, yt = X[:t], y[:t]
        fast = sm.WLS(yt, Xt, weights=wf).fit()
        slow = sm.WLS(yt, Xt, weights=ws).fit()
        ns = ws.sum() ** 2 / (ws**2).sum()
        s2 = (ws * slow.resid**2).sum() / ws.sum() * ns / (ns - Xt.shape[1])
        # The difference is `D e`, `D = A_f X'W_f - A_s X'W_s`: its variance
        # over the noise's is `D D'`.
        D = np.linalg.solve((Xt * wf[:, None]).T @ Xt, (Xt * wf[:, None]).T) - np.linalg.solve(
            (Xt * ws[:, None]).T @ Xt, (Xt * ws[:, None]).T
        )
        d = fast.params - slow.params
        want = d @ np.linalg.solve(D @ D.T, d) / s2
        worst = max(worst, rel(out["break_wald_y"][t], want))
    print(f"{kind}: break_wald worst {worst:.2g}, at the break {out['break_wald_y'][650]:.1f}")
    assert out["break_wald_y"][650] > chi2.isf(0.01, X.shape[1]), "the break passes chi2(k)'s 1%"
    assert worst <= TOL
