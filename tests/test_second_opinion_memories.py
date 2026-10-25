"""A diagnostic's memory of ``inf`` beside a fit that forgets: the run-once
statistic of the rows it folded (docs/PLAN.md task 278, batch 5).

A diagnostic with a memory of its own folds each scored row once the
stream has settled on the fit's memory (95% of the weight the fit can
hold), and with ``inf`` it forgets none of them. So beside a fit at a
finite half-life, its fields on row ``t`` are the classical statistic of
the folded rows before ``t``, computed here by statsmodels and scipy: the
calibration is ``OLS`` of ``y`` on ``pred`` and twice its ``f_test``'s F,
the tails ``scipy.stats``' ``skew`` and ``kurtosis`` and statsmodels'
``jarque_bera`` of the recursive residuals, Ljung and Box's statistic
``acorr_ljungbox``, and the robust standard errors the sandwich with
statsmodels' HAC kernel ``S_hac_simple`` (the rows up to and including
``t``, as ``coef`` reads them). The other memories' second opinions, at the
fit's memory and at memories of their own, are in
``tests/test_second_opinion_diagnostics.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "mixed"

INF = float("inf")
HALF_LIFE = 120.0


def _rows(n: int, seed: int) -> pl.DataFrame:
    """Two features, heavy-tailed noise whose spread grows with the first."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    e = rng.standard_t(5, size=n) * (0.5 + 0.5 * np.abs(x[:, 0]))
    y = 0.4 + x @ np.array([1.2, -0.8]) + e
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


#: The models the limit is held on: ``ewridge`` in the essentials, and two
#: whose residuals come another way, one with no error inflation.
MODELS = [
    pytest.param(po.spec.ewridge, {"emit_error_inflation": True}, id="ewridge"),
    pytest.param(
        po.spec.rls,
        {"emit_error_inflation": True},
        id="rls",
        marks=pytest.mark.extended(reason="a grid over the models"),
    ),
    pytest.param(
        po.spec.sgd, {}, id="sgd", marks=pytest.mark.extended(reason="a grid over the models")
    ),
]


def _run(build, extra: dict[str, Any], df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
    spec = build(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALF_LIFE,
        min_weight=10.0,
        **extra,
        **kw,
    )
    return po.ModelBank([spec]).fit_predict(df).unnest("m")


def _folded(out: pl.DataFrame) -> np.ndarray:
    """The rows a diagnostic folds: those the stream had settled by (the
    fit's 95%), and that were scored. The evidence the memory is not the
    fit's: the first rows of the stream are not among them."""
    ready = out["settled_frac"].fill_null(0.0).fill_nan(0.0).to_numpy() >= 0.95
    first = int(np.argmax(ready))
    # 4.3 half-lives of 120 rows (1 - 2^-4.32 = 0.95).
    assert 500 < first < 560, first
    return ready & np.isfinite(out["resid_y"].to_numpy())


class TestAnInfiniteMemoryIsTheRunOnceOracle:
    @pytest.mark.parametrize(("build", "extra"), MODELS)
    def test_the_calibration_is_ols_and_its_f_test(self, build, extra):
        import statsmodels.api as sm

        df = _rows(1_500, 2781)
        out = _run(build, extra, df, emit_calibration=True, calibration_half_life=INF)
        ok = _folded(out)
        pred, y = out["pred_y"].to_numpy(), df["y"].to_numpy()
        checked = 0
        for t in (700, 1_100, 1_499):
            k = ok[:t]
            fit = sm.OLS(y[:t][k], sm.add_constant(pred[:t][k])).fit()
            f = float(fit.f_test((np.eye(2), np.array([0.0, 1.0]))).fvalue)
            row = out.row(t, named=True)
            # Measured 2026-10-25 on ewridge: 4e-14 relative at most.
            np.testing.assert_allclose(row["calibration_slope_y"], fit.params[1], rtol=1e-10)
            np.testing.assert_allclose(row["calibration_intercept_y"], fit.params[0], rtol=1e-9)
            np.testing.assert_allclose(row["calibration_wald_y"], 2.0 * f, rtol=1e-9)
            checked += 1
        assert checked == 3

    @pytest.mark.parametrize(("build", "extra"), MODELS)
    def test_the_tails_are_scipys_and_jarque_beras(self, build, extra):
        from scipy import stats
        from statsmodels.stats.stattools import jarque_bera

        df = _rows(1_500, 2782)
        out = _run(build, extra, df, emit_tails=True, tails_half_life=INF)
        ok = _folded(out)
        # The recursive residual: over the error inflation where the model
        # reads one, the residual itself where it does not (`sgd`).
        resid = out["resid_y"].to_numpy()
        if extra.get("emit_error_inflation"):
            resid = resid / out["error_inflation_y"].to_numpy()
        for t in (700, 1_100, 1_499):
            v = resid[:t][ok[:t]]
            np.testing.assert_allclose(out["skew_y"][t], stats.skew(v), rtol=1e-10)
            np.testing.assert_allclose(out["kurtosis_y"][t], stats.kurtosis(v), rtol=1e-10)
            np.testing.assert_allclose(out["jarque_bera_y"][t], jarque_bera(v)[0], rtol=1e-10)

    @pytest.mark.parametrize(("build", "extra"), MODELS)
    def test_ljung_box_is_acorr_ljungbox(self, build, extra):
        from statsmodels.stats.diagnostic import acorr_ljungbox

        df = _rows(1_500, 2783)
        out = _run(
            build,
            extra,
            df,
            emit_specification=True,
            specification_half_life=INF,
            ljung_box_lags=6,
        )
        ok = _folded(out)
        resid = out["resid_y"].to_numpy()
        for t in (700, 1_100, 1_499):
            want = acorr_ljungbox(resid[:t][ok[:t]], lags=[6])["lb_stat"].iloc[0]
            np.testing.assert_allclose(out["ljung_box_y"][t], want, rtol=1e-10)

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(po.spec.ewridge, id="ewridge"),
            pytest.param(
                po.spec.rls, id="rls", marks=pytest.mark.extended(reason="a grid over the models")
            ),
        ],
    )
    def test_the_sandwich_is_statsmodels_kernel(self, build):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import S_hac_simple

        lags = 4
        df = _rows(1_500, 2784)
        out = _run(
            build,
            {},
            df,
            emit_robust_se=True,
            robust_se_half_life=INF,
            robust_se_lags=lags,
            coef_every=0,
        )
        ok = _folded(out)
        resid = out["resid_y"].to_numpy()
        X = sm.add_constant(df.select("x0", "x1").to_numpy())
        checked = 0
        for t in (800, 1_100, 1_499):
            k = ok[: t + 1]
            Xk, ek = X[: t + 1][k], resid[: t + 1][k]
            bi = np.linalg.inv(Xk.T @ Xk)
            for nlags, field in ((0, "se_coef_hc0"), (lags, "se_coef_hac")):
                want = np.sqrt(np.diag(bi @ S_hac_simple(Xk * ek[:, None], nlags=nlags) @ bi))
                # Measured 2026-10-25: 6e-15 relative at most.
                np.testing.assert_allclose(out[field][t].to_numpy(), want, rtol=1e-10)
                checked += 1
        assert checked == 6
