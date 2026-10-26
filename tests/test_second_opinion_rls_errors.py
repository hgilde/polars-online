"""Second opinions for ``rls``'s standard errors and influence, and
``kalman``'s influence (task 278, batch 23; PARAMS B5).

``ewridge``'s ``se_coef``, ``se_coef_hc0`` / ``se_coef_hac`` and
``influence`` are held to statsmodels in ``tests/test_second_opinion.py``
and ``tests/test_second_opinion_diagnostics.py``; ``rls`` reads the same
three from its own recursion and had no oracle, and ``kalman``'s influence
none either. Each test here runs the model where it is least squares -- no
decay, a vanishing prior -- and compares with statsmodels on the raw rows:

- ``se_coef / sigma`` is ``sqrt(diag((X'X)⁻¹))``, statsmodels' ``OLS``
  ``cov_params() / scale``, the intercept's included;
- ``se_coef_hc0`` and ``se_coef_hac`` are the sandwich ``(X'ΩX)⁻¹ S
  (X'ΩX)⁻¹`` with ``S`` statsmodels' ``S_hac_simple`` over the scores of the
  bank's own out-of-sample residuals, run once and under a memory and
  weights, as ``ewridge``'s are;
- ``influence`` on row ``t`` is ``OLSInfluence(...).dffits`` of the fit over
  rows ``0..=t`` at its last row.

The prior is what parts the model from least squares, and the gap is linear
in it: ``rls``'s ``delta`` adds ``delta·I`` to ``X'X`` and ``kalman``'s
``p0`` a prior precision of ``1/p0``. Each tolerance below is that measured
gap with headroom; the gap moves by the factor the prior does (measured
2026-10-26: influence 1.2e-7 / 1.2e-8 / 1.2e-9 at ``delta`` 1e-8 / 1e-9 /
1e-10, and 1.2e-5 / 1.2e-6 at ``p0`` 1e6 / 1e7). ``kalman``'s covariance
form loses digits past ``p0 = 1e7`` (4.6e-6 at 1e8, the same at any
``obs_var``), so it is read at 1e7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def _leverage_rows(n: int = 300, seed: int = 81) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Two features away from zero, with rows of high leverage."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([4.0, -2.0])
    x[::37, 0] += 5.0
    y = 0.5 + x @ np.array([1.0, -1.0]) + rng.normal(size=n)
    return x, y, pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


def _sandwich_rows(n: int, seed: int) -> pl.DataFrame:
    """Two features far from zero, a residual whose spread grows with the
    first, and uneven weights with a zero now and then."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([3.0, -1.0])
    e = rng.normal(size=n) * (0.5 + np.abs(x[:, 0] - 3.0))
    y = 0.5 + x @ np.array([1.0, -2.0]) + e
    w = rng.uniform(0.5, 1.5, size=n)
    w[rng.random(n) < 0.05] = 0.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})


class TestRlsSeCoefIsLeastSquares:
    def test_se_over_sigma_is_the_least_squares_covariance(self):
        """With no decay and ``delta = 1e-10``, ``rls``'s ``se_coef`` over
        ``sigma`` is statsmodels' ``(X'X)⁻¹`` over the rows so far. As in
        ``ewridge``'s test, ``sigma`` is the row's (the residuals before it)
        and ``se_coef`` the fit after it (measured 2026-10-26: 7.3e-11
        relative at most, at the earliest row read)."""
        import statsmodels.api as sm

        x, y, df = _leverage_rows(200, 53)
        spec = po.spec.rls(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            delta=1e-10,
            min_weight=3.0,
            emit_se_coef=True,
            emit_sigma=True,
            coef_every=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        se, sigma = out["se_coef"].to_list(), out["sigma_y"].to_numpy()
        checked = 0
        for i in (8, 20, 60, 199):
            fit = sm.OLS(y[: i + 1], sm.add_constant(x[: i + 1])).fit()
            want = np.sqrt(np.diag(fit.cov_params() / fit.scale))
            np.testing.assert_allclose(np.asarray(se[i]) / sigma[i], want, rtol=1e-9)
            checked += 1
        assert checked == 4


class TestRlsRobustStandardErrorsAreTheSandwich:
    @pytest.mark.parametrize(("half_life", "weighted"), [(float("inf"), False), (150.0, True)])
    def test_the_sandwich_is_statsmodels_kernel_on_the_out_of_sample_residuals(
        self, half_life, weighted
    ):
        """The folded rows are those at least 95% settled (all of them run
        once); each is weighed by its weight aged at the fit's memory.
        Measured 2026-10-26: 1.1e-14 relative at most: the prior has
        decayed or is read against 700 rows."""
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import S_hac_simple

        lags = 4
        df = _sandwich_rows(900, 51)
        kw: dict[str, Any] = {"half_life": half_life}
        if weighted:
            kw["weight"] = "w"
        spec = po.spec.rls(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            delta=1e-10,
            min_weight=10.0,
            emit_robust_se=True,
            robust_se_lags=lags,
            coef_every=0,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df if weighted else df.drop("w")).unnest("m")
        resid = out["resid_y"].to_numpy()
        w = df["w"].to_numpy() if weighted else np.ones(df.height)
        X = sm.add_constant(df.select("x0", "x1").to_numpy())
        ready = out["settled_frac"].fill_null(1.0).fill_nan(1.0).to_numpy() >= 0.95
        checked = 0
        for t in (700, 800, 899):
            omega = w[: t + 1] * 0.5 ** ((t - np.arange(t + 1)) / half_life)
            keep = np.isfinite(resid[: t + 1]) & (omega > 0) & ready[: t + 1]
            assert keep.sum() >= 40, keep.sum()
            Xk, ek, om = X[: t + 1][keep], resid[: t + 1][keep], omega[keep]
            bread = np.linalg.inv((Xk * om[:, None]).T @ Xk)
            for nlags, field in ((0, "se_coef_hc0"), (lags, "se_coef_hac")):
                S = S_hac_simple(Xk * (om * ek)[:, None], nlags=nlags)
                want = np.sqrt(np.diag(bread @ S @ bread))
                np.testing.assert_allclose(out[field][t].to_numpy(), want, rtol=1e-10)
                checked += 1
        assert checked == 6


class TestInfluenceIsDffitsAtTheNewestRow:
    """``influence`` is ``(v / s) * sqrt(h)``, ``h = error_inflation**2 - 1``:
    run once with a vanishing prior it is statsmodels' DFFITS at the last row
    of the prefix, for ``rls`` and for ``kalman`` with its noise given and
    its features unstandardized (standardized, its prior sits on moving
    moments, and the fit is not least squares in the raw features)."""

    @pytest.mark.parametrize(
        ("kind", "rtol"),
        [("rls", 1e-8), ("kalman", 1e-5)],
        ids=["rls", "kalman"],
    )
    def test_each_rows_influence_is_its_prefixs_last_dffits(self, kind, rtol):
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence

        x, y, df = _leverage_rows()
        common: dict[str, Any] = dict(
            targets=["y"], features=["x0", "x1"], half_life=float("inf"), emit_influence=True
        )
        if kind == "rls":
            spec = po.spec.rls("m", delta=1e-10, min_weight=3.0, **common)
        else:
            # No `min_weight`: how it meets kalman's learning is being
            # changed elsewhere; with the noise given, the first prediction
            # comes as soon as the fit is determined either way.
            spec = po.spec.kalman(
                "m",
                coef_half_life=float("inf"),
                p0=1e7,
                obs_var=1.0,
                standardize=False,
                **common,
            )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        got = out["influence_y"].to_numpy().astype(float)
        X = sm.add_constant(x)
        rows = np.flatnonzero(np.isfinite(got))
        # The first rows past the three the fit needs carry the prior's
        # largest share: the bound is set there.
        assert rows[0] <= 6 and rows.size > 290, (rows[0], rows.size)
        # Every row where the prior's share is largest, every fifth after.
        for t in rows[(rows < 30) | (rows % 5 == 0)]:
            want = OLSInfluence(sm.OLS(y[: t + 1], X[: t + 1]).fit()).dffits[0][t]
            np.testing.assert_allclose(got[t], want, rtol=rtol, err_msg=f"row {t}")
