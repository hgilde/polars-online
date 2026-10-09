"""Second opinions for the diagnostics with a memory of their own (docs/PLAN.md
tasks 221 and 232), split from ``tests/test_second_opinion.py`` at the
250 KB a source file is held to.

Each test compares a number the bank reports with the same number computed
from the raw rows by statsmodels, scipy or numpy, independently of this
library's own arithmetic: ``Calibration`` is Mincer and Zarnowitz's least
squares, the breaks Brown, Durbin and Evans', the robust standard errors and
the Newey-West forms statsmodels' HAC kernel, the specification tests
statsmodels' own, the tails scipy's, the influence statsmodels' DFFITS, and
the nested comparisons their papers'.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def _calibration_rows(n: int, seed: int) -> pl.DataFrame:
    """A fit whose predictions are miscalibrated in warm-up and drift, with
    uneven weights and a zero now and then."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 0.4 + x @ np.array([1.2, -0.8]) + rng.normal(size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    w[rng.random(n) < 0.05] = 0.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})


class TestCalibrationIsMincerZarnowitz:
    """Task 221 (a): ``calibration_*`` is the least-squares regression of
    ``y`` on the scored ``pred`` (Mincer and Zarnowitz 1969), and its Wald
    statistic the joint test of slope 1 and intercept 0. Run once
    (``calibration_half_life=inf``) with unit weights it is ``statsmodels``'
    ``OLS`` and twice its ``f_test``'s F; with a memory and weights its
    slope and intercept are ``WLS`` at each scored row's weight times its
    decay, and the statistic is the definition at Kish's size, written by
    hand from ``WLS``'s residuals. Each row's fields read the rows before
    it, never the row."""

    def test_run_once_is_ols_and_its_f_test(self):
        import statsmodels.api as sm

        df = _calibration_rows(600, 11).drop("w")
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=20.0,
            emit_calibration=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        pred, y = out["pred_y"].to_numpy(), df["y"].to_numpy()
        checked = 0
        for t in (60, 250, 599):
            ok = np.isfinite(pred[:t])
            fit = sm.OLS(y[:t][ok], sm.add_constant(pred[:t][ok])).fit()
            f = float(fit.f_test((np.eye(2), np.array([0.0, 1.0]))).fvalue)
            row = out.row(t, named=True)
            np.testing.assert_allclose(row["calibration_intercept_y"], fit.params[0], rtol=1e-9)
            np.testing.assert_allclose(row["calibration_slope_y"], fit.params[1], rtol=1e-9)
            np.testing.assert_allclose(row["calibration_wald_y"], 2.0 * f, rtol=1e-8)
            checked += 1
        assert checked == 3

    def test_a_memory_and_weights_are_wls_at_kish_size(self):
        import statsmodels.api as sm

        h_model, h_calib = 40.0, 120.0
        df = _calibration_rows(700, 12)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=h_model,
            weight="w",
            min_weight=10.0,
            emit_calibration=True,
            calibration_half_life=h_calib,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        pred, y, w = out["pred_y"].to_numpy(), df["y"].to_numpy(), df["w"].to_numpy()
        # Folded once the stream is 95% settled (task 232 (1)).
        ready = out["settled_frac"].to_numpy() >= 0.95
        checked = 0
        for t in (250, 400, 699):
            s = np.arange(t)
            # Row s folded at its weight, aged by every row after it up to t - 1.
            omega = w[:t] * 0.5 ** ((t - 1 - s) / h_calib)
            ok = np.isfinite(pred[:t]) & (omega > 0) & ready[:t]
            X = sm.add_constant(pred[:t][ok])
            fit = sm.WLS(y[:t][ok], X, weights=omega[ok]).fit()
            om = omega[ok]
            n_kish = om.sum() ** 2 / (om**2).sum()
            s2 = (om * fit.resid**2).sum() / om.sum() * n_kish / (n_kish - 2.0)
            d = fit.params - np.array([0.0, 1.0])
            m = (X * om[:, None]).T @ X / om.sum()
            wald = n_kish * d @ m @ d / s2
            row = out.row(t, named=True)
            np.testing.assert_allclose(row["calibration_intercept_y"], fit.params[0], rtol=1e-8)
            np.testing.assert_allclose(row["calibration_slope_y"], fit.params[1], rtol=1e-8)
            np.testing.assert_allclose(row["calibration_wald_y"], wald, rtol=1e-7)
            checked += 1
        assert checked == 3

    def test_an_embargo_folds_the_prediction_each_row_was_scored_with(self):
        """Under ``embargo=3`` on a row clock, row ``s`` is released as row
        ``s + 3`` arrives, before that row is scored: row ``t``'s fields are
        ``OLS`` over the rows up to ``t - 3``, each at the prediction it was
        shown (C21), not the one the model gives it at its release."""
        import statsmodels.api as sm

        df = _calibration_rows(400, 13).drop("w")
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=10.0,
            emit_calibration=True,
            embargo=3.0,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        pred, y = out["pred_y"].to_numpy(), df["y"].to_numpy()
        for t in (100, 399):
            upto = t - 2
            ok = np.isfinite(pred[:upto])
            fit = sm.OLS(y[:upto][ok], sm.add_constant(pred[:upto][ok])).fit()
            np.testing.assert_allclose(out["calibration_slope_y"][t], fit.params[1], rtol=1e-9)
            np.testing.assert_allclose(
                out["calibration_intercept_y"][t], fit.params[0], rtol=1e-8, atol=1e-12
            )


def _break_rows(n: int, seed: int, shift: float = 0.0, at: int | None = None) -> pl.DataFrame:
    """Two features and a target; the intercept moves by ``shift`` noise
    standard deviations from row ``at`` on."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    level = 0.5 + (shift * (np.arange(n) >= at) if at is not None else 0.0)
    y = level + x @ np.array([1.0, -1.0]) + rng.normal(size=n)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


class TestBreaksAreBrownDurbinAndEvans:
    """Task 221 (b). Run once with no ridge, ``resid / error_inflation`` is
    the recursive residual of Brown, Durbin and Evans (1975), which
    ``statsmodels``' ``recursive_olsresiduals`` computes by its own update of
    the inverse Gram; ``studentized`` is it over the spread of those before
    it, so ``statsmodels``' residuals give it back through the definition.
    ``cusum`` on row ``t`` is the CUSUM path of the studentized residuals
    before ``t`` over the square root of their count, and
    ``breaks_cusumolsresid`` reads the same statistic from both libraries'
    recursive residuals. ``break_wald``'s two fits are ``WLS`` at the two
    memories."""

    @staticmethod
    def _run(df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            ridge=1e-12,
            emit_breaks=True,
            emit_error_inflation=True,
            **{"half_life": float("inf"), **kw},
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("m")

    def test_the_studentized_residual_is_the_recursive_residual_over_its_spread(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import breaks_cusumolsresid, recursive_olsresiduals

        df = _break_rows(500, 21)
        out = self._run(df)
        X = sm.add_constant(df.select("x0", "x1").to_numpy())
        ols = sm.OLS(df["y"].to_numpy(), X).fit()
        scaled = recursive_olsresiduals(ols, skip=3)[4]
        mine = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        np.testing.assert_allclose(mine[3:], scaled[3:], rtol=0, atol=1e-8)
        # Each studentized residual against the spread of those before it,
        # once that spread has ten rows.
        z = out["studentized_y"].to_numpy().astype(float)
        checked = 0
        for t in range(3 + 10, 500):
            want = scaled[t] / np.sqrt(np.mean(scaled[3:t] ** 2))
            np.testing.assert_allclose(z[t], want, rtol=1e-7)
            checked += 1
        assert np.isnan(z[: 3 + 10]).all()
        assert checked == 487
        # `cusum` on row t: the path before t over its count's square root.
        cusum = out["cusum_y"].to_numpy().astype(float)
        ok = np.isfinite(z)
        path, count = np.cumsum(np.where(ok, z, 0.0)), np.cumsum(ok)
        np.testing.assert_allclose(
            cusum[15:], path[14:-1] / np.sqrt(count[14:-1]), rtol=1e-10, atol=1e-12
        )
        # Ploberger and Kramer's statistic, from either library's residuals.
        theirs = breaks_cusumolsresid(scaled[3:], ddof=0)[0]
        ours = breaks_cusumolsresid(mine[3:], ddof=0)[0]
        np.testing.assert_allclose(ours, theirs, rtol=1e-8)

    def test_the_cusum_crosses_where_statsmodels_does(self):
        """An intercept break of a noise sd at row 400 of 800, and none: the
        CUSUM path crosses Brown, Durbin and Evans' 5% boundary on the same
        streams as ``statsmodels``' ``rcusum`` does against its
        ``rcusumci`` -- every break stream, and the one stream with none of
        the five that both flag, a false alarm at the 5% level."""
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import recursive_olsresiduals

        flagged = []
        for seed, shift in (
            (31, 1.0),
            (34, 1.0),
            (32, 0.0),
            (33, 0.0),
            (35, 0.0),
            (36, 0.0),
            (37, 0.0),
        ):
            df = _break_rows(800, seed, shift, 400)
            z = self._run(df)["studentized_y"].to_numpy().astype(float)
            ok = np.isfinite(z)
            path, r = np.cumsum(np.where(ok, z, 0.0)), np.cumsum(ok)
            t = r[-1]
            ours = bool((ok & (np.abs(path) > 0.948 * (np.sqrt(t) + 2 * r / np.sqrt(t)))).any())
            X = sm.add_constant(df.select("x0", "x1").to_numpy())
            rr = recursive_olsresiduals(sm.OLS(df["y"].to_numpy(), X).fit(), skip=3)
            theirs = bool((np.abs(rr[5][1:]) > rr[6][1]).any())
            assert ours == theirs, (seed, ours, theirs)
            flagged.append(ours)
        assert flagged == [True, True, False, True, False, False, False]

    def test_the_wald_distance_is_two_wls_fits(self):
        import statsmodels.api as sm

        h = 60.0
        df = _break_rows(900, 41, 1.0, 600)
        out = self._run(df, half_life=h)
        x = df.select("x0", "x1").to_numpy()
        y = df["y"].to_numpy()
        X = sm.add_constant(x)
        checked = 0
        for t in (300, 650, 899):
            age = (t - 1) - np.arange(t)
            wf, ws = 0.5 ** (age / h), 0.5 ** (age / (4 * h))
            fast = sm.WLS(y[:t], X[:t], weights=wf).fit()
            slow = sm.WLS(y[:t], X[:t], weights=ws).fit()
            ns = ws.sum() ** 2 / (ws**2).sum()
            s2 = (ws * slow.resid**2).sum() / ws.sum() * ns / (ns - 3)
            d = fast.params - slow.params
            # The difference is `D ε`, `D = A_f X'W_f − A_s X'W_s`: its exact
            # variance over the noise's is `D D'` (task 232 (5)).
            Xt = X[:t]
            D = np.linalg.solve((Xt * wf[:, None]).T @ Xt, (Xt * wf[:, None]).T) - np.linalg.solve(
                (Xt * ws[:, None]).T @ Xt, (Xt * ws[:, None]).T
            )
            want = d @ np.linalg.solve(D @ D.T, d) / s2
            np.testing.assert_allclose(out["break_wald_y"][t], want, rtol=1e-6)
            checked += 1
        assert checked == 3
        assert out["break_wald_y"][650] > 21.1, "the break passes chi2(3)'s 0.01% value"


def _sandwich_rows(n: int, seed: int) -> pl.DataFrame:
    """Two features far from zero and a residual whose spread grows with the
    first: heteroskedastic, as HC0 is for."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([3.0, -1.0])
    e = rng.normal(size=n) * (0.5 + np.abs(x[:, 0] - 3.0))
    y = 0.5 + x @ np.array([1.0, -2.0]) + e
    w = rng.uniform(0.5, 1.5, size=n)
    w[rng.random(n) < 0.05] = 0.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})


class TestRobustStandardErrorsAreTheSandwich:
    """Task 221 (c). ``se_coef_hc0`` and ``se_coef_hac`` are the
    least-squares sandwich ``(X'ΩX)⁻¹ S (X'ΩX)⁻¹`` with ``S``
    ``statsmodels``' own HAC kernel, ``S_hac_simple`` at ``nlags = 0`` and
    at ``robust_se_lags``, over the scores ``ω e x`` of the bank's
    out-of-sample residuals: run once, and under a memory and weights, each
    row at its present weight. Against ``OLS.fit(cov_type="HC0" / "HAC")``,
    which reads the final fit's in-sample residuals, they agree to the
    estimation error the out-of-sample residuals carry."""

    @staticmethod
    def _run(df: pl.DataFrame, lags: int, **kw: Any) -> pl.DataFrame:
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            ridge=1e-12,
            min_weight=10.0,
            emit_robust_se=True,
            robust_se_lags=lags,
            coef_every=0,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("m")

    @pytest.mark.parametrize(("half_life", "weighted"), [(float("inf"), False), (150.0, True)])
    def test_the_sandwich_is_statsmodels_kernel_on_the_out_of_sample_residuals(
        self, half_life, weighted
    ):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import S_hac_simple

        lags = 4
        df = _sandwich_rows(900, 51)
        kw: dict[str, Any] = {"half_life": half_life}
        if weighted:
            kw["weight"] = "w"
        out = self._run(df if weighted else df.drop("w"), lags, **kw)
        resid = out["resid_y"].to_numpy()
        w = df["w"].to_numpy() if weighted else np.ones(df.height)
        X = sm.add_constant(df.select("x0", "x1").to_numpy())
        # Folded once the stream is 95% settled (task 232 (1)); run once it
        # never settles, and every row folds.
        settled = out["settled_frac"].fill_null(1.0).fill_nan(1.0).to_numpy()
        ready = settled >= 0.95
        checked = 0
        for t in (700, 800, 899):
            age = t - np.arange(t + 1)
            omega = w[: t + 1] * (0.5 ** (age / half_life))
            keep = np.isfinite(resid[: t + 1]) & (omega > 0) & ready[: t + 1]
            Xk, ek, om = X[: t + 1][keep], resid[: t + 1][keep], omega[keep]
            bi = np.linalg.inv((Xk * om[:, None]).T @ Xk)
            for nlags, field in ((0, "se_coef_hc0"), (lags, "se_coef_hac")):
                S = S_hac_simple(Xk * (om * ek)[:, None], nlags=nlags)
                want = np.sqrt(np.diag(bi @ S @ bi))
                np.testing.assert_allclose(out[field][t].to_numpy(), want, rtol=1e-8)
                checked += 1
        assert checked == 6

    def test_run_once_it_is_statsmodels_robust_fit_to_the_estimation_error(self):
        import statsmodels.api as sm

        df = _sandwich_rows(1500, 52).drop("w")
        out = self._run(df, 3, half_life=float("inf"))
        ok = np.isfinite(out["resid_y"].to_numpy())
        X = sm.add_constant(df.select("x0", "x1").to_numpy())[ok]
        y = df["y"].to_numpy()[ok]
        for cov, kw, field in (
            ("HC0", {}, "se_coef_hc0"),
            ("HAC", {"maxlags": 3}, "se_coef_hac"),
        ):
            fit = sm.OLS(y, X).fit(cov_type=cov, cov_kwds=kw or None)
            ratio = out[field][-1].to_numpy() / fit.bse
            # Out of sample: a little larger, by the estimation error.
            assert ((ratio > 1.0) & (ratio < 1.03)).all(), (cov, ratio)

    def test_overlapping_labels_put_back_about_the_square_root_of_the_horizon(self):
        """A target summing the next 10 rows' shocks against an AR(1)
        feature at 0.95, learned under ``embargo=10``: across 300 streams the
        coefficient's spread is about ``sqrt(10)`` times ``se_coef``, and
        ``se_coef_hac`` at its default lags, twice the horizon, reads it to
        within 15%; HC0, which assumes no correlation, does not."""
        h, n, g = 10, 3000, 300
        rng = np.random.default_rng(221)
        frames = []
        for gi in range(g):
            u = rng.normal(size=n + 200)
            x = np.zeros(n + 200)
            for t in range(1, n + 200):
                x[t] = 0.95 * x[t - 1] + np.sqrt(1 - 0.95**2) * u[t]
            eps = rng.normal(size=n + 200 + h)
            noise = np.convolve(eps, np.ones(h), "valid")[1 : n + 201]
            frames.append(
                pl.DataFrame({"g": np.full(n, gi), "x0": x[200:], "y": 0.2 * x[200:] + noise[200:]})
            )
        df = pl.concat(frames)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            half_life=float("inf"),
            group="g",
            embargo=float(h),
            min_weight=20.0,
            emit_robust_se=True,
            emit_se_coef=True,
            emit_sigma=True,
            coef_every=0,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        last = out.group_by("g", maintain_order=True).last()
        slope = np.array([c[1] for c in last["coef"].to_list()])
        se = np.array([c[1] for c in last["se_coef"].to_list()]).mean()
        hc0 = np.array([c[1] for c in last["se_coef_hc0"].to_list()]).mean()
        hac = np.array([c[1] for c in last["se_coef_hac"].to_list()]).mean()
        spread = slope.std(ddof=1)
        assert 0.6 * np.sqrt(h) < spread / se < 1.2 * np.sqrt(h), spread / se
        assert abs(hac / spread - 1.0) < 0.15, hac / spread
        assert hc0 / spread < 0.5, hc0 / spread


def _spec_rows(
    n: int, seed: int, curve: float = 0.0, lookahead: int = 0, hetero: bool = True
) -> pl.DataFrame:
    """Two features; noise whose spread grows with the first; a square of
    it the fit may miss; and, with ``lookahead``, a target summing that many
    rows' shocks."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    u = rng.normal(size=n + max(lookahead, 1))
    e = np.convolve(u, np.ones(lookahead), "valid")[1 : n + 1] if lookahead else u[:n]
    if hetero:
        e = e * np.maximum(0.2, 1.0 + 0.6 * x[:, 0])
    y = 0.5 + x @ np.array([1.0, -1.0]) + curve * x[:, 0] ** 2 + e
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


class TestSpecificationTestsAreStatsmodels:
    """Task 221 (d). Run once, on the bank's out-of-sample residuals of the
    rows before each row: ``ljung_box`` is ``acorr_ljungbox``,
    ``breusch_pagan`` is ``het_breuschpagan``'s ``n R2``, and ``reset`` is
    ``compare_lm_test`` of the residual on ``p, p2, p3`` against ``p``.
    Past a horizon ``ljung_box`` is the definition with Bartlett's
    covariance, from ``statsmodels``' ``acf``; on planted curvature it
    decides as ``linear_reset`` does on the in-sample fit."""

    @staticmethod
    def _run(df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=10.0,
            emit_specification=True,
            ljung_box_lags=6,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("m")

    def test_the_three_statistics_are_statsmodels(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

        df = _spec_rows(700, 61, curve=0.2)
        out = self._run(df)
        r, p = out["resid_y"].to_numpy(), out["pred_y"].to_numpy()
        x = df.select("x0", "x1").to_numpy()
        for t in (150, 699):
            ok = np.isfinite(r[:t])
            rt, pt = r[:t][ok], p[:t][ok]
            lb = acorr_ljungbox(rt, lags=[6])["lb_stat"].iloc[0]
            np.testing.assert_allclose(out["ljung_box_y"][t], lb, rtol=1e-9)
            bp = het_breuschpagan(rt, sm.add_constant(x[:t][ok]))[0]
            np.testing.assert_allclose(out["breusch_pagan_y"][t], bp, rtol=1e-8)
            q = pt - pt[0]
            full = sm.OLS(rt, sm.add_constant(np.column_stack([q, q**2, q**3]))).fit()
            base = sm.OLS(rt, sm.add_constant(q)).fit()
            np.testing.assert_allclose(out["reset_y"][t], full.compare_lm_test(base)[0], rtol=1e-7)

    def test_past_a_horizon_it_reads_bartletts_covariance(self):
        from statsmodels.tsa.stattools import acf

        h = 4
        df = _spec_rows(900, 62, lookahead=h)
        out = self._run(df, embargo=float(h))
        r = out["resid_y"].to_numpy()
        t = 899
        # Under the embargo, row t has learned the rows up to t - h.
        released = r[: t - h + 1]
        rt = released[np.isfinite(released)]
        n = len(rt)
        rho = acf(rt, nlags=h - 1 + 6, fft=False)
        lags = np.arange(h, h + 6)
        rr = rho[lags] * np.sqrt((n + 2) / (n - lags))
        s = h - 1

        def at(j: int) -> float:
            return float(rho[abs(j)]) if abs(j) <= s else 0.0

        def gamma(v: int) -> float:
            return sum(at(j) * at(j + v) for j in range(-s, s + 1))

        c = np.array([[gamma(abs(a - b)) for b in range(6)] for a in range(6)])
        want = n * rr @ np.linalg.solve(c, rr)
        np.testing.assert_allclose(out["ljung_box_y"][t], want, rtol=1e-8)

    def test_reset_decides_as_linear_reset(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import linear_reset

        for seed, curve in ((63, 0.4), (64, 0.0), (65, 0.4), (66, 0.0)):
            df = _spec_rows(1500, seed, curve=curve, hetero=False)
            ours = self._run(df)["reset_y"][-1] > 5.991
            X = sm.add_constant(df.select("x0", "x1").to_numpy())
            fit = sm.OLS(df["y"].to_numpy(), X).fit()
            theirs = linear_reset(fit, power=3, test_type="fitted", use_f=False).statistic > 5.991
            assert ours == theirs == (curve > 0), (seed, ours, theirs)


class TestUnderAHorizonTheTestsAreNeweyWest:
    """Task 232 (3). Under a horizon of ``h`` rows the calibration's,
    Breusch and Pagan's and RESET's tests are Wald's with Newey and West's
    variance at ``2h`` lags: ``d' V⁻¹ d`` with ``V = B⁻¹ S B⁻¹``, ``B`` the
    regression's ``Z'Z`` and ``S`` ``statsmodels``' own HAC kernel,
    ``S_hac_simple``, over the scores ``z r`` of the residual under each
    null -- ``y − pred``; ``e²`` less its mean over the rows before; ``e``
    less its least-squares fit on ``p`` over the rows before. Run once on a
    three-row look-ahead target, each row reading the rows released before
    it, at the prediction each was scored with."""

    def test_each_is_wald_with_statsmodels_hac_kernel(self):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import S_hac_simple

        h, n = 3, 600
        rng = np.random.default_rng(232)
        x = rng.normal(size=(n, 2))
        u = rng.normal(size=n + h)
        fwd = np.convolve(u, np.ones(h), "valid")[1 : n + 1] / np.sqrt(h)
        y = 0.4 + x @ np.array([1.2, -0.8]) + fwd * (1.0 + 0.3 * np.abs(x[:, 0]))
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            embargo=float(h),
            min_weight=10.0,
            emit_calibration=True,
            emit_specification=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        p, e = out["pred_y"].to_numpy(), out["resid_y"].to_numpy()
        L = 2 * h

        def wald(Z, r, coef, pick):
            bi = np.linalg.inv(Z.T @ Z)
            V = bi @ S_hac_simple(Z * r[:, None], nlags=L) @ bi
            sub = V[np.ix_(pick, pick)]
            return float(coef @ np.linalg.solve(sub, coef))

        checked = 0
        for t in (200, 400, 599):
            # Row t reads the rows released before it: up to t - h.
            keep = np.isfinite(p[: t - h + 1])
            ps, es = p[: t - h + 1][keep], e[: t - h + 1][keep]
            xs, ys = x[: t - h + 1][keep], y[: t - h + 1][keep]
            # The calibration: y on (1, p), against a = 0, b = 1.
            Z = sm.add_constant(ps)
            fit = sm.OLS(ys, Z).fit()
            want = wald(Z, ys - ps, fit.params - [0.0, 1.0], [0, 1])
            np.testing.assert_allclose(out["calibration_wald_y"][t], want, rtol=1e-7)
            # Breusch and Pagan: e² on (1, x), its slopes against 0, the meat
            # from e² less its mean over the rows before each.
            u2 = es**2
            before = np.concatenate([[np.nan], np.cumsum(u2)[:-1] / np.arange(1, len(u2))])
            r = np.where(np.isfinite(before), u2 - before, 0.0)
            Zx = sm.add_constant(xs)
            gamma = sm.OLS(u2, Zx).fit().params[1:]
            want = wald(Zx, r, gamma, [1, 2])
            np.testing.assert_allclose(out["breusch_pagan_y"][t], want, rtol=1e-7)
            # RESET: e on (1, q, q², q³), q about the first prediction; the
            # coefficients of q² and q³ against 0, the meat from e less its
            # least-squares fit on q over the rows before each.
            q = ps - ps[0]
            r = np.zeros(len(q))
            for i in range(1, len(q)):
                qb, eb = q[:i], es[:i]
                vq = qb.var()
                b = ((qb - qb.mean()) * (eb - eb.mean())).mean() / vq if vq > 0 else 0.0
                r[i] = es[i] - eb.mean() - b * (q[i] - qb.mean())
            Zq = np.column_stack([np.ones_like(q), q, q**2, q**3])
            beta = sm.OLS(es, Zq).fit().params
            want = wald(Zq, r, beta[2:], [2, 3])
            np.testing.assert_allclose(out["reset_y"][t], want, rtol=1e-6)
            checked += 1
        assert checked == 3


class TestTailsAreScipyAndJarqueBera:
    """Task 221 (e). Run once, the tails of the recursive residuals before
    each row, ``resid / error_inflation``, are ``scipy.stats``' biased
    ``skew`` and ``kurtosis`` and ``statsmodels``' ``jarque_bera``; with a
    memory and weights, the same definitions at each row's present weight
    and Kish's size, written by hand from ``numpy.average``."""

    def test_run_once_and_windowed(self):
        from scipy import stats
        from statsmodels.stats.stattools import jarque_bera

        rng = np.random.default_rng(71)
        n = 1200
        x = rng.normal(size=(n, 2))
        w = rng.uniform(0.5, 1.5, size=n)
        w[rng.random(n) < 0.05] = 0.0
        y = x @ np.array([1.0, -1.0]) + rng.standard_t(5, size=n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})
        common = dict(
            targets=["y"],
            features=["x0", "x1"],
            ridge=1e-12,
            min_weight=3.0,
            emit_tails=True,
            emit_error_inflation=True,
        )
        once = po.ModelBank([po.spec.ewridge("m", half_life=float("inf"), **common)])
        out = once.fit_predict(df.drop("w")).unnest("m")
        v = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        for t in (200, 1199):
            vt = v[:t][np.isfinite(v[:t])]
            np.testing.assert_allclose(out["skew_y"][t], stats.skew(vt), rtol=1e-9)
            np.testing.assert_allclose(out["kurtosis_y"][t], stats.kurtosis(vt), rtol=1e-9)
            np.testing.assert_allclose(out["jarque_bera_y"][t], jarque_bera(vt)[0], rtol=1e-9)
        h = 150.0
        spec = po.spec.ewridge("m", half_life=60.0, tails_half_life=h, weight="w", **common)
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        v = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        # Folded once the stream is 95% settled on the model's half-life
        # (task 232 (1)).
        ready = out["settled_frac"].to_numpy() >= 0.95
        for t in (400, 1199):
            om = w[:t] * 0.5 ** (((t - 1) - np.arange(t)) / h)
            keep = np.isfinite(v[:t]) & (om > 0) & ready[:t]
            vt, ot = v[:t][keep], om[keep]
            mu = np.average(vt, weights=ot)
            m = [np.average((vt - mu) ** j, weights=ot) for j in (2, 3, 4)]
            skew, kurt = m[1] / m[0] ** 1.5, m[2] / m[0] ** 2 - 3
            n_kish = ot.sum() ** 2 / (ot**2).sum()
            np.testing.assert_allclose(out["skew_y"][t], skew, rtol=1e-8)
            np.testing.assert_allclose(out["kurtosis_y"][t], kurt, rtol=1e-8)
            np.testing.assert_allclose(
                out["jarque_bera_y"][t], n_kish / 6 * (skew**2 + kurt**2 / 4), rtol=1e-8
            )

    def test_under_an_embargo_each_row_keeps_its_scored_inflation(self):
        """Under an embargo of ``E`` rows, row ``t`` reads the rows up to
        ``t - E``, each as it was scored: its residual over the error
        inflation of the fit that scored it, both as the row's own fields
        show them (review round 6, A-6)."""
        from scipy import stats

        rng = np.random.default_rng(4)
        n, embargo = 400, 7
        x = rng.normal(size=(n, 2))
        y = x @ np.array([1.0, -0.5]) + rng.standard_t(5, size=n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            embargo=float(embargo),
            min_weight=5.0,
            emit_tails=True,
            emit_error_inflation=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        v = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
        for t in (60, 61, 200, 399):
            released = v[: t - embargo + 1]
            vt = released[np.isfinite(released)]
            np.testing.assert_allclose(out["skew_y"][t], stats.skew(vt), rtol=1e-9)
            np.testing.assert_allclose(out["kurtosis_y"][t], stats.kurtosis(vt), rtol=1e-9)


class TestInfluenceIsDffitsAtTheNewestRow:
    """Task 221 (f). Run once with no ridge, ``influence`` on row ``t`` is
    ``statsmodels``' ``OLSInfluence(...).dffits`` of the fit over rows
    ``0..=t`` at its last row: that row's in-sample leverage is ``h/(1+h)``
    and its externally studentized residual the recursive residual over
    the recursive residuals' mean square before it, which is the fit
    without it's ``RSS/(n-k)``. Where the definitions part -- an earlier
    row's in-sample DFFITS reads the rows after it -- nothing is compared."""

    def test_each_rows_influence_is_its_prefixs_last_dffits(self):
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence

        rng = np.random.default_rng(81)
        n = 300
        x = rng.normal(size=(n, 2)) + np.array([4.0, -2.0])
        x[::37, 0] += 5.0  # rows of high leverage
        y = 0.5 + x @ np.array([1.0, -1.0]) + rng.normal(size=n)
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            ridge=1e-12,
            min_weight=3.0,
            emit_influence=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        X = sm.add_constant(x)
        checked = 0
        for t in range(8, n, 5):
            want = OLSInfluence(sm.OLS(y[: t + 1], X[: t + 1]).fit()).dffits[0][t]
            np.testing.assert_allclose(out["influence_y"][t], want, rtol=1e-8)
            checked += 1
        assert checked == 59


class TestFeatureHealthIsTwoWeightedMoments:
    """Task 221 (g). ``spread_ratio_<f>`` and ``mean_shift_<f>`` are
    ``statsmodels``' ``DescrStatsW`` weighted mean and standard deviation
    (``ddof=0``) of the rows before each row, at the memory and at four
    times it, each row at its weight times its decay."""

    def test_the_ratio_and_the_shift_are_descrstatsw(self):
        from statsmodels.stats.weightstats import DescrStatsW

        rng = np.random.default_rng(91)
        n = 900
        x = rng.normal(size=(n, 2)) + np.array([50.0, -3.0])
        x[500:, 1] += 2.0
        w = rng.uniform(0.5, 1.5, size=n)
        w[rng.random(n) < 0.05] = 0.0
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": x[:, 0] + rng.normal(size=n), "w": w})
        h = 60.0
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=h,
            weight="w",
            emit_feature_health=True,
        )
        out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        for t in (100, 560, 899):
            age = (t - 1) - np.arange(t)
            fast = DescrStatsW(x[:t], weights=w[:t] * 0.5 ** (age / h), ddof=0)
            slow = DescrStatsW(x[:t], weights=w[:t] * 0.5 ** (age / (4 * h)), ddof=0)
            for i, f in enumerate(("x0", "x1")):
                np.testing.assert_allclose(
                    out[f"spread_ratio_{f}"][t], fast.std[i] / slow.std[i], rtol=1e-9
                )
                np.testing.assert_allclose(
                    out[f"mean_shift_{f}"][t],
                    (fast.mean[i] - slow.mean[i]) / slow.std[i],
                    rtol=1e-7,
                    atol=1e-10,
                )


def _nested_out(n: int, seed: int, beta2: float, groups: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), n)
    x = rng.normal(size=(groups * n, 2))
    y = x[:, 0] + beta2 * x[:, 1] + rng.normal(size=groups * n)
    df = pl.DataFrame({"g": g, "x0": x[:, 0], "x1": x[:, 1], "y": y})
    common = dict(targets=["y"], half_life=float("inf"), min_weight=20.0, group="g")
    small = po.spec.ewridge("small", features=["x0"], **common)
    big = po.spec.ewridge("big", features=["x0", "x1"], **common)
    return po.ModelBank([small, big]).fit_predict(df)


class TestNestedComparisonsAreTheirPapers:
    """Task 221 (h). ``po.eval.diebold_mariano`` is Diebold and Mariano's
    (1995) statistic, the mean squared-error differential over its standard
    error, and ``po.eval.clark_west`` Clark and West's (2007) adjusted
    differential, each written here from the paper with Newey and West's
    variance from ``statsmodels``' ``S_hac_simple``; the ``half_life`` form
    is the same at each row's present weight, the scores ``ω (d - m)``."""

    @staticmethod
    def _hac(v: np.ndarray, lags: int, weights: np.ndarray | None = None) -> float:
        from statsmodels.stats.sandwich_covariance import S_hac_simple

        w = np.ones_like(v) if weights is None else weights
        m = (w * v).sum() / w.sum()
        s = float(np.asarray(S_hac_simple(w * (v - m), nlags=lags)).ravel()[0])
        return m * w.sum() / np.sqrt(s)

    def test_the_statistics_are_their_definitions(self):
        out = _nested_out(1500, 101, 0.05, groups=2)
        for lags in (0, 3):
            dm = po.eval.diebold_mariano(out, a="big", b="small", lags=lags, group="g")
            cw = po.eval.clark_west(out, big="big", small="small", lags=lags, group="g")
            for gi in range(2):
                part = out.filter(pl.col("g") == gi)
                ra = part["big"].struct.field("resid_y").to_numpy()
                rb = part["small"].struct.field("resid_y").to_numpy()
                ok = np.isfinite(ra) & np.isfinite(rb)
                ra, rb = ra[ok], rb[ok]
                want_dm = self._hac(rb**2 - ra**2, lags)
                want_cw = self._hac(rb**2 - ra**2 + (ra - rb) ** 2, lags)
                got = dm.filter(pl.col("g") == gi)
                np.testing.assert_allclose(got["dm"][0], want_dm, rtol=1e-9)
                assert got["n"][0] == ok.sum()
                np.testing.assert_allclose(
                    cw.filter(pl.col("g") == gi)["cw"][0], want_cw, rtol=1e-9
                )

    def test_the_exponentially_weighted_form_at_each_row(self):
        h, lags = 250.0, 2
        out = _nested_out(1200, 102, 0.1)
        ew = po.eval.clark_west(out, big="big", small="small", lags=lags, half_life=h)
        got = ew["clark_west"].struct.field("cw_y").to_numpy()
        ra = out["big"].struct.field("resid_y").to_numpy()
        rb = out["small"].struct.field("resid_y").to_numpy()
        ok = np.isfinite(ra) & np.isfinite(rb)
        f = (rb**2 - ra**2 + (ra - rb) ** 2)[ok]
        rows = np.flatnonzero(ok)
        lam = 0.5 ** (1 / h)
        for j in (50, 400, len(f) - 1):
            om = lam ** (j - np.arange(j + 1))
            want = self._hac(f[: j + 1], lags, om)
            np.testing.assert_allclose(got[rows[j]], want, rtol=1e-7)
        assert np.isnan(got[~ok]).all() or all(v is None for v in got[~ok])

    def test_clark_west_finds_what_diebold_mariano_leans_away_from(self):
        """A larger model whose extra coefficient is 0.1: Clark and West
        reject the smaller one at 5% on every one of 20 streams, Diebold and
        Mariano on fewer; with the coefficient 0, Clark and West on few."""
        alt = _nested_out(3000, 103, 0.1, groups=20)
        cw = po.eval.clark_west(alt, big="big", small="small", group="g")["cw"].to_numpy()
        dm = po.eval.diebold_mariano(alt, a="big", b="small", group="g")["dm"].to_numpy()
        assert (cw > 1.645).mean() == 1.0
        assert (dm > 1.645).mean() < (cw > 1.645).mean()
        null = _nested_out(3000, 104, 0.0, groups=20)
        cw0 = po.eval.clark_west(null, big="big", small="small", group="g")["cw"].to_numpy()
        assert (cw0 > 1.645).mean() <= 0.15

    def test_refusals(self):
        out = _nested_out(200, 105, 0.0)
        with pytest.raises(ValueError, match="lags must be >= 0"):
            po.eval.diebold_mariano(out, a="big", b="small", lags=-1)
        with pytest.raises(ValueError, match="half_life must be > 0"):
            po.eval.clark_west(out, big="big", small="small", half_life=0.0)
        with pytest.raises(ValueError, match="names no residual"):
            po.eval.diebold_mariano(out, a="big", b="small", targets=["nope"])
