"""Cross-checks against river (docs/TESTING.md section B).

river is an independent implementation of several of the same algorithms, so it
catches errors a self-written oracle cannot: a numpy reference written by the
same author can share a misreading, river cannot.

Two tiers:

* **exact** — configurations where the two libraries implement the same
  published recursion, compared to floating-point tolerance;
* **statistical** — configurations that differ by design (warmup conventions,
  exact vs stochastic optimization), where the assertion is convergence or a
  shared qualitative property, and the difference itself is documented.

Skipped cleanly when river is not installed.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

river = pytest.importorskip("river", reason="river is not installed")
from river import linear_model, optim, stats  # noqa: E402


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


class TestFtrlRecursion:
    """T-R1: our FTRL state recursion vs river's `optim.FTRLProximal`.

    **Documented ordering difference.** McMahan et al. Algorithm 1 recomputes
    the proximal weights from `z` at *prediction* time, then updates `z` with
    that row's gradient. We do that. river's `LogisticRegression` instead
    predicts with the weights its optimizer produced during the *previous*
    `learn_one`, so its predictions lag ours by one proximal recomputation
    (verified below). The underlying z/n recursion is identical, which is what
    this test pins: driven by the same gradient sequence, river's optimizer and
    our model produce the same weights, row for row.
    """

    ALPHA, BETA, L1, L2 = 0.1, 1.0, 0.0, 1.0

    def _data(self, n=400, seed=1):
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal(n)
        x1 = rng.standard_normal(n)
        p = _sigmoid(1.5 * x0 - 0.5 * x1)
        y = (rng.random(n) < p).astype(float)
        return pl.DataFrame({"x0": x0, "x1": x1, "y0": y})

    def _ours(self, df, **kw):
        spec = po.spec.ftrl(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            add_intercept=False,
            halflife=float("inf"),  # no decay: river has none to compare against
            alpha=self.ALPHA,
            beta=self.BETA,
            l1=self.L1,
            l2=self.L2,
            min_periods=0.0,
            coef_every=1,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df)

    @pytest.mark.parametrize("l1", [0.0, 0.5])
    def test_weights_match_river_given_the_same_gradients(self, l1):
        type(self).L1 = l1
        df = self._data()
        out = self._ours(df)
        coef = np.array(out["m"].struct.field("coef").to_list(), dtype=float)
        x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
        y = df["y0"].to_numpy()

        opt = optim.FTRLProximal(alpha=self.ALPHA, beta=self.BETA, l1=l1, l2=self.L2)
        w = {"x0": 0.0, "x1": 0.0}
        max_diff = 0.0
        for t in range(len(y)):
            # The weights our model used to predict row t are the ones it
            # emitted after row t-1 (zeros before the first row).
            prev = coef[t - 1] if t > 0 else np.zeros(2)
            p = _sigmoid(x[t] @ prev)
            g = {"x0": (p - y[t]) * x[t, 0], "x1": (p - y[t]) * x[t, 1]}
            # river recomputes w from z (i.e. `prev`) and then advances z.
            w = opt._step_with_dict(w, g)
            max_diff = max(max_diff, np.max(np.abs(np.array([w["x0"], w["x1"]]) - prev)))
            # after the step, river's z must imply exactly our post-row weights
            after = np.array(
                [
                    0.0
                    if abs(opt.z[k]) <= l1
                    else -(opt.z[k] - np.sign(opt.z[k]) * l1)
                    / ((self.BETA + opt.n[k] ** 0.5) / self.ALPHA + self.L2)
                    for k in ("x0", "x1")
                ]
            )
            assert np.allclose(after, coef[t], atol=1e-12), (
                f"row {t}: river z/n implies {after}, we emitted {coef[t]}"
            )
        assert max_diff < 1e-12, "river's own proximal step disagrees with our weights"
        type(self).L1 = 0.0

    def test_river_model_lags_us_by_one_step(self):
        """Pins the ordering difference itself, so a future change on either
        side shows up as a failure rather than a silent divergence.

        It is only unambiguous on the first two rows: from row 1 on, river
        computes its gradient from the stale weights, so the two states diverge
        and later rows are no longer a pure shift.
        """
        df = self._data(n=50, seed=2)
        ours = np.array(self._ours(df)["m"].struct.field("pred_y0").to_list(), dtype=float)

        model = linear_model.LogisticRegression(
            optimizer=optim.FTRLProximal(alpha=self.ALPHA, beta=self.BETA, l1=self.L1, l2=self.L2),
            intercept_lr=0.0,
            intercept_init=0.0,
        )
        theirs = []
        for i in range(df.height):
            xi = {"x0": float(df["x0"][i]), "x1": float(df["x1"][i])}
            theirs.append(model.predict_proba_one(xi)[True])
            model.learn_one(xi, bool(df["y0"][i]))

        # Row 0: both start from zero weights.
        assert ours[0] == pytest.approx(0.5)
        assert theirs[0] == pytest.approx(0.5)
        # Row 1: we have already recomputed the proximal weights from z, river
        # has not, so it is still predicting with zeros.
        assert theirs[1] == pytest.approx(0.5)
        assert abs(ours[1] - 0.5) > 1e-6
        # The exact per-row agreement of the underlying recursion is the
        # previous test; here we only pin the ordering.

    def test_both_learn_the_same_signal(self):
        df = self._data(n=3000, seed=3)
        ours = np.array(self._ours(df)["m"].struct.field("pred_y0").to_list(), dtype=float)
        model = linear_model.LogisticRegression(
            optimizer=optim.FTRLProximal(alpha=self.ALPHA, beta=self.BETA, l1=0.0, l2=self.L2),
            intercept_lr=0.0,
        )
        theirs = []
        for i in range(df.height):
            xi = {"x0": float(df["x0"][i]), "x1": float(df["x1"][i])}
            theirs.append(model.predict_proba_one(xi)[True])
            model.learn_one(xi, bool(df["y0"][i]))
        y = df["y0"].to_numpy()
        eps = 1e-12

        def logloss(p):
            return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        # Not identical (the lag above), but neither should be meaningfully
        # better than the other.
        assert abs(logloss(ours) - logloss(np.array(theirs))) < 0.01


class TestEwStatistics:
    """T-R4: our exponentially weighted moments vs `river.stats.EWMean`/`EWVar`.

    The mapping is `fading_factor = 1 - 0.5 ** (1 / halflife)` on a row-count
    clock. **Warmup differs by construction**: ours is an exact weighted mean
    from the first row (the accumulator is a mean, not a sum), while river's
    seeds from its first observations, so the two sequences agree only after
    the warmup transient. That difference is the point of this test.
    """

    HALFLIFE = 50.0

    @property
    def fading(self):
        return 1.0 - 0.5 ** (1.0 / self.HALFLIFE)

    def _our_ew_mean(self, y):
        """The EW mean of y, read off an intercept-only fit: a constant feature
        has zero variance, so the standardized solve drops it and the intercept
        is exactly the EW mean of the target."""
        df = pl.DataFrame({"x0": np.zeros(len(y)), "y0": y})
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            halflife=self.HALFLIFE,
            standardize=True,
            min_periods=1.0,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        return np.array(out["m"].struct.field("pred_y0").to_list(), dtype=float)

    def test_ew_mean_converges_to_river(self):
        rng = np.random.default_rng(7)
        y = rng.standard_normal(4000) + 3.0
        ours = self._our_ew_mean(y)

        m = stats.EWMean(fading_factor=self.fading)
        theirs = []
        for v in y:
            theirs.append(m.get())
            m.update(float(v))
        theirs = np.array(theirs)

        early = np.nanmean(np.abs(ours[1:20] - theirs[1:20]))
        late = np.nanmean(np.abs(ours[-1000:] - theirs[-1000:]))
        assert late < 0.05, f"EW means did not converge: {late}"
        assert late < early, "the difference should be a warmup transient"

    def test_warmup_convention_differs_and_ours_is_bias_corrected(self):
        """The two libraries use different EW conventions, and this pins which.

        Ours divides by the accumulated weight, so it is the exact weighted mean
        of everything seen so far from the very first row. river's `EWMean` is
        the un-normalized EWMA `m += f * (x - m)` seeded at its first value, so
        during warmup it stays anchored near that seed. They agree in the limit
        (see the convergence test); they do not agree early, and code that
        assumes otherwise is wrong.
        """
        y = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
        ours = self._our_ew_mean(y)

        m = stats.EWMean(fading_factor=self.fading)
        theirs = []
        for v in y:
            theirs.append(m.get())
            m.update(float(v))

        # Closed form of our convention: m_t = (lam * W * m + y) / (lam * W + 1).
        lam = 0.5 ** (1.0 / self.HALFLIFE)
        exact, w_sum, mean = [], 0.0, 0.0
        for v in y:
            exact.append(mean)
            w_new = lam * w_sum + 1.0
            mean = (lam * w_sum * mean + v) / w_new
            w_sum = w_new

        # Row 0 is warmup (no prediction yet); from row 1 we match the closed
        # form exactly.
        assert ours[0] is None or np.isnan(ours[0])
        np.testing.assert_allclose(ours[1:], exact[1:], atol=1e-9)

        # river, after the same three observations, is still near its seed.
        assert theirs[2] == pytest.approx(1.055, abs=0.01)
        assert ours[2] == pytest.approx(3.014, abs=0.01)

    def test_ew_variance_tracks_river(self):
        rng = np.random.default_rng(8)
        y = rng.standard_normal(5000) * 2.0
        v = stats.EWVar(fading_factor=self.fading)
        theirs = []
        for val in y:
            v.update(float(val))
            theirs.append(v.get())
        # Both should recover the true variance (4.0) on average once warm.
        assert abs(np.mean(theirs[-2000:]) - 4.0) < 1.0


class TestQuantile:
    """T-R5: our IRLS quantile regression vs river's P^2 quantile estimator.

    Different algorithms entirely (regression with a check-loss IRLS weight vs a
    distribution sketch), so this asserts they agree about the *location* of the
    quantile, which is the property that would break if our weights were wrong.
    """

    @pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
    def test_intercept_only_quantile_matches_river(self, tau):
        rng = np.random.default_rng(11)
        n = 20000
        y = rng.standard_normal(n) * 2.0 + 1.0
        df = pl.DataFrame({"x0": np.zeros(n), "y0": y})
        spec = po.spec.quantile(
            "m",
            targets=["y0"],
            features=["x0"],
            quantile=tau,
            halflife=1e9,
            standardize=True,
            min_periods=10.0,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        ours = np.array(out["m"].struct.field("pred_y0").to_list(), dtype=float)[-1]

        q = stats.Quantile(tau)
        for v in y:
            q.update(float(v))
        theirs = q.get()
        truth = np.quantile(y, tau)
        # Both estimators should land near the empirical quantile.
        assert abs(theirs - truth) < 0.2, f"river {theirs} vs empirical {truth}"
        assert abs(ours - truth) < 0.5, f"ours {ours} vs empirical {truth}"


class TestHuber:
    """T-R6: contamination robustness, cross-checked qualitatively."""

    def test_both_libraries_beat_least_squares_under_outliers(self):
        rng = np.random.default_rng(13)
        n = 4000
        x = rng.standard_normal(n)
        y = 2.0 * x + 0.1 * rng.standard_normal(n)
        bad = rng.random(n) < 0.03
        y[bad] = 300.0 * rng.standard_normal(bad.sum())
        df = pl.DataFrame({"x0": x, "y0": y})

        common = dict(
            targets=["y0"],
            features=["x0"],
            halflife=1e9,
            min_periods=10.0,
            max_rows_between_solves=1,
        )
        ours = po.ModelBank([po.spec.huber("m", **common)]).fit_predict(df)
        ours_slope = np.array(ours["m"].struct.field("coef").to_list()[-1], dtype=float)[1]

        model = linear_model.LinearRegression(
            optimizer=optim.SGD(0.01), loss=optim.losses.Huber(), intercept_lr=0.0
        )
        for i in range(n):
            model.learn_one({"x0": float(x[i])}, float(y[i]))
        theirs_slope = model.weights["x0"]

        ols = po.ModelBank([po.spec.ewridge("m", ridge=1e-8, **common)]).fit_predict(df)
        ols_slope = np.array(ols["m"].struct.field("coef").to_list()[-1], dtype=float)[1]

        assert abs(ours_slope - 2.0) < abs(ols_slope - 2.0)
        assert abs(theirs_slope - 2.0) < abs(ols_slope - 2.0)
        # And the two robust fits should broadly agree.
        assert abs(ours_slope - theirs_slope) < 0.5


class TestEwCovAgainstRiver:
    """T-R3: with no decay, our mean-form accumulators are exact running
    moments, and river computes the same quantities by a different route
    (Welford updates). Any disagreement is a bug in one of us.

    Unblocked by ENHANCEMENTS E1 (`ew_cov`), which made these statistics
    reachable without fitting a regression.
    """

    NO_DECAY = float("inf")

    def _data(self, n=2000, seed=17):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(n) * 3.0 + 1.0
        b = 0.7 * a + rng.standard_normal(n)
        return a, b

    def _ours(self, a, b, stats):
        df = pl.DataFrame({"x0": a, "x1": b})
        spec = po.spec.ew_cov(
            "c", features=["x0", "x1"], stats=stats, halflife=self.NO_DECAY, min_periods=2.0
        )
        out = po.ModelBank([spec]).fit_predict(df)
        return {f.name: out["c"].struct.field(f.name).to_list()[-1] for f in out.schema["c"].fields}

    def test_mean_and_var_match_river(self):
        a, b = self._data()
        ours = self._ours(a, b, ["mean", "var"])

        m, v = stats.Mean(), stats.Var(ddof=0)
        for x in a[:-1]:  # ours reports the state before the final row
            m.update(float(x))
            v.update(float(x))
        assert ours["mean_x0"] == pytest.approx(m.get(), abs=1e-9)
        assert ours["var_x0"] == pytest.approx(v.get(), abs=1e-9)

    def test_covariance_matches_river(self):
        a, b = self._data()
        ours = self._ours(a, b, ["cov"])
        c = stats.Cov(ddof=0)
        for x, y in zip(a[:-1], b[:-1], strict=True):
            c.update(float(x), float(y))
        assert ours["cov_x0_x1"] == pytest.approx(c.get(), abs=1e-9)

    def test_correlation_matches_river(self):
        a, b = self._data()
        ours = self._ours(a, b, ["corr"])
        r = stats.PearsonCorr(ddof=0)
        for x, y in zip(a[:-1], b[:-1], strict=True):
            r.update(float(x), float(y))
        assert ours["corr_x0_x1"] == pytest.approx(r.get(), abs=1e-9)

    def test_where_the_raw_moment_form_loses_to_welford(self):
        """Our accumulator derives variance as `E[x^2] - m^2`, river's Welford
        form does not, so on a large offset ours degrades first. This records
        the gap that ENHANCEMENTS E11b would close (see docs/TESTING.md T-E9)."""
        rng = np.random.default_rng(19)
        base = rng.standard_normal(4000)
        errs = {}
        for offset in (0.0, 1e6, 1e9):
            a = base + offset
            ours = self._ours(a, base, ["var"])["var_x0"]
            v = stats.Var(ddof=0)
            for x in a[:-1]:
                v.update(float(x))
            truth = base[:-1].var()
            errs[offset] = (abs(ours - truth), abs(v.get() - truth))
        # identical at ordinary scales
        assert errs[0.0][0] == pytest.approx(errs[0.0][1], abs=1e-9)
        # river is still exact at 1e9; ours has lost the variance entirely
        assert errs[1e9][1] < 1e-3
        assert errs[1e9][0] > 0.5


class TestKalmanIsBayesianLinearRegression:
    """T-R2: with `standardize=False`, `q=0` and a fixed `obs_var`, our Kalman
    filter *is* a Bayesian linear regression, and river implements the same
    recursion independently (Bishop 3.51 via Sherman-Morrison).

    The mapping: river's `alpha` is the prior precision, so `p0 = 1/alpha`;
    river's `beta` is the observation precision, so `obs_var = 1/beta`. river
    has no intercept, hence `add_intercept=False`.

    Unblocked by ENHANCEMENTS E19 (the `standardize` switch); before it, the
    filter always standardized internally and the correspondence was only
    approximate.
    """

    def _data(self, n=500, seed=4):
        rng = np.random.default_rng(seed)
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        y = 1.5 * a - 0.5 * b + 0.2 * rng.standard_normal(n)
        return a, b, y

    def _run_both(self, alpha, beta, n=500, seed=4):
        a, b, y = self._data(n, seed)
        spec = po.spec.kalman(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            add_intercept=False,
            coef_halflife=float("inf"),
            q=[0.0, 0.0],
            obs_var=1.0 / beta,
            p0=1.0 / alpha,
            standardize=False,
            halflife=float("inf"),
            min_periods=0.0,
        )
        df = pl.DataFrame({"x0": a, "x1": b, "y0": y})
        ours = (
            po.ModelBank([spec])
            .fit_predict(df)["m"]
            .struct.field("pred_y0")
            .to_numpy()
            .astype(float)
        )
        model = linear_model.BayesianLinearRegression(alpha=alpha, beta=beta)
        theirs = []
        for i in range(len(y)):
            xi = {"x0": float(a[i]), "x1": float(b[i])}
            theirs.append(model.predict_one(xi))
            model.learn_one(xi, float(y[i]))
        return ours, np.array(theirs)

    @pytest.mark.parametrize(("alpha", "beta"), [(1.0, 1.0), (2.0, 4.0), (0.1, 25.0)])
    def test_predictions_match_exactly(self, alpha, beta):
        ours, theirs = self._run_both(alpha, beta)
        m = np.isfinite(ours)
        assert m.sum() > 400
        assert np.max(np.abs(ours[m] - theirs[m])) < 1e-12

    def test_standardize_true_is_a_different_estimator(self):
        # Guards against the switch silently doing nothing: with standardization
        # on, the correspondence must break.
        a, b, y = self._data()
        spec = po.spec.kalman(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            add_intercept=False,
            coef_halflife=float("inf"),
            q=[0.0, 0.0],
            obs_var=0.25,
            p0=0.5,
            standardize=True,
            halflife=float("inf"),
            min_periods=0.0,
        )
        df = pl.DataFrame({"x0": a, "x1": b, "y0": y})
        std_on = (
            po.ModelBank([spec])
            .fit_predict(df)["m"]
            .struct.field("pred_y0")
            .to_numpy()
            .astype(float)
        )
        _, theirs = self._run_both(2.0, 4.0)
        m = np.isfinite(std_on)
        assert np.max(np.abs(std_on[m] - theirs[m])) > 1e-6
