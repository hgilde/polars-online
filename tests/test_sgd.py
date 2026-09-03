"""E16: SGD with pluggable losses (ENHANCEMENTS E16).

The cheap baseline — one gradient step per row, no solves — and the only model
here that takes count targets, via the Poisson loss with a log link.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from expr_plugin import requires_expr_plugin


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        halflife=float("inf"),
        min_periods=10.0,
        learning_rate=0.05,
    )
    d.update(kw)
    return po.spec.sgd("m", **d)


def _fit(df, **kw):
    out = po.ModelBank([_spec(coef_every=1, **kw)]).fit_predict(df)
    return np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float), out


def _linear(n=20000, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    return pl.DataFrame(
        {"x0": x0, "x1": x1, "y0": 1.5 * x0 - 0.5 * x1 + 0.25 + noise * rng.standard_normal(n)}
    )


class TestLosses:
    def test_squared_recovers_the_coefficients(self):
        c, _ = _fit(_linear())
        assert c[0] == pytest.approx(0.25, abs=0.05)
        assert c[1] == pytest.approx(1.5, abs=0.05)
        assert c[2] == pytest.approx(-0.5, abs=0.05)

    def test_poisson_recovers_a_log_rate(self):
        rng = np.random.default_rng(0)
        n = 30000
        x = rng.standard_normal(n)
        y = rng.poisson(np.exp(0.4 + 0.8 * x)).astype(float)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        c, out = _fit(df, loss="poisson", learning_rate=0.02)
        assert c[0] == pytest.approx(0.4, abs=0.15), f"log-intercept {c[0]}"
        assert c[1] == pytest.approx(0.8, abs=0.15), f"log-slope {c[1]}"
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        assert np.nanmin(p) >= 0.0, "a Poisson rate cannot be negative"
        # Compare the *converged* predictions: the mean over the whole stream is
        # dominated by the early rows, where the fit is still finding the scale.
        assert np.nanmean(p[-10000:]) == pytest.approx(y[-10000:].mean(), rel=0.2)

    def test_poisson_needs_the_default_gradient_clip(self):
        # The default clip_gradient is finite precisely because of this: with a
        # log link one large count makes the next gradient exponentially bigger.
        rng = np.random.default_rng(0)
        n = 30000
        x = rng.standard_normal(n)
        y = rng.poisson(np.exp(0.4 + 0.8 * x)).astype(float)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        default, _ = _fit(df, loss="poisson", learning_rate=0.02)
        unclipped, _ = _fit(df, loss="poisson", learning_rate=0.02, clip_gradient=1e12)
        assert abs(default[0]) < 1.0, "the default should be stable"
        assert abs(unclipped[0]) > 1e3, "expected the unclipped fit to diverge"

    def test_clip_does_not_bind_for_squared_loss(self):
        df = _linear(n=5000)
        with_clip, _ = _fit(df)
        without, _ = _fit(df, clip_gradient=1e12)
        np.testing.assert_allclose(with_clip, without, rtol=0, atol=0)

    @pytest.mark.parametrize(("tau", "other"), [(0.1, 0.9)])
    def test_quantile_levels_are_ordered(self, tau, other):
        rng = np.random.default_rng(3)
        n = 20000
        df = pl.DataFrame({"x0": np.zeros(n), "x1": np.zeros(n), "y0": 1.0 + 2.0 * rng.random(n)})
        lo, _ = _fit(df, loss="quantile", quantile=tau)
        hi, _ = _fit(df, loss="quantile", quantile=other)
        assert lo[0] < hi[0]

    def test_huber_beats_squared_under_contamination(self):
        rng = np.random.default_rng(4)
        n = 20000
        x = rng.standard_normal(n)
        y = 2.0 * x
        bad = rng.random(n) < 0.05
        y[bad] = 500.0 * rng.standard_normal(bad.sum())
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        hub, _ = _fit(df, loss="huber", huber_delta=1.0)
        sq, _ = _fit(df, loss="squared")
        assert abs(hub[1] - 2.0) < abs(sq[1] - 2.0)

    def test_logistic_predicts_probabilities(self):
        rng = np.random.default_rng(5)
        n = 10000
        x = rng.standard_normal(n)
        y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(float)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        _, out = _fit(df, loss="logistic")
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        finite = p[np.isfinite(p)]
        assert ((finite >= 0) & (finite <= 1)).all()

    def test_epsilon_insensitive_fits_with_either_schedule(self):
        # Both reach the slope on this data. Which one wins depends on the
        # hyperparameters, so no ordering is asserted here; the controlled
        # demonstration that the sign-valued subgradient needs annealing to
        # *settle* lives in the Rust unit tests.
        df = _linear(n=30000, noise=0.1)
        const, _ = _fit(df, loss="epsilon_insensitive", eps=0.2, learning_rate=0.01)
        anneal, _ = _fit(
            df,
            loss="epsilon_insensitive",
            eps=0.2,
            learning_rate=0.5,
            schedule="inv_scaling",
            power=0.5,
        )
        for name, c in (("constant", const), ("annealed", anneal)):
            assert c[1] == pytest.approx(1.5, abs=0.1), f"{name}: {c}"
            assert c[2] == pytest.approx(-0.5, abs=0.1), f"{name}: {c}"


class TestSchedules:
    @pytest.mark.parametrize(
        ("schedule", "lr"), [("constant", 0.05), ("adagrad", 0.5), ("inv_scaling", 0.5)]
    )
    def test_all_schedules_converge(self, schedule, lr):
        c, _ = _fit(_linear(n=30000), schedule=schedule, learning_rate=lr, power=0.25)
        assert c[1] == pytest.approx(1.5, abs=0.15), f"{schedule}: {c}"
        assert c[2] == pytest.approx(-0.5, abs=0.15), f"{schedule}: {c}"

    def test_l2_shrinks(self):
        df = _linear(n=5000)
        plain, _ = _fit(df)
        shrunk, _ = _fit(df, l2=1.0)
        assert abs(shrunk[1]) < abs(plain[1])


class TestPlumbing:
    def test_chunk_invariance(self):
        df = _linear(n=400, seed=9)
        spec = _spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    @requires_expr_plugin
    def test_expression_equals_bank(self):
        df = _linear(n=400, seed=9)
        spec = _spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        keep = [c for c in one.columns if not c.startswith("coef")]
        expr = df.select(
            pl.col("y0").online.sgd(
                features=["x0", "x1"],
                halflife=float("inf"),
                min_periods=10.0,
                learning_rate=0.05,
            )
        ).unnest("y0")
        assert one.select(keep).equals(expr.select(keep), null_equal=True)

    def test_save_load(self, tmp_path):
        df = _linear(n=400, seed=10)
        spec = _spec(schedule="adagrad", learning_rate=0.5)
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 200))
        p = tmp_path / "s.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(200, 200)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_out_of_sample_on_noise(self):
        rng = np.random.default_rng(11)
        n = 5000
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "y0": rng.standard_normal(n),
            }
        )
        _, out = _fit(df, halflife=2000.0)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(p)
        assert abs(np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]) < 0.06

    def test_bad_config_rejected(self):
        with pytest.raises(ValueError, match="unknown sgd loss"):
            _spec(loss="hinge")
        with pytest.raises(ValueError, match="unknown sgd schedule"):
            _spec(schedule="cosine")
        with pytest.raises(ValueError, match="learning_rate"):
            _spec(learning_rate=0.0)
        with pytest.raises(ValueError, match="needs a .quantile. level"):
            _spec(loss="quantile")


class TestFeatureScaling:
    """E24: `scale_features` standardizes inputs against their running moments.

    Gradient methods are the ones that need it: a single learning rate has to
    suit every coordinate, so a feature in thousands and one in thousandths
    cannot both converge. The exact solvers do not care.
    """

    def _fit(self, scale):
        rng = np.random.default_rng(0)
        n = 20000
        x0 = 1000.0 * rng.standard_normal(n)
        x1 = 0.001 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x0, "x1": x1, "y0": 0.002 * x0 + 900.0 * x1})
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            learning_rate=0.01,
            halflife=float("inf"),
            min_periods=0.0,
            scale_features=scale,
            coef_every=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        return np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float)

    @staticmethod
    def _rel_err(c):
        return abs(c[1] - 0.002) / 0.002 + abs(c[2] - 900.0) / 900.0

    def test_rescues_badly_scaled_features(self):
        plain, scaled = self._fit(False), self._fit(True)
        assert self._rel_err(scaled) < self._rel_err(plain)
        assert self._rel_err(scaled) < 0.2, f"scaled fit still poor: {scaled}"

    def test_coefficients_come_back_in_original_units(self):
        c = self._fit(True)
        assert c[1] == pytest.approx(0.002, rel=0.15)
        assert c[2] == pytest.approx(900.0, rel=0.15)

    def test_off_by_default(self):
        spec = po.spec.sgd("m", targets=["y0"], features=["x0"], halflife=100.0, learning_rate=0.01)
        assert spec["model"]["scale_features"] is False

    def test_chunk_invariance(self):
        rng = np.random.default_rng(3)
        n = 400
        x0 = 100.0 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x0, "x1": rng.standard_normal(n), "y0": 0.01 * x0})
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            learning_rate=0.05,
            halflife=float("inf"),
            min_periods=5.0,
            scale_features=True,
        )
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)
