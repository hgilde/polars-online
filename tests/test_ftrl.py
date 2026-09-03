"""Task 14: online logistic regression / FTRL-proximal (docs/PLAN.md 4.6)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _pred(out, col="pred_y0"):
    return out["m"].struct.field(col).to_numpy().astype(float)


def _binary_df(n=6000, seed=21, noise=0.0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    logit = 1.5 * x0 - 0.5 * x1
    if noise:
        logit = logit + noise * rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return pl.DataFrame({"x0": x0, "x1": x1, "y0": y})


def _spec(**kw):
    d = dict(targets=["y0"], features=["x0", "x1"], halflife=1e9, min_periods=50.0)
    d.update(kw)
    return po.spec.ftrl("m", **d)


def test_predictions_are_probabilities():
    out = po.ModelBank([_spec()]).fit_predict(_binary_df())
    p = _pred(out)
    finite = p[np.isfinite(p)]
    assert finite.size > 1000
    assert (finite >= 0).all() and (finite <= 1).all()


def test_beats_the_base_rate_on_a_learnable_target():
    df = _binary_df()
    out = po.ModelBank([_spec()]).fit_predict(df)
    p = _pred(out)
    y = df["y0"].to_numpy()
    m = np.isfinite(p)
    # Log loss must beat always predicting the base rate.
    eps = 1e-12
    ll = -np.mean(y[m] * np.log(p[m] + eps) + (1 - y[m]) * np.log(1 - p[m] + eps))
    base = y[m].mean()
    ll_base = -np.mean(y[m] * np.log(base) + (1 - y[m]) * np.log(1 - base))
    assert ll < ll_base, f"log loss {ll} should beat base rate {ll_base}"


def test_resid_is_y_minus_p():
    df = _binary_df(n=2000)
    out = po.ModelBank([_spec()]).fit_predict(df)
    p = _pred(out)
    r = out["m"].struct.field("resid_y0").to_numpy().astype(float)
    y = df["y0"].to_numpy()
    m = np.isfinite(p) & np.isfinite(r)
    np.testing.assert_allclose(r[m], y[m] - p[m], atol=1e-12)


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(22)
    n = 5000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y0": (rng.random(n) < 0.5).astype(float),
        }
    )
    out = po.ModelBank([_spec(halflife=2000.0)]).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
    assert abs(ic) < 0.06, f"IC {ic}: predictions are not out-of-sample"


def test_forgets_a_regime_flip_on_the_clock():
    rng = np.random.default_rng(23)
    n = 4000
    x0 = rng.standard_normal(2 * n)
    y = np.concatenate([(x0[:n] > 0).astype(float), (x0[n:] < 0).astype(float)])
    df = pl.DataFrame({"x0": x0, "x1": np.zeros(2 * n), "y0": y})
    out = po.ModelBank([_spec(halflife=300.0)]).fit_predict(df)
    p = _pred(out)
    late = slice(2 * n - 500, 2 * n)
    hi = p[late][x0[late] > 0.5]
    assert np.nanmean(hi) < 0.4, "model did not follow the flipped regime"


def _plumbing_case():
    df, _ = synthetic(seed=64, n_groups=2, n_rows=200, k=2, null_frac=0.0)
    df = df.with_columns(y0=(pl.col("y0") > 0).cast(pl.Float64))
    kw = dict(
        features=["x0", "x1"],
        halflife=300.0,
        clock="t",
        max_dclock=50.0,
        min_periods=10.0,
    )
    return df, kw, po.spec.ftrl("m", targets=["y0"], group="group", **kw)


def test_chunk_invariance():
    df, _, spec = _plumbing_case()
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_expression_equals_bank():
    df, kw, spec = _plumbing_case()
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    keep = [c for c in one.columns if not c.startswith("coef")]
    expr = df.select(pl.col("y0").online.ftrl(**kw).over("group")).unnest("y0")
    assert one.select(keep).equals(expr.select(keep), null_equal=True)


def test_strict_binary_rejects_non_binary_targets():
    df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "x1": [0.0, 0.0, 0.0], "y0": [0.0, 0.5, 1.0]})
    out = po.ModelBank([_spec(strict_binary=True, min_periods=0.0)]).fit_predict(df)
    assert _pred(out).size == 3  # runs; the 0.5 row simply does not train


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="alpha"):
        _spec(alpha=0.0)
    with pytest.raises(ValueError, match="l1"):
        _spec(l1=-1.0)


class TestSquaredLoss:
    """E18: FTRL with a squared loss — sparse linear regression, no solves."""

    def _spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0", "x1", "x2"],
            loss="squared",
            alpha=0.5,
            l1=0.0,
            l2=0.01,
            halflife=float("inf"),
            min_periods=10.0,
        )
        d.update(kw)
        return po.spec.ftrl("m", **d)

    def _data(self, n=20000, seed=0, noise=0.1):
        rng = np.random.default_rng(seed)
        x0, x1, x2 = (rng.standard_normal(n) for _ in range(3))
        # x2 is pure noise
        y = 1.5 * x0 - 0.5 * x1 + noise * rng.standard_normal(n)
        return pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y0": y})

    def test_recovers_the_coefficients(self):
        df = self._data()
        out = po.ModelBank([self._spec(coef_every=1)]).fit_predict(df)
        c = np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float)
        assert c[1] == pytest.approx(1.5, abs=0.1)
        assert c[2] == pytest.approx(-0.5, abs=0.1)

    def test_l1_zeroes_the_noise_feature(self):
        df = self._data()
        out = po.ModelBank([self._spec(l1=0.5, coef_every=1)]).fit_predict(df)
        c = np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float)
        assert abs(c[3]) < 0.05, f"noise feature survived L1: {c[3]}"
        assert abs(c[1]) > 1.0, "the signal feature should survive"

    def test_predictions_are_not_squashed_into_a_probability(self):
        # The logistic link would cap these at 1.
        rng = np.random.default_rng(5)
        n = 5000
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "x2": np.zeros(n), "y0": 20.0 * x})
        out = po.ModelBank([self._spec(min_periods=5.0)]).fit_predict(df)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        assert np.nanmax(p) > 5.0

    def test_logistic_is_still_the_default(self):
        df = self._data(n=2000)
        df = df.with_columns(y0=(pl.col("y0") > 0).cast(pl.Float64))
        out = po.ModelBank(
            [
                po.spec.ftrl(
                    "m",
                    targets=["y0"],
                    features=["x0", "x1", "x2"],
                    halflife=float("inf"),
                    min_periods=10.0,
                )
            ]
        ).fit_predict(df)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        finite = p[np.isfinite(p)]
        assert ((finite >= 0) & (finite <= 1)).all()

    def test_out_of_sample_on_noise(self):
        rng = np.random.default_rng(6)
        n = 5000
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y0": rng.standard_normal(n),
            }
        )
        out = po.ModelBank([self._spec(halflife=2000.0)]).fit_predict(df)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(p)
        ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
        assert abs(ic) < 0.06

    def test_chunk_invariance(self):
        df = self._data(n=400, seed=7)
        spec = self._spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    def test_expression_equals_bank(self):
        df = self._data(n=400, seed=7)
        spec = self._spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        keep = [c for c in one.columns if not c.startswith("coef")]
        expr = df.select(
            pl.col("y0").online.ftrl(
                features=["x0", "x1", "x2"],
                loss="squared",
                alpha=0.5,
                l1=0.0,
                l2=0.01,
                halflife=float("inf"),
                min_periods=10.0,
            )
        ).unnest("y0")
        assert one.select(keep).equals(expr.select(keep), null_equal=True)

    def test_strict_binary_is_rejected(self):
        with pytest.raises(ValueError, match="strict_binary"):
            self._spec(strict_binary=True)

    def test_unknown_loss_is_rejected(self):
        with pytest.raises(ValueError, match="unknown ftrl loss"):
            self._spec(loss="hinge")
