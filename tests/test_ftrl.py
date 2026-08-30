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


def test_chunk_invariance_and_expression_equality():
    df, _ = synthetic(seed=64, n_groups=2, n_rows=200, k=2, null_frac=0.0)
    df = df.with_columns(y0=(pl.col("y0") > 0).cast(pl.Float64))
    kw = dict(
        features=["x0", "x1"],
        halflife=300.0,
        clock="t",
        max_dclock=50.0,
        min_periods=10.0,
    )
    spec = po.spec.ftrl("m", targets=["y0"], group="group", **kw)
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)

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
