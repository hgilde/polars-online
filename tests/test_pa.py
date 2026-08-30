"""E17: passive-aggressive regression."""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        halflife=float("inf"),
        min_periods=10.0,
    )
    d.update(kw)
    return po.spec.pa("m", **d)


def _fit(df, **kw):
    out = po.ModelBank([_spec(coef_every=1, **kw)]).fit_predict(df)
    return np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float), out


def _linear(n=5000, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    return pl.DataFrame(
        {"x0": x0, "x1": x1, "y0": 1.5 * x0 - 0.5 * x1 + 0.25 + noise * rng.standard_normal(n)}
    )


@pytest.mark.parametrize("mode", ["pa", "pa1", "pa2"])
def test_recovers_a_noiseless_relationship(mode):
    c, _ = _fit(_linear(), mode=mode, eps=0.01, c=1.0)
    assert c[0] == pytest.approx(0.25, abs=0.05), f"{mode}: {c}"
    assert c[1] == pytest.approx(1.5, abs=0.05), f"{mode}: {c}"
    assert c[2] == pytest.approx(-0.5, abs=0.05), f"{mode}: {c}"


def test_no_learning_rate_is_needed():
    # The point of PA: it reaches the answer with no rate to tune, where SGD at
    # a badly chosen rate does not.
    df = _linear(n=3000)
    pa_c, _ = _fit(df, eps=0.01)
    sgd_out = po.ModelBank(
        [
            po.spec.sgd(
                "m",
                targets=["y0"],
                features=["x0", "x1"],
                learning_rate=1e-4,
                halflife=float("inf"),
                min_periods=10.0,
                coef_every=1,
            )
        ]
    ).fit_predict(df)
    sgd_c = np.array(sgd_out["m"].struct.field("coef").to_list()[-1], dtype=float)
    assert abs(pa_c[1] - 1.5) < abs(sgd_c[1] - 1.5)


def test_bounded_variants_resist_outliers():
    """Measured over the stream, not at the last row.

    Plain PA satisfies each row's constraint *exactly*, so after an outlier the
    next clean row pulls it straight back and the final coefficient looks fine.
    The cost is paid in between: the predictions it makes right after each
    outlier are wild. Averaging the out-of-sample error over the whole stream is
    what shows the difference.
    """
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.standard_normal(n)
    clean = 2.0 * x
    y = clean.copy()
    bad = rng.random(n) < 0.04
    y[bad] = 500.0 * rng.standard_normal(bad.sum())
    df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})

    def mean_abs_err(**kw):
        _, out = _fit(df, eps=0.01, **kw)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(p) & ~bad
        return float(np.mean(np.abs(p[m] - clean[m])))

    capped = mean_abs_err(mode="pa1", c=0.05)
    unbounded = mean_abs_err(mode="pa")
    assert capped < unbounded, f"pa1 {capped} should beat pa {unbounded}"
    # and the capped variant is genuinely close to the clean signal
    assert capped < 0.5


def test_wide_tube_is_passive():
    df = _linear(n=500)
    c, _ = _fit(df, eps=1e6)
    assert np.abs(c).max() == 0.0, "nothing should move inside a huge tube"


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(5)
    n = 5000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )
    _, out = _fit(df, mode="pa1", c=0.05)
    p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
    m = np.isfinite(p)
    assert abs(np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]) < 0.06


def test_chunk_invariance_and_expression_equality():
    df = _linear(n=400, seed=9, noise=0.1)
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

    expr = df.select(
        pl.col("y0").online.pa(features=["x0", "x1"], halflife=float("inf"), min_periods=10.0)
    ).unnest("y0")
    assert one.select(keep).equals(expr.select(keep), null_equal=True)


def test_save_load(tmp_path):
    df = _linear(n=400, seed=10, noise=0.1)
    spec = _spec()
    a = po.ModelBank([spec])
    a.fit_predict(df.slice(0, 200))
    p = tmp_path / "pa.state"
    a.save(p)
    b = po.ModelBank.load(p, specs=[spec])
    rest = df.slice(200, 200)
    assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="unknown pa mode"):
        _spec(mode="pa3")
    with pytest.raises(ValueError, match="pa c must be > 0"):
        _spec(c=0.0)
    with pytest.raises(ValueError, match="pa eps must be >= 0"):
        _spec(eps=-1.0)
