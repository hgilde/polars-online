"""E25: Holt's linear trend method — the no-features forecasting baseline."""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _spec(**kw):
    d = dict(targets=["y0"], clock="t", max_dclock=100.0, halflife=5.0, min_periods=3.0)
    d.update(kw)
    return po.spec.holt("m", **d)


def _run(df, **kw):
    return po.ModelBank([_spec(coef_every=1, **kw)]).fit_predict(df)


def _trending(n=500, slope=2.0, noise=0.5, step=1.0, seed=0):
    t = np.arange(float(n)) * step
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"t": t, "y0": 3.0 + slope * t + noise * rng.standard_normal(n)})


def test_needs_no_features():
    spec = _spec()
    assert spec["features"] == []
    assert po.spec.output_fields(spec) == ["pred_y0", "resid_y0", "n_eff", "coef"]


def test_recovers_a_linear_trend():
    out = _run(_trending())
    level, trend = out["m"].struct.field("coef").to_list()[-1]
    assert trend == pytest.approx(2.0, abs=0.1)
    assert level == pytest.approx(3.0 + 2.0 * 499, rel=0.01)


def test_predicts_the_next_value():
    df = _trending()
    out = _run(df)
    pred = out["m"].struct.field("pred_y0").to_list()[-1]
    assert pred == pytest.approx(df["y0"].to_list()[-1], abs=2.0)


def test_a_pinned_trend_lags_a_trending_series():
    df = _trending()
    with_trend = _run(df)["m"].struct.field("pred_y0").to_list()[-1]
    level_only = _run(df, trend_halflife=float("inf"))["m"].struct.field("pred_y0").to_list()[-1]
    actual = df["y0"].to_list()[-1]
    assert abs(with_trend - actual) < abs(level_only - actual)


def test_irregular_clock_extrapolates_the_right_distance():
    # The trend is per clock unit, so a 5-unit gap must forecast 5 units ahead.
    df = _trending(n=400, step=5.0, noise=0.0)
    out = _run(df, halflife=20.0)
    pred = out["m"].struct.field("pred_y0").to_list()[-1]
    assert pred == pytest.approx(df["y0"].to_list()[-1], rel=0.01)


def test_flat_series_has_no_trend():
    df = pl.DataFrame({"t": np.arange(200.0), "y0": np.full(200, 7.0)})
    level, trend = _run(df)["m"].struct.field("coef").to_list()[-1]
    assert level == pytest.approx(7.0, abs=1e-6)
    assert abs(trend) < 1e-6


def test_is_a_baseline_a_regression_should_beat():
    # On data with a real feature relationship, ewridge must beat Holt; on a
    # pure trend with no informative feature, Holt should win.
    rng = np.random.default_rng(3)
    n = 2000
    t = np.arange(float(n))
    x = rng.standard_normal(n)
    df = pl.DataFrame({"t": t, "x0": x, "y0": 0.05 * t + 3.0 * x + 0.1 * rng.standard_normal(n)})

    holt_out = _run(df, halflife=20.0)
    ridge_out = po.ModelBank(
        [
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                clock="t",
                max_dclock=100.0,
                halflife=20.0,
                min_periods=3.0,
                max_rows_between_solves=1,
            )
        ]
    ).fit_predict(df)

    def mse(out):
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        y = df["y0"].to_numpy()
        m = np.isfinite(p)
        return float(np.mean((p[m] - y[m]) ** 2))

    assert mse(ridge_out) < mse(holt_out), "a real feature should beat the baseline"


def test_chunk_invariance_and_save_load(tmp_path):
    df = _trending(n=300, seed=4)
    spec = _spec()
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 31)) for i in range(0, df.height, 31)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)

    a = po.ModelBank([spec])
    a.fit_predict(df.slice(0, 150))
    p = tmp_path / "h.state"
    a.save(p)
    b = po.ModelBank.load(p, specs=[spec])
    rest = df.slice(150, 150)
    assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="level_halflife"):
        _spec(level_halflife=0.0)
    with pytest.raises(ValueError, match="trend_halflife"):
        _spec(trend_halflife=0.0)


def test_other_models_still_require_features():
    with pytest.raises(ValueError, match="features must be non-empty"):
        po.spec.ewridge("m", targets=["y0"], features=[], halflife=10.0)
