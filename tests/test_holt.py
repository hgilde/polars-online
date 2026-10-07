"""E25: Holt's linear trend method — the no-features forecasting baseline."""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _spec(**kw):
    d = dict(targets=["y0"], clock="t", gap_cap=100.0, half_life=5.0, min_weight=3.0)
    d.update(kw)
    return po.spec.holt("m", **d)


def _run(df, **kw):
    return po.ModelBank([_spec(coef_every=0, **kw)]).fit_predict(df)


def _trending(n=500, slope=2.0, noise=0.5, step=1.0, seed=0):
    t = np.arange(float(n)) * step
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"t": t, "y0": 3.0 + slope * t + noise * rng.standard_normal(n)})


def test_needs_no_features():
    spec = _spec()
    assert spec["features"] == []
    assert po.spec.output_fields(spec) == [
        "pred_y0",
        "resid_y0",
        "weight_sum",
        "settled_frac",
        "withheld_reason",
        "coef",
    ]


def test_recovers_a_linear_trend():
    out = _run(_trending())
    level, trend = out["m"].struct.field("coef").to_list()[-1]
    assert trend == pytest.approx(2.0, abs=0.1)
    assert level == pytest.approx(3.0 + 2.0 * 499, rel=0.01)


def test_predicts_the_next_value():
    df = _trending()
    out = _run(df)
    pred = out["m"].struct.field("pred_y0").to_list()[-1]
    # 0.343 off measured; a forecast one step stale is the slope, 2.0, off,
    # so a bound of 2.0 let that through.
    assert pred == pytest.approx(df["y0"].to_list()[-1], abs=1.0)


def test_an_infinite_trend_halflife_learns_the_whole_history_drift():
    """``trend_half_life=inf`` forgets no slope, as ``inf`` means everywhere
    else here. It used to pin the trend at zero, so the fit lagged a trending
    series (review 2026-09-12, S30)."""
    out = _run(_trending(), trend_half_life=float("inf"))
    _, trend = out["m"].struct.field("coef").to_list()[-1]
    assert trend == pytest.approx(2.0, abs=0.1)


def test_lam_one_is_an_infinite_halflife():
    """``lam=1`` is no forgetting, as for every other model: the same fit as
    ``half_life=inf``. It became ``-inf`` on its way to a half-life and was
    refused in ``level_half_life``'s name (review 2026-09-12, S30)."""
    df = _trending(n=200)
    kw = dict(targets=["y0"], clock="t", gap_cap=100.0, min_weight=3.0)
    kw["trend_half_life"] = float("inf")
    by_lam = po.ModelBank([po.spec.holt("m", lam=1.0, **kw)]).fit_predict(df)
    by_inf = po.ModelBank([po.spec.holt("m", half_life=float("inf"), **kw)]).fit_predict(df)
    assert by_lam.equals(by_inf, null_equal=True)


def test_the_level_halflife_has_one_name():
    """``half_life`` is holt's level half-life, ``inf`` included, under one
    name: ``level_half_life``, a second name for the same knob that built a
    second dict shape (review 2026-10-06, TA6), is refused naming it
    (docs/PLAN.md task 196, N16). Under two names, the builder had taken
    ``inf`` under one and refused it under the other (review 2026-09-12,
    S27)."""
    kw = dict(targets=["y0"], clock="t", gap_cap=100.0, min_weight=3.0)
    with pytest.raises(TypeError, match="level_half_life was renamed half_life"):
        po.spec.holt("m", level_half_life=float("inf"), **kw)
    spec = po.spec.holt("m", half_life=float("inf"), **kw)
    assert "level_half_life" not in spec["model"]
    out = po.ModelBank([spec]).fit_predict(_trending(n=200))
    assert out["m"].struct.field("weight_sum")[-1] == 199.0, "no forgetting"


def test_irregular_clock_extrapolates_the_right_distance():
    # The trend is per clock unit, so a 5-unit gap must forecast 5 units ahead.
    df = _trending(n=400, step=5.0, noise=0.0)
    out = _run(df, half_life=20.0)
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

    holt_out = _run(df, half_life=20.0)
    ridge_out = po.ModelBank(
        [
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                clock="t",
                gap_cap=100.0,
                half_life=20.0,
                min_weight=3.0,
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


def test_the_spec_halflife_alone_is_enough(tmp_path):
    """The level's half-life is the spec's ``half_life``, which satisfies the
    "one of half-life/lam is required" rule alone. The README's Holt
    example gave a second name for it, and used to be refused (IMPROVEMENTS
    U6); the second name is gone (docs/PLAN.md task 196)."""
    df = _trending(n=200)
    d = dict(targets=["y0"], clock="t", gap_cap=100.0, min_weight=3.0, trend_half_life=80.0)
    by_halflife = po.spec.holt("m", half_life=20.0, **d)
    a = po.ModelBank([by_halflife]).fit_predict(df)
    assert a["m"].struct.field("pred_y0").drop_nulls().len() > 150
    # And the field names stay ungridded -- no `@h` suffix from a phantom grid.
    assert [f.name for f in a.schema["m"].fields] == po.spec.output_fields(by_halflife)


def test_a_holt_spec_still_needs_one_of_them():
    with pytest.raises(ValueError, match="one of half_life/lam is required"):
        po.spec.holt("m", targets=["y0"], trend_half_life=100.0)
    with pytest.raises(ValueError, match="half_life must be > 0"):
        po.spec.holt("m", targets=["y0"], half_life=-1.0)


def test_other_models_do_not_get_the_exemption():
    with pytest.raises(ValueError, match="one of half_life/lam is required"):
        po.spec.kalman("m", targets=["y0"], features=["x0"], coef_half_life=50.0)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="half_life"):
        _spec(half_life=0.0)
    with pytest.raises(ValueError, match="trend_half_life"):
        _spec(trend_half_life=0.0)


def test_other_models_still_require_features():
    with pytest.raises(ValueError, match="features must be non-empty"):
        po.spec.ewridge("m", targets=["y0"], features=[], half_life=10.0)
