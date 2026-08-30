"""Task 13: robust models - Huber and quantile (docs/PLAN.md section 4.5)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _pred(out, col="pred_y0"):
    return out["m"].struct.field(col).to_numpy().astype(float)


def test_huber_beats_least_squares_under_outliers():
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.1 * rng.standard_normal(n)
    contaminated = rng.random(n) < 0.03
    y[contaminated] = 300.0 * rng.standard_normal(contaminated.sum())
    df = pl.DataFrame({"x0": x, "y0": y})
    clean = 2.0 * x  # what a perfect model would predict

    common = dict(
        targets=["y0"],
        features=["x0"],
        halflife=1e9,
        min_periods=10.0,
        # default solve_every is halflife/50; with halflife 1e9 that is one
        # solve ever, so ask for a real cadence.
        max_rows_between_solves=1,
    )
    hub = po.ModelBank([po.spec.huber("m", huber_delta=1.5, **common)]).fit_predict(df)
    ols = po.ModelBank([po.spec.ewridge("m", ridge=1e-8, **common)]).fit_predict(df)

    ok = np.isfinite(_pred(hub)) & np.isfinite(_pred(ols)) & ~contaminated
    e_hub = np.mean((_pred(hub)[ok] - clean[ok]) ** 2)
    e_ols = np.mean((_pred(ols)[ok] - clean[ok]) ** 2)
    assert e_hub < e_ols, f"huber {e_hub} should beat ols {e_ols}"


def test_huge_delta_reduces_to_least_squares():
    df, _ = synthetic(seed=61, n_groups=1, n_rows=300, k=2, null_frac=0.0)
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        halflife=1e9,
        min_periods=10.0,
        max_rows_between_solves=1,
    )
    a = po.ModelBank([po.spec.huber("m", huber_delta=1e9, **common)]).fit_predict(df)
    b = po.ModelBank([po.spec.ewridge("m", ridge=1e-6, **common)]).fit_predict(df)
    pa, pb = _pred(a), _pred(b)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert np.max(np.abs(pa[m] - pb[m])) < 1e-6


def test_quantile_levels_are_ordered():
    rng = np.random.default_rng(12)
    n = 6000
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": 1.0 + 2.0 * rng.random(n)})
    common = dict(
        targets=["y0"],
        features=["x0"],
        halflife=1e9,
        min_periods=20.0,
        max_rows_between_solves=1,
    )
    preds = {}
    for tau in (0.1, 0.5, 0.9):
        out = po.ModelBank([po.spec.quantile("m", quantile=tau, **common)]).fit_predict(df)
        preds[tau] = np.nanmean(_pred(out))
    assert preds[0.1] < preds[0.5] < preds[0.9], preds


def test_quantile_coverage_is_roughly_right():
    rng = np.random.default_rng(13)
    n = 8000
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n)})
    out = po.ModelBank(
        [
            po.spec.quantile(
                "m",
                quantile=0.9,
                targets=["y0"],
                features=["x0"],
                halflife=2000.0,
                min_periods=50.0,
                max_rows_between_solves=1,
            )
        ]
    ).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    below = (df["y0"].to_numpy()[m] < p[m]).mean()
    # IRLS quantile regression is approximate online; require the right side of
    # the median and in the neighbourhood of 0.9.
    assert 0.6 < below < 0.99, below


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(14)
    n = 3000
    df = pl.DataFrame(
        {"x0": rng.standard_normal(n), "x1": rng.standard_normal(n), "y0": rng.standard_normal(n)}
    )
    out = po.ModelBank(
        [
            po.spec.huber(
                "m",
                targets=["y0"],
                features=["x0", "x1"],
                halflife=300.0,
                min_periods=20.0,
                max_rows_between_solves=1,
            )
        ]
    ).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
    assert abs(ic) < 0.06


def test_chunk_invariance_and_expression_equality():
    df, _ = synthetic(seed=62, n_groups=2, n_rows=200, k=2, null_frac=0.0)
    kw = dict(
        targets=["y0"],
        features=["x0", "x1"],
        halflife=300.0,
        clock="t",
        max_dclock=50.0,
        min_periods=10.0,
        huber_delta=1.5,
    )
    spec = po.spec.huber("m", group="group", **kw)
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)

    expr = df.select(
        pl.col("y0").online.huber(**{k: v for k, v in kw.items() if k != "targets"}).over("group")
    ).unnest("y0")
    assert one.select(keep).equals(expr.select(keep), null_equal=True)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="quantile"):
        po.spec.quantile("m", quantile=0.0, targets=["y0"], features=["x0"], halflife=10.0)
    with pytest.raises(ValueError, match="huber_delta"):
        po.spec.huber("m", huber_delta=-1.0, targets=["y0"], features=["x0"], halflife=10.0)
