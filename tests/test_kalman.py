"""Task 11: Kalman / random-walk-beta (docs/PLAN.md section 4.4)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _spec(**kw):
    defaults = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        coef_half_life=100.0,
        half_life=500.0,
        min_weight=20.0,
    )
    defaults.update(kw)
    return po.spec.kalman("m", **defaults)


def _pred(out, col="pred_y0"):
    return out["m"].struct.field(col).to_numpy().astype(float)


def test_tracks_time_varying_beta_better_than_a_pinned_filter():
    # The synthetic generator's beta is a random walk, which is exactly the
    # Kalman model's assumption; a responsive filter must beat a pinned one.
    df, _ = synthetic(seed=41, n_groups=1, n_rows=1500, k=3, null_frac=0.0, beta_sigma=0.03)
    fast = po.ModelBank([_spec(coef_half_life=50.0)]).fit_predict(df)
    pinned = po.ModelBank([_spec(coef_half_life=float("inf"))]).fit_predict(df)
    y = df["y0"].to_numpy()
    for out, name in ((fast, "fast"), (pinned, "pinned")):
        assert np.isfinite(_pred(out)).sum() > 1000, name
    e_fast = np.nanmean((y - _pred(fast)) ** 2)
    e_pin = np.nanmean((y - _pred(pinned)) ** 2)
    assert e_fast < e_pin, f"fast {e_fast} vs pinned {e_pin}"


def _differ(a, b) -> float:
    pa, pb = _pred(a), _pred(b)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.sum() > 100, "too few rows scored to compare"
    return float(np.abs(pa[m] - pb[m]).max())


def test_per_factor_halflife_and_pinning():
    """Each slot takes its own half-life: the per-factor filter is neither
    the scalar one nor the same one with x1's half-life changed. It once
    checked only that a prediction came out (review 2026-10-05, TB6)."""
    df, _ = synthetic(seed=42, n_groups=1, n_rows=400, k=3, null_frac=0.0)
    # intercept pinned, x0 slow, x1 fast, x2 pinned
    spec = _spec(coef_half_life=[float("inf"), 500.0, 30.0, float("inf")])
    out = po.ModelBank([spec]).fit_predict(df)
    scalar = po.ModelBank([_spec(coef_half_life=100.0)]).fit_predict(df)
    x1_slow = _spec(coef_half_life=[float("inf"), 500.0, 300.0, float("inf")])
    assert _differ(out, scalar) > 1e-6
    assert _differ(out, po.ModelBank([x1_slow]).fit_predict(df)) > 1e-6


def test_explicit_q_overrides_halflife():
    df, _ = synthetic(seed=43, n_groups=1, n_rows=200, k=3, null_frac=0.0)
    a = po.ModelBank([_spec(q=[0.0, 0.0, 0.0, 0.0])]).fit_predict(df)
    b = po.ModelBank([_spec(q=[0.01, 0.01, 0.01, 0.01])]).fit_predict(df)
    # zero process noise converges; nonzero keeps moving, so they differ
    pa, pb = _pred(a), _pred(b)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert np.abs(pa[m] - pb[m]).max() > 1e-6


def test_share_p_runs_and_differs_from_per_target():
    df, _ = synthetic(seed=44, n_groups=1, n_rows=300, k=3, n_targets=2, null_frac=0.0)
    kw = dict(targets=["y0", "y1"])
    a = po.ModelBank([_spec(share_p=False, **kw)]).fit_predict(df)
    b = po.ModelBank([_spec(share_p=True, **kw)]).fit_predict(df)
    # One P driven by the mean sigma^2 is not two driven by their own.
    for col in ("pred_y0", "pred_y1"):
        pa, pb = _pred(a, col), _pred(b, col)
        m = np.isfinite(pa) & np.isfinite(pb)
        assert m.sum() > 100 and np.abs(pa[m] - pb[m]).max() > 1e-6, col


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(9)
    n = 3000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )
    out = po.ModelBank([_spec(coef_half_life=200.0, half_life=500.0)]).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
    assert abs(ic) < 0.06, f"IC {ic}: predictions are not out-of-sample"


def test_chunk_invariance():
    df, _ = synthetic(seed=45, n_groups=2, n_rows=200, k=3, null_frac=0.0)
    spec = _spec(group="group", clock="t", gap_cap=50.0, weight="w")
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 30)) for i in range(0, df.height, 30)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="coef_half_life"):
        _spec(coef_half_life=[1.0, 2.0])  # wrong length for k=3 + intercept
    with pytest.raises(ValueError, match="obs_var"):
        _spec(obs_var=0.0)
