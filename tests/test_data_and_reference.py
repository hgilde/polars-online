"""Task 2: the synthetic generator and the numpy oracles are self-consistent."""

import numpy as np
import polars as pl

from data import public_intraday_or_skip, synthetic
from reference import compute_dclock, ewridge_ref, rls_ref


def _arrays(df: pl.DataFrame, k: int = 3):
    x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(k)])
    y = df["y0"].to_numpy().reshape(-1, 1)
    return x, y, df["t"].to_numpy(), df["session"].to_numpy(), df["w"].to_numpy()


def test_synthetic_is_deterministic():
    df1, b1 = synthetic(seed=7)
    df2, b2 = synthetic(seed=7)
    assert df1.equals(df2)
    assert all(np.array_equal(b1[g], b2[g]) for g in b1)
    df3, _ = synthetic(seed=8)
    assert not df1.equals(df3)


def test_synthetic_shape_and_clock():
    df, betas = synthetic(n_groups=2, n_rows=100, k=4, n_targets=2)
    assert df.height == 200
    assert betas["g0"].shape == (100, 2, 4)
    for _, g in df.group_by("group"):
        t = g["t"].to_numpy()
        assert (np.diff(t) > 0).all()
        # volume clock resets at each session break
        vol = g["vol"].to_numpy()
        ses = g["session"].to_numpy()
        breaks = np.nonzero(np.diff(ses))[0] + 1
        assert (vol[breaks] < vol[breaks - 1]).all()


def test_ewridge_oracle_recovers_static_beta():
    rng = np.random.default_rng(0)
    n, k = 500, 3
    beta = np.array([0.5, -1.0, 2.0])
    x = rng.standard_normal((n, k))
    y = (x @ beta + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    dc = np.ones(n)
    dc[0] = 0.0
    out = ewridge_ref(x, y, dc, np.ones(n), halflife=1e6, ridge=1e-8)
    np.testing.assert_allclose(out["coef"][-1, 0, 1:], beta, atol=1e-3)
    assert abs(out["coef"][-1, 0, 0]) < 1e-2  # intercept ~ 0


def test_ewridge_pred_is_out_of_sample():
    # Pure-noise target => oracle predictions must not correlate with y.
    rng = np.random.default_rng(1)
    n, k = 2000, 2
    x = rng.standard_normal((n, k))
    y = rng.standard_normal((n, 1))
    dc = np.ones(n)
    dc[0] = 0.0
    out = ewridge_ref(x, y, dc, np.ones(n), halflife=200.0)
    m = ~np.isnan(out["pred"][:, 0])
    ic = np.corrcoef(out["pred"][m, 0], y[m, 0])[0, 1]
    assert abs(ic) < 0.08


def test_ridge_decay_matches_rls_exactly():
    df, _ = synthetic(n_groups=1, n_rows=300, k=3, null_frac=0.0)
    x, y, t, s, w = _arrays(df)
    dc, rs = compute_dclock(t, s, len(df), max_dclock=50.0, session_gap=25.0)
    a = ewridge_ref(x, y, dc, w, rs, halflife=300.0, ridge=1.0, ridge_decay=True)
    b = rls_ref(x, y, dc, w, rs, halflife=300.0, ridge=1.0)
    m = ~(np.isnan(a["pred"][:, 0]) | np.isnan(b["pred"][:, 0]))
    assert m.sum() > 250
    np.testing.assert_allclose(a["pred"][m, 0], b["pred"][m, 0], atol=1e-12)


def test_compute_dclock_semantics():
    t = np.array([0.0, 10.0, 5.0, 6.0, 200.0])
    d, r = compute_dclock(t, None, 5, max_dclock=50.0, on_clock_reset="max")
    np.testing.assert_allclose(d, [0.0, 10.0, 50.0, 1.0, 50.0])
    assert not r.any()
    d, r = compute_dclock(t, None, 5, max_dclock=50.0, on_clock_reset="zero")
    assert d[2] == 0.0
    d, r = compute_dclock(t, None, 5, max_dclock=50.0, on_clock_reset="reset_state")
    assert r[2] and d[2] == 0.0
    ses = np.array([0, 0, 1, 1, 1])
    d, r = compute_dclock(t, ses, 5, max_dclock=50.0, session_gap=7.5)
    assert d[2] == 7.5  # session change overrides the negative delta
    d, r = compute_dclock(t, ses, 5, max_dclock=50.0, session_gap="reset")
    assert r[2]


def test_null_policy_in_oracle():
    df, _ = synthetic(n_groups=1, n_rows=200, k=2, null_frac=0.0)
    x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
    y = df["y0"].to_numpy().reshape(-1, 1)
    n = len(df)
    dc = np.ones(n)
    dc[0] = 0.0
    x_null = x.copy()
    x_null[50, 0] = np.nan  # feature null: row skipped
    y_null = y.copy()
    y_null[60, 0] = np.nan  # target null: predict-only
    out = ewridge_ref(x_null, y_null, dc, np.ones(n), halflife=100.0)
    assert np.isnan(out["pred"][50, 0]) and np.isnan(out["n_eff"][50])
    assert np.isfinite(out["pred"][60, 0]) and np.isnan(out["resid"][60, 0])


def test_public_intraday_download():
    df = public_intraday_or_skip()
    assert df.height > 1000
    assert (np.diff(df["t"].to_numpy()) > 0).all()
    assert df["close"].null_count() == 0
