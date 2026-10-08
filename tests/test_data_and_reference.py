"""Task 2: the synthetic generator and the numpy oracles are self-consistent."""

import numpy as np
import polars as pl
import pytest

from data import public_intraday_or_skip, synthetic
from reference import compute_dclock, ewridge_ref, kalman_ref, rls_ref

TIER = "mixed"


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
    out = ewridge_ref(x, y, dc, np.ones(n), half_life=1e6, ridge=1e-8)
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
    out = ewridge_ref(x, y, dc, np.ones(n), half_life=200.0)
    m = ~np.isnan(out["pred"][:, 0])
    ic = np.corrcoef(out["pred"][m, 0], y[m, 0])[0, 1]
    assert abs(ic) < 0.08


def test_ridge_decay_matches_rls_exactly():
    df, _ = synthetic(n_groups=1, n_rows=300, k=3, null_frac=0.0)
    x, y, t, s, w = _arrays(df)
    dc, rs = compute_dclock(t, s, len(df), gap_cap=50.0, session_gap=25.0)
    a = ewridge_ref(x, y, dc, w, rs, half_life=300.0, ridge=1.0, ridge_scale=True)
    b = rls_ref(x, y, dc, w, rs, half_life=300.0, ridge=1.0)
    m = ~(np.isnan(a["pred"][:, 0]) | np.isnan(b["pred"][:, 0]))
    assert m.sum() > 250
    np.testing.assert_allclose(a["pred"][m, 0], b["pred"][m, 0], atol=1e-12)


def test_compute_dclock_semantics():
    """Mirrors `caps_and_negative_deltas` in crates/online-core/src/clock.rs."""
    t = np.array([0.0, 10.0, 5.0, 6.0, 200.0])
    with pytest.raises(ValueError, match="row 2: the clock goes backwards by 5"):
        compute_dclock(t, None, 5, gap_cap=50.0)
    d, r = compute_dclock(t, None, 5, gap_cap=50.0, restart_after_step_back=2.0)
    assert r[2] and d[2] == 0.0
    np.testing.assert_allclose(d, [0.0, 10.0, 0.0, 1.0, 50.0])
    # Inclusive: a step back as large as the minimum is a late row.
    with pytest.raises(ValueError, match="row 2: a late row"):
        compute_dclock(t, None, 5, gap_cap=50.0, restart_after_step_back=5.0)
    ses = np.array([0, 0, 1, 1, 1])
    d, r = compute_dclock(t, ses, 5, gap_cap=50.0, session_gap=7.5)
    assert d[2] == 7.5  # session change overrides the negative delta
    d, r = compute_dclock(t, ses, 5, gap_cap=50.0, session_gap="reset")
    assert r[2]


def test_a_zero_weight_row_leaves_the_oracles_finite():
    """Hard rule 9 holds of the oracles too: a row of weight 0 advances the
    clock and teaches nothing, the stream's first row and its first scored
    row included, and is never a 0/0 that poisons what follows. On this
    stream ``kalman_ref`` divided its feature moments by the weight after a
    zero-weight first row, so none of its 120 predictions was a number, and
    ``ewridge_ref``'s sigma2 by its weight after a zero-weight first scored
    row, NaN on each of the 114 rows after it (review 2026-10-06, TA1 and
    TA10)."""
    df, _ = synthetic(seed=71, n_groups=1, n_rows=120, k=3, null_frac=0.0)
    x, y, t, _, w = _arrays(df)
    dc, _ = compute_dclock(t, None, len(df), gap_cap=50.0)

    first_row = w.copy()
    first_row[0] = 0.0
    pred = kalman_ref(x, y, dc, first_row, gap_cap=50.0)["pred"][:, 0]
    scored = np.flatnonzero(np.isfinite(pred))
    assert scored.size > 100, scored.size
    assert np.isfinite(pred[scored[0] :]).all()

    first_scored = int(np.argmax(np.isfinite(ewridge_ref(x, y, dc, w, gap_cap=50.0)["pred"][:, 0])))
    zeroed = w.copy()
    zeroed[first_scored] = 0.0
    sig2 = ewridge_ref(x, y, dc, zeroed, gap_cap=50.0)["sig2"][:, 0]
    assert np.isfinite(sig2).all()
    # The rows after it still add their residuals.
    added = np.flatnonzero(sig2 > 0.0)
    assert added.size > 100 and added[0] > first_scored, added
    assert (sig2[added[0] :] > 0.0).all()


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
    out = ewridge_ref(x_null, y_null, dc, np.ones(n), half_life=100.0)
    assert np.isnan(out["pred"][50, 0]) and np.isnan(out["weight_sum"][50])
    assert np.isfinite(out["pred"][60, 0]) and np.isnan(out["resid"][60, 0])


@pytest.mark.extended(reason="the network: the public intraday days")
def test_public_intraday_download():
    df = public_intraday_or_skip()
    assert df.height > 1000
    assert (np.diff(df["t"].to_numpy()) > 0).all()
    assert df["close"].null_count() == 0


def test_a_truncated_download_is_retried_and_a_dead_one_is_offline(monkeypatch):
    """`http.client.IncompleteRead` is not an `OSError`; it once escaped as a
    crash from a CI run instead of the retry (or the skip) it deserves."""
    import http.client
    import io

    import data

    calls: list[str] = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise http.client.IncompleteRead(b"partial")
        return _Resp(b"whole")

    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    monkeypatch.setattr(data.urllib.request, "urlopen", flaky)
    assert data._download("http://example.invalid/x") == b"whole"
    assert len(calls) == 3

    def dead(url, timeout):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(data.urllib.request, "urlopen", dead)
    with pytest.raises(RuntimeError, match="offline"):
        data._download("http://example.invalid/x")


def _outcome(fetch):
    """What a fetch does: ``("skipped", reason)``, ``(exception name,
    exception)``, or ``("returned", None)``."""
    try:
        fetch()
    except pytest.skip.Exception as e:
        return "skipped", str(e)
    except Exception as e:
        return type(e).__name__, e
    return "returned", None


@pytest.mark.parametrize(
    "fetch",
    [
        lambda data: data.public_intraday_or_skip(("1999-01-01",)),
        lambda data: data.public_quotes_and_trades_or_skip("ZZZUSDT", "1999-01-01"),
    ],
    ids=["intraday", "quotes"],
)
def test_only_the_network_is_offline(monkeypatch, tmp_path, fetch):
    """Review 2026-10-05, TA3 and TB2. An HTTP error is an ``OSError``, so a
    404 or a 403 was retried and then called offline, and the test skipped
    instead of failing: a moved file would have hidden its tests for good.
    A 4xx is the server's answer, not the network's: it is raised at once.
    A 5xx or a 429 is retried as a dropped connection is, and offline after
    the last attempt; only offline skips."""
    import io
    import urllib.error

    import data

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    calls: list[str] = []

    def status(code: int):
        def urlopen(url, timeout):
            calls.append(url)
            raise urllib.error.HTTPError(url, code, "status", {}, io.BytesIO(b""))

        return urlopen

    def unreachable(url, timeout):
        calls.append(url)
        raise urllib.error.URLError("network unreachable")

    for code in (403, 404, 410):
        calls.clear()
        monkeypatch.setattr(data.urllib.request, "urlopen", status(code))
        kind, err = _outcome(lambda: fetch(data))
        assert kind == "HTTPError" and err.code == code, (code, kind, err)
        assert len(calls) == 1, f"a {code} was retried"
    for answer in (status(500), status(503), status(429), unreachable):
        calls.clear()
        monkeypatch.setattr(data.urllib.request, "urlopen", answer)
        kind, reason = _outcome(lambda: fetch(data))
        assert kind == "skipped" and "offline" in reason, (kind, reason)
        assert len(calls) == 3, "a transient failure was not retried"
