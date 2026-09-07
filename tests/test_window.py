"""`window`: an exponentially weighted accumulator with a hard cutoff.

The promise is narrow and testable: with `window = w`, a row older than `w`
clock units contributes *nothing*, where the exponential weight alone would
leave `0.5^(age/halflife)` of it. Inside the window the weights are still
exponential — this is not a flat window, and the tests say so by comparing
against a weighted sum rather than a plain mean.

The oracle is polars, which computes the same thing a different way: one
`rolling().agg()` per row, `O(n·W)` where the accumulator is `O(n)`.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import polars_online as po

HALFLIFE = 40.0


def stream(n=240, seed=0, step=None):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, 7, n)).astype(np.int64) if step is None else np.arange(n) * step
    x = rng.standard_normal(n) * 2.0 + 5.0
    return pl.DataFrame({"t": t.astype(float), "x": x})


def spec(window=None, **kw):
    d = dict(
        features=["x"],
        clock="t",
        halflife=HALFLIFE,
        max_dclock=1e12,
        min_periods=0.0,
        stats=["mean"],
    )
    d.update(kw)
    return po.spec.ew_cov("w", window=window, **d)


def run(df, window=None, chunks=1, **kw):
    bank = po.ModelBank([spec(window, **kw)])
    parts = [df] if chunks == 1 else [d for d in df.iter_slices(max(1, len(df) // chunks))]
    return pl.concat([bank.fit_predict(p) for p in parts]).unnest("w")


def oracle(df, window):
    """The same statistic in polars: exponential weights over the rows inside
    the window, read at the previous row's clock, which is where every
    statistic in this library is referenced."""
    lam = 0.5 ** (1 / HALFLIFE)
    t, x = df["t"].to_numpy(), df["x"].to_numpy()
    out = []
    for i in range(len(t)):
        if i == 0:
            out.append(None)
            continue
        ref = t[i - 1]
        keep = (t[:i] >= ref - window) if window is not None else np.ones(i, bool)
        if not keep.any():
            out.append(None)
            continue
        w = lam ** (ref - t[:i][keep])
        out.append(float((w * x[:i][keep]).sum() / w.sum()))
    return out


def test_the_window_is_the_weighted_mean_of_what_it_covers():
    df = stream()
    for window in (20.0, 90.0, 400.0):
        got = run(df, window)["mean_x"].to_list()
        want = oracle(df, window)
        # `ew_cov` withholds a mean until two effective rows, so the first
        # rows are null by design; everything after must be the definition.
        nulls = 0
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            if a is None or b is None:
                nulls += 1
                continue
            assert a == pytest.approx(b, abs=1e-9), f"window {window} row {i}"
        assert nulls <= 3, f"window {window}: {nulls} rows reported nothing"


def test_a_row_older_than_the_window_cannot_move_the_answer():
    """The guarantee, stated as an experiment: replace everything outside the
    window with a number that would swamp any average it still touched."""
    df = stream(step=1.0)
    window = 60.0
    clean = run(df, window)
    poisoned = df.with_columns(
        x=pl.when(pl.col("t") < df["t"][-1] - 2 * window).then(1e6).otherwise(pl.col("x"))
    )
    dirty = run(poisoned, window)
    assert clean["mean_x"][-1] == pytest.approx(dirty["mean_x"][-1], abs=1e-8)
    # ... and the plain accumulator is wrecked by the same data, which is what
    # makes the guarantee worth having.
    assert run(poisoned)["mean_x"][-1] > 1e4


def test_one_chunk_and_many_agree():
    df = stream()
    one = run(df, 90.0, chunks=1)
    many = run(df, 90.0, chunks=40)
    assert one.equals(many)


def test_a_window_no_stream_reaches_is_the_plain_accumulator():
    df = stream()
    assert run(df, 1e12).equals(run(df, None))


def test_a_saved_bank_resumes_mid_window(tmp_path):
    df = stream()
    half = len(df) // 2
    whole = run(df, 90.0)

    bank = po.ModelBank([spec(90.0)])
    bank.fit_predict(df[:half])
    path = tmp_path / "w.state"
    bank.save(path)
    resumed = po.ModelBank.load(path)
    second = resumed.fit_predict(df[half:]).unnest("w")
    assert second.equals(whole[half:])


def test_the_window_shrinks_n_eff_to_what_it_covers():
    df = stream(step=1.0)
    lam = 0.5 ** (1 / HALFLIFE)
    windowed = run(df, 60.0)["n_eff"][-1]
    plain = run(df)["n_eff"][-1]
    # A geometric sum over the window, against one over the whole stream.
    assert windowed == pytest.approx((1 - lam**60) / (1 - lam), rel=0.02)
    assert plain > windowed * 1.4


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"window": 0.0}, "window must be finite and > 0"),
        ({"window": -1.0}, "window must be finite and > 0"),
        ({"window": float("inf")}, "window must be finite"),
        ({"window_every": 5}, "window_every needs"),
        ({"window": 10.0, "window_every": 0}, "window_every must be >= 1"),
        ({"window": 10.0, "lags": [1]}, "window and lags do not combine"),
        ({"window": 10.0, "mahal_quantiles": [0.99]}, "mahal_quantiles"),
    ],
)
def test_a_bad_window_is_refused_by_name(kw, msg):
    with pytest.raises(Exception, match=re.escape(msg)):
        po.ModelBank([spec(**kw)]).fit_predict(stream(20))


@pytest.mark.parametrize("model", ["rls", "kalman", "sgd", "holt", "hmm", "bocpd"])
def test_only_the_models_that_can_honour_it_accept_it(model):
    """The identity holds where the state is a sum of per-row contributions.
    Everywhere else the keyword is refused, naming the model, rather than
    accepted and quietly ignored."""
    with pytest.raises(TypeError, match=f"{model}.*unexpected keyword argument 'window'"):
        getattr(po.spec, model)("m", targets=["y"], features=["x"], window=10.0)


# --- the same cutoff on a regression -----------------------------------------


def ridge_spec(window=None, **kw):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        halflife=HALFLIFE,
        max_dclock=1e12,
        min_periods=2.0,
        ridge=1e-8,
        max_rows_between_solves=1,
    )
    d.update(kw)
    return po.spec.ewridge("w", window=window, **d)


def regime_stream(n=300, flip=200, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.where(np.arange(n) < flip, 3.0 * x + 1.0, -2.0 * x + 5.0)
    return pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": y})


def fit(df, window=None, chunks=1, **kw):
    bank = po.ModelBank([ridge_spec(window, **kw)])
    parts = [df] if chunks == 1 else list(df.iter_slices(max(1, len(df) // chunks)))
    for p in parts:
        bank.fit_predict(p)
    return bank.coef("w")["coef"].to_list()


def test_a_windowed_fit_forgets_the_regime_the_window_excludes():
    """The point of the feature, as an experiment: a relationship that ended
    before the window cannot bend the coefficients inside it."""
    df = regime_stream()
    intercept, slope = fit(df, window=40.0)
    assert slope == pytest.approx(-2.0, abs=1e-4)
    assert intercept == pytest.approx(5.0, abs=1e-4)
    # Without one, the decayed tail of the old regime is still in the fit.
    _, plain_slope = fit(df)
    assert abs(plain_slope + 2.0) > 0.5, "the plain fit should still be contaminated"


def test_the_windowed_fit_is_the_weighted_least_squares_of_its_rows():
    """Against the normal equations over exactly the rows inside the window."""
    df = regime_stream(n=260, flip=170, seed=4)
    window = 50.0
    got = fit(df, window=window)
    t, x, y = df["t"].to_numpy(), df["x"].to_numpy(), df["y"].to_numpy()
    now, lam = t[-1], 0.5 ** (1 / HALFLIFE)
    keep = (now - t) < window
    w = lam ** (now - t[keep])
    z = np.column_stack([np.ones(keep.sum()), x[keep]])
    wz = z * w[:, None]
    beta = np.linalg.solve(wz.T @ z + 1e-8 * w.sum() * np.eye(2), wz.T @ y[keep])
    assert got == pytest.approx(list(beta), rel=1e-5)


def test_a_windowed_fit_is_chunk_invariant():
    df = regime_stream()
    assert fit(df, 60.0, chunks=1) == fit(df, 60.0, chunks=30)


def test_a_windowed_fit_resumes_from_a_saved_bank(tmp_path):
    df = regime_stream()
    half = len(df) // 2
    whole = fit(df, 60.0)
    bank = po.ModelBank([ridge_spec(60.0)])
    bank.fit_predict(df[:half])
    bank.save(tmp_path / "r.state")
    resumed = po.ModelBank.load(tmp_path / "r.state")
    resumed.fit_predict(df[half:])
    assert resumed.coef("w")["coef"].to_list() == whole


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"ridge_decay": True}, "window and ridge_decay do not combine"),
        (
            {
                "session_shrink": 0.5,
                "long_halflife": 500.0,
                "session": "t",
                "session_gap": 10.0,
            },
            "window and session_shrink",
        ),
    ],
)
def test_a_window_is_refused_where_the_identity_does_not_hold(kw, msg):
    with pytest.raises(Exception, match=re.escape(msg)):
        po.ModelBank([ridge_spec(30.0, **kw)]).fit_predict(regime_stream(40))


# --- and on a path that selects ----------------------------------------------


def lasso_spec(window=None, **kw):
    d = dict(
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        halflife=HALFLIFE,
        max_dclock=1e12,
        min_periods=2.0,
        lasso_path=[0.05],
        max_rows_between_solves=1,
    )
    d.update(kw)
    return po.spec.lasso("w", window=window, **d)


def two_regime_features(n=300, flip=200, seed=2):
    """`x0` drives the stream until `flip`, `x1` after it."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    y = np.where(np.arange(n) < flip, 3.0 * x0, 2.5 * x1)
    return pl.DataFrame({"t": np.arange(n).astype(float), "x0": x0, "x1": x1, "y": y})


def lasso_coef(df, window=None, chunks=1):
    bank = po.ModelBank([lasso_spec(window)])
    for part in [df] if chunks == 1 else list(df.iter_slices(max(1, len(df) // chunks))):
        bank.fit_predict(part)
    return bank.coef("w")["coef"].to_list()


def test_a_windowed_path_drops_the_support_the_window_excludes():
    """The window changes which features are *selected*, not just their size:
    with no evidence for `x0` inside it, the penalty takes it to exactly
    zero, where the decayed fit still carries it."""
    df = two_regime_features()
    _, x0, x1 = lasso_coef(df, window=40.0)
    assert x0 == 0.0, "a feature with no in-window evidence should be dropped"
    assert x1 == pytest.approx(2.5, abs=0.1)
    _, plain_x0, _ = lasso_coef(df)
    assert abs(plain_x0) > 0.3, "the plain path should still carry the stale feature"


def test_a_windowed_path_is_chunk_invariant():
    df = two_regime_features()
    assert lasso_coef(df, 60.0, chunks=1) == lasso_coef(df, 60.0, chunks=25)


def test_a_lasso_window_no_stream_reaches_is_the_plain_path():
    df = two_regime_features()
    assert lasso_coef(df, 1e12) == lasso_coef(df)


# --- and on the pairwise screen ----------------------------------------------


def test_a_windowed_screen_sees_the_relationship_the_full_history_cancels():
    """Two regimes of opposite sign average to nothing, so the unwindowed
    screen reports no relationship where there is a strong one."""
    n, flip = 300, 200
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n)
    y = np.where(np.arange(n) < flip, 2.0 * x, -1.0 * x) + rng.standard_normal(n) * 0.1
    df = pl.DataFrame({"t": np.arange(n).astype(float), "x": x, "y": y})

    def screen(window):
        spec = po.spec.marginal(
            "m",
            targets=["y"],
            features=["x"],
            clock="t",
            # Slow enough that the old regime is still half the weight: that
            # is what makes the two signs cancel.
            halflife=60.0,
            max_dclock=1e12,
            min_periods=2.0,
            window=window,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        return bank.marginal("m")

    windowed = screen(40.0)
    assert windowed["beta"][0] == pytest.approx(-1.0, abs=0.05)
    assert windowed["corr"][0] < -0.9
    # The full history cancels the two regimes to nothing at all.
    assert abs(screen(None)["corr"][0]) < 0.1
