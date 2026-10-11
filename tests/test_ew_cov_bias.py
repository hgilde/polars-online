"""``ew_cov(bias=)``: Polars' name, default and correction (PLAN task 244).

``bias=False``, the default as in Polars' ``ewm_var`` and ``ewm_std``,
multiplies every second moment by ``c = W**2 / (W**2 - Q)``, ``W`` the
weight and ``Q`` the sum of squared weights behind the moments: Polars'
correction in ``polars-compute``'s ``ewm/cov.rs`` (``EwmCovState``), the
reliability weights' ``V1**2 / (V1**2 - V2)``, and numpy's
``cov(aweights=w, ddof=1)``. ``bias=True`` is the weighted mean the model
keeps. The oracles here are Polars itself on a clock of one row a step, the
same over a frame with a null at every tick the clock skips (Polars decays
over a null under ``ignore_nulls=False``), Polars' rolling windows weighted
by hand, and numpy for row weights, zero-weight rows and windows. ``ew_cov``
reads a row's statistics before learning it, so its row ``i`` is the
reference's row ``i - 1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

#: Each value is held to 1e-12 of the reference, relative for a variance and
#: a standard deviation, and of ``sqrt(var_a * var_b)`` for a covariance,
#: which crosses zero.
TOL = 1e-12


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    a = 3.0 * rng.standard_normal(n) + 1.0
    b = 0.4 * a + rng.standard_normal(n) - 2.0
    return pl.DataFrame({"a": a, "b": b})


def _fit(df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
    spec = po.spec.ew_cov("m", features=["a", "b"], min_weight=0.0, **kw)
    return po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()


def _close(got: np.ndarray, want: np.ndarray, scale: np.ndarray, what: str) -> None:
    live = np.isfinite(got)
    assert live.sum() > 0.9 * len(got), (what, int(live.sum()))
    assert np.isfinite(want[live]).all(), what
    err = np.max(np.abs(got[live] - want[live]) / scale[live])
    assert err <= TOL, (what, err)


def _polars_moments(frame: pl.DataFrame, **kw: Any) -> pl.DataFrame:
    """Polars' variances of ``a`` and ``b`` and, by polarization, their
    covariance, ``(var(a + b) - var(a - b)) / 4``: the four sums share the
    weights, so the correction is the same factor in each."""
    return frame.select(
        var_a=pl.col("a").ewm_var(**kw),
        var_b=pl.col("b").ewm_var(**kw),
        std_a=pl.col("a").ewm_std(**kw),
        cov_a_b=(
            (pl.col("a") + pl.col("b")).ewm_var(**kw) - (pl.col("a") - pl.col("b")).ewm_var(**kw)
        )
        / 4.0,
    )


def _as_read(col: pl.Series) -> np.ndarray:
    """A reference column moved one row on, as ``ew_cov`` reads it."""
    return np.concatenate([[np.nan], col.to_numpy().astype(float)[:-1]])


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("half_life", [2.0, 7.5, 20.0, 200.0])
def test_var_std_and_cov_are_polars_ewm_var_and_ewm_std(bias: bool, half_life: float) -> None:
    """On a clock of one row a step, Polars' ``ewm_var``, ``ewm_std`` and
    the covariance their ``ewm_var`` gives by polarization, with its default
    ``adjust=True``, at a half-life of 2 rows, where the correction is 21%
    in steady state, up to 200."""
    df = _frame(600, seed=31)
    got = _fit(df, half_life=half_life, bias=bias, stats=["var", "std", "cov"])
    want = _polars_moments(df, half_life=half_life, bias=bias)
    var_a, var_b = _as_read(want["var_a"]), _as_read(want["var_b"])
    for name in ("var_a", "var_b", "std_a"):
        ref = _as_read(want[name])
        _close(got[name].to_numpy().astype(float), ref, np.abs(ref), f"{name} h={half_life}")
    _close(
        got["cov_a_b"].to_numpy().astype(float),
        _as_read(want["cov_a_b"]),
        np.sqrt(var_a * var_b),
        f"cov h={half_life}",
    )


def test_the_default_is_unbiased_and_one_number_moves_by_the_correction() -> None:
    """The reviewer's case: 400 rows of N(0, 1) at a half-life of 20. The
    default is Polars' default (0.782974 at the last read), ``bias=True`` its
    ``bias=True`` (0.769407), and the two differ by ``n_kish / (n_kish - 1)``
    at the Kish size ``(1 + lam) / (1 - lam)``, 57.7 rows in steady state."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    spec = po.spec.ew_cov("c", features=["a", "b"], half_life=20, stats=["var", "std", "cov"])
    assert spec["model"]["bias"] is False
    unbiased = po.ModelBank([spec]).fit_predict(df)["c"].struct.field("var_a")
    biased = _fit(df, half_life=20, bias=True, stats=["var"])["var_a"]
    assert unbiased[-1] == pytest.approx(df["a"][:-1].ewm_var(half_life=20)[-1], rel=1e-12)
    assert biased[-1] == pytest.approx(df["a"][:-1].ewm_var(half_life=20, bias=True)[-1], rel=1e-12)
    assert round(unbiased[-1], 6) == 0.782974 and round(biased[-1], 6) == 0.769407
    lam = 0.5 ** (1 / 20)
    n_kish = (1 + lam) / (1 - lam)
    assert unbiased[-1] / biased[-1] == pytest.approx(n_kish / (n_kish - 1), rel=1e-6)


@pytest.mark.parametrize("bias", [False, True])
def test_an_irregular_clock_is_polars_over_a_null_at_every_skipped_tick(bias: bool) -> None:
    """On an integer clock that skips ticks, the moments are Polars' over a
    frame with a row at every tick and a null where the stream has none:
    under ``ignore_nulls=False`` a null decays the weight and the sum of its
    squares, as a clock step of more than one does here. Polars has no
    ``ewm_var_by``; this is its decay on the same clock."""
    rng = np.random.default_rng(32)
    steps = rng.integers(1, 5, 500)
    steps[0] = 0
    t = np.cumsum(steps)
    df = _frame(500, seed=33).with_columns(pl.Series("t", t, dtype=pl.Int64))
    half_life = 6.0
    got = _fit(df, half_life=half_life, bias=bias, clock="t", gap_cap=1e9, stats=["var", "cov"])
    ticks = pl.DataFrame({"t": np.arange(t[-1] + 1, dtype=np.int64)}).join(df, on="t", how="left")
    want = _polars_moments(ticks, half_life=half_life, bias=bias, ignore_nulls=False)
    want = pl.concat([ticks.select("t"), want], how="horizontal").filter(
        pl.col("t").is_in(t.tolist())
    )
    assert want.height == df.height
    var_a, var_b = _as_read(want["var_a"]), _as_read(want["var_b"])
    _close(got["var_a"].to_numpy().astype(float), var_a, np.abs(var_a), "var_a")
    _close(got["var_b"].to_numpy().astype(float), var_b, np.abs(var_b), "var_b")
    _close(
        got["cov_a_b"].to_numpy().astype(float),
        _as_read(want["cov_a_b"]),
        np.sqrt(var_a * var_b),
        "cov",
    )


def _weighted_rows(n: int, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Rows on an irregular float clock with weights that are 0 on a fifth
    of the rows, the first two among them, and spread over two decades
    elsewhere."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.25, 2.5, n))
    w = 10.0 ** rng.uniform(-1.0, 1.0, n)
    w[rng.random(n) < 0.2] = 0.0
    w[:2] = 0.0
    return _frame(n, seed).with_columns(pl.Series("t", t), pl.Series("w", w)), t


@pytest.mark.parametrize("window", [None, 12.0])
@pytest.mark.parametrize("bias", [False, True])
def test_row_weights_and_windows_are_numpy_cov_with_aweights(
    bias: bool, window: float | None
) -> None:
    """With row weights, rows of weight 0 and a ``window_size``, the moments
    are ``numpy.cov`` of the rows inside the window under ``aweights``, each
    row's weight times ``0.5 ** (age / half_life)``: ``ddof=1`` is the
    correction, ``ddof=0`` the weighted mean. The principal component's
    variance is ``numpy.linalg.eigh`` of the same matrix."""
    df, t = _weighted_rows(400, seed=34)
    h = 5.0
    got = _fit(
        df,
        half_life=h,
        bias=bias,
        clock="t",
        gap_cap=1e9,
        weight="w",
        window_size=window,
        stats=["var", "std", "cov"],
        pca=1,
        pca_every=0,
    )
    x = df.select("a", "b").to_numpy()
    w = df["w"].to_numpy()
    checked = 0
    for i in range(20, len(t)):
        age = t[i - 1] - t[:i]
        keep = (w[:i] > 0) & (np.ones(i, bool) if window is None else age < window)
        if keep.sum() < 3:
            continue
        aw = w[:i][keep] * 0.5 ** (age[keep] / h)
        if aw.sum() < 2.0:
            # Below the floor the bank puts under `min_weight`.
            assert got["var_a"][i] is None, i
            continue
        cov = np.cov(x[:i][keep].T, aweights=aw, ddof=0 if bias else 1)
        scale = np.sqrt(cov[0, 0] * cov[1, 1])
        assert abs(got["var_a"][i] - cov[0, 0]) <= TOL * cov[0, 0], i
        assert abs(got["std_b"][i] - np.sqrt(cov[1, 1])) <= TOL * np.sqrt(cov[1, 1]), i
        assert abs(got["cov_a_b"][i] - cov[0, 1]) <= TOL * scale, i
        # The components were refreshed after the row before, from the same
        # moments.
        assert got["pc0_var"][i] == pytest.approx(np.linalg.eigh(cov)[0][-1], rel=1e-10), i
        checked += 1
    assert checked > 300, checked


@pytest.mark.parametrize("window", [None, 40.0])
def test_one_weighted_row_reads_null_and_leaves_nothing_behind(window: float | None) -> None:
    """Hard rule 9. One row of weight has no spread to correct: ``W**2 - Q``
    is 0, and Polars gives null. So the rows read while the history holds one
    row of weight -- the first, rows of weight 0 before it, and rows of weight
    0 after it that age it -- read null, where ``bias=True`` reads a variance
    of 0. Once a second row is learned every value is the definition's, so
    nothing undefined reached the state; ``weight_sum`` is the same under
    both."""
    w = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 1.0]
    rng = np.random.default_rng(35)
    n = len(w)
    df = pl.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "w": w})
    kw: dict[str, Any] = dict(
        half_life=3.0, weight="w", window_size=window, stats=["var", "std", "cov", "corr"]
    )
    unbiased = _fit(df, **kw)
    biased = _fit(df, bias=True, **kw)
    # Rows 3 to 6 are read with the one row of weight 5 learned (weight_sum
    # 5, 5 lam, ...), past the floor of 2.
    one_row = [3, 4, 5, 6]
    assert all(biased["weight_sum"][i] >= 2.0 for i in one_row)
    for name in ("var_a", "std_a", "cov_a_b"):
        assert unbiased[name][one_row].is_null().all(), (name, unbiased[name].to_list())
        assert (biased[name][one_row] == 0.0).all(), (name, biased[name].to_list())
    assert unbiased["weight_sum"].equals(biased["weight_sum"])
    x = df.select("a", "b").to_numpy()
    for i in range(7, n):
        keep = np.array(w[:i]) > 0
        aw = np.array(w[:i])[keep] * 0.5 ** ((i - 1 - np.arange(i)[keep]) / 3.0)
        cov = np.cov(x[:i][keep].T, aweights=aw, ddof=1)
        assert unbiased["var_a"][i] == pytest.approx(cov[0, 0], rel=1e-12), i
        assert unbiased["cov_a_b"][i] == pytest.approx(cov[0, 1], rel=1e-10, abs=1e-12), i


def test_the_correlations_and_the_weight_do_not_move_and_mahal_moves_by_the_root() -> None:
    """A common factor leaves every correlation where it was: ``corr``,
    ``partial_corr``, ``lag_corr``, and the components' shares, loadings and
    scores, to the bit or the last place. The variances move by ``c``, the
    component's variance by ``c`` as of its refresh, ``mahal`` by ``1 /
    sqrt(c)`` (the ridge on the co-moments is scaled with them), and
    ``weight_sum``, the weight before the row, not at all (hard rule 8)."""
    df = _frame(500, seed=36)
    kw: dict[str, Any] = dict(
        half_life=15.0,
        stats=["var", "corr", "partial_corr", "mahal", "lag_corr"],
        precision_prior=0.01,
        lags=[1, 2],
        pca=2,
        pca_every=0,
    )
    a = _fit(df, **kw)
    b = _fit(df, bias=True, **kw)
    exact = ["weight_sum", "corr_a_b", "partial_corr_a_b", "lag_corr_a_b_l1", "lag_corr_b_a_l2"]
    exact += [f"pc{j}_loading_{c}" for j in (0, 1) for c in ("a", "b")]
    exact += ["pc0_score", "pc1_score"]
    for name in exact:
        assert a[name].equals(b[name]), name
    np.testing.assert_allclose(
        a["pc0_share"].to_numpy().astype(float)[3:],
        b["pc0_share"].to_numpy().astype(float)[3:],
        rtol=4e-16,
    )
    lam = 0.5 ** (1 / 15.0)
    for i in range(10, df.height, 37):
        # Row i reads the moments after i rows of unit weight.
        big_w = (1 - lam**i) / (1 - lam)
        big_q = (1 - lam ** (2 * i)) / (1 - lam**2)
        c = big_w**2 / (big_w**2 - big_q)
        assert a["var_a"][i] == pytest.approx(c * b["var_a"][i], rel=1e-12), i
        assert a["pc0_var"][i] == pytest.approx(c * b["pc0_var"][i], rel=1e-12), i
        assert a["mahal"][i] == pytest.approx(b["mahal"][i] / np.sqrt(c), rel=1e-12), i


@pytest.mark.parametrize("window", [None, 25.0])
def test_chunk_invariance_and_a_mid_stream_save_to_the_bit(
    window: float | None, tmp_path: Path
) -> None:
    """Hard rule 3 under the default: 1, 7 and 600 chunks give the same bits,
    and a bank saved mid-stream and loaded goes on to the same bits as one
    that never stopped."""
    df, _ = _weighted_rows(600, seed=37)
    spec = po.spec.ew_cov(
        "m",
        features=["a", "b"],
        half_life=8.0,
        clock="t",
        gap_cap=1e9,
        weight="w",
        window_size=window,
        stats=["var", "std", "cov", "corr", "mahal"],
        precision_prior=0.01,
        pca=1,
        min_weight=0.0,
    )
    whole = po.ModelBank([spec]).fit_predict(df)
    for chunks in (7, 600):
        bank = po.ModelBank([spec])
        size = -(-df.height // chunks)
        parts = [bank.fit_predict(df.slice(o, size)) for o in range(0, df.height, size)]
        assert pl.concat(parts).equals(whole), chunks
    first = po.ModelBank([spec])
    head = first.fit_predict(df.head(311))
    first.save(tmp_path / "bank.msgpack")
    resumed = po.ModelBank.load(tmp_path / "bank.msgpack", specs=[spec])
    tail = resumed.fit_predict(df.slice(311))
    assert pl.concat([head, tail]).equals(whole)
    assert whole["m"].struct.field("var_a").null_count() < 10


def test_bias_is_a_bool_and_reaches_the_spec_and_the_cli_form() -> None:
    """``bias`` is refused by name unless a bool, defaults to ``False`` in
    the spec it writes, and a spec without it -- a TOML file's -- reads as
    the default."""
    with pytest.raises(TypeError, match="bias must be"):
        po.spec.ew_cov("m", features=["a", "b"], half_life=5.0, bias=1)  # type: ignore[arg-type]
    spec = po.spec.ew_cov("m", features=["a", "b"], half_life=5.0)
    assert spec["model"]["bias"] is False
    assert po.spec.ew_cov("m", features=["a", "b"], half_life=5.0, bias=True)["model"]["bias"]
    df = _frame(80, seed=38)
    stripped = {**spec, "model": {k: v for k, v in spec["model"].items() if k != "bias"}}
    got = po.ModelBank([stripped]).fit_predict(df)
    assert got.equals(po.ModelBank([spec]).fit_predict(df))
