"""E57: a block's realised covariance, robust to microstructure noise.

Three estimators, each held to its own definition written out in numpy on
the same returns — not to a golden number, and not to each other. What ties
them together is `plain`: it equals `n` times an `ew_cov(lam=1)`'s uncentred
second moment **to the bit**, so it is the reference the noise-robust two are
measured against, and the gap between them is the thing they remove.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

COLS = ["x0", "x1"]


def ticks(n=600, k=2, noise=0.002, seed=0, blocks=2):
    """An efficient price plus i.i.d. noise, differenced: the returns a noisy
    tick stream produces, split into `blocks` groups.

    Two by default: the last group never closes, so a single-block frame
    emits nothing and every test here reads group ``0``."""
    rng = np.random.default_rng(seed)
    level = np.cumsum(rng.standard_normal((n, k)) * 0.01, axis=0)
    obs = level + noise * rng.standard_normal((n, k))
    ret = np.diff(obs, axis=0, prepend=obs[:1])
    return pl.DataFrame(
        {**{f"x{i}": ret[:, i] for i in range(k)}, "b": np.arange(n) // (n // blocks)}
    )


def spec(**kw):
    d = {
        "features": COLS,
        "group": "b",
        "group_close": "monotone",
        "block_rows": 600,
    }
    d.update(kw)
    return po.spec.rcov("r", **d)


def block(df, **kw):
    bank = po.ModelBank([spec(**kw)])
    bank.fit_predict(df)
    return bank.closed_groups(drop=False), bank


def unvech(v, k):
    m = np.zeros((k, k))
    m[np.triu_indices(k)] = v
    return m + np.triu(m, 1).T


# --- plain: the reference ----------------------------------------------------


def test_plain_is_n_times_the_raw_second_moment_to_the_bit():
    df = ticks(blocks=2)
    rows, _ = block(df, kind="plain")
    first = df.filter(pl.col("b") == 0)
    ref = po.ModelBank([po.spec.ew_cov("c", features=COLS, lam=1.0, stats=[])])
    ref.fit_predict(first.select(COLS))
    g = ref.gram("c")[0]
    raw = g["comoments"] + np.outer(g["means"], g["means"])
    want = raw * first.height
    got = unvech(rows["rcov"][0].to_list(), 2)
    assert np.allclose(got, want, rtol=1e-9, atol=0)
    assert rows["rcov_n"][0] == first.height


def test_the_noise_robust_kinds_differ_from_plain_and_from_each_other():
    """Not a tolerance test: the whole point is that they disagree, because
    `plain` is biased upward by the noise."""
    df = ticks(noise=0.01)
    var = {}
    for kind in ("plain", "kernel", "preavg"):
        rows, _ = block(df, kind=kind)
        var[kind] = unvech(rows["rcov"][0].to_list(), 2)[0, 0]
    assert var["plain"] > var["kernel"] * 1.5, var
    assert var["plain"] > var["preavg"] * 1.5, var


# --- the kernel --------------------------------------------------------------


def effective(ret, m):
    """The jittered effective return series, longhand."""
    n = len(ret)
    lead = sum((1.0 if i == m else i / m) * ret[i - 1] for i in range(1, m + 1))
    trail = sum((n - i) / m * ret[i] for i in range(n - m, n))
    return np.array([lead, *list(ret[m : n - m]), trail])


def parzen(x):
    x = abs(x)
    if x <= 0.5:
        return 1 - 6 * x**2 + 6 * x**3
    if x <= 1.0:
        return 2 * (1 - x) ** 3
    return 0.0


def test_the_kernel_is_its_definition():
    df = ticks(n=300)
    h, m = 4, 2
    rows, _ = block(df, kind="kernel", bandwidth=h, jitter=m)
    ret = df.filter(pl.col("b") == 0).select(COLS).to_numpy()
    y = effective(ret, m)
    want = np.zeros((2, 2))
    for lag in range(h + 1):
        w = parzen(lag / (h + 1))
        g = sum(np.outer(y[t], y[t - lag]) for t in range(lag, len(y)))
        want += w * g
        if lag:
            want += w * g.T
    got = unvech(rows["rcov"][0].to_list(), 2)
    assert np.allclose(got, (want + want.T) / 2, rtol=1e-9, atol=1e-15)
    assert rows["rcov_n"][0] == len(y)
    assert rows["bandwidth_used"][0] == h


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_the_jitter_moves_the_estimate_barely(m):
    """BNHLS: `m = 1` is mean-square optimal and `m = 1..4` move the
    estimate by under 0.5 %, which is why the default of 2 is immaterial.

    That is a claim about a *block*, not about a handful of rows: the jitter
    is an end effect, and at 300 returns per block it is worth 0.8 %. At
    1000 it is 0.11 %, and the paper's bound holds."""
    df = ticks(n=2000)
    kw = {"kind": "kernel", "bandwidth": 6, "block_rows": 1000}
    rows, _ = block(df, jitter=m, **kw)
    got = unvech(rows["rcov"][0].to_list(), 2)[0, 0]
    base, _ = block(df, jitter=1, **kw)
    ref = unvech(base["rcov"][0].to_list(), 2)[0, 0]
    assert abs(got / ref - 1.0) < 0.005, (m, got, ref)


def test_the_auto_bandwidth_is_reported_and_clipped_to_the_ring():
    df = ticks(n=400)
    rows, _ = block(df, kind="kernel", block_rows=400)
    h = rows["bandwidth_used"][0]
    assert h is not None and h >= 1
    assert rows["omega2"][0].to_list()[0] > 0
    assert rows["iv_sparse"][0].to_list()[0] > 0
    # A tiny ring clips it.
    tight, _ = block(df, kind="kernel", block_rows=400, max_bandwidth=2)
    assert tight["bandwidth_used"][0] == 2


# --- pre-averaging -----------------------------------------------------------


def test_the_preaveraged_estimate_is_its_definition():
    df = ticks(n=400)
    kn = 12
    rows, _ = block(df, kind="preavg", preavg_rows=kn, psd=False)
    ret = df.filter(pl.col("b") == 0).select(COLS).to_numpy()
    n = len(ret)
    g = np.array([min(j / kn, 1 - j / kn) for j in range(kn)])
    psi1 = kn * sum(
        (min(i / kn, 1 - i / kn) - min((i - 1) / kn, 1 - (i - 1) / kn)) ** 2
        for i in range(1, kn + 1)
    )
    psi2 = sum(min(i / kn, 1 - i / kn) ** 2 for i in range(1, kn)) / kn
    ybar = np.array(
        [sum(g[j] * ret[start + j] for j in range(1, kn)) for start in range(n - kn + 1)]
    )
    want = n / (n - kn + 2) / (psi2 * kn) * sum(np.outer(y, y) for y in ybar) - psi1 / psi2 / (
        2 * n
    ) * sum(np.outer(x, x) for x in ret)
    got = unvech(rows["rcov"][0].to_list(), 2)
    assert np.allclose(got, (want + want.T) / 2, rtol=1e-9, atol=1e-15)
    assert rows["rcov_n"][0] == len(ybar)


def test_the_psd_form_is_a_longer_window_without_the_bias_term():
    df = ticks(n=400)
    strict, _ = block(df, kind="preavg", psd=False, block_rows=400)
    repaired, _ = block(df, kind="preavg", psd=True, block_rows=400)
    # A longer window means fewer pre-averaged blocks.
    assert repaired["rcov_n"][0] < strict["rcov_n"][0]


# --- the shared contract -----------------------------------------------------


@pytest.mark.parametrize("kind", ["plain", "kernel", "preavg"])
@pytest.mark.parametrize("size", [1, 17, 250, 600])
def test_chunk_invariance(kind, size):
    df = ticks(n=600, blocks=2)
    want, _ = block(df, kind=kind)
    bank = po.ModelBank([spec(kind=kind)])
    for i in range(0, df.height, size):
        bank.fit_predict(df[i : i + size])
    assert bank.closed_groups().equals(want)


@pytest.mark.parametrize("kind", ["plain", "kernel", "preavg"])
def test_save_load_mid_block(kind, tmp_path):
    df = ticks(n=400, blocks=2)
    want, _ = block(df, kind=kind)
    s = spec(kind=kind)
    bank = po.ModelBank([s])
    bank.fit_predict(df[:150])
    path = tmp_path / "b.state"
    bank.save(path)
    again = po.ModelBank.load(path, [s])
    again.fit_predict(df[150:])
    assert again.closed_groups().equals(want)


def test_a_zero_weight_row_is_not_a_return():
    df = ticks(n=300).with_columns(w=pl.lit(1.0))
    want, _ = block(df, kind="kernel", weight="w")
    padded = pl.concat(
        [
            df[:150],
            df[:1].with_columns(x0=pl.lit(1e6), x1=pl.lit(-1e6), w=pl.lit(0.0)),
            df[150:],
        ]
    )
    got, _ = block(padded, kind="kernel", weight="w")
    # The row was fed, so the summary counts it; nothing else moved.
    block_cols = ["rcov", "rcorr", "rcov_n", "bandwidth_used", "n_eff"]
    assert got.select(block_cols).equals(want.select(block_cols))
    assert got["rows_fed"][0] == want["rows_fed"][0] + 1


def test_a_short_block_gives_nulls():
    df = ticks(n=6, blocks=3)
    rows, _ = block(df, kind="kernel", jitter=3, block_rows=6)
    assert rows["rcov"][0] is None and rows["rcorr"][0] is None
    assert rows["rcov_n"][0] == 0


def test_the_correlation_is_the_covariance_scaled():
    df = ticks(n=400)
    rows, _ = block(df, kind="plain")
    cov = unvech(rows["rcov"][0].to_list(), 2)
    corr = unvech(rows["rcorr"][0].to_list(), 2)
    sd = np.sqrt(np.diag(cov))
    assert np.allclose(corr, cov / np.outer(sd, sd), rtol=1e-12)
    assert np.allclose(np.diag(corr), 1.0)


def test_the_block_survives_a_refresh_time_grid():
    """The pipeline E57 is for: asynchronous ticks, a refresh-time grid,
    differences, then a block estimate."""
    from polars_online import prep

    rng = np.random.default_rng(4)
    rows = []
    for s, rate in (("a", 1.0), ("b", 0.6)):
        t = np.cumsum(rng.exponential(1.0 / rate, 400))
        rows.append(
            pl.DataFrame(
                {"series": [s] * 400, "t": t, "px": np.cumsum(rng.standard_normal(400) * 0.01)}
            )
        )
    long = pl.concat(rows).sort("t")
    grid = prep.refresh_time(
        long, series="series", names=["a", "b"], time="t", value="px"
    ).collect()
    ret = grid.select(
        pl.col("a_value").diff().fill_null(0.0).alias("x0"),
        pl.col("b_value").diff().fill_null(0.0).alias("x1"),
        pl.lit(0).alias("b"),
    )
    # Two blocks so the first one closes.
    ret = pl.concat([ret, ret.with_columns(b=pl.lit(1))])
    rows_out, _ = block(ret, kind="kernel", block_rows=grid.height)
    assert rows_out["rcov"][0] is not None
    assert rows_out["rcov_n"][0] > 0


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        # `group_close`'s own check reaches a missing group first, and says
        # the same thing about it.
        ({"group": None}, "group_close needs a group column"),
        ({"group_close": None}, "needs `group` and `group_close`"),
        ({"halflife": 50.0}, "do not apply to rcov"),
        ({"lam": 0.99}, "do not apply to rcov"),
        ({"kernel": "bartlett"}, "not consistent"),
        ({"kind": "nope"}, "unknown rcov kind"),
        ({"jitter": 0}, "jitter must be >= 1"),
        ({"block_rows": None}, "needs `block_rows`"),
        ({"kind": "plain", "bandwidth": 3}, "bandwidth applies to"),
        ({"kind": "kernel", "preavg_rows": 3}, "window applies to"),
        ({"emit_sigma": True}, "does not apply to rcov"),
        # docs/REVIEW-E54-E64.md R9: settings that used to be accepted and
        # then quietly gave a block that never accumulates, or a kernel with
        # no lags in its ring.
        ({"kind": "preavg", "preavg_rows": 0}, "window must be >= 2"),
        ({"kind": "preavg", "preavg_rows": 1}, "window must be >= 2"),
        ({"block_rows": 0}, "block_rows is the block's expected length"),
        ({"max_bandwidth": 0, "bandwidth": 4}, "caps the ring below bandwidth"),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


def test_a_clock_break_splits_the_block_into_stretches():
    """A gap over ``max_dclock`` says the returns on either side of it are
    not adjacent, and a covariance of adjacent returns is the statistic. The
    block is then the two stretches added, and every return is still emitted
    exactly once (docs/REVIEW-E54-E64.md R2).

    The tail used to be dropped instead: the returns waiting in the
    end-jitter ring were lost and the first return after the gap was emitted
    ``jitter + 1`` times.
    """
    df = ticks(n=200, blocks=2)
    n = df.height
    t = np.arange(float(n))
    t[50:] += 1000.0  # one gap, inside the first block
    df = df.with_columns(t=pl.Series(t))
    kw = dict(kind="kernel", bandwidth=0, max_bandwidth=0, clock="t", max_dclock=5.0)

    broken, _ = block(df, **kw)
    row = broken.filter(pl.col("group") == "0").row(0, named=True)

    # The same two stretches as blocks of their own, with no gap in either.
    halves = [df.head(50), df.slice(50, 50)]
    got = np.zeros(3)
    total = 0
    for i, half in enumerate(halves):
        piece = half.with_columns(b=pl.lit(i, dtype=pl.Int64), t=pl.lit(None, dtype=pl.Float64))
        piece = pl.concat([piece, piece.tail(1).with_columns(b=pl.lit(9, dtype=pl.Int64))])
        out, _ = block(piece, kind="kernel", bandwidth=0, max_bandwidth=0)
        r = out.filter(pl.col("group") == str(i)).row(0, named=True)
        got += np.asarray(r["rcov"])
        total += r["rcov_n"]
    assert row["rcov_n"] == total
    assert np.allclose(row["rcov"], got)


def test_a_fractional_weight_is_refused_naming_the_row():
    df = ticks(n=50).with_columns(w=pl.when(pl.int_range(pl.len()) == 20).then(0.5).otherwise(1.0))
    bank = po.ModelBank([spec(kind="plain", weight="w")])
    with pytest.raises(ValueError, match="row 20; rcov takes 0 or 1"):
        bank.fit_predict(df)
