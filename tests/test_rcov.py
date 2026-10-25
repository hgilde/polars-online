"""E57: a block's realised covariance, robust to microstructure noise.

Three estimators, each held to its own definition written out in numpy on
the same returns — not to a golden number, and not to each other. What ties
them together is `plain`: it equals `n` times an `ew_cov(lam=1)`'s uncentred
second moment **to the bit**, so it is the reference the noise-robust two are
measured against, and the gap between them is the thing they remove.
"""

import math
import re

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

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
    assert rows["rcov_bandwidth_used"][0] == h


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
    h = rows["rcov_bandwidth_used"][0]
    assert h is not None and h >= 1
    assert rows["rcov_omega2"][0].to_list()[0] > 0
    assert rows["rcov_iv_sparse"][0].to_list()[0] > 0
    # A tiny ring clips it.
    tight, _ = block(df, kind="kernel", block_rows=400, max_bandwidth=2)
    assert tight["rcov_bandwidth_used"][0] == 2


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
    # CKP Eq. 9: Ȳ_i over Δ_{i+j}, j = 1..k_n−1, for i = 0..n−k_n+1 -- n − k_n + 2
    # terms, the first over the first k_n − 1 returns (task 159, D1: the model
    # once skipped it and scaled as if it had it).
    ybar = np.array(
        [sum(g[j] * ret[start + j - 1] for j in range(1, kn)) for start in range(n - kn + 2)]
    )
    # CKP's bias term at the theta their Eq. 7 defines: k_n / sqrt(n), the window
    # run over this block's n rows.
    theta2 = kn**2 / n
    bias = psi1 / (theta2 * psi2) / (2 * n)
    want = n / (n - kn + 2) / (psi2 * kn) * sum(np.outer(y, y) for y in ybar) - bias * sum(
        np.outer(x, x) for x in ret
    )
    # And rescaled by 1 / (1 - psi1 / (theta^2 psi2) / (2n)), their footnote 1:
    # sum(x x') holds the integrated covariance beside 2n Psi, so the bias term
    # takes that share of it away too (review 2026-10-06, CE1).
    want /= 1 - bias
    got = unvech(rows["rcov"][0].to_list(), 2)
    assert np.allclose(got, (want + want.T) / 2, rtol=1e-9, atol=1e-15)
    assert rows["rcov_n"][0] == len(ybar)


def test_the_psd_form_is_a_longer_window_without_the_bias_term():
    """``psd = True``: CKP's estimator over the longer window ``k_n =
    ceil(theta * block_rows^0.6)``, with no bias term, written out as the
    test above writes Eq. 9. It is a sum of outer products, so nothing is
    clipped. The test once checked only that ``rcov_n`` was smaller
    (review 2026-10-05, TA9)."""
    df = ticks(n=400)
    strict, _ = block(df, kind="preavg", psd=False, block_rows=400)
    repaired, _ = block(df, kind="preavg", psd=True, block_rows=400)
    ret = df.filter(pl.col("b") == 0).select(COLS).to_numpy()
    n = len(ret)  # 200: the block's own rows, half the block_rows hint
    theta = 1.0  # the default
    kn = math.ceil(theta * 400**0.6)
    assert kn == 37
    g = np.array([min(j / kn, 1 - j / kn) for j in range(kn)])
    psi2 = sum(min(i / kn, 1 - i / kn) ** 2 for i in range(1, kn)) / kn
    ybar = np.array(
        [sum(g[j] * ret[start + j - 1] for j in range(1, kn)) for start in range(n - kn + 2)]
    )
    want = n / (n - kn + 2) / (psi2 * kn) * sum(np.outer(y, y) for y in ybar)
    got = unvech(repaired["rcov"][0].to_list(), 2)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-15)
    # A longer window means fewer pre-averaged blocks.
    assert repaired["rcov_n"][0] == len(ybar) < strict["rcov_n"][0]
    assert repaired["rcov_psd_repaired"][0] is False


@pytest.mark.parametrize("kind", ["plain", "kernel"])
def test_a_negative_eigenvalue_is_clipped_and_reported(kind):
    """``rcov_psd_repaired``'s other side: a block with fewer returns than
    features has a covariance of lower rank, whose zero eigenvalues come out
    of the arithmetic a hair either side of 0. Under ``psd = True`` each
    negative one is clipped and the block says so, and the matrix reported is
    positive semi-definite and is the unrepaired estimate (``psd = False``)
    with its negative eigenvalues set to 0, by numpy's ``eigh``. A block the
    repair did not touch is the unrepaired estimate to the bit. Which of the
    sixteen null eigenvalues land below 0 is rounding, so the platform's;
    measured here, the repair fired on every such block for each estimator.
    Only ``False`` was ever asserted (review 2026-10-06, TB13)."""
    k, rows, blocks = 20, 4, 12
    rng = np.random.default_rng(5)
    ret = rng.standard_normal((rows * blocks, k)) * 0.01
    cols = [f"x{i}" for i in range(k)]
    df = pl.DataFrame(
        {**{c: ret[:, i] for i, c in enumerate(cols)}, "b": np.arange(rows * blocks) // rows}
    )
    extra = {"bandwidth": 1} if kind == "kernel" else {}

    def close(psd):
        bank = po.ModelBank([spec(features=cols, kind=kind, block_rows=rows, psd=psd, **extra)])
        bank.fit_predict(df)
        return bank.closed_groups(drop=False)

    repaired, raw = close(True), close(False)
    assert repaired.height == blocks - 1  # the last group never closes
    fired = repaired["rcov_psd_repaired"].to_list()
    assert raw["rcov_psd_repaired"].to_list() == [False] * (blocks - 1)
    assert any(fired), fired
    for got, was, did in zip(repaired["rcov"].to_list(), raw["rcov"].to_list(), fired, strict=True):
        if not did:
            assert got == was
            continue
        got, was = unvech(got, k), unvech(was, k)
        values, vectors = np.linalg.eigh(was)
        scale = values.max()
        assert np.linalg.eigvalsh(got).min() >= -1e-12 * scale
        want = (vectors * np.maximum(values, 0.0)) @ vectors.T
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * scale)


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
    block_cols = ["rcov", "rcorr", "rcov_n", "rcov_bandwidth_used", "weight_sum"]
    assert got.select(block_cols).equals(want.select(block_cols))
    assert got["rows_fed"][0] == want["rows_fed"][0] + 1


def test_a_short_block_gives_nulls():
    df = ticks(n=6, blocks=3)
    rows, _ = block(df, kind="kernel", jitter=3, block_rows=6)
    assert rows["rcov"][0] is None and rows["rcorr"][0] is None
    assert rows["rcov_n"][0] == 0


@pytest.mark.parametrize("kind", ["plain", "kernel", "preavg"])
def test_min_weight_withholds_a_block_with_fewer_returns(kind):
    """``min_weight`` on a block counts its returns: each row weighs 0 or 1,
    so the closed row's ``weight_sum`` is the returns accepted, and below the
    floor ``rcov`` and ``rcorr`` are null, as for a block too short to
    estimate from, while every other field says what the block held. It was
    accepted and did nothing: ``min_weight=1e9`` closed the same row as the
    default (task 278, the user's decision of 2026-10-25)."""
    df = ticks(n=300, blocks=2).with_columns(
        # Ten zero-weight rows in block 0: they advance the clock and are no
        # return, so the block holds 140 returns over 150 rows.
        w=pl.when(pl.int_range(pl.len()) % 15 == 7).then(0.0).otherwise(1.0)
    )
    base, _ = block(df, kind=kind, weight="w")
    assert base["weight_sum"][0] == 140.0 and base["rows_fed"][0] == 150
    assert base["rcov"][0] is not None and base["rcorr"][0] is not None
    estimate = ["rcov", "rcorr"]
    for floor, withheld in (
        (0.0, False),
        (139.0, False),
        (140.0, False),
        (141.0, True),
        (1e9, True),
    ):
        got, _ = block(df, kind=kind, weight="w", min_weight=floor)
        rest = [c for c in got.columns if c not in estimate]
        # The other fields never move, and the gate moves nothing it reads.
        assert got.select(rest).equals(base.select(rest)), floor
        if withheld:
            assert got["rcov"][0] is None and got["rcorr"][0] is None, floor
        else:
            assert got.select(estimate).equals(base.select(estimate)), floor


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
    from polars_online import stream

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
    grid = stream.refresh_time(long, series="series", names=["a", "b"], clock="t", value="px")
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
        ({"half_life": 50.0}, "do not apply to rcov"),
        ({"lam": 0.99}, "do not apply to rcov"),
        ({"kernel": "bartlett"}, "not consistent"),
        ({"kind": "nope"}, 'kind must be one of "kernel", "preavg", "plain"; got "nope"'),
        ({"jitter": 0}, "jitter must be >= 1"),
        ({"block_rows": None}, "needs `block_rows`"),
        ({"kind": "plain", "bandwidth": 3}, "bandwidth applies to"),
        ({"kind": "kernel", "preavg_rows": 3}, "preavg_rows applies to"),
        ({"emit_sigma": True}, "does not apply to rcov"),
        # docs/REVIEW-E54-E64.md R9: settings that used to be accepted and
        # then quietly gave a block that never accumulates, or a kernel with
        # no lags in its ring.
        ({"kind": "preavg", "preavg_rows": 0}, "preavg_rows must be >= 2"),
        ({"kind": "preavg", "preavg_rows": 1}, "preavg_rows must be >= 2"),
        ({"block_rows": 0}, "block_rows is the block's expected length"),
        ({"max_bandwidth": 0, "bandwidth": 4}, "caps the ring below bandwidth"),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


def test_a_clock_break_splits_the_block_into_stretches():
    """A gap over ``gap_cap`` says the returns on either side of it are
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
    kw = dict(kind="kernel", bandwidth=0, max_bandwidth=0, clock="t", gap_cap=5.0)

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


# --- phase-4 coverage: three columns (every test above is k = 2) -------------


def test_plain_at_three_columns_is_the_raw_second_moment_over_six_pairs():
    """The vech layout and the plain estimate at k = 3: six entries of a
    3x3 block, not the three of a 2x2 (review 2026-09-18, phase 4)."""
    cols3 = ["x0", "x1", "x2"]
    df = ticks(k=3, blocks=2)
    bank = po.ModelBank(
        [
            po.spec.rcov(
                "r",
                features=cols3,
                group="b",
                group_close="monotone",
                block_rows=600,
                kind="plain",
            )
        ]
    )
    bank.fit_predict(df)
    rows = bank.closed_groups(drop=False)
    first = df.filter(pl.col("b") == 0)
    ref = po.ModelBank([po.spec.ew_cov("c", features=cols3, lam=1.0, stats=[])])
    ref.fit_predict(first.select(cols3))
    g = ref.gram("c")[0]
    raw = np.array(g["comoments"]) + np.outer(g["means"], g["means"])
    want = raw * first.height
    got = unvech(rows["rcov"][0].to_list(), 3)
    assert got.shape == (3, 3)
    assert np.allclose(got, want, rtol=1e-9, atol=0)
    assert rows["rcov_n"][0] == first.height


def test_the_noise_robust_kinds_differ_from_plain_at_three_columns():
    cols3 = ["x0", "x1", "x2"]
    df = ticks(k=3, noise=0.01)
    var = {}
    for kind in ("plain", "kernel", "preavg"):
        bank = po.ModelBank(
            [
                po.spec.rcov(
                    "r",
                    features=cols3,
                    group="b",
                    group_close="monotone",
                    block_rows=600,
                    kind=kind,
                    **({"bandwidth": 4} if kind == "kernel" else {}),
                )
            ]
        )
        bank.fit_predict(df)
        var[kind] = unvech(bank.closed_groups(drop=False)["rcov"][0].to_list(), 3)[0, 0]
    assert var["plain"] > var["kernel"] * 1.5, var
    assert var["plain"] > var["preavg"] * 1.5, var


# --- the parameters Python never passed, and the relations no test held -----
#
# Task 278 (b11): `noise_stride`, `iv_stride` and `theta` were held only by
# the Rust tests, against a configuration built by hand; nothing showed them
# reaching the model through the spec. Each is shown taking effect where it
# should and nowhere else, and refused at the spec layer where it must be.


def moved(a, b):
    """The closed-row fields that differ between two runs."""
    return [c for c in a.columns if not a[c].equals(b[c])]


C_STAR = (144.0 / 0.269) ** 0.2  # BNHLS's Parzen constant


def by_offsets(ret, stride):
    """BNHLS's subsampled realised variance, longhand: for each offset ``o``
    the returns from ``o`` on, summed over complete steps of ``stride``; the
    mean over the offsets with a step of ``RV / (2 n)`` (the noise variance),
    of ``RV`` (the sparse integrated variance) and of ``(n / 3) sum v^4``
    (the quarticity proxy), per column."""
    noise, iv, iq = [], [], []
    for o in range(stride):
        steps = (len(ret) - o) // stride
        if steps == 0:
            continue
        v = ret[o : o + steps * stride].reshape(steps, stride, -1).sum(axis=1)
        rv = (v**2).sum(axis=0)
        noise.append(rv / (2 * steps))
        iv.append(rv)
        iq.append(steps / 3 * (v**4).sum(axis=0))
    return np.mean(noise, axis=0), np.mean(iv, axis=0), np.mean(iq, axis=0)


@pytest.mark.parametrize(("noise_stride", "iv_stride"), [(1, 20), (3, 5), (7, 40), (2, 1)])
def test_the_subsampled_variances_are_their_definitions(noise_stride, iv_stride):
    """The three fields behind the automatic bandwidth, at strides given
    through the spec, against their definitions over the block's returns."""
    df = ticks(n=1200)
    rows, _ = block(df, kind="kernel", noise_stride=noise_stride, iv_stride=iv_stride)
    ret = df.filter(pl.col("b") == 0).select(COLS).to_numpy()
    noise, _, _ = by_offsets(ret, noise_stride)
    _, iv, iq = by_offsets(ret, iv_stride)
    row = rows.row(0, named=True)
    np.testing.assert_allclose(row["rcov_omega2"], noise, rtol=1e-12, atol=0)
    np.testing.assert_allclose(row["rcov_iv_sparse"], iv, rtol=1e-12, atol=0)
    np.testing.assert_allclose(row["rcov_iq"], iq, rtol=1e-12, atol=0)


def test_the_noise_stride_reaches_the_estimate_through_the_automatic_bandwidth():
    """A coarser dense grid sums the noise over fewer, longer steps, so
    ``omega2`` grows, the bandwidth with it, and the estimate moves; the
    sparse grid's fields do not. Measured 2026-10-25: ``H`` 10 -> 16."""
    df = ticks(n=1200)
    base, _ = block(df, kind="kernel")
    got, _ = block(df, kind="kernel", noise_stride=3)
    assert moved(base, got) == ["rcov", "rcorr", "rcov_bandwidth_used", "rcov_omega2"]
    assert (base["rcov_bandwidth_used"][0], got["rcov_bandwidth_used"][0]) == (10, 16)
    assert all(
        a > b for a, b in zip(got["rcov_omega2"][0].to_list(), base["rcov_omega2"][0], strict=True)
    )


def test_the_iv_stride_reaches_the_estimate_through_the_automatic_bandwidth():
    """``iv_sparse`` is the denominator of BNHLS's ``xi^2``: a denser
    sparse grid lets more noise into it, so ``iv_sparse`` grows and the
    bandwidth falls; a sparser one, the reverse. The dense grid's
    ``omega2`` does not move. At this noise the default ``H`` is far enough
    from a ceiling for the stride to move it (measured 2026-10-25: 15 at
    the default 20, 10 at 1, 16 at 40; at ``noise = 0.002`` stride 5 moved
    ``iv_sparse`` too little to cross one)."""
    df = ticks(n=1200, noise=0.01)
    base, _ = block(df, kind="kernel")
    assert base["rcov_bandwidth_used"][0] == 15
    for stride, h in ((1, 10), (40, 16)):
        got, _ = block(df, kind="kernel", iv_stride=stride)
        assert moved(base, got) == [
            "rcov",
            "rcorr",
            "rcov_bandwidth_used",
            "rcov_iv_sparse",
            "rcov_iq",
        ], stride
        assert got["rcov_bandwidth_used"][0] == h, stride
    dense, _ = block(df, kind="kernel", iv_stride=1)
    assert all(
        a > b for a, b in zip(dense["rcov_iv_sparse"][0], base["rcov_iv_sparse"][0], strict=True)
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"kind": "kernel", "bandwidth": 5},
        {"kind": "kernel", "bandwidth": 0, "max_bandwidth": 0},
        {"kind": "plain"},
        {"kind": "preavg"},
        {"kind": "preavg", "psd": False},
    ],
    ids=["kernel-H5", "kernel-H0", "plain", "preavg", "preavg-balanced"],
)
def test_without_an_automatic_bandwidth_a_stride_moves_only_its_own_fields(kw):
    """The strides feed the automatic bandwidth and nothing else. Where
    the bandwidth is fixed, or the kind has none, ``noise_stride`` moves
    ``rcov_omega2`` alone and ``iv_stride`` moves ``rcov_iv_sparse`` and
    ``rcov_iq`` alone: the estimate stays to the bit."""
    df = ticks(n=1200)
    base, _ = block(df, **kw)
    noise, _ = block(df, noise_stride=3, **kw)
    iv, _ = block(df, iv_stride=5, **kw)
    both, _ = block(df, noise_stride=7, iv_stride=40, **kw)
    assert moved(base, noise) == ["rcov_omega2"]
    assert moved(base, iv) == ["rcov_iv_sparse", "rcov_iq"]
    assert moved(base, both) == ["rcov_omega2", "rcov_iv_sparse", "rcov_iq"]


def test_the_automatic_bandwidth_is_its_formula_over_the_reported_fields():
    """``H = ceil(mean_i c* (omega2_i / iv_sparse_i)^(2/5) n^(3/5))`` over the
    columns, with ``n`` the block's ``rcov_n``, recomputed from the fields the
    closed row reports, over blocks of different noise and the strides that
    move it. Every block here sits below its ring, so the formula alone sets
    ``H``; the cases give twelve distinct values (measured 2026-10-25: 10
    to 30)."""
    seen = set()
    for noise, seed in ((0.002, 0), (0.01, 1), (0.02, 2)):
        df = ticks(n=1800, noise=noise, seed=seed, blocks=3)
        for strides in (
            {},
            {"noise_stride": 3},
            {"iv_stride": 1},
            {"noise_stride": 7, "iv_stride": 40},
        ):
            rows, _ = block(df, kind="kernel", **strides)
            ring = math.ceil(C_STAR * 600**0.6)
            for row in rows.iter_rows(named=True):
                per = [
                    C_STAR * (w / v) ** 0.4 * row["rcov_n"] ** 0.6
                    for w, v in zip(row["rcov_omega2"], row["rcov_iv_sparse"], strict=True)
                ]
                want = math.ceil(sum(per) / len(per))
                assert want < ring, (noise, strides, want)
                assert row["rcov_bandwidth_used"] == want, (noise, strides, row["group"])
                seen.add(want)
    assert len(seen) >= 10, sorted(seen)


def test_a_short_block_rows_clips_the_automatic_bandwidth_to_its_ring():
    """``block_rows`` is a hint that sizes the automatic bandwidth's ring,
    ``ceil(c* block_rows^(3/5))``; a block longer than the hint runs, with
    ``H`` clipped to that ring, and is the same block, to the bit, as the
    ring given outright as ``max_bandwidth`` beside an ample hint. The clip
    was held only through ``max_bandwidth``. Here the formula's ``H`` is 10
    (above), and a hint of 600 rows (ring 81) never reaches it."""
    df = ticks(n=1200)
    free, _ = block(df, kind="kernel")
    assert free["rcov_bandwidth_used"][0] == 10
    for hint in (1, 2, 3, 4):
        ring = math.ceil(C_STAR * hint**0.6)
        assert ring < 10, hint
        got, _ = block(df, kind="kernel", block_rows=hint)
        assert got["rcov_bandwidth_used"][0] == ring, hint
        given, _ = block(df, kind="kernel", max_bandwidth=ring)
        assert got.equals(given), hint
        assert moved(free, got) == ["rcov", "rcorr", "rcov_bandwidth_used"], hint
    # A ring at or past the formula's H moves nothing.
    for hint in (5, 6, 300, 600):
        got, _ = block(df, kind="kernel", block_rows=hint)
        assert math.ceil(C_STAR * hint**0.6) >= 10, hint
        assert got.equals(free), hint


def window(theta, psd, block_rows=600):
    """The pre-averaging window ``theta`` derives: ``ceil(theta b^0.6)``
    under ``psd`` and ``floor(theta sqrt(b))`` without, at least 2 and 3."""
    if psd:
        return max(math.ceil(theta * block_rows**0.6), 2)
    return max(math.floor(theta * math.sqrt(block_rows)), 3)


@pytest.mark.parametrize("psd", [True, False])
def test_theta_sets_the_window_and_is_the_window_given_outright(psd):
    """``theta`` through the spec: a larger one is a longer window and so
    fewer pre-averaged terms, ``rcov_n = n - k_n + 2`` over the block's ``n =
    600`` returns. And it sets the window and nothing else: the window it
    derives, given outright as ``preavg_rows``, is the same block to the bit,
    since the bias term reads its theta from the window run (``k_n /
    sqrt(n)``), not from the keyword. Measured 2026-10-25 under ``psd``: k_n
    24, 47, 70 and 140 at theta 0.5, 1, 1.5 and 3."""
    df = ticks(n=1200)
    default, _ = block(df, kind="preavg", psd=psd)
    counts = []
    for theta in (0.5, 1.0, 1.5, 3.0):
        kn = window(theta, psd)
        got, _ = block(df, kind="preavg", psd=psd, theta=theta)
        assert got["rcov_n"][0] == 600 - kn + 2, theta
        given, _ = block(df, kind="preavg", psd=psd, preavg_rows=kn)
        assert got.equals(given), theta
        counts.append(got["rcov_n"][0])
        # The default is 1, and any other theta moves the estimate alone.
        want = [] if theta == 1.0 else ["rcov", "rcorr", "rcov_n"]
        assert moved(default, got) == want, theta
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 4, counts


def test_theta_beside_preavg_rows_is_inert_today():
    """Today's behaviour, pinned for a decision rather than endorsed:
    ``preavg_rows`` fixes the window, so a ``theta`` given beside it is
    read by nothing, and is not refused, even at a value whose own window
    would be past ``block_rows`` and refused alone (theta 13 below). Every
    other key its configuration does not read is refused by name (task 278
    (b11) reports it; the coordinator decides between a refusal and this)."""
    df = ticks(n=1200)
    base, _ = block(df, kind="preavg", preavg_rows=12)
    for theta in (0.5, 3.0, 13.0):
        got, _ = block(df, kind="preavg", preavg_rows=12, theta=theta)
        assert got.equals(base), theta
    with pytest.raises(ValueError, match="theta = 13 sets is 6.04e2 returns"):
        spec(kind="preavg", theta=13.0)


@pytest.mark.parametrize(
    ("kw", "error", "message"),
    [
        # The strides: counts of at least one, and at most the ceiling every
        # size this model allocates from before the first row is held to.
        ({"noise_stride": 0}, ValueError, "noise_stride must be >= 1"),
        ({"iv_stride": 0}, ValueError, "iv_stride must be >= 1"),
        ({"noise_stride": -1}, ValueError, "noise_stride must be >= 1, got -1"),
        ({"iv_stride": -3}, ValueError, "iv_stride must be >= 1, got -3"),
        ({"jitter": -1}, ValueError, "jitter must be >= 1, got -1"),
        ({"noise_stride": 2**20 + 1}, ValueError, "noise_stride = 1048577 is above 1048576"),
        ({"iv_stride": 2**20 + 1}, ValueError, "iv_stride = 1048577 is above 1048576"),
        ({"noise_stride": 1.5}, TypeError, "noise_stride must be an int, got float 1.5"),
        ({"iv_stride": True}, TypeError, "iv_stride must be an int, got bool True"),
        ({"iv_stride": "20"}, TypeError, "iv_stride must be an int, got str '20'"),
        # theta: pre-averaging's alone, finite and above 0, and no window
        # longer than the block it sizes.
        ({"theta": 1.0}, ValueError, 'theta applies to kind = "preavg", not "kernel"'),
        (
            {"kind": "plain", "theta": 1.0},
            ValueError,
            'theta applies to kind = "preavg", not "plain"',
        ),
        ({"kind": "preavg", "theta": 0.0}, ValueError, "theta must be finite and > 0"),
        ({"kind": "preavg", "theta": -1.0}, ValueError, "theta must be finite and > 0"),
        ({"kind": "preavg", "theta": float("nan")}, ValueError, "theta must not be NaN"),
        (
            {"kind": "preavg", "theta": float("inf")},
            ValueError,
            "theta must be finite, got float inf",
        ),
        ({"kind": "preavg", "theta": "1"}, TypeError, "theta must be a number, got str '1'"),
        # The first theta past the block: ceil(13 * 600^0.6) = 604 under psd,
        # floor(24.6 * sqrt(600)) = 602 without.
        ({"kind": "preavg", "theta": 13.0}, ValueError, "theta = 13 sets is 6.04e2 returns"),
        (
            {"kind": "preavg", "theta": 24.6, "psd": False},
            ValueError,
            "theta = 24.6 sets is 6.02e2 returns, which is more than block_rows = 600",
        ),
    ],
)
def test_the_strides_and_theta_are_refused_at_the_spec(kw, error, message):
    with pytest.raises(error, match=re.escape(message)):
        spec(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"noise_stride": 2**20},
        {"iv_stride": 2**20},
        {"noise_stride": 1, "iv_stride": 1},
        # The widest windows that fit: ceil(12.9 * 600^0.6) = 600 and
        # floor(24.49 * sqrt(600)) = 599.
        {"kind": "preavg", "theta": 12.9},
        {"kind": "preavg", "theta": 24.49, "psd": False},
        {"kind": "preavg", "theta": 2},
    ],
)
def test_the_strides_and_theta_are_accepted_up_to_their_bounds(kw):
    s = spec(**kw)
    for key, value in kw.items():
        assert s["model"][key] == value, key
    if "theta" in kw:
        assert window(kw["theta"], kw.get("psd", True)) <= 600


def test_the_widest_window_theta_allows_runs():
    """The window at the bound, as long as the block: one block of 600
    returns gives ``n - k_n + 2 = 2`` pre-averaged terms."""
    df = ticks(n=1200)
    got, _ = block(df, kind="preavg", theta=12.9)
    assert window(12.9, True) == 600
    assert got["rcov_n"][0] == 2 and got["rcov"][0] is not None


# --- plain against numpy, and the scale every kind keeps ---------------------


def test_plain_is_numpy_x_transpose_x_over_every_block():
    """``plain`` is ``X'X`` over the block's returns, by numpy directly
    rather than through ``ew_cov``, at three columns and over five blocks
    (measured 2026-10-25: at most 1.6e-15 relative; the sums run in a
    different order)."""
    cols3 = ["x0", "x1", "x2"]
    df = ticks(n=1800, k=3, blocks=6)
    rows, _ = block(df, kind="plain", features=cols3, block_rows=300)
    assert rows.height == 5
    for row in rows.iter_rows(named=True):
        x = df.filter(pl.col("b") == int(row["group"])).select(cols3).to_numpy()
        want = x.T @ x
        np.testing.assert_allclose(unvech(row["rcov"], 3), want, rtol=1e-13, atol=0)
        sd = np.sqrt(np.diag(want))
        np.testing.assert_allclose(
            unvech(row["rcorr"], 3), want / np.outer(sd, sd), rtol=1e-13, atol=0
        )
        assert row["rcov_n"] == len(x)


SCALED = {"rcov": 2, "rcov_omega2": 2, "rcov_iv_sparse": 2, "rcov_iq": 4}
KINDS = {
    "plain": {"kind": "plain"},
    "kernel-auto": {"kind": "kernel"},
    "kernel-H4": {"kind": "kernel", "bandwidth": 4},
    "preavg": {"kind": "preavg"},
    "preavg-balanced": {"kind": "preavg", "psd": False},
}


@pytest.mark.parametrize("kw", list(KINDS.values()), ids=list(KINDS))
def test_scaling_the_returns_scales_the_estimate_and_nothing_else(kw):
    """Returns times ``c``: the estimate, ``omega2`` and ``iv_sparse`` times
    ``c^2``, the quarticity times ``c^4``, and the correlation, ``rcov_n``,
    the bandwidth (which reads the ratio ``omega2 / iv_sparse``) and every
    other field unchanged. At a power of two each step is exact, so it holds
    to the bit; at other factors to rounding (measured 2026-10-25 at 1e-6, 3
    and 1e6: at most 1.4e-14 relative on any entry, under ``rtol=1e-13``)."""
    df = ticks(n=1200, noise=0.01)
    base, _ = block(df, **kw)
    assert base["rcov"][0] is not None
    for c in (2.0**-20, 2.0**-3, 2.0**7, 2.0**20, 1e-6, 3.0, 1e6):
        got, _ = block(df.with_columns(pl.col(COLS) * c), **kw)
        same = [f for f in base.columns if f not in SCALED and f != "rcorr"]
        assert got.select(same).equals(base.select(same)), c
        exact = math.log2(c).is_integer()
        for field, power in SCALED.items():
            want = [v * c**power for v in base[field][0]]
            if exact:
                assert got[field][0].to_list() == want, (c, field)
            else:
                np.testing.assert_allclose(got[field][0].to_list(), want, rtol=1e-13, atol=0)
        if exact:
            assert got["rcorr"].equals(base["rcorr"]), c
        else:
            np.testing.assert_allclose(
                got["rcorr"][0].to_list(), base["rcorr"][0].to_list(), rtol=1e-13, atol=0
            )
