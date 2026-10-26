"""E55: dynamic equicorrelation (Engle & Kelly 2012), one number for the whole
matrix.

Three things have to be earned. The row estimate `u` is *their* closed form,
held against the longhand pair sum. The level `rho` is `EwCov`'s mean form on
that sequence, held to it **bit for bit**. And `loglik` is the Gaussian
density under an equicorrelation matrix, held against a dense `numpy` one --
which is what makes the Woodbury path in the blocked case worth having.

The plan's fourth claim did not survive measurement, and the correction is a
test of its own: `rho` is **not** an `ew_cov`'s `corr` on the same
standardised columns. See `test_rho_is_not_the_correlation_of_the_columns`.
"""

import json

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

HALFLIFE = 100.0


def frame(n=800, k=4, rho=0.5, seed=0):
    """A one-factor stream: every pair has correlation `rho` by construction."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    x = np.sqrt(rho) * f + np.sqrt(1.0 - rho) * rng.standard_normal((n, k))
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def cols(k):
    return [f"x{i}" for i in range(k)]


def standardised(df, half_life=HALFLIFE):
    """The rows as the model's `EwDiag` sees them: each value against the
    means and variances *before* it. The oracle everything else is built on."""
    x = df.to_numpy()
    lam = 2.0 ** (-1.0 / half_life)
    m = np.zeros(x.shape[1])
    c = np.zeros(x.shape[1])
    w = 0.0
    out = []
    for row in x:
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(c > 0.0, (row - m) / np.sqrt(np.maximum(c, 0.0)), np.nan))
        w_new = lam * w + 1.0
        a, b = lam * w / w_new, 1.0 / w_new
        d = row - m
        c = a * c + a * b * d * d
        m = m + b * d
        w = w_new
    return np.asarray(out)


def run(spec, df):
    return po.ModelBank([spec]).fit_predict(df)[spec["name"]].struct.unnest()


# --- the row estimate --------------------------------------------------------


def test_u_is_the_longhand_pair_sum():
    """Lemma 2.3: the mean off-diagonal product over the mean squared entry."""
    df = frame()
    out = run(po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0), df)
    r = standardised(df)
    n = r.shape[1]
    off = np.einsum("ti,tj->t", r, r) - (r * r).sum(1)
    want = off / ((n - 1) * (r * r).sum(1))
    got = out["u"].to_numpy()
    live = np.isfinite(want) & np.isfinite(got)
    assert live.sum() > 700
    assert np.allclose(got[live], want[live], rtol=0, atol=1e-12)
    # And the slots the model does not report are exactly the ones the
    # standardiser cannot make.
    assert np.array_equal(np.isnan(got), ~np.isfinite(want))


def test_u_stays_inside_its_bounds():
    k = 5
    out = run(po.spec.deco("d", features=cols(k), half_life=HALFLIFE, min_weight=0.0), frame(k=k))
    u = out["u"].drop_nulls().to_numpy()
    assert u.size > 700
    assert (u > -1.0 / (k - 1) - 1e-12).all() and (u < 1.0).all()


# --- the level ---------------------------------------------------------------


def test_rho_is_ew_covs_mean_of_u_to_the_bit():
    """The recursion is `EwCov::update`'s mean form -- `rho + b*(u - rho)`,
    not the algebraically equal `a*rho + b*u` -- so feeding the `u` sequence
    to an `ew_cov(stats=["mean"])` at the same half-life reproduces `rho`
    exactly, not nearly."""
    df = frame()
    out = run(po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0), df)
    u = out["u"].to_numpy()
    rho = out["rho"].to_numpy()
    live = np.isfinite(u)
    mean = run(
        po.spec.ew_cov("m", features=["u"], half_life=HALFLIFE, stats=["mean"], min_weight=0.0),
        pl.DataFrame({"u": u[live]}),
    )["mean_u"].to_numpy()
    a, b = rho[live], mean
    # The two warm up differently -- `deco`'s level exists one row after its
    # first finite `u`, `ew_cov`'s mean after its own `min_weight` -- so the
    # comparison starts where both are live.
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 700
    assert np.array_equal(a[both], b[both]), "the mean forms have drifted apart"


def test_rho_is_not_the_correlation_of_the_columns():
    """**A correction to `docs/PLAN.md` §11a as written.** The plan said a
    one-pair `"ew"` `deco` equals `ew_cov(stats=["corr"])` on the same
    standardised columns *to the bit*. It does not, and not by a little: the
    EW mean of a ratio is not the ratio of EW means, and `u` is the downward
    biased estimator Engle & Kelly themselves flag. Measured on a two-column
    stream with a true correlation of 0.6, `rho` settles near 0.32 where
    `corr` settles near 0.60.

    The test is here so nobody "fixes" the model to close a gap that is the
    estimator's definition."""
    df = frame(n=4000, k=2, rho=0.6, seed=1)
    rho = run(po.spec.deco("d", features=cols(2), half_life=200.0, min_weight=0.0), df)[
        "rho"
    ].to_numpy()
    r = standardised(df, 200.0)
    corr = run(
        po.spec.ew_cov("c", features=["r0", "r1"], half_life=200.0, stats=["corr"], min_weight=0.0),
        pl.DataFrame({"r0": r[:, 0], "r1": r[:, 1]}),
    )["corr_r0_r1"].to_numpy()
    assert rho[-1] == pytest.approx(0.32, abs=0.05)
    assert corr[-1] == pytest.approx(0.60, abs=0.05)
    assert abs(rho[-1] - corr[-1]) > 0.2


def test_the_linear_dynamics_targets_the_ew_level():
    """`rho' = (1 - a - b)*rho_bar' + a*u + b*rho`, against the recursion
    written out, with `rho_bar` the `"ew"` level of the same stream."""
    df = frame()
    alpha, beta = 0.05, 0.9
    out = run(
        po.spec.deco(
            "d",
            features=cols(4),
            half_life=HALFLIFE,
            min_weight=0.0,
            dynamics="linear",
            alpha=alpha,
            beta=beta,
        ),
        df,
    )
    u = out["u"].to_numpy()
    got = out["rho"].to_numpy()
    lam = 2.0 ** (-1.0 / HALFLIFE)
    bar = np.nan
    rho = np.nan
    w = 0.0
    want = np.full(len(u), np.nan)
    for i, ui in enumerate(u):
        want[i] = rho
        if not np.isfinite(ui):
            continue
        w_new = lam * w + 1.0
        b = 1.0 / w_new
        bar = ui if not np.isfinite(bar) else bar + b * (ui - bar)
        rho = ui if not np.isfinite(rho) else (1 - alpha - beta) * bar + alpha * ui + beta * rho
        w = w_new
    live = np.isfinite(want)
    assert live.sum() > 700
    assert np.allclose(got[live], want[live], rtol=0, atol=1e-12)


# --- the density -------------------------------------------------------------


def dense_loglik(r, rho, blocks):
    """`-0.5 * (n ln 2pi + ln det R + r'R^-1 r)` on the dense matrix, and how
    far rounding can move it.

    The bound (task 273): LAPACK's `slogdet` and `solve` are backward
    stable, so each of `ln det R` and `r'R^-1 r` carries a few `n 2^-53`
    times the condition number `cond(R)` of its own size; the model's
    factorization (Woodbury under blocks) does the same; and `r` and `rho`
    carry the decay factors' rounding, the platform's `exp2` on one side
    and Python's `pow` in `standardised` on the other. So
    `|got - want| <= 64 * 2^-53 * cond(R) * (n ln 2pi + |ln det R| +
    |r'R^-1 r|) / 2`. Measured (one block of 4 and the 3 + 2 blocks, every
    row): at most 2.6 and 1.4 times `2^-53 cond(R)` of that scale on macOS,
    9.2 and 6.3 with every `exp`, `exp2`, `ln` and `powf` the wheel calls
    moved by 1 or 2 ulp (a scratch build), where `cond(R)` is at most 6.4.
    The 1e-8 and 1e-9 relative bounds this replaced allowed a millionfold
    of that; and with the libm moved, one row of the blocked stream whose
    matrix is singular to rounding (smallest eigenvalue 1.5e-16) came out
    positive definite to `slogdet`, and was compared: 0.88 apart, relative,
    which the condition number of 2e16 says is no density at all. The
    model reads no density there either since task 278 (`past_the_edge`)."""
    n = len(r)
    of = {i: bi for bi, b in enumerate(blocks) for i in b}
    k = len(blocks)
    pairs = {}
    p = k
    for i in range(k):
        for j in range(i + 1, k):
            pairs[(i, j)] = pairs[(j, i)] = rho[p]
            p += 1
    m = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i, j] = rho[of[i]] if of[i] == of[j] else pairs[(of[i], of[j])]
    sign, logdet = np.linalg.slogdet(m)
    assert sign > 0
    q = r @ np.linalg.solve(m, r)
    scale = 0.5 * (n * np.log(2 * np.pi) + abs(logdet) + abs(q))
    return -0.5 * (n * np.log(2 * np.pi) + logdet + q), 64 * 2.0**-53 * np.linalg.cond(m) * scale


def past_the_edge(rho, sizes):
    """Whether block levels `rho` (within, then each pair) are past the edge
    at which `deco` gives no density: the block-mean matrix `T` (`T_AA = 1 +
    (n_A - 1) rho_AA`, `T_AB = sqrt(n_A n_B) rho_AB`, whose eigenvalues are
    the dense `R`'s other than its `1 - rho_AA`) equilibrated to `H`, read
    by numpy's `eigvalsh`. The model refuses where its factorization of
    `H`'s scaling needs jitter, or where `3 K gamma_{3K+1} ||H^-1||_1 >= 1`
    by an estimate that is never above `||H^-1||_1 <= sqrt(K) / lambda_min`.
    So a refused row has `lambda_min(H) <= sqrt(K) / limit`, up to the
    rounding of forming `H` and of `eigvalsh` (a few `K 2^-53`; 64 of them
    allowed)."""
    k = len(sizes)
    t = np.eye(k)
    p = k
    for i in range(k):
        t[i, i] = 1.0 + (sizes[i] - 1) * rho[i]
    for i in range(k):
        for j in range(i + 1, k):
            t[i, j] = t[j, i] = np.sqrt(sizes[i] * sizes[j]) * rho[p]
            p += 1
    d = np.sqrt(np.diag(t))
    h = t / np.outer(d, d)
    u = 2.0**-53
    m = (3 * k + 1) * u
    limit = (1 - m) / (3 * k * m)
    return np.linalg.eigvalsh(h).min() <= np.sqrt(k) / limit + 64 * k * u


def test_loglik_is_the_dense_gaussian_density():
    df = frame(k=4)
    spec = po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0)
    out = run(spec, df)
    r = standardised(df)
    rho = out["rho"].to_numpy()
    got = out["loglik"].to_numpy()
    checked = 0
    for i in range(len(rho)):
        if not (np.isfinite(rho[i]) and np.isfinite(r[i]).all()):
            continue
        want, bound = dense_loglik(r[i], [rho[i]], [[0, 1, 2, 3]])
        assert abs(got[i] - want) <= bound, (i, got[i], want, bound)
        checked += 1
    assert checked > 700


def test_the_blocked_loglik_is_the_dense_gaussian_density():
    """The Woodbury path: a `K x K` factorization where the dense form needs
    `n x n`. Same number."""
    df = frame(k=5)
    spec = po.spec.deco(
        "d",
        features=cols(5),
        half_life=HALFLIFE,
        min_weight=0.0,
        blocks={"a": ["x0", "x1", "x2"], "b": ["x3", "x4"]},
    )
    out = run(spec, df)
    r = standardised(df)
    rho = np.column_stack([out["rho_a"], out["rho_b"], out["rho_a_b"]])
    got = out["loglik"].to_numpy()
    checked = 0
    past = []
    for i in range(len(got)):
        if not (np.isfinite(rho[i]).all() and np.isfinite(r[i]).all()):
            continue
        # A level matrix past the edge has no density (task 278): the model
        # reads NaN there, and only there.
        if np.isnan(got[i]):
            assert past_the_edge(rho[i], [3, 2]), (i, rho[i])
            past.append(i)
            continue
        want, bound = dense_loglik(r[i], rho[i], [[0, 1, 2], [3, 4]])
        assert abs(got[i] - want) <= bound, (i, got[i], want, bound)
        checked += 1
    assert checked > 500
    # Row 3, in the warm-up, is the one past the edge on this stream: its
    # density read -6.2e12 before.
    assert past == [3], past


def test_one_block_of_everything_reproduces_the_unblocked_model():
    df = frame(k=4)
    plain = run(po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0), df)
    blocked = run(
        po.spec.deco(
            "d",
            features=cols(4),
            half_life=HALFLIFE,
            min_weight=0.0,
            blocks={"all": cols(4)},
        ),
        df,
    )
    assert plain["u"].to_list() == blocked["u_all"].to_list()
    assert plain["rho"].to_list() == blocked["rho_all"].to_list()
    assert plain["loglik"].to_list() == blocked["loglik"].to_list()


def test_the_blocked_within_and_between_values_are_their_closed_forms():
    df = frame(k=5)
    spec = po.spec.deco(
        "d",
        features=cols(5),
        half_life=HALFLIFE,
        min_weight=0.0,
        blocks={"a": ["x0", "x1", "x2"], "b": ["x3", "x4"]},
    )
    out = run(spec, df)
    r = standardised(df)
    for name, idx in (("a", [0, 1, 2]), ("b", [3, 4])):
        s1 = r[:, idx].sum(1)
        s2 = (r[:, idx] ** 2).sum(1)
        want = (s1**2 - s2) / ((len(idx) - 1) * s2)
        got = out[f"u_{name}"].to_numpy()
        live = np.isfinite(want) & np.isfinite(got)
        assert np.allclose(got[live], want[live], rtol=0, atol=1e-12)
    sa, sb = r[:, [0, 1, 2]].sum(1), r[:, [3, 4]].sum(1)
    qa, qb = (r[:, [0, 1, 2]] ** 2).sum(1), (r[:, [3, 4]] ** 2).sum(1)
    want = sa * sb / np.sqrt(3 * 2 * qa * qb)
    got = out["u_a_b"].to_numpy()
    live = np.isfinite(want) & np.isfinite(got)
    assert np.allclose(got[live], want[live], rtol=0, atol=1e-12)


# --- the shared contract -----------------------------------------------------


@pytest.mark.parametrize("size", [1, 7, 250, 800])
def test_chunk_invariance(size):
    df = frame()
    spec = po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0)
    want = run(spec, df)
    bank = po.ModelBank([spec])
    parts = [bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)]
    got = pl.concat(parts)["d"].struct.unnest()
    # `coef` is reported on each chunk's last row, so its *cadence* is the
    # one thing chunking moves; the values are not.
    assert want.drop("coef", "support_coef", strict=False).equals(
        got.drop("coef", "support_coef", strict=False)
    )
    assert want["coef"][-1].to_list() == got["coef"][-1].to_list()


def test_save_load_mid_stream():
    df = frame()
    spec = po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0)
    want = run(spec, df)
    bank = po.ModelBank([spec])
    bank.fit_predict(df[:400])
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    got = pl.concat([bank.fit_predict(df[:0]), again.fit_predict(df[400:])])
    assert want[400:].equals(got["d"].struct.unnest())


def test_a_null_feature_skips_the_row_and_a_zero_weight_learns_nothing():
    df = frame(n=200).with_columns(w=pl.lit(1.0))
    spec = po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0, weight="w")
    want = run(spec, df)
    # A null feature on row 100: that row is skipped, and so is every output.
    nulled = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 100).then(None).otherwise(pl.col("x0")).alias("x0")
    )
    out = run(spec, nulled)
    assert out["u"][100] is None and out["rho"][100] is None and out["weight_sum"][100] is None
    # A zero weight on the same row: the row is scored but teaches nothing,
    # so every later `rho` matches the stream with that row's weight at zero.
    zeroed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 100).then(0.0).otherwise(pl.col("w")).alias("w")
    )
    z = run(spec, zeroed)
    assert z["u"][100] is not None, "a zero-weight row is still scored"
    assert z["rho"][100] == want["rho"][100]


def test_a_constant_column_is_left_out_and_every_other_value_learns():
    """A column without spread has no standardized value, and is left out of
    its block's sums (docs/PLAN.md task 115 (h), decided 2026-09-29). While
    ``x3`` is constant -- from the first row, the only way its variance is
    exactly zero -- block ``a`` learns as a ``deco`` over its own columns
    does, to the bit; the pair learns from ``x2``; block ``b``, left with one
    column that has a spread, has no ``rho``; and ``loglik``, which needs
    every column, is null. Until the decision no value learned on such a
    row. What held before holds still: ``weight_sum`` advances, nothing NaN
    reaches the state, and once the column moves every value learns (task
    112)."""
    df = frame(n=900)
    i = pl.int_range(pl.len())
    df = df.with_columns(x3=pl.when(i < 300).then(pl.lit(2.0)).otherwise(pl.col("x3")))
    spec = po.spec.deco(
        "d",
        features=cols(4),
        half_life=HALFLIFE,
        min_weight=0.0,
        blocks={"a": ["x0", "x1"], "b": ["x2", "x3"]},
    )
    out = run(spec, df)
    weight_sum = out["weight_sum"].to_numpy()
    assert (np.diff(weight_sum) > 0).all(), "the clock and the weight move on"
    alone = run(po.spec.deco("d", features=["x0", "x1"], half_life=HALFLIFE, min_weight=0.0), df)
    np.testing.assert_array_equal(out["rho_a"].to_numpy(), alone["rho"].to_numpy())
    flat = out[10:300]
    assert flat["rho_b"].null_count() == flat.height, "b has one column with a spread"
    assert flat["rho_a_b"].null_count() == 0
    assert np.ptp(flat["rho_a_b"].to_numpy()) > 1e-3, "the pair learns from x2"
    assert flat["loglik"].null_count() == flat.height, "no density without every column"
    for col in ("rho_a", "rho_b", "rho_a_b"):
        late = out[col][400:].to_numpy()
        assert np.isfinite(late).all(), col
        assert np.ptp(late) > 1e-3, f"{col} learns once the column moves"
    assert out["loglik"][400:].is_finite().all()


def test_a_zero_weight_first_row_is_legal():
    df = frame(n=100).with_columns(w=pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(1.0))
    spec = po.spec.deco("d", features=cols(4), half_life=HALFLIFE, min_weight=0.0, weight="w")
    out = run(spec, df)
    assert out["weight_sum"][0] == 0.0
    assert out["rho"].drop_nulls().len() > 90, "and it keeps learning after"
    assert out["rho"].drop_nulls().is_finite().all()


@pytest.mark.parametrize("c", [2.0, 8.0, 3.0])
def test_the_clock_is_a_number(c):
    """Scaling the clock column and the half-life by the same factor leaves
    `d_clock / half-life` -- and so every output -- where it was; bit for bit
    at a power of two, where `exp2` is exact."""
    df = frame(n=300).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    base = po.spec.deco(
        "d", features=cols(4), half_life=HALFLIFE, min_weight=0.0, clock="t", gap_cap=1e12
    )
    scaled = po.spec.deco(
        "d",
        features=cols(4),
        half_life=HALFLIFE * c,
        min_weight=0.0,
        clock="t",
        gap_cap=1e12,
    )
    want = run(base, df)
    got = run(scaled, df.with_columns(t=pl.col("t") * c))
    if c in (2.0, 8.0):
        assert want.equals(got)
    else:
        for col in ("u", "rho", "loglik"):
            a = want[col].to_numpy()
            b = got[col].to_numpy()
            live = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[live], b[live], rtol=1e-12, atol=1e-12)


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"features": ["x0"]}, "at least two columns"),
        ({"dynamics": "nope"}, 'dynamics must be one of "ew", "linear"; got "nope"'),
        ({"dynamics": "linear"}, "needs both alpha and beta"),
        ({"dynamics": "linear", "alpha": 0.5, "beta": 0.6}, "alpha \\+ beta must be < 1"),
        ({"alpha": 0.5}, "alpha/beta belong to"),
        ({"blocks": {"a": ["x0"], "b": ["x1", "x2", "x3"]}}, "needs at least two"),
        ({"blocks": {"a": ["x0", "x1"]}}, "is in no block"),
        ({"blocks": {"a": ["x0", "x1"], "b": ["x1", "x2"]}}, "in more than one block"),
        ({"blocks": {"a": ["x0", "nope"], "b": ["x1", "x2"]}}, "not a feature of this spec"),
        ({"emit_sigma": True}, "does not apply to deco"),
        ({"conformal": 0.9}, "does not apply to deco"),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    opts = {"features": cols(4), "half_life": HALFLIFE}
    opts.update(kw)
    with pytest.raises(ValueError, match=message):
        po.spec.deco("d", **opts)


def test_a_block_list_written_by_hand_is_checked_too():
    """The Python builder takes ``blocks`` as a dict, so it cannot express a
    duplicate name or an empty list -- but a TOML or JSON spec writes an
    array of pairs and can. Both are refused where the block list is read,
    saying what is wrong with the list (docs/REVIEW-E54-E64.md D1)."""
    base = po.spec.deco(
        "d", features=cols(4), half_life=HALFLIFE, blocks={"a": ["x0", "x1"], "b": ["x2", "x3"]}
    )
    for blocks, message in (
        ([["a", ["x0", "x1"]], ["a", ["x2", "x3"]]], "named twice"),
        ([], "blocks is empty"),
    ):
        raw = json.loads(json.dumps(base))
        raw["model"]["blocks"] = blocks
        with pytest.raises(ValueError, match=message):
            po.ModelBank([raw])


def test_label_delay_is_accepted_as_ew_cov_accepts_it():
    """`docs/PLAN.md` §11a said the new no-target models should refuse
    `embargo`, "nothing to hold back". `ew_cov` -- whose shape `deco`
    copies -- accepts it today, and it does hold something back: the
    accumulator update. Refusing it here and not there would be a surprise,
    so it is accepted, and §11a records the departure.

    What it holds back is checked, not only accepted (review 2026-10-05,
    TB12): the embargoed run differs from the plain one, and row ``i`` of it
    is what a plain bank that learned only the rows five clock units behind
    ``i`` predicts for row ``i`` -- the delay applied by hand."""
    df = frame(n=100).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    kw = dict(features=cols(4), half_life=HALFLIFE, clock="t", gap_cap=1e9)
    spec = po.spec.deco("d", embargo=5.0, **kw)
    assert spec["embargo"] == 5.0
    held = run(spec, df)
    plain = po.spec.deco("d", **kw)
    fields = ("u", "rho", "loglik", "weight_sum")
    assert not held.select(fields).equals(run(plain, df).select(fields), null_equal=True)
    assert held["weight_sum"][:5].to_list() == [0.0] * 5
    for i in range(5, df.height):
        bank = po.ModelBank([plain])
        bank.fit_predict(df.slice(0, i - 4))
        want = bank.predict(df.slice(i, 1))["d"].struct.unnest()
        assert held.select(fields)[i].equals(want.select(fields), null_equal=True), i
    assert held["rho"].drop_nulls().len() > 50


def test_a_decay_is_required():
    with pytest.raises(ValueError, match="half_life/lam"):
        po.spec.deco("d", features=cols(4))


# --- phase-4 coverage: a wide block (the tests above are k <= 5) --------------


def test_a_wide_block_matches_ew_covs_mean_of_u_to_the_bit():
    """`deco` folds `u` into `rho` through the same mean-form recursion at any
    width; at k = 24 this also drives the Woodbury path in the density. `rho`
    still reproduces `ew_cov`'s mean of the `u` sequence exactly (review
    2026-09-18, phase 4)."""
    df = frame(n=1200, k=24, rho=0.4, seed=7)
    out = run(po.spec.deco("d", features=cols(24), half_life=HALFLIFE, min_weight=0.0), df)
    u = out["u"].to_numpy()
    rho = out["rho"].to_numpy()
    live = np.isfinite(u)
    assert live.sum() > 1000
    mean = run(
        po.spec.ew_cov("m", features=["u"], half_life=HALFLIFE, stats=["mean"], min_weight=0.0),
        pl.DataFrame({"u": u[live]}),
    )["mean_u"].to_numpy()
    both = np.isfinite(rho[live]) & np.isfinite(mean)
    assert both.sum() > 1000
    assert np.array_equal(rho[live][both], mean[both]), "the mean forms drifted at width 24"
    # `loglik` is finite once the level exists (i.e. wherever `rho` is).
    loglik = out["loglik"].to_numpy()
    assert np.isfinite(loglik[np.isfinite(rho)]).all()


def test_coef_is_the_level_after_the_row():
    """Task 239: the docstring said every field is read before the row,
    ``coef`` included. ``coef`` is the level after the row, as every model's
    ``coef`` is the fit after it: ``coef`` on a row is ``rho`` on the next,
    per block and pair."""
    df = frame(k=4)
    spec = po.spec.deco(
        "d",
        features=cols(4),
        half_life=HALFLIFE,
        min_weight=0.0,
        coef_every=0,
        blocks={"a": ["x0", "x1"], "b": ["x2", "x3"]},
    )
    out = run(spec, df)
    coef = np.array([c if c is not None else [np.nan] * 3 for c in out["coef"].to_list()])
    rho = np.column_stack([out["rho_a"], out["rho_b"], out["rho_a_b"]]).astype(float)
    live = np.isfinite(rho[1:]).all(axis=1)
    assert live.sum() > 700
    assert np.array_equal(coef[:-1][live], rho[1:][live])


# --- task 278 (b29): the linear dynamics' own parameters ---------------------


def _messy(n=1200, k=4, seed=3):
    """A one-factor stream on an irregular clock with one gap far past the
    cap, uneven row weights and a zero weight on every 23rd row."""
    rng = np.random.default_rng(seed)
    df = frame(n=n, k=k, rho=0.5, seed=seed)
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    t[n // 2 :] += 300.0
    w = rng.uniform(0.25, 2.0, n)
    w[7::23] = 0.0
    return df.with_columns(t=pl.Series(t), w=pl.Series(w))


MESSY = dict(clock="t", gap_cap=4.0, weight="w")


def longhand(df, k, half_life, *, alpha=None, beta=None, t=None, w=None, gap_cap=np.inf):
    """`deco`'s recursion written out, one block of `k` columns: the
    standardiser (`EwDiag`) and the level `rho_bar` decay on the clock,
    `lam = 2^(-min(dt, gap_cap) / half_life)`, and weigh each row by its
    weight; under `"linear"` the level `rho` takes one step a row,
    `(1 - alpha - beta) rho_bar' + alpha u + beta rho`, its `alpha u` term
    unweighted, and a row of weight 0 leaves it where it was. Returns the
    row's `u` and the level before the row, as the model emits them."""
    x = df.select(cols(k)).to_numpy()
    n = len(x)
    w = np.ones(n) if w is None else np.asarray(w, float)
    m = np.zeros(k)
    c = np.zeros(k)
    W = 0.0
    bar, rho, wb = np.nan, np.nan, 0.0
    us = np.full(n, np.nan)
    rhos = np.full(n, np.nan)
    for i in range(n):
        d = 0.0 if (t is None or i == 0) else min(t[i] - t[i - 1], gap_cap)
        lam = 2.0 ** (-d / half_life)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(c > 0.0, (x[i] - m) / np.sqrt(np.maximum(c, 0.0)), np.nan)
        s1, s2 = r.sum(), (r * r).sum()
        u = (s1 * s1 - s2) / ((k - 1) * s2)
        us[i], rhos[i] = u, rho
        if np.isfinite(u):
            wb_new = lam * wb + w[i]
            if wb_new > 0.0:
                b = w[i] / wb_new
                bar = u if not np.isfinite(bar) else bar + b * (u - bar)
                if alpha is None:
                    rho = bar
                elif w[i] > 0.0:
                    rho = (
                        u
                        if not np.isfinite(rho)
                        else (1 - alpha - beta) * bar + alpha * u + beta * rho
                    )
            wb = wb_new
        W_new = lam * W + w[i]
        if W_new > 0.0:
            a, b = lam * W / W_new, w[i] / W_new
            dev = x[i] - m
            c = a * c + a * b * dev * dev
            m = m + b * dev
        W = W_new
    return us, rhos


@pytest.mark.parametrize("messy", [False, True], ids=["plain", "clock_gap_weights"])
def test_linear_at_alpha_and_beta_zero_is_ew_to_the_bit(messy):
    """At `alpha = beta = 0` the linear recursion is `rho' = rho_bar'`, the
    `"ew"` level itself: every field to the bit, on a plain stream and on one
    with a capped gap, uneven weights and rows of weight 0 (probed
    2026-10-25: bit-equal both ways)."""
    df = _messy() if messy else frame(n=1200)
    kw = dict(features=cols(4), half_life=50.0, min_weight=0.0, **(MESSY if messy else {}))
    ew = run(po.spec.deco("d", **kw), df)
    lin = run(po.spec.deco("d", dynamics="linear", alpha=0.0, beta=0.0, **kw), df)
    assert ew.equals(lin, null_equal=True)
    assert ew["rho"].drop_nulls().len() > 1100


@pytest.mark.parametrize("dynamics", ["ew", "linear"])
def test_the_recursion_with_weights_and_a_capped_gap_is_the_longhand_one(dynamics):
    """The reference above, on a clock with a gap far past `gap_cap` and
    weights with zeros: `rho_bar` decays across the gap by the capped step,
    `rho`'s own terms take one row's step, and `alpha u` is not weighted.
    The `"linear"` row is checked to read the gap: the same rows on a clock
    without it part from it after the gap."""
    df = _messy()
    t, w = df["t"].to_numpy(), df["w"].to_numpy()
    lin = dict(alpha=0.05, beta=0.9) if dynamics == "linear" else {}
    spec = po.spec.deco(
        "d", features=cols(4), half_life=50.0, min_weight=0.0, dynamics=dynamics, **lin, **MESSY
    )
    out = run(spec, df)
    u, rho = longhand(df, 4, 50.0, t=t, w=w, gap_cap=4.0, **lin)
    got_u, got_rho = out["u"].to_numpy(), out["rho"].to_numpy()
    assert np.array_equal(np.isnan(got_u), np.isnan(u))
    assert np.array_equal(np.isnan(got_rho), np.isnan(rho))
    live = np.isfinite(rho)
    assert live.sum() > 1100
    assert np.allclose(got_u[live], u[live], rtol=0, atol=1e-12)
    assert np.allclose(got_rho[live], rho[live], rtol=0, atol=1e-12)
    # The gap reaches the level: on the same clock without it the longhand
    # parts from the row after it.
    half = df.height // 2
    no_gap = t.copy()
    no_gap[half:] -= 300.0
    _, rho_no_gap = longhand(df, 4, 50.0, t=no_gap, w=w, gap_cap=4.0, **lin)
    assert np.allclose(rho[: half + 1], rho_no_gap[: half + 1], rtol=0, atol=0, equal_nan=True)
    # Measured 2026-10-25: 5.3e-4 at most under "linear", where the level
    # sees the gap only through `(1 - alpha - beta) rho_bar`.
    assert np.nanmax(np.abs(rho[half + 1 :] - rho_no_gap[half + 1 :])) > 1e-4


def test_a_weight_scale_moves_no_linear_level():
    """Under `"linear"` a weight reaches `rho` through `rho_bar`'s `w / W'`
    alone, a ratio a common factor cancels from: every weight times 4, a
    power of two so the ratio is exact, gives the same frame to the bit."""
    df = _messy()
    spec = po.spec.deco(
        "d",
        features=cols(4),
        half_life=50.0,
        min_weight=0.0,
        dynamics="linear",
        alpha=0.05,
        beta=0.9,
        **MESSY,
    )
    one = run(spec, df).drop("weight_sum", "settled_frac")
    four = run(spec, df.with_columns(w=pl.col("w") * 4.0)).drop("weight_sum", "settled_frac")
    assert one.equals(four, null_equal=True)


def _rho_steps(alpha, beta, df):
    spec = po.spec.deco(
        "d",
        features=cols(4),
        half_life=100.0,
        min_weight=0.0,
        dynamics="linear",
        alpha=alpha,
        beta=beta,
    )
    return np.nanstd(np.diff(run(spec, df)["rho"].to_numpy()))


def test_alpha_moves_the_level_more_and_beta_less():
    """`alpha` is the weight on the row's estimate and `beta` the
    persistence: the spread of `rho`'s row-to-row steps rises with `alpha`
    at each `beta` and falls with `beta` at each `alpha`. Measured
    2026-10-25 on this stream: 0.0126, 0.0226, 0.0338, 0.0511 over alpha at
    beta 0, and 0.0338 down to 0.0213 over beta at alpha 0.05."""
    df = frame(n=1200)
    for beta in (0.0, 0.5, 0.9):
        steps = [_rho_steps(a, beta, df) for a in (0.01, 0.03, 0.05, 0.08)]
        assert np.all(np.diff(steps) > 0), (beta, steps)
    for alpha in (0.0, 0.02, 0.05):
        steps = [_rho_steps(alpha, b, df) for b in (0.0, 0.3, 0.6, 0.9, 0.94)]
        assert np.all(np.diff(steps) < 0), (alpha, steps)


def test_alpha_near_one_tracks_the_last_estimate_and_beta_near_one_freezes():
    """The two corners, with bounds from the recursion rather than measured:
    `u` and `rho_bar` both lie in `[-1/(k-1), 1]`, so at `alpha = 1 - e,
    beta = 0` the level after a row is `u + e (rho_bar' - u)`, within `e k /
    (k-1)` of that row's `u`; and at `alpha = 0, beta = 1 - e` a row moves
    it by `e (rho_bar' - rho)`, at most `e k / (k-1)`. Rows of weight 1."""
    k, e = 4, 1e-3
    df = frame(n=1200, k=k)
    bound = e * k / (k - 1) * (1 + 1e-12)
    out = run(
        po.spec.deco(
            "d",
            features=cols(k),
            half_life=100.0,
            min_weight=0.0,
            dynamics="linear",
            alpha=1 - e,
            beta=0.0,
        ),
        df,
    )
    u, rho = out["u"].to_numpy(), out["rho"].to_numpy()
    live = np.isfinite(rho[1:]) & np.isfinite(u[:-1])
    assert live.sum() > 1100
    gap = np.abs(rho[1:] - u[:-1])[live]
    assert gap.max() <= bound and gap.max() > bound / 100, gap.max()
    frozen = run(
        po.spec.deco(
            "d",
            features=cols(k),
            half_life=100.0,
            min_weight=0.0,
            dynamics="linear",
            alpha=0.0,
            beta=1 - e,
        ),
        df,
    )["rho"].to_numpy()
    step = np.abs(np.diff(frozen))
    step = step[np.isfinite(step)]
    assert step.size > 1100 and step.max() <= bound, step.max()


@pytest.mark.parametrize(
    ("alpha", "beta", "message"),
    [
        (0.25, 0.75, r"alpha \+ beta must be < 1 \(got 1\)"),
        (float("nan"), 0.5, "alpha must not be NaN"),
        (0.5, float("nan"), "beta must not be NaN"),
        (-0.1, 0.5, "alpha and beta must be >= 0"),
        (0.5, -0.1, "alpha and beta must be >= 0"),
    ],
)
def test_the_linear_weights_are_bounded(alpha, beta, message):
    """`alpha + beta` exactly 1 is refused (the intercept would vanish), as
    is a NaN or a negative weight; the sum one ulp under 1 is the largest
    legal one, and `alpha = beta = 0` the smallest."""
    with pytest.raises(ValueError, match=message):
        po.spec.deco(
            "d", features=cols(4), half_life=HALFLIFE, dynamics="linear", alpha=alpha, beta=beta
        )


@pytest.mark.parametrize(("alpha", "beta"), [(0.5, 0.5 - 2.0**-53), (0.0, 0.0), (0.0, 0.999)])
def test_the_edges_of_the_linear_weights_are_accepted_and_finite(alpha, beta):
    assert alpha + beta < 1.0
    spec = po.spec.deco(
        "d",
        features=cols(4),
        half_life=HALFLIFE,
        min_weight=0.0,
        dynamics="linear",
        alpha=alpha,
        beta=beta,
    )
    rho = run(spec, frame(n=300))["rho"].to_numpy()
    assert np.isfinite(rho[3:]).all()


LINEAR = dict(dynamics="linear", alpha=0.05, beta=0.9)
BLOCKS = dict(blocks={"p": ["x0", "x1"], "q": ["x2", "x3"]})


@pytest.mark.parametrize(
    "extra", [LINEAR, BLOCKS, {**LINEAR, **BLOCKS}], ids=["linear", "blocks", "linear_blocks"]
)
@pytest.mark.parametrize("size", [1, 7, 250, 800])
def test_chunk_invariance_under_linear_and_blocks(extra, size):
    """Hard rule 3 beyond the default spec, on the clocked, weighted stream:
    every field but `coef`, whose cadence is the chunk's last row, to the
    bit."""
    df = _messy()
    spec = po.spec.deco("d", features=cols(4), half_life=50.0, min_weight=0.0, **MESSY, **extra)
    want = run(spec, df)
    bank = po.ModelBank([spec])
    got = pl.concat([bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)])[
        "d"
    ].struct.unnest()
    assert want.drop("coef").equals(got.drop("coef"), null_equal=True)
    assert want["coef"][-1].to_list() == got["coef"][-1].to_list()


@pytest.mark.parametrize(
    "extra", [LINEAR, BLOCKS, {**LINEAR, **BLOCKS}], ids=["linear", "blocks", "linear_blocks"]
)
@pytest.mark.parametrize("cut", [5, 601, 1111])
def test_save_load_mid_stream_under_linear_and_blocks(extra, cut):
    """A state saved mid-stream, one cut on each side of the gap, continues
    to the bit; the rows after the cut are the whole run's, `coef` too."""
    df = _messy()
    spec = po.spec.deco("d", features=cols(4), half_life=50.0, min_weight=0.0, **MESSY, **extra)
    want = run(spec, df)
    bank = po.ModelBank([spec])
    bank.fit_predict(df[:cut])
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    assert want[cut:].equals(again.fit_predict(df[cut:])["d"].struct.unnest(), null_equal=True)


def test_the_order_of_the_blocks_and_of_their_columns_moves_only_the_names():
    """The blocks and the columns inside each listed in another order: each
    `u` and `rho` to the bit under its own name (a pair is named in the
    order the blocks are listed, so `rho_p_q` reads as `rho_q_p`). `loglik`
    goes through a `K x K` factorization whose rows come in the blocks'
    order, so it agrees within the conditioning bound `dense_loglik` gives
    (task 273). A row whose level matrix is past the edge has no density in
    either order (on this stream, row 3: smallest eigenvalue -1.5e-17, the
    two orders' readings -2.3e8 and -2.6e12 until task 278 made both NaN)."""
    df = frame(n=1200)
    kw = dict(features=cols(4), half_life=100.0, min_weight=0.0)
    a = run(po.spec.deco("d", blocks={"p": ["x0", "x1"], "q": ["x2", "x3"]}, **kw), df)
    b = run(po.spec.deco("d", blocks={"q": ["x3", "x2"], "p": ["x1", "x0"]}, **kw), df)
    renamed = b.rename({"u_q_p": "u_p_q", "rho_q_p": "rho_p_q"})
    for col in ("u_p", "u_q", "u_p_q", "rho_p", "rho_q", "rho_p_q", "weight_sum"):
        assert a[col].equals(renamed[col], null_equal=True), col
    r = standardised(df)
    rho = np.column_stack([a["rho_p"], a["rho_q"], a["rho_p_q"]])
    la, lb = a["loglik"].to_numpy(), b["loglik"].to_numpy()
    assert (np.isnan(la) == np.isnan(lb)).all(), np.flatnonzero(np.isnan(la) != np.isnan(lb))
    checked = 0
    past = []
    for i in range(df.height):
        if not (np.isfinite(rho[i]).all() and np.isfinite(r[i]).all()):
            continue
        if np.isnan(la[i]):
            assert past_the_edge(rho[i], [2, 2]), (i, rho[i])
            past.append(i)
            continue
        _, bound = dense_loglik(r[i], rho[i], [[0, 1], [2, 3]])
        assert abs(la[i] - lb[i]) <= 2 * bound, (i, la[i], lb[i], bound)
        checked += 1
    assert checked > 1100
    assert past == [3], past
