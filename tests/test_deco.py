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

HALFLIFE = 100.0


def frame(n=800, k=4, rho=0.5, seed=0):
    """A one-factor stream: every pair has correlation `rho` by construction."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    x = np.sqrt(rho) * f + np.sqrt(1.0 - rho) * rng.standard_normal((n, k))
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def cols(k):
    return [f"x{i}" for i in range(k)]


def standardised(df, halflife=HALFLIFE):
    """The rows as the model's `EwDiag` sees them: each value against the
    means and variances *before* it. The oracle everything else is built on."""
    x = df.to_numpy()
    lam = 2.0 ** (-1.0 / halflife)
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
    out = run(po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0), df)
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
    out = run(po.spec.deco("d", features=cols(k), halflife=HALFLIFE, min_periods=0.0), frame(k=k))
    u = out["u"].drop_nulls().to_numpy()
    assert u.size > 700
    assert (u > -1.0 / (k - 1) - 1e-12).all() and (u < 1.0).all()


# --- the level ---------------------------------------------------------------


def test_rho_is_ew_covs_mean_of_u_to_the_bit():
    """The recursion is `EwCov::update`'s mean form -- `rho + b*(u - rho)`,
    not the algebraically equal `a*rho + b*u` -- so feeding the `u` sequence
    to an `ew_cov(stats=["mean"])` at the same halflife reproduces `rho`
    exactly, not nearly."""
    df = frame()
    out = run(po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0), df)
    u = out["u"].to_numpy()
    rho = out["rho"].to_numpy()
    live = np.isfinite(u)
    mean = run(
        po.spec.ew_cov("m", features=["u"], halflife=HALFLIFE, stats=["mean"], min_periods=0.0),
        pl.DataFrame({"u": u[live]}),
    )["mean_u"].to_numpy()
    a, b = rho[live], mean
    # The two warm up differently -- `deco`'s level exists one row after its
    # first finite `u`, `ew_cov`'s mean after its own `min_periods` -- so the
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
    rho = run(po.spec.deco("d", features=cols(2), halflife=200.0, min_periods=0.0), df)[
        "rho"
    ].to_numpy()
    r = standardised(df, 200.0)
    corr = run(
        po.spec.ew_cov("c", features=["r0", "r1"], halflife=200.0, stats=["corr"], min_periods=0.0),
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
            halflife=HALFLIFE,
            min_periods=0.0,
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
    """`-0.5 * (n ln 2pi + ln det R + r'R^-1 r)` on the dense matrix."""
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
    return -0.5 * (n * np.log(2 * np.pi) + logdet + r @ np.linalg.solve(m, r))


def test_loglik_is_the_dense_gaussian_density():
    df = frame(k=4)
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0)
    out = run(spec, df)
    r = standardised(df)
    rho = out["rho"].to_numpy()
    got = out["loglik"].to_numpy()
    checked = 0
    for i in range(len(rho)):
        if not (np.isfinite(rho[i]) and np.isfinite(r[i]).all()):
            continue
        want = dense_loglik(r[i], [rho[i]], [[0, 1, 2, 3]])
        assert got[i] == pytest.approx(want, rel=1e-9)
        checked += 1
    assert checked > 700


def test_the_blocked_loglik_is_the_dense_gaussian_density():
    """The Woodbury path: a `K x K` factorization where the dense form needs
    `n x n`. Same number."""
    df = frame(k=5)
    spec = po.spec.deco(
        "d",
        features=cols(5),
        halflife=HALFLIFE,
        min_periods=0.0,
        blocks={"a": ["x0", "x1", "x2"], "b": ["x3", "x4"]},
    )
    out = run(spec, df)
    r = standardised(df)
    rho = np.column_stack([out["rho_a"], out["rho_b"], out["rho_a_b"]])
    got = out["loglik"].to_numpy()
    checked = 0
    for i in range(len(got)):
        if not (np.isfinite(rho[i]).all() and np.isfinite(r[i]).all()):
            continue
        # A dense matrix that is not positive definite has no density; the
        # model clamps for the density and `slogdet` would refuse, so those
        # rows are skipped rather than compared to nothing.
        try:
            want = dense_loglik(r[i], rho[i], [[0, 1, 2], [3, 4]])
        except AssertionError:
            continue
        assert got[i] == pytest.approx(want, rel=1e-8)
        checked += 1
    assert checked > 500


def test_one_block_of_everything_reproduces_the_unblocked_model():
    df = frame(k=4)
    plain = run(po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0), df)
    blocked = run(
        po.spec.deco(
            "d",
            features=cols(4),
            halflife=HALFLIFE,
            min_periods=0.0,
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
        halflife=HALFLIFE,
        min_periods=0.0,
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
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0)
    want = run(spec, df)
    bank = po.ModelBank([spec])
    parts = [bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)]
    got = pl.concat(parts)["d"].struct.unnest()
    # `coef` is reported on each chunk's last row, so its *cadence* is the
    # one thing chunking moves; the values are not.
    assert want.drop("coef").equals(got.drop("coef"))
    assert want["coef"][-1].to_list() == got["coef"][-1].to_list()


def test_save_load_mid_stream():
    df = frame()
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0)
    want = run(spec, df)
    bank = po.ModelBank([spec])
    bank.fit_predict(df[:400])
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
    got = pl.concat([bank.fit_predict(df[:0]), again.fit_predict(df[400:])])
    assert want[400:].equals(got["d"].struct.unnest())


def test_a_null_feature_skips_the_row_and_a_zero_weight_learns_nothing():
    df = frame(n=200).with_columns(w=pl.lit(1.0))
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0, weight="w")
    want = run(spec, df)
    # A null feature on row 100: that row is skipped, and so is every output.
    nulled = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 100).then(None).otherwise(pl.col("x0")).alias("x0")
    )
    out = run(spec, nulled)
    assert out["u"][100] is None and out["rho"][100] is None and out["n_eff"][100] is None
    # A zero weight on the same row: the row is scored but teaches nothing,
    # so every later `rho` matches the stream with that row's weight at zero.
    zeroed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 100).then(0.0).otherwise(pl.col("w")).alias("w")
    )
    z = run(spec, zeroed)
    assert z["u"][100] is not None, "a zero-weight row is still scored"
    assert z["rho"][100] == want["rho"][100]


def test_a_zero_weight_first_row_is_legal():
    df = frame(n=100).with_columns(w=pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(1.0))
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0, weight="w")
    out = run(spec, df)
    assert out["n_eff"][0] == 0.0
    assert out["rho"].drop_nulls().len() > 90, "and it keeps learning after"
    assert out["rho"].drop_nulls().is_finite().all()


@pytest.mark.parametrize("c", [2.0, 8.0, 3.0])
def test_the_clock_is_a_number(c):
    """Scaling the clock column and the halflife by the same factor leaves
    `d_clock / halflife` -- and so every output -- where it was; bit for bit
    at a power of two, where `exp2` is exact."""
    df = frame(n=300).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    base = po.spec.deco(
        "d", features=cols(4), halflife=HALFLIFE, min_periods=0.0, clock="t", max_dclock=1e12
    )
    scaled = po.spec.deco(
        "d",
        features=cols(4),
        halflife=HALFLIFE * c,
        min_periods=0.0,
        clock="t",
        max_dclock=1e12,
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


def test_the_expression_equals_the_bank():
    df = frame(n=300)
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, min_periods=0.0)
    want = run(spec, df)
    with pytest.warns(po.InMemoryExpressionWarning):
        got = df.select(
            pl.col("x0").online.deco(cols(4)[1:], halflife=HALFLIFE, min_periods=0.0).alias("d")
        )["d"].struct.unnest()
    assert want.equals(got)


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"features": ["x0"]}, "at least two columns"),
        ({"dynamics": "nope"}, "unknown deco dynamics"),
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
    opts = {"features": cols(4), "halflife": HALFLIFE}
    opts.update(kw)
    with pytest.raises(ValueError, match=message):
        po.spec.deco("d", **opts)


def test_a_block_list_written_by_hand_is_checked_too():
    """The Python builder takes ``blocks`` as a dict, so it cannot express a
    duplicate name or an empty list -- but a TOML or JSON spec writes an
    array of pairs and can. Both are refused where the block list is read,
    saying what is wrong with the list (docs/REVIEW-E54-E64.md D1)."""
    base = po.spec.deco(
        "d", features=cols(4), halflife=HALFLIFE, blocks={"a": ["x0", "x1"], "b": ["x2", "x3"]}
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
    `label_delay`, "nothing to hold back". `ew_cov` -- whose shape `deco`
    copies -- accepts it today, and it does hold something back: the
    accumulator update. Refusing it here and not there would be a surprise,
    so it is accepted, and §11a records the departure."""
    spec = po.spec.deco("d", features=cols(4), halflife=HALFLIFE, label_delay=5.0)
    assert spec["label_delay"] == 5.0
    out = run(spec, frame(n=100))
    assert out["n_eff"].drop_nulls().max() > 0


def test_a_decay_is_required():
    with pytest.raises(ValueError, match="halflife/lam"):
        po.spec.deco("d", features=cols(4))
