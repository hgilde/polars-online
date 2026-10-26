"""E60: a Gaussian hidden Markov model, filtered online.

`ew_class` answers "which regime does this row look like". An `hmm` answers
"which regime are we in", and the difference is the transition matrix. Two
things earn that here: the filter reduces to the classifier when the chain
is uniform, and it is the longhand Hamilton recursion at fixed parameters.
"""

import re

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def blobs(n=2000, run=100, sep=4.0, seed=0, d=2):
    """Two well-separated blobs, visited in runs of `run` rows."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n) // run) % 2
    c = np.where(g == 0, -sep / 2, sep / 2)[:, None]
    x = c + rng.standard_normal((n, d))
    return pl.DataFrame({**{f"x{i}": x[:, i] for i in range(d)}, "g": g})


def spec(**kw):
    d = {
        "features": ["x0", "x1"],
        "n_components": 2,
        "precision_prior": 1e-2,
        "half_life": 1e9,
    }
    # The warm-up's parameters where a warm-up runs: with the states given
    # nothing is seeded, and they are refused (review 2026-10-06, CE4).
    if "means" not in kw and "covs" not in kw:
        d |= {"warm_rows": 100, "init": "lloyd"}
    d.update(kw)
    return po.spec.hmm("h", **d)


def run(df, **kw):
    return po.ModelBank([spec(**kw)]).fit_predict(df)["h"].struct.unnest()


def test_the_filter_finds_the_states_and_the_chain():
    df = blobs(n=4000, run=200)
    out = run(df, warm_rows=200)
    # `coef` is null except where the schedule reports it, so it is dropped
    # before asking which rows are live.
    live = out.drop("coef", "withheld_reason").drop_nulls()
    assert live.height > 3000
    # The reported state tracks the blob, up to which state is which.
    state = out["state"].to_numpy()
    truth = df["g"].to_numpy()
    ok = np.isfinite(state.astype(float))
    agree = (state[ok] == truth[ok]).mean()
    assert max(agree, 1 - agree) > 0.95, agree
    # The chain is sticky, and `p` is a distribution on every live row.
    p = live.select("filtered_state0", "filtered_state1").to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()


def test_the_filter_is_the_longhand_hamilton_recursion():
    """At fixed parameters, `learn=False`, against the recursion written
    out — the same numbers the model computes, in a different order."""
    df = blobs(n=600, run=40, d=2, seed=1)
    means = [-2.0, -2.0, 2.0, 2.0]
    covs = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    pi = np.array([[0.95, 0.05], [0.1, 0.9]])
    out = run(
        df,
        learn=False,
        means=means,
        covs=covs,
        transition=pi.flatten().tolist(),
        transition_prior=1.0,
    )
    ridge = 1e-2
    mu = np.array(means).reshape(2, 2)
    p = np.array([0.5, 0.5])
    x = df.select("x0", "x1").to_numpy()
    for t in range(len(x)):
        pred = p @ pi
        f = np.array(
            [
                np.exp(
                    -0.5
                    * (
                        2 * np.log(2 * np.pi)
                        + 2 * np.log(1.0 + ridge)
                        + ((x[t] - mu[st]) ** 2 / (1.0 + ridge)).sum()
                    )
                )
                for st in range(2)
            ]
        )
        z = float(pred @ f)
        assert out["filtered_state0"][t] == pytest.approx(p[0], abs=1e-12)
        assert out["predicted_state0"][t] == pytest.approx(pred[0], abs=1e-12)
        assert out["loglik"][t] == pytest.approx(np.log(z), rel=1e-9)
        assert out["state"][t] == int(np.argmax(pred))
        p = pred * f / z


def test_a_given_transition_is_the_prior_mean():
    """`docs/PLAN.md` §11a said the *counts* start at `tau*K*Pi_0`; that
    gives `(K Pi_0 + 1)/(2K)`, not `Pi_0`. The shape belongs in the prior,
    and then a given matrix is exactly what `Pi` reads before any row."""
    df = blobs(n=200, run=20, seed=4)
    pi = [0.9, 0.1, 0.3, 0.7]
    out = run(
        df,
        learn=False,
        means=[-2.0, -2.0, 2.0, 2.0],
        covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        transition=pi,
    )
    # p starts uniform, so p1 = mean of the two rows of Pi.
    assert out["predicted_state0"][0] == pytest.approx(0.5 * (pi[0] + pi[2]))
    assert out["predicted_state1"][0] == pytest.approx(0.5 * (pi[1] + pi[3]))


def test_nothing_is_reported_before_the_states_are_seeded():
    out = run(blobs(n=300, run=30), warm_rows=100)
    assert out["filtered_state0"][:100].null_count() == 100
    assert out["filtered_state0"][150] is not None


def test_the_state_means_are_the_coef():
    df = blobs(n=2000, run=100, sep=8.0, seed=5)
    bank = po.ModelBank([spec(warm_rows=200)])
    bank.fit_predict(df)
    coef = bank.coef("h")["coef"].to_list()
    centres = sorted([coef[0], coef[2]])
    assert centres[0] == pytest.approx(-4.0, abs=0.4)
    assert centres[1] == pytest.approx(4.0, abs=0.4)
    names = po.spec.coef_fields(spec())["name"].to_list()
    assert names == ["coef_state0_x0", "coef_state0_x1", "coef_state1_x0", "coef_state1_x1"]


@pytest.mark.parametrize("size", [1, 37, 500, 2000])
def test_chunk_invariance(size):
    df = blobs(n=2000, run=100, seed=6)
    want = run(df, warm_rows=200)
    bank = po.ModelBank([spec(warm_rows=200)])
    parts = [bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)]
    got = pl.concat(parts)["h"].struct.unnest()
    assert want.drop("coef").equals(got.drop("coef"))


def test_save_load_mid_stream():
    df = blobs(n=1200, run=100, seed=7)
    s = spec(warm_rows=200)
    want = run(df, warm_rows=200)
    bank = po.ModelBank([s])
    bank.fit_predict(df[:600])
    again = po.ModelBank.load_bytes(bank.save_bytes(), [s])
    got = again.fit_predict(df[600:])["h"].struct.unnest()
    assert want[600:].drop("coef").equals(got.drop("coef"))


def test_a_zero_weight_row_teaches_nothing():
    # At an infinite half-life the decay factor is exactly 1, so the only
    # thing the extra row could change is what it teaches -- which is
    # nothing.
    df = blobs(n=800, run=50, seed=8).with_columns(w=pl.lit(1.0))
    want = run(df, warm_rows=100, weight="w", half_life=float("inf"))
    padded = pl.concat(
        [
            df[:400],
            df[:1].with_columns(x0=pl.lit(1e6), x1=pl.lit(-1e6), w=pl.lit(0.0)),
            df[400:],
        ]
    )
    got = run(padded, warm_rows=100, weight="w", half_life=float("inf"))
    assert got["filtered_state0"][:400].to_list() == want["filtered_state0"][:400].to_list()
    assert got["filtered_state0"][401:].to_list() == want["filtered_state0"][400:].to_list()


@pytest.mark.parametrize("c", [2.0, 8.0, 3.0])
def test_the_clock_is_a_number(c):
    df = blobs(n=600, run=50, seed=9).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    base = spec(half_life=200.0, clock="t", gap_cap=1e12, warm_rows=100)
    scaled = spec(half_life=200.0 * c, clock="t", gap_cap=1e12, warm_rows=100)
    want = po.ModelBank([base]).fit_predict(df)["h"].struct.unnest()
    got = (
        po.ModelBank([scaled]).fit_predict(df.with_columns(t=pl.col("t") * c))["h"].struct.unnest()
    )
    if c in (2.0, 8.0):
        assert want.equals(got)
    else:
        for col in ("filtered_state0", "predicted_state0", "loglik"):
            a, b = want[col].to_numpy(), got[col].to_numpy()
            live = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[live], b[live], rtol=1e-10, atol=1e-10)


def test_exog_tvtp_reads_the_column():
    df = blobs(n=600, run=50, seed=10).with_columns(
        z=pl.when(pl.int_range(pl.len()) < 300).then(0.0).otherwise(1.0)
    )
    out = run(
        df,
        learn=False,
        means=[-2.0, -2.0, 2.0, 2.0],
        covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        exog_tvtp="z",
        tvtp_coef=[[0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]],
    )
    assert out["predicted_state0"].drop_nulls().len() > 0
    # z = 0: the base transition. z = 1: it leans hard to state 1, so a row
    # in the z = 1 stretch predicts state 1 more strongly than one in the
    # z = 0 stretch. The old assertion had an `or predicted_state0[50] < 1.0` disjunct
    # that is true on any non-degenerate row, so nothing failed if the column
    # was ignored (review 2026-09-18, T5).
    assert out["predicted_state1"][350] > out["predicted_state1"][50], (
        out["predicted_state1"][350],
        out["predicted_state1"][50],
    )
    # Control: the same stream with z = 0 everywhere is the base transition,
    # so if the column were read at all its predicted_state1 at row 350 must differ.
    base = run(
        df.with_columns(z=pl.lit(0.0)),
        learn=False,
        means=[-2.0, -2.0, 2.0, 2.0],
        covs=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        exog_tvtp="z",
        tvtp_coef=[[0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]],
    )
    assert abs(out["predicted_state1"][350] - base["predicted_state1"][350]) > 0.05, (
        out["predicted_state1"][350],
        base["predicted_state1"][350],
    )


def test_min_periods_gates_the_report_not_the_update():
    """`min_weight` withholds output and nothing else, as it does in every
    other model here. It used to withhold the row from the *filter* too, so
    a warm-up row was never learned from -- and under decay an `weight_sum` that
    plateaued below it meant a filter that never learned at all
    (docs/REVIEW-E54-E64.md H1)."""
    df = blobs(n=600, run=50, seed=12)
    base = run(df, min_weight=0.0)
    gated = run(df, min_weight=200.0)
    # Once past the threshold the two are the same filter, bit for bit.
    for col in ("filtered_state0", "filtered_state1", "state", "loglik"):
        assert base[col][250:].equals(gated[col][250:]), col
    # And below it, nulls rather than numbers.
    assert gated["filtered_state0"][:150].null_count() == 150

    # Under decay `weight_sum` plateaus, at 29.4 for a half-life of 20 rows: a
    # `min_weight` under it opens the report at the first row whose weight
    # before it meets it, row 56 by the recursion, and every row from there
    # reports. A short warm-up, so the gate that binds is this one (with 100
    # warm-up rows the seeding held the report to row 100 and a check of
    # "some row reports" could not tell; review 2026-10-06, TB10).
    decayed = run(df, half_life=20.0, min_weight=25.0, warm_rows=20)
    lam = 0.5 ** (1 / 20)
    before = (1 - lam ** np.arange(df.height)) / (1 - lam)
    opens = int(np.argmax(before >= 25.0))
    assert opens == 56
    reported = decayed["filtered_state0"].is_not_null().to_numpy()
    assert not reported[:opens].any() and reported[opens:].all()


def test_predict_reads_the_exogenous_column():
    """`exog_tvtp` rides in the targets slot, which `predict` sees too: the
    transition matrix is a function of it, so `predict` must give the step's
    answer for that row and not the one for `z = 0`
    (docs/REVIEW-E54-E64.md H2, C1)."""
    rng = np.random.default_rng(3)
    df = blobs(n=400, run=40, seed=13).with_columns(z=rng.standard_normal(400) * 3.0)
    kw = dict(
        exog_tvtp="z",
        tvtp_coef=[[2.0, -2.0, -2.0, 2.0], [1.5, -1.5, -1.5, 1.5]],
        warm_rows=50,
    )
    step = run(df, **kw)
    bank = po.ModelBank([spec(**kw)])
    bank.fit_predict(df.head(300))
    got = bank.predict(df.slice(300, 1))["h"].struct.unnest()
    for col in ("filtered_state0", "predicted_state0", "predicted_state1", "state", "loglik"):
        assert got[col][0] == pytest.approx(step[col][300], rel=1e-12, abs=1e-12), col


def test_a_missing_exogenous_value_is_the_base_transition():
    """Null or non-finite means "no value here", which is `z = 0`; the row
    is otherwise an ordinary row."""
    df = blobs(n=200, run=40, seed=14).with_columns(z=pl.lit(0.0))
    kw = dict(
        exog_tvtp="z",
        tvtp_coef=[[2.0, -2.0, -2.0, 2.0], [1.5, -1.5, -1.5, 1.5]],
        warm_rows=50,
    )
    zeros = run(df, **kw)
    nulls = run(df.with_columns(z=pl.lit(None, dtype=pl.Float64)), **kw)
    assert zeros["filtered_state0"].equals(nulls["filtered_state0"])


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"n_components": 1}, "n_components must be >= 2"),
        ({"precision_prior": 0.0}, "precision_prior must be finite"),
        ({"learn": False}, "nothing to filter with"),
        ({"means": [0.0] * 4}, "means and covs go together"),
        ({"transition": [0.5, 0.4, 0.5, 0.5]}, "not a distribution"),
        ({"warm_rows": 1}, "warm_rows must be at least n_components"),
        (
            {"init": "nope"},
            re.escape('init must be one of "first", "farthest", "k-means++", "lloyd"; got "nope"'),
        ),
        ({"tvtp_coef": [[0.0] * 4, [0.0] * 4]}, "tvtp_coef needs exog_tvtp"),
        ({"exog_tvtp": "z"}, "needs tvtp_coef"),
        ({"emit_sigma": True}, "does not apply to hmm"),
        # docs/REVIEW-E54-E64.md H4: given states that cannot be filtered
        # with. Every row would be a solve failure and every output a null.
        (
            {
                "means": [-1.0, -1.0, 1.0, 1.0],
                "covs": [1.0, 0.5, 0.4, 1.0, 1.0, 0.0, 0.0, 1.0],
            },
            "covs.0. must be symmetric",
        ),
        (
            {
                "means": [-1.0, -1.0, 1.0, 1.0],
                "covs": [1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            },
            "covs.0. must be positive definite",
        ),
        (
            {
                "means": [float("nan"), -1.0, 1.0, 1.0],
                "covs": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            },
            "means must not be NaN",
        ),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


# --- phase-4 coverage: more states and more features (tests above are k = 2,
# d = 2) ---------------------------------------------------------------------


def test_the_filter_at_three_states_and_one_feature_is_the_hamilton_recursion():
    """Every longhand test above is k = 2, d = 2. Three states over one
    feature, at fixed parameters, against the recursion (review 2026-09-18,
    phase 4)."""
    df = blobs(n=400, run=40, d=1, seed=2)
    k, d = 3, 1
    means = [-3.0, 0.0, 3.0]
    covs = [1.0, 1.0, 1.0]
    pi = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    out = run(
        df,
        n_components=k,
        features=["x0"],
        learn=False,
        means=means,
        covs=covs,
        transition=pi.flatten().tolist(),
        transition_prior=1.0,
    )
    ridge = 1e-2
    mu = np.array(means).reshape(k, d)
    p = np.full(k, 1.0 / k)
    x = df.select("x0").to_numpy()
    for t in range(len(x)):
        pred = p @ pi
        f = np.array(
            [
                np.exp(
                    -0.5
                    * (
                        d * np.log(2 * np.pi)
                        + d * np.log(1.0 + ridge)
                        + ((x[t] - mu[st]) ** 2 / (1.0 + ridge)).sum()
                    )
                )
                for st in range(k)
            ]
        )
        z = float(pred @ f)
        assert out["filtered_state0"][t] == pytest.approx(p[0], abs=1e-12)
        assert out["predicted_state0"][t] == pytest.approx(pred[0], abs=1e-12)
        assert out["loglik"][t] == pytest.approx(np.log(z), rel=1e-9)
        p = pred * f / z


def test_the_full_covariance_filter_recovers_states_at_four_features():
    """The `full` shape solves a d x d system per state; the tests above only
    reach d = 2. Four features, two well-separated blobs (review 2026-09-18,
    phase 4)."""
    df = blobs(n=4000, run=200, sep=4.0, d=4, seed=3)
    out = run(df, features=["x0", "x1", "x2", "x3"], warm_rows=200, covariance_type="full")
    live = out.drop("coef", "withheld_reason").drop_nulls()
    assert live.height > 3000
    state = out["state"].to_numpy()
    truth = df["g"].to_numpy()
    ok = np.isfinite(state.astype(float))
    agree = (state[ok] == truth[ok]).mean()
    assert max(agree, 1 - agree) > 0.9, agree
    p = live.select("filtered_state0", "filtered_state1").to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((p >= 0) & (p <= 1)).all()


def test_state_is_the_most_likely_under_the_predicted_probabilities():
    """Task 239: ``state`` is the argmax of ``predicted_state<j>``, the row's
    prediction from the rows before it, not of ``filtered_state<j>``, which the
    rows before left; on a stream that switches the two differ."""
    out = run(blobs(n=3000, run=50), warm_rows=200, transition_prior=0.1)
    live = out.drop("coef", "withheld_reason").drop_nulls()
    assert live.height > 2500
    pred = live.select("predicted_state0", "predicted_state1").to_numpy()
    assert (live["state"].to_numpy() == pred.argmax(axis=1)).all()
    filt = live.select("filtered_state0", "filtered_state1").to_numpy()
    assert (pred.argmax(axis=1) != filt.argmax(axis=1)).sum() > 0


# --- task 278 (b08): the parameters no third-party library computes. Each
# shown taking effect where it should, and to rounding nowhere else; the
# transition prior's formula itself is written by hand in the Rust tests
# (`hmm.rs`, `the_transition_is_the_decayed_counts_plus_the_prior_written_by_hand`),
# where `cargo mutants` sees it. -------------------------------------------

GIVEN_MEANS = [-1.5, -1.5, 1.5, 1.5]
GIVEN_COVS = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
GIVEN_PI = [0.9, 0.1, 0.2, 0.8]
PROBS = ("filtered_state0", "filtered_state1", "predicted_state0", "predicted_state1")


def gap(a, b, cols=PROBS):
    """The largest difference between two runs over `cols`, which must be
    null on the same rows."""
    worst = 0.0
    for c in cols:
        assert a[c].is_null().equals(b[c].is_null()), c
        x, y = a[c].drop_nulls().to_numpy(), b[c].drop_nulls().to_numpy()
        worst = max(worst, float(np.abs(x - y).max()))
    return worst


def given(**kw):
    return {"means": GIVEN_MEANS, "covs": GIVEN_COVS, "transition": GIVEN_PI} | kw


def test_transition_prior_takes_effect_where_the_counts_are_learned():
    """Under ``learn=True`` the pseudo-count weighs against the learned
    counts, so each value gives its own filter: measured 2026-10-25 against
    the default 1, max |d p| 0.72 at 0, 0.12 at 0.3 and 0.063 at 5, and no
    two of the four closer than that 0.063. Without learning there are no counts to weigh against,
    and a row of ``Pi`` is ``tau K Pi0 / (tau K)``, the given matrix: every
    value is the default's filter to rounding (2.2e-16 on the probabilities,
    1.8e-15 on ``loglik``), 0 included."""
    df = blobs(n=1500, run=60, seed=21)
    runs = {tau: run(df, **given(transition_prior=tau)) for tau in (0.0, 0.3, 1.0, 5.0)}
    taus = list(runs)
    for i, a in enumerate(taus):
        for b in taus[:i]:
            assert gap(runs[a], runs[b]) > 1e-2, (a, b)
    fixed = run(df, **given(learn=False))
    for tau in (0.0, 0.3, 0.7, 3.0, 5.0, 1e12):
        other = run(df, **given(learn=False, transition_prior=tau))
        assert gap(fixed, other) <= 1e-15, tau
        assert gap(fixed, other, ("loglik",)) <= 1e-14, tau


def test_a_large_transition_prior_is_the_fixed_matrix_at_rate_one_over_tau():
    """As ``tau`` grows the learned counts weigh nothing against it, and
    ``Pi`` is the given matrix while the states go on learning: the limit is
    ``tvtp`` with ``B = 0`` and ``A = log Pi``, which learns the states and no
    count. The distance falls as ``1/tau`` (the counts, ~``n`` rows, over
    ``tau K``): measured 2026-10-25 at 3.28e-4, 3.28e-7 and 3.28e-10 for
    ``tau`` = 1e6, 1e9 and 1e12."""
    df = blobs(n=1500, run=60, seed=21).with_columns(z=pl.lit(0.37))
    log_pi = np.log(GIVEN_PI).tolist()
    limit = run(
        df,
        means=GIVEN_MEANS,
        covs=GIVEN_COVS,
        exog_tvtp="z",
        tvtp_coef=[log_pi, [0.0] * 4],
        coef_every=0,
    )
    # Evidence the states learned in the limit run: they left where they
    # were given.
    assert np.abs(np.array(limit["coef"][-1].to_list()) - GIVEN_MEANS).max() > 0.3
    gaps = [gap(limit, run(df, **given(transition_prior=tau))) for tau in (1e6, 1e9, 1e12)]
    assert 1e-5 < gaps[0] < 1e-3, gaps
    for wide, narrow in zip(gaps[:-1], gaps[1:], strict=True):
        assert 500 < wide / narrow < 2000, gaps


def test_tvtp_with_no_slope_is_the_given_transition():
    """``softmax(log Pi + 0 z)`` is ``Pi``, so with the states fixed the
    ``tvtp`` filter is the one given ``transition = Pi``, which the hmmlearn
    oracle holds (``tests/test_second_opinion.py``): to 8.9e-16 measured
    2026-10-25, whatever ``z`` is."""
    df = blobs(n=1500, run=60, seed=21)
    want = run(df, **given(learn=False))
    log_pi = np.log(GIVEN_PI).tolist()
    for z in (0.0, 0.37, -40.0):
        got = run(
            df.with_columns(z=pl.lit(z)),
            means=GIVEN_MEANS,
            covs=GIVEN_COVS,
            learn=False,
            exog_tvtp="z",
            tvtp_coef=[log_pi, [0.0] * 4],
        )
        assert gap(want, got) <= 1e-14, z
        assert gap(want, got, ("loglik",)) <= 1e-13, z


@pytest.mark.parametrize("size", [1, 7, 600])
def test_tvtp_is_chunk_and_save_invariant(size):
    """Under ``tvtp``, learning the states from a warm-up, with a ``z`` that
    moves: chunked at 1, 7 and 600 rows and saved inside the warm-up and
    after it, the run is the one-chunk run."""
    rng = np.random.default_rng(31)
    df = blobs(n=900, run=60, seed=22).with_columns(z=rng.standard_normal(900))
    kw = dict(
        exog_tvtp="z", tvtp_coef=[[2.0, -2.0, -2.0, 2.0], [1.5, -1.5, -1.5, 1.5]], warm_rows=100
    )
    want = run(df, **kw).drop("coef")
    bank = po.ModelBank([spec(**kw)])
    parts = [bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)]
    assert want.equals(pl.concat(parts)["h"].struct.unnest().drop("coef"))
    for cut in (60, 450):
        bank = po.ModelBank([spec(**kw)])
        head = bank.fit_predict(df[:cut])["h"].struct.unnest().drop("coef")
        again = po.ModelBank.load_bytes(bank.save_bytes(), [spec(**kw)])
        tail = again.fit_predict(df[cut:])["h"].struct.unnest().drop("coef")
        assert want.equals(pl.concat([head, tail])), cut


def test_given_diagonal_covs_are_one_filter_in_every_shape():
    """Equal diagonal ``covs``, fixed: the full matrix is diagonal, the
    pooled one is each state's, so ``full``, ``tied`` and ``diag`` are one
    filter, to 3.6e-15 measured 2026-10-25 (``loglik`` included). With
    unequal diagonal ``covs`` the pooled matrix is neither state's, and
    ``tied`` parts from the other two while ``diag`` still is ``full``."""
    df = blobs(n=1500, run=60, seed=21)
    cols = (*PROBS, "loglik")

    def shapes(covs):
        return {
            c: run(
                df,
                means=GIVEN_MEANS,
                covs=covs,
                transition=GIVEN_PI,
                learn=False,
                covariance_type=c,
            )
            for c in ("full", "tied", "diag")
        }

    equal = shapes([1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0])
    assert gap(equal["full"], equal["tied"], cols) <= 1e-13
    assert gap(equal["full"], equal["diag"], cols) <= 1e-13
    unequal = shapes([1.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 3.0])
    assert gap(unequal["full"], unequal["diag"], cols) <= 1e-13
    assert gap(unequal["full"], unequal["tied"]) > 1e-2


def test_learn_false_keeps_the_given_means_exactly():
    """``learn=False`` holds the given states exactly: the ``coef`` written
    on every row is the given means to the bit. Learning from the same start
    moves them (to about -1.95 and 2.02 here) and the probabilities, by up
    to 0.35."""
    df = blobs(n=1500, run=60, seed=21)
    held = run(df, **given(learn=False, coef_every=0))
    assert held["coef"].null_count() == 0
    assert all(c == GIVEN_MEANS for c in held["coef"].to_list())
    moving = run(df, **given(coef_every=0))
    assert np.abs(np.array(moving["coef"][-1].to_list()) - GIVEN_MEANS).max() > 0.3
    assert gap(held, moving) > 0.1


def three_blobs(n=300, seed=11):
    """Three overlapping blobs in two columns, the labels drawn at random:
    the four seeding rules each place the states differently here."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
    x = centres[rng.integers(0, 3, n)] + rng.standard_normal((n, 2))
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})


def seeded_means(df, model, **kw):
    """The centres or state means right after the seeding: the whole frame
    is the warm-up, and at no decay the replay weighs every row as one."""
    if model == "hmm":
        s = po.spec.hmm("m", features=["x0", "x1"], n_components=3, precision_prior=1e-2, **kw)
    else:
        s = po.spec.kmeans("m", features=["x0", "x1"], n_clusters=3, standardize=False, **kw)
    bank = po.ModelBank([s])
    bank.fit_predict(df)
    return np.array(bank.coef("m")["coef"].to_list())


def test_each_seeding_rule_seeds_the_states_kmeans_seeds():
    """The states are seeded by ``kmeans``' rules: for each ``init`` and
    ``seed``, the state means after the warm-up are the centres ``kmeans``
    places on the same rows (unstandardized, so both measure plain
    distances), to 4.4e-16 measured 2026-10-25. ``kmeans`` seeds from the
    rows within its far cut; ``farthest`` agreeing, which picks the row
    furthest out, is the evidence no row was cut here. The rules disagree
    with each other, ``k-means++`` moves with ``seed`` (four seeds, four
    seedings), and ``lloyd`` reaches one optimum from its ten starts, the
    states possibly relabelled."""
    df = three_blobs()
    kw = dict(half_life=float("inf"), warm_rows=df.height)
    seen = {}
    for rule in ("first", "farthest", "k-means++", "lloyd"):
        for seed in (0, 1, 2, 5):
            h = seeded_means(df, "hmm", init=rule, seed=seed, **kw)
            k = seeded_means(df, "kmeans", init=rule, seed=seed, **kw)
            assert np.abs(h - k).max() <= 1e-14, (rule, seed)
            seen[rule, seed] = h
    as_set = {key: sorted(map(tuple, np.round(v, 12).reshape(3, 2))) for key, v in seen.items()}
    for rule in ("first", "farthest"):
        assert all((seen[rule, s] == seen[rule, 0]).all() for s in (1, 2, 5)), rule
    assert len({tuple(map(tuple, as_set["k-means++", s])) for s in (0, 1, 2, 5)}) == 4
    assert all(as_set["lloyd", s] == as_set["lloyd", 0] for s in (1, 2, 5))
    distinct = {
        tuple(map(tuple, as_set[r, 0])) for r in ("first", "farthest", "k-means++", "lloyd")
    }
    assert len(distinct) == 4


@pytest.mark.parametrize("rule", ["first", "farthest", "k-means++", "lloyd"])
def test_seed_is_reproducible_none_is_zero_and_moves_only_the_drawn_rules(rule):
    """The same ``seed`` gives the same run, ``None`` is 0, and the default
    rule is ``lloyd``. ``k-means++`` and ``lloyd`` draw their starts from the
    seed, so 0 and 5 seed differently (``lloyd`` here only in which state is
    which); ``first`` and ``farthest`` draw nothing and do not move."""
    df = blobs(n=600, run=50, seed=3)

    def go(**kw):
        return run(df, warm_rows=200, init=rule, **kw)

    zero = go(seed=0)
    assert zero.equals(go(seed=None))
    assert go(seed=5).equals(go(seed=5))
    if rule in ("first", "farthest"):
        assert zero.equals(go(seed=5))
    else:
        assert not zero.equals(go(seed=5))
    if rule == "lloyd":
        default = (
            po.ModelBank(
                [
                    po.spec.hmm(
                        "h",
                        features=["x0", "x1"],
                        n_components=2,
                        precision_prior=1e-2,
                        half_life=1e9,
                        warm_rows=200,
                    )
                ]
            )
            .fit_predict(df)["h"]
            .struct.unnest()
        )
        assert default.equals(zero)


def plain(df, **kw):
    """An ``hmm`` with every seeding parameter at its default."""
    s = po.spec.hmm(
        "h", features=["x0", "x1"], n_components=2, precision_prior=1e-2, half_life=1e9, **kw
    )
    return po.ModelBank([s]).fit_predict(df)["h"].struct.unnest()


def test_warm_rows_defaults_to_1000():
    """The default buffer is 1,000 rows: the run is ``warm_rows=1000``'s,
    the first 1,000 rows report null with ``below_warm_rows``, and row 1,000
    is the first to report."""
    df = blobs(n=1300, run=50, seed=4)
    out = plain(df)
    assert out.equals(plain(df, warm_rows=1000))
    reported = out["filtered_state0"].is_not_null().to_numpy()
    assert not reported[:1000].any() and reported[1000:].all()
    assert (out["withheld_reason"][:1000] == "below_warm_rows").all()
    assert out["withheld_reason"][1000:].null_count() == 300


def test_warm_rows_defaults_to_n_components_above_1000():
    """Past 1,000 states the default buffer is ``n_components`` rows, the
    fewest that seed every state: 1,001 states report from row 1,001, and a
    smaller ``warm_rows`` is refused."""
    df = pl.DataFrame({"x0": np.random.default_rng(1).standard_normal(1010)})

    def big(**kw):
        return po.spec.hmm(
            "h",
            features=["x0"],
            n_components=1001,
            precision_prior=1e-2,
            half_life=1e9,
            init="first",
            **kw,
        )

    out = po.ModelBank([big()]).fit_predict(df)["h"].struct.unnest()
    reported = out["filtered_state0"].is_not_null().to_numpy()
    assert not reported[:1001].any() and reported[1001:].all()
    with pytest.raises(
        ValueError, match=re.escape("warm_rows must be at least n_components (1001)")
    ):
        big(warm_rows=1000)


def test_warm_rows_at_n_components_seeds_from_that_many_rows():
    out = plain(blobs(n=300, run=30), warm_rows=2)
    reported = out["filtered_state0"].is_not_null().to_numpy()
    assert not reported[:2].any() and reported[2:].all()


@pytest.mark.parametrize("size", [1, 7])
def test_chunks_and_a_save_inside_the_warm_up(size):
    """Cut inside the warm-up, where the rows wait in the buffer: chunked at
    1 and 7 rows, and saved at row 120 of 200, the run is the one-chunk
    run, the seeding included."""
    df = blobs(n=400, run=40, seed=23)
    want = run(df, warm_rows=200).drop("coef")
    bank = po.ModelBank([spec(warm_rows=200)])
    parts = [bank.fit_predict(df[i : min(i + size, 260)]) for i in range(0, 260, size)]
    parts.append(bank.fit_predict(df[260:]))
    assert want.equals(pl.concat(parts)["h"].struct.unnest().drop("coef"))
    bank = po.ModelBank([spec(warm_rows=200)])
    head = bank.fit_predict(df[:120])["h"].struct.unnest().drop("coef")
    again = po.ModelBank.load_bytes(bank.save_bytes(), [spec(warm_rows=200)])
    tail = again.fit_predict(df[120:])["h"].struct.unnest().drop("coef")
    assert want.equals(pl.concat([head, tail]))


def test_a_group_shorter_than_warm_rows_reports_null_throughout():
    """Each group seeds from its own first ``warm_rows`` rows: at the
    default 1,000, a group of 1,200 rows reports from its 1,001st, and one of
    300 rows, interleaved with it, never seeds and reports null with
    ``below_warm_rows`` on every row."""
    long = blobs(n=1200, run=50, seed=24).with_columns(g=pl.lit("long"))
    short = blobs(n=300, run=50, seed=25).with_columns(g=pl.lit("short"))
    df = pl.concat([long[:300], short, long[300:]]).with_row_index("i")
    out = df.select("i", "g").hstack(plain(df.drop("i"), group="g"))
    s = out.filter(pl.col("g") == "short")
    assert s["filtered_state0"].null_count() == 300
    assert (s["withheld_reason"] == "below_warm_rows").all()
    lg = out.filter(pl.col("g") == "long")["filtered_state0"].is_not_null().to_numpy()
    assert not lg[:1000].any() and lg[1000:].all()
