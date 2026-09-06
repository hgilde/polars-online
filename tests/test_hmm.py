"""E60: a Gaussian hidden Markov model, filtered online.

`ew_class` answers "which regime does this row look like". An `hmm` answers
"which regime are we in", and the difference is the transition matrix. Two
things earn that here: the filter reduces to the classifier when the chain
is uniform, and it is the longhand Hamilton recursion at fixed parameters.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


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
        "k": 2,
        "precision_prior": 1e-2,
        "halflife": 1e9,
        "warm_rows": 100,
        "seed_rule": "lloyd",
    }
    d.update(kw)
    return po.spec.hmm("h", **d)


def run(df, **kw):
    return po.ModelBank([spec(**kw)]).fit_predict(df)["h"].struct.unnest()


def test_the_filter_finds_the_states_and_the_chain():
    df = blobs(n=4000, run=200)
    out = run(df, warm_rows=200)
    # `coef` is null except where the schedule reports it, so it is dropped
    # before asking which rows are live.
    live = out.drop("coef").drop_nulls()
    assert live.height > 3000
    # The reported state tracks the blob, up to which state is which.
    state = out["state"].to_numpy()
    truth = df["g"].to_numpy()
    ok = np.isfinite(state.astype(float))
    agree = (state[ok] == truth[ok]).mean()
    assert max(agree, 1 - agree) > 0.95, agree
    # The chain is sticky, and `p` is a distribution on every live row.
    p = live.select("p_0", "p_1").to_numpy()
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
        assert out["p_0"][t] == pytest.approx(p[0], abs=1e-12)
        assert out["p1_0"][t] == pytest.approx(pred[0], abs=1e-12)
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
    assert out["p1_0"][0] == pytest.approx(0.5 * (pi[0] + pi[2]))
    assert out["p1_1"][0] == pytest.approx(0.5 * (pi[1] + pi[3]))


def test_nothing_is_reported_before_the_states_are_seeded():
    out = run(blobs(n=300, run=30), warm_rows=100)
    assert out["p_0"][:100].null_count() == 100
    assert out["p_0"][150] is not None


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
    # At an infinite halflife the decay factor is exactly 1, so the only
    # thing the extra row could change is what it teaches -- which is
    # nothing.
    df = blobs(n=800, run=50, seed=8).with_columns(w=pl.lit(1.0))
    want = run(df, warm_rows=100, weight="w", halflife=float("inf"))
    padded = pl.concat(
        [
            df[:400],
            df[:1].with_columns(x0=pl.lit(1e6), x1=pl.lit(-1e6), w=pl.lit(0.0)),
            df[400:],
        ]
    )
    got = run(padded, warm_rows=100, weight="w", halflife=float("inf"))
    assert got["p_0"][:400].to_list() == want["p_0"][:400].to_list()
    assert got["p_0"][401:].to_list() == want["p_0"][400:].to_list()


@pytest.mark.parametrize("c", [2.0, 8.0, 3.0])
def test_the_clock_is_a_number(c):
    df = blobs(n=600, run=50, seed=9).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    base = spec(halflife=200.0, clock="t", max_dclock=1e12, warm_rows=100)
    scaled = spec(halflife=200.0 * c, clock="t", max_dclock=1e12, warm_rows=100)
    want = po.ModelBank([base]).fit_predict(df)["h"].struct.unnest()
    got = (
        po.ModelBank([scaled]).fit_predict(df.with_columns(t=pl.col("t") * c))["h"].struct.unnest()
    )
    if c in (2.0, 8.0):
        assert want.equals(got)
    else:
        for col in ("p_0", "p1_0", "loglik"):
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
    assert out["p1_0"].drop_nulls().len() > 0
    # z = 0: row 0 of Pi is uniform. z = 1: it leans hard to state 1, so a
    # row that was in state 0 predicts state 1.
    assert out["p1_1"][350] > out["p1_1"][50] or out["p1_0"][50] < 1.0


def test_min_periods_gates_the_report_not_the_update():
    """`min_periods` withholds output and nothing else, as it does in every
    other model here. It used to withhold the row from the *filter* too, so
    a warm-up row was never learned from -- and under decay an `n_eff` that
    plateaued below it meant a filter that never learned at all
    (docs/REVIEW-E54-E64.md H1)."""
    df = blobs(n=600, run=50, seed=12)
    base = run(df, min_periods=0.0)
    gated = run(df, min_periods=200.0)
    # Once past the threshold the two are the same filter, bit for bit.
    for col in ("p_0", "p_1", "state", "loglik"):
        assert base[col][250:].equals(gated[col][250:]), col
    # And below it, nulls rather than numbers.
    assert gated["p_0"][:150].null_count() == 150

    # Under decay `n_eff` plateaus; a `min_periods` above the plateau must
    # still be reached by the row count, not silently never.
    decayed = run(df, halflife=20.0, min_periods=25.0)
    assert decayed["p_0"].null_count() < df.height


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
    for col in ("p_0", "p1_0", "p1_1", "state", "loglik"):
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
    assert zeros["p_0"].equals(nulls["p_0"])


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"k": 1}, "k must be >= 2"),
        ({"precision_prior": 0.0}, "precision_prior must be finite"),
        ({"learn": False}, "nothing to filter with"),
        ({"means": [0.0] * 4}, "means and covs go together"),
        ({"transition": [0.5, 0.4, 0.5, 0.5]}, "not a distribution"),
        ({"warm_rows": 1}, "warm_rows must be at least k"),
        ({"seed_rule": "nope"}, "unknown hmm seed_rule"),
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


def test_the_expression_equals_the_bank():
    df = blobs(n=400, run=40, seed=11)
    want = run(df, warm_rows=100)
    with pytest.warns(po.InMemoryExpressionWarning):
        got = df.select(
            pl.col("x0")
            .online.hmm(
                ["x1"], k=2, precision_prior=1e-2, halflife=1e9, warm_rows=100, seed_rule="lloyd"
            )
            .alias("h")
        )["h"].struct.unnest()
    assert want.equals(got)
