"""E61: how long has the current regime lasted?

Every other detector here answers "has something changed?" with a statistic.
`bocpd` keeps a posterior over the *run length* -- how many rows since the
last break -- so the answer carries the age of the regime with it.

The recursion is Adams & MacKay (2007) Algorithm 1, and the first test is
that recursion written out in numpy and checked at every row. The rest are
the properties a caller would rely on: a variance step found, the cost of
truncation, what the robust emission actually buys, and the arithmetic of
`P(r = 0)` that is the reason the reported number is `P(r <= 1)`.
"""

from math import lgamma

import numpy as np
import polars as pl
import pytest

import polars_online as po


def run(x, **kw):
    """`x` is `(n, d)`; returns the unnested output frame."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    df = pl.DataFrame({f"x{i}": x[:, i] for i in range(x.shape[1])})
    for extra in ("h", "w", "g"):
        if extra in kw:
            df = df.with_columns(pl.Series(extra, kw.pop(extra)))
    kw.setdefault("features", [f"x{i}" for i in range(x.shape[1])])
    spec = po.spec.bocpd("b", **kw)
    return po.ModelBank([spec]).fit_predict(df)["b"].struct.unnest()


def normals(n, seed=0, loc=0.0, scale=1.0):
    return np.random.default_rng(seed).normal(loc, scale, n)


# --- Algorithm 1, written out ------------------------------------------------


def longhand(x, hazard, k0=1.0, nu0=2.0, psi0=1.0):
    """Adams & MacKay Algorithm 1 in plain probabilities, diagonal
    normal-inverse-gamma emission, no truncation. Returns, per row, the full
    run-length posterior *after* the row and the row's log score.

    Line 6 of their algorithm is the whole of the bookkeeping:
    `nu^(r+1)_{t+1} = nu^(r)_t + u(x_t)` and `nu^(0)_{t+1} = nu_prior`. Slot
    `j` holds exactly the `j` rows the hypothesis `r_t = j` says come before
    the next row in its run, so the slot pushed at the front holds nothing.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float).T).T
    h = 1.0 / hazard
    runs = [(0.0, np.zeros(x.shape[1]), np.zeros(x.shape[1]))]
    joint = np.array([1.0])
    posts, scores = [], []
    for row in x:
        pi = []
        for cnt, sx, ss in runs:
            kn, nun = k0 + cnt, nu0 + cnt
            mun = sx / kn
            xbar = sx / cnt if cnt > 0 else np.zeros_like(sx)
            psin = psi0 + (ss - cnt * xbar**2) + k0 * cnt / kn * xbar**2
            var = psin * (kn + 1.0) / (nun * kn)
            z = (row - mun) / np.sqrt(var)
            ln = (
                lgamma((nun + 1.0) / 2.0)
                - lgamma(nun / 2.0)
                - 0.5 * np.log(nun * np.pi * var)
                - 0.5 * (nun + 1.0) * np.log1p(z * z / nun)
            )
            pi.append(np.exp(ln.sum()))
        pi = np.array(pi)
        pre = joint / joint.sum()
        scores.append(np.log((pre * pi).sum()))
        new = np.empty(len(runs) + 1)
        new[1:] = joint * pi * (1.0 - h)
        new[0] = (joint * pi * h).sum()
        posts.append(new / new.sum())
        zero = (0.0, np.zeros(x.shape[1]), np.zeros(x.shape[1]))
        runs = [zero] + [(c + 1, sx + row, ss + row**2) for c, sx, ss in runs]
        joint = new
    return posts, np.array(scores)


def test_the_posterior_is_the_longhand_algorithm_one():
    """Every reported number against the recursion written out, at every
    row, with the truncation off."""
    x = np.concatenate([normals(60, 1), normals(60, 2, loc=3.0)])
    out = run(x, hazard=50.0, prior_nu=2.0, prior_scale=[1.0], truncate=0.0)
    posts, scores = longhand(x, 50.0)
    # Row one is gated out: `P(r <= 1)` is 1 there whatever the data, so
    # every comparison starts at row two -- the row is still *learned*.
    assert out["logscore"][0] is None
    assert out["logscore"].to_list()[1:] == pytest.approx(list(scores[1:]), abs=1e-12)
    p_change, mode, mean = [], [], []
    for p in posts:
        p_change.append(p[0] + (p[1] if len(p) > 1 else 0.0))
        mode.append(int(np.argmax(p)))
        mean.append(float((np.arange(len(p)) * p).sum()))
    assert out["p_change"][0] is None
    assert out["p_change"].to_list()[1:] == pytest.approx(p_change[1:], abs=1e-12)
    # `run_mode` and `run_mean` are read from the posterior *before* the
    # row, which is the longhand's previous entry; and slot `j` holds `j`
    # rows, so the slot index is the run length.
    assert out["run_mode"].to_list()[1:] == mode[:-1]
    assert out["run_mean"].to_list()[1:] == pytest.approx(mean[:-1], abs=1e-10)


def test_the_mass_at_run_length_zero_is_exactly_the_hazard():
    """Why `p_change` is `P(r <= 1)` and not `P(r = 0)`: the changepoint
    branch and the growth branch share the same predictive, so the
    normalised mass at `r = 0` is `H` on every row whatever the data. It
    carries no information at all, which the longhand shows outright."""
    x = np.concatenate([normals(40, 3), normals(40, 4, loc=8.0)])
    for hazard in (25.0, 250.0):
        posts, _ = longhand(x, hazard)
        assert [p[0] for p in posts] == pytest.approx([1.0 / hazard] * len(posts), abs=1e-14)


def test_two_features_are_a_product_of_student_ts():
    """The diagonal emission on `d > 1`, against the same longhand."""
    rng = np.random.default_rng(11)
    x = np.column_stack([rng.normal(0, 1, 80), rng.normal(0, 2, 80)])
    out = run(x, hazard=40.0, prior_nu=2.0, prior_scale=[1.0], truncate=0.0)
    _, scores = longhand(x, 40.0)
    assert out["logscore"].to_list()[1:] == pytest.approx(list(scores[1:]), abs=1e-12)


# --- what it detects ---------------------------------------------------------


def test_a_variance_step_is_found():
    """Adams & MacKay's own finance fixture shape: zero-mean rows whose
    variance steps, their gamma prior on the inverse variance (`a = 1`,
    `b = 1e-4`, i.e. `prior_nu = 2a` and `prior_scale = 2b`) and their
    `hazard = 250`."""
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.normal(0, 0.005, 300), rng.normal(0, 0.05, 300), rng.normal(0, 0.005, 300)]
    )
    out = run(x, hazard=250.0, prior_nu=2.0, prior_scale=[2e-4], truncate=1e-6, max_run=600)
    p = np.asarray(out["p_change"].fill_null(0.0).to_list())
    mode = out["run_mode"].to_list()
    quiet = float(np.median(p[50:300]))
    assert quiet < 2.0 / 250.0, "a quiet stretch sits near the hazard"
    assert p[300:310].max() > 0.5, "the step up was missed"
    assert 320 - mode[320] == 300, "and the run it started is dated to the step"
    # **The two directions are not symmetric**, and that is the model rather
    # than this implementation: a variance *increase* makes the next row
    # wildly unlikely under the old run, so the evidence is immediate; a
    # *decrease* only makes it unsurprising, and the old wide predictive
    # still explains it. The alarm barely moves (2.7x the quiet level, not
    # 150x); what finds it is the run length.
    assert p[600:700].max() < 0.1 < p[300:310].max()
    assert p[600:700].max() > 2.0 * quiet
    assert abs((660 - mode[660]) - 600) <= 1, "the step down is dated too"


def test_the_run_mode_says_where_it_broke():
    """`run_mode` is the run length before the row, so **`t - run_mode` is
    the row the current run began on** -- the useful output, and the one
    that is stable. `p_change` is the alarm and it is spiky: on a mean shift
    with a diffuse prior it barely lifts, because a fresh run's *prior*
    predictive is not much better at explaining one four-sigma row than the
    old run is."""
    x = np.concatenate([normals(200, 7), normals(200, 8, loc=4.0)])
    out = run(x, hazard=250.0, prior_scale=[1.0], prior_nu=2.0)
    mode = out["run_mode"].to_list()
    assert mode[199] == 199, "before: one run, as old as the stream"
    assert mode[202] < 20, "after: the old run is abandoned two rows in"
    for t in (203, 210):
        assert abs((t - mode[t]) - 200) <= 1, f"dated to within a row at {t}"
    for t in (220, 250, 399):
        assert t - mode[t] == 200, f"and settled on the break row by {t}"
    assert max(out["p_change"].to_list()[200:210]) < 0.05, "the alarm hardly moves"


def test_the_predictive_mean_follows_the_regime():
    x = np.concatenate([normals(150, 9), normals(150, 10, loc=5.0)])
    out = run(x, hazard=100.0, prior_scale=[1.0], prior_nu=2.0)
    pred = out["pred_x0"].to_list()
    assert abs(pred[149]) < 0.3, "the old regime's mean"
    assert abs(pred[190] - 5.0) < 0.6, "the new one's, 40 rows later"


def test_the_log_score_prefers_the_model_that_saw_the_break():
    """The point of the run-length posterior: a model that can restart
    scores the second regime better than one that cannot (`hazard` huge)."""
    x = np.concatenate([normals(150, 12), normals(150, 13, loc=6.0)])
    quick = run(x, hazard=50.0, prior_scale=[1.0], prior_nu=2.0)
    stuck = run(x, hazard=1e9, prior_scale=[1.0], prior_nu=2.0)
    # Right after the break, where it matters: 16 nats over 50 rows. Give it
    # long enough and the stuck model's single run drags itself over to the
    # new mean too, and the gap closes.
    after = slice(150, 200)
    assert sum(quick["logscore"].to_list()[after]) > sum(stuck["logscore"].to_list()[after]) + 5


# --- the knobs ---------------------------------------------------------------


def test_truncation_changes_little_and_max_run_caps_the_state():
    x = np.concatenate([normals(200, 14), normals(200, 15, loc=3.0)])
    base = dict(hazard=100.0, prior_scale=[1.0], prior_nu=2.0)
    exact = run(x, truncate=0.0, **base)
    cut = run(x, truncate=1e-4, **base)
    worst = max(
        abs(a - b)
        for a, b in zip(exact["p_change"].to_list()[1:], cut["p_change"].to_list()[1:], strict=True)
    )
    # Runs holding up to `truncate` each are dropped, so `P(r <= 1)` can move
    # by a small multiple of it.
    assert worst < 1e-3, worst
    # And the run length still means rows, not slots: `max_run` folds the
    # tail, and truncation punches holes in the middle of the vector, so
    # each run carries its own length rather than being found by position.
    assert exact["run_mode"].to_list() == cut["run_mode"].to_list()
    capped = run(x, truncate=0.0, max_run=25, **base)
    assert max(v for v in capped["run_mode"].to_list() if v is not None) <= 25


def test_the_prior_scale_sets_what_counts_as_surprising():
    """The one parameter a caller must set from their data. `prior_scale` is
    the prior guess at the variance, and the failure when it is too large is
    silence: the prior predictive is so wide that no row is ever surprising
    under it, so a real break is never found. Here the data has variance
    1e-6 and a four-sigma break at row 150; a `prior_scale` of 1e-6 dates it
    to the row, and 1e-3 and up never see it at all."""
    x = np.concatenate([normals(150, 16, scale=1e-3), normals(150, 17, loc=4e-3, scale=1e-3)])
    fitting = run(x, hazard=250.0, prior_nu=2.0, prior_scale=[1e-6])
    mode = fitting["run_mode"].to_list()
    found = next(t for t in range(150, 300) if mode[t] < 20)
    assert found <= 152 and found - mode[found] == 150
    assert max(fitting["p_change"].to_list()[150:200]) > 0.5
    for scale in (1e-3, 1.0, 100.0):
        blind = run(x, hazard=250.0, prior_nu=2.0, prior_scale=[scale])
        assert min(blind["run_mode"].to_list()[151:]) > 20, f"{scale} saw it"
        assert max(blind["p_change"].to_list()[150:200]) < 0.05


def test_the_hazard_can_be_read_from_a_column():
    """A per-row hazard, declared in the target slot the way a weight is."""
    x = normals(200, 17)
    never = run(x, hazard_col="h", h=np.full(200, 1e9), prior_scale=[1.0], prior_nu=2.0)
    often = run(x, hazard_col="h", h=np.full(200, 5.0), prior_scale=[1.0], prior_nu=2.0)
    # One break every billion rows a priori: on a stationary stream nothing
    # ever restarts. One in five: the posterior never commits to a long run.
    assert never["run_mode"][199] == 199
    assert max(often["run_mode"].to_list()[20:]) < 20
    # And a column of one constant is that constant.
    flat = run(x, hazard_col="h", h=np.full(200, 40.0), prior_scale=[1.0], prior_nu=2.0)
    plain = run(x, hazard=40.0, prior_scale=[1.0], prior_nu=2.0)
    assert flat["p_change"].to_list()[1:] == pytest.approx(plain["p_change"].to_list()[1:])
    # A hazard that switches: the same stream, patient and then twitchy.
    h = np.where(np.arange(200) < 100, 1e9, 5.0)
    mixed = run(x, hazard_col="h", h=h, prior_scale=[1.0], prior_nu=2.0)
    assert mixed["run_mode"][99] == 99
    assert mixed["run_mode"][199] < 40


def test_an_unusable_hazard_in_the_column_is_refused_naming_the_row():
    """A hazard is the expected rows between changepoints, so 1 or less is
    not one. Such a row used to report nulls and vanish from the posterior
    with nothing said (docs/REVIEW-E54-E64.md B1); null still means "no
    value here" and falls back to the spec's own hazard."""
    x = normals(60, 21)
    h = np.full(60, 40.0)
    h[17] = 0.5
    with pytest.raises(ValueError, match="row 17; a hazard is the expected rows"):
        run(x, hazard_col="h", h=h, prior_scale=[1.0], prior_nu=2.0)

    # Null falls back, and the fallback is the ordinary run.
    h = np.full(60, 40.0)
    h[17] = np.nan
    df = pl.DataFrame({"x0": x, "h": h})
    spec = po.spec.bocpd(
        "b", features=["x0"], hazard=40.0, hazard_col="h", prior_scale=[1.0], prior_nu=2.0
    )
    got = po.ModelBank([spec]).fit_predict(df)["b"].struct.unnest()
    want = run(x, hazard=40.0, prior_scale=[1.0], prior_nu=2.0)
    assert got["p_change"].to_list()[1:] == pytest.approx(want["p_change"].to_list()[1:])


def test_predict_reads_the_hazard_column():
    """The hazard rides in the targets slot, which `predict` sees too: it
    must give the step's answer for that row, not the one for the spec's own
    hazard (docs/REVIEW-E54-E64.md B3, C1)."""
    x = normals(200, 23)
    h = np.where(np.arange(200) % 2 == 0, 20.0, 500.0)
    df = pl.DataFrame({"x0": x, "h": h})
    spec = po.spec.bocpd(
        "b", features=["x0"], hazard=50.0, hazard_col="h", prior_scale=[1.0], prior_nu=2.0
    )
    step = po.ModelBank([spec]).fit_predict(df)["b"].struct.unnest()
    bank = po.ModelBank([spec])
    bank.fit_predict(df.head(150))
    got = bank.predict(df.slice(150, 1))["b"].struct.unnest()
    for col in ("p_change", "run_mode", "run_mean", "pred_x0", "logscore"):
        assert got[col][0] == pytest.approx(step[col][150], rel=1e-12, abs=1e-12), col


def test_a_zero_weight_row_advances_nothing():
    x = normals(60, 19)
    w = np.ones(60)
    w[30] = 0.0
    kept = run(x, hazard=50.0, prior_scale=[1.0], prior_nu=2.0, weight="w", w=w)
    dropped = run(np.delete(x, 30), hazard=50.0, prior_scale=[1.0], prior_nu=2.0)
    assert kept["run_mode"].to_list()[31:] == dropped["run_mode"].to_list()[30:]
    assert kept["n_eff"][31] == 30.0


def test_the_first_row_of_every_group_is_silent():
    x = normals(40, 20)
    g = ["a"] * 20 + ["b"] * 20
    out = run(x, hazard=50.0, prior_scale=[1.0], prior_nu=2.0, group="g", g=g)
    assert out["p_change"][0] is None and out["p_change"][20] is None
    assert out["p_change"][1] is not None and out["p_change"][21] is not None
    # And the groups are independent: the second starts its run over.
    assert out["run_mode"][21] == 1
    assert out["n_eff"][20] == 0.0


# --- the robust emission -----------------------------------------------------


def test_the_robust_emission_refuses_to_learn_an_outlier():
    """One 20-sigma row *is* a changepoint to the plain model -- the `r = 0`
    hypothesis is the prior predictive, which explains it far better than a
    fitted run does -- and the run it starts carries the outlier in its
    mean, so the next predictive is nine sigma out. Under `"robust"` the row
    is atypical for every run, every tempered likelihood is about 1, and
    nothing moves: not the posterior, not the statistics."""
    x = normals(200, 21)
    x[100] = 20.0
    base = dict(hazard=250.0, prior_scale=[1.0], prior_nu=2.0)
    plain = run(x, **base)
    robust = run(x, emission="robust", **base)
    assert plain["p_change"][100] > 0.5 and plain["run_mode"][101] == 1
    assert robust["p_change"][100] < 0.02 and robust["run_mode"][101] == 101
    assert plain["pred_x0"][101] > 5.0, "and the plain run's mean is the outlier"
    assert abs(robust["pred_x0"][101]) < 0.1


@pytest.mark.parametrize("beta", [0.05, 0.1])
def test_a_real_shift_survives_a_small_robust_beta(beta):
    """What `robust_beta` costs. Tempering is not free -- a whole new regime
    is a run of individually forgiven rows -- so the knob trades the outlier
    off against the break. Measured: at 0.05 and 0.1 a four-sigma shift is
    still found within three rows and dated to the right one, and a
    20-sigma row is still a non-event; from about 0.3 up, nothing is ever
    detected again. 0.1 is the default."""
    x = np.concatenate([normals(150, 22), normals(150, 23, loc=4.0)])
    out = run(x, emission="robust", robust_beta=beta, hazard=250.0, prior_scale=[1.0], prior_nu=2.0)
    mode = out["run_mode"].to_list()
    found = next(t for t in range(150, 300) if mode[t] < 20)
    assert found <= 153
    assert found - mode[found] == 150


def test_too_much_tempering_detects_nothing():
    x = np.concatenate([normals(150, 22), normals(150, 23, loc=4.0)])
    out = run(x, emission="robust", robust_beta=0.5, hazard=250.0, prior_scale=[1.0], prior_nu=2.0)
    assert min(out["run_mode"].to_list()[151:]) > 20, "the note in the docstring is stale"


def test_the_gaussian_emission_sees_the_correlation_the_diagonal_one_cannot():
    """`"gaussian"` is a normal-inverse-Wishart over all the features, so a
    break in the *dependence* alone is visible to it; `"diag"` is a
    normal-inverse-gamma per feature and cannot see it by construction. The
    marginals here are unchanged -- only the correlation moves, from 0.05 to
    0.95 -- and note that neither alarm spikes: a correlation change is not
    a surprising row, it is a slow accumulation of unsurprising ones, so the
    run length is the whole of the signal."""
    rng = np.random.default_rng(24)
    f = rng.standard_normal((200, 1))
    before = rng.standard_normal((200, 2))
    r = 0.95
    after = np.sqrt(r) * f + np.sqrt(1 - r) * rng.standard_normal((200, 2))
    x = np.vstack([before, after])
    base = dict(hazard=250.0, prior_scale=[1.0], truncate=1e-8)
    full = run(x, emission="gaussian", prior_nu=4.0, **base)
    diag = run(x, emission="diag", prior_nu=2.0, **base)
    assert full["run_mode"][260] < 100, "the full one abandoned the old run"
    assert abs((260 - full["run_mode"][260]) - 200) < 15, "and dates it near the break"
    assert diag["run_mode"][260] == 260, "the diagonal one never noticed"
    assert max(diag["p_change"].to_list()[200:260]) < 0.05


# --- the plumbing ------------------------------------------------------------


def test_chunks_and_a_reload_do_not_move_it():
    x = np.concatenate([normals(120, 25), normals(120, 26, loc=2.0)])
    df = pl.DataFrame({"x0": x})
    spec = po.spec.bocpd("b", features=["x0"], hazard=60.0, prior_scale=[1.0])
    whole = po.ModelBank([spec]).fit_predict(df)
    bank = po.ModelBank([spec])
    parts = [bank.fit_predict(df[i : i + 7]) for i in range(0, 240, 7)]
    assert pl.concat(parts).equals(whole)
    # And across a save/load in the middle.
    warm = po.ModelBank([spec])
    first = warm.fit_predict(df[:100])
    reloaded = po.ModelBank.load_bytes(warm.save_bytes())
    second = reloaded.fit_predict(df[100:])
    assert pl.concat([first, second]).equals(whole)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(halflife=10.0), "do not apply to bocpd"),
        (dict(emission="student"), "unknown bocpd emission"),
        (dict(prior_scale=[1.0, 2.0]), "prior_scale is a scalar"),
        (dict(prior_mean=[0.0, 0.0]), "prior_mean must be 1 values"),
        (dict(hazard=0.5), "hazard"),
        (dict(prior_kappa=0.0), "prior_kappa"),
        (dict(truncate=1.0), "truncate must be in"),
        (dict(max_run=0), "max_run"),
        (dict(robust_beta=-1.0), "robust_beta"),
        # docs/REVIEW-E54-E64.md B2: priors that give a predictive with no
        # density, so every row would report nulls with nothing said.
        (dict(prior_scale=[0.0]), "scalar prior_scale"),
        (dict(prior_scale=[-1.0]), "scalar prior_scale"),
        (dict(prior_mean=[float("nan")]), "prior_mean"),
    ],
)
def test_bad_parameters_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        po.spec.bocpd("b", features=["x0"], **kwargs)
