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

import itertools
from datetime import timedelta
from math import expm1, inf, lgamma, log, pi

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
    out = run(x, hazard=50.0, prior_mean=[0.0], prior_nu=2.0, prior_scale=[1.0], prune_below=0.0)
    posts, scores = longhand(x, 50.0)
    # Row one is gated out: `P(r <= 1)` is 1 there whatever the data, so
    # every comparison starts at row two -- the row is still *learned*.
    assert out["loglik"][0] is None
    assert out["loglik"].to_list()[1:] == pytest.approx(list(scores[1:]), abs=1e-12)
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


def longhand_gaussian(x, hazard, k0=1.0):
    """Algorithm 1 again, with the ``gaussian`` emission: a normal-inverse-
    Wishart run whose predictive is a multivariate Student-t, written from the
    textbook -- raw sums, ``numpy.linalg.solve`` and ``slogdet`` -- where the
    model keeps centred sums and a Cholesky factor. The prior is ``mu0 = 0``,
    ``nu0 = d + 2``, ``Psi0 = I``, given whole to the spec it is held against
    (left out, the first rows set ``mu0`` and ``Psi0``, docs/PLAN.md task 195).
    Returns the row's log score and ``P(r <= 1)`` after it, per row."""
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    nu0, psi0, mu0 = d + 2.0, np.eye(d), np.zeros(d)
    h = 1.0 / hazard
    runs = [(0.0, np.zeros(d), np.zeros((d, d)))]
    joint = np.array([1.0])
    scores, p_change = [], []
    for row in x:
        pi = []
        for n, sx, sxx in runs:
            kn, nun = k0 + n, nu0 + n
            xbar = sx / n if n > 0 else np.zeros(d)
            scatter = sxx - n * np.outer(xbar, xbar)
            mun = (k0 * mu0 + n * xbar) / kn
            psin = psi0 + scatter + k0 * n / kn * np.outer(xbar - mu0, xbar - mu0)
            dof = nun - d + 1.0
            sigma = psin * (kn + 1.0) / (kn * dof)
            delta = row - mun
            q = delta @ np.linalg.solve(sigma, delta)
            _, logdet = np.linalg.slogdet(sigma)
            ln = (
                lgamma((dof + d) / 2.0)
                - lgamma(dof / 2.0)
                - 0.5 * d * np.log(dof * np.pi)
                - 0.5 * logdet
                - 0.5 * (dof + d) * np.log1p(q / dof)
            )
            pi.append(np.exp(ln))
        pi = np.array(pi)
        pre = joint / joint.sum()
        scores.append(np.log((pre * pi).sum()))
        new = np.empty(len(runs) + 1)
        new[1:] = joint * pi * (1.0 - h)
        new[0] = (joint * pi * h).sum()
        post = new / new.sum()
        p_change.append(post[0] + (post[1] if len(post) > 1 else 0.0))
        zero = (0.0, np.zeros(d), np.zeros((d, d)))
        runs = [zero] + [(n + 1, sx + row, sxx + np.outer(row, row)) for n, sx, sxx in runs]
        joint = new
    return np.array(scores), np.array(p_change)


def test_the_gaussian_emission_is_the_longhand_at_three_features():
    """The full normal-inverse-Wishart emission at ``d = 3`` against the
    textbook multivariate-t predictive in Algorithm 1, at every row, with the
    truncation off (REVIEW-2026-09-18 §6; docs/PLAN.md task 112: it had a
    smoke test and a shift invariance, no oracle). The columns are
    correlated and shift together at row 60, so the covariance is what the
    predictive turns on: the diagonal emission's numbers differ here, which
    is what makes the comparison a test of the full one."""
    rng = np.random.default_rng(7)
    cov = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
    x = np.concatenate(
        [
            rng.multivariate_normal(np.zeros(3), cov, 60),
            rng.multivariate_normal([2.0, -1.0, 1.5], cov, 60),
        ]
    )
    common = dict(
        hazard=50.0, prune_below=0.0, prior_mean=[0.0] * 3, prior_scale=[1.0], prior_nu=5.0
    )
    out = run(x, emission="gaussian", **common)
    scores, p_change = longhand_gaussian(x, 50.0)
    assert out["loglik"].to_list()[1:] == pytest.approx(list(scores[1:]), abs=1e-10)
    assert out["p_change"].to_list()[1:] == pytest.approx(list(p_change[1:]), abs=1e-10)
    diag = run(x, emission="diag", **common)["loglik"].to_numpy()[1:]
    assert np.max(np.abs(diag - scores[1:])) > 1e-2, "the covariance matters here"


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
    out = run(
        x, hazard=40.0, prior_mean=[0.0] * 2, prior_nu=2.0, prior_scale=[1.0], prune_below=0.0
    )
    _, scores = longhand(x, 40.0)
    assert out["loglik"].to_list()[1:] == pytest.approx(list(scores[1:]), abs=1e-12)


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
    out = run(x, hazard=250.0, prior_nu=2.0, prior_scale=[2e-4], prune_below=1e-6, max_run=600)
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
    assert sum(quick["loglik"].to_list()[after]) > sum(stuck["loglik"].to_list()[after]) + 5


# --- the knobs ---------------------------------------------------------------


def test_truncation_changes_little_and_max_run_caps_the_state():
    x = np.concatenate([normals(200, 14), normals(200, 15, loc=3.0)])
    base = dict(hazard=100.0, prior_mean=[0.0], prior_scale=[1.0], prior_nu=2.0)
    exact = run(x, prune_below=0.0, **base)
    cut = run(x, prune_below=1e-4, **base)
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
    capped = run(x, prune_below=0.0, max_run=25, **base)
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


def test_a_spec_that_leaves_targets_out_reads_the_hazard_column_it_names():
    """``hazard_col`` rides in the targets slot. A spec that named it and
    left ``targets`` out had them filled from ``features[0]``, so the feature
    was read as the hazard: refused at 1 or below, and above it silently the
    expected run length (review 2026-09-12, C20)."""
    x = normals(200, 17) + 10.0
    h = np.where(np.arange(200) < 100, 1e9, 5.0)
    df = pl.DataFrame({"x0": x, "h": h})
    built = po.spec.bocpd(
        "b", features=["x0"], hazard_col="h", prior_mean=[10.0], prior_scale=[1.0], prior_nu=2.0
    )
    hand = {k: v for k, v in built.items() if k != "targets"}
    a = po.ModelBank([built]).fit_predict(df)["b"].struct.unnest()
    b = po.ModelBank([hand]).fit_predict(df)["b"].struct.unnest()
    assert a["p_change"].to_list() == b["p_change"].to_list()


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
    for col in ("p_change", "run_mode", "run_mean", "pred_x0", "loglik"):
        assert got[col][0] == pytest.approx(step[col][150], rel=1e-12, abs=1e-12), col


def test_a_zero_weight_row_advances_nothing():
    x = normals(60, 19)
    w = np.ones(60)
    w[30] = 0.0
    kept = run(x, hazard=50.0, prior_scale=[1.0], prior_nu=2.0, weight="w", w=w)
    dropped = run(np.delete(x, 30), hazard=50.0, prior_scale=[1.0], prior_nu=2.0)
    assert kept["run_mode"].to_list()[31:] == dropped["run_mode"].to_list()[30:]
    assert kept["weight_sum"][31] == 30.0


def test_the_first_row_of_every_group_is_silent():
    x = normals(40, 20)
    g = ["a"] * 20 + ["b"] * 20
    out = run(x, hazard=50.0, prior_mean=[0.0], prior_scale=[1.0], prior_nu=2.0, group="g", g=g)
    assert out["p_change"][0] is None and out["p_change"][20] is None
    assert out["p_change"][1] is not None and out["p_change"][21] is not None
    # And the groups are independent: the second starts its run over.
    assert out["run_mode"][21] == 1
    assert out["weight_sum"][20] == 0.0


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
    out = run(
        x,
        emission="robust",
        robust_beta=beta,
        hazard=250.0,
        prior_mean=[0.0],
        prior_scale=[1.0],
        prior_nu=2.0,
    )
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
    base = dict(hazard=250.0, prior_mean=[0.0, 0.0], prior_scale=[1.0], prune_below=1e-8)
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
        (dict(half_life=10.0), "do not apply to bocpd"),
        (dict(emission="student"), "unknown bocpd emission"),
        (dict(prior_scale=[1.0, 2.0]), "prior_scale is a scalar"),
        (dict(prior_mean=[0.0, 0.0]), "prior_mean must be 1 values"),
        (dict(hazard=0.5), "hazard"),
        (dict(prior_kappa=0.0), "prior_kappa"),
        (dict(prune_below=1.0), "prune_below must be in"),
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


# --- phase-4 coverage: three features (the diagonal tests above reach d = 2) --


def test_three_features_are_a_product_of_student_ts():
    """The diagonal emission on `d = 3`, against the same longhand as the
    `d = 2` test (review 2026-09-18, phase 4)."""
    rng = np.random.default_rng(12)
    x = np.column_stack([rng.normal(0, 1, 80), rng.normal(0, 2, 80), rng.normal(0, 0.5, 80)])
    out = run(
        x, hazard=40.0, prior_mean=[0.0] * 3, prior_nu=2.0, prior_scale=[1.0], prune_below=0.0
    )
    _, scores = longhand(x, 40.0)
    assert out["loglik"].to_list()[1:] == pytest.approx(list(scores[1:]), abs=1e-12)


# --- the hazard on the clock (task 179) --------------------------------------

#: The prior the oracle below shares with the specs it is held against:
#: ``prior_mean`` 0, ``prior_kappa`` 1, and ``prior_nu = 2a = 2``,
#: ``prior_scale = 2b = 1`` in the gamma parametrisation.
_MU0, _KAPPA0, _ALPHA0, _BETA0 = 0.0, 1.0, 1.0, 0.5

#: 2024-01-02 09:30:00 UTC, in epoch seconds: where the clocks below start.
_START = 1_704_187_800.0


def _log_marginal(xs) -> float:
    """``ln p(x_1..x_n)`` of one segment's rows under that normal-inverse-gamma
    prior, in closed form (Murphy 2007, "Conjugate Bayesian analysis of the
    Gaussian distribution", eq. 95). The model keeps runs and evaluates
    Student-t predictives one row at a time; this does neither."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    if n == 0:
        return 0.0
    xbar = xs.mean()
    s = float(((xs - xbar) ** 2).sum())
    kn, an = _KAPPA0 + n, _ALPHA0 + n / 2.0
    bn = _BETA0 + s / 2.0 + _KAPPA0 * n * (xbar - _MU0) ** 2 / (2.0 * kn)
    return (
        lgamma(an)
        - lgamma(_ALPHA0)
        + _ALPHA0 * log(_BETA0)
        - an * log(bn)
        + 0.5 * log(_KAPPA0 / kn)
        - 0.5 * n * log(2.0 * pi)
    )


def clock_oracle(x, secs, tau, w=None):
    """What a hazard on the clock states, written from its definition and
    nothing of the model's: breaks arrive in time at rate ``1 / tau``, so one
    falls between two learned rows with probability ``1 - exp(-gap / tau)``,
    independently of every other gap, and the rows between breaks are iid
    normal under the prior above. Every segmentation is enumerated and
    weighed by its prior and the closed-form marginal of its segments.

    Per row ``k``, against the learned rows before it, returns

    - ``A[k]``: the chance row ``k`` began a segment, given its own value too,
      which is the bank's ``p_change``;
    - ``B[k]``, ``M[k]``: the mean and the first mode of how many learned rows
      of row ``k``'s segment came before it, given only the rows before:
      ``run_mean`` and ``run_mode``.

    A row of weight 0 is read as a learned row would be, which is what the
    bank reports on it, and then left out of what follows: only its time
    counts, so the gap it splits is one gap. ``expm1`` evaluates the
    definition where ``1 - exp`` would cancel."""
    x = np.asarray(x, dtype=float)
    secs = np.asarray(secs, dtype=float)
    learned = np.ones(len(x), bool) if w is None else np.asarray(w) > 0
    a_out, b_out, m_out = [], [], []
    for k in range(len(x)):
        rows = [i for i in range(k) if learned[i]] + [k]
        m = len(rows)
        if m == 1:
            a_out.append(1.0)
            b_out.append(0.0)
            m_out.append(0)
            continue
        gaps = [secs[rows[j + 1]] - secs[rows[j]] for j in range(m - 1)]
        brk = [log(-expm1(-g / tau)) if g > 0 else -inf for g in gaps]
        stay = [-g / tau for g in gaps]
        xr = x[rows]
        memo: dict[tuple[int, int], float] = {}

        def seg(a: int, b: int, xr=xr, memo=memo) -> float:
            """Rows at positions ``a..b`` of ``rows``, inclusive; none is 0."""
            if a > b:
                return 0.0
            if (a, b) not in memo:
                memo[(a, b)] = _log_marginal(xr[a : b + 1])
            return memo[(a, b)]

        num = den = -inf
        runs: dict[int, float] = {}
        for bits in itertools.product((0, 1), repeat=m - 1):
            prior = sum(brk[j] if bit else stay[j] for j, bit in enumerate(bits))
            if prior == -inf:
                continue
            starts = [0] + [j + 1 for j, bit in enumerate(bits) if bit]
            ends = [s - 1 for s in starts[1:]] + [m - 1]
            head = sum(seg(s, e) for s, e in zip(starts[:-1], ends[:-1], strict=True))
            with_k = prior + head + seg(starts[-1], m - 1)
            den = np.logaddexp(den, with_k)
            if bits[-1]:
                num = np.logaddexp(num, with_k)
            before_k = prior + head + seg(starts[-1], m - 2)
            run = (m - 1) - starts[-1]
            runs[run] = np.logaddexp(runs.get(run, -inf), before_k)
        a_out.append(0.0 if num == -inf else float(np.exp(num - den)))
        total = np.logaddexp.reduce(list(runs.values()))
        b_out.append(sum(r * float(np.exp(v - total)) for r, v in runs.items()))
        top = max(runs.values())
        m_out.append(min(r for r, v in runs.items() if v == top))
    return np.array(a_out), np.array(b_out), np.array(m_out)


def _clock_frame(x, secs, w=None, **cols) -> pl.DataFrame:
    """``x0`` on a microsecond ``Datetime`` clock ``t`` at ``secs``."""
    stamps = (np.asarray(secs, dtype=float) * 1e6).round().astype(np.int64)
    df = pl.DataFrame(
        {"t": pl.Series(stamps).cast(pl.Datetime("us")), "x0": np.asarray(x, dtype=float)}
    )
    if w is not None:
        df = df.with_columns(pl.Series("w", np.asarray(w, dtype=float)))
    return df.with_columns(**{k: pl.Series(v) for k, v in cols.items()})


def run_clock(x, secs, hazard, w=None, **kw):
    """The bank on that clock, with the oracle's prior, nothing pruned and
    every row reported; returns the unnested output frame."""
    df = _clock_frame(x, secs, w)
    if w is not None:
        kw["weight"] = "w"
    kw = dict(dict(gap_cap="1d", prune_below=0.0, max_run=10_000, min_weight=0.0), **kw)
    spec = po.spec.bocpd(
        "b",
        features=["x0"],
        hazard=hazard,
        clock="t",
        prior_mean=[0.0],
        prior_nu=2.0,
        prior_scale=[1.0],
        **kw,
    )
    return po.ModelBank([spec]).fit_predict(df)["b"].struct.unnest()


def _gap_case():
    """Seven rows a minute apart, four quiet hours, seven more a minute
    apart: the level moves by four sigma across the gap. ``tau`` = 1 h."""
    rng = np.random.default_rng(179)
    secs = np.concatenate([np.arange(7) * 60.0, 4 * 3600.0 + 360.0 + np.arange(7) * 60.0])
    x = np.concatenate([rng.normal(0, 1, 7), rng.normal(4, 1, 7)])
    return x, _START + secs, None, "1h", 3600.0


def _duplicate_case():
    """Rows a minute apart, but rows 5 and 6 share a stamp, and row 7 is a
    20-sigma row a minute after them. ``tau`` = 10 min."""
    rng = np.random.default_rng(1792)
    secs = np.arange(12) * 60.0
    secs[5] = secs[4]
    secs[6:] = secs[4] + 60.0 * np.arange(1, 7)
    x = rng.normal(0, 1, 12)
    x[6] = 20.0
    return x, _START + secs, None, "10m", 600.0


def _weightless_case():
    """The gap case with a row of weight 0 in the gap, a minute before the
    first row after it."""
    x, secs, _, text, tau = _gap_case()
    x = np.concatenate([x[:7], [0.0], x[7:]])
    secs = np.concatenate([secs[:7], [secs[7] - 60.0], secs[7:]])
    w = np.concatenate([np.ones(7), [0.0], np.ones(7)])
    return x, secs, w, text, tau


def _first_step_case():
    """The bank's first step is 0, and the chance of a break between the
    first two rows is the second row's: three hours after the first, then
    rows ten minutes apart. ``tau`` = 1 h."""
    rng = np.random.default_rng(1793)
    secs = np.concatenate([[0.0], 3 * 3600.0 + np.arange(11) * 600.0])
    x = np.concatenate([[3.0], rng.normal(0, 1, 11)])
    return x, _START + secs, None, "1h", 3600.0


def _irregular_case():
    """Fourteen rows at irregular whole seconds, one pair sharing a stamp,
    and a level that moves halfway. ``tau`` = 30 min."""
    rng = np.random.default_rng(1794)
    gaps = rng.exponential(600.0, 14).round()
    gaps[0] = 0.0
    gaps[9] = 0.0
    x = np.concatenate([rng.normal(0, 1, 7), rng.normal(2.5, 1, 7)])
    return x, _START + np.cumsum(gaps), None, "30m", 1800.0


CLOCK_CASES = {
    "a break across a four-hour gap": _gap_case,
    "a duplicate stamp, then a 20-sigma row": _duplicate_case,
    "a row of weight 0 inside the gap": _weightless_case,
    "the first step": _first_step_case,
    "irregular steps": _irregular_case,
}


@pytest.mark.parametrize("case", list(CLOCK_CASES))
def test_a_hazard_on_the_clock_is_the_model_it_states(case):
    """The duration form against the enumeration above, at every row. The
    step into row ``k`` of ``d`` clock units carries the chance ``1 -
    exp(-d / tau)`` of a break, applied before row ``k`` is read; so
    ``p_change`` is the chance that row ``k`` began a run, and ``run_mean``
    and ``run_mode`` count the learned rows of its run before it.

    Agreement measured over the five cases: ``p_change`` to 1.1e-14,
    ``run_mean`` to 8e-14, ``run_mode`` exactly. Taking the chance from the
    row's own step on the boundary after the row, as this was first briefed,
    missed by 0.93 and 6.5 rows on the gap case and dated the break a row
    late (task 179)."""
    x, secs, w, text, tau = CLOCK_CASES[case]()
    out = run_clock(x, secs, text, w)
    a, b, m = clock_oracle(x, secs, tau, w)
    assert out["p_change"].to_numpy() == pytest.approx(a, abs=1e-13, rel=0)
    assert out["run_mean"].to_numpy() == pytest.approx(b, abs=1e-12, rel=0)
    assert out["run_mode"].to_list() == m.tolist()
    if case == "a break across a four-hour gap":
        # The first row after the gap began a run, almost surely.
        assert a[7] > 0.99 and a[8] < 0.01
    if case == "a duplicate stamp, then a 20-sigma row":
        # No time passed before row 6: it cannot begin a run, exactly.
        assert out["p_change"][5] == 0.0
        # And the wild row a minute later can: its chance, 0.095 before it is
        # read, is lifted more than five times by what it holds.
        assert a[6] > 5 * -expm1(-60.0 / tau)
    if case == "a row of weight 0 inside the gap":
        # What a weightless row leaves is its time: the learned rows read as
        # the stream without it, the gap read whole.
        x0, secs0, _, _, _ = _gap_case()
        plain = run_clock(x0, secs0, text)
        kept = out.filter(pl.Series(w > 0))
        for col in ("p_change", "run_mean", "pred_x0", "loglik"):
            assert kept[col].to_numpy() == pytest.approx(plain[col].to_numpy(), abs=1e-12), col
        assert kept["run_mode"].to_list() == plain["run_mode"].to_list()
    if case == "the first step":
        assert a[1] > 0.9, "three hours before row two"


def test_a_number_hazard_on_a_temporal_clock_is_still_per_row():
    """A plain number keeps its meaning on any spec: the expected rows
    between changepoints, whatever the clock says. The same stream with and
    without a temporal clock reads the same."""
    x, secs, _, _, _ = _irregular_case()
    with_clock = run_clock(x, secs, 50.0)
    spec = po.spec.bocpd(
        "b",
        features=["x0"],
        hazard=50.0,
        prior_mean=[0.0],
        prior_nu=2.0,
        prior_scale=[1.0],
        prune_below=0.0,
        max_run=10_000,
        min_weight=0.0,
    )
    without = po.ModelBank([spec]).fit_predict(_clock_frame(x, secs))["b"].struct.unnest()
    assert with_clock.equals(without)


def test_each_way_of_writing_a_duration_is_the_same_hazard():
    x, secs, _, _, _ = _irregular_case()
    want = run_clock(x, secs, "30m")
    for form in (timedelta(minutes=30), pl.duration(minutes=30), "1800s"):
        assert run_clock(x, secs, form).equals(want), form


def test_a_gap_past_gap_cap_counts_as_the_cap():
    """The step a row's chance is read from is the decayed clock's: a gap
    past ``gap_cap`` is the cap, as it is to every decay. A nine-hour gap
    under a two-hour cap reads as a two-hour gap."""
    x, secs, _, text, _ = _gap_case()
    capped = run_clock(x, secs, text, gap_cap="2h")
    short = secs.copy()
    short[7:] -= secs[7] - secs[6] - 7200.0
    at_the_cap = run_clock(x, short, text, gap_cap="2h")
    assert capped.equals(at_the_cap)
    assert not capped.equals(run_clock(x, secs, text, gap_cap="12h")), "the cap acted"


def test_a_hazard_on_the_clock_is_chunk_invariant_and_resumes():
    """1, 7 and 37 rows a chunk, and a save and a load at three rows, give the
    stream fed whole, over irregular steps with stamps shared, gaps past the
    cap, rows of weight 0 and two groups."""
    rng = np.random.default_rng(1795)
    n = 300
    gaps = rng.exponential(120.0, n).round()
    gaps[rng.random(n) < 0.08] = 0.0
    gaps[rng.random(n) < 0.03] = 6 * 3600.0
    secs = _START + np.cumsum(gaps)
    x = np.where(np.arange(n) < 150, rng.normal(0, 1, n), rng.normal(3, 1, n))
    w = np.where(rng.random(n) < 0.1, 0.0, 1.0)
    g = np.where((np.arange(n) // 50) % 2 == 0, "a", "b")
    df = _clock_frame(x, secs, w, g=g)
    spec = po.spec.bocpd(
        "b",
        features=["x0"],
        hazard="20m",
        clock="t",
        gap_cap="2h",
        weight="w",
        group="g",
        prior_nu=2.0,
        prior_scale=[1.0],
    )
    whole = po.ModelBank([spec]).fit_predict(df)
    for size in (1, 7, 37):
        bank = po.ModelBank([spec])
        parts = [bank.fit_predict(df[i : i + size]) for i in range(0, n, size)]
        assert pl.concat(parts).equals(whole), size
    for at in (1, 100, 211):
        warm = po.ModelBank([spec])
        first = warm.fit_predict(df[:at])
        second = po.ModelBank.load_bytes(warm.save_bytes()).fit_predict(df[at:])
        assert pl.concat([first, second]).equals(whole), at
    # The stream did what it is here for: rows that could not begin a run
    # (a stamp shared with the row before), and a break found.
    p = whole["b"].struct.field("p_change")
    assert (p == 0.0).sum() >= 5
    assert p.max() > 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(hazard="1h"), "hazard is a duration, which needs a clock column"),
        (
            dict(hazard="1h", clock="t", gap_cap=300.0),
            "hazard is a duration but gap_cap is a plain number",
        ),
        (dict(hazard="0s", clock="t", gap_cap="1h"), "hazard .* must be > 0"),
        (dict(hazard="-5m", clock="t", gap_cap="1h"), "hazard .* must be > 0"),
        (
            dict(hazard=timedelta(hours=1), hazard_col="h", clock="t", gap_cap="1h"),
            "hazard_col .* per-row hazard.* hazard is a duration",
        ),
    ],
)
def test_a_duration_hazard_is_refused_where_it_cannot_run(kwargs, message):
    with pytest.raises(ValueError, match=message):
        po.spec.bocpd("b", features=["x0"], **kwargs)


# --- the prior from the first rows (docs/PLAN.md task 195, U4 and U5) ---------


def stepping(level, sd, n=600, seed=11):
    """Two features whose means step up 4 spreads at row 200 and back at row
    400, at `level` with a spread of `sd`: a shift `robust` dates, where a
    larger one it forgives row by row."""
    rng = np.random.default_rng(seed)
    shift = np.where((np.arange(n) >= 200) & (np.arange(n) < 400), 4.0, 0.0)
    return np.column_stack(
        [level + sd * (shift + rng.normal(size=n)), level + sd * (0.5 * shift + rng.normal(size=n))]
    )


def breaks(out):
    """The rows a break is dated to: where the pre-row run mode falls under
    20 from 20 or over, at `t - mode`, the row the run began on."""
    mode = out["run_mode"].to_numpy()
    found, last = [], 0.0
    for t, m in enumerate(mode):
        if np.isfinite(m):
            if m < 20 and last >= 20:
                found.append(t - int(m))
            last = m
    return found


@pytest.mark.parametrize("emission", ["diag", "gaussian", "robust"])
def test_the_prior_from_the_first_rows_finds_the_same_breaks_at_any_level(emission):
    """With no `prior_mean` and no `prior_scale`, the first `warm_rows` rows
    set both: their sample mean and covariance. So a stream at a level of
    1e4 with a spread of 1e-2 is dated as the same stream at 0 with a spread
    of 1 is, under ``"diag"`` and ``"gaussian"``, whose ``p_change`` agree to
    1e-9. The prior was a mean of 0 and the identity in the data's units,
    under which a run beginning on any row of the first stream had no
    density, and no break was found (review round 4, CE5). ``"robust"`` is
    held at level 0 alone: its tempered message ``pi ** w`` carries the
    data's units as a factor ``c ** -w`` that differs from run to run, so its
    posterior moves with the scale whatever the prior (reported beside task
    195, not changed by it)."""
    unit = run(stepping(0.0, 1.0), emission=emission, hazard=250.0)
    assert all(any(abs(b - e) <= 1 for b in breaks(unit)) for e in (200, 400)), breaks(unit)
    if emission != "robust":
        scaled = run(stepping(1e4, 1e-2), emission=emission, hazard=250.0)
        assert breaks(scaled) == breaks(unit)
        gap = (unit["p_change"] - scaled["p_change"]).abs().max()
        assert gap < 1e-9, gap


def test_the_first_rows_report_nothing_and_are_replayed():
    """The `warm_rows` rows (default `d + 2`) are buffered and report null;
    then the runs see them, so the row after them is read as one more row of
    a model that saw them all. A prior given whole runs from the first row."""
    x = stepping(0.0, 1.0, n=50)
    out = run(x)
    assert out["p_change"][:4].null_count() == 4
    assert out["p_change"][4:].null_count() == 0
    out = run(x, warm_rows=10)
    assert out["p_change"][:10].null_count() == 10
    assert out["p_change"][10:].null_count() == 0
    # Given whole, the prior reads from the first row, which `min_weight`'s
    # default of 1 withholds alone.
    given = run(x, prior_mean=[0.0, 0.0], prior_scale=[1.0])
    assert given["p_change"][0] is None
    assert given["p_change"][1:].null_count() == 0


def test_the_warm_up_is_chunk_invariant_and_resumes():
    """The buffer is state: a stream cut inside the warm-up, or saved and
    loaded there, goes on as the stream that never stopped."""
    x = stepping(3.0, 0.5, n=120)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
    spec = po.spec.bocpd("b", features=["x0", "x1"], warm_rows=7)
    whole = po.ModelBank([spec]).fit_predict(df)
    bank = po.ModelBank([spec])
    parts = pl.concat([bank.fit_predict(df.slice(i, 3)) for i in range(0, df.height, 3)])
    assert parts.equals(whole)
    for at in (2, 6, 7, 8):
        warm = po.ModelBank([spec])
        first = warm.fit_predict(df[:at])
        second = po.ModelBank.load_bytes(warm.save_bytes()).fit_predict(df[at:])
        assert pl.concat([first, second]).equals(whole), at


def test_a_constant_feature_does_not_break_the_prior():
    """A feature that does not move over the first rows gives the prior no
    variance; it takes a floor, and the model still finds the break in the
    other feature, at level 0 and away from it."""
    rng = np.random.default_rng(5)
    n = 400
    for level in (0.0, 1e4):
        x = np.column_stack(
            [np.full(n, level), np.where(np.arange(n) < 200, 0.0, 6.0) + rng.normal(size=n)]
        )
        out = run(x, hazard=250.0)
        assert out["p_change"][10:].null_count() == 0, level
        assert any(abs(b - 200) <= 1 for b in breaks(out)), level
