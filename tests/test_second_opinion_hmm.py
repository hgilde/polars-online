"""``hmm``'s online learning (``learn=True``, the default) against hmmlearn
and scipy, under a finite half-life and at none.

The rule the model states is an online EM on the *filtered* posterior. Each
state ``l``'s moments take row ``s`` at weight ``w_s p_l(s)``, ``p(s)`` the
filtered posterior after the row, and the given pair (``means=``,
``covs=``) at weight 1, every weight decaying by ``lam`` a clock step, so
after ``t`` rows

    W_l = lam^T + sum_s lam^(T - T_s) w_s p_l(s)
    mu_l = (lam^T mu0_l + sum_s lam^(T - T_s) w_s p_l(s) x_s) / W_l
    C_l = (lam^T (Sigma0_l + (mu0_l - mu_l)(mu0_l - mu_l)') +
           sum_s lam^(T - T_s) w_s p_l(s) (x_s - mu_l)(x_s - mu_l)') / W_l

(``T`` the decayed clock, no decay before the first row) and the density
reads ``C_l + r_l I`` with the ridge ``r_l = precision_prior lam^T / W_l``,
the given pair's share of the state's weight. The transition counts are the
decayed sum of the filtered joint of consecutive states,

    A_kl = sum_s lam^(T - T_s) w_s xi_kl(s)
    xi_kl(s) = p_k(s-1) Pi_kl f_l(x_s) / sum_k'l' p_k'(s-1) Pi_k'l' f_l'(x_s)

decayed over a row's clock before the row reads ``Pi = (A + tau) /
rowsum``. A past row's share is fixed when it is read and never revised.

That is not Cappé's (2011) online EM for hidden Markov models, nor Mongillo
and Denève's (2008), which run EM on *smoothed* functionals: they carry an
auxiliary statistic ``rho_t(k)`` through the backward kernel ``r_t(j | k) =
phi_{t-1}(j) q(j, k) / sum_j' phi_{t-1}(j') q(j', k)`` (``rho_t(k) =
gamma_t s(k, x_t) + (1 - gamma_t) sum_j rho_{t-1}(j) r_t(j | k)``, read as
``sum_k rho_t(k) phi_t(k)``), so a past row's attribution to a state is
revised as each later row arrives. With ``gamma_t = 1 - lam`` the forgetting
is the same; the attribution is not. Neither paper's recursion is this
library's, so neither can hold it exactly.

So the first test rebuilds the stated rule from scratch, row by row, with
none of the model's arithmetic: each state's moments from numpy over the
rows' decayed responsibilities, each row's filtered posterior from
hmmlearn's ``GaussianHMM.predict_proba`` (started at that row's predicted
state) and each joint from scipy's Gaussian densities. Its responsibilities
are its own, never read back from the model. It holds the forgetting form,
the learned covariances and the learned transitions to round-off.

The second holds the rule's limit at ``half_life = inf`` against hmmlearn's
batch EM (Baum-Welch) on long streams, for the means, covariances and
transitions: statistical, since the two are different estimators of the
same parameters, with the tolerance derived below.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import pytest
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import ledger
import polars_online as po

TIER = "mixed"

SEEDS = range(50)

#: The ridge on every state's covariance (``precision_prior``), and the
#: default Dirichlet pseudo-count per cell (``transition_prior``).
RIDGE = 1e-2
TAU = 1.0

#: The filtered and predicted probabilities (absolute), ``loglik``
#: (relative to ``1 + |loglik|``), the learned means and covariances
#: (relative to ``1 + |x|``) and counts (relative to the largest count).
#: Measured 2026-11-04 over ``SEEDS`` and every case (aarch64, macOS): the
#: worst 2.2e-11, a filtered probability at a half-life of 5 rows, where a
#: state holds two or three rows' weight in two dimensions and its
#: covariance, whose ridge has faded with them, is nearly singular, so the
#: two sides' rounding in its quadratic form is amplified; 4.4e-13 at every
#: longer half-life. The bound leaves five times the worst.
EM_TOL = 1e-10

#: (covariance, states, half-life): a memory of a few rows to one of most of
#: the stream, at every covariance shape.
EM_CASES = [
    ("full", 2, 30.0),
    ("full", 3, 10.0),
    ("full", 3, 300.0),
    ("full", 2, 5.0),
    ("diag", 3, 30.0),
    ("tied", 3, 30.0),
    ("diag", 2, 300.0),
    ("tied", 2, 10.0),
]


def hmm_stream(seed: int, k: int, n: int = 300, d: int = 2) -> dict[str, Any]:
    """A sticky chain over ``k`` states with random means and covariances,
    its rows, their weights (0.2 to 3, about one row in seven at 0, the
    second row among them) and a given start: the true means moved by about
    half a unit, identity covariances."""
    rng = np.random.default_rng(seed)
    pi = np.full((k, k), 0.1 / (k - 1))
    np.fill_diagonal(pi, 0.9)
    pi *= rng.uniform(0.5, 1.5, (k, k))
    pi /= pi.sum(axis=1, keepdims=True)
    means = rng.normal(0.0, 1.5, (k, d))
    a = rng.normal(size=(k, d, d)) * 0.6
    covs = a @ a.transpose(0, 2, 1) + 0.3 * np.eye(d)
    state = np.zeros(n, dtype=int)
    state[0] = rng.integers(k)
    for t in range(1, n):
        state[t] = rng.choice(k, p=pi[state[t - 1]])
    noise = np.einsum("tij,tj->ti", np.linalg.cholesky(covs)[state], rng.normal(size=(n, d)))
    w = rng.uniform(0.2, 3.0, n)
    w[rng.random(n) < 1 / 7] = 0.0
    w[1] = 0.0
    return {
        "x": means[state] + noise,
        "w": w,
        "m0": means + rng.normal(0.0, 0.5, (k, d)),
        "c0": np.array([np.eye(d)] * k),
    }


def model_run(
    s: dict[str, Any], half_life: float, cov: str = "full"
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """The model's rows, and its learned means, co-moments and transition
    counts after the last row, read from the state the bank writes."""
    k, d = s["m0"].shape
    features = [f"x{i}" for i in range(d)]
    spec = po.spec.hmm(
        "h",
        features=features,
        n_components=k,
        precision_prior=RIDGE,
        covariance_type=cov,
        half_life=half_life,
        means=s["m0"].ravel().tolist(),
        covs=s["c0"].ravel().tolist(),
        weight="w",
    )
    frame = pl.DataFrame({f: s["x"][:, i] for i, f in enumerate(features)}).with_columns(w=s["w"])
    bank = po.ModelBank([spec])
    out = bank.fit_predict(frame)["h"].struct.unnest()
    model = json.loads(bank.write_json())["states"][0][0][1]["models"][0]["model"]["Hmm"]
    learned = {
        "means": np.array([st["m"] for st in model["states"]]),
        "covs": np.array([np.reshape(st["c"], (d, d)) for st in model["states"]]),
        "counts": np.reshape(model["a"], (k, k)),
    }
    return out, learned


def moments(
    u: np.ndarray, x: np.ndarray, m0: np.ndarray, c0: np.ndarray, w0: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """A state's mean and biased covariance over the given pair at weight
    ``w0`` and the rows at weights ``u``, numpy's weighted average and
    ``cov(aweights=, bias=True)``, with the given covariance's share."""
    weights = np.concatenate([[w0], u])
    points = np.vstack([m0, x])
    total = weights.sum()
    mean = np.average(points, axis=0, weights=weights)
    cov = np.atleast_2d(np.cov(points, rowvar=False, aweights=weights, bias=True))
    return mean, cov + w0 * c0 / total, total


def rule_from_scratch(s: dict[str, Any], half_life: float, cov: str) -> dict[str, np.ndarray]:
    """The stated rule, rebuilt from the rows: every row's ``filtered_state``,
    ``predicted_state`` and ``loglik`` as the model reads them before the
    row, and the learned means, covariances and counts after the last.

    Each state learns its full moments whatever the shape; the shape is how
    the density reads them: ``"full"`` each state's ``C_l + r_l I``,
    ``"diag"`` its diagonal, and ``"tied"`` one matrix, ``sum_l W_l (C_l +
    r_l I) / sum_l W_l``, the states pooled by their weights."""
    x, w, m0, c0 = s["x"], s["w"], s["m0"], s["c0"]
    n, d = x.shape
    k = m0.shape[0]
    # ln of the decay through each row: none before the first, lam a row after.
    log_lam = np.log(0.5) / half_life
    clock = np.concatenate([[0.0], np.arange(n) * log_lam])  # after t rows: (t - 1) steps

    def state_params(t: int) -> tuple[np.ndarray, np.ndarray]:
        ages = np.exp(clock[t] - clock[1 : t + 1])
        means, covs, totals = np.zeros((k, d)), np.zeros((k, d, d)), np.zeros(k)
        for st in range(k):
            mean, c, total = moments(
                ages * w[:t] * resp[:t, st], x[:t], m0[st], c0[st], np.exp(clock[t])
            )
            means[st], covs[st] = mean, c + RIDGE * np.exp(clock[t]) / total * np.eye(d)
            totals[st] = total
        if cov == "diag":
            covs = covs * np.eye(d)
        elif cov == "tied":
            covs = np.broadcast_to(np.einsum("s,sij->ij", totals / totals.sum(), covs), covs.shape)
        return means, covs

    resp = np.zeros((n, k))
    joints = np.zeros((n, k, k))
    p = np.full(k, 1.0 / k)
    filtered, predicted, loglik = np.zeros((n, k)), np.zeros((n, k)), np.zeros(n)
    for t in range(n):
        means, covs = state_params(t)
        # The counts decayed over the row's own clock before it reads them.
        counts = np.einsum(
            "s,skl->kl", np.exp(clock[t + 1] - clock[1 : t + 1]), w[:t, None, None] * joints[:t]
        )
        pi = (counts + TAU) / (counts + TAU).sum(axis=1, keepdims=True)
        hmm = GaussianHMM(n_components=k, covariance_type=cov, init_params="", params="")
        hmm.startprob_, hmm.transmat_, hmm.means_ = p @ pi, pi, means
        hmm.covars_ = {"full": covs, "diag": np.diagonal(covs, axis1=1, axis2=2), "tied": covs[0]}[
            cov
        ]
        filtered[t], predicted[t], loglik[t] = p, p @ pi, hmm.score(x[t : t + 1])
        if w[t] > 0:
            logf = np.array(
                [multivariate_normal(means[st], covs[st]).logpdf(x[t]) for st in range(k)]
            )
            joint = p[:, None] * pi * np.exp(logf - logf.max())[None, :]
            joints[t] = joint / joint.sum()
            resp[t] = hmm.predict_proba(x[t : t + 1])[0]
            p = resp[t]
    ages = np.exp(clock[n] - clock[1 : n + 1])
    learned = [
        moments(ages * w * resp[:, st], x, m0[st], c0[st], np.exp(clock[n])) for st in range(k)
    ]
    return {
        "filtered": filtered,
        "predicted": predicted,
        "loglik": loglik,
        "means": np.array([m for m, _, _ in learned]),
        "covs": np.array([c for _, c, _ in learned]),
        "counts": np.einsum("s,skl->kl", ages, w[:, None, None] * joints),
    }


ledger.guard(
    "hmm_online_em",
    lambda: [
        model_run(hmm_stream(seed, k), hl, cov) for seed in range(10) for cov, k, hl in EM_CASES
    ],
    oracles=["hmmlearn", "scipy", "numpy"],
    bits=52,
)


def em_errors(seed: int, cov: str, k: int, half_life: float) -> dict[str, float]:
    """Each quantity's worst error against the rule rebuilt, over ``EM_TOL``
    (at most 1 passes), and two numbers that show the paths ran: how far
    the learned means moved from the given ones, and the largest count."""
    s = hmm_stream(seed, k)
    out, learned = model_run(s, half_life, cov)
    want = rule_from_scratch(s, half_life, cov)

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float((np.abs(a - b) / (1.0 + np.abs(b))).max())

    filtered = out.select(f"filtered_state{j}" for j in range(k)).to_numpy()
    predicted = out.select(f"predicted_state{j}" for j in range(k)).to_numpy()
    return {
        "filtered": float(np.abs(filtered - want["filtered"]).max()) / EM_TOL,
        "predicted": float(np.abs(predicted - want["predicted"]).max()) / EM_TOL,
        "loglik": rel(out["loglik"].to_numpy(), want["loglik"]) / EM_TOL,
        "means": rel(learned["means"], want["means"]) / EM_TOL,
        "covs": rel(learned["covs"], want["covs"]) / EM_TOL,
        "counts": float(np.abs(learned["counts"] - want["counts"]).max() / want["counts"].max())
        / EM_TOL,
        "moved": float(np.abs(learned["means"] - s["m0"]).max()),
        "largest_count": float(want["counts"].max()),
    }


ERRORS = ("filtered", "predicted", "loglik", "means", "covs", "counts")


class TestOnlineEmIsItsRuleRebuilt:
    """Under a finite half-life, the learned states and transitions and the
    filter that reads them are the rule rebuilt with hmmlearn, scipy and
    numpy, on uneven and zero row weights."""

    @pytest.mark.parametrize(("cov", "k", "half_life"), EM_CASES)
    def test_one_seed(self, cov, k, half_life):
        e = em_errors(0, cov, k, half_life)
        print({key: f"{v:.3g}" for key, v in e.items()})
        assert e["moved"] > 0.3 and e["largest_count"] > 1.0
        assert max(e[key] for key in ERRORS) <= 1.0, e

    @pytest.mark.oracle_grid(reason="50 seeds x 8 cases, the rule rebuilt from scratch per row")
    def test_fifty_seeds(self):
        rows = [em_errors(seed, *case) for seed in SEEDS for case in EM_CASES]
        worst = np.array([max(r[key] for key in ERRORS) for r in rows])
        p95 = np.quantile(worst, 0.95)
        print(f"worst over the tolerance {worst.max():.3g}, 95th percentile {p95:.3g}")
        print(f"least move {min(r['moved'] for r in rows):.3g}")
        ledger.record("hmm_online_em", "error_over_tolerance", worst, floor=0.05)
        assert min(r["moved"] for r in rows) > 0.1
        assert worst.max() <= 1.0


# --- the limit at half_life = inf against Baum-Welch ------------------------------


def long_stream(seed: int, n: int, k: int = 3, d: int = 2) -> dict[str, Any]:
    """A sticky chain over ``k`` states whose means sit on a circle of radius
    3 (5.2 apart), with random covariances of standard deviation 0.63 to
    about 1: its rows, the true parameters, and a start ``0.7`` off the
    true means."""
    rng = np.random.default_rng(seed)
    pi = np.full((k, k), 0.05 / (k - 1))
    np.fill_diagonal(pi, 0.95)
    pi *= rng.uniform(0.7, 1.3, (k, k))
    pi /= pi.sum(axis=1, keepdims=True)
    angle = np.arange(k) * 2.0 * np.pi / k
    means = 3.0 * np.c_[np.cos(angle), np.sin(angle)]
    a = rng.normal(size=(k, d, d)) * SPREAD
    covs = a @ a.transpose(0, 2, 1) + 0.4 * np.eye(d)
    state = np.zeros(n, dtype=int)
    state[0] = rng.integers(k)
    for t in range(1, n):
        state[t] = rng.choice(k, p=pi[state[t - 1]])
    noise = np.einsum("tij,tj->ti", np.linalg.cholesky(covs)[state], rng.normal(size=(n, d)))
    return {
        "x": means[state] + noise,
        "w": np.ones(n),
        "pi": pi,
        "means": means,
        "covs": covs,
        "m0": means + rng.normal(0.0, 0.7, (k, d)),
        "c0": np.array([np.eye(d)] * k),
    }


#: The scale of each state's random covariance beyond its floor of 0.4.
SPREAD = 0.2

#: Rows of each long stream.
LONG_ROWS = 20_000

ledger.guard(
    "hmm_batch_limit",
    lambda: [model_run(long_stream(seed, LONG_ROWS), np.inf)[1] for seed in range(10)],
    oracles=["hmmlearn"],
)


def batch_limit(seed: int) -> dict[str, float]:
    """The learned parameters at ``half_life = inf`` against hmmlearn's
    Baum-Welch on the same rows, each over its tolerance (at most 1 passes).

    The two estimators read the same rows, so their difference is not a
    sampling error; it is two terms, each bounded here.

    1. *The rule's definition.* Batch EM attributes each row by its smoothed
       posterior, the rule by its filtered one. At hmmlearn's fit, one
       M-step on the filtered posteriors (a forward pass over scipy's
       densities) against the fit itself is that gap, computed, not assumed.
    2. *The online transient.* The rule reads row ``s`` under the parameters
       of ``s - 1`` rows, which miss the fixed point by about their standard
       error ``sigma / sqrt(s)``; a responsibility moves by at most its
       parameters' error over the states' separation, under 1 here, so the
       average over ``n`` rows is at most ``sum_s sigma / sqrt(s) / n = 2
       sigma / sqrt(n)``: twice each parameter's standard error at ``n``
       rows. The given pair, one row's weight among ``n``, and the
       pseudo-count ``tau`` per cell move the parameters by ``O(1/n)``,
       inside that.

    The second term needs the rule to settle at ``1/sqrt(n)``. At
    ``half_life = inf`` it is a stochastic approximation with step ``w /
    W``, about ``1/n``, which settles as ``n^-lambda`` when ``lambda``, the
    least share of the information the states' attribution leaves observed
    (one minus the batch EM's rate), is below 1/2. States that overlap
    lower it: at ``SPREAD = 0.5`` one stream in fifty (seed 13, a state of
    variance 1.9 reaching toward another) still held a covariance 0.09 off
    Baum-Welch's after 200,000 rows, at neither fixed point, and one batch
    step of the filtered rule from where the model stood landed on
    Baum-Welch's fit to 4 digits: rows attributed while the states were
    young are never revisited at an infinite half-life, and with steps of
    ``1/n`` their weight fades slowly. The states here are separated (5.2
    apart at standard deviations up to about 1), where the attribution is
    nearly certain and the rule settles fast.

    The standard errors are each state's at its smoothed weight ``n_l``:
    ``sqrt(Sigma_ii / n_l)`` for a mean, ``sqrt((Sigma_ii Sigma_jj +
    Sigma_ij^2) / n_l)`` for a covariance entry, and ``sqrt(Pi_kl (1 -
    Pi_kl) / n_k)`` for a transition."""
    s = long_stream(seed, LONG_ROWS)
    x = s["x"]
    k, d = s["means"].shape
    _, learned = model_run(s, np.inf)
    counts = learned["counts"]
    learned_pi = (counts + TAU) / (counts + TAU).sum(axis=1, keepdims=True)
    batch = GaussianHMM(
        n_components=k, covariance_type="full", init_params="", n_iter=500, tol=1e-10
    )
    batch.startprob_ = np.full(k, 1.0 / k)
    batch.transmat_, batch.means_, batch.covars_ = s["pi"], s["means"], s["covs"]
    batch.fit(x)
    assert batch.monitor_.converged
    smoothed = batch.predict_proba(x)
    logf = np.column_stack(
        [multivariate_normal(batch.means_[st], batch.covars_[st]).logpdf(x) for st in range(k)]
    )
    filtered = np.zeros_like(smoothed)
    xi = np.zeros((k, k))
    p = batch.startprob_
    for t in range(len(x)):
        prior = p @ batch.transmat_ if t else p
        if t:
            joint = p[:, None] * batch.transmat_ * np.exp(logf[t] - logf[t].max())[None, :]
            xi += joint / joint.sum()
        post = np.log(prior) + logf[t]
        filtered[t] = p = np.exp(post - logsumexp(post))

    def m_step(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.array([np.average(x, axis=0, weights=r[:, st]) for st in range(k)])
        covs = np.array([np.cov(x, rowvar=False, aweights=r[:, st], bias=True) for st in range(k)])
        return means, covs

    means_f, covs_f = m_step(filtered)
    pi_f = xi / xi.sum(axis=1, keepdims=True)
    n_l = smoothed.sum(axis=0)
    var = np.diagonal(batch.covars_, axis1=1, axis2=2)
    se_mean = np.sqrt(var / n_l[:, None])
    se_cov = np.sqrt((var[:, :, None] * var[:, None, :] + batch.covars_**2) / n_l[:, None, None])
    se_pi = np.sqrt(batch.transmat_ * (1.0 - batch.transmat_) / n_l[:, None])

    def over(got: np.ndarray, fit: np.ndarray, gap_at: np.ndarray, se: np.ndarray) -> float:
        return float((np.abs(got - fit) / (np.abs(gap_at - fit) + 2.0 * se)).max())

    return {
        "means": over(learned["means"], batch.means_, means_f, se_mean),
        "covs": over(learned["covs"], batch.covars_, covs_f, se_cov),
        "transitions": over(learned_pi, batch.transmat_, pi_f, se_pi),
        "moved": float(np.abs(learned["means"] - s["m0"]).max()),
    }


LIMITS = ("means", "covs", "transitions")


class TestTheLimitIsBaumWelch:
    """At ``half_life = inf``, on 20,000 rows, the learned means,
    covariances and transitions are hmmlearn's Baum-Welch fit within the
    filtered-versus-smoothed gap and twice their standard errors."""

    def test_one_seed(self):
        r = batch_limit(0)
        print({key: f"{v:.3g}" for key, v in r.items()})
        assert r["moved"] > 0.3
        assert max(r[key] for key in LIMITS) <= 1.0, r

    @pytest.mark.oracle_grid(reason="50 streams of 20,000 rows against Baum-Welch")
    def test_fifty_seeds(self):
        rows = [batch_limit(seed) for seed in SEEDS]
        for key in LIMITS:
            values = [r[key] for r in rows]
            print(
                f"{key}: worst {max(values):.3g}, 95th percentile {np.quantile(values, 0.95):.3g}"
            )
            ledger.record("hmm_batch_limit", f"{key}_over_tolerance", values)
        assert max(max(r[key] for key in LIMITS) for r in rows) <= 1.0
