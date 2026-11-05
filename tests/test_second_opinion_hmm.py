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
the learned covariances and the learned transitions to round-off: each
error against a bound derived from the rule's own quantities at that
row, not measured on one platform (:func:`rule_error_bound`). Two correct
evaluations of the rule in doubles differ by more than a few units of
rounding where the rule itself amplifies them: a state whose weight has
decayed far below a row's weight hands that row most of its moments, so
the *relative* error of a posterior as small as ``1e-5`` -- the two
sides' rounding in that row's density, read through ``Sigma^-1 delta`` --
becomes the moments' error, and the next row on that state reads it
again through ``Sigma^-1 delta``. The sides' last bits differ in
different places on each platform (libm, BLAS, fused multiply-add,
numpy's SIMD kernels), and the chain multiplies whichever bits differ by
``1e5`` or more, so the same stream measures ``1e-11`` on one platform
and ``4e-10`` on another; the bound covers every pattern.

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

#: The unit roundoff of a double, ``u = 2^-53``: every basic operation
#: returns its exact result times ``1 + delta``, ``|delta| <= u`` (Higham,
#: *Accuracy and Stability of Numerical Algorithms*, 2nd ed., (2.4)).
U = 2.0**-53

#: The units of ``u`` a platform's ``ln``, ``exp`` or ``exp2`` may be off,
#: relative: four ulps. Apple's and glibc's are within one and Windows'
#: has not been measured to the ulp; numpy's own AVX512 kernels for
#: ``exp`` and ``log`` on doubles, which the rebuild's ages and densities
#: go through on such a CPU, state no bound (its float32 ones state 2.52
#: and 3.83 ulps), so the bound leaves four.
LIBM = 8.0

#: scipy's ``multivariate_normal.logpdf`` reads the covariance through its
#: eigendecomposition. LAPACK states that backward error as ``p(d) u
#: ||Sigma||`` with ``p`` a modest polynomial it does not fix; ``10 d`` is
#: taken here, the one constant in the bound that is not an operation count.
C_EIG = 10.0


def gamma(m: int) -> float:
    """``gamma_m = m u / (1 - m u)``: ``m`` roundings in a row move a result
    by at most ``gamma_m`` of itself (Higham, Lemma 3.1), and a sum of ``m
    + 1`` terms, in any order, by at most ``gamma_m`` of the sum of their
    magnitudes (Higham, (3.5) and section 3.1)."""
    return m * U / (1.0 - m * U)


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


def rule_from_scratch(s: dict[str, Any], half_life: float, cov: str) -> dict[str, Any]:
    """The stated rule, rebuilt from the rows: every row's ``filtered_state``,
    ``predicted_state`` and ``loglik`` as the model reads them before the
    row, and the learned means, covariances and counts after the last.

    Each state learns its full moments whatever the shape; the shape is how
    the density reads them: ``"full"`` each state's ``C_l + r_l I``,
    ``"diag"`` its diagonal, and ``"tied"`` one matrix, ``sum_l W_l (C_l +
    r_l I) / sum_l W_l``, the states pooled by their weights.

    Under ``"rows"``, what each row read and learned, which the error bound
    (:func:`rule_error_bound`) walks: the states' means, moment matrices
    ``C_l``, weights ``W_l``, ridges ``r_l`` and the covariances as the
    density read them, before every row and after the last; the counts and
    ``Pi`` as each row read them; each row's responsibilities and joint."""
    x, w, m0, c0 = s["x"], s["w"], s["m0"], s["c0"]
    n, d = x.shape
    k = m0.shape[0]
    # ln of the decay through each row: none before the first, lam a row after.
    log_lam = np.log(0.5) / half_life
    clock = np.concatenate([[0.0], np.arange(n) * log_lam])  # after t rows: (t - 1) steps
    rows: dict[str, np.ndarray] = {
        "means": np.zeros((n + 1, k, d)),
        "moments": np.zeros((n + 1, k, d, d)),
        "totals": np.zeros((n + 1, k)),
        "ridges": np.zeros((n + 1, k)),
        "sigma": np.zeros((n + 1, k, d, d)),
        "ordered": np.zeros((n + 1, k, d)),
        "ordered_w": np.zeros((n + 1, k)),
        "abs1": np.zeros((n + 1, k, d)),
        "dev1": np.zeros((n + 1, k, d)),
        "dev2": np.zeros((n + 1, k, d, d)),
        "counts": np.zeros((n, k, k)),
        "pi": np.zeros((n, k, k)),
        "resp": np.zeros((n, k)),
        "joints": np.zeros((n, k, k)),
    }

    def state_params(t: int) -> tuple[np.ndarray, np.ndarray]:
        ages = np.exp(clock[t] - clock[1 : t + 1])
        means, covs, totals = np.zeros((k, d)), np.zeros((k, d, d)), np.zeros(k)
        # Each point's number of partial sums in numpy's row-ordered sum,
        # the pair first: t + 1 for the pair, 1 for the newest row.
        remaining = np.arange(t + 1, 0, -1, dtype=float)
        points = np.vstack([np.zeros((1, d)), x[:t]])
        for st in range(k):
            weights = ages * w[:t] * resp[:t, st]
            mean, c, total = moments(weights, x[:t], m0[st], c0[st], np.exp(clock[t]))
            ridge = RIDGE * np.exp(clock[t]) / total
            means[st], covs[st] = mean, c + ridge * np.eye(d)
            totals[st] = total
            rows["moments"][t, st], rows["ridges"][t, st] = c, ridge
            points[0] = m0[st]
            share = np.concatenate([[np.exp(clock[t])], weights]) / total
            dev = np.abs(points - mean)
            rows["ordered"][t, st] = remaining @ (share[:, None] * np.abs(points))
            rows["ordered_w"][t, st] = remaining @ share
            rows["abs1"][t, st] = share @ np.abs(points)
            rows["dev1"][t, st] = share @ dev
            rows["dev2"][t, st] = np.einsum("s,si,sj->ij", share, dev, dev)
        rows["means"][t], rows["totals"][t] = means, totals
        if cov == "diag":
            covs = covs * np.eye(d)
        elif cov == "tied":
            covs = np.broadcast_to(np.einsum("s,sij->ij", totals / totals.sum(), covs), covs.shape)
        rows["sigma"][t] = covs
        return means, covs

    resp = rows["resp"]
    joints = rows["joints"]
    p = np.full(k, 1.0 / k)
    filtered, predicted, loglik = np.zeros((n, k)), np.zeros((n, k)), np.zeros(n)
    for t in range(n):
        means, covs = state_params(t)
        # The counts decayed over the row's own clock before it reads them.
        counts = np.einsum(
            "s,skl->kl", np.exp(clock[t + 1] - clock[1 : t + 1]), w[:t, None, None] * joints[:t]
        )
        pi = (counts + TAU) / (counts + TAU).sum(axis=1, keepdims=True)
        rows["counts"][t], rows["pi"][t] = counts, pi
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
    state_params(n)
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
        "rows": rows,
    }


def density_rounding(sigma: np.ndarray, y: np.ndarray, delta: np.ndarray, eigen: bool) -> float:
    """The most one side's own rounding moves ``ln f = -(d ln 2 pi + ln det
    Sigma + delta' Sigma^-1 delta) / 2``, ``y = Sigma^-1 delta``.

    Through a Cholesky factor (the model's ``faer``, hmmlearn's LAPACK): the
    solve's backward error is ``|dSigma| <= gamma_{3d+1} (1 + gamma_{d+1})
    S``, ``S_ij = sqrt(Sigma_ii Sigma_jj)`` (Higham, Theorems 10.3, 10.4 and
    Cauchy-Schwarz), which moves the quadratic form by ``|y|' dSigma |y|``
    and ``ln det`` by ``tr(|Sigma^-1| dSigma)``; the form's sum rounds
    ``gamma_d`` of ``|delta|' |y|``; each ``ln l_ii`` is the platform's and
    their sum rounds ``gamma_d``, on terms between ``ln Sigma_ii`` and ``-ln
    (Sigma^-1)_ii``. Through an eigendecomposition (scipy's ``logpdf``): a
    dense backward error ``C_EIG u ||Sigma||_F`` in place of ``S``. Then the
    combination: the platform's ``ln 2 pi`` and ``gamma_2`` of the three
    terms."""
    d = len(y)
    inv = np.linalg.inv(sigma)
    q = float(abs(delta @ y))
    log_det = float(np.linalg.slogdet(sigma)[1])
    if eigen:
        backward = np.full((d, d), C_EIG * U * np.linalg.norm(sigma))
        factor = backward
    else:
        s = np.sqrt(np.outer(np.diagonal(sigma), np.diagonal(sigma)))
        backward = gamma(3 * d + 1) * (1.0 + gamma(d + 1)) * s
        factor = gamma(d + 1) * (1.0 + gamma(d + 1)) * s
    ay = np.abs(y)
    d_form = float(ay @ backward @ ay) + gamma(d) * float(np.abs(delta) @ ay)
    logs = float(np.sum(np.abs(np.log(np.diagonal(sigma))) + np.abs(np.log(np.diagonal(inv)))))
    d_log_det = float(np.sum(np.abs(inv) * factor)) + (LIBM * U + gamma(d)) * logs
    c = d * np.log(2.0 * np.pi)
    return 0.5 * (d_form + d_log_det) + (LIBM + 1.0) * U * c + gamma(2) * (c + abs(log_det) + q)


def rule_error_bound(
    s: dict[str, Any],
    half_life: float,
    cov: str,
    want: dict[str, Any],
    detail: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """How far two correct evaluations of the rule in doubles may differ --
    the model's recursion against the rebuild's batch sums -- derived to
    first order from the rule's own quantities, not measured.

    Each place either side rounds is a *symbol* ``e_i`` of known magnitude
    ``m_i``, a few units of ``u`` from its operation count; every quantity
    the rule carries is written as its exact value plus ``sum_i J_i e_i``
    and the sensitivities ``J_i`` are propagated, with their signs, through
    the rule's own linearization. A bound is then ``sum_i |J_i| m_i``, the
    worst case over every pattern of rounding. Taking absolute values at
    each row instead compounds the loop posterior -> moments -> density ->
    posterior, whose gain in the worst alignment is the row's Mahalanobis
    distance, to ``1e100`` over the stream; the signed sum is what the rule
    itself does with its rounding. The quantities are the rebuild's
    (``want["rows"]``): either side's are within the error itself of the
    exact ones, which moves a first-order bound by a second-order amount.

    *Where a density reads the moments.* ``ln f_l`` has gradient ``y_l =
    Sigma_l^-1 delta_l`` in the mean and ``-(Sigma_l^-1 - y_l y_l') / 2``
    in the covariance. ``|y_l|^2 <= q_l / lambda_min(Sigma_l)``: a row far
    from a thin state reads its moments' error magnified, in the row's
    Mahalanobis distance over the covariance's least eigenvalue. Each
    side's own rounding of the density is :func:`density_rounding`; the
    filter's, ``gamma_k`` for ``p Pi`` and the platform's ``ln`` and
    ``exp`` on ``ln p~_l + ln f_l``, ``(LIBM + 2) u`` of each.

    *The filter.* ``post_l = p~_l f_l / Z``, ``p~ = p Pi``: ``dln post_l =
    a_l - sum_m post_m a_m`` with ``a_l = dln p~_l + dln f_l`` and ``dln
    p~_l = sum_k c_kl (dln p_k + dln Pi_kl)``, ``c_kl = p_k Pi_kl / p~_l``.
    A state whose posterior has underflowed carries any error at no
    weight: ``c_kl`` and its share are 0. ``dp_l = p_l dln p_l``.

    *Where the moments read a posterior.* The model's step is ``mu_l' =
    mu_l + s delta``, ``C_l' = (1 - s) (C_l + s delta delta')``, ``W_l' =
    lam W_l + w post_l``, ``s = w post_l / W_l'``; the rebuild's weighted
    sums are the same numbers. So ``dln s = (1 - s) (dln post_l - dln
    W_l)`` and ``dmu' = (1 - s) dmu + s delta dln s``, ``dC' = (1 - s) dC +
    s ((1 - 2s) delta delta' - C) dln s - s (1 - s) (delta dmu' + dmu
    delta')``. ``s = w post_l / W_l'`` is what amplifies: a state whose
    weight has decayed far below a row's weight hands that row most of its
    moments, however small the row's posterior, so the *relative* error of
    a posterior as small as ``1e-5`` becomes the moments' error, and the
    next row on that state reads it through ``y_l``. The counts ``A' =
    lam A + w xi`` take ``xi``'s relative error the same way, and ``Pi =
    (A + tau) / rowsum`` reads ``A``'s against the pseudo-count. The ridge
    ``r_l = precision_prior lam^T / W_l`` follows ``W_l``.

    *The ages.* The model decays by ``lam = exp2(-1 / h)`` once a row and
    the rebuild by ``exp((t - s) ln lam)``: one symbol for each platform
    call, ``LIBM u`` on ``lam`` and ``(LIBM + 1) u |ln lam|`` on ``ln
    lam``, enters every row's weight once per row of age, so the oldest
    rows are tilted most against the newest; the model's product and the
    rebuild's clock arithmetic and ``exp`` round once more per row.

    *Each side's own moment arithmetic.* The model's step: ``s`` from four
    operations, the product ``s delta``, a compensated add that is exact
    while the step is below the mean, ``7 u`` of ``s |delta|``; the
    covariance's ``a C + a b d_i d_j``, ``7 u`` of ``s |delta_i delta_j|``
    and ``2 u`` of ``|C|`` where the step is not absorbed (``s > u / 4``).
    The rebuild's ``average`` and ``cov`` over ``t + 1`` terms are fresh
    at every row: ``gamma_{t+4}`` of ``sum share |x_s|`` and
    ``gamma_{t+6}`` of ``sum share |x_s - mu| |x_s - mu|'`` (any order of
    summation, Higham section 3.1), the latter bounded through ``|x_s| +
    |mu|``, and they reach the chain only through that row's posterior.

    Returns, per quantity the test compares, the bound in the test's own
    measure: ``filtered`` and ``predicted`` absolute per row and state,
    ``loglik`` relative to ``1 + |loglik|`` per row, the learned ``means``
    and ``covs`` relative to ``1 + |x|``, ``counts`` relative to the largest
    count. ``detail``, when given, receives the final sensitivities
    (``jac``), the symbols' magnitudes (``mag``), the row each symbol
    belongs to (``row``) and the state's layout, for inspection."""
    x, w, m0 = s["x"], s["w"], s["m0"]
    n, d = x.shape
    k = m0.shape[0]
    rows = want["rows"]
    log_lam = np.log(0.5) / half_life
    eye = np.eye(d)
    gauss = d * np.log(2.0 * np.pi)
    # The state's error vector: each state's mean, moment matrix, ln W and
    # ln ridge, the counts, ln p.
    o_mu, o_c = 0, k * d
    o_w = o_c + k * d * d
    o_r = o_w + k
    o_a = o_r + k
    o_p = o_a + k * k
    z = o_p + k

    def i_mu(st: int, i: int) -> int:
        return o_mu + st * d + i

    def i_c(st: int, i: int, j: int) -> int:
        return o_c + (st * d + i) * d + j

    # The symbols: the two platform decay factors, then per row k each of
    # psi (the row's fresh density and filter error), chi (the joint's own),
    # zeta (the stored posterior's own rounding, uncentred), e1..e4 (the
    # weight's product, the weight's add, the share's product and division,
    # the ridge's division), d of the mean's step, d^2 of the covariance's,
    # and k^2 of the counts'.
    per_row = k * (7 + d + d * d) + k * k
    psi_parts = np.zeros((n, k, 4))
    big = 2 + n * per_row
    jac = np.zeros((z, big))
    mag = np.zeros(big)
    mag[0] = LIBM * U
    mag[1] = (LIBM + 1.0) * U * abs(log_lam)
    symbol_row = np.full(big, -1)
    used = 2
    out = {
        "filtered": np.zeros((n, k)),
        "predicted": np.zeros((n, k)),
        "loglik": np.zeros(n),
    }

    def fresh(t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The rebuild's own errors as row ``t`` reads the moments: numpy's
        ``average`` sums the ``t + 1`` weighted points in row order, so its
        error is ``u`` of each partial sum (Higham, (4.3)), the recent heavy
        rows in few of them; ``cov``'s BLAS dot sums in an order of its own,
        so ``gamma_t`` of its terms' magnitudes (Higham, section 3.1); the
        clock arithmetic and ``exp`` of each age; the weights' and the
        ridges' (relative)."""
        clock = (t - 1) * abs(log_lam) if t else 0.0
        flat = (2.0 * clock + LIBM) * U
        mu = np.abs(rows["means"][t])
        c = np.abs(rows["moments"][t])
        ordered = U * (1.0 + gamma(t + 3))
        f_w = flat + ordered * rows["ordered_w"][t]
        f_mu = ordered * rows["ordered"][t] + gamma(3) * rows["abs1"][t]
        f_mu = f_mu + (f_w + U)[:, None] * mu + flat * rows["dev1"][t]
        f_c = (gamma(t + 3) + flat) * rows["dev2"][t] + (3.0 * U + f_w[:, None, None] + flat) * c
        f_r = rows["ridges"][t] * ((2.0 * clock + LIBM + 1.0) * U + f_w)
        return f_mu, f_c, f_w, f_r

    for t in range(n):
        mu, c, totals = rows["means"][t], rows["moments"][t], rows["totals"][t]
        ridge, sigma = rows["ridges"][t], rows["sigma"][t]
        counts, pi = rows["counts"][t], rows["pi"][t]
        p, pred = want["filtered"][t], want["predicted"][t]
        # This row's symbols, appended as unit rows of an augmented Jacobian.
        new = used + np.arange(per_row)
        symbol_row[new] = t
        s_psi, s_chi, s_zeta = new[:k], new[k : 2 * k], new[2 * k : 3 * k]
        s_e = new[3 * k : 7 * k].reshape(k, 4)
        s_mu = new[7 * k : 7 * k + k * d].reshape(k, d)
        s_c = new[7 * k + k * d : 7 * k + k * d + k * d * d].reshape(k, d, d)
        s_a = new[7 * k + k * d + k * d * d :].reshape(k, k)
        width = used + per_row
        aug = np.zeros((z + per_row, width))
        aug[:z, :used] = jac[:, :used]
        aug[z:, used:width] = np.eye(per_row)
        unit = np.eye(z + per_row)

        fresh_rows = unit[z:]

        # Ageing: the states from the second row on, the counts from the
        # first, each by the model's factor against the rebuild's ages.
        tilt = np.zeros(width)
        tilt[0], tilt[1] = 1.0, -1.0
        if t >= 2:
            for st in range(k):
                aug[o_w + st] += tilt + aug[z + s_e[st, 0] - used]
                mag[s_e[st, 0]] = U
        if t >= 1:
            for kk in range(k):
                for ll in range(k):
                    aug[o_a + kk * k + ll] += counts[kk, ll] * (tilt + aug[z + s_a[kk, ll] - used])
                    mag[s_a[kk, ll]] = 2.0 * U * counts[kk, ll]
        # The covariances as read, as functionals of the state, and their
        # fresh errors; the densities' functionals and the rows' symbols.
        f_mu, f_c, f_w, f_r = fresh(t)
        share = totals / totals.sum()
        sig_fun = np.zeros((k, d, d, z + per_row))
        sig_fresh = np.zeros((k, d, d))
        for st in range(k):
            for i in range(d):
                for j in range(d):
                    if cov == "diag" and i != j:
                        continue
                    sig_fun[st, i, j, i_c(st, i, j)] = 1.0
                    sig_fresh[st, i, j] = f_c[st, i, j]
                sig_fun[st, i, i, o_r + st] += ridge[st]
                sig_fresh[st, i, i] += f_r[st]
        if cov == "tied":
            each = np.abs(c + ridge[:, None, None] * eye)
            pooled = np.einsum("s,sijz->ijz", share, sig_fun)
            pooled_fresh = np.einsum("s,sij->ij", share, sig_fresh)
            for st in range(k):
                d_share = np.zeros(z + per_row)
                d_share[o_w + st] += share[st]
                d_share[o_w : o_w + k] -= share[st] * share
                pooled = pooled + each[st][:, :, None] * d_share
                pooled_fresh = pooled_fresh + each[st] * share[st] * (f_w[st] + f_w.max())
            pooled_fresh = pooled_fresh + 2.0 * gamma(k + 2) * np.abs(sigma[0])
            sig_fun = np.broadcast_to(pooled, sig_fun.shape)
            sig_fresh = np.broadcast_to(pooled_fresh, sig_fresh.shape)
        logf = np.zeros(k)
        f_fun = np.zeros((k, z + per_row))
        for st in range(k):
            delta = x[t] - mu[st]
            y = np.linalg.solve(sigma[st], delta)
            inv = np.linalg.inv(sigma[st])
            q = float(delta @ y)
            logf[st] = -0.5 * (gauss + np.linalg.slogdet(sigma[st])[1] + q)
            grad_sigma = -0.5 * (inv - np.outer(y, y))
            f_fun[st, i_mu(st, 0) : i_mu(st, 0) + d] = y
            f_fun[st] += np.einsum("ij,ijz->z", grad_sigma, sig_fun[st])
            read = float(np.abs(y) @ f_mu[st]) + float(np.sum(np.abs(grad_sigma) * sig_fresh[st]))
            cholesky = density_rounding(sigma[st], y, delta, eigen=False)
            eigen = density_rounding(sigma[st], y, delta, eigen=True)
            filter_round = gamma(k) + (LIBM + 2.0) * U * (abs(np.log(pred[st])) + abs(logf[st]))
            mag[s_psi[st]] = read + 2.0 * (cholesky + filter_round + gamma(k + 2))
            psi_parts[t, st] = (
                float(np.abs(y) @ f_mu[st]),
                float(np.sum(np.abs(grad_sigma) * sig_fresh[st])),
                2.0 * cholesky,
                2.0 * filter_round,
            )
            mag[s_chi[st]] = eigen + 2.0 * gamma(k * k + 2)
        mag[s_chi] += 2.0 * (LIBM + 2.0) * U * (np.abs(logf) + abs(logf.max()))
        # ln Pi and ln p~ as functionals; a_l = ln p~_l + ln f_l + psi_l.
        row_sum = counts.sum(axis=1) + k * TAU
        ln_pi = np.zeros((k, k, z + per_row))
        for kk in range(k):
            for ll in range(k):
                ln_pi[kk, ll, o_a + kk * k + ll] += 1.0 / (counts[kk, ll] + TAU)
                ln_pi[kk, ll, o_a + kk * k : o_a + kk * k + k] -= 1.0 / row_sum[kk]
        weight = p[:, None] * pi / pred[None, :]
        ln_pred = np.einsum("kl,kz->lz", weight, unit[o_p : o_p + k]) + np.einsum(
            "kl,klz->lz", weight, ln_pi
        )
        a_fun = ln_pred + f_fun + unit[z + s_psi - used]
        scaled = np.exp(logf - logf.max())
        post = pred * scaled / float(pred @ scaled)
        # The outputs the row reports, each in the test's own measure.
        filt = p[:, None] * aug[o_p : o_p + k]
        out["filtered"][t] = np.abs(filt) @ mag[:width]
        out["predicted"][t] = np.abs(pred[:, None] * (ln_pred @ aug)) @ mag[:width]
        out["predicted"][t] += 2.0 * gamma(k) * pred
        loglik = float(np.abs((post @ a_fun) @ aug) @ mag[:width])
        loglik += 2.0 * gamma(k + 1) + 2.0 * LIBM * U * abs(want["loglik"][t])
        out["loglik"][t] = loglik / (1.0 + abs(want["loglik"][t]))
        if w[t] <= 0.0:
            jac[:, :width] = aug[:z]
            used = width
            continue
        # The learning step as a linear map of the augmented state.
        # The posterior's log error, centred, plus each stored component's
        # own rounding, which a centred error gives the dominant component
        # at no weight: each side normalizes at the magnitude of ``ln p~_l
        # f_l`` and ``ln Z`` (hmmlearn's ``a_l - (max + ln sum exp(a -
        # max))``, the model's ``exp(v_l - top)``), then its exp, sum and
        # division.
        ln_post = a_fun - post @ a_fun + unit[z + s_zeta - used]
        mag[s_zeta] = 2.0 * (
            (LIBM + 2.0) * U * (np.abs(np.log(pred) + logf) + abs(want["loglik"][t]))
            + gamma(k + 1)
            + LIBM * U
        )
        step = np.zeros((z, z + per_row))
        step[o_p : o_p + k] = ln_post
        shares = w[t] * post / rows["totals"][t + 1]
        for st in range(k):
            sh = shares[st]
            w_aged = unit[o_w + st]
            e1, e2, e3, e4 = (fresh_rows[s_e[st, i] - used] for i in range(4))
            mag[s_e[st, 1]], mag[s_e[st, 2]], mag[s_e[st, 3]] = U, 2.0 * U, U
            ln_share = (1.0 - sh) * (ln_post[st] - w_aged) - e2 + e3
            step[o_w + st] = (1.0 - sh) * w_aged + sh * ln_post[st] + e2
            step[o_r + st] = unit[o_r + st] + sh * (w_aged - ln_post[st]) - e2 - e4
            delta = x[t] - mu[st]
            for i in range(d):
                step[i_mu(st, i)] = (1.0 - sh) * unit[i_mu(st, i)] + sh * delta[i] * ln_share
                step[i_mu(st, i)] += fresh_rows[s_mu[st, i] - used]
                mag[s_mu[st, i]] = 7.0 * U * sh * abs(delta[i])
                for j in range(d):
                    coef = sh * ((1.0 - 2.0 * sh) * delta[i] * delta[j] - c[st, i, j])
                    step[i_c(st, i, j)] = (1.0 - sh) * unit[i_c(st, i, j)] + coef * ln_share
                    step[i_c(st, i, j)] -= sh * (1.0 - sh) * delta[i] * unit[i_mu(st, j)]
                    step[i_c(st, i, j)] -= sh * (1.0 - sh) * delta[j] * unit[i_mu(st, i)]
                    step[i_c(st, i, j)] += fresh_rows[s_c[st, i, j] - used]
                    mag[s_c[st, i, j]] = 7.0 * U * sh * abs(delta[i] * delta[j]) + 2.0 * U * min(
                        1.0, 4.0 * sh / U
                    ) * abs(c[st, i, j] + sh * delta[i] * delta[j])
        xi = rows["joints"][t]
        ln_xi = np.zeros((k, k, z + per_row))
        for kk in range(k):
            for ll in range(k):
                ln_xi[kk, ll] = unit[o_p + kk] + ln_pi[kk, ll] + f_fun[ll]
                ln_xi[kk, ll] += fresh_rows[s_psi[ll] - used] + fresh_rows[s_chi[ll] - used]
        ln_xi = ln_xi - np.einsum("kl,klz->z", xi, ln_xi)
        for kk in range(k):
            for ll in range(k):
                step[o_a + kk * k + ll] = (
                    unit[o_a + kk * k + ll] + w[t] * xi[kk, ll] * ln_xi[kk, ll]
                )
        jac[:, :width] = step @ aug
        used = width
    f_mu, f_c, _, _ = fresh(n)
    d_mean = np.abs(jac[o_mu:o_c, :used]) @ mag[:used] + f_mu.ravel()
    d_cov = np.abs(jac[o_c:o_w, :used]) @ mag[:used] + f_c.ravel()
    out["means"] = d_mean.reshape(k, d) / (1.0 + np.abs(want["means"]))
    out["covs"] = d_cov.reshape(k, d, d) / (1.0 + np.abs(want["covs"]))
    final = want["counts"]
    clock = (n - 1) * abs(log_lam)
    d_counts = np.abs(jac[o_a:o_p, :used]) @ mag[:used]
    d_counts += ((2.0 * clock + LIBM + 3.0) * U + gamma(n + 3)) * final.ravel()
    out["counts"] = d_counts.reshape(k, k) / final.max()
    if detail is not None:
        detail.update(
            jac=jac[:, :used],
            mag=mag[:used],
            row=symbol_row[:used],
            layout=(o_mu, o_c, o_w, o_r, o_a, o_p),
            psi_parts=psi_parts,
        )
    return out


def test_numpy_sums_rows_in_order():
    """The premise of the bound's ``average`` term: numpy reduces a
    two-dimensional array over its rows one row after another, first to
    last (its pairwise summation serves a contiguous inner reduce only), so
    the rounding of each partial sum is the error, and the oldest, lightest
    rows come first. ``1e16`` then 298 ones then ``-1e16`` sum to 0 in that
    order; a pairwise or reordered sum keeps some of the ones."""
    a = np.ones((300, 2))
    a[0], a[-1] = 1e16, -1e16
    assert a.sum(axis=0).tolist() == [0.0, 0.0]
    assert (np.average(a, axis=0, weights=np.ones(300)) == 0.0).all()
    a[0], a[-2], a[-1] = 1.0, 1e16, -1e16
    assert a.sum(axis=0).tolist() == [298.0, 298.0]


ledger.guard(
    "hmm_online_em",
    lambda: [
        model_run(hmm_stream(seed, k), hl, cov) for seed in range(10) for cov, k, hl in EM_CASES
    ],
    oracles=["hmmlearn", "scipy", "numpy"],
    bits=52,
)


def em_errors(seed: int, cov: str, k: int, half_life: float) -> dict[str, float]:
    """Each quantity's worst error against the rule rebuilt, over its derived
    bound (:func:`rule_error_bound`; at most 1 passes), the bound floored at
    one rounding ``u`` where the two sides agree to the bit; and two numbers
    that show the paths ran: how far the learned means moved from the given
    ones, and the largest count."""
    s = hmm_stream(seed, k)
    out, learned = model_run(s, half_life, cov)
    want = rule_from_scratch(s, half_life, cov)
    bound = rule_error_bound(s, half_life, cov, want)

    def over(error: np.ndarray, key: str) -> float:
        return float((error / np.maximum(bound[key], U)).max())

    def rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a - b) / (1.0 + np.abs(b))

    filtered = out.select(f"filtered_state{j}" for j in range(k)).to_numpy()
    predicted = out.select(f"predicted_state{j}" for j in range(k)).to_numpy()
    return {
        "filtered": over(np.abs(filtered - want["filtered"]), "filtered"),
        "predicted": over(np.abs(predicted - want["predicted"]), "predicted"),
        "loglik": over(rel(out["loglik"].to_numpy(), want["loglik"]), "loglik"),
        "means": over(rel(learned["means"], want["means"]), "means"),
        "covs": over(rel(learned["covs"], want["covs"]), "covs"),
        "counts": over(np.abs(learned["counts"] - want["counts"]) / want["counts"].max(), "counts"),
        "moved": float(np.abs(learned["means"] - s["m0"]).max()),
        "largest_count": float(want["counts"].max()),
    }


ERRORS = ("filtered", "predicted", "loglik", "means", "covs", "counts")


class TestOnlineEmIsItsRuleRebuilt:
    """Under a finite half-life, the learned states and transitions and the
    filter that reads them are the rule rebuilt with hmmlearn, scipy and
    numpy, on uneven and zero row weights, each quantity within the error
    bound derived from the rule's own conditioning at each row."""

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
        print(f"worst over its bound {worst.max():.3g}, 95th percentile {p95:.3g}")
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
