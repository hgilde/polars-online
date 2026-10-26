"""The regime models against third-party libraries past the one case each
had (task 278, batch 30).

``tests/test_second_opinion.py`` held ``hmm`` to hmmlearn at two states and
``covariance_type="full"`` only, ``bocpd`` to ``bayesian_changepoint_detection``
at ``mean_precision_prior = 1`` and a ``degrees_of_freedom_prior`` given, and
``corrchange``'s ``monitor`` critical value to three published table entries.
This module carries each to the parameters those left out:

- ``hmm`` under ``"diag"`` and ``"tied"`` covariances and at three states,
  against hmmlearn's ``GaussianHMM``;
- ``bocpd`` at ``mean_precision_prior`` 0.3 and 5 with the default
  ``degrees_of_freedom_prior``, against the package's ``StudentT`` run-length
  recursion;
- the ``monitor`` kind's critical value against ``scipy.stats.kstwobign``,
  the Kolmogorov distribution, at every level from ``1e-12`` to
  ``1 - 1e-6`` and under Bonferroni's share.

The libraries are the oracles, taken as given; what is tested is this
library's filter, recursion and quantile. Each oracle reads the rows the
model learned from: neither library takes a row weight, so a stream's
weights are those the model's arithmetic makes no difference to -- every
positive weight under a filter that does not learn, a constant weight in
``bocpd`` -- and its zero-weight rows, which teach the model nothing, are
left out of the oracle's rows.

The single seeds run in the essentials; the 50-seed grids are
``oracle_grid``, judged by the worst seed.
"""

from __future__ import annotations

import functools
from typing import Any

import bayesian_changepoint_detection.online_changepoint_detection as bcd
import numpy as np
import polars as pl
import pytest
from hmmlearn.hmm import GaussianHMM
from scipy.stats import kstwobign

import ledger
import polars_online as po

TIER = "mixed"

SEEDS = range(50)

#: The unit roundoff of a double.
U = 2.0**-53


# --- hmm against hmmlearn ---------------------------------------------------------

#: Every row's filtered and predicted probabilities, absolute, and its
#: log-likelihood, relative to ``1 + |loglik|``. Measured 2026-10-25 over
#: ``SEEDS`` at every shape and both state counts (aarch64, macOS): the
#: worst 1.85e-13 (seed 3, ``"full"`` at three states), the 95th percentile
#: of the 300 cases' worst 1.16e-13; ``TestHmmIsHmmlearns`` measured 1.4e-13 at two
#: states. The bound leaves fifty times that for another CPU's last bits.
HMM_TOL = 1e-11

#: The ridge on every state's covariance.
HMM_RIDGE = 1e-2

HMM_CASES = [(cov, k) for cov in ("full", "diag", "tied") for k in (2, 3)]


def hmm_stream(seed: int, k: int, n: int = 300, d: int = 2) -> dict[str, Any]:
    """A sticky chain over ``k`` states with random means and covariances,
    its rows, and their weights: drawn from 0.2 to 3, with about one row in
    seven at weight 0, the stream's second row among them."""
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
    x = means[state] + noise
    w = rng.uniform(0.2, 3.0, n)
    w[rng.random(n) < 1 / 7] = 0.0
    w[1] = 0.0
    return {"x": x, "w": w, "pi": pi, "means": means, "covs": covs}


def hmm_run(s: dict[str, Any], cov: str) -> pl.DataFrame:
    k, d = s["means"].shape
    features = [f"x{i}" for i in range(d)]
    spec = po.spec.hmm(
        "h",
        features=features,
        n_components=k,
        precision_prior=HMM_RIDGE,
        covariance_type=cov,
        half_life=1e9,
        learn=False,
        means=s["means"].ravel().tolist(),
        covs=s["covs"].ravel().tolist(),
        transition=s["pi"].ravel().tolist(),
        weight="w",
    )
    frame = pl.DataFrame({f: s["x"][:, i] for i, f in enumerate(features)}).with_columns(w=s["w"])
    return po.ModelBank([spec]).fit_predict(frame)["h"].struct.unnest()


#: The ledger's inputs (tests/ledger.py): the filter's every field on ten
#: seeds at every shape, where the grid reads fifty. The oracle agrees to
#: round-off, so the hash keeps the doubles themselves.
ledger.guard(
    "hmm_shapes",
    lambda: [hmm_run(hmm_stream(s, k), cov) for s in range(10) for cov, k in HMM_CASES],
    oracles=["hmmlearn"],
    bits=52,
)


def hmmlearn_model(s: dict[str, Any], cov: str) -> GaussianHMM:
    """hmmlearn's ``GaussianHMM`` at the given parameters. Each state's
    covariance is the given one plus the ridge on the diagonal; ``"diag"``
    reads its diagonal, and ``"tied"`` pools the states by their weights,
    equal here because each given state enters at one row's weight and a
    filter that does not learn adds none. The filter starts uniform, so
    hmmlearn's start is ``uniform @ Pi``."""
    k, d = s["means"].shape
    model = GaussianHMM(n_components=k, covariance_type=cov, init_params="", params="")
    model.startprob_ = np.full(k, 1.0 / k) @ s["pi"]
    model.transmat_ = s["pi"]
    model.means_ = s["means"]
    if cov == "full":
        model.covars_ = s["covs"] + HMM_RIDGE * np.eye(d)
    elif cov == "diag":
        model.covars_ = np.diagonal(s["covs"], axis1=1, axis2=2) + HMM_RIDGE
    else:
        model.covars_ = s["covs"].mean(axis=0) + HMM_RIDGE * np.eye(d)
    return model


def hmm_errors(seed: int, cov: str, k: int) -> tuple[float, float, float]:
    """The worst error of the filtered probabilities, the predicted ones and
    the log-likelihoods against hmmlearn on the rows of positive weight.

    Everything is read before the row. On a learned row ``t``,
    ``filtered_state`` is hmmlearn's filtered probability after the learned
    row before it (the last of ``predict_proba`` over the learned rows up to
    it), ``predicted_state`` is that times ``Pi``, and ``loglik`` is the
    difference of ``score`` over the learned rows up to ``t`` and up to the
    one before. A zero-weight row reads the same filter and is left out of
    it: its ``filtered_state`` is the learned rows' so far, and its
    ``loglik`` is ``log p(x_t | the learned rows before)``."""
    s = hmm_stream(seed, k)
    out = hmm_run(s, cov)
    model = hmmlearn_model(s, cov)
    x, pi = s["x"], s["pi"]
    learned = np.flatnonzero(s["w"] > 0)
    rows = x[learned]
    # The filter after 0, 1, ..., len(learned) learned rows.
    after = np.vstack(
        [np.full(k, 1.0 / k)] + [model.predict_proba(rows[: j + 1])[-1] for j in range(len(rows))]
    )
    scores = np.array([0.0] + [model.score(rows[: j + 1]) for j in range(len(rows))])
    # How many learned rows come before each row.
    before = np.searchsorted(learned, np.arange(len(x)))
    want_filtered = after[before]
    want_loglik = np.diff(scores)[np.minimum(before, len(learned) - 1)]
    for t in np.flatnonzero(s["w"] == 0):
        prefix = rows[: before[t]]
        with_t = np.vstack([prefix, x[t : t + 1]])
        want_loglik[t] = model.score(with_t) - (model.score(prefix) if len(prefix) else 0.0)
    filtered = out.select(f"filtered_state{j}" for j in range(k)).to_numpy()
    predicted = out.select(f"predicted_state{j}" for j in range(k)).to_numpy()
    loglik = out["loglik"].to_numpy()
    return (
        float(np.abs(filtered - want_filtered).max()),
        float(np.abs(predicted - want_filtered @ pi).max()),
        float((np.abs(loglik - want_loglik) / (1.0 + np.abs(want_loglik))).max()),
    )


def hmm_evidence(seed: int, cov: str, k: int) -> tuple[float, float, int]:
    """That the paths the oracle checks were taken: how far the filter moves
    from uniform (it tells the states apart), how far this shape's
    log-likelihoods are from ``"full"``'s on the same stream (the shape was
    read), and how many zero-weight rows the stream holds."""
    s = hmm_stream(seed, k)
    out = hmm_run(s, cov)
    filtered = out.select(f"filtered_state{j}" for j in range(k)).to_numpy()
    full = hmm_run(s, "full") if cov != "full" else out
    spread = float(np.abs(out["loglik"].to_numpy() - full["loglik"].to_numpy()).max())
    return float(np.abs(filtered - 1.0 / k).max()), spread, int((s["w"] == 0).sum())


class TestHmmIsHmmlearnsAtEveryShape:
    """``hmm`` with given states and ``learn = False`` is Hamilton's filter
    at fixed parameters, which hmmlearn's forward pass computes at each
    ``covariance_type`` and any number of states. A positive weight moves
    nothing in a filter that does not learn, so the stream's weights vary
    from 0.2 to 3; a zero-weight row moves nothing at all, so hmmlearn reads
    only the learned rows."""

    @pytest.mark.parametrize(("cov", "k"), [("diag", 2), ("tied", 2), ("full", 3)])
    def test_one_seed(self, cov, k):
        moved, spread, zeros = hmm_evidence(0, cov, k)
        print(
            f"{cov} k={k}: filter from uniform {moved:.3f}, from full {spread:.3f}, {zeros} zeros"
        )
        assert moved > 0.4 and zeros > 10
        if cov != "full":
            assert spread > 0.1
        errors = hmm_errors(0, cov, k)
        assert max(errors) <= HMM_TOL, errors

    @pytest.mark.oracle_grid(reason="50 seeds x 6 cases, each with hmmlearn on every prefix")
    def test_fifty_seeds(self):
        # Measured 2026-10-25: the errors at HMM_TOL's comment; on every case
        # the filter moved at least 0.273 from uniform, "diag" and "tied" at
        # least 0.314 in loglik from "full", and at least 32 rows weighed 0.
        rows = [
            (max(hmm_errors(seed, *case)), *hmm_evidence(seed, *case), seed, case)
            for seed in SEEDS
            for case in HMM_CASES
        ]
        worst = max(rows)
        errors = np.array([r[0] for r in rows])
        print(
            f"hmm: worst error {worst[0]:.2e} (seed {worst[4]}, case {worst[5]}), "
            f"95th percentile {np.quantile(errors, 0.95):.2e}"
        )
        # Round-off: the ledger holds it at about twice the measured 1.85e-13.
        ledger.record("hmm_shapes", "error", errors, floor=4e-13)
        moved = min(r[1] for r in rows)
        spread = min(r[2] for r in rows if r[5][0] != "full")
        zeros = min(r[3] for r in rows)
        print(f"hmm: least move from uniform {moved:.3f}, from full {spread:.3f}, zeros {zeros}")
        assert moved > 0.2 and spread > 0.1 and zeros > 10
        assert worst[0] <= HMM_TOL, worst


# --- bocpd against bayesian_changepoint_detection ------------------------------------

#: The prior every case shares but for ``mean_precision_prior``: the hazard,
#: the prior scale, and the default ``degrees_of_freedom_prior`` -- 3 under
#: ``"diag"``, the package's ``alpha = 3/2``.
BOCPD_HAZARD = 40.0
BOCPD_PSI0 = 1.0
BOCPD_NU0_DEFAULT = 3.0

#: A constant weight, which ``bocpd`` divides out (a row teaches at its
#: weight over the mean weight).
BOCPD_WEIGHT = 2.5

BOCPD_CASES = [(kappa0, offset) for kappa0 in (0.3, 5.0) for offset in (0.0, 1e6)]


def bocpd_stream(seed: int, offset: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Three mean shifts of 1.5 to 4 noise standard deviations at random
    rows, at a level ``offset``, and the weights: ``BOCPD_WEIGHT`` with about
    one row in nine at 0, the stream's second row among them."""
    rng = np.random.default_rng(seed)
    level = np.zeros(n)
    for cut in np.sort(rng.choice(np.arange(20, n - 20), 3, replace=False)):
        level[cut:] += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 4.0)
    x = offset + level + rng.normal(0.0, 1.0, n)
    w = np.full(n, BOCPD_WEIGHT)
    w[rng.random(n) < 1 / 9] = 0.0
    w[1] = 0.0
    return x, w


def bocpd_run(x: np.ndarray, w: np.ndarray, kappa0: float, offset: float) -> pl.DataFrame:
    spec = po.spec.bocpd(
        "b",
        features=["x"],
        hazard=BOCPD_HAZARD,
        covariance_type="diag",
        mean_prior=[offset],
        mean_precision_prior=kappa0,
        covariance_prior=[BOCPD_PSI0],
        prune_below=0.0,
        max_run=len(x) + 2,
        min_weight=0.0,
        weight="w",
    )
    return po.ModelBank([spec]).fit_predict(pl.DataFrame({"x": x, "w": w}))["b"].struct.unnest()


ledger.guard(
    "bocpd_priors",
    lambda: [
        bocpd_run(*bocpd_stream(s, offset), kappa0, offset)
        for s in range(10)
        for kappa0, offset in BOCPD_CASES
    ],
    oracles=["bayesian-changepoint-detection"],
    bits=52,
)


def bocpd_errors(seed: int, kappa0: float, offset: float) -> float:
    """The worst error over the learned rows against the package, each over
    its tolerance: at most 1 passes.

    The package's ``StudentT(alpha, beta, kappa, mu)`` is the
    normal-inverse-gamma model ``"diag"`` is at one feature, with ``alpha =
    nu0/2``, ``beta = covariance_prior/2``, ``kappa = mean_precision_prior``
    and ``mu = mean_prior``, through Adams and MacKay's recursion at a
    constant hazard with nothing pruned. Over the learned rows, ``R[0, j+1]
    + R[1, j+1]`` is ``p_change`` on the ``j``-th, ``argmax R[:, j]`` its
    ``run_mode`` and ``sum r R[r, j]`` its ``run_mean``. The tolerance is
    ``TestBocpdAtALevel``'s, ``1e-9 + 1e-14 L``: the data is resolved to ``L
    eps`` before either side reads it."""
    x, w = bocpd_stream(seed, offset)
    out = bocpd_run(x, w, kappa0, offset)
    learned = w > 0
    rows = x[learned]
    m = len(rows)
    r, modes = bcd.online_changepoint_detection(
        rows,
        functools.partial(bcd.constant_hazard, BOCPD_HAZARD),
        bcd.StudentT(BOCPD_NU0_DEFAULT / 2.0, BOCPD_PSI0 / 2.0, kappa0, offset),
    )
    tol = 1e-9 + 1e-14 * abs(offset)
    p_change = out["p_change"].to_numpy()[learned]
    run_mean = out["run_mean"].to_numpy()[learned]
    run_mode = out["run_mode"].to_numpy()[learned]
    want_mean = np.arange(m + 1) @ r[:, :m]
    np.testing.assert_array_equal(run_mode, modes[:m])
    return max(
        float(np.abs(p_change - (r[0, 1:] + r[1, 1:])).max()) / tol,
        float((np.abs(run_mean - want_mean) / np.maximum(1.0, np.abs(want_mean))).max()) / tol,
    )


def bocpd_evidence(seed: int, kappa0: float, offset: float) -> tuple[float, int]:
    """That ``mean_precision_prior`` was read -- the largest move of
    ``p_change`` against the same stream at 1 -- and how many times
    ``run_mode`` falls, the run-length posterior finding a break."""
    x, w = bocpd_stream(seed, offset)
    out = bocpd_run(x, w, kappa0, offset)
    at_one = bocpd_run(x, w, 1.0, offset)
    moved = float(np.abs(out["p_change"].to_numpy() - at_one["p_change"].to_numpy()).max())
    mode = out["run_mode"].to_numpy()[w > 0]
    return moved, int((np.diff(mode) < -5).sum())


class TestBocpdIsThePackagesAtEveryPrecisionPrior:
    """``bocpd`` at ``mean_precision_prior`` 0.3 and 5, beside the 1 the
    package's case ran, and with ``degrees_of_freedom_prior`` left at its
    default: a default other than 3 fails every row. Under a weight column
    of one constant, with zero-weight rows that the package does not read."""

    @pytest.mark.parametrize("kappa0", [0.3, 5.0])
    def test_one_seed(self, kappa0):
        moved, breaks = bocpd_evidence(0, kappa0, 0.0)
        print(f"kappa0={kappa0}: p_change moved {moved:.3f} from kappa0=1, {breaks} breaks found")
        assert moved > 1e-3 and breaks >= 2
        ratio = bocpd_errors(0, kappa0, 0.0)
        assert ratio <= 1.0, ratio

    @pytest.mark.oracle_grid(reason="50 seeds x 4 cases against the package's recursion")
    def test_fifty_seeds(self):
        # Measured 2026-10-25 over SEEDS and every case (aarch64, macOS): the
        # worst 0.166 of the tolerance (seed 30, kappa0 5 at 1e6), the 95th
        # percentile 0.110; every run_mode the package's; the least move of
        # p_change from kappa0 = 1 was 0.047, and every stream found a break.
        rows = [
            (bocpd_errors(seed, *case), *bocpd_evidence(seed, *case), seed, case)
            for seed in SEEDS
            for case in BOCPD_CASES
        ]
        worst = max(rows)
        ratios = np.array([r[0] for r in rows])
        print(
            f"bocpd: worst error over its tolerance {worst[0]:.3f} (seed {worst[3]}, {worst[4]}), "
            f"95th percentile {np.quantile(ratios, 0.95):.3f}"
        )
        # Round-off: held at about twice the measured 0.166 of the tolerance.
        ledger.record("bocpd_priors", "error_over_tolerance", ratios, floor=0.35)
        moved = min(r[1] for r in rows)
        breaks = min(r[2] for r in rows)
        print(f"bocpd: least move from kappa0=1 {moved:.3f}, fewest breaks found {breaks}")
        assert moved > 1e-3 and breaks >= 1
        assert worst[0] <= 1.0, worst


# --- corrchange's monitor quantile against scipy -----------------------------------------


def monitor_crit(alpha: float, k: int = 2, alpha_adjust: str = "none") -> float:
    """The critical value a ``monitor`` span of 8 rows reports where it closes."""
    rng = np.random.default_rng(0)
    features = [f"x{i}" for i in range(k)]
    x = rng.standard_normal((8, k))
    spec = po.spec.corrchange(
        "c", features=features, window_rows=8, alpha=alpha, alpha_adjust=alpha_adjust
    )
    frame = pl.DataFrame({f: x[:, i] for i, f in enumerate(features)})
    return po.ModelBank([spec]).fit_predict(frame)["c"].struct.field("crit")[7]


#: The levels from the end at which the model leaves the bisection on ``1 -
#: alpha`` for a series that keeps the tail (``KOLMOGOROV_TAIL`` in
#: ``corrchange.rs``).
KOLMOGOROV_TAIL = 1e-2


def crit_over_bound(alpha: float, k: int = 2, alpha_adjust: str = "none") -> float:
    """The critical value against ``kstwobign.isf`` of the pair's level, over
    its bound: at most 1 passes.

    Between 0.01 and 0.99 the model solves ``F(x) = 1 - alpha`` by bisection
    on the series ``F(x) = 1 - 2 sum (-1)^(k-1) exp(-2 k^2 x^2)``. Its ``1 -
    alpha`` and its series are each within ``2u`` of the true ``F``, so its
    root is within ``4u / f(x)`` of the true quantile, ``f`` the
    distribution's density (``kstwobign.pdf``).

    Below 0.01 it solves ``S(x) = alpha`` on the survival series ``S(x) = 2
    sum (-1)^(k-1) exp(-2 k^2 x^2)``, relative error ``(4x^2 + 4)u``: the
    exponent ``2x^2`` is rounded to ``2u`` of itself, which ``exp`` turns
    into ``4x^2 u`` of the term, and ``exp``, the sum and the doubling add
    up to ``4u``. Above 0.99 it solves on the theta series ``F(x) =
    sqrt(2 pi) / x sum exp(-(2k-1)^2 pi^2 / (8x^2))``, relative error
    ``(pi^2 / (2x^2) + 5)u`` by the same count, its exponent ``pi^2 /
    (8x^2)`` rounded to ``4u``. Either error, times the tail, over ``f``.
    Each bisection ends between adjacent doubles, ``u x``. The bound adds
    ``1e-14 x`` for scipy's own quantile, which differs from a 50-digit sum
    of the same series by ``2.1e-15 x`` at ``alpha = 0.5`` and by at most
    ``1.4e-16 x`` at each other level here (measured 2026-10-25)."""
    pairs = k * (k - 1) // 2
    level = alpha / pairs if alpha_adjust == "bonferroni" else alpha
    want = kstwobign.isf(level)
    density = kstwobign.pdf(want)
    if level < KOLMOGOROV_TAIL:
        solved = (4.0 * want**2 + 4.0) * U * level / density
    elif level > 1.0 - KOLMOGOROV_TAIL:
        solved = (np.pi**2 / (2.0 * want**2) + 5.0) * U * (1.0 - level) / density
    else:
        solved = 4.0 * U / density
    bound = solved + U * want + 1e-14 * want
    return abs(monitor_crit(alpha, k, alpha_adjust) - want) / bound


#: Levels from far in the upper tail to far in the lower one.
ALPHAS = [
    1e-300,
    1e-20,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    0.01,
    0.05,
    0.1,
    0.3,
    0.5,
    0.9,
    0.99,
    0.999,
    1 - 1e-6,
]


class TestTheMonitorsQuantileIsScipysKolmogorov:
    """The ``monitor`` kind's critical value is the ``1 - alpha / npairs``
    quantile of the Kolmogorov distribution, ``sup |B|`` of a Brownian
    bridge, under Bonferroni (the default), and at ``alpha`` with
    ``alpha_adjust = "none"``. ``scipy.stats.kstwobign`` is that
    distribution. ``tests/test_corrchange.py`` held three published values
    to 1e-3."""

    def test_at_the_usual_levels_and_under_bonferroni(self):
        # 1e-12 and 1 - 1e-6 are each a tail's own series: solved on ``1 -
        # alpha`` they were 5.8e7 and 317 of the bound (task 278, kq).
        for alpha in (1e-12, 0.01, 0.05, 0.1, 1 - 1e-6):
            assert crit_over_bound(alpha) <= 1.0, alpha
        for k in (3, 5):
            assert crit_over_bound(0.05, k, "bonferroni") <= 1.0, k
        # Bonferroni took effect: three pairs' share is a larger value.
        assert monitor_crit(0.05, 3, "bonferroni") > monitor_crit(0.05) + 0.1

    @pytest.mark.extended(reason="seventeen levels and four Bonferroni shares, each a bank")
    def test_at_every_level_into_either_tail(self):
        """Measured 2026-10-25 (aarch64, macOS): the worst 0.21 of the bound,
        at ``alpha = 0.5``, where scipy's own quantile is ``2.1e-15 x`` off;
        0.045 at three pairs' share of 0.05, at most 0.014 in either tail's
        series. Solved on ``1 - alpha`` at every level, as it was before
        task 278 (kq), the critical value was no better than that
        subtraction: relative ``5.9e-7`` at ``1e-12``, ``0.097`` at
        ``1e-20``, ``3.3e-12`` at ``1 - 1e-6``; over this bound, ``5.8e7``
        at ``1e-12``, 2.2 at ``1e-4``, 1.4 at ``0.999``."""
        ratios = [(crit_over_bound(a), a) for a in ALPHAS]
        ratios += [(crit_over_bound(0.05, k, "bonferroni"), k) for k in (3, 5, 20, 64)]
        print("monitor crit over its bound:", [(f"{r:.3f}", a) for r, a in ratios])
        worst = max(ratios)
        assert worst[0] <= 1.0, worst
