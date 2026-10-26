"""Task 278 (c), class 11: truth recovery for clustering and regimes.

This library's own ``kmeans``, ``micro``, ``hmm``, ``bocpd``,
``corrchange``, ``deco`` and ``rcov``, measured against the truth that made
a stream, or against a third-party oracle whose answer is taken as given
(scikit-learn's batch ``KMeans`` inertia, hmmlearn's Baum-Welch means).
Nothing here scores those libraries themselves (the user, 2026-10-25: "we do
not want to retest existing libraries just our part"): a public set is an
input to our models.

Every case is judged by its worst seed or ordering, over 50 of them in the
``oracle_grid`` tier, which ``.github/workflows/oracles.yml`` runs on every
push and no local gate takes; the essentials hold one cheap case per model. Every bound is
the measurement with headroom, 2026-10-25, macOS arm64, quoted beside it.

`docs/REGIMES.md` measures `hmm`, `corrchange`, `bocpd`, `deco` and `rcov`
on 8 to 20 seeds and reports means and medians; the tests here take the
same designs (``scripts/regime_experiments.py``'s streams) to 50 seeds and
assert the tails.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import ledger
import polars_online as po
from polars_online import sim
from public_benchmarks import franti_or_skip, tcpd_or_skip

TIER = "mixed"

INF = float("inf")
SEEDS = range(50)


def centroid_index(found: np.ndarray, truth: np.ndarray) -> int:
    """Fränti's centroid index: the centres of one set that no centre of the
    other has as its nearest, the larger of the two directions. 0 is the
    right structure; each unit is a cluster missed (and another split)."""

    def orphans(a: np.ndarray, b: np.ndarray) -> int:
        nearest = np.argmin(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), axis=1)
        return len(b) - len(set(nearest.tolist()))

    return max(orphans(found, truth), orphans(truth, found))


def inertia(points: np.ndarray, centres: np.ndarray) -> float:
    return float(((points[:, None] - centres[None]) ** 2).sum(-1).min(axis=1).sum())


def fit_kmeans(
    points: np.ndarray, k: int, order: np.ndarray | None = None, **kw
) -> tuple[np.ndarray, np.ndarray]:
    """The centres after one pass over ``points`` in ``order`` (as given when
    ``None``), and each point's cluster read from them in the points' own
    order (``predict``: the state as it stands, nothing learned)."""
    names = [f"x{j}" for j in range(points.shape[1])]
    df = pl.DataFrame({n: points[:, j] for j, n in enumerate(names)})
    bank = po.ModelBank([po.spec.kmeans("m", features=names, n_clusters=k, half_life=INF, **kw)])
    bank.fit(df if order is None else df[order])
    centres = bank.coef()["coef"].to_numpy().reshape(k, points.shape[1])
    return centres, bank.predict(df).unnest("m")["cluster"].to_numpy()


# --- kmeans and micro on synthetic blobs ----------------------------------------

BLOB_K, BLOB_P, BLOB_N = 5, 3, 3000


def blobs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five isotropic unit Gaussians in three dimensions, centres drawn in a
    cube of side 20 at least 6 apart, 3,000 rows in a random order. The
    nearest true centre classifies every row of every seed here (the Bayes
    partition's ARI is 1.0)."""
    rng = np.random.default_rng(seed)
    while True:
        c = rng.uniform(-10, 10, (BLOB_K, BLOB_P))
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) + np.eye(BLOB_K) * 99
        if d.min() > 6:
            break
    label = rng.integers(0, BLOB_K, BLOB_N)
    return c[label] + rng.standard_normal((BLOB_N, BLOB_P)), label, c


def _blob_scores(seed: int, **kw) -> tuple[float, float]:
    x, label, truth = blobs(seed)
    centres, got = fit_kmeans(x, BLOB_K, **kw)
    d = np.sqrt(((centres[:, None] - truth[None]) ** 2).sum(-1))
    r, c = linear_sum_assignment(d)
    return adjusted_rand_score(label, got), float(d[r, c].max())


#: The ledger's inputs (tests/ledger.py): this library's fits on the grids'
#: own streams, here and beside each grid below.
ledger.guard(
    "kmeans_blobs",
    lambda: [
        fit_kmeans(blobs(s)[0], BLOB_K, standardize=sd) for s in SEEDS for sd in (False, True)
    ],
    oracles=["scikit-learn", "scipy"],
)


def test_kmeans_recovers_separated_blobs():
    """The essential case: three seeds, the metric in the features' units."""
    for seed in range(3):
        ari, err = _blob_scores(seed, standardize=False)
        assert ari > 0.99 and err < 0.3, (seed, ari, err)


@pytest.mark.oracle_grid(reason="50 seeds of 3,000 rows, twice")
def test_kmeans_recovers_separated_blobs_on_the_worst_seed():
    """In the features' own units every seed's partition is the truth's.
    Under the default ``standardize``, distances are in each feature's own
    standard deviation: isotropic blobs whose features spread unequally are
    stretched, and the stretched Voronoi cells move rows between two nearby
    blobs. Measured, worst of 50: ARI 0.948 standardized (seed 13, whose
    nearest-true-centre partition is the truth's) and 0.9975 raw; the worst
    centre's error 0.31 standardized and 0.16 raw, against unit noise.
    Bounds: raw 0.99 and 0.3; standardized 0.9 and 0.5."""
    raw = np.array([_blob_scores(s, standardize=False) for s in SEEDS])
    std = np.array([_blob_scores(s) for s in SEEDS])
    print(f"raw: ARI worst {raw[:, 0].min():.4f}, centre err worst {raw[:, 1].max():.3f}")
    print(
        f"standardized: ARI worst {std[:, 0].min():.4f} (seed {std[:, 0].argmin()}), "
        f"centre err worst {std[:, 1].max():.3f}"
    )
    ledger.record("kmeans_blobs", "raw_ari", raw[:, 0], ideal=1.0)
    ledger.record("kmeans_blobs", "raw_centre_error", raw[:, 1])
    ledger.record("kmeans_blobs", "standardized_ari", std[:, 0], ideal=1.0)
    ledger.record("kmeans_blobs", "standardized_centre_error", std[:, 1])
    assert raw[:, 0].min() > 0.99 and raw[:, 1].max() < 0.3
    assert std[:, 0].min() > 0.9 and std[:, 1].max() < 0.5


def micro_blobs(seed: int) -> pl.DataFrame:
    """``micro`` at ``eps = 0.1`` over one seed's blobs: each row's cluster
    and whether it is an outlier, read after the pass."""
    names = [f"x{j}" for j in range(BLOB_P)]
    x, _, _ = blobs(seed)
    df = pl.DataFrame({n: x[:, j] for j, n in enumerate(names)})
    bank = po.ModelBank([po.spec.micro("m", features=names, eps=0.1, half_life=INF)])
    bank.fit(df)
    return bank.predict(df).unnest("m").select("cluster", "outlier")


ledger.guard("micro_blobs", lambda: [micro_blobs(s) for s in SEEDS], oracles=["scikit-learn"])


@pytest.mark.oracle_grid(reason="50 seeds of 3,000 rows")
def test_micro_finds_separated_blobs_on_the_worst_seed():
    """``micro`` at ``eps = 0.1``, with no fixed count: the clusters it links,
    the rows no summary takes counted as a cluster of their own. Measured,
    worst of 50 (seed 7): ARI 0.776, from 4 to 7 clusters linked, at most
    8.2% outliers; 5% of seeds below 0.889. Bounds 0.7, 0.85 at the 5th
    percentile, 12% outliers."""
    scores = []
    for s in SEEDS:
        _, label, _ = blobs(s)
        out = micro_blobs(s)
        scores.append(
            (
                adjusted_rand_score(label, out["cluster"].fill_null(-1).to_numpy()),
                out["outlier"].mean(),
            )
        )
    a = np.array(scores)
    print(
        f"micro: ARI worst {a[:, 0].min():.4f} p05 {np.quantile(a[:, 0], 0.05):.4f}, "
        f"outliers {a[:, 1].max():.3f}"
    )
    ledger.record("micro_blobs", "ari", a[:, 0], ideal=1.0)
    ledger.record("micro_blobs", "outlier_share", a[:, 1])
    assert a[:, 0].min() > 0.7
    assert np.quantile(a[:, 0], 0.05) > 0.85
    assert a[:, 1].max() < 0.12


# --- kmeans and micro on Fränti's benchmark -------------------------------------

#: Per set, the bounds on the worst of 50 orderings: ARI at least, the
#: centroid index at most, and the inertia over the batch oracle's at most.
#: Measured over the 50, 2026-10-25 (the comment: the worst ARI, the worst
#: centroid index and in how many orders it is above 0, the worst inertia
#: ratio and its median; before greedy k-means++ and Lloyd to its fixed
#: point, task 278, D7 and F3, after the arrow), bounds with headroom.
FRANTI = {
    # 0.886 -> 0.986; CI 1 in 10 -> 0 in 0; 2.19x -> 1.001x, median 1.001
    "s1": (0.95, 0, 1.05),
    # 0.854 -> 0.936; CI 1 in 13 -> 0 in 0; 1.29x -> 1.001x, median 1.001
    "s2": (0.9, 0, 1.05),
    # 0.664 -> 0.672; CI 1 in 19 -> 1 in 1; 1.14x -> 1.12x, median 1.003
    # (heavy overlap: the truth's labels are not the best partition's)
    "s3": (0.63, 1, 1.25),
    # 0.557 -> 0.594; CI 2 in 22 -> 1 in 6; 1.18x -> 1.09x, median 1.011
    "s4": (0.55, 1, 1.2),
    # 0.822 -> 0.865; CI 2 in 39 -> 1 in 10; 1.56x -> 1.30x, median 1.009
    "a1": (0.83, 1, 1.45),
    # 0.860 -> 0.929; CI 3 in 50 -> 1 in 30; 1.51x -> 1.24x, median 1.101
    "a2": (0.89, 1, 1.4),
    # 0.871 -> 0.913; CI 4 in 50 -> 3 in 44; 1.71x -> 1.50x, median 1.104
    "a3": (0.87, 3, 1.65),
}


def franti_order(points: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(len(points))


#: The first three orderings of each set: the fits are the slow part here
#: (A3's 7,500 rows at 50 clusters), and seven sets of three cover every
#: path the fifty take.
ledger.guard(
    "kmeans_franti",
    lambda: {
        name: [
            fit_kmeans(p, len(t), franti_order(p, s))
            for p, _, t in [franti_or_skip(name)]
            for s in range(3)
        ]
        for name in sorted(FRANTI)
    },
    oracles=["scikit-learn"],
    dataset="franti:" + ",".join(sorted(FRANTI)),
)


@pytest.mark.oracle_grid(reason="the network, and 50 orderings of each of seven sets")
@pytest.mark.parametrize("name", sorted(FRANTI))
def test_kmeans_on_frantis_benchmark_by_the_worst_ordering(name):
    """One pass in each of 50 shuffled orders, ``n_clusters`` the truth's.
    The oracle for the inertia is scikit-learn's batch ``KMeans``, taken as
    given: the lower of its fit from ten greedy k-means++ starts and its fit
    from the published centres, Lloyd to convergence in both. Ten starts
    miss A3's optimum, which the fit from the centres finds 0.933 times
    lower; on the other sets the two agree within 1e-4.

    ``kmeans`` seeds from its first ``warm_rows`` (500) rows: ten greedy
    k-means++ starts, each run to its fixed point. Every S1 and S2 order
    finds every cluster. The A-sets' 20 to 50 clusters leave 10 to 25
    buffered rows a cluster, and 10, 30 and 44 of their 50 orders still end
    in a local minimum, a cluster missed and another split; the median
    order is 1.009, 1.101 and 1.104 times the optimum. With every row
    buffered the medians are 1.006, 1.001 and 1.065, and 2, 14 and 30
    orders still miss a cluster. When the seeding ran ten plain k-means++
    starts with ten Lloyd iterations each, the A-sets' median order was
    1.26 to 1.35 times the optimum, and 10 of S1's 50 orders missed a
    cluster (2026-10-25)."""
    points, label, truth = franti_or_skip(name)
    k = len(truth)
    min_ari, max_ci, max_ratio = FRANTI[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = min(
            KMeans(k, n_init=10, random_state=0).fit(points).inertia_,
            KMeans(k, init=truth, n_init=1).fit(points).inertia_,
        )
    rows = []
    for s in SEEDS:
        centres, got = fit_kmeans(points, k, franti_order(points, s))
        rows.append(
            (
                adjusted_rand_score(label, got),
                centroid_index(centres, truth),
                inertia(points, centres) / best,
            )
        )
    a = np.array(rows)
    print(
        f"{name}: ARI worst {a[:, 0].min():.4f} p05 {np.quantile(a[:, 0], 0.05):.4f}; "
        f"CI max {a[:, 1].max():.0f}, >0 in {(a[:, 1] > 0).sum()}; "
        f"inertia/oracle worst {a[:, 2].max():.3f} "
        f"median {np.median(a[:, 2]):.3f}"
    )
    ledger.record("kmeans_franti", f"{name}_ari", a[:, 0], ideal=1.0)
    ledger.record("kmeans_franti", f"{name}_centroid_index", a[:, 1], exact=True)
    ledger.record("kmeans_franti", f"{name}_inertia_ratio", a[:, 2], ideal=1.0)
    assert a[:, 0].min() >= min_ari
    assert a[:, 1].max() <= max_ci
    assert a[:, 2].max() <= max_ratio
    assert a[:, 2].min() >= 1 - 1e-9, "below the batch optimum: the oracle is wrong"


def micro_s1(points: np.ndarray, seed: int) -> pl.DataFrame:
    """``micro`` at ``eps = 0.02`` after one pass over S1 in one ordering:
    each point's cluster and whether it is an outlier, in the set's order."""
    order = franti_order(points, seed)
    df = pl.DataFrame({"x0": points[order, 0], "x1": points[order, 1]})
    bank = po.ModelBank([po.spec.micro("m", features=["x0", "x1"], eps=0.02, half_life=INF)])
    bank.fit(df)
    out = bank.predict(pl.DataFrame({"x0": points[:, 0], "x1": points[:, 1]})).unnest("m")
    return out.select("cluster", "outlier")


ledger.guard(
    "micro_franti",
    lambda: [micro_s1(franti_or_skip("s1")[0], s) for s in range(10)],
    oracles=["scikit-learn"],
    dataset="franti:s1",
)


@pytest.mark.oracle_grid(reason="the network, and 50 orderings")
def test_micro_on_s1_by_the_worst_ordering():
    """``micro`` on S1, the least overlapped S-set, at ``eps = 0.02``: its
    15 Gaussians touch, so a coarser ``eps`` chains them (``eps = 0.07``
    links 5 to 12 clusters, worst ARI 0.24) and a finer one leaves most rows
    outliers. Measured over 50 orders: ARI worst 0.937, at most 44.5% of
    the rows outliers (17 to 21 clusters linked over 10). Bounds 0.9 and
    50%."""
    points, label, _ = franti_or_skip("s1")
    rows = []
    for s in SEEDS:
        out = micro_s1(points, s)
        rows.append(
            (
                adjusted_rand_score(label, out["cluster"].fill_null(-1).to_numpy()),
                out["outlier"].mean(),
            )
        )
    a = np.array(rows)
    print(f"micro s1: ARI worst {a[:, 0].min():.4f}, outliers worst {a[:, 1].max():.3f}")
    ledger.record("micro_franti", "s1_ari", a[:, 0], ideal=1.0)
    ledger.record("micro_franti", "s1_outlier_share", a[:, 1])
    assert a[:, 0].min() > 0.9
    assert a[:, 1].max() < 0.5


# --- hmm --------------------------------------------------------------------------

HMM_K = 3
#: ``hmm``'s default ``warm_rows`` at three states.
HMM_WARM_ROWS = 1_000
HMM_MEANS = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
HMM_STAY = 0.98


def regimes(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Three regimes that differ in location, three noise standard deviations
    apart, unit noise, each lasting 50 rows on average (stay 0.98)."""
    rng = np.random.default_rng(seed)
    trans = np.full((HMM_K, HMM_K), (1 - HMM_STAY) / (HMM_K - 1)) + np.eye(HMM_K) * (
        HMM_STAY - (1 - HMM_STAY) / (HMM_K - 1)
    )
    state = np.empty(n, dtype=int)
    state[0] = 0
    u = rng.random(n)
    cum = trans.cumsum(axis=1)
    for i in range(1, n):
        state[i] = int(np.searchsorted(cum[state[i - 1]], u[i]))
    return HMM_MEANS[state] + rng.standard_normal((n, 2)), state


def _matched(a: np.ndarray, b: np.ndarray) -> float:
    return min(float(np.abs(a[list(p)] - b).max()) for p in itertools.permutations(range(len(b))))


def hmm_fit(x: np.ndarray, **kw) -> tuple[np.ndarray, np.ndarray]:
    """``hmm``'s filtered state on each row (-1 where null) and its learned
    means after the pass."""
    spec = po.spec.hmm(
        "h", features=["x0", "x1"], n_components=HMM_K, precision_prior=0.01, half_life=INF, **kw
    )
    bank = po.ModelBank([spec])
    out = bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})).unnest("h")
    means = bank.coef("h")["coef"].to_numpy().reshape(HMM_K, 2)
    return out["state"].fill_null(-1).to_numpy(), means


def hmm_scores(seed: int, n: int, oracle: bool = True, **kw) -> tuple[float, float, float]:
    """The share of rows after the warm-up in their true state (the better
    labelling of the states), and the largest error of a learned mean
    against the truth and, when ``oracle``, against hmmlearn's Baum-Welch on
    the same rows (NaN without)."""
    from hmmlearn.hmm import GaussianHMM

    x, state = regimes(seed, n)
    got, means = hmm_fit(x, **kw)
    after = slice(kw.get("warm_rows", HMM_WARM_ROWS), n)
    share = max(
        float(np.mean(np.array(p)[got[after]] == state[after]))
        for p in itertools.permutations(range(HMM_K))
    )
    if not oracle:
        return share, _matched(means, HMM_MEANS), float("nan")
    # The oracle: Baum-Welch to convergence from k-means centres, which on
    # these well-separated regimes is the maximum likelihood.
    bw = GaussianHMM(
        HMM_K, covariance_type="full", n_iter=500, tol=1e-10, init_params="", params="stmc"
    )
    bw.startprob_ = np.full(HMM_K, 1 / HMM_K)
    bw.transmat_ = np.full((HMM_K, HMM_K), 0.05) + np.eye(HMM_K) * 0.85
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bw.means_ = KMeans(HMM_K, n_init=10, random_state=0).fit(x).cluster_centers_
        bw.covars_ = np.stack([np.eye(2)] * HMM_K)
        bw.fit(x)
    return share, _matched(means, HMM_MEANS), _matched(means, bw.means_)


def test_hmm_recovers_location_regimes():
    """The essential case: one seed, seeded from 2,000 rows."""
    share, truth, _ = hmm_scores(0, 5_000, oracle=False, warm_rows=2_000)
    assert share > 0.93 and truth < 0.25, (share, truth)


#: Ten seeds of the fifty: 10 ms a seed.
ledger.guard(
    "hmm_regimes",
    lambda: [hmm_fit(regimes(s, 5_000)[0], warm_rows=2_000) for s in range(10)],
    oracles=["hmmlearn", "scikit-learn"],
)


@pytest.mark.oracle_grid(reason="50 seeds of 5,000 rows, each with a Baum-Welch fit")
def test_hmm_recovers_location_regimes_on_the_worst_seed():
    """Seeded from 2,000 rows, which hold every regime. Measured over 50
    seeds: the worst seed's share in the true state 0.945 (seed 34), the
    worst means 0.25 from the truth and 0.21 from Baum-Welch's; the filter
    with the true parameters can do little better at three noise standard
    deviations. Bounds 0.92, 0.35, 0.3."""
    a = np.array([hmm_scores(s, 5_000, warm_rows=2_000) for s in SEEDS])
    print(
        f"hmm: share worst {a[:, 0].min():.4f} (seed {a[:, 0].argmin()}), means vs truth worst "
        f"{a[:, 1].max():.3f}, vs Baum-Welch worst {a[:, 2].max():.3f}"
    )
    ledger.record("hmm_regimes", "share_in_true_state", a[:, 0], ideal=1.0)
    ledger.record("hmm_regimes", "means_from_truth", a[:, 1])
    ledger.record("hmm_regimes", "means_from_baum_welch", a[:, 2])
    assert a[:, 0].min() > 0.92
    assert a[:, 1].max() < 0.35
    assert a[:, 2].max() < 0.3


def test_hmm_at_its_default_warm_up_recovers_sticky_regimes():
    """Seeded from the default ``warm_rows``, 1,000 rows, which regimes
    lasting 50 rows on average fill about twenty times over. Measured over 20
    seeds at 5,000 rows: the worst share in the true state 0.951 (seed 0;
    0.947 over 50 seeds, and 0.963 over 20 at 50,000 rows). The default was
    50 rows, which such regimes fill with one or two of them: the seeding
    placed two states inside one regime and online EM never moved them out,
    the worst share 0.53 over the same 20 seeds (0.46 over 50, 0.52 at
    50,000 rows; task 278 (c), D9). A buffer of 500 is not enough either: a
    regime with a few percent of its rows is cheaper, by inertia, to share
    than to own (0.81 over 50 seeds). Bound 0.9, and the old default still
    below 0.7, which shows the case is the one that failed."""
    worst = min(hmm_scores(s, 5_000, oracle=False)[0] for s in range(20))
    old = min(hmm_scores(s, 5_000, oracle=False, warm_rows=50)[0] for s in range(20))
    print(f"hmm: worst share {worst:.4f} at the default, {old:.4f} at 50 warm-up rows")
    assert old < 0.7, old
    assert worst > 0.9, worst


# --- bocpd ------------------------------------------------------------------------


def bocpd_out(x: np.ndarray, **kw) -> pl.DataFrame:
    """``bocpd``'s every field on each row of ``x``."""
    spec = po.spec.bocpd("b", features=["x"], **kw)
    return po.ModelBank([spec]).fit_predict(pl.DataFrame({"x": x})).unnest("b")


def bocpd_starts(x: np.ndarray, **kw) -> tuple[list[int], list[int | None]]:
    """The rows on which ``bocpd``'s dated start of the current run jumps
    forward by more than 5 rows (an alarm), and each row's dated start,
    ``t - run_mode``."""
    out = bocpd_out(x, **kw)
    began = [None if m is None else i - m for i, m in enumerate(out["run_mode"].to_list())]
    alarms = [
        i
        for i in range(1, len(began))
        if began[i] is not None and began[i - 1] is not None and began[i] - began[i - 1] > 5
    ]
    return alarms, began


def bocpd_streams(seed: int, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """``bocpd_break``'s two streams: the shift of ``delta`` at row 500 of
    1,000, and the quiet 1,000 drawn after it."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(1_000)
    x[500:] += delta
    return x, rng.standard_normal(1_000)


def bocpd_break(seed: int, delta: float) -> tuple[int | None, int | None, int]:
    """A mean shift of ``delta`` noise standard deviations at row 500 of
    1,000, then a quiet stream of 1,000: the delay to the first alarm at or
    after the break, the error of the start the run-length posterior dates
    on the stream's last row, with all the evidence, and the alarms before
    the break and on the quiet stream. The most likely run length moves
    between hypotheses on its way there: at 3 sd, seed 48's first alarm
    (row 501) dates the run from row 473 and the next row from 500; at 1
    sd, seed 30's dates it from 501, then from row 0 for 25 rows, then from
    487."""
    x, quiet = bocpd_streams(seed, delta)
    alarms, began = bocpd_starts(x, hazard=250.0)
    false = sum(1 for a in alarms if a < 500) + len(bocpd_starts(quiet, hazard=250.0)[0])
    after = [a for a in alarms if a >= 500]
    if not after:
        return None, None, false
    return after[0] - 500, abs(began[-1] - 500), false


def test_bocpd_finds_a_large_mean_shift():
    delay, dated, _ = bocpd_break(0, 3.0)
    assert delay is not None and delay <= 8 and dated <= 2, (delay, dated)


#: Per shift, in noise standard deviations: the worst delay over 50 seeds,
#: and a bound on the dating error at a quantile of them. Measured over the
#: 50 (2026-10-25): delay worst 63, 10 and 6 at 1, 2 and 3 sd; dated
#: within 2 and 1 rows on every seed at 2 and 3 sd; at 1 sd the 95th
#: percentile 16, and the worst 495, seed 31, whose last rows hold an alarm
#: of their own (rows 918 to 999 raise six), so its run is dated from row
#: 995; 140 alarms in 75,000 quiet rows (1.9 per 1,000) at each.
BOCPD_BOUNDS = {1.0: (100, 30, 0.95), 2.0: (20, 4, 1.0), 3.0: (12, 3, 1.0)}


#: Every field on three seeds' streams at each shift, where the grid reads
#: the dated starts on fifty: a run of 1,000 rows costs 17 ms, the posterior
#: over run lengths growing with the run, so fifty seeds would take five
#: seconds of the gate; any change to the recursion moves the posterior's
#: fields on the first seeds as on the rest.
ledger.guard(
    "bocpd_shift",
    lambda: [
        bocpd_out(x, hazard=250.0)
        for d in sorted(BOCPD_BOUNDS)
        for s in range(3)
        for x in bocpd_streams(s, d)
    ],
)


@pytest.mark.oracle_grid(reason="50 seeds of 2,000 rows at three shifts")
@pytest.mark.parametrize("delta", sorted(BOCPD_BOUNDS))
def test_bocpd_finds_every_mean_shift_and_dates_it(delta):
    rows = [bocpd_break(s, delta) for s in SEEDS]
    missed = [s for s, r in zip(SEEDS, rows, strict=True) if r[0] is None]
    delays = [r[0] for r in rows if r[0] is not None]
    dated = [r[1] for r in rows if r[1] is not None]
    false = sum(r[2] for r in rows)
    print(
        f"bocpd {delta} sd: delay worst {max(delays)} p95 {np.quantile(delays, 0.95):.0f}, "
        f"dated worst {max(dated)} p95 {np.quantile(dated, 0.95):.0f}, "
        f"false alarms {false} in {len(rows) * 1500} quiet rows"
    )
    tag = f"{delta:g}sd"
    ledger.record("bocpd_shift", f"delay_{tag}", delays, exact=True)
    ledger.record("bocpd_shift", f"dating_error_{tag}", dated, exact=True)
    ledger.record("bocpd_shift", f"false_alarms_{tag}", false, exact=True)
    max_delay, max_dated, q = BOCPD_BOUNDS[delta]
    assert not missed, missed
    assert max(delays) <= max_delay
    assert np.quantile(dated, q) <= max_dated
    # 140 in 75,000 measured: 1.9 per 1,000 quiet rows; bound 4.
    assert false <= 4 * len(rows) * 1.5, false


#: Per TCPD series, the F1 of ``bocpd`` at its default ``hazard`` (250) on
#: the raw series, TCPDBench's measure at its margin of 5 rows. Measured
#: (2026-10-25), identical with the series scaled by 1e-3 and 1e3: well_log
#: 0.787, quality_control_1 0.571, quality_control_2 1.000,
#: quality_control_3 0.667, nile 1.000, run_log 0.990. One series, one
#: ordering, deterministic: each bound is the measurement less 0.05.
TCPD = {
    "well_log": 0.73,
    "quality_control_1": 0.52,
    "quality_control_2": 0.95,
    "quality_control_3": 0.61,
    "nile": 0.95,
    "run_log": 0.94,
}


def tcpd_f1(predicted: list[int], annotations: dict[str, list[int]], margin: int = 5) -> float:
    """TCPDBench's F1 (van den Burg and Williams 2020, eqs. 3-4): precision
    against the union of the annotators' change points, recall averaged over
    the annotators, each point matched at most once within ``margin``, and
    row 0 a change point of every set."""

    def hits(truth: list[int], pred: list[int]) -> int:
        used: set[int] = set()
        n = 0
        for t in truth:
            near = [p for p in pred if abs(p - t) <= margin and p not in used]
            if near:
                used.add(min(near, key=lambda p: abs(p - t)))
                n += 1
        return n

    pred = sorted({0, *predicted})
    sets = [sorted({0, *a}) for a in annotations.values()]
    union = sorted(set().union(*sets))
    precision = hits(union, pred) / len(pred)
    recall = float(np.mean([hits(a, pred) / len(a) for a in sets]))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


#: Every field on each series. The bounds sit 0.05 under the measured F1,
#: so a change could lose most of that without failing: the ledger holds
#: each series' F1 as measured.
ledger.guard(
    "bocpd_tcpd",
    lambda: [bocpd_out(tcpd_or_skip(name)[0], hazard=250.0) for name in sorted(TCPD)],
    dataset="tcpd:" + ",".join(sorted(TCPD)),
)


@pytest.mark.oracle_grid(reason="the network: TCPD's annotated series, each F1 held to the ledger")
@pytest.mark.parametrize("name", sorted(TCPD))
def test_bocpd_on_tcpd_series(name):
    x, annotations = tcpd_or_skip(name)
    alarms, began = bocpd_starts(x, hazard=250.0)
    found = sorted({began[a] for a in alarms if began[a] is not None})
    score = tcpd_f1(found, annotations)
    print(f"{name}: F1 {score:.3f}, found {found}")
    ledger.record("bocpd_tcpd", f"{name}_f1", score, ideal=1.0)
    assert score >= TCPD[name], (score, found)


# --- corrchange --------------------------------------------------------------------


def corr_stream(seed: int, broken: bool) -> pl.DataFrame:
    """`docs/REGIMES.md` section 5's design: three series, 2,000 rows, the
    innovation correlation 0.2, or 0.2 then 0.7 from row 1,000."""
    out = sim.regimes(
        3,
        states=[0.2, 0.7] if broken else [0.2, 0.2],
        transition=[[0.0, 1.0], [1.0, 0.0]],
        n_blocks=2,
        rows_per_block=1000,
        durations=[1, 1],
        seed=seed,
    )
    rets = out["rows"].select(pl.col("x_1", "x_2", "x_3").diff()).drop_nulls()
    return rets.rename({"x_1": "x0", "x_2": "x1", "x_3": "x2"})


WINDOW = po.spec.corrchange(
    "w",
    features=["x0", "x1", "x2"],
    kind="window",
    window_rows=100,
    n_perm=100,
    permute_every_rows=100,
    alpha=0.05,
    seed=0,
)


def _flags(df: pl.DataFrame) -> np.ndarray:
    return po.ModelBank([WINDOW]).fit_predict(df).unnest("w")["flag"].fill_null(False).to_numpy()


def test_corrchange_flags_a_correlation_break():
    flags = np.flatnonzero(_flags(corr_stream(1_000, True)))
    after = flags[flags >= 999]
    assert len(after) and after[0] - 999 <= 100, after[:3]


#: Every field on ten seeds, broken and quiet, where the grid reads the
#: flags on fifty: the permutation test costs 10 ms a stream.
ledger.guard(
    "corrchange_break",
    lambda: [
        po.ModelBank([WINDOW]).fit_predict(corr_stream(1_000 + s, broken)).unnest("w")
        for s in range(10)
        for broken in (False, True)
    ],
)


@pytest.mark.oracle_grid(reason="50 seeds of 2,000 rows, broken and quiet")
def test_corrchange_flags_every_break_and_holds_its_size():
    """The window test at ``alpha = 0.05``. Measured over 50 seeds: no break
    missed; delay worst 80 rows, 95th percentile 62, median 33 (a window of
    100); on the quiet streams the share of rows flagged 0.053 on average
    (the nominal 0.05: a sliding window flags in runs) and 0.113 on the
    worst seed. Bounds: delay 120, mean size 0.07, worst seed 0.2."""
    delays, quiet = [], []
    for s in SEEDS:
        quiet.append(_flags(corr_stream(1_000 + s, False)).mean())
        flags = np.flatnonzero(_flags(corr_stream(1_000 + s, True)))
        after = flags[flags >= 999]
        delays.append(after[0] - 999 if len(after) else None)
    missed = [s for s, d in zip(SEEDS, delays, strict=True) if d is None]
    print(
        f"corrchange: delay worst {max(d for d in delays if d is not None)}, "
        f"quiet mean {np.mean(quiet):.4f}, worst {max(quiet):.4f}"
    )
    ledger.record("corrchange_break", "delay", [d for d in delays if d is not None], exact=True)
    ledger.record("corrchange_break", "quiet_share_flagged", quiet)
    ledger.record("corrchange_break", "quiet_share_flagged_mean", np.mean(quiet))
    assert not missed, missed
    assert max(delays) <= 120
    assert np.mean(quiet) < 0.07 and max(quiet) < 0.2


# --- deco ---------------------------------------------------------------------------

DECO_M = 5


def deco_rho(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``deco``'s ``rho`` on each row of one seed's stream, five series whose
    equicorrelation alternates 0.2 / 0.6 every 1,200 rows, and each row's
    true state."""
    feats = [f"x{i}" for i in range(DECO_M)]
    out = sim.regimes(
        DECO_M,
        states=[0.2, 0.6],
        transition=[[0.0, 1.0], [1.0, 0.0]],
        durations=[3, 3],
        n_blocks=12,
        rows_per_block=400,
        seed=2_000 + seed,
    )
    rets = out["rows"].select(pl.col([f"x_{i + 1}" for i in range(DECO_M)]).diff()).drop_nulls()
    df = rets.rename({f"x_{i + 1}": f"x{i}" for i in range(DECO_M)})
    want = np.asarray(out["truth_rows"]["state"].to_list()[1:], dtype=int)
    rho = po.ModelBank([po.spec.deco("d", features=feats, half_life=100.0)]).fit_predict(df)
    return rho.unnest("d")["rho"].to_numpy().astype(float), want


#: Ten seeds of the fifty: 13 ms a seed.
ledger.guard("deco_level", lambda: [deco_rho(s) for s in range(10)])


@pytest.mark.parametrize(
    "seeds",
    [
        pytest.param(range(3), id="3 seeds"),
        pytest.param(
            SEEDS, id="50 seeds", marks=pytest.mark.oracle_grid(reason="50 seeds of 4,800 rows")
        ),
    ],
)
def test_deco_level_settles_low_and_moves_at_its_half_life(seeds):
    """`docs/REGIMES.md` section 7's design at 50 seeds, half-life 100: five
    series whose equicorrelation alternates 0.2 / 0.6 every 1,200 rows.
    ``rho`` is an EW mean of a row estimate biased low (Engle and Kelly), so
    the truth is the band it settles in and the speed it moves at, an EW
    mean's step response reaching the midpoint in one half-life. Measured:
    settled at a true 0.2 between 0.086 and 0.167 (mean 0.129), at 0.6
    between 0.321 and 0.430 (0.376), every segment of every seed; rows to
    the midpoint median 95, 95th percentile 147, worst 206, none missed.
    Bounds: the bands 0.05-0.2 and 0.28-0.48, the median within 70-130 rows
    and the worst 300. The essentials run three of the seeds."""
    low, high, delays = [], [], []
    for s in seeds:
        rho, want = deco_rho(s)
        cuts = np.flatnonzero(np.diff(want)) + 1
        segments = list(zip(np.r_[0, cuts], np.r_[cuts, len(want)], strict=True))
        settled = {0: [], 1: []}
        for a, b in segments:
            settled[int(want[a])].append(float(np.nanmean(rho[b - 200 : b])))
        low += settled[0]
        high += settled[1]
        mid = (np.mean(settled[0]) + np.mean(settled[1])) / 2
        for a, b in segments[1:]:
            past = np.flatnonzero(rho[a:b] > mid if want[a] == 1 else rho[a:b] < mid)
            delays.append(int(past[0]) if len(past) else None)
    assert None not in delays, delays
    print(
        f"deco: low {min(low):.3f}-{max(low):.3f}, high {min(high):.3f}-{max(high):.3f}, "
        f"delay median {np.median(delays):.0f} worst {max(delays)}"
    )
    if len(seeds) == len(SEEDS):
        ledger.record("deco_level", "settled_at_0_2", low, ideal=0.2)
        ledger.record("deco_level", "settled_at_0_6", high, ideal=0.6)
        ledger.record("deco_level", "rows_to_midpoint", delays, exact=True)
        ledger.record("deco_level", "rows_to_midpoint_median", np.median(delays))
    assert min(low) > 0.05 and max(low) < 0.2
    assert min(high) > 0.28 and max(high) < 0.48
    assert 70 <= np.median(delays) <= 130 and max(delays) <= 300


# --- rcov ---------------------------------------------------------------------------


def rcov_blocks(seed: int) -> tuple[pl.DataFrame, dict[int, float], dict[int, int]]:
    """``rcov``'s closed blocks on one seed's stream of two series, 20 blocks
    of 2,000 rows, each block's correlation drawn from 0.3 and 0.8: the
    closed groups, each block's true correlation and its rows."""
    out = sim.regimes(
        2,
        states=[0.3, 0.8],
        transition=[[0.5, 0.5], [0.5, 0.5]],
        n_blocks=20,
        rows_per_block=2_000,
        seed=3_000 + seed,
    )
    truth = {
        int(b): c[1]
        for b, c in zip(out["truth_blocks"]["block"], out["truth_blocks"]["corr"], strict=True)
    }
    rows = out["rows"].with_columns(block=out["truth_rows"]["block"])
    rets = (
        rows.with_columns(pl.col("x_1", "x_2").diff().over("block"))
        .drop_nulls(["x_1", "x_2"])
        .select(pl.col("x_1").alias("x0"), pl.col("x_2").alias("x1"), "block")
    )
    sizes = {int(b): n for b, n in rets.group_by("block").len().iter_rows()}
    bank = po.ModelBank(
        [
            po.spec.rcov(
                "r", features=["x0", "x1"], group="block", group_close="monotone", kind="plain"
            )
        ]
    )
    bank.fit_predict(rets)
    return bank.closed_groups(), truth, sizes


#: Every closed block's every field on five seeds, where the grid reads the
#: correlations on fifty: a seed's 40,000 rows cost 90 ms.
ledger.guard("rcov_plain", lambda: [rcov_blocks(s)[0] for s in range(5)])


@pytest.mark.oracle_grid(reason="50 seeds of 20 blocks of 2,000 rows")
def test_rcov_plain_matches_each_blocks_correlation():
    """`docs/REGIMES.md` section 8's clean condition at 50 seeds: two series,
    20 blocks of 2,000 rows, each block's correlation drawn from 0.3 and
    0.8. Each closed block's ``plain`` realized correlation against its
    truth, in units of the sample correlation's standard error ``(1 - rho^2)
    / sqrt(n - 1)``. Measured over 950 blocks: bias +0.0007, the worst error
    0.057 in correlation and 3.20 standard errors, where the largest of 950
    normals sits near 3.3. Bounds: |bias| 0.005, the worst standardized
    error 4.5."""
    errors, z = [], []
    for s in SEEDS:
        closed, truth, sizes = rcov_blocks(s)
        for b, rc in zip(closed["group"].to_list(), closed["rcorr"].to_list(), strict=True):
            rho = truth[int(b)]
            errors.append(rc[1] - rho)
            z.append((rc[1] - rho) / ((1 - rho**2) / np.sqrt(sizes[int(b)] - 1)))
    e, z = np.array(errors), np.abs(np.array(z))
    print(
        f"rcov: {len(e)} blocks, bias {e.mean():+.4f}, worst {np.abs(e).max():.4f}, "
        f"worst z {z.max():.2f}"
    )
    ledger.record("rcov_plain", "error", e)
    ledger.record("rcov_plain", "bias", e.mean())
    ledger.record("rcov_plain", "standardized_error", z)
    assert len(e) == 50 * 19
    assert abs(e.mean()) < 0.005
    assert z.max() < 4.5
