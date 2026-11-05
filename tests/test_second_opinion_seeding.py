"""How ``hmm`` seeds its states (``init=``, ``seed=``, ``warm_rows=``), each
rule against its definition or a third party.

``hmm`` and ``kmeans`` seed through the same function over the first
``warm_rows`` learned rows. ``hmm`` then reads the buffer once through the
seeds it chose: each buffered row joins the state of its nearest seed at
its weight, so after the ``warm_rows``-th row each state holds the weighted
mean and weight of the rows nearest its seed. That is what is observable,
and it is a function of the seeds alone; the tests compute it from seeds
chosen by each rule's definition, or by a third party, and compare.

- ``"first"``: the first ``k`` distinct rows.
- ``"farthest"``: Gonzalez's (1985) farthest-first traversal from the first
  row, each next seed the row farthest from those chosen (first among
  equals).
- ``"k-means++"``: Arthur and Vassilvitskii's (2007) greedy variant, the
  first seed a row drawn by weight, each later one the best by potential
  of ``2 + floor(ln k)`` rows drawn with probability ``w D^2``, which is
  scikit-learn's ``kmeans_plusplus`` with ``sample_weight``. Its random
  stream is not numpy's, so the seeds can only be compared in
  distribution: over 500 seeds each, the potential ``sum w min D^2`` of
  the states the seeds lead to, against the same map of scikit-learn's
  seeds, by a two-sample Kolmogorov-Smirnov test.
- ``"lloyd"``, the default: the best by inertia of ten k-means++ starts,
  each run by weighted Lloyd iterations to a fixed point, which is
  scikit-learn's ``KMeans(n_init=10, algorithm="lloyd")`` with
  ``sample_weight``. At a fixed point each centre is the weighted mean of
  its rows, so the states after the warm-up are the centres themselves.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from scipy.stats import ks_2samp, kstwo
from sklearn.cluster import KMeans, kmeans_plusplus

import ledger
import polars_online as po

TIER = "mixed"

SEEDS = range(50)

#: Means and weights against the definition, relative to ``1 + |x|``.
#: Measured 2026-11-04 over ``SEEDS`` (aarch64, macOS): the worst 3.6e-15.
#: Both sides sum at most 200 weighted rows in a different order.
TOL = 1e-12


def blobs(seed: int, n: int = 200, k: int = 4, d: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``k`` blobs of spread 0.6 at random centres in a box of side 12, and
    row weights from 0.3 to 2."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-6.0, 6.0, (k, d))
    x = centres[rng.integers(k, size=n)] + rng.normal(0.0, 0.6, (n, d))
    return x, rng.uniform(0.3, 2.0, n)


def states_after(
    x: np.ndarray, w: np.ndarray, k: int, init: str, seed: int = 0, rows: int | None = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The states' means and weights after ``rows`` rows (all by default),
    with ``warm_rows`` the whole stream, and whether they are seeded."""
    n, d = x.shape
    rows = n if rows is None else rows
    features = [f"x{i}" for i in range(d)]
    spec = po.spec.hmm(
        "h",
        features=features,
        n_components=k,
        precision_prior=1e-2,
        half_life=float("inf"),
        warm_rows=n,
        init=init,
        seed=seed,
        weight="w",
    )
    frame = pl.DataFrame({f: x[:rows, i] for i, f in enumerate(features)}).with_columns(w=w[:rows])
    bank = po.ModelBank([spec])
    bank.fit_predict(frame)
    model = json.loads(bank.write_json())["states"][0][0][1]["models"][0]["model"]["Hmm"]
    means = np.array([s["m"] for s in model["states"]])
    return means, np.array([s["w_sum"] for s in model["states"]]), model["seeded"]


def through(x: np.ndarray, w: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row to its nearest seed (first among equals): each seed's
    weighted mean of its rows (numpy's ``average``) and their weight."""
    nearest = ((x[:, None, :] - seeds[None]) ** 2).sum(axis=-1).argmin(axis=1)
    means = np.array(
        [np.average(x[nearest == j], axis=0, weights=w[nearest == j]) for j in range(len(seeds))]
    )
    return means, np.array([w[nearest == j].sum() for j in range(len(seeds))])


def first(x: np.ndarray, k: int) -> np.ndarray:
    """The first ``k`` distinct rows."""
    out: list[np.ndarray] = []
    for row in x:
        if not any((row == c).all() for c in out):
            out.append(row)
        if len(out) == k:
            break
    return np.array(out)


def farthest(x: np.ndarray, k: int) -> np.ndarray:
    """Gonzalez's traversal from the first row."""
    out = [x[0]]
    dist = ((x - x[0]) ** 2).sum(axis=1)
    while len(out) < k:
        i = int(np.argmax(dist))
        out.append(x[i])
        dist = np.minimum(dist, ((x - x[i]) ** 2).sum(axis=1))
    return np.array(out)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a - b) / (1.0 + np.abs(b))).max())


ledger.guard(
    "seeding_kmeanspp",
    lambda: [states_after(*blobs(1, 300, 8), 6, "k-means++", seed=s)[0] for s in range(50)],
    oracles=["scikit-learn"],
)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("init", "rule"), [("first", first), ("farthest", farthest)])
def test_the_deterministic_rules_are_their_definitions(init, rule, seed):
    """``"first"`` and ``"farthest"``, read through the buffer, are their
    definitions' seeds read through it; nothing is seeded a row before
    ``warm_rows``."""
    x, w = blobs(seed)
    k = 4
    *_, early = states_after(x, w, k, init, rows=len(x) - 1)
    means, weights, seeded = states_after(x, w, k, init)
    want_means, want_weights = through(x, w, rule(x, k))
    assert not early and seeded
    assert rel(means, want_means) <= TOL
    assert rel(weights, want_weights) <= TOL


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_default_rule_is_scikit_learns_kmeans(seed):
    """``"lloyd"``, the default: the states after the warm-up are
    scikit-learn's ``KMeans(n_init=10)`` centres on the same weighted rows,
    up to the order of the states (both reach the same optimum on these
    separated blobs: the inertia printed is the same to rounding)."""
    x, w = blobs(seed)
    k = 4
    means, _, _ = states_after(x, w, k, "lloyd")
    fit = KMeans(k, n_init=10, init="k-means++", algorithm="lloyd", tol=0.0, random_state=seed)
    centres = fit.fit(x, sample_weight=w).cluster_centers_
    order = [int(np.argmin(((centres - m) ** 2).sum(axis=1))) for m in means]
    inertia = float((w * ((x[:, None] - means[None]) ** 2).sum(-1).min(1)).sum())
    print(f"inertia {inertia:.12g} against scikit-learn's {fit.inertia_:.12g}")
    assert sorted(order) == list(range(k))
    assert rel(means, centres[order]) <= TOL


#: Seeds each side draws in the k-means++ comparison.
DRAWS = 500

#: The level at which the two-sample test is read. The test is
#: deterministic (fixed seeds), so this is the chance that a correct rule
#: would have failed it on the one sample drawn.
LEVEL = 1e-3


def kmeanspp_potentials(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The potential of the states each of ``DRAWS`` k-means++ seedings
    leads to, the model's (``seed`` 0 to ``DRAWS - 1``) and
    scikit-learn's (``random_state`` likewise), on overlapping blobs where
    the seeding varies."""
    x, w = blobs(1 + seed, n=300, k=8)
    x = x / 1.5  # eight blobs, spread 0.4 in a box of side 8: they overlap
    k = 6

    def potential(means: np.ndarray) -> float:
        # The same partition gives the two sides' potentials within about
        # 1e-13 of each other (their means are summed in a different
        # order). A potential takes a few values, each with much of the
        # mass, so unrounded those last bits would split every value in two,
        # one side's copy always just above the other's, and the test would
        # read a shift that is not there. Rounded to ten significant digits,
        # the same partition is one value.
        return float(f"{(w * ((x[:, None] - means[None]) ** 2).sum(-1).min(1)).sum():.10g}")

    ours = np.array(
        [potential(states_after(x, w, k, "k-means++", seed=s)[0]) for s in range(DRAWS)]
    )
    theirs = np.array(
        [
            potential(through(x, w, kmeans_plusplus(x, k, sample_weight=w, random_state=s)[0])[0])
            for s in range(DRAWS)
        ]
    )
    return ours, theirs


def ks_over_critical(seed: int) -> tuple[float, float]:
    """The two-sample statistic over its critical value at ``LEVEL`` (at
    most 1 passes; the two-sample ``D`` of equal samples of 500 is read
    against ``kstwo(250)``, the one-sample law at the samples' effective
    size ``n m / (n + m)``, whose quantile is conservative where the
    potentials take few values),
    and the potentials' coefficient of variation, which shows the seeding
    varies."""
    ours, theirs = kmeanspp_potentials(seed)
    stat = ks_2samp(ours, theirs).statistic
    critical = kstwo.isf(LEVEL, DRAWS // 2)
    return float(stat / critical), float(ours.std() / ours.mean())


class TestKmeansPlusPlusIsScikitLearnsInDistribution:
    def test_one_stream(self):
        ratio, spread = ks_over_critical(0)
        print(f"KS over its critical value {ratio:.3f}, potentials' variation {spread:.3f}")
        assert spread > 0.02
        assert ratio <= 1.0

    @pytest.mark.oracle_grid(reason="10 streams x 500 seedings a side")
    def test_ten_streams(self):
        rows = [ks_over_critical(seed) for seed in range(10)]
        ratios = [r[0] for r in rows]
        print(f"KS over critical: worst {max(ratios):.3f}")
        ledger.record("seeding_kmeanspp", "ks_over_critical", ratios)
        assert min(r[1] for r in rows) > 0.02
        assert max(ratios) <= 1.0
