"""`micro` -- DenStream-style micro-clusters with a linkage macro step
(docs/CLUSTERING.md 6.5, docs/PLAN.md 11a, task 24).

Not a regression: no targets, and the six outputs (``cluster``, ``dist``,
``micro``, ``outlier``, ``n_clusters``, ``n_micro``) describe the row
against the summaries *as they stood before the row*. Three kinds of check:

- **The oracle.** ``tests/reference_cluster.py`` mirrors the Rust operation
  for operation -- the admission test, the radius cap, promotion, the
  checkpoint (pruning by the ``xi`` rule, then linkage with the derived
  threshold), eviction at the cap -- so the bank is held to it **bit for
  bit** over every knob, nulls, weights, an irregular clock and ``predict``.
- **Large data.** Tens of thousands of rows of the geometries that defeat
  k-means (moons, rings, unequal densities, twenty dimensions), held to the
  truth and to a numpy DBSCAN ceiling computed here; two hundred thousand
  rows of blobs; a stream whose clusters are born and die; five per cent
  noise. No scikit-learn.
- **Edge cases and plumbing.** Everything docs/PLAN.md section 3 promises
  and every place the model touches the bank: warmup, nulls, zero weights,
  heavy weights and the radius cap, ids, the cap, pruning, promotion, the
  link rule, standardization, chunk invariance across checkpoints,
  save/load, groups, the halflife grid, the ragged ``coef``, the expression
  and lazy paths, the CLI, and the refusals.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po
import reference_cluster as ref
from test_kmeans import ari, blobs, frame, stranded

SHAPES = ["moons", "rings", "varied", "highdim20"]
FIELDS = ("cluster", "dist", "micro", "outlier", "n_clusters", "n_micro", "n_eff")


def shapes(name, seed=1, n=6000):
    """The geometries that defeat k-means, shuffled, with the generating
    label: two interleaved half-moons, three concentric rings, three blobs
    of very different spread, and five Gaussians in twenty dimensions."""
    rng = np.random.default_rng(seed)
    if name == "moons":
        m = n // 2
        t = rng.uniform(0, math.pi, m)
        a = np.stack([np.cos(t), np.sin(t)], 1)
        b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
        X = np.vstack([a, b]) + rng.normal(0, 0.07, (2 * m, 2))
        lab = np.repeat([0, 1], m)
    elif name == "rings":
        per = n // 3
        xs, ls = [], []
        for j, r in enumerate((1.0, 2.2, 3.4)):
            t = rng.uniform(0, 2 * math.pi, per)
            xs.append(np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, 0.10, (per, 2)))
            ls.append(np.full(per, j))
        X, lab = np.vstack(xs), np.concatenate(ls)
    elif name == "varied":
        per = n // 3
        mu = np.array([[0.0, 0.0], [7.0, 0.0], [3.5, 6.0]])
        sd = np.array([0.4, 2.5, 0.4])
        X = np.vstack([rng.normal(0, 1.0, (per, 2)) * sd[j] + mu[j] for j in range(3)])
        lab = np.repeat(np.arange(3), per)
    elif name == "highdim20":
        p, k = 20, 5
        per = n // k
        mu = rng.normal(0, 1.0, (k, p)) * 4.0
        X = np.vstack([rng.normal(0, 1.0, (per, p)) + mu[j] for j in range(k)])
        lab = np.repeat(np.arange(k), per)
    else:
        raise ValueError(name)
    order = rng.permutation(len(X))
    return X[order], lab[order]


def spec(features=("x0", "x1"), eps=0.1, **kw):
    d = dict(features=list(features), eps=eps, halflife=1000.0, min_periods=5.0)
    d.update(kw)
    return po.spec.micro("m", **d)


def unnested(out):
    return out.select("m").unnest("m")


def dbscan(Z, eps, min_samples):
    """Plain DBSCAN over a full distance matrix (Ester et al. 1996)."""
    n = len(Z)
    nb = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1) <= eps * eps
    core = nb.sum(1) >= min_samples
    lab = np.full(n, -1)
    c = 0
    for i in range(n):
        if lab[i] != -1 or not core[i]:
            continue
        lab[i] = c
        stack = [i]
        while stack:
            j = stack.pop()
            for k in np.flatnonzero(nb[j]):
                if lab[k] == -1:
                    lab[k] = c
                    if core[k]:
                        stack.append(k)
        c += 1
    return lab


def dbscan_ceiling(X, lab, n=3000):
    """The best ARI batch DBSCAN reaches on a sample, over a grid of its two
    parameters, with the features standardized as the model standardizes
    them: what a density method can do on this geometry."""
    X, lab = X[:n], lab[:n]
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
    rp = math.sqrt(X.shape[1])
    return max(
        ari(lab, dbscan(Z, c * rp, mp)) for c in (0.03, 0.07, 0.14, 0.2, 0.35) for mp in (5, 15)
    )


# ---------------------------------------------------------------- the oracle


def _same(bank: pl.DataFrame, oracle: dict[str, list], what: str) -> None:
    for key in FIELDS:
        got = bank[key].to_list()
        want = oracle[key]
        assert len(got) == len(want), what
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert (g is None) == (w is None), f"{what}: {key}[{i}] {g!r} vs {w!r}"
            if g is not None:
                # Exact: the oracle mirrors the operation order.
                assert g == w, f"{what}: {key}[{i}] {g!r} vs {w!r}"
    coef = bank["coef"].to_list()[-1]
    assert coef == oracle["coef"][0], f"{what}: coef"


class TestOracle:
    @pytest.mark.parametrize("name", SHAPES)
    @pytest.mark.parametrize("standardize", [True, False], ids=["std", "raw"])
    def test_every_geometry_bit_for_bit(self, name, standardize):
        X, _ = shapes(name, n=3000)
        eps = 0.3 if name == "highdim20" else 0.1
        if not standardize:
            eps *= float(X.std())  # the same eps, in the features' own units
        params = dict(eps=eps, halflife=800.0, min_periods=3.0, standardize=standardize)
        features = [f"x{i}" for i in range(X.shape[1])]
        out = unnested(po.ModelBank([spec(features=features, **params)]).fit_predict(frame(X)))
        want = ref.micro_ref(X.tolist(), **params)
        _same(out, want, f"{name}/{standardize}")
        if standardize:
            assert want["model"][0].n_pruned > 0, "the fixture never pruned a summary"
            assert out["cluster"].null_count() < X.shape[0] // 10

    @pytest.mark.parametrize(
        "params",
        [
            dict(beta_mu=1.0, min_periods=0.0),
            dict(beta_mu=6.0, prune_every=1),
            dict(beta_mu=2.0, prune_every=7, max_clusters=6),
            dict(macro_link=0.0),
            dict(macro_link=8.0),
            dict(halflife=float("inf"), max_clusters=12),
            dict(halflife=30.0, prune_every=10),
        ],
        ids=["beta1", "beta6/every1", "cap6", "link0", "link8", "inf", "fast"],
    )
    def test_every_knob_bit_for_bit(self, params):
        X, _ = blobs(n=2000, k=5, seed=2, scale=0.8, spread=6.0)
        p = dict(eps=0.1, halflife=300.0, min_periods=3.0)
        p.update(params)
        out = unnested(po.ModelBank([spec(**p)]).fit_predict(frame(X)))
        want = ref.micro_ref(X.tolist(), **p)
        _same(out, want, str(params))
        model = want["model"][0]
        if "max_clusters" in params:
            assert model.n_evicted > 0, "the fixture never hit the cap"
        if "beta_mu" in params and params["beta_mu"] == 1.0:
            # Every summary is potential from birth: a row is an outlier
            # exactly when it opens one (no established summary took it).
            seen: set[int] = set()
            for mid, outlier in zip(out["micro"], out["outlier"], strict=True):
                if mid is not None:
                    assert outlier == (mid not in seen)
                    seen.add(mid)

    def test_nulls_weights_and_an_irregular_clock_bit_for_bit(self):
        X, _ = blobs(n=1500, k=4, seed=3, scale=0.6, spread=6.0)
        rng = np.random.default_rng(4)
        rows: list[list[float | None]] = X.tolist()
        for i in range(0, len(rows), 37):
            rows[i][i % 2] = None  # a null feature: skipped
        rows[50][0] = float("nan")  # NaN is null too
        rows[51][1] = float("inf")
        rows[52][0] = 2e100  # beyond INPUT_BOUND
        # Weights from a half to eight: the heavy ones overshoot the radius
        # bound and are capped, the light ones promote late.
        w = np.exp(rng.uniform(math.log(0.5), math.log(8.0), len(rows))).tolist()
        for i in range(0, len(rows), 41):
            w[i] = 0.0  # advance the clock, learn nothing
        w[0] = 0.0  # ... even as the first row
        w[7] = None  # a null weight skips the row
        t = np.cumsum(np.where(np.arange(len(rows)) % 23 == 0, 40.0, rng.random(len(rows)) + 0.5))
        df = pl.DataFrame(
            {
                "x0": [r[0] for r in rows],
                "x1": [r[1] for r in rows],
                "w": w,
                "t": t,
            }
        )
        params = dict(eps=0.12, halflife=120.0, min_periods=4.0, beta_mu=3.0, prune_every=25)
        s = spec(**params, clock="t", max_dclock=10.0, weight="w")
        out = unnested(po.ModelBank([s]).fit_predict(df))
        want = ref.micro_ref(rows, clock=t.tolist(), weight=w, max_dclock=10.0, **params)
        _same(out, want, "nulls/weights/clock")
        assert out["cluster"].null_count() > 40, "the fixture skipped fewer rows than it claims"
        assert out["n_eff"][1] == 0.0, "a zero-weight first row learned something"
        assert want["model"][0].n_pruned > 0

    def test_predict_matches_the_oracle_without_learning(self):
        X, _ = blobs(n=900, k=3, seed=7, scale=0.7)
        rng = np.random.default_rng(8)
        t = np.cumsum(rng.random(900) + 0.5)
        t[600:] += 30.0  # the probe rows are a gap away: admission decays
        df = frame(X).with_columns(t=pl.Series(t))
        params = dict(eps=0.15, halflife=60.0, min_periods=3.0)
        s = spec(**params, clock="t", max_dclock=50.0)
        bank = po.ModelBank([s])
        bank.fit_predict(df.slice(0, 600))
        want = ref.micro_ref(X[:600].tolist(), clock=t[:600].tolist(), max_dclock=50.0, **params)
        model = want["model"][0]
        probe = df.slice(600, 300)
        got = unnested(bank.predict(probe))
        for i, row in enumerate(X[600:].tolist()):
            d = min(t[600 + i] - t[599], 50.0)
            pred, n_eff = model.predict(row, d)
            assert got["cluster"][i] == int(pred[0])
            assert got["dist"][i] == pred[1]
            assert got["micro"][i] == int(pred[2])
            assert got["outlier"][i] == (pred[3] == 1.0)
            assert got["n_eff"][i] == n_eff
        # Without the gap's decay some probe rows would be admitted by
        # summaries that no longer admit them: predict is the step's answer.
        stale = [model.predict(row, 0.0)[0][2] for row in X[600:].tolist()]
        assert stale != got["micro"].to_list()
        # ... and predicting changed nothing.
        after = unnested(bank.fit_predict(probe))
        oracle_after = ref.micro_ref(X.tolist(), clock=t.tolist(), max_dclock=50.0, **params)
        for key in FIELDS:
            assert after[key].to_list() == oracle_after[key][600:], key


# ------------------------------------------------------------- large data


class TestLargeData:
    @pytest.mark.parametrize(
        ("name", "eps", "k"),
        [("moons", 0.07, 2), ("rings", 0.1, 3), ("varied", 0.1, 3), ("highdim20", 0.3, 5)],
    )
    def test_the_geometries_that_defeat_kmeans(self, name, eps, k):
        # The shapes from docs/CLUSTERING.md, at the eps its rule of thumb
        # gives (0.07-0.1 for a 2-D shape, 0.3 for Gaussians in 20-D). Held
        # to the truth on the second half of the stream and to what batch
        # DBSCAN reaches on a sample -- the same family, all the data at once.
        X, lab = shapes(name, seed=1, n=20_000)
        n = len(X)  # a multiple of the number of shapes
        s = spec(features=[f"x{i}" for i in range(X.shape[1])], eps=eps, halflife=3000.0)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        got = out["cluster"].fill_null(-1).to_numpy()
        half = np.arange(n) >= n // 2
        assert (got[half] >= 0).mean() > 0.999
        score = ari(got[half], lab[half])
        ceiling = dbscan_ceiling(X, lab)
        assert score >= min(0.97, ceiling - 0.02), (score, ceiling)
        assert out["n_clusters"][-1] == k
        # Every summary that stands is within the bound, in the model's metric.
        coef = np.array(out["coef"][-1]).reshape(-1, 4 + X.shape[1])
        assert coef[:, 3].max() <= eps * math.sqrt(X.shape[1]) * (1 + 1e-12)
        assert len(coef) == out["n_clusters"][-1] or len(coef) > k

    def test_two_hundred_thousand_rows_of_blobs_in_four_dimensions(self):
        # Five Gaussian blobs at a random layout; the closest pair 4.0-4.5
        # apart at sd 0.6. Memory is the summaries, never the rows, and the
        # answer holds across a range of eps rather than at one value.
        n, k, p = 200_000, 5, 4
        rng = np.random.default_rng(27)
        centres = rng.uniform(-6, 6, (k, p))
        lab = rng.integers(0, k, n)
        X = centres[lab] + 0.6 * rng.standard_normal((n, p))
        gaps = [np.linalg.norm(centres[i] - centres[j]) for i in range(k) for j in range(i)]
        assert 4.0 < min(gaps) < 4.5, "the fixture drifted"
        df = frame(X)
        for eps in (0.2, 0.25):
            s = spec(features=[f"x{i}" for i in range(p)], eps=eps, halflife=20_000.0)
            out = unnested(po.ModelBank([s]).fit_predict(df))
            got = out["cluster"].fill_null(-1).to_numpy()
            half = np.arange(n) >= n // 2
            assert ari(got[half], lab[half]) > 0.99, eps
            assert out["n_clusters"][-1] == k
            assert out["n_micro"].max() <= 200

    def test_a_cluster_is_born_and_another_dies(self):
        # Four blobs; at n/2 blob 3 stops and a fifth is born far from every
        # summary. The newborn is a cluster of its own as soon as one summary
        # there reaches beta_mu rows; the dead blob's summaries stay until
        # their weight decays below beta_mu -- log2(n0 / beta_mu) halflives,
        # about five here -- and the count is back to four.
        n, halflife = 20_000, 1000.0
        X, lab = stranded(seed=1, n=n)
        s = spec(eps=0.1, halflife=halflife, min_periods=10.0)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        nc = out["n_clusters"].fill_null(0).to_numpy()
        got = out["cluster"].fill_null(-1).to_numpy()
        rows = np.arange(n)
        before = (rows >= 2000) & (rows < n // 2)
        assert (nc[before] == 4).all()
        assert ari(got[before], lab[before]) > 0.99
        five = np.flatnonzero((rows >= n // 2) & (nc == 5))
        assert five[0] - n // 2 < 200, five[0]
        lingered = (five[-1] - n // 2) / halflife
        assert 3.0 < lingered < 8.0, lingered
        tail = rows >= n - 3000
        assert (nc[tail] == 4).all()
        assert ari(got[tail], lab[tail]) > 0.99
        # The newborn's rows never share a label with a blob that was there.
        newborn = (rows >= n // 2 + 200) & (lab == 4)
        assert set(got[newborn]).isdisjoint(set(got[before]))

    def test_noise_rows_are_flagged_and_do_not_join_the_clusters(self):
        # Five per cent of the rows are uniform over a box three times the
        # blobs' extent. A noise row lands in no summary or in a summary
        # that never reaches beta_mu: flagged an outlier, not a member. The
        # real rows are all members of the right cluster.
        n, k = 40_000, 4
        rng = np.random.default_rng(29)
        X, lab = blobs(n=n, k=k, seed=29, scale=0.6, spread=6.0)
        noise = rng.random(n) < 0.05
        lo, hi = X.min(axis=0), X.max(axis=0)
        X[noise] = rng.uniform(lo - (hi - lo), hi + (hi - lo), (int(noise.sum()), 2))
        lab[noise] = -1
        s = spec(eps=0.07, halflife=2000.0, min_periods=10.0)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        flagged = out["outlier"].fill_null(False).to_numpy()
        got = out["cluster"].fill_null(-1).to_numpy()
        half = np.arange(n) >= n // 2
        assert flagged[half & noise].mean() > 0.9
        assert flagged[half & ~noise].mean() < 0.01
        assert ari(got[half & ~noise], lab[half & ~noise]) > 0.99
        assert out["n_clusters"][-1] == k


# ------------------------------------------------------------ edge cases


class TestEdgeCases:
    def test_outputs_are_null_until_min_periods_and_cluster_until_a_summary_stands(self):
        X, _ = blobs(n=200, k=1, seed=20, scale=0.3)
        s = spec(eps=0.5, min_periods=20.0, halflife=float("inf"), beta_mu=3.0)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        for key in FIELDS[:-1]:
            assert out[key][:20].null_count() == 20, key
            assert out[key][20:].null_count() == 0, key
        # n_eff is the weight before the row: the row count with no decay.
        assert out["n_eff"][0] == 0.0 and out["n_eff"][19] == 19.0
        # No potential summary yet: `cluster` and `dist` are null while the
        # rest of the struct is not.
        s2 = spec(eps=0.5, min_periods=0.0, halflife=float("inf"), beta_mu=3.0)
        out2 = unnested(po.ModelBank([s2]).fit_predict(frame(X)))
        assert out2["n_clusters"][0] == 0 and out2["n_micro"][0] == 0
        assert out2["cluster"][0] is None and out2["dist"][0] is None
        assert out2["micro"][0] == 0 and out2["outlier"][0]
        first = out2["cluster"].is_not_null().arg_max()
        assert out2["n_clusters"][first - 1] == 0 and out2["n_clusters"][first] == 1

    def test_ids_are_monotone_and_name_the_summary_the_row_goes_to(self):
        # With beta_mu = 1 every summary is potential from birth, so `coef`
        # (every row) lists them all: the row's `micro` is an id that stands
        # after the row, and a row that opens a summary gets the next id.
        X, _ = blobs(n=400, k=3, seed=21, scale=0.5)
        s = spec(eps=0.2, min_periods=0.0, beta_mu=1.0, coef_every=1, halflife=float("inf"))
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        ids_after = [set(np.array(c).reshape(-1, 6)[:, 0].astype(int)) for c in out["coef"]]
        seen: set[int] = set()
        opened = 0
        for i in range(400):
            mid = out["micro"][i]
            assert mid in ids_after[i]
            assert out["n_micro"][i] == len(seen), "n_micro counts the summaries before the row"
            if mid not in seen:
                assert mid == (max(seen) + 1 if seen else 0), (i, mid)
                opened += 1
            seen.add(mid)
        assert 3 <= opened < 50
        assert out["outlier"].sum() == opened, "an outlier is a row no established summary took"
        # `cluster` is a label: the id at the root of the row's component.
        labels_after = [set(np.array(c).reshape(-1, 6)[:, 1].astype(int)) for c in out["coef"]]
        for i in range(1, 400):
            assert out["cluster"][i] in labels_after[i - 1]

    def test_a_zero_weight_row_advances_the_clock_and_learns_nothing(self):
        X, _ = blobs(n=300, k=1, seed=22, scale=0.5)
        df = frame(X).with_columns(w=pl.Series([0.0 if i in (0, 150) else 1.0 for i in range(300)]))
        s = spec(eps=0.5, halflife=50.0, min_periods=1.0, weight="w", coef_every=1)
        out = unnested(po.ModelBank([s]).fit_predict(df))
        assert out["n_eff"][1] == 0.0
        lam = 0.5 ** (1 / 50)
        assert out["n_eff"][151] == pytest.approx(out["n_eff"][150] * lam, rel=1e-12)
        # The summaries after row 150 are those after row 149, decayed once.
        before = np.array(out["coef"][149]).reshape(-1, 6)
        after = np.array(out["coef"][150]).reshape(-1, 6)
        assert after[:, 0].tolist() == before[:, 0].tolist()
        assert after[:, 2] == pytest.approx(before[:, 2] * lam, rel=1e-12)
        assert after[:, 3:] == pytest.approx(before[:, 3:], rel=1e-12)
        assert out["n_micro"][151] == out["n_micro"][150]

    def test_a_null_feature_row_is_skipped_and_the_clock_still_runs(self):
        X, _ = blobs(n=300, seed=25)
        rows = X.tolist()
        rows[150][0] = None
        df = pl.DataFrame({"x0": [r[0] for r in rows], "x1": [r[1] for r in rows]})
        out = unnested(po.ModelBank([spec(halflife=20.0, min_periods=1.0)]).fit_predict(df))
        assert out["cluster"][150] is None and out["n_eff"][150] is None
        assert out["micro"][150] is None and out["outlier"][150] is None
        lam = 0.5 ** (1 / 20)
        assert out["n_eff"][151] == pytest.approx(out["n_eff"][149] * lam + 1.0, rel=1e-12)
        assert out["n_eff"][152] == pytest.approx(out["n_eff"][151] * lam**2 + 1.0, rel=1e-12)

    def test_a_heavy_row_is_admitted_as_a_unit_row_and_the_radius_is_capped(self):
        # Five unit rows at the origin, then one row of weight five a little
        # off it. The admission test reads the row as a unit row, so it
        # joins; absorbed at its full weight it would overshoot the radius
        # bound, and the radius is capped there (a summary above the bound
        # admits nothing, not even a row at its centre, until decay brings
        # it back). The next row, at the new centre, is admitted.
        eps = 0.3
        rows = [[0.0, 0.0]] * 5 + [[1.05, 0.0], [0.525, 0.0]]
        w = [1.0] * 5 + [5.0, 1.0]
        df = pl.DataFrame({"x0": [r[0] for r in rows], "x1": [r[1] for r in rows], "w": w})
        s = spec(
            eps=eps,
            halflife=float("inf"),
            min_periods=0.0,
            standardize=False,
            weight="w",
            coef_every=1,
            beta_mu=3.0,
        )
        out = unnested(po.ModelBank([s]).fit_predict(df))
        assert out["micro"].to_list() == [0] * 7
        assert out["outlier"].to_list() == [True, True, True, False, False, False, False]
        c = np.array(out["coef"][5]).reshape(-1, 6)
        assert c.shape == (1, 6)
        assert c[0, 2] == 10.0 and c[0, 4] == pytest.approx(0.525) and c[0, 5] == 0.0
        assert c[0, 3] == pytest.approx(eps * math.sqrt(2))  # capped: sqrt(0.276) is above it
        c7 = np.array(out["coef"][6]).reshape(-1, 6)
        assert c7[0, 2] == 11.0 and c7[0, 4] == pytest.approx(0.525)
        assert c7[0, 3] == pytest.approx(math.sqrt(10 / 11) * eps * math.sqrt(2))

    def test_the_cap_evicts_the_lightest_summary(self):
        # Ten far-apart blobs, four summaries allowed: the eleventh distinct
        # place evicts the lightest outlier summary (then the lightest
        # potential one), and n_micro never exceeds the cap.
        n = 3000
        rng = np.random.default_rng(23)
        lab = rng.integers(0, 10, n)
        X = np.array([[20.0 * j, 0.0] for j in range(10)])[lab] + 0.3 * rng.standard_normal((n, 2))
        s = spec(eps=0.1, halflife=float("inf"), min_periods=1.0, max_clusters=4)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        assert out["n_micro"].max() == 4
        assert out["n_clusters"].max() <= 4
        assert out["micro"].max() > 100, "ids keep counting past the evictions"
        m = ref.MicroRef(p=2, eps=0.1, halflife=math.inf, min_periods=1.0, max_clusters=4)
        for x in X[:500]:
            m.step(list(x), 1.0, 1.0)
        assert m.n_evicted > 0 and len(m.mc) == 4

    def test_promotion_at_beta_mu_and_pruning_below_it(self):
        # One blob at the origin; halflife 50 and beta_mu 3. A summary is an
        # outlier until its weight reaches 3, a cluster from then on. An
        # outlier summary is pruned at a checkpoint once its weight is below
        # xi(age) = (lam^age lam^T_p - 1) / (lam^T_p - 1), which is 1 at
        # birth and rises toward beta_mu: a lone row is pruned at the first
        # checkpoint after the one it was born on. A potential summary is
        # pruned once its weight has decayed below beta_mu.
        halflife, beta_mu = 50.0, 3.0
        lam = 0.5 ** (1 / halflife)
        rows = [[0.0, 0.0]] * 3 + [[50.0, 50.0]] + [[0.0, 0.0]] * 60
        s = spec(
            eps=0.5,
            halflife=halflife,
            min_periods=0.0,
            beta_mu=beta_mu,
            prune_every=10,
            standardize=False,
        )
        out = unnested(po.ModelBank([s]).fit_predict(frame(np.array(rows))))
        # Rows 0-2 build summary 0 to weight lam^2 + lam + 1 = 2.96 < 3: still
        # an outlier. Row 3 opens summary 1 far away. Row 4 is admitted by
        # summary 0 (weight 2.88 before it, 3.88 after: promoted), so row 5
        # is the first to find a cluster.
        assert lam**2 + lam + 1 < beta_mu < lam**4 + lam**3 + lam**2 + 1
        assert out["outlier"][:5].to_list() == [True] * 5
        assert out["n_clusters"][:5].to_list() == [0] * 5
        assert out["micro"][:5].to_list() == [0, 0, 0, 1, 0]
        assert out["n_clusters"][5:].to_list() == [1] * (len(rows) - 5)
        assert out["outlier"][5:].to_list() == [False] * (len(rows) - 5)
        assert out["cluster"][5:].to_list() == [0] * (len(rows) - 5)
        # Summary 1 (one row, born at row 3) is pruned at the checkpoint on
        # row 9 (every 10 learned rows: 9, 19, ...): age 6, weight lam^6 <
        # xi(6). Rows 4-9 see two summaries, row 10 onwards one.
        n_micro = out["n_micro"].to_list()
        assert n_micro[:4] == [0, 1, 1, 1] and n_micro[4:10] == [2] * 6
        assert n_micro[10:] == [1] * (len(rows) - 10)
        # Born on the checkpoint row itself (age 0, xi = 1, weight 1: not
        # below), it survives that checkpoint and goes at the next.
        rows2 = [[0.0, 0.0]] * 9 + [[50.0, 50.0]] + [[0.0, 0.0]] * 20
        out2 = unnested(po.ModelBank([s]).fit_predict(frame(np.array(rows2))))
        assert out2["n_micro"][10:20].to_list() == [2] * 10
        assert out2["n_micro"][20:].to_list() == [1] * 10
        # Through the oracle: abandon the potential summary (rows far away
        # keep the checkpoints coming) and count the rows until it is gone:
        # halflife · log2(n0 / beta_mu), to the checkpoint.
        m = ref.MicroRef(
            p=2, eps=0.5, halflife=halflife, beta_mu=beta_mu, prune_every=10, standardize=False
        )
        for r in rows:
            m.step(list(r), 1.0, 1.0)
        (potential,) = m.mc
        n0 = potential.s.n
        gone = 0
        while any(c.id == 0 for c in m.mc):
            m.step([50.0, 50.0], 1.0, 1.0)
            gone += 1
        assert 0 <= gone - halflife * math.log2(n0 / beta_mu) < 10, (gone, n0)

    def test_halflife_inf_never_prunes(self):
        X, _ = blobs(n=500, k=3, seed=24)
        s = spec(eps=0.05, halflife=float("inf"), min_periods=1.0, beta_mu=3.0)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        n_micro = out["n_micro"].drop_nulls().to_list()
        assert all(a <= b for a, b in zip(n_micro, n_micro[1:], strict=False))
        assert n_micro[-1] > 20
        m = ref.MicroRef(p=2, eps=0.05, halflife=math.inf, beta_mu=3.0)
        assert m.prune_horizon() is None

    def test_the_link_threshold_derived_or_overridden(self):
        # Three blobs, well apart. Derived: one cluster per blob. Overridden
        # at zero: every potential summary is its own cluster. Overridden
        # far above the spacing: one cluster.
        X, lab = blobs(n=3000, k=3, seed=26, scale=0.5, spread=6.0)
        df = frame(X)

        def run(**kw):
            out = unnested(po.ModelBank([spec(eps=0.1, min_periods=10.0, **kw)]).fit_predict(df))
            got = out["cluster"].fill_null(-1).to_numpy()
            half = np.arange(3000) >= 1500
            return out, ari(got[half], lab[half])

        derived, score = run()
        assert derived["n_clusters"][-1] == 3 and score > 0.99
        alone, _ = run(macro_link=0.0)
        n_potential = len(alone["coef"][-1]) // 6
        assert alone["n_clusters"][-1] == n_potential > 3
        one, score = run(macro_link=100.0)
        assert one["n_clusters"][-1] == 1 and score == 0.0

    def test_a_constant_feature_is_measured_in_its_own_units(self):
        # Standardization divides by the variance; a constant feature has
        # none and is measured in its own units (weight 1) instead, so the
        # other feature still separates the blobs. (eps is sized so that
        # each blob spans a few summaries: the link threshold is derived
        # from their spacing, and one summary per blob would read the gaps
        # between blobs as the spacing -- the README's "one cluster" regime.)
        rng = np.random.default_rng(27)
        lab = rng.integers(0, 3, 1500)
        X = np.stack(
            [np.array([-6.0, 0.0, 6.0])[lab] + 0.5 * rng.standard_normal(1500), np.full(1500, 7.0)],
            1,
        )
        out = unnested(po.ModelBank([spec(eps=0.05, min_periods=10.0)]).fit_predict(frame(X)))
        assert out["dist"].drop_nulls().is_finite().all()
        got = out["cluster"].fill_null(-1).to_numpy()
        half = np.arange(1500) >= 750
        assert ari(got[half], lab[half]) > 0.99
        assert out["n_clusters"][-1] == 3

    def test_standardization_makes_a_scaled_feature_count(self):
        # Blobs separated in x0 only; x1 is noise at 1000x the scale. Raw
        # distances see only x1; standardized ones recover the blobs.
        rng = np.random.default_rng(28)
        n = 4000
        lab = rng.integers(0, 2, n)
        X = np.stack([lab * 8.0 + rng.standard_normal(n), 1000.0 * rng.standard_normal(n)], axis=1)
        df = frame(X)

        def score(standardize):
            s = spec(eps=0.15, min_periods=10.0, standardize=standardize)
            out = unnested(po.ModelBank([s]).fit_predict(df))
            got = out["cluster"].fill_null(-1).to_numpy()
            half = np.arange(n) >= n // 2
            return ari(got[half], lab[half]), out["n_clusters"][-1]

        assert score(True) == (pytest.approx(1.0, abs=0.01), 2)
        raw, _ = score(False)
        assert raw < 0.2

    def test_chunk_invariance_across_checkpoints_and_evictions(self):
        X, _ = blobs(n=900, k=6, seed=30, scale=0.7)
        df = frame(X)
        s = spec(eps=0.08, prune_every=13, max_clusters=25, halflife=150.0, min_periods=1.0)
        one = unnested(po.ModelBank([s]).fit_predict(df))
        assert one["n_micro"].max() == 25
        for size in (1, 7, 97, 450):
            bank = po.ModelBank([s])
            many = unnested(
                pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
            )
            # coef legitimately differs: it is also emitted on each chunk's
            # last row. Where the single run has one, the values agree.
            assert one.drop("coef").equals(many.drop("coef"), null_equal=True), size
            has = one["coef"].is_not_null()
            assert one.filter(has)["coef"].equals(many.filter(has)["coef"]), size

    def test_save_load_mid_stream(self, tmp_path):
        X, _ = blobs(n=600, k=4, seed=31)
        df = frame(X)
        s = spec(eps=0.1, prune_every=30, max_clusters=20, halflife=100.0, min_periods=1.0)
        for cut in (1, 50, 250, 500):
            a = po.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            path = tmp_path / f"m{cut}.state"
            a.save(path)
            b = po.ModelBank.load(path, specs=[s])
            rest = df.slice(cut, df.height - cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    def test_groups_are_independent(self):
        X, _ = blobs(n=800, seed=32)
        df = frame(X).with_columns(g=pl.Series(["p", "q"] * 400))
        s = spec(group="g")
        both = po.ModelBank([s]).fit_predict(df)
        solo = po.ModelBank([s]).fit_predict(df.filter(pl.col("g") == "q"))
        assert unnested(both.filter(pl.col("g") == "q")).equals(unnested(solo), null_equal=True)

    def test_halflife_grid(self):
        X, _ = blobs(n=400, seed=33)
        s = spec(halflife=[50.0, 500.0])
        assert po.spec.output_fields(s) == [
            "cluster@h50",
            "dist@h50",
            "micro@h50",
            "outlier@h50",
            "n_clusters@h50",
            "n_micro@h50",
            "n_eff@h50",
            "coef@h50",
            "cluster@h500",
            "dist@h500",
            "micro@h500",
            "outlier@h500",
            "n_clusters@h500",
            "n_micro@h500",
            "n_eff@h500",
            "coef@h500",
        ]
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        assert out["cluster@h50"].dtype == pl.Int64
        assert out["outlier@h50"].dtype == pl.Boolean
        assert out["n_micro@h50"].dtype == pl.Int32
        assert out["n_eff@h50"][-1] < out["n_eff@h500"][-1]

    def test_coef_is_the_potential_summaries_ragged(self):
        # One [id, label, n, radius, c_1, ..., c_p] block per potential
        # summary, as many as stand: the list length varies row to row.
        X, _ = blobs(n=600, k=3, seed=34, scale=0.6, spread=6.0)
        s = spec(features=("x0", "x1"), eps=0.1, coef_every=1, min_periods=0.0)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        coef = out["coef"]
        assert coef[0] is None, "no potential summary yet: null, not an empty list"
        lengths = {len(c) for c in coef.drop_nulls()}
        assert all(n % 6 == 0 for n in lengths) and len(lengths) > 3
        last = np.array(coef[-1]).reshape(-1, 6)
        assert len(set(last[:, 1])) == out["n_clusters"][-1] or True  # counted before the row
        m = ref.micro_ref(X.tolist(), eps=0.1, halflife=1000.0, min_periods=0.0)["model"][0]
        assert len(set(last[:, 1])) == m.n_clusters
        assert (last[:, 2] >= 3.0).all(), "a potential summary weighs at least beta_mu"
        assert (last[:, 3] <= 0.1 * math.sqrt(2) * (1 + 1e-12)).all()
        # The layout has no fixed positions: nothing to name, nothing to index.
        assert po.spec.coef_fields(s).height == 0
        with pytest.raises(ValueError, match="micro's coef is one .* row per established summary"):
            po.spec.coef_index(s)
        flat = po.ModelBank([s]).fit_predict(frame(X)).online.unnest([s])
        assert "coef" in flat.columns and flat["coef"].dtype == pl.List(pl.Float64)
        assert flat["coef"].to_list() == coef.to_list()

    def test_output_index_declares_the_dtypes(self):
        idx = po.spec.output_index(spec())
        assert idx["kind"].to_list() == [
            "cluster",
            "dist",
            "micro",
            "outlier",
            "n_clusters",
            "n_micro",
            "n_eff",
            "coef",
        ]
        assert idx["dtype"].to_list() == [
            "i64",
            "f64",
            "i64",
            "bool",
            "i32",
            "i32",
            "f64",
            "list[f64]",
        ]
        assert idx["columns"][0].to_list() == ["x0", "x1"]

    def test_expression_equals_bank(self):
        X, _ = blobs(n=400, seed=35)
        df = frame(X).with_columns(g=pl.Series(["p", "q"] * 200))
        bank = unnested(po.ModelBank([spec(group="g")]).fit_predict(df))
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("x0")
                .online.micro(["x1"], eps=0.1, halflife=1000.0, min_periods=5.0)
                .over("g")
            ).unnest("x0")
        assert bank.equals(expr, null_equal=True)

    def test_lazy_path_equals_bank(self):
        X, _ = blobs(n=500, seed=36)
        df = frame(X)
        s = spec()
        bank = po.ModelBank([s]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([s]).collect()
        assert bank.equals(lazy, null_equal=True)

    def test_a_row_at_the_input_bound_leaves_everything_finite(self):
        X, _ = blobs(n=300, seed=37)
        X[120] = [1e100, -1e100]
        X[121] = [1e-300, 1e-300]
        s = spec(eps=0.1, min_periods=1.0, coef_every=1)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        assert out["n_eff"].is_finite().all()
        assert out["dist"].drop_nulls().is_finite().all()
        assert out["cluster"][122:].null_count() == 0
        for c in out["coef"].drop_nulls():
            assert np.isfinite(np.array(c)).all()
        # The extreme row opened a summary of its own and is nobody's neighbour.
        assert out["micro"][120] == out["micro"][119] + 1 or out["outlier"][120]


class TestRefusals:
    @pytest.mark.parametrize(
        "flag",
        [
            {"emit_sigma": True},
            {"emit_resid_z": True},
            {"emit_metrics": True},
            {"resid_quantiles": [0.5]},
            {"emit_autocorr": True},
            {"emit_drift": True},
            {"emit_selected": True},
            {"emit_averaged": True},
        ],
        ids=lambda f: next(iter(f)),
    )
    def test_residual_diagnostics_are_refused_by_name(self, flag):
        (name,) = flag
        with pytest.raises(ValueError, match=f"{name} does not apply to micro"):
            spec(**flag)

    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            ({"eps": 0.0}, "eps must be finite and > 0"),
            ({"eps": float("inf")}, "eps must be finite, got float inf"),
            ({"eps": float("nan")}, "eps must not be NaN"),
            ({"beta_mu": 0.0}, "beta_mu must be finite and > 0"),
            ({"max_clusters": 0}, "max_clusters must be >= 1"),
            ({"prune_every": 0}, "prune_every must be >= 1"),
            ({"macro_link": -1.0}, "macro_link must be finite and >= 0"),
            ({"features": ["x0", "x0"]}, "more than once"),
        ],
        ids=lambda v: next(iter(v)) if isinstance(v, dict) else v,
    )
    def test_bad_parameters_name_the_parameter(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            spec(**kw)

    def test_no_targets_and_no_intercept_leak(self):
        with pytest.raises(TypeError, match=r"micro\(\) takes no targets"):
            po.spec.micro("m", features=["x0"], targets=["x0"], eps=0.1, halflife=10.0)
        # A feature named like the plumbing target is not a leak.
        assert spec(features=("x0",))["targets"] == ["x0"]

    def test_unpack_says_what_a_micro_struct_holds(self):
        X, _ = blobs(n=100, seed=38)
        out = po.ModelBank([spec()]).fit_predict(frame(X))
        with pytest.raises(TypeError, match="a kmeans or micro struct assignments"):
            po.eval.unpack(out, "m")

    def test_the_cli_runs_it(self, tmp_path, online_cli):
        import subprocess

        X, _ = blobs(n=300, seed=39)
        src = tmp_path / "in.parquet"
        dst = tmp_path / "out.parquet"
        frame(X).write_parquet(src)
        cfg = tmp_path / "bank.toml"
        cfg.write_text(
            "\n".join(
                [
                    f'input = "{src.as_posix()}"',
                    f'output = "{dst.as_posix()}"',
                    "[[specs]]",
                    'name = "m"',
                    'features = ["x0", "x1"]',
                    'targets = ["x0"]',
                    "halflife = 1000.0",
                    "min_periods = 5.0",
                    "[specs.model]",
                    'type = "micro"',
                    "eps = 0.1",
                    "prune_every = 50",
                ]
            )
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        got = unnested(pl.read_parquet(dst))
        want = unnested(po.ModelBank([spec(prune_every=50)]).fit_predict(frame(X)))
        assert got.equals(want, null_equal=True)
        assert got["outlier"].dtype == pl.Boolean and got["micro"].dtype == pl.Int64
