"""`kmeans` -- exponentially weighted k-means (docs/CLUSTERING.md 6.2,
docs/PLAN.md 11a, task 23).

Not a regression: no targets, and the outputs (``cluster``, ``dist``,
``dist_second``) are the assignment of each row to the centres *as they stood
before the row*. Four kinds of check:

- **The transcription.** ``tests/reference_cluster.py`` mirrors the Rust
  operation for operation, so the bank is held to it **bit for bit** --
  across every seeding rule, both metrics, mini-batches, split-merge, nulls,
  weights and an irregular clock -- not to a tolerance. A hard assignment
  that flips on a rounding difference would move whole rows between
  centres, and a tolerance would hide exactly that. Being a transcription,
  it is a regression check, and can share a mistake with the code.
- **The definitions.** ``TestDefinitions`` recomputes each checkpoint's
  centres, weights and radii from the raw rows assigned to them, at their
  decayed weights, and holds the far, merge and dead decisions to the
  module doc's formulas, row by row, against the exported state; to 1e-9.
- **Large data.** Hundreds of thousands of rows, held to the truth (ARI
  against the generating labels), to batch Lloyd's answer on the same rows
  computed here in numpy, and to scikit-learn's ``KMeans``.
- **Edge cases and plumbing.** Everything docs/PLAN.md section 3 promises,
  and every place the model touches the bank: warmup, nulls, zero weights,
  ``k = 1``, duplicate rows under the ``first`` rule, constant features,
  chunk invariance across the seeding boundary, save/load, groups, the
  half-life grid, ``coef`` and its index, the expression path, the lazy
  path, and the refusals.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import polars as pl
import pytest

import polars_online as po
import reference_cluster as ref

TIER = "mixed"

RULES = ["first", "farthest", "k-means++", "lloyd"]


def blobs(n=3000, k=3, p=2, seed=0, scale=0.7, spread=5.0):
    """`n` rows around `k` centres on a circle of radius `spread` (in the
    first two dimensions), with the generating label."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centres = np.zeros((k, p))
    centres[:, 0] = spread * np.cos(ang)
    centres[:, 1 % p] = spread * np.sin(ang) if p > 1 else centres[:, 0]
    lab = rng.integers(0, k, n)
    X = centres[lab] + scale * rng.standard_normal((n, p))
    return X, lab


def frame(X, **cols):
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(X.shape[1])})
    return df.with_columns(**{k: pl.Series(v) for k, v in cols.items()}) if cols else df


def spec(features=("x0", "x1"), k=3, **kw):
    d = dict(features=list(features), n_clusters=k, half_life=200.0, min_weight=5.0, warm_rows=30)
    d.update(kw)
    return po.spec.kmeans("m", **d)


def unnested(out):
    return out.select("m").unnest("m")


def ari(a, b) -> float:
    """Adjusted Rand index of two labelings (Hubert & Arabie 1985), from the
    contingency table -- the same number scikit-learn reports."""
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_ij = comb2(table).sum()
    sum_a = comb2(table.sum(axis=1)).sum()
    sum_b = comb2(table.sum(axis=0)).sum()
    n = comb2(len(a))
    expected = sum_a * sum_b / n
    top = (sum_a + sum_b) / 2.0 - expected
    if top == 0.0:
        return 1.0
    return float((sum_ij - expected) / top)


# ---------------------------------------------------------------- the oracle


def _same(bank: pl.DataFrame, oracle: dict[str, list], what: str) -> None:
    for key in ("cluster", "dist", "dist_second", "weight_sum"):
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
    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("standardize", [True, False], ids=["std", "raw"])
    def test_every_seeding_rule_bit_for_bit(self, rule, standardize):
        X, _ = blobs(n=1500, seed=1)
        # Unequal scales, so the metric matters.
        X[:, 1] *= 40.0
        df = frame(X)
        params = dict(
            k=3,
            half_life=150.0,
            min_weight=3.0,
            warm_rows=40,
            init=rule,
            seed=11,
            standardize=standardize,
        )
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(df))
        want = ref.kmeans_ref(X.tolist(), **params)
        _same(out, want, f"{rule}/{standardize}")
        assert out["cluster"].null_count() < df.height

    @pytest.mark.parametrize("update_every_rows", [1, 7, 64])
    @pytest.mark.parametrize("split_merge", [0.0, 0.5, 2.5])
    def test_mini_batches_and_split_merge_bit_for_bit(self, update_every_rows, split_merge):
        X, _ = blobs(n=2500, k=4, seed=2)
        # Two clusters appear only after seeding, so split-merge has work.
        X[:600] = (
            X[:600] * 0.0
            + np.array([[3.0, 3.0]])
            + 0.5 * np.random.default_rng(5).standard_normal((600, 2))
        )
        df = frame(X)
        params = dict(
            k=4,
            half_life=300.0,
            min_weight=3.0,
            warm_rows=100,
            update_every_rows=update_every_rows,
            split_merge=split_merge,
            split_merge_every_rows=50,
            # The dead rule runs at a split-merge check: without one it is
            # 0, and above it refused (review 2026-10-06, CF6).
            dead_frac=0.1 if split_merge > 0.0 else 0.0,
            seed=3,
        )
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(df))
        want = ref.kmeans_ref(X.tolist(), **params)
        _same(out, want, f"every={update_every_rows}/sm={split_merge}")
        model = want["model"][0]
        if split_merge > 0.0:
            assert model.n_merges + model.n_dead > 0, "the fixture never triggered a re-placement"

    def test_nulls_weights_and_an_irregular_clock_bit_for_bit(self):
        X, _ = blobs(n=1200, seed=3)
        rng = np.random.default_rng(4)
        rows: list[list[float | None]] = X.tolist()
        for i in range(0, len(rows), 37):
            rows[i][i % 2] = None  # a null feature: skipped
        rows[50][0] = float("nan")  # NaN is null too
        rows[51][1] = float("inf")
        rows[52][0] = 2e100  # beyond INPUT_BOUND
        w = (0.5 + rng.random(len(rows))).tolist()
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
        params = dict(k=3, half_life=60.0, min_weight=4.0, warm_rows=25, seed=9)
        s = spec(**params, clock="t", gap_cap=10.0, weight="w")
        out = unnested(po.ModelBank([s]).fit_predict(df))
        want = ref.kmeans_ref(rows, clock=t.tolist(), weight=w, gap_cap=10.0, **params)
        _same(out, want, "nulls/weights/clock")
        assert out["cluster"].null_count() > 40, "the fixture skipped fewer rows than it claims"
        assert out["weight_sum"][1] == 0.0, "a zero-weight first row learned something"

    def test_first_rule_waits_for_distinct_rows_then_gives_up_at_the_cap(self):
        # Only two distinct rows for the first 1100 rows: `first` with k=3
        # cannot seed until the buffer cap (1000), where it takes duplicates.
        rows = [[0.0, 0.0] if i % 2 else [1.0, 1.0] for i in range(1100)]
        X, _ = blobs(n=400, seed=6)
        rows += X.tolist()
        df = frame(np.array(rows))
        params = dict(k=3, half_life=float("inf"), min_weight=1.0, warm_rows=5, init="first")
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(df))
        want = ref.kmeans_ref(rows, **params)
        _same(out, want, "first at the cap")
        first_scored = out["cluster"].is_not_null().arg_max()
        assert first_scored == ref.BUF_CAP, first_scored

    def test_predict_matches_the_oracle_without_learning(self):
        X, _ = blobs(n=600, seed=7)
        df = frame(X)
        params = dict(k=3, half_life=100.0, min_weight=3.0, warm_rows=20)
        bank = po.ModelBank([spec(**params)])
        bank.fit_predict(df.slice(0, 400))
        want = ref.kmeans_ref(X[:400].tolist(), **params)
        model = want["model"][0]
        probe = df.slice(400, 200)
        got = unnested(bank.predict(probe))
        for i, row in enumerate(X[400:].tolist()):
            pred, weight_sum = model.predict(row)
            assert got["cluster"][i] == int(pred[0])
            assert got["dist"][i] == pred[1]
            assert got["weight_sum"][i] == weight_sum
        # ... and predicting changed nothing.
        after = unnested(bank.fit_predict(probe))
        oracle_after = ref.kmeans_ref(X.tolist(), **params)
        for key in ("cluster", "dist", "weight_sum"):
            assert after[key].to_list() == oracle_after[key][400:], key


# ------------------------------------------------------------- large data


class TestLargeData:
    def test_recovers_five_blobs_in_four_dimensions(self):
        # Five Gaussian blobs at a random layout in four dimensions, the
        # closest pair 5.4 sd apart. One k-means++ start followed by Lloyd
        # lands in the split-one-merge-two optimum here (ARI 0.72) in 9 of
        # 100 seeds, 28 of 100 from plain k-means++ (2026-10-25); the
        # `lloyd` rule's ten restarts pick the right partition by its cost.
        n, k, p = 200_000, 5, 4
        rng = np.random.default_rng(27)
        centres = rng.uniform(-6, 6, (k, p))
        lab = rng.integers(0, k, n)
        X = centres[lab] + 0.8 * rng.standard_normal((n, p))
        gaps = [np.linalg.norm(centres[i] - centres[j]) for i in range(k) for j in range(i)]
        assert 4.0 < min(gaps) < 4.5, "the fixture drifted"
        df = frame(X)
        s = po.spec.kmeans(
            "m",
            features=[f"x{i}" for i in range(p)],
            n_clusters=k,
            half_life=20_000.0,
            min_weight=100.0,
            warm_rows=2000,
        )
        out = unnested(po.ModelBank([s]).fit_predict(df))
        got = out["cluster"].to_numpy()
        scored = out["cluster"].is_not_null().to_numpy()
        assert scored.sum() == n - 2000
        assert ari(got[scored], lab[scored]) > 0.96
        # Against batch Lloyd started from the truth on the same rows.
        c = centres.copy()
        for _ in range(30):
            d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            a = d.argmin(axis=1)
            c = np.stack([X[a == j].mean(axis=0) for j in range(k)])
        assert ari(got[scored], a[scored]) > 0.97

    def test_agrees_with_batch_lloyd_on_the_same_rows(self):
        n, k = 100_000, 6
        X, lab = blobs(n=n, k=k, seed=12, scale=0.9, spread=6.0)
        df = frame(X)
        s = spec(k=k, half_life=float("inf"), min_weight=50.0, warm_rows=3000, standardize=False)
        out = unnested(po.ModelBank([s]).fit_predict(df))
        got = out["cluster"].to_numpy()
        scored = out["cluster"].is_not_null().to_numpy()
        # Batch Lloyd from the truth, converged: the best possible answer.
        c = np.stack([X[lab == j].mean(axis=0) for j in range(k)])
        for _ in range(50):
            d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            a = d.argmin(axis=1)
            c = np.stack([X[a == j].mean(axis=0) for j in range(k)])
        batch = d.argmin(axis=1)
        assert ari(got[scored], batch[scored]) > 0.98
        # The centres, after the last row, sit within a tenth of a noise sd
        # of the batch means (no decay: the EW means *are* the means of the
        # rows each centre was assigned, which is what Lloyd converges to).
        coef = np.array(out["coef"][-1]).reshape(k, 2)
        rows = np.linalg.norm(coef[:, None, :] - c[None, :, :], axis=2).min(axis=1)
        assert rows.max() < 0.09, rows

    def test_a_stranded_centre_is_re_placed_on_the_blob_born_later(self):
        # Four blobs; at n/2 one dies and a new one is born far from every
        # centre. Its rows are far from the centre they fall nearest to, so
        # that centre does not drift toward them; the dead rule re-places
        # the stranded centre on them once its weight has decayed below
        # dead_frac of an equal share: log2(1/dead_frac) half-lives after
        # its blob vanished. Without split-merge the stranded centre stays
        # where it was and the new blob shares a centre with its neighbour.
        n, half_life = 20_000, 1000.0
        X, lab = stranded(seed=1, n=n)
        df = frame(X)

        def run(**kw):
            s = spec(
                k=4,
                half_life=half_life,
                min_weight=1.0,
                warm_rows=200,
                split_merge_every_rows=100,
                **kw,
            )
            got = unnested(po.ModelBank([s]).fit_predict(df))["cluster"].fill_null(-1).to_numpy()
            tail = np.arange(n) >= n - n // 4
            recovered = next(
                (
                    st - n // 2
                    for st in range(n // 2, n, 500)
                    if ari(lab[st : st + 500], got[st : st + 500]) > 0.95
                ),
                None,
            )
            return ari(lab[tail], got[tail]), recovered

        tail, rows = run(split_merge=0.5)
        assert tail > 0.95, tail
        assert rows is not None and abs(rows - math.log2(1 / 0.05) * half_life) < half_life, rows
        tail, rows = run(split_merge=0.5, dead_frac=0.25)
        assert tail > 0.95, tail
        assert rows is not None and abs(rows - math.log2(1 / 0.25) * half_life) < half_life, rows
        tail, rows = run(split_merge=0.0)
        assert tail < 0.8 and rows is None, (tail, rows)

    def test_outliers_neither_spoil_the_seeds_nor_start_a_cascade(self):
        # Five per cent of the rows are uniform over a box three times the
        # blobs' extent. Rows far from the buffer's mean do not pick the
        # seeds; a freed centre is placed only on far rows that are a share
        # of the window and at least a few rows, so single outliers never
        # get a centre (a centre on one outlier has no radius, and the pair
        # criterion would then merge the two nearest real blobs).
        n, k = 40_000, 4
        rng = np.random.default_rng(29)
        X, lab = blobs(n=n, k=k, seed=29, scale=0.6, spread=6.0)
        out = rng.random(n) < 0.05
        lo, hi = X.min(axis=0), X.max(axis=0)
        X[out] = rng.uniform(lo - (hi - lo), hi + (hi - lo), (int(out.sum()), 2))
        lab[out] = -1
        df = frame(X)
        for split_merge in (0.5, 0.0):
            s = spec(
                k=k,
                half_life=2000.0,
                min_weight=1.0,
                warm_rows=500,
                split_merge=split_merge,
                split_merge_every_rows=100,
            )
            got = unnested(po.ModelBank([s]).fit_predict(df))["cluster"].fill_null(-1).to_numpy()
            keep = (lab >= 0) & (got >= 0)
            for st in range(500, n, 5000):
                m = keep & (np.arange(n) >= st) & (np.arange(n) < st + 5000)
                assert ari(lab[m], got[m]) > 0.97, (split_merge, st, ari(lab[m], got[m]))
        # Through the oracle (bit-for-bit with the bank): no move at all.
        m = ref.KMeansRef(
            p=2, k=k, half_life=2000.0, min_weight=1.0, warm_rows=500, split_merge_every_rows=100
        )
        for x in X[:10_000]:
            m.step(list(x), 1.0, 1.0)
        assert (m.n_merges, m.n_dead) == (0, 0)


class TestScaleFloor:
    """``scale_floor`` (docs/PLAN.md task 102): a feature quiet for thirty
    half-lives, then moving again. The bank is the oracle bit for bit at
    every floor, and the floor binds only in the quiet spell."""

    @staticmethod
    def stream():
        X, _ = blobs(n=6000, k=3, p=3, seed=3)
        X = X.copy()
        X[2000:5000, 2] = X[2000, 2]
        return X

    @staticmethod
    def spec_at(floor):
        return spec(
            features=("x0", "x1", "x2"), k=3, half_life=100.0, warm_rows=50, scale_floor=floor
        )

    @pytest.mark.parametrize("floor", [0.0, 0.1, 1.0])
    def test_the_bank_is_the_oracle_at_every_floor(self, floor):
        X = self.stream()
        got = unnested(po.ModelBank([self.spec_at(floor)]).fit_predict(frame(X)))
        want = ref.kmeans_ref(
            X.tolist(), k=3, half_life=100.0, min_weight=5.0, warm_rows=50, scale_floor=floor
        )
        _same(got, want, f"floor {floor}")

    def test_the_floor_binds_only_in_the_quiet_spell(self):
        X = self.stream()
        dist = {
            floor: unnested(po.ModelBank([self.spec_at(floor)]).fit_predict(frame(X)))["dist"]
            .fill_null(-1.0)
            .to_numpy()
            for floor in (0.0, 0.1)
        }
        # The EW variance of the held feature falls below a tenth of its
        # long-run one some 3.3 half-lives into the spell, not before.
        assert np.array_equal(dist[0.0][:2200], dist[0.1][:2200])
        assert not np.array_equal(dist[0.0][2500:5000], dist[0.1][2500:5000])


# ---------------------------------------------------------- the definitions
#
# Independent of `reference_cluster.py`, which transcribes the Rust and is
# held bit for bit above: each check here is written from the definitions in
# docs/CLUSTERING.md §6.1-6.3 and in the module docs of
# `crates/online-core/src/cluster/kmeans.rs` and `summary.rs`, recomputes
# what it needs from the raw rows, and agrees to 1e-9 (review 2026-10-05,
# TB1). They run without standardization, so a distance is the Euclidean one
# in the features' own units, and read the weights, radii and far summaries
# the bank does not emit from its exported state.


def _state(bank: po.ModelBank) -> dict:
    """The model's exported state: one group, one instance."""
    model = json.loads(bank.write_json())["states"][0][0][1]["models"][0]["model"]
    return model["KMeans"]


def _decay(dt, half_life):
    return 0.5 ** (np.asarray(dt, dtype=float) / half_life)


class TestDefinitions:
    """Assignment, centres, weights and radii from the rows; the far, merge
    and dead criteria from the formulas; scikit-learn's ``KMeans`` on a large
    case without decay."""

    @staticmethod
    def stream(n=900, seed=21):
        X, _ = blobs(n=n, k=3, seed=seed, scale=0.8)
        rng = np.random.default_rng(seed)
        steps = rng.exponential(1.0, n)
        steps[::97] = 15.0  # gaps
        t = np.cumsum(steps)
        w = rng.uniform(0.5, 1.5, n)
        w[28::29] = 0.0  # scored, not learned
        w[:3] = 1.0  # the three rows the `first` rule seeds from
        return X, t, w

    @staticmethod
    def by_definition(X, t, w, k, half_life, update_every_rows, metric=None):
        """Sequential k-means as docs/CLUSTERING.md §6.2-6.3 define it, with
        the `first` rule and no split-merge. The first `k` learned rows are
        the seeds, and replaying the buffer assigns each to itself. Every
        later row is assigned to the nearest centre as it stood (the first
        minimum wins), and at every `update_every_rows`-th learned row each centre
        is recomputed as the weighted mean of every row assigned to it, at
        the weight it has decayed to. Its radius² is the same weighted mean
        of each row's squared distance to the centre it was assigned to
        (`summary.rs`, `absorb_plain`). ``metric``, one row per row, is the
        diagonal metric each row's distances are read in; the Euclidean one
        when it is not given."""
        learned = [i for i in range(len(w)) if w[i] > 0.0]
        seeds = learned[:k]
        rows: list[list[int]] = [[i] for i in seeds]
        d2_at: dict[int, float] = dict.fromkeys(seeds, 0.0)

        def summarise(at, upto):
            out = []
            for js in rows:
                idx = np.array([i for i in js if i <= upto])
                wt = w[idx] * _decay(t[at] - t[idx], half_life)
                n = wt.sum()
                c = (wt[:, None] * X[idx]).sum(axis=0) / n
                r2 = (wt * np.array([d2_at[i] for i in idx])).sum() / n
                out.append((n, c, r2))
            return out

        checkpoints = [seeds[-1]]
        centres = np.array([c for _, c, _ in summarise(seeds[-1], seeds[-1])])
        scored: dict[int, tuple[int, float, float]] = {}
        since = 0
        for i in range(seeds[-1] + 1, len(w)):
            mw = 1.0 if metric is None else metric[i]
            d2 = ((X[i] - centres) ** 2 * mw).sum(axis=1)
            j = int(np.argmin(d2))
            scored[i] = (j, math.sqrt(d2[j]), math.sqrt(np.sort(d2)[1]))
            if w[i] <= 0.0:
                continue
            rows[j].append(i)
            d2_at[i] = float(d2[j])
            since += 1
            if since == update_every_rows:
                since = 0
                checkpoints.append(i)
                centres = np.array([c for _, c, _ in summarise(i, i)])
        return scored, checkpoints, summarise

    @staticmethod
    def numpy_metric(X, t, w, half_life):
        """The standardized metric each row is read in, from numpy: ``1/v``
        with ``v`` the biased weighted variance of each feature over the
        rows learned before it (weight above 0), each at its weight decayed
        on the clock, ``np.cov(aweights=, bias=True)``; 1 where ``v`` is 0
        or there are fewer than two such rows."""
        out = np.ones(X.shape)
        for i in range(1, len(X)):
            idx = np.flatnonzero(w[:i] > 0.0)
            if idx.size < 2:
                continue
            aw = w[idx] * _decay(t[i - 1] - t[idx], half_life)
            v = np.diag(np.cov(X[idx].T, aweights=aw, bias=True))
            out[i] = np.where(v > 0.0, 1.0 / np.where(v > 0.0, v, 1.0), 1.0)
        return out

    @pytest.mark.parametrize("standardize", [False, True], ids=["raw", "standardized"])
    @pytest.mark.parametrize("update_every_rows", [1, 7])
    def test_each_checkpoint_is_the_decayed_mean_of_its_rows(self, update_every_rows, standardize):
        """Standardized, the default, with the second feature in units fifty
        times the first's: every distance is read in the metric numpy's
        weighted variance gives (:meth:`numpy_metric`), and the definitions
        are otherwise the same. The metric's floor, ``scale_floor``, does not
        bind on this stream (the outputs at the default floor and at none
        are the same, checked below), so what is held is the variance."""
        X, t, w = self.stream()
        k, half_life = 3, 60.0
        if standardize:
            X = X * np.array([1.0, 50.0])
        params = dict(
            k=k,
            half_life=half_life,
            min_weight=0.0,
            warm_rows=k,
            init="first",
            split_merge=0.0,
            standardize=standardize,
            update_every_rows=update_every_rows,
            clock="t",
            gap_cap=1e9,
            weight="w",
            coef_every=0,
        )
        s = spec(**params)
        df = frame(X, t=t, w=w)
        metric = self.numpy_metric(X, t, w, half_life) if standardize else None
        scored, checkpoints, summarise = self.by_definition(
            X, t, w, k, half_life, update_every_rows, metric
        )
        out = unnested(po.ModelBank([s]).fit_predict(df))
        if standardize:
            unfloored = unnested(po.ModelBank([spec(**params, scale_floor=0.0)]).fit_predict(df))
            assert out.equals(unfloored), "the floor binds: the case holds the variance alone"
            # The metric moved: the second feature's weight is about 1/2500
            # of the first's, and neither is constant.
            assert np.median(metric[100:, 1] / metric[100:, 0]) < 1e-3
            assert metric[100:].std(axis=0).min() > 0.0
        # The assignment, and the distances to the nearest and the runner-up.
        # Tolerances here (task 252 (b)): definitions computed another way
        # (batch means of the raw rows at their decayed weights, against the
        # model's running compensated means) through libm (`exp2` in the
        # bank, `pow` here, over an irregular clock).
        assert out["cluster"][: checkpoints[0] + 1].null_count() == checkpoints[0] + 1
        for i, (j, d1, d2) in scored.items():
            assert out["cluster"][i] == j, i
            assert math.isclose(out["dist"][i], d1, rel_tol=1e-9), i
            assert math.isclose(out["dist_second"][i], d2, rel_tol=1e-9), i
        # The centres after every checkpoint, as `coef` reports them.
        for i in checkpoints:
            want = np.concatenate([c for _, c, _ in summarise(i, i)])
            np.testing.assert_allclose(out["coef"][i].to_numpy(), want, rtol=1e-9, atol=1e-12)
        # The weights and radii, from the exported state between checkpoints:
        # the last checkpoint's centres and radii, its rows' weights decayed
        # to the row the state was read after.
        bank = po.ModelBank([s])
        compared = 0
        for a in range(0, df.height, 23):
            bank.fit_predict(df.slice(a, 23))
            last = min(a + 23, df.height) - 1
            done = [c for c in checkpoints if c <= last]
            if not done:
                continue
            state = _state(bank)
            if standardize and last + 1 < df.height:
                # The metric the state holds for the next row is numpy's.
                np.testing.assert_allclose(state["mw"], metric[last + 1], rtol=1e-12)
            got = state["clusters"]
            for g, (n, c, r2) in zip(got, summarise(last, done[-1]), strict=True):
                assert math.isclose(g["n"], n, rel_tol=1e-9), (last, g["n"], n)
                np.testing.assert_allclose(g["c"], c, rtol=1e-9, atol=1e-12)
                assert math.isclose(g["r2"], r2, rel_tol=1e-9, abs_tol=1e-12), (last, g, r2)
            compared += 1
        assert compared > 30
        assert len(checkpoints) > (df.height - 100) // (update_every_rows * 2)

    # The constants `kmeans.rs` names in its module doc.
    FAR_SIGMAS, FAR_SHARE, FAR_ROWS, RADIUS_ROWS = 4.0, 0.05, 3, 10

    @classmethod
    def typical(cls, clusters, rows):
        """``R̃``: the mean of the positive radii² from at least
        ``RADIUS_ROWS`` rows, leaving out the largest; ``None`` when there
        is none, and nothing is far."""
        r2 = [c["r2"] for c, n in zip(clusters, rows, strict=True) if n >= cls.RADIUS_ROWS]
        r2 = [v for v in r2 if v > 0.0]
        if not r2:
            return None
        return r2[0] if len(r2) == 1 else (sum(r2) - max(r2)) / (len(r2) - 1)

    @staticmethod
    def absorb_plain(c, z, w, d2):
        """`summary.rs`: ``n' = n + w``, ``c' = c + b (z - c)``, ``r2' = r2 +
        b (d2 - r2)``, ``b = w / n'``."""
        c["n"] += w
        b = w / c["n"]
        c["c"] = c["c"] + b * (z - c["c"])
        c["r2"] += b * (d2 - c["r2"])

    @staticmethod
    def welford(c, z, w, r2_o=0.0):
        """`summary.rs`: ``a = n / n'``, ``b = w / n'``, ``c' = c + b δ``,
        ``r2' = a r2 + b r2_o + a b |δ|²``; a row is a summary of radius 0."""
        n = c["n"] + w
        if n <= 0.0:
            return
        a, b = c["n"] / n, w / n
        delta = z - c["c"]
        c["c"] = c["c"] + b * delta
        c["r2"] = a * c["r2"] + b * r2_o + a * b * float(delta @ delta)
        c["n"] = n

    @classmethod
    def split(cls, C, F, rows, typ, target, source):
        """The freed centre ``target`` goes to far summary ``source``'s
        mean with the typical radius² and half its cluster's weight."""
        C[source]["n"] *= 0.5
        C[target] = dict(n=C[source]["n"], c=F[source]["c"].copy(), r2=typ or 0.0)
        rows[target] = cls.RADIUS_ROWS

    def replay_criteria(self, X, **params):
        """Step the bank a row at a time and hold each row to the module
        doc's rules applied to the state before it: the row is far when its
        squared distance to the nearest centre is above the cut, and then
        goes to that centre's far summary and not into the centre; the cut
        is ``f R̃``, ``f = 1 + FAR_SIGMAS sqrt(2/p)``; and at every
        ``split_merge_every_rows``-th learned row, the radii take the far rows in
        as if at the cut, the closest pair by ``d_ij / (r_i + r_j)`` below
        ``split_merge`` is merged when the heaviest far summary (counting
        ``F_j`` as ``F_i``'s) holds ``FAR_ROWS`` rows and ``FAR_SHARE`` of
        the weight learned since the last check, and with no such pair the
        lightest centre, under ``dead_frac n_eff / k``, is dead. Either
        freed centre goes to the heaviest far summary's mean, with ``R̃``
        and half that cluster's weight, and every far summary starts
        again."""
        k, p = params["k"], X.shape[1]
        lam = 0.5 ** (1.0 / params["half_life"])
        f = 1.0 + self.FAR_SIGMAS * math.sqrt(2.0 / p)
        sm, every, dead_frac = (
            params["split_merge"],
            params["split_merge_every_rows"],
            params["dead_frac"],
        )
        bank = po.ModelBank([spec(min_weight=1.0, standardize=False, **params)])
        df = frame(X)
        seen = dict.fromkeys(("far", "checks", "merges", "dead"), 0)

        def summaries(state, key):
            return [
                dict(n=float(c["n"]), c=np.array(c["c"], dtype=float), r2=float(c["r2"]))
                for c in state[key]
            ]

        prev = None
        for i in range(df.height):
            bank.fit_predict(df.slice(i, 1))
            now = _state(bank)
            if prev is None or not prev["clusters"]:
                prev = now
                continue
            assert float(prev["far_factor"]) == f
            x = X[i]
            C, F = summaries(prev, "clusters"), summaries(prev, "far")
            rows, far_rows = list(prev["rows"]), list(prev["far_rows"])
            W = lam * float(prev["moments"]["w"]) + 1.0
            V = lam * float(prev["window_w"]) + 1.0
            for s in C + F:
                s["n"] *= lam
            d2 = np.array([float((x - c["c"]) @ (x - c["c"])) for c in C])
            j = int(np.argmin(d2))
            if d2[j] > float(prev["far_cut"]):
                self.welford(F[j], x, 1.0)
                far_rows[j] += 1
                seen["far"] += 1
            else:
                self.absorb_plain(C[j], x, 1.0, d2[j])
                rows[j] += 1
            typ = self.typical(C, rows)
            cut = math.inf if typ is None else f * typ
            merged = dead = False
            if prev["since_sm"] + 1 >= every:
                seen["checks"] += 1
                if math.isfinite(cut):
                    for c, fs in zip(C, F, strict=True):
                        if fs["n"] > 0.0:
                            c["r2"] = (c["n"] * c["r2"] + fs["n"] * cut) / (c["n"] + fs["n"])
                typ = self.typical(C, rows)
                cut = math.inf if typ is None else f * typ
                close = False
                if k >= 3:
                    best, pair = math.inf, (0, 0)
                    for a in range(k):
                        for b in range(a + 1, k):
                            den = math.sqrt(max(C[a]["r2"], 0.0)) + math.sqrt(max(C[b]["r2"], 0.0))
                            d = math.sqrt(float((C[a]["c"] - C[b]["c"]) @ (C[a]["c"] - C[b]["c"])))
                            ratio = 0.0 if d == 0.0 else (d / den if den > 0.0 else math.inf)
                            if ratio < best:
                                best, pair = ratio, (a, b)
                    if best < sm:
                        close = True
                        a, b = pair
                        src, n_src, rows_src = a, F[a]["n"] + F[b]["n"], far_rows[a] + far_rows[b]
                        for m in range(k):
                            if m not in pair and F[m]["n"] > n_src:
                                src, n_src, rows_src = m, F[m]["n"], far_rows[m]
                        if (
                            n_src > 0.0
                            and rows_src >= self.FAR_ROWS
                            and n_src >= self.FAR_SHARE * V
                        ):
                            self.welford(C[a], C[b]["c"], C[b]["n"], C[b]["r2"])
                            rows[a] += rows[b]
                            self.welford(F[a], F[b]["c"], F[b]["n"], F[b]["r2"])
                            self.split(C, F, rows, typ, b, src)
                            merged = True
                if not close and k >= 2 and dead_frac > 0.0:
                    jd = int(np.argmin([c["n"] for c in C]))
                    pools = [m for m in range(k) if F[m]["n"] > 0.0]
                    if C[jd]["n"] < dead_frac * W / k and pools:
                        heaviest = max(pools, key=lambda m: (F[m]["n"], -m))
                        self.split(C, F, rows, typ, jd, heaviest)
                        dead = True
                F = [dict(n=0.0, c=np.zeros(p), r2=0.0) for _ in range(k)]
                far_rows = [0] * k
                V = 0.0
            seen["merges"] += merged
            seen["dead"] += dead
            what = f"row {i}"
            assert now["n_merges"] == prev["n_merges"] + merged, what
            assert now["n_dead"] == prev["n_dead"] + dead, what
            assert now["rows"] == rows, what
            assert now["far_rows"] == far_rows, what
            # Tolerances (task 252 (b)): definitions computed another way
            # from the raw rows, through libm (`pow` here, `exp2` in the bank).
            assert float(now["far_cut"]) == pytest.approx(cut, rel=1e-9), what
            assert float(now["window_w"]) == pytest.approx(V, rel=1e-9), what
            pairs = zip(summaries(now, "clusters") + summaries(now, "far"), C + F, strict=True)
            for got, want in pairs:
                assert got["n"] == pytest.approx(want["n"], rel=1e-9), what
                np.testing.assert_allclose(got["c"], want["c"], rtol=1e-9, atol=1e-12)
                assert got["r2"] == pytest.approx(want["r2"], rel=1e-9, abs=1e-12), what
            prev = now
        return seen

    @pytest.mark.extended(
        reason="a second or more: the criteria against their formulas row by row (1.4 s)"
    )
    def test_the_far_and_merge_criteria_are_the_formulas(self):
        # Two blobs born after seeding, so a centre seeded on one blob is
        # freed for them by a merge.
        X, _ = blobs(n=2500, k=4, seed=2)
        X[:600] = np.array([[3.0, 3.0]]) + 0.5 * np.random.default_rng(5).standard_normal((600, 2))
        for split_merge in (0.5, 2.5):
            seen = self.replay_criteria(
                X,
                k=4,
                half_life=300.0,
                warm_rows=100,
                split_merge=split_merge,
                split_merge_every_rows=50,
                dead_frac=0.1,
                seed=3,
            )
            assert seen["far"] > 100 and seen["checks"] > 40 and seen["merges"] >= 3, seen

    @pytest.mark.extended(
        reason="a second or more: the criterion against its formula row by row (1.1 s)"
    )
    def test_the_dead_criterion_is_the_formula(self):
        # A blob dies and one is born far away: its centre is re-placed on
        # the new blob's far rows once it falls under dead_frac * n_eff / k.
        X, _ = stranded(seed=1, n=4000)
        seen = self.replay_criteria(
            X,
            k=4,
            half_life=200.0,
            warm_rows=200,
            split_merge=0.5,
            split_merge_every_rows=100,
            dead_frac=0.25,
        )
        assert seen["dead"] >= 1 and seen["far"] > 100, seen

    def test_a_large_case_without_decay_is_scikit_learns_kmeans(self):
        """With no decay and no split-merge each centre is the mean of the
        rows assigned to it, the fixed point batch Lloyd's algorithm reaches.
        scikit-learn's ``KMeans``, ten k-means++ starts over the same rows,
        is the second opinion: the same partition (ARI 0.99995 measured)
        and the same centres (5.4e-4 apart at most, the noise sd 0.9)."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        n, k = 100_000, 6
        X, _ = blobs(n=n, k=k, seed=12, scale=0.9, spread=6.0)
        s = spec(
            k=k,
            half_life=float("inf"),
            min_weight=50.0,
            warm_rows=3000,
            standardize=False,
            split_merge=0.0,
        )
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        got = out["cluster"].to_numpy()
        scored = out["cluster"].is_not_null().to_numpy()
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        score = ari(got[scored], km.labels_[scored])
        # Tolerance (task 252 (b)): scikit-learn's ARI, a second opinion
        # computed its own way.
        assert score == pytest.approx(adjusted_rand_score(got[scored], km.labels_[scored]))
        assert score > 0.999
        coef = np.array(out["coef"][-1]).reshape(k, 2)
        dist = np.linalg.norm(coef[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        assert sorted(dist.argmin(axis=1)) == list(range(k)), "not one centre each"
        assert dist.min(axis=1).max() < 0.01, dist.min(axis=1)

    def test_a_large_standardized_case_without_decay_is_scikit_learns_on_scaled_rows(self):
        """The default, ``standardize=True``, on the same blobs with the
        second feature in units a hundred times the first's. Without decay
        the metric settles to one over each feature's variance over the
        stream, so the fixed point is batch k-means on the rows scaled by
        their standard deviation: scikit-learn's ``StandardScaler(with_mean=
        False)`` then ``KMeans``, its centres scaled back. The centres stay
        in the features' own units here (the metric is scaled, not the
        rows). Measured 2026-11-04: ARI 0.99995, and the centres 1.7e-4
        apart at most in units of each feature's standard deviation (4.3
        and 434; the noise is 0.21 of them). The control: unstandardized,
        the second feature's units decide, and the partition is not
        scikit-learn's (ARI 0.40)."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        n, k = 100_000, 6
        X, _ = blobs(n=n, k=k, seed=12, scale=0.9, spread=6.0)
        X = X * np.array([1.0, 100.0])
        params = dict(k=k, half_life=float("inf"), min_weight=50.0, warm_rows=3000, split_merge=0.0)
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(frame(X)))
        got = out["cluster"].to_numpy()
        scored = out["cluster"].is_not_null().to_numpy()
        scaler = StandardScaler(with_mean=False).fit(X)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(scaler.transform(X))
        assert ari(got[scored], km.labels_[scored]) > 0.999
        coef = np.array(out["coef"][-1]).reshape(k, 2) / scaler.scale_
        dist = np.linalg.norm(coef[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        assert sorted(dist.argmin(axis=1)) == list(range(k)), "not one centre each"
        assert dist.min(axis=1).max() < 1e-3, dist.min(axis=1)
        raw = unnested(po.ModelBank([spec(**params, standardize=False)]).fit_predict(frame(X)))
        lost = raw["cluster"].to_numpy()
        assert ari(lost[scored], km.labels_[scored]) < 0.9


class TestScaleFloorDefinition:
    """``scale_floor`` from its definition (task 320, rank 2): the metric is
    ``1 / max(v, scale_floor · v_long)``, where ``v_long`` is each feature's
    long-run variance -- the same weighted moments at eight times the
    half-life, started from the medians of the feature's first five learned
    rows (the values' median, the median squared deviation from it, and for
    its weight five times the median weight). No library keeps such a
    reference, so its pieces come from numpy: ``np.median`` for the start,
    and for every later row the biased weighted moments
    (``np.average``) over the start, as a pseudo-sample of two points at
    ``m₀ ± √v₀`` sharing its weight, and the rows after it, each at its
    weight decayed on the clock at eight half-lives. The reference also
    clips a row's weight at its own and a squared deviation at a hundred
    times its variance; this stream reaches neither clip (checked row by
    row from the oracle), so what is held is the moments and the floor.
    The distances are then :meth:`TestDefinitions.by_definition`'s in that
    floored metric."""

    LONG_HALFLIVES, START_ROWS, CLIP = 8.0, 5, 100.0

    @staticmethod
    def stream():
        """`TestDefinitions.stream` at 1,500 rows, the second feature in
        units fifty times the first's and held still on rows 500-1199,
        about twelve half-lives."""
        X, t, w = TestDefinitions.stream(n=1500, seed=23)
        X = X * np.array([1.0, 50.0])
        X[500:1200, 1] = X[500, 1]
        return X, t, w

    @classmethod
    def by_numpy(cls, X, t, w, half_life, scale_floor):
        """Per row ``i``: the variance ``v`` and the reference ``v_long``
        each feature has over the rows learned before ``i`` (nan before
        there are any), and the metric ``1/max(v, scale_floor · v_long)``
        it reads row ``i`` in (1 where that is not positive). Asserts that
        no learned row after the start reaches either clip."""
        n, p = X.shape
        learned = np.flatnonzero(w > 0.0)
        s = cls.START_ROWS
        start = learned[:s]
        m0 = np.median(X[start], axis=0)
        v0 = np.median((X[start] - m0) ** 2, axis=0)
        w0 = s * np.median(w[start])
        long_hl = cls.LONG_HALFLIVES * half_life
        v = np.full((n, p), np.nan)
        v_long = np.full((n, p), np.nan)
        metric = np.ones((n, p))
        after = learned[s:]
        for i in range(1, n):
            idx = learned[learned < i]
            if idx.size >= 2:
                aw = w[idx] * _decay(t[i - 1] - t[idx], half_life)
                mu = np.average(X[idx], axis=0, weights=aw)
                v[i] = np.average((X[idx] - mu) ** 2, axis=0, weights=aw)
            if start[-1] < i:
                rows = after[after < i]
                pts = np.vstack([m0 - np.sqrt(v0), m0 + np.sqrt(v0), X[rows]])
                at = np.concatenate([[t[start[-1]]] * 2, t[rows]])
                aw = np.concatenate([[w0 / 2, w0 / 2], w[rows]]) * _decay(t[i - 1] - at, long_hl)
                mu_l = np.average(pts, axis=0, weights=aw)
                v_long[i] = np.average((pts - mu_l) ** 2, axis=0, weights=aw)
                if i in after:
                    # Row i's step: its weight against the reference's
                    # weight decayed to it, its squared deviation against
                    # a hundred times the reference's variance.
                    assert w[i] <= aw.sum() * _decay(t[i] - t[i - 1], long_hl), i
                    assert np.all((X[i] - mu_l) ** 2 <= cls.CLIP * v_long[i]), i
            if idx.size >= 2:
                floored = np.maximum(v[i], np.nan_to_num(scale_floor * v_long[i]))
                metric[i] = np.where(
                    floored > 0.0, 1.0 / np.where(floored > 0.0, floored, 1.0), 1.0
                )
        return v, v_long, metric

    def test_the_floored_metric_and_the_distances_are_the_definition(self):
        """Measured 2026-11-04: the state's ``var``, ``var_long`` and metric
        within 5.1e-15 of numpy's, relative, at every checkpoint, and the
        distances within 3.1e-14; held at 1e-13 and 1e-12 (running
        compensated means against batch means, through ``exp2`` in the bank
        and ``pow`` here, over an irregular clock). The floor binds on a third of the rows, all
        in the quiet spell, and moves the distances there: without it the
        quiet feature's weight grows 2^Q over Q half-lives."""
        X, t, w = self.stream()
        k, half_life, scale_floor = 3, 60.0, 0.1
        params = dict(
            k=k,
            half_life=half_life,
            min_weight=0.0,
            warm_rows=k,
            init="first",
            split_merge=0.0,
            update_every_rows=1,
            clock="t",
            gap_cap=1e9,
            weight="w",
            coef_every=0,
            scale_floor=scale_floor,
        )
        s = spec(**params)
        df = frame(X, t=t, w=w)
        v, v_long, metric = self.by_numpy(X, t, w, half_life, scale_floor)
        binds = scale_floor * v_long > v
        rows_bound = np.flatnonzero(binds.any(axis=1))
        print(f"the floor binds on {rows_bound.size} of {len(X)} rows")
        assert rows_bound.size > len(X) // 4
        assert rows_bound.min() >= 500 and rows_bound.max() < 1200 + 100
        # The state, read after every 23rd row: the reference and the metric
        # it holds for the next row.
        bank = po.ModelBank([s])
        compared = 0
        for a in range(0, df.height, 23):
            bank.fit_predict(df.slice(a, 23))
            nxt = min(a + 23, df.height)
            if nxt >= df.height or nxt <= self.START_ROWS + 3:
                continue
            state = _state(bank)
            np.testing.assert_allclose(state["moments"]["var_long"], v_long[nxt], rtol=1e-13)
            np.testing.assert_allclose(state["moments"]["var"], v[nxt], rtol=1e-13)
            np.testing.assert_allclose(state["mw"], metric[nxt], rtol=1e-13)
            compared += 1
        assert compared > 50
        # The assignment and the distances in the floored metric.
        scored, _, _ = TestDefinitions.by_definition(X, t, w, k, half_life, 1, metric)
        out = unnested(po.ModelBank([s]).fit_predict(df))
        for i, (j, d1, d2) in scored.items():
            assert out["cluster"][i] == j, i
            assert math.isclose(out["dist"][i], d1, rel_tol=1e-12), i
            assert math.isclose(out["dist_second"][i], d2, rel_tol=1e-12), i
        unfloored = unnested(po.ModelBank([spec(**{**params, "scale_floor": 0.0})]).fit_predict(df))
        moved = (unfloored["dist"] != out["dist"]).to_numpy()
        assert moved[rows_bound].mean() > 0.9 and not moved[:500].any()

    def test_micro_holds_the_same_floored_metric(self):
        """`micro` keeps the same moments and floor: its state's ``var_long``
        and metric are numpy's at every checkpoint, to the same 1e-13."""
        X, t, w = self.stream()
        half_life, scale_floor = 60.0, 0.1
        s = po.spec.micro(
            "m",
            features=["x0", "x1"],
            eps=0.5,
            half_life=half_life,
            clock="t",
            gap_cap=1e9,
            weight="w",
            scale_floor=scale_floor,
        )
        df = frame(X, t=t, w=w)
        v, v_long, metric = self.by_numpy(X, t, w, half_life, scale_floor)
        bank = po.ModelBank([s])
        compared = 0
        for a in range(0, df.height, 23):
            bank.fit_predict(df.slice(a, 23))
            nxt = min(a + 23, df.height)
            if nxt >= df.height or nxt <= self.START_ROWS + 3:
                continue
            model = json.loads(bank.write_json())["states"][0][0][1]["models"][0]["model"]
            state = model["Micro"]
            np.testing.assert_allclose(state["moments"]["var_long"], v_long[nxt], rtol=1e-13)
            np.testing.assert_allclose(state["mw"], metric[nxt], rtol=1e-13)
            compared += 1
        assert compared > 50
        assert (scale_floor * v_long > v).any(axis=1).sum() > len(X) // 4


def stranded(seed, n=20_000, sd=0.6, radius=6.0, born=(9.0, 9.0)):
    """Four blobs on a circle; from n/2 on, blob 3 is replaced by a fifth
    blob at `born`, far from every centre. Labels are the generating blob."""
    rng = np.random.default_rng(seed)
    ang = np.arange(4) * 2 * np.pi / 4
    centres = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    centres = np.vstack([centres, np.array(born)])
    lab = rng.integers(0, 4, n)
    lab[(np.arange(n) >= n // 2) & (lab == 3)] = 4
    X = centres[lab] + rng.normal(0.0, sd, (n, 2))
    return X, lab


class TestEdgeCases:
    def test_outputs_are_null_until_seeded_and_until_min_periods(self):
        X, _ = blobs(n=200, seed=20)
        s = spec(warm_rows=40, min_weight=60.0, half_life=float("inf"))
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        cl = out["cluster"]
        assert cl[:60].null_count() == 60
        assert cl[60:].null_count() == 0
        # weight_sum is the weight before the row: the row count with no decay.
        assert out["weight_sum"][0] == 0.0 and out["weight_sum"][59] == 59.0

    def test_k_equals_one_has_no_runner_up(self):
        X, _ = blobs(n=100, seed=21)
        out = unnested(po.ModelBank([spec(k=1, warm_rows=5, min_weight=1.0)]).fit_predict(frame(X)))
        assert out["dist_second"].null_count() == 100
        assert out["dist"].null_count() == 5
        assert set(out["cluster"].drop_nulls().to_list()) == {0}

    def test_warm_rows_below_k_is_refused_and_k_rows_seed_k_centres(self):
        """``max(warm_rows, k)`` rows were buffered, so ``warm_rows=1`` beside
        ``k=10`` meant 10 without a word; it is refused by name, as ``hmm``
        refuses it (review 2026-10-06, PC8). At ``warm_rows = k`` the model
        seeds on the k-th row, as the floor did."""
        with pytest.raises(
            ValueError,
            match=r'spec "m": warm_rows must be at least n_clusters \(10\) to seed '
            r"that many centers, but it is 1",
        ):
            spec(k=10, warm_rows=1, min_weight=1.0)
        X, _ = blobs(n=100, k=3, seed=22)
        out = unnested(
            po.ModelBank([spec(k=10, warm_rows=10, min_weight=1.0)]).fit_predict(frame(X))
        )
        assert out["cluster"].is_not_null().arg_max() == 10
        assert out["coef"][-1] is not None and len(out["coef"][-1]) == 20

    def test_a_constant_feature_is_measured_in_its_own_units(self):
        # Standardization divides by the variance; a constant feature has
        # none and is measured in its own units (weight 1) instead, so the
        # other feature still separates the blobs.
        rng = np.random.default_rng(23)
        lab = rng.integers(0, 3, 800)
        X = np.stack(
            [np.array([-6.0, 0.0, 6.0])[lab] + 0.7 * rng.standard_normal(800), np.full(800, 7.0)], 1
        )
        out = unnested(po.ModelBank([spec(warm_rows=50, min_weight=1.0)]).fit_predict(frame(X)))
        assert out["dist"].drop_nulls().is_finite().all()
        got = out["cluster"].to_numpy()
        scored = out["cluster"].is_not_null().to_numpy()
        assert ari(got[scored], lab[scored]) > 0.9

    def test_standardization_makes_a_scaled_feature_count(self):
        # Blobs separated in x0 only; x1 is noise at 1000x the scale. Raw
        # distances see only x1; standardized ones recover the blobs.
        rng = np.random.default_rng(24)
        n = 4000
        lab = rng.integers(0, 2, n)
        X = np.stack([lab * 6.0 + rng.standard_normal(n), 1000.0 * rng.standard_normal(n)], axis=1)
        df = frame(X)

        def score(standardize):
            s = spec(k=2, warm_rows=200, min_weight=1.0, standardize=standardize)
            out = unnested(po.ModelBank([s]).fit_predict(df))
            m = out["cluster"].is_not_null().to_numpy()
            return ari(out["cluster"].to_numpy()[m], lab[m])

        assert score(True) > 0.9
        assert score(False) < 0.2

    def test_a_null_feature_row_is_skipped_and_the_clock_still_runs(self):
        X, _ = blobs(n=300, seed=25)
        rows = X.tolist()
        rows[150][0] = None
        df = pl.DataFrame({"x0": [r[0] for r in rows], "x1": [r[1] for r in rows]})
        out = unnested(
            po.ModelBank([spec(half_life=20.0, warm_rows=10, min_weight=1.0)]).fit_predict(df)
        )
        assert out["cluster"][150] is None and out["weight_sum"][150] is None
        # The skipped row's tick is folded into the next one: row 151 reads
        # the weight row 150 would have (weight_sum is the weight *before* the
        # row), and row 152 sees two ticks of decay between them.
        # Tolerances (task 252 (b)): libm, Python's `pow` against the bank's
        # `exp2`, and `lam ** 2` against one factor for the two ticks.
        lam = 0.5 ** (1 / 20)
        assert out["weight_sum"][151] == pytest.approx(
            out["weight_sum"][149] * lam + 1.0, rel=1e-12
        )
        assert out["weight_sum"][152] == pytest.approx(
            out["weight_sum"][151] * lam**2 + 1.0, rel=1e-12
        )

    def test_chunk_invariance_across_seeding_and_checkpoints(self):
        X, _ = blobs(n=700, seed=26)
        df = frame(X)
        s = spec(warm_rows=100, update_every_rows=13, split_merge_every_rows=40, min_weight=1.0)
        one = unnested(po.ModelBank([s]).fit_predict(df))
        for size in (1, 7, 97, 350):
            bank = po.ModelBank([s])
            many = unnested(
                pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
            )
            # coef legitimately differs: it is also emitted on each chunk's
            # last row. Where the single run has one, the values agree.
            assert one.drop("coef").equals(many.drop("coef"), null_equal=True), size
            has = one["coef"].is_not_null()
            assert one.filter(has)["coef"].equals(many.filter(has)["coef"]), size

    def test_save_load_mid_warmup_and_after(self, tmp_path):
        X, _ = blobs(n=600, seed=27)
        df = frame(X)
        s = spec(warm_rows=200, update_every_rows=5, split_merge_every_rows=30, min_weight=1.0)
        for cut in (100, 250, 500):
            a = po.ModelBank([s])
            a.fit_predict(df.slice(0, cut))
            path = tmp_path / f"k{cut}.state"
            a.save(path)
            b = po.ModelBank.load(path, specs=[s])
            rest = df.slice(cut, df.height - cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    def test_groups_are_independent(self):
        X, _ = blobs(n=800, seed=28)
        df = frame(X).with_columns(g=pl.Series(["p", "q"] * 400))
        s = spec(group="g", warm_rows=50)
        both = po.ModelBank([s]).fit_predict(df)
        solo = po.ModelBank([s]).fit_predict(df.filter(pl.col("g") == "q"))
        assert unnested(both.filter(pl.col("g") == "q")).equals(unnested(solo), null_equal=True)

    def test_halflife_grid(self):
        X, _ = blobs(n=400, seed=29)
        s = spec(half_life=[50.0, 500.0], warm_rows=20)
        assert po.spec.output_fields(s) == [
            "cluster@h50",
            "dist@h50",
            "dist_second@h50",
            "weight_sum@h50",
            "settled_frac@h50",
            "withheld_reason@h50",
            "coef@h50",
            "cluster@h500",
            "dist@h500",
            "dist_second@h500",
            "weight_sum@h500",
            "settled_frac@h500",
            "withheld_reason@h500",
            "coef@h500",
        ]
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        assert out["cluster@h50"].dtype == pl.Int64
        assert out["weight_sum@h50"][-1] < out["weight_sum@h500"][-1]

    def test_coef_is_the_centres_on_the_cadence(self):
        X, _ = blobs(n=300, seed=30)
        s = spec(warm_rows=20, coef_every=50)
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        coef = out["coef"]
        # Every 50th learned row, and the last row of the chunk.
        assert [i for i in range(300) if coef[i] is not None] == sorted({*range(49, 300, 50), 299})
        assert len(coef[299]) == 6
        # Before seeding there are no centres: null, not an empty list.
        s2 = spec(warm_rows=200, coef_every=50)
        out2 = unnested(po.ModelBank([s2]).fit_predict(frame(X)))
        assert out2["coef"][49] is None and out2["coef"][199] is not None

    def test_coef_index_and_unnest_name_the_centres(self):
        s = spec(features=("a", "b", "c"), k=2, half_life=[10.0, 20.0])
        cf = po.spec.coef_fields(s)
        assert cf["name"].to_list()[:6] == [
            "coef_cluster0_a@h10",
            "coef_cluster0_b@h10",
            "coef_cluster0_c@h10",
            "coef_cluster1_a@h10",
            "coef_cluster1_b@h10",
            "coef_cluster1_c@h10",
        ]
        ci = po.spec.coef_index(s)
        assert ci["cluster"].to_list() == [0] * 3 + [1] * 3
        assert ci["term"].to_list() == ["a", "b", "c"] * 2
        X, _ = blobs(n=200, seed=31, p=3)
        df = pl.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
        s1 = spec(features=("a", "b", "c"), k=2, warm_rows=20)
        bank = po.ModelBank([s1])
        flat = bank.fit_predict(df).online.unnest([s1])
        names = po.spec.coef_fields(s1)["name"].to_list()
        assert names == [
            "coef_cluster0_a",
            "coef_cluster0_b",
            "coef_cluster0_c",
            "coef_cluster1_a",
            "coef_cluster1_b",
            "coef_cluster1_c",
        ]
        assert set(names) <= set(flat.columns) and "coef" not in flat.columns
        # The bank's own `coef` reads the same centres, one row per position.
        c = bank.coef("m")
        assert c["cluster"].to_list() == [0] * 3 + [1] * 3
        assert c["coef"].to_list() == [flat[n][-1] for n in names]

    def test_output_index_declares_the_dtypes(self):
        idx = po.spec.output_index(spec())
        tail = ["weight_sum", "settled_frac", "withheld_reason", "coef"]
        assert idx["kind"].to_list() == ["cluster", "dist", "dist_second", *tail]
        assert idx["dtype"].to_list() == ["i64", "f64", "f64", "f64", "f64", "enum", "list[f64]"]
        assert idx["columns"][0].to_list() == ["x0", "x1"]

    def test_lazy_path_equals_bank(self):
        X, _ = blobs(n=500, seed=33)
        df = frame(X)
        s = spec(warm_rows=30)
        bank = po.ModelBank([s]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([s]).collect()
        assert bank.equals(lazy, null_equal=True)

    def test_far_rows_neither_drag_the_centre_nor_widen_its_radius(self):
        # After a blob is learned, a burst of rows far outside it (more than
        # far_factor times the typical squared radius from the centre) is
        # summarised, not learned: the centre does not move at all, and the
        # radius grows only as if every far row sat at the cut.
        X, _ = blobs(n=600, k=1, seed=35, scale=0.5)
        Y = X.copy()
        Y[300:330] = [40.0, 40.0]
        s = spec(
            k=1,
            half_life=float("inf"),
            min_weight=1.0,
            warm_rows=50,
            split_merge_every_rows=100,
            standardize=False,
        )
        plain = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        burst = unnested(po.ModelBank([s]).fit_predict(frame(Y)))
        # Same centre before and after the burst (coef on the update cadence).
        assert burst["coef"][299] == plain["coef"][299]
        assert burst["coef"][329] == plain["coef"][299]
        # The far rows are scored, at their distance, and weight_sum counts them
        # (it is the model's clock, not the clusters' weight).
        assert burst["cluster"][300:330].to_list() == [0] * 30
        assert burst["dist"][310] > 40.0
        # Exact (task 252 (b)): no decay, unit weights, so a count.
        assert burst["weight_sum"][330] == 330.0
        # Through the oracle: the cluster weight excludes them.
        m = ref.KMeansRef(
            p=2,
            k=1,
            half_life=math.inf,
            min_weight=1.0,
            warm_rows=50,
            split_merge_every_rows=100,
            standardize=False,
        )
        for x in Y[:330]:
            m.step(list(x), 1.0, 1.0)
        # (Two ordinary rows were far as well: the cut is far_factor = 5
        # times the squared radius in 2-D, which a Gaussian blob crosses
        # 0.7% of the time. They counted in the radius as if at the cut,
        # nothing else, and were cleared at the check before the burst.)
        assert m.far[0].n == 30.0
        assert m.clusters[0].n == 298.0
        # ... and the radius widens at the check as if each far row sat at the cut.
        r2_before, cut = m.clusters[0].r2, m.far_cut
        for x in Y[330:400]:
            m.step(list(x), 1.0, 1.0)
        assert r2_before < m.clusters[0].r2 < cut
        assert m.far[0].n == 0.0

    @pytest.mark.parametrize("k", [1, 2])
    def test_a_jumped_blob_is_followed(self, k):
        # k = 1: the only radius is the typical one, so it widens toward the
        # far rows until they are learned, and the centre drifts over.
        # k = 2: the other radius sets the cut, which never widens; the
        # stranded centre is re-placed by the dead rule instead, once its
        # weight has decayed to dead_frac of an equal share.
        n, half_life = 6000, 200.0
        rng = np.random.default_rng(36)
        lab = rng.integers(0, k, n)
        centres = np.array([[0.0, 0.0], [12.0, 0.0]])[:k]
        X = centres[lab] + 0.5 * rng.standard_normal((n, 2))
        jump = np.arange(n) >= n // 2
        X[jump & (lab == 0)] += [0.0, 20.0]
        s = spec(
            k=k,
            half_life=half_life,
            min_weight=1.0,
            warm_rows=100,
            split_merge_every_rows=100,
            dead_frac=0.25,
            standardize=False,
        )
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        coef = np.array(out["coef"][-1]).reshape(k, 2)
        want = np.array([[0.0, 20.0], [12.0, 0.0]])[:k]
        # Each true centre has a centre within half a noise sd (in any order).
        assert np.linalg.norm(want[:, None] - coef[None], axis=2).min(axis=1).max() < 0.5
        dist = out["dist"].to_numpy()
        late = jump & (lab == 0)
        # The first rows after the jump are far, the last ones close.
        assert dist[late][:100].min() > 10.0
        assert dist[late][-500:].max() < 3.0
        first_close = np.flatnonzero(late & (dist < 3.0))[0] - n // 2
        if k == 1:
            # The radius about doubles per check (a third of the rows are
            # far, each counted at five times the radius) until the cut
            # covers the jump: log2(20^2 / 0.5) = 11 checks at most.
            assert first_close < 11 * 100, first_close
        else:
            # log2(1/dead_frac) half-lives from an equal share (the stranded
            # centre's is a little above it), rounded up to a check.
            assert 300 < first_close <= math.log2(1 / 0.25) * half_life + 200, first_close

    def test_a_row_at_the_input_bound_leaves_everything_finite(self):
        X, _ = blobs(n=300, seed=34)
        X[120] = [1e100, -1e100]
        X[121] = [1e-300, 1e-300]
        out = unnested(po.ModelBank([spec(warm_rows=30, min_weight=1.0)]).fit_predict(frame(X)))
        assert out["weight_sum"].is_finite().all()
        assert out["dist"].drop_nulls().is_finite().all()
        coef = np.array(out["coef"][-1])
        assert np.isfinite(coef).all()
        # The centres recover: after 178 ordinary rows the two extreme rows
        # are a small part of an EW mean at half-life 200.
        assert np.abs(coef).max() < 1e100 * 0.01


class TestRefusals:
    @pytest.mark.parametrize(
        "flag",
        [
            {"emit_sigma": True},
            {"emit_zscore": True},
            {"emit_metrics": True},
            {"resid_quantiles": [0.5]},
            {"conformal": 0.9},
            {"emit_autocorr": True},
            {"emit_drift": True},
            {"emit_selected": True},
            {"emit_averaged": True},
        ],
        ids=lambda f: next(iter(f)),
    )
    def test_residual_diagnostics_are_refused_by_name(self, flag):
        (name,) = flag
        with pytest.raises(ValueError, match=f"{name} does not apply to kmeans"):
            spec(**flag)
        with pytest.raises(ValueError, match=f"{name} does not apply to ew_cov"):
            po.spec.ew_cov("c", features=["x0", "x1"], half_life=10.0, **flag)

    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            ({"n_clusters": 0}, "n_clusters must be at least 1, but it is 0"),
            (
                {"init": "random"},
                re.escape(
                    'init must be "first", "farthest", "k-means++" or "lloyd", but it is "random"'
                ),
            ),
            ({"update_every_rows": 0}, "update_every_rows must be at least 1, but it is 0"),
            (
                {"split_merge_every_rows": 0},
                "split_merge_every_rows must be at least 1, but it is 0",
            ),
            ({"split_merge": -1.0}, "split_merge must be finite and at least 0"),
            ({"dead_frac": -0.1}, "dead_frac must be at least 0 and below 0.5"),
            ({"scale_floor": -0.5}, "scale_floor must be finite and at least 0"),
            ({"features": ["x0", "x0"]}, "more than once"),
        ],
        ids=lambda v: next(iter(v)) if isinstance(v, dict) else v,
    )
    def test_bad_parameters_name_the_parameter(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            spec(**kw)

    def test_no_targets_and_no_intercept_leak(self):
        with pytest.raises(TypeError, match=r"kmeans\(\) takes no targets"):
            po.spec.kmeans("m", features=["x0"], targets=["x0"], n_clusters=2, half_life=10.0)
        # A feature named like the plumbing target is not a leak.
        assert spec(features=("x0",), k=2)["targets"] == ["x0"]

    def test_unpack_says_what_a_kmeans_struct_holds(self):
        X, _ = blobs(n=100, seed=35)
        out = po.ModelBank([spec(warm_rows=10)]).fit_predict(frame(X))
        with pytest.raises(TypeError, match="a kmeans or micro struct assignments"):
            po.eval.unpack(out, "m")

    def test_the_cli_runs_it(self, tmp_path, online_cli):
        import subprocess

        X, _ = blobs(n=300, seed=36)
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
                    "half_life = 200.0",
                    "min_weight = 5.0",
                    "[specs.model]",
                    'type = "kmeans"',
                    "k = 3",
                    "warm_rows = 30",
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        got = unnested(pl.read_parquet(dst))
        want = unnested(po.ModelBank([spec(warm_rows=30)]).fit_predict(frame(X)))
        assert got.equals(want, null_equal=True)
        assert not math.isnan(got["dist"][-1])


# ------------------------------------------- the parameters (task 278, b22)


def _differ(a: pl.DataFrame, b: pl.DataFrame) -> list[str]:
    return [c for c in a.columns if not a[c].equals(b[c], null_equal=True)]


class TestTheSeed:
    """``seed`` drives the k-means++ draws, and through them ``lloyd``'s ten
    starts; ``first`` and ``farthest`` draw nothing. Overlapping blobs, so
    the draws matter."""

    @staticmethod
    def run(**kw):
        X, _ = blobs(n=2000, k=4, seed=2, scale=1.4, spread=3.0)
        s = spec(k=4, warm_rows=60, **kw)
        return unnested(po.ModelBank([s]).fit_predict(frame(X)))

    @pytest.mark.parametrize("rule", RULES)
    def test_reproducible_and_none_is_zero(self, rule):
        zero = self.run(init=rule, seed=0)
        assert _differ(zero, self.run(init=rule, seed=0)) == []
        assert _differ(zero, self.run(init=rule)) == []
        assert _differ(zero, self.run(init=rule, seed=None)) == []

    @pytest.mark.parametrize("seed", [1, 9, 2**64 - 1])
    def test_it_moves_the_draws_and_only_the_draws(self, seed):
        zero = {rule: self.run(init=rule, seed=0) for rule in RULES}
        other = {rule: self.run(init=rule, seed=seed) for rule in RULES}
        assert {"cluster", "dist"} <= set(_differ(zero["k-means++"], other["k-means++"]))
        for rule in ("first", "farthest"):
            assert _differ(zero[rule], other[rule]) == [], rule
        # lloyd keeps the best of its ten starts: on this stream every seed
        # tried finds the same partition, under other labels. So the
        # distances hold to the bit and the labels are one relabelling.
        moved = _differ(zero["lloyd"], other["lloyd"])
        assert moved == ["cluster", "coef"], moved
        pairs = set(
            zip(
                zero["lloyd"]["cluster"].drop_nulls(),
                other["lloyd"]["cluster"].drop_nulls(),
                strict=True,
            )
        )
        assert len(pairs) == 4 == len({a for a, _ in pairs}) == len({b for _, b in pairs})


class TestMergesAcrossChunksAndSaves:
    """Chunk invariance and save/load where split-merge has fired, proven
    from the state's own counters: the stationary-blob tests elsewhere never
    show a merge or a re-placement."""

    @staticmethod
    def stream():
        # Six hundred rows of a blob at (3, 3), then four overlapping blobs:
        # two centres seeded on the first blob merge once it is gone.
        X, _ = blobs(n=2000, k=4, seed=2, scale=1.4, spread=3.0)
        rng = np.random.default_rng(5)
        X[:600] = np.array([[3.0, 3.0]]) + 0.5 * rng.standard_normal((600, 2))
        return frame(X)

    SPEC = staticmethod(
        lambda: spec(
            k=4,
            warm_rows=100,
            half_life=300.0,
            split_merge_every_rows=50,
            dead_frac=0.1,
        )
    )

    def test_chunk_invariance(self):
        df = self.stream()
        bank = po.ModelBank([self.SPEC()])
        one = unnested(bank.fit_predict(df))
        st = _state(bank)
        # Measured 2026-10-25: three merges, between rows 600 and 800.
        assert st["n_merges"] + st["n_dead"] > 0, st
        for size in (1, 7, 600):
            many = unnested(
                pl.concat(po.ModelBank([self.SPEC()]).fit_predict_batches(df, chunk_size=size))
            )
            assert one.drop("coef").equals(many.drop("coef"), null_equal=True), size
            has = one["coef"].is_not_null()
            assert one.filter(has)["coef"].equals(many.filter(has)["coef"]), size

    def test_save_load_before_between_and_after_the_merges(self):
        df = self.stream()
        counts = []
        for cut in (550, 700, 1500):
            a = po.ModelBank([self.SPEC()])
            a.fit_predict(df.head(cut))
            st = _state(a)
            counts.append(st["n_merges"] + st["n_dead"])
            b = po.ModelBank.load_bytes(a.save_bytes(), specs=[self.SPEC()])
            assert _state(b) == st, cut
            rest = df.slice(cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut
            assert _state(a) == _state(b), cut
        # A cut before any merge, one after the first, one after them all.
        assert counts[0] == 0 < counts[1] <= counts[2], counts


class TestTheDefaults:
    """Every keyword left out is the value the docstring names, and that
    value is not inert: the other value given beside it moves the output.
    On the stream whose first blob vanishes, so that split-merge acts."""

    @staticmethod
    def run(k=4, **kw):
        df = TestMergesAcrossChunksAndSaves.stream()
        d = dict(features=["x0", "x1"], n_clusters=k, half_life=200.0, min_weight=5.0)
        d.update(kw)
        return unnested(po.ModelBank([po.spec.kmeans("m", **d)]).fit_predict(df))

    def test_each_default_spelled_out(self):
        default = self.run()
        assert default["cluster"].is_not_null().arg_max() == 500, "warm_rows is 500"
        for given, other in (
            (dict(warm_rows=500), dict(warm_rows=400)),
            (dict(init="lloyd"), dict(init="first")),
            (dict(seed=0), dict(seed=9)),
            (dict(update_every_rows=1), dict(update_every_rows=10)),
            # Measured 2026-10-25: the default merges once; 0.1 never does
            # (0.3 to 5 merge the same pair on the same check).
            (dict(split_merge=0.5), dict(split_merge=0.1)),
            (dict(split_merge_every_rows=100), dict(split_merge_every_rows=10)),
            (dict(dead_frac=0.05), dict(dead_frac=0.45)),
            (dict(standardize=True), dict(standardize=False)),
            (dict(scale_floor=0.1), dict(scale_floor=1.0)),
        ):
            assert _differ(default, self.run(**given)) == [], given
            assert _differ(default, self.run(**other)) != [], other

    def test_warm_rows_is_k_past_five_hundred(self):
        # 501 centres need 501 buffered rows; `first` keeps the seeding cheap.
        wide = self.run(k=501, init="first")
        assert wide["cluster"].is_not_null().arg_max() == 501
        assert _differ(wide, self.run(k=501, init="first", warm_rows=501)) == []

    def test_dead_frac_is_zero_beside_split_merge_zero(self):
        off = self.run(split_merge=0.0)
        assert _differ(off, self.run(split_merge=0.0, dead_frac=0.0)) == []
        assert _differ(off, self.run()) != []
        with pytest.raises(
            ValueError,
            match='spec "m": dead_frac = 0.05 re-places a center only at a split-merge check, but '
            "split_merge = 0 runs none",
        ):
            po.ModelBank([spec(split_merge=0.0, dead_frac=0.05)])


class TestTheBoundaries:
    """The ceilings and floors through Python, each value on both sides."""

    def test_n_clusters_ceiling(self):
        po.ModelBank([po.spec.kmeans("m", features=["x0"], n_clusters=65_536, half_life=10.0)])
        with pytest.raises(ValueError, match="n_clusters must be at most 65536, but it is 65537"):
            po.spec.kmeans("m", features=["x0"], n_clusters=65_537, half_life=10.0)

    def test_the_warm_up_buffer_ceiling(self):
        # A buffered row of two features is its vector (24 bytes on a
        # 64-bit target), its two values and its weight: 48 bytes, so
        # 5,592,405 rows fit 256 MiB and one more is past it.
        po.ModelBank([spec(warm_rows=5_592_405)])
        with pytest.raises(
            ValueError,
            match=r"the first 5592406 rows, which seed the centers, would take 256\.00003 MiB at 2 "
            r"features, over the 256 MiB limit\. Set warm_rows to 5592405 or below",
        ):
            po.ModelBank([spec(warm_rows=5_592_406)])

    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            ({"split_merge": float("inf")}, "split_merge must be finite, but it is inf"),
            ({"dead_frac": float("nan")}, "dead_frac must be a number, but it is NaN"),
            ({"scale_floor": float("inf")}, "scale_floor must be finite"),
            ({"warm_rows": 2}, "warm_rows"),
        ],
        ids=lambda v: next(iter(v)) if isinstance(v, dict) else v,
    )
    def test_a_value_past_its_bound_is_refused_by_name(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            po.ModelBank([spec(**kw)])

    def test_dead_frac_just_below_one_half_runs(self):
        # 0.5 and above are refused (a re-placed centre would be born at the
        # floor); just below it is legal and re-places on the formula.
        X, _ = blobs(n=600, seed=7)
        out = unnested(po.ModelBank([spec(dead_frac=0.499, warm_rows=50)]).fit_predict(frame(X)))
        assert out["dist"].drop_nulls().is_finite().all()
        with pytest.raises(ValueError, match="dead_frac must be below 0.5, but it is 0.5"):
            po.ModelBank([spec(dead_frac=0.5, warm_rows=50)])


class TestScaleEquivariance:
    """Under ``standardize`` distances are in units of each feature's EW
    standard deviation, so a change of units moves no assignment: a power
    of two per feature to the bit (the centres scale exactly), an affine
    change with the distances to rounding. Without standardization the
    units count, and one power of two for every feature scales every
    distance exactly."""

    SCALE = np.array([2.0**10, 2.0**-7])

    @staticmethod
    def run(X, **kw):
        return unnested(po.ModelBank([spec(k=4, warm_rows=60, **kw)]).fit_predict(frame(X)))

    @staticmethod
    def stream():
        X, _ = blobs(n=3000, k=4, seed=2, scale=0.8, spread=6.0)
        return X

    def test_a_power_of_two_per_feature_is_exact(self):
        X = self.stream()
        base, scaled = self.run(X), self.run(X * self.SCALE)
        assert _differ(base, scaled) == ["coef"]
        for a, b in zip(base["coef"], scaled["coef"], strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                assert np.array_equal(
                    np.array(a).reshape(-1, 2) * self.SCALE, np.array(b).reshape(-1, 2)
                )

    def test_an_affine_change_holds_the_assignments(self):
        X = self.stream()
        a, b = np.array([3.7, 0.013]), np.array([1e3, -42.0])
        base, moved = self.run(X), self.run(X * a + b)
        assert base["cluster"].equals(moved["cluster"], null_equal=True)
        assert base["weight_sum"].equals(moved["weight_sum"])
        # Measured 2026-10-25: 6.7e-14 at most on either distance.
        for key in ("dist", "dist_second"):
            diff = np.abs(base[key].to_numpy() - moved[key].to_numpy())
            assert np.nanmax(diff) < 1e-10, key
        centres = np.array(base["coef"][-1]).reshape(-1, 2) * a + b
        np.testing.assert_allclose(centres, np.array(moved["coef"][-1]).reshape(-1, 2), rtol=1e-12)

    def test_without_standardization_the_units_count(self):
        X = self.stream()
        base = self.run(X, standardize=False)
        assert "cluster" in _differ(base, self.run(X * self.SCALE, standardize=False))
        c = 2.0**6
        uniform = self.run(X * c, standardize=False)
        assert _differ(base, uniform) == ["dist", "dist_second", "coef"]
        assert (uniform["dist"] == base["dist"] * c).all()
        assert (uniform["dist_second"] == base["dist_second"] * c).all()
