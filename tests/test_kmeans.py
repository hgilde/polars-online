"""`kmeans` -- exponentially weighted k-means (docs/CLUSTERING.md 6.2,
docs/PLAN.md 11a, task 23).

Not a regression: no targets, and the outputs (``cluster``, ``dist``,
``dist2``) are the assignment of each row to the centres *as they stood
before the row*. Three kinds of check:

- **The oracle.** ``tests/reference_cluster.py`` mirrors the Rust operation
  for operation, so the bank is held to it **bit for bit** -- across every
  seeding rule, both metrics, mini-batches, split-merge, nulls, weights and
  an irregular clock -- not to a tolerance. A hard assignment that flips on
  a rounding difference would move whole rows between centres, and a
  tolerance would hide exactly that.
- **Large data.** Hundreds of thousands of rows, held to the truth (ARI
  against the generating labels) and to batch Lloyd's answer on the same
  rows, computed here in numpy. No scikit-learn.
- **Edge cases and plumbing.** Everything docs/PLAN.md section 3 promises,
  and every place the model touches the bank: warmup, nulls, zero weights,
  ``k = 1``, duplicate rows under the ``first`` rule, constant features,
  chunk invariance across the seeding boundary, save/load, groups, the
  halflife grid, ``coef`` and its index, the expression path, the lazy
  path, and the refusals.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po
import reference_cluster as ref

RULES = ["first", "farthest", "kmeanspp", "lloyd"]


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
    d = dict(features=list(features), k=k, halflife=200.0, min_periods=5.0, warm_rows=30)
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
    for key in ("cluster", "dist", "dist2", "n_eff"):
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
            halflife=150.0,
            min_periods=3.0,
            warm_rows=40,
            seed_rule=rule,
            seed=11,
            standardize=standardize,
        )
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(df))
        want = ref.kmeans_ref(X.tolist(), **params)
        _same(out, want, f"{rule}/{standardize}")
        assert out["cluster"].null_count() < df.height

    @pytest.mark.parametrize("update_every", [1, 7, 64])
    @pytest.mark.parametrize("split_merge", [0.0, 0.5, 2.5])
    def test_mini_batches_and_split_merge_bit_for_bit(self, update_every, split_merge):
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
            halflife=300.0,
            min_periods=3.0,
            warm_rows=100,
            update_every=update_every,
            split_merge=split_merge,
            sm_every=50,
            dead_frac=0.1,
            seed=3,
        )
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(df))
        want = ref.kmeans_ref(X.tolist(), **params)
        _same(out, want, f"every={update_every}/sm={split_merge}")
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
        params = dict(k=3, halflife=60.0, min_periods=4.0, warm_rows=25, seed=9)
        s = spec(**params, clock="t", max_dclock=10.0, weight="w")
        out = unnested(po.ModelBank([s]).fit_predict(df))
        want = ref.kmeans_ref(rows, clock=t.tolist(), weight=w, max_dclock=10.0, **params)
        _same(out, want, "nulls/weights/clock")
        assert out["cluster"].null_count() > 40, "the fixture skipped fewer rows than it claims"
        assert out["n_eff"][1] == 0.0, "a zero-weight first row learned something"

    def test_first_rule_waits_for_distinct_rows_then_gives_up_at_the_cap(self):
        # Only two distinct rows for the first 1100 rows: `first` with k=3
        # cannot seed until the buffer cap (1000), where it takes duplicates.
        rows = [[0.0, 0.0] if i % 2 else [1.0, 1.0] for i in range(1100)]
        X, _ = blobs(n=400, seed=6)
        rows += X.tolist()
        df = frame(np.array(rows))
        params = dict(k=3, halflife=float("inf"), min_periods=1.0, warm_rows=5, seed_rule="first")
        out = unnested(po.ModelBank([spec(**params)]).fit_predict(df))
        want = ref.kmeans_ref(rows, **params)
        _same(out, want, "first at the cap")
        first_scored = out["cluster"].is_not_null().arg_max()
        assert first_scored == ref.BUF_CAP, first_scored

    def test_predict_matches_the_oracle_without_learning(self):
        X, _ = blobs(n=600, seed=7)
        df = frame(X)
        params = dict(k=3, halflife=100.0, min_periods=3.0, warm_rows=20)
        bank = po.ModelBank([spec(**params)])
        bank.fit_predict(df.slice(0, 400))
        want = ref.kmeans_ref(X[:400].tolist(), **params)
        model = want["model"][0]
        probe = df.slice(400, 200)
        got = unnested(bank.predict(probe))
        for i, row in enumerate(X[400:].tolist()):
            pred, n_eff = model.predict(row)
            assert got["cluster"][i] == int(pred[0])
            assert got["dist"][i] == pred[1]
            assert got["n_eff"][i] == n_eff
        # ... and predicting changed nothing.
        after = unnested(bank.fit_predict(probe))
        oracle_after = ref.kmeans_ref(X.tolist(), **params)
        for key in ("cluster", "dist", "n_eff"):
            assert after[key].to_list() == oracle_after[key][400:], key


# ------------------------------------------------------------- large data


class TestLargeData:
    def test_recovers_five_blobs_in_four_dimensions(self):
        # Five Gaussian blobs at a random layout in four dimensions, the
        # closest pair 5.4 sd apart. One k-means++ start followed by Lloyd
        # lands in the split-one-merge-two optimum here (ARI 0.72); the
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
            k=k,
            halflife=20_000.0,
            min_periods=100.0,
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
        s = spec(k=k, halflife=float("inf"), min_periods=50.0, warm_rows=3000, standardize=False)
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
        # dead_frac of an equal share: log2(1/dead_frac) halflives after
        # its blob vanished. Without split-merge the stranded centre stays
        # where it was and the new blob shares a centre with its neighbour.
        n, halflife = 20_000, 1000.0
        X, lab = stranded(seed=1, n=n)
        df = frame(X)

        def run(**kw):
            s = spec(k=4, halflife=halflife, min_periods=1.0, warm_rows=200, sm_every=100, **kw)
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
        assert rows is not None and abs(rows - math.log2(1 / 0.05) * halflife) < halflife, rows
        tail, rows = run(split_merge=0.5, dead_frac=0.25)
        assert tail > 0.95, tail
        assert rows is not None and abs(rows - math.log2(1 / 0.25) * halflife) < halflife, rows
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
                halflife=2000.0,
                min_periods=1.0,
                warm_rows=500,
                split_merge=split_merge,
                sm_every=100,
            )
            got = unnested(po.ModelBank([s]).fit_predict(df))["cluster"].fill_null(-1).to_numpy()
            keep = (lab >= 0) & (got >= 0)
            for st in range(500, n, 5000):
                m = keep & (np.arange(n) >= st) & (np.arange(n) < st + 5000)
                assert ari(lab[m], got[m]) > 0.97, (split_merge, st, ari(lab[m], got[m]))
        # Through the oracle (bit-for-bit with the bank): no move at all.
        m = ref.KMeansRef(p=2, k=k, halflife=2000.0, min_periods=1.0, warm_rows=500, sm_every=100)
        for x in X[:10_000]:
            m.step(list(x), 1.0, 1.0)
        assert (m.n_merges, m.n_dead) == (0, 0)


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
        s = spec(warm_rows=40, min_periods=60.0, halflife=float("inf"))
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        cl = out["cluster"]
        assert cl[:60].null_count() == 60
        assert cl[60:].null_count() == 0
        # n_eff is the weight before the row: the row count with no decay.
        assert out["n_eff"][0] == 0.0 and out["n_eff"][59] == 59.0

    def test_k_equals_one_has_no_runner_up(self):
        X, _ = blobs(n=100, seed=21)
        out = unnested(
            po.ModelBank([spec(k=1, warm_rows=5, min_periods=1.0)]).fit_predict(frame(X))
        )
        assert out["dist2"].null_count() == 100
        assert out["dist"].null_count() == 5
        assert set(out["cluster"].drop_nulls().to_list()) == {0}

    def test_k_larger_than_warm_rows_waits_for_k_rows(self):
        X, _ = blobs(n=100, k=3, seed=22)
        out = unnested(
            po.ModelBank([spec(k=10, warm_rows=1, min_periods=1.0)]).fit_predict(frame(X))
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
        out = unnested(po.ModelBank([spec(warm_rows=50, min_periods=1.0)]).fit_predict(frame(X)))
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
            s = spec(k=2, warm_rows=200, min_periods=1.0, standardize=standardize)
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
            po.ModelBank([spec(halflife=20.0, warm_rows=10, min_periods=1.0)]).fit_predict(df)
        )
        assert out["cluster"][150] is None and out["n_eff"][150] is None
        # The skipped row's tick is folded into the next one: row 151 reads
        # the weight row 150 would have (n_eff is the weight *before* the
        # row), and row 152 sees two ticks of decay between them.
        lam = 0.5 ** (1 / 20)
        assert out["n_eff"][151] == pytest.approx(out["n_eff"][149] * lam + 1.0, rel=1e-12)
        assert out["n_eff"][152] == pytest.approx(out["n_eff"][151] * lam**2 + 1.0, rel=1e-12)

    def test_chunk_invariance_across_seeding_and_checkpoints(self):
        X, _ = blobs(n=700, seed=26)
        df = frame(X)
        s = spec(warm_rows=100, update_every=13, sm_every=40, min_periods=1.0)
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
        s = spec(warm_rows=200, update_every=5, sm_every=30, min_periods=1.0)
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
        s = spec(halflife=[50.0, 500.0], warm_rows=20)
        assert po.spec.output_fields(s) == [
            "cluster@h50",
            "dist@h50",
            "dist2@h50",
            "n_eff@h50",
            "coef@h50",
            "cluster@h500",
            "dist@h500",
            "dist2@h500",
            "n_eff@h500",
            "coef@h500",
        ]
        out = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        assert out["cluster@h50"].dtype == pl.Int32
        assert out["n_eff@h50"][-1] < out["n_eff@h500"][-1]

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
        s = spec(features=("a", "b", "c"), k=2, halflife=[10.0, 20.0])
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
        assert ci["target"].to_list() == ["cluster0"] * 3 + ["cluster1"] * 3
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
        assert c["target"].to_list() == ["cluster0"] * 3 + ["cluster1"] * 3
        assert c["coef"].to_list() == [flat[n][-1] for n in names]

    def test_output_index_declares_the_dtypes(self):
        idx = po.spec.output_index(spec())
        assert idx["kind"].to_list() == ["cluster", "dist", "dist2", "n_eff", "coef"]
        assert idx["dtype"].to_list() == ["i32", "f64", "f64", "f64", "list[f64]"]
        assert idx["columns"][0].to_list() == ["x0", "x1"]

    def test_expression_equals_bank(self):
        X, _ = blobs(n=400, seed=32)
        df = frame(X).with_columns(g=pl.Series(["p", "q"] * 200))
        bank = unnested(po.ModelBank([spec(group="g", warm_rows=30)]).fit_predict(df))
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("x0")
                .online.kmeans(["x1"], k=3, halflife=200.0, min_periods=5.0, warm_rows=30)
                .over("g")
            ).unnest("x0")
        assert bank.equals(expr, null_equal=True)

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
            halflife=float("inf"),
            min_periods=1.0,
            warm_rows=50,
            sm_every=100,
            standardize=False,
        )
        plain = unnested(po.ModelBank([s]).fit_predict(frame(X)))
        burst = unnested(po.ModelBank([s]).fit_predict(frame(Y)))
        # Same centre before and after the burst (coef on the update cadence).
        assert burst["coef"][299] == plain["coef"][299]
        assert burst["coef"][329] == plain["coef"][299]
        # The far rows are scored, at their distance, and n_eff counts them
        # (it is the model's clock, not the clusters' weight).
        assert burst["cluster"][300:330].to_list() == [0] * 30
        assert burst["dist"][310] > 40.0
        assert burst["n_eff"][330] == pytest.approx(330.0)
        # Through the oracle: the cluster weight excludes them.
        m = ref.KMeansRef(
            p=2,
            k=1,
            halflife=math.inf,
            min_periods=1.0,
            warm_rows=50,
            sm_every=100,
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
        n, halflife = 6000, 200.0
        rng = np.random.default_rng(36)
        lab = rng.integers(0, k, n)
        centres = np.array([[0.0, 0.0], [12.0, 0.0]])[:k]
        X = centres[lab] + 0.5 * rng.standard_normal((n, 2))
        jump = np.arange(n) >= n // 2
        X[jump & (lab == 0)] += [0.0, 20.0]
        s = spec(
            k=k,
            halflife=halflife,
            min_periods=1.0,
            warm_rows=100,
            sm_every=100,
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
            # log2(1/dead_frac) halflives from an equal share (the stranded
            # centre's is a little above it), rounded up to a check.
            assert 300 < first_close <= math.log2(1 / 0.25) * halflife + 200, first_close

    def test_a_row_at_the_input_bound_leaves_everything_finite(self):
        X, _ = blobs(n=300, seed=34)
        X[120] = [1e100, -1e100]
        X[121] = [1e-300, 1e-300]
        out = unnested(po.ModelBank([spec(warm_rows=30, min_periods=1.0)]).fit_predict(frame(X)))
        assert out["n_eff"].is_finite().all()
        assert out["dist"].drop_nulls().is_finite().all()
        coef = np.array(out["coef"][-1])
        assert np.isfinite(coef).all()
        # The centres recover: after 178 ordinary rows the two extreme rows
        # are a small part of an EW mean at halflife 200.
        assert np.abs(coef).max() < 1e100 * 0.01


class TestRefusals:
    @pytest.mark.parametrize(
        "flag",
        [
            {"emit_sigma": True},
            {"emit_resid_z": True},
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
            po.spec.ew_cov("c", features=["x0", "x1"], halflife=10.0, **flag)

    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            ({"k": 0}, "k must be >= 1"),
            ({"seed_rule": "random"}, "unknown kmeans seed_rule"),
            ({"update_every": 0}, "update_every must be >= 1"),
            ({"sm_every": 0}, "sm_every must be >= 1"),
            ({"split_merge": -1.0}, "split_merge must be finite and >= 0"),
            ({"dead_frac": -0.1}, "dead_frac must be finite and >= 0"),
            ({"features": ["x0", "x0"]}, "more than once"),
        ],
        ids=lambda v: next(iter(v)) if isinstance(v, dict) else v,
    )
    def test_bad_parameters_name_the_parameter(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            spec(**kw)

    def test_no_targets_and_no_intercept_leak(self):
        with pytest.raises(TypeError, match=r"kmeans\(\) takes no targets"):
            po.spec.kmeans("m", features=["x0"], targets=["x0"], k=2, halflife=10.0)
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
                    "halflife = 200.0",
                    "min_periods = 5.0",
                    "[specs.model]",
                    'type = "kmeans"',
                    "k = 3",
                    "warm_rows = 30",
                ]
            )
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        got = unnested(pl.read_parquet(dst))
        want = unnested(po.ModelBank([spec(warm_rows=30)]).fit_predict(frame(X)))
        assert got.equals(want, null_equal=True)
        assert not math.isnan(got["dist"][-1])
