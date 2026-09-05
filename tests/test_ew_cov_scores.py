"""E37 + E38 (task 26): `mahal` and `pca` on `ew_cov`.

Two row-level scores off the same EW moments (docs/ENHANCEMENTS.md E37/E38):

* ``mahal``: the row's Mahalanobis distance from the decayed history,
  ``sqrt(delta' (C + s*prior*I)^-1 delta)`` with ``delta = x - m``, read
  before the row is learned. With ``k`` Gaussian columns ``mahal**2`` is
  chi-squared with ``k`` degrees of freedom. ``mahal_quantiles`` adds P²
  running quantiles of the past scores (``mahal_q0.99``).
* ``pca=r``: the top ``r`` eigenpairs of the centred co-moments, refreshed
  every ``pca_every`` learned rows *after* the row is folded in, frozen in
  between; per component ``pc<j>_var``, ``pc<j>_share``, one loading per
  column and the row's score ``v_j . (x - m)`` about the live mean. Each
  loading vector is signed for continuity with the previous refresh
  (largest-magnitude entry positive on the first).

The oracle is a numpy replay of the accumulator (`EwCov::update`, the
Welford form in the same operation order, so the moments are bit-exact),
with `np.linalg.solve` for the distance and `np.linalg.eigh` for the
components.
"""

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

NO_DECAY = float("inf")

#: chi-squared 0.99 quantiles for k = 1..12 (scipy.stats.chi2.ppf(0.99, k)).
CHI2_99 = {
    1: 6.634896601021214,
    2: 9.21034037197618,
    3: 11.344866730144373,
    4: 13.276704135987622,
    5: 15.08627246938899,
    6: 16.811893829770927,
    8: 20.090235029663233,
    12: 26.216967305535853,
}


# --------------------------------------------------------------------- oracle


def replay(X, lam, w, *, prior, accepted=None):
    """Replay `EwCov::update` row by row and return, per row, the state
    *before* the row: (means, centred co-moments, precision_scale, w_sum).

    The Welford form, in the crate's operation order:
        w_new = lam*w_sum + w;  a = lam*w_sum/w_new;  b = w/w_new
        c_ij  = a*c_ij + a*b*d_i*d_j        (d = x - m, the OLD mean)
        m    += b*(x - m)
        precision_scale = 1 if a <= 0 else precision_scale*a
    A skipped row (`accepted` false) leaves the state alone.
    """
    n, k = X.shape
    m = np.zeros(k)
    c = np.zeros((k, k))
    w_sum = 0.0
    scale = 1.0
    states = []
    for i in range(n):
        states.append((m.copy(), c.copy(), scale, w_sum))
        if accepted is not None and not accepted[i]:
            continue
        w_new = lam[i] * w_sum + w[i]
        if w_new <= 0.0:
            w_sum = w_new
            continue
        a = lam[i] * w_sum / w_new
        b = w[i] / w_new
        d = X[i] - m
        for r in range(k):
            for s in range(k):
                c[r, s] = a * c[r, s] + a * b * d[r] * d[s]
        m = m + b * (X[i] - m)
        scale = 1.0 if a <= 0.0 else scale * a
        w_sum = w_new
    return states


def mahal_oracle(x, m, c, scale, prior):
    k = len(x)
    M = c + prior * scale * np.eye(k)
    d = x - m
    sol = np.linalg.solve(M, d)
    return math.sqrt(max(0.0, float(d @ sol)))


def pca_oracle(c, r, prev=None):
    """Top-`r` eigenpairs of `c`, largest first, each loading vector signed
    for continuity with `prev` (its dot product non-negative) or, without
    one, with its largest-magnitude entry positive: the crate's convention."""
    w, v = np.linalg.eigh(c)
    order = np.argsort(w)[::-1][:r]
    eig = w[order]
    load = v[:, order].T.copy()
    for j in range(r):
        d = float(prev[2][j] @ load[j]) if prev is not None and j < len(prev[0]) else 0.0
        if d != 0.0:
            if d < 0:
                load[j] = -load[j]
            continue
        lead = int(np.argmax(np.abs(load[j])))
        if load[j, lead] < 0:
            load[j] = -load[j]
    return eig, float(np.trace(c)), load


def field(out, name, col="c"):
    return out[col].struct.field(name).to_numpy()


def frame(X, **extra):
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(X.shape[1])} | extra)


def spec(k, **kw):
    d = dict(
        features=[f"x{i}" for i in range(k)],
        stats=["mean", "mahal"],
        precision_prior=1e-6,
        halflife=NO_DECAY,
        min_periods=5.0,
    )
    d.update(kw)
    return po.spec.ew_cov("c", **d)


def gaussian(n, k, seed, *, chol=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    if chol is None:
        chol = np.tril(rng.uniform(0.2, 1.0, (k, k)))
        np.fill_diagonal(chol, np.arange(1, k + 1) / 2.0)
    return Z @ chol.T, chol @ chol.T


# ------------------------------------------------------------ the mahal oracle


class TestMahalOracle:
    def test_matches_the_replayed_precision_solve(self):
        n, k = 400, 4
        X, _ = gaussian(n, k, 1)
        prior = 1e-3
        out = po.ModelBank([spec(k, precision_prior=prior, halflife=50.0)]).fit_predict(frame(X))
        got = field(out, "mahal")
        lam = np.full(n, math.exp2(-1.0 / 50.0))
        lam[0] = 1.0
        states = replay(X, lam, np.ones(n), prior=prior)
        for i in range(n):
            m, c, scale, w_sum = states[i]
            if w_sum < 5.0:
                assert np.isnan(got[i]), i
                continue
            want = mahal_oracle(X[i], m, c, scale, prior)
            assert got[i] == pytest.approx(want, rel=1e-9, abs=1e-12), i
        # The means the crate reports are the replayed ones, bit for bit.
        for j in range(k):
            mine = np.array([s[0][j] for s in states])
            theirs = field(out, f"mean_x{j}")
            ok = np.isfinite(theirs)
            np.testing.assert_array_equal(theirs[ok], mine[ok])

    def test_on_a_messy_stream_with_weights_a_clock_and_skipped_rows(self):
        # Irregular clock with gaps past max_dclock, weights including zeros
        # and nulls, a null feature every 37th row (skipped: the state is
        # untouched and its clock delta folds into the next accepted row).
        n, k = 600, 3
        rng = np.random.default_rng(2)
        X, _ = gaussian(n, k, 3)
        t = np.cumsum(rng.integers(1, 4, n)).astype(float)
        t[200:] += 100.0  # one gap far past max_dclock
        w = rng.choice([0.0, 0.5, 1.0, 2.0], n)
        w_col = [None if i % 97 == 0 else float(w[i]) for i in range(n)]
        x1 = [None if i % 37 == 0 else float(X[i, 1]) for i in range(n)]
        df = frame(X).with_columns(
            x1=pl.Series(x1, dtype=pl.Float64),
            t=pl.Series(t),
            w=pl.Series(w_col, dtype=pl.Float64),
        )
        prior, halflife, max_dclock = 1e-2, 40.0, 10.0
        s = spec(
            k,
            precision_prior=prior,
            clock="t",
            max_dclock=max_dclock,
            halflife=halflife,
            weight="w",
            min_periods=3.0,
        )
        out = po.ModelBank([s]).fit_predict(df)
        got = field(out, "mahal")
        accepted = np.array([x1[i] is not None and w_col[i] is not None for i in range(n)])
        lam = np.ones(n)
        pending = 0.0
        prev_t = None
        for i in range(n):
            d = 0.0 if prev_t is None else min(t[i] - prev_t, max_dclock)
            prev_t = t[i]
            pending += d
            if accepted[i]:
                lam[i] = math.exp2(-(pending / halflife))
                pending = 0.0
        ww = np.array([0.0 if v is None else v for v in w_col])
        states = replay(X, lam, ww, prior=prior, accepted=accepted)
        checked = 0
        for i in range(n):
            m, c, scale, w_sum = states[i]
            if not accepted[i]:
                assert np.isnan(got[i]), "a skipped row has no output"
                continue
            if w_sum < 3.0:
                assert np.isnan(got[i]), i
                continue
            want = mahal_oracle(X[i], m, c, scale, prior)
            assert got[i] == pytest.approx(want, rel=1e-9, abs=1e-12), i
            checked += 1
        assert checked > 500

    def test_one_column_is_the_absolute_z_score(self):
        n = 300
        rng = np.random.default_rng(4)
        x = 3.0 + 2.0 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x})
        out = po.ModelBank(
            [spec(1, stats=["mean", "std", "mahal"], precision_prior=1e-12)]
        ).fit_predict(df)
        got = field(out, "mahal")
        z = np.abs((x - field(out, "mean_x0")) / field(out, "std_x0"))
        ok = np.isfinite(got)
        assert ok.sum() > 290
        np.testing.assert_allclose(got[ok], z[ok], rtol=1e-8)

    def test_is_read_before_the_row_and_equals_the_precision_quadratic_form(self):
        # The same state feeds `partial_corr` (the precision matrix) and
        # `mahal`; an out-of-sample shock on row 300 is scored against the
        # history that has not seen it yet.
        n, k = 400, 3
        X, _ = gaussian(n, k, 5)
        X[300] += 8.0
        out = po.ModelBank([spec(k, stats=["mahal"])]).fit_predict(frame(X))
        got = field(out, "mahal")
        assert got[300] > 10 * np.nanmedian(got)
        assert got[301] < got[300] / 3, (
            "the row after the shock is scored on a history that now includes it"
        )


# ---------------------------------------------------------- the pca oracle


class TestPcaOracle:
    @pytest.mark.parametrize("every", [1, 7])
    def test_matches_eigh_at_every_refresh(self, every):
        n, k, r = 300, 4, 2
        X, _ = gaussian(n, k, 6)
        halflife = 60.0
        s = spec(k, stats=["mean"], precision_prior=None, pca=r, pca_every=every, halflife=halflife)
        out = po.ModelBank([s]).fit_predict(frame(X))
        lam = np.full(n, math.exp2(-1.0 / halflife))
        lam[0] = 1.0
        states = replay(X, lam, np.ones(n), prior=1.0)
        # After row i's update the state is states[i + 1]; the refresh after
        # the update on the row that makes `since_pca == every` (counted
        # from the previous refresh, the first refresh as soon as n_eff
        # reaches min_periods) is what rows i+1.. are scored on.
        frozen = None
        since = 0
        checked = 0
        for i in range(n):
            m, c, scale, w_sum = states[i]
            if frozen is None:
                for j in range(r):
                    assert np.isnan(field(out, f"pc{j}_var")[i]), i
            else:
                eig, trace, load = frozen
                for j in range(r):
                    assert field(out, f"pc{j}_var")[i] == pytest.approx(eig[j], rel=1e-9)
                    assert field(out, f"pc{j}_share")[i] == pytest.approx(eig[j] / trace, rel=1e-9)
                    for col in range(k):
                        assert field(out, f"pc{j}_x{col}")[i] == pytest.approx(
                            load[j, col], abs=1e-9
                        ), (i, j, col)
                    score = float(load[j] @ (X[i] - m))
                    assert field(out, f"pc{j}_score")[i] == pytest.approx(score, abs=1e-9)
                checked += 1
            # Row i is learned; maybe refresh.
            m2, c2, _, w2 = states[i + 1] if i + 1 < n else (None, None, None, None)
            if m2 is None:
                break
            since += 1
            if w2 >= 5.0 and (frozen is None or since >= every):
                frozen = pca_oracle(c2, r, frozen)
                since = 0
        assert checked > 280

    def test_recovers_a_planted_three_factor_structure(self):
        # k = 12 columns driven by 3 latent factors of variances 9, 4, 1
        # plus isotropic noise 0.01: the top three components span the
        # factor loadings, their variances are the factor variances, the
        # rest is noise.
        n, k, f = 200_000, 12, 3
        rng = np.random.default_rng(7)
        Q, _ = np.linalg.qr(rng.standard_normal((k, f)))
        F = rng.standard_normal((n, f)) * np.array([3.0, 2.0, 1.0])
        X = F @ Q.T + 0.1 * rng.standard_normal((n, k))
        s = spec(k, stats=["mean"], precision_prior=None, pca=4, pca_every=1000, halflife=NO_DECAY)
        out = po.ModelBank([s]).fit_predict(frame(X))
        last = out["c"][-1]
        var = np.array([last[f"pc{j}_var"] for j in range(4)])
        np.testing.assert_allclose(var[:3], [9.0, 4.0, 1.0], rtol=0.03)
        assert var[3] < 0.02, "the fourth component is noise"
        share = np.array([last[f"pc{j}_share"] for j in range(4)])
        np.testing.assert_allclose(share[:3], var[:3] / (14.0 + 0.01 * k), rtol=0.03)
        V = np.array([[last[f"pc{j}_x{c}"] for c in range(k)] for j in range(3)])
        # Each loading lies in the factor subspace: its projection has norm 1.
        proj = V @ Q
        np.testing.assert_allclose(np.linalg.norm(proj, axis=1), 1.0, atol=0.01)
        # And the scores have the component variances.
        for j in range(3):
            sc = field(out, f"pc{j}_score")[-50_000:]
            assert np.var(sc) == pytest.approx(var[j], rel=0.05)

    def test_rotation_equivariance(self):
        # Rotating the columns rotates the loadings and leaves the eigenvalues,
        # the shares and the scores unchanged (up to the sign convention).
        n, k = 2000, 3
        X, _ = gaussian(n, k, 8)
        R, _ = np.linalg.qr(np.random.default_rng(9).standard_normal((k, k)))
        Y = X @ R.T
        s = spec(k, stats=["mean"], precision_prior=None, pca=k, pca_every=50)
        a = po.ModelBank([s]).fit_predict(frame(X))
        b = po.ModelBank([s]).fit_predict(frame(Y))
        for j in range(k):
            np.testing.assert_allclose(
                field(a, f"pc{j}_var"), field(b, f"pc{j}_var"), rtol=1e-8, equal_nan=True
            )
            va = np.column_stack([field(a, f"pc{j}_x{c}") for c in range(k)])
            vb = np.column_stack([field(b, f"pc{j}_x{c}") for c in range(k)])
            ok = np.isfinite(va[:, 0])
            rotated = va[ok] @ R.T
            # Same line; the sign convention may pick either direction.
            sign = np.sign(np.sum(rotated * vb[ok], axis=1))
            np.testing.assert_allclose(rotated * sign[:, None], vb[ok], atol=1e-7)
            sa, sb = field(a, f"pc{j}_score")[ok], field(b, f"pc{j}_score")[ok]
            np.testing.assert_allclose(sa * sign, sb, atol=1e-7)

    def test_sign_continuity_keeps_loadings_stable_across_refreshes(self):
        # Seed 10's covariance has two loadings of nearly equal size on the
        # first component that trade the lead as the moments drift; the
        # max-abs rule alone flips the vector there (measured: a dot product
        # of −0.99999 between consecutive refreshes). Continuity does not.
        n, k = 3000, 5
        X, _ = gaussian(n, k, 10)
        s = spec(k, stats=["mean"], precision_prior=None, pca=2, pca_every=1, halflife=200.0)
        out = po.ModelBank([s]).fit_predict(frame(X))
        V1 = np.column_stack([field(out, f"pc1_x{c}") for c in range(k)])[100:]
        lead = np.argmax(np.abs(V1), axis=1)
        assert len(set(lead.tolist())) > 1, (
            "the lead entry must actually change for this to test anything"
        )
        flips = np.flatnonzero(lead[1:] != lead[:-1])
        assert flips.size > 0
        for j in range(2):
            V = np.column_stack([field(out, f"pc{j}_x{c}") for c in range(k)])[100:]
            dots = np.sum(V[1:] * V[:-1], axis=1)
            assert dots.min() > 0.9, dots.min()
        # Where the lead traded places, max-abs signing would have flipped.
        V1_maxabs = V1 * np.sign(V1[np.arange(V1.shape[0]), lead])[:, None]
        assert (np.sum(V1_maxabs[1:] * V1_maxabs[:-1], axis=1)[flips] < -0.9).any()


# --------------------------------------------------- calibration at scale


class TestCalibration:
    def test_mahal_squared_is_chi_squared_on_gaussian_columns(self):
        n, k = 200_000, 8
        X, _ = gaussian(n, k, 11)
        s = spec(
            k, stats=["mahal"], mahal_quantiles=[0.5, 0.99], halflife=20_000.0, min_periods=50.0
        )
        out = po.ModelBank([s]).fit_predict(frame(X))
        d2 = field(out, "mahal")[10_000:] ** 2
        assert np.mean(d2) == pytest.approx(k, rel=0.02)
        assert np.mean(d2 > CHI2_99[k]) == pytest.approx(0.01, abs=0.002)
        # The running quantiles sit at the empirical ones.
        q99 = field(out, "mahal_q0.99")[-1]
        q50 = field(out, "mahal_q0.5")[-1]
        m = field(out, "mahal")[50:-1]
        assert q99 == pytest.approx(np.quantile(m[-100_000:], 0.99), rel=0.03)
        assert q50 == pytest.approx(np.quantile(m[-100_000:], 0.5), rel=0.02)
        assert q99 == pytest.approx(math.sqrt(CHI2_99[k]), rel=0.03)

    def test_injected_outliers_are_flagged(self):
        n, k = 100_000, 6
        X, _ = gaussian(n, k, 12)
        rng = np.random.default_rng(13)
        bad = rng.choice(np.arange(5_000, n), 500, replace=False)
        X[bad] += rng.standard_normal((500, k)) * 4.0
        s = spec(k, stats=["mahal"], mahal_quantiles=[0.99], halflife=NO_DECAY, min_periods=50.0)
        out = po.ModelBank([s]).fit_predict(frame(X))
        d2 = field(out, "mahal") ** 2
        flag = d2 > CHI2_99[k]
        assert flag[bad].mean() > 0.9
        clean = np.ones(n, bool)
        clean[bad] = False
        clean[:5_000] = False
        assert flag[clean].mean() < 0.02
        # The stream's own 0.99 quantile flags them too.
        q = field(out, "mahal_q0.99")
        own = field(out, "mahal") > q
        assert own[bad].mean() > 0.85

    def test_decayed_scores_follow_a_covariance_change(self):
        # Halfway the covariance rotates; a short halflife re-learns it and
        # the score distribution returns to chi-squared.
        n, k = 60_000, 4
        rng = np.random.default_rng(14)
        A, _ = gaussian(n // 2, k, 15)
        R, _ = np.linalg.qr(rng.standard_normal((k, k)))
        B = 2.5 * gaussian(n // 2, k, 16)[0] @ R.T
        X = np.vstack([A, B])
        s = spec(k, stats=["mahal"], halflife=500.0, min_periods=50.0)
        d2 = field(po.ModelBank([s]).fit_predict(frame(X)), "mahal") ** 2
        assert np.mean(d2[n // 2 : n // 2 + 200]) > 1.5 * k, "the switch is visible"
        assert np.mean(d2[-10_000:]) == pytest.approx(k, rel=0.1)


# ------------------------------------------------------------- the contract


class TestStreamContract:
    def _df(self, n=800, k=4, seed=17):
        X, _ = gaussian(n, k, seed)
        return frame(X)

    def _spec(self, **kw):
        d = dict(
            stats=["mean", "mahal"],
            precision_prior=1e-4,
            mahal_quantiles=[0.9],
            pca=2,
            pca_every=7,
            halflife=100.0,
        )
        d.update(kw)
        return spec(4, **d)

    def test_chunk_invariance_with_a_refresh_cadence(self):
        df = self._df()
        s = self._spec()
        one = po.ModelBank([s]).fit_predict(df).select("c").unnest("c")
        for size in (1, 7, 13, 100):
            bank = po.ModelBank([s])
            many = (
                pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
                .select("c")
                .unnest("c")
            )
            assert one.equals(many, null_equal=True), size

    def test_save_load(self, tmp_path):
        df = self._df()
        s = self._spec()
        a = po.ModelBank([s])
        a.fit_predict(df.slice(0, 401))  # mid-cadence: 401 % 7 != 0
        a.save(tmp_path / "c.state")
        b = po.ModelBank.load(tmp_path / "c.state", specs=[s])
        rest = df.slice(401)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_predict_reads_frozen_components_and_moves_nothing(self):
        df = self._df()
        s = self._spec()
        bank = po.ModelBank([s])
        bank.fit_predict(df.slice(0, 400))
        snap = bank.save_bytes()
        scored = bank.predict(df.slice(400))
        assert bank.save_bytes() == snap
        for j in range(2):
            assert np.ptp(field(scored, f"pc{j}_var")) == 0.0
            for c in range(4):
                assert np.ptp(field(scored, f"pc{j}_x{c}")) == 0.0
        assert np.ptp(field(scored, "mahal_q0.9")) == 0.0
        assert np.ptp(field(scored, "mean_x0")) == 0.0
        # Scores and distances still vary row by row: they are about the row.
        assert np.ptp(field(scored, "pc0_score")) > 0.1
        assert np.ptp(field(scored, "mahal")) > 0.1
        learned = bank.fit_predict(df.slice(400, 1))
        for name in ("mahal", "mahal_q0.9", "pc0_score", "pc1_var"):
            np.testing.assert_array_equal(field(scored, name)[:1], field(learned, name))

    def test_expression_equals_bank(self):
        df = self._df().with_columns(g=pl.Series(["p", "q"] * 400))
        s = self._spec(group="g")
        bank = po.ModelBank([s]).fit_predict(df).select("c").unnest("c")
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("x0")
                .online.ew_cov(
                    ["x1", "x2", "x3"],
                    stats=["mean", "mahal"],
                    precision_prior=1e-4,
                    mahal_quantiles=[0.9],
                    pca=2,
                    pca_every=7,
                    halflife=100.0,
                    min_periods=5.0,
                )
                .over("g")
            ).unnest("x0")
        assert bank.equals(expr, null_equal=True)

    def test_halflife_grid_names_every_slot(self):
        s = spec(
            2, halflife=[50.0, 500.0], stats=["mahal"], mahal_quantiles=[0.5], pca=1, pca_every=1
        )
        assert po.spec.output_fields(s) == [
            "mahal@h50",
            "mahal_q0.5@h50",
            "pc0_var@h50",
            "pc0_share@h50",
            "pc0_x0@h50",
            "pc0_x1@h50",
            "pc0_score@h50",
            "n_eff@h50",
            "mahal@h500",
            "mahal_q0.5@h500",
            "pc0_var@h500",
            "pc0_share@h500",
            "pc0_x0@h500",
            "pc0_x1@h500",
            "pc0_score@h500",
            "n_eff@h500",
        ]
        out = po.ModelBank([s]).fit_predict(self._df(k=2))
        assert field(out, "pc0_var@h50")[-1] != field(out, "pc0_var@h500")[-1]

    def test_the_runner_agrees_with_the_bank(self, tmp_path):
        df = self._df(n=500).with_columns(g=pl.Series(["p", "q", "r", "s", "t"] * 100))
        s = self._spec(group="g")
        want = po.ModelBank([s]).fit_predict(df)
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        df.write_parquet(src)
        po.run(input=str(src), output=str(dst), specs=[s])
        got = pl.read_parquet(dst)
        assert want.select("c").unnest("c").equals(got.select("c").unnest("c"), null_equal=True)

    def test_a_toml_config_carries_the_new_keys(self, tmp_path):
        df = self._df(n=300)
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        df.write_parquet(src)
        cfg = tmp_path / "bank.toml"
        cfg.write_text(
            f'input = "{src.as_posix()}"\n'
            f'output = "{dst.as_posix()}"\n'
            "\n[[specs]]\n"
            'name = "c"\n'
            'targets = ["x0"]\n'
            'features = ["x0", "x1", "x2", "x3"]\n'
            "halflife = 100.0\n"
            "min_periods = 5.0\n"
            "\n[specs.model]\n"
            'type = "ew_cov"\n'
            'stats = ["mean", "mahal"]\n'
            "precision_prior = 1e-4\n"
            "mahal_quantiles = [0.9]\n"
            "pca = 2\n"
            "pca_every = 7\n"
        )
        po.run(cfg)
        got = pl.read_parquet(dst).select("c").unnest("c")
        want = po.ModelBank([self._spec()]).fit_predict(df).select("c").unnest("c")
        assert got.equals(want, null_equal=True)


# ------------------------------------------------------------------- fields


class TestFields:
    def test_order_and_index(self):
        s = spec(
            3, stats=["mean", "mahal", "corr"], mahal_quantiles=[0.5, 0.99], pca=1, pca_every=1
        )
        assert po.spec.output_fields(s) == [
            "mean_x0",
            "mean_x1",
            "mean_x2",
            "mahal",
            "corr_x0_x1",
            "corr_x0_x2",
            "corr_x1_x2",
            "mahal_q0.5",
            "mahal_q0.99",
            "pc0_var",
            "pc0_share",
            "pc0_x0",
            "pc0_x1",
            "pc0_x2",
            "pc0_score",
            "n_eff",
        ]
        idx = po.spec.output_index(s)
        rows = {r["field"]: r for r in idx.to_dicts()}
        assert rows["mahal"]["kind"] == "mahal"
        assert rows["mahal"]["columns"] == ["x0", "x1", "x2"]
        assert rows["mahal_q0.99"]["kind"] == "mahal_q"
        assert rows["mahal_q0.99"]["quantile"] == 0.99
        assert rows["mahal_q0.5"]["quantile"] == 0.5
        assert rows["pc0_var"]["kind"] == "pc_var"
        assert rows["pc0_share"]["kind"] == "pc_share"
        assert rows["pc0_x1"]["kind"] == "pc_loading"
        assert rows["pc0_x1"]["columns"] == ["x1"]
        assert rows["pc0_score"]["kind"] == "pc_score"
        assert rows["pc0_score"]["columns"] == ["x0", "x1", "x2"]
        assert set(idx["dtype"].to_list()) == {"f64"}

    def test_absent_by_default(self):
        assert po.spec.output_fields(spec(2, stats=["mean"], precision_prior=None)) == [
            "mean_x0",
            "mean_x1",
            "n_eff",
        ]

    def test_kwargs_are_typed(self):
        from polars_online._kwargs import EwCovKwargs

        keys = EwCovKwargs.__annotations__
        assert {"mahal_quantiles", "pca", "pca_every"} <= set(keys)


# --------------------------------------------------------------- validation


class TestValidation:
    def test_mahal_needs_a_prior(self):
        with pytest.raises(ValueError, match="mahal needs `precision_prior`"):
            spec(2, precision_prior=None)

    def test_quantiles_need_mahal(self):
        with pytest.raises(ValueError, match='mahal_quantiles needs "mahal"'):
            spec(2, stats=["mean"], mahal_quantiles=[0.9])
        with pytest.raises(ValueError, match='mahal_quantiles needs "mahal"'):
            spec(2, stats=None, mahal_quantiles=[0.9])

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_quantile_levels_are_open_interval(self, bad):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            spec(2, mahal_quantiles=[0.5, bad])

    def test_quantile_levels_must_be_finite_floats(self):
        with pytest.raises(ValueError, match="must be finite"):
            spec(2, mahal_quantiles=[float("inf")])
        with pytest.raises(TypeError, match="mahal_quantiles must be"):
            spec(2, mahal_quantiles=0.9)

    def test_pca_bounds(self):
        with pytest.raises(ValueError, match="pca asks for 3 components of 2 features"):
            spec(2, pca=3)
        with pytest.raises(ValueError, match="pca_every must be >= 1"):
            spec(2, pca=1, pca_every=0)
        with pytest.raises(ValueError, match="pca_every needs `pca`"):
            spec(2, pca_every=3)
        with pytest.raises(ValueError, match="pca_every needs `pca`"):
            spec(2, pca=0, pca_every=3)
        with pytest.raises(ValueError, match="pca must be >= 0"):
            spec(2, pca=-1)
        with pytest.raises(TypeError, match="pca must be"):
            spec(2, pca=1.5)

    def test_pca_zero_is_off(self):
        s = spec(2, stats=["mean"], precision_prior=None, pca=0)
        assert po.spec.output_fields(s) == ["mean_x0", "mean_x1", "n_eff"]

    def test_hand_edited_dicts_are_checked_at_the_bank(self):
        s = spec(2)
        s["model"]["mahal_quantiles"] = [2.0]
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            po.ModelBank([s])
        s = spec(2, stats=["mean"], precision_prior=None, pca=1)
        s["model"]["pca"] = 5
        with pytest.raises(ValueError, match="5 components of 2 features"):
            po.ModelBank([s])

    def test_mahal_is_listed_as_a_statistic(self):
        with pytest.raises(ValueError, match="mean, var, std, cov, corr, partial_corr, mahal"):
            spec(2, stats=["nonsense"])


# --------------------------------------------------------------- edge cases


class TestEdgeCases:
    def test_single_column_pca_is_the_column(self):
        n = 200
        x = np.random.default_rng(18).standard_normal(n) * 3.0
        s = spec(1, stats=["var"], precision_prior=None, pca=1, pca_every=1)
        out = po.ModelBank([s]).fit_predict(pl.DataFrame({"x0": x}))
        ok = np.isfinite(field(out, "pc0_var"))
        assert ok.sum() > 190
        np.testing.assert_allclose(field(out, "pc0_x0")[ok], 1.0)
        np.testing.assert_allclose(field(out, "pc0_share")[ok], 1.0)
        # The variance in force was refreshed after the previous row; the
        # `var` field is read live. They agree one row apart.
        var, pcv = field(out, "var_x0"), field(out, "pc0_var")
        np.testing.assert_allclose(pcv[ok][1:], var[ok][1:], rtol=1e-12)

    def test_a_constant_column_is_a_zero_variance_direction(self):
        n = 300
        X, _ = gaussian(n, 2, 19)
        X[:, 1] = 7.0
        s = spec(2, stats=["mean"], precision_prior=None, pca=2, pca_every=1)
        out = po.ModelBank([s]).fit_predict(frame(X))
        assert field(out, "pc1_var")[-1] == pytest.approx(0.0, abs=1e-12)
        assert field(out, "pc0_share")[-1] == pytest.approx(1.0)
        assert abs(field(out, "pc0_x0")[-1]) == pytest.approx(1.0)
        assert field(out, "pc1_score")[-1] == pytest.approx(0.0, abs=1e-9)

    def test_an_all_constant_stream_has_nan_shares_and_a_finite_mahal(self):
        df = pl.DataFrame({"x0": [1.0] * 40, "x1": [2.0] * 40})
        s = spec(2, stats=["mahal"], precision_prior=1e-3, pca=1, pca_every=1)
        out = po.ModelBank([s]).fit_predict(df)
        assert np.isnan(field(out, "pc0_share")[-1]), "0/0 is NaN, not 0"
        assert field(out, "pc0_var")[-1] == 0.0
        assert field(out, "mahal")[-1] == 0.0, "a row at the mean is at distance 0"
        # An off-mean row on a degenerate history: the prior alone sets the
        # scale, so the distance is finite and large.
        d = po.ModelBank([s]).fit_predict(df.vstack(pl.DataFrame({"x0": [2.0], "x1": [2.0]})))
        last = field(d, "mahal")[-1]
        assert np.isfinite(last) and last > 10.0

    def test_zero_weight_rows_score_but_do_not_move_the_state(self):
        n, k = 300, 3
        X, _ = gaussian(n, k, 20)
        w = np.ones(n)
        w[150:170] = 0.0
        df = frame(X, w=w)
        s = spec(k, weight="w", pca=1, pca_every=1, precision_prior=1e-4)
        out = po.ModelBank([s]).fit_predict(df)
        for name in ("mean_x0", "pc0_var", "pc0_x0"):
            v = field(out, name)
            assert np.ptp(v[150:171]) == 0.0, name
        assert np.ptp(field(out, "mahal")[150:170]) > 0
        assert np.ptp(field(out, "pc0_score")[150:170]) > 0

    def test_quantiles_lag_the_score_by_one_row(self):
        n, k = 200, 2
        X, _ = gaussian(n, k, 21)
        s = spec(k, mahal_quantiles=[0.5], min_periods=2.0)
        out = po.ModelBank([s]).fit_predict(frame(X))
        m, q = field(out, "mahal"), field(out, "mahal_q0.5")
        first_score = int(np.argmax(np.isfinite(m)))
        first_q = int(np.argmax(np.isfinite(q)))
        assert first_q == first_score + 5, (
            "P² needs five scores, and the row's own is not one of them"
        )

    def test_large_k_runs_and_is_calibrated(self):
        n, k = 5_000, 40
        X, _ = gaussian(n, k, 22)
        s = spec(k, stats=["mahal"], pca=3, pca_every=100, halflife=NO_DECAY, min_periods=200.0)
        out = po.ModelBank([s]).fit_predict(frame(X))
        d2 = field(out, "mahal")[1000:] ** 2
        assert np.mean(d2) == pytest.approx(k, rel=0.05)
        share = [field(out, f"pc{j}_share")[-1] for j in range(3)]
        assert share[0] >= share[1] >= share[2] > 0

    def test_precision_prior_is_the_only_regularizer_and_fades(self):
        # A large prior shrinks every distance; its scale `s` decays as data
        # accumulates (`s = 1/n` without decay), so by the last row a prior
        # of 1e3 is a ridge of 1e3/400 = 2.5 on the diagonal, no more.
        n, k = 400, 3
        X, _ = gaussian(n, k, 23)
        small = field(po.ModelBank([spec(k, precision_prior=1e-9)]).fit_predict(frame(X)), "mahal")
        big = field(po.ModelBank([spec(k, precision_prior=1e3)]).fit_predict(frame(X)), "mahal")
        ok = np.isfinite(small)
        assert (big[ok] < small[ok]).all()
        lam = np.ones(n)
        m, c, scale, w_sum = replay(X, lam, np.ones(n), prior=1e3)[-1]
        assert scale == pytest.approx(1.0 / (n - 1), rel=1e-12)
        assert big[-1] == pytest.approx(mahal_oracle(X[-1], m, c, scale, 1e3), rel=1e-9)
        # With the ridge nearly gone the two agree.
        assert small[-1] == pytest.approx(mahal_oracle(X[-1], m, c, scale, 1e-9), rel=1e-9)
