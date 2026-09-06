"""E1: `ew_cov` — EW moments of the feature columns (docs/PLAN.md 4.7).

Not a regression: no targets, no coefficients, just running statistics decayed
on the same clock as every model here.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def _df(n=3000, seed=0, rho=0.6):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    c = rng.standard_normal(n)
    return pl.DataFrame({"x0": a, "x1": b, "x2": c, "t": np.arange(float(n))})


#: `inf` is *exactly* no decay (the factor short-circuits to 1.0); 1e9 is not,
#: and over a few thousand rows the drift is ~1e-6, enough to break an exact
#: comparison against numpy.
NO_DECAY = float("inf")


def _spec(features=("x0", "x1"), **kw):
    d = dict(features=list(features), halflife=NO_DECAY, min_periods=5.0)
    d.update(kw)
    return po.spec.ew_cov("c", **d)


def _last(out, field):
    return out["c"].struct.field(field).to_list()[-1]


class TestStatistics:
    def test_recovers_mean_std_and_correlation(self):
        df = _df(rho=0.6)
        out = po.ModelBank([_spec()]).fit_predict(df)
        a, b = df["x0"].to_numpy(), df["x1"].to_numpy()
        # no decay => these are the ordinary sample statistics of all but the
        # final row (values are read before each row is folded in)
        assert _last(out, "mean_x0") == pytest.approx(a[:-1].mean(), abs=1e-9)
        assert _last(out, "std_x1") == pytest.approx(b[:-1].std(), abs=1e-9)
        assert _last(out, "corr_x0_x1") == pytest.approx(
            np.corrcoef(a[:-1], b[:-1])[0, 1], abs=1e-9
        )

    def test_cov_and_var_variants(self):
        df = _df()
        out = po.ModelBank([_spec(stats=["var", "cov"])]).fit_predict(df)
        a, b = df["x0"].to_numpy()[:-1], df["x1"].to_numpy()[:-1]
        assert _last(out, "var_x0") == pytest.approx(a.var(), abs=1e-9)
        assert _last(out, "cov_x0_x1") == pytest.approx(np.cov(a, b, bias=True)[0, 1], abs=1e-9)

    def test_all_pairs_are_emitted(self):
        spec = _spec(features=("x0", "x1", "x2"), stats=["corr"])
        assert po.spec.output_fields(spec) == [
            "corr_x0_x1",
            "corr_x0_x2",
            "corr_x1_x2",
            "n_eff",
        ]

    def test_correlation_is_bounded(self):
        # x0 with itself would be exactly 1; a near-duplicate must not exceed it
        df = _df().with_columns(x1=pl.col("x0") + 1e-15)
        out = po.ModelBank([_spec(stats=["corr"])]).fit_predict(df)
        vals = np.array([v for v in out["c"].struct.field("corr_x0_x1").to_list() if v is not None])
        assert (np.abs(vals) <= 1.0).all()

    def test_decay_tracks_a_regime_change(self):
        # correlation flips sign halfway; a short halflife must follow it
        rng = np.random.default_rng(3)
        n = 4000
        a = rng.standard_normal(n)
        b = np.concatenate([0.9 * a[: n // 2], -0.9 * a[n // 2 :]]) + 0.2 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": a, "x1": b})
        out = po.ModelBank([_spec(halflife=200.0, stats=["corr"])]).fit_predict(df)
        corr = out["c"].struct.field("corr_x0_x1").to_numpy().astype(float)
        assert corr[n // 2 - 10] > 0.8
        assert corr[-1] < -0.8

    def test_values_are_read_before_the_row(self):
        # A huge outlier must not inflate the std reported on its own row.
        df = _df(n=500)
        x = df["x0"].to_list()
        x[400] = 500.0
        df = df.with_columns(x0=pl.Series(x))
        out = po.ModelBank([_spec(stats=["std"])]).fit_predict(df)
        s = out["c"].struct.field("std_x0").to_numpy().astype(float)
        # Row 400's own value is not in row 400's statistic: it moves no more
        # than an ordinary row would (one extra point out of ~400).
        assert s[400] == pytest.approx(s[399], rel=0.02)
        # From the next row on, the outlier dominates.
        assert s[401] > s[400] * 5


class TestPlumbing:
    def test_warmup_and_null_policy(self):
        df = pl.DataFrame({"x0": [1.0, 2.0, None, 4.0, 5.0], "x1": [1.0, 3.0, 2.0, 4.0, 6.0]})
        out = po.ModelBank([_spec(min_periods=2.0)]).fit_predict(df)
        n_eff = out["c"].struct.field("n_eff").to_list()
        assert n_eff[2] is None, "a null feature must skip the row"
        assert out["c"].struct.field("mean_x0").to_list()[0] is None, "warmup"

    def test_chunk_invariance(self):
        df = _df(n=300)
        spec = _spec()
        one = po.ModelBank([spec]).fit_predict(df).select("c").unnest("c")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 31)) for i in range(0, df.height, 31)])
            .select("c")
            .unnest("c")
        )
        assert one.equals(many, null_equal=True)

    def test_save_load(self, tmp_path):
        df = _df(n=300)
        spec = _spec()
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 150))
        p = tmp_path / "c.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(150, 150)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_expression_equals_bank(self):
        df = _df(n=400).with_columns(g=pl.Series(["p", "q"] * 200))
        spec = _spec(group="g")
        bank = po.ModelBank([spec]).fit_predict(df).select("c").unnest("c")
        expr = df.select(
            pl.col("x0").online.ew_cov(["x1"], halflife=NO_DECAY, min_periods=5.0).over("g")
        ).unnest("x0")
        assert bank.equals(expr, null_equal=True)

    def test_groups_are_independent(self):
        df = _df(n=400).with_columns(g=pl.Series(["p", "q"] * 200))
        spec = _spec(group="g")
        both = po.ModelBank([spec]).fit_predict(df)
        solo = po.ModelBank([spec]).fit_predict(df.filter(pl.col("g") == "p"))
        a = both.filter(pl.col("g") == "p").select("c").unnest("c")
        assert a.equals(solo.select("c").unnest("c"), null_equal=True)

    def test_halflife_grid(self):
        df = _df(n=200)
        spec = _spec(halflife=[50.0, 500.0], stats=["corr"])
        fields = po.spec.output_fields(spec)
        assert fields == ["corr_x0_x1@h50", "n_eff@h50", "corr_x0_x1@h500", "n_eff@h500"]
        out = po.ModelBank([spec]).fit_predict(df)
        assert out["c"].struct.field("corr_x0_x1@h50").null_count() < df.height

    def test_bad_stat_is_rejected(self):
        with pytest.raises(ValueError, match="unknown ew_cov statistic"):
            _spec(stats=["nonsense"])

    def test_pairwise_stats_need_two_columns(self):
        with pytest.raises(ValueError, match="at least two features"):
            _spec(features=("x0",), stats=["corr"])


class TestPartialCorrelation:
    """E2: `partial_corr`, read off the regularized precision matrix.

    The matrix is solved from the co-moments on each row it is read (a
    Sherman-Morrison inverse used to be tracked; IMPROVEMENTS C5 explains why
    it is gone).
    """

    def _driver_data(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        d = rng.standard_normal(n)
        return pl.DataFrame(
            {
                "x0": d,
                "x1": d + 0.1 * rng.standard_normal(n),
                "x2": d + 0.1 * rng.standard_normal(n),
            }
        )

    def _last(self, df, **kw):
        spec = po.spec.ew_cov(
            "c",
            features=["x0", "x1", "x2"],
            halflife=NO_DECAY,
            min_periods=5.0,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df)["c"][-1]

    def test_removes_a_spurious_link(self):
        row = self._last(self._driver_data(), stats=["corr", "partial_corr"], precision_prior=1e-6)
        # x1 and x2 are both driven by x0, so they correlate marginally...
        assert row["corr_x1_x2"] > 0.9
        # ...but not once x0 is controlled for
        assert abs(row["pcorr_x1_x2"]) < 0.1
        # and each child keeps its genuine link to the driver
        assert abs(row["pcorr_x0_x1"]) > 0.5

    def test_keeps_a_direct_link(self):
        # A chain x0 -> x1 with x2 independent: pcorr(x0, x1) survives.
        rng = np.random.default_rng(2)
        n = 20000
        x0 = rng.standard_normal(n)
        df = pl.DataFrame(
            {
                "x0": x0,
                "x1": 2 * x0 + rng.standard_normal(n),
                "x2": rng.standard_normal(n),
            }
        )
        row = self._last(df, stats=["partial_corr"], precision_prior=1e-6)
        assert abs(row["pcorr_x0_x1"]) > 0.7
        assert abs(row["pcorr_x0_x2"]) < 0.1

    def test_is_bounded(self):
        row = self._last(self._driver_data(seed=3), stats=["partial_corr"], precision_prior=1e-6)
        for k, v in row.items():
            if k.startswith("pcorr_"):
                assert -1.0 <= v <= 1.0, f"{k} = {v}"

    def test_field_names(self):
        spec = po.spec.ew_cov(
            "c",
            features=["a", "b", "c"],
            stats=["partial_corr"],
            precision_prior=1e-6,
            halflife=NO_DECAY,
        )
        assert po.spec.output_fields(spec) == [
            "pcorr_a_b",
            "pcorr_a_c",
            "pcorr_b_c",
            "n_eff",
        ]

    def test_requires_a_precision_prior(self):
        with pytest.raises(ValueError, match="needs .precision_prior."):
            po.spec.ew_cov("c", features=["x0", "x1"], stats=["partial_corr"], halflife=NO_DECAY)

    def test_rejects_a_bad_prior(self):
        with pytest.raises(ValueError, match="precision_prior"):
            po.spec.ew_cov(
                "c",
                features=["x0", "x1"],
                stats=["partial_corr"],
                precision_prior=0.0,
                halflife=NO_DECAY,
            )

    def test_chunk_invariance_and_save_load(self, tmp_path):
        df = self._driver_data(n=400, seed=4)
        spec = po.spec.ew_cov(
            "c",
            features=["x0", "x1", "x2"],
            stats=["partial_corr"],
            precision_prior=1e-4,
            halflife=NO_DECAY,
            min_periods=5.0,
        )
        one = po.ModelBank([spec]).fit_predict(df).select("c").unnest("c")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 31)) for i in range(0, df.height, 31)])
            .select("c")
            .unnest("c")
        )
        assert one.equals(many, null_equal=True)

        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 200))
        p = tmp_path / "pc.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(200, 200)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)


class TestAccumulateOnly:
    """E43: ``stats=[]`` learns the same moments and emits nothing but ``n_eff``.

    The spec's value is its state, so every accessor that reads the state --
    ``gram``, ``describe``, ``summary`` -- must agree with a spec that also
    emitted, on every surface that runs a spec.
    """

    def _pair(self):
        bare = _spec(("x0", "x1", "x2"), stats=[])
        full = po.spec.ew_cov(
            "f", features=["x0", "x1", "x2"], stats=["mean", "corr"], halflife=NO_DECAY
        )
        return bare, full

    def test_emits_only_n_eff_and_keeps_the_same_state(self):
        df = _df(n=1200)
        bare, full = self._pair()
        assert po.spec.output_fields(bare) == ["n_eff"]
        bank = po.ModelBank([bare, full])
        out = bank.fit_predict(df)
        assert out.schema["c"] == pl.Struct({"n_eff": pl.Float64})
        assert out["c"].struct.field("n_eff").to_list()[-1] == df.height - 1
        g, f = bank.gram("c")[0], bank.gram("f")[0]
        assert np.array_equal(g["comoments"], f["comoments"])
        assert np.array_equal(g["means"], f["means"])
        assert g["n_eff"] == f["n_eff"] == df.height
        # And against numpy: the state is the product, so it is what is tested.
        x = df.select("x0", "x1", "x2").to_numpy()
        assert np.allclose(g["comoments"], np.cov(x, rowvar=False, bias=True))
        assert np.allclose(g["means"], x.mean(axis=0))
        desc = bank.describe("c")
        assert desc.height == 3 and desc["count"].to_list() == [df.height] * 3
        assert bank.summary("c")["rows_learned"].item() == df.height

    def test_every_surface_runs_it(self, tmp_path):
        df = _df(n=900)
        bare, _ = self._pair()
        ref = po.ModelBank([bare])
        one = ref.fit_predict(df).select("c").unnest("c")
        # Chunked, lazy, runner and expression: the same n_eff column, and the
        # same Gram wherever a state comes out.
        bank = po.ModelBank([bare])
        many = pl.concat([bank.fit_predict(df.slice(i, 101)) for i in range(0, df.height, 101)])
        assert many.select("c").unnest("c").equals(one)
        assert np.array_equal(bank.gram("c")[0]["comoments"], ref.gram("c")[0]["comoments"])
        lazy = df.lazy().online.fit_predict([bare]).collect().select("c").unnest("c")
        assert lazy.equals(one)
        src, dst, state = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "s.state"
        df.write_parquet(src)
        po.run(input=src, output=dst, specs=[bare], save_state=state)
        assert pl.read_parquet(dst).select("c").unnest("c").equals(one)
        loaded = po.ModelBank.load(state)
        assert np.array_equal(loaded.gram("c")[0]["comoments"], ref.gram("c")[0]["comoments"])
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("x0").online.ew_cov(["x1", "x2"], stats=[], halflife=NO_DECAY).alias("c")
            )
        assert expr.select("c").unnest("c").equals(one)

    def test_pca_and_mahal_stand_without_a_statistic(self):
        df = _df(n=600)
        spec = _spec(("x0", "x1", "x2"), stats=[], pca=1, pca_every=50)
        fields = po.spec.output_fields(spec)
        assert fields[0] == "pc0_var" and fields[-1] == "n_eff" and "mean_x0" not in fields
        out = po.ModelBank([spec]).fit_predict(df)
        assert out["c"].struct.field("pc0_score").drop_nulls().len() > 0
        with pytest.raises(ValueError, match='needs "mahal"'):
            _spec(("x0", "x1"), stats=[], mahal_quantiles=[0.9])


class TestLaggedComoments:
    """E56: `E_w[d_t d'_{t-l}]` beside the contemporaneous co-moments.

    The recursion centres **both** legs at the mean before the row, which is
    the choice that makes lag 0 coincide with `comoments`; the oracle below
    is that recursion written out, not some other definition of an EW lagged
    covariance.
    """

    @staticmethod
    def _oracle(x, lags, halflife, weights=None):
        """The recursion longhand: means, co-moments and lagged matrices."""
        lam = 1.0 if np.isinf(halflife) else 2.0 ** (-1.0 / halflife)
        n, k = x.shape
        w_sum = 0.0
        m = np.zeros(k)
        c = np.zeros((k, k))
        lag = {ell: np.zeros((k, k)) for ell in lags}
        ring: list[np.ndarray] = []
        for t in range(n):
            w = 1.0 if weights is None else float(weights[t])
            w_new = lam * w_sum + w
            if w_new <= 0.0:
                continue
            a, b = lam * w_sum / w_new, w / w_new
            d = x[t] - m
            for ell in lags:
                if len(ring) >= ell:
                    lag[ell] = a * lag[ell] + a * b * np.outer(d, ring[-ell] - m)
                else:
                    lag[ell] = a * lag[ell]
            c = a * c + a * b * np.outer(d, d)
            m = m + b * d
            w_sum = w_new
            if w > 0.0:
                ring.append(x[t].copy())
                ring = ring[-max(lags) :]
        return m, c, lag

    def test_the_recursion_is_the_longhand_one(self):
        df = _df(n=800)
        lags = [1, 3]
        spec = _spec(("x0", "x1", "x2"), lags=lags, halflife=200.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]
        x = df.select("x0", "x1", "x2").to_numpy()
        m, c, lag = self._oracle(x, lags, 200.0)
        assert np.allclose(g["means"], m, rtol=0, atol=1e-12)
        assert np.allclose(g["comoments"], c, rtol=0, atol=1e-12)
        assert g["lags"] == lags
        for i, ell in enumerate(lags):
            assert np.allclose(g["lag_comoments"][i], lag[ell], rtol=1e-12, atol=1e-14)

    def test_a_lagged_matrix_is_not_symmetric(self):
        """`b` is `a` one row back, so `C_1[b, a]` is the variance and
        `C_1[a, b]` is not. A symmetric implementation would pass every
        magnitude test and this one."""
        n = 600
        rng = np.random.default_rng(3)
        a = rng.standard_normal(n)
        df = pl.DataFrame({"x0": a, "x1": np.concatenate([[0.0], a[:-1]])})
        spec = _spec(("x0", "x1"), lags=[1], stats=["lagcorr"], halflife=150.0)
        out = po.ModelBank([spec]).fit_predict(df)
        assert _last(out, "lagcorr_x1_x0_l1") > 0.98
        assert abs(_last(out, "lagcorr_x0_x1_l1")) < 0.15

    def test_lags_leave_the_contemporaneous_moments_bit_identical(self):
        df = _df(n=700)
        cols = ("x0", "x1", "x2")
        plain = po.ModelBank([_spec(cols, halflife=200.0)])
        lagged = po.ModelBank([_spec(cols, lags=[1, 2, 4], halflife=200.0)])
        a = plain.fit_predict(df)["c"].struct.unnest()
        b = lagged.fit_predict(df)["c"].struct.unnest()
        assert a.equals(b), "the emitted statistics moved"
        ga, gb = plain.gram("c")[0], lagged.gram("c")[0]
        for key in ("n_eff", "n_kish"):
            assert ga[key] == gb[key]
        assert np.array_equal(ga["means"], gb["means"])
        assert np.array_equal(ga["comoments"], gb["comoments"])
        assert ga["lags"] is None and gb["lags"] == [1, 2, 4]

    @pytest.mark.parametrize("size", [1, 13, 300, 800])
    def test_chunk_invariance(self, size):
        df = _df(n=800)
        spec = _spec(("x0", "x1"), lags=[1, 5], stats=["lagcorr"], halflife=200.0)
        want = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        got = pl.concat([bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)])
        assert want.equals(got)

    def test_the_ring_clears_on_a_capped_gap_a_session_change_and_a_reset(self):
        """Task 47's events, and a fourth that is not one: a gap just under
        `max_dclock` leaves the ring alone."""
        n = 60
        rng = np.random.default_rng(5)
        base = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "t": np.arange(float(n)),
                "s": ["m"] * n,
            }
        )

        # `inf` is exactly no decay, so a clock gap changes *nothing* about
        # the numbers except through the ring: any difference below is the
        # clearing, not the decay.
        def run(df, **kw):
            kw.setdefault("max_dclock", 5.0)
            spec = _spec(
                ("x0", "x1"),
                lags=[1],
                stats=["lagcorr"],
                halflife=NO_DECAY,
                clock="t",
                min_periods=0.0,
                **kw,
            )
            out = po.ModelBank([spec]).fit_predict(df)
            # The row after the break is the one with no partner one row back,
            # so its lag matrix only aged: the value it reports is the level
            # before it, and the row after *that* is the first to pair again.
            return out["c"].struct.field("lagcorr_x0_x0_l1").to_list()

        plain = run(base)
        # A gap of 4 at row 30: under the ceiling, so nothing is cleared.
        under = base.with_columns(
            t=pl.when(pl.int_range(pl.len()) >= 30).then(pl.col("t") + 4.0).otherwise(pl.col("t"))
        )
        assert run(under) == plain, "a gap under max_dclock is not a break"
        # The same frame with a ceiling above the gap: also not a break, which
        # separates "the gap was capped" from "the gap was long".
        # A gap of 50: capped, so the ring goes.
        over = base.with_columns(
            t=pl.when(pl.int_range(pl.len()) >= 30).then(pl.col("t") + 50.0).otherwise(pl.col("t"))
        )
        assert run(over, max_dclock=100.0) == plain
        capped = run(over)
        assert capped[:30] == plain[:30]
        assert capped[31] != plain[31], "the row after a capped gap saw a stale partner"
        # A session change at the same row.
        sess = base.with_columns(
            s=pl.when(pl.int_range(pl.len()) >= 30).then(pl.lit("a")).otherwise(pl.lit("m"))
        )
        changed = run(sess, session="s", session_gap=1.0)
        assert changed[:30] == plain[:30]
        assert changed[31] != plain[31]
        # And a reset rebuilds the model, ring included.
        reset = run(sess, session="s", session_gap="reset")
        assert reset[31] != plain[31]

    def test_a_state_without_lags_resumes_under_a_spec_that_has_them(self):
        df = _df(n=200)
        bare = _spec(("x0", "x1"), halflife=200.0)
        bank = po.ModelBank([bare])
        bank.fit_predict(df[:100])
        data = bank.save_bytes()
        # The same spec plus lags: the moments come through, the ring starts
        # empty and the matrices from zero.
        with_lags = _spec(("x0", "x1"), lags=[1], halflife=200.0)
        # The saved specs are checked, so the resume is explicitly unchecked.
        resumed = po.ModelBank.load_bytes(data)
        assert resumed.gram("c")[0]["lags"] is None
        fresh = po.ModelBank([with_lags])
        fresh.fit_predict(df[:100])
        assert np.array_equal(fresh.gram("c")[0]["comoments"], resumed.gram("c")[0]["comoments"])

    def test_merge_reports_no_lags_and_subset_slices_them(self):
        df = _df(n=400)
        spec = _spec(("x0", "x1", "x2"), lags=[1, 2], halflife=1e9)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]
        pooled = po.gram.merge([g, g])
        assert pooled["lags"] is None and pooled["lag_comoments"] is None
        sub = po.gram.subset(g, ["x0", "x2"])
        assert sub["lag_comoments"].shape == (2, 2, 2)
        assert np.array_equal(
            sub["lag_comoments"][0], g["lag_comoments"][0][np.ix_([0, 2], [0, 2])]
        )

    def test_a_bad_lag_list_is_refused_by_name(self):
        for lags, message in (
            ([0, 1], "must be >= 1"),
            ([2, 1], "strictly increasing"),
            ([1, 1], "strictly increasing"),
        ):
            with pytest.raises(ValueError, match=message):
                _spec(("x0", "x1"), lags=lags)
        with pytest.raises(ValueError, match="lagcorr needs `lags`"):
            _spec(("x0", "x1"), stats=["lagcorr"])

    def test_the_closed_row_carries_the_lags(self):
        df = _df(n=200).with_columns(g=pl.int_range(pl.len()) // 100)
        spec = po.spec.ew_cov(
            "c",
            features=["x0", "x1"],
            halflife=1e9,
            lags=[1, 2],
            group="g",
            group_close="monotone",
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        row = bank.closed_groups()
        assert row["lags"][0].to_list() == [1, 2]
        assert len(row["lag_comoments"][0]) == 2 * 2 * 2
        g = po.gram.from_row(row)
        assert g["lags"] == [1, 2] and g["lag_comoments"].shape == (2, 2, 2)
