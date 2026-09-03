"""E1: `ew_cov` — EW moments of the feature columns (docs/PLAN.md 4.7).

Not a regression: no targets, no coefficients, just running statistics decayed
on the same clock as every model here.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from expr_plugin import requires_expr_plugin


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

    @requires_expr_plugin
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
