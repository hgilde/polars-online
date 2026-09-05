"""E30 and E45: the EW accumulators, exposed -- all of them.

The claim these tests have to earn is that `gram()` returns *the same matrices
the model solves against* -- not something merely similar. So they solve the
returned system by hand and check it reproduces the model's own coefficients,
and separately check the moments against a from-scratch weighted computation
over the same rows.

E45 (task 38) completes the export: `n_kish`, `target_means`, `target_vars`
and `target_n_kish`, which are what turns the Gram from half a sufficient
statistic into a whole one -- a residual variance, an R^2 or a standard error
needs `Var[y]`, and until task 38 the export had no way to give it.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po


def stream(n=4000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    beta = np.array([2.0, -1.0, 0.5])[:k]
    y = X @ beta + 3.0 + 0.1 * rng.standard_normal(n)
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(k)}).with_columns(y=pl.Series(y))
    return df, beta


class TestTheMatrixIsTheOneTheModelSolves:
    def test_hand_solving_the_gram_reproduces_the_models_coefficients(self):
        """The whole point of E30. No ridge, no standardization, so the normal
        equations are exactly `C beta = r` on the centered columns."""
        df, _ = stream()
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            ridge=1e-12,
            halflife=1e9,
            min_periods=5.0,
            max_rows_between_solves=1,
            standardize=False,
        )
        bank = po.ModelBank([spec])
        out = bank.fit_predict(df)
        g = bank.gram("m")[0]

        k = len(g["means"])
        # Intercept column is constant 1: zero variance, so drop it and
        # recover the intercept from the means, as the model does.
        # The documented bridge: the solve pairs the *raw* second moment with
        # the uncentered cross-moments. Using `comoments` directly here would
        # be silently wrong, which is why the identity is worth a test.
        raw = g["comoments"] + np.outer(g["means"], g["means"])
        beta = np.linalg.solve(raw, g["cross_moments"][0])

        model_coef = np.array(out["m"].struct.field("coef").to_list()[-1])
        # `coef` is read before the final row is folded in (out-of-sample), so
        # the accumulators are one row ahead of it.
        assert beta == pytest.approx(model_coef, rel=2e-3)
        assert k == len(model_coef)

    def test_moments_match_a_from_scratch_computation(self):
        """With halflife enormous, the EW moments are the plain unweighted
        ones, so numpy can check them directly."""
        df, _ = stream(n=2000, k=3, seed=1)
        spec = po.spec.ew_cov(
            "c", features=["x0", "x1", "x2"], stats=["cov"], halflife=1e12, min_periods=2.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]

        X = df.select("x0", "x1", "x2").to_numpy()
        assert g["means"] == pytest.approx(X.mean(axis=0), rel=1e-6)
        Xc = X - X.mean(axis=0)
        assert g["comoments"] == pytest.approx(Xc.T @ Xc / len(X), rel=1e-5)

    def test_uncentered_second_moment_identity(self):
        """The documented relation: raw = centered + outer(means, means)."""
        df, _ = stream(n=1500, k=3, seed=2)
        spec = po.spec.ew_cov(
            "c", features=["x0", "x1", "x2"], stats=["cov"], halflife=1e12, min_periods=2.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]
        X = df.select("x0", "x1", "x2").to_numpy()
        raw = g["comoments"] + np.outer(g["means"], g["means"])
        assert raw == pytest.approx(X.T @ X / len(X), rel=1e-5)


class TestShape:
    def test_one_entry_per_group_and_instance(self):
        df, _ = stream(n=900)
        df = df.with_columns(gid=pl.Series(["a", "b", "c"] * 300))
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=[100.0, 500.0],
            min_periods=3.0,
            group="gid",
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        rows = bank.gram("m")
        assert len(rows) == 6  # 3 groups x 2 halflives
        assert sorted({r["group"] for r in rows}) == ["a", "b", "c"]
        assert sorted({r["instance"] for r in rows}) == ["@h100", "@h500"]
        assert len(bank.gram("m", group="b")) == 2

    def test_multi_target_cross_moments(self):
        df, _ = stream()
        df = df.with_columns(z=2 * pl.col("y"))
        spec = po.spec.ewridge(
            "m", targets=["y", "z"], features=["x0", "x1"], halflife=1e9, min_periods=3.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("m")[0]
        assert g["cross_moments"].shape == (2, len(g["means"]))
        assert len(g["target_weights"]) == 2
        # z = 2y, so its cross-moments are exactly twice y's.
        assert g["cross_moments"][1] == pytest.approx(2 * g["cross_moments"][0], rel=1e-9)

    def test_ew_cov_has_no_cross_moments(self):
        df, _ = stream()
        spec = po.spec.ew_cov(
            "c", features=["x0", "x1"], stats=["corr"], halflife=100.0, min_periods=2.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]
        assert g["cross_moments"].shape == (0, 2)
        assert len(g["target_weights"]) == 0

    def test_lasso_reports(self):
        df, _ = stream()
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            lasso_path=[0.1, 0.0],
            halflife=1e9,
            min_periods=3.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        assert len(bank.gram("m")) == 1

    @pytest.mark.parametrize("model", ["rls", "kalman", "ftrl", "sgd", "pa", "huber"])
    def test_models_without_a_comoment_matrix_report_nothing(self, model):
        """Silence rather than a fabricated matrix: rls and kalman track an
        inverse, the gradient models track no second moment at all."""
        df, _ = stream(n=300)
        kw = dict(targets=["y"], features=["x0", "x1"], halflife=100.0, min_periods=3.0)
        if model == "kalman":
            kw["coef_halflife"] = 500.0
        spec = getattr(po.spec, model)("m", **kw)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        assert bank.gram("m") == []

    def test_spec_by_name_or_index(self):
        df, _ = stream(n=300)
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=100.0, min_periods=3.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        assert len(bank.gram(0)) == len(bank.gram("m")) == 1

    def test_it_tracks_the_stream_rather_than_a_snapshot(self):
        """n_eff grows with the stream: these are live accumulators, and
        reading them mid-stream is the point."""
        df, _ = stream(n=2000)
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=1e9, min_periods=3.0)
        bank = po.ModelBank([spec])
        seen = []
        for chunk in df.iter_slices(500):
            bank.fit_predict(chunk)
            seen.append(bank.gram("m")[0]["n_eff"])
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(2000, rel=1e-6)


def test_missing_numpy_says_what_to_do(monkeypatch):
    """numpy is an extra, not a dependency, so the failure has to be
    actionable rather than a bare ModuleNotFoundError from an inner import."""
    import builtins

    df, _ = stream(n=200)
    spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=100.0, min_periods=3.0)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)

    real_import = builtins.__import__

    def no_numpy(name, *a, **kw):
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    with pytest.raises(ModuleNotFoundError, match=r"polars-online\[numpy\]"):
        bank.gram("m")


class TestTheCompleteSufficientStatistic:
    """E45: the target moments and the Kish sizes."""

    def test_target_moments_match_a_from_scratch_computation(self):
        df, _ = stream(n=2000, k=3, seed=3)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            halflife=1e12,
            min_periods=3.0,
            standardize=False,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("m")[0]
        y = df["y"].to_numpy()
        assert g["target_means"] == pytest.approx([y.mean()], rel=1e-9)
        assert g["target_vars"] == pytest.approx([y.var()], rel=1e-7)
        # Unit weights, no nulls: the Kish size is the row count.
        assert g["n_kish"] == pytest.approx(len(y), rel=1e-9)
        assert g["target_n_kish"] == pytest.approx([len(y)], rel=1e-9)

    def test_a_targets_variance_is_the_one_ew_cov_reports_to_the_bit(self):
        """The same arithmetic, not merely the same number: `TargetMoments`
        takes the `a`/`b` the cross-moment update computed."""
        df, _ = stream(n=1200, k=2, seed=4)
        df = df.with_columns(w=pl.Series([0.5 + (i % 4) for i in range(df.height)]))
        common = dict(halflife=80.0, weight="w", min_periods=3.0)
        ridge = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], standardize=False, **common
        )
        cov = po.spec.ew_cov("c", features=["y"], stats=["var"], **common)
        bank = po.ModelBank([ridge, cov])
        bank.fit_predict(df)
        g, c = bank.gram("m")[0], bank.gram("c")[0]
        assert g["target_vars"][0] == c["comoments"][0, 0]
        assert g["target_means"][0] == c["means"][0]
        assert g["target_n_kish"][0] == c["n_kish"]

    def test_kish_is_a_row_count_where_n_eff_is_a_weight(self):
        n, lam = 6000, 0.5 ** (1 / 100.0)
        df, _ = stream(n=n, k=1, seed=5)
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=100.0, min_periods=3.0)
        heavy = po.spec.ewridge(
            "h", targets=["y"], features=["x0"], halflife=100.0, min_periods=3.0, weight="w"
        )
        bank = po.ModelBank([spec, heavy])
        bank.fit_predict(df.with_columns(w=pl.lit(7.0)))
        light, weighty = bank.gram("m")[0], bank.gram("h")[0]
        # The closed form for an exponentially weighted window of unit rows.
        assert light["n_kish"] == pytest.approx((1 + lam) / (1 - lam), rel=1e-6)
        # Seven times the weight is the same information.
        assert weighty["n_eff"] == pytest.approx(7 * light["n_eff"], rel=1e-9)
        assert weighty["n_kish"] == pytest.approx(light["n_kish"], rel=1e-9)

    def test_a_target_that_stops_arriving_keeps_its_sample_size(self):
        """`n_kish` is scale-free: pure decay scales `W` and `Q` together, so
        a stale target's moments still represent the rows they averaged. It is
        `target_weights` that collapses, and that is what says how old they
        are -- reading staleness off `n_kish` would read it off the wrong
        number."""
        n = 900
        df, _ = stream(n=n, k=2, seed=6)
        df = df.with_columns(
            y2=pl.when(pl.int_range(pl.len()) < n // 2).then(pl.col("y")).otherwise(None)
        )
        spec = po.spec.ewridge(
            "m", targets=["y", "y2"], features=["x0", "x1"], halflife=50.0, min_periods=3.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df.head(n // 2))
        at_the_break = bank.gram("m")[0]
        bank.fit_predict(df.tail(n - n // 2))
        g = bank.gram("m")[0]
        # Frozen where it stood. `W` and `Q` keep decaying, so their ratio
        # moves by rounding alone; the moments themselves are untouched and
        # so are bit-equal.
        assert g["target_n_kish"][1] == pytest.approx(at_the_break["target_n_kish"][1], rel=1e-12)
        assert g["target_means"][1] == at_the_break["target_means"][1]
        assert g["target_vars"][1] == at_the_break["target_vars"][1]
        # The weight behind it is what decayed away: 450 more rows at a
        # halflife of 50 is nine halvings, and nothing else touched it.
        assert g["target_weights"][1] == pytest.approx(
            at_the_break["target_weights"][1] / 2**9, rel=1e-9
        )
        assert g["target_weights"][1] < 0.01 * g["target_weights"][0]
        # The live target keeps counting, alongside the features.
        assert g["n_kish"] == pytest.approx(g["target_n_kish"][0], rel=1e-9)

    def test_the_residual_variance_the_docstring_promises(self):
        """The reason the target moments exist: R^2 from a saved Gram."""
        df, _ = stream(n=3000, k=3, seed=7)
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            ridge=1e-12,
            halflife=1e12,
            min_periods=5.0,
            standardize=False,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("m")[0]
        raw = g["comoments"] + np.outer(g["means"], g["means"])
        beta = np.linalg.solve(raw, g["cross_moments"][0])
        slopes = beta[1:]
        resid_var = g["target_vars"][0] - slopes @ g["comoments"][1:, 1:] @ slopes
        r2 = 1 - resid_var / g["target_vars"][0]
        X = df.select("x0", "x1", "x2").to_numpy()
        y = df["y"].to_numpy()
        want = np.corrcoef(X @ np.linalg.lstsq(X - X.mean(0), y - y.mean(), rcond=None)[0], y)[0, 1]
        assert r2 == pytest.approx(want**2, rel=1e-6)
        assert 0.99 < r2 < 1.0

    def test_lasso_reports_them_too(self):
        df, _ = stream(n=1500, k=2, seed=8)
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            lasso_path=[0.1, 0.0],
            halflife=1e12,
            min_periods=3.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("m")[0]
        y = df["y"].to_numpy()
        assert g["target_means"] == pytest.approx([y.mean()], rel=1e-9)
        assert g["target_vars"] == pytest.approx([y.var()], rel=1e-7)
        assert g["target_n_kish"] == pytest.approx([len(y)], rel=1e-9)

    def test_ew_cov_has_no_targets_so_the_lists_are_empty_not_missing(self):
        """Empty says "no targets"; None would say "this state cannot tell
        you", which is a different answer."""
        df, _ = stream(n=500, k=2)
        spec = po.spec.ew_cov("c", features=["x0", "x1"], stats=["var"], halflife=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]
        assert g["n_kish"] is not None
        assert len(g["target_means"]) == len(g["target_vars"]) == 0
        assert len(g["target_n_kish"]) == 0

    def test_nothing_learned_yet_has_no_sample_size(self):
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], halflife=100.0)
        bank = po.ModelBank([spec])
        bank.fit_predict(
            pl.DataFrame({"x0": [1.0], "y": [None]}, schema_overrides={"y": pl.Float64})
        )
        g = bank.gram("m")[0]
        assert g["n_kish"] == pytest.approx(1.0)
        assert np.isnan(g["target_n_kish"][0]), "the target saw no weighted row"

    def test_they_survive_a_save_and_load(self, tmp_path):
        df, _ = stream(n=800, k=2, seed=9)
        spec = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], halflife=200.0, min_periods=3.0
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        want = bank.gram("m")[0]
        bank.save(tmp_path / "b.msgpack")
        got = po.ModelBank.load(tmp_path / "b.msgpack").gram("m")[0]
        for key in ("n_kish", "target_means", "target_vars", "target_n_kish"):
            assert np.asarray(got[key]) == pytest.approx(np.asarray(want[key])), key

    def test_chunking_cannot_move_them(self):
        df, _ = stream(n=1000, k=2, seed=10)
        spec = po.spec.ewridge(
            "m", targets=["y"], features=["x0", "x1"], halflife=120.0, min_periods=3.0
        )
        one = po.ModelBank([spec])
        one.fit_predict(df)
        many = po.ModelBank([spec])
        for chunk in df.iter_slices(37):
            many.fit_predict(chunk)
        a, b = one.gram("m")[0], many.gram("m")[0]
        assert a["n_kish"] == b["n_kish"]
        for key in ("target_means", "target_vars", "target_n_kish"):
            assert list(a[key]) == list(b[key]), key
