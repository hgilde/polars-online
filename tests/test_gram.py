"""E30: the EW accumulators, exposed.

The claim these tests have to earn is that `gram()` returns *the same matrices
the model solves against* -- not something merely similar. So they solve the
returned system by hand and check it reproduces the model's own coefficients,
and separately check the moments against a from-scratch weighted computation
over the same rows.
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
