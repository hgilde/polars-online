"""E46: `polars_online.gram`, the accumulators read back.

Every function here has to be held against the thing that already computes
the same quantity -- the model's own solve, the model's own lasso path, or
the Gram of the whole stream -- rather than against a second implementation
of the same formula, which would only test that two copies of my arithmetic
agree.

The one tolerance worth naming: `solve` and the model's `solve` do the same
algebra, but `faer`'s Cholesky and LAPACK's LU round differently in the last
place or two, so they agree to a few ulps rather than exactly. The tests
below measure the gap; it is ~1e-16 relative on a well-conditioned system,
and the assertions are set an order or two looser than that, not at 1e-3.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import gram as pg


def stream(n=4000, k=3, seed=0, collinear=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    if collinear and k >= 3:
        X[:, 2] = X[:, 0] + 1e-3 * rng.standard_normal(n)
    beta = np.array([2.0, -1.0, 0.5, 0.25, -0.75])[:k]
    y = X @ beta + 3.0 + 0.1 * rng.standard_normal(n)
    df = pl.DataFrame({f"x{i}": X[:, i] for i in range(k)}).with_columns(y=pl.Series(y))
    return df, beta


def fit(df, **kw):
    # `lam=1.0` is no decay at all, so an accumulator is the plain weighted
    # moments of the rows it saw and two shards share a weighting exactly.
    if "halflife" not in kw:
        kw.setdefault("lam", 1.0)
    kw.setdefault("min_periods", 5.0)
    features = kw.pop("features", [c for c in df.columns if c.startswith("x")])
    targets = kw.pop("targets", ["y"])
    spec = po.spec.ewridge("m", targets=targets, features=features, **kw)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank


class TestSolveIsTheModelsSolve:
    """`solve` has to be the fit the spec would report, not a fit."""

    @pytest.mark.parametrize("standardize", [False, True])
    @pytest.mark.parametrize("ridge", [0.0, 0.3, 5.0])
    def test_it_reproduces_the_models_coefficients(self, standardize, ridge):
        df, _ = stream()
        bank = fit(df, ridge=ridge, standardize=standardize, max_rows_between_solves=1)
        g = bank.gram("m")[0]
        model = bank.coef("m")["coef"].to_numpy()
        mine = pg.solve(g, ridge=ridge, standardize=standardize)
        assert mine == pytest.approx(model, rel=1e-12), np.max(np.abs(mine - model))

    def test_without_an_intercept_too(self):
        df, _ = stream()
        bank = fit(df, ridge=0.2, standardize=False, add_intercept=False, max_rows_between_solves=1)
        g = bank.gram("m")[0]
        assert pg.INTERCEPT not in g["columns"]
        assert pg.solve(g, ridge=0.2) == pytest.approx(bank.coef("m")["coef"].to_numpy(), rel=1e-12)

    def test_the_gram_is_ahead_of_a_stale_solve(self):
        """`gram()` is as of the last row; `coef()` is as of the last *solve*,
        which the spec's schedule decides. On a spec that solves rarely the
        two differ, and it is the Gram that is current -- worth knowing before
        reading a disagreement as a bug in one of them."""
        df, _ = stream()
        rare = fit(
            df,
            ridge=0.2,
            standardize=False,
            solve_every=1e9,
            max_rows_between_solves=4_000_000,
        )
        eager = fit(df, ridge=0.2, standardize=False, max_rows_between_solves=1)
        stale = rare.coef("m")["coef"].to_numpy()
        current = eager.coef("m")["coef"].to_numpy()
        assert np.max(np.abs(stale - current)) > 1e-3, "the fixture did not go stale"
        assert pg.solve(rare.gram("m")[0], ridge=0.2) == pytest.approx(current, rel=1e-12)

    def test_a_ridge_grid_is_one_call(self):
        df, _ = stream()
        bank = fit(df, standardize=True)
        g = bank.gram("m")[0]
        grid = [0.0, 0.01, 0.1, 1.0]
        rows = pg.solve(g, ridge=grid, standardize=True)
        assert rows.shape == (4, 4)
        for i, r in enumerate(grid):
            assert rows[i] == pytest.approx(pg.solve(g, ridge=r, standardize=True), rel=1e-12)
        # More ridge is less slope, monotonically.
        slopes = np.abs(rows[:, 1:]).sum(axis=1)
        assert list(slopes) == sorted(slopes, reverse=True)

    def test_a_grid_of_one_is_still_a_row(self):
        df, _ = stream(n=500)
        g = fit(df).gram("m")[0]
        assert pg.solve(g, ridge=[0.1]).shape == (1, 4)
        assert pg.solve(g, ridge=0.1).shape == (4,)

    def test_features_narrows_the_regression(self):
        df, _ = stream()
        bank = fit(df, standardize=False)
        g = bank.gram("m")[0]
        narrowed = pg.solve(g, ridge=1e-9, features=["x0", "x1"])
        # Zero where the column was left out, and equal to the same fit run
        # on the subset -- a marginal Gram is a sub-block of the joint one.
        assert narrowed[3] == 0.0
        sub = pg.subset(g, [pg.INTERCEPT, "x0", "x1"])
        assert pg.solve(sub, ridge=1e-9) == pytest.approx(narrowed[[0, 1, 2]], rel=1e-9)
        # And it is the fit a spec over those two features reports.
        two = fit(
            df,
            features=["x0", "x1"],
            ridge=1e-9,
            standardize=False,
            max_rows_between_solves=1,
        )
        assert narrowed[[0, 1, 2]] == pytest.approx(two.coef("m")["coef"].to_numpy(), rel=1e-9)

    def test_a_constant_column_is_dropped_not_singular(self):
        df, _ = stream(n=800, k=2)
        df = df.with_columns(flat=pl.lit(4.0))
        bank = fit(df, features=["x0", "x1", "flat"], standardize=True)
        g = bank.gram("m")[0]
        b = pg.solve(g, standardize=True)
        assert b[g["columns"].index("flat")] == 0.0
        assert np.isfinite(b).all()

    def test_the_intercept_is_not_a_feature(self):
        df, _ = stream(n=300)
        g = fit(df).gram("m")[0]
        with pytest.raises(ValueError, match="constant column, not a feature"):
            pg.solve(g, features=[pg.INTERCEPT, "x0"])

    def test_target_by_name_or_position(self):
        df, _ = stream(n=900, k=2)
        df = df.with_columns(z=3.0 - 2.0 * pl.col("x0"))
        bank = fit(df, targets=["y", "z"], standardize=False, max_rows_between_solves=1)
        g = bank.gram("m")[0]
        assert pg.solve(g, target="z") == pytest.approx(pg.solve(g, target=1), rel=0)
        # z is exactly linear in x0, so the fit finds it.
        b = pg.solve(g, target="z")
        assert b == pytest.approx([3.0, -2.0, 0.0], abs=1e-8)
        with pytest.raises(KeyError, match="no target 'nope'"):
            pg.solve(g, target="nope")
        with pytest.raises(IndexError, match="target 5 out of range"):
            pg.solve(g, target=5)


class TestLassoPathIsTheModelsPath:
    def test_it_reproduces_the_lasso_models_coefficients(self):
        df, _ = stream(n=3000, k=5, seed=2)
        lambdas = [0.5, 0.1, 0.02, 0.0]
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=[f"x{i}" for i in range(5)],
            lasso_path=lambdas,
            halflife=1e12,
            min_periods=5.0,
            max_rows_between_solves=1,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("m")[0]
        want = (
            bank.coef("m")
            .sort("lambda", descending=True, nulls_last=True)["coef"]
            .to_numpy()
            .reshape(len(lambdas), -1)
        )
        got = pg.lasso_path(g, lambdas)
        assert got == pytest.approx(want, abs=1e-9), np.max(np.abs(got - want))

    def test_a_bigger_lambda_is_a_sparser_fit(self):
        df, _ = stream(n=2000, k=5, seed=3)
        g = fit(df, standardize=True).gram("m")[0]
        path = pg.lasso_path(g, [2.0, 0.5, 0.05, 0.0])
        nonzero = [(np.abs(row[1:]) > 1e-12).sum() for row in path]
        assert nonzero == sorted(nonzero)
        assert nonzero[0] < nonzero[-1]

    def test_l1_ratio_zero_is_a_ridge(self):
        """Elastic net with no L1 is ridge on the standardized scale, which
        is exactly what `solve(standardize=True)` computes."""
        df, _ = stream(n=1500, k=3, seed=4)
        g = fit(df).gram("m")[0]
        (enet,) = pg.lasso_path(g, [0.4], l1_ratio=0.0, max_iter=20000, tol=1e-14)
        assert enet == pytest.approx(pg.solve(g, ridge=0.4, standardize=True), rel=1e-6)

    def test_penalty_weights_spare_a_column(self):
        df, _ = stream(n=2000, k=3, seed=5)
        g = fit(df).gram("m")[0]
        heavy = pg.lasso_path(g, [1.5])[0]
        spared = pg.lasso_path(g, [1.5], penalty_weights=[0.0, 1.0, 1.0])[0]
        assert abs(heavy[1]) < abs(spared[1]), "x0 should survive an exemption"
        assert spared[1] == pytest.approx(2.0, rel=0.2)

    def test_penalty_weights_must_match_the_features(self):
        df, _ = stream(n=300)
        g = fit(df).gram("m")[0]
        with pytest.raises(ValueError, match="one entry per feature"):
            pg.lasso_path(g, [0.1], penalty_weights=[1.0])


class TestMerge:
    def test_merging_a_split_gives_the_whole(self):
        """The claim: pooling shards is exact, not approximate."""
        df, _ = stream(n=3000, k=3, seed=6)
        whole = fit(df).gram("m")[0]
        worst = {}
        for parts in (2, 7, 100):
            grams = []
            for chunk in np.array_split(np.arange(df.height), parts):
                bank = fit(df[chunk.tolist()])
                grams.append(bank.gram("m")[0])
            merged = pg.merge(grams)
            for key in (
                "n_eff",
                "means",
                "comoments",
                "cross_moments",
                "target_means",
                "target_vars",
                "target_weights",
                "n_kish",
                "target_n_kish",
            ):
                got = np.asarray(merged[key], dtype=float)
                want = np.asarray(whole[key], dtype=float)
                assert got == pytest.approx(want, rel=1e-8), (parts, key)
                worst[key] = max(
                    worst.get(key, 0.0),
                    float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-300))),
                )
        # The point of the pairwise form: a hundred parts is no worse than
        # two. A running "sum of squares minus the square of the sum" would
        # lose a digit or more over the same split.
        assert max(worst.values()) < 1e-8, worst

    def test_a_merged_gram_solves_to_the_whole_streams_fit(self):
        df, _ = stream(n=2400, k=3, seed=7)
        whole = fit(df, ridge=1e-9, standardize=False, max_rows_between_solves=1)
        halves = [fit(df.head(1200)).gram("m")[0], fit(df.tail(1200)).gram("m")[0]]
        merged = pg.merge(halves)
        assert pg.solve(merged, ridge=1e-9) == pytest.approx(
            whole.coef("m")["coef"].to_numpy(), rel=1e-9
        )

    def test_two_halves_of_a_decayed_stream_are_not_the_stream(self):
        """The documented caveat, held as a test rather than only as prose:
        with a finite halflife each part's weights are relative to its own
        last row, so a naive merge over-weights the earlier one -- and the
        rescaling the docstring gives is what fixes it."""
        n, half = 2400, 1200
        df, _ = stream(n=n, k=2, seed=18)
        hl = 400.0
        lam = 0.5 ** (1.0 / hl)
        whole = fit(df, halflife=hl).gram("m")[0]
        early = fit(df.head(half), halflife=hl).gram("m")[0]
        late = fit(df.tail(n - half), halflife=hl).gram("m")[0]

        naive = pg.merge([early, late])
        # Both halves sit at their own steady state, so the pool is nearly
        # twice the weight the whole stream carries: the early rows have not
        # been aged by the 1200 clock units that passed after them.
        assert naive["n_eff"] > 1.7 * whole["n_eff"], "the early half is not yet aged"

        # The recipe: the early part is `half` clock units older, so its
        # weight sum decays by lam**half and its sum of squares by lam**(2*half).
        aged = dict(early)
        aged["n_eff"] = early["n_eff"] * lam**half
        aged["n_kish"] = early["n_kish"]  # scale-free: W and Q decay together
        aged["target_weights"] = np.asarray(early["target_weights"]) * lam**half
        fixed = pg.merge([aged, late])
        assert fixed["n_eff"] == pytest.approx(whole["n_eff"], rel=1e-9)
        assert fixed["means"] == pytest.approx(whole["means"], rel=1e-7)
        assert fixed["comoments"] == pytest.approx(whole["comoments"], rel=1e-6)

    def test_it_pools_groups(self):
        """The everyday use: per-group accumulators combined into the pooled
        one, without a second pass."""
        df, _ = stream(n=1200, k=2, seed=8)
        df = df.with_columns(gid=pl.Series(["a", "b", "c"] * 400))
        spec = po.spec.ewridge(
            "g", targets=["y"], features=["x0", "x1"], halflife=1e12, min_periods=3.0, group="gid"
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        pooled = pg.merge(bank.gram("g"))
        ungrouped = fit(df.drop("gid")).gram("m")[0]
        assert pooled["means"] == pytest.approx(ungrouped["means"], rel=1e-9)
        assert pooled["comoments"] == pytest.approx(ungrouped["comoments"], rel=1e-7)
        assert pooled["group"] is None and pooled["instance"] is None

    def test_one_part_is_itself_and_none_is_an_error(self):
        df, _ = stream(n=400)
        g = fit(df).gram("m")[0]
        assert pg.merge([g])["means"] == pytest.approx(g["means"], rel=0)
        with pytest.raises(ValueError, match="at least one Gram"):
            pg.merge([])

    def test_mismatched_parts_are_refused(self):
        df, _ = stream(n=400, k=3)
        a = fit(df, features=["x0", "x1"]).gram("m")[0]
        b = fit(df, features=["x0", "x2"]).gram("m")[0]
        with pytest.raises(ValueError, match="same columns"):
            pg.merge([a, b])

    def test_an_empty_part_contributes_nothing(self):
        df, _ = stream(n=600, k=2)
        whole = fit(df).gram("m")[0]
        empty = po.ModelBank(
            [po.spec.ewridge("m", targets=["y"], features=["x0", "x1"], halflife=1e12)]
        ).gram("m")
        assert empty == []  # nothing fed, no accumulator
        merged = pg.merge([fit(df).gram("m")[0], fit(df.head(0)).gram("m")[0]])
        assert merged["means"] == pytest.approx(whole["means"], rel=1e-12)


class TestReadingTheMatrix:
    def test_correlation_matches_numpy(self):
        df, _ = stream(n=2000, k=3, seed=9)
        g = fit(df).gram("m")[0]
        r = pg.correlation(g)
        want = np.corrcoef(df.select("x0", "x1", "x2").to_numpy().T)
        assert r[1:, 1:] == pytest.approx(want, rel=1e-6)
        # The intercept is constant: no correlation with anything.
        assert np.isnan(r[0]).all()

    def test_subset_is_a_sub_block(self):
        df, _ = stream(n=800, k=3)
        g = fit(df).gram("m")[0]
        s = pg.subset(g, ["x2", "x0"])
        assert s["columns"] == ["x2", "x0"]
        assert s["means"] == pytest.approx([g["means"][3], g["means"][1]], rel=0)
        assert s["comoments"][0, 1] == g["comoments"][3, 1]
        assert s["cross_moments"][0].tolist() == [
            g["cross_moments"][0][3],
            g["cross_moments"][0][1],
        ]
        assert s["targets"] == g["targets"]

    def test_subset_by_position_and_the_errors(self):
        df, _ = stream(n=300, k=2)
        g = fit(df).gram("m")[0]
        assert pg.subset(g, [0, 2])["columns"] == [pg.INTERCEPT, "x1"]
        with pytest.raises(KeyError, match="no column 'nope'"):
            pg.subset(g, ["nope"])
        with pytest.raises(IndexError, match="column 9 out of range"):
            pg.subset(g, [9])

    def test_vif_finds_the_collinear_pair(self):
        df, _ = stream(n=4000, k=3, seed=10, collinear=True)
        g = fit(df).gram("m")[0]
        v = pg.vif(g)
        assert len(v) == 3, "the intercept is not a regressor"
        assert v[0] > 1e4 and v[2] > 1e4, v
        assert v[1] < 1.1, v
        # The textbook identity: VIF_j = 1 / (1 - R2 of column j on the rest).
        X = df.select("x0", "x1", "x2").to_numpy()
        Xc = X - X.mean(0)
        others, target = Xc[:, [0, 1]], Xc[:, 2]
        resid = target - others @ np.linalg.lstsq(others, target, rcond=None)[0]
        r2 = 1 - resid @ resid / (target @ target)
        assert v[2] == pytest.approx(1 / (1 - r2), rel=1e-3)

    def test_condition_names_the_columns_in_the_dependency(self):
        df, _ = stream(n=4000, k=3, seed=11, collinear=True)
        g = fit(df).gram("m")[0]
        c = pg.condition(g)
        assert c["columns"] == g["columns"]
        assert c["kappa"] > 100
        assert list(c["singular_values"]) == sorted(c["singular_values"], reverse=True)
        # Every column's variance is fully accounted for across components.
        assert c["proportions"].sum(axis=0) == pytest.approx(np.ones(4), rel=1e-9)
        # Belsley's reading: the worst component carries most of x0 and x2 and
        # little of x1, which is exactly the dependency that was built in.
        worst = c["proportions"][-1]
        assert worst[1] > 0.5 and worst[3] > 0.5
        assert worst[2] < 0.1

    def test_condition_of_an_orthogonal_design_is_near_one(self):
        df, _ = stream(n=6000, k=3, seed=12)
        g = fit(df).gram("m")[0]
        assert pg.condition(g, features=["x0", "x1", "x2"])["kappa"] < 1.2


class TestCoefStats:
    def test_r2_and_residual_variance_against_the_rows(self):
        df, _ = stream(n=5000, k=3, seed=13)
        bank = fit(df, ridge=1e-12, standardize=False, max_rows_between_solves=1)
        g = bank.gram("m")[0]
        b = pg.solve(g, ridge=1e-12)
        st = pg.coef_stats(g, b)
        X = df.select("x0", "x1", "x2").to_numpy()
        y = df["y"].to_numpy()
        resid = y - (b[0] + X @ b[1:])
        assert st["resid_var"] == pytest.approx(resid.var(), rel=1e-6)
        assert st["r2"] == pytest.approx(1 - resid.var() / y.var(), rel=1e-6)
        assert st["n"] == pytest.approx(len(y), rel=1e-9)

    def test_standard_errors_against_the_textbook_formula(self):
        df, _ = stream(n=5000, k=3, seed=14)
        bank = fit(df, ridge=1e-12, standardize=False, max_rows_between_solves=1)
        g = bank.gram("m")[0]
        b = pg.solve(g, ridge=1e-12)
        st = pg.coef_stats(g, b)
        X = df.select("x0", "x1", "x2").to_numpy()
        y = df["y"].to_numpy()
        n, k = X.shape
        design = np.column_stack([np.ones(n), X])
        resid = y - design @ b
        s2 = resid @ resid / (n - k - 1)
        want = np.sqrt(np.diag(s2 * np.linalg.inv(design.T @ design)))
        assert st["se"][1:] == pytest.approx(want[1:], rel=1e-4)
        assert np.isnan(st["se"][0]), "the intercept's error needs the design, not the Gram"
        assert st["t"][1:] == pytest.approx(b[1:] / want[1:], rel=1e-4)

    def test_a_real_coefficient_and_a_noise_one(self):
        n = 4000
        rng = np.random.default_rng(15)
        df = pl.DataFrame(
            {"x0": rng.standard_normal(n), "noise": rng.standard_normal(n)}
        ).with_columns(y=2.0 * pl.col("x0") + pl.Series(rng.standard_normal(n)))
        bank = fit(df, features=["x0", "noise"], ridge=1e-12, standardize=False)
        g = bank.gram("m")[0]
        st = pg.coef_stats(g, pg.solve(g, ridge=1e-12))
        assert abs(st["t"][1]) > 50, "the real one"
        assert abs(st["t"][2]) < 4, "the noise one"

    def test_weights_are_not_a_sample_size(self):
        """Ten times the weight on every row is the same information, so the
        standard errors must not shrink -- which they would if `n_eff` were
        used where `n_kish` is."""
        df, _ = stream(n=2000, k=2, seed=16)
        plain = fit(df.with_columns(w=pl.lit(1.0)), weight="w", ridge=1e-12, standardize=False)
        heavy = fit(df.with_columns(w=pl.lit(10.0)), weight="w", ridge=1e-12, standardize=False)
        a, b = plain.gram("m")[0], heavy.gram("m")[0]
        assert b["n_eff"] == pytest.approx(10 * a["n_eff"], rel=1e-9)
        sa = pg.coef_stats(a, pg.solve(a, ridge=1e-12))
        sb = pg.coef_stats(b, pg.solve(b, ridge=1e-12))
        assert sb["n"] == pytest.approx(sa["n"], rel=1e-9)
        assert sb["se"][1:] == pytest.approx(sa["se"][1:], rel=1e-9)

    def test_it_refuses_a_gram_that_cannot_answer(self):
        df, _ = stream(n=300, k=2)
        g = dict(fit(df).gram("m")[0])
        g["target_vars"] = None
        with pytest.raises(ValueError, match="no target moments"):
            pg.coef_stats(g, np.zeros(3))

    def test_it_checks_the_coefficient_count(self):
        df, _ = stream(n=300, k=2)
        g = fit(df).gram("m")[0]
        with pytest.raises(ValueError, match="one entry per Gram column"):
            pg.coef_stats(g, [1.0, 2.0])


class TestItWorksOnEveryGramItIsGiven:
    def test_an_ew_cov_gram_has_no_targets(self):
        df, _ = stream(n=900, k=3)
        spec = po.spec.ew_cov("c", features=["x0", "x1", "x2"], stats=[], halflife=1e12)
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("c")[0]
        assert g["columns"] == ["x0", "x1", "x2"] and g["targets"] == []
        # Everything that does not need a target still works.
        assert pg.correlation(g).shape == (3, 3)
        assert len(pg.vif(g)) == 3
        assert pg.condition(g)["kappa"] > 0
        assert pg.subset(g, ["x1"])["columns"] == ["x1"]
        assert pg.merge([g, g])["n_eff"] == pytest.approx(2 * g["n_eff"], rel=1e-9)
        with pytest.raises(IndexError, match="target 0 out of range"):
            pg.solve(g)

    def test_a_loaded_state_is_the_same_gram(self, tmp_path):
        df, _ = stream(n=1000, k=3, seed=17)
        bank = fit(df, standardize=False, max_rows_between_solves=1)
        bank.save(tmp_path / "b.state")
        loaded = po.ModelBank.load(tmp_path / "b.state")
        assert pg.solve(loaded.gram("m")[0]) == pytest.approx(pg.solve(bank.gram("m")[0]), rel=0)

    @pytest.mark.parametrize(
        ("label", "kw"),
        [
            ("ridge", {}),
            ("no intercept", {"add_intercept": False}),
            ("feature sets", {"feature_sets": {"a": ["x0"], "b": ["x1", "x2"]}}),
        ],
    )
    def test_the_axes_are_named_for_every_shape_of_spec(self, label, kw):
        """`columns` and `targets` are derived from the spec, not from the
        matrix, so they have to agree with it however the spec is shaped --
        an intercept or not, a feature-set grid or not."""
        df, _ = stream(n=600, k=3, seed=19)
        g = fit(df, **kw).gram("m")[0]
        k = len(g["means"])
        assert len(g["columns"]) == k
        assert g["comoments"].shape == (k, k)
        assert len(g["targets"]) == len(g["cross_moments"]) == len(g["target_weights"])
        assert (pg.INTERCEPT in g["columns"]) is kw.get("add_intercept", True)
        assert pg.solve(g).shape == (k,)

    def test_a_lasso_gram_names_its_axes_too(self):
        df, _ = stream(n=800, k=2, seed=20)
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            lasso_path=[0.1, 0.0],
            lam=1.0,
            min_periods=3.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        g = bank.gram("m")[0]
        assert g["columns"] == [pg.INTERCEPT, "x0", "x1"]
        assert g["targets"] == ["y"]
        assert pg.lasso_path(g, [0.05]).shape == (1, 3)

    def test_a_gram_from_a_dict_without_columns_says_so(self):
        with pytest.raises(KeyError, match="no 'columns'"):
            pg.vif({"comoments": np.eye(2), "means": np.zeros(2)})

    def test_the_namespace_is_the_documented_one(self):
        assert po.gram is pg
        assert sorted(pg.__all__) == pg.__all__
        for name in pg.__all__:
            assert getattr(pg, name).__doc__, name
