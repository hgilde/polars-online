"""Pooling Grams within parts (docs/PLAN.md task 240): ``po.gram.merge(...,
within=True)`` and ``solve_subsets(..., within=True)``.

The within merge is held to the Gram of the rows demeaned part by part, read
into memory with numpy; its fit to statsmodels' OLS with a dummy per part,
the textbook fixed-effects regression, for the coefficients, their standard
errors and the degrees of freedom; and ``solve_subsets``' merges in Rust to a
loop of ``merge(within=True)`` and the single-Gram fits.

The blocks' levels move together on purpose: each block's mean of ``y``
follows its mean of ``x0``, so pooling the blocks as one union reads a slope
of ``x0`` the rows inside the blocks do not have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import gram as pg

TIER = "essential"

FEATURES = ["x0", "x1", "x2", "x3"]
BETA = np.array([0.5, -0.3, 0.0, 0.2])


def frame(n_blocks: int = 6, per: int = 300, seed: int = 0, absent: int | None = 3) -> pl.DataFrame:
    """Blocks of rows whose levels differ: each block's features sit at a
    level of their own, and its ``y`` at 4 times its level of ``x0``. ``y2``
    is missing on a share of the rows, and on every row of block
    ``absent``."""
    rng = np.random.default_rng(seed)
    n = n_blocks * per
    block = np.repeat(np.arange(n_blocks), per)
    level = 3.0 * rng.standard_normal((n_blocks, len(FEATURES)))
    x = rng.standard_normal((n, len(FEATURES))) + level[block]
    y = x @ BETA + 4.0 * level[block, 0] + 10.0 + rng.standard_normal(n)
    y2 = x[:, ::-1] @ BETA - 2.0 * level[block, 1] + 0.5 * rng.standard_normal(n)
    gaps = (np.arange(n) % 5 == 0) | (block == (-1 if absent is None else absent))
    return pl.DataFrame(
        {
            "block": block,
            **{f: x[:, j] for j, f in enumerate(FEATURES)},
            "y": y,
            "y2": np.where(gaps, np.nan, y2),
        }
    ).with_columns(pl.col("y2").fill_nan(None))


def blocks(df: pl.DataFrame, **gram) -> list[dict[str, Any]]:
    spec = po.spec.ewridge(
        "m",
        targets=["y", "y2"],
        features=FEATURES,
        half_life=float("inf"),
        group="block",
        min_weight=5.0,
        target_gaps="pairwise",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m", **gram)


def demeaned(df: pl.DataFrame) -> dict[str, Any]:
    """The moments of the rows demeaned block by block, two-pass in numpy:
    the co-moments of the features less their block means; for each target,
    on its rows, the features less their block means there and the target
    less its own; and per target the blocks it has rows in."""
    x = df.select(FEATURES).to_numpy()
    b = df["block"].to_numpy()
    dx = x - np.stack([x[b == v].mean(axis=0) for v in b])
    out: dict[str, Any] = {"comoments": dx.T @ dx / len(dx), "weight_sum": float(len(dx))}
    for name in ("y", "y2"):
        y = df[name].to_numpy().astype(float)
        own = ~np.isnan(y)
        dxt, dy = np.zeros((0, len(FEATURES))), np.zeros(0)
        for v in np.unique(b[own]):
            rows = own & (b == v)
            dxt = np.vstack([dxt, x[rows] - x[rows].mean(axis=0)])
            dy = np.concatenate([dy, y[rows] - y[rows].mean()])
        out[name] = {
            "cross_centred": dxt.T @ dy / len(dy),
            "var": dy @ dy / len(dy),
            "weight": float(len(dy)),
            "parts": len(np.unique(b[own])),
        }
    return out


def close(got, want, tol: float = 1e-12) -> None:
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    scale = 1.0 + np.abs(want).max()
    assert np.abs(got - want).max() <= tol * scale, np.abs(got - want).max() / scale


class TestMergeWithin:
    def test_it_is_the_gram_of_the_rows_demeaned_per_part(self):
        df = frame()
        g = pg.merge(blocks(df), within=True)
        want = demeaned(df)
        assert g["weight_sum"] == want["weight_sum"]
        close(g["comoments"][1:, 1:], want["comoments"])
        assert np.array_equal(g["comoments"][0], np.zeros(5)), "the intercept's row is 0"
        assert np.array_equal(g["means"], [1.0, 0, 0, 0, 0]), "the demeaned rows' means"
        assert np.array_equal(g["means_by_target"], np.tile(g["means"], (2, 1)))
        assert np.array_equal(g["target_means"], [0.0, 0.0])
        for t, name in enumerate(["y", "y2"]):
            close(g["cross_centred"][t][1:], want[name]["cross_centred"])
            assert np.array_equal(g["cross_moments"][t], g["cross_centred"][t])
            assert g["cross_centred"][t][0] == 0.0
            close(g["target_vars"][t], want[name]["var"])
            assert g["target_weights"][t] == want[name]["weight"]
            assert g["target_n_kish"][t] == pytest.approx(want[name]["weight"], rel=1e-12)
        assert g["n_kish"] == pytest.approx(want["weight_sum"], rel=1e-12)
        assert g["within_parts"] == 6
        assert g["target_within_parts"].tolist() == [6, 5], "y2 has no rows in block 3"
        # Least squares on the demeaned rows: the slopes, and no intercept.
        b = pg.solve(g)
        assert np.isnan(b[0])
        x = df.select(FEATURES).to_numpy()
        blk = df["block"].to_numpy()
        means = np.stack([x[blk == v].mean(axis=0) for v in blk])
        y = df["y"].to_numpy()
        ymeans = np.array([y[blk == v].mean() for v in blk])
        lstsq = np.linalg.lstsq(x - means, y - ymeans, rcond=None)[0]
        close(b[1:], lstsq)

    def test_its_fit_is_statsmodels_ols_with_a_dummy_per_part(self):
        """The fixed-effects regression with its intercepts written out:
        the slopes, the standard errors and t with ``N - p - P`` degrees of
        freedom, and the residual variance."""
        import statsmodels.api as sm

        df = frame(seed=1)
        g = pg.merge(blocks(df), within=True)
        b = pg.solve(g)
        stats = pg.coef_stats(g, b)
        x = df.select(FEATURES).to_numpy()
        dummies = (df["block"].to_numpy()[:, None] == np.arange(6)).astype(float)
        ols = sm.OLS(df["y"].to_numpy(), np.hstack([x, dummies])).fit()
        assert b[1:] == pytest.approx(ols.params[:4], rel=1e-10)
        assert stats["se"][1:] == pytest.approx(ols.bse[:4], rel=1e-9)
        assert stats["t"][1:] == pytest.approx(ols.tvalues[:4], rel=1e-9)
        assert stats["sigma2"] == pytest.approx(ols.scale, rel=1e-9)
        assert ols.df_resid == df.height - 4 - 6
        assert np.isnan(stats["se"][0]) and np.isnan(stats["t"][0])
        # The union's fit reads the blocks' levels: x0's slope is pulled
        # toward the 4 the levels move by, where the rows inside the blocks
        # give 0.5.
        union = pg.solve(pg.merge(blocks(df)))
        assert abs(union[1] - 0.5) > 1.0 > 10 * abs(b[1] - 0.5)

    @pytest.mark.parametrize(
        "form",
        [dict(layout="packed"), dict(dtype="float32"), dict(dtype="float32", layout="packed")],
        ids=lambda f: "+".join(f.values()),
    )
    def test_every_form_merges_within_in_float64(self, form):
        df = frame(seed=2)
        whole = pg.merge(blocks(df), within=True)
        got = pg.merge(blocks(df, **form), within=True)
        packed = form.get("layout") == "packed"
        assert np.ndim(got["comoments"]) == (1 if packed else 2)
        assert got["comoments"].dtype == np.float64
        k = len(whole["columns"])
        if form == dict(layout="packed"):
            assert np.array_equal(got["comoments"], whole["comoments"][np.triu_indices(k)])
        else:
            np.testing.assert_allclose(
                pg._comoments(np, got), whole["comoments"], rtol=2e-7, atol=1e-6
            )
        for key in ("cross_centred", "target_vars", "target_weights", "target_within_parts"):
            assert np.array_equal(got[key], whole[key]), key

    def test_one_part_is_its_own_moments_about_its_means(self):
        part = blocks(frame(seed=3))[0]
        got = pg.merge([part], within=True)
        for key in ("comoments", "cross_centred", "target_vars", "target_weights", "target_n_kish"):
            assert np.array_equal(got[key], part[key]), key
        assert got["weight_sum"] == part["weight_sum"] and got["n_kish"] == part["n_kish"]
        assert np.array_equal(got["cross_moments"], part["cross_centred"])
        assert got["within_parts"] == 1 and got["target_within_parts"].tolist() == [1, 1]
        # The same slopes as the part's own fit, and no intercept.
        mine, its = pg.solve(got), pg.solve(part)
        assert np.array_equal(mine[1:], its[1:])
        assert np.isnan(mine[0]) and np.isfinite(its[0])
        # merge() of one part without `within` is still the part itself.
        assert pg.merge([part]) is not part and pg.merge([part]).keys() == part.keys()

    def test_parts_with_equal_means_pool_within_as_they_pool_together(self):
        """Where every gap between the parts' means is 0, the union adds
        nothing to the parts' moments, and the two merges are the same
        numbers, to the bit."""
        grams = blocks(frame(seed=4, absent=None))
        first = grams[0]
        same = [first] + [
            dict(
                g,
                means=first["means"],
                means_by_target=first["means_by_target"],
                target_means=first["target_means"],
                cross_moments=np.where(
                    np.arange(5) == 0, first["cross_moments"], g["cross_moments"]
                ),
            )
            for g in grams[1:]
        ]
        within, union = pg.merge(same, within=True), pg.merge(same)
        for key in (
            "weight_sum",
            "n_kish",
            "comoments",
            "cross_centred",
            "target_weights",
            "target_vars",
            "target_n_kish",
        ):
            assert np.array_equal(within[key], union[key]), key
        b_within, b_union = pg.solve(within), pg.solve(union)
        assert np.array_equal(b_within[1:], b_union[1:])

    def test_a_part_whose_target_is_absent_adds_to_the_comoments_and_not_to_the_target(self):
        grams = blocks(frame(seed=5))
        assert grams[3]["target_weights"][1] == 0.0, "y2 has no rows in block 3"
        every = pg.merge(grams, within=True)
        without = pg.merge(grams[:3] + grams[4:], within=True)
        assert every["weight_sum"] == without["weight_sum"] + grams[3]["weight_sum"]
        assert not np.allclose(every["comoments"], without["comoments"])
        for key in ("cross_centred", "target_vars", "target_weights", "target_n_kish"):
            assert np.array_equal(every[key][1], without[key][1]), key
        assert every["target_within_parts"].tolist() == [6, 5]
        assert without["target_within_parts"].tolist() == [5, 5]

    def test_a_part_pooled_within_pools_again_as_its_parts(self):
        grams = blocks(frame(seed=6))
        first = pg.merge(grams[:2], within=True)
        nested = pg.merge([first, *grams[2:]], within=True)
        flat = pg.merge(grams, within=True)
        for key, value in flat.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(nested[key], value), key
            else:
                assert nested[key] == value, key

    def test_a_union_of_several_refuses_a_part_pooled_within(self):
        grams = blocks(frame(seed=7))
        pooled = pg.merge(grams[:3], within=True)
        with pytest.raises(ValueError, match=r"part 0 was pooled within parts \(within_parts=3\)"):
            pg.merge([pooled, *grams[3:]])
        with pytest.raises(ValueError, match="pass within=True"):
            pg.solve_subsets([pooled, *grams[3:]], [[0, 1]])
        # Alone it is its own union, and beside others within it pools.
        assert pg.merge([pooled])["within_parts"] == 3
        assert pg.solve_subsets([pooled, *grams[3:]], [[0]])[0][0]["coef"][1:].all()


class TestTheFitsReadAWithinGram:
    def test_the_intercept_is_nan_and_the_slopes_are_the_within_ones(self):
        df = frame(seed=8)
        g = pg.merge(blocks(df), within=True)
        b = pg.solve(g, ridge=[0.0, 0.1], standardize=True)
        assert np.isnan(b[:, 0]).all() and np.isfinite(b[:, 1:]).all()
        # Without the intercept column the demeaned rows' raw moments are
        # their centred ones: the same slopes.
        origin = pg.subset(g, FEATURES)
        assert pg.solve(origin)[:] == pytest.approx(pg.solve(g)[1:], rel=1e-12)
        so, sg = pg.coef_stats(origin, pg.solve(origin)), pg.coef_stats(g, pg.solve(g))
        assert so["sigma2"] == pytest.approx(sg["sigma2"], rel=1e-12)
        lasso = pg.lasso_path(g, [0.1, 1e-6])
        assert np.isnan(lasso[:, 0]).all()
        assert lasso[1, 1:] == pytest.approx(pg.solve(g)[1:], rel=1e-4)
        path = pg.lars_path(g, target="y2")
        assert np.isnan(path["coef"][:, 0]).all()
        assert path["coef"][-1, 1:] == pytest.approx(pg.solve(g, target="y2")[1:], rel=1e-9)

    def test_a_degree_of_freedom_per_part_the_target_has_rows_in(self):
        g = pg.merge(blocks(frame(seed=9)), within=True)
        for t, parts in enumerate([6, 5]):
            b = pg.solve(g, target=t)
            st = pg.coef_stats(g, b, target=t)
            n = g["target_n_kish"][t]
            assert st["sigma2"] == pytest.approx(st["resid_var"] * n / (n - 4 - parts), rel=1e-14)

    def test_correlation_vif_and_condition_read_the_within_design(self):
        df = frame(seed=10)
        g = pg.merge(blocks(df), within=True)
        r = pg.correlation(g)[1:, 1:]
        want = demeaned(df)["comoments"]
        s = np.sqrt(np.diag(want))
        close(r, want / np.outer(s, s))
        assert np.isfinite(pg.vif(g)).all()
        cond = pg.condition(g)
        assert np.isfinite(cond["kappa"])


SUBSETS = [[0], [1, 2], [0, 2, 4], [5, 3, 1], [0, 1, 2, 3, 4, 5]]


class TestSolveSubsetsWithin:
    """``solve_subsets(..., within=True)`` against ``merge(within=True)``
    and the single-Gram fits on each subset: the merge is the same
    arithmetic, so a path is the same to the bit; a ridge is a Cholesky
    against numpy's eigendecomposition, held at ``1e-10``."""

    @pytest.mark.parametrize(
        "case",
        [dict(), dict(ridge=0.3, standardize=True), dict(features=["x3", "x1"])],
        ids=lambda c: ",".join(c) or "plain",
    )
    def test_each_ridge_is_merge_within_then_solve_and_coef_stats(self, case):
        grams = blocks(frame(seed=11))
        got = pg.solve_subsets(grams, SUBSETS, within=True, **case)
        for subset, per in zip(SUBSETS, got, strict=True):
            merged = pg.merge([grams[i] for i in subset], within=True)
            for t, fit in enumerate(per):
                coef = pg.solve(merged, target=t, **case)
                stats = pg.coef_stats(merged, coef, target=t, features=case.get("features"))
                assert np.isnan(fit["coef"][0]) and np.isnan(coef[0])
                close(fit["coef"][1:], coef[1:], 1e-10)
                for key in ("resid_var", "sigma2", "r2", "n"):
                    assert fit[key] == pytest.approx(stats[key], rel=1e-10, abs=1e-14), key
                for key in ("se", "t"):
                    assert np.array_equal(np.isnan(fit[key]), np.isnan(stats[key])), key
                    ok = ~np.isnan(stats[key])
                    np.testing.assert_allclose(fit[key][ok], stats[key][ok], rtol=1e-10)

    def test_a_path_is_lars_path_on_the_merge_within_to_the_bit(self):
        grams = blocks(frame(seed=12))
        got = pg.solve_subsets(grams, SUBSETS, path={"max_active": 3}, within=True)
        merged = [pg.merge([grams[i] for i in subset], within=True) for subset in SUBSETS]
        paths = pg.lars_paths(merged, max_active=3)
        for subset, per, want_per in zip(SUBSETS, got, paths, strict=True):
            for t, (path, want) in enumerate(zip(per, want_per, strict=True)):
                assert np.array_equal(path["coef"], want["coef"], equal_nan=True)
                assert np.isnan(path["coef"][:, 0]).all()
                assert np.array_equal(path["penalties"], want["penalties"])
                assert path["active"] == want["active"]
                single = pg.lars_path(merged[SUBSETS.index(subset)], target=t, max_active=3)
                assert np.array_equal(single["coef"], path["coef"], equal_nan=True)

    def test_a_subset_of_one_within_is_that_grams_slopes(self):
        grams = blocks(frame(seed=13))
        (fit, _) = pg.solve_subsets(grams, [[2]], within=True, ridge=0.1)[0]
        want = pg.solve(grams[2], ridge=0.1)
        assert np.isnan(fit["coef"][0])
        close(fit["coef"][1:], want[1:], 1e-12)
        plain = pg.coef_stats(grams[2], want)
        # One part: its own intercept, one degree of freedom, as before.
        assert fit["sigma2"] == pytest.approx(plain["sigma2"], rel=1e-10)

    def test_without_within_the_fits_are_the_unions(self):
        grams = blocks(frame(seed=14))
        union = pg.solve_subsets(grams, [[0, 1, 2]])[0][0]
        within = pg.solve_subsets(grams, [[0, 1, 2]], within=True)[0][0]
        assert np.isfinite(union["coef"][0]) and np.isnan(within["coef"][0])
        assert abs(union["coef"][1] - within["coef"][1]) > 0.1
