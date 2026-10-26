"""The Gram's offline fits in Rust (docs/PLAN.md task 226): ``lasso_path``,
``lars_path`` and ``lars_paths``.

``lasso_path`` moved from a Python loop to Rust with its signature kept, so
the Python loop it replaced is kept here as the reference it is held to
(:func:`python_lasso_path`, the 0.13.0 code verbatim). scikit-learn's second
opinions on both paths are in ``tests/test_second_opinion.py``
(``TestLarsPathIsSklearns``).
"""

from __future__ import annotations

import itertools
import re
import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import gram as pg

TIER = "essential"


def stream(n=3000, k=5, seed=0, constant=False, offset=0.0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    X = rng.standard_normal((n, k)) + 0.6 * base[:, None] + offset
    if constant:
        X[:, 1] = 4.0
    beta = np.array([1.5, -1.0, 0.6, 0.0, 0.3, 0.0, -0.4, 0.2][:k])
    y = X @ beta + 2.0 + 0.5 * rng.standard_normal(n)
    y2 = X[:, ::-1] @ beta - 1.0 + 0.5 * rng.standard_normal(n)
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(k)}).with_columns(
        y=pl.Series(y), y2=pl.Series(y2)
    )


def gram(df, targets=("y",), **kw) -> dict[str, Any]:
    kw.setdefault("lam", 1.0)
    features = [c for c in df.columns if c.startswith("x")]
    spec = po.spec.ewridge("m", targets=list(targets), features=features, min_weight=5.0, **kw)
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m")[0]


def python_lasso_path(
    g,
    penalties,
    *,
    l1_ratio=1.0,
    penalty_weights=None,
    target=0,
    features=None,
    max_iter=1000,
    tol=1e-10,
):
    """``po.gram.lasso_path`` as 0.13.0 ran it, in Python: the reference the
    Rust descent is held to. Its defaults follow the Rust function's (``tol``
    1e-10 since 1.0, the ``lasso`` model's; 0.13.0 ran 1e-7), and so does its
    stopping rule since task 278 (D3): a sweep's largest move against ``tol``
    times the target's spread -- its standard deviation with an intercept,
    its root mean square through the origin, never below the largest
    ``|d_i|`` -- or a sweep that moves nothing."""
    t = pg._target_index(g, target)
    slots, icept = pg._feature_slots(g, features)
    k = len(pg._columns(g))
    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = np.asarray(g["comoments"], dtype=float)
    m, ybar = means, 0.0
    # A mapping without target moments has no spread: the floor is the scale.
    tv, tm = g.get("target_vars"), g.get("target_means")
    var_y = np.nan if tv is None else float(np.asarray(tv, dtype=float)[t])
    mean_y = np.nan if tm is None else float(np.asarray(tm, dtype=float)[t])
    second = var_y if icept >= 0 else var_y + mean_y * mean_y
    spread = 0.0 if np.isnan(second) else max(second, 0.0) ** 0.5
    if icept >= 0:
        m = pg._means_of(np, g, t)
        ybar = cross[icept]
        c = como[np.ix_(slots, slots)]
        rhs = pg._cross_centered(np, g, t, m, ybar)[slots]
    else:
        c = como[np.ix_(slots, slots)] + np.outer(means[slots], means[slots])
        rhs = cross[slots]
    s = np.sqrt(np.clip(np.diag(c), 0.0, None))
    live = s > 0.0
    scale = np.where(live, s, 1.0)
    corr = c / np.outer(scale, scale)
    corr[~live, :] = 0.0
    corr[:, ~live] = 0.0
    corr[~live, ~live] = 1.0
    d = np.where(live, rhs / scale, 0.0)
    limit = tol * max(spread, float(np.abs(d).max(initial=0.0)))
    pw = np.ones(len(slots)) if penalty_weights is None else np.asarray(penalty_weights, float)
    out = np.zeros((len(penalties), k))
    b = np.zeros(len(slots))
    for li, lam in enumerate(penalties):
        l1, l2 = lam * l1_ratio * pw, lam * (1.0 - l1_ratio) * pw
        for _ in range(max_iter):
            delta = 0.0
            for i in range(len(slots)):
                if not live[i]:
                    b[i] = 0.0
                    continue
                rho = d[i] - (corr[i] @ b - corr[i, i] * b[i])
                new = np.sign(rho) * max(abs(rho) - l1[i], 0.0) / (corr[i, i] + l2[i])
                delta = max(delta, abs(new - b[i]))
                b[i] = new
            if delta < limit or delta == 0.0:
                break
        out[li, slots] = np.where(live, b / scale, 0.0)
        if icept >= 0:
            out[li, icept] = ybar - out[li, slots] @ m[slots]
    return out


PENALTIES = [0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]


class TestLassoPathIsTheLoopItReplaced:
    """The Rust descent runs the Python loop's sweeps in the same order, so
    the two agree to rounding: the sum over the other columns is the one
    thing ordered differently. Measured: within 4e-15 of the coefficients'
    size on every case here."""

    @pytest.mark.parametrize(
        "case",
        [
            dict(),
            dict(fit_intercept=False, offset=1.5),
            dict(constant=True),
            dict(l1_ratio=0.5),
            dict(penalty_weights=[1.0, 0.0, 2.0, 0.5, 1.0]),
            dict(features=["x3", "x0", "x2"]),
            dict(target="y2", offset=-3.0),
            dict(max_iter=3),
        ],
        ids=lambda c: ",".join(c) or "plain",
    )
    def test_it_is_the_python_loop_to_rounding(self, case):
        case = dict(case)
        data = {key: case.pop(key) for key in ("constant", "offset") if key in case}
        spec = {key: case.pop(key) for key in ("fit_intercept",) if key in case}
        g = gram(stream(seed=3, **data), targets=("y", "y2"), **spec)
        if "max_iter" in case:
            # Three sweeps leave penalties short of the rule, which is said.
            with pytest.warns(po.ReadinessWarning, match="ran out of sweeps"):
                got = pg.lasso_path(g, PENALTIES, **case)
        else:
            got = pg.lasso_path(g, PENALTIES, **case)
        want = python_lasso_path(g, PENALTIES, **case)
        scale = 1.0 + np.abs(want).max()
        assert np.abs(got - want).max() <= 4e-15 * scale, np.abs(got - want).max()

    def test_a_target_with_gaps_reads_its_own_means(self):
        df = stream(seed=4).with_columns(
            y2=pl.when(pl.int_range(pl.len()) % 3 == 0).then(None).otherwise(pl.col("y2"))
        )
        g = gram(df, targets=("y", "y2"), target_gaps="pairwise")
        assert not np.allclose(g["means_by_target"][0], g["means_by_target"][1])
        got = pg.lasso_path(g, PENALTIES, target="y2")
        want = python_lasso_path(g, PENALTIES, target="y2")
        assert np.abs(got - want).max() <= 4e-15 * (1.0 + np.abs(want).max())

    def test_a_mapping_without_target_moments_stops_on_the_largest_cross_moment(self):
        # The stopping rule's scale is the target's spread; without the
        # target's moments it is the largest `|d_i|`, no larger by
        # Cauchy-Schwarz, so the rule keeps the target's units. A `tol` at
        # which the two scales stop on other sweeps here (the target's
        # spread is 2.1, the largest `|d_i|` smaller), so they give two paths.
        g = gram(stream(seed=5))
        bare = dict(g, target_vars=None, target_means=None)
        got = pg.lasso_path(bare, PENALTIES, tol=1e-2)
        want = python_lasso_path(bare, PENALTIES, tol=1e-2)
        assert np.abs(got - want).max() <= 4e-15 * (1.0 + np.abs(want).max())
        assert np.abs(got - pg.lasso_path(g, PENALTIES, tol=1e-2)).max() > 1e-3

    def test_a_mapping_without_the_centred_forms_forms_them(self):
        g = gram(stream(seed=5))
        bare = dict(g, means_by_target=None, cross_centered=None)
        got = pg.lasso_path(bare, PENALTIES)
        assert np.abs(got - python_lasso_path(bare, PENALTIES)).max() < 1e-12
        assert got == pytest.approx(pg.lasso_path(g, PENALTIES), abs=1e-9)


class TestLarsPath:
    def test_each_knot_is_the_lasso_at_its_penalty(self):
        """Where both are defined: ``lasso_path`` at a knot's penalty, run to
        a tight ``tol``, is the knot's row."""
        g = gram(stream(seed=6, k=6))
        path = pg.lars_path(g)
        assert path["stop"] == "end"
        assert path["penalties"][-1] == 0.0
        assert np.all(np.diff(path["penalties"]) < 0.0)
        cd = pg.lasso_path(g, list(path["penalties"]), tol=1e-14, max_iter=100_000)
        assert np.abs(cd - path["coef"]).max() < 1e-9, np.abs(cd - path["coef"]).max()

    def test_the_first_knot_is_the_intercept_alone_and_the_last_is_least_squares(self):
        g = gram(stream(seed=7))
        path = pg.lars_path(g)
        first = path["coef"][0]
        assert np.all(first[1:] == 0.0)
        assert first[0] == pytest.approx(g["target_means"][0], rel=1e-12)
        assert len(path["active"][0]) == 1, "the first column enters at the first knot"
        assert path["coef"][-1] == pytest.approx(pg.solve(g), rel=1e-9)

    def test_active_names_columns_in_the_order_they_entered(self):
        g = gram(stream(seed=8))
        path = pg.lars_path(g)
        for before, after in zip(path["active"], path["active"][1:], strict=False):
            if len(after) > len(before):
                assert after[: len(before)] == before, (before, after)
        # A knot's nonzero coefficients are the set before it, less any
        # column leaving at it; the last knot's (l = 0) are its own set.
        sets = [set(a) for a in path["active"]]
        for n, row in enumerate(path["coef"]):
            nonzero = {c for c, v in zip(g["columns"][1:], row[1:], strict=True) if v != 0.0}
            if n == 0:
                want = set()
            elif n == len(sets) - 1:
                want = sets[n]
            else:
                want = sets[n - 1] & sets[n]
            assert nonzero == want, n

    def test_it_stops_after_max_iter_or_at_max_active(self):
        g = gram(stream(seed=9, k=8))
        full = pg.lars_path(g)
        steps = pg.lars_path(g, max_iter=3)
        assert steps["stop"] == "max_iter"
        assert len(steps["penalties"]) == 4
        assert np.array_equal(steps["coef"], full["coef"][:4])
        active = pg.lars_path(g, max_active=2)
        assert active["stop"] == "max_active"
        assert len(active["active"][-1]) == 2
        assert all(len(a) < 2 for a in active["active"][:-1])
        assert np.count_nonzero(active["coef"][-1][1:]) == 1, "the second enters at 0"
        assert np.array_equal(active["coef"], full["coef"][: len(active["coef"])])

    def test_columns_that_tie_enter_at_one_knot(self):
        """Two uncorrelated columns, equally correlated with the target, from
        rows the bank accumulated (task 236): both enter at the first knot,
        and the path ends at least squares. Taken one at a time, the second's
        step was 0 and it never entered."""
        n = 1000
        a = np.tile([1.0, 1.0, -1.0, -1.0], n // 4)
        b = np.tile([1.0, -1.0, 1.0, -1.0], n // 4)
        e = np.tile([1.0, -1.0, -1.0, 1.0], n // 4)
        df = pl.DataFrame({"x0": 0.1 * a, "x1": 0.1 * b, "y": 2.0 + 0.3 * (a + b) + 0.05 * e})
        g = gram(df)
        path = pg.lars_path(g)
        assert path["active"] == [["x0", "x1"], ["x0", "x1"]]
        assert path["penalties"][-1] == 0.0
        assert path["coef"][-1] == pytest.approx(pg.solve(g), rel=1e-12)
        assert path["coef"][-1][1:] == pytest.approx([3.0, 3.0], rel=1e-12)
        one = pg.lars_path(g, max_active=1)
        assert one["stop"] == "max_active"
        assert one["active"] == [["x0", "x1"]], "a tie passes max_active"

    def test_a_constant_column_never_enters(self):
        g = gram(stream(seed=10, constant=True))
        path = pg.lars_path(g)
        assert all("x1" not in a for a in path["active"])
        assert np.all(path["coef"][:, 2] == 0.0)

    def test_without_an_intercept_it_reads_the_raw_moments(self):
        g = gram(stream(seed=11, offset=2.0), fit_intercept=False)
        path = pg.lars_path(g)
        cd = pg.lasso_path(g, list(path["penalties"]), tol=1e-14, max_iter=100_000)
        assert np.abs(cd - path["coef"]).max() < 1e-9

    def test_penalty_weights_are_the_lasso_paths(self):
        g = gram(stream(seed=12))
        w = [1.0, 3.0, 0.5, 1.0, 2.0]
        path = pg.lars_path(g, penalty_weights=w)
        cd = pg.lasso_path(
            g, list(path["penalties"]), penalty_weights=w, tol=1e-14, max_iter=100_000
        )
        assert np.abs(cd - path["coef"]).max() < 1e-9

    def test_features_narrow_it(self):
        g = gram(stream(seed=13))
        narrow = pg.lars_path(g, features=["x2", "x0"])
        sub = pg.lars_path(pg.subset(g, ["intercept", "x2", "x0"]))
        assert np.array_equal(narrow["penalties"], sub["penalties"])
        assert np.array_equal(narrow["coef"][:, [0, 3, 1]], sub["coef"])
        assert narrow["active"] == sub["active"]

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            (dict(max_iter=0), "max_iter must be a whole number >= 1"),
            (dict(max_active=True), "max_active must be a whole number >= 1"),
            (dict(max_active=2.0), "max_active must be a whole number >= 1"),
            (dict(penalty_weights=[1.0, 0.0, 1.0, 1.0, 1.0]), "use lasso_path for it"),
            (dict(penalty_weights=[1.0]), "one entry per feature"),
            (dict(features=["intercept"]), "constant column"),
        ],
    )
    def test_it_refuses(self, kw, match):
        g = gram(stream(seed=14, n=200))
        with pytest.raises(ValueError, match=match):
            pg.lars_path(g, **kw)


class TestLarsPaths:
    def test_each_path_is_lars_paths_to_the_bit(self):
        df = stream(seed=15, k=6).with_columns(grp=pl.int_range(pl.len()) % 4)
        spec = po.spec.ewridge(
            "m",
            targets=["y", "y2"],
            features=[f"x{i}" for i in range(6)],
            lam=1.0,
            group="grp",
            min_weight=5.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        grams = bank.gram("m")
        assert len(grams) == 4
        many = pg.lars_paths(grams, max_iter=4)
        assert [len(per) for per in many] == [2, 2, 2, 2]
        for g, per in zip(grams, many, strict=True):
            for t, path in zip(["y", "y2"], per, strict=True):
                one = pg.lars_path(g, target=t, max_iter=4)
                assert np.array_equal(path["coef"], one["coef"])
                assert np.array_equal(path["penalties"], one["penalties"])
                assert path["active"] == one["active"]
        named = pg.lars_paths(grams, targets=["y2"], max_iter=4)
        assert [len(per) for per in named] == [1, 1, 1, 1]
        assert np.array_equal(named[2][0]["coef"], many[2][1]["coef"])

    def test_grams_with_different_targets_each_run_their_own(self):
        """Under ``target_gaps="own_rows"`` a group whose rows never hold
        ``y2`` has a Gram without it, and the others have both. The
        docstring's example runs on them, and each Gram's paths are its own
        targets', as documented (review 6, E-4: every Gram had to have the
        same targets, and the example was refused)."""
        rng = np.random.default_rng(3)
        n = 600
        x = rng.normal(size=(n, 2))
        block = np.repeat(["a", "b", "c"], 200)
        y2 = np.where(block == "a", np.nan, x @ [0.5, 0.5] + 0.1 * rng.normal(size=n))
        df = pl.DataFrame(
            {
                "x0": x[:, 0],
                "x1": x[:, 1],
                "y": x @ [1.0, -1.0] + 0.1 * rng.normal(size=n),
                "y2": y2,
                "stock_id": block,
            }
        ).with_columns(pl.col("y2").fill_nan(None))
        # The docstring's example, with a second target.
        spec = po.spec.ewridge(
            "ridge", targets=["y", "y2"], features=["x0", "x1"], half_life=100.0, group="stock_id"
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        grams = bank.gram("ridge")
        assert len({tuple(g["targets"]) for g in grams}) > 1, [g["targets"] for g in grams]
        paths = po.gram.lars_paths(bank.gram("ridge"), max_active=1)  # one list per group
        firsts = [per_gram[0]["active"][-1] for per_gram in paths]
        assert len(firsts) == len(grams)
        for g, per in zip(grams, paths, strict=True):
            assert len(per) == len(g["targets"])
            for t, path in zip(g["targets"], per, strict=True):
                one = pg.lars_path(g, target=t, max_active=1)
                assert np.array_equal(path["coef"], one["coef"])
        first = pg.lars_paths(grams, targets=[0], max_active=1)
        assert [len(per) for per in first] == [1] * len(grams)
        # A target one Gram lacks is named as missing there.
        lacking = next(t for t in ("y", "y2") if any(t not in g["targets"] for g in grams))
        with pytest.raises(KeyError, match=f"no target '{lacking}'"):
            pg.lars_paths(grams, targets=[lacking])

    def test_no_grams_is_no_paths_and_mismatched_columns_are_refused(self):
        assert pg.lars_paths([]) == []
        g = gram(stream(seed=16, n=200))
        other = pg.subset(g, ["intercept", "x0"])
        with pytest.raises(ValueError, match="same columns"):
            pg.lars_paths([g, other])


def blocks(n_blocks=6, seed=20, gaps=False, **kw) -> list[dict[str, Any]]:
    """One Gram per block of rows, ``half_life=inf`` so the blocks share a
    weighting and merge exactly."""
    df = stream(n=600 * n_blocks, seed=seed, offset=kw.pop("offset", 0.0)).with_columns(
        block=pl.int_range(pl.len()) // 600
    )
    if gaps:
        df = df.with_columns(
            y2=pl.when(pl.int_range(pl.len()) % 4 == 0).then(None).otherwise(pl.col("y2"))
        )
        kw.setdefault("target_gaps", "pairwise")
    spec = po.spec.ewridge(
        "m",
        targets=["y", "y2"],
        features=[c for c in df.columns if c.startswith("x")],
        half_life=float("inf"),
        group="block",
        min_weight=5.0,
        **kw,
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m")


SUBSETS = [[0], [1, 2], [0, 2, 4], [5, 3, 1], [0, 1, 2, 3, 4, 5]]


class TestSolveSubsets:
    """``solve_subsets`` against ``merge`` then ``solve`` and ``coef_stats``
    on each subset. The merge is the same arithmetic; the solves are a
    Cholesky against numpy's eigendecomposition and inverse, so they agree
    to the system's condition number times the rounding: measured within
    ``2e-13`` of each number's size here, held at ``1e-10``."""

    @pytest.mark.parametrize(
        "case",
        [
            dict(),
            dict(ridge=0.3, standardize=True),
            dict(ridge=0.05),
            dict(features=["x4", "x1"]),
            dict(fit_intercept=False, offset=2.0),
            dict(gaps=True, ridge=1e-6),
            dict(targets=["y2"]),
        ],
        ids=lambda c: ",".join(c) or "plain",
    )
    def test_each_subset_is_merge_then_solve_and_coef_stats(self, case):
        case = dict(case)
        make = {key: case.pop(key) for key in ("fit_intercept", "offset", "gaps") if key in case}
        grams = blocks(**make)
        got = pg.solve_subsets(grams, SUBSETS, **case)
        targets = case.pop("targets", ["y", "y2"])
        assert [len(per) for per in got] == [len(targets)] * len(SUBSETS)
        for subset, per in zip(SUBSETS, got, strict=True):
            merged = pg.merge([grams[i] for i in subset])
            for t, fit in zip(targets, per, strict=True):
                coef = pg.solve(merged, target=t, **case)
                stats = pg.coef_stats(merged, coef, target=t, features=case.get("features"))
                scale = 1.0 + np.abs(coef).max()
                assert np.abs(fit["coef"] - coef).max() <= 1e-10 * scale
                for key in ("resid_var", "sigma2", "r2", "n"):
                    assert fit[key] == pytest.approx(stats[key], rel=1e-10, abs=1e-14), key
                for key in ("se", "t"):
                    assert np.array_equal(np.isnan(fit[key]), np.isnan(stats[key])), key
                    ok = ~np.isnan(stats[key])
                    np.testing.assert_allclose(fit[key][ok], stats[key][ok], rtol=1e-10)

    def test_a_path_is_lars_path_on_the_merge_to_the_bit(self):
        grams = blocks()
        got = pg.solve_subsets(grams, SUBSETS, path={"max_active": 3})
        for subset, per in zip(SUBSETS, got, strict=True):
            merged = pg.merge([grams[i] for i in subset])
            for t, path in zip(["y", "y2"], per, strict=True):
                want = pg.lars_path(merged, target=t, max_active=3)
                assert np.array_equal(path["coef"], want["coef"])
                assert np.array_equal(path["penalties"], want["penalties"])
                assert path["active"] == want["active"]
                assert path["stop"] == want["stop"] == "max_active"

    def test_a_subset_of_one_is_that_gram(self):
        grams = blocks(n_blocks=3)
        alone = pg.solve_subsets(grams, [[0], [1], [2]], path={"max_iter": 4})
        paths = pg.lars_paths(grams, max_iter=4)
        for a, p in zip(alone, paths, strict=True):
            for x, y in zip(a, p, strict=True):
                assert np.array_equal(x["coef"], y["coef"])
        fits = pg.solve_subsets(grams, [[1]], ridge=0.1, standardize=True)
        want = pg.solve(grams[1], target="y", ridge=0.1, standardize=True)
        assert np.abs(fits[0][0]["coef"] - want).max() < 1e-12

    def test_a_system_that_is_not_positive_definite_is_nan(self):
        """A constant column and no ridge: its pivot is exactly 0, so the
        factorization fails and every coefficient is ``nan``."""
        df = stream(n=1200, seed=21).with_columns(
            x4=pl.lit(3.0), block=pl.int_range(pl.len()) // 600
        )
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=[f"x{i}" for i in range(5)],
            half_life=float("inf"),
            group="block",
            min_weight=5.0,
            ridge=1e-3,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        grams = bank.gram("m")
        (fit,) = pg.solve_subsets(grams, [[0, 1]])[0]
        assert np.all(np.isnan(fit["coef"]))
        (fit,) = pg.solve_subsets(grams, [[0, 1]], ridge=0.01)[0]
        want = pg.solve(pg.merge(grams), ridge=0.01)
        assert np.abs(fit["coef"] - want).max() < 1e-9
        # The ridge makes the coefficients finite, and the standard errors
        # read the unridged moments of the columns with variance: the
        # constant's own is nan, as ``coef_stats`` gives, never 0, which
        # reads as an exact coefficient (review 6, E-1: ``f64::max(NaN, 0)``
        # is 0), and the others' are the fit's without it, where a factor
        # of the whole system made every one nan (review round 8, A-5).
        stats = pg.coef_stats(pg.merge(grams), want)
        for key in ("se", "t"):
            assert np.isnan(stats[key][5]) and np.isnan(fit[key][5]), (key, fit[key])
            assert np.isfinite(fit[key][:5]).all(), (key, fit[key])
            np.testing.assert_allclose(fit[key][:5], stats[key][:5], rtol=1e-9)
        for kw in (dict(ridge=0.1, standardize=True), dict(standardize=True)):
            (fit,) = pg.solve_subsets(grams, [[0, 1]], **kw)[0]
            assert not np.any(fit["se"] == 0.0), (kw, fit["se"])

    @pytest.mark.parametrize(
        ("args", "kw", "match"),
        [
            ([[]], {}, "names no Gram"),
            ([[0, 9]], {}, "out of range"),
            ([[1, 1]], {}, "names a Gram twice"),
            ([[0]], dict(ridge=-1.0), "ridge must be finite"),
            ([[0]], dict(path={"max_knots": 3}), "path takes"),
            ([[0]], dict(path={"max_iter": 0}), "max_iter must be"),
            ([[0]], dict(features=["intercept"]), "constant column"),
        ],
    )
    def test_it_refuses(self, args, kw, match):
        grams = blocks(n_blocks=2)
        with pytest.raises(ValueError, match=match):
            pg.solve_subsets(grams, args, **kw)

    def test_it_refuses_grams_that_do_not_share_their_axes_or_lack_the_centred_forms(self):
        grams = blocks(n_blocks=2)
        with pytest.raises(ValueError, match="needs at least one Gram"):
            pg.solve_subsets([], [[0]])
        with pytest.raises(ValueError, match="same columns and targets"):
            pg.solve_subsets([grams[0], pg.subset(grams[1], ["intercept", "x0"])], [[0, 1]])
        bare = dict(grams[1], cross_centered=None)
        with pytest.raises(ValueError, match="cross_centered"):
            pg.solve_subsets([grams[0], bare], [[0, 1]])
        old = dict(grams[1], target_vars=None)
        with pytest.raises(ValueError, match="no target moments"):
            pg.solve_subsets([grams[0], old], [[0, 1]])


class TestCompactGram:
    """``bank.gram(dtype="float32")`` and ``layout="packed"`` (docs/PLAN.md
    task 229): the co-moments rounded to float32, or their upper triangle,
    and every function of ``po.gram`` reading each form in float64."""

    @staticmethod
    def bank(df=None, **kw):
        df = stream(n=2000, k=6, seed=30, offset=50.0) if df is None else df
        spec = po.spec.ewridge(
            "m",
            targets=["y", "y2"],
            features=[c for c in df.columns if c.startswith("x")],
            half_life=500.0,
            min_weight=5.0,
            **kw,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        return bank

    def test_float32_is_the_float64_rounded_to_nearest(self):
        bank = self.bank()
        (g64,), (g32,) = bank.gram("m"), bank.gram("m", dtype="float32")
        c64, c32 = g64["comoments"], g32["comoments"]
        assert c32.dtype == np.float32 and c32.shape == c64.shape
        assert np.array_equal(c32, c64.astype(np.float32))
        live = c64 != 0.0
        rel = np.abs(c32[live].astype(np.float64) - c64[live]) / np.abs(c64[live])
        assert rel.max() <= 2.0**-24, rel.max()
        assert c32.nbytes * 2 == c64.nbytes
        for key, v in g64.items():
            if key != "comoments" and isinstance(v, np.ndarray):
                assert np.array_equal(g32[key], v, equal_nan=True), key
                assert g32[key].dtype == v.dtype, key
        # The polars and numpy spellings of the dtype are the same.
        (gp,) = bank.gram("m", dtype=pl.Float32)
        (gn,) = bank.gram("m", dtype=np.float32)
        assert np.array_equal(gp["comoments"], c32) and np.array_equal(gn["comoments"], c32)

    def test_packed_is_the_upper_triangle_and_expands_to_the_matrix(self):
        bank = self.bank()
        (g,), (gp,), (gq,) = (
            bank.gram("m"),
            bank.gram("m", layout="packed"),
            bank.gram("m", dtype="float32", layout="packed"),
        )
        c = g["comoments"]
        k = c.shape[0]
        iu = np.triu_indices(k)
        assert gp["comoments"].shape == (k * (k + 1) // 2,)
        assert np.array_equal(gp["comoments"], c[iu]), "the upper triangle, bit for bit"
        # The index rule: (i, j), i <= j, at i * (2k - i + 1) // 2 + (j - i).
        for i, j in [(0, 0), (1, 3), (k - 1, k - 1), (2, 2), (0, k - 1)]:
            assert gp["comoments"][i * (2 * k - i + 1) // 2 + (j - i)] == c[i, j]
        full = pg._unpack(np, gp["comoments"], k)
        assert np.array_equal(np.triu(full), np.triu(c))
        assert np.array_equal(full, full.T)
        # Its lower triangle is the mirror, which the accumulator's own lower
        # triangle matches to the last bit or so (E48), not to the bit.
        assert np.abs(full - c).max() <= 1e-15 * np.abs(c).max()
        assert np.array_equal(gq["comoments"], c[iu].astype(np.float32))
        n = c.nbytes
        assert gp["comoments"].nbytes == n * (k + 1) // (2 * k)
        assert gq["comoments"].nbytes == n * (k + 1) // (4 * k)
        # A closed row packs the same way: its expansion reads this.
        assert np.array_equal(pg._unvech(np, list(gp["comoments"]), k), full)

    def test_a_lagged_matrix_takes_the_dtype_and_is_never_packed(self):
        df = stream(n=800, k=3, seed=31)
        spec = po.spec.ew_cov("c", features=["x0", "x1", "x2"], lam=0.99, lags=[1, 2])
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        (g,), (h,) = bank.gram("c"), bank.gram("c", dtype="float32", layout="packed")
        assert h["lag_comoments"].shape == g["lag_comoments"].shape == (2, 3, 3)
        assert np.array_equal(h["lag_comoments"], g["lag_comoments"].astype(np.float32))
        assert h["comoments"].shape == (6,)

    @pytest.mark.parametrize(
        "form",
        [dict(layout="packed"), dict(dtype="float32"), dict(dtype="float32", layout="packed")],
    )
    def test_every_function_reads_each_form(self, form):
        bank = self.bank(group=None)
        (g,) = bank.gram("m")
        (h,) = bank.gram("m", **form)
        f32 = form.get("dtype") == "float32"
        # Rounding to float32 moves a solve by about the condition number
        # times 6e-8; the packed form by the matrix's last-bit asymmetry.
        tol = 1e-4 if f32 else 1e-10
        assert pg.solve(h, ridge=0.1) == pytest.approx(pg.solve(g, ridge=0.1), rel=tol)
        beta = pg.solve(g, ridge=0.1)
        for key in ("resid_var", "sigma2", "r2"):
            assert pg.coef_stats(h, beta)[key] == pytest.approx(
                pg.coef_stats(g, beta)[key], rel=tol
            )
        np.testing.assert_allclose(pg.correlation(h), pg.correlation(g), rtol=tol, atol=tol)
        np.testing.assert_allclose(pg.vif(h), pg.vif(g), rtol=tol)
        assert pg.condition(h)["kappa"] == pytest.approx(pg.condition(g)["kappa"], rel=tol)
        np.testing.assert_allclose(
            pg.lasso_path(h, PENALTIES), pg.lasso_path(g, PENALTIES), rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            pg.lars_path(h, max_iter=4)["coef"],
            pg.lars_path(g, max_iter=4)["coef"],
            rtol=tol,
            atol=tol,
        )
        sub = pg.subset(h, ["intercept", "x3", "x1"])
        assert np.ndim(sub["comoments"]) == (1 if form.get("layout") == "packed" else 2)
        assert sub["comoments"].dtype == np.float64
        np.testing.assert_allclose(
            pg._comoments(np, sub),
            pg.subset(g, ["intercept", "x3", "x1"])["comoments"],
            rtol=tol,
            atol=tol,
        )

    def test_merges_and_subset_fits_read_each_form_in_float64(self):
        bank_forms = {}
        df = stream(n=1800, seed=20).with_columns(block=pl.int_range(pl.len()) // 600)
        spec = po.spec.ewridge(
            "m",
            targets=["y", "y2"],
            features=[c for c in df.columns if c.startswith("x")],
            half_life=float("inf"),
            group="block",
            min_weight=5.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        for name, form in {
            "full": {},
            "packed": dict(layout="packed"),
            "f32": dict(dtype="float32"),
            "both": dict(dtype="float32", layout="packed"),
        }.items():
            bank_forms[name] = bank.gram("m", **form)
        grams = bank_forms["full"]
        k = grams[0]["comoments"].shape[0]
        whole = pg.merge(grams)
        packed = pg.merge(bank_forms["packed"])
        assert np.ndim(packed["comoments"]) == 1 and packed["comoments"].dtype == np.float64
        assert np.array_equal(packed["comoments"], whole["comoments"][np.triu_indices(k)])
        f32 = pg.merge(bank_forms["f32"])
        assert f32["comoments"].dtype == np.float64
        np.testing.assert_allclose(f32["comoments"], whole["comoments"], rtol=2e-7, atol=1e-12)
        both = pg.merge(bank_forms["both"])
        np.testing.assert_allclose(
            pg._comoments(np, both), pg._comoments(np, f32), rtol=1e-15, atol=1e-15
        )
        mixed = pg.merge([bank_forms["packed"][0], *grams[1:]])
        assert np.ndim(mixed["comoments"]) == 2
        # solve_subsets reads every form; packed is the full form's upper
        # triangle, so its merge and path are the full one's on a symmetric
        # matrix, and float32 within its rounding.
        subsets = [[0, 1], [2], [0, 1, 2]]
        want = pg.solve_subsets(grams, subsets, ridge=0.01)
        for name in ("packed", "f32", "both"):
            got = pg.solve_subsets(bank_forms[name], subsets, ridge=0.01)
            tol = 1e-10 if name == "packed" else 1e-4
            for g_per, w_per in zip(got, want, strict=True):
                for a, b in zip(g_per, w_per, strict=True):
                    np.testing.assert_allclose(a["coef"], b["coef"], rtol=tol, atol=tol)
                    np.testing.assert_allclose(a["t"][1:], b["t"][1:], rtol=tol)
            path = pg.solve_subsets(bank_forms[name], subsets, path={"max_iter": 3})
            assert len(path) == 3

    @pytest.mark.parametrize(
        ("scale", "says"),
        [
            (1e-25, "below float32's smallest normal number"),
            (1e20, "past float32's largest number"),
        ],
    )
    @pytest.mark.parametrize("layout", ["full", "packed"])
    def test_an_entry_outside_float32s_range_is_refused_by_name(self, scale, says, layout):
        """A co-moment that float32 would flush to zero, or below its normal
        range, or round to infinity is refused, naming the column (review 6,
        E-3: a column at 1e-25 read a variance of 0, one at 1e20 inf, and
        the solves went on silently wrong). float64 hands it over."""
        rng = np.random.default_rng(0)
        n = 500
        x = rng.normal(size=(n, 2)) * [scale, 1.0]
        df = pl.DataFrame({"x0": x[:, 1], "x1": x[:, 0], "y": x[:, 1] + rng.normal(size=n)})
        bank = po.ModelBank([po.spec.ewridge("r", targets=["y"], features=["x0", "x1"], lam=1.0)])
        bank.fit_predict(df)
        (g,) = bank.gram("r", layout=layout)
        assert np.isfinite(g["comoments"]).all()
        with pytest.raises(ValueError, match=f"'x1'.*{says}"):
            bank.gram("r", dtype="float32", layout=layout)
        lagged = po.ModelBank([po.spec.ew_cov("c", features=["x0", "x1"], lam=1.0, lags=[1])])
        lagged.fit_predict(df)
        with pytest.raises(ValueError, match=says):
            lagged.gram("c", dtype="float32")

    def test_a_lagged_entry_outside_float32s_range_is_refused_by_name(self):
        """A lagged co-moment is held to the same range: at 3e-19, ``x1``'s
        variance, 8.7e-38, is a float32 normal number and its lag-1
        co-moment, 3.8e-39, is not."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 2)) * [1.0, 3e-19]
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
        bank = po.ModelBank([po.spec.ew_cov("c", features=["x0", "x1"], lam=1.0, lags=[1])])
        bank.fit_predict(df)
        (g,) = bank.gram("c")
        assert np.float32(g["comoments"][1, 1]) >= np.finfo(np.float32).tiny
        with pytest.raises(ValueError, match="lag-1 co-moment of 'x1' with itself"):
            bank.gram("c", dtype="float32")
        no_lags = po.ModelBank([po.spec.ew_cov("c", features=["x0", "x1"], lam=1.0)])
        no_lags.fit_predict(df)
        (h,) = no_lags.gram("c", dtype="float32")
        assert h["comoments"][1, 1] == np.float32(g["comoments"][1, 1])

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            (dict(dtype="float16"), 'dtype must be "float64" or "float32"'),
            (dict(dtype="int64"), 'dtype must be "float64" or "float32"'),
            (dict(dtype=[1]), 'dtype must be "float64" or "float32"'),
            (dict(layout="lower"), 'layout must be "full" or "packed"'),
        ],
    )
    def test_it_refuses(self, kw, match):
        bank = self.bank()
        with pytest.raises(ValueError, match=match):
            bank.gram("m", **kw)


# ------------------------------------------------- task 278 (b12): the paths' own keywords


def grouped_grams(seed: int = 15, k: int = 6) -> list[dict[str, Any]]:
    """Four Grams of one spec, one per group, two targets each."""
    df = stream(seed=seed, k=k).with_columns(grp=pl.int_range(pl.len()) % 4)
    spec = po.spec.ewridge(
        "m",
        targets=["y", "y2"],
        features=[f"x{i}" for i in range(k)],
        lam=1.0,
        group="grp",
        min_weight=5.0,
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m")


def same_path(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        np.array_equal(a["coef"], b["coef"])
        and np.array_equal(a["penalties"], b["penalties"])
        and a["active"] == b["active"]
        and a["stop"] == b["stop"]
    )


class TestLarsPathsTakeLarsPathsArguments:
    """``lars_paths`` runs ``lars_path`` per Gram and target, so each
    keyword it passes on means what it means there, to the bit. No test
    passed ``penalty_weights`` or ``features`` to ``lars_paths`` before."""

    @pytest.mark.parametrize(
        "kw",
        [
            dict(penalty_weights=[0.5, 1.0, 2.0, 1.0, 1.0, 3.0]),
            dict(features=["x3", "x0", "x5"]),
            dict(features=["x4", "x1"], penalty_weights=[2.0, 0.25], max_active=1),
            dict(penalty_weights=[1.0, 4.0, 1.0, 1.0, 0.5, 1.0], max_iter=3),
        ],
        ids=["weights", "features", "both", "weights,max_iter"],
    )
    def test_each_gram_and_target_is_its_lars_path(self, kw):
        grams = grouped_grams()
        many = pg.lars_paths(grams, **kw)
        plain = pg.lars_paths(grams)
        for g, per, unweighted in zip(grams, many, plain, strict=True):
            for t, path, bare in zip(["y", "y2"], per, unweighted, strict=True):
                assert same_path(path, pg.lars_path(g, target=t, **kw))
                # The keyword took effect: the path is not the plain one.
                assert not same_path(path, bare)

    def test_its_refusals_are_lars_paths(self):
        grams = grouped_grams()
        with pytest.raises(ValueError, match="use lasso_path for it"):
            pg.lars_paths(grams, penalty_weights=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="one entry per feature"):
            pg.lars_paths(grams, penalty_weights=[1.0])


def leaving_gram(seed: int = 0) -> dict[str, Any]:
    """A design whose columns mix their signs, so that columns leave the
    path (the lasso modification): ``TestLarsPathIsSklearns``' rows, with
    four to seven knots where one leaves on these seeds."""
    rng = np.random.default_rng(seed)
    k = 10
    mix = np.eye(k) + rng.standard_normal((k, k))
    x = rng.standard_normal((200, k)) @ mix + 5.0
    y = x @ rng.standard_normal(k) + 1.0 + 2.0 * rng.standard_normal(200)
    df = pl.DataFrame({f"x{i}": x[:, i] for i in range(k)}).with_columns(y=pl.Series(y))
    return gram(df)


class TestAStoppedLarsPathIsAPrefix:
    """Stopping a LARS path early changes nothing it has already walked:
    each knot is computed from the ones before it alone, so the path that
    stops after ``max_iter`` knots or at ``max_active`` columns is the full
    path's first rows, to the bit, and a limit past the path's end is the
    full path. Held on every limit, on a path where columns leave."""

    @pytest.mark.parametrize("make", ["stream", "leaving"])
    def test_every_max_iter_and_max_active_is_a_prefix(self, make):
        g = gram(stream(seed=9, k=8)) if make == "stream" else leaving_gram(0)
        full = pg.lars_path(g)
        knots = len(full["penalties"])
        sets = full["active"]
        leaves = sum(len(b) < len(a) for a, b in zip(sets, sets[1:], strict=False))
        print(f"{make}: {knots} knots, {leaves} where a column leaves")
        assert full["stop"] == "end"
        if make == "leaving":
            assert leaves > 0, "the lasso modification did not run"
        for max_iter in range(1, knots + 3):
            p = pg.lars_path(g, max_iter=max_iter)
            n = len(p["penalties"])
            assert n == min(max_iter + 1, knots), max_iter
            assert p["stop"] == ("end" if max_iter + 1 >= knots else "max_iter"), max_iter
            assert np.array_equal(p["penalties"], full["penalties"][:n])
            assert np.array_equal(p["coef"], full["coef"][:n])
            assert p["active"] == sets[:n]
        widest = max(len(a) for a in sets)
        for max_active in range(1, widest + 2):
            p = pg.lars_path(g, max_active=max_active)
            n = len(p["penalties"])
            assert np.array_equal(p["penalties"], full["penalties"][:n])
            assert np.array_equal(p["coef"], full["coef"][:n])
            assert p["active"] == sets[:n]
            if max_active > widest:
                assert same_path(p, full)
            else:
                assert p["stop"] == "max_active"
                assert len(p["active"][-1]) >= max_active
                assert all(len(a) < max_active for a in p["active"][:-1])

    def test_lars_paths_stops_each_path_the_same_way(self):
        grams = grouped_grams()
        full = pg.lars_paths(grams)
        for max_iter in (1, 2, 4, 50):
            for per, whole in zip(pg.lars_paths(grams, max_iter=max_iter), full, strict=True):
                for p, f in zip(per, whole, strict=True):
                    n = len(p["penalties"])
                    assert n == min(max_iter + 1, len(f["penalties"]))
                    assert np.array_equal(p["coef"], f["coef"][:n])


def rows_system(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The lasso's problem written from the rows (unit weights, as the
    Gram's ``lam=1`` accumulator holds them), not from the Gram: ``C`` the
    features' correlation matrix, ``d_i = cov(x_i, y) / s_i``, ``s`` their
    population spreads and the target's spread. The objective is
    ``b'Cb / 2 - d'b + l |b|_1`` over the standardized ``b = coef * s``."""
    x = df.select(sorted(c for c in df.columns if c.startswith("x"))).to_numpy()
    y = df["y"].to_numpy()
    m = x.mean(0)
    s = np.sqrt(((x - m) ** 2).mean(0))
    zs = (x - m) / s
    return zs.T @ zs / len(y), zs.T @ (y - y.mean()) / len(y), s, float(np.std(y))


def kkt_violation(c: np.ndarray, d: np.ndarray, lam: float, b: np.ndarray) -> np.ndarray:
    """Each coordinate's distance from its KKT condition: ``|g_i + l
    sign(b_i)|`` where ``b_i`` is not 0, ``max(|g_i| - l, 0)`` where it is,
    ``g = C b - d``."""
    g = c @ b - d
    return np.where(b != 0, np.abs(g + lam * np.sign(b)), np.maximum(np.abs(g) - lam, 0.0))


def gamma(m: int) -> float:
    """Higham's ``gamma_m = m u / (1 - m u)``."""
    u = 2.0**-53
    return m * u / (1.0 - m * u)


def certificate(
    c: np.ndarray, d: np.ndarray, lam: float, b: np.ndarray, tol: float, s_y: float, n: int
) -> np.ndarray:
    """What a descent that met its stopping rule certifies, per coordinate.
    The last sweep moved no coordinate by ``tol * s_y`` or more, and
    coordinate ``i`` met its own condition when it was set, so the moves of
    the others since bound its violation by ``tol * s_y * sum_j |C_ij|``.
    The rest is the rounding of the Gram's moments against these from the
    rows: ``8 gamma(4n)`` of the terms' sizes."""
    bound = tol * s_y * np.abs(c).sum(1)
    return bound + 8 * gamma(4 * n) * (np.abs(c) @ np.abs(b) + np.abs(d) + lam)


def short_penalties(caught: list[warnings.WarningMessage]) -> list[float]:
    """The penalties a ``ReadinessWarning`` names as out of sweeps."""
    said = [str(w.message) for w in caught if issubclass(w.category, po.ReadinessWarning)]
    assert len(said) <= 1, said
    if not said:
        return []
    listed = re.search(r"penalties, \[([^\]]*)\]", said[0])
    assert listed, said[0]
    return [float(v) for v in listed.group(1).split(",")]


def path_and_warnings(g, penalties, **kw) -> tuple[np.ndarray, list[float]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = pg.lasso_path(g, penalties, **kw)
    return path, short_penalties(caught)


class TestLassoPathsStoppingRule:
    """``lasso_path``'s ``tol`` and ``max_iter`` -- the coordinate descent's
    stopping rule and its cap -- against the problem written from the rows.
    No test showed a loose ``tol`` changing the answer before (every one
    tightened it), nor what a binding ``max_iter`` leaves; the warning that
    names a penalty out of sweeps has its own test in
    ``test_convergence_iterations.py``."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_tol_meets_its_certificate_and_a_tighter_one_lands_nearer(self, seed):
        df = stream(seed=seed, k=6)
        g = gram(df)
        c, d, s, s_y = rows_system(df)
        ref = pg.lasso_path(g, PENALTIES, tol=1e-15, max_iter=1_000_000)
        dists = []
        for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
            path, short = path_and_warnings(g, PENALTIES, tol=tol)
            assert short == [], (tol, short)
            for row, lam in zip(path, PENALTIES, strict=True):
                b = row[1:] * s
                v = kkt_violation(c, d, lam, b)
                assert np.all(v <= certificate(c, d, lam, b, tol, s_y, df.height)), (tol, lam)
            dists.append(np.abs(path - ref).max())
        print(f"seed {seed}: distance from the converged path by tol {dists}")
        # Measured 2026-10-25 on seeds 0-2: 4.1e-2..4.4e-2 at tol 1e-1 down to
        # 5.1e-13..5.9e-13 at 1e-12, falling at every step. The descent's
        # path does not depend on tol; a smaller one only stops it on a
        # later sweep, so the distance shrinks with it here.
        assert all(a >= b for a, b in zip(dists, dists[1:], strict=False)), dists
        assert dists[0] > 1e-2 and dists[-1] < 1e-11, dists

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_binding_max_iter_is_named_and_every_other_penalty_met_its_rule(self, seed):
        df = stream(seed=seed, k=6)
        g = gram(df)
        c, d, s, s_y = rows_system(df)
        ref = pg.lasso_path(g, PENALTIES, tol=1e-15, max_iter=1_000_000)
        counts, dists, over = [], [], 0.0
        for max_iter in [1, 2, 3, 5, 10, 20, 1000]:
            path, short = path_and_warnings(g, PENALTIES, max_iter=max_iter)
            assert set(short) <= set(PENALTIES)
            for row, lam in zip(path, PENALTIES, strict=True):
                b = row[1:] * s
                v = kkt_violation(c, d, lam, b)
                ok = np.all(v <= certificate(c, d, lam, b, 1e-10, s_y, df.height))
                # A penalty not named stopped by the rule, so it holds the
                # rule's certificate.
                assert ok or lam in short, (max_iter, lam)
                if max_iter == 1 and lam in short:
                    over = max(over, float(v.max()))
            counts.append(len(short))
            dists.append(np.abs(path - ref).max())
        print(f"seed {seed}: penalties named {counts}, distance {dists}, one sweep's KKT {over}")
        # The hard path: one sweep names every penalty and leaves them off
        # their certificate (measured 2026-10-25: 8 of 8 named, the
        # violation above 1e-3); 1,000 sweeps name none.
        assert counts[0] == len(PENALTIES) and over > 1e-3
        assert counts[-1] == 0
        assert all(a >= b for a, b in zip(counts, counts[1:], strict=False)), counts
        # Measured 2026-10-25 on seeds 0-2: 8.5e-2..9.1e-2 after one sweep,
        # 4.0e-8..2.8e-7 after ten, 3.8e-11..4.5e-11 once the rule stops it.
        assert all(a >= b for a, b in zip(dists, dists[1:], strict=False)), dists

    @pytest.mark.parametrize("lam", [0.2, 0.02])
    def test_one_penalty_s_objective_never_rises_with_another_sweep(self, lam):
        """Cyclic coordinate descent minimizes the objective exactly along
        each coordinate in turn, so no sweep raises it: from the same start
        (zero, one penalty) the row after ``j + 1`` sweeps is no worse than
        after ``j``. ``tol=1e-300`` keeps the rule from stopping any sooner
        than a sweep that moves nothing, so a call runs out of sweeps, and
        says so, until the descent stands still. The slack is the rounding
        of the Gram's moments against these from the rows."""
        df = stream(seed=0, k=6)
        g = gram(df)
        c, d, s, _ = rows_system(df)
        objs, slack = [], []
        for max_iter in range(1, 40):
            path, short = path_and_warnings(g, [lam], max_iter=max_iter, tol=1e-300)
            # Measured 2026-10-25: still moving after five sweeps.
            assert short == [lam] if max_iter <= 5 else short in ([], [lam])
            b = path[0, 1:] * s
            objs.append(0.5 * b @ c @ b - d @ b + lam * np.abs(b).sum())
            terms = 0.5 * np.abs(b) @ np.abs(c) @ np.abs(b) + np.abs(d) @ np.abs(b)
            slack.append(8 * gamma(4 * df.height) * (terms + lam * np.abs(b).sum()))
        rises = np.diff(objs)
        print(f"objective {objs[0]:.6f} -> {objs[-1]:.6f}, largest rise {rises.max():.1e}")
        assert objs[0] - objs[-1] > 1e-3, "the sweeps did not descend"
        assert np.all(rises <= np.array(slack[1:])), rises.max()

    def test_weights_times_a_power_of_two_are_the_penalties_times_it(self):
        """Each column's penalty is ``l * pw_i``, so doubling every weight is
        doubling every penalty, to the bit: a power of two scales each
        product exactly, and the stopping rule reads neither."""
        g = gram(stream(seed=3, k=5))
        w = np.array([1.0, 0.0, 2.0, 0.5, 3.0])
        pens = np.array(PENALTIES)
        for c in (0.25, 2.0, 8.0):
            a = pg.lasso_path(g, list(pens), penalty_weights=list(w * c))
            b = pg.lasso_path(g, list(pens * c), penalty_weights=list(w))
            assert np.array_equal(a, b), c
        assert not np.array_equal(
            pg.lasso_path(g, PENALTIES, penalty_weights=list(w)), pg.lasso_path(g, PENALTIES)
        )


class TestMergeOrder:
    """``merge`` pools its parts' moments; the result is the same Gram in
    any order, to rounding, and each fit from it the same fit. Associativity
    of the within merge had a test; the order of the parts had none.
    Measured 2026-10-25 over all 120 orders of five blocks: the moments
    within ``3.2e-16`` of ``1 + |value|``, the weights equal to the bit, the
    solve within ``1.2e-15``; held at ``4 gamma(5)`` (2.2e-15) and ``1e-14``."""

    MOMENTS = ("means", "comoments", "cross_moments", "cross_centered", "target_vars")

    @pytest.mark.parametrize("within", [False, True])
    @pytest.mark.parametrize("gaps", [False, True])
    def test_every_order_of_the_parts_is_the_same_gram(self, within, gaps):
        grams = blocks(n_blocks=5, seed=20, gaps=gaps)
        base = pg.merge(grams, within=within)
        fit = pg.solve(base, target="y2")
        for order in itertools.permutations(range(5)):
            m = pg.merge([grams[i] for i in order], within=within)
            assert m["columns"] == base["columns"] and m["targets"] == base["targets"]
            assert np.array_equal(m["weight_sum"], base["weight_sum"])
            for key in self.MOMENTS:
                a = np.asarray(base[key], dtype=float)
                b = np.asarray(m[key], dtype=float)
                assert np.all(np.abs(a - b) <= 4 * gamma(5) * (1.0 + np.abs(a))), (order, key)
            got = pg.solve(m, target="y2")
            ok = ~np.isnan(fit)
            assert np.array_equal(np.isnan(got), ~ok)
            assert np.abs(got[ok] - fit[ok]).max() <= 1e-14 * (1.0 + np.abs(fit[ok]).max())
