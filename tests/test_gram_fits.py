"""The Gram's offline fits in Rust (docs/PLAN.md task 226): ``lasso_path``,
``lars_path`` and ``lars_paths``.

``lasso_path`` moved from a Python loop to Rust with its signature kept, so
the Python loop it replaced is kept here as the reference it is held to
(:func:`python_lasso_path`, the 0.13.0 code verbatim). scikit-learn's second
opinions on both paths are in ``tests/test_second_opinion.py``
(``TestLarsPathIsSklearns``).
"""

from __future__ import annotations

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
    tol=1e-7,
):
    """``po.gram.lasso_path`` as 0.13.0 ran it, in Python: the reference the
    Rust descent is held to."""
    t = pg._target_index(g, target)
    slots, icept = pg._feature_slots(g, features)
    k = len(pg._columns(g))
    means = np.asarray(g["means"], dtype=float)
    cross = np.asarray(g["cross_moments"], dtype=float)[t]
    como = np.asarray(g["comoments"], dtype=float)
    m, ybar = means, 0.0
    if icept >= 0:
        m = pg._means_of(np, g, t)
        ybar = cross[icept]
        c = como[np.ix_(slots, slots)]
        rhs = pg._cross_centred(np, g, t, m, ybar)[slots]
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
            if delta < tol:
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

    def test_a_mapping_without_the_centred_forms_forms_them(self):
        g = gram(stream(seed=5))
        bare = dict(g, means_by_target=None, cross_centred=None)
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

    def test_it_stops_after_max_steps_or_at_max_active(self):
        g = gram(stream(seed=9, k=8))
        full = pg.lars_path(g)
        steps = pg.lars_path(g, max_steps=3)
        assert steps["stop"] == "max_steps"
        assert len(steps["penalties"]) == 4
        assert np.array_equal(steps["coef"], full["coef"][:4])
        active = pg.lars_path(g, max_active=2)
        assert active["stop"] == "max_active"
        assert len(active["active"][-1]) == 2
        assert all(len(a) < 2 for a in active["active"][:-1])
        assert np.count_nonzero(active["coef"][-1][1:]) == 1, "the second enters at 0"
        assert np.array_equal(active["coef"], full["coef"][: len(active["coef"])])

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
            (dict(max_steps=0), "max_steps must be a whole number >= 1"),
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
        many = pg.lars_paths(grams, max_steps=4)
        assert [len(per) for per in many] == [2, 2, 2, 2]
        for g, per in zip(grams, many, strict=True):
            for t, path in zip(["y", "y2"], per, strict=True):
                one = pg.lars_path(g, target=t, max_steps=4)
                assert np.array_equal(path["coef"], one["coef"])
                assert np.array_equal(path["penalties"], one["penalties"])
                assert path["active"] == one["active"]
        named = pg.lars_paths(grams, targets=["y2"], max_steps=4)
        assert [len(per) for per in named] == [1, 1, 1, 1]
        assert np.array_equal(named[2][0]["coef"], many[2][1]["coef"])

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
        alone = pg.solve_subsets(grams, [[0], [1], [2]], path={"max_steps": 4})
        paths = pg.lars_paths(grams, max_steps=4)
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

    @pytest.mark.parametrize(
        ("args", "kw", "match"),
        [
            ([[]], {}, "names no Gram"),
            ([[0, 9]], {}, "out of range"),
            ([[1, 1]], {}, "names a Gram twice"),
            ([[0]], dict(ridge=-1.0), "ridge must be finite"),
            ([[0]], dict(path={"max_iter": 3}), "path takes"),
            ([[0]], dict(path={"max_steps": 0}), "max_steps must be"),
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
        bare = dict(grams[1], cross_centred=None)
        with pytest.raises(ValueError, match="cross_centred"):
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
            pg.lars_path(h, max_steps=4)["coef"],
            pg.lars_path(g, max_steps=4)["coef"],
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
            path = pg.solve_subsets(bank_forms[name], subsets, path={"max_steps": 3})
            assert len(path) == 3

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
