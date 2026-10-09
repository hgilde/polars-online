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
