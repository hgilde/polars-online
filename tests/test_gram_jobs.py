"""``po.gram.solve_subsets(..., jobs=...)`` (docs/PLAN.md task 256): a fit per
job, each job its own subset, target, draw of signs and columns.

Each job is held, to the bit, to the loop it replaces: ``merge`` of its
subset's Grams (within, the draw's signs on each block's targets), ``subset``
on its columns with the intercept, and ``lars_path`` on that. The signs are
put on the blocks' centred cross-moments before the merge, as negating each
block's target rows does. A job without columns is the call's ``features``;
the jobs agree with the same call without them; and the pool's size moves no
bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import gram as pg

TIER = "essential"

K = 10
FEATURES = [f"x{j}" for j in range(K)]
SUBSETS = [[0, 1, 2], [3, 4, 5], [0, 2, 4], [1], [5, 3, 1, 0]]
SIGNS = np.array(
    [[1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, 1, -1]], dtype=float
)


def frame(n_blocks: int = 6, per: int = 200, seed: int = 0) -> pl.DataFrame:
    """Blocks of rows at levels of their own, the features correlated
    through a shared factor, ``y`` on four of them and ``y2`` on three
    others, missing on a fifth of the rows (fitted pairwise)."""
    rng = np.random.default_rng(seed)
    n = n_blocks * per
    block = np.repeat(np.arange(n_blocks), per)
    level = 2.0 * rng.standard_normal((n_blocks, K))
    x = rng.standard_normal((n, K)) + 0.6 * rng.standard_normal((n, 1)) + level[block]
    y = x[:, [0, 3, 5, 8]] @ [0.6, -0.4, 0.3, 0.2] + level[block, 0] + rng.standard_normal(n)
    y2 = x[:, [1, 2, 9]] @ [0.5, 0.5, -0.3] + 0.5 * rng.standard_normal(n)
    return pl.DataFrame(
        {
            "block": block,
            **{f: x[:, j] for j, f in enumerate(FEATURES)},
            "y": y,
            "y2": np.where(np.arange(n) % 5 == 0, np.nan, y2),
        }
    ).with_columns(pl.col("y2").fill_nan(None))


def blocks(df: pl.DataFrame) -> list[dict[str, Any]]:
    spec = po.spec.ewridge(
        "m",
        targets=["y", "y2"],
        features=FEATURES,
        half_life=float("inf"),
        group="block",
        target_gaps="pairwise",
    )
    bank = po.ModelBank([spec])
    bank.fit_predict(df)
    return bank.gram("m")


def screens(seed: int, n_jobs: int) -> list[list[str]]:
    """A screen per job: between 2 and ``K`` of the features, in an order
    of their own, so a tie among them is settled in that order."""
    rng = np.random.default_rng(seed)
    return [
        [FEATURES[j] for j in rng.permutation(K)[: rng.integers(2, K + 1)]] for _ in range(n_jobs)
    ]


def signed(parts: list[dict[str, Any]], signs: np.ndarray) -> list[dict[str, Any]]:
    """Each Gram with its targets negated where its sign is -1: its
    cross-moments, centred and raw, which is all a negated target moves in
    a merge within the parts."""
    return [
        p
        if s > 0
        else {
            **p,
            "cross_centered": -np.asarray(p["cross_centered"]),
            "cross_moments": -np.asarray(p["cross_moments"]),
        }
        for p, s in zip(parts, signs, strict=True)
    ]


def reference(
    grams: list[dict[str, Any]],
    job: dict[str, Any],
    *,
    within: bool = True,
    signs: np.ndarray | None = None,
    **path: Any,
) -> dict[str, Any]:
    """The loop a job replaces: its subset merged (within, the draw's signs
    on the targets), the intercept and its columns taken, and the path."""
    parts = [grams[i] for i in SUBSETS[job["subset"]]]
    if signs is not None:
        parts = signed(parts, signs[SUBSETS[job["subset"]], job["draw"]])
    merged = pg.merge(parts, within=within)
    feats = job.get("features") or FEATURES
    return pg.lars_path(pg.subset(merged, ["intercept", *feats]), target=job["target"], **path)


def same_path(got: dict[str, Any], want: dict[str, Any], feats: list[str]) -> None:
    """``got``, over every column of the Grams, is ``want``, over the
    intercept and ``feats``, to the bit, and 0 on every other column."""
    cols = ["intercept", *FEATURES]
    at = [cols.index(c) for c in ["intercept", *feats]]
    assert np.array_equal(got["penalties"], want["penalties"])
    assert np.array_equal(got["coef"][:, at], want["coef"], equal_nan=True)
    rest = [i for i in range(len(cols)) if i not in at]
    assert not got["coef"][:, rest].any()
    assert got["active"] == want["active"]
    assert got["stop"] == want["stop"]


def signed_jobs(seed: int) -> list[dict[str, Any]]:
    """A job per subset, draw and target, each with a screen of its own,
    in an order that is not the nested one."""
    order = [
        (s, d, t) for s in range(len(SUBSETS)) for d in range(SIGNS.shape[1]) for t in ("y", "y2")
    ]
    rng = np.random.default_rng(seed)
    feats = screens(seed + 1, len(order))
    return [
        {"subset": s, "draw": d, "target": t, "features": feats[i]}
        for i, (s, d, t) in (lambda o: [(i, o[i]) for i in rng.permutation(len(o))])(order)
    ]


class TestPaths:
    def test_each_job_is_lars_path_on_its_columns_of_its_signed_merge_to_the_bit(self):
        grams = blocks(frame(seed=1))
        jobs = signed_jobs(2)
        got = pg.solve_subsets(
            grams,
            SUBSETS,
            path={"max_active": 4},
            within=True,
            target_signs=SIGNS,
            jobs=jobs,
        )
        assert len(got) == len(jobs)
        stops = set()
        for job, path in zip(jobs, got, strict=True):
            want = reference(grams, job, signs=SIGNS, max_active=4)
            same_path(path, want, job["features"])
            stops.add(path["stop"])
        # The limit stopped some paths and a short screen ran others to 0.
        assert stops == {"max_active", "end"}

    def test_a_path_to_its_end_and_one_stopped_after_its_knots(self):
        grams = blocks(frame(seed=3))
        jobs = [
            {"subset": s, "target": t, "features": f}
            for (s, t), f in zip([(0, "y"), (1, "y2"), (4, 0), (3, 1)], screens(4, 4), strict=True)
        ]
        for limits in ({}, {"max_iter": 2}):
            got = pg.solve_subsets(grams, SUBSETS, path=limits, within=True, jobs=jobs)
            for job, path in zip(jobs, got, strict=True):
                same_path(path, reference(grams, job, **limits), job["features"])

    def test_the_union_merge_is_merge_then_subset_then_lars_path(self):
        grams = blocks(frame(seed=5))
        jobs = [
            {"subset": s, "target": t, "features": f}
            for (s, t), f in zip(
                [(3, "y"), (3, "y2"), (3, "y"), (2, "y")], screens(6, 4), strict=True
            )
        ]
        got = pg.solve_subsets(grams, SUBSETS, path={"max_active": 3}, jobs=jobs)
        for job, path in zip(jobs, got, strict=True):
            want = reference(grams, job, within=False, max_active=3)
            same_path(path, want, job["features"])
        assert np.isfinite(got[0]["coef"][:, 0]).all()

    def test_a_job_without_columns_reads_the_calls_features(self):
        grams = blocks(frame(seed=7))
        base = {"subset": 2, "target": "y", "draw": 1}
        kw: dict[str, Any] = dict(path={"max_active": 5}, within=True, target_signs=SIGNS)
        (every, named, none) = pg.solve_subsets(
            grams,
            SUBSETS,
            **kw,
            jobs=[base, {**base, "features": FEATURES}, {**base, "features": None}],
        )
        same_path(every, reference(grams, base, signs=SIGNS, max_active=5), FEATURES)
        for path in (named, none):
            same_path(path, reference(grams, base, signs=SIGNS, max_active=5), FEATURES)
        call = ["x4", "x0", "x7"]
        (default, own) = pg.solve_subsets(
            grams, SUBSETS, **kw, features=call, jobs=[base, {**base, "features": ["x1", "x2"]}]
        )
        same_path(
            default, reference(grams, {**base, "features": call}, signs=SIGNS, max_active=5), call
        )
        same_path(
            own,
            reference(grams, {**base, "features": ["x1", "x2"]}, signs=SIGNS, max_active=5),
            ["x1", "x2"],
        )

    def test_jobs_of_every_fit_are_the_call_without_them(self):
        grams = blocks(frame(seed=8))
        kw: dict[str, Any] = dict(path={"max_iter": 6}, within=True, target_signs=SIGNS)
        nested = pg.solve_subsets(grams, SUBSETS, **kw)
        jobs = [
            {"subset": s, "draw": d, "target": t}
            for d in range(SIGNS.shape[1])
            for s in range(len(SUBSETS))
            for t in range(2)
        ]
        flat = pg.solve_subsets(grams, SUBSETS, **kw, jobs=jobs)
        want = [path for draw in nested for per in draw for path in per]
        for path, w in zip(flat, want, strict=True):
            same_path(path, w, FEATURES)

    def test_a_job_takes_penalty_weights_for_its_own_columns(self):
        grams = blocks(frame(seed=9))
        feats = ["x5", "x0", "x3", "x8"]
        weights = [1.5, 0.5, 2.0, 1.0]
        job = {"subset": 0, "target": "y", "features": feats, "penalty_weights": weights}
        (got,) = pg.solve_subsets(grams, SUBSETS, path={"max_active": 3}, within=True, jobs=[job])
        want = reference(grams, job, max_active=3, penalty_weights=weights)
        same_path(got, want, feats)
        # The path's own weights apply to every job that has none.
        (shared,) = pg.solve_subsets(
            grams,
            SUBSETS,
            path={"max_active": 3, "penalty_weights": weights},
            within=True,
            jobs=[{**job, "penalty_weights": None}],
        )
        same_path(shared, want, feats)
        unweighted = reference(grams, job, max_active=3)
        assert not np.array_equal(got["penalties"], unweighted["penalties"])

    def test_a_subset_of_one_gram_is_lars_path_on_that_gram(self):
        grams = blocks(frame(seed=10))
        feats = screens(11, len(grams))
        jobs = [{"subset": i, "target": "y2", "features": f} for i, f in enumerate(feats)]
        got = pg.solve_subsets(
            grams, [[i] for i in range(len(grams))], path={"max_active": 3}, jobs=jobs
        )
        for g, f, path in zip(grams, feats, got, strict=True):
            want = pg.lars_path(g, target="y2", features=f, max_active=3)
            for key in ("penalties", "coef"):
                assert np.array_equal(path[key], want[key]), key
            assert path["active"] == want["active"] and path["stop"] == want["stop"]

    def test_the_pools_size_moves_no_bit(self):
        grams = blocks(frame(seed=12))
        jobs = signed_jobs(13)
        kw: dict[str, Any] = dict(path={"max_active": 4}, within=True, target_signs=SIGNS)
        one = pg.solve_subsets(grams, SUBSETS, **kw, jobs=jobs, threads=1)
        four = pg.solve_subsets(grams, SUBSETS, **kw, jobs=jobs, threads=4)
        for a, b in zip(one, four, strict=True):
            assert np.array_equal(a["penalties"], b["penalties"])
            assert np.array_equal(a["coef"], b["coef"], equal_nan=True)
            assert a["active"] == b["active"] and a["stop"] == b["stop"]

    def test_a_frame_of_jobs_is_its_rows(self):
        grams = blocks(frame(seed=14))
        jobs = signed_jobs(15)[:12]
        kw: dict[str, Any] = dict(path={"max_active": 3}, within=True, target_signs=SIGNS)
        rows = pg.solve_subsets(grams, SUBSETS, **kw, jobs=jobs)
        table = pg.solve_subsets(grams, SUBSETS, **kw, jobs=pl.DataFrame(jobs))
        for a, b in zip(rows, table, strict=True):
            assert np.array_equal(a["coef"], b["coef"], equal_nan=True)
            assert a["active"] == b["active"]

    def test_no_jobs_fit_nothing(self):
        grams = blocks(frame(seed=16))
        assert pg.solve_subsets(grams, SUBSETS, path={}, within=True, jobs=[]) == []
        assert pg.solve_subsets(grams, SUBSETS, within=True, jobs=[]) == []


class TestRidge:
    def test_each_job_is_the_call_on_its_columns_target_and_draw(self):
        grams = blocks(frame(seed=17))
        # Jobs with columns of their own, and two that share theirs, whose
        # fits share one factorization and are each its own fit still.
        jobs = signed_jobs(18)[:20] + [
            {"subset": 0, "draw": 1, "target": t, "features": None} for t in ("y2", "y")
        ]
        for kw in (dict(ridge=0.0), dict(ridge=0.2, standardize=True), dict(ridge=[0.0, 0.5])):
            got = pg.solve_subsets(grams, SUBSETS, within=True, target_signs=SIGNS, jobs=jobs, **kw)
            for job, fit in zip(jobs, got, strict=True):
                # One draw, one subset, one target.
                (((want,),),) = pg.solve_subsets(
                    grams,
                    [SUBSETS[job["subset"]]],
                    within=True,
                    target_signs=SIGNS[:, [job["draw"]]],
                    features=job["features"],
                    targets=[job["target"]],
                    **kw,
                )
                assert fit.keys() == want.keys()
                for key, value in want.items():
                    assert np.array_equal(fit[key], value, equal_nan=True), (key, kw)


@pytest.mark.parametrize(
    ("job", "kw", "error", "match"),
    [
        ("x", {}, ValueError, r"jobs\[0\] is a str, not a mapping"),
        ({"subset": 0, "target": "y", "lag": 1}, {}, ValueError, r"jobs\[0\] has \['lag'\]"),
        ({"target": "y"}, {}, ValueError, r"jobs\[0\] names no 'subset'"),
        ({"subset": 0}, {}, ValueError, r"jobs\[0\] names no 'target'"),
        ({"subset": 5, "target": "y"}, {}, ValueError, r"'subset' is a position .*0\.\.4; got 5"),
        ({"subset": True, "target": "y"}, {}, ValueError, r"'subset' is a position"),
        ({"subset": 0, "target": "z"}, {}, KeyError, r"jobs\[0\]: no target 'z'"),
        ({"subset": 0, "target": 2}, {}, IndexError, r"jobs\[0\]: target 2 out of range"),
        ({"subset": 0, "target": "y", "draw": 0}, {}, ValueError, r"needs target_signs"),
        ({"subset": 0, "target": "y"}, {"target_signs": SIGNS}, ValueError, r"names no 'draw'"),
        (
            {"subset": 0, "target": "y", "draw": 3},
            {"target_signs": SIGNS},
            ValueError,
            r"'draw' is a column of target_signs, 0\.\.2; got 3",
        ),
        ({"subset": 0, "target": "y", "features": ["x0", "w"]}, {}, KeyError, r"no column 'w'"),
        ({"subset": 0, "target": "y", "features": [0, 99]}, {}, IndexError, r"column 99 out"),
        (
            {"subset": 0, "target": "y", "features": ["x0", "intercept"]},
            {},
            ValueError,
            r"'intercept' is a constant column",
        ),
        ({"subset": 0, "target": "y", "features": "x0"}, {}, ValueError, r"not the one name"),
        (
            {"subset": 0, "target": "y", "features": ["x0", "x1"], "penalty_weights": [1.0]},
            {"path": {}},
            ValueError,
            r"jobs\[0\]: penalty_weights must have one entry per feature \(2\)",
        ),
        (
            {"subset": 0, "target": "y", "features": ["x0", "x1"]},
            {"path": {"penalty_weights": [1.0] * 3}},
            ValueError,
            r"one entry per feature \(2\)",
        ),
        (
            {"subset": 0, "target": "y", "features": ["x0"], "penalty_weights": [0.0]},
            {"path": {}},
            ValueError,
            r"finite and > 0",
        ),
        (
            {"subset": 0, "target": "y", "penalty_weights": [1.0] * K},
            {},
            ValueError,
            r"only a path reads",
        ),
        ({"subset": 0, "target": "y"}, {"targets": ["y"]}, ValueError, r"takes no targets="),
    ],
)
def test_a_malformed_job_is_refused_by_name(job, kw, error, match):
    grams = blocks(frame(seed=19))
    kw = {"within": True, **kw}
    with pytest.raises(error, match=match):
        pg.solve_subsets(grams, SUBSETS, jobs=[job], **kw)


def test_jobs_that_are_no_list_of_jobs_are_refused():
    grams = blocks(frame(seed=20))
    with pytest.raises(ValueError, match=r"jobs is a list of mappings.*got int"):
        pg.solve_subsets(grams, SUBSETS, within=True, jobs=3)
    with pytest.raises(ValueError, match=r"jobs\[1\] names no 'target'"):
        pg.solve_subsets(
            grams, SUBSETS, within=True, jobs=[{"subset": 0, "target": 0}, {"subset": 1}]
        )


# --- a column named twice (task 269, D-3) -------------------------------------
#
# The merged design holds a column once, so a column named twice was solved
# twice -- singular: `nan`, or an arbitrary split by rounding -- and each
# slot's coefficient written to the one output column, the last write winning
# beside the fit's own `r2`. Refused by name wherever a fit reads a column
# set; `subset()` is a selection and keeps taking one.


def test_a_column_named_twice_in_features_is_refused_by_name():
    grams = blocks(frame(seed=21))
    x1 = grams[0]["columns"].index("x1")  # named, then by its position
    calls = {
        "solve_subsets": lambda f: pg.solve_subsets(grams, [[0]], features=f),
        "solve_subsets path": lambda f: pg.solve_subsets(grams, [[0]], features=f, path={}),
        "solve": lambda f: pg.solve(grams[0], features=f),
        "lasso_path": lambda f: pg.lasso_path(grams[0], penalties=[0.1], features=f),
        "lars_path": lambda f: pg.lars_path(grams[0], features=f),
        "lars_paths": lambda f: pg.lars_paths(grams, features=f),
        "coef_stats": lambda f: pg.coef_stats(grams[0], [0.0] * (K + 1), features=f),
        "vif": lambda f: pg.vif(grams[0], features=f),
    }
    for call in calls.values():
        for feats in (["x0", "x0"], ["x0", "x1", "x1"], ["x1", x1]):
            with pytest.raises(ValueError, match=r"names column 'x\d' twice"):
                call(feats)
    # A selection is not a fit: `subset` keeps taking a column twice.
    assert pg.subset(grams[0], ["x0", "x0"])["columns"][-2:] == ["x0", "x0"]


@pytest.mark.parametrize("path", [None, {}])
def test_a_job_naming_a_column_twice_is_refused_naming_the_job(path):
    grams = blocks(frame(seed=22))
    jobs = [
        {"subset": 0, "target": "y", "features": ["x0", "x1"]},
        {"subset": 1, "target": "y", "features": ["x0", "x1", "x1"]},
    ]
    with pytest.raises(ValueError, match=r"jobs\[1\]: 'features' names column 'x1' twice"):
        pg.solve_subsets(grams, SUBSETS, within=True, jobs=jobs, path=path)


def test_the_native_fits_refuse_a_column_twice():
    """The native calls, reached without the Python check, refuse a slot
    named twice too."""
    from polars_online import _polars_online as native

    grams = blocks(frame(seed=23))
    cols = grams[0]["columns"]
    icept = cols.index("intercept")
    x0, x1 = cols.index("x0"), cols.index("x1")
    gin = [pg._native_gram(np, g, icept) for g in grams]
    calls = {
        "gram_ridge_subsets": lambda s: native.gram_ridge_subsets(
            gin, [[0]], [0], s, icept, 0.0, False, False, []
        ),
        "gram_path_subsets": lambda s: native.gram_path_subsets(
            gin, [[0]], [0], s, icept, [], None, None, False, []
        ),
        "gram_lars_paths": lambda s: native.gram_lars_paths(
            gin[:1], [[0]], s, icept, [], None, None
        ),
        "gram_cd_path": lambda s: native.gram_cd_path(
            gin[0], 0, s, icept, [0.1], 1.0, [1.0] * len(s), 100, 1e-7
        ),
        "gram_ridge_jobs": lambda s: native.gram_ridge_jobs(
            gin, [[0]], [s], [(0, 0, 0, 0)], icept, 0.0, False, False, []
        ),
        "gram_path_jobs": lambda s: native.gram_path_jobs(
            gin, [[0]], [s], [[]], [(0, 0, 0, 0)], icept, None, None, False, []
        ),
    }
    for call in calls.values():
        call([x0, x1])  # the same call with each column once runs
        with pytest.raises(ValueError, match=rf"column {x1} .*twice"):
            call([x0, x1, x1])


# --- numpy integers (task 269, D-6) ------------------------------------------


def test_a_job_takes_numpy_integers_and_refuses_a_bool():
    """A job's `target` and `features` take numpy's integers, as its
    `subset` and `draw` do and as a DataFrame of jobs hands over; a bool is
    no position."""
    grams = blocks(frame(seed=24))
    x1 = grams[0]["columns"].index("x1")
    plain = {"subset": 0, "target": 0, "features": ["x0", x1]}
    numpy = {"subset": np.int64(0), "target": np.int64(0), "features": ["x0", np.int32(x1)]}
    a = pg.solve_subsets(grams, SUBSETS, within=True, jobs=[plain])
    b = pg.solve_subsets(grams, SUBSETS, within=True, jobs=[numpy])
    assert np.array_equal(a[0]["coef"], b[0]["coef"], equal_nan=True)
    with pytest.raises(KeyError, match=r"jobs\[0\]: no target True"):
        pg.solve_subsets(grams, SUBSETS, within=True, jobs=[{"subset": 0, "target": True}])
    with pytest.raises(KeyError, match=r"jobs\[0\]: no column True"):
        pg.solve_subsets(
            grams, SUBSETS, within=True, jobs=[{"subset": 0, "target": 0, "features": [True]}]
        )
