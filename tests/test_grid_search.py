"""`po.eval.grid_search` (task 336): every setting scored in one pass, and the
choice scored on rows it had not seen.

Each claim is held against a computation that does not share the code's
arithmetic:

1. the spans are scikit-learn's `TimeSeriesSplit`, and the grid's points
   scikit-learn's `ParameterGrid`, in its order;
2. every score, choice and count is the brute-force computation over the
   whole output frame held in memory (`by_hand`), on the rows every setting
   scored, with and without a `gap`;
3. the choice is read from earlier rows only: on a stream built so that the
   setting best before the cut is not the one best after it, the choice is
   the first, and a choice read from the span itself would be the second;
4. the result is the same for any chunk size, to rounding, and for a frame
   as for a query;
5. every refusal names what to give instead.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

import polars_online as po
from polars_online.eval import _points, _time_series_folds

TIER = "essential"

pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")


def wander(path, n=3_000, seed=9):
    """y = b x0 + 0.5 x1 + noise, b a random walk, saved and scanned."""
    rng = np.random.default_rng(seed)
    x0, x1, e = rng.standard_normal((3, n))
    b = 1.0 + np.cumsum(rng.normal(0, 0.03, n))
    pl.DataFrame(
        {"t": np.arange(n, dtype=float), "x0": x0, "x1": x1, "y": b * x0 + 0.5 * x1 + e}
    ).write_parquet(path)
    return pl.scan_parquet(path)


def by_hand(out, specs, n_splits, gap=0, field_prefix="pred_y", target="y", loglik=False):
    """The search computed from the whole output: each setting's squared error
    (or loglik) on every row, the rows every setting has, the spans as
    `TimeSeriesSplit` gives them, and each span's choice from the rows before
    it less the gap."""
    columns = {}
    for spec in specs:
        fields = [
            f.name
            for f in out.schema[spec["name"]].fields
            if (f.name.startswith("loglik") if loglik else f.name.startswith(field_prefix))
            and "__selected" not in f.name
            and "__averaged" not in f.name
        ]
        for field in fields:
            value = out[spec["name"]].struct.field(field).to_numpy()
            if not loglik:
                value = (out[target].to_numpy() - value) ** 2
            columns[(spec["name"], field)] = value
    keys = list(columns)
    matrix = np.column_stack([columns[k] for k in keys])
    usable = np.isfinite(matrix).all(axis=1)
    n = out.height
    best = np.argmax if loglik else np.argmin
    picks, honest = [], []
    tests = np.zeros(n, dtype=bool)
    if n_splits == 1:  # scikit-learn wants two; one is the halves
        start = n - n // 2
        splits = [(np.arange(start - gap), np.arange(start, n))]
    else:
        splits = TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.zeros(n))
    for train, test in splits:
        rows = train[usable[train]]
        pick = int(best(matrix[rows].mean(axis=0)))
        picks.append(keys[pick])
        honest.extend(matrix[test[usable[test]], pick])
        tests[test] = True
    scored = tests & usable
    means = dict(zip(keys, matrix[scored].mean(axis=0), strict=True))
    return {
        "picks": picks,
        "honest": float(np.mean(honest)),
        "means": means,
        "n": int(scored.sum()),
    }


# -- 1. the spans and the grid, against scikit-learn ---------------------------------


@pytest.mark.parametrize(
    ("n", "n_splits", "gap"),
    [
        (100, 2, 0),
        (101, 3, 0),
        (1_000, 5, 0),
        (1_000, 5, 7),
        (37, 4, 2),
        (12, 11, 0),
        (999, 2, 300),
    ],
)
def test_the_spans_are_time_series_splits(n, n_splits, gap):
    ours = _time_series_folds(n, n_splits, gap)
    theirs = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.zeros(n)))
    assert len(ours) == len(theirs) == n_splits
    for (choose_end, start, end), (train, test) in zip(ours, theirs, strict=True):
        assert (0, choose_end) == (train[0], train[-1] + 1)
        assert (start, end) == (test[0], test[-1] + 1)


def test_one_split_chooses_on_the_first_half():
    """`TimeSeriesSplit` wants two splits or more; one is the first half against
    the second, which the guide's hand form computes."""
    assert _time_series_folds(40_000, 1, 0) == [(20_000, 20_000, 40_000)]
    assert _time_series_folds(41, 1, 0) == [(21, 21, 41)]


def test_the_points_are_a_parameter_grid_in_its_order():
    grid = {"learning_rate": [0.1, 0.01, 0.3], "half_life": [10.0, 50.0], "eps": [0.5]}
    assert _points(grid) == list(ParameterGrid(grid))


# -- 2. every number against the whole output --------------------------------------


@pytest.mark.parametrize(("n_splits", "gap"), [(1, 0), (4, 0), (5, 25)])
def test_a_grid_in_one_spec_is_scored_as_by_hand(tmp_path, n_splits, gap):
    lf = wander(tmp_path / "w.parquet")
    spec = po.spec.ewridge(
        "g",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=5.0,
        half_life=[10.0, 40.0, 160.0],
        ridge=[1e-6, 0.5],
        emit_selected=True,
    )
    table = po.eval.grid_search(lf, spec, n_splits=n_splits, gap=gap)
    want = by_hand(lf.online.fit_predict([spec]).collect(), [spec], n_splits, gap)
    settings = table.drop_nulls("rank")
    assert settings.height == 6
    for row in settings.iter_rows(named=True):
        assert row["n"] == want["n"]
        assert row["mse"] == pytest.approx(want["means"][("g", row["slot"])], rel=1e-12)
        assert row["folds_chosen"] == sum(p == ("g", row["slot"]) for p in want["picks"])
    assert set(settings["half_life"]) == {10.0, 40.0, 160.0}
    assert set(settings["ridge"]) == {1e-6, 0.5}
    chosen = table.filter(pl.col("spec").is_null()).row(0, named=True)
    assert chosen["slot"] == "pred_y__chosen"
    assert chosen["n"] == want["n"]
    assert chosen["mse"] == pytest.approx(want["honest"], rel=1e-12)
    # The rank is the order of the scores over the test spans, 1 the best.
    order = settings.sort("mse")["rank"].to_list()
    assert order == sorted(order) and order[0] == 1
    # emit_selected's prediction is scored on the same rows, and never ranked.
    selected = table.filter(slot="pred_y__selected").row(0, named=True)
    assert selected["rank"] is None and selected["folds_chosen"] is None
    assert selected["n"] == want["n"]


def test_specs_that_warm_up_apart_are_scored_on_the_rows_they_share(tmp_path):
    """One setting withholds its first 400 rows: every setting is scored only
    where both predict, and a span's choice reads only those rows."""
    lf = wander(tmp_path / "w.parquet")
    builder = functools.partial(
        po.spec.ewridge,
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=5.0,
        half_life=50.0,
    )
    table = po.eval.grid_search(lf, builder, {"min_weight": [3.0, 60.0]}, n_splits=3)
    specs = [builder("min_weight=3.0", min_weight=3.0), builder("min_weight=60.0", min_weight=60.0)]
    want = by_hand(lf.online.fit_predict(specs).collect(), specs, 3)
    for row in table.drop_nulls("rank").iter_rows(named=True):
        assert row["n"] == want["n"]
        assert row["mse"] == pytest.approx(want["means"][(row["spec"], "pred_y")], rel=1e-12)
    assert table.filter(pl.col("spec").is_null())["mse"].item() == pytest.approx(
        want["honest"], rel=1e-12
    )
    assert table["min_weight"].to_list() == [3.0, 60.0, None]


def test_a_builder_names_each_spec_by_its_point(tmp_path):
    lf = wander(tmp_path / "w.parquet", n=1_200)
    sgd = functools.partial(po.spec.sgd, targets=["y"], features=["x0", "x1"], half_life=200.0)
    grid = {"learning_rate": [0.003, 0.03], "l2": [0.0, 0.1]}
    table = po.eval.grid_search(lf, sgd, grid, n_splits=2)
    points = list(ParameterGrid(grid))
    assert table["spec"].to_list()[:4] == [
        ", ".join(f"{k}={v}" for k, v in p.items()) for p in points
    ]
    assert table.columns == [
        "spec",
        "l2",
        "learning_rate",
        "slot",
        "target",
        "n",
        "mse",
        "rank",
        "folds_chosen",
    ]
    assert table["l2"].to_list()[:4] == [p["l2"] for p in points]


def test_loglik_is_scored_highest_best(tmp_path):
    rng = np.random.default_rng(14)
    n = 3_000
    regime = np.cumsum(rng.random(n) < 1 / 200)
    x = rng.normal(0, 2.0, regime.max() + 1)[regime] + rng.standard_normal(n)
    pl.DataFrame({"x": x}).write_parquet(tmp_path / "j.parquet")
    lf = pl.scan_parquet(tmp_path / "j.parquet")
    bocpd = functools.partial(
        po.spec.bocpd,
        features=["x"],
        mean_prior=[0.0],
        covariance_prior=[1.0],
        mean_precision_prior=0.25,
        degrees_of_freedom_prior=2.0,
    )
    hazards = [10.0, 200.0, 5_000.0]
    table = po.eval.grid_search(lf, bocpd, {"hazard": hazards}, n_splits=3, scoring="loglik")
    specs = [bocpd(f"hazard={h}", hazard=h) for h in hazards]
    want = by_hand(lf.online.fit_predict(specs).collect(), specs, 3, loglik=True)
    settings = table.drop_nulls("rank")
    for row in settings.iter_rows(named=True):
        assert row["slot"] == "loglik" and row["target"] is None
        assert row["loglik"] == pytest.approx(want["means"][(row["spec"], "loglik")], rel=1e-12)
    assert settings.sort("loglik", descending=True)["rank"][0] == 1
    chosen = table.filter(pl.col("spec").is_null()).row(0, named=True)
    assert chosen["slot"] == "loglik__chosen"
    assert chosen["loglik"] == pytest.approx(want["honest"], rel=1e-12)


# -- 3. the choice reads earlier rows only -----------------------------------------


def test_the_choice_reads_only_the_rows_before_its_span(tmp_path):
    """The coefficient sits still for the first half and jumps every 50 rows in
    the second, so a long memory is best before the cut and a short one after.
    Chosen from the rows before the span, the long one runs on it; a choice
    read from the span itself would run the short one, at the lower score the
    short one shows there."""
    rng = np.random.default_rng(2)
    n = 4_000
    x, e = rng.standard_normal((2, n))
    b = np.where(np.arange(n) < n // 2, 1.0, rng.choice([-2.0, 2.0], n // 50).repeat(50)[:n])
    b[: n // 2] = 1.0
    pl.DataFrame({"x": x, "y": b * x + e}).write_parquet(tmp_path / "cut.parquet")
    lf = pl.scan_parquet(tmp_path / "cut.parquet")
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=[5.0, 2_000.0])
    table = po.eval.grid_search(lf, spec, n_splits=1)
    long, short = (table.filter(slot=s).row(0, named=True) for s in ("pred_y@h2000", "pred_y@h5"))
    assert long["folds_chosen"] == 1 and short["folds_chosen"] == 0
    assert short["rank"] == 1  # best on the span, in hindsight
    chosen = table.filter(pl.col("spec").is_null())["mse"].item()
    assert chosen == pytest.approx(long["mse"], rel=1e-12)
    assert chosen > 1.5 * short["mse"]


def test_the_gap_keeps_the_choice_from_rows_not_yet_known(tmp_path):
    """With a gap of g, a span's choice reads the rows before its start less g:
    the same as by hand with `TimeSeriesSplit(gap=g)`, and a different choice
    here from the one with no gap, as the planted best flips just before the
    span starts."""
    rng = np.random.default_rng(4)
    n = 2_000
    x, e = rng.standard_normal((2, n))
    b = np.ones(n)
    b[850:1_000] = rng.choice([-3.0, 3.0], 6).repeat(
        25
    )  # moves in the 150 rows before the span at 1,000
    pl.DataFrame({"x": x, "y": b * x + 0.5 * e}).write_parquet(tmp_path / "g.parquet")
    lf = pl.scan_parquet(tmp_path / "g.parquet")
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=[3.0, 1_000.0])
    out = lf.online.fit_predict([spec]).collect()
    picks = {}
    for gap in (0, 150):
        table = po.eval.grid_search(lf, spec, n_splits=1, gap=gap)
        want = by_hand(out, [spec], 1, gap)
        picks[gap] = table.filter(pl.col("folds_chosen") == 1)["slot"].item()
        assert ("m", picks[gap]) == want["picks"][0]
        assert table.filter(pl.col("spec").is_null())["mse"].item() == pytest.approx(
            want["honest"], rel=1e-12
        )
    assert picks == {0: "pred_y@h3", 150: "pred_y@h1000"}


# -- 4. chunk sizes, frames and queries --------------------------------------------


def test_the_result_is_the_same_for_any_chunk_size_and_for_a_frame(tmp_path):
    lf = wander(tmp_path / "w.parquet", n=900)
    spec = po.spec.ewridge(
        "g", targets=["y"], features=["x0", "x1"], half_life=[10.0, 80.0], ridge=[1e-6, 1.0]
    )
    whole = po.eval.grid_search(lf, spec, n_splits=4)
    for table in (
        po.eval.grid_search(lf, spec, n_splits=4, chunk_size=1),
        po.eval.grid_search(lf, spec, n_splits=4, chunk_size=97),
        po.eval.grid_search(lf.collect(), spec, n_splits=4),
    ):
        assert table.drop("mse").equals(whole.drop("mse"))
        for a, b in zip(table["mse"], whole["mse"], strict=True):
            assert math.isclose(a, b, rel_tol=1e-12)


# -- 5. refusals --------------------------------------------------------------------


def test_refusals_name_what_to_give(tmp_path):
    lf = wander(tmp_path / "w.parquet", n=600)
    spec = po.spec.ewridge("g", targets=["y"], features=["x0"], half_life=[10.0, 80.0])
    sgd = functools.partial(po.spec.sgd, targets=["y"], features=["x0"], half_life=50.0)
    with pytest.raises(ValueError, match="scoring is one of"):
        po.eval.grid_search(lf, spec, scoring="r2")
    with pytest.raises(ValueError, match="n_splits must be"):
        po.eval.grid_search(lf, spec, n_splits=0)
    with pytest.raises(ValueError, match="gap must be"):
        po.eval.grid_search(lf, spec, gap=-1)
    with pytest.raises(ValueError, match="too few"):
        po.eval.grid_search(lf, spec, n_splits=2, gap=200)
    with pytest.raises(TypeError, match="param_grid needs a spec builder"):
        po.eval.grid_search(lf, spec, {"half_life": [1.0]})
    with pytest.raises(TypeError, match="param_grid is a dict"):
        po.eval.grid_search(lf, sgd)
    with pytest.raises(TypeError, match="list of the values"):
        po.eval.grid_search(lf, sgd, {"learning_rate": 0.1})
    with pytest.raises(TypeError, match="list of the values"):
        po.eval.grid_search(lf, sgd, {"loss": "huber"})
    with pytest.raises(ValueError, match="one of the result's own columns"):
        po.eval.grid_search(lf, sgd, {"rank": [1.0]})
    with pytest.raises(ValueError, match="named by its first argument"):
        po.eval.grid_search(
            lf,
            lambda name, learning_rate: sgd("x", learning_rate=learning_rate),
            {"learning_rate": [0.1]},
        )
    with pytest.raises(TypeError, match="LazyFrame or DataFrame"):
        po.eval.grid_search({"y": [1.0]}, spec)
    with pytest.raises(TypeError, match="a spec, a list of them, or a builder"):
        po.eval.grid_search(lf, [])
    with pytest.raises(ValueError, match='scoring="loglik"'):
        po.eval.grid_search(lf, po.spec.ew_cov("c", features=["x0", "x1"], half_life=50.0))
    with pytest.raises(ValueError, match="writes no loglik"):
        po.eval.grid_search(lf, spec, scoring="loglik")
    late = po.spec.ewridge(
        "late", targets=["y"], features=["x0"], half_life=[10.0, 80.0], min_weight=1e9
    )
    with pytest.raises(ValueError, match="nothing to choose by"):
        po.eval.grid_search(lf, late, n_splits=2)
