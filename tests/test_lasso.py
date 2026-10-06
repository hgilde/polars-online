"""Task 10: lasso path + online lambda selection (docs/PLAN.md section 4.3)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _spec(path, **kw):
    defaults = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        lasso_path=path,
        half_life=1e9,
        max_rows_between_solves=1,
        min_weight=10.0,
    )
    defaults.update(kw)
    return po.spec.lasso("m", **defaults)


def test_output_fields_include_lam_selected():
    spec = _spec([1.0, 0.1, 0.0])
    fields = po.spec.output_fields(spec)
    assert "penalty_selected_y0" in fields
    assert "pred_y0__l1" in fields and "pred_y0__l0" in fields


def test_zero_penalty_matches_ewridge():
    # lambda = 0 lasso == unpenalized least squares == ew_ridge with tiny ridge.
    df, _ = synthetic(seed=31, n_groups=1, n_rows=300, k=3, null_frac=0.0)
    common = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        half_life=1e9,
        max_rows_between_solves=1,
        min_weight=10.0,
    )
    a = po.ModelBank([po.spec.lasso("m", lasso_path=[0.0], **common)]).fit_predict(df)
    b = po.ModelBank([po.spec.ewridge("m", ridge=1e-12, standardize=True, **common)]).fit_predict(
        df
    )
    # lasso fields are always suffixed by their path point
    pa = a["m"].struct.field("pred_y0__l0").to_numpy().astype(float)
    pb = b["m"].struct.field("pred_y0").to_numpy().astype(float)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.sum() > 200
    assert np.max(np.abs(pa[m] - pb[m])) < 1e-6


def test_heavy_penalty_predicts_the_mean():
    df, _ = synthetic(seed=32, n_groups=1, n_rows=200, k=3, null_frac=0.0)
    out = po.ModelBank([_spec([1e6, 0.0])]).fit_predict(df)
    # All slopes zeroed => prediction is the running EW mean of y, so it must be
    # constant across rows with very different x.
    pred = out["m"].struct.field("pred_y0__l1000000").to_numpy().astype(float)
    finite = pred[np.isfinite(pred)]
    assert finite.std() < 0.5 * df["y0"].std()


def test_lam_selected_is_on_the_path():
    df, _ = synthetic(seed=33, n_groups=2, n_rows=250, k=3, null_frac=0.0)
    path = [1.0, 0.3, 0.1, 0.0]
    out = po.ModelBank([_spec(path, group="group", half_life=200.0)]).fit_predict(df)
    sel = out["m"].struct.field("penalty_selected_y0").to_numpy().astype(float)
    assert set(np.unique(sel[np.isfinite(sel)])).issubset(set(path))


def test_selection_prefers_penalty_when_features_are_noise():
    # y is pure noise: no feature helps, so a penalty that zeroes everything
    # must beat lambda = 0 on out-of-sample error most of the time.
    rng = np.random.default_rng(5)
    n = 3000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )
    out = po.ModelBank([_spec([10.0, 0.0], half_life=300.0)]).fit_predict(df)
    sel = out["m"].struct.field("penalty_selected_y0").to_numpy().astype(float)
    sel = sel[np.isfinite(sel)]
    assert (sel == 10.0).mean() > 0.8


def test_an_infinite_select_halflife_selects_on_the_plain_mean():
    """``select_half_life = inf`` scores each penalty by its plain mean
    squared out-of-sample error over every row so far (review 2026-09-12,
    S27); the builder refused it. numpy over the bank's own residual columns:
    the selection a row reports is the argmin coming into it."""
    rng = np.random.default_rng(5)
    n = 900
    x = rng.standard_normal((n, 3))
    # Signal, then none: the unpenalized fit leads, and the plain mean hands
    # the lead to a penalty long after an EW mean would have.
    y = np.where(np.arange(n) < 300, 1.5 * x[:, 0], 0.0) + rng.standard_normal(n)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y0": y})
    path = [1.0, 0.1, 0.0]
    out = po.ModelBank([_spec(path, select_half_life=float("inf"))]).fit_predict(df)["m"]
    resid = np.array(
        [out.struct.field(f"resid_y0__l{lab}").to_numpy() for lab in ("1", "0.1", "0")], float
    )
    sel = out.struct.field("penalty_selected_y0").to_numpy().astype(float)
    counted = np.isfinite(resid[0])
    e2 = np.where(counted, resid, 0.0) ** 2
    before_sum = np.concatenate([np.zeros((3, 1)), np.cumsum(e2, axis=1)[:, :-1]], axis=1)
    before_n = np.concatenate([[0], np.cumsum(counted)[:-1]])
    picked = []
    for i in np.flatnonzero((before_n > 0) & np.isfinite(sel)):
        mean = before_sum[:, i] / before_n[i]
        lo, second = np.sort(mean)[:2]
        if second - lo <= 1e-9 * second:
            continue  # a near tie: the running mean's rounding may break it
        assert sel[i] == path[int(np.argmin(mean))], i
        picked.append(sel[i])
    assert len(picked) > 500 and len(set(picked)) > 1, set(picked)


def test_chunk_invariance():
    df, _ = synthetic(seed=34, n_groups=2, n_rows=180, k=3, null_frac=0.0)
    spec = _spec([0.5, 0.0], group="group", half_life=200.0)
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")

    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_path_must_be_decreasing():
    with pytest.raises(ValueError, match="decreasing"):
        _spec([0.1, 1.0])


class TestEachTargetsSelectionCountsFromItsOwnMinWeight:
    """A target's ``penalty_selected`` counts its errors from the row where its
    own weight reaches its own ``min_weight`` (review 2026-10-05, CA3). They
    counted from the model's first prediction, which the list's smallest
    threshold gates, so one target's choice moved with another's threshold:
    ``[0, 60]`` and ``[60, 60]`` chose apart for ``y1`` on 84 rows from row 60."""

    @staticmethod
    def _out(min_weight) -> pl.DataFrame:
        rng = np.random.default_rng(4)
        n = 300
        x = rng.normal(size=(n, 3))
        y0 = 0.5 + 2.0 * x[:, 0] + 0.3 * rng.normal(size=n)
        y1 = -1.0 + x[:, 1] + 0.5 * x[:, 2] + 0.6 * rng.normal(size=n)
        df = pl.DataFrame(
            {"t": np.arange(n, dtype=float), "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2]}
            | {"y0": y0, "y1": y1}
        )
        spec = po.spec.lasso(
            "l",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=5.0,
            half_life=float("inf"),
            lasso_path=[0.5, 0.1, 0.01, 0.0],
            select_half_life=float("inf"),
            min_weight=min_weight,
            solve_every=0.0,
            max_rows_between_solves=1,
        )
        return po.ModelBank([spec]).fit_predict(df)["l"].struct.unnest()

    def test_another_targets_threshold_does_not_move_a_targets_choice(self):
        a, b = (
            self._out(mw)["penalty_selected_y1"].to_numpy() for mw in ([0.0, 60.0], [60.0, 60.0])
        )
        wrong = np.flatnonzero(a != b)
        assert wrong.size == 0, f"{wrong.size} rows differ, the first {wrong[:5]}"
        assert len(set(a[60:])) > 1, "the choice never moves, so this compares nothing"

    def test_a_list_moves_only_the_targets_above_its_minimum(self):
        listed, low = self._out([0.0, 60.0]), self._out([0.0, 0.0])
        # `y0`'s threshold is the list's minimum both times: nothing of it
        # moves, and the model's predictions are the same ones throughout.
        assert listed["penalty_selected_y0"].equals(low["penalty_selected_y0"])
        shown = listed["pred_y1__l0"].is_not_null()
        for c in (c for c in listed.columns if c.startswith("pred_")):
            got, want = listed[c], low[c]
            if c.startswith("pred_y1"):
                got, want = got.filter(shown), want.filter(shown)
            assert got.equals(want), c
        # `y1`'s is above it, so its choice moves.
        assert not listed["penalty_selected_y1"].equals(low["penalty_selected_y1"])

    def test_a_scalar_and_a_list_of_equal_values_agree(self):
        assert self._out(30.0).equals(self._out([30.0, 30.0]), null_equal=True)
