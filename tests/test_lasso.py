"""Task 10: lasso path + online lambda selection (docs/PLAN.md section 4.3)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic

TIER = "essential"


def _spec(path, **kw):
    defaults = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        penalties=path,
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
    assert "pred_y0__p1" in fields and "pred_y0__p0" in fields


def test_zero_penalty_matches_ewridge():
    # lambda = 0 lasso == unpenalized least squares == ewridge with tiny ridge.
    df, _ = synthetic(seed=31, n_groups=1, n_rows=300, k=3, null_frac=0.0)
    common = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        half_life=1e9,
        max_rows_between_solves=1,
        min_weight=10.0,
    )
    a = po.ModelBank([po.spec.lasso("m", penalties=[0.0], **common)]).fit_predict(df)
    b = po.ModelBank([po.spec.ewridge("m", ridge=1e-12, standardize=True, **common)]).fit_predict(
        df
    )
    # a path of one point writes no suffix, as a single ridge writes none
    pa = a["m"].struct.field("pred_y0").to_numpy().astype(float)
    pb = b["m"].struct.field("pred_y0").to_numpy().astype(float)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.sum() > 200
    assert np.max(np.abs(pa[m] - pb[m])) < 1e-6


def test_heavy_penalty_predicts_the_mean():
    df, _ = synthetic(seed=32, n_groups=1, n_rows=200, k=3, null_frac=0.0)
    out = po.ModelBank([_spec([1e6, 0.0])]).fit_predict(df)
    # All slopes zeroed => prediction is the running EW mean of y, so it must be
    # constant across rows with very different x.
    pred = out["m"].struct.field("pred_y0__p1000000").to_numpy().astype(float)
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
    # `min_weight` that lets out the rows from the model's floor, four rows
    # (a row per coefficient), from which the selection counts whatever
    # `min_weight` says: the rows the output scores are then the rows the
    # selection does.
    # (3.5, as the decay of a half-life of 1e9 leaves four rows' weight
    # short of 4.)
    spec = _spec(path, select_half_life=float("inf"), min_weight=3.5)
    out = po.ModelBank([spec]).fit_predict(df)["m"]
    resid = np.array(
        [out.struct.field(f"resid_y0__p{lab}").to_numpy() for lab in ("1", "0.1", "0")], float
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


class TestEachTargetsSelectionCountsFromTheFloor:
    """A target's ``penalty_selected`` counts its errors from the row the fit
    holds a row of it per coefficient, the model's floor, whatever
    ``min_weight`` says: the gate withholds the output alone (task 278, D10).
    They counted from the row where its own weight reached its own
    ``min_weight`` (review 2026-10-05, CA3), and before that from the model's
    first prediction, which the list's smallest threshold gates, so one
    target's choice moved with another's threshold: ``[0, 60]`` and ``[60,
    60]`` chose apart for ``y1`` on 84 rows from row 60."""

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
            penalties=[0.5, 0.1, 0.01, 0.0],
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
        wrong = np.flatnonzero((a != b) & ~(np.isnan(a) & np.isnan(b)))
        assert wrong.size == 0, f"{wrong.size} rows differ, the first {wrong[:5]}"
        # From the floor's row, 4, where its errors start to count.
        low = self._out([0.0, 0.0])["penalty_selected_y1"].to_numpy()
        assert len(set(low[4:])) > 1, "the choice never moves, so this compares nothing"

    def test_no_threshold_moves_a_choice_or_a_prediction_it_shows(self):
        low = self._out([0.0, 0.0])
        for mw in ([0.0, 60.0], [60.0, 60.0], 200.0):
            gated = self._out(mw)
            for j in ("y0", "y1"):
                col = f"penalty_selected_{j}"
                assert gated[col].equals(low[col], null_equal=True), (mw, col)
            # A prediction the gate lets out is the ungated one, and it
            # withholds rows the selection scored.
            withheld = 0
            for c in (c for c in gated.columns if c.startswith("pred_")):
                shown = gated[c].is_not_null()
                withheld += int((~shown & low[c].is_not_null()).sum())
                assert gated[c].filter(shown).equals(low[c].filter(shown)), (mw, c)
            assert withheld > 0, f"{mw}: the gate withheld nothing"
        assert len(set(low["penalty_selected_y1"].to_list()[4:])) > 1, "the choice moves"

    def test_a_scalar_and_a_list_of_equal_values_agree(self):
        assert self._out(30.0).equals(self._out([30.0, 30.0]), null_equal=True)


def test_on_the_row_clock_solve_every_is_its_row_cap():
    """Without a clock column the clock is the row's number, the first row
    being 1, so ``solve_every=n`` solves on the rows
    ``max_rows_between_solves=n`` does: the first period ends on the
    ``n``-th row, solving on the default weight-share schedule before it --
    every row at this half-life -- and then every ``n``-th row (task 278,
    "Default cadence first"; until then both counted from the first solve,
    on the third row, the floor of a row per coefficient). Measured from the
    first row, whose step is 0, it solved a row later, the whole stream
    long, which ``coef`` showed (task 278, D13)."""
    import warnings

    rng = np.random.default_rng(0)
    n = 60
    x = rng.standard_normal(n)
    df = pl.DataFrame({"x0": x, "x1": rng.standard_normal(n), "y0": 2.0 * x})

    def run(**kw):
        spec = po.spec.lasso(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            penalties=[0.01],
            half_life=20.0,
            min_weight=1e9,
            max_rows_between_coefs=1,
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            return po.ModelBank([spec]).fit_predict(df).unnest("m")

    clock = run(solve_every=7.0)
    assert clock.equals(run(solve_every=1e9, max_rows_between_solves=7), null_equal=True)
    c = clock["coef"].to_list()
    assert [i for i in range(1, n) if c[i] != c[i - 1]] == [1, 2, 3, 4, 5, *range(6, n, 7)]


# ---- the keywords against each other (task 278, batch 27) ----


def _relations_frame(seed: int = 278, n: int = 2000) -> pl.DataFrame:
    """Three features of unlike scales and means, a slope that changes half
    way, a second target null where ``x0`` is high, uneven weights with a
    zero on the first row and every 37th, and a clock that is the row
    number."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = rng.standard_normal((n, 3)) * [1.0, 2.0, 0.5] + [0.5, -1.0, 3.0]
    beta = np.array([2.0, -1.0, 0.5])
    y = 1.0 + x @ beta + np.where(t < n / 2, 0.0, 1.5) * x[:, 0] + rng.standard_normal(n)
    z = 0.5 - x @ beta[::-1] + rng.standard_normal(n)
    z[x[:, 0] > 1.2] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[::37] = 0.0
    return pl.DataFrame(
        {"t": t, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y, "w": w}
        | {"z": pl.Series(z).fill_nan(None)}
    )


_REL = dict(
    targets=["y", "z"],
    features=["x0", "x1", "x2"],
    clock="t",
    gap_cap=1e9,
    weight="w",
    half_life=200.0,
    min_weight=10.0,
    coef_every=0,
    max_rows_between_solves=1,
)
#: A descent run to the double's rounding, so two routes to one optimum can
#: be compared at a bound far under the default rule's.
_TIGHT = dict(tol=1e-14, max_iter=100_000)


def _rel_run(build, df, **kw) -> tuple[pl.DataFrame, po.ModelBank]:
    bank = po.ModelBank([build("m", **{**_REL, **kw})], threads=1)
    return bank.fit_predict(df).unnest("m"), bank


def _diff(a: pl.Series, b: pl.Series) -> float:
    pa, pb = a.to_numpy().astype(float), b.to_numpy().astype(float)
    np.testing.assert_array_equal(np.isnan(pa), np.isnan(pb))
    assert np.isfinite(pa).sum() > 1500
    return float(np.nanmax(np.abs(pa - pb)))


@pytest.mark.parametrize("penalty", [0.5, 0.05])
def test_an_elastic_net_with_no_l1_is_the_standardized_ridge(penalty):
    """``l1_ratio=0`` leaves the penalty ``l |b|^2 / 2`` on the standardized
    slopes, which is ``ewridge(standardize=True, ridge=l)``: the descent
    reaches the ridge's closed form. Measured 2026-10-26 at ``tol=1e-14``:
    5.3e-15 and 2.0e-14 apart in a prediction for the two penalties, no
    descent short of its rule; held to 1e-12."""
    df = _relations_frame()
    net, bank = _rel_run(po.spec.lasso, df, penalties=[penalty], l1_ratio=0.0, **_TIGHT)
    ridge, _ = _rel_run(po.spec.ewridge, df, ridge=penalty, standardize=True)
    assert int(bank.solve_failures()["solve_failures"].sum()) == 0
    for t in ("y", "z"):
        d = _diff(net[f"pred_{t}"], ridge[f"pred_{t}"])
        print(f"l1_ratio 0 at {penalty}, {t}: {d:.3g}")
        assert d < 1e-12, t


@pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
def test_a_path_point_is_the_lone_penalty(l1_ratio):
    """Each point of the path is its own fit from the same sums. The first
    point's descent starts where a lone spec's does, so it is that spec, to
    the bit; a later one warm-starts from the point before it, so it
    reaches the same optimum by another route: 3.4e-14 (lasso) and 3.6e-14
    (elastic net) apart at ``tol=1e-14``, measured 2026-10-26; held to
    1e-12."""
    df = _relations_frame()
    kw = dict(l1_ratio=l1_ratio, **_TIGHT)
    path, _ = _rel_run(po.spec.lasso, df, penalties=[0.5, 0.05], **kw)
    first, _ = _rel_run(po.spec.lasso, df, penalties=[0.5], **kw)
    later, _ = _rel_run(po.spec.lasso, df, penalties=[0.05], **kw)
    for t in ("y", "z"):
        assert path[f"pred_{t}__p0.5"].equals(first[f"pred_{t}"]), t
        d = _diff(path[f"pred_{t}__p0.05"], later[f"pred_{t}"])
        print(f"l1_ratio {l1_ratio}, {t}: the later point {d:.3g} from its lone spec")
        assert d < 1e-12, t
    # The two points are two fits.
    assert _diff(path["pred_y__p0.5"], path["pred_y__p0.05"]) > 1e-3


def test_one_sweep_is_a_failure_for_each_target_point_and_solve():
    """``max_iter=1`` with a ``tol`` no move can be under: every descent that
    moves a coefficient runs out of sweeps, and ``solve_failures`` counts
    one per target and path point, in each group. On this stream every
    coefficient clears at every point, so on each row whose solve moved the
    fit (``coef`` holds every target's every point) all of them moved: the
    count is targets x points x those rows, per group. A group's first row
    runs no descent (one row gives no slope), so its fit is not counted."""
    rng = np.random.default_rng(0)
    n = 400
    a, b, c = rng.normal(size=(3, n))
    x = np.column_stack([2.0 + a, 2.0 + a + 0.3 * b, -1.0 + c])
    df = pl.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "g": np.arange(n) % 2}
        | {
            "y": 0.7 + x @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=n),
            "z": -0.3 + x @ np.array([0.5, 1.0, -1.0]) + 0.1 * rng.normal(size=n),
        }
    )
    points = [0.01, 0.001, 0.0]
    spec = po.spec.lasso(
        "m",
        targets=["y", "z"],
        features=["x0", "x1", "x2"],
        penalties=points,
        half_life=float("inf"),
        max_iter=1,
        tol=1e-300,
        group="g",
        coef_every=0,
    )
    bank = po.ModelBank([spec], threads=1)
    out = bank.fit_predict(df).unnest("m")
    got = dict(bank.solve_failures().select("group", "solve_failures").iter_rows())
    for g in (0, 1):
        coef = out.filter(pl.col("g") == g)["coef"].to_list()
        moved = sum(
            1
            for i, cf in enumerate(coef[1:], start=1)
            if cf is not None and (coef[i - 1] is None or cf != coef[i - 1])
        )
        print(f"group {g}: {moved} solves moved the fit, {got[str(g)]} counted")
        assert moved > n // 2 - 5
        assert got[str(g)] == 2 * len(points) * moved, g


def test_select_half_life_given_as_the_half_life_is_the_default():
    """The selection's memory defaults to the model's half-life: naming it
    is the same model, to the bit; another memory chooses otherwise."""
    df = _relations_frame()
    path = [0.5, 0.05, 0.0]
    default, _ = _rel_run(po.spec.lasso, df, penalties=path)
    named, _ = _rel_run(po.spec.lasso, df, penalties=path, select_half_life=200.0)
    assert default.equals(named, null_equal=True)
    other, _ = _rel_run(po.spec.lasso, df, penalties=path, select_half_life=20.0)
    assert not other["penalty_selected_y"].equals(default["penalty_selected_y"])
