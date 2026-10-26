"""``po.eval`` on the paths no other test took (task 252): each refusal of an
argument, by name; the merged spelling of a spec beside the new one; and a
frame that already holds a column of the name a test computes in, which the
test steps around rather than overwrites."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def _two_specs(n: int = 300, seed: int = 7) -> tuple[pl.DataFrame, dict, dict]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = x[:, 0] + 0.3 * x[:, 1] + rng.normal(size=n)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
    common = dict(targets=["y"], half_life=float("inf"), min_weight=5.0)
    small = po.spec.ewridge("small", features=["x0"], **common)
    big = po.spec.ewridge("big", features=["x0", "x1"], **common)
    return po.ModelBank([small, big]).fit_predict(df), small, big


@pytest.fixture(scope="module")
def fitted():
    return _two_specs()


def test_a_frame_that_is_not_a_polars_frame_is_refused(fitted):
    out, _, _ = fitted
    with pytest.raises(TypeError, match="po.eval.metrics takes a polars DataFrame, got dict"):
        po.eval.metrics(out.to_dict(), "big")  # type: ignore[arg-type]


def test_a_spec_that_is_neither_a_name_nor_a_spec_dict_is_refused(fitted):
    out, _, _ = fitted
    with pytest.raises(
        TypeError, match="po.eval.metrics: spec is a spec's name or its dict, got int"
    ):
        po.eval.metrics(out, 3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec is a spec's name or its dict, got dict"):
        po.eval.metrics(out, {"model": {}})  # type: ignore[arg-type]


def test_the_merged_spellings_of_a_spec_beside_the_new_one(fitted):
    """``spec_name`` was merged into ``spec``: both given is refused, and a
    name beside a spec dict of another name is refused; of the same name it
    is read, with a deprecation warning."""
    out, small, big = fitted
    with pytest.raises(
        TypeError, match="po.eval.metrics: spec_name was merged into spec, and both are given"
    ):
        po.eval.metrics(out, "big", spec_name="big")
    with pytest.raises(
        ValueError,
        match="po.eval.metrics: spec 'big' and the spec dict named 'small' disagree; give the "
        "dict alone as spec",
    ):
        po.eval.metrics(out, "big", spec=small)
    with pytest.raises(
        TypeError,
        match="po.eval.compare_specs: spec_names was merged into specs, and both are given",
    ):
        po.eval.compare_specs(out, ["big"], spec_names=["big"])
    with pytest.warns(po.PolarsOnlineDeprecationWarning, match="po.eval.compare_specs"):
        old = po.eval.compare_specs(out, ["small", "big"], specs=[big])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = po.eval.compare_specs(out, ["small", big])
    assert old.equals(new)


def test_a_spec_dict_whose_target_column_is_missing_is_refused_by_name(fitted):
    out, _, big = fitted
    with pytest.raises(
        ValueError,
        match=r"cannot find column 'y', which slot 'pred_y' reads for target 'y'; the frame has",
    ):
        po.eval.metrics(out.drop("y"), big)


def test_compare_specs_refuses_a_logistic_fit_beside_a_regression(fitted):
    out, _, big = fitted
    labelled = out.with_columns(label=(pl.col("y") > 0).cast(pl.Float64))
    logit = po.spec.ftrl("logit", targets=["label"], features=["x0", "x1"], half_life=100.0)
    both = po.ModelBank([logit]).fit_predict(labelled.select("x0", "x1", "label"))
    frame = labelled.with_columns(both["logit"])
    with pytest.raises(
        ValueError,
        match=r"po.eval.compare_specs: specs \['big', 'logit'\] mix a logistic fit with others, "
        "whose metrics differ in kind; compare them in two calls, or set binary",
    ):
        po.eval.compare_specs(frame, [big, logit])


def test_two_specs_sharing_no_residual_are_refused_by_name(fitted):
    out, _, _ = fitted
    with pytest.raises(
        ValueError,
        match=r"diebold_mariano: 'big' and 'small' share no residual field \(with a_suffix "
        r"'_z', b_suffix ''\)",
    ):
        po.eval.diebold_mariano(out, a="big", b="small", a_suffix="_z")


def test_clark_west_refuses_negative_lags(fitted):
    out, _, _ = fitted
    with pytest.raises(ValueError, match="clark_west: lags must be >= 0, got -1"):
        po.eval.clark_west(out, big="big", small="small", lags=-1)


def test_a_column_of_the_working_name_is_stepped_around(fitted):
    """The tests compute in columns of their own (``__po_d``, ``__po_row``,
    ``__po_seqtest_*``); a frame that already has one gets the same result,
    and keeps its column untouched."""
    out, _, _ = fitted
    taken = out.with_columns(
        pl.lit(-1.0).alias("__po_d"),
        pl.lit(-2.0).alias("__po_row"),
        pl.lit(-3.0).alias("__po_seqtest_weight_sum"),
    )
    plain = po.eval.diebold_mariano(out, a="big", b="small", min_samples=1)
    assert po.eval.diebold_mariano(taken, a="big", b="small", min_samples=1).equals(plain)
    ew = po.eval.diebold_mariano(out, a="big", b="small", half_life=50.0)
    ew_taken = po.eval.diebold_mariano(taken, a="big", b="small", half_life=50.0)
    assert ew_taken["diebold_mariano"].equals(ew["diebold_mariano"])
    assert ew_taken["__po_d"].unique().to_list() == [-1.0]
    assert ew_taken["__po_row"].unique().to_list() == [-2.0]
    seq = po.eval.seqtest(out, a="big", b="small")
    seq_taken = po.eval.seqtest(taken, a="big", b="small")
    assert seq_taken["seqtest"].equals(seq["seqtest"])
    assert seq_taken["__po_seqtest_weight_sum"].unique().to_list() == [-3.0]


def test_a_window_of_no_length_is_refused_on_a_temporal_clock(fitted):
    out, _, _ = fitted
    stamped = out.with_columns(
        t=pl.datetime_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 1) + timedelta(minutes=out.height - 1),
            "1m",
            eager=True,
        )
    )
    for every in (timedelta(0), pl.duration(minutes=-5)):
        with pytest.raises(ValueError, match="window_metrics: every must be > 0, got"):
            po.eval.window_metrics(stamped, "big", clock="t", every=every)


# ----------------------------- task 278 (b12): the comparisons' own keywords


def _grid_specs(n: int = 600, seed: int = 7) -> pl.DataFrame:
    """Two specs, each a grid of two instances (``@h50`` and ``@hinf``), on
    rows in two keys: ``thin`` (the first 60) and ``wide``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = x[:, 0] + 0.3 * x[:, 1] + rng.normal(size=n)
    df = pl.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "y": y, "g": np.where(np.arange(n) < 60, "thin", "wide")}
    )
    common = dict(targets=["y"], half_life=[50.0, float("inf")], min_weight=5.0)
    small = po.spec.ewridge("small", features=["x0"], **common)
    big = po.spec.ewridge("big", features=["x0", "x1"], **common)
    return po.ModelBank([small, big]).fit_predict(df)


@pytest.fixture(scope="module")
def grid():
    return _grid_specs()


def _plain(frame: pl.DataFrame, side: str, suffix: str) -> pl.DataFrame:
    """``frame`` with ``side`` replaced by a struct holding one field,
    ``resid_y``, its ``resid_y<suffix>``: what a spec of that instance
    alone would have written."""
    resid = frame[side].struct.field(f"resid_y{suffix}").alias("resid_y")
    return frame.with_columns(pl.struct(resid).alias(side))


PAIRS = [("@h50", "@hinf"), ("@hinf", "@h50"), ("@h50", "@h50")]


@pytest.mark.parametrize(("first", "second"), PAIRS)
def test_each_suffix_picks_its_instance(grid, first, second):
    """``a_suffix``/``b_suffix`` and ``big_suffix``/``small_suffix`` pick a
    grid instance: the statistic is the one on a frame whose plain residual
    is that instance, to the bit, per key and per row. No test passed
    ``b_suffix``, ``big_suffix`` or ``small_suffix`` before."""
    frame = _plain(_plain(grid, "big", first), "small", second)
    for kw in (dict(group="g", min_samples=1, lags=2), dict(half_life=40.0)):
        dm = po.eval.diebold_mariano(
            grid, a="big", b="small", a_suffix=first, b_suffix=second, **kw
        )
        want = po.eval.diebold_mariano(frame, a="big", b="small", **kw)
        cw = po.eval.clark_west(
            grid, big="big", small="small", big_suffix=first, small_suffix=second, **kw
        )
        cw_want = po.eval.clark_west(frame, big="big", small="small", **kw)
        if "half_life" in kw:
            assert dm["diebold_mariano"].equals(want["diebold_mariano"])
            assert cw["clark_west"].equals(cw_want["clark_west"])
        else:
            assert dm.equals(want) and cw.equals(cw_want)


def test_the_two_instances_differ(grid):
    """The suffix takes effect: each side's two instances give two
    statistics."""
    for side in ("a", "b"):
        stats = {
            s: po.eval.diebold_mariano(
                grid,
                a="big",
                b="small",
                **{"a_suffix": "@h50", "b_suffix": "@h50", f"{side}_suffix": s},
            )["dm"][0]
            for s in ("@h50", "@hinf")
        }
        assert stats["@h50"] != stats["@hinf"], side
    for side in ("big", "small"):
        stats = {
            s: po.eval.clark_west(
                grid,
                big="big",
                small="small",
                **{"big_suffix": "@h50", "small_suffix": "@h50", f"{side}_suffix": s},
            )["cw"][0]
            for s in ("@h50", "@hinf")
        }
        assert stats["@h50"] != stats["@hinf"], side


def test_name_names_the_per_row_struct_and_the_batch_form_ignores_it(grid):
    sides = dict(a="big", b="small", a_suffix="@h50", b_suffix="@h50")
    nested = dict(big="big", small="small", big_suffix="@h50", small_suffix="@h50")
    dm = po.eval.diebold_mariano(grid, **sides, half_life=40.0, name="zz")
    cw = po.eval.clark_west(grid, **nested, half_life=40.0, name="zz")
    seq = po.eval.seqtest(grid, **sides, name="zz")
    for out, default in ((dm, "diebold_mariano"), (cw, "clark_west"), (seq, "seqtest")):
        assert out.columns == [*grid.columns, "zz"] and default not in out.columns
    want = po.eval.diebold_mariano(grid, **sides, half_life=40.0)
    assert dm["zz"].equals(want["diebold_mariano"].alias("zz"))
    assert seq["zz"].equals(po.eval.seqtest(grid, **sides)["seqtest"].alias("zz"))
    # Without half_life the result is one row per key, and name is not read.
    batch = po.eval.diebold_mariano(grid, **sides, name="zz")
    assert batch.equals(po.eval.diebold_mariano(grid, **sides))
    assert po.eval.clark_west(grid, **nested, name="zz").equals(po.eval.clark_west(grid, **nested))


def test_a_name_equal_to_an_input_column_replaces_that_column(grid):
    """Measured 2026-10-25, and pinned as it stands: the result is ``df``
    with the struct added by Polars' ``with_columns``, which replaces a
    column of the same name in its place -- the target ``y``, or a spec's
    own output struct -- and says nothing. (The working columns these tests
    compute in are stepped around instead; see above.)"""
    sides = dict(a="big", b="small", a_suffix="@h50", b_suffix="@h50")
    for name in ("y", "big"):
        dm = po.eval.diebold_mariano(grid, **sides, half_life=40.0, name=name)
        seq = po.eval.seqtest(grid, **sides, name=name)
        for out, first in ((dm, "dm_y"), (seq, "log_e_a_y")):
            assert out.columns == grid.columns
            assert out.schema[name].fields[0].name == first  # type: ignore[union-attr]


def test_clark_west_min_samples_nulls_a_thin_key_and_moves_no_other_value(grid):
    """The thin key has 54 scored rows, the wide one 540: a ``min_samples``
    between keeps the thin key's row with ``n`` and nulls its scores, and
    leaves the wide key's row as it was. No test bound CW's before."""
    nested = dict(big="big", small="small", big_suffix="@h50", small_suffix="@h50", group="g")
    loose = po.eval.clark_west(grid, **nested, min_samples=1)
    strict = po.eval.clark_west(grid, **nested, min_samples=100)
    assert loose["n"].to_list() == strict["n"].to_list() == [54, 540]
    thin = strict.row(0, named=True)
    assert thin["mean"] is None and thin["cw"] is None and thin["p_value"] is None
    assert loose.row(0, named=True)["cw"] is not None
    assert strict[1:].equals(loose[1:])
    sides = dict(a="big", b="small", a_suffix="@h50", b_suffix="@h50", group="g")
    dm_strict = po.eval.diebold_mariano(grid, **sides, min_samples=100)
    assert dm_strict[1:].equals(po.eval.diebold_mariano(grid, **sides, min_samples=1)[1:])
    assert dm_strict.row(0, named=True)["dm"] is None


@pytest.mark.parametrize("lags", [0, 3])
def test_swapping_the_sides(grid, lags):
    """Diebold-Mariano's differential ``resid_b^2 - resid_a^2`` changes sign
    with the sides, exactly (a float subtraction is antisymmetric), so its
    mean and ``dm`` are negated to the bit and the two-sided ``p_value``
    kept, per key and per row. Clark-West's is not antisymmetric: the two
    orders' ``f`` add to ``2 (resid_a - resid_b)^2``, which holds of the
    means to rounding."""
    kw = dict(lags=lags, group="g", min_samples=1)
    ab = po.eval.diebold_mariano(grid, a="big", b="small", a_suffix="@h50", b_suffix="@hinf", **kw)
    ba = po.eval.diebold_mariano(grid, a="small", b="big", a_suffix="@hinf", b_suffix="@h50", **kw)
    assert ab["n"].equals(ba["n"]) and ab["p_value"].equals(ba["p_value"])
    assert ab["dm"].equals(-ba["dm"]) and ab["mean"].equals(-ba["mean"])
    ew = dict(lags=lags, half_life=40.0)
    one = po.eval.diebold_mariano(grid, a="big", b="small", a_suffix="@h50", b_suffix="@hinf", **ew)
    two = po.eval.diebold_mariano(grid, a="small", b="big", a_suffix="@hinf", b_suffix="@h50", **ew)
    stat = one["diebold_mariano"].struct.field("dm_y")
    assert stat.drop_nulls().len() > 500
    assert stat.equals(-two["diebold_mariano"].struct.field("dm_y"))
    big = dict(big="big", small="small", big_suffix="@h50", small_suffix="@hinf")
    cw_ab = po.eval.clark_west(grid, **big, **kw)
    small = dict(big="small", small="big", big_suffix="@hinf", small_suffix="@h50")
    cw_ba = po.eval.clark_west(grid, **small, **kw)
    ra = grid["big"].struct.field("resid_y@h50")
    rb = grid["small"].struct.field("resid_y@hinf")
    both = pl.DataFrame({"g": grid["g"], "d": (ra - rb) ** 2}).drop_nulls()
    want = both.group_by("g", maintain_order=True).agg(pl.col("d").mean() * 2.0)["d"]
    got = cw_ab["mean"] + cw_ba["mean"]
    assert np.abs((got - want).to_numpy()).max() <= 1e-14 * np.abs(want.to_numpy()).max()


def test_swapping_seqtest_s_sides_swaps_its_fields(grid):
    """The sign tested, ``|resid_b| - |resid_a|``, flips with the sides, so
    ``a``'s gambler becomes ``b``'s: each field is the other's, to the bit."""
    one = po.eval.seqtest(grid, a="big", b="small", a_suffix="@h50", b_suffix="@hinf", group="g")
    two = po.eval.seqtest(grid, a="small", b="big", a_suffix="@hinf", b_suffix="@h50", group="g")
    u, v = one["seqtest"].struct.unnest(), two["seqtest"].struct.unnest()
    assert u["wins_a_y"][-1] > 0 and u["wins_b_y"][-1] > 0
    for left, right in (("log_e_a_y", "log_e_b_y"), ("wins_a_y", "wins_b_y")):
        assert u[left].equals(v[right]) and u[right].equals(v[left])
    assert u["weight_sum"].equals(v["weight_sum"])


def test_seqtest_s_b_suffix_is_the_banks():
    """The eval twin's ``b_suffix`` reads the grid instance the bank's
    ``seqtest`` spec reads with the same ``b_suffix``, to the bit. The
    twin's ``b_suffix`` had only a refusal test."""
    rng = np.random.default_rng(22)
    n = 3000
    x = rng.normal(size=(n, 2))
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": x @ [1.0, -0.5] + rng.normal(size=n)})
    common = dict(targets=["y"], features=["x0", "x1"], min_weight=5.0)
    grid = po.spec.ewridge("grid", half_life=[20.0, 400.0], **common)
    kalman = po.spec.kalman("kalman", half_life=100.0, coef_half_life=50.0, **common)
    picked = po.spec.seqtest("c", targets=["y"], a="kalman", b="grid", b_suffix="@h20")
    out = po.ModelBank([grid, kalman, picked]).fit_predict(df)
    twin = po.eval.seqtest(out, a="kalman", b="grid", b_suffix="@h20")["seqtest"].struct.unnest()
    bank = out["c"].struct.unnest()
    # The bank adds its readiness fields; the e-processes are the twin's.
    assert bank.select(twin.columns).equals(twin, null_equal=True)
    assert twin["wins_a_y"][-1] > 1000 and twin["wins_b_y"][-1] > 1000
    other = po.eval.seqtest(out, a="kalman", b="grid", b_suffix="@h400")["seqtest"]
    assert not other.struct.unnest().equals(twin)


@pytest.mark.parametrize("lags", [0, 2])
def test_a_half_life_far_past_the_rows_is_the_batch_statistic(grid, lags):
    """The per-row form weighs each earlier row by ``0.5 ** (age /
    half_life)``; as ``half_life`` grows every weight tends to 1, and the
    last row's statistic to the batch one over the same rows (``inf`` is
    refused: Polars' ``ewm_mean`` takes no ``alpha`` of 0). Measured
    2026-10-25 on 594 rows, at lags 0 and 2: off by 1.5e-5 and 2.1e-5 of
    the statistic at 1e6, 1.5e-7 and 2.1e-7 at 1e8, 1.5e-11 and 2.1e-11 at
    1e12 -- first order in ``1 / half_life``, as the weights' distance from
    1 is; held at 1e-9 at 1e12."""
    sides = dict(a="big", b="small", a_suffix="@h50", b_suffix="@h50", lags=lags)
    batch = po.eval.diebold_mariano(grid, **sides, min_samples=1)["dm"][0]
    gaps = {}
    for half_life in (1e6, 1e8, 1e12):
        ew = po.eval.diebold_mariano(grid, **sides, half_life=half_life)
        gaps[half_life] = abs(ew["diebold_mariano"].struct.field("dm_y")[-1] - batch) / abs(batch)
    print(f"lags {lags}: batch dm {batch:.12f}, relative distance by half_life {gaps}")
    assert gaps[1e12] <= 1e-9
    assert gaps[1e6] > 50 * gaps[1e8] > 50 * 50 * gaps[1e12], gaps
    nested = dict(big="big", small="small", big_suffix="@h50", small_suffix="@h50", lags=lags)
    cw = po.eval.clark_west(grid, **nested, min_samples=1)["cw"][0]
    ew_cw = po.eval.clark_west(grid, **nested, half_life=1e12)["clark_west"]
    assert abs(ew_cw.struct.field("cw_y")[-1] - cw) <= 1e-9 * abs(cw)
