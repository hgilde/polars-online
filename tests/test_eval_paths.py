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
