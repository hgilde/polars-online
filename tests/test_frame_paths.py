"""The frame namespaces on the paths no other test took (task 252): the
``closed_groups`` sidecar of ``df.online.fit_predict``, a sidecar whose
directory is not there, a join that keeps the right side's order or both
sides', and a Polars whose ``register_io_source`` takes no ``explain_name``
(py-polars 1.x, still within the supported range)."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import _frame

TIER = "essential"


def _groups() -> pl.DataFrame:
    """Three groups, one run of 30 rows each, in key order."""
    rng = np.random.default_rng(5)
    keys = [k for k in "abc" for _ in range(30)]
    return pl.DataFrame(
        {
            "g": keys,
            "x0": rng.normal(size=90),
            "x1": rng.normal(size=90),
        }
    )


SPEC = po.spec.ew_cov("c", features=["x0", "x1"], half_life=20.0, group="g", group_close="monotone")


def test_the_frame_namespace_writes_the_closed_groups_sidecar(tmp_path):
    """The sidecar holds what a bank fed the same chunks drains, in the
    format its extension names; the frame's output is the bank's."""
    df = _groups()
    side = tmp_path / "closed.parquet"
    out = df.online.fit_predict([SPEC], closed_groups=side, chunk_size=25)
    driver = po.ModelBank([SPEC])
    frames, outs = [], []
    for i in range(0, df.height, 25):
        outs.append(driver.fit_predict(df[i : i + 25]))
        frames.append(driver.closed_groups())
    want = pl.concat(frames)
    assert want.height == 2, "a and b closed, c is still open"
    assert pl.read_parquet(side).equals(want)
    assert out.equals(pl.concat(outs))


def test_a_closed_groups_sidecar_in_a_missing_directory_is_refused_before_the_fit(tmp_path):
    side = tmp_path / "no" / "such" / "closed.parquet"
    with pytest.raises(FileNotFoundError, match=r"closed_groups: '.*such' is not a directory"):
        _groups().online.fit_predict([SPEC], closed_groups=side)
    assert not side.parent.exists()


SPEC_ORDER = po.spec.ewridge(
    "m",
    targets=["y"],
    features=["x0"],
    clock="t",
    half_life=10.0,
    gap_cap=5.0,
    min_weight=1.0,
    restart_after_step_back=0.0,
)


def _left() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "k": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "t": [float(i) for i in range(9)],
            "x0": [0.1 * i for i in range(9)],
            "y": [0.2 * i for i in range(9)],
        }
    )


def _shuffled() -> pl.LazyFrame:
    """A side whose order is unspecified: a ``group_by`` without
    ``maintain_order``."""
    return pl.LazyFrame({"k": [1, 2, 3, 1], "z": [1, 2, 3, 4]}).group_by("k").agg(pl.col("z").sum())


def _flagged(lf: pl.LazyFrame) -> list[str]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        po.ModelBank([SPEC_ORDER]).fit(lf)
    return [str(w.message) for w in rec if issubclass(w.category, po.OrderNotGuaranteedWarning)]


def test_a_join_that_keeps_the_right_order_reads_the_right_side_alone():
    """A join with ``maintain_order="right"`` delivers the right side's
    order, so a hazard there is reported and one on the left is not."""
    ordered = pl.LazyFrame({"k": [1, 2, 3], "z": [10, 20, 30]})
    right_hazard = ordered.join(_left(), on="k", maintain_order="right").sort("t")
    assert _flagged(right_hazard) == [], "the sort settles everything beneath it"
    right_hazard = _left().join(_shuffled(), on="k", maintain_order="right")
    found = _flagged(right_hazard)
    assert len(found) == 1 and "group_by" in found[0]
    left_hazard = _shuffled().join(_left(), on="k", maintain_order="right")
    assert _flagged(left_hazard) == []


@pytest.mark.parametrize("mode", ["left_right", "right_left"])
def test_a_join_that_keeps_both_orders_reads_both_sides(mode):
    for lf in (
        _left().join(_shuffled(), on="k", maintain_order=mode),
        _shuffled().join(_left(), on="k", maintain_order=mode),
    ):
        found = _flagged(lf)
        assert len(found) == 1 and "group_by" in found[0], found


def test_a_polars_without_explain_name_gets_no_explain_keywords(monkeypatch):
    """``explain_name`` and ``explain_detail`` are passed to
    ``register_io_source`` only where it takes them (py-polars 2.0 on); on
    one without them the plan is built exactly as before."""

    def register_io_source(io_source, *, schema, validate_schema=False):  # the 1.x shape
        raise AssertionError("not called here")

    monkeypatch.setattr(_frame, "register_io_source", register_io_source)
    assert _frame._explain_kwargs([SPEC]) == {}
    assert _frame._explain_named("anything") == {}
    monkeypatch.undo()
    if "explain_name" in inspect.signature(_frame.register_io_source).parameters:
        want = {"explain_name": "polars-online", "explain_detail": "x"}
        assert _frame._explain_named("x") == want
