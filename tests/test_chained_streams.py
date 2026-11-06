"""This package's sources hand Polars each chunk's output in slices of a
bounded size (task 348). Polars keeps a number of frames from a source in
flight before a node that runs on every thread, not a number of bytes, so a
source in front of a slower consumer -- one ``fit_predict`` feeding another
through a ``with_columns`` -- ran that many whole chunks ahead of it. The
memory this bounds is held by ``tests/test_budgets_memory.py``
(``test_two_fit_predicts_in_series_hold_memory_flat_in_the_streams_length``);
here, that the slicing is exact: every slice within the bound, every row
once and in order, and a chained query's output the same as the two banks
driven by a loop, with the bound small enough that every source slices."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import _frame, stream

TIER = "essential"


def _frame_of(rows: int) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.standard_normal((rows, 3))
    return pl.DataFrame(
        {
            "t": np.arange(rows, dtype=np.float64),
            "g": np.where(np.arange(rows) % 3 == 0, "a", "b"),
            "x0": x[:, 0],
            "x1": x[:, 1],
            "x2": x[:, 2],
            "y": x @ np.array([0.5, -1.0, 0.25]) + 0.3 * rng.standard_normal(rows),
        }
    )


@pytest.mark.parametrize("bound", [1, 64, 1_000, 10_000, 1 << 20])
def test_a_frame_comes_out_in_near_equal_slices_within_the_bound(monkeypatch, bound):
    df = _frame_of(1_001)
    monkeypatch.setattr(_frame, "_MORSEL_BYTES", bound)
    parts = list(_frame._in_morsels(df))
    assert pl.concat(parts).equals(df)
    size = df.estimated_size()
    if size <= bound:
        assert len(parts) == 1
        assert parts[0] is df
        return
    # As many rows a slice as fit at the frame's mean bytes a row, one at
    # the least, the last slice taking the rest.
    rows = max(1, df.height * bound // size)
    heights = [p.height for p in parts]
    assert heights[:-1] == [rows] * (len(parts) - 1)
    assert 1 <= heights[-1] <= rows
    assert all(p.estimated_size() <= bound for p in parts if p.height > 1)


def test_an_empty_frame_is_yielded_once_as_it_is():
    empty = _frame_of(0)
    assert [p.height for p in _frame._in_morsels(empty)] == [0]


def test_a_frame_within_the_bound_is_not_sliced():
    df = _frame_of(100)
    assert df.estimated_size() < _frame._MORSEL_BYTES
    assert [p.height for p in _frame._in_morsels(df)] == [100]


STAGE1 = po.spec.ewridge("fit", targets=["y"], features=["x0", "x1"], half_life=50.0, group="g")
STAGE2 = po.spec.ewridge(
    "second",
    targets=["y"],
    features=["resid_y", "x2"],
    half_life=80.0,
    clock="t",
    gap_cap=10.0,
    group="g",
)


@pytest.mark.parametrize("chunk_size", [37, 400])
def test_two_fit_predicts_in_series_give_what_a_loop_over_the_two_banks_gives(
    monkeypatch, chunk_size
):
    """A bound of 2 KiB slices every chunk's output of both stages several
    ways; the chain's rows, every field, are the loop's."""
    monkeypatch.setattr(_frame, "_MORSEL_BYTES", 2_048)
    df = _frame_of(3_000)
    yielded: list[int] = []
    real = _frame._in_morsels

    def counted(out: pl.DataFrame):
        for part in real(out):
            yielded.append(part.height)
            yield part

    monkeypatch.setattr(_frame, "_in_morsels", counted)
    resid = pl.col("fit").struct.field("resid_y")
    chain = (
        df.lazy()
        .online.fit_predict([STAGE1], chunk_size=chunk_size)
        .with_columns(resid)
        .online.fit_predict([STAGE2], chunk_size=chunk_size)
        .collect()
    )
    # The bound took effect: more slices than either stage had chunks.
    chunks = -(-df.height // chunk_size)
    assert len(yielded) > 2 * chunks
    first, second = po.ModelBank([STAGE1]), po.ModelBank([STAGE2])
    loop = pl.concat(
        second.fit_predict(first.fit_predict(df[i : i + chunk_size]).with_columns(resid))
        for i in range(0, df.height, chunk_size)
    )
    assert chain.equals(loop)


def test_a_window_source_feeding_a_fit_through_unnest_gives_the_eager_result(monkeypatch):
    """`with_windows` slices its output too: the chain the two-stage worked
    example runs -- windows, a fit, `unnest`, a second fit -- is the same
    under a bound that slices every chunk as under none."""
    df = _frame_of(2_000)
    fit = po.spec.ewridge(
        "fit", targets=["y"], features=["x0", "dev"], half_life=50.0, clock="t", gap_cap=10.0
    )
    second = po.spec.ewridge("second", targets=["y"], features=["resid_y"], half_life=80.0)

    def run() -> pl.DataFrame:
        return (
            df.lazy()
            .online.with_windows(
                dev=pl.col("x1") - po.ewm_mean("x1", half_life=20.0, window_size=60.0),
                clock="t",
                gap_cap=10.0,
            )
            .online.fit_predict([fit], chunk_size=300)
            .online.unnest([fit])
            .online.fit_predict([second], chunk_size=300)
            .collect()
        )

    whole = run()
    monkeypatch.setattr(_frame, "_MORSEL_BYTES", 1_024)
    windows: list[int] = []
    real = stream._in_morsels

    def counted(out: pl.DataFrame):
        for part in real(out):
            windows.append(part.height)
            yield part

    monkeypatch.setattr(stream, "_in_morsels", counted)
    assert run().equals(whole)
    # The window source's own output was sliced, past its seven chunks.
    assert len(windows) > 2 * 7
    assert sum(windows) == df.height
