"""``po.stream`` on the paths no other test took (task 252): ``with_windows``
under a filter Polars pushes into its source and a ``head`` longer than the
stream, a spent single-use source under ``with_windows``, and the refusals
of an expression that is not one, a NaN clock quantity, a chunk size below
1 and a delay of no length, each by name."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.io.plugins import register_io_source

import polars_online as po

TIER = "essential"


def _frame(n: int = 40) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x": [float((i * 7) % 11) - 5.0 for i in range(n)],
        }
    )


def test_a_pushed_filter_is_applied_by_the_source():
    """Polars pushes a filter on the output into a Python source and does
    not apply it again, so the source applies it: the filtered plan gives
    the rows of the whole output that pass it."""
    lf = _frame().lazy()
    windows = po.stream.with_windows(lf, m=po.ewm_mean("x", half_life=4.0))
    whole = windows.collect()
    got = windows.filter(pl.col("x") > 0).collect()
    assert got.equals(whole.filter(pl.col("x") > 0))
    assert 0 < got.height < whole.height


def test_a_head_longer_than_the_stream_gives_the_whole_stream():
    """A slice Polars pushes into the source counts output rows; one longer
    than the stream ends with the rows the input's end resolves."""
    lf = _frame().lazy()
    ahead = po.ewm_mean("x", half_life=4.0, window_size=5.0, reverse=True)
    windows = po.stream.with_windows(lf, clock="t", gap_cap=10.0, m=ahead)
    whole = windows.collect()
    assert windows.head(1_000).collect().equals(whole)
    assert windows.head(10).collect().equals(whole.head(10))


def _single_use(frame: pl.DataFrame) -> pl.LazyFrame:
    """A plan that yields its rows once and nothing after, as a consumed
    Arrow C stream does."""
    spent: list[bool] = []

    def io_source(with_columns, predicate, n_rows, batch_size):  # noqa: ANN001, ANN202
        if spent:
            return
        spent.append(True)
        out = frame
        if n_rows is not None:
            out = out.head(n_rows)
        if predicate is not None:
            out = out.filter(predicate)
        if with_columns is not None:
            out = out.select(with_columns)
        yield out

    return register_io_source(io_source=io_source, schema=frame.schema)


def test_with_windows_over_a_spent_single_use_source_warns():
    windows = po.stream.with_windows(_single_use(_frame()), m=po.ewm_mean("x", half_life=4.0))
    assert windows.collect().height == 40
    with pytest.warns(
        po.ConsumedSourceWarning,
        match="with_windows: the plan yielded no rows, and its source is a Python scan",
    ):
        assert windows.collect().height == 0


def test_with_windows_refuses_what_is_not_an_expression_or_not_a_number():
    lf = _frame().lazy()
    with pytest.raises(
        TypeError,
        match="with_windows: an expression is a pl.Expr over the operators, got int",
    ):
        po.stream.with_windows(lf, 3)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match="with_windows: m is a pl.Expr over the operators, got str 'x'"
    ):
        po.stream.with_windows(lf, m="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="with_windows: gap_cap must not be NaN"):
        po.stream.with_windows(
            lf, clock="t", gap_cap=float("nan"), m=po.ewm_mean("x", half_life=4.0)
        )


def test_a_chunk_size_below_one_is_refused():
    lf = _frame().lazy()
    with pytest.raises(ValueError, match="chunk_size must be at least 1, got 0"):
        po.stream.with_windows(lf, chunk_size=0, m=po.ewm_mean("x", half_life=4.0))
    ticks = pl.DataFrame({"s": ["a", "b"], "t": [1.0, 2.0], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="chunk_size must be at least 1, got 0"):
        po.stream.refresh_time(
            ticks.lazy(), series="s", names=["a", "b"], clock="t", value="v", chunk_size=0
        )


def test_an_embargo_of_no_length_is_refused_on_a_temporal_clock():
    df = pl.DataFrame(
        {
            "t": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(5)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    for delay in (timedelta(0), pl.duration(minutes=-1)):
        with pytest.raises(ValueError, match="embargo: delay must be > 0, got"):
            po.stream.embargo(df, clock="t", delay=delay)
