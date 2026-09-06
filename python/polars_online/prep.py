"""Frame preparation: streams whose labels arrive late, and series that tick
at their own times (docs/ENHANCEMENTS.md E47, E58).

:func:`embargo` turns a frame into the doubled stream
that a forward-looking target needs: every row appears twice, once as a
prediction at its own clock with zero weight, and once as a lesson at
``clock + delay``, the two merged back into clock order. It is the recipe a
spec's ``label_delay`` runs natively, written out in Polars -- useful for
seeing what the delay does, for a model that has no ``label_delay``, and as
the oracle the native path is tested against.

:func:`refresh_time` puts asynchronous series on a common grid by
Barndorff-Nielsen, Hansen, Lunde & Shephard's refresh-time rule: a grid point
wherever every series has ticked at least once since the last one. The scan
is a Rust operator, wrapped here as a lazy source.

Everything here is lazy and streaming -- ``merge_sorted`` on two sorted halves
of the same frame, a chunk-fed operator for the grid -- so a stream too long
to hold is still too long to hold and this does not change that.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _polars_online as _native

__all__ = ["embargo", "refresh_time"]

#: Column :func:`embargo` adds to say which copy of a row this is.
ROLE = "_online_role"


def embargo(
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    clock: str,
    delay: float,
    weight: str | None = None,
    role: str = ROLE,
) -> pl.LazyFrame:
    """The doubled stream for a target that is only known ``delay`` later.

    Every row comes back twice, in clock order:

    - a **predict** row at ``clock``, with its weight forced to 0, so the
      model scores it and learns nothing from it;
    - a **learn** row at ``clock + delay``, carrying the same features and
      target at full weight.

    A ``role`` column says which is which (``"predict"`` / ``"learn"``), so
    the output is filtered back down with
    ``out.filter(pl.col(role) == "predict")``.

    Why bother: a target that is a forward quantity over ``delay`` clock
    units is not known at the row it sits on. A stream that learns it there
    has seen ``delay`` of the future before predicting the rows in between,
    and every "out-of-sample" number after that is contaminated -- with an
    autocorrelated feature, even a pure noise column will show a correlation
    with its target. Zero-weight rows are legal and mean "advance the clock,
    learn nothing", so the doubled stream says exactly what is wanted:
    predict here, learn later.

    ``weight`` names an existing weight column; without it the function adds
    one (named ``role + "_weight"``) that is 1 on learn rows and 0 on predict
    rows -- pass that name to the spec's ``weight=``.

    The frame must already be in ``clock`` order, as a stream must be. The
    result is sorted by ``clock`` with **learn rows before predict rows** at
    the same clock value: a label whose ``delay`` has just run out is known
    at that instant, so a prediction made then may use it. A spec's
    ``label_delay`` releases in the same order, which is what lets the two
    be compared row for row.

    ``delay`` must be finite and positive; ``0`` would be the undoubled
    stream, and negative is a label from the past, which is not what this is
    for.

    A spec's ``label_delay=`` does the same thing in the stream with no
    doubling and no filtering, which is cheaper and does not need the frame
    rewritten. Reach for this when a delay has to be visible in the data --
    an oracle, a demonstration, or an engine that is not this one.
    """
    if not (delay > 0.0) or delay == float("inf"):
        msg = f"embargo: delay must be finite and > 0, got {delay!r}"
        raise ValueError(msg)
    lazy = lf.lazy()
    schema = lazy.collect_schema()
    if clock not in schema:
        msg = f"embargo: no clock column {clock!r} in the frame; it has {schema.names()}"
        raise ValueError(msg)
    if weight is not None and weight not in schema:
        msg = f"embargo: no weight column {weight!r} in the frame; it has {schema.names()}"
        raise ValueError(msg)
    if role in schema:
        msg = f"embargo: the frame already has a column named {role!r}; pass another `role=`"
        raise ValueError(msg)

    wcol = weight if weight is not None else f"{role}_weight"
    # `merge_sorted` needs both halves sorted on the key it merges by. Each
    # half is the input in its own order, so a single key sorts both: the
    # clock, with the learn copy first at a tie.
    predict = lazy.with_columns(
        pl.lit("predict").alias(role),
        (pl.col(weight) * 0.0 if weight is not None else pl.lit(0.0)).alias(wcol),
        pl.lit(1, pl.UInt8).alias("__embargo_order"),
    )
    learn = lazy.with_columns(
        pl.lit("learn").alias(role),
        (pl.col(weight) if weight is not None else pl.lit(1.0)).alias(wcol),
        (pl.col(clock) + delay).alias(clock),
        pl.lit(0, pl.UInt8).alias("__embargo_order"),
    )
    # One sort key, so the merge is by (clock, order): at a tie the lesson
    # lands before the prediction that may use it.
    key = "__embargo_key"
    both = [
        f.with_columns(
            pl.struct(pl.col(clock), pl.col("__embargo_order")).alias(key),
        )
        for f in (predict, learn)
    ]
    return (
        both[0]
        .merge_sorted(both[1], key=key)
        .drop(key, "__embargo_order")
        .select(*schema.names(), *([] if weight is not None else [wcol]), role)
    )


def refresh_time(
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    series: str,
    names: Sequence[str],
    time: str,
    value: str,
    by: str | None = None,
    pairs: bool = False,
    keep: Sequence[str] = (),
    chunk_rows: int | None = None,
) -> pl.LazyFrame:
    """Asynchronous series on a common grid, by refresh time (E58).

    Series observed at their own times cannot be correlated directly: the
    Epps effect attenuates a correlation computed over a fine grid, and
    filling forward invents observations. Barndorff-Nielsen, Hansen, Lunde &
    Shephard's rule places a grid point at the first instant by which
    **every** series has ticked at least once since the previous point, and
    takes each series' last value there:

    .. code-block:: text

        tau_0     = max_i (first tick of series i)
        tau_{j+1} = max_i (first tick of series i strictly after tau_j)

    Nothing is interpolated -- every value in the output was observed -- and
    the grid adapts to the slowest series rather than carrying a stale value
    across an interval.

    The input is **long**: one row per tick, with a ``series`` column naming
    it, a ``time`` and a ``value``. A wide frame is already synchronised;
    ``lf.unpivot(index=[time], variable_name="series", value_name="value")``
    is the line that makes one from the other.

    ``names`` is required and gives the series in output order. It is not
    discovered from the data because a lazy plan has to declare its schema
    before a row is read, and the output columns are named after the series.
    A row whose ``series`` is not in ``names`` is an error naming it: dropping
    it would hide a misspelling.

    Output, one row per grid point:

    ``time_refresh``
        The completing tick's time -- the max over series of their last
        update, their Definition 1.
    ``<s>_value``
        Each series' last value at that instant.
    ``n_ticks_<s>``
        Ticks of ``s`` since the previous point, the *first* of which is the
        one on the grid.
    ``retained_fraction``
        ``m / sum(n_ticks)``: how many of the interval's ticks the grid kept.
        Look at it before trusting a correlation computed on the result.

    plus the ``by`` column and any ``keep`` columns, at their value on the
    completing tick. ``pairs=True`` runs an independent two-series grid per
    unordered pair instead -- which keeps far more of the data when one
    series is slow -- and returns the long frame ``(by?, pair,
    time_refresh, a_value, b_value, n_ticks_a, n_ticks_b,
    retained_fraction)`` with ``pair = "a|b"`` in ``names`` order.

    **The staleness caveat** (their §2.1): the output looks synchronous and
    is not. A refresh vector is treated as observed at ``time_refresh``, but
    each series' value is up to one of its own inter-tick intervals old.
    ``n_ticks_<s>`` is that staleness made visible: the series with the
    largest count is the one holding the grid up, and the one whose value is
    freshest.

    Rows must be in ``time`` order within each ``by`` key, as a stream must
    be; a time below the previous row's is a ``ValueError`` naming the row. A
    null ``value`` is a tick that observed nothing, so it does not update the
    series. Feeding the input in one chunk or a thousand gives the same grid:
    a point is a property of the ticks up to it.

    ``ValueError`` for fewer than two ``names`` or a duplicate, and for a
    column the frame has not got.
    """
    lazy = lf.lazy()
    in_schema = lazy.collect_schema()
    names = list(names)
    keep = list(keep)
    for role, col in [("series", series), ("time", time), ("value", value)] + (
        [("by", by)] if by is not None else []
    ):
        if col not in in_schema:
            msg = f"refresh_time: no {role} column {col!r} in the frame; it has {in_schema.names()}"
            raise ValueError(msg)
    for col in keep:
        if col not in in_schema:
            msg = f"refresh_time: no keep column {col!r} in the frame; it has {in_schema.names()}"
            raise ValueError(msg)
    rows = chunk_rows if chunk_rows is not None else _native.default_chunk_rows()
    if rows < 1:
        msg = f"chunk_rows must be at least 1, got {rows}"
        raise ValueError(msg)

    def build() -> _native.RefreshTime:
        return _native.RefreshTime(names, series, time, value, by, pairs, keep)

    # The schema is what the operator says it is, taken from a run over no
    # rows -- so a name that cannot be a column is reported while the plan is
    # built, as polars reports its own schema errors.
    schema = build().feed(pl.DataFrame(schema=in_schema)).schema

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # Polars does not re-apply the three pushdowns after a Python source,
        # so each is honoured here, and in the order `_frame.py` explains:
        # the slice counts *output* rows, since the grid is what the query
        # sliced.
        rt = build()
        seen = 0
        for chunk in lazy.collect_batches(chunk_size=rows, maintain_order=True):
            out = rt.feed(chunk)
            if n_rows is not None:
                out = out.head(n_rows - seen)
            seen += out.height
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            yield out
            if n_rows is not None and seen >= n_rows:
                break

    return register_io_source(source, schema=schema, validate_schema=True)
