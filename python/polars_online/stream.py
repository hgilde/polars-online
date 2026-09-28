"""Streaming transforms of a plan: streams whose labels arrive late, and series
that tick at their own times.

Everything here takes a ``LazyFrame`` or a ``DataFrame`` and gives back the same
kind, in O(state) memory, before or beside a bank. Every function follows the
same rules (docs/PLAN.md task 105):

- **Frame first, the same kind back.** Chain with Polars' own
  ``lf.pipe(po.stream.embargo, clock="t", delay=5.0)``.
- **One vocabulary, the specs'.** ``clock`` and ``group`` mean what they mean
  on a spec, and a clock parameter takes a duration on a temporal clock. With
  no ``clock``, one unit is one row.
- **A stateful transform resumes** with ``load_state`` and ``save_state``, and
  takes ``chunk_rows`` as everywhere. :func:`embargo` is a pure Polars plan with
  nothing to save, and takes neither.

:func:`embargo` turns a frame into the doubled stream a forward-looking target
needs: every row appears twice, once as a prediction at its own clock with
zero weight, and once as a lesson at ``clock + delay``, the two merged back
into clock order. It is the recipe a spec's ``label_delay`` runs natively,
written out in polars: useful for seeing what the delay does, for a model that
has no ``label_delay``, and as the oracle the native path is tested against.

:func:`refresh_time` puts asynchronous series on a common grid by
Barndorff-Nielsen, Hansen, Lunde & Shephard's refresh-time rule: a grid point
wherever every series has ticked at least once since the last one. The scan is
a Rust operator, wrapped here as a lazy source.

Everything here streams (``merge_sorted`` on two sorted halves of the same
frame, a chunk-fed operator for the grid), so a stream too long to hold is
still too long to hold, and this does not change that.

This module was ``polars_online.prep`` until 0.12.0, and ``refresh_time``'s
``time=`` and ``by=`` were its ``clock=`` and ``group=``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _polars_online as _native
from polars_online._duration import Duration, clock_nanoseconds
from polars_online._frame import State, _read_state, _save_path

__all__ = ["embargo", "refresh_time"]

#: Column :func:`embargo` adds to say which copy of a row this is.
ROLE = "_online_role"


@overload
def embargo(
    lf: pl.LazyFrame,
    *,
    clock: str,
    delay: float | Duration,
    weight: str | None = None,
    role: str = ROLE,
) -> pl.LazyFrame: ...


@overload
def embargo(
    lf: pl.DataFrame,
    *,
    clock: str,
    delay: float | Duration,
    weight: str | None = None,
    role: str = ROLE,
) -> pl.DataFrame: ...


def embargo(
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    clock: str,
    delay: float | Duration,
    weight: str | None = None,
    role: str = ROLE,
) -> pl.LazyFrame | pl.DataFrame:
    """The doubled stream for a target that is only known ``delay`` later.

    Why: a target that is a forward quantity over ``delay`` clock units is not
    known at the row it sits on. A stream that learns it there has seen ``delay``
    of the future before predicting the rows in between, and every "out-of-sample"
    number after that is contaminated; with an autocorrelated feature, even a pure
    noise column will show a correlation with its target. Zero-weight rows are
    legal and mean "advance the clock, learn nothing", so the doubled stream says
    exactly what is wanted: predict here, learn later.

    Every row comes back twice, in clock order, in the kind of frame it came in:

    - a predict row at ``clock``, with its weight forced to 0, so the model scores
      it and learns nothing from it;
    - a learn row at ``clock + delay``, carrying the same features and target at
      full weight.

    A ``role`` column says which is which (``"predict"`` / ``"learn"``), so the
    output is filtered back down with ``out.filter(pl.col(role) == "predict")``.
    ``weight`` names an existing weight column; without it the function adds one,
    named ``role + "_weight"``, that is 1 on learn rows and 0 on predict rows;
    pass that name to the spec's ``weight=``.

    .. code-block:: python

        doubled = po.stream.embargo(lf, clock="t", delay=5.0)  # adds two role columns
        scored = doubled.online.fit_predict(
            [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", max_dclock=10.0,
                             halflife=50.0, weight="_online_role_weight")]
        ).filter(pl.col("_online_role") == "predict").collect()

    The frame must already be in ``clock`` order across all its rows, not only
    within each group, which is all a stream needs: the two copies are merged by the
    clock alone, so a frame sorted within its groups but not across them comes back
    with a group's rows out of order, and a bank then refuses it under the default
    ``on_clock_reset`` (task 120). Sort by the clock first (``lf.sort(clock,
    maintain_order=True)``), or embargo each group and concatenate. The result is
    sorted by ``clock`` with learn rows before predict rows at the same clock value:
    a label whose ``delay`` has just run out is known at that instant, so a
    prediction made then may use it. A spec's ``label_delay`` releases in the same
    order, which is what lets the two be compared row for row. A spec's
    ``label_delay=`` does the same thing in the stream with no doubling and no
    filtering, which is cheaper and does not need the frame rewritten; reach for
    this when a delay has to be visible in the data (an oracle, a demonstration, or
    an engine that is not this one).

    ``delay`` is measured the way a spec's clock parameters are: a number of the
    clock's own units for a numeric clock, and a duration for a ``Datetime``,
    ``Date`` or ``Duration`` one (``pl.duration(minutes=5)``, ``timedelta`` or
    ``"5m"``). A duration must be a whole number of the column's own steps, since
    the learn copy's clock is the column plus the delay: a ``Date`` clock takes
    whole days, where ``"12h"`` would be cut to nothing.

    ``ValueError`` for a ``delay`` that is not finite and positive (``0`` would be
    the undoubled stream, and negative a label from the past), for a ``delay``
    of the wrong kind for the clock, for a ``clock`` or ``weight`` column the
    frame has not got, and for a frame that already has a column named ``role``.
    """
    lazy = lf.lazy()
    schema = lazy.collect_schema()
    if clock not in schema:
        msg = f"embargo: no clock column {clock!r} in the frame; it has {schema.names()}"
        raise ValueError(msg)
    ns = clock_nanoseconds(delay, schema[clock], "embargo", "delay", clock)
    if ns is None:
        if not (delay > 0.0) or delay == float("inf"):  # type: ignore[operator]
            msg = f"embargo: delay must be finite and > 0, got {delay!r}"
            raise ValueError(msg)
        later = pl.col(clock) + delay
    else:
        # Exact: the delay is whole steps of the column, so the cast back to
        # its own dtype drops nothing.
        later = (pl.col(clock) + pl.duration(nanoseconds=ns)).cast(schema[clock])
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
        later.alias(clock),
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
    out = (
        both[0]
        .merge_sorted(both[1], key=key)
        .drop(key, "__embargo_order")
        .select(*schema.names(), *([] if weight is not None else [wcol]), role)
    )
    return out if isinstance(lf, pl.LazyFrame) else out.collect()


@overload
def refresh_time(
    lf: pl.LazyFrame,
    *,
    series: str,
    names: Sequence[str],
    clock: str,
    value: str,
    group: str | None = None,
    pairs: bool = False,
    keep: Sequence[str] = (),
    chunk_rows: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
) -> pl.LazyFrame: ...


@overload
def refresh_time(
    lf: pl.DataFrame,
    *,
    series: str,
    names: Sequence[str],
    clock: str,
    value: str,
    group: str | None = None,
    pairs: bool = False,
    keep: Sequence[str] = (),
    chunk_rows: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
) -> pl.DataFrame: ...


def refresh_time(
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    series: str,
    names: Sequence[str],
    clock: str,
    value: str,
    group: str | None = None,
    pairs: bool = False,
    keep: Sequence[str] = (),
    chunk_rows: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Asynchronous series on a common grid, by refresh time.

    Series observed at their own times cannot be correlated directly: the Epps
    effect attenuates a correlation computed over a fine grid, and filling forward
    invents observations. Barndorff-Nielsen, Hansen, Lunde & Shephard's rule
    places a grid point at the first instant by which every series has ticked at
    least once since the previous point, and takes each series' last value there:

    .. code-block:: text

        tau_0     = max_i (first tick of series i)
        tau_{j+1} = max_i (first tick of series i strictly after tau_j)

    Nothing is interpolated, every value in the output was observed, and the grid
    adapts to the slowest series rather than carrying a stale value across an
    interval.

    The input is long: one row per tick, with a ``series`` column naming it, a
    ``clock`` and a ``value``. A wide frame is already synchronised;
    ``lf.unpivot(index=[clock], variable_name="series", value_name="value")``
    is the line that makes one from the other. ``names`` is required and gives the
    series in output order; it is not discovered from the data because a lazy plan
    has to declare its schema before a row is read, and the output columns are
    named after the series. A row whose ``series`` is not in ``names`` is an error
    naming it, since dropping it would hide a misspelling.

    Returns the kind of frame it was given, with one row per grid point:

    ``time_refresh``
        The completing tick's clock: the max over series of their last update,
        their Definition 1.
    ``<s>_value``
        Each series' last value at that instant.
    ``n_obs_<s>``
        Ticks of ``s`` since the previous point, the first of which is the one on
        the grid.
    ``retained_fraction``
        ``m / sum(n_obs)``: how many of the interval's ticks the grid kept. Look
        at it before trusting a correlation computed on the result.

    plus the ``group`` column, in the dtype it came in as, and any ``keep``
    columns at their value on the completing tick. ``pairs=True`` runs an
    independent two-series grid per unordered pair instead, which keeps far more
    of the data when one series is slow, and returns the long frame ``(group?,
    pair, time_refresh, a_value, b_value, n_obs_a, n_obs_b, retained_fraction)``
    with ``pair = "a|b"`` in ``names`` order.

    .. code-block:: python

        grid = po.stream.refresh_time(      # a DataFrame in, a DataFrame out
            ticks, series="symbol", names=["AAA", "BBB", "CCC"], clock="t", value="px"
        )
        returns = grid.select(pl.col("^.*_value$").diff())   # what rcov and ew_cov take

    The staleness caveat (their section 2.1): the output looks synchronous and is
    not. A refresh vector is treated as observed at ``time_refresh``, but each
    series' value is up to one of its own inter-tick intervals old. ``n_obs_<s>``
    is that staleness made visible: the series with the largest count is the one
    holding the grid up, and the one whose value is freshest.

    Rows must be in ``clock`` order within each ``group``, as a stream must be; a
    clock below the previous row's is a ``ValueError`` naming the row, as under a
    spec's default ``on_clock_reset``. A temporal clock is compared exactly, in
    integer nanoseconds. A null ``value`` is a tick that observed nothing, so it
    does not update the series. Feeding the input in one chunk or a thousand
    gives the same grid, since a point is a property of the ticks up to it. Ties
    are broken by row order: "strictly after ``tau_j``" is read against the row
    sequence, so a tick carrying the same clock value as the one that just closed
    a point, but later in the frame, belongs to the next interval. That is what
    lets a point be emitted the moment its last series ticks, which is what makes
    the result chunk-invariant; sort the input by ``clock`` and by the order you
    want within a clock value.

    **It resumes.** ``save_state`` writes the sampler's state once the input is
    fed: every group's grid, part-way through an interval or not, and its last
    clock, whole or not at all. Under a slice of the output (``.head(n)``), the
    input is read up to the tick that completed the last point returned and no
    further, so the state saved is the state after that tick, whatever the chunk
    size, as a bank's is after the rows a ``head(n)`` pulled: a run resumed on the
    input after that tick goes on with the next point. With ``pairs=True`` one
    tick can complete several pairs' points at once; if the slice ends among
    them, the rest are in the state and not in the output. ``load_state`` is read
    when the plan is built and goes on from there, so feeding a stream in two runs
    gives the grid one run gives. It must have been saved with the same
    ``names`` and ``pairs``, on the same kind of clock (a temporal state resumes
    on any unit), and a row before its group's saved clock is refused like any
    other step back.

    ``ValueError`` for fewer than two ``names`` or a duplicate, for a column the
    frame has not got, for ``chunk_rows`` below 1, and for a ``load_state`` that
    is not such a state or was saved with other ``names`` or ``pairs``;
    ``FileNotFoundError`` for a ``load_state`` that is not there or a
    ``save_state`` whose directory is not.
    """
    lazy = lf.lazy()
    in_schema = lazy.collect_schema()
    names = list(names)
    keep = list(keep)
    for role, col in [("series", series), ("clock", clock), ("value", value)] + (
        [("group", group)] if group is not None else []
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
    # Read now, as a bank's `load_state` is: a plan collected twice goes on
    # from the same state, whatever the file holds by then.
    loaded = _read_state(load_state) if load_state is not None else None
    save_path = _save_path(save_state)

    def build() -> _native.RefreshTime:
        if loaded is not None:
            return _native.RefreshTime.load_bytes(
                loaded, names, series, clock, value, group, pairs, keep
            )
        return _native.RefreshTime(names, series, clock, value, group, pairs, keep)

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
        # sliced. Under a slice the sampler stops at the tick that completed
        # the last point wanted (`feed(limit=)`), so the state is the state
        # after the input behind the rows returned, as a bank's is after a
        # `head(n)`, and the same whatever the chunk size.
        rt = build()
        seen = 0
        for chunk in lazy.collect_batches(chunk_size=rows, maintain_order=True):
            out = rt.feed(chunk, None if n_rows is None else n_rows - seen)
            if n_rows is not None:
                # A tick that completes several pairs' points is taken whole.
                out = out.head(n_rows - seen)
            seen += out.height
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            yield out
            if n_rows is not None and seen >= n_rows:
                break
        # Reached once the input is fed, or the slice is: not on a run the
        # caller abandons, nor one the sampler ended with an error.
        if save_path is not None:
            rt.save(save_path)

    plan = register_io_source(source, schema=schema, validate_schema=True)
    return plan if isinstance(lf, pl.LazyFrame) else plan.collect()
