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
needs. Every row appears twice: once as a prediction at its own clock with
zero weight, and once as a lesson at ``clock + delay``, the two merged back
into clock order. It is the recipe a spec's ``embargo`` runs natively,
written out in polars. It is useful for seeing what the delay does, for a
model that has no ``embargo``, and as the oracle the native path is tested
against.

:func:`with_windows` adds exponentially weighted means with a hard cutoff, looking
back or ahead, described with :mod:`polars_online.ops`: any number of them
in one pass, each row out once every window over it has closed.

:func:`refresh_time` puts asynchronous series on a common grid by
Barndorff-Nielsen, Hansen, Lunde & Shephard's refresh-time rule: a grid point
wherever every series has ticked at least once since the last one. The scan is
a Rust operator, wrapped here as a lazy source.

Everything here streams: ``merge_sorted`` on two sorted halves of the same
frame, a chunk-fed operator for the grid. A stream too long to hold is still
too long to hold, and this does not change that.
"""

from __future__ import annotations

import json
import math
import numbers
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, overload

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _formula
from polars_online import _polars_online as _native
from polars_online._duration import Duration, clock_nanoseconds, duration_text
from polars_online._frame import (
    ConsumedSourceWarning,
    State,
    _explain_named,
    _is_python_scan,
    _plan_text,
    _read_state,
    _save_path,
    _source_started,
    _sources_ran_since,
    _user_stacklevel,
    _warn_if_order_unspecified,
)
from polars_online._spec import _RENAMED

__all__ = ["embargo", "refresh_time", "with_windows"]

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
    known at the row it sits on. A stream that learns it there has seen
    ``delay`` of the future before predicting the rows in between, and every
    "out-of-sample" number after that is contaminated. With an autocorrelated
    feature, even a pure noise column will show a correlation with its target.
    Zero-weight rows are legal and mean "advance the clock, learn nothing", so
    the doubled stream says exactly what is wanted: predict here, learn later.

    Every row comes back twice, in clock order, in the kind of frame it came in:

    .. list-table::
       :header-rows: 1
       :widths: 22 24 54

       * - the row
         - at
         - carrying
       * - a predict row
         - ``clock``
         - its weight forced to 0, so the model scores it and learns nothing
           from it
       * - a learn row
         - ``clock + delay``
         - the same features and target at full weight

    A ``role`` column says which is which (``"predict"`` / ``"learn"``), so the
    output is filtered back down with ``out.filter(pl.col(role) == "predict")``.
    ``weight`` names an existing weight column, of any numeric dtype, which
    comes back as ``Float64``, the number the bank reads it as. Without it the
    function adds one, named ``role + "_weight"``, that is 1 on learn rows and
    0 on predict rows; pass that name to the spec's ``weight=``.

    It reads its input twice, once for each copy, merging the two as they
    stream. A source that can be read once only -- what
    ``pl.scan_arrow_c_stream`` builds over a DuckDB relation or a pyarrow
    reader -- gives the second read nothing, and half the stream would go
    missing with no error. Whether a source is spent cannot be seen until
    the plan runs, so an input whose source is a Python scan, which such a
    stream is, is warned about with :class:`~polars_online.ConsumedSourceWarning`
    as the plan is built: write it to a file and scan that, or collect it
    first.

    .. code-block:: python

        doubled = po.stream.embargo(lf, clock="t", delay=5.0)  # adds two role columns
        scored = doubled.online.fit_predict(
            [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0,
                             half_life=50.0, weight="_online_role_weight")]
        ).filter(pl.col("_online_role") == "predict").collect()

    The frame must already be in ``clock`` order across all its rows, not only
    within each group, which is all a stream needs. The two copies are merged
    by the clock alone, so a frame sorted within its groups but not across
    them comes back with a group's rows out of order. A bank then refuses it,
    as it refuses every step back unless ``restart_after_step_back`` says it
    is a new start (task 120). Sort by the clock first (``lf.sort(clock,
    maintain_order=True)``), or embargo each group and concatenate. The result
    is sorted by ``clock`` with learn rows before predict rows at the same
    clock value: a label whose ``delay`` has just run out is known at that
    instant, so a prediction made then may use it. A spec's ``embargo``
    releases in the same order, which is what lets the two be compared row
    for row. A spec's ``embargo=`` does the same thing in the stream with no
    doubling and no filtering: no second copy of every row, and no frame
    rewritten. Reach for this when a delay has to be visible in the data (an
    oracle, a demonstration, or an engine that is not this one).

    ``delay`` is measured the way a spec's clock parameters are: a number of
    the clock's own units for a numeric clock, and a duration for a
    ``Datetime``, ``Date`` or ``Duration`` one (``pl.duration(minutes=5)``,
    ``timedelta`` or ``"5m"``). A duration must be a whole number of the
    column's own steps, since the learn copy's clock is the column plus the
    delay: a ``Date`` clock takes whole days, where ``"12h"`` would be cut to
    nothing.

    On an integer clock a whole ``delay``, ``5.0`` as well as ``5``, keeps the
    clock's dtype.

    ``ValueError`` for:

    - a ``delay`` that is not finite and positive (``0`` would be the
      undoubled stream, and negative a label from the past);
    - a ``delay`` of the wrong kind for the clock, or one that is not a whole
      number on an integer clock;
    - a ``clock`` or ``weight`` column the frame has not got;
    - a frame that already has a column named ``role``, or, without
      ``weight``, one named ``role + "_weight"``.

    ``TypeError`` for a ``clock`` column that is neither numeric nor temporal,
    as :func:`polars_online.eval.rolling_metrics` refuses it.
    """
    lazy = lf.lazy()
    schema = lazy.collect_schema()
    if clock not in schema:
        msg = f"embargo: no clock column {clock!r} in the frame; it has {schema.names()}"
        raise ValueError(msg)
    if not (schema[clock].is_numeric() or schema[clock].is_temporal()):
        # Named before the plan is built; a String clock failed inside polars'
        # arithmetic when the plan ran (review round 4, YB15).
        msg = f"embargo: clock column {clock!r} must be numeric or temporal, got {schema[clock]}"
        raise TypeError(msg)
    ns = clock_nanoseconds(delay, schema[clock], "embargo", "delay", clock)
    if ns is None:
        if not (delay > 0.0) or delay == float("inf"):  # type: ignore[operator]
            msg = f"embargo: delay must be finite and > 0, got {delay!r}"
            raise ValueError(msg)
        later = pl.col(clock) + delay
        if schema[clock].is_integer():
            # The learn copy's clock must keep the column's dtype, or the
            # merge of the two copies dies in a SchemaError; `delay=5.0`, the
            # example above, did (review 2026-10-05, YB3).
            if not float(delay).is_integer():  # type: ignore[arg-type]
                msg = (
                    f"embargo: delay {delay!r} is not a whole number, and clock column "
                    f"{clock!r} is {schema[clock]}, which holds whole numbers; give a whole "
                    "delay, or cast the clock to a float"
                )
                raise ValueError(msg)
            later = (pl.col(clock) + int(delay)).cast(schema[clock])  # type: ignore[arg-type]
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
    if weight is None and wcol in schema:
        # The weight column added beside the role, which polars refused as a
        # duplicate (review 2026-10-05, YB10).
        msg = (
            f"embargo: the frame already has a column named {wcol!r}, the weight column this "
            "adds; pass another `role=`, or name that column as the `weight=` to zero"
        )
        raise ValueError(msg)
    if _is_python_scan(lazy):
        # The two copies are two reads of the input (review round 4, YB3).
        warnings.warn(
            ConsumedSourceWarning(
                "embargo: the input's source is a Python scan, and embargo reads its input "
                "twice, once for each copy. A source that can be read once only -- what "
                "`pl.scan_arrow_c_stream` builds over a DuckDB relation, a pyarrow reader, or "
                "anything exposing `__arrow_c_stream__` -- gives the second read nothing, and "
                "half the doubled stream goes missing with no error. Write the input to a file "
                "and scan that, or collect it first. If the source can be read twice, silence "
                'this with warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning).'
            ),
            stacklevel=_user_stacklevel(),
        )
    # `merge_sorted` needs both halves sorted on the key it merges by. Each
    # half is the input in its own order, so a single key sorts both: the
    # clock, with the learn copy first at a tie. The weight is a float in
    # both, so the two schemas agree whatever the column's dtype; `w * 0.0`
    # beside an integer `w` died in the merge (review round 4, YB2).
    weight_value = pl.col(weight).cast(pl.Float64) if weight is not None else None
    predict = lazy.with_columns(
        pl.lit("predict").alias(role),
        (weight_value * 0.0 if weight_value is not None else pl.lit(0.0)).alias(wcol),
        pl.lit(1, pl.UInt8).alias("__embargo_order"),
    )
    learn = lazy.with_columns(
        pl.lit("learn").alias(role),
        (weight_value if weight_value is not None else pl.lit(1.0)).alias(wcol),
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

    Series observed at their own times cannot be correlated directly: the
    Epps effect attenuates a correlation computed over a fine grid, and
    filling forward invents observations. Barndorff-Nielsen, Hansen, Lunde &
    Shephard's rule places a grid point at the first instant by which every
    series has ticked at least once since the previous point, and takes each
    series' last value there:

    .. code-block:: text

        tau_0     = max_i (first tick of series i)
        tau_{j+1} = max_i (first tick of series i strictly after tau_j)

    Nothing is interpolated, every value in the output was observed, and the
    grid adapts to the slowest series rather than carrying a stale value
    across an interval.

    The input is long: one row per tick, with a ``series`` column naming it,
    a ``clock`` and a ``value``. A wide frame is already synchronised;
    ``lf.unpivot(index=[clock], variable_name="series", value_name="value")``
    is the line that makes one from the other. ``names`` is required and
    gives the series in output order. It is not discovered from the data,
    because a lazy plan has to declare its schema before a row is read, and
    the output columns are named after the series. A row whose ``series`` is
    not in ``names`` is an error naming it, since dropping it would hide a
    misspelling.

    Returns the kind of frame it was given, with one row per grid point:

    ``time_refresh``
        The completing tick's clock: the max over series of their last
        update, their Definition 1.
    ``<s>_value``
        Each series' last value at that instant.
    ``n_obs_<s>``
        Ticks of ``s`` since the previous point. The grid keeps the last of
        them, the series' value at the refresh time.
    ``retained_fraction``
        ``m / sum(n_obs)``: how many of the interval's ticks the grid kept.
        Look at it before trusting a correlation computed on the result.

    plus the ``group`` column, in the dtype it came in as, and any ``keep``
    columns at their value on the completing tick. ``pairs=True`` runs an
    independent two-series grid per unordered pair instead, which keeps far
    more of the data when one series is slow. It returns the long frame
    ``(group?, pair, time_refresh, a_value, b_value, n_obs_a, n_obs_b,
    retained_fraction)`` with ``pair = "a|b"`` in ``names`` order.

    .. code-block:: python

        ticks = pl.DataFrame({              # one row per tick, each series at its own times
            "symbol": ["AAA", "BBB", "AAA", "CCC", "BBB", "AAA", "AAA", "CCC"],
            "t": [0.4, 0.9, 1.3, 1.6, 2.2, 2.5, 2.8, 3.1],
            "px": [100.0, 20.0, 100.2, 50.0, 20.1, 100.1, 100.4, 49.9],
        })
        grid = po.stream.refresh_time(      # a DataFrame in, a DataFrame out
            ticks, series="symbol", names=["AAA", "BBB", "CCC"], clock="t", value="px"
        )
        returns = grid.select(pl.col("^.*_value$").diff())   # what rcov and ew_cov take

    The staleness caveat (their section 2.1): the output looks synchronous
    and is not. A refresh vector is treated as observed at ``time_refresh``,
    but each series' value is up to one of its own inter-tick intervals old.
    ``n_obs_<s>`` counts the ticks of each series the point folded in, of
    which it kept one: a large count is ticks the grid dropped, and the
    series holding the grid up, the one whose tick completes each point,
    sits near 1.

    Rows must be in ``clock`` order within each ``group``, as a stream must
    be. A clock below the previous row's is refused naming the row, as a spec
    refuses one with ``restart_after_step_back`` unset. The refusal is a
    ``ValueError``, which py-polars 1.x hands on inside its ``ComputeError``
    because the sampler runs as a polars source, for a ``DataFrame`` too. A temporal
    clock is compared exactly, in integer nanoseconds. A ``value`` that is
    missing -- null, NaN, infinite, or past 1e100 in magnitude, the rule every
    spec column follows -- is a tick that observed nothing, so it does not
    update the series. Feeding the input in one chunk or a thousand gives the
    same grid, since a point is a property of the ticks up to it. Ties are
    broken by row order: "strictly after ``tau_j``" is read against the row
    sequence. So a tick carrying the same clock value as the one that just
    closed a point, but later in the frame, belongs to the next interval.
    That is what lets a point be emitted the moment its last series ticks,
    which is what makes the result chunk-invariant. Sort the input by
    ``clock`` and by the order you want within a clock value.

    **It resumes.** ``save_state`` writes the sampler's state once the input
    is fed: every group's grid, part-way through an interval or not, and its
    last clock, whole or not at all. Under a slice of the output
    (``.head(n)``), the input is read up to the tick that completed the last
    point returned and no further. So the state saved is the state after
    that tick, whatever the chunk size, as a bank's is after the rows a
    ``head(n)`` pulled. A run resumed on the input after that tick goes on
    with the next point. With ``pairs=True`` one tick can complete several
    pairs' points at once; if the slice ends among them, the rest are in the
    state and not in the output. ``load_state`` is read when the plan is
    built and goes on from there, so feeding a stream in two runs gives the
    grid one run gives. It must have been saved with the same ``names`` and
    ``pairs``, on the same kind of clock (a temporal state resumes on any
    unit). A row before its group's saved clock is refused like any other
    step back.

    ``ValueError`` for:

    - fewer than two ``names``, or a duplicate;
    - a column the frame has not got;
    - a ``value`` column that is not numeric (a boolean and a column of nulls
      are);
    - ``chunk_rows`` below 1;
    - a ``load_state`` that is not such a state, or was saved with other
      ``names`` or ``pairs``.

    ``FileNotFoundError`` for a ``load_state`` that is not there, or a
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
    # A number, as a spec's feature is: a boolean or a column of nulls too
    # (review round 4, PC7). Read through a non-strict cast, text came out
    # all null, an empty grid with nothing said, and a `Date` as its days.
    value_dtype = in_schema[value]
    if not (value_dtype.is_numeric() or value_dtype in (pl.Boolean, pl.Null)):
        msg = (
            f"refresh_time: value column {value!r} has dtype {value_dtype}; it must be numeric "
            f"(cast it, e.g. pl.col({value!r}).cast(pl.Float64))"
        )
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


#: The clock keywords ``like=`` takes from a spec, and refuses beside it.
_CLOCK_KEYS = (
    "clock",
    "gap_cap",
    "restart_after_step_back",
    "session",
    "session_gap",
    "group",
)


def _clock_value(value: Any, who: str, key: str) -> Any:
    """A clock quantity as the JSON the Rust side reads: a duration as its
    text, an infinity as ``"inf"``, NaN refused."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        x = float(value)
        if math.isnan(x):
            raise ValueError(f"{who}: {key} must not be NaN")
        return ("inf" if x > 0 else "-inf") if math.isinf(x) else value
    return duration_text(value, who, key)


#: The clock keywords of ``with_windows``, reserved: not output names.
_WINDOW_KEYS = ("clock", "gap_cap", "restart_after_step_back", "session", "session_gap", "group")


def _formulas(who: str, exprs: tuple[Any, ...], named: dict[str, Any]) -> list[dict[str, Any]]:
    """The call's expressions as named trees: a keyword names its expression,
    a positional one is named by its alias."""
    out: list[dict[str, Any]] = []
    for e in exprs:
        if not isinstance(e, pl.Expr):
            raise TypeError(
                f"{who}: an expression is a pl.Expr over the operators, got {type(e).__name__}"
            )
        tree = _formula.to_tree(e)
        name = e.meta.output_name()
        if name.startswith(_formula.PREFIX):
            raise ValueError(
                f"{who}: an expression whose name would be an operator's needs a name: give it "
                f'with .alias("...") or as a keyword, with_windows(name=expr)'
            )
        out.append({"name": name, "tree": tree})
    for name, e in named.items():
        if name in _RENAMED:
            raise TypeError(f"{who}: {name} was renamed {_RENAMED[name]}")
        if not isinstance(e, pl.Expr):
            raise TypeError(
                f"{who}: {name} is a pl.Expr over the operators, got {type(e).__name__} {e!r}"
            )
        out.append({"name": name, "tree": _formula.to_tree(e)})
    for f in out:
        if not _formula.operators(f["tree"]):
            raise ValueError(
                f"{who}: {f['name']!r} holds no operator; a formula of the row alone is Polars' "
                f"with_columns"
            )
    return out


@overload
def with_windows(
    lf: pl.LazyFrame,
    *exprs: pl.Expr,
    clock: str | None = None,
    gap_cap: float | Duration | None = None,
    restart_after_step_back: float | Duration | None = None,
    session: str | None = None,
    session_gap: float | Duration | None = None,
    group: str | None = None,
    like: dict[str, Any] | None = None,
    chunk_rows: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
    **named: pl.Expr,
) -> pl.LazyFrame: ...


@overload
def with_windows(
    lf: pl.DataFrame,
    *exprs: pl.Expr,
    clock: str | None = None,
    gap_cap: float | Duration | None = None,
    restart_after_step_back: float | Duration | None = None,
    session: str | None = None,
    session_gap: float | Duration | None = None,
    group: str | None = None,
    like: dict[str, Any] | None = None,
    chunk_rows: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
    **named: pl.Expr,
) -> pl.DataFrame: ...


def with_windows(
    lf: pl.LazyFrame | pl.DataFrame,
    *exprs: pl.Expr,
    clock: str | None = None,
    gap_cap: float | Duration | None = None,
    restart_after_step_back: float | Duration | None = None,
    session: str | None = None,
    session_gap: float | Duration | None = None,
    group: str | None = None,
    like: dict[str, Any] | None = None,
    chunk_rows: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
    **named: pl.Expr,
) -> pl.LazyFrame | pl.DataFrame:
    """Formulas over the window operators, computed over a stream in one
    pass, looking back and ahead (:mod:`polars_online.ops`).

    Reads like ``with_columns``: each expression adds a column named by its
    keyword or its ``.alias()``, and the input's columns come through
    unchanged, in order. The operators -- :func:`polars_online.ewm_mean`,
    :func:`polars_online.rewm_mean`, :func:`polars_online.ewm_sum`,
    :func:`polars_online.rewm_sum`, :func:`polars_online.ewm_rate`,
    :func:`polars_online.rewm_rate` and :func:`polars_online.increment` --
    are stateful kernels computed by the core, each distinct one once. The
    formula around them is element-wise Polars, evaluated on each chunk
    emitted. Over ``trades``, quotes with a ``mid`` and trades between them
    with a ``side``, a ``quantity`` and a ``price``, a trailing mean of the
    mid and forward VWAPs of all trades and of the buys:

    .. code-block:: python

        notional = pl.col("price") * pl.col("quantity")
        buys = pl.when(pl.col("side") == "buy")
        out = po.stream.with_windows(
            trades,
            mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
            fwd_vwap=po.rewm_sum(notional, half_life="10s", window_size="1m")
                     / po.rewm_sum("quantity", half_life="10s", window_size="1m"),
            fwd_buy_vwap=po.rewm_sum(buys.then(notional), half_life="10s", window_size="1m")
                         / po.rewm_sum(buys.then("quantity"), half_life="10s", window_size="1m"),
            clock="ts", gap_cap="5m", group="symbol",
        )

    **The clock is a spec's**, in the same words:

    .. list-table::
       :header-rows: 1
       :widths: 32 68

       * - keyword
         - what it does here
       * - ``clock``
         - the clock column; with none, one unit is one row
       * - ``gap_cap``
         - required with a clock; a longer gap is a break
       * - ``restart_after_step_back``
         - unset, a step back is refused; given, one larger than it starts
           over, and one no larger is a late row, refused
       * - ``session``, ``session_gap``
         - a number, a duration or ``"reset"``
       * - ``group``
         - groups that each have their own clock, sessions and windows; a
           window looking ahead under ``group`` needs ``clock``

    A window is measured on that policy clock, after the cap and the session
    gap. A gap longer than ``gap_cap`` or a session change ends every window
    open across it, under each operator's ``partial``; a reset discards
    them: null, never dropped. A window looking ahead under
    ``closed="right"`` or ``"both"`` whose far edge is the last row before
    either holds every row it covers, so it is whole, and gives its value
    under any ``partial``. The rows must be in clock order across
    groups, as one stream. The clock is also read in input order, under the
    same policy, so a step back there is refused or, past
    ``restart_after_step_back``, resets every group, and a gap there ends
    every group's windows. With ``group`` a session is each group's, read on
    its own clock, so a clock that starts over with a session is a step back
    on the stream's; without ``group`` the stream is the one group. A group
    silent past ``gap_cap`` is cut as the stream's clock passes the cap,
    before its next row can say its session changed, so a
    ``session_gap="reset"`` there discards only what is still open.
    ``like=spec`` takes all of this from a spec instead, and with it the
    rule for the rows the spec learns from. A forward window over a row the
    spec would skip (a null feature or weight) is null, as the model never
    learns its target. These keywords are the call's, not output names: an
    expression passed as ``session=...`` is refused; name it with
    ``.alias()``.

    **A formula is element-wise.** Columns, literals, arithmetic,
    comparisons, ``log``, ``exp``, ``abs``, ``sqrt``, ``pow``, ``clip``,
    ``fill_null``, ``is_null``, ``when/then/otherwise``, ``cast`` and
    ``alias`` over the operators. A ``shift``, a cumulative or rolling
    function, ``over`` or an aggregation would depend on the chunking, and
    is refused by name while the plan is built. An operator's input is the
    same kind of formula, ``increment`` included, not another operator. The
    formula is kept as a compact tree of its own, in the plan and in a saved
    state (docs/PLAN.md task 143).

    **Rows leave in input order**, each once every window over it has
    resolved. A backward window under ``closed="right"`` or ``"both"`` waits
    for the next distinct stamp, and a look-ahead for its window to pass, so
    the output trails the input by the longest window. A group that falls
    silent holds every later row for at most ``gap_cap`` of the stream's
    time: past that its next row is certain to open with a gap past the cap,
    so its windows end then. Only a clock column has a cap, so a look-ahead
    under ``group`` needs one, and is refused without it: each group's clock
    would count only its own rows, and a silent group would hold every later
    row to the end of the input. The rows a look-ahead holds are kept as the
    input's own chunks, not copied, so the memory is one window of input
    whatever its width. The kernels' own sums are one window of the rows
    that count in them.

    **It resumes.** ``save_state`` writes the kernels' sums and the rows
    still waiting, which the next run, given ``load_state``, emits first:
    feeding a stream in two runs gives what one run gives. Without it, a
    row whose window has not passed when the input ends is emitted
    unresolved, null. Any resumed run refuses by name an input whose first
    row is at the last stamp the state read, unless that row starts a new
    session: a file boundary inside a tied stamp cannot be told from an
    input that repeats rows the state read, which would come out twice. So
    cut a stream into files between stamps. Without a clock column there is
    no stamp to compare, and a repeated row is not caught.

    Under a slice of the output (``.head(n)``) the input
    is read only up to the row that resolved the *n*-th row, and the state
    records how many rows of the input were consumed so far. A run resumed
    with ``load_state`` on the *same input*, unsliced, skips those rows and
    goes on, so any chain of sliced runs gives what one run gives, whatever
    the chunk size. Such a state knows its input by its first row's clock
    and session, and by the last rows it read: the rows it holds that are
    the input's, or the last row alone where it holds none. An input whose
    first row differs is the next file when the clock policy takes that
    row as a step forward or a new start, which is a step back past
    ``restart_after_step_back`` or a new session. The run then skips
    nothing and first returns the rows the state held. Besides the input
    at the last stamp read, a state saved under a slice refuses by name:

    - one that steps back where the policy refuses it: the same input
      sliced by hand, or an overlapping file (a hand slice that starts
      after the last clock read is taken as the next file);
    - without a clock column, one that does not begin with a new session;
    - one that starts as the saved input did but differs where the state
      was cut, and one shorter than the rows consumed.

    A clock that starts over at the same stamp each day needs a session
    column for the next day's file to differ from the saved input at its
    first row. Without a clock column a row-count clock steps forward at
    every row, so the next file is one that begins with a new session --
    under ``group``, a new session of a group the state has read -- and
    not with the saved input's first session, which marks the saved input
    itself; without a session column either, there is no next file:
    resume on the same input. An input whose rows match the state's where
    it was cut is taken as the same input, so resume a stream whose rows
    can repeat (a daily grid with the same values) on the same input only.
    A state resumes only the call that saved it, on the same kind of clock.

    ``ValueError`` for a formula, a clock policy or a column that cannot
    run (a look-ahead under ``group`` with no ``clock`` among them), an
    output name that collides, and a ``load_state`` another call saved.
    What the run refuses once the plan is running -- a step back,
    naming the row, or a state resumed on another input -- surfaces as
    ``polars.exceptions.ComputeError`` with the message inside under
    py-polars 1.x, since the rows come from a Python source, and as the
    ``ValueError`` itself under 2.0. ``TypeError`` for something that
    is not an expression, or a clock keyword beside ``like=``.
    ``FileNotFoundError`` for a ``load_state`` that is not there, or a
    ``save_state`` whose directory is not.
    """
    who = "with_windows"
    given = {
        "clock": clock,
        "gap_cap": gap_cap,
        "restart_after_step_back": restart_after_step_back,
        "session": session,
        "session_gap": session_gap,
        "group": group,
    }
    for key, value in given.items():
        if isinstance(value, pl.Expr):
            raise TypeError(
                f"{who}: {key} is a clock keyword of the call, not an output name; name the "
                f"expression with .alias({key!r}) or another keyword"
            )
    formulas = _formulas(who, exprs, named)
    config: dict[str, Any] = {"formulas": formulas}
    if like is not None:
        if not (isinstance(like, dict) and isinstance(like.get("features"), list)):
            raise TypeError(
                f"{who}: like must be a spec, such as po.spec.ewridge(...), got {like!r}"
            )
        clash = [k for k, v in given.items() if v is not None]
        if clash:
            raise TypeError(
                f"{who}: like= takes the clock policy from spec {like.get('name')!r}; leave out "
                f"{', '.join(clash)}"
            )
        # A hand-written dict with a clock key under its old name would be
        # read as not setting it (review R1, F1): refused naming the new one.
        if old := [k for k in like if k in _RENAMED]:
            raise TypeError(
                f"{who}: like= spec {like.get('name')!r}: {old[0]} was renamed {_RENAMED[old[0]]}"
            )
        policy = {k: like.get(k) for k in _CLOCK_KEYS}
        weight = like.get("weight")
        config["like"] = {
            "spec": str(like.get("name")),
            "accept": [*like["features"], *([weight] if weight is not None else [])],
        }
    else:
        policy = given
    for key, value in policy.items():
        if value is None:
            continue
        if key in ("gap_cap", "restart_after_step_back", "session_gap"):
            value = _clock_value(value, who, key)
        config[key] = value
    config_json = json.dumps(config)

    lazy = lf.lazy()
    in_schema = lazy.collect_schema()
    rows = chunk_rows if chunk_rows is not None else _native.default_chunk_rows()
    if rows < 1:
        msg = f"chunk_rows must be at least 1, got {rows}"
        raise ValueError(msg)
    # The two plan checks a bank makes of its input, made here of this one's:
    # a bank after this sees only this source, and not the plan beneath it.
    plan_text = _plan_text(lazy)
    _warn_if_order_unspecified(
        lazy, who, plan_text, reads="a window is taken over rows in row order", before="the windows"
    )
    python_scan = _is_python_scan(lazy, plan_text)
    # Read now, as a bank's `load_state` is: a plan collected twice goes on
    # from the same state, whatever the file holds by then.
    loaded = _read_state(load_state) if load_state is not None else None
    save_path = _save_path(save_state)
    empty = pl.DataFrame(schema=in_schema)

    def build() -> _native.Windows:
        if loaded is not None:
            return _native.Windows.load_bytes(loaded, config_json, empty)
        return _native.Windows(config_json, empty)

    # Built once now, so a formula or a column that cannot run is reported
    # while the plan is built, as polars reports its own errors.
    first = build()
    schema = first.feed(empty).schema
    needed = set(first.needed())

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        this_run = _source_started()
        # Polars does not re-apply the three pushdowns after a Python source,
        # so each is honoured here (`_frame.py` explains the order). The
        # slice counts *output* rows, and the run reads the input only up to
        # the row that resolved the last one wanted, so a saved state goes on
        # from there whatever the chunk size. A projection reaches the input
        # only when no state is loaded or saved: the rows a state holds are
        # the input's whole rows, for whatever the next run asks of them.
        w = build()
        plan = lazy
        if with_columns is not None and loaded is None and save_path is None:
            wanted = set(with_columns) | needed
            plan = plan.select([c for c in in_schema.names() if c in wanted])
        seen = 0
        read = 0

        def done(out: pl.DataFrame) -> pl.DataFrame:
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            return out

        sliced = False
        for chunk in plan.collect_batches(chunk_size=rows, maintain_order=True):
            read += chunk.height
            out = w.feed(chunk, None if n_rows is None else n_rows - seen)
            seen += out.height
            yield done(out)
            if n_rows is not None and seen >= n_rows:
                sliced = True
                break
        if not sliced and save_path is None:
            # The input's end: what still waits for a window is unresolved.
            out = w.finish()
            if n_rows is not None:
                out = out.head(n_rows - seen)
            yield done(out)
        # Reached once the input is fed, or the slice is: not on a run the
        # caller abandons, nor one the windows ended with an error. Under a
        # slice -- asked for, satisfied or not (review R5, C2) -- the state
        # records the rows of the input consumed so far, and a run resumed
        # on that input skips them (review R2, W4; R4, B1; R5, C1).
        if save_path is not None:
            w.save(save_path, w.consumed() if n_rows is not None else 0, not sliced)
        # As the bank warns (`ConsumedSourceWarning`): no rows at all from a
        # Python scan is the shape of a spent single-use stream.
        if python_scan and read == 0 and n_rows != 0 and not _sources_ran_since(this_run):
            warnings.warn(
                ConsumedSourceWarning(
                    f"{who}: the plan yielded no rows, and its source is a Python scan -- "
                    "which is what `pl.scan_arrow_c_stream` builds over a DuckDB relation, a "
                    "pyarrow reader, or anything exposing `__arrow_c_stream__`. Such a stream "
                    "is consumed once: collected a second time it yields nothing, with no "
                    "error. Rebuild the plan per collect (call `pl.scan_arrow_c_stream(...)` "
                    "inside the loop, not outside it). If the source really is empty, "
                    'silence this with warnings.simplefilter("ignore", '
                    "polars_online.ConsumedSourceWarning)."
                ),
                stacklevel=_user_stacklevel(),
            )

    plan = register_io_source(
        source,
        schema=schema,
        validate_schema=True,
        is_pure=save_path is None,
        **_explain_named(f"with_windows: {len(formulas)} formula(s)"),
    )
    return plan if isinstance(lf, pl.LazyFrame) else plan.collect()
