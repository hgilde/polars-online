"""Streaming transforms of a query before a bank: targets known only after a
delay, windows looking back or ahead, and series that tick at their own times.

Each function takes a ``LazyFrame`` or a ``DataFrame`` and gives back the same
kind, holding a bounded state in memory rather than the stream, before or
beside a bank. They share three rules:

- **Frame first, the same kind back.** Chain one with Polars' own ``pipe``, as
  ``lf.pipe(po.stream.embargo, clock="t", delay=5.0)``.
- **The clock settings mean what they mean on a spec.** ``clock`` and
  ``group`` read their columns as a spec does, and a clock parameter takes a
  duration on a temporal clock. ``session_gap`` is the one difference, as
  :func:`with_windows` says. With no ``clock``, which only
  :func:`with_windows` allows, one unit is one row.
- **A stateful transform resumes** with ``load_state`` and ``save_state``, and
  takes ``chunk_size``, as a bank does. :func:`embargo` is a plain Polars query
  with nothing to save, and takes neither.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - function
     - what it does
   * - :func:`embargo`
     - doubles a stream for a target known only ``delay`` after its row:
       each row once as a prediction at its own clock with weight 0, and
       once as a learn row at ``clock + delay``, merged back into clock order.
       It is the delay a spec's ``embargo`` runs inside the bank, written out
       in Polars, for seeing what the delay does, for a model with no
       ``embargo``, and as an independent check of the bank's own
       ``embargo``.
   * - :func:`with_windows`
     - adds formulas over the window operators of :mod:`polars_online.ops`
       (exponentially weighted means, sums, rates and quantiles looking back
       or ahead, and variances looking back, each with an optional hard
       cutoff), any number in one pass, each row out once every window over
       it has closed
   * - :func:`refresh_time`
     - puts series that tick at their own times on a common grid by
       Barndorff-Nielsen, Hansen, Lunde & Shephard's refresh-time rule: a
       grid point wherever every series has ticked at least once since the
       last one

Each streams: :func:`embargo` merges two sorted copies of the same input with
Polars' ``merge_sorted``, and the other two run as an operator of the Rust core
fed one chunk at a time, read as a lazy source.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, overload

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _formula, _warnings
from polars_online import _polars_online as _native
from polars_online._bank import _refuse_uint128
from polars_online._duration import Duration, clock_nanoseconds, duration_text
from polars_online._errors import ColumnNotFoundError, RefusedValueError, WrongTypeError
from polars_online._frame import (
    ConsumedSourceWarning,
    State,
    _and_list,
    _check_chunk_size,
    _explain_named,
    _in_morsels,
    _is_python_scan,
    _plan_text,
    _read_state,
    _save_path,
    _source_started,
    _sources_ran_since,
    _user_stacklevel,
    _warn_if_order_unspecified,
)
from polars_online._spec import _RENAMED, _RENAMED_KEYWORDS, _renamed_keywords
from polars_online._warnings import forward_deprecated, warn_unstable

__all__ = ["ROLE", "embargo", "refresh_time", "with_windows"]

#: Column :func:`embargo` adds to say which copy of a row this is.
ROLE = "_online_role"

#: The integer dtypes a clock may have: every one an ``Int64`` holds, or a
#: ``UInt64`` refused by row past it. A wider one is refused by name.
_INTEGER_CLOCKS = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


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
    """Double a stream so that each row's target is learned ``delay`` after its
    row.

    Use it when a target is a forward quantity, known only ``delay`` clock
    units after its row, and you want the delay visible in the data: an
    independent check of a spec's own ``embargo``, a demonstration, or a model
    or engine with no ``embargo`` of its own.
    Otherwise give the spec ``embargo=delay``, which does the same in the
    stream with no second copy of each row, or a ``label_clock`` column when
    each row's target becomes known at a time of its own (Notes).

    .. rubric:: Parameters

    ``lf``
        The stream, a ``LazyFrame`` or a ``DataFrame``, already in ``clock``
        order across all its rows, not only within each group. The two copies
        are merged by the clock alone, so a frame sorted within its groups but
        not across them comes back with a group's clock stepping back. Sort it
        first with ``lf.sort(clock, maintain_order=True)``, or embargo each
        group and concatenate.
    ``clock``
        The clock column: numeric, ``Datetime``, ``Date`` or ``Duration``.
    ``delay``
        The time from predicting a row to learning it, finite and above 0. It
        is measured as a spec's clock parameters are: a number of the clock's
        own units on a numeric clock, and a duration
        (``pl.duration(minutes=5)``, a ``timedelta`` or ``"5m"``) on a temporal
        one. A duration must be a whole number of the column's own steps, since
        the learn copy's clock is the column plus the delay: a ``Date`` clock
        takes whole days, where ``"12h"`` would be cut to nothing. On an
        integer clock a whole ``delay``, ``5.0`` as well as ``5``, keeps the
        clock's dtype.
    ``weight``
        An existing weight column, of any numeric dtype, to carry onto the
        learn rows and set to 0 on the predict rows. It comes back as
        ``Float64``, the number a bank reads it as. The default, ``None``, adds
        a weight column named ``role + "_weight"``: 1 on learn rows and 0 on
        predict rows. Pass that name to the spec's ``weight=``.
    ``role``
        The name of the column that says which copy a row is, ``"predict"`` or
        ``"learn"``. Default ``"_online_role"`` (:data:`ROLE`).

    .. rubric:: Returns

    The kind of frame it was given, with every row twice, in clock order:

    .. list-table::
       :header-rows: 1
       :widths: 22 24 54

       * - the row
         - at
         - carrying
       * - a predict row
         - ``clock``
         - its weight forced to 0, so the model predicts it and learns
           nothing from it
       * - a learn row
         - ``clock + delay``
         - the same features and target at the row's own weight (1 without
           ``weight``)

    The columns are the input's, then the weight column it adds (without
    ``weight``), then ``role``. At the same clock value a learn row comes
    before a predict row, since a target whose ``delay`` has just run out is
    known at that instant, and a prediction made then may use it. A spec's
    ``embargo`` releases in the same order, so the two can be compared row for
    row. Keep the predictions with ``out.filter(pl.col(role) == "predict")``.

    .. rubric:: Raises

    :class:`~polars_online.ColumnNotFoundError`
        The frame has no ``clock`` or ``weight`` column of that name. It is
        both a ``ValueError`` and Polars' ``ColumnNotFoundError``.
    :class:`~polars_online.WrongTypeError`
        The ``clock`` column is neither numeric nor temporal, is an integer
        wider than 64 bits (``Int128``), which a bank refuses as a clock, or
        is a ``Time``, since a time of day starts again at midnight; or, on
        a numeric clock, ``delay`` is neither a number nor a duration, such
        as ``None``. It is both a ``TypeError`` and a ``ValueError``.
    ``ValueError``
        When:

        - ``delay`` is not finite and above 0 (``0`` would be the undoubled
          stream, and a negative delay a target learned before its row);
        - ``delay`` is of the wrong kind for the clock, or is not a whole
          number on an integer clock;
        - ``delay`` is a duration that is not a whole number of the clock
          column's steps, such as ``"12h"`` on a ``Date``;
        - the frame already has a column named ``role``, or, without
          ``weight``, one named ``role + "_weight"``.
    ``TypeError``
        ``delay`` is an expression that reads a column or does not give one
        duration.
    ``polars.exceptions.InvalidOperationError``
        When the query runs, if an integer clock within ``delay`` of its dtype's
        top cannot hold its learn copy. Polars names the column, where wrapping
        the clock around would put the learn copy before the stream.
    :class:`~polars_online.ConsumedSourceWarning`
        A warning, not an error, issued as the query is built when the input
        reads a source written in Python (Notes).

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0]})
        doubled = po.stream.embargo(rows, clock="t", delay=1.0)   # adds a weight and a role column
        for row in doubled.rows():                                # (t, weight, role)
            print(row)

    .. code-block:: text

        (0.0, 0.0, 'predict')
        (1.0, 1.0, 'learn')
        (1.0, 0.0, 'predict')
        (2.0, 1.0, 'learn')
        (2.0, 0.0, 'predict')
        (3.0, 1.0, 'learn')

    A bank fed the doubled stream learns each row's target 5 rows late, and the
    predict rows carry its predictions. This code reads ``lf`` from the
    README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        doubled = po.stream.embargo(lf, clock="t", delay=5.0)
        scored = doubled.online.fit_predict(
            [po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", gap_cap=10.0,
                             half_life=50.0, weight="_online_role_weight")]
        ).filter(pl.col("_online_role") == "predict").collect()
        pred = scored["m"].struct.field("pred_y")
        print(scored.height, pred.count())         # predict rows, and those with a prediction

    .. code-block:: text

        400 393

    .. rubric:: See also

    - The spec's ``embargo`` and ``label_clock`` (:mod:`polars_online.spec`),
      which delay the learning inside the bank.
    -
      `Learn targets known late
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/learn-targets-known-late.md>`_,
      the how-to: each of the three ways, with a check that it worked.

    .. rubric:: Notes

    A target that is a forward quantity over ``delay`` clock units is not known
    at the row it sits on. A stream that learns it there has seen ``delay`` of
    the future before predicting the rows in between, and every out-of-sample
    number after that is contaminated. With an autocorrelated feature, even a
    pure-noise column shows a correlation with its target. A row of weight 0 is
    legal and means "advance the clock, learn nothing", so the doubled stream
    says what is wanted: predict here, learn later.

    When each row's target becomes known at a time of its own, give the spec a
    ``label_clock`` column of those times rather than doubling by hand. A
    doubled stream learns the same targets at the same clocks, so its predict
    rows' predictions are the ones a ``label_clock`` bank serves. But on each
    learn row the bank predicts again before learning, and computes every
    residual diagnostic from that second prediction, not the one the predict
    row was given. On New York's 2013 flights its 90% interval covered 88.0%
    of the delays it was shown while reporting 90.0%. ``label_clock`` scores
    each target against the prediction and interval its row was served.

    The function reads its input twice, once for each copy, merging the two as
    they stream. A source that can be read once only, such as what
    ``pl.scan_arrow_c_stream`` builds over a DuckDB relation or a pyarrow
    reader, gives the second read nothing, and half the stream would go missing
    with no error. Whether a source is spent cannot be seen until the query
    runs. So where the query reads a source written in Python, as such a
    stream does, the function issues a
    :class:`~polars_online.ConsumedSourceWarning` as the query is built. Write
    such an input to a file and read that file as a query, or pass it as a
    frame.

    A bank refuses a doubled stream in which a group's clock steps back, as it
    refuses every step back unless ``restart_after_step_back`` says it is a new
    start.
    """
    lazy = lf.lazy()
    schema = lazy.collect_schema()
    if clock not in schema:
        msg = (
            f"po.stream.embargo: the input has no clock column {clock!r}, only {schema.names()}. "
            "Set clock to one of those columns."
        )
        raise ColumnNotFoundError(msg)
    if not (schema[clock].is_numeric() or schema[clock].is_temporal()):
        # Named before the plan is built; a String clock failed inside polars'
        # arithmetic when the plan ran (review round 4, YB15).
        msg = (
            f"po.stream.embargo: clock column {clock!r} must be numeric or temporal, but it is "
            f"{schema[clock]}. Cast it to a number or a Datetime."
        )
        raise WrongTypeError(msg)
    if schema[clock].is_integer() and schema[clock] not in _INTEGER_CLOCKS:
        # An integer clock is held as an Int64 (task 200), so a bank refuses
        # an Int128 by name; said here, before the doubled frame is built
        # for a bank that would refuse it (review round 5, B3).
        msg = (
            f"po.stream.embargo: clock column {clock!r} is {schema[clock]}, wider than the Int64 "
            "an integer clock is held in. Cast it to Int64 if its values fit, or after "
            "subtracting an origin."
        )
        raise WrongTypeError(msg)
    ns = clock_nanoseconds(delay, schema[clock], "po.stream.embargo", "delay", clock)
    if ns is None:
        if isinstance(delay, bool) or not isinstance(delay, numbers.Real):
            msg = (
                f"po.stream.embargo: delay must be a number on numeric clock column {clock!r}, "
                f"but its type is {type(delay).__name__}. Set delay to the clock units from "
                "predicting a row to learning it, such as 5."
            )
            raise WrongTypeError(msg)
        if not (delay > 0.0) or delay == float("inf"):
            msg = (
                f"po.stream.embargo: delay must be finite and above 0, but it is {delay!r}. Set "
                "delay to the time from predicting a row to learning it, such as 5 clock units."
            )
            raise ValueError(msg)
        later = pl.col(clock) + delay
        if schema[clock].is_integer():
            # The learn copy's clock must keep the column's dtype, or the
            # merge of the two copies dies in a SchemaError; `delay=5.0`, the
            # example above, did (review 2026-10-05, YB3).
            if not float(delay).is_integer():
                msg = (
                    f"po.stream.embargo: delay {delay!r} is not a whole number, but clock column "
                    f"{clock!r} is {schema[clock]}, which holds whole numbers. Give a whole "
                    "delay, or cast the clock to a float."
                )
                raise ValueError(msg)
            # Widened before the add, and cast back strictly: Polars adds
            # integers in the column's width and wraps at its top, so a
            # clock within `delay` of it put the learn copy before the
            # stream, where a bank learned the row before scoring it (review
            # round 5, B2). Past 64 bits the add is in Int128; either way the
            # cast back refuses a sum the column cannot hold when the plan
            # runs. Measured the same on the 1.34.0 floor and on 1.44.2.
            wide = pl.Int128 if schema[clock] in (pl.Int64, pl.UInt64) else pl.Int64
            later = (pl.col(clock).cast(wide) + int(delay)).cast(schema[clock], strict=True)
    else:
        # Exact: the delay is whole steps of the column, so the cast back to
        # its own dtype drops nothing.
        later = (pl.col(clock) + pl.duration(nanoseconds=ns)).cast(schema[clock])
    if weight is not None and weight not in schema:
        msg = (
            f"po.stream.embargo: the input has no weight column {weight!r}, only {schema.names()}. "
            "Set weight to one of those columns, or leave it out."
        )
        raise ColumnNotFoundError(msg)
    if role in schema:
        msg = (
            f"po.stream.embargo: the input already has a column named {role!r}, the name of the "
            "column this adds. Pass another role=."
        )
        raise ValueError(msg)

    wcol = weight if weight is not None else f"{role}_weight"
    if weight is None and wcol in schema:
        # The weight column added beside the role, which polars refused as a
        # duplicate (review 2026-10-05, YB10).
        msg = (
            f"po.stream.embargo: the input already has a column named {wcol!r}, the name of the "
            "weight column this adds. Pass another role=, or pass that column as weight= to have "
            "it set to 0 on the rows that only predict."
        )
        raise ValueError(msg)
    if _is_python_scan(lazy):
        # The two copies are two reads of the input (review round 4, YB3).
        warnings.warn(
            ConsumedSourceWarning(
                "po.stream.embargo: the query reads a source written in Python, and embargo reads "
                "its input twice, once for each copy, so a source that gives its rows once, such "
                "as `pl.scan_arrow_c_stream` over a DuckDB relation or a pyarrow reader, gives the "
                "second read nothing and half the doubled stream goes missing with no error. "
                "Write the input to a file and read that file as a query, or pass its result "
                "as a frame, or, if the source can be read twice, silence this warning with "
                '`warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning)`.'
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
    chunk_size: int | None = None,
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
    chunk_size: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
) -> pl.DataFrame: ...


@_renamed_keywords
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
    chunk_size: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Put series that tick at their own times on a common grid of refresh times.

    Use it before a correlation or a covariance model, such as ``rcov`` or
    ``ew_cov``, over series observed at their own times. A correlation over a
    fine grid of such series is pulled toward 0 (the Epps effect), and filling
    each series forward invents observations. For series already observed at
    the same times, a wide frame needs no grid.

    .. rubric:: Parameters

    ``lf``
        The ticks, a ``LazyFrame`` or a ``DataFrame``, in long form: one row
        per tick, with a ``series`` column naming it, a ``clock`` and a
        ``value``.
        ``lf.unpivot(index=[clock], variable_name="series", value_name="value")``
        makes the long form from a wide one. Rows must be in ``clock`` order
        within each ``group``, as a stream must be.
    ``series``
        The column that names each tick's series.
    ``names``
        The series, at least two and each once, in the order of the output
        columns. It is required, not discovered from the data, because a lazy
        query declares its schema before a row is read, and the output columns
        are named after the series. A row whose ``series`` is not in ``names``
        is refused, naming it, since dropping it would hide a misspelling.
    ``clock``
        The clock column: numeric or temporal (a Boolean and a column of nulls
        count as numbers). A temporal clock is compared exactly, in integer
        nanoseconds.
    ``value``
        The column of the ticks' values, numeric (a Boolean and a column of
        nulls count as numbers). A value that is null, NaN, infinite or past
        1e100 in magnitude is missing, the rule every spec column follows: the
        tick observed nothing, so it does not update its series.
    ``group``
        A column whose every value gets a grid of its own, such as a trading
        day. Default ``None``: one grid.
    ``pairs``
        ``True`` runs an independent grid of two series for each unordered
        pair, which keeps far more of the data when one series is slow. Default
        ``False``: one grid of every series. With two ``names`` the one pair's
        grid is the joint grid.
    ``keep``
        Columns to carry to the output, at their value on the tick that
        completes each grid point. Default: none.
    ``chunk_size``
        The rows read at a time, at least 1. The default, ``None``, is the
        library's default chunk size. It changes no output.
    ``load_state``
        The path of a state :func:`refresh_time` saved, to go on from. It is
        read when the query is built. It must have been saved with the same
        ``names`` and ``pairs``, on the same kind of clock (a temporal state
        resumes on any unit). Default ``None``: start with no state loaded.
    ``save_state``
        The path of a file to save the state to once the input is fed. Default
        ``None``: save nothing (Notes).

    .. rubric:: Returns

    The kind of frame it was given, with one row per grid point:

    ``time_refresh``
        The clock of the tick that completed the point: the latest, over the
        series, of their last update (the paper's Definition 1).
    ``<s>_value``
        Each series' last value at that instant.
    ``n_obs_<s>``
        The ticks of ``s`` since the previous point. The grid keeps the last of
        them, the series' value at the refresh time.
    ``retained_fraction``
        ``m / sum(n_obs)``, with ``m`` the number of series: the share of the
        interval's ticks the grid kept. Read it before trusting a correlation
        computed on the result.

    The ``group`` column comes first, in the dtype it came in as, and the
    ``keep`` columns last. Under ``pairs=True`` it is one long frame of the
    columns ``group`` (with a group), ``pair``, ``time_refresh``, ``a_value``,
    ``b_value``, ``n_obs_a``, ``n_obs_b``, ``retained_fraction`` and the
    ``keep`` columns, with ``pair = "a|b"`` and ``a`` the pair's series listed
    first in ``names``. Each group runs its own pairs.

    .. rubric:: Raises

    :class:`~polars_online.ColumnNotFoundError`
        The frame has no ``series``, ``clock``, ``value``, ``group`` or
        ``keep`` column of that name. It is both a ``ValueError`` and Polars'
        ``ColumnNotFoundError``.
    :class:`~polars_online.WrongTypeError`
        The ``clock`` column is neither numeric nor temporal (a Boolean and a
        column of nulls are numbers), is an ``Int128``, wider than the
        ``Int64`` an integer clock is held in, or is a ``Time``, which starts
        again at midnight; the ``value`` column is not numeric; or a column,
        read or not, is ``UInt128``, which polars-online cannot read (the
        message names a cast to make). It is both a ``TypeError`` and a
        ``ValueError``.
    ``ValueError``
        When:

        - ``names`` lists fewer than two series, or one twice;
        - ``chunk_size`` is below 1, or is not a whole number, which raises
          :class:`~polars_online.WrongTypeError`;
        - ``load_state`` is not such a state, or was saved with other ``names``
          or ``pairs``, or on the other kind of clock (numeric or temporal), or
          with a ``group`` where this call has none, or the reverse;
        - when the query runs, a row's ``series`` is not in ``names``, or a
          clock steps below the previous row's in its group, the row named, as
          a spec refuses one with ``restart_after_step_back`` unset. A row
          before its group's saved clock is refused the same way. Under
          py-polars 1.x the error arrives inside Polars' ``ComputeError``,
          since the grid runs as a Polars source, for a ``DataFrame`` too.
          Under 2.0 it is the ``ValueError`` itself.
    ``TypeError``
        A keyword under its old name, such as ``chunk_rows``.
    ``FileNotFoundError``
        ``load_state`` is not there, or the folder of ``save_state`` is not.
    ``IsADirectoryError``
        ``save_state`` is a folder, on every system. A ``load_state`` that is
        a folder raises it too on macOS and Linux, and ``PermissionError`` on
        Windows, where opening a folder fails that way.

    .. rubric:: Examples

    .. code-block:: python

        ticks = pl.DataFrame({              # one row per tick, each series at its own times
            "symbol": ["AAA", "BBB", "AAA", "CCC", "BBB", "AAA", "AAA", "CCC"],
            "t": [0.4, 0.9, 1.3, 1.6, 2.2, 2.5, 2.8, 3.1],
            "px": [100.0, 20.0, 100.2, 50.0, 20.1, 100.1, 100.4, 49.9],
        })
        grid = po.stream.refresh_time(      # a DataFrame in, a DataFrame out
            ticks, series="symbol", names=["AAA", "BBB", "CCC"], clock="t", value="px"
        )
        for row in grid.rows():             # each point: CCC's tick completes it
            print(row)
        returns = grid.select(pl.col("^.*_value$").diff())   # what rcov and ew_cov take

    .. code-block:: text

        (1.6, 100.2, 20.0, 50.0, 2, 1, 1, 0.75)
        (3.1, 100.4, 20.1, 49.9, 2, 1, 1, 0.75)

    .. rubric:: See also

    - ``po.spec.rcov`` and ``po.spec.ew_cov``, the models a grid's returns
      feed.
    - :func:`polars_online.corr.epps_invert`, which corrects a correlation for
      the Epps effect instead.
    -
      `Series that tick at their own times
      <https://github.com/hgilde/polars-online#series-that-tick-at-their-own-times>`_,
      the README section.

    .. rubric:: Notes

    The grid is Barndorff-Nielsen, Hansen, Lunde & Shephard's refresh-time
    rule. A grid point is the first instant by which every series has ticked at
    least once since the previous point, and each series' last value there is
    taken:

    .. code-block:: text

        tau_0     = max_i (first tick of series i)
        tau_{j+1} = max_i (first tick of series i strictly after tau_j)

    Nothing is interpolated, every value in the output was observed, and the
    grid adapts to the slowest series rather than carry a stale value across an
    interval.

    The staleness caveat (their section 2.1): the output looks synchronous and
    is not. A refresh vector is treated as observed at ``time_refresh``, but
    each series' value is up to one of its own inter-tick intervals old.
    ``n_obs_<s>`` counts the ticks of each series the point folded in, of which
    it kept one. A large count is ticks the grid dropped, and the series
    holding the grid up, the one whose tick completes each point, sits near 1.

    Ties are broken by row order: "strictly after ``tau_j``" is read against
    the row sequence. So a tick carrying the same clock value as the one that
    just closed a point, but later in the frame, belongs to the next interval.
    That lets a point be emitted the moment its last series ticks, which makes
    the result chunk-invariant: feeding the input in one chunk or a thousand
    gives the same grid, since a point is a property of the ticks up to it.
    Sort the input by ``clock``, and by the order you want within a clock
    value.

    **It resumes.** ``save_state`` writes the grid's state once the input is
    fed: every group's grid, part-way through an interval or not, and its last
    clock. The state is written in full or not at all. Under a slice of the
    output (``.head(n)``), the input is read up to the tick that completed the
    last point returned and no further. So the state saved is the state after
    that tick, whatever the chunk size, as a bank's is after the rows a
    ``head(n)`` pulled. A run resumed on the input after that tick goes on
    with the next point. With ``pairs=True`` one tick can complete several
    pairs' points at once. If the slice ends among them, the state holds the
    tick's points the slice left out, and a run resumed on the input after
    that tick returns them first. The slice and the resumed run together give
    the one run's grid, wherever the slice ends. ``load_state`` goes on from
    where its run stopped, so feeding a stream in two runs gives the grid one
    run gives.
    """
    lazy = lf.lazy()
    in_schema = lazy.collect_schema()
    names = list(names)
    keep = list(keep)
    for role, col in [("series", series), ("clock", clock), ("value", value)] + (
        [("group", group)] if group is not None else []
    ):
        if col not in in_schema:
            msg = (
                f"po.stream.refresh_time: the input has no {role} column {col!r}, "
                f"only {in_schema.names()}. Set {role} to one of those columns."
            )
            raise ColumnNotFoundError(msg)
    for col in keep:
        if col not in in_schema:
            msg = (
                f"po.stream.refresh_time: keep names column {col!r}, but the input does not "
                f"have it, only {in_schema.names()}. List only those columns in keep."
            )
            raise ColumnNotFoundError(msg)
    # Every column of a chunk crosses to the extension, which cannot take a
    # `UInt128` one in.
    _refuse_uint128(in_schema, "po.stream.refresh_time", (clock,))
    # A number or a temporal column, as a spec's clock is: a boolean or a
    # column of nulls is a number (task 197, PC7's rule). Read through a
    # non-strict cast, text of digits was taken as a clock.
    clock_dtype = in_schema[clock]
    if not (
        clock_dtype.is_numeric()
        or clock_dtype.is_temporal()
        or clock_dtype in (pl.Boolean, pl.Null)
    ):
        msg = (
            f"po.stream.refresh_time: clock column {clock!r} must be numeric or temporal, "
            f"but it is {clock_dtype}. Cast it, such as with pl.col({clock!r}).cast(pl.Float64) or "
            f"pl.col({clock!r}).str.to_datetime()."
        )
        raise WrongTypeError(msg)
    # A number, as a spec's feature is: a boolean or a column of nulls too
    # (review round 4, PC7). Read through a non-strict cast, text came out
    # all null, an empty grid with nothing said, and a `Date` as its days.
    value_dtype = in_schema[value]
    if not (value_dtype.is_numeric() or value_dtype in (pl.Boolean, pl.Null)):
        msg = (
            f"po.stream.refresh_time: value column {value!r} must be numeric, "
            f"but it is {value_dtype}. Cast it, such as with pl.col({value!r}).cast(pl.Float64)."
        )
        raise WrongTypeError(msg)
    checked = _check_chunk_size(chunk_size, "po.stream.refresh_time")
    rows = checked if checked is not None else _native.default_chunk_size()
    # Read now, as a bank's `load_state` is: a plan collected twice goes on
    # from the same state, whatever the file holds by then.
    loaded = _read_state(load_state) if load_state is not None else None
    save_path = _save_path(save_state, "po.stream.refresh_time")

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

        def emit(out: pl.DataFrame) -> pl.DataFrame:
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            return out

        for chunk in lazy.collect_batches(chunk_size=rows, maintain_order=True):
            # Exactly the points still wanted: the rest of a tick's points
            # past them are held, in the state too, for the next feed.
            out = rt.feed(chunk, None if n_rows is None else n_rows - seen)
            seen += out.height
            yield from _in_morsels(emit(out))
            if n_rows is not None and seen >= n_rows:
                break
        else:
            # Points a resumed state held come out on the first feed; an
            # input of no rows gives no chunk to feed, so a frame of none
            # takes them.
            out = rt.feed(pl.DataFrame(schema=in_schema), None if n_rows is None else n_rows - seen)
            if out.height:
                yield from _in_morsels(emit(out))
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
            raise ValueError(
                f"{who}: {key} is NaN, which is no number of clock units. Set {key} to a number "
                "or a duration."
            )
        return ("inf" if x > 0 else "-inf") if math.isinf(x) else value
    return duration_text(value, who, key)


def _warn_session_gap_ignored(who: str, value: Any) -> None:
    """The :class:`~polars_online.PolarsOnlineDeprecationWarning` for a
    ``session_gap`` that is a number or a duration, which no window reads,
    pointed at the first line outside the package."""
    warnings.warn(
        f"{who}: session_gap = {value!r} is deprecated and ignored until the next major "
        "version, which refuses it: every window ends at a session change, so no window "
        "measures the clock's step across one. Leave session_gap out, or set it to "
        '"reset" to discard the windows still open at a session change rather than cut them.',
        _warnings.PolarsOnlineDeprecationWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


#: The package's directory: a warning skips its frames, so it names the
#: caller's line through any wrapper (the frame namespace's method).
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


#: The clock keywords of ``with_windows``, reserved: not output names.
_WINDOW_KEYS = ("clock", "gap_cap", "restart_after_step_back", "session", "session_gap", "group")


def _formulas(who: str, exprs: tuple[Any, ...], named: dict[str, Any]) -> list[dict[str, Any]]:
    """The call's expressions as named trees: a keyword names its expression,
    a positional one is named by its alias."""
    out: list[dict[str, Any]] = []
    for e in exprs:
        if not isinstance(e, pl.Expr):
            raise TypeError(
                f"{who}: each value passed by position must be a Polars expression (a column "
                f"computation, "
                f'such as pl.col("x") + 1) over the window operators, but one is of type '
                f"{type(e).__name__}. "
                'Pass expressions such as po.ewm_mean("x", half_life=10).alias("m"), or name '
                'each as a keyword, as in with_windows(m=po.ewm_mean("x", half_life=10)).'
            )
        tree = _formula.to_tree(e, who)
        name = e.meta.output_name()
        if name.startswith(_formula.PREFIX):
            raise ValueError(
                f"{who}: an expression has no name of its own, only its window operator's. "
                f'Name it with .alias("...") or as a keyword, as in with_windows(name=expr).'
            )
        out.append({"name": name, "tree": tree})
    for name, e in named.items():
        if name in _RENAMED_KEYWORDS:
            new = _RENAMED_KEYWORDS[name]
            raise TypeError(f"{who}: {name} was renamed {new}. Use {new} instead.")
        # Only the old name of one of this call's own keywords is refused: any
        # other old name of a spec's parameter, such as `window`, is a valid
        # output column, which was refused (task 352).
        if name in _RENAMED and _RENAMED[name] in _WINDOW_KEYS:
            raise TypeError(
                f"{who}: {name} was renamed {_RENAMED[name]}. Use {_RENAMED[name]} instead."
            )
        if not isinstance(e, pl.Expr):
            raise TypeError(
                f"{who}: the keyword {name} names an output column, so it must be a pl.Expr "
                f"over the window operators, but it is of type {type(e).__name__} ({e!r}). "
                "Pass an expression such as po.ewm_mean(...)."
            )
        out.append({"name": name, "tree": _formula.to_tree(e, who)})
    for f in out:
        if not _formula.operators(f["tree"]):
            raise ValueError(
                f"{who}: {f['name']!r} holds no window operator, so it reads only its own row. "
                "Compute it with Polars' with_columns instead."
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
    chunk_size: int | None = None,
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
    chunk_size: int | None = None,
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
    chunk_size: int | None = None,
    load_state: State | None = None,
    save_state: State | None = None,
    **named: pl.Expr,
) -> pl.LazyFrame | pl.DataFrame:
    """Add columns of formulas over the window operators, computed over a stream
    in one pass.

    Use it, as you would Polars' ``with_columns``, for features or targets
    that look back or ahead over the stream's clock, built from the operators
    of :mod:`polars_online.ops`, with one window of rows in memory, never the
    whole stream. For a formula that reads only its own row, use Polars'
    ``with_columns``. A spec also takes a formula over the operators as its
    target, computed inside the bank.

    .. rubric:: Parameters

    ``lf``
        The stream, a ``LazyFrame`` or a ``DataFrame``, in clock order across
        groups, as one stream.
    ``*exprs``
        Formulas passed by position, each named by its ``.alias()``.
    ``clock``
        The clock column. The default, ``None``, counts one unit a row.
    ``gap_cap``
        The longest step of the clock between two rows of one stretch (the
        rows between two clock breaks): a longer step is a clock break, which
        ends every window open across it. Required with a
        clock. A number of clock units, or a duration on a temporal clock: text
        such as ``"5m"``, a ``timedelta``, or an expression that reads no
        column, such as ``pl.duration(minutes=5)``.
    ``restart_after_step_back``
        By default a step back of the clock is refused. If it is set, a step
        back larger than it resets every window, and one no larger is still
        refused. In the units ``gap_cap`` takes. Needs ``clock``.
    ``session``
        A column whose change of value ends every window open across it.
    ``session_gap``
        ``"reset"``, to discard the windows a session change leaves open rather
        than cut them. Needs ``session``. A number or a duration is deprecated
        here, since no window reads it (Notes).
    ``group``
        A column whose every value has its own clock, sessions and windows. A
        window looking ahead under ``group`` needs ``clock``.
    ``like``
        A spec to take the clock settings from, all six above, in place of
        giving them. With them it takes the rule for the rows the spec learns
        from: a window looking ahead over a row the spec would skip is null
        (Notes). Default ``None``.
    ``chunk_size``
        The rows read at a time, at least 1. The default, ``None``, is the
        library's default chunk size. It changes no output.
    ``load_state``
        The path of a state this call saved, to go on from. It is read when the
        query is built. Default ``None``: start with no state loaded.
    ``save_state``
        The path of a file to save the state to once the input is fed, or the
        slice of the output is. Default ``None``: save nothing (Notes).
    ``**named``
        Formulas by keyword, each named by its keyword. The six clock keywords
        (``clock`` to ``group``) are the call's clock settings, not output
        names: to add a column named
        ``session``, name its expression with ``.alias("session")`` and pass it
        by position.

    A formula is element-wise Polars over the operators: columns, literals,
    arithmetic, negation, comparisons, ``&`` and ``|``, ``log``, ``exp``,
    ``abs``, ``sqrt``, ``pow``, ``clip``, ``fill_null``, ``is_null``,
    ``is_not_null``, ``when/then/otherwise``, ``cast`` and ``alias``. Each
    formula must hold at least one operator. A ``shift``, a cumulative or
    rolling function, ``over`` or an aggregation would depend on the chunking,
    and is refused by name when the query is built.

    .. rubric:: Returns

    The kind of frame it was given: the input's columns, unchanged and in
    order, then a column for each formula. Rows leave in input order, each once
    every window over it has resolved. A row whose window has not passed when
    the input ends gives null, unless ``save_state`` keeps it for the next run.

    .. rubric:: Raises

    ``TypeError``
        When the query is built: a value that is not an expression, an
        expression given to ``clock``, ``session`` or ``group``, a ``gap_cap``,
        ``restart_after_step_back`` or ``session_gap`` expression that is not
        one duration reading no column, a keyword under its old name, or a
        ``like`` that is not a spec.
    :class:`~polars_online.RefusedValueError`
        A clock keyword given beside ``like=``. It is both a ``ValueError`` and
        a ``TypeError``.
    :class:`~polars_online.WrongTypeError`
        When the query is built, a window operator's input is not a number,
        or an increment's is neither a number nor temporal; or a column, read
        or not, is ``UInt128``, which polars-online cannot read (the message
        names a cast to make). It is both a ``TypeError`` and a
        ``ValueError``.
    ``ValueError``
        When the query is built:

        - a formula, a clock policy or a column that cannot run, such as a
          window looking ahead under ``group`` with no ``clock``, or
          ``half_life=float("inf")`` with no ``window_size`` for a mean, a
          variance or a standard deviation;
        - a formula that holds no window operator, or a formula by position
          that has no name of its own;
        - a ``session_gap`` without ``session``, or a
          ``restart_after_step_back`` without ``clock``, which would do
          nothing;
        - an output name that collides, or ``chunk_size`` below 1;
        - a ``load_state`` that another call saved.

        And when the query runs: a step back of the clock, naming the row, or a
        state resumed on another input (Notes). Under py-polars 1.x these
        arrive inside Polars' ``ComputeError``, since the rows come from a
        Python source. Under 2.0 each is the ``ValueError`` itself.
    ``FileNotFoundError``
        ``load_state`` is not there, or the folder of ``save_state`` is not.
    ``IsADirectoryError``
        ``save_state`` is a folder, on every system. A ``load_state`` that is
        a folder raises it too on macOS and Linux, and ``PermissionError`` on
        Windows, where opening a folder fails that way.
    :class:`~polars_online.PolarsOnlineDeprecationWarning`
        A warning, not an error: a ``session_gap`` that is a number or a
        duration, which no window reads.
    :class:`~polars_online.UnstableWarning`
        A warning, under ``POLARS_ONLINE_WARN_UNSTABLE=1``: a call with
        ``load_state`` or ``save_state``, whose file format is unstable.
    :class:`~polars_online.OrderNotGuaranteedWarning`
        A warning: the query before this call does not keep its rows in order,
        while a window is taken over rows in row order.
    :class:`~polars_online.ConsumedSourceWarning`
        A warning: a run that reads a source written in Python gave no rows,
        the shape of a stream that gives its rows once and was spent by an
        earlier run.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 5.0, 6.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = po.stream.with_windows(
            rows,
            trend=pl.col("x") - po.ewm_mean("x", half_life=1.0),   # x above its trailing mean
            ahead=po.ewm_sum("x", half_life=float("inf"), window_size=1.0, reverse=True),
            clock="t", gap_cap=2.0,                    # the gap from 2 to 5 is a clock break
        )
        print(out["trend"].round(4).to_list())   # 0 at row 3: the windows start over
        print(out["ahead"].to_list())            # next unit's sum: null if cut, or open at the end

    .. code-block:: text

        [0.0, 0.5, 0.75, 0.0, 0.5]
        [2.0, 3.0, None, None, None]

    Over ``trades``, quotes with a ``mid`` and trades between them with a
    ``side``, a ``quantity`` and a ``price``, a trailing mean of the mid and
    forward VWAPs of all trades and of the buys. This code reads ``trades``
    from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        notional = pl.col("price") * pl.col("quantity")
        buys = pl.when(pl.col("side") == "buy")
        ahead = {"half_life": "10s", "window_size": "1m", "reverse": True}
        out = po.stream.with_windows(
            trades,
            mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
            fwd_vwap=po.ewm_sum(notional, half_life="10s", window_size="1m", reverse=True)
                     / po.ewm_sum("quantity", half_life="10s", window_size="1m", reverse=True),
            fwd_buy_vwap=po.ewm_sum(buys.then(notional), **ahead)
                         / po.ewm_sum(buys.then("quantity"), **ahead),
            clock="ts", gap_cap="5m", group="symbol",
        )
        print(out.height, out.columns[-3:], out["fwd_vwap"].count())

    .. code-block:: text

        3000 ['mid_trend', 'fwd_vwap', 'fwd_buy_vwap'] 2890

    .. rubric:: See also

    - :mod:`polars_online.ops`, the operators, and the tables of ``closed`` and
      ``partial``.
    - ``lf.online.with_windows`` and ``df.online.with_windows``, the same call
      on a frame's ``online`` namespace.
    -
      `Make features that look back over the stream
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/make-window-features.md>`_,
      `Make a target that looks ahead
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/make-a-target-that-looks-ahead.md>`_
      and `Saving and resuming a window run
      <https://github.com/hgilde/polars-online/blob/main/docs/WINDOWS.md#saving-and-resuming-a-window-run>`_,
      the guides.

    .. rubric:: Notes

    The operators are stateful kernels the core computes, each distinct one
    once. The formula around them is element-wise Polars, evaluated on each
    chunk the call emits. The formula is kept as a compact tree of the
    library's own, in the query and in a saved state, so a state saved under one
    Polars version loads under another: Polars' own serialized form of an
    expression changes between versions.

    **The clock.** A window is measured on the clock with each step capped at
    ``gap_cap``. A gap longer than ``gap_cap`` or a session change ends every
    window open across it, under each operator's ``partial``. A reset discards
    them: each gives null, and its row stays in the output. Only a window
    looking ahead is still open at a session change, so
    ``session_gap="reset"`` changes those windows and nothing else. Since
    every window ends at a session change, no window measures the clock's step
    across one, and the number a spec takes there would change nothing. So a
    session needs no ``session_gap`` here. A window looking ahead under
    ``closed="right"`` or ``"both"`` whose far edge is the last row before a
    cut or a reset holds every row it covers, so it is whole, and gives its
    value under any ``partial``.

    The clock is also read in input order, under the same settings. So a step
    back there is refused or, past ``restart_after_step_back``, resets every
    group, and a gap there ends every group's windows. With ``group`` a session
    is each group's, read on its own clock, so a clock that starts over with a
    session is a step back on the stream's. Without ``group`` the stream is the
    one group. A group silent past ``gap_cap`` is cut as the stream's clock
    passes the cap, before its next row can say its session changed, so a
    ``session_gap="reset"`` there discards only what is still open.

    Under ``like=spec``, a window looking ahead over a row the spec would skip
    is null, as the model never learns its target. The spec skips a row whose
    feature or weight it cannot use: null, NaN, infinite or past ``1e100`` in
    magnitude, each of which counts as null.

    **The rows' order and memory.** A backward window under ``closed="right"``
    or ``"both"`` waits for the next distinct stamp (a clock value), and a
    look-ahead for its window to pass, so the output trails the input by
    one clock step for a backward window and by the longest window for a
    look-ahead. A
    group that falls silent holds every later row for at most ``gap_cap`` of
    the stream's time. Past that its next row is certain to open with a gap
    past the cap, so its windows end then. Only a clock column has a cap, so a
    look-ahead under ``group`` needs one, and is refused without it: each
    group's clock would count only its own rows, and a silent group would hold
    every later row to the end of the input. The rows a look-ahead holds are
    kept as the input's own chunks, not copied, so the memory is one window of
    input whatever its width. The kernels' own sums are one window of the rows
    that count in them.

    **It resumes.** ``save_state`` writes the kernels' sums and the rows still
    waiting, which the next run, given ``load_state``, emits first. Feeding a
    stream in two runs gives what one run gives. Any resumed run refuses by
    name an input whose first row is at the last stamp the state read, unless
    that row starts a new session. A file boundary inside a tied stamp cannot
    be told from an input that repeats rows the state read, which would come
    out twice. So cut a stream into files between stamps. Without a clock
    column there is no stamp to compare, and a repeated row is not caught.

    .. warning::
        The state file's format is considered **unstable**: it may change in
        any release without that counting as a breaking change, and a state one
        release saved need not load in the next. So is the formula tree it
        holds. Under ``POLARS_ONLINE_WARN_UNSTABLE=1``, a call with
        ``load_state`` or ``save_state`` issues
        :class:`polars_online.UnstableWarning`.

    Under a slice of the output (``.head(n)``) the input is read only up to the
    row that resolved the *n*-th row, and the state records how many rows of
    the input were consumed so far. A run resumed with ``load_state`` on the
    *same input*, unsliced, skips those rows and goes on. So any chain of
    sliced runs gives what one run gives, whatever the chunk size. Such a state
    knows its input by its first row's clock and session, and by the last rows
    it read: the rows it holds that are the input's, or the last row alone
    where it holds none. An input whose first row differs is the next file when
    the clock settings take that row as a step forward or a new start, which is
    a step back past ``restart_after_step_back`` or a new session. The run then
    skips nothing and first returns the rows the state held. Besides the input
    at the last stamp read, a state saved under a slice refuses by name:

    - an input that steps back where the clock settings refuse it: the same input
      sliced by hand, or an overlapping file (a hand slice that starts after
      the last clock read is taken as the next file);
    - without a clock column, an input that does not begin with a new session;
    - an input that starts as the saved input did but differs where the state
      was cut, and one shorter than the rows consumed.

    A clock that starts over at the same stamp each day needs a session column
    for the next day's file to differ from the saved input at its first row.
    Without a clock column a row-count clock steps forward at every row. So the
    next file is one that begins with a new session (under ``group``, a new
    session of a group the state has read), and not with the saved input's
    first session, which marks the saved input itself. Without a session column
    either, there is no next file: resume on the same input. An input whose
    rows match the state's where it was cut is taken as the same input, so
    resume a stream whose rows can repeat (a daily grid with the same values)
    on the same input only. A state resumes only the call that saved it, on the
    same kind of clock.
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
        if isinstance(value, pl.Expr) and key in ("clock", "session", "group"):
            example = {"clock": "ts", "session": "day", "group": "symbol"}[key]
            raise TypeError(
                f"{who}: {key} takes a column name, but it is given an expression (a column "
                f'computation, such as pl.col("x") + 1). Give the column\'s name, as '
                f'{key}="{example}", or, for an output column named {key!r}, name the '
                f"expression with .alias({key!r}) and pass it by position."
            )
        if isinstance(value, pl.Expr):
            # gap_cap, restart_after_step_back and session_gap: each a span
            # of the clock, which takes a duration expression as a spec's
            # does (task 352), read here as its text.
            try:
                given[key] = duration_text(value, who, key)
            except (TypeError, ValueError):
                raise TypeError(
                    f"{who}: {key} takes a number or a duration, but it is given an expression "
                    f"that is not one duration reading no column. Give {key} as "
                    f'pl.duration(minutes=5) or "5m", or, for an output column named {key!r}, '
                    f"name the expression with .alias({key!r}) and pass it by position."
                ) from None
    formulas = _formulas(who, exprs, named)
    config: dict[str, Any] = {"formulas": formulas}
    if like is not None:
        if not (isinstance(like, dict) and isinstance(like.get("features"), list)):
            raise TypeError(
                f"{who}: like must be a spec, such as po.spec.ewridge(...), but it is "
                f"{like!r}. Pass a spec."
            )
        clash = [k for k, v in given.items() if v is not None]
        if clash:
            raise RefusedValueError(
                f'po.stream.with_windows: like= takes the clock settings from spec "'
                f'{like.get("name")}", but '
                f"{_and_list(clash)} {'is' if len(clash) == 1 else 'are'} given too. Leave out "
                f"{_and_list(clash)}, or like=."
            )
        # A hand-written dict with a clock key under its old name would be
        # read as not setting it (review R1, F1): refused naming the new one,
        # or, renamed after 1.0, read as it with a warning.
        like = forward_deprecated(
            f"{who}: like= spec {like.get('name')!r}", like, _warnings._DEPRECATED
        )
        if old := [k for k in like if k in _RENAMED]:
            raise TypeError(
                f"{who}: like= spec {like.get('name')!r}: {old[0]} was renamed "
                f"{_RENAMED[old[0]]}. Use {_RENAMED[old[0]]} in the spec."
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
    # Before the empty frame below crosses to the extension, which cannot
    # take a `UInt128` column in.
    _refuse_uint128(in_schema, who, (policy.get("clock"),))
    rows = chunk_size if chunk_size is not None else _native.default_chunk_size()
    if rows < 1:
        msg = (
            f"{who}: chunk_size must be at least 1, but it is {rows}. Set chunk_size to 1 or "
            "more, or leave it out."
        )
        raise ValueError(msg)
    # The two plan checks a bank makes of its input, made here of this one's:
    # a bank after this sees only this source, and not the plan beneath it.
    plan_text = _plan_text(lazy)
    _warn_if_order_unspecified(
        lazy, who, plan_text, reads="a window is taken over rows in row order", before="the windows"
    )
    python_scan = _is_python_scan(lazy, plan_text)
    if load_state is not None or save_state is not None:
        warn_unstable(f"{who}'s state file (load_state, save_state)")
    # Read now, as a bank's `load_state` is: a plan collected twice goes on
    # from the same state, whatever the file holds by then.
    loaded = _read_state(load_state) if load_state is not None else None
    save_path = _save_path(save_state, who)
    empty = pl.DataFrame(schema=in_schema)

    def build() -> _native.Windows:
        if loaded is not None:
            return _native.Windows.load_bytes(loaded, config_json, empty)
        return _native.Windows(config_json, empty)

    # Built once now, so a formula or a column that cannot run is reported
    # while the plan is built, as polars reports its own errors.
    first = build()
    # Once the call is known to run, so a gap the run refuses is refused in
    # its own words first (task 278, the user's decision of 2026-10-25).
    if like is None and given["session_gap"] is not None and given["session_gap"] != "reset":
        _warn_session_gap_ignored(who, given["session_gap"])
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
            yield from _in_morsels(done(out))
            if n_rows is not None and seen >= n_rows:
                sliced = True
                break
        if not sliced and save_path is None:
            # The input's end: what still waits for a window is unresolved.
            out = w.finish()
            if n_rows is not None:
                out = out.head(n_rows - seen)
            yield from _in_morsels(done(out))
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
                    f"{who}: the query gave no rows, and it reads a source written in Python, "
                    "such as `pl.scan_arrow_c_stream` over a DuckDB relation, a pyarrow reader or "
                    "anything with `__arrow_c_stream__`, which gives its rows once, so a second "
                    "run of the query reads nothing. Build the query again for each run, calling "
                    "`pl.scan_arrow_c_stream(...)` inside the loop, or, if the source "
                    "really is empty, silence this warning with "
                    '`warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning)`.'
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
