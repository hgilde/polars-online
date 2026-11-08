"""A model bank inside a Polars query: ``lf.online.fit_predict(specs)``.

A query in, a query out. When the query runs, its rows go through a bank
that starts with nothing learned, one chunk at a time, so memory grows with
a chunk and the bank's state, never with the length of the stream. The bank
is a source of the query, the kind of step Polars reads chunks of rows from,
and every step after it, such as a filter, a ``select``, a join or writing
the result to a file, is Polars' own. A filter after the bank never changes
what the bank learns from. To keep the bank from learning rows, give them
weight 0 in the spec's ``weight`` column.

The query is pure: every run starts from the same state (the specs', or
``load_state``'s, read when the query is built), so running it twice gives
the same frame. ``save_state`` writes the state the run ends in, after the
last row the query gave the bank, atomically. Because the query is pure,
that write gives the same bytes every time. Polars runs a query's source
once per run, and twice, at the same time, when one query uses it twice (a
self-join, ``pl.concat``, or ``pl.collect_all`` of two queries), and every
run ends in the same state.

``df.online.fit_predict(specs)`` does the same for a frame in memory, as
``ModelBank(specs).fit_predict(df)`` in one call. ``online.unnest(specs)``
turns each spec's output column into one column per field, with the
``coef`` list as one named column per coefficient
(:func:`polars_online.spec.coef_fields`). ``online.with_windows(*exprs, **named)``
adds exponentially weighted means, sums, rates, variances and quantiles,
looking back or ahead, as columns
(:func:`polars_online.stream.with_windows`), so a query can add them and
fit on them in one chain. Both namespaces are attached when the package is
imported, which no type checker can see. So :func:`~polars_online.fit_predict`,
:func:`~polars_online.predict`, :func:`~polars_online.unnest` and
:func:`polars_online.stream.with_windows` are the same calls with the query
or the frame as the first argument, visibly typed.
"""

from __future__ import annotations

import itertools
import json
import numbers
import operator
import os
import re
import sys
import threading
import warnings
from collections.abc import Callable, Iterable, Iterator
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _polars_online as _native
from polars_online._bank import ModelBank, _check_threads, _refuse_uint128
from polars_online._errors import WrongTypeError
from polars_online._spec import _renamed_keywords, coef_fields, output_index, target_columns
from polars_online._warnings import PolarsOnlineWarning

if TYPE_CHECKING:
    from polars_online._duration import Duration

__all__ = [
    "ConsumedSourceWarning",
    "DataFrameOnlineNamespace",
    "LazyFrameOnlineNamespace",
    "OrderNotGuaranteedWarning",
    "ReadinessWarning",
    "fit_predict",
    "predict",
    "unnest",
]

Specs = Iterable[dict[str, Any]]
State = str | os.PathLike[str]

# The input columns a spec reads, by key (crates/online-polars/src/bank.rs
# `extract`); the rest of the frame is carried through.
_SCALAR_COLUMN_KEYS = ("clock", "session", "weight", "group")


def _bank(
    specs: Specs | None,
    load_state: State | ModelBank | None,
    what: str,
    threads: int | None = None,
) -> Callable[[], ModelBank]:
    """How to make the bank a plan starts from, from the specs, a state file
    or a bank.

    A state file is read here, once, when the plan is built -- the plan
    carries the bytes, as ``df.lazy()`` carries the frame -- and each
    execution deserialises them. Read at run time instead, a plan collected
    twice would not be the same frame if the file changed in between, and
    ``load_state=p, save_state=p`` used twice in one query would race the
    second run's load against the first run's write. A bank is snapshotted
    the same way, so the plan learns on a copy and the caller's bank does not
    move (task 244, M-25: ``predict(bank=)`` already took a bank).

    ``threads`` is the bank's pool (task 253); a copied bank keeps its own
    count unless one is given.
    """
    threads = _check_threads(threads)
    if isinstance(load_state, ModelBank):
        state = load_state.save_bytes()
        count = load_state.threads if threads is None else threads
        return lambda: ModelBank.load_bytes(state, specs, threads=count)
    if load_state is not None:
        state = _read_state(load_state)
        return lambda: ModelBank.load_bytes(state, specs, threads=threads)
    if specs is None:
        who = what
        msg = (
            f"{who}: neither specs nor load_state is given, so there is no spec to fit. Pass "
            "specs, or load_state= to take them from a saved state."
        )
        raise ValueError(msg)
    return lambda: ModelBank(specs, threads=threads)


def _check_chunk_size(chunk_size: Any, who: str) -> int | None:
    """``chunk_size`` as an int, or ``None`` left out, once it is checked to
    be a whole number of at least 1: any whole number, numpy's integers
    included, but not a bool. ``who`` is the caller's name for the messages."""
    if chunk_size is None:
        return None
    wrong_type = isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral)
    if wrong_type:
        msg = (
            f"{who}: chunk_size must be a whole number of at least 1, but it is {chunk_size!r}. "
            "Set chunk_size to a whole number, such as 10_000, or leave it out."
        )
        raise WrongTypeError(msg)
    if operator.index(chunk_size) < 1:
        msg = (
            f"{who}: chunk_size must be at least 1, but it is {chunk_size}. Set chunk_size to "
            "1 or more, or leave it out."
        )
        raise ValueError(msg)
    return operator.index(chunk_size)


def _in_chunks(
    df: pl.DataFrame,
    chunk_size: int | None,
    run: Callable[[pl.DataFrame, int], pl.DataFrame],
) -> pl.DataFrame:
    """``run`` over ``df`` in ``chunk_size`` slices, each fed as rows of the
    whole frame so an error names the frame's row; the frame whole without
    ``chunk_size``, and an empty frame as itself, which has no slice."""
    if chunk_size is None or df.height == 0:
        return run(df, 0)
    outs = []
    fed = 0
    for chunk in df.iter_slices(n_rows=chunk_size):
        outs.append(run(chunk, fed))
        fed += chunk.height
    return pl.concat(outs)


def _read_state(state: State) -> bytes:
    with open(os.fspath(state), "rb") as f:
        return f.read()


def _save_path(save_state: State | None, who: str) -> str | None:
    """The ``save_state`` path, checked while the plan is built: a directory
    that is not there, and a path that is itself a directory, are reported
    now, not after the stream (review 2026-10-05, YB7: the second failed once
    every row had been read). Raised here, so the error is the same on every
    OS, where opening a directory raises a different one on Windows."""
    if save_state is None:
        return None
    path = os.fspath(save_state)
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        msg = (
            f"{who}: save_state names a file in the folder {parent!r}, which does not exist. "
            "Make the folder, or name a file in a folder that does."
        )
        raise FileNotFoundError(msg)
    if os.path.isdir(path):
        msg = f"{who}: save_state {path!r} is a folder, not a file. Name a file in it."
        raise IsADirectoryError(msg)
    return path


_WRITE: dict[str, str] = {
    "parquet": "write_parquet",
    "ipc": "write_ipc",
    "ndjson": "write_ndjson",
}


def _closed_path(
    closed_groups: State | None, specs: Iterable[dict[str, Any]], who: str
) -> str | None:
    """The ``closed_groups`` sidecar path, checked while the plan is built:
    a directory that is not there, an extension that names no format, and a
    bank with nothing to close are all reported now (E54)."""
    if closed_groups is None:
        return None
    if not isinstance(closed_groups, (str, os.PathLike)):
        msg = (
            f"{who}: closed_groups must be a path, but its type is "
            f"{type(closed_groups).__name__}. Pass the file to write the closed groups to, "
            'such as "closed.parquet", or leave closed_groups out.'
        )
        raise TypeError(msg)
    path = os.fspath(closed_groups)
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        msg = (
            f"{who}: closed_groups names a file in the folder {parent!r}, which does not exist. "
            "Make the folder, or name a file in a folder that does."
        )
        raise FileNotFoundError(msg)
    # Raises ValueError naming the extensions it knows.
    _native.format_of_path(path, f"{who}: closed_groups")
    if not any(spec.get("group_close") for spec in specs):
        msg = (
            f"{who}: closed_groups names a file, but no spec closes groups, so the file would "
            'stay empty. Add group_close="monotone" or "session" to the spec whose closed '
            "groups you want, or leave closed_groups out."
        )
        raise ValueError(msg)
    return path


def _check_closed_format(path: str | None, schema: pl.DataFrame, who: str) -> None:
    """A ``closed_groups`` file the closed rows' columns can be written to,
    checked before the fit: CSV carries a list column only as JSON text,
    and a nested column that is not such a list not at all (the error
    names it)."""
    if path is not None and _native.format_of_path(path) == "csv":
        _native.csv_flat(schema, f'{who}: closed_groups "{path}"')


def _write_closed(path: str, frames: list[pl.DataFrame], schema: pl.DataFrame) -> None:
    """The drained closed rows as one file, in the format the extension
    names. An empty run writes the empty frame with the schema, as an empty
    output does. A CSV goes through the command line's own writer: every
    closed row has a list column, which Polars' CSV writer refuses, and the
    runner writes each list as JSON text (task 269, D-1), so the file is the
    CLI's to the byte."""
    df = pl.concat(frames) if frames else schema
    fmt = _native.format_of_path(path)
    if fmt == "csv":
        _native.write_frame(df, path)
    else:
        getattr(df, _WRITE[fmt])(path)


def _explain_kwargs(specs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """`explain_name` / `explain_detail` for `register_io_source`, when the
    installed polars takes them (py-polars 2.0 added them; 1.x has neither).

    Without them a plan holding a bank shows as `PYTHON SCAN []` in
    `LazyFrame.explain()`, which says nothing about which models are in it.
    Passed by name and only when supported, so this is additive: on a polars
    without them the plan is exactly what it was.
    """
    import inspect

    try:
        takes = inspect.signature(register_io_source).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C or wrapped callable
        return {}
    if "explain_name" not in takes:
        return {}
    names = [str(spec.get("name", "?")) for spec in specs]
    detail = ", ".join(names)
    return _explain_named(f"{len(names)} spec(s): {detail}" if names else "no specs")


def _explain_named(detail: str) -> dict[str, Any]:
    """:func:`_explain_kwargs` for any of this package's sources: the name
    ``polars-online`` and ``detail``, when the installed polars takes them."""
    import inspect

    try:
        takes = inspect.signature(register_io_source).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C or wrapped callable
        return {}
    if "explain_name" not in takes:
        return {}
    return {"explain_name": "polars-online", "explain_detail": detail}


def _spec_columns(specs: Iterable[dict[str, Any]]) -> set[str]:
    """Every column the specs read, which a plan must keep: a table target's
    column and its reference among them (review 2026-09-26, D1/F1: a table
    was added to the set as a dict, and every LazyFrame path raised)."""
    cols: set[str] = set()
    for spec in specs:
        for t in spec.get("targets") or ():
            cols.update(target_columns(t))
        cols.update(spec.get("features") or ())
        for key in _SCALAR_COLUMN_KEYS:
            if spec.get(key) is not None:
                cols.add(spec[key])
        # One label-time column, or one per target.
        label = spec.get("label_clock")
        if label is not None:
            cols.update([label] if isinstance(label, str) else label)
    return cols


class OrderNotGuaranteedWarning(PolarsOnlineWarning, UserWarning):
    """A warning that Polars may change the row order of a query a bank or a window run reads.

    An online model learns in row order, so the order a query delivers is
    part of the model. :meth:`ModelBank.fit`,
    :meth:`ModelBank.fit_predict_batches`, ``lf.online.fit_predict``,
    ``lf.online.predict``, and :func:`polars_online.stream.with_windows` with
    its namespace form ``lf.online.with_windows`` have Polars run a query a
    chunk at a time. Each issues this warning when the query has a step
    whose row order Polars may then change. Fix the step it names, or, for
    a query whose order is fixed by other means, filter the warning out
    with
    ``warnings.simplefilter("ignore", polars_online.OrderNotGuaranteedWarning)``.

    The warning names each such step and its fix:

    .. list-table::
       :header-rows: 1
       :widths: 35 65

       * - the step
         - its fix
       * - a ``join``
         - ``maintain_order="left"``
       * - a ``group_by``
         - ``maintain_order=True``
       * - a ``sort`` by several keys
         - ``maintain_order=True``: without it, rows with equal keys come out
           in no particular order
       * - a ``unique``
         - a ``sort`` after it, since Polars does not honor its
           ``maintain_order`` when it runs a query a chunk at a time
       * - any of them
         - a sort before the bank or the window run

    When Polars runs the query a chunk at a time, such a step without an
    order guarantee delivers a different stream than ``lf.collect()``
    gives, and the order may differ between runs (see Notes). So a query
    checked by running it and reading its result learns something else when
    a bank is fed from it.

    :meth:`ModelBank.fit` does not issue it when every spec keeps running
    sums with no decay: ``ewridge`` or ``rls`` at ``lam=1.0`` or
    ``half_life=inf``. Each such spec must also have none of the settings
    that make a fit depend on the row order, such as a ``window_size``, a
    session, a weight column, an embargo, a drift reset or a diagnostic
    (the full list is in :meth:`ModelBank.fit`'s Notes). Those sums
    commute, so the state after that fit is the same whatever the order,
    to rounding, and ``fit`` keeps only the state. Every other
    call and spec gets the warning:

    - ``fit_predict_batches`` and ``lf.online.fit_predict`` over the same
      specs, whose predictions are out-of-sample and so move with the order
      even where the coefficients do not;
    - every other model, whose step at each row does not commute: ``sgd``, ``pa``,
      ``ftrl``, ``quantile`` and a reweighting ``huber`` all differ
      materially with no decay at all;
    - ``lasso``, which selects its penalty by an out-of-sample error.

    So a spec with no half-life is not by itself one whose output ignores
    the row order. A spec setting the check does not recognize counts as
    unsafe, so a setting added to the library later cannot quietly become
    exempt.

    .. rubric:: Examples

    This code reads ``ticks.parquet`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        import warnings

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        sides = pl.scan_parquet("ticks.parquet").select("t", "x0", "x1")
        # A join with no maintain_order: Polars may deliver its rows in any order.
        rows = pl.scan_parquet("ticks.parquet").select("t", "y").join(sides, on="t")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows.online.fit_predict([spec])
        print([type(w.message).__name__ for w in caught])

    .. code-block:: text

        ['OrderNotGuaranteedWarning']

    .. rubric:: See also

    - `Put rows in clock order
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/put-rows-in-clock-order.md>`_,
      how to give a bank each group's rows in order, and how to read this
      warning.
    - :class:`PolarsOnlineWarning`, the base class every warning of this
      library shares.

    .. rubric:: Notes

    Measured on 200,000 rows, ``collect()`` kept the input order and
    ``collect_batches()`` did not. Over 200 rows in two orders, the state an
    order-free ``fit`` leaves differed by 6.7e-16 for ``ewridge`` and 1.4e-15
    for ``rls``, and the predictions of ``fit_predict_batches`` by up to
    0.087 for ``ewridge`` and 0.65 for ``rls``.

    The check is best effort. The query's ``explain`` text is read first.
    Only when it names a join, an aggregation, a ``unique`` or a sort is the
    query read through ``LazyFrame.serialize``, a format Polars has
    deprecated, and walked. A query Polars cannot serialize is let through
    without a warning. Polars 1.44.2 serializes a query that already holds a
    bank, its Python source included, so a join stacked after a bank is
    reported there. A Polars that cannot serialize it says nothing about it.
    The check never fails a run.
    """


class ReadinessWarning(PolarsOnlineWarning, UserWarning):
    """A warning that a model is not ready, or may never become ready.

    It points once at a state every row already carries in
    ``withheld_reason``, ``settled_frac`` and ``support_coef``, so you can
    act before reading the output. Where the finding is intended, filter it
    out with ``warnings.simplefilter("ignore", polars_online.ReadinessWarning)``.
    `Is it ready?
    <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready>`_
    sets each finding beside the gates that withhold a prediction.

    The learning calls issue it: :meth:`ModelBank.fit`,
    :meth:`ModelBank.fit_predict`, :meth:`ModelBank.fit_predict_batches` and
    ``lf.online.fit_predict``. ``predict`` never does. Apart from the two
    findings about a horizon, it is issued once per spec and group. Its
    message begins with the spec, and the group where the spec has one, as
    ``spec "m" group "g":``. Under ``group_close="session"`` each session
    span is a stream of its own, so the warning is issued once per span,
    which the message names by its session value, as
    ``spec "m" group "g" session "a":`` (``session null`` for a null value).
    Two spans with the same finding then give two texts, and Python's
    default filter, which shows a text once per line of code, shows both.

    It reports five findings:

    - **A coefficient the ridge determined more than the data did**
      (``support_coef`` below 0.5), named, on a row the gates let through.
      The features do not determine such a coefficient: a duplicated or
      constant column, or a ridge as large as the feature's variance does
      this. While the columns stay collinear, the predictions do not depend
      on the split. The split among such
      columns is arbitrary, and it moves the moment the collinearity breaks.
    - **The ``max_error_inflation`` gate has not been met.** The gate has withheld every row
      for a further half-life, and at the row rate so far ``error_inflation``
      stays above ``max_error_inflation``, so predictions stay withheld
      unless the rows come faster. The message says how far, and gives a
      half-life above which the gate is met, or a looser ratio. Where the
      model has not fit at all, its ``weight_sum`` short of a row per
      coefficient, the message gives where ``weight_sum`` tops out instead.
      A ``kalman`` never issues it, since its gate reads each row's own
      value, which a row nearer the design's center reads lower.
    - **A** ``min_weight`` **has not been met.** The floor has withheld
      every row for a further half-life, and the weight a target reads tops
      out below its ``min_weight``. The message names that ceiling
      (``weight_sum_settled`` in :meth:`ModelBank.summary`), and the
      ``min_weight`` or ``half_life`` that meets it. A target that no row
      with a positive weight has given a value has no rate to project from,
      so its message says that instead. A target whose own weight is below
      ``weight_sum`` has not been learned from every row, and its weight
      falls on each row without it. Its message says so, with the ceiling
      the floor would have if every row carried the target, where that
      ceiling is below the floor.
    - **A diagnostic has no horizon.** A spec with a ``clock`` and an
      ``embargo`` but no ``horizon_rows`` has a diagnostic that reads the
      target's horizon (``emit_calibration``, ``emit_breaks``,
      ``emit_specification`` or ``emit_robust_se``). There ``embargo`` is in
      clock units, so the horizon is 0, and a target that looks ahead flags
      the test on most rows. The message names ``horizon_rows``, once per
      diagnostic, on the bank's first learning call.
    - **The Newey-West lags were capped at 64.** Without a clock,
      ``embargo`` is in rows and gives the horizon. A diagnostic that reads
      the horizon takes twice the horizon in Newey-West lags, and at most 64
      when they are read from ``embargo``. The
      message gives the lags asked for, and names ``horizon_rows`` (to take
      them all) and ``robust_se_lags``. It begins with the spec alone, and
      comes on the bank's first learning call.

    Offline, :func:`polars_online.gram.lasso_path` issues it once per call
    for the penalties whose coordinate descent ran out of its ``max_iter``
    sweeps before it met its stopping rule, naming them. Each such
    penalty's row of the path is where its descent stopped. The ``lasso``
    model counts the same event in
    :meth:`ModelBank.solve_failures`.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        import warnings

        # 400 rows a step apart, with a half-life of 20 steps: the weight tops out near 29.
        spec = po.spec.ewridge(
            "ridge", targets=["y"], features=["x0", "x1"], clock="t", half_life=20.0,
            gap_cap=5.0, min_weight=50.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            po.ModelBank([spec]).fit_predict(df)
        for w in caught:
            print(type(w.message).__name__, str(w.message).split(" has not")[0])

    .. code-block:: text

        ReadinessWarning spec "ridge": min_weight = 50

    .. rubric:: See also

    - `Is it ready?
      <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#is-it-ready>`_,
      the gates and what each finding means for the output.
    - :meth:`ModelBank.summary`, whose ``weight_sum_settled`` is the
      ceiling the ``min_weight`` finding names.
    - :class:`PolarsOnlineWarning`, the base class every warning of this
      library shares.

    .. rubric:: Notes

    The first three findings judge a model only once its ``settled_frac``
    reaches 0.95, the model within 5% of its steady state. Before then, its
    early rows still move what the warning would say, and a warning cannot
    be taken back: the first rows' noisy feature variances, for example,
    can read a share of the data below 0.5 that later rows do not. Where
    nothing decays, ``settled_frac`` is null, so the model is never judged,
    and its ``support_coef``, ``withheld_reason`` and
    :meth:`ModelBank.summary` carry what it reads.

    The noise-gate and ``min_weight`` findings wait a half-life during which
    their gate withholds every row, on the clock the learned rows have
    covered. A row the gate lets through starts the wait again. The wait is
    in the state file: a bank saved and loaded inside it goes on waiting,
    and issues its warning on the row an unbroken stream would. Under
    ``embargo`` both findings read the ``settled_frac`` of the rows the
    model has learned from, on the clock its weight has covered. The row's
    own ``settled_frac`` counts the held rows' clock too.

    Both findings project a ceiling from the rows so far, and rows that come
    faster can beat it, so neither says the gate is unmet for good. At the
    rows' spacing ``d`` and weight so far, the ``min_weight`` ceiling is
    ``1 / (1 - 2 ** (-d / half_life))`` times that weight. With a clock,
    ``d`` is known only from the rows, so the ceiling cannot be read from the spec before
    the rows arrive. A warning on a row of
    weight 0, or on rows too uneven to project (one row far heavier than the
    rest, or a stream whose weight is still its first row's), gives no
    figure. A target the rows stopped carrying gets no figure where rows
    that each carried it would meet the floor, since its weight falls
    toward 0 once the rows stop carrying it.
    """


class ConsumedSourceWarning(PolarsOnlineWarning, UserWarning):
    """A warning that a query whose source is written in Python may give its rows only once.

    An Arrow C stream is consumed once: the Arrow specification says a
    stream's capsule "can only be consumed once". A query built by
    ``pl.scan_arrow_c_stream(...)`` over a DuckDB relation, a pyarrow
    ``RecordBatchReader`` or anything else with ``__arrow_c_stream__`` gives
    its rows the first time it runs, and then gives none, with no error from
    Polars. A bank fed the same query twice learns from every row and then
    from none, and the second run leaves a state that looks finished and is
    empty. Build the query again for each run, calling
    ``pl.scan_arrow_c_stream(con.sql(q))`` inside the loop, since the
    producer makes a new stream each time.

    It is a warning, not an error, because a source may have no rows: an
    empty query is not a mistake. The calls that read a query
    (:meth:`ModelBank.fit`, :meth:`ModelBank.fit_predict_batches`,
    ``lf.online.fit_predict`` and ``lf.online.predict``) issue it only for a query whose source is
    written in Python and which gave no rows, the shape a consumed stream
    has. A frame's ``.lazy()`` and a ``pl.scan_parquet`` never issue it, and
    both can be run again. :func:`polars_online.stream.with_windows`, and its namespace form
    ``lf.online.with_windows``, issue it the same way, for a run that gave no
    rows. :func:`polars_online.stream.embargo` issues it
    earlier, while it builds its query, whenever its input's source is
    written in Python: the query it returns is Polars' own, reads its input
    twice, and runs after the function returns.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        import warnings


        class Once:  # a producer of one Arrow stream, as a database cursor's result is
            def __init__(self, frame):
                self.stream = frame.__arrow_c_stream__()

            def __arrow_c_stream__(self, requested_schema=None):
                return self.stream


        rows = pl.scan_arrow_c_stream(Once(df))
        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bank.fit(rows)                       # learns all 400 rows
            bank.fit(rows)                       # the stream is spent: no rows
        print(bank.rows_fed, [type(w.message).__name__ for w in caught])

    .. code-block:: text

        400 ['ConsumedSourceWarning']

    .. rubric:: See also

    - :meth:`ModelBank.fit`, which reads a query a chunk at a time.
    - :class:`PolarsOnlineWarning`, the base class every warning of this
      library shares.
    """


#: A Python source in ``explain``, in both spellings polars uses. 1.x writes
#: ``PYTHON SCAN []``; 2.0 writes ``PYTHON[<name>] SCAN []`` for a source given
#: an ``explain_name``, which is what ``_explain_kwargs`` gives this package's
#: own plan form. A literal ``"PYTHON SCAN" in text`` therefore stopped
#: recognising our plan on 2.0 -- measured on 2.0.0-rc.1, where
#: `scan_arrow_c_stream` is *not* named and still reads ``PYTHON SCAN``, so the
#: spent-stream guard kept working and only our own plan form was missed.
#: Matching both keeps the behaviour the same on either polars rather than
#: quietly different.
_PYTHON_SCAN = re.compile(r"PYTHON(?:\[[^\]]*\])?\s+SCAN")


#: Bumped at every run of one of this package's own sources. A source that
#: got no rows from a Python scan tells a spent single-use stream from one of
#: our own plan forms over an input that was simply empty by whether another
#: of our sources ran during its collect (task 159, P2: a chain of two plan
#: forms over an empty frame warned). Best-effort, as the warning is: a
#: source running in another thread at the same time silences it too.
_SOURCE_RUNS = itertools.count(1)
_SOURCE_RUNS_LOCK = threading.Lock()
_SOURCE_RUNS_LAST = 0


def _source_started() -> int:
    """This run's number; `_sources_ran_since(n)` says whether another ran."""
    global _SOURCE_RUNS_LAST
    with _SOURCE_RUNS_LOCK:
        _SOURCE_RUNS_LAST = next(_SOURCE_RUNS)
        return _SOURCE_RUNS_LAST


def _sources_ran_since(run: int) -> bool:
    with _SOURCE_RUNS_LOCK:
        return run < _SOURCE_RUNS_LAST


def _source_runs() -> int:
    """The last run's number, without starting one: what a reader of a plan
    that is not itself a source -- a bank's ``fit`` -- hands
    `_sources_ran_since` after reading it (review 2026-10-05, YB6)."""
    with _SOURCE_RUNS_LOCK:
        return _SOURCE_RUNS_LAST


#: The most bytes of a chunk's output one of this package's sources hands
#: Polars at a time (task 348). Polars' streaming engine bounds what it holds
#: from a source by a number of frames, not of bytes: between a Python source
#: and a node that runs on every thread (a `with_columns`, a `select` that
#: computes, `.online.unnest`) it keeps up to about ten frames a thread in
#: flight, however slowly the node after them consumes -- 148 on a 14-thread
#: Mac, 28 under `POLARS_MAX_THREADS=2`, on py-polars 1.37.0 and 2.0.0 alike
#: (a scan straight into `collect_batches` keeps 6). So a source in front of
#: a slower consumer, one `fit_predict` feeding another through a
#: `with_columns`, runs that many frames ahead of it. As whole 100,000-row
#: chunks of three factor ridges' output, 29 MB each, that was the whole of
#: a 4,000,000-row stream: 1,379 MB at its peak, where a loop over the same
#: two banks held 311. In slices of at most 1 MiB the same frames are a few
#: chunks' rows, and the query held 455 MB at 1,000,000 rows and 629 MB at
#: 4,000,000, flat from its first second to the input's end (2 MiB: 774 MB
#: at 4,000,000, still filling). The chunk the bank is fed, and every value
#: out, are unchanged; a slice shares its chunk's buffers. Each extra frame
#: costs Polars about 30 µs, 3-4% of the fastest single bank's time
#: (an `ewridge` on two features, 0.17 µs a row).
_MORSEL_BYTES = 1 << 20


def _in_morsels(out: pl.DataFrame) -> Iterator[pl.DataFrame]:
    """``out`` in consecutive slices of as many rows as fit in
    `_MORSEL_BYTES` at its mean bytes a row by ``estimated_size()`` (one at
    the least), the last taking the rest: what a source yields in place of
    ``out``. A frame within the bound, an empty one included, is yielded
    whole."""
    size = out.estimated_size()
    if size <= _MORSEL_BYTES:
        yield out
        return
    rows = max(1, int(out.height * _MORSEL_BYTES // size))
    for start in range(0, out.height, rows):
        yield out.slice(start, rows)


def _is_python_scan(lf: pl.LazyFrame, plan_text: str | None = None) -> bool:
    """Whether ``lf``'s source is a Python scan, which is what an Arrow C
    stream and this package's own plan form both are.

    Read from ``explain``, which does not execute the plan and so cannot
    consume the very stream this is here to protect (checked: a plan still
    yields its rows after being explained). Any failure to read the plan is
    "not a python scan", so the guard never turns a working run into an
    error. ``plan_text`` is that output when the caller already has it, so a
    run reads the plan once for this and the order check together."""
    if plan_text is None:
        plan_text = _plan_text(lf)
    return bool(plan_text and _PYTHON_SCAN.search(plan_text))


def _user_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside
    this package, so it points at the caller's line and not at the library's.
    Counted from the function that calls this and then ``warnings.warn``."""
    here = os.path.dirname(os.path.abspath(__file__)) + os.sep
    frame: FrameType | None = sys._getframe(1)
    level = 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(here):
        frame = frame.f_back
        level += 1
    return level


def _walk(node: Any, found: list[str]) -> None:
    """Collect the nodes of a serialized plan whose output order is unspecified,
    top-down. A ``Sort`` settles the order of everything beneath it, so the
    walk stops there -- up to ties, which a sort by several keys without
    ``maintain_order`` leaves in no particular order, and that is reported; a
    join that keeps one side's order is followed on that side alone. Anything
    unrecognised is descended into generically."""
    if isinstance(node, list):
        for item in node:
            _walk(item, found)
        return
    if not isinstance(node, dict):
        return
    if len(node) == 1:
        ((tag, body),) = node.items()
        # A plan's sort has an `input`; an expression's (`pl.col(..).sort()`)
        # is tagged `Sort` too, with `expr` and `options`, and is descended as
        # any other node (review 2026-09-28).
        if tag == "Sort" and isinstance(body, dict) and "input" in body:
            # Measured on polars 1.44.2 (task 120): sorted by two keys without
            # maintain_order, 2,109 of 10,000 rows came out in another order
            # than the stable sort's -- rows of one group at one clock value,
            # reordered -- and by one key, none did. Polars promises neither,
            # and `test_order_hazards.py` pins both, so a change shows.
            opts = body.get("sort_options", {})
            keys = body.get("by_column", [])
            if len(keys) > 1 and not opts.get("maintain_order", False):
                found.append(
                    "a sort by several keys without maintain_order=True, which gives rows "
                    "with equal keys in no particular order"
                )
            return
        if tag == "Join" and isinstance(body, dict):
            args = body.get("options", {}).get("args", {})
            mode = str(args.get("maintain_order", "None")).lower()
            if mode == "none":
                found.append("a join without maintain_order")
                _walk(body.get("input_left"), found)
                _walk(body.get("input_right"), found)
            elif mode == "left":
                _walk(body.get("input_left"), found)
            elif mode == "right":
                _walk(body.get("input_right"), found)
            else:
                _walk(body.get("input_left"), found)
                _walk(body.get("input_right"), found)
            return
        if tag == "GroupBy" and isinstance(body, dict):
            if not body.get("maintain_order", False):
                found.append("a group_by without maintain_order=True")
            _walk(body.get("input"), found)
            return
        if tag == "Distinct" and isinstance(body, dict):
            found.append(
                "a unique(), whose order Polars does not keep when it runs a query in "
                "chunks, even under maintain_order=True"
            )
            _walk(body.get("input"), found)
            return
    for value in node.values():
        _walk(value, found)


#: The plan tags :func:`_walk` reports, each with the substring polars'
#: ``explain`` prints for it. Two things read this: the walk, which is the
#: authority on what counts as a hazard, and the pre-filter below, which skips
#: the deprecated JSON read when none of these can be in the plan. **They must
#: not drift apart** -- a tag the walk handles but this omits would stop being
#: warned about, silently, which is the worst shape this code can fail in, so
#: ``test_order_hazards.py`` fails if the two ever diverge. ``Sort`` is here
#: since task 120: a sort by several keys without ``maintain_order`` reports.
#: Checked, filter against raw walk,
#: across every join form -- the seven ``how=`` variants plus ``join_asof`` and
#: ``join_where`` -- and ``group_by``, ``group_by_dynamic``, ``rolling``,
#: three ``unique`` forms and ``merge_sorted``: no disagreement in any, each
#: hazard carries its marker, and a plan with no hazard carries none.
_HAZARD_TAGS: dict[str, str] = {
    "Join": "JOIN",
    "GroupBy": "AGGREGATE",
    "Distinct": "UNIQUE",
    "Sort": "SORT BY",
}


def _plan_text(lf: pl.LazyFrame) -> str | None:
    """``explain`` once per run, for the two checks that read a plan.

    Both :func:`_is_python_scan` and :func:`_order_hazards` want this text, and
    reading it twice costs more than the JSON scan it saves on a small plan
    (0.105 ms against 0.074 ms, measured) -- so a caller reads it once and
    hands it to both. ``None`` when the plan cannot be explained, which every
    reader treats as "no finding" rather than as an error."""
    try:
        return lf.explain(optimized=False)
    except Exception:  # a plan that cannot be explained is not one to inspect
        return None


def _order_hazards(lf: pl.LazyFrame, plan_text: str | None = None) -> list[str]:
    """The nodes of ``lf`` whose output order is unspecified, each as a phrase
    naming the fix; ``[]`` when there are none -- or when the plan cannot be
    read, since this inspects a deprecated polars format and is best-effort by
    design, never a reason for a run to fail.

    ``plan_text`` is ``explain``'s output when the caller already has it. It is
    read first as a filter: no hazard marker in it means no node this reports,
    so the deprecated ``serialize(format="json")`` is never touched -- which on
    a 200-column plan is 2.13 ms saved, and on any plan is one less use of a
    format polars has said it is removing. A plan that cannot be explained, or
    that explains to nothing, falls through to the JSON path rather than being
    skipped, so the filter can only ever make the check cheaper, never blinder.
    (``if plan_text``, not ``is not None``: an empty string would otherwise
    read as "inspected, no markers" and skip a real hazard -- unreachable
    today, since polars never explains a plan to nothing, and closed anyway.)"""
    if plan_text is None:
        plan_text = _plan_text(lf)
    if plan_text and not any(m in plan_text for m in _HAZARD_TAGS.values()):
        return []
    try:
        with warnings.catch_warnings():
            # polars' own deprecation of the json format is the library's to
            # hear, not the caller's.
            warnings.filterwarnings("ignore", message="'json' serialization format")
            text = lf.serialize(format="json")
        plan = json.loads(text)
        found: list[str] = []
        _walk(plan, found)
    except Exception:  # any failure to read the plan is "no finding", by design
        return []
    return found


#: Models whose fit is an accumulation, so the state after a ``fit()`` is the
#: same whatever order the rows arrived in -- **to rounding, never to the
#: bit**: the Gram sums commute mathematically but not in floating point.
#: Measured over 200 rows with no decay: ``ewridge`` 6.7e-16, ``rls`` 1.4e-15
#: (re-measured 2026-10-10; 3.3e-16 and 8.9e-16 when first listed).
#: Every other model moves materially even with no decay at all, because its
#: update is not commutative -- ``sgd`` 5.9e-03, ``pa`` 5.1e-02, ``ftrl``
#: 3.3e-02, ``quantile`` 2.8e-03 -- so "no half-life" is not on its own a
#: reason to expect order not to matter. ``huber`` and ``lasso`` were listed
#: here until 2026-09-29 (docs/PLAN.md task 139). ``huber`` read 6.7e-16 on
#: rows no residual reached ``delta * sigma``, where every weight is 1; once
#: it reweights, the fit before each row sets that row's weight, and with one
#: row in ten lifted by 5 a shuffle moved its coefficients by 1.05e-02.
#: ``lasso``'s path points commute to rounding, but ``penalty_selected`` ranks
#: them by out-of-sample error, and on the same rows shuffles moved the
#: penalty it selects from 0.01 to 0.1 and to 0.001
#: (``tests/test_order_hazards.py``).
_ORDER_FREE_MODELS = frozenset({"ewridge", "rls"})

#: Spec keys that change the *path* a fit takes, with the only values that
#: leave it order-free. Measured on the same rows: ``window_size`` 8.3e-03,
#: ``gram_block_rows`` 6.3e-04 (which row sits in the pending block when a
#: solve fires depends on arrival order), ``embargo`` 4.3e-04, and
#: ``drift_action="reset"`` **8.9e-01** -- the largest of all, and the one that
#: read as harmless until the fixture actually made drift fire. The rest are
#: denied as *unproven* rather than refuted: their code path never ran in the
#: probe, and a green result from a path that did not execute is not evidence.
_ORDER_FREE_ONLY_WHEN: dict[str, tuple[Any, ...]] = {
    # Decay off, either way the README names it: `lam=1.0` (with no
    # `half_life`), or `half_life` infinite (with no `lam`), as a float or as
    # the text a hand-written dict may carry. Only `lam=1.0` passed, so a fit
    # with `half_life=inf` warned where the README says it does not (review
    # round 4, TB2). A spec gives one of the two, so a `None` here is the
    # other's.
    "lam": (None, 1.0),
    "half_life": (None, float("inf"), "inf"),
    "window_size": (None,),
    # The window's edge, at its default, which is all a spec with no window
    # may carry (docs/PLAN.md task 196).
    "closed": ("right",),
    "window_budget": (None,),
    "snapshot_every": (None,),
    "max_rows_between_snapshots": (None,),
    "embargo": (None,),
    "label_clock": (None,),
    "gram_block_rows": (None,),
    "drift_action": ("flag",),
    "session": (None,),
    "session_gap": (None,),
    "session_shrink": (None,),
    "long_half_life": (None,),
    "ridge_scale": ("mean",),
    "conformal": (None,),
    "conformal_rate": (None,),
    "average_eta": (None,),
    "emit_averaged": (False,),
    "group_close": (None,),
    "weight": (None,),
    # Unset, a step back is refused, loudly; given, a step back that large
    # starts over, which is order itself.
    "restart_after_step_back": (None,),
    "target_gaps": ("own_rows",),
    "coef_every": (None,),
    "max_rows_between_coefs": (None,),
    "emit_autocorr": (False,),
    "emit_metrics": (False,),
    "emit_zscore": (False,),
    "emit_selected": (False,),
    "emit_sigma": (False,),
    # `sigma` times a fit's own variances: order-free no more than `sigma`.
    "emit_se_coef": (False,),
    "resid_autocorr_lag": (None,),
    "resid_quantiles": (None,),
    # Task 221's diagnostics keep a state of their own, whose order-freeness
    # is unmeasured: denied until it is.
    "emit_calibration": (False,),
    "calibration_half_life": (None,),
    "emit_breaks": (False,),
    "breaks_half_life": (None,),
    "emit_robust_se": (False,),
    "robust_se_half_life": (None,),
    "robust_se_lags": (None,),
    "emit_specification": (False,),
    "specification_half_life": (None,),
    "ljung_box_lags": (None,),
    "horizon_rows": (None,),
    "emit_tails": (False,),
    "tails_half_life": (None,),
    "emit_influence": (False,),
    "influence_half_life": (None,),
    "emit_feature_health": (False,),
    "feature_health_half_life": (None,),
}

#: Keys free to hold any value without touching order-freeness: what the fit
#: reads, how it solves, and (for ``emit_drift``, measured at 3.3e-16 with
#: ``drift_action`` left at ``"flag"``) what it reports. ``group`` is here
#: because each group accumulates independently, and ``standardize`` and
#: ``select_half_life`` because both measured at rounding.
_ORDER_FREE_ANY = frozenset(
    {
        "name",
        "type",
        "targets",
        "features",
        "feature_sets",
        "min_weight",
        # The readiness gates withhold output and change no number a fit
        # produces, and the per-row field is read from the state; order-free
        # where the fit is (docs/WARMUP-AND-CONVERGENCE.md).
        "min_settled_frac",
        "max_error_inflation",
        "emit_error_inflation",
        # The clock fields are read from the state and change no number a
        # fit produces (task 152).
        "emit_clocks",
        "group",
        "clock",
        "gap_cap",
        "fit_intercept",
        "standardize",
        "ridge",
        # `rls`'s prior strength, its `ridge` until task 195 (N11).
        "delta",
        "coef_prior",
        "solve_every",
        "max_rows_between_solves",
        "emit_drift",
        "drift_delta",
        "drift_threshold",
        # How many threads the Gram's update runs on: the same bits at every
        # count (docs/PLAN.md task 225, tests/test_gram_threads.py).
        "gram_threads",
    }
)


def _spec_items(spec: Any) -> Any:
    """Every ``(key, value)`` of a spec, with the nested ``model`` dict walked
    in place -- ``window_size`` lives *there* for ``ewridge`` and ``lasso``, not at
    the top level, so a top-level-only check would miss a windowed spec, which
    is the worst case after a drift reset."""
    for key, value in spec.items():
        if key == "model" and isinstance(value, dict):
            yield from value.items()
        else:
            yield key, value


def _order_free(specs: Any) -> bool:
    """Whether ``ModelBank.fit`` over these specs reaches the same state
    whatever order the rows arrive in -- to rounding, not to the bit.

    Deliberately narrow, and **unknown means no**: a key in neither table
    fails the check, so a spec option added later cannot quietly become exempt
    from the warning. Relaxing an entry is a measurement, not a guess.

    This is about ``fit()`` alone, whose product is the state. It is never
    about a prediction: ``pred`` is out-of-sample by construction, so row *i*
    is predicted from the rows before it, and reordering moves every
    prediction even where the coefficients commute (measured 2026-10-10: by
    up to 0.087 on a no-decay ``ewridge`` whose coefficients agreed to
    6.7e-16)."""
    if not specs:
        return False
    for spec in specs:
        model = spec.get("model")
        if not isinstance(model, dict) or model.get("type") not in _ORDER_FREE_MODELS:
            return False
        for key, value in _spec_items(spec):
            if key in _ORDER_FREE_ONLY_WHEN:
                if not any(value == ok for ok in _ORDER_FREE_ONLY_WHEN[key]):
                    return False
            elif key == "targets":
                # A target that is a window expression reads the rows ahead,
                # so the order is the target (review R2, P2).
                if any(
                    isinstance(t, pl.Expr) or (isinstance(t, dict) and "formula" in t)
                    for t in (value or ())
                ):
                    return False
            elif key not in _ORDER_FREE_ANY:
                return False
    return True


def _warn_if_order_unspecified(
    lf: pl.LazyFrame,
    what: str,
    plan_text: str | None = None,
    *,
    reads: str = "an online model learns in row order",
    before: str = "the bank",
) -> None:
    """Warn, naming ``what`` the caller called, when ``lf`` has a node whose
    output order is not guaranteed. ``plan_text`` is ``explain``'s output when
    the caller already has it (see :func:`_plan_text`). ``reads`` says why the
    order matters to the caller, and ``before`` names it in the fix."""
    hazards = _order_hazards(lf, plan_text)
    if not hazards:
        return
    msg = (
        f"{what}: the query's row order is not guaranteed, and {reads}. Its steps that "
        "leave the order open (" + "; ".join(hazards) + ") let Polars give the rows in "
        "another order when it runs the query, so the output can change from run to run. "
        'Fix the order with maintain_order="left" on a join, maintain_order=True on a '
        f"group_by, or a sort after a unique() or before {before}, or, where the order is "
        "fixed by other means, silence this warning with "
        '`warnings.simplefilter("ignore", polars_online.OrderNotGuaranteedWarning)`.'
    )
    warnings.warn(OrderNotGuaranteedWarning(msg), stacklevel=_user_stacklevel())


def _source(
    lf: pl.LazyFrame,
    make_bank: Callable[[], ModelBank],
    step: Literal["fit_predict", "predict"],
    chunk_size: int | None,
    save_state: State | None = None,
    closed_groups: State | None = None,
) -> pl.LazyFrame:
    """``lf`` streamed through the bank method ``step`` names, on a bank from
    ``make_bank``, as a plan. Each chunk is fed as rows of the whole input, so an
    error names the input's row."""
    chunk_size = _check_chunk_size(chunk_size, f"lf.online.{step}")
    # At build time, before anything runs: the order the plan will deliver is
    # decided here, and a caller should hear about it before the first chunk.
    called = f"lf.online.{step}"
    run = ModelBank._fit_predict_from if step == "fit_predict" else ModelBank._predict_from
    # One `explain` for both plan checks: the order hazards below and the
    # Python-scan test further down. Reading it twice would cost more than the
    # JSON scan the first one skips.
    plan_text = _plan_text(lf)
    _warn_if_order_unspecified(lf, called, plan_text)
    # Also decided here, off the plan as handed over: whether its source is a
    # Python scan, which is the shape `pl.scan_arrow_c_stream` has and so the
    # shape a single-use stream arrives in. Read now because `explain` must
    # see the caller's plan, and acted on below only if the run sees no rows.
    python_scan = _is_python_scan(lf, plan_text)
    rows = chunk_size or _native.default_chunk_size()
    save_path = _save_path(save_state, called)
    in_schema = lf.collect_schema()
    # The output schema, from a bank run on no rows. This is also where a spec
    # naming a column the input lacks is reported: while the plan is built,
    # as polars reports its own schema errors, not when it runs.
    bank = make_bank()
    # Before the empty frame crosses, in this call's own name: a `UInt128`
    # column cannot cross at all.
    _refuse_uint128(in_schema, called, bank._clocks())
    schema = run(bank, pl.DataFrame(schema=in_schema), 0).schema
    needed = _spec_columns(bank.specs)
    closed_path = _closed_path(closed_groups, bank.specs, called)
    # Peeked, not drained: `make_bank` can hand back the caller's own bank
    # (`lf.online.predict(bank)`), whose queue building a plan must leave
    # alone (review 2026-09-12, C4).
    closed_schema = bank.closed_groups(drop=False).clear()
    _check_closed_format(closed_path, closed_schema, called)

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        this_run = _source_started()
        # Projection pushdown reaches the input: read only the columns the
        # bank needs plus the ones the query asked for. Polars does not
        # re-apply any of the three pushdowns after a Python source, so each
        # is honoured here, and in this order: `n_rows` counts rows *before*
        # the predicate, because polars pushes a slice into a Python scan
        # only while the scan has no predicate yet (slice pushdown runs first,
        # `slice_pushdown_lp.rs`), so both present means the plan sliced
        # before it filtered. (Polars' own `pl.defer` filters first, and
        # returns 100 rows for `head(100).filter(..)`.) The slice is applied
        # to the *input*, so the bank is fed exactly the rows the query
        # pulled and no more: the state it ends in -- what `save_state`
        # writes -- is the state after those rows, whatever the chunk size.
        plan = lf
        if with_columns is not None:
            wanted = set(with_columns) | needed
            plan = plan.select([c for c in in_schema if c in wanted])
        bank = make_bank()
        seen = 0
        closed: list[pl.DataFrame] = []
        for chunk in plan.collect_batches(chunk_size=rows, maintain_order=True):
            if n_rows is not None:
                chunk = chunk.head(n_rows - seen)
            out = run(bank, chunk, seen)
            seen += chunk.height
            if closed_path is not None:
                # Drained per chunk so the bank's queue stays bounded; the
                # file is written once, at the end, where the CLI hands each
                # drain to its writer as it comes (review 2026-09-12, P5).
                # So the sidecar's rows -- one per closed (group, instance)
                # -- are held until then: bounded by the number of closes,
                # not by the input, and empty drains cost nothing
                # (docs/REVIEW-E54-E64.md G3).
                drained = bank.closed_groups()
                if drained.height:
                    closed.append(drained)
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            yield from _in_morsels(out)
            if n_rows is not None and seen >= n_rows:
                break
        # Reached only when the source has fed the bank its last row: the
        # input's end, or the rows a `head(n)` asked for. Not in a `finally`:
        # a run the caller abandons is closed whenever polars drops it -- on
        # some versions when the plan object goes -- and a run the bank
        # ended with an error never gets here, so the file, if any, stands.
        # A node after this one failing does not stop this one (polars
        # drains a Python source first), so the state is written even then;
        # the CLI saves after its output is committed, for callers who need
        # the two tied together. The sidecar goes first, as the CLI writes
        # it, so a state file has its closed rows beside it; but a sidecar
        # that cannot be written does not take the fit's state with it
        # (task 269, D-1: a refused CSV once lost both).
        try:
            if closed_path is not None:
                _write_closed(closed_path, closed, closed_schema)
        finally:
            if save_path is not None:
                bank.save(save_path)
        # Last, so a state file is written before anything is said about it.
        # `n_rows == 0` is a `head(0)` pushed into the scan, where no rows is
        # what the query asked for rather than a stream that is spent.
        if python_scan and seen == 0 and n_rows != 0 and not _sources_ran_since(this_run):
            warnings.warn(
                ConsumedSourceWarning(
                    f"{called}: the query gave no rows, and it reads a source written in Python, "
                    "such as `pl.scan_arrow_c_stream` over a DuckDB relation, a pyarrow reader or "
                    "anything with `__arrow_c_stream__`, which gives its rows once, so a second "
                    "run of the query reads nothing and this bank learns from nothing. Build the "
                    "query again for each run, calling "
                    "`pl.scan_arrow_c_stream(...)` inside the loop, or, if the source "
                    "really is empty, silence this warning with "
                    '`warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning)`.'
                ),
                stacklevel=_user_stacklevel(),
            )

    # `is_pure` tells polars two occurrences of this scan in one plan are the
    # same node, which lets it drop one of them: in the source it is the only
    # thing that can make two `PythonOptions` compare equal
    # (`polars-plan/src/plans/ir/equality.rs`), and node equality is what CSE
    # and plan dedup are keyed on. Dropping an occurrence drops its *effects*
    # too, so the claim is only ours to make when a run has none.
    #
    # The rows are pure either way: `make_bank()` is called inside `source`,
    # so every execution starts from the same bytes `load_state` fixed when
    # the plan was built (R3) and feeds them the same rows in the same order.
    # That is R2's idempotence, and it is why the two runs of an impure plan
    # write identical bytes rather than racing to a different answer
    # (measured: byte-identical to a single ordinary run).
    #
    # But identical bytes are not no bytes. `save_state` and the
    # `closed_groups` sidecar are writes, and a source that writes is not
    # pure whatever its rows do. So it is declared exactly when there is
    # nothing to write -- which is every `predict` and every fit that keeps
    # its state in memory. When there is, polars runs the source twice,
    # concurrently, and `atomic.rs`' counter makes the two writers safe
    # (docs/STATE-WORKFLOW.md R2).
    #
    # We cannot dedupe those two runs ourselves. Polars hands the source
    # callable `(with_columns, predicate, n_rows, batch_size)` and nothing
    # that identifies an execution, so a second concurrent run of one query
    # is indistinguishable from a later `collect()` -- which must re-run.
    # Overlap in time is the only signal left, and sharing rows between two
    # consumers pulling at their own rates means buffering the whole stream,
    # which is the memory bound this library exists to hold.
    pure = save_path is None and closed_path is None
    return register_io_source(
        source,
        schema=schema,
        validate_schema=True,
        is_pure=pure,
        **_explain_kwargs(bank.specs),
    )


def _fit_predict_lazy(
    lf: pl.LazyFrame,
    specs: Specs | None,
    load_state: State | ModelBank | None,
    save_state: State | None,
    chunk_size: int | None,
    closed_groups: State | None = None,
    threads: int | None = None,
) -> pl.LazyFrame:
    specs = list(specs) if specs is not None else None
    return _source(
        lf,
        _bank(specs, load_state, "lf.online.fit_predict", threads),
        "fit_predict",
        chunk_size,
        save_state,
        closed_groups,
    )


def _predict_lazy(
    lf: pl.LazyFrame, bank: ModelBank | State, chunk_size: int | None, threads: int | None = None
) -> pl.LazyFrame:
    if bank is None:
        # Read as no load_state, it would be refused for the specs it has not
        # got; the bank is what is missing.
        msg = (
            "lf.online.predict: bank must be a ModelBank or the path of a state file, but it is "
            "None. Pass a bank."
        )
        raise ValueError(msg)
    if not isinstance(bank, ModelBank) or _check_threads(threads) is not None:
        # A bank given with a count of its own is scored as a copy on that
        # pool: the caller's bank keeps its pool.
        return _source(lf, _bank(None, bank, "lf.online.predict", threads), "predict", chunk_size)

    def own() -> ModelBank:
        # `predict` leaves a bank as it was, so the caller's own is safe to
        # share with the plan; it scores as the bank stands when the plan runs.
        return bank

    return _source(lf, own, "predict", chunk_size)


def _specs_of(specs: Specs | ModelBank | State, what: str) -> list[dict[str, Any]]:
    """The spec dicts behind ``specs``: a list of them, a bank's, or a saved
    bank's (the file carries them; it is read here, once)."""
    if isinstance(specs, ModelBank):
        return specs.specs
    if isinstance(specs, (str, os.PathLike)):
        return ModelBank.load(os.fspath(specs)).specs
    who = f"online.{what}"
    if isinstance(specs, dict):
        msg = (
            f"{who}: specs must be a list of specs, a ModelBank or the path of a state file, "
            "but it is one spec. Pass it in a list, as [spec]."
        )
        raise TypeError(msg)
    out = list(specs)
    bad = [spec for spec in out if not isinstance(spec, dict)]
    if bad:
        msg = (
            f"{who}: specs must be a list of specs, a ModelBank or the path of a state file, "
            f"but one of its items is of type {type(bad[0]).__name__}. Pass a list whose every "
            "item is a spec."
        )
        raise TypeError(msg)
    return out


#: The fields laid out like ``coef``, one value per coefficient, longest
#: prefix first.
_BESIDE_COEF = ("support_coef", "se_coef_hc0", "se_coef_hac", "se_coef")


def _unnest_exprs(
    schema: pl.Schema, specs: list[dict[str, Any]], separator: str | None = None
) -> list[pl.Expr]:
    """The columns of ``schema`` with each spec's struct replaced, in place, by
    its fields -- the ``coef`` lists as one column per coefficient, named by
    :func:`polars_online.spec.coef_fields`. With a ``separator``, each column
    is named ``{spec}{separator}{name}``, as Polars 2.0's
    ``DataFrame.unnest(separator=)`` names a struct's fields.

    A spec whose column the frame has not got, or whose struct lacks a field
    the spec produces, is reported here, while the plan is built.
    """
    if separator is not None and not isinstance(separator, str):
        msg = (
            f"online.unnest: separator must be a str or None, but it is {separator!r}. Pass a "
            "str, or None to keep the field names as they are."
        )
        raise TypeError(msg)
    by_name: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec["name"] in by_name:
            msg = f'online.unnest: spec "{spec["name"]}" is given twice. Pass each spec once.'
            raise ValueError(msg)
        by_name[spec["name"]] = spec
    exprs: list[pl.Expr] = []
    # Each output column's name and where it comes from: a spec's name, or
    # None for a column of the frame kept as it is.
    origins: list[tuple[str, str | None]] = []
    # The output names that are a struct field taken whole, by name, with
    # the spec they come from: the only ones `struct.field` can take again.
    plain: set[tuple[str, str]] = set()
    for column, dtype in schema.items():
        if column not in by_name:
            exprs.append(pl.col(column))
            origins.append((column, None))
            continue
        spec = by_name.pop(column)
        if not isinstance(dtype, pl.Struct):
            msg = (
                f"online.unnest: column {column!r} is {dtype}, not the output column of spec "
                f'"{column}". Rename the column, or leave spec "{column}" out of specs.'
            )
            raise ValueError(msg)
        fields = [f.name for f in dtype.fields]
        dtypes = {f.name: f.dtype for f in dtype.fields}
        idx = output_index(spec)
        missing = [f for f in idx["field"] if f not in fields]
        if missing:
            msg = (
                f'online.unnest: column {column!r} lacks the fields spec "{column}" writes: '
                f"{', '.join(missing)}. Pass the spec the column was written by."
            )
            raise ValueError(msg)
        coefs = coef_fields(spec)
        # A `coef` with no named positions (`micro`: one row per established
        # summary, as many as there are) stays the list it is.
        lists = set(idx.filter(pl.col("kind") == "coef")["field"]) & set(coefs["field"])

        def named(name: str, col: str = column) -> str:
            return name if separator is None else f"{col}{separator}{name}"

        for field in fields:
            col = pl.col(column).struct.field(field)
            # `support_coef`, `se_coef` and the robust `se_coef_hc0` and
            # `se_coef_hac` are laid out like `coef` -- one value per
            # coefficient (docs/WARMUP-AND-CONVERGENCE.md §2.2, docs/PLAN.md
            # tasks 116 and 221 (c)) -- so they unnest the same way,
            # `support_coef_<t>_<term>`, `se_coef_<t>_<term>` and
            # `se_coef_hc0_<t>_<term>`. The longer prefixes are tried first.
            beside = next((b for b in _BESIDE_COEF if field.startswith(b)), "")
            base = "coef" + field.removeprefix(beside) if beside else field
            if base not in lists:
                exprs.append(col.alias(named(field)))
                origins.append((named(field), column))
                if separator is None:
                    plain.add((field, column))
                continue
            # A `List` as the bank writes it, or an `Array` as a file written
            # by a build that wrote one (task 244, reverted by task 250).
            get = col.arr.get if isinstance(dtypes[field], pl.Array) else col.list.get
            for position, name in (
                coefs.filter(pl.col("field") == base).select("position", "name").iter_rows()
            ):
                out = beside + "_" + name.removeprefix("coef_") if beside else name
                exprs.append(get(position, null_on_oob=False).alias(named(out)))
                origins.append((named(out), column))
    if by_name:
        missing = [f'"{name}"' for name in by_name]
        specs_word = "spec" if len(missing) == 1 else "specs"
        msg = (
            f"online.unnest: the input has no output column of {specs_word} "
            f"{_and_list(missing)}. Pass only the specs whose output columns the input holds."
        )
        raise ValueError(msg)
    _refuse_shared_names(origins, separator, plain)
    return exprs


def _and_list(words: list[str]) -> str:
    """``words`` as one English list: ``a``, ``a and b``, ``a, b and c``."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _refuse_shared_names(
    origins: list[tuple[str, str | None]],
    separator: str | None,
    plain: set[tuple[str, str]],
) -> None:
    """Refuse an unnest that would name two columns alike: two specs with a
    field of the same name (three fits of one kind each have ``weight_sum``),
    or a spec's field and a column of the frame. Polars would refuse the
    selection with its ``DuplicateError`` naming one column and neither
    source, after the plan was built; this names every shared field, the
    specs it comes from, and the two ways out. The error is Polars'
    ``DuplicateError`` still, so a handler for Polars' own catches it."""
    sources: dict[str, list[str | None]] = {}
    for name, origin in origins:
        sources.setdefault(name, []).append(origin)
    # The shared names, grouped by the specs (and frame columns) they come
    # from, so three specs sharing five fields read as one clause.
    shared: dict[tuple[str | None, ...], list[str]] = {}
    for name, origin_list in sources.items():
        if len(origin_list) > 1:
            shared.setdefault(tuple(dict.fromkeys(origin_list)), []).append(name)
    if not shared:
        return
    clauses = []
    for where, names in shared.items():
        specs = [f'"{w}"' for w in where if w is not None]
        one = len(names) == 1
        whose = "a field of " if one else "fields of "
        whose += ("spec " if len(specs) == 1 else "specs ") + _and_list(specs)
        if None in where:
            whose += " and a column of the input" if one else " and columns of the input"
        # Two specs of one target share their coefficients' names too, a
        # dozen fields or more: the first few name the clash.
        shown = names if len(names) <= 6 else [*names[:5], f"{len(names) - 5} more"]
        clauses.append(f"{_and_list(shown)} {'is' if one else 'are'} {whose}")
    # A field to take again by name: the first shared one that is a spec's
    # struct field whole, not a coefficient's element or a prefixed name.
    field = next(
        (
            (w, n)
            for where, names in shared.items()
            for n in names
            for w in where
            if w is not None and (n, w) in plain
        ),
        None,
    )
    other = (
        'separator="." to name each column after its spec'
        if separator is None
        else f"a separator other than {separator!r}"
    )
    take = (
        ""
        if field is None
        else f', or take the fields you need with pl.col("{field[0]}").struct.field("{field[1]}")'
    )
    msg = (
        f"online.unnest: {'; '.join(clauses)}, so the columns unnested would share a name. "
        f"Pass {other}{take}."
    )
    raise pl.exceptions.DuplicateError(msg)


def _unnest_lazy(
    lf: pl.LazyFrame, specs: Specs | ModelBank | State, separator: str | None = None
) -> pl.LazyFrame:
    return lf.select(_unnest_exprs(lf.collect_schema(), _specs_of(specs, "unnest"), separator))


@pl.api.register_lazyframe_namespace("online")
class LazyFrameOnlineNamespace:
    """The ``online`` methods of a query, which run a model bank over its rows as the query runs."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    @_renamed_keywords
    def fit_predict(
        self,
        specs: Specs | None = None,
        *,
        load_state: State | ModelBank | None = None,
        save_state: State | None = None,
        closed_groups: State | None = None,
        chunk_size: int | None = None,
        threads: int | None = None,
    ) -> pl.LazyFrame:
        """Add each spec's output column to a query's rows, learning the rows as the query runs.

        Use it for a stream too large to hold in memory, or to fit inside a
        query that goes on to filter, join or write the result. For a frame
        already in memory, use ``df.online.fit_predict``. To keep the bank
        and feed it more chunks later, use
        :meth:`~polars_online.ModelBank.fit_predict_batches`, and to learn
        without keeping the output, :meth:`~polars_online.ModelBank.fit`.

        .. rubric:: Parameters

        ``specs``
            The specs of a model bank that starts with nothing learned, as
            :class:`~polars_online.ModelBank` takes them. Leave it out to
            resume from ``load_state``. Given with ``load_state``, they are
            checked against the specs the state was saved with.
        ``load_state``
            A bank to resume from: the path of a saved state, or a
            :class:`~polars_online.ModelBank`. The file is read, or the bank
            copied, when the query is built, so a bank given here never
            moves, and each run of the query starts from the same state.
        ``save_state``
            The path to write the state the run ends in, after the last row
            the query gave the bank. The write is atomic
            (:meth:`~polars_online.ModelBank.save`): the file is the old
            state or the new one, never half of either, so it may be
            ``load_state``'s path. Nothing is written unless the query gives
            its last row (see Notes). A rerun whose ``load_state`` this
            query already wrote learns the same rows twice, and Notes say
            how to name the states so that it does not.
        ``closed_groups``
            The path of a file to write the groups that closed during the
            run to (:meth:`~polars_online.ModelBank.closed_groups`), in the
            format its extension names. It needs a spec with
            ``group_close``. The bank's queue of closed groups is emptied
            after every chunk, so the bank's memory stays bounded. The file
            is written once per run, under ``save_state``'s rules and before
            it. A run in which no group closed writes an empty frame with
            the schema. A CSV holds each list column, such as ``coef``, as
            JSON text, the bytes the command line writes: read it back with
            ``str.json_decode``.
        ``chunk_size``
            How many rows the bank learns at a time: 100,000 by default.
            Memory grows with a chunk and the bank's state, never with the
            length of the stream. A chunk's size never changes the numbers,
            only which rows ``coef`` is written on: each group's last row in
            each chunk.
        ``threads``
            How many threads the bank runs on, as
            :class:`~polars_online.ModelBank` takes it. ``None``, the
            default, shares the process's threads. A bank given as
            ``load_state`` keeps its own count unless one is given here. The
            output is the same at any count. The threads Polars reads the
            input with are set by ``POLARS_MAX_THREADS``.

        .. rubric:: Returns

        A query holding the input's rows, in their order, with one output
        column added per spec, named after the spec. Its fields are what
        :meth:`~polars_online.ModelBank.fit_predict` writes, which
        :mod:`polars_online.spec` describes under *What a spec writes*. The
        rows must arrive in each group's clock order, as for the bank.

        .. rubric:: Raises

        While the query is built, as Polars reports its own schema errors:

        ``ValueError``
            When:

            - neither ``specs`` nor ``load_state`` is given;
            - ``chunk_size`` or ``threads`` is below 1;
            - the bank refuses a spec;
            - a column a spec reads shares the spec's name, with the
              message of :meth:`~polars_online.ModelBank.fit_predict`;
            - ``closed_groups``'s extension names no format, or no spec has
              ``group_close``;
            - ``load_state`` is not a state this build loads, or was saved
              with specs other than ``specs``
              (:meth:`~polars_online.ModelBank.load`).
        :class:`~polars_online.ColumnNotFoundError`
            A column a spec reads is not in the query. It is a
            ``ValueError`` too.
        :class:`~polars_online.WrongTypeError`
            A column a spec reads is of a type its role cannot take, such as
            a text feature, with the messages of
            :meth:`~polars_online.ModelBank.fit_predict`, or a column is
            ``UInt128``, which no frame can carry into the bank. It is both
            a ``TypeError`` and a ``ValueError``.
        ``TypeError``
            ``threads`` is not an integer.
        ``FileNotFoundError``
            ``load_state`` is not there, or the directory of ``save_state``
            or ``closed_groups`` is not.
        ``IsADirectoryError``
            ``save_state`` is a directory.
        :class:`~polars_online.OrderNotGuaranteedWarning`
            A warning, not an error: the query has a step whose row order
            Polars does not guarantee when it runs a query a chunk at a
            time, which the warning names.

        While the query runs, an error the rows' values cause (a null clock,
        a negative weight, a step back): under py-polars 1.x it arrives as
        Polars' ``ComputeError`` carrying the bank's message, and under 2.0
        as the bank's ``ValueError`` itself, since how a query wraps an
        error is Polars' to change. A ``save_state`` that cannot be written
        when the run ends is reported the same way, as the ``OSError`` that
        names the path, wrapped under 1.x.

        .. rubric:: Examples

        This code reads ``lf`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            fitted = lf.online.fit_predict([spec], save_state="fit.state").collect()
            # Every field as a column of its own, coef as one column per coefficient.
            flat = lf.online.fit_predict([spec]).online.unnest([spec]).collect()
            print(fitted.columns[-1], flat.select("^coef_.*$").columns)

        .. code-block:: text

            ridge ['coef_y_intercept', 'coef_y_x0', 'coef_y_x1']

        .. rubric:: See also

        - ``df.online.fit_predict``, the same over a frame in memory.
        - :meth:`~polars_online.ModelBank.fit_predict_batches`, the same
          with a bank you keep.
        - :meth:`predict`, which predicts and learns nothing.
        - :meth:`unnest`, which turns each field into a column.
        - :func:`polars_online.fit_predict`, this method as a function a
          type checker can see.

        .. rubric:: Notes

        The query is pure: every run starts from the same state, the specs'
        or ``load_state``'s, so running it twice gives the same frame. So
        ``save_state``'s write gives the same bytes whenever it happens.
        Polars runs a query's source once per run, and twice when one query
        uses it twice (a self-join, ``pl.concat``, or ``pl.collect_all`` of
        two queries that write files), and each run writes the same bytes.
        Nothing is written unless the query gives its last row: a run
        abandoned before then, or one the bank ended with an error, leaves
        the file as it was. An error in a step after the bank does not stop
        the bank, so the state is written then. Where the output and the
        state must be written together or not at all, use the command line,
        which saves the state only after its output is written. A
        ``closed_groups`` file that cannot be written still leaves the state
        saved before its error is raised.

        To rerun a day's input without learning its rows twice, name the
        state's path after the input the run learned, such as
        ``state-2026-11-07`` for that day's files, and pass the previous
        day's state as ``load_state``. A rerun then starts from the state
        before that day.

        Polars pushes a filter, a selection or a ``head`` applied after the
        bank into it. A filter never changes what the bank learns from: to
        keep the bank from learning rows, give them weight 0 in the spec's
        ``weight`` column. A filter before the bank makes Polars hold several
        blocks of each parquet file per thread, where one after the bank
        keeps the query running a chunk at a time. A selection is read from
        the input, so a wide file read as a query gives only the columns the
        specs and the query need. ``head(n)`` feeds the bank the first ``n`` rows and no
        more.

        Unnest a wide spec before writing the result with
        ``sink_parquet()``. Polars' parquet writer encodes an output
        column's fields at several times the cost of the same values as
        plain columns. An ``ew_cov`` over 20 series writes 233 fields: 2,785
        ns a row as one output column, 365 unnested, where the bank's own
        work is 550. Over 50 series, 1,328 fields, the output column took
        44,642 ns a row against 2,914, and the writer set the run's pace. So
        write ``lf.online.fit_predict([spec]).online.unnest([spec]).sink_parquet(path)``.
        ``sink_ipc()`` writes the output column at about the cost of plain
        columns.

        An online model learns in row order, so the order Polars delivers
        the rows in is part of the result. Polars runs this query a chunk at
        a time, where a ``join``, ``group_by``, ``unique`` or a sort by
        several keys without an
        order guarantee delivers a different stream than ``lf.collect()``
        gives, and may differ between runs. Give a join
        ``maintain_order="left"``, a ``group_by`` ``maintain_order=True``,
        a sort by several keys ``maintain_order=True``, and sort after a
        ``unique``, or sort before the bank.
        """
        return _fit_predict_lazy(
            self._lf, specs, load_state, save_state, chunk_size, closed_groups, threads
        )

    @_renamed_keywords
    def predict(
        self,
        bank: ModelBank | State,
        *,
        chunk_size: int | None = None,
        threads: int | None = None,
    ) -> pl.LazyFrame:
        """Predict a query's rows from a bank as it stands, learning nothing.

        Use it to serve a query's rows, such as a file of new rows read with
        ``pl.scan_parquet``, from a bank that has learned. For a frame in memory, use
        ``df.online.predict``, and to predict rows and then learn them, use
        :meth:`fit_predict`.

        .. rubric:: Parameters

        ``bank``
            The bank to predict from: a :class:`~polars_online.ModelBank`,
            read as it stands each time the query runs, or the path of a
            saved state, read once, when the query is built. Predicting
            leaves a bank as it was, so a bank can be shared with the query.
            To pick up a newer state file, build the query again.
        ``chunk_size``
            How many rows are predicted at a time: 100,000 by default.
            ``coef`` is written on each group's last row in each chunk that
            is not skipped, whatever ``coef_every`` and
            ``max_rows_between_coefs`` say.
        ``threads``
            How many threads to predict on, as
            :class:`~polars_online.ModelBank` takes it. A bank given with a
            count is predicted from as a copy, so its own threads stay its
            own. Without one, a bank predicts on its own threads, and a path
            on the process's.

        .. rubric:: Returns

        A query holding the input's rows with one output column per spec:
        what :meth:`~polars_online.ModelBank.predict` gives each row, as the
        next row of its group's stream. Target columns are optional, as they
        are for :meth:`~polars_online.ModelBank.predict`.

        .. rubric:: Raises

        While the query is built:

        ``FileNotFoundError``
            ``bank`` is a path that is not there.
        ``ValueError``
            When:

            - the file is not a state this build loads
              (:meth:`~polars_online.ModelBank.load`);
            - ``bank`` is ``None``;
            - ``chunk_size`` or ``threads`` is below 1.
        :class:`~polars_online.ColumnNotFoundError`
            A column the bank reads, other than a target, is not in the
            query. It is a ``ValueError`` too.
        :class:`~polars_online.WrongTypeError`
            A column the bank reads is of a type its role cannot take, such
            as a text feature, or a column is ``UInt128``, which no frame can
            carry into the bank. It is both a ``TypeError`` and a
            ``ValueError``.
        ``TypeError``
            ``bank`` is neither a bank nor a path, or ``threads`` is not an
            integer.
        :class:`~polars_online.OrderNotGuaranteedWarning`
            A warning, not an error: the query has a step whose row order
            Polars does not guarantee when it runs a query a chunk at a
            time, which the warning names.

        While the query runs, a value the bank refuses (a null clock, a
        negative weight): under py-polars 1.x it arrives as Polars'
        ``ComputeError`` carrying :meth:`~polars_online.ModelBank.predict`'s
        message, and under 2.0 as its ``ValueError`` itself, since how a
        query wraps an error is Polars' to change.

        .. rubric:: Examples

        This code reads ``lf`` and ``later`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(lf)                              # learn df's 400 rows
            bank.save("trained.state")
            scored = later.online.predict("trained.state").collect()   # the next 400, unlearned
            print(scored["ridge"].struct.field("pred_y").count(), bank.rows_fed)

        .. code-block:: text

            400 400

        .. rubric:: See also

        - :meth:`~polars_online.ModelBank.predict`, which this runs over
          each chunk, and the fields it gives.
        - :meth:`fit_predict`, which learns each row after predicting it.
        - :func:`polars_online.predict`, this method as a function a type
          checker can see.
        """
        return _predict_lazy(self._lf, bank, chunk_size, threads)

    def unnest(
        self, specs: Specs | ModelBank | State, *, separator: str | None = None
    ) -> pl.LazyFrame:
        """Turn each spec's output column in a query into one column per field.

        Use it on a query whose output a bank wrote, from
        :meth:`fit_predict` or from a parquet file the command line wrote,
        to read its fields as plain columns or to write them to a file. For
        a frame in memory, use ``df.online.unnest``. To read one field
        without unnesting, use ``pl.col("ridge").struct.field("pred_y")``,
        with the spec's name and the field's.

        .. rubric:: Parameters

        ``specs``
            The specs whose output columns to unnest: the spec dicts, a
            :class:`~polars_online.ModelBank` (its specs), or the path of a
            saved state, which carries them. Each output column takes the
            place of its spec's column. The rest of the query, and the
            output column of a spec not named, are left as they are.
        ``separator``
            A text to name each column after its spec too, as
            ``{spec}{separator}{name}``: ``separator="."`` gives
            ``ridge.pred_y`` and ``ridge.coef_y_x1``, as Polars 2.0's
            ``LazyFrame.unnest(separator=)`` names a struct's fields. Give
            it whenever two specs write a field of the same name, as almost
            any two do: most models write ``weight_sum``, ``settled_frac``
            and ``withheld_reason``. ``None``, the default, names each column by its
            field alone. The names are made here rather than by Polars' own
            ``separator``, so it works on every Polars this library supports.

        .. rubric:: Returns

        A query with each output column replaced by its fields as columns:

        - each scalar field under its own name, such as ``pred_y`` or
          ``weight_sum@h500``;
        - each ``coef`` list as one column per coefficient, named
          ``coef_{target}_{term}{combo}{instance}``, such as
          ``coef_y_intercept`` or ``coef_y_x1__r0.5@h500``, as
          :func:`polars_online.spec.coef_fields` lists them;
        - ``support_coef``, the share of each coefficient the data
          determined, the same way, as ``support_coef_{target}_{term}...``;
        - ``se_coef`` as ``se_coef_{target}_{term}...``, and the robust
          ``se_coef_hc0`` and ``se_coef_hac`` as
          ``se_coef_hc0_{target}_{term}...``.

        .. rubric:: Raises

        While the query is built:

        ``ValueError``
            When:

            - a spec's output column is not in the query, is not a struct,
              or lacks a field the spec writes;
            - a spec is given twice, or is not valid;
            - ``specs`` is the path of a file that is not a state this build
              loads (:meth:`~polars_online.ModelBank.load`).
        ``TypeError``
            ``specs`` is none of the three.
        ``FileNotFoundError``
            ``specs`` is a path that is not there.
        Polars' ``DuplicateError``
            Two columns would get the same name, without a ``separator``.
            The message names each shared field and the specs it comes from.

        .. rubric:: Examples

        This code reads ``lf`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            betas = (
                lf.online.fit_predict([spec])
                .online.unnest([spec], separator=".")
                .select("t", "^ridge.coef_.*$")
                .collect()
            )
            print(betas.columns)

        .. code-block:: text

            ['t', 'ridge.coef_y_intercept', 'ridge.coef_y_x0', 'ridge.coef_y_x1']

        .. rubric:: See also

        - :func:`polars_online.spec.coef_fields`, each coefficient's column
          name and where it sits in ``coef``.
        - :func:`polars_online.spec.output_fields`, the fields a spec writes.
        - :func:`polars_online.unnest`, this method as a function a type
          checker can see.
        """
        return _unnest_lazy(self._lf, specs, separator)

    def with_windows(
        self,
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
    ) -> pl.LazyFrame:
        """Add one column per window operator's expression to a query's rows, as the query runs.

        It is :func:`polars_online.stream.with_windows` as a method, so a
        query can add window operators' values and fit on them in one
        chain. For a frame in memory, use ``df.online.with_windows``.

        .. rubric:: Parameters

        ``*exprs``, ``**named`` and every keyword
            As :func:`polars_online.stream.with_windows` takes them, with the
            query as its input.

        .. rubric:: Returns

        A query holding the input's rows with one column added per
        expression, as :func:`polars_online.stream.with_windows` returns it.

        .. rubric:: Raises

        What :func:`polars_online.stream.with_windows` raises.

        .. rubric:: Examples

        This code reads ``trades.parquet`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            fitted = (
                pl.scan_parquet("trades.parquet")
                .online.with_windows(
                    mid_5s=po.ewm_mean("mid", half_life="5s", window_size="1m"),
                    clock="ts", gap_cap="5m", group="symbol",
                )
                .collect()
            )
            print(fitted.columns[-1], fitted.height)

        .. code-block:: text

            mid_5s 3000

        .. rubric:: See also

        - :func:`polars_online.stream.with_windows`, every parameter and
          what it refuses.
        - :mod:`polars_online.ops`, the window operators.
        """
        from polars_online.stream import with_windows

        return with_windows(
            self._lf,
            *exprs,
            clock=clock,
            gap_cap=gap_cap,
            restart_after_step_back=restart_after_step_back,
            session=session,
            session_gap=session_gap,
            group=group,
            like=like,
            chunk_size=chunk_size,
            load_state=load_state,
            save_state=save_state,
            **named,
        )


@pl.api.register_dataframe_namespace("online")
class DataFrameOnlineNamespace:
    """The ``online`` methods of a frame, which run a model bank over its rows at once."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def fit_predict(
        self,
        specs: Specs | None = None,
        *,
        load_state: State | ModelBank | None = None,
        save_state: State | None = None,
        closed_groups: State | None = None,
        chunk_size: int | None = None,
        threads: int | None = None,
    ) -> pl.DataFrame:
        """Add each spec's output column to a frame's rows, learning them, in one call.

        It is ``ModelBank(specs).fit_predict(df)`` in one call, from a bank
        that is then dropped. Use it for a frame already in memory, when you
        do not need the bank afterwards. To feed the same bank more rows
        later, keep a :class:`~polars_online.ModelBank` of your own, and for
        a query, use ``lf.online.fit_predict``.

        .. rubric:: Parameters

        ``specs``
            The specs of a bank that starts with nothing learned. Leave it
            out to start from ``load_state``.
        ``load_state``
            What to start from in place of the specs: the path of a saved
            state, or a :class:`~polars_online.ModelBank`, which is copied
            and does not move.
        ``save_state``
            The path to save the bank to before it is dropped
            (:meth:`~polars_online.ModelBank.save`).
        ``closed_groups``
            The path of a file to write the groups that closed to, in the
            format its extension names, before ``save_state`` is written
            (:meth:`~polars_online.ModelBank.closed_groups`). A CSV holds
            each list column as JSON text, as ``lf.online.fit_predict``
            writes it. A file that cannot be written still leaves the state
            saved before its error is raised.
        ``chunk_size``
            How many rows the bank learns at a time, as
            ``lf.online.fit_predict`` reads its input. ``None``, the
            default, feeds the frame as one chunk. The numbers are the same
            either way: only the rows ``coef`` is written on move, to each
            chunk's last row of each group.
        ``threads``
            How many threads the bank runs on, as
            :meth:`LazyFrameOnlineNamespace.fit_predict` takes it.

        .. rubric:: Returns

        The frame with one output column added per spec, as
        :meth:`~polars_online.ModelBank.fit_predict` returns it.

        .. rubric:: Raises

        What :class:`~polars_online.ModelBank`,
        :meth:`~polars_online.ModelBank.fit_predict`,
        :meth:`~polars_online.ModelBank.load` and
        :meth:`~polars_online.ModelBank.save` raise, and:

        ``ValueError``
            Neither ``specs`` nor ``load_state`` is given, ``chunk_size`` is
            below 1, or ``closed_groups`` is a file that
            :meth:`LazyFrameOnlineNamespace.fit_predict` refuses, checked
            before the fit.
        ``FileNotFoundError``
            The directory of ``save_state`` or ``closed_groups`` is not
            there, checked before the fit rather than after it.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            out = df.online.fit_predict([spec], save_state="ridge.state")
            print(out["ridge"].struct.field("pred_y").count(), "of", out.height, "rows predicted")

        .. code-block:: text

            396 of 400 rows predicted

        .. rubric:: See also

        - :meth:`LazyFrameOnlineNamespace.fit_predict`, the same over a
          query, run as the query runs.
        - :meth:`~polars_online.ModelBank.fit_predict`, the same with a bank
          you keep.
        - :func:`polars_online.fit_predict`, this method as a function a
          type checker can see.
        """
        chunk_size = _check_chunk_size(chunk_size, "df.online.fit_predict")
        save_path = _save_path(save_state, "df.online.fit_predict")
        bank = _bank(specs, load_state, "df.online.fit_predict", threads)()
        closed_path = _closed_path(closed_groups, bank.specs, "df.online.fit_predict")
        if closed_path is not None:
            _check_closed_format(
                closed_path, bank.closed_groups(drop=False).clear(), "df.online.fit_predict"
            )
        out = _in_chunks(self._df, chunk_size, bank._fit_predict_from)
        try:
            if closed_path is not None:
                # One drain, and its empty frame the schema: two calls were
                # right only by the order Python evaluates arguments in (C4).
                drained = bank.closed_groups()
                _write_closed(closed_path, [drained], drained.clear())
        finally:
            # Saved even when the sidecar fails, as the plan form saves it.
            if save_path is not None:
                bank.save(save_path)
        return out

    def predict(
        self,
        bank: ModelBank | State,
        *,
        chunk_size: int | None = None,
        threads: int | None = None,
    ) -> pl.DataFrame:
        """Predict a frame's rows from a bank as it stands, learning nothing.

        It is :meth:`~polars_online.ModelBank.predict` over the frame, from a
        bank or a saved state. Use it to serve rows in memory
        from a bank that has learned. For a query, use ``lf.online.predict``.

        .. rubric:: Parameters

        ``bank``
            The bank to predict from: a :class:`~polars_online.ModelBank`,
            which does not move, or the path of a saved state.
        ``chunk_size``
            How many rows are predicted at a time, as ``lf.online.predict``
            reads its input. ``None``, the default, predicts the frame as one
            chunk. The numbers are the same either way. ``coef`` is written
            on each group's last row in each chunk that is not skipped,
            whatever ``coef_every`` and ``max_rows_between_coefs`` say.
        ``threads``
            How many threads to predict on, as
            :meth:`LazyFrameOnlineNamespace.predict` takes it.

        .. rubric:: Returns

        The frame with one output column added per spec, as
        :meth:`~polars_online.ModelBank.predict` returns it.

        .. rubric:: Raises

        What :meth:`~polars_online.ModelBank.load` (for a path) and
        :meth:`~polars_online.ModelBank.predict` raise, and:

        ``TypeError``
            ``bank`` is neither a bank nor a path, or ``chunk_size`` is not a
            whole number, as :class:`~polars_online.WrongTypeError`, which is
            a ``ValueError`` too.
        ``ValueError``
            ``chunk_size`` is below 1.

        .. rubric:: Examples

        This code reads ``df`` and ``today`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            bank = po.ModelBank([spec])
            bank.fit(df)                                  # learn df's 400 rows
            scored = today.online.predict(bank)           # the next 400, unlearned
            print(scored["ridge"].struct.field("pred_y").count(), bank.rows_fed)

        .. code-block:: text

            400 400

        .. rubric:: See also

        - :meth:`LazyFrameOnlineNamespace.predict`, the same over a query.
        - :func:`polars_online.predict`, this method as a function a type
          checker can see.
        """
        chunk_size = _check_chunk_size(chunk_size, "df.online.predict")
        threads = _check_threads(threads)
        if not isinstance(bank, ModelBank):
            bank = ModelBank.load(os.fspath(bank), threads=threads)
        elif threads is not None:
            # Scored as a copy on that pool: the caller's bank keeps its own.
            bank = ModelBank.load_bytes(bank.save_bytes(), threads=threads)
        return _in_chunks(self._df, chunk_size, bank._predict_from)

    def unnest(
        self, specs: Specs | ModelBank | State, *, separator: str | None = None
    ) -> pl.DataFrame:
        """Turn each spec's output column in a frame into one column per field.

        It is :meth:`LazyFrameOnlineNamespace.unnest` for a frame in memory,
        run at once.

        .. rubric:: Parameters

        ``specs``, ``separator``
            As :meth:`LazyFrameOnlineNamespace.unnest` takes them: the specs
            whose output columns to unnest, and a text to name each column
            after its spec too, as ``{spec}{separator}{name}``, as Polars
            2.0's ``DataFrame.unnest(separator=)`` names a struct's fields.

        .. rubric:: Returns

        The frame with each output column replaced by its fields: each scalar
        field under its own name, and each ``coef`` list as one named column
        per coefficient, as :meth:`LazyFrameOnlineNamespace.unnest` lists
        them.

        .. rubric:: Raises

        What :meth:`LazyFrameOnlineNamespace.unnest` raises, on the call.

        .. rubric:: Examples

        This code reads ``df`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
            flat = df.online.fit_predict([spec]).online.unnest([spec])
            print(flat.select("^coef_.*$").columns)

        .. code-block:: text

            ['coef_y_intercept', 'coef_y_x0', 'coef_y_x1']

        .. rubric:: See also

        - :func:`polars_online.spec.coef_fields`, each coefficient's column
          name and where it sits in ``coef``.
        - :func:`polars_online.unnest`, this method as a function a type
          checker can see.
        """
        return self._df.select(
            _unnest_exprs(self._df.schema, _specs_of(specs, "unnest"), separator)
        )

    def with_windows(
        self,
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
    ) -> pl.DataFrame:
        """Add one column per window operator's expression to a frame's rows.

        It is :func:`polars_online.stream.with_windows` as a method, as
        :meth:`LazyFrameOnlineNamespace.with_windows` is for a query.

        .. rubric:: Parameters

        ``*exprs``, ``**named`` and every keyword
            As :func:`polars_online.stream.with_windows` takes them, with the
            frame as its input.

        .. rubric:: Returns

        The frame with one column added per expression.

        .. rubric:: Raises

        What :func:`polars_online.stream.with_windows` raises.

        .. rubric:: Examples

        This code reads ``trades`` from the README's `Example data
        <https://github.com/hgilde/polars-online#example-data>`_:

        .. code-block:: python

            out = trades.online.with_windows(
                mid_5s=po.ewm_mean("mid", half_life="5s", window_size="1m"),
                clock="ts", gap_cap="5m", group="symbol",
            )
            print(out.columns[-1], out.height)

        .. code-block:: text

            mid_5s 3000

        .. rubric:: See also

        - :func:`polars_online.stream.with_windows`, every parameter and
          what it refuses.
        """
        from polars_online.stream import with_windows

        return with_windows(
            self._df,
            *exprs,
            clock=clock,
            gap_cap=gap_cap,
            restart_after_step_back=restart_after_step_back,
            session=session,
            session_gap=session_gap,
            group=group,
            like=like,
            chunk_size=chunk_size,
            load_state=load_state,
            save_state=save_state,
            **named,
        )


@overload
def fit_predict(
    frame: pl.LazyFrame,
    specs: Specs | None = None,
    *,
    load_state: State | ModelBank | None = None,
    save_state: State | None = None,
    closed_groups: State | None = None,
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.LazyFrame: ...


@overload
def fit_predict(
    frame: pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | ModelBank | None = None,
    save_state: State | None = None,
    closed_groups: State | None = None,
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.DataFrame: ...


@_renamed_keywords
def fit_predict(
    frame: pl.LazyFrame | pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | ModelBank | None = None,
    save_state: State | None = None,
    closed_groups: State | None = None,
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Add each spec's output column to a query's or a frame's rows, learning them.

    It is ``frame.online.fit_predict(...)`` as a plain function, so that a
    type checker can see it: the ``online`` namespaces are attached when
    the package is imported, which a type checker does not follow. The
    namespace methods take the same arguments.

    .. rubric:: Parameters

    ``frame``
        A query or a frame. A query gives a query that learns the rows when
        it runs (:meth:`~polars_online._frame.LazyFrameOnlineNamespace.fit_predict`).
        A frame gives the frame with the bank's columns
        (:meth:`~polars_online._frame.DataFrameOnlineNamespace.fit_predict`).
    ``specs``
        The specs of a bank that starts with nothing learned.
    ``load_state``
        What to start from in place of the specs: the path of a saved
        state, or a :class:`ModelBank`, which is copied.
    ``save_state``
        The path to write the state the bank ends in.
    ``closed_groups``
        The path of a file to write the groups that closed to.
    ``chunk_size``
        How many rows the bank learns at a time, from a query or a frame
        alike.
    ``threads``
        How many threads the bank runs on, as
        :class:`~polars_online.ModelBank` takes it.

    .. rubric:: Returns

    A query for a query, and a frame for a frame, with one output column
    added per spec.

    .. rubric:: Raises

    ``TypeError``
        ``frame`` is neither a query nor a frame.

    Otherwise, what the namespace method raises.

    .. rubric:: Examples

    This code reads ``lf`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.fit_predict(lf, [spec])     # a LazyFrame to a type checker too
        print(out.collect()["ridge"].struct.field("pred_y").count())

    .. code-block:: text

        396

    .. rubric:: See also

    - :meth:`~polars_online._frame.LazyFrameOnlineNamespace.fit_predict`,
      each parameter in full, for a query.
    - :meth:`~polars_online._frame.DataFrameOnlineNamespace.fit_predict`,
      the same for a frame.
    """
    if isinstance(frame, pl.LazyFrame):
        return _fit_predict_lazy(
            frame, specs, load_state, save_state, chunk_size, closed_groups, threads
        )
    _check_frame(frame, "fit_predict")
    return DataFrameOnlineNamespace(frame).fit_predict(
        specs,
        load_state=load_state,
        save_state=save_state,
        closed_groups=closed_groups,
        chunk_size=chunk_size,
        threads=threads,
    )


@overload
def predict(
    frame: pl.LazyFrame,
    bank: ModelBank | State,
    *,
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.LazyFrame: ...


@overload
def predict(
    frame: pl.DataFrame,
    bank: ModelBank | State,
    *,
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.DataFrame: ...


@_renamed_keywords
def predict(
    frame: pl.LazyFrame | pl.DataFrame,
    bank: ModelBank | State,
    *,
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Predict a query's or a frame's rows from a bank as it stands, learning nothing.

    It is ``frame.online.predict(bank)`` as a plain function, so that a type
    checker can see it: the ``online`` namespaces are attached when the
    package is imported, which a type checker does not follow.

    .. rubric:: Parameters

    ``frame``
        A query, which gives a query
        (:meth:`~polars_online._frame.LazyFrameOnlineNamespace.predict`), or
        a frame, which gives a frame
        (:meth:`~polars_online._frame.DataFrameOnlineNamespace.predict`).
    ``bank``
        A :class:`~polars_online.ModelBank`, or the path of a saved state.
    ``chunk_size``
        How many rows are predicted at a time, from a query or a frame
        alike.
    ``threads``
        How many threads to predict on, as the namespace method takes it.

    .. rubric:: Returns

    A query for a query, and a frame for a frame, with one output column
    added per spec.

    .. rubric:: Raises

    ``TypeError``
        ``frame`` is neither a query nor a frame.

    Otherwise, what the namespace method raises.

    .. rubric:: Examples

    This code reads ``df`` and ``today`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        bank = po.ModelBank([spec])
        bank.fit(df)
        scored = po.predict(today, bank)     # a DataFrame to a type checker too
        print(scored["ridge"].struct.field("pred_y").count())

    .. code-block:: text

        400

    .. rubric:: See also

    - :meth:`~polars_online._frame.LazyFrameOnlineNamespace.predict`, each
      parameter in full.
    - :meth:`~polars_online.ModelBank.predict`, the fields it gives.
    """
    if isinstance(frame, pl.LazyFrame):
        return _predict_lazy(frame, bank, chunk_size, threads)
    _check_frame(frame, "predict")
    return DataFrameOnlineNamespace(frame).predict(bank, chunk_size=chunk_size, threads=threads)


@overload
def unnest(
    frame: pl.LazyFrame, specs: Specs | ModelBank | State, *, separator: str | None = None
) -> pl.LazyFrame: ...


@overload
def unnest(
    frame: pl.DataFrame, specs: Specs | ModelBank | State, *, separator: str | None = None
) -> pl.DataFrame: ...


def unnest(
    frame: pl.LazyFrame | pl.DataFrame,
    specs: Specs | ModelBank | State,
    *,
    separator: str | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Turn each spec's output column in a query or a frame into one column per field.

    It is ``frame.online.unnest(specs)`` as a plain function, so that a type
    checker can see it: the ``online`` namespaces are attached when the
    package is imported, which a type checker does not follow.

    .. rubric:: Parameters

    ``frame``
        A query, which gives a query
        (:meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest`), or a
        frame, which gives a frame
        (:meth:`~polars_online._frame.DataFrameOnlineNamespace.unnest`).
    ``specs``
        The specs whose output columns to unnest: the spec dicts, a bank, or
        the path of a saved state.
    ``separator``
        A text to name each column after its spec too, as
        ``{spec}{separator}{name}``, as for the namespace method.

    .. rubric:: Returns

    A query for a query, and a frame for a frame, with each output column
    replaced by its fields, and each ``coef`` list by one named column per
    coefficient.

    .. rubric:: Raises

    ``TypeError``
        ``frame`` is neither a query nor a frame.

    Otherwise, what the namespace method raises.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        flat = po.unnest(df.online.fit_predict([spec]), [spec], separator=".")
        print(flat.select("^ridge.coef_.*$").columns)

    .. code-block:: text

        ['ridge.coef_y_intercept', 'ridge.coef_y_x0', 'ridge.coef_y_x1']

    .. rubric:: See also

    - :meth:`~polars_online._frame.LazyFrameOnlineNamespace.unnest`, the
      columns it writes, in full.
    - :func:`polars_online.spec.coef_fields`, each coefficient's column
      name.
    """
    if isinstance(frame, pl.LazyFrame):
        return _unnest_lazy(frame, specs, separator)
    _check_frame(frame, "unnest")
    return DataFrameOnlineNamespace(frame).unnest(specs, separator=separator)


def _check_frame(frame: object, what: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        who = f"po.{what}"
        msg = (
            f"{who}: it takes a Polars DataFrame or LazyFrame, but it was given an object of "
            f"type {type(frame).__name__}. Pass a DataFrame or a LazyFrame."
        )
        raise TypeError(msg)
