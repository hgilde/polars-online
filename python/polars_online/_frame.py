"""The bank as a polars source: ``lf.online.fit_predict(specs)`` (ENHANCEMENTS E33).

A ``LazyFrame`` in, a ``LazyFrame`` out. Executing the plan streams the input
through a fresh :class:`ModelBank` in ``chunk_rows`` chunks, so a query with
the bank in it is O(chunk) in memory however long the stream is -- where a
user *expression* in the same query is O(data): polars calls it once with
its whole column, and its streaming engine collects the column to do so
(docs/PERFORMANCE.md section 11; that is why the expression namespace is
dormant, docs/PLAN.md section 6). This is polars' IO-plugin mechanism
(``polars.io.plugins.register_io_source``): the bank is registered as a
*source*, the kind of node the engine pulls batches from, and what comes
after it -- filters, selects, joins, ``sink_parquet`` -- is polars' own.

The plan is pure: every execution starts from the same state (the specs'
initial state, or ``load_state``), so collecting twice gives the same frame,
and nothing is saved. For the state after the stream use :func:`run` with
``save_state=``, or a bank of your own over ``collect_batches``.

``df.online.fit_predict(specs)`` is the eager twin, ``ModelBank(specs)
.fit_predict(df)`` in one call. Both namespaces are attached at import, which
no type checker can see; :func:`fit_predict` and :func:`predict` are the same
calls with the frame as the first argument, visibly typed.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any, overload

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _polars_online as _native
from polars_online._bank import ModelBank

__all__ = ["DataFrameOnlineNamespace", "LazyFrameOnlineNamespace", "fit_predict", "predict"]

Specs = Iterable[dict[str, Any]]
State = str | os.PathLike[str]

# The input columns a spec reads, by key (crates/online-polars/src/bank.rs
# `extract`); the rest of the frame is carried through.
_COLUMN_KEYS = ("targets", "features")
_SCALAR_COLUMN_KEYS = ("clock", "session", "weight", "group")


def _bank(specs: Specs | None, load_state: State | None, what: str) -> ModelBank:
    if load_state is not None:
        return _load(load_state, specs)
    if specs is None:
        msg = f"online.{what} needs specs, or load_state= to take them from a saved bank"
        raise ValueError(msg)
    return ModelBank(specs)


def _spec_columns(specs: Iterable[dict[str, Any]]) -> set[str]:
    cols: set[str] = set()
    for spec in specs:
        for key in _COLUMN_KEYS:
            cols.update(spec.get(key) or ())
        for key in _SCALAR_COLUMN_KEYS:
            if spec.get(key) is not None:
                cols.add(spec[key])
    return cols


def _source(
    lf: pl.LazyFrame,
    make_bank: Callable[[], ModelBank],
    step: Callable[[ModelBank, pl.DataFrame], pl.DataFrame],
    chunk_rows: int | None,
) -> pl.LazyFrame:
    """``lf`` streamed through ``step`` on a bank from ``make_bank``, as a plan."""
    if chunk_rows is not None and chunk_rows < 1:
        msg = f"chunk_rows must be at least 1, got {chunk_rows}"
        raise ValueError(msg)
    rows = chunk_rows or _native.default_chunk_rows()
    in_schema = lf.collect_schema()
    # The output schema, from a bank run on no rows. This is also where a spec
    # naming a column the input lacks is reported: while the plan is built,
    # as polars reports its own schema errors, not when it runs.
    bank = make_bank()
    schema = step(bank, pl.DataFrame(schema=in_schema)).schema
    needed = _spec_columns(bank.specs)

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # Projection pushdown reaches the input: read only the columns the
        # bank needs plus the ones the query asked for. Polars does not
        # re-apply any of the three pushdowns after a Python source, so each
        # is honoured here, and in this order: `n_rows` counts rows *before*
        # the predicate, because polars pushes a slice into a Python scan
        # only while the scan has no predicate yet (slice pushdown runs first,
        # `slice_pushdown_lp.rs`), so both present means the plan sliced
        # before it filtered. (Polars' own `pl.defer` filters first, and
        # returns 100 rows for `head(100).filter(..)`.)
        plan = lf
        if with_columns is not None:
            wanted = set(with_columns) | needed
            plan = plan.select([c for c in in_schema if c in wanted])
        bank = make_bank()
        seen = 0
        for chunk in plan.collect_batches(chunk_size=rows, maintain_order=True):
            out = step(bank, chunk)
            done = False
            if n_rows is not None:
                take = min(out.height, n_rows - seen)
                out = out.head(take)
                seen += take
                done = seen >= n_rows
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            yield out
            if done:
                return

    return register_io_source(source, schema=schema, validate_schema=True)


def _fit_predict_lazy(
    lf: pl.LazyFrame, specs: Specs | None, load_state: State | None, chunk_rows: int | None
) -> pl.LazyFrame:
    specs = list(specs) if specs is not None else None
    return _source(
        lf,
        lambda: _bank(specs, load_state, "fit_predict"),
        ModelBank.fit_predict,
        chunk_rows,
    )


def _predict_lazy(
    lf: pl.LazyFrame, bank: ModelBank | State, chunk_rows: int | None
) -> pl.LazyFrame:
    def make() -> ModelBank:
        # `predict` leaves a bank as it was, so the caller's own is safe to
        # share with the plan; it scores as the bank stands when the plan runs.
        return bank if isinstance(bank, ModelBank) else _load(bank)

    return _source(lf, make, ModelBank.predict, chunk_rows)


def _load(state: State, specs: Specs | None = None) -> ModelBank:
    return ModelBank.load(os.fspath(state), specs)


@pl.api.register_lazyframe_namespace("online")
class LazyFrameOnlineNamespace:
    """A model bank over the plan's rows, as a plan that streams."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def fit_predict(
        self,
        specs: Specs | None = None,
        *,
        load_state: State | None = None,
        chunk_rows: int | None = None,
    ) -> pl.LazyFrame:
        """The plan's rows plus one struct column per spec, learning as it goes.

        Executing the returned plan -- ``collect()``, ``collect_batches()``,
        ``sink_parquet()`` and the rest -- streams this plan's rows through a
        new :class:`ModelBank` in ``chunk_rows`` chunks (default 100,000;
        chunking never changes the numbers, only ``coef``'s reporting cadence),
        so memory is O(chunk + state) whatever the length of the stream. Rows
        must arrive in stream order, as for the bank. Filters, selections and
        ``head`` applied after are pushed into the source: a filter never
        changes what the bank learns from -- filter *before* to do that -- and
        a selection is read from the input, so a wide scan reads only the
        columns the specs and the query need.

        ``specs`` are the bank's, or ``load_state`` names a saved bank to
        resume from (with ``specs``, they are checked against the file). Each
        execution starts from that state afresh and saves nothing, so a plan
        collected twice gives the same frame; the state after the stream is
        :func:`polars_online.run`'s ``save_state=`` or your own bank's
        ``save``. An error in the bank surfaces as a polars ``ComputeError``
        carrying its message.
        """
        return _fit_predict_lazy(self._lf, specs, load_state, chunk_rows)

    def predict(self, bank: ModelBank | State, *, chunk_rows: int | None = None) -> pl.LazyFrame:
        """The plan's rows scored against ``bank`` as it stands, learning nothing.

        Each row gets :meth:`ModelBank.predict`'s struct: what the bank would
        report for it as the next row of its group's stream, from the current
        state, which the plan never moves. ``bank`` is a :class:`ModelBank`
        (scored as it stands each time the plan runs; ``predict`` leaves it
        untouched, so sharing it with a plan is safe) or a path to a saved
        state, loaded each time the plan runs. Target columns are optional,
        as for ``predict``; ``chunk_rows`` is the read chunk.
        """
        return _predict_lazy(self._lf, bank, chunk_rows)


@pl.api.register_dataframe_namespace("online")
class DataFrameOnlineNamespace:
    """A model bank over the frame's rows, in one call."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def fit_predict(
        self, specs: Specs | None = None, *, load_state: State | None = None
    ) -> pl.DataFrame:
        """``ModelBank(specs).fit_predict(df)`` -- the frame plus one struct
        column per spec, from a bank that is then dropped; keep a bank of your
        own to save its state or to feed it more rows. ``load_state`` starts
        it from a saved bank instead."""
        return _bank(specs, load_state, "fit_predict").fit_predict(self._df)

    def predict(self, bank: ModelBank | State) -> pl.DataFrame:
        """:meth:`ModelBank.predict` over the frame: scored against ``bank`` --
        a :class:`ModelBank`, or the path of a saved one -- as it stands, which
        does not move."""
        if not isinstance(bank, ModelBank):
            bank = _load(bank)
        return bank.predict(self._df)


@overload
def fit_predict(
    frame: pl.LazyFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    chunk_rows: int | None = None,
) -> pl.LazyFrame: ...


@overload
def fit_predict(
    frame: pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    chunk_rows: int | None = None,
) -> pl.DataFrame: ...


def fit_predict(
    frame: pl.LazyFrame | pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    chunk_rows: int | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """``frame.online.fit_predict(...)``, spelled so that a type checker can see it.

    A ``LazyFrame`` gives a plan that streams the rows through a bank when it
    runs (:meth:`LazyFrameOnlineNamespace.fit_predict`); a ``DataFrame`` gives
    the frame with the bank's columns
    (:meth:`DataFrameOnlineNamespace.fit_predict`). ``chunk_rows`` is the
    plan's read chunk; a frame already in memory is fitted in one call.
    """
    if isinstance(frame, pl.LazyFrame):
        return _fit_predict_lazy(frame, specs, load_state, chunk_rows)
    _check_frame(frame, "fit_predict")
    return DataFrameOnlineNamespace(frame).fit_predict(specs, load_state=load_state)


@overload
def predict(
    frame: pl.LazyFrame, bank: ModelBank | State, *, chunk_rows: int | None = None
) -> pl.LazyFrame: ...


@overload
def predict(
    frame: pl.DataFrame, bank: ModelBank | State, *, chunk_rows: int | None = None
) -> pl.DataFrame: ...


def predict(
    frame: pl.LazyFrame | pl.DataFrame, bank: ModelBank | State, *, chunk_rows: int | None = None
) -> pl.LazyFrame | pl.DataFrame:
    """``frame.online.predict(bank)``, spelled so that a type checker can see it.

    Scores the rows against ``bank`` as it stands and learns nothing: a plan
    from a ``LazyFrame`` (:meth:`LazyFrameOnlineNamespace.predict`), a frame
    from a ``DataFrame`` (:meth:`DataFrameOnlineNamespace.predict`).
    """
    if isinstance(frame, pl.LazyFrame):
        return _predict_lazy(frame, bank, chunk_rows)
    _check_frame(frame, "predict")
    return DataFrameOnlineNamespace(frame).predict(bank)


def _check_frame(frame: object, what: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        msg = f"online.{what} takes a polars DataFrame or LazyFrame, got {type(frame).__name__}"
        raise TypeError(msg)
