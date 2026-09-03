"""The bank as a polars source: ``lf.online.fit_predict(specs)`` (ENHANCEMENTS E33).

A ``LazyFrame`` in, a ``LazyFrame`` out. Executing the plan streams the input
through a fresh :class:`ModelBank` in ``chunk_rows`` chunks, so a query with
the bank in it is O(chunk) in memory however long the stream is -- where the
expression form, ``pl.col("y").online.<model>(...)``, in the same query is
O(data): polars calls a user expression once with its whole column, and its
streaming engine collects the column to do so (docs/PERFORMANCE.md section
11; the expression warns about it, :mod:`polars_online._expr`). This is
polars' IO-plugin mechanism (``polars.io.plugins.register_io_source``): the
bank is registered as a *source*, the kind of node the engine pulls batches
from, and what comes after it -- filters, selects, joins, ``sink_parquet`` --
is polars' own.

The plan is pure: every execution starts from the same state (the specs'
initial state, or ``load_state``, read when the plan is built), so collecting
twice gives the same frame. ``save_state`` writes the state the execution
ends in -- after the last row the source fed the bank -- atomically, and
because the plan is pure that write is idempotent: polars runs a plan's
source once per execution and twice, concurrently, when one query uses the
plan twice (a self-join, ``pl.concat``, ``pl.collect_all`` of two sinks), and
every run ends in the same state (docs/STATE-WORKFLOW.md).

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


def _bank(specs: Specs | None, load_state: State | None, what: str) -> Callable[[], ModelBank]:
    """How to make the bank a plan starts from, from the specs or a state file.

    A state file is read here, once, when the plan is built -- the plan
    carries the bytes, as ``df.lazy()`` carries the frame -- and each
    execution deserialises them. Read at run time instead, a plan collected
    twice would not be the same frame if the file changed in between, and
    ``load_state=p, save_state=p`` used twice in one query would race the
    second run's load against the first run's write.
    """
    if load_state is not None:
        state = _read_state(load_state)
        return lambda: ModelBank.load_bytes(state, specs)
    if specs is None:
        msg = f"online.{what} needs specs, or load_state= to take them from a saved bank"
        raise ValueError(msg)
    return lambda: ModelBank(specs)


def _read_state(state: State) -> bytes:
    with open(os.fspath(state), "rb") as f:
        return f.read()


def _save_path(save_state: State | None) -> str | None:
    """The ``save_state`` path, checked while the plan is built: a directory
    that is not there is reported now, not after the stream."""
    if save_state is None:
        return None
    path = os.fspath(save_state)
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        msg = f"save_state: {parent!r} is not a directory"
        raise FileNotFoundError(msg)
    return path


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
    save_state: State | None = None,
) -> pl.LazyFrame:
    """``lf`` streamed through ``step`` on a bank from ``make_bank``, as a plan."""
    if chunk_rows is not None and chunk_rows < 1:
        msg = f"chunk_rows must be at least 1, got {chunk_rows}"
        raise ValueError(msg)
    rows = chunk_rows or _native.default_chunk_rows()
    save_path = _save_path(save_state)
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
        for chunk in plan.collect_batches(chunk_size=rows, maintain_order=True):
            if n_rows is not None:
                chunk = chunk.head(n_rows - seen)
            out = step(bank, chunk)
            seen += chunk.height
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            yield out
            if n_rows is not None and seen >= n_rows:
                break
        # Reached only when the source has fed the bank its last row: the
        # input's end, or the rows a `head(n)` asked for. Not in a `finally`:
        # a run the caller abandons is closed whenever polars drops it -- on
        # some versions when the plan object goes -- and a run the bank
        # ended with an error never gets here, so the file, if any, stands.
        # A node after this one failing does not stop this one (polars
        # drains a Python source first), so the state is written even then;
        # `po.run` saves after its output is committed, for callers who need
        # the two tied together.
        if save_path is not None:
            bank.save(save_path)

    return register_io_source(source, schema=schema, validate_schema=True)


def _fit_predict_lazy(
    lf: pl.LazyFrame,
    specs: Specs | None,
    load_state: State | None,
    save_state: State | None,
    chunk_rows: int | None,
) -> pl.LazyFrame:
    specs = list(specs) if specs is not None else None
    return _source(
        lf, _bank(specs, load_state, "fit_predict"), ModelBank.fit_predict, chunk_rows, save_state
    )


def _predict_lazy(
    lf: pl.LazyFrame, bank: ModelBank | State, chunk_rows: int | None
) -> pl.LazyFrame:
    if not isinstance(bank, ModelBank):
        return _source(lf, _bank(None, bank, "predict"), ModelBank.predict, chunk_rows)

    def own() -> ModelBank:
        # `predict` leaves a bank as it was, so the caller's own is safe to
        # share with the plan; it scores as the bank stands when the plan runs.
        return bank

    return _source(lf, own, ModelBank.predict, chunk_rows)


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
        save_state: State | None = None,
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
        changes what the bank learns from -- filter *before* to do that -- a
        selection is read from the input, so a wide scan reads only the
        columns the specs and the query need, and ``head(n)`` feeds the bank
        the first ``n`` rows and no more.

        ``specs`` are the bank's, or ``load_state`` names a saved bank to
        resume from (with ``specs``, they are checked against the file). The
        file is read when the plan is built, so the plan carries that state:
        each execution starts from it afresh, and a plan collected twice
        gives the same frame. ``save_state`` writes the state the execution
        ends in -- after the last row the source fed the bank -- to that path
        when it ends, atomically (:meth:`ModelBank.save`), so the file is the
        old state or the new one and never half of either; ``load_state`` and
        ``save_state`` may be the same path. Because the plan is pure the
        write is the same whenever it happens: a plan used twice in one query
        (a self-join, ``pl.concat``, ``pl.collect_all`` of two sinks) runs
        twice and writes the same bytes twice. Nothing is written unless the
        source reaches the last row: a run abandoned before then, or one the
        bank ended with an error, leaves the file as it was; a node *after*
        the bank failing does not stop the bank, so the state is written then
        (docs/STATE-WORKFLOW.md) -- :func:`polars_online.run` saves only after
        its output is committed, for the case where the two must be tied
        together, and a dated ``save_state`` per batch of data keeps a rerun
        from learning it twice.

        What the schema decides is reported while the plan is built, as
        polars reports its own schema errors: ``ValueError`` for neither
        ``specs`` nor ``load_state``, for ``chunk_rows`` below 1, for a spec
        the bank refuses, and for a spec whose column the plan has not got,
        is not numeric, or shares the spec's name (the checks of
        :class:`ModelBank` and :meth:`ModelBank.fit_predict`, with the same
        messages); ``FileNotFoundError`` for a ``load_state`` that is not
        there or a ``save_state`` whose directory is not, ``ValueError`` for
        a ``load_state`` that is not a bank this build loads or whose specs
        are not ``specs`` (:meth:`ModelBank.load`). What only the values
        decide -- a null clock, a negative weight, a clock running backwards
        -- is reported when the plan runs, as polars' ``ComputeError``
        carrying the bank's message, and so is a ``save_state`` that cannot
        be written when the run ends, carrying the ``OSError``'s message and
        the path.
        """
        return _fit_predict_lazy(self._lf, specs, load_state, save_state, chunk_rows)

    def predict(self, bank: ModelBank | State, *, chunk_rows: int | None = None) -> pl.LazyFrame:
        """The plan's rows scored against ``bank`` as it stands, learning nothing.

        Each row gets :meth:`ModelBank.predict`'s struct: what the bank would
        report for it as the next row of its group's stream, from the current
        state, which the plan never moves. ``bank`` is a :class:`ModelBank`
        (scored as it stands each time the plan runs; ``predict`` leaves it
        untouched, so sharing it with a plan is safe) or a path to a saved
        state, read when the plan is built -- build the plan again to pick up
        a newer file. Target columns are optional, as for ``predict``;
        ``chunk_rows`` is the read chunk.

        Reported while the plan is built: ``FileNotFoundError`` for a path
        that is not there and ``ValueError`` for a file that is not a bank
        this build loads (:meth:`ModelBank.load`), ``TypeError`` for a
        ``bank`` that is neither a bank nor a path, ``ValueError`` for
        ``chunk_rows`` below 1 and for a column the bank reads that the plan
        has not got or that is not numeric (a missing target is fine). A
        value the bank refuses -- a null clock, a negative weight -- is
        reported when the plan runs, as polars' ``ComputeError`` carrying
        :meth:`ModelBank.predict`'s message.
        """
        return _predict_lazy(self._lf, bank, chunk_rows)


@pl.api.register_dataframe_namespace("online")
class DataFrameOnlineNamespace:
    """A model bank over the frame's rows, in one call."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def fit_predict(
        self,
        specs: Specs | None = None,
        *,
        load_state: State | None = None,
        save_state: State | None = None,
    ) -> pl.DataFrame:
        """``ModelBank(specs).fit_predict(df)`` -- the frame plus one struct
        column per spec, from a bank that is then dropped, or saved to
        ``save_state`` first (:meth:`ModelBank.save`); ``load_state`` starts
        it from a saved bank instead of the specs. Keep a bank of your own to
        feed it more rows.

        Raises what :class:`ModelBank`, :meth:`ModelBank.fit_predict`,
        :meth:`ModelBank.load` and :meth:`ModelBank.save` raise, and
        ``ValueError`` for neither ``specs`` nor ``load_state``. A
        ``save_state`` whose directory is not there is ``FileNotFoundError``
        before the fit, not after it."""
        save_path = _save_path(save_state)
        bank = _bank(specs, load_state, "fit_predict")()
        out = bank.fit_predict(self._df)
        if save_path is not None:
            bank.save(save_path)
        return out

    def predict(self, bank: ModelBank | State) -> pl.DataFrame:
        """:meth:`ModelBank.predict` over the frame: scored against ``bank`` --
        a :class:`ModelBank`, or the path of a saved one -- as it stands, which
        does not move. Raises what :meth:`ModelBank.load` (for a path) and
        :meth:`ModelBank.predict` raise, and ``TypeError`` for a ``bank`` that
        is neither."""
        if not isinstance(bank, ModelBank):
            bank = ModelBank.load(os.fspath(bank))
        return bank.predict(self._df)


@overload
def fit_predict(
    frame: pl.LazyFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    save_state: State | None = None,
    chunk_rows: int | None = None,
) -> pl.LazyFrame: ...


@overload
def fit_predict(
    frame: pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    save_state: State | None = None,
    chunk_rows: int | None = None,
) -> pl.DataFrame: ...


def fit_predict(
    frame: pl.LazyFrame | pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    save_state: State | None = None,
    chunk_rows: int | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """``frame.online.fit_predict(...)`` as a plain function, so that a type checker can see it.

    A ``LazyFrame`` gives a plan that streams the rows through a bank when it
    runs (:meth:`LazyFrameOnlineNamespace.fit_predict`); a ``DataFrame`` gives
    the frame with the bank's columns
    (:meth:`DataFrameOnlineNamespace.fit_predict`). ``load_state`` starts the
    bank from a saved one and ``save_state`` writes where it ends up;
    ``chunk_rows`` is the plan's read chunk, and a frame already in memory is
    fitted in one call. ``TypeError`` for a ``frame`` that is neither;
    otherwise raises what the namespace method does.
    """
    if isinstance(frame, pl.LazyFrame):
        return _fit_predict_lazy(frame, specs, load_state, save_state, chunk_rows)
    _check_frame(frame, "fit_predict")
    return DataFrameOnlineNamespace(frame).fit_predict(
        specs, load_state=load_state, save_state=save_state
    )


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
    """``frame.online.predict(bank)`` as a plain function, so that a type checker can see it.

    Scores the rows against ``bank`` as it stands and learns nothing: a plan
    from a ``LazyFrame`` (:meth:`LazyFrameOnlineNamespace.predict`), a frame
    from a ``DataFrame`` (:meth:`DataFrameOnlineNamespace.predict`).
    ``TypeError`` for a ``frame`` that is neither; otherwise raises what the
    namespace method does.
    """
    if isinstance(frame, pl.LazyFrame):
        return _predict_lazy(frame, bank, chunk_rows)
    _check_frame(frame, "predict")
    return DataFrameOnlineNamespace(frame).predict(bank)


def _check_frame(frame: object, what: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        msg = f"online.{what} takes a polars DataFrame or LazyFrame, got {type(frame).__name__}"
        raise TypeError(msg)
