"""The bank as a polars source: ``lf.online.fit_predict(specs)``.

A ``LazyFrame`` in, a ``LazyFrame`` out. When the plan runs, its rows go
through a bank that starts with nothing learned, one chunk at a time, so a
query with the bank in it is O(chunk) in memory however long the stream is.
The bank is registered as a polars source, the kind of node the engine pulls
batches from, and what comes after it -- filters, selects, joins, writing the
result to a file -- is polars' own. A filter after the bank never changes what
the bank learns from; one that should belongs before it.

The plan is pure: every execution starts from the same state (the specs'
initial state, or ``load_state``, read when the plan is built), so collecting
twice gives the same frame. ``save_state`` writes the state the execution ends
in, after the last row the source fed the bank, atomically. Because the plan
is pure that write is idempotent: polars runs a plan's source once per
execution, and twice, concurrently, when one query uses the plan twice (a
self-join, ``pl.concat``, ``pl.collect_all`` of two sinks), and every run ends
in the same state.

``df.online.fit_predict(specs)`` is the eager twin,
``ModelBank(specs).fit_predict(df)`` in one call. ``online.unnest(specs)``
takes a bank's output apart: each spec's struct column becomes its fields as
columns, with the ``coef`` list as one named column per coefficient
(:func:`polars_online.spec.coef_fields`). Both namespaces are attached at
import, which no type checker can see; :func:`fit_predict`, :func:`predict`
and :func:`unnest` are the same calls with the frame as the first argument,
visibly typed.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator
from types import FrameType
from typing import Any, overload

import polars as pl
from polars.io.plugins import register_io_source

from polars_online import _polars_online as _native
from polars_online._bank import ModelBank
from polars_online._spec import coef_fields, output_index

__all__ = [
    "ConsumedSourceWarning",
    "DataFrameOnlineNamespace",
    "LazyFrameOnlineNamespace",
    "OrderNotGuaranteedWarning",
    "fit_predict",
    "predict",
    "unnest",
]

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


_WRITE: dict[str, str] = {
    "parquet": "write_parquet",
    "ipc": "write_ipc",
    "csv": "write_csv",
    "ndjson": "write_ndjson",
}


def _closed_path(closed_groups: State | None, specs: Iterable[dict[str, Any]]) -> str | None:
    """The ``closed_groups`` sidecar path, checked while the plan is built:
    a directory that is not there, an extension that names no format, and a
    bank with nothing to close are all reported now (E54)."""
    if closed_groups is None:
        return None
    path = os.fspath(closed_groups)
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        msg = f"closed_groups: {parent!r} is not a directory"
        raise FileNotFoundError(msg)
    # Raises ValueError naming the extensions it knows.
    _native.format_of_path(path)
    if not any(spec.get("group_close") for spec in specs):
        msg = (
            "closed_groups names a file but no spec closes groups; add "
            'group_close = "monotone" or "session" to the spec whose groups should be emitted'
        )
        raise ValueError(msg)
    return path


def _write_closed(path: str, frames: list[pl.DataFrame], schema: pl.DataFrame) -> None:
    """The drained closed rows as one file, in the format the extension
    names. An empty run writes the empty frame with the schema, as an empty
    output does."""
    df = pl.concat(frames) if frames else schema
    getattr(df, _WRITE[_native.format_of_path(path)])(path)


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
    out: dict[str, Any] = {
        "explain_name": "polars-online",
        "explain_detail": f"{len(names)} spec(s): {detail}" if names else "no specs",
    }
    return out


def _spec_columns(specs: Iterable[dict[str, Any]]) -> set[str]:
    cols: set[str] = set()
    for spec in specs:
        for key in _COLUMN_KEYS:
            cols.update(spec.get(key) or ())
        for key in _SCALAR_COLUMN_KEYS:
            if spec.get(key) is not None:
                cols.add(spec[key])
    return cols


class OrderNotGuaranteedWarning(UserWarning):
    """A plan handed to a bank has a row order polars does not guarantee.

    An online model learns in row order, so the order a plan delivers is part of
    the model. :meth:`ModelBank.fit`, :meth:`ModelBank.fit_predict_batches` and
    ``lf.online.fit_predict`` run a plan through polars' streaming engine, and a
    ``join``, ``group_by`` or ``unique`` without an order guarantee delivers a
    different stream there than ``lf.collect()`` gives. Measured on 200,000
    rows: ``collect()`` kept the input order and ``collect_batches()`` did not,
    so a plan checked by collecting it learns something else when it is fed.
    The order may also differ between runs.

    The warning names each such node and its fix: ``maintain_order="left"`` on
    a join, ``maintain_order=True`` on a ``group_by``, a ``sort`` after a
    ``unique`` (the streaming engine does not honour its ``maintain_order``), or
    a sort before the bank.

    Best-effort by design. The plan is read through ``LazyFrame.serialize``,
    a format polars has deprecated, and a plan polars cannot serialize -- one
    already holding a bank, for instance -- is let through in silence: the
    inspection never fails a run. For a plan whose order is fixed by other
    means, ``warnings.simplefilter("ignore", polars_online.OrderNotGuaranteedWarning)``.

    **Not raised by** :meth:`ModelBank.fit` **when every spec is an accumulator
    with no decay** -- ``ewridge``, ``rls``, ``huber`` or ``lasso`` at
    ``lam=1.0``, no ``window``, no session, no drift reset. Those sums commute,
    so the state after that fit is the same whatever the order (to rounding:
    3.3e-16 over 200 rows), and ``fit`` keeps only the state. The exception is
    that narrow on purpose. It does not extend to ``fit_predict_batches`` or to
    the plan form, whose predictions are out-of-sample and so move with the
    order even where the coefficients do not (1.33 on the same rows); nor to a
    model whose update does not commute, which is every other one -- ``sgd``,
    ``pa``, ``ftrl`` and ``quantile`` all differ materially with no decay at
    all, so "no halflife" is not by itself a reason to expect order not to
    matter. A spec option the check does not recognise counts as unsafe, so a
    key added later cannot quietly become exempt.
    """


class ConsumedSourceWarning(UserWarning):
    """A plan that reads a single-use source fed a bank no rows at all.

    An Arrow C stream is consumed once: the specification says a capsule "can
    only be consumed once", and a producer hands its buffers away. So a
    ``LazyFrame`` built by ``pl.scan_arrow_c_stream(...)`` -- over a DuckDB
    relation, a ``pyarrow.RecordBatchReader``, anything exposing
    ``__arrow_c_stream__`` -- works the first time it is collected and then
    yields **nothing**, silently, with no error from polars. A bank fed the
    same plan twice therefore learns from every row and then from none, and
    the second run leaves a state that looks finished and is empty.

    This warns rather than raises, because a source may legitimately have no
    rows -- an empty query is not a mistake. It is raised only for a plan
    whose source is a Python scan *and* which yielded nothing, which is the
    shape a consumed stream has; an in-memory frame's ``.lazy()`` and a
    ``scan_parquet`` cannot trip it, and both are safely reusable.

    The fix is to rebuild the plan per run -- ``pl.scan_arrow_c_stream(con.sql(q))``
    inside the loop rather than outside it -- since the producer will make a
    fresh stream.
    """


def _is_python_scan(lf: pl.LazyFrame) -> bool:
    """Whether ``lf``'s source is a Python scan, which is what an Arrow C
    stream and this package's own plan form both are.

    Read from ``explain``, which does not execute the plan and so cannot
    consume the very stream this is here to protect (checked: a plan still
    yields its rows after being explained). Any failure to read the plan is
    "not a python scan", so the guard never turns a working run into an
    error."""
    try:
        return "PYTHON SCAN" in lf.explain(optimized=False)
    except Exception:  # a plan that cannot be explained is not one to guard
        return False


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
    walk stops there; a join that keeps one side's order is followed on that
    side alone. Anything unrecognised is descended into generically."""
    if isinstance(node, list):
        for item in node:
            _walk(item, found)
        return
    if not isinstance(node, dict):
        return
    if len(node) == 1:
        ((tag, body),) = node.items()
        if tag == "Sort":
            return
        if tag == "Join" and isinstance(body, dict):
            args = body.get("options", {}).get("args", {})
            mode = str(args.get("maintain_order", "None")).lower()
            if mode == "none":
                found.append(
                    'a join without maintain_order (pass maintain_order="left" to keep the '
                    "order of the left input)"
                )
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
                "a unique(): the streaming engine does not honour maintain_order here, "
                "so sort after it"
            )
            _walk(body.get("input"), found)
            return
    for value in node.values():
        _walk(value, found)


def _order_hazards(lf: pl.LazyFrame) -> list[str]:
    """The nodes of ``lf`` whose output order is unspecified, each as a phrase
    naming the fix; ``[]`` when there are none -- or when the plan cannot be
    read, since this inspects a deprecated polars format and is best-effort by
    design, never a reason for a run to fail."""
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
#: Measured over 200 rows with no decay: ``ewridge`` 3.3e-16, ``rls`` 8.9e-16,
#: ``huber`` 6.7e-16, ``lasso`` 7.8e-16. Every other model moves materially
#: even with no decay at all, because its update is not commutative --
#: ``sgd`` 5.9e-03, ``pa`` 5.1e-02, ``ftrl`` 3.3e-02, ``quantile`` 2.8e-03 --
#: so "no halflife" is not on its own a reason to expect order not to matter.
_ORDER_FREE_MODELS = frozenset({"ew_ridge", "rls", "huber", "lasso"})

#: Spec keys that change the *path* a fit takes, with the only values that
#: leave it order-free. Measured on the same rows: ``window`` 8.3e-03,
#: ``gram_block_rows`` 6.3e-04 (which row sits in the pending block when a
#: solve fires depends on arrival order), ``label_delay`` 4.3e-04, and
#: ``drift_action="reset"`` **8.9e-01** -- the largest of all, and the one that
#: read as harmless until the fixture actually made drift fire. The rest are
#: denied as *unproven* rather than refuted: their code path never ran in the
#: probe, and a green result from a path that did not execute is not evidence.
_ORDER_FREE_ONLY_WHEN: dict[str, tuple[Any, ...]] = {
    "lam": (1.0,),
    "halflife": (None,),
    "window": (None,),
    "window_budget": (None,),
    "window_every": (None,),
    "label_delay": (None,),
    "gram_block_rows": (None,),
    "drift_action": ("flag",),
    "session": (None,),
    "session_gap": (None,),
    "session_shrink": (None,),
    "long_halflife": (None,),
    "ridge_decay": (False,),
    "conformal": (None,),
    "conformal_rate": (None,),
    "average_eta": (None,),
    "emit_averaged": (False,),
    "group_close": (None,),
    "weight": (None,),
    "on_clock_reset": ("max",),
    "target_gaps": ("own_rows",),
    "coef_every": (0,),
    "emit_autocorr": (False,),
    "emit_metrics": (False,),
    "emit_resid_z": (False,),
    "emit_selected": (False,),
    "emit_sigma": (False,),
    "resid_autocorr_lag": (None,),
    "resid_quantiles": (None,),
}

#: Keys free to hold any value without touching order-freeness: what the fit
#: reads, how it solves, and (for ``emit_drift``, measured at 3.3e-16 with
#: ``drift_action`` left at ``"flag"``) what it reports. ``group`` is here
#: because each group accumulates independently, and ``standardize`` and
#: ``select_halflife`` because both measured at rounding.
_ORDER_FREE_ANY = frozenset(
    {
        "name",
        "type",
        "targets",
        "features",
        "feature_sets",
        "min_periods",
        "group",
        "clock",
        "max_dclock",
        "add_intercept",
        "standardize",
        "ridge",
        "coef_prior",
        "solve_every",
        "max_rows_between_solves",
        "huber_delta",
        "cd_tol",
        "l1_ratio",
        "lasso_path",
        "max_cd_iters",
        "select_halflife",
        "emit_drift",
        "drift_delta",
        "drift_threshold",
    }
)


def _spec_items(spec: Any) -> Any:
    """Every ``(key, value)`` of a spec, with the nested ``model`` dict walked
    in place -- ``window`` lives *there* for ``ewridge`` and ``lasso``, not at
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
    prediction even where the coefficients commute (measured: 1.33 on a
    no-decay ``ewridge`` whose coefficients agreed to 3.3e-16)."""
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
            elif key not in _ORDER_FREE_ANY:
                return False
    return True


def _warn_if_order_unspecified(lf: pl.LazyFrame, what: str) -> None:
    """Warn, naming ``what`` the caller called, when ``lf`` has a node whose
    output order is not guaranteed."""
    hazards = _order_hazards(lf)
    if not hazards:
        return
    msg = (
        f"{what}: the plan's row order is not guaranteed, and an online model learns in "
        f"row order. {len(hazards)} node(s) leave it unspecified: " + "; ".join(hazards) + ". "
        "The streaming engine that runs the plan may deliver rows in an order lf.collect() "
        "would not, and it may differ between runs. Give the plan a fixed order -- "
        'maintain_order="left" on a join, maintain_order=True on a group_by, a sort after '
        "a unique() or before the bank -- or, for a plan whose order is fixed by other "
        'means, warnings.simplefilter("ignore", polars_online.OrderNotGuaranteedWarning).'
    )
    warnings.warn(OrderNotGuaranteedWarning(msg), stacklevel=_user_stacklevel())


def _source(
    lf: pl.LazyFrame,
    make_bank: Callable[[], ModelBank],
    step: Callable[[ModelBank, pl.DataFrame], pl.DataFrame],
    chunk_rows: int | None,
    save_state: State | None = None,
    closed_groups: State | None = None,
) -> pl.LazyFrame:
    """``lf`` streamed through ``step`` on a bank from ``make_bank``, as a plan."""
    if chunk_rows is not None and chunk_rows < 1:
        msg = f"chunk_rows must be at least 1, got {chunk_rows}"
        raise ValueError(msg)
    # At build time, before anything runs: the order the plan will deliver is
    # decided here, and a caller should hear about it before the first chunk.
    called = f"lf.online.{getattr(step, '__name__', 'fit_predict')}"
    _warn_if_order_unspecified(lf, called)
    # Also decided here, off the plan as handed over: whether its source is a
    # Python scan, which is the shape `pl.scan_arrow_c_stream` has and so the
    # shape a single-use stream arrives in. Read now because `explain` must
    # see the caller's plan, and acted on below only if the run sees no rows.
    python_scan = _is_python_scan(lf)
    rows = chunk_rows or _native.default_chunk_rows()
    save_path = _save_path(save_state)
    in_schema = lf.collect_schema()
    # The output schema, from a bank run on no rows. This is also where a spec
    # naming a column the input lacks is reported: while the plan is built,
    # as polars reports its own schema errors, not when it runs.
    bank = make_bank()
    schema = step(bank, pl.DataFrame(schema=in_schema)).schema
    needed = _spec_columns(bank.specs)
    closed_path = _closed_path(closed_groups, bank.specs)
    # Peeked, not drained: `make_bank` can hand back the caller's own bank
    # (`lf.online.predict(bank)`), whose queue building a plan must leave
    # alone (review 2026-09-12, C4).
    closed_schema = bank.closed_groups(drop=False).clear()

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
        closed: list[pl.DataFrame] = []
        for chunk in plan.collect_batches(chunk_size=rows, maintain_order=True):
            if n_rows is not None:
                chunk = chunk.head(n_rows - seen)
            out = step(bank, chunk)
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
        # the CLI saves after its output is committed, for callers who need
        # the two tied together.
        if closed_path is not None:
            _write_closed(closed_path, closed, closed_schema)
        if save_path is not None:
            bank.save(save_path)
        # Last, so a state file is written before anything is said about it.
        # `n_rows == 0` is a `head(0)` pushed into the scan, where no rows is
        # what the query asked for rather than a stream that is spent.
        if python_scan and seen == 0 and n_rows != 0:
            warnings.warn(
                ConsumedSourceWarning(
                    f"{called}: the plan yielded no rows, and its source is a Python scan -- "
                    "which is what `pl.scan_arrow_c_stream` builds over a DuckDB relation, a "
                    "pyarrow reader, or anything exposing `__arrow_c_stream__`. Such a stream "
                    "is consumed once: collected a second time it yields nothing, with no "
                    "error, so this bank has learned from nothing. Rebuild the plan per "
                    "collect (call `pl.scan_arrow_c_stream(...)` inside the loop, not outside "
                    "it). If the source really is empty, silence this with "
                    'warnings.simplefilter("ignore", polars_online.ConsumedSourceWarning).'
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
    load_state: State | None,
    save_state: State | None,
    chunk_rows: int | None,
    closed_groups: State | None = None,
) -> pl.LazyFrame:
    specs = list(specs) if specs is not None else None
    return _source(
        lf,
        _bank(specs, load_state, "fit_predict"),
        ModelBank.fit_predict,
        chunk_rows,
        save_state,
        closed_groups,
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


def _specs_of(specs: Specs | ModelBank | State, what: str) -> list[dict[str, Any]]:
    """The spec dicts behind ``specs``: a list of them, a bank's, or a saved
    bank's (the file carries them; it is read here, once)."""
    if isinstance(specs, ModelBank):
        return specs.specs
    if isinstance(specs, (str, os.PathLike)):
        return ModelBank.load(os.fspath(specs)).specs
    out = list(specs)
    if not all(isinstance(spec, dict) for spec in out):
        msg = f"online.{what} takes specs, a ModelBank or the path of a saved one"
        raise TypeError(msg)
    return out


def _unnest_exprs(schema: pl.Schema, specs: list[dict[str, Any]]) -> list[pl.Expr]:
    """The columns of ``schema`` with each spec's struct replaced, in place, by
    its fields -- the ``coef`` lists as one column per coefficient, named by
    :func:`polars_online.spec.coef_fields`.

    A spec whose column the frame has not got, or whose struct lacks a field
    the spec produces, is reported here, while the plan is built.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec["name"] in by_name:
            msg = f"online.unnest: spec {spec['name']!r} given twice"
            raise ValueError(msg)
        by_name[spec["name"]] = spec
    exprs: list[pl.Expr] = []
    for column, dtype in schema.items():
        if column not in by_name:
            exprs.append(pl.col(column))
            continue
        spec = by_name.pop(column)
        if not isinstance(dtype, pl.Struct):
            msg = f"online.unnest: column {column!r} is {dtype}, not spec {column!r}'s struct"
            raise ValueError(msg)
        fields = [f.name for f in dtype.fields]
        idx = output_index(spec)
        missing = [f for f in idx["field"] if f not in fields]
        if missing:
            msg = (
                f"online.unnest: column {column!r} lacks the field(s) spec {column!r} "
                f"produces: {', '.join(missing)}"
            )
            raise ValueError(msg)
        coefs = coef_fields(spec)
        # A `coef` with no named positions (`micro`: one row per established
        # summary, as many as there are) stays the list it is.
        lists = set(idx.filter(pl.col("kind") == "coef")["field"]) & set(coefs["field"])
        for field in fields:
            col = pl.col(column).struct.field(field)
            if field not in lists:
                exprs.append(col)
                continue
            for position, name in (
                coefs.filter(pl.col("field") == field).select("position", "name").iter_rows()
            ):
                exprs.append(col.list.get(position, null_on_oob=False).alias(name))
    if by_name:
        msg = f"online.unnest: the frame has no column(s) {', '.join(map(repr, by_name))}"
        raise ValueError(msg)
    return exprs


def _unnest_lazy(lf: pl.LazyFrame, specs: Specs | ModelBank | State) -> pl.LazyFrame:
    return lf.select(_unnest_exprs(lf.collect_schema(), _specs_of(specs, "unnest")))


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
        closed_groups: State | None = None,
        chunk_rows: int | None = None,
    ) -> pl.LazyFrame:
        """The plan's rows plus one struct column per spec, learning as it goes.

        Executing the returned plan (``collect()``, ``collect_batches()``,
        ``sink_parquet()`` and the rest) streams this plan's rows through a new
        :class:`ModelBank` in ``chunk_rows`` chunks, so memory is O(chunk + state)
        whatever the length of the stream. ``chunk_rows`` defaults to 100,000, and
        chunking never changes the numbers, only ``coef``'s reporting cadence. Rows
        must arrive in stream order, as for the bank. The struct columns are what
        :meth:`ModelBank.fit_predict` writes: one per spec, named after it, with the
        fields :mod:`polars_online.spec` describes under *What a spec writes*.

        .. code-block:: python

            fitted = lf.online.fit_predict([spec], save_state="fit.state").collect()
            # one column per field, coef as one column per coefficient
            flat = lf.online.fit_predict([spec]).online.unnest([spec]).collect()

        Filters, selections and ``head`` applied after are pushed into the source. A
        filter never changes what the bank learns from; filter before to do that. A
        selection is read from the input, so a wide scan reads only the columns the
        specs and the query need. ``head(n)`` feeds the bank the first ``n`` rows and
        no more.

        ``specs`` are the bank's, or ``load_state`` names a saved bank to resume from
        (with ``specs``, they are checked against the file). The file is read when the
        plan is built, so the plan carries that state: each execution starts from it
        afresh, and a plan collected twice gives the same frame. ``save_state`` writes
        the state the execution ends in, after the last row the source fed the bank,
        to that path when it ends. The write is atomic (:meth:`ModelBank.save`), so
        the file is the old state or the new one and never half of either, and
        ``load_state`` and ``save_state`` may be the same path. Because the plan is
        pure the write is the same whenever it happens: a plan used twice in one query
        (a self-join, ``pl.concat``, ``pl.collect_all`` of two sinks) runs twice and
        writes the same bytes twice. Nothing is written unless the source reaches the
        last row: a run abandoned before then, or one the bank ended with an error,
        leaves the file as it was. A node after the bank failing does not stop the
        bank, so the state is written then. The ``online`` CLI saves only after
        its output is committed, for the case where the two must be tied together, and
        a dated ``save_state`` per batch of data keeps a rerun from learning it twice.

        The plan runs through polars' streaming engine, and an online model learns
        in row order, so the order the engine delivers is part of the result. A
        ``join``, ``group_by`` or ``unique`` without an order guarantee delivers a
        different stream there than ``lf.collect()`` gives, and may differ between
        runs; give a join ``maintain_order="left"``, a ``group_by``
        ``maintain_order=True``, and sort after a ``unique``, or sort before the
        bank. A plan with such a node raises :class:`OrderNotGuaranteedWarning`
        when it is built, naming the node.

        ``closed_groups`` writes the groups that finished during the run to a sidecar
        file, in the format its extension names (:meth:`ModelBank.closed_groups`). The
        queue is drained after every chunk, so the bank stays bounded. The one file is
        written where ``save_state`` is written and under the same rules: only when
        the source reaches its last row, once per execution of the plan, and twice
        with the same bytes for a plan used twice in one query. It needs a spec with
        ``group_close``; a run in which nothing closed writes an empty frame with the
        schema.

        What the schema decides is reported while the plan is built, as polars reports
        its own schema errors. ``ValueError`` for neither ``specs`` nor
        ``load_state``, for ``chunk_rows`` below 1, for a spec the bank refuses, and
        for a spec whose column the plan has not got, is not numeric, or shares the
        spec's name (the checks of :class:`ModelBank` and
        :meth:`ModelBank.fit_predict`, with the same messages). ``FileNotFoundError``
        for a ``load_state`` that is not there or a ``save_state`` whose directory is
        not. ``ValueError`` for a ``load_state`` that is not a bank this build loads
        or whose specs are not ``specs`` (:meth:`ModelBank.load`). What only the
        values decide (a null clock, a negative weight, a clock running backwards) is
        reported when the plan runs, as polars' ``ComputeError`` carrying the bank's
        message. So is a ``save_state`` that cannot be written when the run ends,
        carrying the ``OSError``'s message and the path.
        """
        return _fit_predict_lazy(self._lf, specs, load_state, save_state, chunk_rows, closed_groups)

    def predict(self, bank: ModelBank | State, *, chunk_rows: int | None = None) -> pl.LazyFrame:
        """The plan's rows scored against ``bank`` as it stands, learning nothing.

        Each row gets :meth:`ModelBank.predict`'s struct: what the bank would report
        for it as the next row of its group's stream, from the current state, which
        the plan never moves. ``bank`` is a :class:`ModelBank`, scored as it stands
        each time the plan runs (``predict`` leaves it untouched, so sharing it with a
        plan is safe), or a path to a saved state, read when the plan is built. Build
        the plan again to pick up a newer file. Target columns are optional, as for
        ``predict``; ``chunk_rows`` is the read chunk.

        .. code-block:: python

            scored = lf.online.predict("bank.state").collect()    # the saved bank, unmoved

        Reported while the plan is built: ``FileNotFoundError`` for a path that is not
        there; ``ValueError`` for a file that is not a bank this build loads
        (:meth:`ModelBank.load`); ``TypeError`` for a ``bank`` that is neither a bank
        nor a path; ``ValueError`` for ``chunk_rows`` below 1, and for a column the
        bank reads that the plan has not got or that is not numeric (a missing target
        is fine). A value the bank refuses (a null clock, a negative weight) is
        reported when the plan runs, as polars' ``ComputeError`` carrying
        :meth:`ModelBank.predict`'s message.
        """
        return _predict_lazy(self._lf, bank, chunk_rows)

    def unnest(self, specs: Specs | ModelBank | State) -> pl.LazyFrame:
        """The plan with each spec's struct column taken apart into columns.

        ``lf.unnest(names)`` with the ``coef`` lists taken apart too: every scalar
        field becomes a column of its own name (``pred_y``, ``n_eff@h500``), and each
        ``coef`` list becomes one column per coefficient, named
        ``coef_{target}_{term}{combo}{instance}`` (``coef_y_intercept``,
        ``coef_y_x1__r0.5@h500``) as :func:`polars_online.spec.coef_fields` lists
        them. The columns take the struct's place; the rest of the frame, and any spec
        column not named, are left as they are. ``specs`` is the spec dicts, a
        :class:`ModelBank` (its specs), or the path of a saved bank (which carries
        them). So a scored plan, or a parquet the CLI wrote, comes back flat:

        .. code-block:: python

            betas = (
                lf.online.fit_predict([spec])
                .online.unnest([spec])
                .select("t", "^coef_.*$")
                .collect()
            )

        Reported while the plan is built: ``ValueError`` for a spec whose column the
        plan has not got, is not a struct, or lacks a field the spec produces, for a
        spec given twice and for a spec that is not valid; ``TypeError`` for ``specs``
        that are none of the three; ``FileNotFoundError`` and :meth:`ModelBank.load`'s
        ``ValueError`` for a path. Two specs that produce a field of the same name
        unnest to the same column name, which polars reports as its
        ``DuplicateError``: unnest them one at a time, or rename the struct's fields
        first (``pl.col("m").name.prefix_fields("m_")``).
        """
        return _unnest_lazy(self._lf, specs)


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
        closed_groups: State | None = None,
    ) -> pl.DataFrame:
        """``ModelBank(specs).fit_predict(df)`` in one call: the frame plus one struct
        column per spec, from a bank that is then dropped.

        ``save_state`` saves the bank first (:meth:`ModelBank.save`), and
        ``load_state`` starts it from a saved bank instead of the specs. Keep a bank
        of your own to feed it more rows. ``closed_groups`` writes the groups that
        finished to a sidecar file in the format its extension names, before
        ``save_state`` (:meth:`ModelBank.closed_groups`).

        .. code-block:: python

            out = df.online.fit_predict([spec])

        Raises what :class:`ModelBank`, :meth:`ModelBank.fit_predict`,
        :meth:`ModelBank.load` and :meth:`ModelBank.save` raise, and ``ValueError``
        for neither ``specs`` nor ``load_state``. A ``save_state`` whose directory is
        not there is ``FileNotFoundError`` before the fit, not after it.
        """
        save_path = _save_path(save_state)
        bank = _bank(specs, load_state, "fit_predict")()
        closed_path = _closed_path(closed_groups, bank.specs)
        out = bank.fit_predict(self._df)
        if closed_path is not None:
            # One drain, and its empty frame the schema: two calls were right
            # only by the order Python evaluates arguments in (C4).
            drained = bank.closed_groups()
            _write_closed(closed_path, [drained], drained.clear())
        if save_path is not None:
            bank.save(save_path)
        return out

    def predict(self, bank: ModelBank | State) -> pl.DataFrame:
        """:meth:`ModelBank.predict` over the frame: scored against ``bank``, a
        :class:`ModelBank` or the path of a saved one, as it stands, which does not
        move.

        Raises what :meth:`ModelBank.load` (for a path) and :meth:`ModelBank.predict`
        raise, and ``TypeError`` for a ``bank`` that is neither.
        """
        if not isinstance(bank, ModelBank):
            bank = ModelBank.load(os.fspath(bank))
        return bank.predict(self._df)

    def unnest(self, specs: Specs | ModelBank | State) -> pl.DataFrame:
        """The frame with each spec's struct column taken apart into columns, as
        :meth:`LazyFrameOnlineNamespace.unnest` does for a plan.

        Scalar fields under their own names, each ``coef`` list as one named column
        per coefficient. Raises what the plan form does, on the call.
        """
        return self._df.select(_unnest_exprs(self._df.schema, _specs_of(specs, "unnest")))


@overload
def fit_predict(
    frame: pl.LazyFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    save_state: State | None = None,
    closed_groups: State | None = None,
    chunk_rows: int | None = None,
) -> pl.LazyFrame: ...


@overload
def fit_predict(
    frame: pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    save_state: State | None = None,
    closed_groups: State | None = None,
    chunk_rows: int | None = None,
) -> pl.DataFrame: ...


def fit_predict(
    frame: pl.LazyFrame | pl.DataFrame,
    specs: Specs | None = None,
    *,
    load_state: State | None = None,
    save_state: State | None = None,
    closed_groups: State | None = None,
    chunk_rows: int | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """``frame.online.fit_predict(...)`` as a plain function, so that a type checker
    can see it.

    A ``LazyFrame`` gives a plan that streams the rows through a bank when it runs
    (:meth:`LazyFrameOnlineNamespace.fit_predict`); a ``DataFrame`` gives the
    frame with the bank's columns (:meth:`DataFrameOnlineNamespace.fit_predict`).
    ``load_state`` starts the bank from a saved one and ``save_state`` writes
    where it ends up; ``chunk_rows`` is the plan's read chunk, and a frame already
    in memory is fitted in one call. ``TypeError`` for a ``frame`` that is
    neither; otherwise raises what the namespace method does.
    """
    if isinstance(frame, pl.LazyFrame):
        return _fit_predict_lazy(frame, specs, load_state, save_state, chunk_rows, closed_groups)
    _check_frame(frame, "fit_predict")
    return DataFrameOnlineNamespace(frame).fit_predict(
        specs, load_state=load_state, save_state=save_state, closed_groups=closed_groups
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
    """``frame.online.predict(bank)`` as a plain function, so that a type checker can
    see it.

    Scores the rows against ``bank`` as it stands and learns nothing: a plan from
    a ``LazyFrame`` (:meth:`LazyFrameOnlineNamespace.predict`), a frame from a
    ``DataFrame`` (:meth:`DataFrameOnlineNamespace.predict`). ``TypeError`` for a
    ``frame`` that is neither; otherwise raises what the namespace method does.
    """
    if isinstance(frame, pl.LazyFrame):
        return _predict_lazy(frame, bank, chunk_rows)
    _check_frame(frame, "predict")
    return DataFrameOnlineNamespace(frame).predict(bank)


@overload
def unnest(frame: pl.LazyFrame, specs: Specs | ModelBank | State) -> pl.LazyFrame: ...


@overload
def unnest(frame: pl.DataFrame, specs: Specs | ModelBank | State) -> pl.DataFrame: ...


def unnest(
    frame: pl.LazyFrame | pl.DataFrame, specs: Specs | ModelBank | State
) -> pl.LazyFrame | pl.DataFrame:
    """``frame.online.unnest(specs)`` as a plain function, so that a type checker can
    see it.

    Takes each spec's struct column apart into columns, the ``coef`` lists as one
    named column per coefficient: a plan from a ``LazyFrame``
    (:meth:`LazyFrameOnlineNamespace.unnest`), a frame from a ``DataFrame``
    (:meth:`DataFrameOnlineNamespace.unnest`). ``TypeError`` for a ``frame`` that
    is neither; otherwise raises what the namespace method does.
    """
    if isinstance(frame, pl.LazyFrame):
        return _unnest_lazy(frame, specs)
    _check_frame(frame, "unnest")
    return DataFrameOnlineNamespace(frame).unnest(specs)


def _check_frame(frame: object, what: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        msg = f"online.{what} takes a polars DataFrame or LazyFrame, got {type(frame).__name__}"
        raise TypeError(msg)
