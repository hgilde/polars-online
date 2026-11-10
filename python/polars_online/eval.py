"""Scores of a bank's output, and tests of one spec's residuals against another's.

The scores are out-of-sample R², IC (the correlation of the prediction with
the target) and hit rate per spec, group and target,
over a whole frame or in windows of the clock. :func:`sums` gives the same
scores one chunk at a time, and :func:`grid_search` chooses a grid point from
a stream's own residuals.

Most calls read the frame a :class:`~polars_online.ModelBank` returns: the
input's columns plus one output column per spec, whose value in each row is
a record of the model's fields, or a query of one. :func:`merge_sums` and
:func:`from_sums` read the frames :func:`sums` makes. :func:`seqtest` can
also test the sign of plain numeric columns, and :func:`grid_search` runs a
bank over a query itself. The calls fall into five kinds:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - kind
     - the calls
     - what they hold in memory, given a query
   * - scores over the rows, in windows of the clock, or of several specs
     - :func:`metrics`, :func:`window_metrics`, :func:`compare_specs`
     - one chunk, and ten numbers per slot and key (and window)
   * - scores one chunk at a time
     - :func:`sums`, :func:`merge_sums`, :func:`from_sums`
     - one chunk, and ten numbers per slot and key
   * - one spec against another
     - :func:`diebold_mariano`, :func:`clark_west`, :func:`seqtest`
     - one chunk, and per key and target the last ``lags`` values and a
       few sums
   * - the long form the scores read
     - :func:`unpack`
     - nothing until the query it returns runs, and then one chunk
   * - a grid point chosen from a stream's own residuals
     - :func:`grid_search`
     - three numbers per slot for each run of rows between the places a
       test span starts or its choice stops,
       and the bank's state and one chunk

**Give a call the query to score an output too long to hold.** A query (a
``LazyFrame``) runs only when its result is asked for, and every call that
reads a bank's output reads a query's result one chunk at a time. The
scores keep the ten sums per key that :func:`sums` keeps, added chunk to
chunk as :func:`merge_sums` adds them. The tests of two specs keep sums of
their own, and the last rows of each key, which the next chunk's first rows
pair with. So scoring the file a bank's output was written to,
``pl.scan_parquet("out.parquet")``, holds one chunk and the sums, not the
file, and Polars what it reads of the file ahead of them, which
`Cut what Polars reads ahead
<https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
bounds. A call whose result has a row for each row of the output returns a
query of that result, as Polars' own calls on a query return a query:
:func:`unpack`, :func:`seqtest`, and :func:`diebold_mariano` and
:func:`clark_west` with ``half_life``. The results are the frame's, as each
function's Notes say. Given a frame, a call reads it in memory. To keep
scores beside the model as it runs, set a spec's ``emit_metrics``, which
keeps exponentially weighted R², IC and hit rate, in memory the size of
the model's state.

**Pass the spec itself as ``spec`` where a target is not a plain column.**
``spec`` is the spec's name, which names its output column, or the spec
dict, which names the column too. Only the dict can say where a target's
values are when they are not in a column of the target's name: a column
renamed with :func:`polars_online.target`, or a formula target (see
:func:`unpack`). The dict also says whether a fit is a classifier, whose
scores take other names (:func:`metrics`' ``binary``).

Every call here that reads a bank's output takes a ``LazyFrame`` as well
as a ``DataFrame``, and refuses a query as it refuses the frame, with the
same message, before the query runs. :func:`merge_sums` and
:func:`from_sums` read the sums a call returned, and given a ``LazyFrame``
raise ``TypeError``: pass ``lf.collect()``, which runs the query and returns
its result. The scores
start from :func:`unpack` and raise what it raises. It raises a
:class:`~polars_online.OutputColumnNotFoundError` (a ``KeyError`` too) for a
spec whose output column the frame does not have, ``TypeError`` for a column
that is not a spec's output or is the output of a model that predicts no
target. A slot whose target column cannot be found raises a
:class:`~polars_online.ColumnNotFoundError` when ``spec`` is the spec dict,
and ``ValueError`` when it is the spec's name alone. A ``group``, ``clock`` or
``weight`` column the frame does not have raises a
:class:`~polars_online.ColumnNotFoundError` too. A ``group`` column of the
scores named like a column they make (:data:`RESERVED`, :data:`SUM_FIELDS`,
a score's name or ``window_start``), and a ``clock`` or ``weight`` column
named like one of :data:`RESERVED`, raise ``ValueError``, since the column
made would take its place: rename it. An ``Object`` ``group`` column raises
:class:`~polars_online.WrongTypeError`, since a query's chunks cannot be
matched by it: cast it to ``String`` or ``Categorical``.

``group`` names one key column, as a spec's ``group`` does, or a list of
them, and ``None``, the default, is no key. ``min_samples`` is the fewest
rows a key is scored on, as Polars' ``min_samples`` is: a key with fewer
rows keeps its row, with its count ``n`` filled and its scores null. The
keywords ``by`` and ``min_obs`` and the function ``rolling_metrics`` are
refused, and the error names the keyword or function that replaced each:
``group``, ``min_samples`` and :func:`window_metrics`. :func:`window_metrics`
refuses ``window_size`` the same way, naming ``every``. ``spec_name``
(``spec_names`` in :func:`compare_specs`) is read as ``spec``, with a
:class:`~polars_online.PolarsOnlineDeprecationWarning`. Given beside a spec
passed by position, it raises ``TypeError``, and a spec name given beside
a spec dict whose name differs raises ``ValueError``.

`docs/DIAGNOSTICS.md <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md>`_
says which question each call and each diagnostic answers, with its
threshold, its false-alarm rate as measured, and a recipe that runs.
"""

from __future__ import annotations

import functools
import itertools
import math
import numbers
import operator
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal, overload

import polars as pl

from polars_online import _polars_online as _native
from polars_online._duration import Duration, clock_nanoseconds
from polars_online._errors import (
    ColumnNotFoundError,
    OutputColumnNotFoundError,
    RefusedValueError,
    WrongTypeError,
)
from polars_online._renamed import renamed_function, renamed_keywords
from polars_online._warnings import PolarsOnlineDeprecationWarning

__all__ = [
    "metrics",
    "window_metrics",
    "compare_specs",
    "grid_search",
    "unpack",
    "seqtest",
    "diebold_mariano",
    "clark_west",
    "sums",
    "merge_sums",
    "from_sums",
    "SUM_FIELDS",
    "RESERVED",
]

# The keywords renamed before 1.0 (docs/PLAN.md task 197): `min_obs` is
# Polars' `min_samples` for the same count, and `by` is `group`, the specs'
# word since 0.12.0 (review round 4, AP6); `window_metrics`' `every` was
# `rolling_metrics`' `window_size` (YB5). Literal tables, each whole: Sphinx's
# source parser fails on a `{**a, **b}` in a decorator, and then drops every
# `#:` comment of the module, so `SUM_FIELDS` and `RESERVED` went unrendered.
_GROUP = {"by": "group"}
_MIN_SAMPLES = {"min_obs": "min_samples"}
_BOTH = {"by": "group", "min_obs": "min_samples"}
_WINDOW = {"by": "group", "min_obs": "min_samples", "window_size": "every"}

#: The columns :func:`sums` produces beside the keys, in order: ten numbers
#: per (key, slot) are all the memory evaluating a stream takes.
SUM_FIELDS = (
    "n",
    "w",
    "mean_y",
    "mean_pred",
    "m2_y",
    "m2_pred",
    "cov",
    "sse",
    "hits",
    "signed",
)


def _frame(who: str, df: object, *, lazy: bool = False) -> None:
    """Refuse anything but a ``DataFrame``, and a ``LazyFrame`` where the
    call reads a query (``lazy``): a ``LazyFrame`` once reached Polars' own
    calls, which warned three times and then failed on ``LazyFrame is not
    subscriptable``."""
    if isinstance(df, pl.DataFrame):
        return
    if lazy:
        if isinstance(df, pl.LazyFrame):
            return
        msg = (
            f"{who}: the frame must be a Polars DataFrame or LazyFrame, but it is of type "
            f"{type(df).__name__}. Pass a Polars DataFrame, or a LazyFrame to read a query."
        )
        raise TypeError(msg)
    if isinstance(df, pl.LazyFrame):
        msg = (
            f"{who}: the frame must be a Polars DataFrame, but it is a LazyFrame. Pass the "
            f"query's result, lf.collect(), which runs the query and returns its rows."
        )
    else:
        msg = (
            f"{who}: the frame must be a Polars DataFrame, but it is of type {type(df).__name__}. "
            f"Pass a Polars DataFrame."
        )
    raise TypeError(msg)


def _spec_of(who: str, spec: str | dict) -> tuple[str, dict | None]:
    """``spec`` as the struct column's name and, when it is the spec dict,
    the dict too."""
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, dict) and isinstance(spec.get("name"), str):
        return spec["name"], spec
    if isinstance(spec, dict):
        msg = (
            f"{who}: spec must be a spec's name or the spec itself, but it is a dict with no "
            f"name, which no po.spec builder returns. Pass a spec built by po.spec, or its name "
            f"as a str."
        )
        raise TypeError(msg)
    msg = (
        f"{who}: spec must be a spec's name or the spec itself, but it is of type "
        f"{type(spec).__name__}. Pass the spec, or its name as a str."
    )
    raise TypeError(msg)


def _warn_merged(who: str, what: str) -> None:
    warnings.warn(
        f"{who}: {what} is deprecated, since spec takes the spec's name or the spec itself, "
        "which carries the name. It is read as before until the next major version, which "
        "refuses it. Pass the spec alone as spec.",
        PolarsOnlineDeprecationWarning,
        stacklevel=3,
    )


def _takes_spec[R](
    who: str, *, lazy: bool = False
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """A decorator for a call whose ``spec_name`` and ``spec=`` became one
    ``spec``, the name or the dict: the old forms, ``spec_name=`` and a name
    beside ``spec=``, are read as the new one, with a
    :class:`~polars_online.PolarsOnlineDeprecationWarning`, and a frame
    that is not a ``DataFrame`` (or a ``LazyFrame``, where ``lazy``) is
    refused by name."""

    def decorate(fn: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(fn)
        def wrapper(df: Any, *args: Any, **kwargs: Any) -> R:
            _frame(who, df, lazy=lazy)
            if "spec_name" in kwargs:
                if args:
                    msg = (
                        f"{who}: spec_name was merged into spec, but both are given. "
                        f"Pass the spec alone as spec."
                    )
                    raise TypeError(msg)
                _warn_merged(who, "spec_name")
                args = (kwargs.pop("spec_name"),)
            if args and isinstance(args[0], str) and isinstance(kwargs.get("spec"), dict):
                written = kwargs.pop("spec")
                if written.get("name") != args[0]:
                    msg = (
                        f"{who}: the spec name {args[0]!r} and the spec named "
                        f"{written.get('name')!r} disagree. Pass the spec alone as spec."
                    )
                    raise ValueError(msg)
                _warn_merged(who, "a spec's name beside the spec dict")
                args = (written, *args[1:])
            return fn(df, *args, **kwargs)

        return wrapper

    return decorate


def _logistic(spec: dict | None) -> bool:
    """Whether ``spec`` names a fit with ``loss = "logistic"``, whose
    ``pred`` is a probability and its target a 0/1 label: ``ftrl``'s default
    and one of ``sgd``'s losses."""
    model = (spec or {}).get("model") or {}
    loss = model.get("loss", "logistic" if model.get("type") == "ftrl" else None)
    return loss == "logistic"


def _binary(binary: bool | None, spec: dict | None) -> bool:
    """``binary`` as given, or read from the spec when ``None``: on for a
    logistic fit, off without a spec, where the loss cannot be known."""
    return _logistic(spec) if binary is None else binary


def _pred_fields(who: str, df: pl.DataFrame, spec_name: str) -> list[str]:
    dtype = df.schema.get(spec_name)
    if dtype is None:
        msg = (
            f'{who}: spec "{spec_name}" has no output column in the frame, whose columns are '
            f"{df.columns}. Name a spec whose output column the frame holds."
        )
        raise OutputColumnNotFoundError(msg)
    if not isinstance(dtype, pl.Struct):
        msg = (
            f"{who}: column {spec_name!r} must be a spec's output column, but it is "
            f"{dtype}. Name a spec whose output column the frame holds."
        )
        raise TypeError(msg)
    fields = [f.name for f in dtype.fields if f.name.startswith("pred_")]
    if not fields:
        msg = (
            f"{who}: output column {spec_name!r} has no prediction fields (pred_*), since its "
            "model writes other fields: an ew_cov model statistics, a kmeans or micro model "
            "cluster assignments, an ew_class model a class and its posterior probabilities, "
            "and a seqtest model evidence. Read those fields with pl.col(...).struct.field(...), "
            "or pass a spec whose model predicts a target."
        )
        raise TypeError(msg)
    return fields


#: `online_core::INPUT_BOUND`: a magnitude beyond it is a missing value to
#: the bank, so it is one to :func:`seqtest` too.
_INPUT_BOUND = 1e100

#: Column names :func:`unpack` produces. Input columns with these names are
#: dropped rather than duplicated (the target's values come back as ``y``).
RESERVED = ("slot", "target", "pred", "y")


@overload
def unpack(
    df: pl.DataFrame, spec: str | dict, *, targets: Sequence[str] | None = None
) -> pl.DataFrame: ...


@overload
def unpack(
    df: pl.LazyFrame, spec: str | dict, *, targets: Sequence[str] | None = None
) -> pl.LazyFrame: ...


@_takes_spec("po.eval.unpack", lazy=True)
def unpack(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict,
    *,
    targets: Sequence[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Turn a spec's output into long form, one row per input row and slot.

    Use it to score or plot predictions with your own Polars code: each row
    holds one prediction beside the value of the target it predicts. For the
    standard scores, use :func:`metrics`, which reads this long form itself.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned: the input's columns and the spec's output
        column. Or a query of one, a ``LazyFrame``, which gives a query of
        the long form back (see Returns).
    ``spec``
        The spec's name, which is its output column's, or the spec dict
        itself, which names the column too. Pass the dict to read each
        slot's target as the spec reads it, through
        :func:`polars_online.spec.output_index` and the spec's ``targets``.
        A plain target is read from its column, and a target renamed with
        :func:`polars_online.target`'s ``name=`` from the column it renames.
        A formula target is read from a column of its own name, which
        :func:`polars_online.stream.with_windows` makes, since the bank
        computes the formula as it learns and does not write it out.
        Given the name alone, a slot is read against the column named
        after it, the longest name that matches. That rule can attribute a
        slot to the wrong target when a column of the frame is named as a
        target followed by a slot's ``__`` or ``@`` suffix. A renamed target
        (``po.target("p", name="price")`` writes ``pred_price``) names no
        column. So only the dict can say what either read.
    ``targets``
        The columns the name-based rule may read as targets; every column
        of ``df`` by default. A bare string is one name, as ``group`` takes
        one. Not read when ``spec`` is the dict.

    .. rubric:: Returns

    A ``DataFrame`` of ``df``'s columns that are not records, one row for
    each row of ``df`` and slot, with four columns added:

    ``slot``
        The name of the field the prediction is in, such as
        ``pred_y__r0.5@h500``.
    ``target``
        The target the slot predicts: the name its output fields carry,
        which for a plain target is its column.
    ``pred``, ``y``
        The prediction and the target's value on the row.

    An input column named like one of these four (:data:`RESERVED`) is
    dropped, so a target column called ``y`` does not collide with the
    ``y`` added.

    Given a ``LazyFrame``, a ``LazyFrame`` of the same rows, which runs
    only when its result is asked for, as Polars' own calls on a query
    return a query. Its rows come slot by slot, as the frame's do, so it
    reads ``df`` once for each slot: a query that runs a bank runs the bank
    once per slot. Write the bank's output to a file first, with
    ``sink_parquet``, and unpack a scan of the file. Written to a file with
    ``sink_parquet``, the long form of a 20-slot output held 123 MiB at a
    million rows, 133 MiB at four million and 172 MiB at sixteen million
    under :func:`metrics`' bounded settings, and 656, 1,137
    and 1,469 MiB under Polars' defaults, which read the scan ahead of the
    writing: `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds that.

    .. rubric:: Raises

    ``TypeError``
        ``df`` is neither a ``DataFrame`` nor a ``LazyFrame``; or the
        spec's column is not a record, or holds no ``pred_*`` field, as an
        ``ew_cov``, ``kmeans``, ``micro``, ``ew_class`` or ``seqtest`` model
        writes; or ``spec`` is neither a name nor a spec dict.
    :class:`~polars_online.OutputColumnNotFoundError`
        ``df`` has no column of the spec's. It is both Polars'
        ``ColumnNotFoundError`` and a ``KeyError``.
    :class:`~polars_online.ColumnNotFoundError`
        Given the spec dict, a slot's target column is not in ``df``, such
        as a formula target's column not yet made. It is both Polars'
        ``ColumnNotFoundError`` and a ``ValueError``.
    ``ValueError``
        Given the name alone, no column of ``df`` (or of ``targets``) is
        named after a slot; or, given the dict, whatever
        :func:`polars_online.spec.output_index` raises for it.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        long = po.eval.unpack(out, spec)                       # one row per row of out and slot
        slot, target, pred, y = long.select("slot", "target", "pred", "y").row(399)
        print(f"{slot} predicts {target}: {pred:.3f} on the last row, where y is {y:.3f}")

    .. code-block:: text

        pred_y predicts y: 1.768 on the last row, where y is 1.479

    .. rubric:: See also

    - :func:`metrics`, the scores of this long form per slot, target and group.
    - :func:`polars_online.spec.output_index`, each output field's slot,
      target and grid point, read from the spec.
    """
    if isinstance(df, pl.LazyFrame):
        return _unpack(df, spec, targets=targets, who="po.eval.unpack")
    return _unpack(df, spec, targets=targets, who="po.eval.unpack")


def _unpack[F: (pl.DataFrame, pl.LazyFrame)](
    df: F,
    spec: str | dict,
    *,
    targets: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    who: str = "po.eval.unpack",
) -> F:
    """:func:`unpack`, carrying of the input's columns only those in
    ``columns`` (all of them when ``None``). A score reads the key columns
    and the clock or weight beside ``pred`` and ``y``; carrying every input
    column once per slot made its long form the frame's width times the
    slots, 1.4 GiB from a 290 MiB frame of 43 columns and 8 slots, where
    the four columns it reads take 137 MiB."""
    probe = _probe(df)
    spec_name, slots = _slots(probe, spec, targets, who)
    keep = _kept(probe, columns)
    frames = [
        df.select(
            *keep,
            pl.lit(slot).alias("slot"),
            pl.lit(target).alias("target"),
            pl.col(spec_name).struct.field(slot).alias("pred"),
            y.alias("y"),
        )
        for slot, target, y in slots
    ]
    return pl.concat(frames)


def _probe(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """``df`` itself, or for a query a frame of no rows with the query's
    schema: what the checks of a spec and its columns read, so a query is
    refused as a frame is, with the same message, before it runs."""
    return df if isinstance(df, pl.DataFrame) else pl.DataFrame(schema=df.collect_schema())


def _slots(
    df: pl.DataFrame, spec: str | dict, targets: Sequence[str] | None, who: str
) -> tuple[str, list[tuple[str, str, pl.Expr]]]:
    """The spec's output column, and each of its ``pred`` fields with the
    target it predicts and the expression of the target's value."""
    spec_name, written = _spec_of(who, spec)
    fields = _pred_fields(who, df, spec_name)
    # With the spec in hand, the slot -> target mapping comes from
    # `output_index` -- the same Rust code that named the fields -- and each
    # target's value from the spec's own table of it, instead of the
    # heuristic name-parsing below, which exists only for callers who have a
    # frame but no spec.
    exact = {} if written is None else _spec_targets(written, df, who)
    slots = []
    for slot in fields:
        if slot in exact:
            target, y = exact[slot]
        else:
            target = _target_of(who, slot, df, targets)
            y = pl.col(target)
        slots.append((slot, target, y))
    return spec_name, slots


def _kept(df: pl.DataFrame, columns: Sequence[str] | None) -> list[str]:
    """The input columns :func:`unpack` carries: every one that is not a
    record or :data:`RESERVED`, or of those only the ones in ``columns``."""
    keep = [c for c, d in df.schema.items() if not isinstance(d, pl.Struct) and c not in RESERVED]
    if columns is not None:
        wanted = set(columns)
        keep = [c for c in keep if c in wanted]
    return keep


def _spec_targets(spec: dict, df: pl.DataFrame, who: str) -> dict[str, tuple[str, pl.Expr]]:
    """Each ``pred`` field of ``spec`` -> its target's name and the
    expression of the target's value on the frame."""
    from polars_online import spec as _spec_mod
    from polars_online._spec import target_name

    tables = {target_name(t): t for t in spec.get("targets") or []}
    idx = _spec_mod.output_index(spec).filter(pl.col("kind") == "pred")
    return {
        r["field"]: (
            r["target"],
            _target_value(who, r["field"], tables.get(r["target"]), r["target"], df),
        )
        for r in idx.iter_rows(named=True)
        if r["target"] is not None
    }


def _target_value(who: str, slot: str, table: Any, name: str, df: pl.DataFrame) -> pl.Expr:
    """A target's value on the frame as the bank reads it: its column, or
    for a formula target the column of its own name that
    :func:`polars_online.stream.with_windows` writes. A value the bank
    would not learn from is dropped by :func:`_scored`."""

    def column(c: str) -> pl.Expr:
        if c not in df.columns:
            msg = (
                f"{who}: slot {slot!r} reads target {name!r} from column {c!r}, but the frame "
                f"has no such column (it has {df.columns}). Pass a frame that holds the target "
                f"column."
            )
            raise ColumnNotFoundError(msg)
        return pl.col(c)

    if table is None or isinstance(table, str):
        return column(name if table is None else table)
    if "formula" in table:
        if name not in df.columns:
            msg = (
                f"{who}: slot {slot!r} predicts formula target {name!r}, which the bank computes "
                f"as it learns and does not write out, and the frame has no column {name!r} to "
                f"score it against. Add the column with po.stream.with_windows before scoring."
            )
            raise ColumnNotFoundError(msg)
        return pl.col(name)
    return column(table["column"])  # a column renamed


def _target_of(who: str, slot: str, df: pl.DataFrame, targets: Sequence[str] | None) -> str:
    """``pred_<target>[__combo][@hN]`` -> ``<target>``. Longest match wins, so
    targets whose names are prefixes of each other still resolve."""
    body = slot[len("pred_") :]
    if isinstance(targets, str):
        # One name, as `group` takes one: iterated, "ret" was the targets
        # "r", "e" and "t" (task 382a, A14).
        targets = [targets]
    candidates = targets if targets is not None else list(df.columns)
    matches = [
        t for t in candidates if body == t or body.startswith(t + "__") or body.startswith(t + "@")
    ]
    if not matches and targets is not None:
        msg = (
            f"{who}: no target in targets matches slot {slot!r}, so its target column is not "
            "known. Add the slot's target to targets, or pass the spec itself as spec."
        )
        raise ValueError(msg)
    if not matches:
        msg = (
            f"{who}: no column of the frame is named after slot {slot!r}, so its target column "
            "is not known. Add the target column to the frame, or, for a target named with "
            "po.target(name=), pass the spec itself as spec."
        )
        raise ValueError(msg)
    return max(matches, key=len)


def _group_keys(group: str | Iterable[str] | None) -> list[str]:
    """``group`` as the list of its key columns: a bare string is one key,
    as in Polars' ``group_by`` and ``over`` and a spec's ``group``. Iterated
    as a sequence, ``group="group"`` would be the keys ``g``, ``r``, ``o``,
    ``u``, ``p``. ``None`` is no key, as a spec's ``group=None`` is."""
    if group is None:
        return []
    return [group] if isinstance(group, str) else list(group)


#: The names of the columns the scores make, beside :data:`RESERVED`'s:
#: :data:`SUM_FIELDS`, each score's name, and a window's ``window_start``.
#: A key, clock or weight column of one of these names collides with the
#: column made (task 382a, A6 and A11).
_SCORE_NAMES = (
    "r2", "ic", "hit_rate", "mse", "rmse", "brier_skill", "accuracy", "log_loss", "window_start",
)  # fmt: skip
_PRODUCED = (*RESERVED, *SUM_FIELDS, *_SCORE_NAMES)

#: The names a clock or a weight column may not take: :data:`RESERVED`,
#: which :func:`unpack` puts in its place, and for a clock the window's
#: ``window_start``, which a time window joins beside it.
_TAKEN_BY_LONG = {"clock": (*RESERVED, "window_start"), "weight": RESERVED}


def _check_columns(
    df: pl.DataFrame, who: str, *, produced: Iterable[str] = (), **roles: str | Iterable[str] | None
) -> None:
    """Each column a score reads beside a spec's output, by its role
    (``group``, ``clock``, ``weight``), checked to be in ``df``: a missing
    one is :class:`~polars_online.ColumnNotFoundError`, a ``ValueError``
    too, as a bank's refusal of a missing column is. A column named in
    ``produced``, a name the call's result or its sums give a column of
    their own, is refused with ``ValueError``: named ``y`` or ``pred`` it
    was replaced by the unpacked target or prediction, so ``group="y"``
    keyed the scores by the target's value, and named ``n`` or ``w`` it
    raised Polars' ``DuplicateError``. An ``Object`` key is refused with
    :class:`~polars_online.WrongTypeError`: it grouped a frame and failed
    over a query on Polars' internals (task 382a, A6, A11 and C9)."""
    produced = set(produced)
    for role, value in roles.items():
        # A key is a column of the result; a clock or a weight is read beside
        # the unpacked long form only, so only its own names collide there.
        taken = produced if role == "group" or not produced else set(_TAKEN_BY_LONG[role])
        for column in _group_keys(value):
            if column not in df.schema:
                msg = (
                    f"{who}: {role} column {column!r} is not in the frame, whose columns are "
                    f"{df.columns}. Name a column of the frame as {role}."
                )
                raise ColumnNotFoundError(msg)
            if column in taken:
                msg = (
                    f"{who}: {role} column {column!r} is named like a column the scores make "
                    f"({', '.join(sorted(taken))}), and would be confused with it. Rename the "
                    f"column, with df.rename({{{column!r}: ...}}), and pass the new name."
                )
                raise ValueError(msg)
            if role == "group" and isinstance(df.schema[column], pl.Object):
                msg = (
                    f"{who}: group column {column!r} is an Object column, which Polars can group "
                    "a frame by but not join or deduplicate a query's chunks by. Cast it to "
                    "String or Categorical."
                )
                raise WrongTypeError(msg)


def _check_min_samples(who: str, min_samples: object) -> None:
    """``min_samples`` a whole number of at least 0, numpy's integers
    included but not a bool: ``None``, ``True`` and -5 scored every key,
    20.5 thinned a key of 20 rows, and "30" raised Polars' error on
    comparing a string with a number (task 382a, A14)."""
    if isinstance(min_samples, bool) or not isinstance(min_samples, numbers.Integral):
        msg = (
            f"{who}: min_samples must be a whole number of rows, but it is {min_samples!r}. "
            "Give a whole number, such as the default 30."
        )
        raise WrongTypeError(msg)
    if min_samples < 0:
        msg = (
            f"{who}: min_samples must be a whole number of at least 0, but it is {min_samples}. "
            "Give 0 to score every key, or more."
        )
        raise ValueError(msg)


def _int_range(dtype: pl.DataType) -> tuple[int, int]:
    """The least and the greatest value of an integer ``dtype``."""
    bits = int(re.sub(r"\D", "", str(dtype)))
    if str(dtype).startswith("U"):
        return 0, 2**bits - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _usable(value: pl.Expr) -> pl.Expr:
    """Whether a float is one the bank learns from: finite, and within its
    input bound of 1e100; null where ``value`` is."""
    return value.is_finite() & (value.abs() <= _INPUT_BOUND)


def _scored(long: pl.DataFrame) -> pl.DataFrame:
    """The rows of :func:`unpack`'s long form whose prediction and target are
    both values the bank would learn from: not null, not NaN, not infinite and
    not past its input bound of 1e100. The bank reads such a target as
    missing and still predicts its row. Dropping nulls alone would keep the
    row, make ``r2``, ``ic`` and ``mse`` NaN, and quietly lower
    ``hit_rate``.

    ``pred`` and ``y`` come back as ``Float64``, so every score is computed
    in doubles: a bank passes a ``Float32`` target through as it is, and its
    deviations were squared in single precision on a frame, 2e-8 apart from
    the same scores of a query (task 382a, A4)."""
    long = long.with_columns(pl.col("pred").cast(pl.Float64), pl.col("y").cast(pl.Float64))
    return long.filter(_usable(pl.col("pred")) & _usable(pl.col("y")))


def _no_hit(spec: dict | None) -> bool:
    """Whether ``spec`` names a fit with no sign to hit, whose ``hit_rate``
    the bank's ``emit_metrics`` nulls: a Poisson ``sgd`` (``loss =
    "poisson"``), where a rate is positive and a count never negative, so
    the sign test about zero agrees on every row and reads 1.0 whatever the
    fit. Without ``spec``
    the loss cannot be known, and the sign test is read."""
    model = (spec or {}).get("model") or {}
    return model.get("type") == "sgd" and model.get("loss") == "poisson"


def _check_labels(long: pl.DataFrame, who: str) -> None:
    """Under ``binary``, refuse a scored row whose target is not 0 or 1: a
    regression target scored as a classifier's read a negative log loss
    and an accuracy with nothing to say why (task 382a, A15)."""
    bad = long.filter(~pl.col("y").is_in([0.0, 1.0])).head(1)
    if bad.height:
        row = bad.row(0, named=True)
        msg = (
            f"{who}: under binary the target is a 0/1 label, but slot {row['slot']!r} reads "
            f"target {row['target']!r}, which is {row['y']} on a scored row, where the log loss, "
            "the accuracy and the Brier skill score have no meaning. Pass binary=False to "
            "score it as a regression, or pass the 0/1 target column."
        )
        raise ValueError(msg)


def _thin(out: pl.DataFrame, min_samples: int, keep: Iterable[str]) -> pl.DataFrame:
    """``out`` with every column but the keys, ``n`` and ``keep`` null where
    ``n`` is below ``min_samples``: Polars' meaning of ``min_samples`` in
    ``rolling_*`` and ``ewm_*``, where a window too thin gives null and never
    leaves the result. The row stays, its count read (task 244, M-6)."""
    held = {"n", *keep}
    thin = pl.col("n") < min_samples
    return out.with_columns(
        pl.when(thin).then(None).otherwise(pl.col(c)).alias(c) for c in out.columns if c not in held
    )


@renamed_keywords("po.eval.metrics", _BOTH)
@_takes_spec("po.eval.metrics", lazy=True)
def metrics(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict,
    *,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    min_samples: int = 30,
    binary: bool | None = None,
) -> pl.DataFrame:
    """Score a spec's out-of-sample predictions over a whole frame.

    Use it to read how well a spec predicted, per slot, target and group,
    over a frame in memory or over a query too long to hold, such as a scan
    of the file a bank's output was written to. For scores in windows of the
    clock, use :func:`window_metrics`; for several specs in one table,
    :func:`compare_specs`; for scores kept beside the model as it runs, the
    spec's own ``emit_metrics``.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding the spec's output column, or a
        query of one, a ``LazyFrame``, which this runs and reads one chunk
        at a time (see Notes).
    ``spec``
        The spec's name, or the spec dict itself, as :func:`unpack` reads
        it. Pass the dict for a target that is not a plain column, and to
        have ``binary`` read from the spec's loss.
    ``group``
        One key column, as a spec's ``group`` is, or a list of them, to
        score each key apart. ``None``, the default, scores every row
        together.
    ``targets``
        :func:`unpack`'s: the columns the name-based rule may read as
        targets.
    ``min_samples``
        The fewest rows a key is scored on, 30 by default. A key with fewer
        rows keeps its row, with ``n`` filled and every score null, as
        Polars' ``rolling_*`` and ``ewm_*`` give null below their
        ``min_samples``.
    ``binary``
        Score the slots as a classifier's: ``pred`` a probability and ``y``
        a 0/1 target, the output of an ``sgd`` or ``ftrl`` fit with
        ``loss="logistic"``. ``None``, the default, reads it from the spec
        dict's loss, and is off given the name alone, since nothing here
        reads the target's values to decide. A target other than 0 or 1 on
        a scored row is refused under it.

    .. rubric:: Returns

    A ``DataFrame`` with one row per ``slot``, ``target`` and ``group``
    key, sorted by them, given a frame or a query alike, and these columns
    after the keys. A slot or key with no scored row has no row, such as a
    slot whose predictions were all withheld by a warm-up.

    ``n``
        The rows scored.
    ``r2``
        The out-of-sample R², against the target's mean over the rows
        scored. Null where the target never varied.
    ``ic``
        The correlation of the prediction with the target. Null where
        either never varied.
    ``hit_rate``
        The share of rows whose sign the prediction got right, over the
        rows where neither ``y`` nor ``pred`` is 0, as the bank's
        ``emit_metrics`` counts them. Null where no row is counted. A
        Poisson ``sgd`` fit (``loss="poisson"``) has no sign to hit, so the
        bank nulls its ``hit_rate``, and so does this, given the spec dict.
    ``mse``, ``rmse``
        The mean squared residual, and its square root, in the target's
        units.

    A score that is undefined is null, not an infinity or a NaN that reads
    as a number. Under ``binary``, two columns
    take other names with their meaning, and one is added:

    ``brier_skill``, in place of ``r2``
        The same formula, which on a 0/1 target is the Brier skill score:
        the share of the Brier score of always predicting the base rate
        that the fit removes.
    ``accuracy``, in place of ``hit_rate``
        The share of rows on the right side of 0.5. Every row is counted.
    ``log_loss``
        The mean of ``-(y*ln(p) + (1-y)*ln(1-p))``, with ``p`` clipped to
        ``[1e-15, 1 - 1e-15]``, so a confident miss is a large number and
        not an infinity.

    ``ic`` keeps its name under ``binary``: on a 0/1 target it is the
    point-biserial correlation, the same formula. ``rmse`` is then the root
    of the Brier score.

    .. rubric:: Raises

    Whatever :func:`unpack` raises for ``df`` and ``spec``, and:

    :class:`~polars_online.ColumnNotFoundError`
        ``df`` does not have a ``group`` column. It is both Polars'
        ``ColumnNotFoundError`` and a ``ValueError``.
    ``ValueError``
        A ``group`` column is named like a column the scores make, in
        :data:`RESERVED` or :data:`SUM_FIELDS`, a score's name, or
        ``window_start``: named ``y``, it would key the scores by the
        target's value. Rename it. Or, under ``binary``, a scored row's
        target is not 0 or 1, which a query raises from the chunk that
        holds it. Or ``min_samples`` is below 0.
    :class:`~polars_online.WrongTypeError`
        A ``group`` column is an ``Object``, which a query's chunks cannot
        be matched by: cast it to ``String`` or ``Categorical``. Or
        ``min_samples`` is not a whole number. It is both a ``TypeError``
        and a ``ValueError``.

    A query is refused as its result would be, with the same message,
    before it runs.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        scores = po.eval.metrics(out, spec, group="stock_id")  # one row per slot, target and stock
        for row in scores.select("stock_id", "n", "r2", "hit_rate").iter_rows():
            print(f"{row[0]}: {row[1]} rows, r2 {row[2]:.3f}, hit rate {row[3]:.3f}")

    .. code-block:: text

        b0: 99 rows, r2 0.939, hit rate 0.899
        b1: 99 rows, r2 0.938, hit rate 0.939
        b2: 99 rows, r2 0.947, hit rate 0.949
        b3: 99 rows, r2 0.948, hit rate 0.929

    .. rubric:: See also

    - :func:`window_metrics`, the same scores in windows of the clock.
    - :func:`compare_specs`, the same scores for several specs in one table.
    - :func:`sums` and :func:`from_sums`, the same scores one chunk at a time.
    - `Is the model working?
      <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md>`_,
      the diagnostics a spec writes beside its predictions.

    .. rubric:: Notes

    **Given a query, this holds one chunk of it and a few sums per key, not
    its rows, and Polars holds what it has read of the query ahead of
    them.** It runs the query once and reads its result 100,000 rows at a
    time, the chunk a bank reads by default. Each chunk is reduced to
    :func:`sums`' ten numbers per slot, target and key, with the sum of the
    log loss under ``binary``, and the chunks' sums are added as
    :func:`merge_sums` adds them. A scan of a parquet file reads blocks of
    it, its row groups, ahead of a slower consumer, as many as Polars'
    thread count sizes, and under Polars' defaults that read-ahead is most
    of what the process holds: `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds it. Measured on macOS (an M4 Pro, py-polars 2.0.0) as the most
    memory the process held, 47 MiB of it Python and Polars before the
    call, scoring a scan of a 20-slot output written in row groups of
    100,000 rows, by a key of four values: under Polars' defaults, 494 MiB
    at a million rows, 1,012 MiB at four million and 1,020 MiB at sixteen
    million; with four threads (``POLARS_MAX_THREADS=4``), the read-ahead
    cut to one row group (``POLARS_ROW_GROUP_PREFETCH_SIZE=1``) and the
    allocator told to return freed pages at once
    (``_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0``), the bounded
    settings, 205, 208 and 211 MiB. The same scores of the collected frame,
    read a slot at a time, held 345 MiB at a million rows under the bounded
    settings, the frame's 168 MiB among them.

    The scores of a query are the frame's. A frame is scored from the same
    centered sums as a chunk, so while the query gives one chunk every
    column is the frame's to the bit. Over many chunks each score is the
    frame's within the rounding of the merge. Each chunk's mean is rounded
    to a double, and that rounding enters the moments about the pooled
    mean at first order: about ``eps * |mean| / (spread * sqrt(N))``
    relative in a key's ``m2_y``, ``m2_pred`` and ``cov``, with ``eps``
    2.2e-16 and ``N`` the key's rows, whatever the number or size of the
    chunks. ``r2`` and ``ic`` then differ from the frame's by about that
    much, not relative to themselves, so a score near 0 keeps it in
    absolute terms (``r2`` by ``1 - r2`` times it). Measured against exact
    rational sums, keys of unit spread over 1,000 to 100,000 rows read in
    parts of 25 rows to a tenth of the key were within 1.4 times that
    figure: 1e-13 at a level of 1e4 over 1,000 rows, 3e-10 at a level of
    1e8 over 10,000 rows. Score a target that sits far above its spread,
    such as a price, as a difference or a return.

    A row is scored only where both the prediction and the target are
    values the bank would learn from: not null, NaN or infinite, and within
    the bank's input bound of 1e100 in magnitude. So the rows of a warm-up,
    whose predictions are withheld, and the rows a model skipped never
    enter the scores. The bank still predicts a row whose target is
    null, and that row is left out here too.

    A target that sits about 1, such as a plain ratio of two prices, is
    positive on every row, and so is its prediction, so ``hit_rate`` reads
    1.0 whatever the fit. Score a return as a difference or a log ratio.
    Given the name alone, the loss of a Poisson ``sgd`` fit cannot be
    known, and the sign test reads 1.0.
    """
    return _metrics(
        df, spec, group=group, targets=targets, min_samples=min_samples, binary=binary,
        who="po.eval.metrics",
    )  # fmt: skip


def _metrics(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict,
    *,
    group: str | Iterable[str] | None,
    targets: Sequence[str] | None,
    min_samples: int,
    binary: bool | None,
    who: str,
) -> pl.DataFrame:
    """:func:`metrics`, refusing in the name of ``who``: the call made,
    :func:`compare_specs`' own where it scores each of its specs."""
    spec_name, written = _spec_of(who, spec)
    _check_min_samples(who, min_samples)
    _check_columns(_probe(df), who, group=group, produced=_PRODUCED)
    binary = _binary(binary, written)
    keys = _keys(group)
    if isinstance(df, pl.LazyFrame):
        job = _job(df, written or spec_name, targets, group, None, binary, who, log_loss=True)
        merged = _query_sums(df, [job], keys=keys, weight=None)[0]
        if merged is not None:
            return _thin(_scores(merged, keys, binary=binary), min_samples, keys)
        df = _probe(df)
    job = _job(df, written or spec_name, targets, group, None, binary, who, log_loss=True)
    s = _frame_sums(df, job, keys, None)
    return _thin(_scores(s, keys, binary=binary), min_samples, keys)


#: Polars' duration text, every unit it takes: ``mo``, ``q`` and ``y`` are
#: the calendar's and have no fixed length.
_EVERY_TEXT = re.compile(r"(?:\d+(?:ns|us|ms|s|m|h|d|w|mo|q|y))+")
_CALENDAR_PART = re.compile(r"\d+(?:mo|q|y)")


def _is_calendar(every: object) -> bool:
    """Whether ``every`` is duration text with a month, quarter or year in
    it, which no number of nanoseconds measures."""
    return isinstance(every, str) and bool(_CALENDAR_PART.search(every.strip()))


def _calendar_every(every: str, dtype: pl.DataType, clock: str) -> int:
    """Check text with a calendar part, as ``group_by_dynamic`` would read
    it on a ``Datetime`` or ``Date`` clock: Polars' grammar, at least one
    unit above 0, and any fixed part a whole number of the column's steps.
    Returns the fixed part's nanoseconds (0 for none)."""
    text = every.strip()
    if not _EVERY_TEXT.fullmatch(text) or not re.search(r"[1-9]", text):
        msg = (
            f"po.eval.window_metrics: every must be a duration above 0 in Polars' text, but it "
            f'is {every!r}. Give every as text such as "1mo", "1q" or "1y2w".'
        )
        raise ValueError(msg)
    fixed = _CALENDAR_PART.sub("", text)
    if fixed and re.search(r"[1-9]", fixed):
        return int(clock_nanoseconds(fixed, dtype, "po.eval.window_metrics", "every", clock) or 0)
    return 0


def _polars_every(every: Any, ns: int | None) -> str | timedelta:
    """``every`` as ``group_by_dynamic`` takes it: text and a ``timedelta``
    as given, so Polars reads them its own way (its ``d``, ``w``, ``mo``,
    ``q`` and ``y`` follow the calendar; a ``timedelta``'s whole days are
    days), and an expression as the nanoseconds it evaluates to, a length of
    time."""
    if isinstance(every, str):
        return every.strip()
    if isinstance(every, timedelta):
        return every
    return f"{ns}ns"


def _dynamic_windows(
    long: pl.DataFrame, clock: str, every: str | timedelta, keys: list[str]
) -> pl.DataFrame:
    """``long`` with each row's ``window_start``: the left edge of its window
    in Polars' ``group_by_dynamic`` over ``keys``, closed and labelled on the
    left, the windows starting where Polars starts them. A row with no clock
    is in no window (its ``window_start`` is null)."""
    idx = "__po_row"
    while idx in long.columns:
        idx = "_" + idx
    long = long.with_row_index(idx)
    starts = (
        long.filter(pl.col(clock).is_not_null())
        .select(*keys, clock, idx)
        .sort(*keys, clock, maintain_order=True)
        .group_by_dynamic(clock, every=every, group_by=keys, closed="left", label="left")
        .agg(pl.col(idx))
        .explode(idx)
        .select(idx, pl.col(clock).alias("window_start"))
    )
    return long.join(starts, on=idx, how="left", maintain_order="left").drop(idx)


@renamed_keywords("po.eval.window_metrics", _WINDOW)
@_takes_spec("po.eval.window_metrics", lazy=True)
def window_metrics(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict,
    *,
    clock: str,
    every: float | Duration,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    min_samples: int = 30,
    binary: bool | None = None,
) -> pl.DataFrame:
    """Score a spec's out-of-sample predictions in windows of the clock.

    Use it to see how a spec's scores move over the stream, such as R² per
    hour or per day. The windows do not overlap: each row falls in one, as
    in Polars' ``group_by_dynamic``. For one score over the whole frame,
    use :func:`metrics`.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding the spec's output column and the
        clock column, or a query of one, a ``LazyFrame``, which this runs
        and reads one chunk at a time, as :func:`metrics` reads one.
    ``spec``
        The spec's name, or the spec dict itself, as :func:`metrics` reads
        it.
    ``clock``
        The column whose values place each row in a window: a number, a
        ``Datetime``, a ``Date`` or a ``Duration``. A row whose clock is
        null is in no window and is not scored, as Polars leaves a null key
        out of a group, and so is a row whose float clock is NaN or
        infinite, which no window holds. A ``Time`` is refused, since it
        starts again at midnight.
    ``every``
        The length of each window, in the clock's units. On a numeric clock
        it is a number, finite and above 0, and a whole number on an
        integer clock, so a window's edges are values of the column. On a
        ``Datetime`` or ``Date`` clock it is a duration: text such as
        ``"1h"`` or ``"1mo"``, a ``timedelta``, or a ``pl.duration``
        expression. On a ``Duration`` clock it is a length of time with no
        months, quarters or years. Notes say where each kind of window
        starts.
    ``group``, ``targets``, ``min_samples``, ``binary``
        :func:`metrics`' own. A window with fewer than ``min_samples`` rows
        keeps its row, with ``n`` filled and the scores null.

    .. rubric:: Returns

    :func:`metrics`' columns for each window, with ``window_start``, the
    window's left edge, after the keys. ``window_start`` has the clock's
    own dtype. One row per ``slot``, ``target``, ``group`` key and window,
    sorted by them.

    .. rubric:: Raises

    Whatever :func:`unpack` raises for ``df`` and ``spec``, and:

    ``ValueError``
        ``every`` is not finite and above 0, is of the wrong kind for the
        clock, is not a whole number on an integer clock, or is a month, a
        quarter or a year on a ``Duration`` clock. Or ``every``, or the
        left edge of a window, is past what an integer clock's dtype holds,
        as a window of 100 that holds -120 starts at -200, past ``Int8``:
        cast the clock to ``Int64``. Or the ``clock`` column is named like
        a column of :func:`unpack`'s long form (:data:`RESERVED`) or
        ``window_start``. And what :func:`metrics` raises for ``group``,
        ``min_samples`` and ``binary``.
    :class:`~polars_online.WrongTypeError`
        The ``clock`` column is neither numeric nor temporal, or is a
        ``Time``, which starts again at midnight. It is both a ``TypeError``
        and a ``ValueError``, as a bank's refusal of the same clock is.
    :class:`~polars_online.ColumnNotFoundError`
        ``df`` does not have the ``clock`` column or a ``group`` column. It
        is both Polars' ``ColumnNotFoundError`` and a ``ValueError``.

    A query is refused as its result would be, with the same message,
    before it runs, and also with ``ValueError`` where a key's clock steps
    back, in a later chunk, before the windows an earlier chunk started for
    it would place it (see Notes).

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        windows = po.eval.window_metrics(out, spec, clock="t", every=100.0)   # 100 units of t each
        for start, n, r2 in windows.select("window_start", "n", "r2").iter_rows():
            print(f"from t = {start:.0f}: {n} rows, r2 {r2:.3f}")
        by_hour = po.eval.window_metrics(out, spec, clock="ts", every="1h")    # a Datetime clock
        print(by_hour.height, "windows of an hour")

    .. code-block:: text

        from t = 0: 96 rows, r2 0.940
        from t = 100: 100 rows, r2 0.941
        from t = 200: 100 rows, r2 0.956
        from t = 300: 100 rows, r2 0.926
        8 windows of an hour

    .. rubric:: See also

    - :func:`metrics`, the same scores over the whole frame.
    - The ``emit_metrics`` switch of a spec, the same scores kept by the
      model as it learns, exponentially weighted.

    .. rubric:: Notes

    The windows are those of Polars' ``group_by_dynamic(clock, every=every,
    closed="left", label="left")`` over each key (slot, target and
    ``group``), so a window here is the window Polars would make of the
    same rows:

    - On a numeric clock, a window starts at
      ``floor(clock / every) * every``, as Polars' integer index does. A
      float clock, which Polars does not window, is windowed the same way,
      in doubles: a label can read a little above the decimal edge it
      stands for, ``0.30000000000000004`` for ``0.3``, and a clock on a
      decimal edge can, rarely, fall in the window before it (7 of 2,000
      rows on a grid of 0.001 with ``every=0.01``). Give the clock in whole
      units, as an integer, where the edges must be exact.
    - On a ``Datetime`` or ``Date`` clock, ``every`` is read as Polars
      reads a duration. Nanoseconds to hours (``"ns"`` to ``"h"``) are
      elapsed time: across a change of a time zone's offset, a ``"5h"``
      window still holds five hours, and ``"24h"`` holds 24 of them. Days,
      weeks, months, quarters and years (``"d"``, ``"w"``, ``"mo"``,
      ``"q"``, ``"y"``) follow the wall calendar: ``"1d"`` runs from
      midnight to midnight, 23 or 25 hours on the day of a change, and
      ``"1mo"`` is a calendar month. A ``timedelta`` is read as Polars
      reads it, so its whole days are calendar days. A ``pl.duration``
      expression is the length of time it evaluates to. A month has no
      fixed length, so a spec's clock parameters refuse one, but a window
      takes it.
    - On a ``Duration`` clock, ``every`` is a length of time with no
      calendar, and a window starts at a whole number of ``every`` from 0.

    **Given a query, this holds one chunk of it and the sums of each window
    it has seen,** ten numbers per slot, key and window, read as
    :func:`metrics` reads a query, and Polars holds what it has read ahead
    of them, which `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds. Scoring a scan of a two-spec output by the day and by a key of
    four values held 102 MiB at a million rows, 108 MiB at four million and
    110 MiB at sixteen million under :func:`metrics`' bounded
    settings, and 236, 416 and 715 MiB under Polars'
    defaults. The scores are the frame's as :func:`metrics`' are.

    Polars starts the windows of a time-zoned clock whose ``every`` is a
    fixed length that does not divide a day, such as ``"90m"``, from a
    key's first clock rather than from midnight. So each chunk of a query
    is windowed from the first clock of the key in the chunks before it,
    and its windows are the frame's. A key whose clock steps back, in a
    later chunk, to before that first clock would be windowed from the new
    one in the frame, so where that moves its windows the query is refused
    with ``ValueError``: sort the query by the clock within each key, or
    pass the collected frame.
    """
    who = "po.eval.window_metrics"
    spec_name, written = _spec_of(who, spec)
    _check_min_samples(who, min_samples)
    probe = _probe(df)
    dtype = probe.schema.get(clock)
    if dtype is None:
        # Named before the window is checked against it (review R2, P9).
        msg = (
            f"po.eval.window_metrics: clock column {clock!r} is not in the frame, whose "
            f"columns are {probe.columns}. Name a column of the frame as clock."
        )
        raise ColumnNotFoundError(msg)
    _check_columns(probe, who, group=group, clock=clock, produced=_PRODUCED)
    # The clock's dtype before `every` is read against it: a String clock was
    # refused for the window, not for itself (task 352).
    if not (dtype.is_numeric() or dtype.is_temporal()):
        msg = (
            f"po.eval.window_metrics: clock column {clock!r} must be numeric or temporal, but "
            f"it is {dtype}. Cast the clock to a number or a Datetime."
        )
        raise WrongTypeError(msg)
    ns: int | None
    narrow: pl.DataType | None = None
    if isinstance(dtype, (pl.Datetime, pl.Date)) and isinstance(every, str) and _is_calendar(every):
        ns = _calendar_every(every, dtype, clock)
    else:
        ns = clock_nanoseconds(every, dtype, "po.eval.window_metrics", "every", clock)
    if ns is None and not every > 0:  # type: ignore[operator]
        msg = (
            f"po.eval.window_metrics: every must be above 0, but it is {every}. Give every a "
            f"positive number of clock units."
        )
        raise ValueError(msg)
    if ns is None and math.isinf(every):  # type: ignore[arg-type]
        # One bucket, whose `window_start` was NaN (review 2026-10-05, YB11).
        msg = (
            f"po.eval.window_metrics: every must be finite, but it is {every}. Give every a "
            "finite number of clock units, or use po.eval.metrics for one window over the "
            "whole frame."
        )
        raise ValueError(msg)
    if ns is None and dtype.is_integer():
        # The bucket's left edge in the clock's own dtype, as a temporal
        # clock's is, in integers, so no edge is rounded through a float; a
        # window that is not whole would put edges between the column's
        # values (review round 4, YB14).
        if not float(every).is_integer():  # type: ignore[arg-type]
            msg = (
                f"po.eval.window_metrics: every must be a whole number, since clock column "
                f"{clock!r} is {dtype}, but it is {every!r}. Give every a whole number, or cast "
                "the clock to Float64."
            )
            raise ValueError(msg)
        whole = int(every)  # type: ignore[arg-type]
        low, high = _int_range(dtype)
        if whole > high:
            # Polars' own cast error, from the supertype the literal took
            # (task 382a, A13).
            msg = (
                f"po.eval.window_metrics: every is {every!r}, which clock column {clock!r}, "
                f"{dtype}, cannot hold, so no window's edge but 0 is a value of the column. Cast "
                "the clock to Int64, or give a smaller every."
            )
            raise ValueError(msg)
        if -(2**63) < low < 0:
            # In Int64, and back to the clock's dtype only where the edge
            # fits: the window of 100 that holds -120 starts at -200, which
            # Int8 wrapped to 56 (task 382a, A13).
            start = pl.col(clock).cast(pl.Int64) // whole * whole
            narrow = dtype
        else:
            start = (pl.col(clock) // whole * whole).cast(dtype)
    elif ns is None:
        start = (pl.col(clock) / every).floor() * every
    elif isinstance(dtype, pl.Duration):
        start = (
            (pl.col(clock).dt.total_nanoseconds() // ns * ns).cast(pl.Duration("ns")).cast(dtype)
        )
    else:
        start = None
    keys = ["slot", "target", *_group_keys(group), "window_start"]
    binary = _binary(binary, written)
    every_text = None if start is not None else _polars_every(every, ns)
    if isinstance(df, pl.LazyFrame):
        job = _job(df, written or spec_name, targets, group, None, binary, who, log_loss=True,
                   also=[clock])  # fmt: skip
        windows = _windowing(clock, start, every_text, keys[:-1], every, narrow)
        merged = _query_sums(df, [job], keys=keys, weight=None, prepare=windows)[0]
        if merged is not None:
            return _thin(_scores(merged, keys, binary=binary), min_samples, keys)
        df = probe
    job = _job(df, written or spec_name, targets, group, None, binary, who, log_loss=True,
               also=[clock])  # fmt: skip
    windows = _windowing(clock, start, every_text, keys[:-1], every, narrow)
    s = _frame_sums(df, job, keys, None, windows)
    return _thin(_scores(s, keys, binary=binary), min_samples, keys)


def _windowing(
    clock: str,
    start: pl.Expr | None,
    every: str | timedelta | None,
    keys: list[str],
    given: object,
    narrow: pl.DataType | None = None,
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """What puts each scored row of a long form in its window: the rows with
    a clock, each with its ``window_start``, by ``start`` or by Polars'
    ``group_by_dynamic`` over ``keys`` (``every``). Called on each chunk of
    a query in turn, it starts a key's windows where its first chunk
    started them: Polars counts a time-zoned clock's windows of a length
    that does not divide a day from a key's first row, so each chunk is
    windowed after a row at that first clock, which is then dropped.
    ``narrow`` is a signed integer clock's dtype narrower than ``Int64``,
    whose ``start`` is computed in ``Int64`` and cast back where it fits."""
    anchors: pl.DataFrame | None = None
    mark = "__po_anchor"

    def windowed(long: pl.DataFrame) -> pl.DataFrame:
        nonlocal anchors
        # A row with no clock is in no window, as Polars leaves a null key out
        # of `group_by_dynamic`; scored as a window of its own, it was a row
        # of null `window_start` (task 359). A float clock's NaN or infinity
        # is as unusable, and made windows labelled NaN, inf and -inf (task
        # 382a, A7).
        has = pl.col(clock).is_not_null()
        if long.schema[clock].is_float():
            has = has & pl.col(clock).is_finite()
        long = long.filter(has)
        if start is not None:
            out = long.with_columns(start.alias("window_start"))
            if narrow is not None:
                edge = out.select(pl.col("window_start").arg_min()).item()
                if edge is not None and out["window_start"][edge] < _int_range(narrow)[0]:
                    msg = (
                        f"po.eval.window_metrics: the window of every={given!r} that holds "
                        f"clock {out[clock][edge]} starts at {out['window_start'][edge]}, which "
                        f"clock column {clock!r}, {narrow}, cannot hold. Cast the clock to Int64."
                    )
                    raise ValueError(msg)
                out = out.with_columns(pl.col("window_start").cast(narrow))
            return out
        assert every is not None
        firsts = long.group_by(keys).agg(pl.col(clock).min())
        if anchors is None:
            anchors = firsts
            return _dynamic_windows(long, clock, every, keys)
        seen = firsts.join(anchors, on=keys, how="inner", nulls_equal=True, suffix="_was")
        back = seen.filter(pl.col(clock) < pl.col(f"{clock}_was"))
        for i in range(back.height):
            _check_step_back(back.slice(i, 1), clock, every, keys, given)
        # Each key's earliest clock so far, and the keys new in this chunk.
        earlier = seen.select(*keys, pl.min_horizontal(clock, f"{clock}_was").alias(clock))
        anchors = pl.concat(
            [
                anchors.join(earlier, on=keys, how="anti", nulls_equal=True),
                earlier,
                firsts.join(anchors, on=keys, how="anti", nulls_equal=True),
            ]
        )
        here = anchors.join(firsts.select(keys), on=keys, how="semi", nulls_equal=True)
        both = pl.concat(
            [
                here.with_columns(pl.lit(True).alias(mark)),
                long.with_columns(pl.lit(False).alias(mark)),
            ],
            how="diagonal",
        )
        return _dynamic_windows(both, clock, every, keys).filter(~pl.col(mark)).drop(mark)

    return windowed


def _check_step_back(
    row: pl.DataFrame, clock: str, every: str | timedelta, keys: list[str], given: object
) -> None:
    """Refuse a key whose clock steps back, in a later chunk of a query, to
    before the first clock of its earlier chunks, where its windows counted
    from that earlier first clock are not the windows counted from the new
    one. A frame in memory is windowed from each key's earliest clock."""
    was, now = row[f"{clock}_was"][0], row[clock][0]
    key = {k: row[k][0] for k in keys}
    pair = pl.concat(
        [row.select(*keys, clock), row.select(*keys, pl.col(f"{clock}_was").alias(clock))]
    )
    chained = _dynamic_windows(pair, clock, every, keys)["window_start"][1]
    alone = _dynamic_windows(pair.tail(1), clock, every, keys)["window_start"][0]
    if chained != alone:
        msg = (
            f"po.eval.window_metrics: clock column {clock!r} steps back to {now} for "
            f"{key}, before {was}, where an earlier chunk of the query started its windows of "
            f"{given!r}, and windows of that length on this clock are counted from a key's "
            "first clock, so the chunks' windows would not be the frame's. Sort the query by "
            "the clock within each key, or run the query first and pass its result."
        )
        raise ValueError(msg)


#: Renamed :func:`window_metrics`, and its ``window_size`` ``every``, before
#: 1.0: the old names are refused, and the error names the new ones.
rolling_metrics = renamed_function(
    "po.eval.rolling_metrics", "po.eval.window_metrics", ", and window_size was renamed every"
)


def _takes_specs[R](fn: Callable[..., R]) -> Callable[..., R]:
    """:func:`compare_specs`' ``spec_names`` and ``specs=`` merged into one
    ``specs``, names or dicts: the old forms read as the new one, with a
    :class:`~polars_online.PolarsOnlineDeprecationWarning`, as
    :func:`_takes_spec` does for one spec."""
    who = "po.eval.compare_specs"

    @functools.wraps(fn)
    def wrapper(df: Any, *args: Any, **kwargs: Any) -> R:
        _frame(who, df, lazy=True)
        if "spec_names" in kwargs:
            if args:
                msg = (
                    f"{who}: spec_names was merged into specs, but both are given. "
                    f"Pass the specs alone as specs."
                )
                raise TypeError(msg)
            _warn_merged(who, "spec_names")
            args = (kwargs.pop("spec_names"),)
        if args and "specs" in kwargs:
            written = {s.get("name"): s for s in kwargs.pop("specs") or ()}
            _warn_merged(who, "a list of names beside specs=")
            args = ([written.get(s, s) if isinstance(s, str) else s for s in args[0]],)
        return fn(df, *args, **kwargs)

    return wrapper


@renamed_keywords("po.eval.compare_specs", _BOTH)
@_takes_specs
def compare_specs(
    df: pl.DataFrame | pl.LazyFrame,
    specs: Iterable[str | dict],
    *,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    min_samples: int = 30,
    binary: bool | None = None,
) -> pl.DataFrame:
    """Score several specs' out-of-sample predictions in one table.

    Use it to read side by side how well each of several specs predicted,
    over a whole frame. To test whether one spec's residuals are smaller
    than another's, use :func:`diebold_mariano`, :func:`clark_west` or
    :func:`seqtest`; to choose among the grid points of one spec, use
    :func:`grid_search`.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding each spec's output column, or a
        query of one, a ``LazyFrame``, which this runs once for every spec
        and reads one chunk at a time, as :func:`metrics` reads one.
    ``specs``
        The specs that wrote the columns, each its name or its dict, read
        as :func:`metrics` reads its ``spec``.
    ``group``, ``targets``, ``min_samples``
        :func:`metrics`' own, applied to every spec alike.
    ``binary``
        :func:`metrics`' own, applied to every spec alike. ``None``, the
        default, reads it from the spec dicts' losses, and refuses specs
        whose losses read it both ways, since a classifier's scores and a
        regression's are different columns.

    .. rubric:: Returns

    :func:`metrics`' table for each spec, one under another, with a
    ``spec`` column first holding the spec's name. No specs give an empty
    frame.

    .. rubric:: Raises

    Whatever :func:`metrics` raises for each spec, in this function's name,
    and ``ValueError`` when ``binary`` is ``None`` and some specs are fits
    with ``loss="logistic"`` and others are not (pass the two sets in two
    calls, or set ``binary``), or when a ``group`` column is named
    ``spec``, as the result's first column is.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        x0_only = po.spec.ewridge("x0_only", targets=["y"], features=["x0"], half_life=100.0)
        out = po.ModelBank([ridge, x0_only]).fit_predict(df)
        table = po.eval.compare_specs(out, [ridge, x0_only])   # one row per spec and slot
        for name, n, r2, mse in table.select("spec", "n", "r2", "mse").iter_rows():
            print(f"{name}: {n} rows, r2 {r2:.3f}, mse {mse:.3f}")

    .. code-block:: text

        ridge: 396 rows, r2 0.944, mse 0.273
        x0_only: 397 rows, r2 0.728, mse 1.330

    .. rubric:: See also

    - :func:`metrics`, the same scores for one spec.
    - :func:`diebold_mariano` and :func:`clark_west`, tests of whether one
      spec's squared error is lower than another's.
    - :func:`seqtest`, evidence that one spec predicts closer on more rows,
      to read at every row.
    - `Compare two models
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/compare-two-models.md>`_,
      with a recipe that compares two models on 200 streams.

    .. rubric:: Notes

    Each spec is scored on its own rows, so two specs whose warm-ups differ
    in length are scored on different rows, as the 396 and 397 rows in the
    example are. Diebold and Mariano's test and Clark and West's read only
    the rows where both specs have a residual. Given a query, every spec's
    sums are read in the one run of it, beside what Polars reads ahead of
    the call, which `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds: scoring two specs of a scan by a key of four values held 91
    MiB at a million rows, 106 MiB at four million and 108 MiB at sixteen
    million under :func:`metrics`' bounded settings, and 242,
    450 and 871 MiB under Polars' defaults.

    **Diebold and Mariano's (1995) test asks whether two specs' squared
    errors differ on average.** Its statistic is the mean of the loss
    differential ``d = resid_b² − resid_a²`` over Newey and West's
    standard error, and is about ``N(0, 1)`` when the two predict equally
    well. :func:`diebold_mariano` computes it. Its ``lags``, 0 by default,
    are right at 0 only where the differentials are uncorrelated from row
    to row: two specs that each leave out something persistent make
    one-step errors that run, and then ``lags=0`` found two equally
    accurate specs to differ at 5% on 24% to 32% of simulated streams.
    For a target that looks ``h`` rows ahead, give about ``2 h``; its
    ``lags`` entry gives the sizes measured.

    **Between two nested specs, Diebold and Mariano's test leans toward
    the smaller one, and Clark and West's (2007) correction removes the
    lean.** When the larger spec's extra coefficients are truly zero, it
    still estimates them, and their noise adds ``(pred_small −
    pred_big)²`` to its squared error on average. So under the null the
    differential favors the smaller spec, and the test rarely finds the
    larger one better even when it is (Clark and McCracken, 2001). Clark
    and West add the term back:

    .. code-block:: text

        f  = resid_small² − resid_big² + (resid_big − resid_small)²
        cw = f̄ · n / sqrt(S)            one-sided: cw > 1.645 rejects at 5%

    Here ``f̄`` is the mean of ``f`` over the ``n`` rows, and ``S`` is
    Newey and West's long-run variance of ``f`` times ``n``, as
    :func:`diebold_mariano` gives it for ``d``. :func:`clark_west` computes
    it. With ``half_life``, each of the two functions gives its statistic on
    every row instead, exponentially weighted.

    :func:`seqtest`, and the `seqtest
    <https://github.com/hgilde/polars-online/blob/main/docs/MODELS.md#seqtest--a-sequential-test-of-a-sign-by-betting>`_
    model a bank runs, give evidence
    that one spec predicts closer on more rows, by betting on the sign of
    ``|resid_b| − |resid_a|``. That is not the question the two tests
    above ask: a spec with rare large errors can be closer on most rows and
    have the larger mean squared error. The evidence can be read at every
    row, as often as you like, without increasing the false-alarm rate:
    reject at the 5% level the first time ``log_e_a_<t>`` reaches ``ln
    20``. The 5% holds where, at every row and whatever came before, the
    next sign is no more likely positive than negative, which is more than
    the two being equally close on average: equally accurate specs whose
    errors run from row to row give signs that run, and both bets grow
    (:func:`seqtest`'s Notes give the rates measured).
    """
    given = [_spec_of("po.eval.compare_specs", s) for s in specs]
    if binary is None:
        reads = {_logistic(written) for _, written in given}
        if len(reads) > 1:
            logistic = [name for name, written in given if _logistic(written)]
            others = [name for name, written in given if not _logistic(written)]
            msg = (
                f'po.eval.compare_specs: specs {logistic} are fits with loss="logistic" and '
                f"{others} are not, and the metrics of the two differ in kind. Pass the two sets "
                "in two calls, or set binary to True or False."
            )
            raise ValueError(msg)
        binary = reads.pop() if reads else False
    who = "po.eval.compare_specs"
    _check_min_samples(who, min_samples)
    _check_columns(_probe(df), who, group=group, produced=(*_PRODUCED, "spec"))
    if isinstance(df, pl.LazyFrame):
        frames = _compare_query(df, given, group, targets, min_samples, binary)
    else:
        frames = [
            _metrics(
                df,
                written or name,
                group=group,
                targets=targets,
                min_samples=min_samples,
                binary=binary,
                who=who,
            ).with_columns(pl.lit(name).alias("spec"))
            for name, written in given
        ]
    cols = ["spec", *frames[0].columns[:-1]] if frames else []
    return pl.concat(frames).select(cols) if frames else pl.DataFrame()


def _compare_query(
    lf: pl.LazyFrame,
    given: list[tuple[str, dict | None]],
    group: str | Iterable[str] | None,
    targets: Sequence[str] | None,
    min_samples: int,
    binary: bool,
) -> list[pl.DataFrame]:
    """:func:`compare_specs`' tables over a query, every spec's sums read in
    one pass: each spec checked as :func:`metrics` checks it, in order, in
    :func:`compare_specs`' name."""
    who = "po.eval.compare_specs"
    probe = _probe(lf)
    jobs = [
        _job(lf, written or name, targets, group, None, binary, who, log_loss=True)
        for name, written in given
    ]
    keys = _keys(group)
    frames = []
    for (name, written), merged in zip(
        given, _query_sums(lf, jobs, keys=keys, weight=None) if jobs else [], strict=True
    ):
        if merged is None:
            out = _metrics(probe, written or name, group=group, targets=targets,
                           min_samples=min_samples, binary=binary, who=who)  # fmt: skip
        else:
            out = _thin(_scores(merged, keys, binary=binary), min_samples, keys)
        frames.append(out.with_columns(pl.lit(name).alias("spec")))
    return frames


#: :func:`grid_search`'s scorings: each names the column it writes and
#: whether a lower value is the better one.
_SCORINGS = {"mse": True, "loglik": False}

#: The settings a spec's own grid varies, as :func:`polars_online.spec.output_index`
#: names them: :func:`grid_search` writes a column for each that varies.
_GRID_COLUMNS = ("half_life", "ridge", "feature_set", "penalty")

#: :func:`grid_search`'s own columns, which a ``param_grid`` key may not take.
_SEARCH_COLUMNS = ("kind", "spec", "slot", "target", "n", "rank", "folds_chosen", *_SCORINGS)

#: What each of :func:`grid_search`'s rows is, by the field's kind in
#: :func:`polars_online.spec.output_index`: a setting to choose among, or a
#: spec's own pick or blend of its grid, scored beside them and never chosen.
_SEARCH_KINDS = {
    "pred": "candidate",
    "loglik": "candidate",
    "pred_selected": "selected",
    "pred_averaged": "averaged",
}


def _points(param_grid: Any) -> list[dict[str, Any]]:
    """``param_grid`` as its list of points, in scikit-learn's
    ``ParameterGrid`` order: the keys sorted, the last key varying fastest."""
    who = "po.eval.grid_search"
    if not isinstance(param_grid, Mapping) or not param_grid:
        msg = (
            f"{who}: param_grid must be a non-empty dict of each keyword's values, but it is "
            f'{param_grid!r}. Pass a dict such as {{"half_life": [50, 500]}}.'
        )
        raise TypeError(msg) if not isinstance(param_grid, Mapping) else RefusedValueError(msg)
    keys = sorted(param_grid)
    for key in keys:
        values = param_grid[key]
        if not isinstance(key, str):
            msg = (
                f"{who}: each key of param_grid must be a keyword of the spec builder, but one "
                f"is {key!r}. Use the builder's keyword names, as str, for keys."
            )
            raise TypeError(msg)
        if key in _SEARCH_COLUMNS:
            msg = (
                f"{who}: param_grid key {key!r} is the name of one of the result's own columns. "
                f"Rename the builder's keyword, or wrap the builder in a function that takes "
                f"another name."
            )
            raise ValueError(msg)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
            msg = (
                f"{who}: param_grid[{key!r}] must be a list of the values to try, but it is "
                f"{values!r}. Give a non-empty list."
            )
            wrong_type = isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
            raise TypeError(msg) if wrong_type else RefusedValueError(msg)
    return [
        dict(zip(keys, point, strict=True))
        for point in itertools.product(*(param_grid[k] for k in keys))
    ]


def _search_specs(
    specs: Any, param_grid: Mapping[str, Sequence[Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The specs to run, and each one's point of the grid (``{}`` without one)."""
    who = "po.eval.grid_search"
    if callable(specs):
        points = _points(param_grid)
        built = []
        for point in points:
            name = ", ".join(f"{k}={v}" for k, v in point.items())
            spec = specs(name, **point)
            if not isinstance(spec, dict) or spec.get("name") != name:
                got = spec.get("name") if isinstance(spec, dict) else type(spec).__name__
                msg = (
                    f"{who}: the spec builder must return a spec named by its first argument, "
                    f"but called with {name!r} it returned {got!r}. Name the spec it returns "
                    f"by its first argument, as po.spec's builders do."
                )
                raise ValueError(msg)
            built.append(spec)
        return built, points
    if param_grid is not None:
        msg = (
            f"{who}: param_grid needs a spec builder to call, but specs holds specs already "
            f"built. Pass a builder, such as po.spec.ewridge, as specs, or leave out param_grid."
        )
        raise RefusedValueError(msg)
    given = [specs] if isinstance(specs, dict) else list(specs)
    if not given or not all(isinstance(s, dict) for s in given):
        if not isinstance(specs, (list, tuple)):
            got = f"of type {type(specs).__name__}"
        elif not given:
            got = "an empty list"
        else:
            got = "a list holding something other than a spec"
        msg = (
            f"{who}: specs must be a spec, a list of specs or a spec builder, but it is {got}. "
            f"Pass one of those."
        )
        raise TypeError(msg)
    return given, [{} for _ in given]


def _time_series_folds(n: int, n_splits: int, gap: int) -> list[tuple[int, int, int]]:
    """Each fold as ``(choose_end, test_start, test_end)``, in rows: scikit-learn's
    ``TimeSeriesSplit(n_splits, gap=gap)`` over ``n`` rows, each fold's choice read
    from the rows before ``choose_end`` and scored on ``test_start:test_end``."""
    who = "po.eval.grid_search"
    # Any whole number, numpy's integers included (task 352), but not a bool.
    wrong_type = isinstance(n_splits, bool) or not isinstance(n_splits, numbers.Integral)
    if wrong_type or operator.index(n_splits) < 1:
        msg = (
            f"{who}: n_splits must be a whole number of at least 1, but it is {n_splits!r}. "
            f"Set n_splits to 1 or more."
        )
        raise WrongTypeError(msg) if wrong_type else ValueError(msg)
    wrong_type = isinstance(gap, bool) or not isinstance(gap, numbers.Integral)
    if wrong_type or operator.index(gap) < 0:
        msg = (
            f"{who}: gap must be a whole number of rows of at least 0, but it is {gap!r}. "
            f"Set gap to 0 or more."
        )
        raise WrongTypeError(msg) if wrong_type else ValueError(msg)
    n_splits, gap = operator.index(n_splits), operator.index(gap)
    test_size = n // (n_splits + 1)
    first = n - n_splits * test_size
    if test_size < 1:
        have = "1 row is" if n == 1 else f"{n} rows are"
        remedy = (
            f"fewer n_splits, {n - 1} at most, or more rows"
            if n >= 2
            else f"{n_splits + 1} rows at least, or fewer n_splits"
        )
        msg = (
            f"{who}: n_splits={n_splits} cuts the rows into {n_splits + 1} spans of a row or "
            f"more, but {have} too few. Give {remedy}."
        )
        raise ValueError(msg)
    if first - gap < 1:
        msg = (
            f"{who}: the first span, rows 0-{first - 1} of {n} at n_splits={n_splits}, is "
            f"{first} row{'' if first == 1 else 's'}, and gap={gap} leaves none of them to "
            f"choose from. Give a smaller gap, below {first}, fewer n_splits, or more rows."
        )
        raise ValueError(msg)
    return [(s - gap, s, s + test_size) for s in range(first, n, test_size)]


def _search_slots(
    specs: list[dict[str, Any]], frame: pl.DataFrame, scoring: str
) -> list[dict[str, Any]]:
    """One record per field :func:`grid_search` scores: its spec, field and
    target, the target's value as an expression, the spec grid's settings,
    and what its row is: ``"candidate"``, a setting to choose among, or
    ``"selected"`` and ``"averaged"``, ``emit_selected``'s and
    ``emit_averaged``'s fields, scored beside them and never chosen."""
    from polars_online import spec as _spec_mod

    slots = []
    for i, spec in enumerate(specs):
        idx = _spec_mod.output_index(spec)
        if scoring == "loglik":
            rows = idx.filter(pl.col("kind") == "loglik")
            if rows.is_empty():
                msg = (
                    f'po.eval.grid_search: spec "{spec["name"]}" writes no loglik field to score '
                    'by scoring="loglik", which only bocpd, hmm and deco write. Use '
                    'scoring="mse" for a spec that predicts a target.'
                )
                raise ValueError(msg)
            for r in rows.iter_rows(named=True):
                slots.append(
                    {
                        "spec": i,
                        "field": r["field"],
                        "target": None,
                        "y": None,
                        "kind": _SEARCH_KINDS[r["kind"]],
                        **{c: r[c] for c in _GRID_COLUMNS},
                    }
                )
            continue
        targets = _spec_targets(spec, frame, "po.eval.grid_search")
        if not targets:
            msg = (
                f'po.eval.grid_search: spec "{spec["name"]}" makes no prediction of a target to '
                'score by scoring="mse", and only bocpd, hmm and deco write the loglik that '
                'scoring="loglik" scores. Pass specs that predict a target, or bocpd, hmm or deco '
                'specs with scoring="loglik".'
            )
            raise ValueError(msg)
        rows = idx.filter(pl.col("kind").is_in(["pred", "pred_selected", "pred_averaged"]))
        for r in rows.iter_rows(named=True):
            if r["kind"] == "pred":
                if r["field"] not in targets:
                    continue
                target, y = targets[r["field"]]
            else:
                target = r["target"]
                y = next(v for t, v in targets.values() if t == target)
            slots.append(
                {
                    "spec": i,
                    "field": r["field"],
                    "target": target,
                    "y": y,
                    "kind": _SEARCH_KINDS[r["kind"]],
                    **{c: r[c] for c in _GRID_COLUMNS},
                }
            )
    _check_same_data(specs, slots, scoring)
    return slots


def _check_same_data(
    specs: list[dict[str, Any]], slots: list[dict[str, Any]], scoring: str
) -> None:
    """Refuse to rank slots that score different data against each other: a
    log density compares two models only as densities of the same columns,
    and a squared error two predictions only of the same quantity. Ranked
    together, ``bocpd`` over a column and over that column times 1000 read
    ranks by the scale alone, and two targets named ``ret`` read from ``y``
    and from a column a thousand times larger got one choice (task 382a,
    C5)."""
    who = "po.eval.grid_search"
    if scoring == "loglik":
        data: dict[frozenset[str], list[str]] = {}
        for spec in specs:
            data.setdefault(frozenset(spec.get("features") or ()), []).append(spec["name"])
        if len(data) > 1:
            sets = "; ".join(f"{names} model {sorted(cols)}" for cols, names in data.items())
            msg = (
                f'{who}: scoring="loglik" ranks log densities, which compare two models only '
                f"as densities of the same columns, but the specs model different ones: {sets}. "
                "Pass the specs of each set of columns in a call of their own."
            )
            raise ValueError(msg)
        return
    read: dict[str, dict[str, list[str]]] = {}
    for slot in slots:
        names = read.setdefault(slot["target"], {}).setdefault(str(slot["y"]), [])
        if specs[slot["spec"]]["name"] not in names:
            names.append(specs[slot["spec"]]["name"])
    for target, columns in read.items():
        if len(columns) > 1:
            ways = "; ".join(f"{names} read {column}" for column, names in columns.items())
            msg = (
                f"{who}: target {target!r} is read from different columns by different specs "
                f"({ways}), so their squared errors are of different quantities and cannot be "
                "ranked together. Name each target after what it reads, with "
                "po.target(column, name=...), or score the specs in separate calls."
            )
            raise ValueError(msg)


def _slot_sums(
    chunk: pl.DataFrame, specs: list[dict[str, Any]], slots: list[dict[str, Any]], scoring: str
) -> tuple[list[tuple[float, int, int]], dict[str | None, int]]:
    """Each slot's sum of squared errors, or of ``loglik``, and its row count,
    over the rows of ``chunk`` where every slot of its target has a value,
    with the rows it has a value on alone; and the rows each target is
    present on (under ``loglik``, every row). The last two only name the
    late setting when no row is common to all."""

    def value(s: dict[str, Any]) -> pl.Expr:
        return pl.col(specs[s["spec"]]["name"]).struct.field(s["field"]).cast(pl.Float64)

    exprs = []
    for target in dict.fromkeys(s["target"] for s in slots):
        mine = [(k, s) for k, s in enumerate(slots) if s["target"] == target]
        common = pl.all_horizontal(
            [_usable(value(s)) for _, s in mine]
            + ([] if scoring == "loglik" else [_usable(mine[0][1]["y"].cast(pl.Float64))])
        ).fill_null(False)
        known = pl.len() if scoring == "loglik" else _usable(mine[0][1]["y"].cast(pl.Float64)).sum()
        exprs.append(known.cast(pl.Int64).alias(f"t{mine[0][0]}"))
        for k, s in mine:
            v = value(s) if scoring == "loglik" else (value(s) - s["y"].cast(pl.Float64)) ** 2
            exprs.append(v.filter(common).sum().alias(f"s{k}"))
            exprs.append(common.sum().cast(pl.Int64).alias(f"n{k}"))
            exprs.append(_usable(value(s)).sum().cast(pl.Int64).alias(f"o{k}"))
    row = chunk.select(exprs).row(0, named=True)
    sums = [(float(row[f"s{k}"]), int(row[f"n{k}"]), int(row[f"o{k}"])) for k in range(len(slots))]
    present = {s["target"]: int(row[f"t{k}"]) for k, s in enumerate(slots) if f"t{k}" in row}
    return sums, present


def grid_search(
    source: pl.LazyFrame | pl.DataFrame,
    specs: dict[str, Any] | Sequence[dict[str, Any]] | Callable[..., dict[str, Any]],
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    *,
    n_splits: int = 5,
    gap: int = 0,
    scoring: Literal["mse", "loglik"] = "mse",
    chunk_size: int | None = None,
    threads: int | None = None,
) -> pl.DataFrame:
    """Score every grid point in one pass over a stream, and choose one from
    the rows before each test span.

    Use it to choose a spec's settings, such as its half-life, ridge or
    learning rate, from the stream's own out-of-sample residuals, and to learn
    what that choice is worth on rows it did not see. To compare a few
    specs you have already run, use :func:`compare_specs`.

    .. rubric:: Parameters

    ``source``
        The stream: a ``LazyFrame`` or a ``DataFrame``. It is read once,
        ``chunk_size`` rows at a time, through one bank of every spec. A
        query is counted first, to place the spans: for a query that reads
        parquet files, Polars reads the count from their footers, and any
        other query runs once more without the bank.
    ``specs``
        A spec, a list of specs, or a spec builder called as
        ``specs(name, **point)`` for each point of ``param_grid``, such as
        ``functools.partial(po.spec.sgd, targets=["y"], features=["x"])``.
        The name a builder is given is the point written out, such as
        ``"learning_rate=0.01"``, and the spec it returns must take that
        name.
    ``param_grid``
        A dict of each builder keyword's values to try, as scikit-learn's
        ``ParameterGrid`` takes one: its keys sorted, the last varying
        fastest, as ``{"learning_rate": [0.001, 0.01]}``. ``None``, the
        default, when ``specs`` are built already. A key may not be one of
        the result's own columns.
    ``n_splits``
        The number of test spans, 5 by default. For 2 or more, the rows
        are cut as scikit-learn's ``TimeSeriesSplit(n_splits, gap=gap)``
        cuts them: the ``n`` rows into ``n_splits + 1`` spans, the last
        ``n_splits`` of ``n // (n_splits + 1)`` rows each. 1, which
        scikit-learn refuses, gives one test span of the last ``n // 2``
        rows. Fewer spans make the first span, the rows the first choice is
        made from, longer.
    ``gap``
        The rows left out between the rows a choice is made from and the
        span it is scored on, 0 by default. Give it the number of rows of
        the stream a target looks ahead, so a choice never reads a row
        whose target was not yet known when its span started. It counts
        rows of the stream, as scikit-learn's ``gap`` counts samples, not
        steps of one series: with ``G`` series interleaved, as under a
        spec's ``group``, a target ``h`` steps ahead in its series looks
        about ``h * G`` rows ahead, and ``gap`` should be that.
    ``scoring``
        ``"mse"``, the default, the mean of ``(y - pred)**2``, lower
        better; or ``"loglik"``, the mean log predictive density of a row,
        higher better, read from the ``loglik`` field that ``bocpd``,
        ``hmm`` and ``deco`` write.
    ``chunk_size``
        The rows read at a time, as
        :meth:`~polars_online.ModelBank.fit_predict_batches` takes it.
        ``None``, the default, reads a query 100,000 rows at a time and
        takes a ``DataFrame`` whole, as that method does.
    ``threads``
        The bank's threads, as :class:`~polars_online.ModelBank` takes them.
        ``None``, the default, runs on the threads banks share, one per core
        unless ``POLARS_ONLINE_MAX_THREADS`` is set.

    .. rubric:: Returns

    A ``DataFrame`` that holds, for each target, one row per slot (one
    target at one grid point of a spec's own grid, or a spec's
    ``emit_selected`` or ``emit_averaged`` prediction) in grid order, then
    the row of the choice:

    ``kind``
        What the row is: ``"candidate"``, a slot to choose among;
        ``"selected"`` or ``"averaged"``, a spec's ``emit_selected`` or
        ``emit_averaged`` prediction, scored beside the others and never
        chosen; ``"choice"``, what choosing from the rows before each span
        scored on that span. Select it with ``pl.col("kind") == "choice"``.
    ``spec``
        The spec's name. Null on the choice's row, which runs whichever
        slot each span chose, from any spec.
    the keys of ``param_grid``, then ``half_life``, ``ridge``, ``feature_set`` and ``penalty``
        The slot's grid point: the point of ``param_grid``, and each
        setting of the specs' own grids that varies among the slots. Null
        on the choice's row.
    ``slot``, ``target``
        The output field scored, and its target (null under
        ``scoring="loglik"``). ``slot`` is null on the choice's row, and
        ``target`` names the target the choice is for.
    ``n``
        The rows scored, over the test spans.
    ``mse`` or ``loglik``
        The score over those rows.
    ``rank``
        1 for the slot with the best score over the test spans, ties
        sharing the lower rank: the grid point to run. Since it is chosen on
        the rows it reports, its score is optimistic. Null on every row but
        a ``"candidate"``.
    ``folds_chosen``
        How many test spans the choice ran the slot on. A tie goes to the
        first in grid order. Null on every row but a ``"candidate"``.

    The choice's row is the out-of-sample score of a rule: run the slot
    best over the rows read before each span (see Notes), and score it on
    the span. How far it falls short of the rank-1 score is the rule's
    regret against the best grid point in hindsight. With no ``gap`` it
    cannot be negative, since following the leader never beats the best
    fixed choice over the same spans; it is an upper bound on how much the
    rank-1 score overstates, and it holds as well what the rule lost by
    choosing from fewer rows, and the first choice from the first span's
    rows, where every model starts up.

    .. rubric:: Raises

    ``TypeError``
        ``source`` is not a ``LazyFrame`` or a ``DataFrame``; ``specs`` is
        not a spec, a list of specs or a builder; ``param_grid`` is not a
        dict; a key is not a str; or a value is not a list.
    :class:`~polars_online.RefusedValueError`
        ``param_grid`` is given beside specs already built, or is empty, or
        one of its lists is empty. It is both a ``ValueError`` and a
        ``TypeError``.
    :class:`~polars_online.WrongTypeError`
        ``n_splits``, ``gap`` or ``chunk_size`` is not a whole number. It
        is both a ``TypeError`` and a ``ValueError``.
    :class:`~polars_online.ColumnNotFoundError`
        A spec's target has no column in the stream. A formula target never
        has one, since the bank computes it as it learns and does not write
        it out, so such a spec cannot be scored here. It is both a
        ``ValueError`` and Polars' ``ColumnNotFoundError``.
    ``ValueError``
        When:

        - ``scoring`` is neither word;
        - ``n_splits`` is below 1, ``gap`` below 0, or ``chunk_size``
          below 1;
        - a builder returns a spec not named by its first argument;
        - a ``param_grid`` key is one of the result's columns;
        - a spec has nothing to score: no prediction of a target under
          ``"mse"``, or no ``loglik`` under ``"loglik"``;
        - the slots score different data, which no rank can compare: under
          ``"loglik"``, specs that model different columns (score each set
          in a call of its own); under ``"mse"``, a target name that two
          specs read from different columns (name each target after what
          it reads, with :func:`polars_online.target`);
        - a query gives a number of rows other than its count when run;
        - the first choice has no row to read. The error names the rows and
          the remedy. The stream is too short for its spans, ``gap`` is as
          long as the first span, or no row of the first span has every
          slot's prediction, as when a warm-up (``min_weight``,
          ``warm_rows``) outlasts it.
    Whatever the bank raises
        For the specs or ``threads``, or for a chunk.

    .. rubric:: Examples

    This code reads ``ticks.parquet`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        import functools

        sgd = functools.partial(po.spec.sgd, targets=["y"], features=["x0", "x1"], half_life=1000.0)
        table = po.eval.grid_search(
            pl.scan_parquet("ticks.parquet"), sgd, {"learning_rate": [0.001, 0.01, 0.1]}, n_splits=4
        )
        points = table.filter(pl.col("kind") == "candidate")     # one row per grid point
        for rate, mse, rank in points.select("learning_rate", "mse", "rank").iter_rows():
            print(f"learning_rate={rate}: mse {mse:.3f}, rank {rank}")
        choice = table.filter(pl.col("kind") == "choice")        # the score of choosing
        print(f"choosing from earlier rows: mse {choice['mse'][0]:.3f}")

    .. code-block:: text

        learning_rate=0.001: mse 3.136, rank 3
        learning_rate=0.01: mse 0.386, rank 2
        learning_rate=0.1: mse 0.304, rank 1
        choosing from earlier rows: mse 0.304

    .. rubric:: See also

    - :func:`compare_specs`, the scores of specs already run, side by side.
    - `Choose a model's settings from the stream's own errors
      <https://github.com/hgilde/polars-online/blob/main/docs/CALIBRATION.md>`_,
      the pattern this follows, for every setting.

    .. rubric:: Notes

    **Why the choice is scored apart.** Each row's prediction is made
    before the row is learned, so the stream's own residuals are
    out-of-sample, and every grid point is scored on the same rows. But
    choosing the grid point whose error is lowest over those rows, and
    reporting that error, is optimistic: each error is the grid point's
    true error plus noise, and the minimum picks the grid point whose noise
    happened to run lowest. So the choice is scored too. Before each test
    span, the slot with the best score over the rows read before it
    (``gap`` rows short of it) is chosen, and scored on the span. The
    first choice reads the first span, the only rows before it. Each later
    choice reads the test spans' rows before it, the rows the rank reads,
    and not the first span's: there every model starts up, and a model
    whose first errors were large, as one with more coefficients to learn,
    would weigh them on every choice and be passed over on spans it was
    best on. Where the gap leaves no test row before a span, or no row
    there that every slot predicts, its choice reads from row 0:

    .. code-block:: text

        size     = n // (n_splits + 1)
        start_k  = n - (n_splits - k) * size                 k = 0 .. n_splits - 1
        from_k   = start_0 where start_k - gap > start_0, else 0
        chosen_k = argbest over candidate slots s of score(s, rows [from_k, start_k - gap))
        choice   = score(chosen_k on rows [start_k, start_k + size), pooled over k)

    **What is scored.** The slots are every prediction a spec writes, one
    per target and grid point of its own grid (``half_life``, ``ridge``,
    ``feature_sets`` or ``penalties`` as lists), across all the specs. A
    row counts for a target only where every slot of that target has a
    value and the target is present. So the grid points are compared on
    the same rows, and the rows any slot withholds while it warms up count
    for none. Every row weighs alike: a spec's ``weight`` column is not
    read here, so a row of weight 0 is scored like any other. To leave
    such rows out of the score, score the specs chosen with
    :func:`sums`' ``weight``. Under ``scoring="loglik"``, the sum
    of a stream's log densities, each read before its row is learned, is
    its log marginal likelihood (the prequential score). It is the
    criterion that compares ``bocpd``'s hazards and priors. For ``hmm``'s number of
    states, see the table under *What this cannot choose*.

    **Memory and rounding.** Only three numbers per slot are kept (its summed
    score, the rows it is scored on and the rows it predicts) for each run
    of rows between the places a test span starts or its choice stops, with
    each target's count of rows.
    So memory is the bank's state and one chunk, however long the stream.
    These sums are added a chunk at a time, so the scores are the same to
    rounding whatever ``chunk_size`` is, and the choices are the same
    unless two grid points tie to rounding.

    **What this cannot choose.** Three settings have no out-of-sample score
    this function can choose them by. One model writes no score, one
    writes a score that was measured not to find the setting, and for one
    the squared error would push it toward the loss it exists to replace:

    .. list-table::
       :header-rows: 1
       :widths: 18 52 30

       * - setting
         - why this does not choose it
         - what to do
       * - ``kmeans``' ``n_clusters``
         - Its only score is ``dist``, the distance to the nearest center,
           and that falls as clusters are added even on rows not yet
           learned: on three planted clusters, its mean square on the
           second half went 1.18, 0.45, 0.38, 0.32 and 0.26 at two to six
           clusters. It writes no likelihood.
         - Choose it by what the clusters are for, or from a Gram or a
           sample with a criterion of your own.
       * - ``hmm``'s ``n_components``
         - ``hmm`` writes ``loglik``, but on a stream planted with three
           Gaussian states of spread 1 at means −2, 0 and 2, four states
           scored −1.60 a row on the second half and three scored −1.67, so
           the score is not shown to find the number of states.
         - Read ``n_components`` from what the states mean.
       * - ``huber_delta``
         - It is the cut, in units of the residuals' spread, past which a
           row's residual counts linearly. The default 1.345 keeps 95% of
           least squares' efficiency when the noise is normal, and choosing
           it by squared error would push it toward least squares.
         - Keep 1.345, or choose it by a robust loss of your own over
           :func:`unpack`'s residuals.
    """
    from polars_online._bank import ModelBank

    who = "po.eval.grid_search"
    if scoring not in _SCORINGS:
        msg = (
            f'{who}: scoring must be "mse" or "loglik", but it is {scoring!r}. '
            f"Set scoring to one of those words."
        )
        raise ValueError(msg)
    wrong_type = chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral)
    )
    if wrong_type or (chunk_size is not None and chunk_size < 1):
        # Refused here, where the bank's own refusal named a method the
        # caller did not call (task 382a, C10).
        msg = (
            f"{who}: chunk_size must be a whole number of rows of at least 1, or None, but it "
            f"is {chunk_size!r}. Give a number of rows, or None for the default."
        )
        raise WrongTypeError(msg) if wrong_type else ValueError(msg)
    built, points = _search_specs(specs, param_grid)
    if isinstance(source, pl.LazyFrame):
        n = int(source.select(pl.len()).collect().item())
    elif isinstance(source, pl.DataFrame):
        n = source.height
    else:
        msg = (
            f"{who}: source must be a query or a frame, but it is of type {type(source).__name__}. "
            f"Pass a Polars LazyFrame or DataFrame."
        )
        raise TypeError(msg)
    folds = _time_series_folds(n, n_splits, gap)
    # The rows split at every place a fold starts or its choice stops, so a
    # span's sums are the sums of whole segments.
    cuts = sorted({0, n, *(c for f in folds for c in f[:2])})
    segments = list(itertools.pairwise(cuts))
    slots: list[dict[str, Any]] | None = None
    totals: list[list[tuple[float, int, int]]] = []
    known: list[dict[str | None, int]] = []
    seen = 0
    bank = ModelBank(built, threads=threads)
    for chunk in bank.fit_predict_batches(source, chunk_size=chunk_size):
        if slots is None:
            slots = _search_slots(built, chunk, scoring)
            totals = [[(0.0, 0, 0)] * len(slots) for _ in segments]
            known = [{} for _ in segments]
        for j, (a, b) in enumerate(segments):
            lo, hi = max(a, seen), min(b, seen + chunk.height)
            if lo < hi:
                part, here = _slot_sums(chunk.slice(lo - seen, hi - lo), built, slots, scoring)
                totals[j] = [
                    (s0 + s1, n0 + n1, o0 + o1)
                    for (s0, n0, o0), (s1, n1, o1) in zip(totals[j], part, strict=True)
                ]
                known[j] = {t: known[j].get(t, 0) + c for t, c in here.items()}
        seen += chunk.height
    if seen != n:
        msg = (
            f"{who}: the query gave {n} rows when counted and {seen} when run, so the spans "
            "placed by the count do not fit the rows scored. Give a query that returns the "
            "same rows each time it runs, or pass its result as a frame."
        )
        raise ValueError(msg)
    assert slots is not None  # n >= 2 rows, so at least one chunk

    def over(k: int, start: int, end: int) -> tuple[float, int, int]:
        parts = [totals[j][k] for j, (a, b) in enumerate(segments) if start <= a and b <= end]
        return math.fsum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts)

    lower = _SCORINGS[scoring]
    rows: list[dict[str, Any]] = []
    for target in dict.fromkeys(s["target"] for s in slots):
        mine = [k for k, s in enumerate(slots) if s["target"] == target]
        chosen_from = [k for k in mine if slots[k]["kind"] == "candidate"]
        # The first choice reads the fewest rows, so it alone can find none.
        first_end = folds[0][0]
        first = {k: over(k, 0, first_end) for k in chosen_from}
        if not first[chosen_from[0]][1]:
            late = min(chosen_from, key=lambda k: first[k][2])
            who_late = f'{slots[late]["field"]!r} of spec "{built[slots[late]["spec"]]["name"]}"'
            rows_read = (
                f"rows 0-{first_end - 1}, the {first_end}" if first_end > 1 else "row 0, the one"
            )
            ahead = (
                f"the gap of {gap} rows ahead of the first test span"
                if gap
                else "the first test span"
            )
            if target is None:
                lacks = "a loglik from every slot"
                fewest = f"{who_late} has one on {first[late][2]}, the fewest"
            else:
                present = sum(
                    known[j].get(target, 0) for j, (_, b) in enumerate(segments) if b <= first_end
                )
                lacks = f"a prediction of target {target!r} from every slot with the target present"
                fewest = (
                    f"{who_late} predicts on {first[late][2]}, the fewest, and the target is "
                    f"present on {present}"
                )
            msg = (
                f"{who}: at n_splits={n_splits}, the first choice of a grid point reads "
                f"{rows_read} before {ahead}. None of them has {lacks}: {fewest}. Give fewer "
                "n_splits, a smaller gap, a shorter warm-up "
                "(min_weight, warm_rows) or more rows."
            )
            raise ValueError(msg)
        picks = []
        honest_sum, honest_n = 0.0, 0
        tested_from = folds[0][1]
        for choose_end, start, end in folds:
            # The first choice reads the first span, the only rows it has. Each
            # later one reads the test spans' rows before it, as the rank reads
            # them, so no model's start-up rows weigh on it: read from row 0, a
            # model whose first errors were large was passed over on every
            # span it was best on (task 382a, C3(b)). With a gap as long as a
            # span, or no row there every slot predicts, it reads from row 0.
            low = tested_from if choose_end > tested_from else 0
            before = {k: over(k, low, choose_end) for k in chosen_from}
            if not before[chosen_from[0]][1]:
                before = {k: over(k, 0, choose_end) for k in chosen_from}
            means = {k: s / c for k, (s, c, _) in before.items()}
            pick = (min if lower else max)(chosen_from, key=lambda k: means[k])
            picks.append(pick)
            s, c, _ = over(pick, start, end)
            honest_sum, honest_n = honest_sum + s, honest_n + c
        tested = {k: over(k, folds[0][1], n) for k in mine}
        scores: dict[int, float | None] = {
            k: (s / c if c else None) for k, (s, c, _) in tested.items()
        }
        ranked = [v for k in chosen_from if (v := scores[k]) is not None]
        for k in mine:
            slot = slots[k]
            score = scores[k]
            candidate = slot["kind"] == "candidate"
            rank = None
            if candidate and score is not None:
                rank = 1 + sum(
                    1 for r in ranked if (r < score if lower else r > score)
                )  # ties share
            rows.append(
                {
                    "kind": slot["kind"],
                    "spec": built[slot["spec"]]["name"],
                    **points[slot["spec"]],
                    **{c: slot[c] for c in _GRID_COLUMNS},
                    "slot": slot["field"],
                    "target": target,
                    "n": tested[k][1],
                    scoring: score,
                    "rank": rank,
                    "folds_chosen": picks.count(k) if candidate else None,
                }
            )
        # The choice is no one spec's field: each span runs whichever candidate
        # the rows before it chose, so its spec, settings and slot are null.
        rows.append(
            {
                "kind": "choice",
                "target": target,
                "n": honest_n,
                scoring: honest_sum / honest_n if honest_n else None,
            }
        )
    keys = list(points[0]) if points else []
    varying = [
        c
        for c in _GRID_COLUMNS
        if c not in keys and len({s[c] for s in slots if s["kind"] == "candidate"}) > 1
    ]
    columns = [
        "kind",
        "spec",
        *keys,
        *varying,
        "slot",
        "target",
        "n",
        scoring,
        "rank",
        "folds_chosen",
    ]
    schema: dict[str, Any] = {
        "kind": pl.String,
        "spec": pl.String,
        "slot": pl.String,
        "target": pl.String,
        "n": pl.Int64,
        scoring: pl.Float64,
        "rank": pl.Int64,
        "folds_chosen": pl.Int64,
    }
    data = {c: [r.get(c) for r in rows] for c in columns}
    return pl.DataFrame(
        {c: pl.Series(c, v, dtype=schema.get(c), strict=False) for c, v in data.items()}
    )


def _resid_fields(who: str, df: pl.DataFrame, spec_name: str) -> list[str]:
    dtype = df.schema.get(spec_name)
    if dtype is None:
        msg = (
            f'{who}: spec "{spec_name}" has no output column in the frame, whose columns are '
            f"{df.columns}. Name a spec whose output column the frame holds."
        )
        raise OutputColumnNotFoundError(msg)
    if not isinstance(dtype, pl.Struct):
        msg = (
            f"{who}: column {spec_name!r} must be a spec's output column, but it is "
            f"{dtype}. Name a spec whose output column the frame holds."
        )
        raise TypeError(msg)
    return [f.name for f in dtype.fields if f.name.startswith("resid_")]


@overload
def seqtest(
    df: pl.DataFrame,
    *,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    group: str | Iterable[str] | None = None,
    min_weight: float = 0.0,
    name: str = "seqtest",
) -> pl.DataFrame: ...


@overload
def seqtest(
    df: pl.LazyFrame,
    *,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    group: str | Iterable[str] | None = None,
    min_weight: float = 0.0,
    name: str = "seqtest",
) -> pl.LazyFrame: ...


@renamed_keywords("po.eval.seqtest", _GROUP)
def seqtest(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    targets: Sequence[str] | None = None,
    a: str | None = None,
    b: str | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    group: str | Iterable[str] | None = None,
    min_weight: float = 0.0,
    name: str = "seqtest",
) -> pl.DataFrame | pl.LazyFrame:
    """Test a sign sequentially by betting, over a frame or a query.

    Use it to gather evidence, row by row, that a column's sign leans one
    way, or that one spec predicts closer than another on more rows, with
    evidence you may read at every row (see Notes for the null that
    holds). It computes what :func:`polars_online.spec.seqtest` writes,
    field for field and row for row, with Polars expressions. For a
    stream too long to hold, give it a query of the output, or run the
    ``seqtest`` model in the bank, whose memory is the size of its state.

    .. rubric:: Parameters

    ``df``
        The frame: in column mode, one holding the columns tested; in
        compare mode, one a bank returned, holding both specs' output
        columns. Or a query of either, a ``LazyFrame``, which gives a query
        back (see Returns).
    ``targets``
        In column mode, the columns whose sign is tested, which column mode
        needs. In compare mode, the targets whose residuals are compared:
        ``t`` means ``resid_<t><a_suffix>`` of ``a`` against
        ``resid_<t><b_suffix>`` of ``b``. ``None``, the default, takes every
        target the two share.
    ``a``, ``b``
        The names of two specs' output columns, for compare mode; give both
        or neither. The sign tested is that of ``|resid_b| - |resid_a|``,
        positive when ``a`` was closer.
    ``a_suffix``, ``b_suffix``
        The suffix of the residual field each side's slot carries, such as
        ``"__r0.1"`` for one grid point of a grid; ``""`` by default.
    ``group``
        One key column, as a spec's ``group`` is, or a list of them, to run
        one test per group in row order (``.over(group)``); ``None``, the
        default, runs one.
    ``min_weight``
        The rows a group must have before a field other than ``weight_sum``
        is filled; 0 by default.
    ``name``
        The name of the column added, ``"seqtest"`` by default.

    .. rubric:: Returns

    ``df`` with a record column ``name`` added, holding per target ``t``,
    each read before the row, as the bank writes them:

    ``log_e_pos_<t>``, ``log_e_neg_<t>``, ``n_pos_<t>``, ``n_neg_<t>``
        In column mode: the log wealth of the bet that the sign is positive
        and of the bet that it is negative, and the signs counted.
    ``log_e_a_<t>``, ``log_e_b_<t>``, ``wins_a_<t>``, ``wins_b_<t>``
        In compare mode: the same, for "``a`` was closer" and "``b`` was
        closer".
    ``weight_sum``
        The rows before this one in its group. Every other field is null
        until it reaches ``min_weight``.

    Given a ``LazyFrame``, a ``LazyFrame`` of the same, which reads ``df``
    one chunk at a time when it runs (see Notes).

    .. rubric:: Raises

    ``ValueError``
        When:

        - ``df`` already has a column ``name``, which the result would
          replace;
        - ``a`` is given without ``b``, or ``b`` without ``a``;
        - column mode has no ``targets``;
        - the two specs share no residual field;
        - a target has no residual on one side (the error names the fields
          that side has).
    :class:`~polars_online.OutputColumnNotFoundError`
        ``df`` has no column ``a`` or ``b``. It is both Polars'
        ``ColumnNotFoundError`` and a ``KeyError``.
    ``TypeError``
        ``df`` is neither a ``DataFrame`` nor a ``LazyFrame``, or ``a`` or
        ``b`` names a column that is not a record.
    :class:`~polars_online.ColumnNotFoundError`
        ``df`` does not have a ``group`` column. It is both Polars'
        ``ColumnNotFoundError`` and a ``ValueError``.
    Polars' ``ColumnNotFoundError``
        ``df`` does not have a column-mode target.

    .. rubric:: Update equation

    Per target, with ``s`` the row's sign (+1, -1 or 0), and ``n_pos`` and
    ``n_neg`` the signs counted before the row:

    .. code-block:: text

        lam_pos = max(0, (n_pos - n_neg) / (n_pos + n_neg + 1))
        lam_neg = max(0, (n_neg - n_pos) / (n_pos + n_neg + 1))
        log_e_pos += log1p(lam_pos * s)
        log_e_neg += log1p(-lam_neg * s)

    ``lam_pos`` and ``lam_neg`` are the shares of wealth each bet stakes on
    the row, read from the signs before it.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        ridge = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        x0_only = po.spec.ewridge("x0_only", targets=["y"], features=["x0"], half_life=100.0)
        out = po.ModelBank([ridge, x0_only]).fit_predict(df)
        evidence = po.eval.seqtest(out, a="ridge", b="x0_only")
        last = evidence["seqtest"].struct.unnest().row(-1, named=True)    # the last row's fields
        print(f"ridge closer on {last['wins_a_y']:.0f} rows, x0_only on {last['wins_b_y']:.0f}")
        log_e = last["log_e_a_y"]                                        # ridge's log evidence
        print(f"log evidence for ridge {log_e:.1f}, past ln 20: {log_e > np.log(20)}")

    .. code-block:: text

        ridge closer on 290 rows, x0_only on 105
        log evidence for ridge 41.8, past ln 20: True

    .. rubric:: See also

    - :func:`polars_online.spec.seqtest`, the same test run by the bank.
    - :func:`diebold_mariano` and :func:`clark_west`, tests of two specs'
      squared errors over a whole frame.
    - :func:`compare_specs`, whose Notes compare the three tests.

    .. rubric:: Notes

    The evidence can be read at every row. Each target has two bets, one on
    each direction: reject a direction at the 5% level the first time its
    bet's log wealth reaches ``ln 20``, about 3.0. Watching both bets at
    once, to test either direction at 5%, needs ``ln 40``. A value of zero bets
    nothing and counts nothing, and so does one the bank would not learn
    from: null, NaN, infinite, or past its input bound of 1e100 in
    magnitude. In compare mode the value is the difference ``|resid_b| -
    |resid_a|``, and the bound is on it, where :func:`diebold_mariano`
    bounds each residual.

    **The 5% holds under a null about every row:** at each row, whatever
    came before, the next sign is no more likely positive than negative.
    Then each bet's wealth is a nonnegative supermartingale started at 1,
    and by Ville's inequality it reaches 20 with probability at most 5%,
    read at every row, not only the last. That null is more than "the two
    specs are equally close": signs that run break it with neither spec
    closer on average, and the bet that follows the run grows, in either
    direction. On simulated streams of 5,000 rows from two equally accurate
    specs, each bet reached ``ln 20`` on 3.6% of streams where their errors
    were independent, and on 61% (``a`` closer) and 62% (``b`` closer)
    where each spec's errors followed an autoregression at 0.9, on 88% one
    or the other at ``ln 40``. Where errors persist, test one row in each
    block of rows longer than their memory: filter the frame to those rows
    first. The bank's model adds what a frame in memory does not have,
    the clock (``session``, ``restart_after_step_back``), and its output is
    held to this function's to the last bit.

    **Given a query, the result is the frame's to the bit, and this holds
    one chunk and a few numbers per key, beside what Polars reads of the
    query ahead of it.** Each chunk carries in, per group,
    the rows and the signs counted before it and each bet's log wealth
    after its last row, and adds that wealth to its own first term, so the
    sums run on in the frame's order. A query is refused as its frame
    would be, with the same message, before it runs. A scan of a parquet
    file reads its row groups ahead of a slower consumer, as many as
    Polars' thread count sizes, and under Polars' defaults that read-ahead
    is most of what the process holds: `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds it. Written to a file with ``sink_parquet``, the test of two
    specs of a scan by a key of four values, the file in row groups of
    100,000 rows, held at most 149 MiB at a million rows, 159 MiB at four
    million and 157 MiB at sixteen million with four threads
    (``POLARS_MAX_THREADS=4``), the read-ahead cut to one row group
    (``POLARS_ROW_GROUP_PREFETCH_SIZE=1``) and the allocator told to return
    freed pages at once (``_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0``);
    under Polars' defaults, 349, 646 and 1,155 MiB. Each figure is the
    most memory the process held, about 48 MiB of it Python and Polars
    before the call, measured on macOS (an M4 Pro, py-polars 2.0.0).
    """
    who = "po.eval.seqtest"
    _frame(who, df, lazy=True)
    probe = _probe(df)
    _check_name("po.eval.seqtest", probe, name)
    _check_columns(probe, who, group=group)
    keys = _group_keys(group)
    if (a is None) != (b is None):
        msg = (
            "po.eval.seqtest: a and b go together, but only one is given. Name both specs to "
            "compare them, or neither."
        )
        raise ValueError(msg)
    if a is not None and b is not None:
        have = {a: _resid_fields(who, probe, a), b: _resid_fields(who, probe, b)}
        if targets is None:
            # `resid_<t><a_suffix>` of a, for every t with `resid_<t><b_suffix>` in b.
            bodies = [f.removeprefix("resid_") for f in have[a] if f.endswith(a_suffix)]
            stems = [x[: len(x) - len(a_suffix)] for x in bodies]
            targets = [t for t in stems if f"resid_{t}{b_suffix}" in have[b]]
            if not targets:
                msg = (
                    f"{who}: {a!r} and {b!r} share no residual field with a_suffix "
                    f"{a_suffix!r} and b_suffix {b_suffix!r}: {a!r} has {have[a]}, and "
                    f"{b!r} has {have[b]}. Name the targets, or set the suffixes to match."
                )
                raise ValueError(msg)
        for t in targets:
            for side, suffix in ((a, a_suffix), (b, b_suffix)):
                want = f"resid_{t}{suffix}"
                if want not in have[side]:
                    msg = (
                        f"{who}: target {t!r} names no residual of {side!r}, which has no field "
                        f"{want!r} (its residual fields are {have[side]}). Set a_suffix and "
                        "b_suffix to the suffixes these fields carry, or name a target both specs "
                        "predict."
                    )
                    raise ValueError(msg)
        signs = {
            t: pl.col(b).struct.field(f"resid_{t}{b_suffix}").abs()
            - pl.col(a).struct.field(f"resid_{t}{a_suffix}").abs()
            for t in targets
        }
        names = ("log_e_a", "log_e_b", "wins_a", "wins_b")
    else:
        if targets is None:
            msg = (
                "po.eval.seqtest: targets is required without a and b, but it is not given. "
                "Name the columns whose sign is tested in targets, or name two specs as a and b."
            )
            raise ValueError(msg)
        signs = {t: pl.col(t) for t in targets}
        names = ("log_e_pos", "log_e_neg", "n_pos", "n_neg")
    test = _SeqTest(signs, names, keys, min_weight, name)
    if isinstance(df, pl.LazyFrame):
        # A column-mode target the query lacks is Polars' own refusal, here
        # as on a frame, before the query runs.
        schema = test.run(probe)[0].schema
        needed = set(keys).union(*(d.meta.root_names() for d in signs.values()))

        def step() -> Callable[[pl.DataFrame], pl.DataFrame]:
            carry: pl.DataFrame | None = None

            def one(chunk: pl.DataFrame) -> pl.DataFrame:
                nonlocal carry
                out, carry = test.run(chunk, carry, carrying=True)
                return out

            return one

        return _row_query(df, needed, schema, step)
    return test.run(df)[0]


class _SeqTest:
    """:func:`seqtest`'s arithmetic over a frame, or over one chunk of a
    query after the chunks before it.

    A chunk carries in, per key, the rows before it, the signs counted, and
    each bet's log wealth after the last of them; the running sums go on
    from those. The log wealth is a sum in row order, so the carried sum
    is added to the chunk's first term and the chunk's sums run on from it,
    the same additions in the same order as over the frame, and the result
    is the frame's to the bit."""

    def __init__(
        self,
        signs: dict[str, pl.Expr],
        names: tuple[str, str, str, str],
        keys: list[str],
        min_weight: float,
        name: str,
    ) -> None:
        self.signs, self.names, self.keys = signs, names, keys
        self.min_weight, self.name = min_weight, name

    def run(
        self, df: pl.DataFrame, carry: pl.DataFrame | None = None, *, carrying: bool = False
    ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """``df`` with the record column added, and, when ``carrying``, what
        the next chunk carries in."""
        keys = self.keys

        def over(e: pl.Expr) -> pl.Expr:
            return e.over(keys) if keys else e

        def before(e: pl.Expr) -> pl.Expr:
            """The running sum of ``e`` over the rows before this one."""
            return over(e.cum_sum().shift(1, fill_value=0))

        # Two passes, so that no window sits inside another: the log wealth is a
        # running sum over the rows before, of a bet sized by counts that are
        # themselves running sums. Polars 1.34.0 refuses a window inside a
        # window ("window expression not allowed in aggregation"; 1.44.2 takes
        # it), which failed every `group` at the declared floor (task 109's floor run,
        # 2026-09-27). The counts go into columns first, then the wealth reads
        # them; the arithmetic is the same.
        taken = {*df.columns, self.name}

        def temp(label: str) -> str:
            column = f"__po_seqtest_{label}"
            while column in taken:
                column = "_" + column
            taken.add(column)
            return column

        n_eff_col = temp("weight_sum")
        count = over(pl.int_range(pl.len()))
        # The carried-in columns: the rows before the chunk, and per target
        # the signs counted and each bet's log wealth (null for a key the
        # chunks before did not have).
        rows_in = temp("rows_in")
        held = [tuple(temp(f"{k}_in_{i}") for k in ("pos", "neg", "lpos", "lneg"))
                for i in range(len(self.signs))]  # fmt: skip
        if carrying:
            counts = [rows_in, *(c for h in held for c in h[:2])]
            df = _carried_in(df, carry, keys, counts, [c for h in held for c in h[2:]])
            count = count + pl.col(rows_in)
        first: list[pl.Expr] = [count.cast(pl.Float64).alias(n_eff_col)]
        staged: dict[str, tuple[str, str, str]] = {}
        for i, (t, d) in enumerate(self.signs.items()):
            # What the bank does not learn from -- null, NaN, an infinity, a
            # magnitude beyond its input bound (docs/PLAN.md section 3) -- is no
            # sign here either. Polars orders NaN above every float, so without
            # this `NaN > 0` would be a positive sign.
            d = d.cast(pl.Float64)
            d = pl.when(d.is_finite() & (d.abs() <= _INPUT_BOUND)).then(d)
            cols = (temp(f"s_{i}"), temp(f"n_pos_{i}"), temp(f"n_neg_{i}"))
            pos = before((d > 0).cast(pl.Int64).fill_null(0))
            neg = before((d < 0).cast(pl.Int64).fill_null(0))
            if carrying:
                pos, neg = pos + pl.col(held[i][0]), neg + pl.col(held[i][1])
            first += [
                pl.when(d > 0).then(1.0).when(d < 0).then(-1.0).otherwise(0.0).alias(cols[0]),
                pos.alias(cols[1]),
                neg.alias(cols[2]),
            ]
            staged[t] = cols
        ready = pl.col(n_eff_col) >= self.min_weight
        fields: list[pl.Expr] = []
        sums: list[pl.Expr] = []
        totals: list[tuple[str, str]] = []
        for i, (t, (s_col, pos_col, neg_col)) in enumerate(staged.items()):
            s, n_pos, n_neg = pl.col(s_col), pl.col(pos_col), pl.col(neg_col)
            n1 = (n_pos + n_neg + 1).cast(pl.Float64)
            lam_pos = pl.max_horizontal((n_pos - n_neg).cast(pl.Float64) / n1, 0.0)
            lam_neg = pl.max_horizontal((n_neg - n_pos).cast(pl.Float64) / n1, 0.0)
            bets = ((lam_pos * s).log1p(), (-lam_neg * s).log1p())
            if carrying:
                # The carried log wealth added to the key's first term in the
                # chunk, so each running sum goes on from it.
                start = over(pl.int_range(pl.len())) == 0
                wealth = []
                for bet, kept, label in zip(bets, held[i][2:], ("lpos", "lneg"), strict=True):
                    total = temp(f"{label}_{i}")
                    lead = pl.when(start & pl.col(kept).is_not_null())
                    sums.append(
                        over(lead.then(bet + pl.col(kept)).otherwise(bet).cum_sum()).alias(total)
                    )
                    wealth.append(
                        over(pl.col(total).shift(1)).fill_null(pl.col(kept).fill_null(0.0))
                    )
                    totals.append((total, label))
                log_e_pos, log_e_neg = wealth
            else:
                log_e_pos, log_e_neg = before(bets[0]), before(bets[1])
            for label, e in zip(self.names, (log_e_pos, log_e_neg, n_pos, n_neg), strict=True):
                fields.append(pl.when(ready).then(e).alias(f"{label}_{t}"))
        fields.append(pl.col(n_eff_col).alias("weight_sum"))
        temps = [n_eff_col, *(c for cols in staged.values() for c in cols)]
        staged_df = df.with_columns(first)
        if sums:
            staged_df = staged_df.with_columns(sums)
        out = staged_df.with_columns(pl.struct(fields).alias(self.name))
        if not carrying:
            return out.drop(temps), None
        last = [
            (pl.col(n_eff_col).last().cast(pl.Int64) + 1).alias(rows_in),
            *(
                e
                for i, (s_col, pos_col, neg_col) in enumerate(staged.values())
                for e in (
                    (pl.col(pos_col).last() + (pl.col(s_col).last() > 0).cast(pl.Int64)).alias(
                        held[i][0]
                    ),
                    (pl.col(neg_col).last() + (pl.col(s_col).last() < 0).cast(pl.Int64)).alias(
                        held[i][1]
                    ),
                    pl.col(totals[2 * i][0]).last().alias(held[i][2]),
                    pl.col(totals[2 * i + 1][0]).last().alias(held[i][3]),
                )
            ),
        ]
        kept_cols = [rows_in, *(c for h in held for c in h)]
        if out.is_empty():
            new = carry
        else:
            now = out.group_by(keys, maintain_order=True).agg(last) if keys else out.select(last)
            new = _carry_on(carry, now, keys)
        drop = [*temps, *kept_cols, *(t for t, _ in totals)]
        return out.drop(drop), new


def _carried_in(
    df: pl.DataFrame,
    carry: pl.DataFrame | None,
    keys: list[str],
    counts: list[str],
    sums: list[str],
) -> pl.DataFrame:
    """``df`` with the carried-in columns beside each row, by its key: the
    ``counts`` 0 and the ``sums`` null for a key the chunks before did not
    have."""
    if carry is None:
        return df.with_columns(
            *(pl.lit(0, dtype=pl.Int64).alias(c) for c in counts),
            *(pl.lit(None, dtype=pl.Float64).alias(c) for c in sums),
        )
    if keys:
        joined = df.join(carry, on=keys, how="left", nulls_equal=True, maintain_order="left")
    else:
        joined = df.with_columns(
            pl.lit(carry[c][0], dtype=carry.schema[c]).alias(c) for c in (*counts, *sums)
        )
    return joined.with_columns(pl.col(c).fill_null(0) for c in counts)


def _carry_on(old: pl.DataFrame | None, now: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """The carry after a chunk: each key's from ``now``, and the keys the
    chunk did not have from ``old``."""
    if old is None or not keys:
        return now
    return pl.concat([old.join(now, on=keys, how="anti", nulls_equal=True), now])


def _row_query(
    lf: pl.LazyFrame,
    needed: set[str],
    schema: pl.Schema,
    step: Callable[[], Callable[[pl.DataFrame], pl.DataFrame]],
) -> pl.LazyFrame:
    """A query of ``lf``'s rows, each chunk passed through a fresh ``step()``
    in order: what a call that adds a column computed row by row returns for
    a query, read in one pass a chunk at a time when it runs.

    A Polars IO source: Polars hands it the columns, the filter and the row
    count the query asks of it, and does not apply them again, so it applies
    all three, the filter and the row count after the column is computed,
    since each row's value reads the rows before it. Each chunk's result is
    handed on in slices of at most a mebibyte, as the package's other
    sources hand theirs."""
    from polars.io.plugins import register_io_source

    from polars_online._frame import _in_morsels

    columns = list(lf.collect_schema())

    def source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        plan = lf
        if with_columns is not None:
            wanted = set(with_columns) | needed
            plan = plan.select([c for c in columns if c in wanted])
        one = step()
        seen = 0
        for chunk in plan.collect_batches(chunk_size=_chunk_rows(), maintain_order=True):
            if n_rows is not None:
                chunk = chunk.head(n_rows - seen)
            out = one(chunk)
            seen += chunk.height
            if predicate is not None:
                out = out.filter(predicate)
            if with_columns is not None:
                out = out.select(with_columns)
            yield from _in_morsels(out)
            if n_rows is not None and seen >= n_rows:
                break

    return register_io_source(source, schema=schema, validate_schema=True, is_pure=True)


@renamed_keywords("po.eval.sums", _GROUP)
@_takes_spec("po.eval.sums", lazy=True)
def sums(
    df: pl.DataFrame | pl.LazyFrame,
    spec: str | dict,
    *,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    weight: str | None = None,
    binary: bool | None = None,
) -> pl.DataFrame:
    """Reduce a bank's output, or one chunk of it, to the sums its scores are read from.

    Use it to keep scores that can be added up: reduce each part of a
    stream to ten numbers per slot, target and group, add the parts with
    :func:`merge_sums`, and read :func:`metrics`' scores with
    :func:`from_sums`. Given a query, it reduces all of the query's rows,
    one chunk at a time. For the scores themselves, :func:`metrics` gives
    them in one call, over a frame or a query.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, or one chunk of it, holding the spec's
        output column. Or a query of one, a ``LazyFrame``, which this runs
        and reads one chunk at a time, as :func:`metrics` reads one.
    ``spec``
        The spec's name, or the spec dict itself, as :func:`unpack` reads
        it.
    ``group``, ``targets``
        :func:`metrics`' own.
    ``weight``
        A column to weight the rows by; ``None``, the default, counts every
        row 1, and then ``w`` equals ``n``. A row whose weight is null,
        NaN, infinite, past the input bound of 1e100, or negative is
        dropped: the bank skips the first four and refuses a negative one.
        A row of weight 0 is kept, counted in ``n`` and not in ``w``, so
        :func:`from_sums`' ``min_samples`` counts it: 30 rows of which 28
        weigh 0 pass ``min_samples=30`` and are scored on two. Filter out
        the rows that are not to count before this call, rather than
        weighting them 0, where ``min_samples`` is to count the rows
        scored.
    ``binary``
        :func:`metrics`' own: whether ``hits`` and ``signed`` count a
        classifier's accuracy at 0.5 or the sign test. Keep the sums of the
        two readings apart (see Returns). A target other than 0 or 1 on a
        scored row is refused under it.

    .. rubric:: Returns

    A ``DataFrame`` with one row per ``slot``, ``target`` and ``group`` key,
    given a frame or a query alike, and these columns after the keys,
    :data:`SUM_FIELDS`. A slot or key with no scored row has no row.

    ``n``, ``w``
        The rows, and the weight behind them.
    ``mean_y``, ``mean_pred``
        The weighted means of the target and the prediction.
    ``m2_y``, ``m2_pred``, ``cov``
        The weighted second moments about those means, and the weighted
        cross-moment about them.
    ``sse``
        The weighted sum of squared residuals.
    ``hits``, ``signed``
        For the hit rate: ``hits`` weighs the rows whose sign the
        prediction got right, and ``signed`` the rows with ``y != 0`` and
        ``pred != 0``, as :func:`metrics` counts them. Under ``binary``,
        ``hits`` weighs the rows on the right side of 0.5, and ``signed``
        every row. With the spec dict naming a Poisson ``sgd`` fit, which
        has no sign to hit, both are 0, and :func:`from_sums` gives the
        null the bank gives. :func:`merge_sums` adds ``hits`` and
        ``signed`` without knowing which reading made them, so add only
        sums made under one ``binary``.

    .. rubric:: Raises

    Whatever :func:`unpack` raises for ``df`` and ``spec``, and
    :class:`~polars_online.ColumnNotFoundError`, both Polars'
    ``ColumnNotFoundError`` and a ``ValueError``, for a ``group`` or
    ``weight`` column ``df`` does not have. What :func:`metrics` raises for
    ``group`` and ``binary``, and ``ValueError`` for a ``weight`` column
    named like a column of :func:`unpack`'s long form (:data:`RESERVED`).
    A query is refused as its result would be, with the same message,
    before it runs.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        part = po.eval.sums(out.head(200), spec)         # ten numbers per slot and target
        rest = po.eval.sums(out.tail(200), spec)
        running = po.eval.merge_sums(part, rest)         # the two halves, added
        scores = po.eval.from_sums(running)
        whole = po.eval.metrics(out, spec)
        print(f"r2 from sums {scores['r2'][0]:.6f}, from the frame {whole['r2'][0]:.6f}")

    .. code-block:: text

        r2 from sums 0.944014, from the frame 0.944014

    .. rubric:: See also

    - :func:`merge_sums`, which adds the sums of several chunks.
    - :func:`from_sums`, which reads the scores from them.
    - :func:`metrics`, the same scores in one call, over a frame or a query.

    .. rubric:: Notes

    A run that compares fifty slots over a billion rows keeps ten numbers
    per key this way, instead of writing the rows out to score them later.
    Rows where the prediction or the target is null are dropped, as
    :func:`metrics` drops them: null, NaN, infinite or past the bank's input
    bound.

    **The moments are kept about the mean, not as raw sums.** The simpler
    form keeps ``sum(y)`` and ``sum(y**2)`` and subtracts, and loses the
    variance entirely when the mean is large against the spread. A target
    of variance 1 about 1e8 has ``var / E[y**2]`` of about 1e-16, and the
    subtraction has nothing left. Keeping the moments about the mean adds
    one term to :func:`merge_sums`, a multiply, and keeps the variance to
    rounding, as its Notes measure.

    Given a query, the sums are those of the query's chunks of 100,000
    rows, each reduced as a frame is and added as :func:`merge_sums` adds
    them, so they hold what :func:`metrics`' Notes measure, and Polars what
    it reads ahead of them, which `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds: 204, 208 and 212 MiB at one, four and sixteen million rows under
    its bounded settings, 493, 990 and 1,072 MiB under Polars'
    defaults. While the query gives one chunk, they are the frame's to the
    bit; over many, within the merge's rounding, which :func:`metrics`'
    Notes measure.
    """
    spec_name, written = _spec_of("po.eval.sums", spec)
    _check_columns(_probe(df), "po.eval.sums", group=group, weight=weight, produced=_PRODUCED)
    binary = _binary(binary, written)
    if isinstance(df, pl.LazyFrame):
        jobs = [_job(df, written or spec_name, targets, group, weight, binary, "po.eval.sums")]
        merged = _query_sums(df, jobs, keys=_keys(group), weight=weight)[0]
        if merged is not None:
            return merged.select(*_keys(group), *SUM_FIELDS)
        df = _probe(df)
    job = _job(df, written or spec_name, targets, group, weight, binary, "po.eval.sums")
    return _frame_sums(df, job, _keys(group), weight).select(*_keys(group), *SUM_FIELDS)


def _keys(group: str | Iterable[str] | None) -> list[str]:
    """The keys a score is read per: the slot, its target and ``group``'s."""
    return ["slot", "target", *_group_keys(group)]


def _weighed(long: pl.DataFrame, weight: str | None) -> pl.DataFrame:
    """``long`` with each row's weight as ``__w``: ``weight``'s column, the
    rows whose weight the bank cannot use left out, or 1.0 without one."""
    if weight is not None:
        # What the bank does with the same weight: it skips a row whose
        # weight it cannot use and refuses a negative one (review round 4,
        # YB21: a null weight counted in `n` and not in `w`).
        wexpr = pl.col(weight).cast(pl.Float64)
        long = long.filter(_usable(wexpr) & (wexpr >= 0.0))
    else:
        wexpr = pl.lit(1.0)
    return long.with_columns(wexpr.alias("__w"))


def _sum_aggs(binary: bool, no_hit: bool, *, weighted: bool = True) -> list[pl.Expr]:
    """:data:`SUM_FIELDS` as aggregations of a long form's ``y``, ``pred``
    and weight ``__w``. Not ``weighted``, every row weighs 1 and ``__w`` is
    not read: the same numbers to the bit, since a product by 1.0 is exact,
    without the products and the filter of rows that weigh, which took a
    frame's ``metrics`` of 20 slots over a million rows from 560 ms to 860
    (task 382a, A5)."""
    y, p = pl.col("y"), pl.col("pred")
    w = pl.col("__w") if weighted else None

    def times(e: pl.Expr) -> pl.Expr:
        return e if w is None else w * e

    tw = pl.len().cast(pl.Float64) if w is None else w.sum()

    def mean(v: pl.Expr) -> pl.Expr:
        # A value the rows that weigh all share is its own mean, exactly: the
        # computed `sum(w * c) / sum(w)` of a constant `c` is `c` only where
        # the sum happens to be exact, and otherwise leaves a spread of
        # rounding, about `n * (eps * c)**2`, that read as a variance: `r2`
        # about -1e32 and `ic` of a constant prediction 0.14 (task 382a, A2).
        held = v if w is None else v.filter(w > 0)
        return pl.when(held.min() == held.max()).then(held.first()).otherwise(times(v).sum() / tw)

    my, mp = mean(y), mean(p)
    if binary:
        hits, signed = times(((p > 0.5) == (y > 0.5)).cast(pl.Float64)), times(pl.lit(1.0))
    else:
        # A target or a prediction of zero has no sign, as in :func:`metrics`
        # and the bank (docs/PLAN.md task 195, S6); a fit with no sign to
        # hit signs no row at all, so :func:`from_sums` gives null (E1).
        scored = pl.lit(False) if no_hit else (y != 0) & (p != 0)
        hits = times(((y.sign() == p.sign()) & scored).cast(pl.Float64))
        signed = times(scored.cast(pl.Float64))
    return [
        pl.len().cast(pl.Int64).alias("n"),
        tw.alias("w"),
        my.alias("mean_y"),
        mp.alias("mean_pred"),
        times((y - my) ** 2).sum().alias("m2_y"),
        times((p - mp) ** 2).sum().alias("m2_pred"),
        times((y - my) * (p - mp)).sum().alias("cov"),
        times((y - p) ** 2).sum().alias("sse"),
        hits.sum().alias("hits"),
        (pl.len().cast(pl.Float64) if binary and w is None else signed.sum()).alias("signed"),
    ]


#: The column a query's sums carry beside :data:`SUM_FIELDS` for
#: :func:`metrics`' ``log_loss``: the sum of each row's log loss.
_LOG_LOSS = "__po_log_loss"

#: How many chunks' sums a query's reading keeps before it merges them into
#: one, so what it holds is bounded by the keys, not by the query's length.
_MERGE_EVERY = 64


def _chunk_rows() -> int:
    """The rows of a query read at once: a bank's default ``chunk_size``."""
    return int(_native.default_chunk_size())


def _log_loss_sum() -> pl.Expr:
    """The sum of :func:`metrics`' log loss over a long form's rows,
    ``-(y*ln(p) + (1-y)*ln(1-p))``, read over the rows rather than kept in a
    spec's streaming ``emit_metrics``: an exponentially weighted sum would
    put a ``ln`` result into model state, and glibc's last bit is not
    Apple's (that difference broke a frozen state fixture on Linux alone).
    ``p`` is clipped to ``[1e-15, 1 - 1e-15]`` so a confident miss, whose
    term is ``-ln(1e-15)``, is a large number rather than ``+inf``."""
    p = pl.col("pred").clip(1e-15, 1.0 - 1e-15)
    y = pl.col("y")
    return (-(y * p.log() + (1.0 - y) * (1.0 - p).log())).sum().alias(_LOG_LOSS)


class _Job:
    """One spec's sums to read from a query: its output column, its slots
    (each field, its target and the target's value), what to count as a hit,
    and whether to keep the log loss."""

    def __init__(
        self,
        spec_name: str,
        slots: list[tuple[str, str, pl.Expr]],
        keep: list[str],
        binary: bool,
        no_hit: bool,
        log_loss: bool,
        who: str,
    ) -> None:
        self.spec_name, self.slots, self.keep = spec_name, slots, keep
        self.binary, self.no_hit, self.log_loss = binary, no_hit, log_loss
        self.who = who

    def extra(self) -> list[str]:
        """The columns its sums carry beside :data:`SUM_FIELDS`."""
        return [_LOG_LOSS] if self.log_loss else []

    def columns(self) -> set[str]:
        """The query's columns this job reads."""
        roots = {r for _, _, y in self.slots for r in y.meta.root_names()}
        return {self.spec_name, *self.keep, *roots}


def _job(
    lf: pl.DataFrame | pl.LazyFrame,
    spec: str | dict,
    targets: Sequence[str] | None,
    group: str | Iterable[str] | None,
    weight: str | None,
    binary: bool,
    who: str,
    *,
    log_loss: bool = False,
    also: Sequence[str] = (),
) -> _Job:
    """``spec``'s :class:`_Job` over ``lf``, checked against the query's
    schema as :func:`_unpack` checks a frame, so it raises what a frame
    raises, before the query runs. ``also`` names more columns to carry,
    such as a clock."""
    probe = _probe(lf)
    spec_name, slots = _slots(probe, spec, targets, who)
    keep = _kept(probe, [*_group_keys(group), *([] if weight is None else [weight]), *also])
    _, written = _spec_of(who, spec)
    return _Job(spec_name, slots, keep, binary, _no_hit(written), log_loss and binary, who)


def _query_sums(
    lf: pl.LazyFrame,
    jobs: list[_Job],
    *,
    keys: list[str],
    weight: str | None,
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> list[pl.DataFrame | None]:
    """Each job's sums over every row of ``lf`` by ``keys``, read in one
    pass a chunk at a time: each chunk reduced to :data:`SUM_FIELDS` per
    key, and the chunks' sums merged as :func:`merge_sums` merges them,
    every :data:`_MERGE_EVERY` chunks and at the end. ``prepare`` takes each
    slot's scored rows in a chunk and adds a key, such as a window's start.
    ``None`` for a job no row of the query was scored for. What it holds is
    one chunk and the sums, not the query's rows."""
    needed = set().union(*(job.columns() for job in jobs))
    plan = lf.select([c for c in lf.collect_schema() if c in needed])
    parts: dict[int, list[pl.DataFrame]] = {id(job): [] for job in jobs}
    for chunk in plan.collect_batches(chunk_size=_chunk_rows(), maintain_order=True):
        for job in jobs:
            got = _chunk_sums(chunk, job, keys, weight, prepare)
            if got is None:
                continue
            held = parts[id(job)]
            held.append(got)
            if len(held) >= _MERGE_EVERY:
                held[:] = [_merge(held, keys, job.extra())]
    return [_merge(parts[id(job)], keys, job.extra()) if parts[id(job)] else None for job in jobs]


def _chunk_sums(
    chunk: pl.DataFrame,
    job: _Job,
    keys: list[str],
    weight: str | None,
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None,
    *,
    empty: bool = False,
) -> pl.DataFrame | None:
    """``job``'s sums of one chunk by ``keys``, a slot at a time: each slot's
    scored rows reduced to :data:`SUM_FIELDS` per key. ``None`` where no row
    of the chunk is scored, or, with ``empty``, a frame of no rows with the
    sums' columns. A frame is scored as one chunk, so its scores are a
    one-chunk query's to the bit; a slot at a time, it holds one slot's long
    form, where the long form of every slot held 2.9 GiB of a 20-slot
    million-row frame's ``metrics`` (task 382a, A5)."""
    aggs = _sum_aggs(job.binary, job.no_hit, weighted=weight is not None)
    if job.log_loss:
        aggs.append(_log_loss_sum())
    got = []
    long = None
    for slot, target, y in job.slots:
        long = chunk.select(
            *job.keep,
            pl.lit(slot).alias("slot"),
            pl.lit(target).alias("target"),
            pl.col(job.spec_name).struct.field(slot).alias("pred"),
            y.alias("y"),
        )
        long = _scored(long)
        if job.binary:
            _check_labels(long, job.who)
        if prepare is not None:
            long = prepare(long)
        long = _weighed(long, weight)
        if not long.is_empty():
            # A `group_by` even without a `group`, though a plain reduction
            # over the column would be about nine times as fast there: Polars
            # sums a whole column and a group's rows in different orders, and
            # a key's scores are the call on its rows alone, to the bit.
            got.append(long.group_by(keys).agg(aggs))
    if got:
        return pl.concat(got)
    if empty and long is not None:
        return long.group_by(keys).agg(aggs)
    return None


def _frame_sums(
    df: pl.DataFrame,
    job: _Job,
    keys: list[str],
    weight: str | None,
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """``job``'s sums of a frame by ``keys``, read as a query of one chunk
    reads them."""
    got = _chunk_sums(df, job, keys, weight, prepare, empty=True)
    assert got is not None  # a spec has a slot, so `empty` gives a frame
    return _merge([got], keys, job.extra())


def merge_sums(first: pl.DataFrame, *rest: pl.DataFrame) -> pl.DataFrame:
    """Add the sums of disjoint sets of rows, as if their rows were one chunk.

    Use it to add the :func:`sums` of the chunks of a stream, in any number
    and any split, before reading the scores with :func:`from_sums`.

    .. rubric:: Parameters

    ``first``, ``*rest``
        Frames :func:`sums` returned, over disjoint sets of rows, each with
        the same key columns. Merging one frame returns its sums again,
        exactly, sorted by the keys.

    .. rubric:: Returns

    A ``DataFrame`` with the same columns: the keys, then
    :data:`SUM_FIELDS`, one row per key, sorted by the keys. A key present
    in one frame and not another is carried through as it is.

    .. rubric:: Raises

    ``TypeError``
        A frame is not a ``DataFrame``.
    ``ValueError``
        A frame lacks a column of :data:`SUM_FIELDS`, such as the scores
        :func:`metrics` or :func:`from_sums` returns; or the frames' key
        columns differ, in their names or their dtypes.

    .. rubric:: Update equation

    The means are pooled by weight, and each moment about the mean picks up
    the parallel-axis term for the distance between its frame's mean and
    the pooled one. With ``g`` running over the frames, and ``mean`` and
    ``m2`` standing for each of ``mean_y`` and ``m2_y``, and ``mean_pred``
    and ``m2_pred``, in turn:

    .. code-block:: text

        w    = sum(w_g)
        mean = sum(w_g * mean_g) / w
        m2   = sum(m2_g + w_g * (mean_g - mean)**2)
        cov  = sum(cov_g + w_g * (mean_y_g - mean_y) * (mean_pred_g - mean_pred))

    ``n``, ``sse``, ``hits`` and ``signed`` are summed. A frame whose rows
    all weigh 0 (``w`` is 0, its means 0/0) adds its ``n`` and nothing
    else. Where every frame that weighs has one mean, the pooled mean is
    that value, exactly, so a constant target or prediction keeps a second
    moment of exactly 0, and its ``r2`` or ``ic`` the null of
    :func:`from_sums`, through any number of merges.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        chunks = [po.eval.sums(out.slice(i, 100), spec) for i in range(0, 400, 100)]
        merged = po.eval.merge_sums(*chunks)                  # four chunks of 100 rows, added
        whole = po.eval.sums(out, spec)                       # the 400 rows as one chunk
        print(f"m2_y {merged['m2_y'][0]:.6f} merged, {whole['m2_y'][0]:.6f} whole")

    .. code-block:: text

        m2_y 1932.454328 merged, 1932.454328 whole

    .. rubric:: See also

    - :func:`sums`, which makes the frames this adds.
    - :func:`from_sums`, which reads the scores from the result.

    .. rubric:: Notes

    This is the n-way form of Chan, Golub and LeVeque's merge. Every frame
    enters as one term of a sum, never as a difference of running totals,
    so merging a thousand chunks loses about as much as merging two.
    Measured on 100,000 rows at a level of 1e4 standard deviations, against
    the exact sums in rational arithmetic, over eight random streams: 2
    chunks merged within 6e-15 of the exact ``m2_y``, ``m2_pred`` and
    ``cov``, relative to each, and 1,000 chunks within 9e-15. Most of that
    is each frame's mean rounded to a double at that level, before any
    merging. A difference of running totals loses the level's square over
    the variance times the rounding, about 1e-8 there.

    That rounding of each frame's mean enters the merged moments at first
    order, about ``eps * |mean| / (spread * sqrt(N))`` relative, with ``N``
    the key's rows: it does not grow with the number of frames, nor shrink
    as they grow larger, but it grows as a key gets shorter, to 1e-13 at a
    level of 1e4 over 1,000 rows, as :func:`metrics`' Notes measure.
    """
    who = "po.eval.merge_sums"
    frames = [first, *rest]
    for f in frames:
        _frame(who, f)
        _check_sums(who, f)
    keys = [c for c in frames[0].columns if c not in SUM_FIELDS]
    for f in frames[1:]:
        other = [c for c in f.columns if c not in SUM_FIELDS]
        if other != keys:
            msg = (
                f"po.eval.merge_sums: every part must have the same key columns, but one has "
                f"{keys} and another {other}. Pass parts from po.eval.sums with the same group."
            )
            raise ValueError(msg)
        for k in keys:
            if f.schema[k] != frames[0].schema[k]:
                msg = (
                    f"po.eval.merge_sums: key column {k!r} is {frames[0].schema[k]} in one part "
                    f"and {f.schema[k]} in another, so their keys cannot be matched. Cast the "
                    "key to one dtype before po.eval.sums."
                )
                raise ValueError(msg)
    return _merge(frames, keys)


def _check_sums(who: str, f: pl.DataFrame) -> None:
    """Refuse a frame that is not one :func:`sums` made: one without every
    column of :data:`SUM_FIELDS` raised Polars' ``ColumnNotFoundError`` or
    ``ShapeError``, and :func:`metrics`' scores beside sums were read as
    keys (task 382a, A12)."""
    missing = [c for c in SUM_FIELDS if c not in f.columns]
    if not missing:
        return
    if any(c in f.columns for c in ("r2", "brier_skill")):
        msg = (
            f"{who}: a frame holds scores, not sums: its columns {f.columns} are those "
            "po.eval.metrics or po.eval.from_sums returns. Pass the frames po.eval.sums returns, "
            "whose scores po.eval.from_sums reads."
        )
    else:
        msg = (
            f"{who}: a frame lacks {missing} of the sums' columns, {list(SUM_FIELDS)}, so it is "
            "not a frame po.eval.sums returned. Pass the frames po.eval.sums returns."
        )
    raise ValueError(msg)


def _merge(frames: list[pl.DataFrame], keys: list[str], extra: Sequence[str] = ()) -> pl.DataFrame:
    """:func:`merge_sums` of ``frames`` by ``keys``, each column of
    ``extra`` added as ``n`` is."""
    w_g = pl.col("w")
    w = w_g.sum()
    # A part whose rows all weigh 0 has means of 0/0, NaN, and moments of
    # NaN; it adds its rows to `n` and nothing else, where its `0 * NaN`
    # terms turned the key's moments to NaN (task 382a, A1).
    weighed = w_g > 0

    def pooled(field: str) -> pl.Expr:
        # Parts whose means are all one value pool to that value exactly, so
        # a constant stays constant through any number of merges, its
        # parallel-axis terms exactly 0 (task 382a, A2).
        held = pl.col(field).filter(weighed)
        return (
            pl.when(held.min() == held.max())
            .then(held.first())
            .otherwise((w_g * pl.col(field)).filter(weighed).sum() / w)
        )

    mean_y, mean_p = pooled("mean_y"), pooled("mean_pred")

    def summed(term: pl.Expr) -> pl.Expr:
        return pl.when(weighed).then(term).otherwise(0.0).sum()

    def merged(field: str, expr: pl.Expr) -> pl.Expr:
        # A key whose weight is in one part alone keeps that part's sums as
        # they are: recomputed as `w * mean / w`, about one mean in ten moved
        # by its last bit, and a small `m2` by `w` times that bit squared
        # (task 359). A key whose parts all weigh 0 keeps the NaN of a
        # frame of such rows.
        return (
            pl.when(weighed.sum() == 1)
            .then(pl.col(field).filter(weighed).first())
            .when(pl.len() == 1)
            .then(pl.col(field).first())
            .when(w > 0)
            .then(expr)
            .otherwise(float("nan"))
            .alias(field)
        )

    return (
        pl.concat(frames)
        .group_by(keys)
        .agg(
            pl.col("n").sum(),
            w.alias("w"),
            merged("mean_y", mean_y),
            merged("mean_pred", mean_p),
            merged("m2_y", summed(pl.col("m2_y") + w_g * (pl.col("mean_y") - mean_y) ** 2)),
            merged(
                "m2_pred", summed(pl.col("m2_pred") + w_g * (pl.col("mean_pred") - mean_p) ** 2)
            ),
            merged(
                "cov",
                summed(
                    pl.col("cov")
                    + w_g * (pl.col("mean_y") - mean_y) * (pl.col("mean_pred") - mean_p)
                ),
            ),
            pl.col("sse").sum(),
            pl.col("hits").sum(),
            pl.col("signed").sum(),
            *(pl.col(c).sum() for c in extra),
        )
        .select(*keys, *SUM_FIELDS, *extra)
        .sort(keys)
    )


@renamed_keywords("po.eval.from_sums", _MIN_SAMPLES)
def from_sums(s: pl.DataFrame, *, min_samples: int = 30, binary: bool = False) -> pl.DataFrame:
    """Read :func:`metrics`' scores from the sums :func:`sums` made.

    Use it last, after :func:`sums` has reduced each chunk of a stream and
    :func:`merge_sums` has added them, to get the scores :func:`metrics`
    would give over every row at once.

    .. rubric:: Parameters

    ``s``
        A frame :func:`sums` or :func:`merge_sums` returned.
    ``min_samples``
        The fewest rows a key is scored on, 30 by default. A key with fewer
        rows keeps its row, with ``n`` filled and the scores null, as
        :func:`metrics` keeps it. It counts rows, ``n``, whatever they
        weigh: a row of weight 0 counts (see :func:`sums`' ``weight``).
    ``binary``
        Say that the sums were made under :func:`sums`' ``binary``, and
        name the columns as :func:`metrics` names them under it:
        ``brier_skill`` for ``r2`` and ``accuracy`` (at 0.5) for
        ``hit_rate``. ``False`` by default. The sums do not record their reading, so this
        cannot read it from them: which reading ``hit_rate`` is was fixed
        when the sums were made, since this divides ``hits`` by ``signed``.

    .. rubric:: Returns

    A ``DataFrame`` of the keys and :func:`metrics`' columns, with the same
    numbers: ``n``, ``r2`` (out-of-sample, against the target's mean),
    ``ic`` (the correlation of the prediction with the target),
    ``hit_rate``, ``mse`` and ``rmse``. Sums made with :func:`sums`'
    ``weight`` give each score weighted, ``mse`` as ``sse / w``. ``r2`` is
    null where the target never varied, and ``ic`` where either the target
    or the prediction never varied, where
    dividing by the variance of 0 would give an infinity that reads as a
    number. Sums whose ``w`` is 0, from rows that all weigh 0, give null
    ``r2``, ``ic``, ``mse`` and ``rmse``, since no row counts. There is no
    ``log_loss`` column under ``binary``: :data:`SUM_FIELDS` carries no sum
    for it. :func:`metrics` gives it, over a frame or a query.

    .. rubric:: Raises

    ``TypeError`` for an ``s`` that is not a ``DataFrame``; ``ValueError``
    for one that lacks a column of :data:`SUM_FIELDS`, such as the scores
    :func:`metrics` returns, and for a ``min_samples`` below 0; and
    :class:`~polars_online.WrongTypeError`, both a ``TypeError`` and a
    ``ValueError``, for a ``min_samples`` that is not a whole number.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        spec = po.spec.ewridge("ridge", targets=["y"], features=["x0", "x1"], half_life=100.0)
        out = po.ModelBank([spec]).fit_predict(df)
        s = po.eval.sums(out, spec, group="stock_id")          # ten numbers per stock
        scores = po.eval.from_sums(s)
        for stock, n, r2 in scores.select("stock_id", "n", "r2").iter_rows():
            print(f"{stock}: {n} rows, r2 {r2:.3f}")

    .. code-block:: text

        b0: 99 rows, r2 0.939
        b1: 99 rows, r2 0.938
        b2: 99 rows, r2 0.947
        b3: 99 rows, r2 0.948

    .. rubric:: See also

    - :func:`sums` and :func:`merge_sums`, which make the sums.
    - :func:`metrics`, the same scores in one call, over a frame or a query.
    """
    _frame("po.eval.from_sums", s)
    _check_sums("po.eval.from_sums", s)
    _check_min_samples("po.eval.from_sums", min_samples)
    keys = [c for c in s.columns if c not in SUM_FIELDS]
    return _thin(_scores(s, keys, binary=binary), min_samples, keys)


def _scores(s: pl.DataFrame, keys: list[str], *, binary: bool) -> pl.DataFrame:
    """:func:`from_sums` before ``min_samples``, and :func:`metrics`'
    ``log_loss`` after ``accuracy`` where ``s`` holds its sum."""
    # Rows that all weigh 0 have no weighted score: their means are 0/0, so
    # every sum but `sse` is NaN, and Polars orders NaN above 0, so each
    # score is null by `w` first, as `hit_rate` is by `signed` (task 359).
    weighed = pl.col("w") > 0
    mse = pl.when(weighed).then(pl.col("sse") / pl.col("w"))
    m2_y, m2_pred = pl.col("m2_y"), pl.col("m2_pred")
    # The roots before the product: `m2_y * m2_pred` overflowed to inf (an
    # `ic` of 0.0) or underflowed to 0 (null) at scales inside the input
    # bound of 1e100, where each root cannot. Clipped, since the roots'
    # rounding read two rows that agree as 1.0000000000000002 (task 382a,
    # A3 and A16).
    ic = (pl.col("cov") / (m2_y.sqrt() * m2_pred.sqrt())).clip(-1.0, 1.0)
    log_loss = [(pl.col(_LOG_LOSS) / pl.col("n")).alias("log_loss")] if _LOG_LOSS in s else []
    return s.select(
        *keys,
        pl.col("n"),
        pl.when(weighed & (m2_y > 0))
        .then(1.0 - pl.col("sse") / m2_y)
        .otherwise(None)
        .alias("brier_skill" if binary else "r2"),
        pl.when(weighed & (m2_y > 0) & (m2_pred > 0)).then(ic).otherwise(None).alias("ic"),
        pl.when(pl.col("signed") > 0)
        .then(pl.col("hits") / pl.col("signed"))
        .otherwise(None)
        .alias("accuracy" if binary else "hit_rate"),
        *log_loss,
        mse.alias("mse"),
        mse.sqrt().alias("rmse"),
    ).sort(keys)


def _paired_residuals(
    df: pl.DataFrame,
    a: str,
    b: str,
    targets: Sequence[str] | None,
    a_suffix: str,
    b_suffix: str,
    who: str,
) -> dict[str, tuple[pl.Expr, pl.Expr]]:
    """Per target, ``a``'s and ``b``'s residual fields, resolved as
    :func:`seqtest` resolves them, NaN, infinities and values past the
    bank's input bound read as null."""
    have = {a: _resid_fields(who, df, a), b: _resid_fields(who, df, b)}
    if targets is None:
        bodies = [f.removeprefix("resid_") for f in have[a] if f.endswith(a_suffix)]
        stems = [x[: len(x) - len(a_suffix)] for x in bodies]
        targets = [t for t in stems if f"resid_{t}{b_suffix}" in have[b]]
        if not targets:
            msg = (
                f"{who}: {a!r} and {b!r} share no residual field with suffixes {a_suffix!r} "
                f"and {b_suffix!r}: {a!r} has {have[a]}, and {b!r} has {have[b]}. Name the "
                f"targets, or set the suffixes to match."
            )
            raise ValueError(msg)
    out: dict[str, tuple[pl.Expr, pl.Expr]] = {}
    for t in targets:
        pair = []
        for side, suffix in ((a, a_suffix), (b, b_suffix)):
            want = f"resid_{t}{suffix}"
            if want not in have[side]:
                msg = (
                    f"{who}: target {t!r} names no residual of {side!r}, which has no field "
                    f"{want!r} (its residual fields are {have[side]}). Set a_suffix and b_suffix "
                    "to the suffixes these fields carry, or name a target both specs predict."
                )
                raise ValueError(msg)
            r = pl.col(side).struct.field(want).cast(pl.Float64)
            pair.append(pl.when(r.is_finite() & (r.abs() <= _INPUT_BOUND)).then(r))
        out[t] = (pair[0], pair[1])
    return out


def _check_name(who: str, df: pl.DataFrame, name: str) -> None:
    """Refuse a result ``name`` the frame already has: ``with_columns`` would
    replace that column -- the target, a spec's output -- with nothing said."""
    if name in df.columns:
        msg = (
            f"{who}: name {name!r} is a column of the frame already, and the result would "
            "replace it. Give the result a name the frame does not have."
        )
        raise ValueError(msg)


def _free(taken: set[str], label: str) -> str:
    """A working column's name that is in ``taken`` neither as the frame's
    column nor as the result's name; added to ``taken``."""
    column = f"__po_{label}"
    while column in taken:
        column = "_" + column
    taken.add(column)
    return column


def _normal_sf(z: float | None) -> float | None:
    """``P(Z > z)`` for a standard normal ``Z``."""
    return None if z is None or math.isnan(z) else 0.5 * math.erfc(z / math.sqrt(2.0))


#: The rows a key needs per lag of the Newey-West variance, ``lags + 1``
#: counting lag 0, before its statistic is read: a rule of thumb (see
#: :func:`diebold_mariano`'s ``lags``).
_ROWS_PER_LAG = 4


def _few_rows(lags: int) -> int:
    """The most rows with a value on which a key's statistic is null for
    its ``lags``: ``4 (lags + 1)``. The deviations from the mean sum to 0,
    so the Bartlett sum of a short key shrinks toward 0 as ``lags`` nears
    its rows, and the statistic grows with no meaning."""
    return _ROWS_PER_LAG * (lags + 1)


def _hac_test(
    df: pl.DataFrame | pl.LazyFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    stat: str,
    min_samples: int,
) -> pl.DataFrame:
    """Per key and target, the mean of each value column and its t
    statistic against Newey and West's variance: with ``d`` the value and
    ``d̄`` its mean over the ``n`` rows that have one,

    ``S = Σ (d − d̄)² + 2 Σ_{l ≤ L} (1 − l/(L+1)) Σ_t (d_t − d̄)(d_{t−l} − d̄)``,
    ``stat = d̄ · n / sqrt(S)``.
    """
    if isinstance(df, pl.LazyFrame):
        got = _hac_query(df, values, keys, lags, stat)
        if got is not None:
            return _thin(got, min_samples, [*keys, "target"])
        df = _probe(df)
    frames = []
    for t, d in values.items():
        col = "__po_d"
        while col in df.columns:
            col = "_" + col
        rows = df.select(*keys, d.alias(col)).filter(pl.col(col).is_not_null())
        dd = pl.col(col)
        dev = dd - dd.mean()
        s = (dev * dev).sum()
        for lag in range(1, lags + 1):
            weight = 1.0 - lag / (lags + 1.0)
            s = s + 2.0 * weight * (dev * (dd.shift(lag) - dd.mean())).sum()
        exprs = [
            pl.len().alias("n"),
            dd.mean().alias("mean"),
            # Null where every value is the same, whose S is 0 but for the
            # rounding of the mean, and on a key too short for its lags.
            pl.when((s > 0) & (dd.min() < dd.max()) & (pl.len() > _few_rows(lags)))
            .then(dd.mean() * pl.len() / s.sqrt())
            .alias(stat),
        ]
        g = rows.group_by(keys, maintain_order=True).agg(exprs) if keys else rows.select(exprs)
        frames.append(g.with_columns(pl.lit(t).alias("target")))
    out = pl.concat(frames).select(*keys, "target", pl.col("n").cast(pl.Int64), "mean", stat)
    return _thin(out, min_samples, [*keys, "target"])


def _hac_query(
    lf: pl.LazyFrame, values: dict[str, pl.Expr], keys: list[str], lags: int, stat: str
) -> pl.DataFrame | None:
    """:func:`_hac_test` over a query, read in one pass a chunk at a time, or
    ``None`` where no row has a value.

    Each lag's sum is a co-moment about the overall mean, of the pairs of
    rows ``l`` apart with a value: with ``x`` the later of a pair and ``y``
    the earlier, ``Σ (x − d̄)(y − d̄) = C + n (x̄ − d̄)(ȳ − d̄)``, where ``C``
    is the pairs' co-moment about their own means. So each chunk keeps, per
    key and lag, the pairs' count, means and co-moment, which merge as
    :func:`merge_sums` merges a chunk's (lag 0 is the rows themselves), and
    the last ``lags`` values of each key, which pair with the next chunk's
    first rows. What it holds is one chunk, and per key and target ``lags``
    values and ``lags + 1`` rows of sums, not the query's rows."""
    schema = lf.collect_schema()
    taken = set(schema)
    col, other, mark, lag_col = (_free(taken, n) for n in ("d", "d_back", "carried", "lag"))
    needed = set(keys).union(*(d.meta.root_names() for d in values.values()))
    plan = lf.select([c for c in schema if c in needed])
    pair_keys = [*keys, lag_col]
    carry: dict[str, pl.DataFrame | None] = dict.fromkeys(values)
    seen: dict[str, pl.DataFrame | None] = dict.fromkeys(values)
    parts: dict[str, list[pl.DataFrame]] = {t: [] for t in values}
    # Each key's least and greatest value: where they are equal, every
    # value is, and the statistic is null.
    bounds: dict[str, pl.DataFrame | None] = dict.fromkeys(values)
    x, y = pl.col(col), pl.col(other)
    lo, hi = _free(taken, "lo"), _free(taken, "hi")
    aggs = [
        pl.len().cast(pl.Int64).alias("n"),
        pl.len().cast(pl.Float64).alias("w"),
        x.mean().alias("mean_y"),
        y.mean().alias("mean_pred"),
        ((x - x.mean()) ** 2).sum().alias("m2_y"),
        ((y - y.mean()) ** 2).sum().alias("m2_pred"),
        ((x - x.mean()) * (y - y.mean())).sum().alias("cov"),
        *(pl.lit(0.0).alias(f) for f in ("sse", "hits", "signed")),
    ]
    for chunk in plan.collect_batches(chunk_size=_chunk_rows(), maintain_order=True):
        for t, d in values.items():
            rows = chunk.select(*keys, d.alias(col)).filter(x.is_not_null())
            if rows.is_empty():
                continue
            ends = [x.min().alias(lo), x.max().alias(hi)]
            spans = rows.group_by(keys).agg(ends) if keys else rows.select(ends)
            was_spans = bounds[t]
            if was_spans is not None:
                spans = pl.concat([was_spans, spans])
                ends = [pl.col(lo).min(), pl.col(hi).max()]
                spans = spans.group_by(keys).agg(ends) if keys else spans.select(ends)
            bounds[t] = spans
            # The keys in the order they first have a value, as the frame's
            # `group_by(maintain_order=True)` gives them.
            first = rows.select(keys).unique(maintain_order=True) if keys else rows.head(1)
            was = seen[t]
            if was is None:
                seen[t] = first
            elif keys:
                fresh = first.join(was, on=keys, how="anti", nulls_equal=True)
                seen[t] = pl.concat([was, fresh])
            both = rows.with_columns(pl.lit(False).alias(mark))
            held = carry[t]
            if held is not None:
                both = pl.concat([held, both])
            got = []
            for lag in range(lags + 1):
                back = x.shift(lag).over(keys) if keys and lag else x.shift(lag)
                pairs = both.with_columns(back.alias(other)).filter(~pl.col(mark) & y.is_not_null())
                summed = pairs.group_by(keys).agg(aggs) if keys else pairs.select(aggs)
                got.append(summed.filter(pl.col("n") > 0).with_columns(pl.lit(lag).alias(lag_col)))
            parts[t].append(pl.concat(got).select(*pair_keys, *SUM_FIELDS))
            if len(parts[t]) >= _MERGE_EVERY:
                parts[t] = [_merge(parts[t], pair_keys)]
            if lags:
                kept = (
                    both.group_by(keys, maintain_order=True).tail(lags) if keys else both.tail(lags)
                )
                carry[t] = kept.with_columns(pl.lit(True).alias(mark))
    frames = []
    for t in values:
        if not parts[t]:
            if not keys:
                # The frame's row for a target no row has a value of.
                types = {"target": pl.String, "n": pl.Int64, "mean": pl.Float64, stat: pl.Float64}
                none = {"target": [t], "n": [0], "mean": [None], stat: [None]}
                frames.append(pl.DataFrame(none, schema=types))
            continue
        merged = _merge(parts[t], pair_keys)
        own = merged.filter(pl.col(lag_col) == 0).select(
            *keys, "n", pl.col("mean_y").alias("mean"), pl.col("m2_y").alias("s")
        )
        s_expr = pl.col("s")
        if lags:
            lagged = merged.filter(pl.col(lag_col) > 0)
            on = own.select(*keys, "mean")
            if keys:
                lagged = lagged.join(on, on=keys, how="left", nulls_equal=True)
            else:
                lagged = lagged.with_columns(pl.lit(own["mean"][0]).alias("mean"))
            weight = 1.0 - pl.col(lag_col) / (lags + 1.0)
            term = (
                2.0
                * weight
                * (
                    pl.col("cov")
                    + pl.col("w")
                    * (pl.col("mean_y") - pl.col("mean"))
                    * (pl.col("mean_pred") - pl.col("mean"))
                )
            )
            extra = (
                lagged.group_by(keys).agg(term.sum().alias("__po_lagged"))
                if keys
                else lagged.select(term.sum().alias("__po_lagged"))
            )
            own = (
                own.join(extra, on=keys, how="left", nulls_equal=True)
                if keys
                else own.with_columns(extra.to_series())
            )
            s_expr = s_expr + pl.col("__po_lagged").fill_null(0.0)
        ends_t = bounds[t]
        if ends_t is not None:
            own = (
                own.join(ends_t, on=keys, how="left", nulls_equal=True)
                if keys
                else own.with_columns(pl.lit(ends_t[c][0]).alias(c) for c in (lo, hi))
            )
        live = (s_expr > 0) & (pl.col(lo) < pl.col(hi)) & (pl.col("n") > _few_rows(lags))
        result = own.select(
            *keys,
            pl.lit(t).alias("target"),
            "n",
            "mean",
            pl.when(live).then(pl.col("mean") * pl.col("n") / s_expr.sqrt()).alias(stat),
        )
        order = seen[t]
        if keys and order is not None:
            result = order.join(
                result, on=keys, how="left", nulls_equal=True, maintain_order="left"
            )
        frames.append(result)
    if not frames:
        return None
    return pl.concat(frames).select(*keys, "target", pl.col("n").cast(pl.Int64), "mean", stat)


@overload
def _ew_test(
    df: pl.DataFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
    min_samples: int,
) -> pl.DataFrame: ...


@overload
def _ew_test(
    df: pl.LazyFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
    min_samples: int,
) -> pl.LazyFrame: ...


def _ew_test(
    df: pl.DataFrame | pl.LazyFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
    min_samples: int,
) -> pl.DataFrame | pl.LazyFrame:
    """Per row, the same t statistic with each row before it weighed by
    ``λ^age``, ``λ = 0.5 ** (1 / half_life)``, one step per row that has a
    value, the row itself included:

    ``m = Σ ω d / Σ ω``, ``S = Σ ω_t² (d_t − m)² + 2 Σ_l (1 − l/(L+1)) Σ_t ω_t ω_{t−l}
    (d_t − m)(d_{t−l} − m)``, ``stat = m Σ ω / sqrt(S)``.

    ``S`` is summed about the weighted means, so no sum in it cancels a
    level against itself. With ``ω_t ω_{t−l} = λ^l ω_t²``, each double sum
    is one at the weights ``ω_t²``, which ``ewm_mean`` and
    ``ewm_var(bias=True)`` at ``α = 1 − λ²`` read: with ``W₂ = Σ ω²``, and
    ``m₂`` and ``v₂`` the mean and variance of ``d`` at those weights, lag
    0's is ``W₂ (v₂ + (m₂ − m)²)``; lag ``l``'s, over the pairs ``(u, v) =
    (d_t, d_{t−l})``, is ``λ^l W_l (c_l + (μ_l − m)(ν_l − m))``, with
    ``W_l`` the pairs' weight, ``μ_l`` and ``ν_l`` their means and ``c_l =
    (var(u + v) − var(u − v)) / 4`` their covariance. The statistic is null
    where ``S`` is no more than its rounding, :func:`_ew_rounding`, and
    until the key has ``min_samples`` rows with a value and more than
    :func:`_few_rows`.
    """
    if isinstance(df, pl.LazyFrame):
        return _ew_query(df, values, keys, lags, half_life, stat, name, min_samples)
    return _ew_chunk(df, values, keys, lags, half_life, stat, name, min_samples, {})


def _ew_rounding(
    lags: int, lam2: float, s0: pl.Expr, r: pl.Expr, w1: pl.Expr, w2: pl.Expr
) -> pl.Expr:
    """The most rounding the per-row ``S`` carries, from ``s0 = W₂ (v₂ +
    (m₂ − m)²)``, its lag-0 sum, ``r = W₂ (v₂ + m₂²) = Σ ω² d²``, and the
    row's weights ``w1 = Σ ω`` and ``w2 = W₂ = Σ ω²``:
    ``E = 10 (1 + L) ((η_r + η_w) s0 + η_r sqrt(s0 r) + η_r² r)``, with
    ``η_r = ε w1`` and ``η_w = ε / ((1 − λ²) W₂)``.

    ``η_r``, the recursions. Each mean and variance is an exponentially
    weighted recursion (``ewm_mean``, ``ewm_var``) that steps by ``(x −
    mean) / weight``, its weight the sum of its own weights so far: at
    most the rows read and at most ``1 / α``, so never above ``w1`` (``m``
    is at ``α = 1 − λ``, whose weight is ``w1``; the rest at ``1 − λ²``,
    whose weight ``W₂`` is below it). Each step rounds by about ``ε`` of the
    value held, and the next step keeps ``1 − 1 / weight`` of that, so the
    roundings rest within ``ε · weight`` of its size; and a mean of a
    constant ``c``, whose step ``(c − mean) / weight`` is below half an ulp
    and rounds to nothing, rests up to ``ε |c| weight / 2`` from it. So
    every mean is within ``η_r`` of its size, and so is ``λ²`` raised to
    the recursion's powers, ``λ²`` being ``λ`` squared to a double. A key
    with few rows has gathered few roundings: ``η_r`` grows with ``w1``
    to ``ε / (1 − λ)`` once the key has had many half-lives, where a
    settled recursion's ``ε / (1 − λ²)`` for every row nulled the rows of
    a young key at a long half-life that were right to 1e-9 and better.

    ``η_w``, the totals. ``W = (1 − μ^n) / (1 − μ)`` is read from the
    count, and ``μ^n`` rounds to within ``ε``, so ``1 − μ^n`` is within
    ``ε`` of its value, ``ε / ((1 − μ) W)`` relative: ``η_w`` for ``W₂``,
    at most ``2 η_w`` for ``w1``. Near ``μ = 1`` it is most of the
    rounding; it is ``ε`` once a key has had many half-lives.

    The sums. On rows of a constant ``c``, where ``S`` is 0, ``|m₂ − m| <=
    2 η_r |c|``, and what is left is: lag 0, ``W₂ ((2 η_r c)² + (η_r
    c)²)``, the difference of the means squared and the variance's floor
    (its mean's offset squared); lag ``l``, ``λ^l W_l ((2 η_r c)² / 4 +
    (2 η_r c)²)``, the floor of ``var(u + v)`` (``u + v = 2c``; ``u − v``
    is 0 exactly) and the product of the means' offsets; each at most ``10
    η_r² W₂ c² = 10 η_r² r``, at Bartlett weights that add up to ``1 +
    L``. Away from a constant the error is relative, ``η_r + η_w`` of each
    sum, and the lag sums are each at most ``s0`` by Cauchy and Schwarz
    (``λ^l Σ ω_t² (d_t − m)² <= s0``, and so for ``d_{t−l}``), the
    covariance's polarisation at most their mean, so ``10 (1 + L) (η_r +
    η_w) s0`` holds the terms' rounding; and the cross term of the means'
    offsets, ``2 W₂ |m₂ − m| 2 η_r |c|``, is at most ``4 η_r sqrt(s0
    r)``. ``E`` adds the three: no less than ``S``'s rounding, so ``S <=
    E`` is ``S`` zero but for rounding, and above it the statistic's own
    rounding is at most about ``E / (2 S)`` of its value."""
    eps = sys.float_info.epsilon
    eta_r = eps * w1
    eta_w = eps / ((1.0 - lam2) * w2)
    return (
        10.0 * (1.0 + lags) * ((eta_r + eta_w) * s0 + eta_r * (s0 * r).sqrt() + eta_r * eta_r * r)
    )


def _ew_query(
    lf: pl.LazyFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
    min_samples: int,
) -> pl.LazyFrame:
    """:func:`_ew_test` over a query: a query of its rows with the record
    column added, read in one pass a chunk at a time when it runs.

    Each of the statistic's weighted means and variances is of the rows
    ``j ≤ t`` at weights ``μ^{t−j}`` (``μ`` is ``λ`` or ``λ²``), whose total
    follows from the count, so a chunk carries in each key's count and its
    means and variances after the rows before it, decays their weight by
    ``μ^{k+1}`` over the chunk's own ``k + 1`` rows so far, and pools them
    with the chunk's own, as two groups' means and variances pool: every
    term non-negative. With them it carries the last ``lags`` values of
    each key, which the chunk's first rows read as the rows ``l`` before
    them. What it holds is one chunk, and per key and target ``lags``
    values and ``3 + 4 lags`` means and variances, not the query's rows."""
    schema = _ew_test(_probe(lf), values, keys, lags, half_life, stat, name, min_samples).schema
    needed = set(keys).union(*(d.meta.root_names() for d in values.values()))

    def step() -> Callable[[pl.DataFrame], pl.DataFrame]:
        carry: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}

        def one(chunk: pl.DataFrame) -> pl.DataFrame:
            return _ew_chunk(chunk, values, keys, lags, half_life, stat, name, min_samples, carry)

        return one

    return _row_query(lf, needed, schema, step)


def _ew_chunk(
    df: pl.DataFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
    min_samples: int,
    carry: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
) -> pl.DataFrame:
    """One chunk of :func:`_ew_query`, after the chunks whose counts, means,
    variances and last values ``carry`` holds per target; ``carry`` is
    updated in place. A key the chunks before did not have reads the
    chunk's own means and variances as they are, so a frame is one chunk
    with nothing carried in, and a query of one chunk gives the frame's
    result to the bit."""
    lam = 0.5 ** (1.0 / half_life)
    lam2 = lam * lam
    taken = {*df.columns, name}
    idx = _free(taken, "row")
    col, mark, count = _free(taken, "d"), _free(taken, "carried"), _free(taken, "count")
    seq = _free(taken, "seq")
    # Each pooled quantity: its name, its decay, the lag whose pairs it is
    # of (0 for the rows themselves), and the mean it is a variance about
    # (None for a mean), as (name, sign) pairs of the means of `u` and `v`.
    means = [("m", lam, 0), ("m2", lam2, 0)]
    variances: list[tuple[str, int, list[tuple[str, float]]]] = [("v2", 0, [("m2", 1.0)])]
    for lag in range(1, lags + 1):
        means += [(f"mu{lag}", lam2, lag), (f"nu{lag}", lam2, lag)]
        variances += [
            (f"vp{lag}", lag, [(f"mu{lag}", 1.0), (f"nu{lag}", 1.0)]),
            (f"vm{lag}", lag, [(f"mu{lag}", 1.0), (f"nu{lag}", -1.0)]),
        ]
    names = [q for q, _, _ in means] + [q for q, _, _ in variances]
    held = {q: _free(taken, f"in_{q}") for q in names}
    now = {q: _free(taken, f"now_{q}") for q in names}
    base = df.with_row_index(idx)
    fields: list[pl.Expr] = []
    temps: list[str] = []

    def over(e: pl.Expr) -> pl.Expr:
        return e.over(keys) if keys else e

    raw = pl.col(col)
    # Rows of the key before the chunk, `c`, and up to this row, `n`; `j`
    # the chunk's own.
    c = pl.col(count).cast(pl.Float64)
    j = pl.col(seq)

    def total(mu: float, rows: pl.Expr) -> pl.Expr:
        """``Σ μ^age`` over ``rows`` rows, 0 over none."""
        return (1.0 - pl.lit(mu).pow(rows.clip(lower_bound=0.0))) / (1.0 - mu)

    def pairs(rows: pl.Expr, lag: int) -> pl.Expr:
        return rows - float(lag)

    def ewm(e: pl.Expr, mu: float, var: bool = False) -> pl.Expr:
        a = 1.0 - mu
        got = (
            e.ewm_var(alpha=a, adjust=True, bias=True) if var else e.ewm_mean(alpha=a, adjust=True)
        )
        return over(got)

    for t, d in values.items():
        rows = base.select(idx, *keys, d.alias(col)).filter(raw.is_not_null())
        carried, tail_in = carry.get(t, (None, None))
        both = rows.with_columns(pl.lit(False).alias(mark))
        if tail_in is not None:
            both = pl.concat([tail_in, both], how="diagonal")
        backs = [over(raw.shift(lag)).alias(f"__po_back{lag}") for lag in range(1, lags + 1)]
        loc = both.with_columns(backs).filter(~pl.col(mark))
        loc = _carried_in(loc, carried, keys, [count], list(held.values()))
        # The chunk's own means and variances, each key's rows from the
        # chunk's first: of the rows, and of each lag's pairs, which start
        # where a row `l` back has a value.
        own: dict[str, pl.Expr] = {
            "m": ewm(raw, lam),
            "m2": ewm(raw, lam2),
            "v2": ewm(raw, lam2, var=True),
        }
        for lag in range(1, lags + 1):
            v = pl.col(f"__po_back{lag}")
            u = pl.when(v.is_not_null()).then(raw)
            own |= {
                f"mu{lag}": ewm(u, lam2),
                f"nu{lag}": ewm(v, lam2),
                f"vp{lag}": ewm(u + v, lam2, var=True),
                f"vm{lag}": ewm(u - v, lam2, var=True),
            }
        loc = loc.with_columns(
            *(e.alias(now[q]) for q, e in own.items()),
            (over(pl.int_range(pl.len())).cast(pl.Float64) + 1.0).alias(seq),
        )

        # Pool the carried quantities, decayed over the chunk's rows so far,
        # with the chunk's own: `A` the carried weight, `B` the chunk's, one
        # pair of columns per decay and lag.
        shares: dict[tuple[float, int], tuple[str, str]] = {}
        for mu, lag in sorted({(mu, lag) for _, mu, lag in means}):
            shares[(mu, lag)] = (_free(taken, f"a{len(shares)}"), _free(taken, f"b{len(shares)}"))

        def share(mu: float, lag: int) -> tuple[pl.Expr, pl.Expr]:
            before = total(mu, pairs(c, lag))
            after = total(mu, pairs(c + j, lag)) - pl.lit(mu).pow(j) * before
            return pl.lit(mu).pow(j) * before, after

        loc = loc.with_columns(
            e.alias(nm)
            for key, names_ab in shares.items()
            for e, nm in zip(share(*key), names_ab, strict=True)
        )

        pooled: dict[str, pl.Expr] = {}
        shift: dict[str, pl.Expr] = {}
        for q, mu, lag in means:
            a, b = (pl.col(nm) for nm in shares[(mu, lag)])
            gap = pl.col(held[q]) - pl.col(now[q])
            shift[q] = gap
            pooled[q] = (
                pl.when(a <= 0.0)
                .then(pl.col(now[q]))
                .when(b <= 0.0)
                .then(pl.col(held[q]))
                .otherwise(pl.col(now[q]) + gap * (a / (a + b)))
            )
        for q, lag, parts in variances:
            a, b = (pl.col(nm) for nm in shares[(lam2, lag)])
            gap = functools.reduce(operator.add, (sign * shift[p] for p, sign in parts))
            pooled[q] = (
                pl.when(a <= 0.0)
                .then(pl.col(now[q]))
                .when(b <= 0.0)
                .then(pl.col(held[q]))
                .otherwise(
                    (a * pl.col(held[q]) + b * pl.col(now[q])) / (a + b)
                    + a * b * gap * gap / ((a + b) * (a + b))
                )
            )
        loc = loc.with_columns(pooled[q].alias(now[q]) for q in names)
        got = loc.with_columns(pl.col(now[q]).alias(held[q]) for q in names)

        def p(q: str) -> pl.Expr:
            return pl.col(held[q])

        n = c + j
        w1, w2 = total(lam, n), total(lam2, n)
        m = p("m")
        s0 = w2 * (p("v2") + (p("m2") - m) * (p("m2") - m))
        s = s0
        for lag in range(1, lags + 1):
            cov = (p(f"vp{lag}") - p(f"vm{lag}")) / 4.0
            term = lam**lag * total(lam2, pairs(n, lag))
            term = term * (cov + (p(f"mu{lag}") - m) * (p(f"nu{lag}") - m))
            weight = 1.0 - lag / (lags + 1.0)
            s = s + 2.0 * weight * pl.when(n > lag).then(term).otherwise(0.0)
        r = w2 * (p("v2") + p("m2") * p("m2"))
        out_col = _free(taken, f"{stat}_{len(temps)}")
        cut = _ew_rounding(lags, lam2, s0, r, w1, w2)
        # `n`, the key's rows with a value up to this one, is a whole number
        # held in a float, so the comparisons are exact.
        live = (s > cut) & (n >= float(min_samples)) & (n > float(_few_rows(lags)))
        got = got.with_columns(pl.when(live).then(m * w1 / s.sqrt()).alias(out_col))
        base = base.join(got.select(idx, out_col), on=idx, how="left", maintain_order="left")
        fields.append(pl.col(out_col).alias(f"{stat}_{t}"))
        temps.append(out_col)
        if not got.is_empty():
            last = [
                (pl.col(count).first() + pl.len().cast(pl.Int64)).alias(count),
                *(pl.col(held[q]).last() for q in names),
            ]
            summed = got.group_by(keys).agg(last) if keys else got.select(last)
            back_tail = both.select(*keys, col, pl.lit(True).alias(mark))
            if lags:
                back_tail = (
                    back_tail.group_by(keys, maintain_order=True).tail(lags)
                    if keys
                    else back_tail.tail(lags)
                )
            else:
                back_tail = back_tail.head(0)
            carry[t] = (
                _carry_on(carried, summed.select(*keys, count, *held.values()), keys),
                back_tail,
            )
    return base.with_columns(pl.struct(fields).alias(name)).drop(idx, *temps)


def _check_half_life(who: str, half_life: float) -> None:
    """The per-row form's ``half_life``: above 0, finite, and short enough
    that ``0.5 ** (1 / half_life)`` is below 1 in a double, up to about
    1.25e16 rows. An infinite one, or a longer one, reached polars'
    ``ewm_mean`` as ``alpha = 0``, which it refuses with no word of the
    parameter."""
    finite = 0 < half_life < math.inf
    if not finite or 0.5 ** (1.0 / half_life) == 1.0:
        why = (
            ""
            if not finite
            else ", longer than a double can tell from no decay (0.5 ** (1 / half_life) is 1.0)"
        )
        msg = (
            f"{who}: half_life must be finite and above 0, but it is {half_life}{why}. Give "
            f"half_life a positive number of rows, or leave it None for one statistic over "
            f"every row."
        )
        raise ValueError(msg)


def _lags(who: str, lags: Any) -> int:
    """``lags`` as an int, once it is checked to be a whole number of at least 0:
    any whole number, numpy's integers included, but not a bool."""
    wrong_type = isinstance(lags, bool) or not isinstance(lags, numbers.Integral)
    if wrong_type:
        msg = (
            f"{who}: lags must be a whole number of at least 0, but it is {lags!r}. Set lags "
            "to a whole number: 0 where the loss differentials are uncorrelated from row to "
            "row, or about twice the horizon for a target that looks ahead."
        )
        raise WrongTypeError(msg)
    if operator.index(lags) < 0:
        msg = f"{who}: lags must be at least 0, but it is {lags}. Set lags to 0 or more."
        raise ValueError(msg)
    return operator.index(lags)


@overload
def diebold_mariano(
    df: pl.DataFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "diebold_mariano",
) -> pl.DataFrame: ...


@overload
def diebold_mariano(
    df: pl.LazyFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: None = None,
    min_samples: int = 30,
    name: str = "diebold_mariano",
) -> pl.DataFrame: ...


@overload
def diebold_mariano(
    df: pl.LazyFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: float,
    min_samples: int = 30,
    name: str = "diebold_mariano",
) -> pl.LazyFrame: ...


@renamed_keywords("po.eval.diebold_mariano", _GROUP)
def diebold_mariano(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    a: str,
    b: str,
    targets: Sequence[str] | None = None,
    a_suffix: str = "",
    b_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "diebold_mariano",
) -> pl.DataFrame | pl.LazyFrame:
    """Test whether two specs' out-of-sample squared errors differ on average,
    by Diebold and Mariano's test.

    Use it to ask whether one spec predicts better than another that is not
    nested in it. It is Diebold and Mariano's (1995) test, with Newey and
    West's (1987) variance. When one spec's features are a subset of the other's,
    use :func:`clark_west`, since this test leans toward the smaller of two
    nested specs. For evidence you may read at every row as the rows come,
    use :func:`seqtest`, which asks a different question: which spec is
    closer on more rows (see Notes).

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding both specs' output columns, or a
        query of one, a ``LazyFrame``, which this reads one chunk at a time
        (see Notes).
    ``a``, ``b``
        The names of the two specs' output columns. The statistic is
        positive when ``a`` was closer.
    ``targets``, ``a_suffix``, ``b_suffix``
        :func:`seqtest`'s, in compare mode: the targets whose residuals are
        compared (every target the two share by default), and the suffix of
        each side's residual field.
    ``lags``
        The lags of the Newey-West variance, 0 by default. 0 is right only
        where the differentials are uncorrelated from row to row, which
        holds where neither spec's one-step errors can be predicted from
        its past errors. Two specs that each leave out something persistent,
        such as a feature that moves slowly, make errors that run from row
        to row, and so differentials that do, however equal their accuracy;
        then ``lags=0`` understates the variance. Two ridges, each on one of
        two features that follow an autoregression at 0.9, equally accurate
        by construction, were found to differ at 5% on 28% of simulated
        streams of 200 rows and 32% of streams of 2,000 at ``lags=0``. Read
        the differentials' autocorrelation, or give ``lags``.

        For a target that looks ``h`` rows ahead, rows fewer than ``h``
        apart share shocks, so their differentials correlate whatever the
        specs. Give about ``2 h``, as many as the bank's diagnostics take
        for the same variance: Newey and West's weights count lag ``l`` at
        ``1 - l / (lags + 1)``, so ``h - 1`` lags count the overlap's
        correlations at less than their weight, and the test rejects too
        often however long the stream. On simulated errors of two equally
        accurate ``h``-step forecasts, at ``h = 5`` and ``h = 20``,
        ``lags=2h`` rejected at 5% on 6% of streams of 20,000 rows, 7% of
        streams of 2,000 and 9% to 15% of streams of 200, where ``h - 1``
        rejected on 8% to 9%, 9% to 10% and 10% to 13%. On short keys no
        number of lags holds 5%: read a p-value near 0.05 there as weaker
        evidence than it says.

        A key with at most ``4 (lags + 1)`` rows with a differential reads
        a null statistic, whatever ``min_samples`` is. The deviations from
        the mean sum to zero, so the variance's sum over a key shrinks
        toward zero as ``lags`` nears the key's rows, and the statistic
        grows with no meaning: on 40 rows of noise, ``lags=10`` rejected at
        5% on 15% of streams and ``lags=19`` on 21%, and on two rows
        ``lags=10`` read 11.7. The 4 is a rule of thumb; give a key many
        times ``lags`` rows.
    ``group``
        :func:`seqtest`'s: the key columns to test each group apart.
    ``half_life``
        ``None``, the default, gives one statistic over every row. A number
        of rows, finite and above 0, gives the statistic on every row
        instead, exponentially weighted. Each row up to it counts
        ``0.5 ** (age / half_life)``, where ``age`` is the number of rows
        with a differential after it, up to this row, so the row itself has
        age 0.
    ``min_samples``
        The fewest rows with a differential a key is tested on, 30 by
        default. Without ``half_life``, a key with fewer rows keeps its
        row, with ``n`` filled and the other three columns null, as Polars
        gives null below its ``min_samples``. With ``half_life``, a key's
        statistic is null on its rows before its ``min_samples``-th row
        with a differential. On a key's first rows the statistic is a mean
        of a few rows over an estimate of its own spread, far from ``N(0,
        1)``: on simulated noise it passed 1.96 on 32% of streams at a
        key's second row and 8% to 9% at its tenth, at ``lags=0``, and on
        72% and 21% to 24% at ``lags=4``.
    ``name``
        With ``half_life``, the name of the record column added,
        ``"diebold_mariano"`` by default.

    .. rubric:: Returns

    Without ``half_life``, a ``DataFrame`` with one row per ``group`` key
    and target: ``n``, the rows read; ``mean``, the mean differential;
    ``dm``, the statistic, null where every differential is the same value;
    and ``p_value``, its two-sided p-value. A
    ``dm`` above 1.96 says ``a`` predicted better at the 5% level, and one
    below -1.96 says ``b`` did.

    With ``half_life``, ``df`` with a record column ``name`` added, holding
    ``dm_<t>`` per target ``t``: null on a row where a side has no
    residual, on a key's rows before its ``min_samples``-th with a
    differential and while it has at most ``4 (lags + 1)`` of them (see
    ``lags``), and where ``S`` is zero but for rounding (see Notes); it has
    a value on every other row. Given a ``LazyFrame``, a ``LazyFrame`` of
    the same, which reads ``df`` one chunk at a time when it runs.

    Each row's value is about ``N(0, 1)`` when the two predict equally
    well, once its weight covers many rows. It rests on about ``2.9 ·
    half_life`` rows however long the stream (``(1 + λ) / (1 - λ)``, with
    ``λ`` the weight's decay a row), so keep ``half_life`` well above
    ``lags`` and above the differentials' memory. On simulated noise it
    passed 1.96 on 8% of rows at ``half_life=5`` (17% at ``lags=4``), on
    5% at ``half_life=20`` (8% at ``lags=4``), and on 5% at
    ``half_life=100`` (5% to 6% at ``lags=4``). It is one test per row, not a sequential test:
    acting the first time it passes 1.96 is not a test at 5%. Over 2,000
    rows of noise some row past the 30th passed 1.96 on 96% of streams at
    ``half_life=100`` and 65% at ``half_life=500``. :func:`seqtest` is the
    test to act on as the rows come.

    .. rubric:: Raises

    Whatever :func:`seqtest` raises for ``a``, ``b``, ``targets``, the
    suffixes and ``group``. ``ValueError`` too, for ``lags`` below 0, for a
    ``half_life`` that is not finite and above 0, or so long that ``0.5 **
    (1 / half_life)`` is 1.0 in a double (from about 1.25e16 rows), and,
    with ``half_life``, for a ``name`` that ``df`` already has as a
    column, which the result would replace. :class:`~polars_online.WrongTypeError`, both a
    ``TypeError`` and a ``ValueError``, for a ``lags`` that is not a whole
    number, such as ``1.5``.

    .. rubric:: Update equation

    Per target ``t``, the loss differential on a row is ``d = resid_b² -
    resid_a²``, over the ``n`` rows where both have a residual. With ``d̄``
    its mean, and Bartlett weights ``1 - l / (lags + 1)`` on lags ``l`` up
    to ``lags``:

    .. code-block:: text

        S  = Σ (d - d̄)² + 2 Σ_{l ≤ lags} (1 - l/(lags+1)) Σ_i (d_i - d̄)(d_{i-l} - d̄)
        dm = d̄ · n / sqrt(S)                  ~ N(0, 1) with equal accuracy

    ``S`` is Newey and West's estimate of the differential's long-run
    variance, times ``n``. ``i`` counts the rows that have a differential:
    a row without one is left out before the lags are taken, so
    ``d_(i-l)`` is the ``l``-th earlier row that has one, not the row ``l``
    places back.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        x0_only = po.spec.ewridge("x0_only", targets=["y"], features=["x0"], half_life=500.0)
        x1_only = po.spec.ewridge("x1_only", targets=["y"], features=["x1"], half_life=500.0)
        out = po.ModelBank([x0_only, x1_only]).fit_predict(df)
        test = po.eval.diebold_mariano(out, a="x0_only", b="x1_only")
        n, dm, p = test.select("n", "dm", "p_value").row(0)
        print(f"{n} rows, dm {dm:.2f}, p {p:.2g}")   # dm above 1.96: x0_only did better, at 5%

    .. code-block:: text

        397 rows, dm 8.62, p 6.7e-18

    .. rubric:: See also

    - :func:`clark_west`, the test for two nested specs.
    - :func:`seqtest`, evidence to read at every row, of which spec is
      closer more often.
    - :func:`compare_specs`, whose Notes say when to use each test.

    .. rubric:: Notes

    **This test asks which spec has the smaller mean squared error;**
    :func:`seqtest` **asks which is closer on more rows.** A spec with rare
    large errors can win the second and lose the first. On a simulated
    stream of 4,999 rows where ``a`` was closer on 3,415 rows but its mean
    squared error was 4.80 against ``b``'s 1.01, :func:`seqtest`'s log
    evidence that ``a`` was closer reached 339 (``ln 20`` is 3.0), and this
    test read ``dm`` -8.7, ``b`` better at a p-value of 4e-18. Each answer
    is right for its question.

    **Given a query, this holds one chunk and, per key and target, the
    last** ``lags`` **differentials and a few sums, beside what Polars
    reads of the query ahead of it.** Without
    ``half_life``, each lag's sum in ``S`` is a co-moment about ``d̄`` of the
    pairs of rows ``l`` apart, and with ``x`` the later of a pair and ``y``
    the earlier, ``Σ (x - d̄)(y - d̄) = C + n (x̄ - d̄)(ȳ - d̄)``, where ``C``
    is the pairs' co-moment about their own means. So each chunk keeps, per
    key and lag, the pairs' count, means and co-moment, added chunk to
    chunk as :func:`merge_sums` adds its sums, and the last ``lags``
    differentials of each key, which pair with the next chunk's first rows.
    The result is the frame's within rounding: within 1e-9 relative, and
    1e-12 at the most measured, on a mean differential near zero read a
    row at a time.

    A scan of a parquet file reads its row groups ahead of a slower
    consumer, as many as Polars' thread count sizes, and under Polars'
    defaults that read-ahead is most of what the process holds: `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds it. Measured on macOS (an M4 Pro, py-polars 2.0.0) as the most
    memory the process held, about 48 MiB of it Python and Polars before
    the call, testing two specs of a scan, a file in row groups of 100,000
    rows, by a key of four values with ``lags=5``: 97 MiB at a million
    rows, 97 MiB at four million and 101 MiB at sixteen million with four
    threads (``POLARS_MAX_THREADS=4``), the read-ahead cut to one row group
    (``POLARS_ROW_GROUP_PREFETCH_SIZE=1``) and the allocator told to return
    freed pages at once (``_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0``),
    the bounded settings; under Polars' defaults, 247, 409 and 793 MiB.
    The collected frame's test held 252 MiB at a million rows under the
    bounded settings.

    With ``half_life``, ``S`` is read from weighted means and variances (see
    below), each over the rows ``j ≤ t`` at weights ``μ^(t-j)``, with ``μ``
    the weight's decay a row or its square, so a chunk carries in each
    key's count, means and variances after the rows before it, decays
    their weight over its own rows, and pools them with its own as two
    groups' means and variances pool. While the query is one chunk the
    result is the frame's to the bit. Over many, frame and query each
    carry the rounding the next paragraph bounds, about ``E / (2 S)`` of
    ``dm``, so they are within about ``E / S`` of each other: near 1e-14
    where the differentials' level is near their spread, and about 1e-10
    at a level a million spreads from zero (7.9e-11 measured at
    ``half_life=5`` in chunks of 7 rows). A row whose ``S`` is within its
    rounding of ``E`` can be null in one and not the other. Written to a
    file with ``sink_parquet``, the test by the row with ``lags=2`` and
    ``half_life=500`` held 157 MiB at a million rows, 161 MiB at four
    million and 161 MiB at sixteen million under the bounded settings, and
    395, 681 and 1,129 MiB under Polars' defaults.

    **With** ``half_life``, ``S`` **is summed about the weighted means, and**
    ``dm`` **is null where** ``S`` **is zero but for rounding.** Each row's
    weight is ``ω``, ``d̄`` is the mean at those weights, and ``ω_t ω_(t-l)
    = λ^l ω_t²``, so each sum in ``S``
    is a sum at the weights ``ω²``: with ``W = Σ ω²``, and ``m₂`` and
    ``v₂`` the mean and variance of ``d`` at those weights, lag 0's is ``W
    (v₂ + (m₂ - d̄)²)``, and lag ``l``'s, over the pairs ``(u, v) = (d_i,
    d_(i-l))``, is ``λ^l W_l (c_l + (μ_l - d̄)(ν_l - d̄))``, with ``W_l``,
    ``μ_l``, ``ν_l`` and ``c_l`` the pairs' weight, means and covariance.
    No sum cancels the differentials' level against itself, so a spread
    far below the level, or one first differential far from the rest, is
    read to the digits the means hold. Each mean and variance is a
    recursion that steps by ``1 / weight`` of the gap to the new value, its
    weight the sum of its weights so far, at most the rows read and at most
    ``Σ ω``. Each step rounds by about ``ε = 2.2e-16`` of the value held,
    and the next keeps ``1 - 1 / weight`` of that, so the roundings rest
    within ``η_r = ε Σ ω`` of its size; a mean of a constant can rest that
    far from it, since a step smaller than half its last digit rounds to
    nothing. A key with few rows has gathered few roundings, whatever the
    half-life. The totals ``Σ ω²`` are read from the count as ``(1 -
    λ^(2n)) / (1 - λ²)``, and ``λ^(2n)`` rounds to within ``ε``, so near
    ``λ = 1`` they are within ``η_w = ε / ((1 - λ²) Σ ω²)``. From there,
    with ``S₀`` lag 0's sum and ``R = Σ ω² d²``, ``S`` is within ``E = 10
    (1 + lags) ((η_r + η_w) S₀ + η_r sqrt(S₀ R) + η_r² R)`` of its value:
    ``η_r²`` of ``R`` where the differentials have been one value (the
    means' offsets squared), and ``η_r + η_w`` of the sums where they have
    not. ``dm`` is null where ``S <= E``: on rows whose differentials have
    all been equal, and once a constant has run long enough that the rows
    before it weigh less than its rounding. Above it, ``dm``'s rounding is
    at most about ``E / (2 S)`` of it, and the mean's, about ``η_r`` of the
    differentials' size. At a half-life near the longest a double can tell
    from no decay, about 1e16 rows, ``η_w`` nears ``1 / n`` and nulls a
    key's first rows, a few dozen at ``lags=3``, though its statistic is
    right there to 1e-12.

    Without ``half_life``, ``dm`` is null where every differential of the
    key is the same value, whose ``S`` is zero but for the rounding of the
    mean. A key whose differentials take values a few units in the last
    place apart reads a statistic only as good as the rounding of that
    spread, 1e16 or more and not the same in a frame and a query, with a
    p-value of 0 either way.
    """
    _frame("po.eval.diebold_mariano", df, lazy=True)
    probe = _probe(df)
    _check_columns(probe, "po.eval.diebold_mariano", group=group)
    keys = _group_keys(group)
    pairs = _paired_residuals(probe, a, b, targets, a_suffix, b_suffix, "po.eval.diebold_mariano")
    values = {t: rb * rb - ra * ra for t, (ra, rb) in pairs.items()}
    lags = _lags("po.eval.diebold_mariano", lags)
    if half_life is not None:
        _check_half_life("po.eval.diebold_mariano", half_life)
        _check_name("po.eval.diebold_mariano", probe, name)
        return _ew_test(df, values, keys, lags, half_life, "dm", name, min_samples)
    out = _hac_test(df, values, keys, lags, "dm", min_samples)
    p = [
        None if z is None or math.isnan(z) else math.erfc(abs(z) / math.sqrt(2.0))
        for z in out["dm"].to_list()
    ]
    return out.with_columns(pl.Series("p_value", p, dtype=pl.Float64))


@overload
def clark_west(
    df: pl.DataFrame,
    *,
    big: str,
    small: str,
    targets: Sequence[str] | None = None,
    big_suffix: str = "",
    small_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "clark_west",
) -> pl.DataFrame: ...


@overload
def clark_west(
    df: pl.LazyFrame,
    *,
    big: str,
    small: str,
    targets: Sequence[str] | None = None,
    big_suffix: str = "",
    small_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: None = None,
    min_samples: int = 30,
    name: str = "clark_west",
) -> pl.DataFrame: ...


@overload
def clark_west(
    df: pl.LazyFrame,
    *,
    big: str,
    small: str,
    targets: Sequence[str] | None = None,
    big_suffix: str = "",
    small_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: float,
    min_samples: int = 30,
    name: str = "clark_west",
) -> pl.LazyFrame: ...


@renamed_keywords("po.eval.clark_west", _GROUP)
def clark_west(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    big: str,
    small: str,
    targets: Sequence[str] | None = None,
    big_suffix: str = "",
    small_suffix: str = "",
    lags: int = 0,
    group: str | Iterable[str] | None = None,
    half_life: float | None = None,
    min_samples: int = 30,
    name: str = "clark_west",
) -> pl.DataFrame | pl.LazyFrame:
    """Test whether a larger spec predicts better than a smaller one nested in
    it, by Clark and West's test.

    Use it when one spec's features are a subset of the other's, such as a
    spec with and without one more feature. It is Clark and West's (2007)
    test, with Newey and West's variance. For two specs that are not
    nested, use :func:`diebold_mariano`. For evidence you may read at every
    row as the rows come, use :func:`seqtest`, which asks which spec is
    closer on more rows, not which has the smaller mean squared error.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding both specs' output columns, or a
        query of one, a ``LazyFrame``, which this reads one chunk at a time
        as :func:`diebold_mariano` reads one.
    ``big``, ``small``
        The names of the larger and the smaller spec's output columns, as
        :func:`diebold_mariano`'s ``a`` and ``b``.
    ``targets``
        The targets whose residuals are compared; every target the two
        share by default.
    ``big_suffix``, ``small_suffix``
        The suffix of each side's residual field, as
        :func:`diebold_mariano`'s ``a_suffix`` and ``b_suffix``; ``""`` by
        default.
    ``lags``, ``group``, ``half_life``, ``min_samples``
        :func:`diebold_mariano`'s own.
    ``name``
        With ``half_life``, the name of the record column added,
        ``"clark_west"`` by default.

    .. rubric:: Returns

    Without ``half_life``, a ``DataFrame`` with one row per ``group`` key
    and target: ``n``, the rows read; ``mean``, the mean of ``f`` (see
    Update equation); ``cw``, the statistic; and ``p_value``, its one-sided p-value. With
    ``half_life``, ``df`` with a record column ``name`` added, holding
    ``cw_<t>`` per target ``t``, the exponentially weighted statistic,
    null on the rows :func:`diebold_mariano`'s ``dm_<t>`` is and read as
    its Returns say, one test per row; given a ``LazyFrame``, a
    ``LazyFrame`` of the same.

    .. rubric:: Raises

    Whatever :func:`diebold_mariano` raises, with ``big`` and ``small`` in
    place of ``a`` and ``b``.

    .. rubric:: Update equation

    Per target, over the ``n`` rows where both have a residual:

    .. code-block:: text

        f  = resid_small² - (resid_big² - (pred_small - pred_big)²)
           = resid_small² - resid_big² + (resid_big - resid_small)²
        cw = f̄ · n / sqrt(S)          S as in diebold_mariano, of f

    ``f̄`` is the mean of ``f``. ``cw`` is about ``N(0, 1)`` under the null
    that the smaller spec is the truth, and the test is one-sided:
    ``cw > 1.645`` rejects the null at 5%. ``pred_small - pred_big`` equals
    ``resid_big - resid_small``, so the two residual fields are all the test
    reads.

    .. rubric:: Examples

    This code reads ``df`` from the README's `Example data
    <https://github.com/hgilde/polars-online#example-data>`_:

    .. code-block:: python

        small = po.spec.ewridge("small", targets=["y"], features=["x0"], half_life=500.0)
        big = po.spec.ewridge("big", targets=["y"], features=["x0", "x1"], half_life=500.0)
        out = po.ModelBank([small, big]).fit_predict(df)
        test = po.eval.clark_west(out, big="big", small="small")
        n, cw, p = test.select("n", "cw", "p_value").row(0)
        print(f"{n} rows, cw {cw:.2f}, p {p:.2g}")   # cw above 1.645: x1 helps, at 5%

    .. code-block:: text

        396 rows, cw 12.24, p 9.3e-35

    .. rubric:: See also

    - :func:`diebold_mariano`, the test for two specs that are not nested.
    - :func:`compare_specs`, whose Notes say when to use each test.

    .. rubric:: Notes

    Diebold and Mariano's differential leans toward the smaller of two
    nested specs. Under the null, the larger spec's extra coefficients are
    zero but estimated, and their noise adds ``(pred_small - pred_big)²``
    to its squared error on average. Clark and West add that term back.

    ``cw`` is null where :func:`diebold_mariano`'s ``dm`` is, of ``f``:
    where every ``f`` of a key is the same value, on a key with at most
    ``4 (lags + 1)`` rows with an ``f``, and, with ``half_life``, before a
    key's ``min_samples``-th row and where ``S`` is zero but for rounding.
    Its ``lags`` are chosen as :func:`diebold_mariano`'s are: 0 only where
    ``f`` is uncorrelated from row to row, and about twice the horizon for
    a target that looks ahead.

    Given a query, this holds and computes what :func:`diebold_mariano`
    does for one, of ``f`` in place of ``d``, with the same agreement with
    the frame, beside what Polars reads of the query ahead of it, which
    `Cut what Polars reads ahead
    <https://github.com/hgilde/polars-online/blob/main/docs/how-to/bound-memory.md#cut-what-polars-reads-ahead>`__
    bounds. Scoring a scan of two specs, a file in row groups of 100,000
    rows, by a key of four values with ``lags=5`` held at most 97 MiB at a
    million rows, 100 MiB at four million and 100 MiB at sixteen million
    with four threads, the read-ahead cut to one row group
    (``POLARS_ROW_GROUP_PREFETCH_SIZE=1``) and the allocator told to return
    freed pages at once, and 241, 425 and 787 MiB under Polars' defaults,
    measured as :func:`diebold_mariano`'s Notes say.
    """
    _frame("po.eval.clark_west", df, lazy=True)
    probe = _probe(df)
    _check_columns(probe, "po.eval.clark_west", group=group)
    keys = _group_keys(group)
    pairs = _paired_residuals(
        probe, big, small, targets, big_suffix, small_suffix, "po.eval.clark_west"
    )
    values = {t: rs * rs - rb * rb + (rb - rs) * (rb - rs) for t, (rb, rs) in pairs.items()}
    lags = _lags("po.eval.clark_west", lags)
    if half_life is not None:
        _check_half_life("po.eval.clark_west", half_life)
        _check_name("po.eval.clark_west", probe, name)
        return _ew_test(df, values, keys, lags, half_life, "cw", name, min_samples)
    out = _hac_test(df, values, keys, lags, "cw", min_samples)
    return out.with_columns(
        pl.Series("p_value", [_normal_sf(z) for z in out["cw"].to_list()], dtype=pl.Float64)
    )
