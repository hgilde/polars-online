"""Scores of a bank's output, and tests of one spec's residuals against another's.

The scores are out-of-sample R², IC (the correlation of the prediction with
the target) and hit rate per spec, group and target,
over a whole frame or in windows of the clock. :func:`sums` gives the same
scores one chunk at a time, and :func:`grid_search` chooses a grid point from
a stream's own residuals.

Most calls read the frame a :class:`~polars_online.ModelBank` returns: the
input's columns plus one output column per spec, whose value in each row is
a record of the model's fields. :func:`merge_sums` and :func:`from_sums` read
the frames :func:`sums` makes. :func:`seqtest` can also test the sign of
plain numeric columns, and :func:`grid_search` runs a bank over a query
itself. The calls
fall into five kinds:

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - kind
     - the calls
     - what they hold in memory
   * - scores over a frame
     - :func:`metrics`, :func:`window_metrics`, :func:`compare_specs`
     - the whole frame
   * - scores one chunk at a time
     - :func:`sums`, :func:`merge_sums`, :func:`from_sums`
     - ten numbers per key
   * - one spec against another
     - :func:`diebold_mariano`, :func:`clark_west`, :func:`seqtest`
     - the whole frame
   * - the long form the scores read
     - :func:`unpack`
     - the whole frame
   * - a grid point chosen from a stream's own residuals
     - :func:`grid_search`
     - three numbers per slot for each run of rows between the places a
       test span starts or its choice stops,
       and the bank's state and one chunk

The scores over a frame are Polars aggregations, so they need the whole
frame. For a stream too long to hold, set a spec's ``emit_metrics``, which
keeps exponentially weighted R², IC and hit rate beside the model, in memory
the size of its state. Or
reduce each chunk with :func:`sums`, add the chunks with :func:`merge_sums`
and read the scores with :func:`from_sums`.

**Pass the spec itself as ``spec`` where a target is not a plain column.**
``spec`` is the spec's name, which names its output column, or the spec
dict, which names the column too. Only the dict can say where a target's
values are when they are not in a column of the target's name: a column
renamed with :func:`polars_online.target`, or a formula target (see
:func:`unpack`). The dict also says whether a fit is a classifier, whose
scores take other names (:func:`metrics`' ``binary``).

Every call here but :func:`grid_search` reads a whole ``DataFrame``. Given a
``LazyFrame``, it raises ``TypeError``: pass ``lf.collect()``, which runs the
query and returns its result. The scores
start from :func:`unpack` and raise what it raises. It raises a
:class:`~polars_online.OutputColumnNotFoundError` (a ``KeyError`` too) for a
spec whose output column the frame does not have, ``TypeError`` for a column
that is not a spec's output or is the output of a model that predicts no
target. A slot whose target column cannot be found raises a
:class:`~polars_online.ColumnNotFoundError` when ``spec`` is the spec dict,
and ``ValueError`` when it is the spec's name alone. A ``group``, ``clock`` or
``weight`` column the frame does not have raises a
:class:`~polars_online.ColumnNotFoundError` too.

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
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal

import polars as pl

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


def _frame(who: str, df: object) -> None:
    """Refuse anything but a ``DataFrame`` here, where every call reads the
    whole frame: a ``LazyFrame`` once reached Polars' own calls, which warned
    three times and then failed on ``LazyFrame is not subscriptable``."""
    if isinstance(df, pl.DataFrame):
        return
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


def _takes_spec[R](who: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """A decorator for a call whose ``spec_name`` and ``spec=`` became one
    ``spec``, the name or the dict: the old forms, ``spec_name=`` and a name
    beside ``spec=``, are read as the new one, with a
    :class:`~polars_online.PolarsOnlineDeprecationWarning`, and a frame
    that is not a ``DataFrame`` is refused by name."""

    def decorate(fn: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(fn)
        def wrapper(df: Any, *args: Any, **kwargs: Any) -> R:
            _frame(who, df)
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


@_takes_spec("po.eval.unpack")
def unpack(
    df: pl.DataFrame,
    spec: str | dict,
    *,
    targets: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Turn a spec's output into long form, one row per input row and slot.

    Use it to score or plot predictions with your own Polars code: each row
    holds one prediction beside the value of the target it predicts. For the
    standard scores, use :func:`metrics`, which reads this long form itself.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned: the input's columns and the spec's output
        column. A ``LazyFrame`` is refused, since every call here reads the
        whole frame: pass ``lf.collect()``, which runs the query and returns
        its result.
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
        of ``df`` by default. Not read when ``spec`` is the dict.

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

    .. rubric:: Raises

    ``TypeError``
        ``df`` is not a ``DataFrame``, such as a ``LazyFrame``; or the
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
    return _unpack(df, spec, targets=targets, who="po.eval.unpack")


def _unpack(
    df: pl.DataFrame,
    spec: str | dict,
    *,
    targets: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    who: str = "po.eval.unpack",
) -> pl.DataFrame:
    """:func:`unpack`, carrying of the input's columns only those in
    ``columns`` (all of them when ``None``). A score reads the key columns
    and the clock or weight beside ``pred`` and ``y``; carrying every input
    column once per slot made its long form the frame's width times the
    slots, 1.4 GiB from a 290 MiB frame of 43 columns and 8 slots, where
    the four columns it reads take 137 MiB."""
    spec_name, written = _spec_of(who, spec)
    fields = _pred_fields(who, df, spec_name)
    keep = [c for c, d in df.schema.items() if not isinstance(d, pl.Struct) and c not in RESERVED]
    if columns is not None:
        wanted = set(columns)
        keep = [c for c in keep if c in wanted]
    # With the spec in hand, the slot -> target mapping comes from
    # `output_index` -- the same Rust code that named the fields -- and each
    # target's value from the spec's own table of it, instead of the
    # heuristic name-parsing below, which exists only for callers who have a
    # frame but no spec.
    exact = {} if written is None else _spec_targets(written, df, who)
    frames = []
    for slot in fields:
        if slot in exact:
            target, y = exact[slot]
        else:
            target = _target_of(who, slot, df, targets)
            y = pl.col(target)
        frames.append(
            df.select(
                *keep,
                pl.lit(slot).alias("slot"),
                pl.lit(target).alias("target"),
                df[spec_name].struct.field(slot).alias("pred"),
                y.alias("y"),
            )
        )
    return pl.concat(frames)


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


def _check_columns(df: pl.DataFrame, who: str, **roles: str | Iterable[str] | None) -> None:
    """Each column a score reads beside a spec's output, by its role
    (``group``, ``clock``, ``weight``), checked to be in ``df``: a missing
    one is :class:`~polars_online.ColumnNotFoundError`, a ``ValueError``
    too, as a bank's refusal of a missing column is."""
    for role, value in roles.items():
        for column in _group_keys(value):
            if column not in df.schema:
                msg = (
                    f"{who}: {role} column {column!r} is not in the frame, whose columns are "
                    f"{df.columns}. Name a column of the frame as {role}."
                )
                raise ColumnNotFoundError(msg)


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
    ``hit_rate``."""
    return long.filter(
        _usable(pl.col("pred").cast(pl.Float64)) & _usable(pl.col("y").cast(pl.Float64))
    )


def _no_hit(spec: dict | None) -> bool:
    """Whether ``spec`` names a fit with no sign to hit, whose ``hit_rate``
    the bank's ``emit_metrics`` nulls: a Poisson ``sgd`` (``loss =
    "poisson"``), where a rate is positive and a count never negative, so
    the sign test about zero agrees on every row and reads 1.0 whatever the
    fit. Without ``spec``
    the loss cannot be known, and the sign test is read."""
    model = (spec or {}).get("model") or {}
    return model.get("type") == "sgd" and model.get("loss") == "poisson"


def _metric_exprs(*, binary: bool = False, hit: bool = True) -> list[pl.Expr]:
    resid = pl.col("y") - pl.col("pred")
    # The centred sums, as :func:`sums` keeps them: a metric is null where
    # one it divides by is 0, as :func:`from_sums` gives it, where this gave
    # -inf and NaN (review 2026-10-05, YB9).
    m2_y = (pl.col("y") - pl.col("y").mean()).pow(2).sum()
    m2_pred = (pl.col("pred") - pl.col("pred").mean()).pow(2).sum()
    exprs = [
        # Every count an Int64, as in every frame of the library (task 244).
        pl.len().cast(pl.Int64).alias("n"),
        # Out-of-sample R^2 against the realized mean of y in the window. When
        # `binary` (`pred` a probability, `y` a 0/1 label) the same formula
        # is the Brier skill score, and is named so (task 244, M-19).
        pl.when(m2_y > 0)
        .then(1.0 - resid.pow(2).sum() / m2_y)
        .alias("brier_skill" if binary else "r2"),
        # Correlation of prediction with target -- the point-biserial
        # correlation when `binary`, again the same formula.
        pl.when(m2_y * m2_pred > 0).then(pl.corr("pred", "y")).alias("ic"),
    ]
    if binary:
        # Accuracy at a 0.5 threshold. Every row scores: 0 is one of the two
        # classes here, not the sign test's excluded case, and `pred` and `y`
        # are both positive by construction, so the sign test below always
        # agrees and is not a hit rate at all on this kind of fit (task 76
        # found it reading exactly 1.0 on a fit that had learned nothing).
        exprs.append(
            (((pl.col("pred") > 0.5) == (pl.col("y") > 0.5)).sum() / pl.len()).alias("accuracy")
        )
        # Log loss: -(y*ln(p) + (1-y)*ln(1-p)), mean over the window. Lives
        # here, over a collected frame, rather than in the streaming
        # `emit_metrics` -- an EW accumulator would put a `ln` result into
        # model state, which `docs/PLAN.md` §11a's B4 rule forbids (glibc's
        # last bit is not Apple's, and that difference broke a frozen state
        # fixture on Linux alone). Clipped so a confident, correct prediction
        # is a strong negative number rather than -inf.
        p = pl.col("pred").clip(1e-15, 1.0 - 1e-15)
        exprs.append(
            (-(pl.col("y") * p.log() + (1.0 - pl.col("y")) * (1.0 - p).log()))
            .mean()
            .alias("log_loss")
        )
    elif not hit:
        # A fit with no sign to hit (`_no_hit`): null, as the bank nulls it.
        exprs.append(pl.lit(None, dtype=pl.Float64).alias("hit_rate"))
    else:
        # Fraction of rows where the sign of pred matches the sign of y
        # (rows with y == 0 or pred == 0 excluded: neither up nor down, not a
        # class), and null where no row has a sign. A prediction of exactly
        # zero is left out as the bank leaves it out (docs/PLAN.md task 195,
        # S6): polars' `sign(0)` is 0, a miss here, where the bank's
        # `signum(+0.0)` was 1, a hit.
        y, pred = pl.col("y"), pl.col("pred")
        scored = (y != 0) & (pred != 0)
        signed = scored.sum()
        hits = ((pred.sign() == y.sign()) & scored).sum()
        exprs.append(pl.when(signed > 0).then(hits.truediv(signed)).alias("hit_rate"))
    exprs.append(resid.pow(2).mean().alias("mse"))
    exprs.append(resid.pow(2).mean().sqrt().alias("rmse"))
    return exprs


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
@_takes_spec("po.eval.metrics")
def metrics(
    df: pl.DataFrame,
    spec: str | dict,
    *,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    min_samples: int = 30,
    binary: bool | None = None,
) -> pl.DataFrame:
    """Score a spec's out-of-sample predictions over a whole frame.

    Use it to read how well a spec predicted, per slot, target and group,
    once its output is in memory. For scores in windows of the clock, use
    :func:`window_metrics`; for several specs in one table,
    :func:`compare_specs`; for a stream too long to hold, :func:`sums`, or
    the spec's own ``emit_metrics``.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding the spec's output column. A
        ``LazyFrame`` is refused: pass ``lf.collect()``, which runs the query
        and returns its result.
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
        reads the target's values to decide.

    .. rubric:: Returns

    A ``DataFrame`` with one row per ``slot``, ``target`` and ``group``
    key, sorted by them, and these columns after the keys:

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

    Whatever :func:`unpack` raises for ``df`` and ``spec``, and
    :class:`~polars_online.ColumnNotFoundError`, both Polars'
    ``ColumnNotFoundError`` and a ``ValueError``, for a ``group`` column
    ``df`` does not have.

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
    spec_name, written = _spec_of("po.eval.metrics", spec)
    _check_columns(df, "po.eval.metrics", group=group)
    binary = _binary(binary, written)
    keys = ["slot", "target", *_group_keys(group)]
    long = _scored(
        _unpack(df, written or spec_name, targets=targets, columns=keys, who="po.eval.metrics")
    )
    exprs = _metric_exprs(binary=binary, hit=not _no_hit(written))
    out = long.group_by(keys).agg(exprs).sort(keys)
    return _thin(out, min_samples, keys)


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
@_takes_spec("po.eval.window_metrics")
def window_metrics(
    df: pl.DataFrame,
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
        clock column. A ``LazyFrame`` is refused: pass ``lf.collect()``, which
        runs the query and returns its result.
    ``spec``
        The spec's name, or the spec dict itself, as :func:`metrics` reads
        it.
    ``clock``
        The column whose values place each row in a window: a number, a
        ``Datetime``, a ``Date`` or a ``Duration``. The rows whose clock is
        null are scored together, in a row of their own whose
        ``window_start`` is null. A ``Time`` is refused, since it starts
        again at midnight.
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
        quarter or a year on a ``Duration`` clock.
    :class:`~polars_online.WrongTypeError`
        The ``clock`` column is neither numeric nor temporal, or is a
        ``Time``, which starts again at midnight. It is both a ``TypeError``
        and a ``ValueError``, as a bank's refusal of the same clock is.
    :class:`~polars_online.ColumnNotFoundError`
        ``df`` does not have the ``clock`` column or a ``group`` column. It
        is both Polars' ``ColumnNotFoundError`` and a ``ValueError``.

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
      ``floor(clock / every) * every``, as Polars' integer index does. A float clock, which Polars
      does not window, is windowed the same way.
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
    """
    spec_name, written = _spec_of("po.eval.window_metrics", spec)
    dtype = df.schema.get(clock)
    if dtype is None:
        # Named before the window is checked against it (review R2, P9).
        msg = (
            f"po.eval.window_metrics: clock column {clock!r} is not in the frame, whose "
            f"columns are {df.columns}. Name a column of the frame as clock."
        )
        raise ColumnNotFoundError(msg)
    _check_columns(df, "po.eval.window_metrics", group=group)
    # The clock's dtype before `every` is read against it: a String clock was
    # refused for the window, not for itself (task 352).
    if not (dtype.is_numeric() or dtype.is_temporal()):
        msg = (
            f"po.eval.window_metrics: clock column {clock!r} must be numeric or temporal, but "
            f"it is {dtype}. Cast the clock to a number or a Datetime."
        )
        raise WrongTypeError(msg)
    ns: int | None
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
    long = _scored(
        _unpack(
            df,
            written or spec_name,
            targets=targets,
            columns=[*keys, clock],
            who="po.eval.window_metrics",
        )
    )
    if start is None:
        long = _dynamic_windows(long, clock, _polars_every(every, ns), keys[:-1])
    else:
        long = long.with_columns(start.alias("window_start"))
    exprs = _metric_exprs(binary=_binary(binary, written), hit=not _no_hit(written))
    out = long.group_by(keys).agg(exprs).sort(keys)
    return _thin(out, min_samples, keys)


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
        _frame(who, df)
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
    df: pl.DataFrame,
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
        A frame a bank returned, holding each spec's output column. A
        ``LazyFrame`` is refused: pass ``lf.collect()``, which runs the query
        and returns its result.
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

    Whatever :func:`metrics` raises for each spec, and ``ValueError`` when
    ``binary`` is ``None`` and some specs are fits with
    ``loss="logistic"`` and others are not. Pass the two sets in two calls,
    or set ``binary``.

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
    - :func:`seqtest`, evidence that one spec predicts closer, to read at
      any row.
    - `Comparing two models
      <https://github.com/hgilde/polars-online/blob/main/docs/DIAGNOSTICS.md#comparing-two-models>`_,
      with a recipe that compares two models on 200 streams.

    .. rubric:: Notes

    Each spec is scored on its own rows, so two specs whose warm-ups differ
    in length are scored on different rows, as the 396 and 397 rows in the
    example are. Diebold and Mariano's test and Clark and West's read only
    the rows where both specs have a residual.

    **Diebold and Mariano's (1995) test asks whether two specs' squared
    errors differ on average.** Its statistic is the mean of the loss
    differential ``d = resid_b² − resid_a²`` over Newey and West's
    standard error, and is about ``N(0, 1)`` when the two predict equally
    well. :func:`diebold_mariano` computes it. Give its ``lags`` at least
    ``h − 1`` for a target that looks ``h`` rows ahead, and 0 for one-step
    predictions.

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
    that one spec predicts closer, by betting on the sign of ``|resid_b| −
    |resid_a|``. The evidence can be read at any row, as often as you like,
    without increasing the false-alarm rate: reject at the 5% level the first
    time ``log_e_a_<t>`` reaches ``ln 20``.
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
    frames = [
        metrics(
            df,
            written or name,
            group=group,
            targets=targets,
            min_samples=min_samples,
            binary=binary,
        ).with_columns(pl.lit(name).alias("spec"))
        for name, written in given
    ]
    cols = ["spec", *frames[0].columns[:-1]] if frames else []
    return pl.concat(frames).select(cols) if frames else pl.DataFrame()


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
    return slots


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
        The number of test spans, 5 by default. The rows are cut as
        scikit-learn's ``TimeSeriesSplit(n_splits, gap=gap)`` cuts them:
        the ``n`` rows into ``n_splits + 1`` spans, the last ``n_splits`` of
        ``n // (n_splits + 1)`` rows each. Fewer spans make the first span,
        the rows the first choice is made from, longer.
    ``gap``
        The rows left out between the rows a choice is made from and the
        span it is scored on, 0 by default. Give it the number of rows a
        target looks ahead, so a choice never reads a row whose target was
        not yet known when its span started.
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

    The choice's row is the out-of-sample score of choosing: what choosing
    the best grid point from the rows before each span scored on that span.
    How far it falls short of the rank-1 score is how much choosing on the
    same rows would have overstated the score.

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
        ``n_splits`` or ``gap`` is not a whole number. It is both a
        ``TypeError`` and a ``ValueError``.
    :class:`~polars_online.ColumnNotFoundError`
        A spec's target has no column in the stream. A formula target never
        has one, since the bank computes it as it learns and does not write
        it out, so such a spec cannot be scored here. It is both a
        ``ValueError`` and Polars' ``ColumnNotFoundError``.
    ``ValueError``
        When:

        - ``scoring`` is neither word;
        - ``n_splits`` is below 1, or ``gap`` below 0;
        - a builder returns a spec not named by its first argument;
        - a ``param_grid`` key is one of the result's columns;
        - a spec has nothing to score: no prediction of a target under
          ``"mse"``, or no ``loglik`` under ``"loglik"``;
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
    - `Calibrating a model
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
    span, the slot with the best score over every row before it (``gap``
    rows short of it) is chosen, and scored on the span:

    .. code-block:: text

        size     = n // (n_splits + 1)
        start_k  = n - (n_splits - k) * size                 k = 0 .. n_splits - 1
        chosen_k = argbest over candidate slots s of score(s, rows [0, start_k - gap))
        choice   = score(chosen_k on rows [start_k, start_k + size), pooled over k)

    **What is scored.** The slots are every prediction a spec writes, one
    per target and grid point of its own grid (``half_life``, ``ridge``,
    ``feature_sets`` or ``penalties`` as lists), across all the specs. A
    row counts for a target only where every slot of that target has a
    value and the target is present. So the grid points are compared on
    the same rows, and the rows any slot withholds while it warms up count
    for none. Every row weighs alike. Under ``scoring="loglik"``, the sum
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
        for choose_end, start, end in folds:
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


@renamed_keywords("po.eval.seqtest", _GROUP)
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
) -> pl.DataFrame:
    """Test a sign sequentially by betting, over a frame in memory.

    Use it to gather evidence, row by row, that a column's sign leans one
    way, or that one spec predicts closer than another, with evidence you
    may read at any row. It computes what :func:`polars_online.spec.seqtest`
    writes, field for field and row for row, with Polars expressions. For a
    stream too long to hold, run the ``seqtest`` model in the bank, whose
    memory is the size of its state.

    .. rubric:: Parameters

    ``df``
        The frame: in column mode, one holding the columns tested; in
        compare mode, one a bank returned, holding both specs' output
        columns. A ``LazyFrame`` is refused: pass ``lf.collect()``, which runs
        the query and returns its result.
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
        ``df`` is not a ``DataFrame``, or ``a`` or ``b`` names a column that
        is not a record.
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

    The evidence can be read at any row. Each target has two bets, one on
    each direction: reject a direction at the 5% level the first time its
    bet's log wealth reaches ``ln 20``, about 3.0. Watching both bets at
    once, to test either direction at 5%, needs ``ln 40``. A value of zero bets
    nothing and counts nothing, and so does one the bank would not learn
    from: null, NaN, infinite, or past its input bound of 1e100 in
    magnitude. The bank's model adds what a frame in memory does not have,
    the clock (``session``, ``restart_after_step_back``), and its output is
    held to this function's to the last bit.
    """
    who = "po.eval.seqtest"
    _frame(who, df)
    _check_name("po.eval.seqtest", df, name)
    _check_columns(df, who, group=group)
    keys = _group_keys(group)
    if (a is None) != (b is None):
        msg = (
            "po.eval.seqtest: a and b go together, but only one is given. Name both specs to "
            "compare them, or neither."
        )
        raise ValueError(msg)
    if a is not None and b is not None:
        have = {a: _resid_fields(who, df, a), b: _resid_fields(who, df, b)}
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
    taken = {*df.columns, name}

    def temp(label: str) -> str:
        column = f"__po_seqtest_{label}"
        while column in taken:
            column = "_" + column
        taken.add(column)
        return column

    n_eff_col = temp("weight_sum")
    first: list[pl.Expr] = [over(pl.int_range(pl.len())).cast(pl.Float64).alias(n_eff_col)]
    staged: dict[str, tuple[str, str, str]] = {}
    for i, (t, d) in enumerate(signs.items()):
        # What the bank does not learn from -- null, NaN, an infinity, a
        # magnitude beyond its input bound (docs/PLAN.md section 3) -- is no
        # sign here either. Polars orders NaN above every float, so without
        # this `NaN > 0` would be a positive sign.
        d = d.cast(pl.Float64)
        d = pl.when(d.is_finite() & (d.abs() <= _INPUT_BOUND)).then(d)
        cols = (temp(f"s_{i}"), temp(f"n_pos_{i}"), temp(f"n_neg_{i}"))
        first += [
            pl.when(d > 0).then(1.0).when(d < 0).then(-1.0).otherwise(0.0).alias(cols[0]),
            before((d > 0).cast(pl.Int64).fill_null(0)).alias(cols[1]),
            before((d < 0).cast(pl.Int64).fill_null(0)).alias(cols[2]),
        ]
        staged[t] = cols
    ready = pl.col(n_eff_col) >= min_weight
    fields: list[pl.Expr] = []
    for t, (s_col, pos_col, neg_col) in staged.items():
        s, n_pos, n_neg = pl.col(s_col), pl.col(pos_col), pl.col(neg_col)
        n1 = (n_pos + n_neg + 1).cast(pl.Float64)
        lam_pos = pl.max_horizontal((n_pos - n_neg).cast(pl.Float64) / n1, 0.0)
        lam_neg = pl.max_horizontal((n_neg - n_pos).cast(pl.Float64) / n1, 0.0)
        log_e_pos = before((lam_pos * s).log1p())
        log_e_neg = before((-lam_neg * s).log1p())
        for label, e in zip(names, (log_e_pos, log_e_neg, n_pos, n_neg), strict=True):
            fields.append(pl.when(ready).then(e).alias(f"{label}_{t}"))
    fields.append(pl.col(n_eff_col).alias("weight_sum"))
    temps = [n_eff_col, *(c for cols in staged.values() for c in cols)]
    return df.with_columns(first).with_columns(pl.struct(fields).alias(name)).drop(temps)


@renamed_keywords("po.eval.sums", _GROUP)
@_takes_spec("po.eval.sums")
def sums(
    df: pl.DataFrame,
    spec: str | dict,
    *,
    group: str | Iterable[str] | None = None,
    targets: Sequence[str] | None = None,
    weight: str | None = None,
    binary: bool | None = None,
) -> pl.DataFrame:
    """Reduce one chunk of a bank's output to the sums its scores are read from.

    Use it to score a stream too long to hold: reduce each chunk to ten
    numbers per slot, target and group, add the chunks with
    :func:`merge_sums`, and read :func:`metrics`' scores with
    :func:`from_sums`. For a frame in memory, :func:`metrics` gives the
    same scores in one call.

    .. rubric:: Parameters

    ``df``
        One chunk a bank returned, holding the spec's output column. A
        ``LazyFrame`` is refused: pass ``lf.collect()``, which runs the query
        and returns its result.
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
        A row of weight 0 is kept, counted in ``n`` and not in ``w``.
    ``binary``
        :func:`metrics`' own: whether ``hits`` and ``signed`` count a
        classifier's accuracy at 0.5 or the sign test. Keep the sums of the
        two readings apart (see Returns).

    .. rubric:: Returns

    A ``DataFrame`` with one row per ``slot``, ``target`` and ``group`` key,
    and these columns after the keys, :data:`SUM_FIELDS`:

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
    ``weight`` column ``df`` does not have.

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
    - :func:`metrics`, the same scores over a frame in memory.

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
    """
    spec_name, written = _spec_of("po.eval.sums", spec)
    _check_columns(df, "po.eval.sums", group=group, weight=weight)
    binary = _binary(binary, written)
    columns = [*_group_keys(group), *([] if weight is None else [weight])]
    long = _scored(
        _unpack(df, written or spec_name, targets=targets, columns=columns, who="po.eval.sums")
    )
    if weight is not None:
        # What the bank does with the same weight: it skips a row whose
        # weight it cannot use and refuses a negative one (review round 4,
        # YB21: a null weight counted in `n` and not in `w`).
        wexpr = pl.col(weight).cast(pl.Float64)
        long = long.filter(_usable(wexpr) & (wexpr >= 0.0))
    else:
        wexpr = pl.lit(1.0)
    long = long.with_columns(wexpr.alias("__w"))
    w, y, p = pl.col("__w"), pl.col("y"), pl.col("pred")
    tw = w.sum()
    my, mp = (w * y).sum() / tw, (w * p).sum() / tw
    if binary:
        hits, signed = w * ((p > 0.5) == (y > 0.5)), w
    else:
        # A target or a prediction of zero has no sign, as in :func:`metrics`
        # and the bank (docs/PLAN.md task 195, S6); a fit with no sign to
        # hit signs no row at all, so :func:`from_sums` gives null (E1).
        scored = pl.lit(False) if _no_hit(written) else (y != 0) & (p != 0)
        hits, signed = w * ((y.sign() == p.sign()) & scored), w * scored
    keys = ["slot", "target", *_group_keys(group)]
    return (
        long.group_by(keys)
        .agg(
            pl.len().cast(pl.Int64).alias("n"),
            tw.alias("w"),
            my.alias("mean_y"),
            mp.alias("mean_pred"),
            (w * (y - my) ** 2).sum().alias("m2_y"),
            (w * (p - mp) ** 2).sum().alias("m2_pred"),
            (w * (y - my) * (p - mp)).sum().alias("cov"),
            (w * (y - p) ** 2).sum().alias("sse"),
            hits.sum().alias("hits"),
            signed.sum().alias("signed"),
        )
        .sort(keys)
    )


def merge_sums(first: pl.DataFrame, *rest: pl.DataFrame) -> pl.DataFrame:
    """Add the sums of disjoint sets of rows, as if their rows were one chunk.

    Use it to add the :func:`sums` of the chunks of a stream, in any number
    and any split, before reading the scores with :func:`from_sums`.

    .. rubric:: Parameters

    ``first``, ``*rest``
        Frames :func:`sums` returned, over disjoint sets of rows, each with
        the same key columns. Merging one frame returns its sums again, to
        rounding: the means are recomputed as ``sum(w * mean) / w``.

    .. rubric:: Returns

    A ``DataFrame`` with the same columns: the keys, then
    :data:`SUM_FIELDS`, one row per key, sorted by the keys. A key present
    in one frame and not another is carried through as it is.

    .. rubric:: Raises

    ``TypeError``
        A frame is not a ``DataFrame``.
    ``ValueError``
        The frames' key columns differ.

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

    ``n``, ``sse``, ``hits`` and ``signed`` are summed.

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
    """
    frames = [first, *rest]
    for f in frames:
        _frame("po.eval.merge_sums", f)
    keys = [c for c in frames[0].columns if c not in SUM_FIELDS]
    for f in frames[1:]:
        other = [c for c in f.columns if c not in SUM_FIELDS]
        if other != keys:
            msg = (
                f"po.eval.merge_sums: every part must have the same key columns, but one has "
                f"{keys} and another {other}. Pass parts from po.eval.sums with the same group."
            )
            raise ValueError(msg)
    w = pl.col("w").sum()
    mean_y = (pl.col("w") * pl.col("mean_y")).sum() / w
    mean_p = (pl.col("w") * pl.col("mean_pred")).sum() / w
    return (
        pl.concat(frames)
        .group_by(keys)
        .agg(
            pl.col("n").sum(),
            w.alias("w"),
            mean_y.alias("mean_y"),
            mean_p.alias("mean_pred"),
            (pl.col("m2_y") + pl.col("w") * (pl.col("mean_y") - mean_y) ** 2).sum().alias("m2_y"),
            (pl.col("m2_pred") + pl.col("w") * (pl.col("mean_pred") - mean_p) ** 2)
            .sum()
            .alias("m2_pred"),
            (
                pl.col("cov")
                + pl.col("w") * (pl.col("mean_y") - mean_y) * (pl.col("mean_pred") - mean_p)
            )
            .sum()
            .alias("cov"),
            pl.col("sse").sum(),
            pl.col("hits").sum(),
            pl.col("signed").sum(),
        )
        .select(*keys, *SUM_FIELDS)
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
        :func:`metrics` keeps it.
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
    number. Sums whose ``w`` is 0, from rows that all weigh 0, give ``NaN``
    for ``r2``, ``ic``, ``mse`` and ``rmse``. There is no ``log_loss`` column under ``binary``:
    :data:`SUM_FIELDS` carries no sum for it, and :func:`metrics` computes
    it only over a frame in memory.

    .. rubric:: Raises

    ``TypeError`` for an ``s`` that is not a ``DataFrame``.

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
    - :func:`metrics`, the same scores over a frame in memory.
    """
    _frame("po.eval.from_sums", s)
    keys = [c for c in s.columns if c not in SUM_FIELDS]
    mse = pl.col("sse") / pl.col("w")
    denom = (pl.col("m2_y") * pl.col("m2_pred")).sqrt()
    out = s.select(
        *keys,
        pl.col("n"),
        pl.when(pl.col("m2_y") > 0)
        .then(1.0 - pl.col("sse") / pl.col("m2_y"))
        .otherwise(None)
        .alias("brier_skill" if binary else "r2"),
        pl.when(denom > 0).then(pl.col("cov") / denom).otherwise(None).alias("ic"),
        pl.when(pl.col("signed") > 0)
        .then(pl.col("hits") / pl.col("signed"))
        .otherwise(None)
        .alias("accuracy" if binary else "hit_rate"),
        mse.alias("mse"),
        mse.sqrt().alias("rmse"),
    ).sort(keys)
    return _thin(out, min_samples, keys)


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


def _hac_test(
    df: pl.DataFrame,
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
            pl.when(s > 0).then(dd.mean() * pl.len() / s.sqrt()).alias(stat),
        ]
        g = rows.group_by(keys, maintain_order=True).agg(exprs) if keys else rows.select(exprs)
        frames.append(g.with_columns(pl.lit(t).alias("target")))
    out = pl.concat(frames).select(*keys, "target", pl.col("n").cast(pl.Int64), "mean", stat)
    return _thin(out, min_samples, [*keys, "target"])


def _ew_test(
    df: pl.DataFrame,
    values: dict[str, pl.Expr],
    keys: list[str],
    lags: int,
    half_life: float,
    stat: str,
    name: str,
) -> pl.DataFrame:
    """Per row, the same t statistic with each row before it weighed by
    ``λ^age``, ``λ = 0.5 ** (1 / half_life)``, one step per row that has a
    value, the row itself included:

    ``m = Σ ω d / Σ ω``, ``S = Σ ω_t² (d_t − m)² + 2 Σ_l (1 − l/(L+1)) Σ_t ω_t ω_{t−l}
    (d_t − m)(d_{t−l} − m)``, ``stat = m Σ ω / sqrt(S)``.

    Each double sum is an exponentially weighted sum at ``λ²`` -- of the
    products, the two sides and the pairs -- read as ``ewm_mean`` times its
    weights, so the statistic is a few window expressions, not a loop.
    """
    lam = 0.5 ** (1.0 / half_life)
    lam2 = lam * lam
    # Working columns step around the frame's and the result's name, so a
    # frame holding `dm_<t>` keeps it and a name spelled like one is kept.
    taken = {*df.columns, name}
    idx = _free(taken, "row")
    base = df.with_row_index(idx)
    fields: list[pl.Expr] = []
    temps: list[str] = []

    def over(e: pl.Expr) -> pl.Expr:
        return e.over(keys) if keys else e

    i = over(pl.int_range(pl.len())).cast(pl.Float64)
    w1 = (1.0 - pl.lit(lam).pow(i + 1.0)) / (1.0 - lam)
    w2 = (1.0 - pl.lit(lam2).pow(i + 1.0)) / (1.0 - lam2)

    def ew_sum2(e: pl.Expr) -> pl.Expr:
        """``Σ λ^{2(T−t)} e_t``, the missing terms as zeros."""
        return over(e.fill_null(0.0).ewm_mean(alpha=1.0 - lam2, adjust=True)) * w2

    for t, d in values.items():
        col = _free(taken, "d")
        rows = base.select(idx, *keys, d.alias(col)).filter(pl.col(col).is_not_null())
        dd = pl.col(col)
        m = over(dd.ewm_mean(alpha=1.0 - lam, adjust=True))
        s = ew_sum2(dd * dd) - 2.0 * m * ew_sum2(dd) + m * m * w2
        for lag in range(1, lags + 1):
            back = over(dd.shift(lag))
            has = back.is_not_null().cast(pl.Float64)
            a_l = ew_sum2(dd * back)
            b_l = ew_sum2(dd * has)
            c_l = ew_sum2(back)
            d_l = ew_sum2(has)
            weight = 1.0 - lag / (lags + 1.0)
            s = s + 2.0 * weight * lam**lag * (a_l - m * (b_l + c_l) + m * m * d_l)
        out_col = _free(taken, f"{stat}_{len(temps)}")
        value = pl.when(s > 0).then(m * w1 / s.sqrt()).alias(out_col)
        got = rows.select(idx, value)
        base = base.join(got, on=idx, how="left", maintain_order="left")
        fields.append(pl.col(out_col).alias(f"{stat}_{t}"))
        temps.append(out_col)
    return base.with_columns(pl.struct(fields).alias(name)).drop(idx, *temps)


def _check_half_life(who: str, half_life: float) -> None:
    """The per-row form's ``half_life``: above 0, and finite. An infinite
    one reached polars' ``ewm_mean`` as ``alpha = 0``, which it refuses
    with no word of the parameter."""
    if not half_life > 0:
        msg = (
            f"{who}: half_life must be finite and above 0, but it is {half_life}. Give "
            f"half_life a positive number of rows, or leave it None for one statistic over "
            f"every row."
        )
        raise ValueError(msg)
    if math.isinf(half_life):
        msg = (
            f"{who}: half_life must be finite and above 0, but it is {half_life}. Give "
            f"half_life a positive number of rows, or leave it None for one statistic over "
            f"every row."
        )
        raise ValueError(msg)


@renamed_keywords("po.eval.diebold_mariano", _GROUP)
def _lags(who: str, lags: Any) -> int:
    """``lags`` as an int, once it is checked to be a whole number of at least 0:
    any whole number, numpy's integers included, but not a bool."""
    wrong_type = isinstance(lags, bool) or not isinstance(lags, numbers.Integral)
    if wrong_type:
        msg = (
            f"{who}: lags must be a whole number of at least 0, but it is {lags!r}. Set lags "
            "to a whole number, such as 0 for one-step forecasts."
        )
        raise WrongTypeError(msg)
    if operator.index(lags) < 0:
        msg = f"{who}: lags must be at least 0, but it is {lags}. Set lags to 0 or more."
        raise ValueError(msg)
    return operator.index(lags)


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
) -> pl.DataFrame:
    """Test whether two specs' out-of-sample squared errors differ on average,
    by Diebold and Mariano's test.

    Use it to ask whether one spec predicts better than another that is not
    nested in it. It is Diebold and Mariano's (1995) test, with Newey and
    West's (1987) variance. When one spec's features are a subset of the other's,
    use :func:`clark_west`, since this test leans toward the smaller of two
    nested specs. For evidence you may read at any row, use
    :func:`seqtest`.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding both specs' output columns. A
        ``LazyFrame`` is refused: pass ``lf.collect()``, which runs the query
        and returns its result.
    ``a``, ``b``
        The names of the two specs' output columns. The statistic is
        positive when ``a`` was closer.
    ``targets``, ``a_suffix``, ``b_suffix``
        :func:`seqtest`'s, in compare mode: the targets whose residuals are
        compared (every target the two share by default), and the suffix of
        each side's residual field.
    ``lags``
        The lags of the Newey-West variance, 0 by default, for one-step
        predictions, whose differentials are not correlated under the null.
        For a target that looks ``h`` rows ahead, give at least ``h - 1``.
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
        The fewest rows a key is tested on, 30 by default. Without
        ``half_life``, a key with fewer rows keeps its row, with ``n``
        filled and the other three columns null, as Polars gives null below
        its ``min_samples``.
    ``name``
        With ``half_life``, the name of the record column added,
        ``"diebold_mariano"`` by default.

    .. rubric:: Returns

    Without ``half_life``, a ``DataFrame`` with one row per ``group`` key
    and target: ``n``, the rows read; ``mean``, the mean differential;
    ``dm``, the statistic; and ``p_value``, its two-sided p-value. A
    ``dm`` above 1.96 says ``a`` predicted better at the 5% level, and one
    below -1.96 says ``b`` did.

    With ``half_life``, ``df`` with a record column ``name`` added, holding
    ``dm_<t>`` per target ``t``, null on a row where a side has no residual.
    It can be read at any row.

    .. rubric:: Raises

    Whatever :func:`seqtest` raises for ``a``, ``b``, ``targets``, the
    suffixes and ``group``. ``ValueError`` too, for ``lags`` below 0, for a
    ``half_life`` that is not finite and above 0, and, with ``half_life``,
    for a ``name`` that ``df`` already has as a column, which the result
    would replace. :class:`~polars_online.WrongTypeError`, both a
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
    variance, times ``n``, and ``d_i`` is the differential on row
    ``i``.

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
    - :func:`seqtest`, evidence to read at any row.
    - :func:`compare_specs`, whose Notes say when to use each test.
    """
    _frame("po.eval.diebold_mariano", df)
    _check_columns(df, "po.eval.diebold_mariano", group=group)
    keys = _group_keys(group)
    pairs = _paired_residuals(df, a, b, targets, a_suffix, b_suffix, "po.eval.diebold_mariano")
    values = {t: rb * rb - ra * ra for t, (ra, rb) in pairs.items()}
    lags = _lags("po.eval.diebold_mariano", lags)
    if half_life is not None:
        _check_half_life("po.eval.diebold_mariano", half_life)
        _check_name("po.eval.diebold_mariano", df, name)
        return _ew_test(df, values, keys, lags, half_life, "dm", name)
    out = _hac_test(df, values, keys, lags, "dm", min_samples)
    p = [
        None if z is None or math.isnan(z) else math.erfc(abs(z) / math.sqrt(2.0))
        for z in out["dm"].to_list()
    ]
    return out.with_columns(pl.Series("p_value", p, dtype=pl.Float64))


@renamed_keywords("po.eval.clark_west", _GROUP)
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
) -> pl.DataFrame:
    """Test whether a larger spec predicts better than a smaller one nested in
    it, by Clark and West's test.

    Use it when one spec's features are a subset of the other's, such as a
    spec with and without one more feature. It is Clark and West's (2007)
    test, with Newey and West's variance. For two specs that are not
    nested, use :func:`diebold_mariano`, and for evidence you may read at
    any row, :func:`seqtest`.

    .. rubric:: Parameters

    ``df``
        A frame a bank returned, holding both specs' output columns. A
        ``LazyFrame`` is refused: pass ``lf.collect()``, which runs the query
        and returns its result.
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
    ``cw_<t>`` per target ``t``, the exponentially weighted statistic on
    every row, as :func:`diebold_mariano` gives it.

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
    """
    _frame("po.eval.clark_west", df)
    _check_columns(df, "po.eval.clark_west", group=group)
    keys = _group_keys(group)
    pairs = _paired_residuals(
        df, big, small, targets, big_suffix, small_suffix, "po.eval.clark_west"
    )
    values = {t: rs * rs - rb * rb + (rb - rs) * (rb - rs) for t, (rb, rs) in pairs.items()}
    lags = _lags("po.eval.clark_west", lags)
    if half_life is not None:
        _check_half_life("po.eval.clark_west", half_life)
        _check_name("po.eval.clark_west", df, name)
        return _ew_test(df, values, keys, lags, half_life, "cw", name)
    out = _hac_test(df, values, keys, lags, "cw", min_samples)
    return out.with_columns(
        pl.Series("p_value", [_normal_sf(z) for z in out["cw"].to_list()], dtype=pl.Float64)
    )
