"""Exponentially weighted means with a hard cutoff, backward and forward.

A description says what to compute and computes nothing; any number of them
run together, in one pass over one stream, with
:func:`polars_online.stream.with_windows`. For row *t* at clock ``tau_t``, over
the rows *j* in its window, taken in stream order:

.. code-block:: text

    y_t    = sum_j w_j * lam**|tau_j - tau_a| * v_j / sum_j w_j * lam**|tau_j - tau_a|
    lam    = 2 ** (-1 / halflife)

:func:`ewm` looks back: the rows at or before *t* less than ``horizon``
older, ``(tau_t - horizon, tau_t]``, the interval a time-indexed
``rolling`` takes by default -- an ordinary EWMA with a hard cutoff, or with
none when ``horizon`` is left out. :func:`lookahead_rewm` looks ahead: the
rows after *t* less than ``horizon`` later, ``(tau_t, tau_t + horizon)``,
weighted most on the next row and less going forward. A weighted mean does
not depend on the anchor ``tau_a``, since moving it multiplies every weight
by one constant; it sits at the near end so every factor is at most 1.

Polars has no cheap form of either: ``rolling(...).agg(...)`` gathers each
window's rows separately, O(rows x rows per window). These keep each window
as running sums in a two-stack queue, O(1) a row, with memory one window of
the rows that count in it.

A row counts in a window when its value is present and its ``weight`` is
present and not zero -- a null, NaN or infinite value or weight counts
nothing, and a weight below zero is refused -- so a market-data row
interleaved with trades never enters a VWAP. A window whose weights sum to
zero is empty: null. ``split=(column, [values])`` gives one output per
listed value of a column, besides the total over every row that counts, so a
buy-side, a sell-side and an all-trades VWAP come from one window.

Each description is a plain dict, as a spec is.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Literal, NotRequired, TypedDict

import polars as pl

from polars_online._duration import Duration, duration_text

__all__ = ["Window", "ewm", "lookahead_rewm"]

Unlisted = Literal["error", "total", "ignore"]
PartialRule = Literal["keep", "null", "drop"]
SameClock = Literal["include", "exclude"]
Span = float | Duration


class Split(TypedDict):
    """The column a window is split by, and its listed values."""

    column: str
    values: list[str | int]


class Window(TypedDict):
    """A window description, as :func:`ewm` and :func:`lookahead_rewm` build
    it."""

    kind: Literal["ewm", "lookahead_rewm"]
    columns: list[str]
    halflife: list[float | str]
    horizon: list[float | str] | None
    weight: str | None
    split: Split | None
    unlisted: Unlisted
    total: bool
    same_clock: NotRequired[SameClock]
    partial: PartialRule
    complete: str | None
    name: str | None


def ewm(
    columns: str | Sequence[str],
    *,
    halflife: Span | Sequence[Span],
    horizon: Span | Sequence[Span] | None = None,
    weight: str | None = None,
    split: tuple[str, Sequence[str | int]] | None = None,
    unlisted: Unlisted = "error",
    total: bool = True,
    partial: PartialRule = "keep",
    complete: str | None = None,
    name: str | None = None,
) -> Window:
    """An exponentially weighted mean over the rows at or before each row, less
    than ``horizon`` older.

    For row *t* at clock ``tau_t``, over the rows *j* with
    ``tau_t - horizon < tau_j <= tau_t`` that count:

    .. code-block:: text

        ewm_t = sum_j w_j * lam**(tau_t - tau_j) * v_j / sum_j w_j * lam**(tau_t - tau_j)
        lam   = 2 ** (-1 / halflife)

    With no ``horizon`` there is no cutoff, and the mean is the running EWMA
    ``S_t = lam**(tau_t - tau_{t-1}) * S_{t-1} + w_t * v_t`` over ``W_t``, the
    same recursion on the weights, kept in two numbers.

    ``columns`` is one column or several, and ``halflife`` and ``horizon`` one
    value or several: the description is one window per column, halflife and
    horizon, in that order. Each is a number of clock units, or a duration on a
    temporal clock (``"10m"``, a ``timedelta``, ``pl.duration(minutes=10)``).
    ``halflife=inf`` weighs the window evenly.

    ``weight`` names a column each row's weight is read from; with none, every
    row that has a value weighs 1. ``split=(column, [values])`` gives one
    output per listed value, each over the rows whose ``column`` has that
    value, and one over every row that counts unless ``total=False``; values
    are matched as text, so ``1`` matches an integer column's ``1``.
    ``unlisted`` says what a row that counts, and whose value is not listed
    (null included), does: ``"error"`` refuses it naming the row; ``"total"``
    counts it in the total only; ``"ignore"`` counts it nowhere.

    ``partial`` is what a window cut short before its horizon passed gives:
    the first ``horizon`` of each group, and of the stream after a session
    change, a gap longer than ``max_dclock`` or a reset. ``"keep"`` (a warm-up
    over what there is), ``"null"``, or ``"drop"``, which removes the row.
    ``complete`` names a Boolean column saying which rows had a full window; a
    window with no cutoff is always full.

    ``name`` is the output columns' template, with the fields ``{column}``,
    ``{halflife}``, ``{horizon}`` and ``{split}`` (``""`` for the total,
    ``"_<value>"`` for a listed value); the default is
    ``"{column}_ewm_{halflife}_{horizon}{split}"``, less ``_{horizon}`` with
    no horizon. ``complete`` may use ``{horizon}``.

    ``TypeError`` for an argument of the wrong type; ``ValueError`` for an
    empty or repeated column list, an empty or repeated split list, and an
    option word that is not one of the listed ones. Values are checked when
    :func:`polars_online.stream.with_windows` builds its plan.
    """
    who = "po.window.ewm"
    return _describe(
        who,
        "ewm",
        columns,
        halflife=halflife,
        horizon=horizon,
        weight=weight,
        split=split,
        unlisted=unlisted,
        total=total,
        same_clock=None,
        partial=_word(who, "partial", partial, ("keep", "null", "drop")),
        complete=complete,
        name=name,
    )


def lookahead_rewm(
    columns: str | Sequence[str],
    *,
    halflife: Span | Sequence[Span],
    horizon: Span | Sequence[Span],
    weight: str | None = None,
    split: tuple[str, Sequence[str | int]] | None = None,
    unlisted: Unlisted = "error",
    total: bool = True,
    same_clock: SameClock = "include",
    partial: PartialRule = "null",
    complete: str | None = None,
    name: str | None = None,
) -> Window:
    """A reverse EWMA over the rows after each row, less than ``horizon`` later:
    weighted most on the next row, and less going forward.

    For row *t* at clock ``tau_t``, over the rows *j* after it with
    ``tau_j - tau_t < horizon`` that count, *f* the first of them:

    .. code-block:: text

        rewm_t = sum_j w_j * lam**(tau_j - tau_f) * v_j / sum_j w_j * lam**(tau_j - tau_f)
        lam    = 2 ** (-1 / halflife)

    It is a label: a row's value is known only once its horizon has passed, so
    :func:`polars_online.stream.with_windows` holds each row until then, and a model
    that learns from it must wait as long (a spec's ``label_delay``).

    ``same_clock`` says whether a later row at *t*'s own clock is one of "the
    rows after": ``"include"`` is stream order, and what interleaved data needs,
    where a trade often prints at the clock of the quote before it;
    ``"exclude"`` is what a time-indexed ``rolling`` does. Two rows share a
    clock when the step between them is zero on the policy clock.

    ``partial`` is what a window cut short gives: a session change, a gap
    longer than ``max_dclock``, ends every window open across it. ``"null"``
    (the default), ``"keep"`` (the mean of what the window saw), or ``"drop"``.
    A reset -- ``session_gap="reset"``, a step back under
    ``on_clock_reset="reset_state"`` -- discards the windows open across it:
    null, whatever ``partial`` says, and never dropped. A row whose horizon has
    not passed when the input ends is unresolved: null, unless
    :func:`~polars_online.stream.with_windows` saves a state, which holds it for the
    next run.

    The default ``name`` is ``"{column}_rewm_{halflife}_{horizon}{split}"``.
    Every other argument is as for :func:`ewm`, and ``horizon`` is required.
    """
    who = "po.window.lookahead_rewm"
    return _describe(
        who,
        "lookahead_rewm",
        columns,
        halflife=halflife,
        horizon=horizon,
        weight=weight,
        split=split,
        unlisted=unlisted,
        total=total,
        same_clock=_word(who, "same_clock", same_clock, ("include", "exclude")),
        partial=_word(who, "partial", partial, ("keep", "null", "drop")),
        complete=complete,
        name=name,
    )


def _describe(
    who: str,
    kind: Literal["ewm", "lookahead_rewm"],
    columns: str | Sequence[str],
    *,
    halflife: Any,
    horizon: Any,
    weight: Any,
    split: Any,
    unlisted: Any,
    total: Any,
    same_clock: Any,
    partial: Any,
    complete: Any,
    name: Any,
) -> Window:
    cols = _names(who, "columns", columns)
    if weight is not None and not isinstance(weight, str):
        raise TypeError(f"{who}: weight must be a column name or None, got {_got(weight)}")
    if not isinstance(total, bool):
        raise TypeError(f"{who}: total must be a bool, got {_got(total)}")
    for key, v in (("complete", complete), ("name", name)):
        if v is not None and not isinstance(v, str):
            raise TypeError(f"{who}: {key} must be a str or None, got {_got(v)}")
    out: Window = {
        "kind": kind,
        "columns": cols,
        "halflife": _spans(who, "halflife", halflife),
        "horizon": None if horizon is None else _spans(who, "horizon", horizon),
        "weight": weight,
        "split": _split(who, split),
        "unlisted": _word(who, "unlisted", unlisted, ("error", "total", "ignore")),
        "total": total,
        "partial": partial,
        "complete": complete,
        "name": name,
    }
    if same_clock is not None:
        out["same_clock"] = same_clock
    return out


def _got(v: Any) -> str:
    return f"{type(v).__name__} {v!r}"


def _word(who: str, key: str, value: Any, words: tuple[str, ...]) -> Any:
    if not isinstance(value, str):
        raise TypeError(
            f"{who}: {key} must be one of {', '.join(map(repr, words))}, got {_got(value)}"
        )
    if value not in words:
        raise ValueError(
            f"{who}: {key} must be one of {', '.join(map(repr, words))}, got {value!r}"
        )
    return value


def _names(who: str, key: str, value: Any) -> list[str]:
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, Sequence) or not all(isinstance(c, str) for c in names):
        raise TypeError(f"{who}: {key} must be a column name or a list of them, got {_got(value)}")
    if not names:
        raise ValueError(f"{who}: {key} is empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{who}: {key} lists a column twice: {list(names)}")
    return list(names)


def _spans(who: str, key: str, value: Any) -> list[float | str]:
    """One clock-unit value or a list of them, each a number or a duration,
    with every duration written as the text the Rust side reads and an
    infinity as ``"inf"``, which JSON has no literal for."""
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"{who}: {key} is empty")
    out: list[float | str] = []
    for v in values:
        if isinstance(v, numbers.Real) and not isinstance(v, bool):
            x = float(v)
            if math.isnan(x):
                raise ValueError(f"{who}: {key} must not be NaN")
            out.append(("inf" if x > 0 else "-inf") if math.isinf(x) else x)
        elif isinstance(v, (str, timedelta, pl.Expr)):
            out.append(duration_text(v, who, key))
        else:
            raise TypeError(
                f"{who}: {key} must be a number or a duration, or a list of them, got {_got(value)}"
            )
    return out


def _split(who: str, split: Any) -> Split | None:
    if split is None:
        return None
    if not (
        isinstance(split, (tuple, list))
        and len(split) == 2
        and isinstance(split[0], str)
        and isinstance(split[1], Sequence)
        and not isinstance(split[1], str)
    ):
        raise TypeError(f"{who}: split must be (column, [values...]), got {_got(split)}")
    column, values = split
    if not all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in values):
        raise TypeError(f"{who}: split values must be text or integers, got {list(values)!r}")
    if not values:
        raise ValueError(f"{who}: split lists no values; give at least one")
    text = [str(v) for v in values]
    if len(set(text)) != len(text):
        raise ValueError(f"{who}: split lists a value twice: {list(values)!r}")
    return {"column": column, "values": list(values)}
