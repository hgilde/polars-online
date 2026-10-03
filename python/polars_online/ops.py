"""The window operators (docs/PLAN.md task 143): exponentially weighted
means, sums and rates over a stream's clock, looking back or ahead, and the
increment of a running sum. Each returns a ``pl.Expr`` that composes with
Polars' own -- ``pl.col`` is the current row -- and is computed by
:func:`polars_online.stream.with_windows` one pass, O(window) memory, or
learned as a target under an embargo (task 104).

Every operator is defined once, by the library that has it, and held to it
in the tests. For row *t* at policy time ``tau_t`` with ``lam = 2 ** (-1 /
half_life)``:

- :func:`ewm_mean`: Polars' ``ewm_mean_by``, the *time-weighted* mean. Row
  *i*'s value is held over ``(t_{i-1}, t_i]``, the first value of a stretch
  from before it; the mean over ``(tau_t - w, tau_t]`` weighs each held
  value by ``integral lam**(tau_t - s) ds`` over its interval inside the
  window. A burst of rows does not outweigh a quiet stretch, and a row at a
  repeated stamp holds no interval, so it moves no mean -- as in Polars. No
  mean counts rows; pandas' ``ewm(times=)`` is that form, and is not
  offered.
- :func:`rewm_mean`: the mirror. Value *j* is held over ``[t_j, t_{j+1})``,
  until the next row, and weighed by ``integral lam**(s - tau_t) ds`` inside
  ``(tau_t, tau_t + w]``.
- :func:`ewm_sum` and :func:`rewm_sum`: ``sum lam**|t_j - tau_t| * x_j`` over
  the rows of the window, each counted once at its own time (Polars'
  ``ewm_sum_by``).
- :func:`ewm_rate` and :func:`rewm_rate`: the sum over the decayed time the
  window covers, ``integral_0^T lam**s ds = half_life / ln 2 * (1 - 2 **
  (-T / half_life))``, ``T`` the window's span inside the stretch -- a
  quantity per unit time.
- :func:`increment`: ``x_i - x_{i-1}`` within the group and session; null on
  a session's first row; seconds on a temporal column.

**Which rows a window holds follows Polars' ``rolling_*_by``**: a window is a
set of timestamps. ``closed="right"``, the default, is ``(tau_t - w, tau_t]``
looking back and ``(tau_t, tau_t + w]`` looking ahead; ``"left"``, ``"both"``
and ``"none"`` move the ends. Every row at one stamp gets the same backward
window, later rows at that stamp included, so under ``"right"`` and
``"both"`` a backward output waits for the next distinct stamp; under
``"left"`` and ``"none"`` the stamp's own rows are outside, and the output
is known as the stamp arrives. A forward window counts a row exactly ``w``
later under ``"right"`` and ``"both"``. ``min_samples`` (Polars' name) nulls
a window holding fewer rows with a value.

**A weighted mean is a ratio of two sums.** A VWAP is decayed notional over
decayed volume, the time mass cancelling; a side's VWAP puts ``when/then``
inside both sums. There is no ``weight=``:

.. code-block:: python

    notional = pl.col("price") * pl.col("quantity")
    buys = pl.when(pl.col("side") == "buy")
    out = po.stream.with_windows(
        trades,
        mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
        fwd_vwap=po.rewm_sum(notional, half_life="10s", window_size="1m")
                 / po.rewm_sum("quantity", half_life="10s", window_size="1m") - pl.col("mid"),
        buy_vwap=po.ewm_sum(buys.then(notional), half_life="10s")
                 / po.ewm_sum(buys.then("quantity"), half_life="10s"),
        clock="ts", gap_cap="5m", group="symbol",
    )

**The clock and its policy** -- ``clock``, ``gap_cap``,
``restart_after_step_back``, ``session``, ``session_gap``, ``group`` -- are
the call's, shared by every operator in it, in a spec's own words
(:mod:`polars_online.spec`). A gap past ``gap_cap`` or a session change ends
every window open across it: ``partial`` says what such a window gives,
``"keep"`` (the value over what it saw; the default looking back),
``"null"`` (the default looking ahead) or ``"drop"`` (the row leaves the
output). A reset discards it: null, never dropped. A forward window still
open when the input ends is null.

An operator's input is a column name or an element-wise expression of the
row, :func:`increment` included -- not another window operator: a formula
over an operator's output is a second call. ``half_life`` and
``window_size`` are in the clock's units: numbers on a numeric clock, and
durations (``"10s"``, a ``timedelta``, ``pl.duration``) on a temporal one;
``half_life=float("inf")`` weighs the window evenly, and needs a
``window_size`` for a mean.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_online._formula import operator

__all__ = ["ewm_mean", "ewm_rate", "ewm_sum", "increment", "rewm_mean", "rewm_rate", "rewm_sum"]


def ewm_mean(
    input: str | pl.Expr,
    *,
    half_life: Any,
    window_size: Any = None,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
) -> pl.Expr:
    """The time-weighted mean of ``input`` over the rows at or before each
    row, less than ``window_size`` older (none: the running mean).

    Polars' ``ewm_mean_by`` recursion, ``y_i = a_i x_i + (1 - a_i) y_{i-1}``
    with ``a_i = 1 - lam ** (t_i - t_{i-1})`` and ``y_1 = x_1``: each value
    weighed by the decayed time of the interval ending at its row. Under a
    ``window_size`` only the part of each interval inside the window counts.
    """
    return operator(
        "ewm_mean",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
    )


def rewm_mean(
    input: str | pl.Expr,
    *,
    half_life: Any,
    window_size: Any,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
) -> pl.Expr:
    """The time-weighted mean of ``input`` over the rows after each row, at
    most ``window_size`` later: the mirror of :func:`ewm_mean`, each value
    held until the next row and weighed by the decayed time from the row.
    A target of the row's future, under an embargo of at least
    ``window_size`` (docs/PLAN.md task 104)."""
    return operator(
        "rewm_mean",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
    )


def ewm_sum(
    input: str | pl.Expr,
    *,
    half_life: Any,
    window_size: Any = None,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
) -> pl.Expr:
    """``sum lam ** (tau_t - t_j) * x_j`` over the rows at or before each row,
    less than ``window_size`` older: Polars' ``ewm_sum_by``, each value
    counted once at its own time. Two sums make a weighted mean."""
    return operator(
        "ewm_sum",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
    )


def rewm_sum(
    input: str | pl.Expr,
    *,
    half_life: Any,
    window_size: Any,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
) -> pl.Expr:
    """``sum lam ** (t_j - tau_t) * x_j`` over the rows after each row, at
    most ``window_size`` later: the mirror of :func:`ewm_sum`."""
    return operator(
        "rewm_sum",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
    )


def ewm_rate(
    input: str | pl.Expr,
    *,
    half_life: Any,
    window_size: Any = None,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
) -> pl.Expr:
    """:func:`ewm_sum` over the decayed time the window covers, ``half_life /
    ln 2 * (1 - 2 ** (-T / half_life))`` with ``T`` the window's span inside
    the stretch: a quantity per unit of clock, such as a volume rate from
    :func:`increment` of a running volume."""
    return operator(
        "ewm_rate",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
    )


def rewm_rate(
    input: str | pl.Expr,
    *,
    half_life: Any,
    window_size: Any,
    closed: str = "right",
    min_samples: int = 1,
    partial: str | None = None,
) -> pl.Expr:
    """:func:`rewm_sum` over the decayed time the window ahead covers: the
    mirror of :func:`ewm_rate`."""
    return operator(
        "rewm_rate",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
    )


def increment(input: str | pl.Expr) -> pl.Expr:
    """``x_i - x_{i-1}`` of ``input`` within the group and session: null on a
    session's first row, and after a step back ``restart_after_step_back``
    reads as a new start; seconds on a temporal column. The input of a sum
    or a rate over a running total, so a day's notional of 1e10 is read as
    its trades (docs/PLAN.md task 143, *Numerics*)."""
    return operator("increment", input)
