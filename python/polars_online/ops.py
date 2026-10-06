"""The window operators (docs/PLAN.md task 143): exponentially weighted
means, sums and rates over a stream's clock, looking back or ahead, and the
increment of a running sum. Each returns a ``pl.Expr`` that composes with
Polars' own -- ``pl.col`` is the current row -- and is computed by
:func:`polars_online.stream.with_windows` one pass, O(window) memory, or
learned as a target under an embargo (task 104).

Every operator is defined once, by the library that has it, and held to it
in the tests. For row *t* at policy time ``tau_t`` with ``lam = 2 ** (-1 /
half_life)``:

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - operator
     - defined as
     - held to
   * - :func:`ewm_mean`
     - the *time-weighted* mean: row *i*'s value is held over ``(t_{i-1},
       t_i]``, the first value of a stretch from before it, and the mean
       over ``(tau_t - w, tau_t]`` weighs each held value by ``integral
       lam**(tau_t - s) ds`` over its interval inside the window
     - Polars' ``ewm_mean_by``. A burst of rows does not outweigh a quiet
       stretch, and a row at a repeated stamp holds no interval, so it moves
       no mean. No mean counts rows; pandas' ``ewm(times=)`` is that form,
       and is not offered
   * - :func:`rewm_mean`
     - the mirror: value *j* is held over ``[t_j, t_{j+1})``, until the next
       row, and weighed by ``integral lam**(s - tau_t) ds`` inside ``(tau_t,
       tau_t + w]``
     - the reversed stream's :func:`ewm_mean`
   * - :func:`ewm_sum`, :func:`rewm_sum`
     - ``sum lam**|t_j - tau_t| * x_j`` over the rows of the window, each
       counted once at its own time
     - Polars' ``ewm_sum_by`` on distinct stamps; at a repeated stamp every
       row holds the stamp's whole window, so each carries the stamp's
       total where ``ewm_sum_by``'s is a running sum
   * - :func:`ewm_rate`, :func:`rewm_rate`
     - the sum over the decayed time the window covers, ``integral_0^T lam**s
       ds = half_life / ln 2 * (1 - 2 ** (-T / half_life))``, ``T`` the
       window's span inside the stretch: a quantity per unit time
     - the sum and the mass, each as above
   * - :func:`increment`
     - ``x_i - x_{i-1}`` within the group and session; null on a session's
       first row; seconds on a temporal column
     - the row before it

**Which rows a window holds follows Polars' ``rolling_*_by``**: a window is a
set of timestamps, and ``closed`` says which ends are in. Every row at one
stamp gets the same backward window, later rows at that stamp included, and
looking ahead the mirror holds:

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - ``closed``
     - looking back
     - looking ahead
   * - ``"right"`` (the default)
     - ``(tau_t - w, tau_t]``: the stamp's own rows are in, so the output
       waits for the next distinct stamp
     - ``(tau_t, tau_t + w]``: none at the stamp is in; a row exactly ``w``
       later is
   * - ``"left"``
     - ``[tau_t - w, tau_t)``: the stamp's own rows are outside, and the
       output is known as the stamp arrives
     - ``[tau_t, tau_t + w)``: every row at the row's own stamp is in, the
       row itself included
   * - ``"both"``
     - both ends in; waits for the next stamp
     - both ends in: the stamp's rows, and a row exactly ``w`` later
   * - ``"none"``
     - neither end; known as the stamp arrives
     - neither end

An edge between two rows is decided from the difference of their clocks,
exact in nanoseconds on a temporal clock, as Polars decides it, at any age
of the stream. ``min_samples`` (Polars' name) nulls a window holding fewer
rows with a value. A value that is NaN, infinite or beyond 1e100 in
magnitude is read as missing, as a null is (the library's input bound,
docs/PLAN.md §3). A row is in a window by its stamp, but a mean weighs it
by its held interval inside the window. So a row exactly one window old
under ``"left"`` or ``"both"`` counts for ``min_samples`` and weighs
nothing, since its interval ends at its row, and a mean with no other value
in the window is null.

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
every window open across it, and ``partial`` says what such a window gives:

.. list-table::
   :header-rows: 1
   :widths: 16 54 30

   * - ``partial``
     - the window gives
     - the default
   * - ``"keep"``
     - the value over what it saw, the window ending at the last row seen
     - looking back
   * - ``"null"``
     - null
     - looking ahead
   * - ``"drop"``
     - the row leaves the output
     -

A reset discards it: null, never dropped. Under ``session_gap="reset"`` a
session change is a reset, so it discards rather than cuts. A forward window
still open when the input ends is null. ``partial`` needs a ``window_size``:
without one no window is cut short, so ``partial`` alone is refused.

An operator's input is a column name or an element-wise expression of the
row, :func:`increment` included -- not another window operator: a formula
over an operator's output is a second call. ``half_life`` and
``window_size`` are in the clock's units: numbers on a numeric clock, and
durations (``"10s"``, a ``timedelta``, ``pl.duration``) on a temporal one.
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
    counted once at its own time, on distinct stamps; at a repeated stamp
    every row carries the stamp's total, where ``ewm_sum_by``'s is a running
    sum. Two sums make a weighted mean."""
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
    """``x_i - x_{i-1}`` of ``input`` within the group and session, ``x_{i-1}``
    the input's last value with a value. A null input makes the row's
    increment null and is skipped, as the operators hold a value from the
    last valued row (``[1, null, 3]`` gives ``[null, null, 2]``, where
    ``diff()`` gives three nulls). Null on a session's first row, and after a
    step back ``restart_after_step_back`` reads as a new start, in the row's
    group or on the stream across groups. Seconds on a temporal column. The
    input of a sum or a rate over a running total, so a day's notional of
    1e10 is read as its trades (docs/PLAN.md task 143, *Numerics*)."""
    return operator("increment", input)
