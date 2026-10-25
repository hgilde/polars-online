"""The window operators: exponentially weighted means, sums, rates and
quantiles over a stream's clock, looking back or ahead, a variance and a
standard deviation looking back, and the increment of a running sum. Each returns a
``pl.Expr`` that composes with Polars' own -- ``pl.col`` is the current
row. It is computed by :func:`polars_online.stream.with_windows` in one
pass, in O(window) memory, or given to a spec as a target. A target that
looks ahead is known only once its window has closed, so the bank learns
it then, under an ``embargo`` (:mod:`polars_online.spec`) of at least the
window.

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
   * - :func:`ewm_mean` with ``reverse=True``
     - the mirror: value *j* is held over ``[t_j, t_{j+1})``, until the next
       row, and weighed by ``integral lam**(s - tau_t) ds`` inside ``(tau_t,
       tau_t + w]``
     - the reversed stream's :func:`ewm_mean`
   * - :func:`ewm_sum`, and with ``reverse=True``
     - ``sum lam**|t_j - tau_t| * x_j`` over the rows of the window, each
       counted once at its own time
     - Polars' ``ewm_sum_by`` on distinct stamps; at a repeated stamp every
       row holds the stamp's whole window, so each carries the stamp's
       total where ``ewm_sum_by``'s is a running sum
   * - :func:`ewm_rate`, and with ``reverse=True``
     - the sum over the decayed time the window covers, ``integral_0^T lam**s
       ds = half_life / ln 2 * (1 - 2 ** (-T / half_life))``, ``T`` the
       window's span inside the stretch: a quantity per unit time
     - the sum and the mass, each as above
   * - :func:`ewm_var`, :func:`ewm_std`
     - the variance about :func:`ewm_mean`, each value under the mean's
       weight ``m_i``: ``sum m_i (x_i - mu) ** 2 / V1``, ``V1 = sum m_i``;
       unless ``bias``, times ``V1 ** 2 / (V1 ** 2 - V2)``, ``V2 = sum
       m_i ** 2``; and its root. Looking back only
     - Polars' ``ewm_var`` and ``ewm_std`` under ``adjust=False`` on a
       clock that steps by 1 a row, the correction for unequal weights
       Polars' own
   * - :func:`ewm_quantile`
     - the smallest ``x`` with ``W(<= x) >= q W``, each row of the window
       weighing ``lam**|t_j - tau_t|``, as in :func:`ewm_sum`, ``W`` their
       total; read from buckets each spanning a ratio of at most
       ``e**(1/64)``, so within ``tanh(1/128)``, 0.78%, of it, relatively
     - numpy's ``inverted_cdf`` quantile over the window's values and
       their weights
   * - :func:`increment`
     - ``x_i - x_{i-1}`` within the group and session; null on a session's
       first row; seconds on a temporal column
     - the row before it

**Which rows a window holds follows Polars' rolling windows**
(``rolling_*_by``): a window is a set of timestamps, and ``closed`` says
which ends are in. Every row at one stamp gets the same backward window,
later rows at that stamp included, and looking ahead the mirror holds:

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
magnitude is read as missing, as a null is: every model reads its inputs
within that bound, where a weight of 1e100 times a square of 1e100 is 1e300,
still inside a double's range. At a row whose value is missing the operator
gives the window as it stands, where Polars' ``ewm_mean_by`` gives null. A
row is in a window by its stamp, but a mean weighs it by its held interval
inside the window. So a row exactly one window old under ``"left"`` or
``"both"`` counts for ``min_samples`` and weighs nothing, since its interval
ends at its row, and a mean with no other value in the window is null.

**A weighted mean is a ratio of two sums.** A VWAP is decayed notional over
decayed volume, the time mass cancelling. A side's VWAP puts ``when/then``
inside both sums. There is no ``weight=``:

.. code-block:: python

    notional = pl.col("price") * pl.col("quantity")
    buys = pl.when(pl.col("side") == "buy")
    out = po.stream.with_windows(
        trades,
        mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
        fwd_vwap=po.ewm_sum(notional, half_life="10s", window_size="1m", reverse=True)
                 / po.ewm_sum("quantity", half_life="10s", window_size="1m", reverse=True)
                 - pl.col("mid"),
        buy_vwap=po.ewm_sum(buys.then(notional), half_life="10s")
                 / po.ewm_sum(buys.then("quantity"), half_life="10s"),
        clock="ts", gap_cap="5m", group="symbol",
    )

**A feature in units of its spread is a ratio of a deviation and an
operator.** ``(x - ewm_mean(x)) / ewm_std(x)`` is the feature's z-score: a
model fitted on it has a coefficient per standard deviation of the feature
as its spread stands on each row, where a model on ``x`` has one per unit.
Where a target responds to a feature relative to its volatility, fit on the
z-score. Every model reads it as a column (``docs/WINDOWS.md``, *Features
in units of their spread*, with the measured gain):

.. code-block:: python

    spread = {"half_life": "30m"}
    z_mid = (pl.col("mid") - po.ewm_mean("mid", **spread)) / po.ewm_std("mid", **spread)
    zscored = pl.scan_parquet("trades.parquet").online.with_windows(
        z_mid=z_mid, clock="ts", gap_cap="5m", group="symbol"
    ).collect()

**The clock and its policy** -- ``clock``, ``gap_cap``,
``restart_after_step_back``, ``session``, ``session_gap``, ``group`` -- are
the call's, shared by every operator in it, in a spec's own words
(:mod:`polars_online.spec`), with one difference: a session needs no
``session_gap``, since no window measures the clock's step across a session
change, and a number given for it changes nothing (only ``"reset"`` acts).
A gap past ``gap_cap`` or a session change ends every window open across
it, and ``partial`` says what such a window gives.
A backward window whose span reaches before its stretch's first row is
partial too, the stream's own start included, so ``partial`` also says what
every stream's first ``window_size`` gives:

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
under ``closed="right"`` or ``"both"`` whose far edge is the last row before
a cut or a reset is whole: every row it covers has arrived, so it gives its
value. A forward window still open when the input ends is null.
Under ``group`` a forward window needs a clock column, since only a clock
ends the windows of a group that falls silent. ``partial`` needs a
``window_size``: without one no window is cut short, so ``partial`` alone is
refused.

An operator's input is a column name or an element-wise expression of the
row, :func:`increment` included -- not another window operator: a formula
over an operator's output is a second call. ``half_life`` and
``window_size`` are in the clock's units: numbers on a numeric clock, and
durations (``"10s"``, a ``timedelta``, ``pl.duration``) on a temporal one.
``half_life=float("inf")`` weighs the window evenly, and needs a
``window_size`` for a mean, a variance or a standard deviation.

An operator checks its own arguments when it is called, and names the one
it refuses:

- ``TypeError`` for an input that is neither a column name nor a
  ``pl.Expr``, a ``half_life`` or ``window_size`` that is neither a number
  nor a duration, a ``bias`` that is not a bool, and a ``quantile`` that is
  not a number;
- ``ValueError`` for a ``half_life`` or ``window_size`` that is NaN or not
  above 0, an infinite ``window_size``, a ``closed`` or ``partial`` it does
  not know, ``partial`` without ``window_size``, a ``min_samples`` below 1
  or above ``2**32 - 1``, and a ``quantile`` not above 0 and below 1;
- ``ValueError`` for an input expression that holds another window
  operator, or a node that is not element-wise, or that reads a column
  whose name starts with ``@po:``, the prefix the operators' own columns
  carry.

:func:`polars_online.stream.with_windows` refuses the rest while its plan is
built: ``half_life=float("inf")`` with no ``window_size`` for a mean, a
variance or a standard deviation, and a window looking ahead under
``group`` with no clock column.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import polars as pl

from polars_online._formula import operator
from polars_online._warnings import deprecated_alias

if TYPE_CHECKING:
    from polars_online._duration import Duration

__all__ = [
    "ewm_mean",
    "ewm_quantile",
    "ewm_rate",
    "ewm_std",
    "ewm_sum",
    "ewm_var",
    "increment",
]


def _window(
    name: str,
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None,
    closed: Literal["right", "left", "both", "none"],
    min_samples: int,
    partial: Literal["keep", "null", "drop"] | None,
    reverse: bool,
    quantile: float | None = None,
) -> pl.Expr:
    """The mean, sum, rate or quantile ``name`` names, looking back, or ahead
    under ``reverse``: the operator over the rows after each row, which the
    tree keeps as ``r<name>``."""
    if not isinstance(reverse, bool):
        raise TypeError(f"po.{name}: reverse must be a bool, got {type(reverse).__name__}")
    if reverse and window_size is None:
        raise TypeError(
            f"po.{name}(reverse=True): window_size is required; the rows ahead of a row have "
            "no end without one"
        )
    return operator(
        f"r{name}" if reverse else name,
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        quantile=quantile,
        who=f"po.{name}(reverse=True)" if reverse else None,
    )


def ewm_mean(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """The time-weighted mean of ``input`` over the rows at or before each
    row, less than ``window_size`` older (none: the running mean).

    Polars' ``ewm_mean_by`` recursion, ``y_i = a_i x_i + (1 - a_i) y_{i-1}``
    with ``a_i = 1 - lam ** (t_i - t_{i-1})`` and ``y_1 = x_1``: each value
    weighed by the decayed time of the interval ending at its row. Under a
    ``window_size`` only the part of each interval inside the window counts.

    ``reverse=True`` looks ahead instead, as Polars' ``cum_sum(reverse=True)``
    runs from the end: the mean over the rows after each row, at most
    ``window_size`` later, which it then requires. It is the mirror, each
    value held until the next row and weighed by the decayed time from the
    row. As a spec's target it is known only once the window has closed, so
    the bank learns it under an ``embargo`` of at least ``window_size``.
    """
    return _window(
        "ewm_mean",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
    )


def ewm_sum(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """``sum lam ** (tau_t - t_j) * x_j`` over the rows at or before each row,
    less than ``window_size`` older. On distinct stamps it is Polars'
    ``ewm_sum_by``, each value counted once at its own time. At a repeated
    stamp every row carries the stamp's total, where ``ewm_sum_by``'s is a
    running sum. Two sums make a weighted mean.

    ``reverse=True`` looks ahead instead: ``sum lam ** (t_j - tau_t) * x_j``
    over the rows after each row, at most ``window_size`` later, which it then
    requires."""
    return _window(
        "ewm_sum",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
    )


def ewm_rate(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """:func:`ewm_sum` over the decayed time the window covers, ``half_life /
    ln 2 * (1 - 2 ** (-T / half_life))``, with ``T`` the window's span inside
    the stretch. It is a quantity per unit of clock, such as a volume rate
    from :func:`increment` of a running volume.

    ``reverse=True`` looks ahead instead: the sum ahead over the decayed time
    the window ahead covers, ``window_size`` then required."""
    return _window(
        "ewm_rate",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
    )


def ewm_quantile(
    input: str | pl.Expr,
    *,
    quantile: float,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """The exponentially weighted quantile at level ``quantile`` of ``input``
    over the rows at or before each row, less than ``window_size`` older
    (none: every row since the stretch began, decayed).

    Each row of the window with a value counts once, at its own time, with
    the weight :func:`ewm_sum` gives it, ``w_j = lam ** (tau_t - t_j)``. The
    quantile at ``q`` is the smallest ``x`` with ``W(<= x) >= q W``, where
    ``W(<= x)`` is the weight of the values at or below ``x`` and ``W`` the
    weight of all of them: numpy's ``inverted_cdf`` quantile over the
    window's values and weights. A value is read as
    the bucket it falls in, each bucket spanning a ratio of at most
    ``e ** (1/64)``, so the result is within ``tanh(1/128)``, 0.78%, of the exact quantile,
    relatively, at any scale and either sign; a magnitude below the smallest
    normal double counts as 0. A row whose value is missing counts for
    nothing and gives the window as it stands, as the sums do.

    **A robust scale.** The median of ``|r|`` over ``Φ⁻¹(3/4) =
    0.6744897501960817`` is the median absolute deviation of ``r`` about 0,
    which estimates a standard deviation for normal ``r`` and holds while up
    to half the window is wild, where a root mean square is moved by one
    row. For a residual, centred on 0 by the fit::

        mad = po.ewm_quantile(pl.col("resid").abs(), quantile=0.5, half_life=500)
        scale = mad / 0.6744897501960817

    ``closed="left"`` leaves the row's own value out of its window, so a
    scale of a target used as a weight on the same row does not see the
    label it weighs.

    ``reverse=True`` looks ahead instead: the quantile of the rows after
    each row, at most ``window_size`` later, which it then requires, each
    weighing ``lam ** (t_j - tau_t)``.
    """
    return _window(
        "ewm_quantile",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
        quantile=quantile,
    )


def _forward(kind: str) -> Callable[..., pl.Expr]:
    """The old name of ``ewm_<kind>(reverse=True)``, forwarded with a
    deprecation warning."""
    new = {"mean": ewm_mean, "sum": ewm_sum, "rate": ewm_rate}[kind]

    @deprecated_alias(f"po.rewm_{kind}", f"po.ewm_{kind}(..., reverse=True)")
    def old(
        input: str | pl.Expr,
        *,
        half_life: float | Duration,
        window_size: float | Duration,
        closed: Literal["right", "left", "both", "none"] = "right",
        min_samples: int = 1,
        partial: Literal["keep", "null", "drop"] | None = None,
    ) -> pl.Expr:
        return new(
            input,
            half_life=half_life,
            window_size=window_size,
            closed=closed,
            min_samples=min_samples,
            partial=partial,
            reverse=True,
        )

    old.__name__ = old.__qualname__ = f"rewm_{kind}"
    old.__doc__ = (
        f"Renamed :func:`ewm_{kind}` with ``reverse=True``: each call raises a "
        ":class:`~polars_online.PolarsOnlineDeprecationWarning`, then gives that."
    )
    return old


rewm_mean = _forward("mean")
rewm_sum = _forward("sum")
rewm_rate = _forward("rate")


def ewm_var(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    bias: bool = False,
) -> pl.Expr:
    """The variance of ``input`` about :func:`ewm_mean`, under the mean's own
    weights, over the rows at or before each row, less than ``window_size``
    older (none: the running variance). It looks back only.

    Each value weighs ``m_i``, the decayed mass of its held interval inside
    the window, as in :func:`ewm_mean`. The variance is ``sum m_i (x_i -
    mu) ** 2 / V1``, with ``mu = sum m_i x_i / V1`` and ``V1 = sum m_i``.
    Unless ``bias``, it is multiplied by ``V1 ** 2 / (V1 ** 2 - V2)``, with
    ``V2 = sum m_i ** 2``: the correction for unequal weights that Polars'
    ``ewm_var`` applies to its own. It is null while one row carries all the
    weight, as Polars' first row is.

    On a clock that steps by 1 a row, with ``half_life`` in rows, this is
    Polars' ``ewm_var(half_life=..., adjust=False, bias=...)``.
    ``adjust=False`` is the form :func:`ewm_mean` already mirrors: it is
    ``ewm_mean_by``, which holds a stretch's first value from before it. So
    the first row weighs ``(1 - alpha) ** (n - 1)``, with
    ``alpha = 1 - lam``, as ``adjust=False`` weighs it. A window that holds
    one value gives 0 exactly, at any level.
    """
    return operator(
        "ewm_var",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        bias=bias,
    )


def ewm_std(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    bias: bool = False,
) -> pl.Expr:
    """The square root of :func:`ewm_var`, as Polars' ``ewm_std`` is the root
    of its ``ewm_var``. ``(x - ewm_mean(x)) / ewm_std(x)`` is a feature in
    units of its current spread: a z-score."""
    return operator(
        "ewm_std",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        bias=bias,
    )


def increment(input: str | pl.Expr) -> pl.Expr:
    """``x_i - x_{i-1}`` of ``input`` within the group and session, ``x_{i-1}``
    the input's last value with a value. A null input makes the row's
    increment null and is skipped, as the operators hold a value from the
    last valued row (``[1, null, 3]`` gives ``[null, null, 2]``, where
    ``diff()`` gives three nulls). Null on a session's first row, and after a
    step back ``restart_after_step_back`` reads as a new start, in the row's
    group or on the stream across groups. Seconds on a temporal column; on a
    time of day, the step in its seconds since midnight, so a step across
    midnight is negative, as Polars' ``diff`` on a ``Time`` gives. On an
    integer column, ``Int8`` to ``Int128`` or ``UInt8`` to ``UInt64``, the
    step is taken in integers, then made a float, rounded once, so a running
    count past ``2**53`` keeps its steps of 1. Give it a running total
    before a sum or a rate: its increments are the trades themselves, where
    a window's notional of 1e6 read as the difference of two running totals
    near 1e10 loses about four of a double's sixteen digits."""
    return operator("increment", input)
