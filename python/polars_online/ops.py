"""The window operators: exponentially weighted (EW) means, sums, rates and
quantiles over a stream's clock, looking back or ahead, and a variance and a
standard deviation looking back. ``increment``, the step of a running total,
gives a window operator an input.

Each operator returns a ``pl.Expr`` that composes with Polars' own
expressions, in which ``pl.col`` reads the current row.
:func:`polars_online.stream.with_windows` computes the expressions in one
pass, holding one window of rows in memory, or a spec takes one as its
target. A target that looks ahead is known only once its window has
closed, so the bank learns it then, under an ``embargo``
(:mod:`polars_online.spec`) of at least the window. The guide `How a window
operator weighs the rows near each row
<https://github.com/hgilde/polars-online/blob/main/docs/WINDOWS.md>`_ explains
each.

Every operator is defined once, by the library that has it, and held to it
in the tests. A *stamp* is a clock value, which several
rows may share. A *clock break* is a gap past ``gap_cap`` or a session
change, and a *stretch* is the rows between two clock breaks. ``tau_t`` is
row *t*'s clock with each step capped at ``gap_cap``, and ``lam = 2 ** (-1 /
half_life)``:

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - operator
     - defined as
     - held to
   * - :func:`ewm_mean`
     - the *time-weighted* mean: row *i*'s value is held over ``(t_{i-1},
       t_i]``, a stretch's first value is held from before the stretch
       begins, and the mean
       over ``(tau_t - w, tau_t]`` weighs each held value by ``integral
       lam**(tau_t - s) ds`` over its interval inside the window
     - Polars' ``ewm_mean_by``. A burst of rows does not outweigh a quiet
       stretch, and a row at a repeated stamp holds no interval, so it moves
       no mean. No operator offers a mean that weighs each row by its count,
       the form pandas' ``ewm(times=)`` computes
   * - :func:`ewm_mean` with ``reverse=True``
     - the mirror: value *j* is held over ``[t_j, t_{j+1})``, until the next
       row, and weighed by ``integral lam**(s - tau_t) ds`` inside ``(tau_t,
       tau_t + w]``
     - the reversed stream's :func:`ewm_mean`
   * - :func:`ewm_sum`, and with ``reverse=True``
     - ``sum lam**|t_j - tau_t| * x_j`` over the rows of the window, each
       counted once at its own time
     - Polars' ``ewm_sum_by`` on distinct stamps. At a repeated stamp every
       row holds the stamp's whole window, so each carries the stamp's
       total, where ``ewm_sum_by``'s is a running sum
   * - :func:`ewm_rate`, and with ``reverse=True``
     - the sum over the decayed time the window covers, ``integral_0^T lam**s
       ds = half_life / ln 2 * (1 - 2 ** (-T / half_life))``, ``T`` the
       window's span inside the stretch: a quantity per unit time
     - the sum and the mass, each as above
   * - :func:`ewm_var`, :func:`ewm_std`
     - the variance about :func:`ewm_mean`, each value under the mean's
       weight ``m_i``: ``sum m_i (x_i - mu) ** 2 / V1``, ``V1 = sum m_i``.
       Unless ``bias``, it is multiplied by ``V1 ** 2 / (V1 ** 2 - V2)``,
       ``V2 = sum m_i ** 2``. :func:`ewm_std` is its root. Looking back only
     - Polars' ``ewm_var`` and ``ewm_std`` under ``adjust=False`` on a
       clock that steps by 1 a row, with Polars' own correction for unequal
       weights
   * - :func:`ewm_quantile`
     - the smallest ``x`` with ``W(<= x) >= q W``, each row of the window
       weighing ``lam**|t_j - tau_t|``, as in :func:`ewm_sum`, ``W`` their
       total. It is read from buckets each spanning a ratio of at most
       ``e**(1/64)``, so it is within ``tanh(1/128)``, 0.78%, of the exact
       quantile, relatively
     - numpy's ``inverted_cdf`` quantile over the window's values and
       their weights
   * - :func:`increment`
     - ``x_i - x_{i-1}`` within the group and session, null on a session's
       first row, and in seconds on a temporal column
     - the row before it

**Which rows a window holds follows Polars' rolling windows**
(``rolling_*_by``). A window is a set of timestamps, and ``closed`` says
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
     - ``(tau_t, tau_t + w]``: no row at the stamp is in, and a row exactly
       ``w`` later is
   * - ``"left"``
     - ``[tau_t - w, tau_t)``: the stamp's own rows are outside, and the
       output is known as the stamp arrives
     - ``[tau_t, tau_t + w)``: every row at the row's own stamp is in, the
       row itself included
   * - ``"both"``
     - both ends in, so the output waits for the next stamp
     - both ends in: the stamp's rows, and a row exactly ``w`` later
   * - ``"none"``
     - neither end, so the output is known as the stamp arrives
     - neither end

An edge between two rows is decided from the difference of their clocks,
exact in nanoseconds on a temporal clock, as Polars decides it, at any age
of the stream. ``min_samples`` (Polars' name) nulls a window holding fewer
rows with a value. A value that is NaN, infinite or beyond 1e100 in
magnitude is read as missing, as a null is. Every model reads its inputs
within that bound, since a weight of 1e100 times a square of 1e100 is
1e300, still inside a double's range. At a row whose value is missing the
operator gives the window as it stands, where Polars' ``ewm_mean_by`` gives
null. A row is in a window by its stamp, but a mean weighs it by its held
interval inside the window. So under ``"left"`` or ``"both"``, a row exactly
one window old counts for ``min_samples`` and weighs nothing, since its
interval ends at its row. A mean with no other value in the window is null.

**A weighted mean is a ratio of two sums.** A volume-weighted average price
(VWAP) is decayed notional over decayed volume, and the time mass cancels.
For one side's VWAP, put ``when/then`` inside both sums. The operators take
no ``weight=``. This code reads ``trades`` from the README's `Example data
<https://github.com/hgilde/polars-online#example-data>`_:

.. code-block:: python

    notional = pl.col("price") * pl.col("quantity")
    buys = pl.when(pl.col("side") == "buy")
    out = po.stream.with_windows(
        trades,
        mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
        fwd_vwap_vs_mid=po.ewm_sum(notional, half_life="10s", window_size="1m", reverse=True)
                        / po.ewm_sum("quantity", half_life="10s", window_size="1m", reverse=True)
                        - pl.col("mid"),
        buy_vwap=po.ewm_sum(buys.then(notional), half_life="10s")
                 / po.ewm_sum(buys.then("quantity"), half_life="10s"),
        clock="ts", gap_cap="5m", group="symbol",
    )

**A feature in units of its spread is a deviation over an operator.**
``(x - ewm_mean(x)) / ewm_std(x)`` is the feature's z-score. A model fitted
on it has a coefficient per standard deviation of the feature, as the
spread stands on each row, where a model on ``x`` has one per unit. Where a
target responds to a feature relative to its volatility, fit on the
z-score. Every model reads it as a column. `Scale a feature by its spread
<https://github.com/hgilde/polars-online/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_
gives the measured gain. This code reads ``trades.parquet`` from the README's
`Example data <https://github.com/hgilde/polars-online#example-data>`_:

.. code-block:: python

    spread = {"half_life": "30m"}
    z_mid = (pl.col("mid") - po.ewm_mean("mid", **spread)) / po.ewm_std("mid", **spread)
    zscored = pl.scan_parquet("trades.parquet").online.with_windows(
        z_mid=z_mid, clock="ts", gap_cap="5m", group="symbol"
    ).collect()

**The clock settings are given to the call**, and every operator in it
shares them: ``clock``, ``gap_cap``, ``restart_after_step_back``, ``session``,
``session_gap`` and ``group``, in a spec's own words
(:mod:`polars_online.spec`). They differ from a spec's in one way: a
session needs no ``session_gap``, since no window measures the clock's step
across a session change. A number given for it changes nothing, and only
``"reset"`` acts. A gap past ``gap_cap`` or a session change ends every
window open across it, and ``partial`` says what such a window gives. A
backward window whose span reaches before its stretch's first row is
partial too, the stream's own start included. So ``partial`` also says what
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

A reset discards a window: it gives null, and the row is never dropped.
Under ``session_gap="reset"`` a session change is a reset, so it discards
the windows rather than cuts them. A forward window under ``closed="right"``
or ``"both"`` whose far edge is the last row before a cut or a reset is
whole: every row it covers has arrived, so it gives its value. A forward
window still open when the input ends is null. Under ``group`` a forward
window needs a clock column, since only a clock ends the windows of a group
that falls silent. ``partial`` needs a ``window_size``: without one no
window is cut short, so ``partial`` alone is refused.

An operator's input is a column name or an element-wise expression of the
row, :func:`increment` included, and never another window operator. For a
formula over an operator's output, add the inner operator as a column with
one call, and pass that column to a second. ``half_life`` and
``window_size`` are in the clock's units: numbers on a numeric clock, and
durations (``"10s"``, a ``timedelta``, ``pl.duration``) on a temporal one.
``half_life=float("inf")`` weighs the window evenly, and needs a
``window_size`` for a mean, a variance or a standard deviation.

An operator checks its own arguments when it is called, and names the one
it refuses. Each operator's *Raises* lists them.
:func:`polars_online.stream.with_windows` refuses the rest when it builds
its query: ``half_life=float("inf")`` with no ``window_size`` for a mean, a
variance or a standard deviation, and a window looking ahead under
``group`` with no clock column.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import polars as pl

from polars_online._errors import RefusedValueError
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
    who = f"po.{name}"
    if not isinstance(reverse, bool):
        raise TypeError(
            f"{who}: reverse must be True or False, but it is {reverse!r}. "
            "Pass reverse=True to read the rows after each row, or reverse=False."
        )
    if reverse and window_size is None:
        raise RefusedValueError(
            f"{who}: reverse=True needs window_size, but it is not set, and the rows after a "
            "row have no end without it. Set window_size to the span to read ahead."
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
    """The time-weighted EW mean of a column over each row's window.

    Use it for a trailing or leading level of a value, such as the mid's recent
    mean, measured on the stream's clock. For a sum or a quantity per unit
    time, use :func:`ewm_sum` or :func:`ewm_rate`, and for the spread about
    this mean, :func:`ewm_std`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly,
        and for this operator needs ``window_size``:
        :func:`~polars_online.stream.with_windows`, or a spec that takes it in
        a formula target, refuses it without one.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`polars_online.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`polars_online.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~polars_online.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~polars_online.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`polars_online.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = po.stream.with_windows(
            rows,
            back=po.ewm_mean("x", half_life=1.0),        # the rows at or before each row
            ahead=po.ewm_mean("x", half_life=1.0, window_size=2.0, reverse=True),   # next 2 units
            clock="t", gap_cap=10.0,
        )
        polars = rows.select(pl.col("x").ewm_mean_by(pl.col("t").cast(pl.Int64), half_life="1i"))
        print(out["back"].round(4).to_list(), polars["x"].round(4).to_list())   # equal
        print(out["ahead"].round(4).to_list())   # null where a window was open at the input's end

    .. code-block:: text

        [1.0, 1.5, 2.25, 3.125, 4.0625] [1.0, 1.5, 2.25, 3.125, 4.0625]
        [2.0, 3.0, None, None, None]

    .. rubric:: See also

    - :func:`ewm_std`, the spread about this mean.
    - :func:`ewm_sum`, for a weighted mean as a ratio of two sums.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.
    - `How a window operator weighs the rows near each row
      <https://github.com/hgilde/polars-online/blob/main/docs/WINDOWS.md>`_,
      the guide.

    .. rubric:: Notes

    Looking back, it is Polars' ``ewm_mean_by`` recursion,
    ``y_i = a_i x_i + (1 - a_i) y_{i-1}`` with
    ``a_i = 1 - lam ** (t_i - t_{i-1})``, ``lam = 2 ** (-1 / half_life)`` and
    ``y_1 = x_1``. Each value is weighed by the decayed time of the interval
    ending at its row. Under a ``window_size`` only the part of each interval
    inside the window counts.

    Looking ahead, it is the mirror: each value is held until the next row, and
    weighed by the decayed time from the row the window is read at.
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
    """The EW sum of a column over each row's window.

    Use it for a decayed total, such as recent notional, and for a weighted
    mean as a ratio of two sums. For a mean weighted by time, use
    :func:`ewm_mean`, and for a total per unit time, :func:`ewm_rate`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`polars_online.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`polars_online.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~polars_online.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~polars_online.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`polars_online.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = po.stream.with_windows(rows, s=po.ewm_sum("x", half_life=1.0), clock="t", gap_cap=10)
        polars = rows.select(pl.col("x").ewm_sum_by(pl.col("t").cast(pl.Int64), half_life="1i"))
        print(out["s"].round(4).to_list(), polars["x"].round(4).to_list())   # the same as Polars'

    .. code-block:: text

        [1.0, 2.5, 4.25, 6.125, 8.0625] [1.0, 2.5, 4.25, 6.125, 8.0625]

    .. rubric:: See also

    - :func:`ewm_rate`, the sum per unit of clock time.
    - :func:`ewm_mean`, the mean weighted by time.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.
    - `Make a weighted mean, such as a VWAP
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/make-window-features.md#make-a-weighted-mean-such-as-a-vwap>`_,
      the forward VWAP built from two sums.

    .. rubric:: Notes

    Looking back it is ``sum lam ** (tau_t - t_j) * x_j`` over the rows at or
    before row *t*, less than ``window_size`` older, with
    ``lam = 2 ** (-1 / half_life)`` and ``tau_t`` the row's time. Looking ahead
    it is ``sum lam ** (t_j - tau_t) * x_j`` over the rows after it, at most
    ``window_size`` later. On distinct stamps it is Polars' ``ewm_sum_by``,
    each value counted once at its own time. At a repeated stamp every row
    carries the stamp's total, where ``ewm_sum_by``'s is a running sum.
    Polars added ``ewm_sum_by`` in py-polars 1.43.0, so the example's
    comparison with it needs that version or later.
    """
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
    """The EW sum of a column per unit of clock time over each row's window.

    Use it for a quantity per unit time, such as a volume rate from
    :func:`increment` of a running volume. For the total itself, use
    :func:`ewm_sum`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`polars_online.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`polars_online.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value, and where the
    window spans no clock time, such as at a stretch's first stamp. It is
    null too where ``partial="null"`` meets a cut window, where a forward
    window is still open at the input's end, and after a reset (the
    module's ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~polars_online.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~polars_online.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`polars_online.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # half_life=inf weighs the window evenly: the sum over the last 2 units, per unit.
        rate = po.ewm_rate("x", half_life=float("inf"), window_size=2.0)
        out = po.stream.with_windows(rows, r=rate, clock="t", gap_cap=10.0)
        print(out["r"].round(4).to_list())   # null on row 0, whose window spans no time yet

    .. code-block:: text

        [None, 3.0, 2.5, 3.5, 4.5]

    .. rubric:: See also

    - :func:`ewm_sum`, the total it divides.
    - :func:`increment`, the steps of a running total.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.
    - `How a window operator weighs the rows near each row
      <https://github.com/hgilde/polars-online/blob/main/docs/WINDOWS.md>`_,
      the guide.

    .. rubric:: Notes

    It is :func:`ewm_sum` over the decayed time the window covers,
    ``half_life / ln 2 * (1 - 2 ** (-T / half_life))``, with ``T`` the window's
    span inside the stretch. Looking ahead it is the sum ahead over the decayed
    time the window ahead covers.
    """
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
    """The EW quantile of a column over each row's window.

    Use it for a level that one wild row does not move, such as a median, or
    for a robust scale (Notes). For a mean, use :func:`ewm_mean`, and for a
    standard deviation, :func:`ewm_std`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``quantile``
        The level of the quantile, above 0 and below 1, such as 0.5 for the
        median. Required.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`polars_online.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`polars_online.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~polars_online.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``;
        - ``quantile`` is not a number.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``quantile`` is not above 0 and below 1;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~polars_online.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`polars_online.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # The median of the last 3 units, weighed evenly.
        med = po.ewm_quantile("x", quantile=0.5, half_life=float("inf"), window_size=3.0)
        out = po.stream.with_windows(rows, med=med, clock="t", gap_cap=10.0)
        print(out["med"].round(3).to_list())   # within 0.78% of the exact medians 1, 1, 2, 3, 4

    .. code-block:: text

        [1.008, 1.008, 2.016, 3.016, 4.031]

    .. rubric:: See also

    - :func:`ewm_sum`, which weighs the rows the same way.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.
    - `Quantiles and a robust scale
      <https://github.com/hgilde/polars-online/blob/main/docs/WINDOWS.md#quantiles-and-a-robust-scale>`_,
      the guide's worked example.

    .. rubric:: Notes

    Each row of the window with a value counts once, at its own time, with the
    weight :func:`ewm_sum` gives it, ``w_j = lam ** |t_j - tau_t|``,
    ``lam = 2 ** (-1 / half_life)``. The quantile at ``q`` is the smallest
    ``x`` with ``W(<= x) >= q W``, where ``W(<= x)`` is the weight of the
    values at or below ``x`` and ``W`` the weight of all of them. That is
    numpy's ``inverted_cdf`` quantile over the window's values and weights.

    A value is read as the bucket it falls in, each bucket spanning a ratio of
    at most ``e ** (1/64)``. So the result is within ``tanh(1/128)``, 0.78%, of
    the exact quantile, relatively, at any scale and either sign. A magnitude
    below the smallest normal double counts as 0. A row whose value is missing
    counts for nothing and gives the window as it stands, as the sums do.

    **A robust scale.** The median of ``|r|`` is the median absolute
    deviation of ``r`` about 0. Divided by ``Φ⁻¹(3/4) = 0.6744897501960817``,
    it estimates a standard deviation for normal ``r``, and holds
    while up to half the window is wild, where a root mean square is moved by
    one row. For a residual, which the fit centers on 0::

        mad = po.ewm_quantile(pl.col("resid").abs(), quantile=0.5, half_life=500)
        scale = mad / 0.6744897501960817

    ``closed="left"`` leaves the row's own value out of its window. So a scale
    of a target, used as a weight on the same row, does not see the target it
    weighs.
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
    """The EW variance of a column about its time-weighted mean, over each row's
    window.

    Use it for the spread of a value as it stands on each row, looking back.
    For its square root, use :func:`ewm_std`. For a spread that one wild row
    does not move, use :func:`ewm_quantile` (its Notes).

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly,
        and for this operator needs ``window_size``:
        :func:`~polars_online.stream.with_windows`, or a spec that takes it in
        a formula target, refuses it without one.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`polars_online.ops`) gives each end,
        looking back.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"``. It needs ``window_size``, since without one no window
        is cut short.
    ``bias``
        ``False`` (the default) multiplies the variance by the correction for
        unequal weights, ``V1 ** 2 / (V1 ** 2 - V2)``, as Polars' ``ewm_var``
        does. ``True`` leaves it out.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`polars_online.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands. While one row carries
    all the weight it is null under the default ``bias=False``, as Polars'
    first row is, and 0 under ``bias=True``.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~polars_online.WrongTypeError`, also a ``ValueError``);
        - ``bias`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.

    :func:`polars_online.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = po.stream.with_windows(rows, v=po.ewm_var("x", half_life=1.0), clock="t", gap_cap=10)
        polars = rows.select(pl.col("x").ewm_var(half_life=1.0, adjust=False))   # 1 a row
        print(out["v"].round(4).to_list(), polars["x"].round(4).to_list())   # the same as Polars'

    .. code-block:: text

        [None, 0.5, 1.1, 1.6905, 2.1588] [None, 0.5, 1.1, 1.6905, 2.1588]

    .. rubric:: See also

    - :func:`ewm_std`, its square root.
    - :func:`ewm_mean`, the mean it is taken about.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.
    - `Scale a feature by its spread
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_,
      the guide.

    .. rubric:: Notes

    Each value weighs ``m_i``, the decayed mass of its held interval inside the
    window, as in :func:`ewm_mean`. The variance is
    ``sum m_i (x_i - mu) ** 2 / V1``, with ``mu = sum m_i x_i / V1`` and
    ``V1 = sum m_i``. Unless ``bias``, it is multiplied by
    ``V1 ** 2 / (V1 ** 2 - V2)``, with ``V2 = sum m_i ** 2``: the correction
    for unequal weights that Polars' ``ewm_var`` applies to its own.

    On a clock that steps by 1 a row, with ``half_life`` in rows, this is
    Polars' ``ewm_var(half_life=..., adjust=False, bias=...)``.
    ``adjust=False`` is the form :func:`ewm_mean` already mirrors, since
    ``ewm_mean_by`` holds a stretch's first value from before it. So the first
    row weighs ``(1 - alpha) ** (n - 1)``, with ``alpha = 1 - lam``, as
    ``adjust=False`` weighs it. A window that holds one value gives null under
    the default ``bias=False`` and 0 exactly, at any level, under ``bias=True``.
    Polars' ``ewm_var`` gives that row null too from py-polars 1.44, and 0
    before it, so on an older Polars the example's first value from Polars
    reads ``0.0``.
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
    """The EW standard deviation of a column, the square root of :func:`ewm_var`.

    Use it to put a feature in units of its current spread:
    ``(x - ewm_mean(x)) / ewm_std(x)`` is a z-score. For the variance itself,
    use :func:`ewm_var`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly,
        and for this operator needs ``window_size``:
        :func:`~polars_online.stream.with_windows`, or a spec that takes it in
        a formula target, refuses it without one.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`polars_online.ops`) gives each end,
        looking back.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"``. It needs ``window_size``, since without one no window
        is cut short.
    ``bias``
        ``False`` (the default) multiplies the variance by the correction for
        unequal weights, ``V1 ** 2 / (V1 ** 2 - V2)``, as Polars' ``ewm_var``
        does. ``True`` leaves it out.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`polars_online.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands. It is null where
    :func:`ewm_var` is.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~polars_online.WrongTypeError`, also a ``ValueError``);
        - ``bias`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.

    :func:`polars_online.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        z = (pl.col("x") - po.ewm_mean("x", half_life=1.0)) / po.ewm_std("x", half_life=1.0)
        sd = po.ewm_std("x", half_life=1.0)
        out = po.stream.with_windows(rows, sd=sd, z=z, clock="t", gap_cap=10.0)
        polars = rows.select(pl.col("x").ewm_std(half_life=1.0, adjust=False))   # 1 a row
        print(out["sd"].round(4).to_list(), polars["x"].round(4).to_list())   # the same as Polars'
        print(out["z"].round(4).to_list())   # x in units of its spread

    .. code-block:: text

        [None, 0.7071, 1.0488, 1.3002, 1.4693] [None, 0.7071, 1.0488, 1.3002, 1.4693]
        [None, 0.7071, 0.7151, 0.673, 0.6381]

    .. rubric:: See also

    - :func:`ewm_var`, whose square root it is, and whose Notes define it.
    - :func:`ewm_mean`, the mean the spread is taken about.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.
    - `Scale a feature by its spread
      <https://github.com/hgilde/polars-online/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_,
      the guide's measured gain.

    .. rubric:: Notes

    It is the square root of :func:`ewm_var`, as Polars' ``ewm_std`` is the
    root of its ``ewm_var``. Before py-polars 1.44, Polars gives a window of
    one value 0 rather than null, so the example's first value from Polars
    reads ``0.0`` there.
    """
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
    """The step of a column from its last value within the group and session.

    Use it to turn a running total, such as a cumulative volume, into the
    amounts each row added, before :func:`ewm_sum` or :func:`ewm_rate`. For a
    difference that is null on the row after a null too, use Polars'
    ``diff()``.

    .. rubric:: Parameters

    ``input``
        The running value: a column name, or an element-wise expression of the
        row. It may hold no window operator.

    .. rubric:: Returns

    A ``pl.Expr`` of ``x_i - x_{i-1}``, ``x_{i-1}`` the input's last value,
    that :func:`polars_online.stream.with_windows` computes as a column, or
    that a window operator takes as its input. It is a float, in seconds on a
    temporal column. It is null on the first row of each group (of the
    stream, without ``group``) and of each session, on a row whose input is
    null, and on a row whose clock steps back by more than
    ``restart_after_step_back``, which starts the stream over, in the row's
    group or on the stream across groups. A step back on the stream across
    groups starts every group over, so the next row of every group is null
    too.

    .. rubric:: Raises

    ``TypeError``
        ``input`` is neither a column name nor a ``pl.Expr``.
    ``ValueError``
        ``input`` holds a window operator, a part that is not element-wise, or
        a column whose name starts with ``@po:``, the prefix the operators' own
        columns carry.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, None, 3.0, 4.0]})
        out = po.stream.with_windows(rows, d=po.increment("x"), clock="t", gap_cap=10.0)
        print(out["d"].to_list())   # the null row is skipped: 3 steps from 1, the last value

    .. code-block:: text

        [None, None, 2.0, 1.0]

    .. rubric:: See also

    - :func:`ewm_sum` and :func:`ewm_rate`, which take its steps.
    - :func:`polars_online.stream.with_windows`, which computes the operators
      over a stream.

    .. rubric:: Notes

    A null input makes the row's increment null and is skipped, as the
    operators hold a value from the last valued row. So ``[1, null, 3]`` gives
    ``[null, null, 2]``, where ``diff()`` gives three nulls. On a time of day
    the step is in its seconds since midnight, so a step across midnight is
    negative, as Polars' ``diff`` on a ``Time`` gives.

    On an integer column, ``Int8`` to ``Int128`` or ``UInt8`` to ``UInt64``,
    the step is taken in integers, then made a float, rounded once. So a
    running count past ``2**53`` keeps its steps of 1.

    Give a sum or a rate the increments of a running total: the increments
    are the trades themselves. The difference of two decayed sums of the
    total loses precision instead. A window's notional of 1e6, read as the difference of two such
    sums near 1e10, loses about four of a double's sixteen digits.
    """
    return operator("increment", input)
