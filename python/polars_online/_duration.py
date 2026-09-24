"""Durations for the clock parameters of a spec (docs/PLAN.md task 88).

A parameter measured in clock units -- ``halflife``, ``max_dclock``,
``window`` and the rest -- is a plain number when the clock column is
numeric, and a duration when it is a ``Datetime``, ``Date`` or ``Duration``
column. A duration may be written three ways: a polars expression such as
``pl.duration(minutes=10)``, a :class:`datetime.timedelta`, or polars'
duration text such as ``"10m"``. The spec keeps it as text, the form a TOML
config writes and a state file stores; a ``timedelta`` or an expression is
written as that text here, by the Rust side's formatter, so the two sides
cannot disagree about it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from polars_online._polars_online import format_duration, parse_duration

#: What a clock parameter may be given as besides a number.
Duration = timedelta | str | pl.Expr

# The words a clock parameter takes in place of a number, which are not
# durations: infinity (no decay, no ceiling) and `session_gap`'s "reset".
_WORDS = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan", "reset"})


def duration_text(value: Any, who: str, key: str) -> Any:
    """``value`` with every duration in it written as text. Numbers, the
    words above and ``None`` pass through; a list is taken value by value."""
    if isinstance(value, (list, tuple)):
        return [duration_text(v, who, key) for v in value]
    if isinstance(value, timedelta):
        return format_duration(_fits(value // timedelta(microseconds=1) * 1_000, who, key))
    if isinstance(value, pl.Expr):
        return format_duration(_fits(_nanoseconds(value, who, key), who, key))
    if isinstance(value, str):
        # The text names a grid's fields (`@h10m`), so no padding travels.
        value = value.strip()
        if value.lower() not in _WORDS:
            try:
                parse_duration(value)
            except ValueError as e:
                raise ValueError(f"{who}: {key} {e}") from None
    return value


def _nanoseconds(expr: pl.Expr, who: str, key: str) -> int:
    """The one duration an expression that reads no column evaluates to."""
    example = f"pl.duration(minutes=10) for {key}"
    try:
        s = pl.select(expr).to_series()
    except Exception as e:  # a column it reads, an argument polars refuses
        raise TypeError(
            f"{who}: {key} must be a duration that reads no column, such as {example}; "
            f"evaluating it failed: {e}"
        ) from None
    if s.len() != 1 or not isinstance(s.dtype, pl.Duration):
        raise TypeError(
            f"{who}: {key} must be one duration, such as {example}; the expression "
            f"gives {s.len()} value(s) of dtype {s.dtype}"
        )
    # The stored integer in the column's own unit, scaled in Python's exact
    # integers: `dt.total_nanoseconds()` wraps past 292 years, so 585 years
    # read as 384 ns (a property test found it, 2026-09-24).
    stored = s.cast(pl.Int64).item()
    if stored is None:
        raise ValueError(f"{who}: {key} is a null duration")
    per = {"ns": 1, "us": 1_000, "ms": 1_000_000}[s.dtype.time_unit]
    return int(stored) * per


#: The longest duration a clock holds: nanoseconds in an i64.
_MAX_NS = 2**63 - 1


def _fits(ns: int, who: str, key: str) -> int:
    """``ns`` if a clock can hold it, else a refusal that names the parameter,
    in the words the text form's refusal uses."""
    if abs(ns) > _MAX_NS:
        raise ValueError(f"{who}: {key} is longer than 292 years, the most a clock can hold")
    return ns


def _tick_ns(dtype: pl.DataType) -> int:
    """The smallest step a temporal column can take, in nanoseconds."""
    if isinstance(dtype, pl.Date):
        return 86_400 * 10**9
    unit = getattr(dtype, "time_unit", "ns")
    return {"ms": 10**6, "us": 10**3, "ns": 1}[unit]


def clock_nanoseconds(value: Any, dtype: pl.DataType, who: str, key: str, clock: str) -> int | None:
    """A helper's parameter measured against a clock column: its length in
    nanoseconds when the clock is temporal, and ``None`` when the clock is
    numeric and the value a plain number. Either mixture is refused, as a
    spec refuses it, and so is a duration that is not a whole number of the
    column's own steps, which adding to the column would cut short."""
    duration = isinstance(value, (timedelta, str, pl.Expr))
    if isinstance(dtype, pl.Time):
        raise ValueError(
            f"{who}: clock column {clock!r} is a time of day, which starts again at "
            "midnight, so it cannot be a clock"
        )
    if not dtype.is_temporal():
        if duration:
            raise ValueError(
                f"{who}: {key} is a duration, but clock column {clock!r} is {dtype}, which "
                f"has no unit to measure it; give {key} as a number of the clock's own units"
            )
        return None
    if not duration:
        raise ValueError(
            f"{who}: clock column {clock!r} is {dtype}, a temporal clock, so {key} must be "
            f'a duration, such as pl.duration(minutes=5), timedelta(minutes=5) or "5m"; '
            f"got {value!r}"
        )
    text = duration_text(value, who, key)
    ns = parse_duration(text)
    if ns <= 0:
        raise ValueError(f"{who}: {key} must be > 0, got {text}")
    tick = _tick_ns(dtype)
    if ns % tick:
        raise ValueError(
            f"{who}: {key} is {text}, which is not a whole number of steps of clock column "
            f"{clock!r} ({dtype}, one step is {format_duration(tick)}); the column cannot "
            "hold the result"
        )
    return int(ns)
