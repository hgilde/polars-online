"""A helper's duration measured against a clock column: its length in
nanoseconds when the clock is temporal, read from the column's Polars dtype.
The durations themselves, which need no Polars, are ``rheon/_duration.py``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl

from rheon._duration import duration_text
from rheon._errors import WrongTypeError
from rheon._rheon import format_duration, parse_duration


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
        raise WrongTypeError(
            f"{who}: clock column {clock!r} is a time of day, which starts again at "
            "midnight, so it cannot be a clock. Combine it with its date into a Datetime, "
            f'such as pl.col("date").dt.combine(pl.col({clock!r})).'
        )
    if not dtype.is_temporal():
        if duration:
            raise ValueError(
                f"{who}: {key} is a duration, but clock column {clock!r} is {dtype}, which "
                f"has no unit to measure it in. Use a numeric clock with {key} as a number of "
                f"its units, or a temporal clock with {key} as a duration."
            )
        return None
    if not duration and isinstance(dtype, pl.Date):
        raise ValueError(
            f"{who}: clock column {clock!r} is {dtype}, a temporal clock, so {key} must be "
            f"a duration, but it is {value!r}. Give {key} as a duration of whole days, "
            f'such as timedelta(days=1) or "1d".'
        )
    if not duration:
        raise ValueError(
            f"{who}: clock column {clock!r} is {dtype}, a temporal clock, so {key} must be "
            f"a duration, but it is {value!r}. Give {key} as a duration, such as "
            f'pl.duration(minutes=5), timedelta(minutes=5) or "5m".'
        )
    text = duration_text(value, who, key)
    try:
        ns = parse_duration(text)
    except ValueError as e:
        # The words a spec's clock parameter takes in place of a number pass
        # the text check, and none is a duration here: the parse says so,
        # under the helper's name and the parameter's (review 2026-10-05, YA6).
        raise ValueError(f"{who}: {key} {e}") from None
    if ns <= 0:
        raise ValueError(
            f"{who}: {key} must be above 0, but it is {text}. Give {key} a positive duration."
        )
    tick = _tick_ns(dtype)
    if ns % tick:
        raise ValueError(
            f"{who}: {key} is {text}, which is not a whole number of steps of clock column "
            f"{clock!r} ({dtype}, one step is {format_duration(tick)}), so the column cannot "
            f"hold the result. Give {key} as a multiple of {format_duration(tick)}."
        )
    return int(ns)
