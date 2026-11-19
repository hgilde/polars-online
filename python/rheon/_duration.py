"""Durations for the clock parameters of a spec (docs/PLAN.md task 88).

A parameter measured in clock units -- ``half_life``, ``gap_cap``,
``window_size`` and the rest -- is a plain number when the clock column is
numeric, and a duration when it is a ``Datetime``, ``Date`` or ``Duration``
column. A duration may be written three ways: a polars expression such as
``pl.duration(minutes=10)``, a :class:`datetime.timedelta`, or polars'
duration text such as ``"10m"``. The spec keeps it as text, the form a TOML
config writes and a state file stores; a ``timedelta`` or an expression is
written as that text here, by the Rust side's formatter, so the two sides
cannot disagree about it.

Nothing here imports Polars: a polars expression exists only once Polars is
imported, so :func:`_is_expr` asks ``sys.modules`` and the one function that
evaluates an expression imports Polars inside. The helpers that measure a
duration against a clock column's dtype are ``rheon/polars/_clock.py``.
"""

from __future__ import annotations

import math
import sys
from datetime import timedelta
from typing import TYPE_CHECKING, Any, TypeGuard

from rheon._rheon import format_duration, parse_duration

if TYPE_CHECKING:
    import polars as pl


class _Expr:
    """What ``Duration`` holds in place of ``pl.Expr`` at run time, where
    ``typing.get_type_hints`` reads the builders' annotations and Polars may
    be absent: a class no value is an instance of. Every check of a value
    against it asks :func:`_is_expr` instead (``_spec._matches``)."""


def _is_expr(value: Any) -> TypeGuard[pl.Expr]:
    """Whether ``value`` is a polars expression: never without Polars
    imported, since an expression is made by Polars."""
    pl = sys.modules.get("polars")
    return pl is not None and isinstance(value, pl.Expr)


#: What a clock parameter may be given as besides a number. A type checker
#: reads ``pl.Expr``; the run time reads :class:`_Expr`, so that the
#: builders' annotations resolve without Polars.
if TYPE_CHECKING:
    Duration = timedelta | str | pl.Expr
else:
    Duration = timedelta | str | _Expr

#: The infinity words, in lower case, and the number each names.
_INFINITY = {
    "inf": math.inf,
    "+inf": math.inf,
    "infinity": math.inf,
    "+infinity": math.inf,
    "-inf": -math.inf,
    "-infinity": -math.inf,
}

# The words a clock parameter takes in place of a number, which are not
# durations: infinity (no decay, no ceiling) and `session_gap`'s "reset".
_WORDS = frozenset({*_INFINITY, "nan", "reset"})


def infinity_as_number(value: Any) -> Any:
    """``value`` with each infinity word -- ``"inf"``, ``"+INF"``,
    ``"infinity"`` and the rest, in any case -- as the number it names, and
    anything else as it is; a list is taken value by value. A spec holds an
    infinity as a number, which is what a bank's ``specs`` read the Rust
    side's ``"inf"`` back as, so the two compare equal (review 2026-10-05,
    YA7: ``half_life="inf"`` did not)."""
    if isinstance(value, (list, tuple)):
        return [infinity_as_number(v) for v in value]
    if isinstance(value, str):
        return _INFINITY.get(value.strip().lower(), value)
    return value


def duration_text(value: Any, who: str, key: str) -> Any:
    """``value`` with every duration in it written as text. Numbers, the
    words above and ``None`` pass through; a list is taken value by value."""
    if isinstance(value, (list, tuple)):
        return [duration_text(v, who, key) for v in value]
    if isinstance(value, timedelta):
        return format_duration(_fits(value // timedelta(microseconds=1) * 1_000, who, key))
    if _is_expr(value):
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
    """The one duration an expression that reads no column evaluates to.
    Reached only with an expression, so Polars is already imported."""
    import polars as pl

    try:
        s = pl.select(expr).to_series()
    except Exception as e:  # a column it reads, an argument polars refuses
        raise TypeError(
            f"{who}: {key} must be a duration that reads no column, but evaluating it failed: "
            f'{e}. Give {key} as a duration such as pl.duration(minutes=10) or "10m".'
        ) from None
    if s.len() != 1 or not isinstance(s.dtype, pl.Duration):
        raise TypeError(
            f"{who}: {key} must be one duration, but the expression (a column computation, "
            f'such as pl.col("x") + 1) gives {s.len()} '
            f"value{'' if s.len() == 1 else 's'} of dtype {s.dtype}. Give {key} as one "
            f"duration, such as pl.duration(minutes=10)."
        )
    # The stored integer in the column's own unit, scaled in Python's exact
    # integers: `dt.total_nanoseconds()` wraps past 292 years, so 585 years
    # read as 384 ns (a property test found it, 2026-09-24).
    stored = s.cast(pl.Int64).item()
    if stored is None:
        raise ValueError(
            f"{who}: {key} is a null duration. Give {key} a duration, such as "
            f'pl.duration(minutes=10) or "10m".'
        )
    per = {"ns": 1, "us": 1_000, "ms": 1_000_000}[s.dtype.time_unit]
    return int(stored) * per


#: The longest duration a clock holds: nanoseconds in an i64.
_MAX_NS = 2**63 - 1


def _fits(ns: int, who: str, key: str) -> int:
    """``ns`` if a clock can hold it, else a refusal that names the parameter,
    in the words the text form's refusal uses."""
    if abs(ns) > _MAX_NS:
        raise ValueError(
            f"{who}: {key} is longer than 292 years, the most a clock can hold. "
            f"Give {key} a shorter duration."
        )
    return ns
