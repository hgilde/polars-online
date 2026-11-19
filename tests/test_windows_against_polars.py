"""The window operators held to Polars itself, on a wider grid than
``test_windows.py``'s one stream: ``po.ewm_mean`` against ``ewm_mean_by``
over every datetime unit, half-lives from a microsecond to an hour, stamps
that repeat, nulls and two epochs; and ``po.ewm_sum(half_life=inf,
window_size=...)`` against ``rolling_sum_by`` under all four values of
``closed``. Polars is the reference, so what differs is stated:

- Polars gives null at a row whose input is null; the operators give the
  window there, which a null row does not move. The comparison is at rows
  with a value.
- ``rolling_sum_by`` gives 0 for a window that holds no row with a value, and
  so does ``po.ewm_sum`` under its default ``min_samples`` of 0, as Polars'
  default is; with ``min_samples=1`` or more both give null there.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import rheon as po
from polars_version import needs_polars

TIER = "essential"

UNITS = {"ns": 1, "us": 1_000, "ms": 1_000_000}
EPOCHS = {"1970": 0, "2024": 1_704_067_200 * 10**9}
HALF_LIVES = ["1ms", "37ms", "1s", "20s", "1h"]


def _stream(n: int, seed: int, unit: str, epoch: str, nulls: float, ties: float) -> pl.DataFrame:
    """Stamps in ``unit`` from ``epoch``, steps from none (a tie) to seconds,
    values with ``nulls`` of them null."""
    rng = np.random.default_rng(seed)
    scale = UNITS[unit]
    # Steps in whole units, never under one unit, so a stamp is exact.
    steps_ns = rng.choice([1, 3, 1_000, 40_000, 7_000_000, 300_000_000, 2_500_000_000], n)
    steps = np.maximum(steps_ns // scale, 1) * scale
    steps = np.where(rng.random(n) < ties, 0, steps)
    ns = EPOCHS[epoch] + np.cumsum(steps)
    x = rng.normal(size=n) * 3 + 1
    x[rng.random(n) < nulls] = np.nan
    return pl.DataFrame(
        {
            "ts": pl.Series(ns // scale, dtype=pl.Int64).cast(pl.Datetime(unit)),  # type: ignore[arg-type]
            "x": x,
        }
    ).with_columns(pl.col("x").fill_nan(None))


def _close(got: list[float | None], want: list[float | None], rows: list[bool], what: str) -> int:
    seen = 0
    for i, (a, b, ok) in enumerate(zip(got, want, rows, strict=True)):
        if not ok:
            continue
        assert a is not None and b is not None, f"{what} row {i}: {a} vs {b}"
        assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"{what} row {i}: {a} vs {b}"
        seen += 1
    return seen


@pytest.mark.parametrize("unit", list(UNITS))
@pytest.mark.parametrize("epoch", list(EPOCHS))
@pytest.mark.parametrize(("nulls", "ties"), [(0.0, 0.0), (0.15, 0.25), (0.4, 0.5)])
def test_the_mean_is_polars_ewm_mean_by(unit: str, epoch: str, nulls: float, ties: float) -> None:
    df = _stream(800, 17, unit, epoch, nulls, ties)
    valued = df["x"].is_not_null().to_list()
    for h in HALF_LIVES:
        out = po.stream.with_windows(
            df, m=po.ewm_mean("x", half_life=h), clock="ts", gap_cap="1000h"
        )
        ref = df.select(pl.col("x").ewm_mean_by("ts", half_life=h))["x"].to_list()
        seen = _close(out["m"].to_list(), ref, valued, f"{unit} {epoch} {h}")
        assert seen == sum(valued)


# `rolling_sum_by` over a column with a null is refused before 1.41.1, and
# gives null for an empty window before it where it now gives 0.
NEEDS_ROLLING = needs_polars("1.41.1", "rolling_sum_by over nulls and for empty windows")


@NEEDS_ROLLING
@pytest.mark.parametrize("unit", list(UNITS))
@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
@pytest.mark.parametrize(("nulls", "ties"), [(0.0, 0.0), (0.2, 0.3), (0.6, 0.0)])
@pytest.mark.parametrize("min_samples", [None, 1, 2])
def test_the_plain_sum_over_a_window_is_polars_rolling_sum_by(
    unit: str, closed: str, nulls: float, ties: float, min_samples: int | None
) -> None:
    """Every row, the empty windows too. Where a window holds no row with a
    value (no row at all, or only nulls) both give 0.0, under their default
    ``min_samples`` of 0 -- up to the dust Polars' sliding sum leaves, 4e-15
    measured, where the operator's is exactly 0.0 -- and both give null under
    ``min_samples=1`` or more, at the same rows."""
    df = _stream(800, 29, unit, "2024", nulls, ties)
    kw = {} if min_samples is None else {"min_samples": min_samples}
    empty = 0
    for w in ("5ms", "1s", "5s"):
        out = po.stream.with_windows(
            df,
            s=po.ewm_sum("x", half_life=math.inf, window_size=w, closed=closed, **kw),
            clock="ts",
            gap_cap="1000h",
        )["s"].to_list()
        ref = df.select(pl.col("x").rolling_sum_by("ts", window_size=w, closed=closed, **kw))[
            "x"
        ].to_list()
        held = df.select(
            pl.col("x")
            .is_not_null()
            .cast(pl.Int64)
            .rolling_sum_by("ts", window_size=w, closed=closed)
        )["x"].to_list()
        for i, (a, b, c) in enumerate(zip(out, ref, held, strict=True)):
            where = f"{unit} {closed} {w} min_samples={min_samples} row {i}: {a} vs {b}"
            if c == 0:
                empty += 1
                if min_samples in (None, 0):
                    assert a == 0.0 and b is not None and abs(b) < 1e-9, where
                else:
                    assert a is None and b is None, where
            elif c < (min_samples or 0):
                assert a is None and b is None, where
            else:
                assert a is not None and b is not None, where
                assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), where
    # The grid does reach windows with no value in them, wherever a window
    # can hold none: under "left" and "none" (the row's own is out) or with nulls.
    assert empty > 0 or (closed in ("right", "both") and nulls == 0.0)


@NEEDS_ROLLING
def test_a_window_with_no_row_sums_to_zero_as_polars_does() -> None:
    """The three cases of an empty window, spelled out on a small stream: no
    row in it (the first row's, under ``"left"``), only nulls in it, and rows
    with values; ``min_samples=1`` nulls the first two."""
    secs = [0, 1, 2, 10, 11, 12, 30]
    base = 1_704_067_200_000
    df = pl.DataFrame(
        {
            "ts": pl.Series([base + 1000 * s for s in secs], dtype=pl.Int64).cast(
                pl.Datetime("ms")
            ),
            "x": [1.0, None, None, None, 5.0, None, None],
        }
    )
    clock = {"clock": "ts", "gap_cap": "1000h"}
    for min_samples in (0, 1):
        got = po.stream.with_windows(
            df,
            s=po.ewm_sum(
                "x", half_life=math.inf, window_size="2s", closed="left", min_samples=min_samples
            ),
            **clock,
        )["s"].to_list()
        want = df.select(
            pl.col("x").rolling_sum_by("ts", "2s", closed="left", min_samples=min_samples)
        )["x"].to_list()
        # No row (rows 0, 3, 6), only nulls (rows 2, 5 hold none either way),
        # and a value (rows 1, 2, 5).
        assert got == want
        if min_samples == 0:
            assert got == [0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 0.0]
        else:
            assert got == [None, 1.0, 1.0, None, None, 5.0, None]
