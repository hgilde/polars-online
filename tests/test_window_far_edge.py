"""A window's far-edge row: counted by its stamp, weighed by nothing.

A row exactly one window back from a row is in that row's window under
``closed="left"`` or ``"both"`` looking back, ``"right"`` or ``"both"``
looking ahead. Its held interval ends at the row, so its mass inside the
window is 0 and a mean (or a variance) whose only valued row it is, is
null. The operators once decided that row's membership from the two rows'
raw clocks but its mass from the policy time (the clock measured from the
stretch's first row, in seconds), where the same distance rounds a last
bit either way: the row weighed about 1e-17 or nothing depending on where
the stretch began, and a mean was its value or null.

The oracle here takes every edge from integer nanoseconds and every
interval's inside part from them, converted once to seconds, so it shares
no arithmetic with the operators' policy time (nor with ``loop`` in
``test_windows.py``, which differences floats).
"""

from __future__ import annotations

import math
from fractions import Fraction

import polars as pl
import pytest

import polars_online as po

TIER = "essential"

#: Milliseconds per stretch start: where the stretch begins moves the policy
#: time of every row after it by a rounding, not by a stamp.
STARTS_MS = (0, 3, 7, 50, 99)
#: The far-edge row's stamp, the window, and the half-lives, as the review
#: that found this swept them.
EDGES_MS = (100, 137, 400, 1500, 2999, 6000)
WINDOWS_MS = (200, 300, 700, 1100)
HALF_LIVES = (float("inf"), "700ms")

#: Each stream: a null start row, then rows at offsets from the edge row, each
#: `(windows, milliseconds)` for `windows * w + milliseconds`, with values.
BACKWARD = {
    "edge row only": (((0, 0), (1, 0), (1, 1)), (5.0, None, None)),
    "edge row and one inside": (
        ((0, 0), (Fraction(1, 2), 0), (1, 0), (1, 1)),
        (5.0, 7.0, None, None),
    ),
    "edge row and one valued on it": (((0, 0), (1, 0), (1, 1)), (5.0, 7.0, None)),
}
#: A forward stream puts the valued row one window after a null row, and a
#: row after it to close the null row's window.
FORWARD = {
    "edge row only": (((-1, 0), (0, 0), (0, 1)), (None, 5.0, None)),
    "edge row and one inside": (
        ((-1, 0), (Fraction(-1, 2), 0), (0, 0), (0, 1)),
        (None, 7.0, 5.0, None),
    ),
}


# --------------------------------------------------------------------------
# The definition, from integer nanoseconds


def _mass(h: float, d_ns: int) -> Fraction | float:
    """``integral_0^d lam**s ds`` for ``d`` nanoseconds: exact for ``h = inf``."""
    if d_ns <= 0:
        return Fraction(0)
    if math.isinf(h):
        return Fraction(d_ns, 10**9)
    d = d_ns / 1e9
    return -h / math.log(2) * math.expm1(-d * math.log(2) / h)


def _between(h: float, forward: bool, t: int, a: int, b: int) -> Fraction | float:
    """The decayed mass of ``[a, b]`` seen from ``t``, every distance an
    integer number of nanoseconds."""
    if b <= a:
        return Fraction(0)
    if forward:
        return _mass(h, b - t) - _mass(h, a - t)
    return _mass(h, t - a) - _mass(h, t - b)


def oracle(
    ts: list[int],
    xs: list[float | None],
    w: int,
    closed: str,
    forward: bool,
    h: float,
    stat: str,
) -> list[float | None]:
    """Each row's ``ewm_mean`` or ``ewm_var(bias=True)`` over a window of ``w``
    nanoseconds, ``min_samples=1``, ``partial="keep"``, on distinct stamps.

    A row is in a window by its stamp; a value weighs the decayed mass of its
    held interval inside the window (backward: from the previous valued row,
    or a window before its own row if it is the first; ahead: until the next
    valued row), so a row exactly one window away weighs nothing. A forward
    window no later row has passed the far edge of is unresolved: null."""
    n = len(ts)
    near = closed in (("left", "both") if forward else ("right", "both"))
    far = closed in (("right", "both") if forward else ("left", "both"))
    out: list[float | None] = []
    for a in range(n):
        if forward:
            members = [
                b
                for b in range(n)
                if (ts[b] - ts[a] >= 0 if near else ts[b] - ts[a] > 0)
                and (ts[b] - ts[a] <= w if far else ts[b] - ts[a] < w)
            ]
            if not any(
                (ts[b] - ts[a] > w) if far else (ts[b] - ts[a] >= w) for b in range(a + 1, n)
            ):
                out.append(None)
                continue
        else:
            members = [
                b
                for b in range(a + 1)
                if (b <= a if near else b < a)
                and ((ts[a] - ts[b] <= w) if far else (ts[a] - ts[b] < w))
            ]
        held: list[tuple[Fraction | float, float]] = []
        for b in members:
            x = xs[b]
            if x is None:
                continue
            if forward:
                nxt = next((c for c in range(b + 1, n) if xs[c] is not None), None)
                end = ts[nxt] if nxt is not None else ts[a] + w
                m = _between(h, True, ts[a], ts[b], min(end, ts[a] + w))
            else:
                prv = next((c for c in range(b - 1, -1, -1) if xs[c] is not None), None)
                start = ts[prv] if prv is not None else ts[b] - w
                m = _between(h, False, ts[a], max(start, ts[a] - w), ts[b])
            held.append((m, x))
        v1 = sum((m for m, _ in held), Fraction(0))
        if not held or not v1 > 0:
            out.append(None)
            continue
        mean = sum((m * Fraction(x) for m, x in held), Fraction(0)) / v1
        if stat == "mean":
            out.append(float(mean))
        else:
            var = sum((m * (Fraction(x) - mean) ** 2 for m, x in held), Fraction(0)) / v1
            out.append(float(var))
    return out


def _stretch(
    start_ms: int, edge_ms: int, w_ms: int, offsets: tuple, values: tuple
) -> tuple[list[int], list[float | None]]:
    """A stream's stamps in nanoseconds: the start row (null), then each
    offset from the edge row's stamp, `(windows, milliseconds)` as written."""
    ns = 10**6
    stamps = [start_ms * ns]
    for windows, ms in offsets:
        stamps.append(int(edge_ms * ns + windows * w_ms * ns + ms * ns))
    return stamps, [None, *values]


def _frame(ts: list[int], xs: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"t": pl.Series(ts, dtype=pl.Int64).cast(pl.Datetime("ns")), "x": xs},
        schema_overrides={"x": pl.Float64},
    )


def _close(got: list[float | None], want: list[float | None], what: str) -> None:
    assert len(got) == len(want), what
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        if a is None or b is None:
            assert a is None and b is None, f"{what}: row {i} is {a}, expected {b}"
        else:
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"{what}: row {i} is {a}, expected {b}"


# --------------------------------------------------------------------------
# The sweep


@pytest.mark.parametrize("h", HALF_LIVES, ids=["flat", "decay"])
@pytest.mark.parametrize("closed", ["both", "left"])
def test_a_backward_window_weighs_its_far_edge_row_at_nothing(closed: str, h: float | str) -> None:
    """Whichever row began the stretch, a mean and a ``bias=True`` variance over a
    window whose only valued row is exactly one window old are null."""
    h_s = h if isinstance(h, float) else 0.7
    cases = 0
    for name, (offsets, values) in BACKWARD.items():
        for w_ms in WINDOWS_MS:
            for a_ms in EDGES_MS:
                for s_ms in STARTS_MS:
                    # The stamps: start row (null), then the edge row at
                    # `a`, the offsets as written for the edge row at 0.
                    ts, xs = _stretch(s_ms, a_ms, w_ms, offsets, values)
                    frame = _frame(ts, xs)
                    got = po.stream.with_windows(
                        frame,
                        mean=po.ewm_mean(
                            "x", half_life=h, window_size=f"{w_ms}ms", closed=closed, partial="keep"
                        ),
                        var=po.ewm_var(
                            "x",
                            half_life=h,
                            window_size=f"{w_ms}ms",
                            closed=closed,
                            partial="keep",
                            bias=True,
                        ),
                        clock="t",
                        gap_cap="1h",
                    )
                    for stat in ("mean", "var"):
                        want = oracle(ts, xs, w_ms * 10**6, closed, False, h_s, stat)
                        _close(
                            got[stat].to_list(),
                            want,
                            f"{stat} {closed} h={h} {name}: start {s_ms} ms, edge {a_ms} ms, "
                            f"window {w_ms} ms",
                        )
                    cases += 1
    assert cases == len(BACKWARD) * len(WINDOWS_MS) * len(EDGES_MS) * len(STARTS_MS)


@pytest.mark.parametrize("h", HALF_LIVES, ids=["flat", "decay"])
@pytest.mark.parametrize("closed", ["both", "right"])
def test_a_forward_window_weighs_its_far_edge_row_at_nothing(closed: str, h: float | str) -> None:
    """The mirror, looking ahead: a valued row exactly one window after a row
    is in that row's window under ``"right"`` and ``"both"`` and weighs nothing."""
    h_s = h if isinstance(h, float) else 0.7
    for name, (offsets, values) in FORWARD.items():
        for w_ms in WINDOWS_MS:
            for a_ms in EDGES_MS:
                for s_ms in STARTS_MS:
                    # The edge row is the first after the start: offsets are
                    # from the valued row, so the window opens a window before.
                    ts, xs = _stretch(s_ms, a_ms + w_ms, w_ms, offsets, values)
                    got = po.stream.with_windows(
                        _frame(ts, xs),
                        mean=po.ewm_mean(
                            "x",
                            half_life=h,
                            window_size=f"{w_ms}ms",
                            closed=closed,
                            partial="keep",
                            reverse=True,
                        ),
                        clock="t",
                        gap_cap="1h",
                    )["mean"].to_list()
                    want = oracle(ts, xs, w_ms * 10**6, closed, True, h_s, "mean")
                    _close(
                        got,
                        want,
                        f"mean ahead {closed} h={h} {name}: start {s_ms} ms, valued row at "
                        f"{a_ms + w_ms} ms, window {w_ms} ms",
                    )


def test_the_reviews_input() -> None:
    """``t = [0, 400, 1500, 1501] ms``, ``x = [null, 5, null, null]``, a window of
    1100 ms under ``"both"``: the row at 1500 ms holds only the row at 400 ms,
    exactly a window back. Null, and null when the stretch begins at 3 ms."""
    for start in (0, 3):
        ts = [start * 10**6, 400 * 10**6, 1500 * 10**6, 1501 * 10**6]
        out = po.stream.with_windows(
            _frame(ts, [None, 5.0, None, None]),
            m=po.ewm_mean("x", half_life="1s", window_size="1100ms", closed="both"),
            clock="t",
            gap_cap="1h",
        )
        assert out["m"].to_list() == [None, 5.0, None, None], start


# --------------------------------------------------------------------------
# Number clocks


def _float_edge_cases() -> list[tuple[float, float, float]]:
    """``(start, edge, window)`` on a seconds clock where the edge row and the
    row a window later differ by exactly the window in doubles, so the row is
    exactly one window old by the clock's own subtraction, while the stretch's
    start moves the policy time's rounding."""
    cases = []
    for w in (0.2, 0.3, 0.7, 1.1):
        for a in (0.1, 0.137, 0.4, 1.5, 2.999, 6.0):
            for s in (0.0, 0.003, 0.007, 0.05):
                if (a + w) - a == w and a - s > 0:
                    cases.append((s, a, w))
    return cases


def test_a_number_clock_weighs_its_far_edge_row_at_nothing() -> None:
    """The same on a clock of floats: the row's age is the exact difference of
    the two stamps, compared with the window, whatever the stretch's start."""
    cases = _float_edge_cases()
    assert len(cases) > 30
    for closed in ("both", "left"):
        for h in HALF_LIVES[0], 0.7:
            for s, a, w in cases:
                frame = pl.DataFrame(
                    {"t": [s, a, a + w, a + w + 0.001], "x": [None, 5.0, None, None]},
                    schema_overrides={"x": pl.Float64},
                )
                out = po.stream.with_windows(
                    frame,
                    m=po.ewm_mean("x", half_life=h, window_size=w, closed=closed, partial="keep"),
                    v=po.ewm_var(
                        "x", half_life=h, window_size=w, closed=closed, partial="keep", bias=True
                    ),
                    clock="t",
                    gap_cap=1e9,
                )
                what = f"{closed} h={h}: start {s}, edge {a}, window {w}"
                assert out["m"].to_list()[2:] == [None, None], what
                assert out["v"].to_list()[2:] == [None, None], what
                assert out["m"][1] == (5.0 if closed == "both" else None), what
