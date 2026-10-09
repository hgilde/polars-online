"""``po.stream.with_windows``: formulas over the window operators, one pass
over a stream (docs/PLAN.md tasks 78, 143 and 144).

Each operator is held to its definition: the time-weighted mean to Polars'
``ewm_mean_by`` recursion and the decayed sum to ``ewm_sum_by``, written out
by hand here from the definition and, where the installed Polars has them,
to Polars itself as a second opinion; the windowed and forward forms, which
no library has, to a brute-force loop from the same definition and to the
time-reversal identity. The rest is the call: compositions on either side
of a Polars operator, what is refused and by what name, the clock policy
and ``like=``, chunk invariance, resuming, and the user's own case --
interleaved trades and quotes, three VWAPs as ratios of two sums.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online._formula import FormulaError, from_tree, to_tree
from polars_version import INSTALLED, needs_polars

TIER = "mixed"

#: Polars' own windowed sums and means are this file's reference, and their
#: results moved on the way from the floor of the declared range, 1.34.0, to
#: the version this repository pins. Measured on the releases between them,
#: 2026-10-07: before 1.41.1 `rolling_sum_by` over a time column refuses a
#: column with a null, or gives null for an empty window where it now gives
#: 0, and a value at rows where it now gives null; before 1.37.0
#: `rolling_mean_by` over an integer index refuses a column with a null;
#: `ewm_sum_by` arrived in 1.43.0.
NEEDS_ROLLING_NULLS = needs_polars(
    "1.41.1", "Polars' rolling_sum_by, the reference, gives other empty-window and null results"
)

CLOCK = {"clock": "t", "gap_cap": 1e9}

#: What a refusal raised while the plan runs arrives as. py-polars 1.x wraps
#: an exception raised inside a Python IO source as its own `ComputeError`;
#: 2.0.0rc2 lets it through unwrapped, so the run's `ValueError` arrives as
#: itself (the canary of 2026-10-05). Both carry the same message, as in
#: `test_frame._REFUSAL`.
_REFUSAL = (pl.exceptions.ComputeError, ValueError)


# --------------------------------------------------------------------------
# The definitions, by hand


def _lam(h: float) -> float:
    return 1.0 if math.isinf(h) else 2.0 ** (-1.0 / h)


def _mass(h: float, d: float) -> float:
    """``integral_0^d lam**s ds``."""
    if d <= 0:
        return 0.0
    # The closed form by expm1, as the core takes it (task 159, W2).
    return d if math.isinf(h) else -h / math.log(2) * math.expm1(-d * math.log(2) / h)


def _between(h: float, forward: bool, t: float, a: float, b: float) -> float:
    """``integral_a^b lam**(t - s) ds`` back, ``integral_a^b lam**(s - t) ds`` ahead."""
    if b <= a:
        return 0.0
    if forward:
        return _mass(h, b - t) - _mass(h, a - t)
    return _mass(h, t - a) - _mass(h, t - b)


def recursion_mean(t: list[float], x: list[float], h: float) -> list[float]:
    """Polars' ``ewm_mean_by``: ``y_i = a_i x_i + (1 - a_i) y_{i-1}``, ``a_i = 1 -
    lam**(t_i - t_{i-1})``, ``y_1 = x_1``, rows with no value skipped."""
    out: list[float] = []
    y = math.nan
    last_t = math.nan
    for ti, xi in zip(t, x, strict=True):
        if xi is None or math.isnan(xi):
            out.append(y)
            continue
        if math.isnan(y):
            y = xi
        else:
            a = 1.0 - _lam(h) ** (ti - last_t)
            y = a * xi + (1.0 - a) * y
        last_t = ti
        out.append(y)
    return out


def recursion_sum(t: list[float], x: list[float], h: float) -> list[float]:
    """Polars' ``ewm_sum_by``: ``y_i = x_i + lam**(t_i - t_{i-1}) y_{i-1}``."""
    out: list[float] = []
    y = 0.0
    last_t = None
    for ti, xi in zip(t, x, strict=True):
        if last_t is not None:
            y *= _lam(h) ** (ti - last_t)
        last_t = ti
        if xi is not None and not math.isnan(xi):
            y += xi
        out.append(y)
    return out


def loop(
    df: pl.DataFrame,
    op: str,
    value: str,
    *,
    half_life: float,
    window_size: float | None = None,
    closed: str = "right",
    min_samples: int = 1,
    clock: str = "t",
    group: str | None = None,
    bias: bool = False,
) -> list[float | None]:
    """Every row's operator from the definition, on a stream with no clock
    event: each group's rows, in stream order, at their own clock. A forward
    window not closed by the end is unresolved: null.

    A variance weighs each value by the mean's weight, its held interval's
    decayed mass ``m_i``, and is taken in two passes: the mean ``mu = sum
    m_i x_i / sum m_i``, then ``sum m_i (x_i - mu) ** 2 / sum m_i``; unless
    ``bias``, times ``V1 ** 2 / (V1 ** 2 - V2)`` with ``V1 = sum m_i`` and
    ``V2 = sum m_i ** 2``, the correction Polars' ``ewm_var`` applies to its
    own unequal weights, null where ``V1 ** 2 <= V2``."""
    forward = op.startswith("rewm")
    stat = op.split("_")[1]
    weighs_held = stat in ("mean", "var", "std")
    t = df[clock].to_list()
    v = [math.nan if x is None else float(x) for x in df[value].to_list()]
    g = df[group].to_list() if group is not None else [None] * df.height
    rows: dict[Any, list[int]] = defaultdict(list)
    for i, key in enumerate(g):
        rows[key].append(i)
    near = closed in ("right", "both") if not forward else closed in ("left", "both")
    far = closed in ("left", "both") if not forward else closed in ("right", "both")
    out: list[float | None] = [None] * df.height
    for idx in rows.values():
        n = len(idx)
        tau = [t[j] - t[idx[0]] for j in idx]
        for a in range(n):
            ta = tau[a]
            if forward:
                assert window_size is not None
                w = window_size
                # A set of timestamps: under "left" and "both" every row at
                # the row's stamp, itself included; under "right" and "none"
                # none at it (review R1, C3).
                members = [
                    b
                    for b in range(n)
                    if (tau[b] - ta >= 0 if near else tau[b] - ta > 0)
                    and (tau[b] - ta <= w if far else tau[b] - ta < w)
                ]
                if not any(
                    (tau[b] - ta > w) if far else (tau[b] - ta >= w) for b in range(a + 1, n)
                ):
                    continue
                span = w
            else:
                w = window_size
                members = [
                    b
                    for b in range(n)
                    if ((b <= a or tau[b] == ta) if near else (b < a and tau[b] != ta))
                    and (w is None or ((ta - tau[b] <= w) if far else (ta - tau[b] < w)))
                ]
                span = ta if w is None else min(w, ta)
            s = mass = count = 0.0
            held: list[tuple[float, float]] = []
            for b in members:
                xb = v[idx[b]]
                if math.isnan(xb):
                    continue
                count += 1
                tb = tau[b]
                if weighs_held:
                    # A value is held from the previous valued row to its own,
                    # or until the next valued row ahead: ewm_mean_by skips a
                    # null.
                    if forward:
                        nxt = next((c for c in range(b + 1, n) if not math.isnan(v[idx[c]])), None)
                        end = math.inf if nxt is None else tau[nxt]
                        m = _between(half_life, True, ta, tb, min(end, ta + w))
                    else:
                        prv = next(
                            (c for c in range(b - 1, -1, -1) if not math.isnan(v[idx[c]])), None
                        )
                        if prv is None:
                            start = tau[0] - w if w is not None else -math.inf
                        else:
                            start = tau[prv]
                        edge = -math.inf if w is None else ta - w
                        m = _between(half_life, False, ta, max(start, edge), tb)
                    s += m * xb
                    mass += m
                    held.append((m, xb))
                else:
                    s += _lam(half_life) ** abs(tb - ta) * xb
            if count < min_samples:
                continue
            if stat == "mean":
                out[idx[a]] = s / mass if mass > 0 else None
            elif stat in ("var", "std"):
                out[idx[a]] = _two_pass_var(held, bias, stat == "std")
            elif stat == "sum":
                out[idx[a]] = s
            else:
                m = _mass(half_life, span)
                out[idx[a]] = s / m if m > 0 else None
    return out


def _two_pass_var(held: list[tuple[float, float]], bias: bool, std: bool) -> float | None:
    """The weighted variance of ``held``'s ``(weight, value)`` pairs, in two
    passes, corrected as ``loop`` says unless ``bias``; its root if ``std``."""
    v1 = sum(m for m, _ in held)
    if not v1 > 0:
        return None
    mu = sum(m * x for m, x in held) / v1
    var = sum(m * (x - mu) ** 2 for m, x in held) / v1
    if not bias:
        # V1 ** 2 - V2 = 2 sum_{i<j} m_i m_j: each weight times the ones
        # before it, every term at or above 0, so nothing cancels.
        before = pairs = 0.0
        for m, _ in held:
            pairs += m * before
            before += m
        if not pairs > 0:
            return None
        var *= v1 * v1 / (2.0 * pairs)
    return math.sqrt(var) if std else var


def assert_close(
    got: list[float | None], want: list[float | None], what: str = "", tol: float = 1e-9
) -> None:
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        if a is None or b is None:
            assert a is None and b is None, f"{what} row {i}: {a} vs {b}"
        else:
            assert abs(a - b) <= tol * max(1.0, abs(a), abs(b)), f"{what} row {i}: {a} vs {b}"


def ticks(n: int, seed: int, *, groups: int = 1, repeats: bool = True) -> pl.DataFrame:
    """Irregular stamps with bursts, gaps and -- with ``repeats`` -- repeated
    stamps, one value column with a few nulls, and a group column."""
    rng = np.random.default_rng(seed)
    steps = (
        rng.choice([0.0, 0.5, 1.0, 1.0, 2.0, 3.0, 6.0], n) if repeats else rng.uniform(0.3, 2.0, n)
    )
    t = np.cumsum(steps)
    x = rng.normal(size=n) * 3 + 1
    x[rng.random(n) < 0.12] = np.nan
    g = rng.integers(0, groups, n)
    return pl.DataFrame({"t": t, "x": x, "g": [f"g{k}" for k in g]}).with_columns(
        pl.col("x").fill_nan(None)
    )


OPS = ("ewm_mean", "rewm_mean", "ewm_sum", "rewm_sum", "ewm_rate", "rewm_rate")


# --------------------------------------------------------------------------
# Each operator against its definition


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_each_operator_matches_the_definition(op: str, closed: str) -> None:
    df = ticks(700, 3)
    fn = getattr(po, op)
    out = po.stream.with_windows(
        df, y=fn("x", half_life=4.0, window_size=7.0, closed=closed, min_samples=2), **CLOCK
    )
    want = loop(df, op, "x", half_life=4.0, window_size=7.0, closed=closed, min_samples=2)
    assert_close(out["y"].to_list(), want, f"{op} {closed}")
    assert sum(v is not None for v in want) > 500


@pytest.mark.parametrize("op", ["ewm_mean", "ewm_sum", "ewm_rate"])
def test_a_backward_operator_without_a_window_runs_over_the_stretch(op: str) -> None:
    df = ticks(400, 5, repeats=False)
    out = po.stream.with_windows(df, y=getattr(po, op)("x", half_life=3.0), **CLOCK)
    assert_close(out["y"].to_list(), loop(df, op, "x", half_life=3.0), op)


def test_the_mean_and_the_sum_are_the_recursions() -> None:
    """``ewm_mean`` is ``ewm_mean_by``'s recursion with ``a_1 = 1``; ``ewm_sum``
    is ``ewm_sum_by``'s. On distinct stamps the recursions are the
    definitions row by row."""
    df = ticks(500, 8, repeats=False)
    out = po.stream.with_windows(
        df, m=po.ewm_mean("x", half_life=2.5), s=po.ewm_sum("x", half_life=2.5), **CLOCK
    )
    t = df["t"].to_list()
    x = [math.nan if v is None else v for v in df["x"].to_list()]
    m = recursion_mean(t, x, 2.5)
    got = out["m"].to_list()
    for i, (a, b) in enumerate(zip(got, m, strict=True)):
        if math.isnan(b):
            assert a is None, i
        else:
            assert a is not None and abs(a - b) < 1e-9, i
    assert_close(out["s"].to_list(), recursion_sum(t, x, 2.5), "sum")


def test_polars_computes_the_same_mean_and_sum() -> None:
    """A second opinion, on a Datetime clock: Polars' own ``ewm_mean_by`` and,
    where this Polars has it (1.44.1 on), ``ewm_sum_by``."""
    df = ticks(600, 11, repeats=False).with_columns(
        ts=pl.from_epoch((pl.col("t") * 1e9).cast(pl.Int64), time_unit="ns")
    )
    out = po.stream.with_windows(
        df,
        m=po.ewm_mean("x", half_life="2s500ms"),
        s=po.ewm_sum("x", half_life="2s500ms"),
        clock="ts",
        gap_cap="1000s",
    )
    # Polars gives null at a null row; this library gives the window there,
    # which a null row does not move. The valued rows are the comparison.
    valued = df["x"].is_not_null().to_list()
    ref = df.select(pl.col("x").ewm_mean_by("ts", half_life="2s500ms"))["x"].to_list()
    assert_close(
        [v for v, ok in zip(out["m"].to_list(), valued, strict=True) if ok],
        [v for v, ok in zip(ref, valued, strict=True) if ok],
        "ewm_mean_by",
    )
    assert sum(valued) > 500
    if hasattr(pl.Expr, "ewm_sum_by"):
        ref = df.select(pl.col("x").ewm_sum_by("ts", half_life="2s500ms"))["x"].to_list()
        assert_close(
            [v for v, ok in zip(out["s"].to_list(), valued, strict=True) if ok],
            [v for v, ok in zip(ref, valued, strict=True) if ok],
            "ewm_sum_by",
        )
    else:
        assert INSTALLED < (1, 43, 0), "ewm_sum_by arrived in 1.43.0"


@pytest.mark.parametrize(
    "closed",
    [
        "right",
        pytest.param("left", marks=NEEDS_ROLLING_NULLS),
        "both",
        pytest.param("none", marks=NEEDS_ROLLING_NULLS),
    ],
)
def test_which_rows_a_window_holds_is_polars_rolling(closed: str) -> None:
    """A window is a set of timestamps, as Polars' ``rolling_sum_by`` has it:
    rows at a repeated stamp share one window, and ``closed`` moves the
    ends. With ``half_life=inf`` the decayed sum is the plain sum."""
    t = [0, 1, 2, 2, 2, 3, 5, 5, 9]
    df = pl.DataFrame({"t": t, "x": [float(v) for v in range(1, 10)]}).with_columns(
        ts=pl.from_epoch(pl.col("t") * 1_000_000_000, time_unit="ns")
    )
    out = po.stream.with_windows(
        df, s=po.ewm_sum("x", half_life=math.inf, window_size=2.0, closed=closed), **CLOCK
    )
    ref = df.select(pl.col("x").rolling_sum_by("ts", window_size="2s", closed=closed))[
        "x"
    ].to_list()
    got = [0.0 if v is None else v for v in out["s"].to_list()]
    assert got == ref, closed


def test_min_samples_counts_rows_with_a_value() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, None, 2.0, 3.0]})
    out = po.stream.with_windows(
        df, s=po.ewm_sum("x", half_life=math.inf, window_size=3.0, min_samples=2), **CLOCK
    )
    assert out["s"].to_list() == [None, None, 3.0, 5.0]


def test_a_forward_window_is_a_backward_one_over_the_reversed_stream() -> None:
    """The mirror: times negated and reversed, the row itself excluded, so a
    forward ``closed="right"`` window is a backward ``"left"`` one there."""
    df = ticks(500, 13, repeats=False)
    fwd = po.stream.with_windows(
        df,
        m=po.rewm_mean("x", half_life=3.0, window_size=5.0),
        s=po.rewm_sum("x", half_life=3.0, window_size=5.0),
        **CLOCK,
    )
    last = df["t"][-1]
    mirror = df.reverse().with_columns(t=last - pl.col("t"))
    back = po.stream.with_windows(
        mirror,
        m=po.ewm_mean("x", half_life=3.0, window_size=5.0, closed="left"),
        s=po.ewm_sum("x", half_life=3.0, window_size=5.0, closed="left"),
        **CLOCK,
    ).reverse()
    for c in ("m", "s"):
        pairs = [
            (a, b)
            for a, b in zip(fwd[c].to_list(), back[c].to_list(), strict=True)
            if a is not None
        ]
        assert len(pairs) > 400
        for a, b in pairs:
            assert b is not None and abs(a - b) <= 1e-9 * (1 + abs(a)), c


# --------------------------------------------------------------------------
# Compositions, and the real case


def trades_and_quotes(n: int, seed: int, quotes_per_trade: int = 2) -> pl.DataFrame:
    """The user's case (2026-09-11): market-data rows interleaved with trades,
    ``side``, ``quantity`` and ``price`` null on the quotes, a trade often at
    the clock of the row before it; one trade in ``quotes_per_trade + 1``."""
    rng = np.random.default_rng(seed)
    trade = rng.random(n) < 1.0 / (quotes_per_trade + 1)
    steps = np.where(trade & (rng.random(n) < 0.5), 0, rng.choice([1, 2, 3], n))
    side = rng.choice(["buy", "sell"], n)
    return pl.DataFrame(
        {
            "t": np.cumsum(steps).astype(np.int64),
            "side": [s if k else None for s, k in zip(side, trade, strict=True)],
            "quantity": [
                float(q) if k else None for q, k in zip(rng.integers(1, 10, n), trade, strict=True)
            ],
            "price": [
                100 + float(p) if k else None
                for p, k in zip(rng.normal(size=n), trade, strict=True)
            ],
            "mid": 100 + rng.normal(size=n) * 0.01,
        }
    )


@pytest.mark.extended(reason="a second or more (1.1 s)")
def test_a_vwap_is_a_ratio_of_two_sums_and_a_side_is_a_when_inside_both() -> None:
    """Three forward VWAPs -- all trades, buys, sells -- as ratios of decayed
    sums, equal to the definition's ratio, unchanged by a trade split in two
    or a zero-volume row, and the buys' null where the window has no buy."""
    df = trades_and_quotes(3000, 7)
    notional = pl.col("price") * pl.col("quantity")
    h = {"half_life": 10.0, "window_size": 60.0}
    buys, sells = pl.when(pl.col("side") == "buy"), pl.when(pl.col("side") == "sell")
    out = po.stream.with_windows(
        df,
        vwap=po.rewm_sum(notional, **h) / po.rewm_sum("quantity", **h),
        buy_vwap=po.rewm_sum(buys.then(notional), **h) / po.rewm_sum(buys.then("quantity"), **h),
        sell_vwap=po.rewm_sum(sells.then(notional), **h) / po.rewm_sum(sells.then("quantity"), **h),
        **CLOCK,
    )
    pq = df.with_columns(pq=notional, bq=buys.then("quantity"), bpq=buys.then(notional))
    num = loop(pq, "rewm_sum", "pq", **h)
    den = loop(pq, "rewm_sum", "quantity", **h)
    want = [
        None if a is None or b is None or b == 0 else a / b for a, b in zip(num, den, strict=True)
    ]
    assert_close(out["vwap"].to_list(), want, "vwap")
    bnum = loop(pq, "rewm_sum", "bpq", **h)
    bden = loop(pq, "rewm_sum", "bq", **h)
    bwant = [
        None if a is None or b is None or b == 0 else a / b for a, b in zip(bnum, bden, strict=True)
    ]
    assert_close(out["buy_vwap"].to_list(), bwant, "buy vwap")
    assert out["buy_vwap"].null_count() > out["vwap"].null_count()
    # A print split in two, and a zero-volume row, change nothing.
    i = next(k for k in range(100, 3000) if df["side"][k] is not None)
    row = df[i]
    split = pl.concat(
        [
            df[:i],
            row.with_columns(quantity=pl.col("quantity") / 2),
            row.with_columns(quantity=pl.col("quantity") / 2),
            df[i + 1 :],
        ]
    )
    zero = pl.concat([df[:i], row.with_columns(quantity=pl.lit(0.0)), df[i:]])
    for other, label in ((split, "split"), (zero, "zero")):
        got = po.stream.with_windows(
            other, vwap=po.rewm_sum(notional, **h) / po.rewm_sum("quantity", **h), **CLOCK
        )
        assert_close(got["vwap"][:i].to_list(), out["vwap"][:i].to_list(), label)


def test_increments_give_a_vwap_from_running_sums_and_a_rate() -> None:
    """``ewm_sum(increment(C_pq)) / ewm_sum(increment(C_q))`` is the backward
    VWAP from prices and quantities where the sums step at the trades; and
    a rate of the running volume is the sum of its increments over the
    decayed time."""
    df = trades_and_quotes(2000, 9).with_columns(
        cum_q=pl.col("quantity").fill_null(0.0).cum_sum(),
        cum_pq=(pl.col("price") * pl.col("quantity")).fill_null(0.0).cum_sum(),
    )
    h = {"half_life": 10.0}
    out = po.stream.with_windows(
        df,
        vwap=po.ewm_sum(po.increment("cum_pq"), **h) / po.ewm_sum(po.increment("cum_q"), **h),
        rate=po.ewm_rate(po.increment("cum_q"), **h, window_size=30.0),
        dq=po.increment("cum_q"),
        **CLOCK,
    )
    pq = df.with_columns(pq=pl.col("price") * pl.col("quantity"))
    num = loop(pq, "ewm_sum", "pq", **h)
    den = loop(pq, "ewm_sum", "quantity", **h)
    want = [
        None if a is None or b is None or b == 0 else a / b for a, b in zip(num, den, strict=True)
    ]
    # The first row's increment is null, so its sums are empty there.
    assert_close(out["vwap"][1:].to_list(), want[1:], "vwap from running sums")
    dq = out["dq"].to_list()
    assert dq[0] is None and all(
        abs(a - b) < 1e-9 for a, b in zip(dq[1:], df["cum_q"].diff()[1:].to_list(), strict=True)
    )
    rate = loop(df.with_columns(dq=pl.col("cum_q").diff()), "ewm_rate", "dq", **h, window_size=30.0)
    assert_close(out["rate"].to_list(), rate, "rate")


def test_a_formula_composes_on_either_side_of_a_polars_operator() -> None:
    df = ticks(300, 21, repeats=False)
    out = po.stream.with_windows(
        df,
        a=po.ewm_mean("x", half_life=3.0) - pl.col("x"),
        b=(pl.col("x") / po.ewm_mean("x", half_life=3.0)).log().clip(-1.0, 1.0),
        c=pl.when(po.ewm_sum("x", half_life=3.0) > 0).then(pl.lit(1)).otherwise(pl.lit(-1)),
        d=po.ewm_mean("x", half_life=3.0).fill_null(0.0).cast(pl.Float32) ** 2,
        **CLOCK,
    )
    base = po.stream.with_windows(
        df, m=po.ewm_mean("x", half_life=3.0), s=po.ewm_sum("x", half_life=3.0), **CLOCK
    )
    want = base.select(
        a=pl.col("m") - pl.col("x"),
        b=(pl.col("x") / pl.col("m")).log().clip(-1.0, 1.0),
        c=pl.when(pl.col("s") > 0).then(pl.lit(1)).otherwise(pl.lit(-1)),
        d=pl.col("m").fill_null(0.0).cast(pl.Float32) ** 2,
    )
    assert out.select("a", "b", "c", "d").equals(want)
    assert out.schema["c"] == pl.Int32 and out.schema["d"] == pl.Float32


def test_one_operator_asked_for_twice_is_computed_once() -> None:
    """Identical operators share a column; operators on one kernel share a
    queue, so the queued rows do not grow with the operators."""
    df = ticks(200, 4, repeats=False)
    bank = {"half_life": 2.0, "window_size": 5.0}
    one = po.stream.with_windows(df, a=po.ewm_sum("x", **bank), **CLOCK)
    four = po.stream.with_windows(
        df,
        a=po.ewm_sum("x", **bank),
        b=po.ewm_sum("x", **bank) * 2,
        c=po.ewm_mean("x", **bank),
        d=po.ewm_rate("x", **bank),
        **CLOCK,
    )
    assert four["a"].equals(one["a"]) and four.select(pl.col("b") == 2 * pl.col("a"))["b"].all()


def test_the_tree_round_trips_every_node_kind() -> None:
    """The compact tree in both directions, for every node kind, under this
    Polars (the floor, 1.34.0, has the same shapes: measured 2026-10-02)."""
    c, d = pl.col("a"), pl.col("b")
    exprs = [
        c - d,
        c / d,
        c * d,
        c + d,
        c**2,
        -c,
        c.log(),
        c.exp(),
        c.abs(),
        c.sqrt(),
        c.clip(0, 1),
        c.clip(lower_bound=0),
        c.clip(upper_bound=1.0),
        (c > 0) & d.is_null(),
        (c <= 1) | d.is_not_null(),
        c - 1.5,
        c - 1,
        c == "x",
        c != 2,
        c >= d,
        pl.when(c > 0).then(d).otherwise(None),
        pl.when(c > 0).then(1).otherwise(pl.lit(True)),
        c.cast(pl.Float64),
        c.cast(pl.Int32),
        c.fill_null(0.0),
        c.alias("z"),
        po.ewm_mean("a", half_life="10s", window_size="1m") - c,
        po.rewm_sum(
            c * d, half_life=5.0, window_size=60.0, closed="both", min_samples=3, partial="keep"
        ),
        po.ewm_rate(po.increment("a"), half_life=30.0),
        po.increment(c + d),
    ]
    for e in exprs:
        tree = to_tree(e)
        again = to_tree(from_tree(tree))
        assert again == tree, e
    # The Rust side reads the same trees: a run accepts every one.
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "a": [1.0, 2.0, 3.0], "b": [0.5, 1.5, 2.5]})
    po.stream.with_windows(
        df,
        *[e.alias(f"f{i}") for i, e in enumerate(exprs[-3:])],
        clock="t",
        gap_cap=100.0,
    )


def test_what_is_not_element_wise_is_refused_by_name() -> None:
    df = ticks(50, 1)
    for expr, says in [
        (po.ewm_mean("x", half_life=2.0).shift(1), "Shift"),
        (po.ewm_mean("x", half_life=2.0).cum_sum(), "CumSum"),
        (po.ewm_mean("x", half_life=2.0).mean(), "Agg"),
        (po.ewm_mean("x", half_life=2.0).rolling_mean(3), "Rolling"),
        # Refused by the node's own name, which Polars spells `Window` before
        # 1.36.1 (measured on 1.34.0 and 1.35.1; 1.36.0 cannot be installed).
        (
            po.ewm_mean("x", half_life=2.0).over("g"),
            "Over" if INSTALLED >= (1, 36, 1) else "Window",
        ),
        # `Date` is read since review 2026-10-05 (YB1), a date literal's dtype.
        (po.ewm_mean("x", half_life=2.0).cast(pl.Datetime("us")), "a cast to Datetime"),
        (~(po.ewm_mean("x", half_life=2.0) > 0), "Not"),
    ]:
        with pytest.raises(FormulaError, match=says):
            po.stream.with_windows(df, y=expr, **CLOCK)
    with pytest.raises(ValueError, match="not another window operator"):
        po.ewm_mean(po.ewm_sum("x", half_life=2.0), half_life=2.0)
    with pytest.raises(ValueError, match="holds no operator"):
        po.stream.with_windows(df, y=pl.col("x") * 2, **CLOCK)
    with pytest.raises(ValueError, match="needs a name"):
        po.stream.with_windows(df, po.ewm_mean("x", half_life=2.0), **CLOCK)
    assert (
        "y"
        in po.stream.with_windows(df, po.ewm_mean("x", half_life=2.0).alias("y"), **CLOCK).columns
    )


def test_the_calls_own_keywords_are_not_output_names() -> None:
    """3(a), the user, 2026-10-02: the clock policy's keywords are reserved."""
    df = ticks(50, 1)
    for key in ("clock", "gap_cap", "restart_after_step_back", "session", "session_gap", "group"):
        with pytest.raises(TypeError, match=f"{key} is a clock keyword"):
            po.stream.with_windows(
                df,
                **{key: po.ewm_mean("x", half_life=2.0)},
                **{k: v for k, v in CLOCK.items() if k != key},
            )
    with pytest.raises(TypeError):
        po.stream.with_windows(df, like=po.ewm_mean("x", half_life=2.0), **CLOCK)  # type: ignore[arg-type]
    out = po.stream.with_windows(df, po.ewm_mean("x", half_life=2.0).alias("session"), **CLOCK)
    assert "session" in out.columns


def test_an_operators_parameters_are_checked_by_name() -> None:
    for kw, says in [
        (dict(half_life=0), "half_life must be above 0"),
        (dict(half_life=-1.0), "half_life must be above 0"),
        (dict(half_life=math.nan), "must not be NaN"),
        (dict(half_life="x"), "half_life"),
        (dict(half_life=2.0, window_size=math.inf), "window_size must be finite"),
        (dict(half_life=2.0, closed="up"), "closed must be"),
        (dict(half_life=2.0, min_samples=0), "min_samples must be"),
        (dict(half_life=2.0, partial="maybe"), "partial must be"),
    ]:
        with pytest.raises((ValueError, TypeError), match=says):
            po.ewm_mean("x", **kw)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="window_size"):
        po.rewm_mean("x", half_life=2.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="half_life"):
        po.ewm_mean("x")  # type: ignore[call-arg]
    df = ticks(50, 1)
    with pytest.raises(ValueError, match="half_life = inf needs a window_size"):
        po.stream.with_windows(df, y=po.ewm_mean("x", half_life=math.inf), **CLOCK)
    with pytest.raises(ValueError, match="input must be a number"):
        po.stream.with_windows(df, y=po.ewm_mean("g", half_life=2.0), **CLOCK)
    with pytest.raises(ValueError, match="no input column"):
        po.stream.with_windows(df, y=po.ewm_mean("nope", half_life=2.0), **CLOCK)
    with pytest.raises(ValueError, match="already a column"):
        po.stream.with_windows(df, x=po.ewm_mean("x", half_life=2.0), **CLOCK)


# --------------------------------------------------------------------------
# The clock policy


def events(**kw: Any) -> dict[str, Any]:
    return {"clock": "t", "gap_cap": 10.0, "session": "s", "session_gap": 1.0} | kw


def event_stream() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "s": ["a"] * 4 + ["a"] * 3 + ["b"] * 4,
        }
    )


def test_a_gap_past_the_cap_and_a_session_change_cut_the_windows() -> None:
    """Row 4 opens with a gap past the cap: the forward windows of rows 0..3
    are cut -- kept, null or dropped as ``partial`` says -- and the backward
    windows start over; row 7's session change does the same."""
    df = event_stream()
    out = po.stream.with_windows(
        df,
        keep=po.rewm_mean("x", half_life=1.0, window_size=5.0, partial="keep"),
        null=po.rewm_mean("x", half_life=1.0, window_size=5.0, partial="null"),
        back=po.ewm_sum("x", half_life=math.inf, window_size=100.0),
        **events(),
    )
    assert out["keep"][0] is not None and out["null"][0] is None
    assert out["keep"][3] is None, "a window with no row after it is null even when kept"
    assert out["back"].to_list() == [1, 3, 6, 10, 5, 11, 18, 8, 17, 27, 38]
    dropped = po.stream.with_windows(
        df, d=po.rewm_mean("x", half_life=1.0, window_size=5.0, partial="drop"), **events()
    )
    # The four rows the gap cut and the three the session change cut leave;
    # the rows whose windows the input's end holds stay. Once `or height <
    # 11`, true of any drop at all (review 2026-10-05, TB9).
    assert dropped["t"].to_list() == [33.0, 34.0, 35.0, 36.0]


def test_a_reset_discards_and_a_late_row_is_refused() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 1.5, 2.5, 3.5], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    with pytest.raises(_REFUSAL, match="goes backwards by 0.5 at row 3"):
        po.stream.with_windows(df, y=po.ewm_sum("x", half_life=1.0), **CLOCK)
    with pytest.raises(_REFUSAL, match="no more than restart_after_step_back = 1"):
        po.stream.with_windows(
            df, y=po.ewm_sum("x", half_life=1.0), restart_after_step_back=1.0, **CLOCK
        )
    out = po.stream.with_windows(
        df,
        f=po.rewm_sum("x", half_life=math.inf, window_size=10.0, partial="keep"),
        b=po.ewm_sum("x", half_life=math.inf),
        restart_after_step_back=0.0,
        **CLOCK,
    )
    assert out["f"][:3].to_list() == [None, None, None], "discarded: null whatever partial says"
    assert out["b"].to_list() == [1, 3, 6, 4, 9, 15]


def test_a_silent_group_holds_the_output_for_at_most_the_cap() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 20.0, 21.0, 40.0],
            "g": ["a", "b", "b", "b", "b", "a"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    out = po.stream.with_windows(
        df,
        f=po.rewm_sum("x", half_life=math.inf, window_size=100.0, partial="keep"),
        clock="t",
        gap_cap=10.0,
        group="g",
    )
    # Group a's row 0 was cut once the stream passed t = 10: its window saw
    # nothing of its own group after it.
    assert out["f"][0] is None and out["t"].to_list() == df["t"].to_list()


def test_like_takes_the_specs_clock_and_its_rule() -> None:
    df = trades_and_quotes(400, 3).with_columns(w=pl.lit(1.0))
    spec = po.spec.ewridge(
        "m", targets=["mid"], features=["price"], half_life=10.0, clock="t", gap_cap=50.0
    )
    out = po.stream.with_windows(
        df, f=po.rewm_mean("mid", half_life=5.0, window_size=10.0), like=spec
    )
    plain = po.stream.with_windows(
        df, f=po.rewm_mean("mid", half_life=5.0, window_size=10.0), clock="t", gap_cap=50.0
    )
    # A row the spec would skip (a null feature: the quotes) gets a null window.
    quotes = (df["price"].is_null()).to_list()
    for a, b, q in zip(out["f"].to_list(), plain["f"].to_list(), quotes, strict=True):
        assert (a is None) if q else (a == b)
    with pytest.raises(TypeError, match="leave out clock"):
        po.stream.with_windows(
            df, f=po.rewm_mean("mid", half_life=5.0, window_size=10.0), like=spec, clock="t"
        )


# --------------------------------------------------------------------------
# Streaming: chunks, state, projection


def mixed() -> dict[str, pl.Expr]:
    h = {"half_life": 2.0}
    return {
        "a": po.ewm_mean("x", **h, window_size=5.0),
        "b": po.rewm_mean("x", **h, window_size=4.0) - pl.col("x"),
        "c": po.ewm_rate(po.increment("x"), **h, window_size=6.0),
        "d": po.rewm_sum(pl.when(pl.col("x") > 0).then("x"), **h, window_size=3.0, closed="left"),
    }


@pytest.mark.parametrize("rows", [1, 7, 64, 10_000])
def test_chunking_changes_nothing(rows: int) -> None:
    df = ticks(800, 17, groups=3)
    one = po.stream.with_windows(
        df, **mixed(), clock="t", gap_cap=20.0, group="g", chunk_size=10_000
    )
    out = po.stream.with_windows(df, **mixed(), clock="t", gap_cap=20.0, group="g", chunk_size=rows)
    assert out.equals(one)


def test_a_save_and_load_at_every_row_is_one_run(tmp_path: Any) -> None:
    df = ticks(60, 19, groups=2)
    one = po.stream.with_windows(df, **mixed(), clock="t", gap_cap=20.0, group="g")
    for at in (1, 2, 17, 33, 59):
        state = tmp_path / f"w{at}.state"
        first = po.stream.with_windows(
            df[:at], **mixed(), clock="t", gap_cap=20.0, group="g", save_state=state
        )
        second = po.stream.with_windows(
            df[at:], **mixed(), clock="t", gap_cap=20.0, group="g", load_state=state
        )
        assert pl.concat([first, second]).equals(one), at


def test_a_state_resumes_only_its_own_call(tmp_path: Any) -> None:
    df = ticks(30, 2)
    state = tmp_path / "w.state"
    po.stream.with_windows(df, a=po.ewm_mean("x", half_life=2.0), save_state=state, **CLOCK)
    with pytest.raises(ValueError, match="another call"):
        po.stream.with_windows(df, a=po.ewm_mean("x", half_life=3.0), load_state=state, **CLOCK)


def test_a_slice_and_a_projection_are_honoured() -> None:
    df = ticks(300, 23)
    plan = df.lazy().online.with_windows(**mixed(), **CLOCK)
    head = plan.head(50).collect()
    assert head.height == 50 and head.equals(plan.collect().head(50))
    narrow = plan.select("b", "d").collect()
    assert narrow.columns == ["b", "d"]
    assert narrow.equals(plan.collect().select("b", "d"))


def test_the_online_namespaces_are_the_same_call() -> None:
    df = ticks(100, 29)
    want = po.stream.with_windows(df, m=po.ewm_mean("x", half_life=2.0), **CLOCK)
    assert df.online.with_windows(m=po.ewm_mean("x", half_life=2.0), **CLOCK).equals(want)
    assert (
        df.lazy()
        .online.with_windows(m=po.ewm_mean("x", half_life=2.0), **CLOCK)
        .collect()
        .equals(want)
    )


def test_window_columns_feed_a_model_in_one_query() -> None:
    df = ticks(400, 31, repeats=False)
    out = (
        df.lazy()
        .online.with_windows(trend=po.ewm_mean("x", half_life=5.0) - pl.col("x"), **CLOCK)
        .online.fit_predict(
            [po.spec.ewridge("m", targets=["x"], features=["trend"], half_life=50.0)]
        )
        .collect()
    )
    assert out["m"].struct.field("pred_x").drop_nulls().len() > 300


def test_a_temporal_clock_takes_durations() -> None:
    df = ticks(200, 37, repeats=False).with_columns(
        ts=pl.from_epoch((pl.col("t") * 1e9).cast(pl.Int64), time_unit="ns")
    )
    want = po.stream.with_windows(df, m=po.ewm_mean("x", half_life=3.0, window_size=7.0), **CLOCK)
    for hl, w in (
        ("3s", "7s"),
        (timedelta(seconds=3), timedelta(seconds=7)),
        (pl.duration(seconds=3), pl.duration(seconds=7)),
    ):
        got = po.stream.with_windows(
            df, m=po.ewm_mean("x", half_life=hl, window_size=w), clock="ts", gap_cap="1000s"
        )
        # Stamps rounded to the nanosecond move each interval by up to 0.5 ns.
        assert_close(got["m"].to_list(), want["m"].to_list(), tol=1e-7)
    with pytest.raises(ValueError, match="half_life is a duration"):
        po.stream.with_windows(df, m=po.ewm_mean("x", half_life="3s"), **CLOCK)
    with pytest.raises(ValueError, match="half_life a number"):
        po.stream.with_windows(df, m=po.ewm_mean("x", half_life=3.0), clock="ts", gap_cap="1000s")


@pytest.mark.extended(reason="the network: a downloaded day of Binance quotes and trades")
def test_the_real_day_runs_and_the_recipes_agree_where_they_should() -> None:
    """One symbol-day of Binance quotes and trades (hard rule 1: downloaded
    and cached): the three forward VWAPs and a trailing mid, under a
    Datetime clock, chunked; the VWAP from running sums equals the one from
    prices and quantities where the sums step at trades."""
    from data import public_quotes_and_trades_or_skip

    day = public_quotes_and_trades_or_skip()  # offline: the one skip, explained
    df = day.head(200_000).with_columns(
        cum_q=pl.col("quantity").fill_null(0.0).cum_sum(),
        cum_pq=(pl.col("price") * pl.col("quantity")).fill_null(0.0).cum_sum(),
    )
    notional = pl.col("price") * pl.col("quantity")
    h = {"half_life": "10s", "window_size": "1m"}
    out = po.stream.with_windows(
        df,
        mid_trend=po.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
        fwd_vwap=po.rewm_sum(notional, **h) / po.rewm_sum("quantity", **h) - pl.col("mid"),
        vwap_pq=po.ewm_sum(notional, half_life="10s") / po.ewm_sum("quantity", half_life="10s"),
        vwap_sums=po.ewm_sum(po.increment("cum_pq"), half_life="10s")
        / po.ewm_sum(po.increment("cum_q"), half_life="10s"),
        clock="ts",
        gap_cap="5m",
        chunk_size=50_000,
    )
    assert out.height == df.height
    assert out["fwd_vwap"].drop_nulls().len() > 150_000
    both = out.select("vwap_pq", "vwap_sums").drop_nulls()
    assert both.height > 100_000
    assert (both["vwap_pq"] - both["vwap_sums"]).abs().max() < 1e-6, (
        "the day's rounding, about 2e-6 at 1e10"
    )


# --------------------------------------------------------------------------
# Review round R1 (2026-10-03): the formula layer and the runner


def test_a_state_saved_under_a_slice_resumes_alike_at_any_chunk_size(tmp_path: Any) -> None:
    """R1-B1: the increments' state advanced over the whole chunk where a
    slice stopped the core short, so a state saved under ``head`` depended on
    ``chunk_size``."""
    df = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "c": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0]}
    )
    outs = []
    for rows in (1, 10):
        state = tmp_path / f"s{rows}.state"
        df.lazy().online.with_windows(
            dc=po.increment("c"), clock="t", gap_cap=100.0, chunk_size=rows, save_state=state
        ).head(2).collect()
        rest = (
            df.slice(2)
            .lazy()
            .online.with_windows(dc=po.increment("c"), clock="t", gap_cap=100.0, load_state=state)
            .collect()
        )
        outs.append(rest["dc"].to_list())
    assert outs[0] == outs[1] == [2.0, 3.0, 4.0, 5.0]


def test_a_cast_is_strict_as_in_polars_unless_told_otherwise() -> None:
    """R1-B2: a formula's ``cast`` was rebuilt non-strict, so an overflow that
    Polars refuses became a silent null."""
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [1000.0, 1.0]})
    with pytest.raises((pl.exceptions.PolarsError, ValueError), match="conversion from"):
        po.stream.with_windows(df, y=po.ewm_sum("x", half_life=1.0).cast(pl.Int8), **CLOCK)
    lax = po.ewm_sum("x", half_life=1.0).cast(pl.Int8, strict=False)
    assert po.stream.with_windows(df, y=lax, **CLOCK)["y"].to_list() == [None, None]
    assert to_tree(pl.col("x").cast(pl.Int8, strict=False))[-1] == "non_strict"
    for e in [pl.col("x").cast(pl.Int8), pl.col("x").cast(pl.Int8, strict=False)]:
        assert from_tree(to_tree(e)).meta.eq(e)
    with pytest.raises(FormulaError, match="options"):
        to_tree(pl.col("x").cast(pl.Int8, wrap_numerical=True))


def test_a_restart_on_the_streams_clock_restarts_every_groups_increments() -> None:
    """R1-S4: the core discards every group's windows at a step back on the
    stream's clock past ``restart_after_step_back``; the increments now start
    over with them."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 100.0, 5.0, 50.0],
            "g": ["A", "B", "A", "A", "B"],
            "c": [1.0, 10.0, 2.0, 3.0, 20.0],
        }
    )
    out = po.stream.with_windows(
        df,
        dc=po.increment("c"),
        b=po.ewm_sum("c", half_life=math.inf),
        clock="t",
        gap_cap=1e9,
        restart_after_step_back=50.0,
        group="g",
    )
    assert out["b"].to_list()[-1] == 20.0, "B's windows started over at the stream's restart"
    assert out["dc"].to_list()[-1] is None, "and so did its increment"


def test_an_increment_skips_a_null_input() -> None:
    """R1-S5, decided: ``x_{i-1}`` is the last value with a value, as the
    operators hold one from the last valued row; ``diff()`` would give three
    nulls."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "c": [1.0, None, 3.0]})
    out = po.stream.with_windows(df, dc=po.increment("c"), **CLOCK)
    assert out["dc"].to_list() == [None, None, 2.0]


def test_the_operators_prefix_is_reserved_on_inputs() -> None:
    """R1-N8: a column named like an operator's hidden column raised a JSON
    error; it is refused by name on both sides."""
    df = pl.DataFrame({"t": [0.0, 1.0], "@po:x": [1.0, 2.0]})
    with pytest.raises(FormulaError, match="reserved"):
        po.stream.with_windows(df, y=po.ewm_sum("@po:x", half_life=1.0), **CLOCK)


def test_a_positional_expression_named_after_a_column_is_told_about_alias() -> None:
    """R1-N9: ``pl.col("x") - op`` is named ``x`` by its leftmost column, and
    the refusal says so."""
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"\.alias\(\)"):
        po.stream.with_windows(df, pl.col("x") - po.ewm_mean("x", half_life=2.0), **CLOCK)


def test_from_tree_takes_a_log_base_that_is_not_a_literal() -> None:
    """R1-N10: a hand-written tree may put an expression under ``log``; the
    Rust side reads it, and so does ``from_tree`` now."""
    got = from_tree(["log", ["col", "x"], ["col", "b"]])
    assert got.meta.eq(pl.col("x").log() / pl.col("b").log())


# --------------------------------------------------------------------------
# Review round R1 (2026-10-03): the window core


def test_a_row_count_clock_reads_every_backward_window() -> None:
    """R1-C1: without a clock, every backward operator under ``"right"`` or
    ``"both"`` was null on every row -- the row never waited for a next
    stamp, and nothing read it."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = po.stream.with_windows(
        df,
        s=po.ewm_sum("x", half_life=2.0),
        m=po.ewm_mean("x", half_life=2.0),
        w=po.ewm_mean("x", half_life=2.0, window_size=2.0),
        b=po.ewm_sum("x", half_life=2.0, closed="both"),
    )
    t = [0.0, 1.0, 2.0, 3.0]
    assert_close(out["s"].to_list(), recursion_sum(t, df["x"].to_list(), 2.0))
    assert_close(out["m"].to_list(), recursion_mean(t, df["x"].to_list(), 2.0))
    assert out["w"].to_list()[0] is not None
    assert out["b"].to_list() == out["s"].to_list()


def test_a_row_exactly_one_window_later_lands_on_the_edge_at_any_age() -> None:
    """R1-C2: the policy time was a sum of rounded steps, so a row exactly
    one window later fell on either side of the edge by the sum's noise; it
    is measured from the stretch's first row, exact in nanoseconds."""
    n = 1003
    ts = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 0, 1, 2000), "1ms", eager=True
    )[:n]
    df = pl.DataFrame({"t": ts, "x": [1.0] * n})
    out = po.stream.with_windows(
        df,
        left=po.ewm_sum("x", half_life=math.inf, window_size="1s", closed="left"),
        fwd=po.rewm_sum("x", half_life=math.inf, window_size="1s"),
        clock="t",
        gap_cap="1h",
    )
    want = df.select(pl.col("x").rolling_sum_by("t", "1s", closed="left"))["x"].to_list()
    assert out["left"].to_list()[1000] == want[1000] == 1000.0
    assert out["fwd"].to_list()[0] == 1000.0, "a forward window counts the row exactly w later"
    # And after 60 s of 1 ms steps, where the sum of steps fell short of 60.
    n = 60_003
    ts = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 1, 0, 3000), "1ms", eager=True
    )[:n]
    df = pl.DataFrame({"t": ts, "x": [1.0] * n})
    out = po.stream.with_windows(
        df,
        right=po.ewm_sum("x", half_life=math.inf, window_size="1m"),
        clock="t",
        gap_cap="1h",
    )
    want = df.select(pl.col("x").rolling_sum_by("t", "1m", closed="right"))["x"].to_list()
    assert out["right"].to_list()[60_001] == want[60_001] == 60_000.0


def test_forward_left_and_both_hold_every_row_at_the_stamp() -> None:
    """R1-C3, decided: a window is a set of timestamps, so under ``"left"``
    and ``"both"`` a forward window holds every row at the row's own stamp,
    the row itself included -- the mirror of a backward ``"right"`` window,
    which holds every row at its stamp."""
    df = pl.DataFrame({"t": [0.0, 0.0, 1.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]})
    out = po.stream.with_windows(
        df,
        left=po.rewm_sum("x", half_life=math.inf, window_size=2.0, closed="left"),
        right=po.rewm_sum("x", half_life=math.inf, window_size=2.0),
        **CLOCK,
    )
    # [0, 2): rows at 0 (both) and 1, for both rows at the stamp.
    assert out["left"].to_list()[:2] == [6.0, 6.0]
    # (0, 2]: row 1 only, for both.
    assert out["right"].to_list()[:2] == [3.0, 3.0]
    # The mirror with repeated stamps: forward "left" is backward "right"
    # over the reversed stream.
    df = ticks(400, 21, repeats=True)
    fwd = po.stream.with_windows(
        df,
        m=po.rewm_mean("x", half_life=3.0, window_size=5.0, closed="left"),
        s=po.rewm_sum("x", half_life=3.0, window_size=5.0, closed="left"),
        **CLOCK,
    )
    last = df["t"][-1]
    mirror = df.reverse().with_columns(t=last - pl.col("t"))
    back = po.stream.with_windows(
        mirror,
        m=po.ewm_mean("x", half_life=3.0, window_size=5.0),
        s=po.ewm_sum("x", half_life=3.0, window_size=5.0),
        **CLOCK,
    ).reverse()
    for c in ("m", "s"):
        pairs = [
            (a, b)
            for a, b in zip(fwd[c].to_list(), back[c].to_list(), strict=True)
            if a is not None
        ]
        assert len(pairs) > 300
        for a, b in pairs:
            assert b is not None and abs(a - b) <= 1e-9 * (1 + abs(a)), c


def test_a_cut_window_ends_at_the_last_row_it_saw() -> None:
    """R1-C4, decided: ``partial="keep"`` is the value over what the window
    saw; it was held ``gap_cap`` past the last row, so a kept rate's span
    grew with the cap."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 100.0, 101.0], "x": [1.0] * 5})
    rates = []
    for cap in (5.0, 300.0):
        out = po.stream.with_windows(
            df,
            r=po.rewm_rate("x", half_life=math.inf, window_size=60.0, partial="keep"),
            m=po.rewm_mean("x", half_life=math.inf, window_size=60.0, partial="keep"),
            clock="t",
            gap_cap=cap,
        )
        rates.append(out["r"].to_list()[0])
        assert out["m"].to_list()[0] == 1.0
    # Cut at t = 2 under the cap of 5: two rows over the two seconds seen,
    # where the old form spread them over seven. Under the cap of 300 the
    # window is not cut: it closes at t = 100 with its full span of 60.
    assert rates[0] == 2.0 / 2.0
    assert rates[1] == 2.0 / 60.0


@needs_polars("1.37.0", "Polars' rolling_mean_by, the reference, refuses a column with a null")
def test_a_row_on_the_far_edge_counts_but_weighs_nothing() -> None:
    """R1-C6, decided: under ``"left"`` or ``"both"`` a row exactly one window
    old is in the window by its stamp, so it counts for ``min_samples``, but
    its held interval -- from the valued row before it to itself -- lies
    before the window, so it weighs nothing in a mean; a mean with no other
    value in the window is null, where Polars' row-weighted
    ``rolling_mean_by`` says 5. The brute force from the definition agrees
    with the core, not with Polars."""
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [5.0, None]})
    out = po.stream.with_windows(
        df, m=po.ewm_mean("x", half_life=math.inf, window_size=1.0, closed="left"), **CLOCK
    )
    assert out["m"].to_list()[1] is None
    assert loop(df, "ewm_mean", "x", half_life=math.inf, window_size=1.0, closed="left")[1] is None
    polars = (
        df.with_columns(pl.col("t").cast(pl.Int64))
        .select(pl.col("x").rolling_mean_by("t", "1i", closed="left"))["x"]
        .to_list()
    )
    assert polars[1] == 5.0
    # With a value at the row itself, the far-edge row still holds nothing
    # inside the window and the mean is the row's own value under "both".
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [5.0, 7.0]})
    out = po.stream.with_windows(
        df, m=po.ewm_mean("x", half_life=math.inf, window_size=1.0, closed="both"), **CLOCK
    )
    assert out["m"].to_list()[1] == 7.0


# --------------------------------------------------------------------------
# Review round R3 (2026-10-03): the window core's edges, sessions, resuming

# 2024-01-01T00:00:00Z in milliseconds: an age at which a double resolves
# an epoch to 0.24 us, so a clock read as seconds would miss every edge.
EPOCH_2024_MS = 1_704_067_200_000


def _ms(ms: list[int]) -> pl.DataFrame:
    """Rows at ``ms`` milliseconds on a Datetime clock, each worth 1."""
    t = pl.Series("t", ms, dtype=pl.Int64).cast(pl.Datetime("ms"))
    return pl.DataFrame({"t": t, "x": [1.0] * len(ms)})


def test_a_long_half_life_agrees_with_the_definition_and_tends_to_the_even_one() -> None:
    """Task 159 (W2): the core's `1 - 2^(-t/h)` cancelled at a half-life far
    past the window -- 6e-5 of the mass at `h/t = 1e12`, all of it from 1e17,
    every mean null -- and the oracle here took the same form. Both take the
    closed form by expm1; at 1e17 the mean is the even window's."""
    df = pl.DataFrame({"t": [float(i) for i in range(20)], "x": [float(i % 5) for i in range(20)]})
    for h in (1e9, 1e12):
        out = po.stream.with_windows(df, y=po.ewm_mean("x", half_life=h, window_size=5.0), **CLOCK)
        want = loop(df, "ewm_mean", "x", half_life=h, window_size=5.0)
        assert_close(out["y"].to_list(), want, f"half_life {h}")
    far = po.stream.with_windows(df, y=po.ewm_mean("x", half_life=1e17, window_size=5.0), **CLOCK)
    even = po.stream.with_windows(
        df, y=po.ewm_mean("x", half_life=math.inf, window_size=5.0), **CLOCK
    )
    assert far["y"].null_count() == even["y"].null_count() < 20
    assert_close(far["y"].to_list(), even["y"].to_list(), "1e17 against inf", tol=1e-12)


def test_a_number_clock_decides_an_edge_from_the_two_rows_clocks() -> None:
    """Task 159 (W3): on a number clock an edge was decided from two
    origin-subtracted policy times, so with `t = [0.1, 0.2, 0.5]` the row
    exactly one window (0.3) back fell outside it: `(0.5 - 0.1) - (0.2 - 0.1)`
    is `0.30000000000000004` where `0.5 - 0.2` is `0.3`. The edge is the
    difference of the two rows' clocks, as on a temporal clock since R2-W1."""
    df = pl.DataFrame({"t": [0.1, 0.2, 0.5], "x": [1.0, 1.0, 1.0]})
    left = po.ewm_sum("x", half_life=math.inf, window_size=0.3, closed="left")
    assert po.stream.with_windows(df, y=left, **CLOCK)["y"].to_list() == [None, 1.0, 1.0]
    both = po.ewm_sum("x", half_life=math.inf, window_size=0.3, closed="both")
    assert po.stream.with_windows(df, y=both, **CLOCK)["y"].to_list() == [1.0, 2.0, 2.0]
    right = po.ewm_sum("x", half_life=math.inf, window_size=0.3, closed="right")
    assert po.stream.with_windows(df, y=right, **CLOCK)["y"].to_list() == [1.0, 2.0, 1.0]


def test_a_subnormal_half_life_is_refused() -> None:
    """Task 159 (W5): a subnormal half-life gave a mean of 0 and a rate of
    inf, where every other bad half-life is refused by name."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [0.3, 0.3, 0.3]})
    with pytest.raises(ValueError, match="half_life must be a normal number"):
        po.stream.with_windows(df, y=po.ewm_rate("x", half_life=5e-324, window_size=2.0), **CLOCK)


@pytest.mark.parametrize("base", [0, EPOCH_2024_MS])
def test_a_row_exactly_one_window_from_another_lands_as_in_polars(base: int) -> None:
    """R2-W1: an edge was decided from two rounded policy times, so a row
    exactly one window from another landed on either side of it (0.4 - 0.1
    is not 0.3 in a double); it is decided from the two rows' clocks, exact
    in nanoseconds, where Polars puts it."""
    clock = {"clock": "t", "gap_cap": "1h"}

    def back(closed: str) -> pl.Expr:
        return po.ewm_sum("x", half_life=math.inf, window_size="300ms", closed=closed)

    # Backward "left": the row 300 ms older is at the far edge, in. Row 0's
    # window is empty: null here under ``min_samples``, where Polars sums
    # nothing to 0.
    df = _ms([base, base + 100, base + 400])
    got = po.stream.with_windows(df, y=back("left"), **clock)["y"].to_list()
    want = df.select(pl.col("x").rolling_sum_by("t", "300ms", closed="left"))["x"].to_list()
    assert got[1:] == want[1:] == [1.0, 1.0] and got[0] is None
    # Backward "right": the row 300 ms older is at the far edge, out.
    df = _ms([base, base + 400, base + 700])
    got = po.stream.with_windows(df, y=back("right"), **clock)["y"].to_list()
    want = df.select(pl.col("x").rolling_sum_by("t", "300ms", closed="right"))["x"].to_list()
    assert got == want == [1.0, 1.0, 1.0]
    # Forward "right": the row 300 ms later is at the far edge, in. A
    # forward window closes only at a stamp past its far edge, so a fourth
    # row is there to close row 1's; rows 2 and 3 are null, one closed
    # empty, one unresolved at the end. Polars has no look-ahead; the
    # mirror is a backward "left" window on the negated clock.
    df = _ms([base, base + 100, base + 400, base + 800])
    fwd = po.rewm_sum("x", half_life=math.inf, window_size="300ms", closed="right")
    got = po.stream.with_windows(df, y=fwd, **clock)["y"].to_list()
    mirror = (
        df.with_columns(u=-pl.col("t").dt.epoch("ms"))
        .sort("u")
        .select(pl.col("x").rolling_sum_by("u", "300i", closed="left"))["x"]
        .to_list()[::-1]
    )
    assert got[:2] == mirror[:2] == [1.0, 1.0] and got[2:] == [None, None]


def test_with_groups_a_session_is_each_groups() -> None:
    """R2-W3: the stream's clock took every row's session, so groups with
    sessions of their own saw a change at every row, and every group's
    windows started over at every row. With a group column a session is
    each group's; without one the stream is the one group, and its session
    change still ends the windows."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "g": ["a", "b", "a", "b"],
            "s": ["a1", "b1", "a1", "b1"],
            "x": [1.0, 10.0, 3.0, 30.0],
        }
    )
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 10.0, "session": "s", "session_gap": 1.0}
    total = po.ewm_sum("x", half_life=math.inf)
    assert po.stream.with_windows(df, y=total, group="g", **kw)["y"].to_list() == [1, 10, 4, 40]
    # A group's own session change ends its windows, and no other group's.
    own = df.with_columns(s=pl.Series(["a1", "b1", "a2", "b1"]))
    assert po.stream.with_windows(own, y=total, group="g", **kw)["y"].to_list() == [1, 10, 3, 40]
    # One group: the stream's session, a change at rows 1, 2 and 3.
    one = df.with_columns(s=pl.Series(["a1", "b1", "a2", "b2"]))
    assert po.stream.with_windows(one, y=total, **kw)["y"].to_list() == [1, 10, 3, 30]


@pytest.mark.parametrize("rows", [1, 7, 100_000])
def test_a_state_saved_under_a_slice_resumes_on_the_same_input(tmp_path: Any, rows: int) -> None:
    """R2-W4, then R4-B1/B2 and R5-C1: under ``head(n)`` the run reads past
    the n-th row (a backward ``"right"`` window waits for the next stamp, a
    look-ahead for its window to pass). The state records the rows of the
    input consumed so far, skipped or fed, and the input's first clock; a run
    resumed with the same input, unsliced, skips them and goes on, so any
    chain of sliced runs gives one run's output, a ``partial="drop"`` row
    (consumed, never returned) and rows held from an earlier run (returned,
    not this input's) included, whatever the chunk size (R5-C1: a loaded
    skip not yet applied when a resumed run satisfied its slice from held
    rows was left out of the count, so the chain broke at ``chunk_size=1``)."""
    df = ticks(60, 19, groups=2)
    dropping = {"e": po.rewm_mean("x", half_life=2.0, window_size=4.0, partial="drop")}
    for exprs, cap in [(mixed(), 20.0), (mixed() | dropping, 4.0)]:
        kw: dict[str, Any] = {"clock": "t", "gap_cap": cap, "group": "g", "chunk_size": rows}
        one = po.stream.with_windows(df, **exprs, **kw)
        assert one.height > 20, "the dropping leg keeps enough rows to slice"
        for at in (1, 2, 17):
            state = tmp_path / f"h{at}-{cap}.state"
            head = df.lazy().online.with_windows(**exprs, save_state=state, **kw).head(at).collect()
            assert head.height == at
            rest = df.lazy().online.with_windows(**exprs, load_state=state, **kw).collect()
            assert pl.concat([head, rest]).equals(one), (at, cap)
        # A chain: six sliced runs, each resumed from the last and saved again,
        # then the rest.
        state = tmp_path / f"chain-{cap}.state"
        parts = [df.lazy().online.with_windows(**exprs, save_state=state, **kw).head(3).collect()]
        for _ in range(5):
            parts.append(
                df.lazy()
                .online.with_windows(**exprs, load_state=state, save_state=state, **kw)
                .head(3)
                .collect()
            )
        parts.append(df.lazy().online.with_windows(**exprs, load_state=state, **kw).collect())
        assert all(p.height == 3 for p in parts[:6]), [p.height for p in parts]
        assert pl.concat(parts).equals(one), cap
        # R5-C2: a chain of slices run until one is not satisfied -- the
        # input ends first -- then the rest. The unsatisfied run consumed
        # every row and saved that, not zero.
        state = tmp_path / f"exhaust-{cap}-{rows}.state"
        parts = [df.lazy().online.with_windows(**exprs, save_state=state, **kw).head(20).collect()]
        while parts[-1].height == 20:
            parts.append(
                df.lazy()
                .online.with_windows(**exprs, load_state=state, save_state=state, **kw)
                .head(20)
                .collect()
            )
        parts.append(df.lazy().online.with_windows(**exprs, load_state=state, **kw).collect())
        assert pl.concat(parts).equals(one), (cap, [p.height for p in parts])


def test_rows_held_from_an_earlier_input_are_not_this_inputs(tmp_path: Any) -> None:
    """R4-B2: a run on the next file first returns the rows the state held
    from the file before, which are not rows of this input; a slice that
    returns them must not count them as consumed."""
    df = ticks(60, 19, groups=2)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = po.stream.with_windows(df, **mixed(), **kw)
    file1, file2 = df.slice(0, 30), df.slice(30)
    state = tmp_path / "two.state"
    a = po.stream.with_windows(file1, **mixed(), save_state=state, **kw)
    assert a.height < 30, "file1 ends with rows held for their windows"
    b = (
        file2.lazy()
        .online.with_windows(**mixed(), load_state=state, save_state=state, **kw)
        .head(5)
        .collect()
    )
    c = file2.lazy().online.with_windows(**mixed(), load_state=state, **kw).collect()
    assert pl.concat([a, b, c]).equals(one)


def test_a_state_saved_under_a_slice_skips_nothing_of_another_input(tmp_path: Any) -> None:
    """R4-B3: the next file starts at another clock, so a state saved under a
    slice of the file before skips none of its rows; the rows the state held
    come out first."""
    df = ticks(60, 19, groups=2)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    file1, file2 = df.slice(0, 30), df.slice(30)
    state = tmp_path / "next.state"
    file1.lazy().online.with_windows(**mixed(), save_state=state, **kw).head(5).collect()
    out = po.stream.with_windows(file2, **mixed(), load_state=state, **kw)
    assert out["t"].to_list()[-file2.height :] == file2["t"].to_list()
    assert out["t"][0] < file2["t"][0], "the rows the state held come out first"


def test_a_windows_state_of_another_version_is_refused_by_its_version(tmp_path: Any) -> None:
    """R2-W5: a state written before round one (version 2) loaded with
    defaults for the fields round one added, and misbehaved; the version
    is read before the rest and refused by number."""
    # msgpack by hand: a map of two, "magic" and "version" 2.
    header = b"\x82" + b"\xa5magic" + b"\xb5polars-online windows" + b"\xa7version" + b"\x02"
    path = tmp_path / "v2.state"
    path.write_bytes(header)
    with pytest.raises(ValueError, match=r"version 2 not supported \(this build reads 9\)"):
        po.stream.with_windows(
            ticks(5, 1), y=po.ewm_sum("x", half_life=1.0), load_state=path, **CLOCK
        )


def test_a_silent_groups_session_reset_after_a_capped_gap_is_a_cut() -> None:
    """R4-A1, a decision recorded: under ``group`` the stream's clock cuts a
    group silent past ``gap_cap`` as soon as another group's row shows the
    silence, before the group's next row can say its session changed, so a
    ``session_gap="reset"`` there discards only what is still open. The same
    rows without a group column discard, since one clock sees the gap and
    the session change at one row, and the reset comes first."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 4.0, 8.0, 9.0],
            "g": ["a", "b", "b", "a", "a", "b"],
            "s": [1, 1, 1, 1, 1, 2],
            "x": [1.0, 10.0, 20.0, 1.0, 1.0, 30.0],
        }
    )
    fwd = po.rewm_sum("x", half_life=math.inf, window_size=10.0, partial="keep")
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 5.0, "session": "s", "session_gap": "reset"}
    grouped = po.stream.with_windows(df, y=fwd, group="g", **kw)
    assert grouped["y"][1] == 20.0, grouped["y"].to_list()
    alone = po.stream.with_windows(df.filter(pl.col("g") == "b"), y=fwd, **kw)
    assert alone["y"][0] is None, alone["y"].to_list()


def test_a_state_saved_under_a_slice_refuses_another_input(tmp_path: Any) -> None:
    """R5-C3/C4: the input a sliced state resumes on is known by the rows it
    held -- the unresolved tail of the consumed prefix, the input's own rows
    -- and by its first clock. An input that starts at the same clock but is
    another file, and the same input sliced by hand (``df.slice(n)``, the old
    contract), are refused by name, not fed with rows skipped or doubled;
    without a clock column too."""
    df = ticks(60, 19, groups=2)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    state = tmp_path / "sliced.state"
    df.lazy().online.with_windows(**mixed(), save_state=state, **kw).head(5).collect()
    # The same first clock, every later row another: the next day's file on
    # a clock that starts over.
    other = df.with_columns(
        t=pl.when(pl.int_range(pl.len()) == 0).then(pl.col("t")).otherwise(pl.col("t") + 0.25),
        x=pl.col("x") * 3,
    )
    with pytest.raises(_REFUSAL, match="another input"):
        po.stream.with_windows(other, **mixed(), load_state=state, **kw)
    with pytest.raises(_REFUSAL, match="another input"):
        po.stream.with_windows(df.slice(5), **mixed(), load_state=state, **kw)
    # An input shorter than the rows consumed: refused at the input's end,
    # and at a save that says the input ended (R6: no test passed
    # ``input_ended`` before).
    with pytest.raises(_REFUSAL, match="another input"):
        po.stream.with_windows(df.head(3), **mixed(), load_state=state, **kw)
    with pytest.raises(_REFUSAL, match="ended after 3"):
        po.stream.with_windows(
            df.head(3), **mixed(), load_state=state, save_state=tmp_path / "short.state", **kw
        )
    # No clock column: the held rows alone identify the input.
    bare = df.drop("t")
    state = tmp_path / "bare.state"
    bare.lazy().online.with_windows(
        f=po.rewm_sum("x", half_life=2.0, window_size=3.0), save_state=state
    ).head(5).collect()
    one = po.stream.with_windows(bare, f=po.rewm_sum("x", half_life=2.0, window_size=3.0))
    head = (
        bare.lazy()
        .online.with_windows(f=po.rewm_sum("x", half_life=2.0, window_size=3.0), save_state=state)
        .head(5)
        .collect()
    )
    rest = po.stream.with_windows(
        bare, f=po.rewm_sum("x", half_life=2.0, window_size=3.0), load_state=state
    )
    assert pl.concat([head, rest]).equals(one)
    with pytest.raises(_REFUSAL, match="another input"):
        po.stream.with_windows(
            bare.slice(5), f=po.rewm_sum("x", half_life=2.0, window_size=3.0), load_state=state
        )


def test_a_damaged_windows_state_says_so(tmp_path: Any) -> None:
    """R5-C6: a state cut short is reported as damaged, not as a state another
    call saved; the version is 4 since round five (C5: round four changed
    what the skip counts and added the identity, and the number did not move)."""
    df = ticks(30, 3)
    state = tmp_path / "w.state"
    po.stream.with_windows(df, y=po.ewm_mean("x", half_life=2.0), save_state=state, **CLOCK)
    whole = state.read_bytes()
    cut = tmp_path / "cut.state"
    cut.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(ValueError, match="damaged"):
        po.stream.with_windows(df, y=po.ewm_mean("x", half_life=2.0), load_state=cut, **CLOCK)


def test_a_sliced_run_on_the_next_file_resumes_with_the_first_files_rows_held(
    tmp_path: Any,
) -> None:
    """R6-D2: a run on the next file under a slice keeps the first file's
    unresolved rows held ahead of its own (rows go out in order, so none of
    its own went out while one of theirs waited), and the state then holds
    more rows than it consumed of this input. The identity is the last
    ``consumed`` of the held rows, this input's own; a resume on the second
    file goes on where round five refused it as another input."""
    df = ticks(60, 19, groups=2)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = po.stream.with_windows(df, **mixed(), **kw)
    for cut, n in [(20, 1), (30, 2), (45, 3)]:
        first, second = df.head(cut), df.slice(cut)
        state = tmp_path / f"{cut}-{n}.state"
        a = po.stream.with_windows(first, **mixed(), save_state=state, **kw)
        assert cut - a.height > n, "the first file's held rows outlast the slice"
        b = (
            second.lazy()
            .online.with_windows(**mixed(), load_state=state, save_state=state, **kw)
            .head(n)
            .collect()
        )
        c = po.stream.with_windows(second, **mixed(), load_state=state, **kw)
        assert pl.concat([a, b, c]).equals(one), (cut, n)


def test_a_sliced_state_that_holds_no_rows_knows_its_input_by_its_last_row(
    tmp_path: Any,
) -> None:
    """R6-D1: a backward operator under ``closed="left"`` (or any operator on
    a row-count clock) resolves a row at its own push, so a sliced run holds
    no row and the held rows identify nothing; round five's fallback to the
    last clock was never reached. The state keeps the last row it read, and
    another input is refused by it, with and without a clock column."""
    df = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]}
    )
    y = {"y": po.ewm_mean("x", half_life=1.0, closed="left")}
    one = po.stream.with_windows(df, **y, **CLOCK)
    state = tmp_path / "left.state"
    head = df.lazy().online.with_windows(**y, save_state=state, **CLOCK).head(3).collect()
    assert head.height == 3
    other = pl.DataFrame({"t": [0.0, 1.5, 2.5, 3.5, 4.5], "x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    with pytest.raises(_REFUSAL, match="another input"):
        po.stream.with_windows(other, **y, load_state=state, **CLOCK)
    rest = po.stream.with_windows(df, **y, load_state=state, **CLOCK)
    assert pl.concat([head, rest]).equals(one)
    # No clock column: the row-count clock repeats no stamp, so every row
    # resolves at its push and the state holds none.
    bare = df.drop("t")
    y = {"y": po.ewm_mean("x", half_life=2.0)}
    one = po.stream.with_windows(bare, **y)
    state = tmp_path / "bare-left.state"
    head = bare.lazy().online.with_windows(**y, save_state=state).head(4).collect()
    assert head.height == 4
    # Another input of the same length, differing where the state was cut
    # (R7-E5: a slice shorter than the skip was refused by the count alone).
    with pytest.raises(_REFUSAL, match="rows it read are not this input"):
        po.stream.with_windows(bare.with_columns(x=pl.col("x") * 3), **y, load_state=state)
    rest = po.stream.with_windows(bare, **y, load_state=state)
    assert pl.concat([head, rest]).equals(one)


def test_a_next_file_starts_after_the_last_stamp_the_state_read(tmp_path: Any) -> None:
    """R6-D3: a sliced state takes an input that starts at the stamp of the
    last row it read for the same input sliced inside a tied stamp, since a
    file boundary inside one cannot be told from it, and refuses it by name;
    one that starts after that stamp is the next file."""
    y = {"y": po.rewm_mean("x", half_life=1.0, window_size=2.0)}
    first = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]})
    state = tmp_path / "tie.state"
    a = first.lazy().online.with_windows(**y, save_state=state, **CLOCK).head(100).collect()
    assert a.height == 1, "rows 1..4 wait for their windows"
    tied = pl.DataFrame({"t": [3.0, 4.0, 5.0], "x": [5.0, 6.0, 7.0]})
    with pytest.raises(_REFUSAL, match="at the last stamp the state read"):
        po.stream.with_windows(tied, **y, load_state=state, **CLOCK)
    second = pl.DataFrame({"t": [4.0, 5.0, 6.0], "x": [5.0, 6.0, 7.0]})
    c = po.stream.with_windows(second, **y, load_state=state, **CLOCK)
    one = po.stream.with_windows(pl.concat([first, second]), **y, **CLOCK)
    assert pl.concat([a, c]).equals(one)


def test_an_unsliced_state_refuses_an_input_that_repeats_its_last_row(tmp_path: Any) -> None:
    """E13 (task 158): a state saved without a slice took an input whose first
    row repeats the last row it read, and that row came out twice: the
    README's `trades.head(2000)` saved, then `trades.slice(1999)` resumed,
    gave 3,001 rows against one run's 3,000. It now refuses an input that
    starts at the last stamp it read, as a sliced state does, unless that row
    starts a new session; one that starts after it is the next file."""
    y = {"y": po.rewm_mean("x", half_life=1.0, window_size=2.0)}
    first = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]})
    state = tmp_path / "unsliced.state"
    a = po.stream.with_windows(first, **y, save_state=state, **CLOCK)
    again = pl.DataFrame({"t": [3.0, 4.0, 5.0], "x": [4.0, 5.0, 6.0]})
    with pytest.raises(_REFUSAL, match="at the last stamp the state read"):
        po.stream.with_windows(again, **y, load_state=state, **CLOCK)
    second = again.slice(1)
    c = po.stream.with_windows(second, **y, load_state=state, **CLOCK)
    one = po.stream.with_windows(pl.concat([first, second]), **y, **CLOCK)
    assert pl.concat([a, c]).equals(one)
    # A new session at that stamp is a new start, which the policy takes.
    days = {**CLOCK, "session": "s", "session_gap": 1.0}
    first_day = first.with_columns(s=pl.lit("d1"))
    next_day = again.with_columns(s=pl.lit("d2"))
    a_day = po.stream.with_windows(first_day, **y, save_state=state, **days)
    c_day = po.stream.with_windows(next_day, **y, load_state=state, **days)
    one_day = po.stream.with_windows(pl.concat([first_day, next_day]), **y, **days)
    assert pl.concat([a_day, c_day]).equals(one_day)


def test_a_sliced_state_takes_a_new_start_by_the_policys_word_for_the_next_file(
    tmp_path: Any,
) -> None:
    """R6-D4: a next-day file on a clock that starts over steps back from the
    last row the state read. Where the clock policy takes that step as a new
    start -- a step back past ``restart_after_step_back``, or a new session
    -- the resumed run takes the file as the next one and skips nothing, as
    an unsliced state does; round five refused it as at or before the last
    row read. R7-E2: a clock that starts over at the *same* stamp each day
    gives the next file the saved input's first clock; the state knows its
    input by the first row's session too, so with a session column the file
    is put to the policy all the same, and without one it is the saved
    input until the rows differ, refused by name (the documented limit)."""
    day1 = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "d": ["a"] * 7,
        }
    )
    y = {"y": po.rewm_mean("x", half_life=1.0, window_size=2.0)}
    for start in (0.5, 0.0):
        day2 = pl.DataFrame(
            {"t": [start + i for i in range(8)], "x": [8.0 + i for i in range(8)], "d": ["b"] * 8}
        )
        for extra in ({"restart_after_step_back": 2.0}, {"session": "d", "session_gap": 1.0}):
            kw = {**CLOCK, **extra}
            state = tmp_path / f"{next(iter(extra))}-{start}.state"
            a = day1.lazy().online.with_windows(**y, save_state=state, **kw).head(100).collect()
            assert a.height < 7, "the last rows wait for their windows"
            if start == 0.0 and "session" not in extra:
                with pytest.raises(_REFUSAL, match="another input"):
                    po.stream.with_windows(day2, **y, load_state=state, **kw)
                continue
            one = po.stream.with_windows(pl.concat([day1, day2]), **y, **kw)
            c = po.stream.with_windows(day2, **y, load_state=state, **kw)
            assert pl.concat([a, c]).equals(one), (start, kw)


def test_without_a_clock_the_next_file_begins_with_a_new_session(tmp_path: Any) -> None:
    """R8-F1: on a row-count clock every row is a step forward, so a step
    forward says nothing about the input; without a clock column the next
    file is one that begins with a new session, and a hand slice that starts
    in the session the state last read is refused, not fed twice."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "s": ["a", "a", "a", "b", "b", "b"]})
    kw: dict[str, Any] = {"session": "s", "session_gap": 1.0}
    y = {"y": po.ewm_mean("x", half_life=1.0)}
    state = tmp_path / "rows.state"
    head = df.lazy().online.with_windows(**y, save_state=state, **kw).head(4).collect()
    assert head.height == 4
    for cut in (3, 5):
        with pytest.raises(_REFUSAL, match="does not begin with a new session"):
            po.stream.with_windows(df.slice(cut), **y, load_state=state, **kw)
    nxt = pl.DataFrame({"x": [7.0, 8.0], "s": ["c", "c"]})
    rest = po.stream.with_windows(nxt, **y, load_state=state, **kw)
    one = po.stream.with_windows(pl.concat([df.head(4), nxt]), **y, **kw)
    assert pl.concat([head, rest]).equals(one)
    same = po.stream.with_windows(df, **y, load_state=state, **kw)
    assert pl.concat([head, same]).equals(po.stream.with_windows(df, **y, **kw))


def test_a_restart_on_the_streams_clock_restarts_every_groups_clock() -> None:
    """Task 159 (W1): a restart on the stream's clock discarded every group's
    windows but left their clocks, so a group's next row within
    ``restart_after_step_back`` of its last -- C at 52 after C at 60, 8 back,
    once the stream restarted at row 3 -- was refused as a late row. Every
    group starts over, clock included, so C's row is the first of its new
    stretch, as it is when C's last row was at 70, 18 back."""

    def run(c_at: float) -> list[float | None]:
        df = pl.DataFrame(
            {
                "t": [0.0, c_at, 100.0, 50.0, 52.0],
                "g": ["A", "C", "A", "B", "C"],
                "x": [1.0] * 5,
            }
        )
        y = po.ewm_sum("x", half_life=math.inf)
        out = po.stream.with_windows(
            df, y=y, clock="t", gap_cap=1e9, group="g", restart_after_step_back=10.0
        )
        return out["y"].to_list()

    assert run(60.0) == run(70.0)
    assert run(60.0)[4] == 1.0


def test_a_state_refuses_a_group_or_session_column_of_another_dtype(tmp_path: Any) -> None:
    """Task 159 (F2): a key is its value's text, so an Int64 group column
    followed by a Float64 file keyed "1" then "1.0" and started every group
    over, silently; a session column's dtype change cut the windows. The
    state carries the two columns' dtypes and refuses another by name."""
    y = {"y": po.ewm_mean("x", half_life=2.0)}
    first = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0], "g": [1, 1, 2, 2]})
    second = pl.DataFrame({"t": [4.0, 5.0], "x": [5.0, 6.0], "g": [1.0, 2.0]})
    state = tmp_path / "keys.state"
    po.stream.with_windows(first, **y, save_state=state, group="g", **CLOCK)
    with pytest.raises(ValueError, match='group column "g": i64.*this input has.*f64'):
        po.stream.with_windows(second, **y, load_state=state, group="g", **CLOCK)
    same = po.stream.with_windows(
        second.with_columns(pl.col("g").cast(pl.Int64)), **y, load_state=state, group="g", **CLOCK
    )
    # The row the state held comes out first, then the new rows.
    assert same["t"].to_list() == [3.0, 4.0, 5.0]
    sessions = {**CLOCK, "session": "s", "session_gap": 1.0}
    po.stream.with_windows(first.with_columns(s=pl.lit(1)), **y, save_state=state, **sessions)
    with pytest.raises(ValueError, match='session column "s": i32.*this input has'):
        po.stream.with_windows(
            second.with_columns(s=pl.lit(1.0)), **y, load_state=state, **sessions
        )


def test_a_typed_literal_keeps_its_dtype() -> None:
    """Task 159 (F3): ``pl.lit(x, dtype=...)``, or a numpy scalar, was rebuilt
    as a dynamic literal, so a formula's dtype and last bits could differ from
    Polars' own; it is carried through a cast, and the two agree."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 4.0, 8.0]})
    mean = po.stream.with_windows(df, m=po.ewm_mean("x", half_life=5.0), **CLOCK)
    forms = [
        lambda c: c * pl.lit(0.1, dtype=pl.Float32),
        lambda c: (
            pl.when(c > 2).then(pl.lit(200, dtype=pl.UInt8)).otherwise(pl.lit(1, dtype=pl.UInt8))
        ),
        lambda c: c.cast(pl.Float32) * pl.lit(2.5, dtype=pl.Float64),
        lambda c: c + pl.lit(np.float32(0.3)),
    ]
    for form in forms:
        got = po.stream.with_windows(df, y=form(po.ewm_mean("x", half_life=5.0)), **CLOCK)["y"]
        want = mean.select(form(pl.col("m")).alias("y"))["y"]
        assert got.dtype == want.dtype, (got.dtype, want.dtype)
        assert got.to_list() == want.to_list()


def test_a_date_literal_keeps_its_dtype() -> None:
    """Review 2026-10-05 (YB1): a ``Date`` or ``Time`` literal was read as its
    bare integer, so ``pl.lit(date).cast(pl.String) == "2024-01-03"`` came out
    false where Polars says true, and ``pl.col("d") - pl.lit(date)`` was
    refused when the plan ran. Each is carried through a cast to its own
    dtype, and the formula gives Polars' own numbers."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 4.0, 8.0]}).with_columns(
        d=pl.lit(date(2024, 1, 1)) + pl.duration(days=pl.col("t").cast(pl.Int64))
    )
    mean = po.stream.with_windows(df, m=po.ewm_mean("x", half_life=5.0), **CLOCK)
    forms = [
        lambda c: (
            pl.when(pl.lit(date(2024, 1, 3)).cast(pl.String) == pl.lit("2024-01-03"))
            .then(c)
            .otherwise(-1.0)
        ),
        lambda c: (pl.col("d") - pl.lit(date(2024, 1, 1))).cast(pl.Int64) + c,
        lambda c: pl.when(pl.col("d") > pl.lit(date(2024, 1, 2))).then(c).otherwise(-1.0),
        lambda c: (
            pl.when(pl.lit(time(12, 30)).cast(pl.String) == pl.lit("12:30:00"))
            .then(c)
            .otherwise(-1.0)
        ),
    ]
    for i, form in enumerate(forms):
        expr = form(po.ewm_mean("x", half_life=5.0))
        got = po.stream.with_windows(df, y=expr, **CLOCK)["y"]
        want = mean.select(form(pl.col("m")).alias("y"))["y"]
        assert got.dtype == want.dtype, (i, got.dtype, want.dtype)
        assert got.to_list() == want.to_list(), i
        assert want.to_list() != [-1.0] * 4, i


@pytest.mark.parametrize(
    ("dtype", "name"),
    [
        (pl.Datetime("us"), "Datetime"),
        (pl.Duration("ms"), "Duration"),
        (pl.Decimal(10, 2), "Decimal"),
        (pl.List(pl.Float64), "List"),
        (pl.Array(pl.Float64, 2), "Array"),
        (pl.Categorical, "Categorical"),
        (pl.Enum(["a"]), "Enum"),
        (pl.Struct({"a": pl.Float64}), "Struct"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_cast_to_a_parametrized_dtype_is_refused_by_name(dtype: Any, name: str) -> None:
    """Review 2026-10-05 (YB2): a dtype with parameters serializes as a
    mapping, which the refusal's lookup could not hash, so the cast died in a
    ``TypeError`` before it was named."""
    with pytest.raises(FormulaError, match=f"a cast to {name} is not read"):
        to_tree(po.ewm_mean("x", half_life=5.0).cast(dtype))


def test_partial_needs_a_window_size() -> None:
    """Review 2026-10-05 (YB5): ``partial`` says what a window cut short by a
    gap or a session change gives, and with no ``window_size`` nothing is cut
    short: it was taken and did nothing. It is refused, as a forward
    operator without a window is."""
    for op in (po.ewm_mean, po.ewm_sum, po.ewm_rate):
        for partial in ("keep", "null", "drop"):
            with pytest.raises(ValueError, match=rf"^po\.{op.__name__}: partial needs window_size"):
                op("x", half_life=2.0, partial=partial)
    po.ewm_mean("x", half_life=2.0, window_size=5.0, partial="drop")


def test_a_literal_that_is_not_finite_is_refused_by_name() -> None:
    """Task 159 (F4): Polars serializes ``inf`` and ``nan`` alike, as a null
    under the float's type, so ``clip(0, inf)`` was refused as "this literal
    (...null)"; the refusal names the reason, and a clip with one bound has
    its own form."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [-1.0, 2.0, 3.0]})
    m = po.ewm_mean("x", half_life=5.0)
    for bad in (m.clip(0, float("inf")), m.clip(float("nan"), 1)):
        with pytest.raises(FormulaError, match="not finite .*inf or nan"):
            po.stream.with_windows(df, y=bad, **CLOCK)
    out = po.stream.with_windows(df, y=m.clip(lower_bound=0), **CLOCK)
    assert out["y"].min() >= 0


@needs_polars("1.43.0", "Expr.ewm_sum_by, the reference, which py-polars added in 1.43.0")
def test_at_a_repeated_stamp_every_row_carries_the_stamps_total() -> None:
    """Task 159 (W4): ``ewm_sum`` is Polars' ``ewm_sum_by`` on distinct stamps
    only. Rows at one stamp share a window, so each carries the stamp's total
    where ``ewm_sum_by`` runs row by row; the docs say so now, and this holds
    the sum to ``ewm_sum_by`` on each stamp's last row and to the stamp's
    total on the others."""
    df = ticks(600, 11, repeats=True).with_columns(
        ts=pl.from_epoch((pl.col("t") * 1e9).cast(pl.Int64), time_unit="ns")
    )
    out = po.stream.with_windows(
        df, s=po.ewm_sum("x", half_life="2s500ms"), clock="ts", gap_cap="1000s"
    )
    ref = df.select(pl.col("x").ewm_sum_by("ts", half_life="2s500ms"))["x"].to_list()
    last = (df["ts"] != df["ts"].shift(-1)).fill_null(True).to_list()
    valued = df["x"].is_not_null().to_list()
    got = out["s"].to_list()
    pairs = [
        (g, r) for g, r, ok, is_last in zip(got, ref, valued, last, strict=True) if ok and is_last
    ]
    assert len(pairs) > 300
    assert_close([g for g, _ in pairs], [r for _, r in pairs], "ewm_sum_by on a stamp's last row")
    tied = 0
    for i in range(len(got) - 1):
        if not last[i] and valued[i] and valued[i + 1]:
            j = i + 1
            while not last[j]:
                j += 1
            assert got[i] == got[j], (i, j)
            tied += 1
    assert tied > 20


# --------------------------------------------------------------------------
# Task 160: the whole-project review of 2026-10-05


def test_a_forward_window_whose_far_edge_is_the_last_row_before_a_break_is_whole() -> None:
    """PC2: a window is the set of the stretch's timestamps in ``(t, t + w]``.
    When its far edge is the stretch's last row, a break after that row
    leaves every one of them arrived -- the backward mirror calls the same
    window complete -- yet the core closed a forward window only on a row
    strictly past its edge, and the break cut it: null by default, and the
    row dropped under ``partial="drop"``. Under ``"left"`` and ``"none"`` the
    far edge is outside, and a row on it closed the window already."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        }
    )

    def run(frame: pl.DataFrame, **kw: Any) -> pl.DataFrame:
        call = {"clock": "t", "gap_cap": 1.5} | kw
        closed = call.pop("closed", "right")
        partial = call.pop("partial", "null")
        f = po.rewm_sum("x", half_life=math.inf, window_size=2.0, closed=closed, partial=partial)
        return po.stream.with_windows(frame, f=f, **call)

    # t = 1: (1, 3] holds 4 + 8; t = 2 and t = 3 reach past the stretch; the
    # last stretch is still open when the input ends.
    assert run(df)["f"].to_list() == [6.0, 12.0, None, None, None, None, None]
    assert run(df, closed="both")["f"].to_list() == [7.0, 14.0, None, None, None, None, None]
    dropped = run(df, partial="drop")
    assert dropped["t"].to_list() == [0.0, 1.0, 10.0, 11.0, 12.0]
    assert dropped["f"].to_list() == [6.0, 12.0, None, None, None]
    # The rule that stands: the far edge outside the window.
    assert run(df, closed="left")["f"].to_list() == [3.0, 6.0, None, None, 48.0, None, None]
    assert run(df, closed="none")["f"].to_list() == [2.0, 4.0, None, None, 32.0, None, None]
    # A session change in place of the gap is the same break.
    sessions = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "s": ["a", "a", "a", "a", "b", "b"],
            "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )
    out = run(sessions, gap_cap=100.0, session="s", session_gap=1.0)
    assert out["f"].to_list()[:2] == [6.0, 12.0]


def test_a_reset_keeps_a_forward_window_whose_far_edge_is_the_last_row_before_it_whole() -> None:
    """Task 173, PC2: a reset -- a step back past ``restart_after_step_back``,
    or a new session under ``session_gap="reset"`` -- discards the windows
    open across it. One whose far edge is the stretch's last row has every
    row it covers, as before a cut (the test above), and is whole under
    every ``partial``; the reset discarded it, null. The windows reaching
    past that row are still discarded: null, and the row never dropped."""
    back = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, -7.0, -6.0, -5.0],
            "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        }
    )
    sessions = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "s": ["a", "a", "a", "a", "b", "b"],
            "x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )

    def run(
        frame: pl.DataFrame, closed: str = "right", partial: str = "null", **kw: Any
    ) -> pl.DataFrame:
        f = po.rewm_sum("x", half_life=math.inf, window_size=2.0, closed=closed, partial=partial)
        return po.stream.with_windows(frame, f=f, clock="t", gap_cap=100.0, **kw)

    step_back: dict[str, Any] = {"restart_after_step_back": 1.0}
    reset: dict[str, Any] = {"session": "s", "session_gap": "reset"}
    # t = 1: (1, 3] holds 4 + 8, and [1, 3] under "both" 2 + 4 + 8; t = 2 and
    # t = 3 reach past the stretch; the last stretch is open at the end.
    assert run(back, **step_back)["f"].to_list() == [6.0, 12.0, None, None, None, None, None]
    assert run(back, "both", **step_back)["f"].to_list() == [7.0, 14.0, *[None] * 5]
    assert run(sessions, **reset)["f"].to_list() == [6.0, 12.0, None, None, None, None]
    assert run(sessions, "both", **reset)["f"].to_list() == [7.0, 14.0, *[None] * 4]
    for partial in ("keep", "drop"):
        out = run(back, partial=partial, **step_back)
        assert out["t"].to_list() == back["t"].to_list(), partial
        assert out["f"].to_list() == [6.0, 12.0, None, None, None, None, None], partial
    # The rule that stands: the far edge outside the window, where the row on
    # it closed the window already.
    assert run(back, "left", **step_back)["f"].to_list() == [3.0, 6.0, None, None, 48.0, None, None]
    assert run(back, "none", **step_back)["f"].to_list() == [2.0, 4.0, None, None, 32.0, None, None]


def test_a_window_looking_ahead_with_group_needs_a_clock_column() -> None:
    """Task 173, PC1: without a clock column each group's clock counts its
    own rows, so a group that fell silent never closed its windows, and
    every later row of every group waited behind them to the end of the
    input -- 500,001 rows held, 117 MiB more over 2M rows -- where the
    call holds about one window. ``gap_cap``, which cuts a silent group on
    a clock, needs a clock. Refused at every door, naming the fix; the
    same calls with a clock column run, and so do a window looking back
    under ``group`` and one looking ahead without it."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0],
            "g": ["a", "b", "b", "b", "b"],
            "x": [1.0, 2.0, 4.0, 8.0, 16.0],
        }
    )
    fwd = po.rewm_sum("x", half_life=math.inf, window_size=2.0)
    msg = (
        "with_windows: a window looking ahead with group needs a clock column.*"
        "Name a clock column, with gap_cap, or leave out group"
    )
    with pytest.raises(ValueError, match=msg):
        po.stream.with_windows(df, f=fwd, group="g")
    with pytest.raises(ValueError, match=msg):
        po.stream.with_windows(df.lazy(), (fwd - pl.col("x")).alias("f"), group="g")
    with pytest.raises(ValueError, match=msg):
        df.lazy().online.with_windows(f=fwd, group="g")
    with pytest.raises(ValueError, match=msg):
        df.online.with_windows(f=fwd, group="g")
    # like= takes its policy from a spec, groups and no clock here.
    like = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0, group="g")
    with pytest.raises(ValueError, match=msg):
        po.stream.with_windows(df, f=fwd, like=like)
    # With a clock column: group a's window is cut once the stream's clock is
    # past the cap; b's row at t = 1 holds 4 + 8; the rest are open at the end.
    clocked = po.stream.with_windows(df, f=fwd, group="g", clock="t", gap_cap=1.5)
    assert clocked["f"].to_list() == [None, 12.0, None, None, None]
    lazy = df.lazy().online.with_windows(f=fwd, group="g", clock="t", gap_cap=1.5).collect()
    assert lazy.equals(clocked)
    assert df.online.with_windows(f=fwd, group="g", clock="t", gap_cap=1.5).equals(clocked)
    like = po.spec.ewridge(
        "m", targets=["y"], features=["x"], half_life=10.0, group="g", clock="t", gap_cap=1.5
    )
    assert po.stream.with_windows(df, f=fwd, like=like).equals(clocked)
    # Looking back under groups, each group's running sum; looking ahead with
    # no group, the one stream's rows.
    back = po.stream.with_windows(df, b=po.ewm_sum("x", half_life=math.inf), group="g")
    assert back["b"].to_list() == [1.0, 2.0, 6.0, 14.0, 30.0]
    assert po.stream.with_windows(df, f=fwd)["f"].to_list() == [6.0, 12.0, None, None, None]


def test_a_state_resumes_under_another_spelling_of_the_same_length(tmp_path: Any) -> None:
    """PC3: a duration was compared as written, so a state saved with
    ``gap_cap="5s"`` refused ``"5000ms"`` as another clock policy, and one
    saved with ``window_size="2s"`` refused ``"2000ms"``. A length is its
    nanoseconds. The forward operator holds rows across the save, so the
    resumed run reads the rows the state holds as well as its own."""
    df = pl.DataFrame(
        {
            "t": pl.Series([0, 1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000]).cast(
                pl.Datetime("ms")
            ),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )

    def run(frame: pl.DataFrame, gap_cap: str, window: str, **kw: Any) -> pl.DataFrame:
        return po.stream.with_windows(
            frame,
            b=po.ewm_sum("x", half_life="1h", window_size=window),
            f=po.rewm_sum("x", half_life="1h", window_size=window),
            clock="t",
            gap_cap=gap_cap,
            **kw,
        )

    whole = run(df, "5s", "2s")
    assert whole["f"].drop_nulls().len() > 3
    state = tmp_path / "w.state"
    first = run(df.head(4), "5s", "2s", save_state=state)
    for gap_cap, window in [("5000ms", "2s"), ("5s", "2000ms"), ("5000000us", "2000ms")]:
        second = run(df.slice(4), gap_cap, window, load_state=state)
        assert pl.concat([first, second]).equals(whole), (gap_cap, window)
    # Another length is still another call.
    with pytest.raises(ValueError, match="another call"):
        run(df.slice(4), "6s", "2s", load_state=state)


def test_a_zoned_datetime_group_is_keyed_by_its_instant() -> None:
    """PA4b: a zoned Datetime group or session column failed inside the
    run with polars' own message, this build formatting no time zone. Its
    keys are its instants: the windows of the UTC instants as a naive
    column, whatever zone shows them -- Amsterdam shows the two groups at
    one wall time -- and the rows come back with the input's own values."""
    from test_error_messages import zoned_keys

    def run(frame: pl.DataFrame) -> pl.DataFrame:
        return po.stream.with_windows(
            frame,
            b=po.ewm_sum("x0", half_life=3.0),
            f=po.rewm_sum("x0", half_life=3.0, window_size=4.0),
            clock="t",
            gap_cap=100.0,
            group="g",
            session="s",
            session_gap=1.0,
        )

    want = run(zoned_keys(None))
    assert want["f"].drop_nulls().len() > 20
    for tz in ("Europe/Amsterdam", "America/New_York"):
        df = zoned_keys(tz)
        out = run(df)
        assert out.select("b", "f").equals(want.select("b", "f")), tz
        assert out.select(df.columns).equals(df), tz


# --------------------------------------------------------------------------
# Review round 4 (PD3-PD10)


@pytest.mark.parametrize(
    "name",
    [
        '@po:["ewm_mean"]',
        '@po:["increment"]',
        "@po:5",
        '@po:{"a":1}',
        '@po:["nope",["col","x"],{}]',
        '@po:["col","x"]',
        '@po:["ewm_mean",["col","x"],{"half_life":1},4]',
    ],
)
def test_a_hand_built_operator_column_that_is_no_operator_is_refused_as_reserved(
    name: str,
) -> None:
    """Review round 4 (PD3): a column under the operators' prefix whose JSON
    was a short list raised a bare ``IndexError`` from the tree's walk, and
    one that was no list at all was refused as "holds no operator". Only a
    list headed by an operator, with its input and its parameters, is an
    operator's column; anything else under the prefix is the reserved-name
    refusal."""
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
    with pytest.raises(FormulaError, match="reserved for the operators' own columns"):
        po.stream.with_windows(df, y=pl.col(name), **CLOCK)
    # The operators' own form, written by hand, is the operator.
    hand = pl.col('@po:["ewm_mean",["col","x"],{"half_life":1}]')
    out = po.stream.with_windows(df, y=hand, **CLOCK)
    assert (
        out["y"].to_list()
        == po.stream.with_windows(df, y=po.ewm_mean("x", half_life=1.0), **CLOCK)["y"].to_list()
    )


def test_min_samples_takes_an_integer_of_any_kind_up_to_its_ceiling() -> None:
    """Review round 4 (PD4): ``min_samples=2**40`` passed here and was refused
    later with the message for a count below 1, and a numpy integer, which
    Polars' ``rolling_*_by`` takes, was refused. Any integer from 1 to the
    ceiling, 4294967295, is taken; past it the refusal names the ceiling, on
    both sides."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, None, 2.0, 3.0]})
    for n in (2, np.int64(2), np.int32(2), np.uint8(2)):
        e = po.ewm_sum("x", half_life=math.inf, window_size=3.0, min_samples=n)
        assert po.stream.with_windows(df, s=e, **CLOCK)["s"].to_list() == [None, None, 3.0, 5.0]
    po.ewm_sum("x", half_life=2.0, min_samples=4294967295)
    for bad in (2**32, 2**40, np.int64(2**40)):
        with pytest.raises(ValueError, match=r"^po\.ewm_sum: min_samples must be .*4294967295"):
            po.ewm_sum("x", half_life=2.0, min_samples=bad)
    for bad in (0, -1, True, 2.0, "2"):
        with pytest.raises(ValueError, match=r"^po\.ewm_sum: min_samples must be an integer"):
            po.ewm_sum("x", half_life=2.0, min_samples=bad)  # type: ignore[arg-type]
    # A tree written elsewhere -- by hand, in TOML, in a state file -- meets
    # the Rust side's check, which names the ceiling too.
    tree = '@po:["ewm_sum",["col","x"],{"half_life":2.0,"min_samples":1099511627776}]'
    with pytest.raises(ValueError, match="min_samples must be .* at most 4294967295"):
        po.stream.with_windows(df, s=pl.col(tree), **CLOCK)


def test_an_increment_of_a_time_of_day_is_seconds_and_negative_across_midnight() -> None:
    """Review round 4 (PD5): ``increment`` of a ``Time`` column was refused
    with "a time column cannot be read as a clock", where the docstring
    promises seconds on a temporal column and the column was the increment's
    input, not the clock. A time of day steps by its seconds since midnight,
    and a step across midnight is negative, as Polars' ``diff`` on a ``Time``
    gives a negative ``Duration``. A ``Time`` clock is still refused, as a
    clock, since it starts again each midnight."""
    tod = [time(9, 0), time(10, 30), None, time(23, 0), time(1, 0, 0, 500)]
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "tod": tod})
    got = po.stream.with_windows(df, d=po.increment("tod"), **CLOCK)["d"].to_list()
    # The null is skipped: 23:00 steps from 10:30, the last time with a value.
    assert got == [None, 5400.0, None, 45000.0, -79199.9995]
    diff = df.select(pl.col("tod").diff().dt.total_nanoseconds() / 1e9)["tod"].to_list()
    assert got[1] == pytest.approx(diff[1], rel=1e-15), diff
    assert got[4] == pytest.approx(diff[4], rel=1e-15), diff
    with pytest.raises(_REFUSAL, match="time column cannot be read as a clock"):
        po.stream.with_windows(df, d=po.increment("t"), clock="tod", gap_cap="1h")


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"half_life": "nan"}, "half_life must not be NaN"),
        ({"half_life": "NaN"}, "half_life must not be NaN"),
        ({"half_life": "-inf"}, "half_life must be finite and above 0, got -inf"),
        ({"half_life": "-Infinity"}, "half_life must be finite and above 0, got -Infinity"),
        ({"half_life": "reset"}, "half_life must be a number or a duration, got 'reset'"),
        ({"half_life": 2.0, "window_size": "inf"}, "window_size must be finite and above 0"),
        ({"half_life": 2.0, "window_size": "+Infinity"}, "window_size must be finite and above"),
        ({"half_life": 2.0, "window_size": "nan"}, "window_size must not be NaN"),
        ({"half_life": timedelta(seconds=-5)}, "half_life must be above 0, got -5s"),
        ({"half_life": timedelta(0)}, "half_life must be above 0, got 0s"),
        ({"half_life": pl.duration(seconds=-5)}, "half_life must be above 0, got -5s"),
        ({"half_life": "-5s"}, "half_life must be above 0, got -5s"),
        ({"half_life": "1s", "window_size": "0s"}, "window_size must be above 0, got 0s"),
        ({"half_life": "1s", "window_size": timedelta(minutes=-1)}, "window_size must be above 0"),
    ],
)
def test_a_word_or_a_duration_not_above_zero_is_refused_where_it_is_written(
    kw: dict[str, Any], says: str
) -> None:
    """Review round 4 (PD9): ``"nan"``, ``"-inf"`` and ``"reset"``, an
    infinite ``window_size`` written as a word, and a duration at or below 0
    passed ``po.<op>`` and were refused only when the plan was built, under a
    serde path ("invalid windows: formulas[0].tree: ..."), where the same
    value as a number is refused at once. Each is refused in ``po.<op>``'s
    voice now, as the number is."""
    with pytest.raises(ValueError, match=rf"^po\.ewm_sum: {says}"):
        po.ewm_sum("x", **kw)


def test_the_infinity_words_are_still_an_unbounded_half_life() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0, 4.0]})
    want = po.stream.with_windows(df, s=po.ewm_sum("x", half_life=math.inf), **CLOCK)["s"]
    for word in ("inf", "+inf", "Infinity", "+infinity", " INF "):
        got = po.stream.with_windows(df, s=po.ewm_sum("x", half_life=word), **CLOCK)["s"]
        assert got.to_list() == want.to_list() == [1.0, 3.0, 6.0, 10.0], word


def test_the_docs_say_what_a_null_row_and_a_streams_start_give() -> None:
    """Review round 4 (PD6, PD7): at a row whose value is missing the
    operator gives the window as it stands, where Polars' ``ewm_mean_by``
    gives null -- stated in a test and in ``increment``'s docstring, and not
    where the operators are defined; and a backward window reaching before
    its stretch's first row is partial, the stream's own start included, so
    ``partial`` governs every stream's first ``window_size`` too."""
    doc = " ".join((po.ops.__doc__ or "").split())
    assert "where Polars' ``ewm_mean_by`` gives null" in doc
    assert "the stream's own start included" in doc
    dn = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, None, 3.0, None]})
    held = po.stream.with_windows(dn, m=po.ewm_mean("x", half_life=1.0), **CLOCK)
    assert held["m"].to_list() == [1.0, 1.0, 2.5, 2.5]
    df = pl.DataFrame({"t": [float(i) for i in range(7)], "x": [1.0] * 7})
    polars = df.select(pl.col("x").rolling_sum_by(pl.col("t").cast(pl.Int64), "3i"))["x"]
    for partial, want in (
        ("keep", polars.to_list()),
        ("null", [None, None, None, 3.0, 3.0, 3.0, 3.0]),
        ("drop", [3.0, 3.0, 3.0, 3.0]),
    ):
        e = po.ewm_sum("x", half_life=math.inf, window_size=3.0, partial=partial)
        assert po.stream.with_windows(df, s=e, **CLOCK)["s"].to_list() == want, partial


@NEEDS_ROLLING_NULLS
@pytest.mark.parametrize("min_samples", [1, 3])
@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_which_rows_a_window_holds_is_polars_rolling_on_a_long_random_stream(
    closed: str, min_samples: int
) -> None:
    """Review round 4 (PD10): "a window holds the rows Polars' ``rolling_*_by``
    would give it" was held on nine hand rows without nulls or
    ``min_samples``, and the brute force shares the membership rule by
    construction. Three seeds of 1,500 random rows, with nulls (12%), bursts
    of repeated stamps and gaps, on a ``Datetime`` clock, against
    ``rolling_sum_by``: every value and every null the same."""
    valued = 0
    for seed in (3, 11, 21):
        df = ticks(1500, seed).with_columns(
            ts=pl.from_epoch((pl.col("t") * 1e9).round().cast(pl.Int64), time_unit="ns")
        )
        e = po.ewm_sum(
            "x", half_life=math.inf, window_size="4s", closed=closed, min_samples=min_samples
        )
        out = po.stream.with_windows(df, s=e, clock="ts", gap_cap="1000s")["s"].to_list()
        ref = df.select(
            pl.col("x").rolling_sum_by(
                "ts", window_size="4s", closed=closed, min_samples=min_samples
            )
        )["x"].to_list()
        assert_close(out, ref, f"seed {seed}")
        valued += sum(v is not None for v in out)
    assert valued > 1000, valued


# --------------------------------------------------------------------------
# The variance and the standard deviation (task 212)


def _var_ops(**kw: Any) -> dict[str, pl.Expr]:
    return {"v": po.ewm_var("x", **kw), "s": po.ewm_std("x", **kw)}


#: Polars' ``ewm_var``/``ewm_std`` under ``bias=False`` give the first row
#: null from 1.44 (measured: 0.0 on 1.42.0, null on 1.44.1 and 2.0.0; 1.43
#: does not install on Python 3.12 to narrow it), the definition
#: ``po.ewm_var`` follows; the canary's floor leg met the old 0.0 on
#: 2026-10-09.
NEEDS_UNBIASED_FIRST_NULL = needs_polars(
    "1.44.1", "Polars' ewm_var(bias=False), the reference, gives the first row null from 1.44"
)


@pytest.mark.parametrize("bias", [pytest.param(False, marks=NEEDS_UNBIASED_FIRST_NULL), True])
@pytest.mark.parametrize("clock", [True, False])
def test_var_and_std_on_a_row_clock_are_polars_ewm_var_and_ewm_std(bias: bool, clock: bool) -> None:
    """On a clock that steps by 1 a row, with ``half_life`` in rows, the
    operators are Polars' ``ewm_var`` and ``ewm_std`` under ``adjust=False``:
    the form ``po.ewm_mean`` mirrors, since ``ewm_mean_by`` holds a stretch's
    first value from before it, which gives it the weight ``(1 - alpha) ** (n
    - 1)`` that ``adjust=False`` gives the first row. ``bias`` and
    ``min_samples`` are Polars' own, and so is the null on the first row
    under ``bias=False``: one value has no spread to correct. With a clock
    column and on the row count."""
    rng = np.random.default_rng(41)
    n = 600
    df = pl.DataFrame({"t": np.arange(n, dtype=float), "x": rng.normal(size=n) * 3 + 1})
    for h, min_samples in ((1.0, 1), (7.5, 3), (60.0, 1)):
        kw: dict[str, Any] = {"half_life": h, "bias": bias, "min_samples": min_samples}
        out = po.stream.with_windows(df, **_var_ops(**kw), **(CLOCK if clock else {}))
        ref = df.select(
            v=pl.col("x").ewm_var(**kw, adjust=False),
            s=pl.col("x").ewm_std(**kw, adjust=False),
        )
        for c in ("v", "s"):
            assert_close(out[c].to_list(), ref[c].to_list(), f"{c} h={h}", tol=1e-12)
        nulls = max(min_samples - 1, 0 if bias else 1)
        assert out["v"].null_count() == nulls


@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
@pytest.mark.parametrize(
    "op",
    [
        "ewm_var",
        # The essentials keep the variance at every edge; the standard
        # deviation is its square root from the same pass.
        pytest.param(
            "ewm_std",
            marks=pytest.mark.extended(reason="a grid: eight cases of 0.32 s, 2.6 s in all"),
        ),
    ],
)
def test_var_and_std_match_the_definition_on_an_irregular_clock(op: str, closed: str) -> None:
    """On an irregular clock -- bursts, gaps, repeated stamps, nulls -- a value
    weighs what it weighs in ``po.ewm_mean``: its held interval's decayed mass
    inside the window. The variance is about that mean, taken in two passes,
    and corrected for the unequal weights as Polars corrects its own
    (``loop``'s docstring)."""
    df = ticks(700, 3)
    fn = getattr(po, op)
    for bias in (False, True):
        for window in (7.0, None):
            e = fn("x", half_life=4.0, window_size=window, closed=closed, min_samples=2, bias=bias)
            out = po.stream.with_windows(df, y=e, **CLOCK)
            want = loop(
                df,
                op,
                "x",
                half_life=4.0,
                window_size=window,
                closed=closed,
                min_samples=2,
                bias=bias,
            )
            assert_close(out["y"].to_list(), want, f"{op} {closed} {window} {bias}")
            assert sum(v is not None for v in want) > 450


def _zscore() -> dict[str, pl.Expr]:
    return {
        "v": po.ewm_var("x", half_life=2.0, window_size=5.0),
        "s": po.ewm_std("x", half_life=3.0, bias=True),
        "z": (pl.col("x") - po.ewm_mean("x", half_life=3.0)) / po.ewm_std("x", half_life=3.0),
        "m": po.ewm_mean("x", half_life=2.0, window_size=5.0, closed="left"),
    }


@pytest.mark.parametrize("rows", [1, 7, 600])
def test_var_and_std_chunking_changes_nothing(rows: int) -> None:
    """Hard rule 3, with a mean on the same kernel as a variance and on one of
    its own."""
    df = ticks(800, 17, groups=3)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = po.stream.with_windows(df, **_zscore(), **kw, chunk_size=10_000)
    out = po.stream.with_windows(df, **_zscore(), **kw, chunk_size=rows)
    assert out.equals(one)
    assert one["z"].is_not_null().sum() > 600
    # A mean on a variance's kernel, whose queue is six wide, gives the
    # bits it gives on its own.
    mean = po.ewm_mean("x", half_life=3.0)
    alone = po.stream.with_windows(df, m=mean, **kw, chunk_size=rows)["m"]
    beside = po.stream.with_windows(df, m=mean, s=po.ewm_std("x", half_life=3.0), **kw)["m"]
    assert alone.equals(beside)


def test_var_and_std_resume_through_a_state_at_any_row(tmp_path: Any) -> None:
    df = ticks(60, 19, groups=2)
    kw: dict[str, Any] = {"clock": "t", "gap_cap": 20.0, "group": "g"}
    one = po.stream.with_windows(df, **_zscore(), **kw)
    for at in (1, 2, 17, 33, 59):
        state = tmp_path / f"v{at}.state"
        first = po.stream.with_windows(df[:at], **_zscore(), **kw, save_state=state)
        second = po.stream.with_windows(df[at:], **_zscore(), **kw, load_state=state)
        assert pl.concat([first, second]).equals(one), at


def test_a_null_row_gives_the_window_as_it_stands_and_the_next_value_its_interval() -> None:
    """Nulls as ``po.ewm_mean`` takes them: a row with no value gives the
    window as it stands, where Polars' ``ewm_var`` gives null, and the next
    value is held from the last valued row, as ``ewm_mean_by`` skips a null.
    So row 2's 4.0 is held over ``(0, 2]``, two units at a half-life of 2,
    and weighs what the 1.0 before it does: the variance of two equal
    weights at 1.0 and 4.0, ``2.25``."""
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "x": [1.0, None, 4.0, 3.0, None, 5.0]})
    out = po.stream.with_windows(df, **_var_ops(half_life=2.0, bias=True), **CLOCK)
    v = out["v"].to_list()
    # The window seen a unit later: its weights all decayed alike, the same
    # variance to a rounding.
    assert v[0] == 0.0 and v[1] == v[0] and v[4] == pytest.approx(v[3], rel=1e-15)
    assert v[2] == pytest.approx(2.25, rel=1e-12)
    assert_close(v, loop(df, "ewm_var", "x", half_life=2.0, bias=True), tol=1e-12)
    assert_close(out["s"].to_list(), [math.sqrt(x) for x in v], tol=1e-15)


@pytest.mark.parametrize("level", [0.1, 1e8 + 0.1, -3e12])
def test_a_constant_input_has_a_variance_of_exactly_zero(level: float) -> None:
    """Every window holds one value however many rows it weighs, so its
    variance is 0, not a rounding of 0, at any level: two partial windows
    are merged about their means, whose difference is 0 (a mean square less
    a squared mean would cancel to noise at a level). Repeated stamps,
    nulls, cuts and every ``closed``; under ``bias=False`` a window weighing
    a single row is null, as in Polars."""
    df = ticks(400, 5).with_columns(x=pl.when(pl.col("x").is_not_null()).then(pl.lit(level)))
    for kw in (
        {"half_life": 3.0},
        {"half_life": 2.0, "window_size": 5.0},
        {"half_life": math.inf, "window_size": 6.0, "closed": "both"},
        {"half_life": 5.0, "window_size": 7.0, "closed": "none"},
    ):
        out = po.stream.with_windows(
            df,
            b=po.ewm_var("x", **kw, bias=True),
            u=po.ewm_var("x", **kw),
            s=po.ewm_std("x", **kw),
            **CLOCK,
        )
        for c in ("b", "u", "s"):
            got = [x for x in out[c].to_list() if x is not None]
            assert len(got) > 250 and all(x == 0.0 for x in got), (kw, c)


def test_a_variance_at_a_level_decays_with_its_history() -> None:
    """The compensated mean (``crate::comp``, docs/PLAN.md task 101): a value
    held row after row is approached as exact arithmetic approaches it. At
    1e8 a double mean stalls a few rounding steps short of a new level, and
    a variance about it settles on that gap's square, about 1e-15, where the
    definition decays as ``p (1 - p)``, ``p = 2 ** (-T / half_life)`` the
    weight left on the old level ``T`` units after its last row (the rows
    before the step hold ``(-inf, t_49]``, the rows after it the rest)."""
    a, n_a, n, h = 1e8, 50, 500, 5.0
    df = pl.DataFrame({"t": np.arange(n, dtype=float), "x": [a] * n_a + [a + 1.0] * (n - n_a)})
    v = po.stream.with_windows(df, v=po.ewm_var("x", half_life=h, bias=True), **CLOCK)["v"]
    for t in (n_a + 20, n_a + 100, n_a + 200, n - 1):
        p = 2.0 ** (-(t - (n_a - 1)) / h)
        want = p * (1.0 - p)
        assert abs(v[t] - want) <= 1e-9 * want, (t, v[t], want)


def test_bias_is_a_variances_own_and_a_boolean() -> None:
    with pytest.raises(TypeError, match="bias"):
        po.ewm_mean("x", half_life=2.0, bias=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="bias must be a bool"):
        po.ewm_var("x", half_life=2.0, bias=1)  # type: ignore[arg-type]
    df = ticks(50, 1)
    with pytest.raises(ValueError, match="half_life = inf needs a window_size"):
        po.stream.with_windows(df, y=po.ewm_std("x", half_life=math.inf), **CLOCK)
    tree = to_tree(po.ewm_std("x", half_life=2.0, bias=True) - pl.col("x"))
    assert to_tree(from_tree(tree)) == tree
    assert tree[1][2]["bias"] is True


@NEEDS_UNBIASED_FIRST_NULL
def test_the_readmes_zscore_recipe_runs_and_is_polars_own_on_a_row_clock(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The README's *Features in units of their spread* block, run as
    written on a ``ticks.parquet`` of its own: the z-scores it makes are
    Polars' ``ewm_mean`` and ``ewm_std`` under ``adjust=False`` on its clock
    that steps by 1, as the paragraph after it says, and the model fits on
    them."""
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    section = readme.split("#### Features in units of their spread\n", 1)[1].split("\n#### ")[0]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {"t": np.arange(400.0), **{c: rng.standard_normal(400) for c in ("x0", "x1", "y")}}
    )
    monkeypatch.chdir(tmp_path)
    df.write_parquet("ticks.parquet")
    ns: dict[str, Any] = {"pl": pl, "po": po}
    exec(compile(code, "README.md: Features in units of their spread", "exec"), ns)
    fitted = ns["fitted"]
    for c in ("x0", "x1"):
        x = pl.col(c)
        z = (x - x.ewm_mean(half_life=100.0, adjust=False)) / x.ewm_std(
            half_life=100.0, adjust=False
        )
        assert_close(fitted[f"z_{c}"].to_list(), df.select(z)[c].to_list(), c, tol=1e-12)
    assert fitted["per_sd"].struct.field("pred_y").is_not_null().sum() > 300
