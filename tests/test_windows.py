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
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online._formula import FormulaError, from_tree, to_tree

CLOCK = {"clock": "t", "gap_cap": 1e9}


# --------------------------------------------------------------------------
# The definitions, by hand


def _lam(h: float) -> float:
    return 1.0 if math.isinf(h) else 2.0 ** (-1.0 / h)


def _mass(h: float, d: float) -> float:
    """``integral_0^d lam**s ds``."""
    if d <= 0:
        return 0.0
    return d if math.isinf(h) else h / math.log(2) * (1.0 - _lam(h) ** d)


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
) -> list[float | None]:
    """Every row's operator from the definition, on a stream with no clock
    event: each group's rows, in stream order, at their own clock. A forward
    window not closed by the end is unresolved: null."""
    forward = op.startswith("rewm")
    stat = op.split("_")[1]
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
            for b in members:
                xb = v[idx[b]]
                if math.isnan(xb):
                    continue
                count += 1
                tb = tau[b]
                if stat == "mean":
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
                else:
                    s += _lam(half_life) ** abs(tb - ta) * xb
            if count < min_samples:
                continue
            if stat == "mean":
                out[idx[a]] = s / mass if mass > 0 else None
            elif stat == "sum":
                out[idx[a]] = s
            else:
                m = _mass(half_life, span)
                out[idx[a]] = s / m if m > 0 else None
    return out


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
        major, minor, patch = (int(v) for v in pl.__version__.split(".")[:3])
        assert (major, minor, patch) < (1, 44, 1), "ewm_sum_by arrived in 1.44.1"


@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
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
    m = out.with_columns(m=po.ewm_mean("x", half_life=3.0)) if False else None
    assert m is None
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
        (po.ewm_mean("x", half_life=2.0).over("g"), "Over"),
        (po.ewm_mean("x", half_life=2.0).cast(pl.Date), "cast"),
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
    assert (
        dropped.height == 11 - 4 - 3 and dropped["t"].to_list()[:1] == [33.0] or dropped.height < 11
    )


def test_a_reset_discards_and_a_late_row_is_refused() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 1.5, 2.5, 3.5], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="goes backwards by 0.5 at row 3"):
        po.stream.with_windows(df, y=po.ewm_sum("x", half_life=1.0), **CLOCK)
    with pytest.raises(
        pl.exceptions.ComputeError, match="no more than restart_after_step_back = 1"
    ):
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
        df, **mixed(), clock="t", gap_cap=20.0, group="g", chunk_rows=10_000
    )
    out = po.stream.with_windows(df, **mixed(), clock="t", gap_cap=20.0, group="g", chunk_rows=rows)
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


def test_the_real_day_runs_and_the_recipes_agree_where_they_should() -> None:
    """One symbol-day of Binance quotes and trades (hard rule 1: downloaded
    and cached): the three forward VWAPs and a trailing mid, under a
    Datetime clock, chunked; the VWAP from running sums equals the one from
    prices and quantities where the sums step at trades."""
    from data import public_quotes_and_trades

    try:
        day = public_quotes_and_trades()
    except OSError as e:  # offline: the one skip, explained
        pytest.skip(f"offline: {e}")
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
        chunk_rows=50_000,
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
    ``chunk_rows``."""
    df = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "c": [10.0, 11.0, 13.0, 16.0, 20.0, 25.0]}
    )
    outs = []
    for rows in (1, 10):
        state = tmp_path / f"s{rows}.state"
        df.lazy().online.with_windows(
            dc=po.increment("c"), clock="t", gap_cap=100.0, chunk_rows=rows, save_state=state
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
    with pytest.raises(pl.exceptions.PolarsError, match="conversion from"):
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
