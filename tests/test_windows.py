"""``po.stream.with_windows`` and ``po.window`` (docs/PLAN.md task 78).

Each direction against a loop written from the definition, and against
polars' own ``rolling`` recipe -- the third-party oracle, on data it can
hold. The loop covers what ``rolling`` cannot express: stream order at one
clock (``same_clock="include"``), groups, and the interleaved trades and
quotes the task was written for. The clock events -- a gap past the cap, a
session change, a reset -- are checked against the rule each follows. The
window core itself is tested alone against a brute-force loop over every
event in ``crates/online-polars/src/windows.rs``.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from polars_online import _polars_online as _native


def _counts(v: Any, w: Any) -> bool:
    return (
        v is not None
        and w is not None
        and math.isfinite(v)
        and math.isfinite(w)
        and abs(v) <= 1e100
        and abs(w) <= 1e100
        and w != 0
    )


def _mean(
    members: list[tuple[float, float, float]], anchor: float, halflife: float
) -> float | None:
    """``sum w lam^|t - a| v / sum w lam^|t - a|`` over ``(t, v, w)``."""
    if not members:
        return None
    lam = 1.0 if math.isinf(halflife) else 2.0 ** (-1.0 / halflife)
    s = sum(w * lam ** abs(t - anchor) * v for t, v, w in members)
    d = sum(w * lam ** abs(t - anchor) for t, _, w in members)
    return s / d if d > 0 else None


def loop(
    df: pl.DataFrame,
    *,
    value: str,
    halflife: float,
    horizon: float | None,
    forward: bool,
    weight: str | None = None,
    same_clock: str = "include",
    clock: str = "t",
    group: str | None = None,
) -> tuple[list[float | None], list[bool]]:
    """Every row's window from the definition, on a stream with no clock
    event: each group's rows, in stream order, at their own clock. A forward
    window whose horizon has not passed by the end is unresolved: null."""
    t = df[clock].to_list()
    v = df[value].to_list()
    w = df[weight].to_list() if weight is not None else [1.0] * df.height
    g = df[group].to_list() if group is not None else [None] * df.height
    rows: dict[Any, list[int]] = defaultdict(list)
    for i, key in enumerate(g):
        rows[key].append(i)
    values: list[float | None] = [None] * df.height
    complete = [False] * df.height
    for idx in rows.values():
        for a, i in enumerate(idx):
            if forward:
                assert horizon is not None
                later = idx[a + 1 :]
                members = [
                    (t[j], v[j], w[j])
                    for j in later
                    if t[j] - t[i] < horizon
                    and (same_clock == "include" or t[j] > t[i])
                    and _counts(v[j], w[j])
                ]
                if any(t[j] - t[i] >= horizon for j in later):
                    complete[i] = True
                    values[i] = _mean(members, members[0][0] if members else 0.0, halflife)
            else:
                members = [
                    (t[j], v[j], w[j])
                    for j in idx[: a + 1]
                    if (horizon is None or t[i] - t[j] < horizon) and _counts(v[j], w[j])
                ]
                complete[i] = horizon is None or t[i] - t[idx[0]] >= horizon
                values[i] = _mean(members, members[-1][0] if members else 0.0, halflife)
    return values, complete


def assert_close(got: list[float | None], want: list[float | None], what: str = "") -> None:
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        if a is None or b is None:
            assert a is None and b is None, f"{what} row {i}: {a} vs {b}"
        else:
            assert abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b)), f"{what} row {i}: {a} vs {b}"


def ticks(n: int, seed: int, *, groups: int = 1, unique: bool = False) -> pl.DataFrame:
    """Irregular ticks at whole clock units -- so a step of the policy clock
    is exact and a row exactly a horizon away is exercised exactly -- with
    repeated stamps unless ``unique``, nulls in the value and zeros and nulls
    in the weight."""
    rng = np.random.default_rng(seed)
    steps = rng.choice([1, 2, 3] if unique else [0, 0, 1, 2, 3], n)
    v = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    v_null = rng.random(n) < 0.1
    w_zero = rng.random(n) < 0.1
    w_null = rng.random(n) < 0.1
    return pl.DataFrame(
        {
            "t": np.cumsum(steps).astype(np.int64),
            "g": rng.integers(0, groups, n).astype(str),
            "x": [None if m else float(a) for a, m in zip(v, v_null, strict=True)],
            "w": [
                None if nn else (0.0 if z else float(a))
                for a, z, nn in zip(w, w_zero, w_null, strict=True)
            ],
        }
    )


CLOCK = {"clock": "t", "max_dclock": 1e6}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("same_clock", ["include", "exclude"])
def test_each_direction_matches_the_definition(seed: int, same_clock: str) -> None:
    df = ticks(400, seed, groups=3)
    out = po.stream.with_windows(
        df,
        [
            po.window.ewm("x", weight="w", halflife=4, horizon=10, complete="b_ok"),
            po.window.ewm("x", halflife=6, name="x_all"),
            po.window.lookahead_rewm(
                "x", weight="w", halflife=3, horizon=7, same_clock=same_clock, complete="f_ok"
            ),
        ],
        group="g",
        **CLOCK,
    )
    for name, ok, kwargs in [
        ("x_ewm_4_10", "b_ok", {"halflife": 4, "horizon": 10, "forward": False, "weight": "w"}),
        ("x_all", None, {"halflife": 6, "horizon": None, "forward": False}),
        (
            "x_rewm_3_7",
            "f_ok",
            {"halflife": 3, "horizon": 7, "forward": True, "weight": "w", "same_clock": same_clock},
        ),
    ]:
        want, complete = loop(df, value="x", group="g", **kwargs)  # type: ignore[arg-type]
        assert_close(out[name].to_list(), want, name)
        if ok is not None:
            assert out[ok].to_list() == complete, name


def rolling_recipe(df: pl.DataFrame, horizon: int, halflife: float, forward: bool) -> pl.Series:
    """The recipe docs/PLAN.md measured: ``rolling`` with each window's
    weights taken from its own anchor, forward over ``(t, t + h)``
    (``closed="none"``) and backward over ``(t - h, t]`` (the default)."""
    lam = 2.0 ** (-1.0 / halflife)
    w = pl.when(pl.col("x").is_not_null()).then(pl.col("w").fill_null(0.0)).otherwise(0.0)
    anchor = pl.col("t").min() if forward else pl.col("t").max()
    f = pl.lit(lam).pow((pl.col("t") - anchor).abs().cast(pl.Float64))
    agg = ((w * f * pl.col("x").fill_null(0.0)).sum() / (w * f).sum()).alias("y")
    if forward:
        r = df.rolling(index_column="t", period=f"{horizon}i", offset="0i", closed="none")
    else:
        r = df.rolling(index_column="t", period=f"{horizon}i")
    y = r.agg(agg)["y"]
    return y.fill_nan(None)


@pytest.mark.parametrize("seed", range(3))
def test_each_direction_matches_polars_rolling(seed: int) -> None:
    """``rolling`` cannot hold two rows at one stamp in stream order, so the
    stamps here are unique; where it can hold the data, it is the oracle."""
    df = ticks(600, seed, unique=True)
    out = po.stream.with_windows(
        df,
        [
            po.window.ewm("x", weight="w", halflife=5, horizon=12),
            po.window.lookahead_rewm(
                "x", weight="w", halflife=5, horizon=12, same_clock="exclude", complete="ok"
            ),
        ],
        **CLOCK,
    )
    assert_close(
        out["x_ewm_5_12"].to_list(), rolling_recipe(df, 12, 5, forward=False).to_list(), "back"
    )
    # A forward window whose horizon passes within the data; `rolling` also
    # gives the unresolved ones at the end, which are null here.
    ok = out["ok"].to_list()
    fwd = rolling_recipe(df, 12, 5, forward=True).to_list()
    assert_close(
        [y for y, k in zip(out["x_rewm_5_12"].to_list(), ok, strict=True) if k],
        [y for y, k in zip(fwd, ok, strict=True) if k],
        "forward",
    )
    assert sum(ok) > 500 and all(
        y is None for y, k in zip(out["x_rewm_5_12"], ok, strict=True) if not k
    )


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


@pytest.mark.parametrize("same_clock", ["include", "exclude"])
def test_interleaved_trades_three_vwaps_and_one_split(same_clock: str) -> None:
    df = trades_and_quotes(3000, 7).with_columns(
        buy_qty=pl.when(pl.col("side") == "buy").then(pl.col("quantity")),
        sell_qty=pl.when(pl.col("side") == "sell").then(pl.col("quantity")),
    )
    common = {"halflife": 10, "horizon": 60, "same_clock": same_clock}
    three = po.stream.with_windows(
        df,
        [
            po.window.lookahead_rewm("price", weight=w, name=f"vwap_{w}", **common)  # type: ignore[arg-type]
            for w in ("quantity", "buy_qty", "sell_qty")
        ],
        **CLOCK,
    )
    for w in ("quantity", "buy_qty", "sell_qty"):
        want, _ = loop(df, value="price", weight=w, forward=True, **common)  # type: ignore[arg-type]
        assert_close(three[f"vwap_{w}"].to_list(), want, w)
    # The windows with no trade of a side are there, and null.
    assert three["vwap_buy_qty"].null_count() > three["vwap_quantity"].null_count()
    split = po.stream.with_windows(
        df,
        [
            po.window.lookahead_rewm(
                "price",
                weight="quantity",
                split=("side", ["buy", "sell"]),
                name="vwap{split}",
                complete="ok",
                **common,  # type: ignore[arg-type]
            )
        ],
        **CLOCK,
    )
    # To the bit: the same rows in the same queues, summed the same way.
    assert split["vwap"].equals(three["vwap_quantity"])
    assert split["vwap_buy"].equals(three["vwap_buy_qty"])
    assert split["vwap_sell"].equals(three["vwap_sell_qty"])


def test_an_empty_window_is_null_and_complete() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 9], "p": [1.0, None, None, 5.0], "q": [1.0, None, None, 2.0]})
    out = po.stream.with_windows(
        df,
        [po.window.lookahead_rewm("p", weight="q", halflife=1, horizon=5, complete="ok")],
        **CLOCK,
    )
    assert out["p_rewm_1_5"].to_list() == [None, None, None, None]
    assert out["ok"].to_list() == [True, True, True, False]


def test_unlisted_under_each_setting() -> None:
    df = pl.DataFrame(
        {
            "t": [0, 1, 2, 3, 4, 20],
            "side": ["buy", "cross", None, None, "sell", "buy"],
            "q": [1.0, 2.0, 4.0, None, 8.0, 1.0],
            "p": [10.0, 20.0, 30.0, 99.0, 50.0, 1.0],
        }
    )

    def run(unlisted: str, total: bool = True) -> pl.DataFrame:
        return po.stream.with_windows(
            df,
            [
                po.window.lookahead_rewm(
                    "p",
                    weight="q",
                    split=("side", ["buy", "sell"]),
                    unlisted=unlisted,  # type: ignore[arg-type]
                    total=total,
                    halflife=math.inf,
                    horizon=10,
                    name="v{split}",
                )
            ],
            **CLOCK,
        )

    # Row 0's window is rows 1-4; row 3, a quote with no quantity, never counts.
    total = run("total")
    assert total["v"][0] == pytest.approx((2 * 20 + 4 * 30 + 8 * 50) / 14)
    assert total["v_sell"][0] == 50.0 and total["v_buy"][0] is None
    ignore = run("ignore")
    assert ignore["v"][0] == 50.0
    # A cross and a null side both count, and are refused, naming the row;
    # the quote with a null side and no quantity is not.
    with pytest.raises(pl.exceptions.ComputeError, match='row 1 has "side" = "cross"'):
        run("error")
    df_ok = df.filter(pl.col("side").is_in(["buy", "sell"]) | pl.col("q").is_null())
    out = po.stream.with_windows(
        df_ok,
        [
            po.window.lookahead_rewm(
                "p", weight="q", split=("side", ["buy", "sell"]), halflife=1, horizon=10
            )
        ],
        **CLOCK,
    )
    assert out.height == df_ok.height
    no_total = run("ignore", total=False)
    assert "v" not in no_total.columns and {"v_buy", "v_sell"} <= set(no_total.columns)
    with pytest.raises(ValueError, match="total=False"):
        run("total", total=False)


def test_a_split_by_integers_matches_as_text() -> None:
    df = pl.DataFrame({"t": [0, 1, 2, 9], "k": [1, 2, 1, 1], "p": [1.0, 2.0, 3.0, 4.0]})
    out = po.stream.with_windows(
        df,
        [po.window.lookahead_rewm("p", split=("k", [1, 2]), halflife=math.inf, horizon=5)],
        **CLOCK,
    )
    assert out["p_rewm_inf_5_1"][0] == 3.0
    assert out["p_rewm_inf_5_2"][0] == 2.0


def events(on_clock_reset: str = "error", session_gap: Any = 1.0, **kw: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {"clock": "t", "max_dclock": 10.0, "on_clock_reset": on_clock_reset}
    if on_clock_reset == "reset_state":
        policy["min_backwards_jump"] = 50.0
    if session_gap is not None:
        policy.update(session="s", session_gap=session_gap)
    return policy | kw


def test_a_gap_past_the_cap_cuts_and_a_session_change_cuts() -> None:
    """Both end every window open across them, as the ``label_delay`` buffer
    releases every waiting row: partial, under ``partial``."""
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 30.0, 31.0, 32.0, 33.0, 60.0],
            "s": [0, 0, 0, 0, 0, 1, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    out = po.stream.with_windows(
        df,
        [
            po.window.lookahead_rewm(
                "x", halflife=math.inf, horizon=5, partial="keep", name="keep", complete="ok"
            ),
            po.window.lookahead_rewm("x", halflife=math.inf, horizon=5, name="null"),
            po.window.ewm("x", halflife=math.inf, horizon=5, name="back", complete="back_ok"),
        ],
        **events(),
    )
    # Row 0 sees rows 1 and 2, then the gap of 28 > 10 cuts it.
    assert out["keep"].to_list()[:3] == [2.5, 3.0, None]
    assert out["null"].to_list()[:3] == [None, None, None]
    assert out["ok"].to_list() == [False] * 8
    # Row 3 sees row 4; the session change at row 5 cuts it.
    assert out["keep"][3] == 5.0 and out["keep"][4] is None
    # The backward windows start over after each: 30 and 32 begin again.
    assert out["back"].to_list() == [1.0, 1.5, 2.0, 4.0, 4.5, 6.0, 6.5, 8.0]
    assert not any(out["back_ok"].to_list())


def test_a_reset_discards_whatever_partial_says() -> None:
    df = pl.DataFrame(
        {"t": [100.0, 101.0, 102.0, 0.0, 1.0, 9.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    policy = events("reset_state", session_gap=None)
    for partial in ["keep", "null", "drop"]:
        out = po.stream.with_windows(
            df,
            [po.window.lookahead_rewm("x", halflife=1, horizon=5, partial=partial, complete="ok")],  # type: ignore[arg-type]
            **policy,
        )
        assert out.height == 6, partial
        assert out["x_rewm_1_5"].to_list()[:3] == [None, None, None], partial
        assert out["ok"].to_list()[:3] == [False, False, False]
    # A step back no larger than the minimum is a late row, refused.
    late = df.with_columns(pl.when(pl.col("t") == 0.0).then(90.0).otherwise(pl.col("t")).alias("t"))
    with pytest.raises(
        pl.exceptions.ComputeError, match="goes backwards by 12 at row 3, no more than"
    ):
        po.stream.with_windows(late, [po.window.ewm("x", halflife=1)], **policy)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r'goes backwards by 102 at row 3 \(on_clock_reset = "error"',
    ):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1)], clock="t", max_dclock=10.0)


def test_session_gap_reset_discards() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "s": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]})
    out = po.stream.with_windows(
        df,
        [po.window.lookahead_rewm("x", halflife=1, horizon=5, partial="keep")],
        **events(session_gap="reset"),
    )
    assert out["x_rewm_1_5"].to_list() == [None, None, None, None]


def test_the_clock_is_read_across_groups() -> None:
    """The rows are one stream: a step back from one group's row to
    another's is refused like any other."""
    df = pl.DataFrame({"t": [5.0, 3.0], "g": ["a", "b"], "x": [1.0, 2.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="goes backwards by 2 at row 1"):
        po.stream.with_windows(
            df, [po.window.ewm("x", halflife=1)], clock="t", max_dclock=10.0, group="g"
        )


def test_a_silent_group_holds_the_output_for_at_most_the_cap() -> None:
    quiet = pl.DataFrame({"t": [0.0, 1.0], "g": ["quiet", "quiet"], "x": [1.0, 2.0]})
    busy = pl.DataFrame(
        {"t": [float(t) for t in range(2, 40)], "g": ["busy"] * 38, "x": [3.0] * 38}
    )
    df = pl.concat([quiet, busy])
    w = _native.Windows(
        '{"windows": [{"kind": "lookahead_rewm", "columns": ["x"], "halflife": [5], '
        '"horizon": [4], "partial": "keep"}], "clock": "t", "max_dclock": 10, "group": "g"}',
        df.clear(),
    )
    held = []
    for i in range(df.height):
        w.feed(df.slice(i, 1))
        held.append(w.held())
    # Every row waits behind the quiet group's row at 1 until the stream's
    # clock is more than 10 past it, at 12.
    assert held[:12] == list(range(1, 13)) and held[12] <= 5


def test_negative_weight_and_bad_clock_are_refused_by_row() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0], "w": [1.0, -1.0, 1.0]})
    with pytest.raises(pl.exceptions.ComputeError, match='row 1 has weight "w" = -1'):
        po.stream.with_windows(df, [po.window.ewm("x", weight="w", halflife=1)], **CLOCK)
    df = pl.DataFrame({"t": [0.0, None], "x": [1.0, 2.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="row 1 has a null or non-finite clock"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1)], **CLOCK)


def mixed(df: pl.DataFrame) -> list[po.window.Window]:
    return [
        po.window.ewm(["x", "y"], halflife=[2, 5], horizon=8, complete="b_ok"),
        po.window.ewm("x", halflife=3, partial="drop", horizon=4, name="dropping"),
        po.window.lookahead_rewm(
            "x", weight="w", halflife=2, horizon=6, partial="keep", complete="f_ok"
        ),
        po.window.lookahead_rewm(
            "y",
            split=("g", ["0", "1"]),
            unlisted="total",
            halflife=1e-3,
            horizon=3,
            name="tiny{split}",
        ),
    ]


def grouped_stream(n: int, seed: int) -> pl.DataFrame:
    df = ticks(n, seed, groups=3)
    rng = np.random.default_rng(seed + 100)
    t = df["t"].to_numpy().astype(float)
    # A gap past the cap now and then, and sessions.
    t = t + np.cumsum(np.where(rng.random(n) < 0.02, 40.0, 0.0))
    return df.with_columns(
        t=pl.Series(t),
        y=pl.lit(2.5),
        s=pl.Series(np.cumsum(rng.random(n) < 0.02)),
    )


POLICY = {"clock": "t", "max_dclock": 10.0, "session": "s", "session_gap": 2.0, "group": "g"}


@pytest.mark.parametrize("rows", [1, 7, 64, 1000])
def test_chunking_changes_nothing(rows: int) -> None:
    df = grouped_stream(500, 3)
    one = po.stream.with_windows(df, mixed(df), chunk_rows=10_000, **POLICY)  # type: ignore[arg-type]
    got = po.stream.with_windows(df, mixed(df), chunk_rows=rows, **POLICY)  # type: ignore[arg-type]
    assert got.equals(one)
    # What the stream covers: drops, both ends, a constant, an underflow.
    assert 0 < one.height < df.height
    assert (one["y_ewm_2_8"].drop_nulls() - 2.5).abs().max() <= 1e-14
    assert one["tiny"].drop_nulls().len() > 0


def test_a_save_and_load_at_every_row_is_one_run(tmp_path: Any) -> None:
    # Seed 28: two gaps past the cap and four session changes in 60 rows.
    df = grouped_stream(60, 28)
    t = df["t"].to_list()
    assert sum(b - a > 10 for a, b in zip(t, t[1:], strict=False)) == 2 and df["s"].n_unique() == 5
    one = po.stream.with_windows(df, mixed(df), **POLICY)  # type: ignore[arg-type]
    state = tmp_path / "w.bin"
    for k in range(df.height + 1):
        a = po.stream.with_windows(df.head(k), mixed(df), save_state=state, **POLICY)  # type: ignore[arg-type]
        b = po.stream.with_windows(df.slice(k), mixed(df), load_state=state, **POLICY)  # type: ignore[arg-type]
        assert pl.concat([a, b]).equals(one), k


def test_a_state_resumes_only_its_own_call(tmp_path: Any) -> None:
    df = grouped_stream(40, 1)
    state = tmp_path / "w.bin"
    po.stream.with_windows(df, mixed(df), save_state=state, **POLICY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="saved by another call"):
        po.stream.with_windows(df, mixed(df)[:1], load_state=state, **POLICY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="saved by another call"):
        po.stream.with_windows(df, mixed(df), load_state=state, **(POLICY | {"max_dclock": 11.0}))  # type: ignore[arg-type]


def test_a_slice_and_a_filter_are_honoured() -> None:
    df = grouped_stream(300, 2)
    full = po.stream.with_windows(df, mixed(df), **POLICY)  # type: ignore[arg-type]
    lf = po.stream.with_windows(df.lazy(), mixed(df), chunk_rows=13, **POLICY)  # type: ignore[arg-type]
    for n in [0, 1, 17, 250]:
        assert lf.head(n).collect().equals(full.head(n)), n
    assert lf.filter(pl.col("x") > 0).collect().equals(full.filter(pl.col("x") > 0))
    assert lf.select("tiny_1").collect().equals(full.select("tiny_1"))


def test_like_takes_the_spec_clock_and_its_rule() -> None:
    df = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0, 20.0],
            "f": [1.0, None, 1.0, 1.0, 1.0],
            "p": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    spec = po.spec.ewridge(
        "m", targets=["p"], features=["f"], halflife=10.0, clock="t", max_dclock=100.0
    )
    out = po.stream.with_windows(
        df, [po.window.lookahead_rewm("p", halflife=math.inf, horizon=5)], like=spec
    )
    # Row 1's features are null: the model never learns its target, so it is
    # null; the row still counts in row 0's window.
    assert out["p_rewm_inf_5"].to_list() == [3.0, None, 4.0, None, None]
    with pytest.raises(TypeError, match="leave out clock"):
        po.stream.with_windows(df, [po.window.ewm("p", halflife=1)], like=spec, clock="t")


def test_names_and_collisions() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0], "y": [1.0, 2.0]})
    out = po.stream.with_windows(
        df,
        [
            po.window.ewm(["x", "y"], halflife=[1, "inf"], horizon=[5, 10]),
            po.window.ewm("x", halflife=0.5),
        ],
        **CLOCK,
    )
    assert out.columns[3:] == [
        "x_ewm_1_5", "x_ewm_1_10", "x_ewm_inf_5", "x_ewm_inf_10",
        "y_ewm_1_5", "y_ewm_1_10", "y_ewm_inf_5", "y_ewm_inf_10",
        "x_ewm_0.5",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="is also"):
        po.stream.with_windows(
            df, [po.window.ewm("x", halflife=1), po.window.ewm("x", halflife=1)], **CLOCK
        )
    with pytest.raises(ValueError, match="already a column"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1, name="y")], **CLOCK)
    with pytest.raises(ValueError, match=r"is also"):
        po.stream.with_windows(df, [po.window.ewm(["x", "y"], halflife=1, name="same")], **CLOCK)
    with pytest.raises(ValueError, match=r"\{bogus\}"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1, name="{bogus}")], **CLOCK)
    with pytest.raises(ValueError, match="has no horizon"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1, name="a{horizon}")], **CLOCK)


def test_a_description_that_cannot_run_is_refused_while_the_plan_is_built() -> None:
    df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0], "s": ["a", "b"]})
    cases: list[tuple[Any, str]] = [
        ([po.window.ewm("x", halflife=0)], "halflife must be above 0"),
        ([po.window.lookahead_rewm("x", halflife=1, horizon=math.inf)], "horizon must be finite"),
        ([po.window.ewm("nope", halflife=1)], "no value column"),
        ([po.window.ewm("s", halflife=1)], "must be numbers"),
        ([po.window.ewm("x", halflife="1m")], "halflife is a duration"),
        ([], "no windows"),
    ]
    for windows, match in cases:
        with pytest.raises(ValueError, match=match):
            po.stream.with_windows(df.lazy(), windows, **CLOCK)
    with pytest.raises(ValueError, match="max_dclock is required"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1)], clock="t")
    with pytest.raises(ValueError, match="session_gap is required"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=1)], session="s", **CLOCK)
    with pytest.raises(TypeError, match="descriptions"):
        po.stream.with_windows(df, ["x"], **CLOCK)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="partial"):
        po.window.ewm("x", halflife=1, partial="maybe")  # type: ignore[arg-type]


def test_a_temporal_clock_takes_durations() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}).with_columns(
        t=pl.datetime_range(
            pl.datetime(2026, 9, 30), pl.datetime(2026, 9, 30, 0, 0, 3), "1s", eager=True
        )
    )
    out = po.stream.with_windows(
        df, [po.window.lookahead_rewm("x", halflife="1s", horizon="2s")], clock="t", max_dclock="1m"
    )
    # Row 0: rows 1 (weight 1) only; the row 2 s later closes it.
    assert out["x_rewm_1s_2s"].to_list() == [2.0, 3.0, None, None]
    with pytest.raises(ValueError, match="plain number"):
        po.stream.with_windows(df, [po.window.ewm("x", halflife=5)], clock="t", max_dclock="1m")


def test_queues_follow_the_rows_that_count() -> None:
    """At a fixed trade rate the queues hold the trades a horizon covers,
    however many quotes come between: 1, 10 and 100 per trade."""
    most = []
    for quotes in [1, 10, 100]:
        df = trades_and_quotes(20_000, 3, quotes_per_trade=quotes)
        # The same clock span per trade: scale the clock by the density.
        df = df.with_columns(t=(pl.col("t") * 2 // (quotes + 1)))
        w = _native.Windows(
            '{"windows": [{"kind": "lookahead_rewm", "columns": ["price"], "weight": "quantity", '
            '"halflife": [10], "horizon": [60]}], "clock": "t", "max_dclock": 1000}',
            df.clear(),
        )
        peak = 0
        for chunk in df.iter_slices(1000):
            w.feed(chunk)
            peak = max(peak, w.queued())
        most.append(peak)
    assert max(most) <= 2 * min(most), most


def test_rows_are_held_once_however_many_windows() -> None:
    """Windows added cost their own sums and nothing more: the held rows are
    one horizon of input for one window or five."""
    df = trades_and_quotes(5000, 4)

    def peak(n: int) -> tuple[int, int]:
        cfg = {
            "windows": [
                {"kind": "lookahead_rewm", "columns": ["price"], "weight": "quantity",
                 "halflife": [h], "horizon": [60]}
                for h in range(1, n + 1)
            ],
            "clock": "t",
            "max_dclock": 1000,
        }  # fmt: skip
        import json

        w = _native.Windows(json.dumps(cfg), df.clear())
        held = queued = 0
        for chunk in df.iter_slices(500):
            w.feed(chunk)
            held, queued = max(held, w.held()), max(queued, w.queued())
        return held, queued

    (h1, q1), (h5, q5) = peak(1), peak(5)
    assert h1 == h5
    assert q5 == 5 * q1


def test_window_columns_feed_a_model_in_one_query() -> None:
    """The README's use: trailing windows as a model's inputs in the same
    query, streaming, give the fit a frame of those columns gives."""
    df = grouped_stream(2000, 11).with_columns(target=pl.col("x").fill_null(0.0) * 2.0)
    factors = [po.window.ewm("x", halflife=[3, 20], horizon=40)]
    spec = po.spec.ewridge(
        "m", targets=["target"], features=["x_ewm_3_40", "x_ewm_20_40"], halflife=200.0, **POLICY
    )
    lf = po.stream.with_windows(df.lazy(), factors, chunk_rows=97, **POLICY)  # type: ignore[arg-type]
    one_query = lf.online.fit_predict([spec], chunk_rows=61).collect()
    columns = po.stream.with_windows(df, factors, **POLICY)  # type: ignore[arg-type]
    two_steps = po.ModelBank([spec]).fit_predict(columns)
    # `coef`, and `support_coef` beside it, are emitted on each group's last
    # row of every chunk, by design, so they differ between chunkings; every
    # other field is the same.
    a, b = (f.unnest("m").drop("coef", "support_coef") for f in (one_query, two_steps))
    assert a.equals(b)
    assert a["pred_target"].drop_nulls().len() > 1500


def test_the_online_namespace_is_the_same_call() -> None:
    """``lf.online.with_windows`` and ``df.online.with_windows`` are
    :func:`po.stream.with_windows` as methods, for a chain."""
    df = grouped_stream(300, 4)
    want = po.stream.with_windows(df, mixed(df), **POLICY)  # type: ignore[arg-type]
    assert df.online.with_windows(mixed(df), **POLICY).equals(want)  # type: ignore[arg-type]
    lazy = df.lazy().online.with_windows(mixed(df), chunk_rows=17, **POLICY)  # type: ignore[arg-type]
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(want)


def test_the_methods_take_exactly_the_functions_arguments() -> None:
    """The namespace methods restate the function's signature, for help() and
    the reference; this keeps the two from drifting apart."""
    import inspect

    from polars_online._frame import DataFrameOnlineNamespace, LazyFrameOnlineNamespace

    want = list(inspect.signature(po.stream.with_windows).parameters.values())[1:]
    for cls in (LazyFrameOnlineNamespace, DataFrameOnlineNamespace):
        got = list(inspect.signature(cls.with_windows).parameters.values())[1:]
        assert [(p.name, p.kind, p.default) for p in got] == [
            (p.name, p.kind, p.default) for p in want
        ], cls.__name__


def test_an_order_hazard_beneath_the_windows_is_reported() -> None:
    """A bank after the windows sees only their source; the windows read in
    row order too, so they check the plan they are given, as a bank does."""
    left = pl.LazyFrame({"k": [1, 2, 3], "t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
    right = pl.LazyFrame({"k": [1, 2, 3], "z": [0.0, 1.0, 0.0]})
    wins = [po.window.ewm("x", halflife=2)]
    with pytest.warns(po.OrderNotGuaranteedWarning, match="with_windows: .* taken over rows"):
        left.join(right, on="k").online.with_windows(wins, clock="t", max_dclock=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        left.join(right, on="k", maintain_order="left").online.with_windows(
            wins, clock="t", max_dclock=10.0
        )


def test_a_spent_stream_beneath_the_windows_is_reported() -> None:
    from polars.io.plugins import register_io_source

    spent = False

    def once(*_: Any) -> Any:
        nonlocal spent
        if not spent:
            spent = True
            yield pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})

    lf = register_io_source(once, schema={"t": pl.Float64, "x": pl.Float64})
    plan = lf.online.with_windows([po.window.ewm("x", halflife=2)], clock="t", max_dclock=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert plan.collect().height == 2
    with pytest.warns(po.ConsumedSourceWarning, match="with_windows: the plan yielded no rows"):
        assert plan.collect().height == 0
