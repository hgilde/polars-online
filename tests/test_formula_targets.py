"""Window expressions as model targets (docs/PLAN.md task 104).

A target may be a formula of the row's future -- ``po.rewm_mean("mid", ...)
- pl.col("mid")`` in a spec's ``targets`` -- and the bank's own window core
resolves it: each row is scored where it sits and learned from once its
window has closed and its ``embargo`` has passed. The claim is parity with
the column form: the same expression written by ``with_windows(...,
like=spec)`` and fed back as a plain column target under the same embargo
gives the same prediction on every row, bit for bit, through every clock
event, on skipped rows and accepted ones, whatever the chunking. Around it:
what ``fit_predict`` refuses and ``fit`` takes, ``predict``, the clocks of
task 152, a state saved mid-window, the plan's projection, the TOML form.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online
from polars_online._polars_online import validate_spec
from test_windows import trades_and_quotes

W = 10.0
H = 5.0


def fwd(**kw: Any) -> pl.Expr:
    """The formula every test learns: the next window's mean of ``mid``,
    less the row's own."""
    return (po.rewm_mean("mid", half_life=H, window_size=W, **kw) - pl.col("mid")).alias("fwd")


def spec(target: Any, *, embargo: float | None = W + 2.5, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = dict(
        features=["x"],
        clock="t",
        gap_cap=100.0,
        half_life=50.0,
        min_weight=3.0,
        max_rows_between_solves=1,
        emit_clocks=True,
        embargo=embargo,
    )
    d.update(kw)
    return po.spec.ewridge("m", targets=[target], **d)


def stream(
    n: int = 500, seed: int = 0, *, groups: int = 1, gap_at: int | None = None
) -> pl.DataFrame:
    """A random-step clock, a feature null now and then, a drifting ``mid``,
    and optionally several groups interleaved on the one clock and a gap
    past the cap."""
    rng = np.random.default_rng(seed)
    steps = 0.25 + rng.random(n) * 2.0
    if gap_at is not None:
        steps[gap_at] = 250.0
    t = np.cumsum(steps)
    x = rng.standard_normal(n)
    x[rng.random(n) < 0.08] = np.nan
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    df = pl.DataFrame({"t": t, "x": x, "mid": mid}).with_columns(pl.col("x").fill_nan(None))
    if groups > 1:
        df = df.with_columns(g=pl.Series(rng.choice([f"g{i}" for i in range(groups)], n)))
    return df


def native(df: pl.DataFrame, s: dict[str, Any], chunk_rows: int | None = None) -> pl.DataFrame:
    bank = po.ModelBank([s])
    if chunk_rows is None:
        return bank.fit_predict(df)
    return pl.concat(bank.fit_predict_batches(df, chunk_rows=chunk_rows))


def column_form(df: pl.DataFrame, s: dict[str, Any], expr: pl.Expr) -> pl.DataFrame:
    """The reference: the formula as a column (``like=`` the spec, so a row
    the spec skips gets a null), then the plain column as the target."""
    name = expr.meta.output_name()
    with_col = po.stream.with_windows(df, expr, like=s)
    plain = dict(s)
    plain["targets"] = [name]
    return po.ModelBank([plain]).fit_predict(with_col)


def field(out: pl.DataFrame, name: str) -> list[Any]:
    return out["m"].struct.field(name).to_list()


# --------------------------------------------------------------------------
# Parity with the column form


def test_a_formula_target_is_the_column_form_fed_back() -> None:
    df = stream(600, 1, gap_at=300)
    want = column_form(df, spec(fwd()), fwd())
    assert sum(p is not None for p in field(want, "pred_fwd")) > 300
    for chunk_rows in [None, 1, 97]:
        got = native(df, spec(fwd()), chunk_rows)
        assert field(got, "pred_fwd") == field(want, "pred_fwd"), chunk_rows
        assert field(got, "learned_clock") == field(want, "learned_clock"), chunk_rows
        assert all(r is None for r in field(got, "resid_fwd")), "not known at the row"
        assert field(got, "weight_sum") == field(want, "weight_sum")


def test_three_vwaps_on_interleaved_trades_and_quotes() -> None:
    """The user's case: three forward VWAPs -- all trades, buys, sells -- as
    targets of one spec, each a ratio of two decayed sums, the quotes' null
    trades skipped by the operators and learned from by the model."""
    df = trades_and_quotes(900, 5).with_columns(x=pl.col("mid") - pl.col("mid").shift(1))
    notional = pl.col("price") * pl.col("quantity")
    w = dict(half_life=H, window_size=W)
    buys = pl.when(pl.col("side") == "buy")
    sells = pl.when(pl.col("side") == "sell")
    targets = [
        (po.rewm_sum(notional, **w) / po.rewm_sum("quantity", **w) - pl.col("mid")).alias("all"),
        (
            po.rewm_sum(buys.then(notional), **w) / po.rewm_sum(buys.then("quantity"), **w)
            - pl.col("mid")
        ).alias("buy"),
        (
            po.rewm_sum(sells.then(notional), **w) / po.rewm_sum(sells.then("quantity"), **w)
            - pl.col("mid")
        ).alias("sell"),
    ]
    s = po.spec.ewridge(
        "m",
        targets=targets,
        features=["x"],
        clock="t",
        gap_cap=100.0,
        half_life=50.0,
        min_weight=3.0,
        max_rows_between_solves=1,
        embargo=W + 1,
        emit_clocks=True,
    )
    with_cols = po.stream.with_windows(df, *targets, like=s)
    plain = dict(s)
    plain["targets"] = ["all", "buy", "sell"]
    want = po.ModelBank([plain]).fit_predict(with_cols)
    got = native(df, s, chunk_rows=50)
    for name in ["all", "buy", "sell"]:
        assert field(got, f"pred_{name}") == field(want, f"pred_{name}"), name
        assert sum(p is not None for p in field(got, f"pred_{name}")) > 400
    assert field(got, "learned_clock") == field(want, "learned_clock")


def event_streams() -> dict[str, tuple[pl.DataFrame, dict[str, Any]]]:
    """A stream per clock event the review names, each with the event on a
    skipped row (a null feature) and on an accepted one, as (frame, the
    spec's clock keywords)."""
    base = stream(240, 3)
    out: dict[str, tuple[pl.DataFrame, dict[str, Any]]] = {}

    def skip(df: pl.DataFrame, rows: list[int]) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.int_range(pl.len()).is_in(rows)).then(None).otherwise(pl.col("x")).alias("x")
        )

    def jump(df: pl.DataFrame, at: int, by: float) -> pl.DataFrame:
        t = df["t"].to_numpy().copy()
        t[at:] += by
        return df.with_columns(t=pl.Series(t))

    # A gap past the cap on an accepted row (60) and on a skipped one (150).
    out["gap"] = (skip(jump(jump(base, 60, 150.0), 150, 150.0), [150, 151]), {})
    # A session change with a finite session_gap, on an accepted row and a
    # skipped one; the clock keeps going.
    sess = base.with_columns(
        s=pl.when(pl.int_range(pl.len()) < 80)
        .then(pl.lit("a"))
        .when(pl.int_range(pl.len()) < 170)
        .then(pl.lit("b"))
        .otherwise(pl.lit("c"))
    )
    out["session_gap"] = (skip(sess, [170]), {"session": "s", "session_gap": 3.0})
    out["session_reset"] = (skip(sess, [170]), {"session": "s", "session_gap": "reset"})
    # A step back the policy restarts on, on an accepted row and a skipped
    # one.
    back = jump(jump(base, 100, -30.0), 180, -30.0)
    out["restart"] = (skip(back, [180]), {"restart_after_step_back": 0.0})
    # Several groups whose clocks restart together at a gap on the stream.
    grouped = stream(300, 4, groups=3, gap_at=150)
    out["groups"] = (grouped, {"group": "g"})
    # Rows at one stamp, a row exactly one window later, and a run of
    # skipped rows totalling more than the cap.
    ints = pl.DataFrame(
        {
            "t": [
                float(v)
                for v in [0, 1, 1, 2, 3, 3, 3, 5, 8, 11, 13, 13, 15, 18, 21, 25, 28, 31, 35, 40]
                + list(range(41, 120, 2))
            ],
            "mid": np.cumsum(np.random.default_rng(9).standard_normal(60)).tolist(),
            "x": np.random.default_rng(10).standard_normal(60).tolist(),
        }
    )
    out["stamps"] = (skip(ints, [2, 5, 6, 17, 18]), {})
    run = jump(skip(base, list(range(120, 150))), 150, 0.0)
    t = run["t"].to_numpy().copy()
    t[120:150] = np.linspace(t[119] + 5, t[119] + 180, 30)
    t[150:] += 180
    out["skipped_run"] = (run.with_columns(t=pl.Series(t)), {})
    return out


@pytest.mark.parametrize("partial", [None, "keep"])
@pytest.mark.parametrize("event", list(event_streams()))
def test_parity_through_every_clock_event(event: str, partial: str | None) -> None:
    """Row by row, the native path and the column form learn the same rows
    at the same time and predict the same numbers, through a gap past the
    cap, a session change (finite gap and reset), a step back the policy
    restarts on, groups restarting together, rows at one stamp, a row
    exactly one window later, a run of skipped rows longer than the cap,
    and the end of the input inside a window -- the embargo covering the
    window by more than any step, so the only question is the events."""
    df, clock = event_streams()[event]
    expr = fwd(partial=partial) if partial else fwd()
    s = spec(expr, embargo=W + 2.5, gap_cap=100.0, **clock)
    want = column_form(df, s, expr)
    # One row a chunk is the leg that caught resolutions made at skipped
    # rows being lost (the plan's task 104, *Built*).
    for chunk_rows in [None, 7, 1]:
        got = native(df, s, chunk_rows)
        assert field(got, "pred_fwd") == field(want, "pred_fwd"), (event, chunk_rows)
        assert field(got, "learned_clock") == field(want, "learned_clock"), (event, chunk_rows)
    learned = [c for c in field(want, "learned_clock") if c is not None]
    assert learned, event
    # The end of the input is inside a window: its rows are never learned.
    assert max(learned) < df["t"].max() - W + 1e-9, event


def test_drop_means_null_for_a_target() -> None:
    """A row of a model is scored and cannot leave the output, so a window
    cut short under ``partial="drop"`` is not learned from, as under
    ``"null"``; the two give the same predictions."""
    df = stream(300, 17, gap_at=150)
    null = native(df, spec(fwd(partial="null")))
    drop = native(df, spec(fwd(partial="drop")))
    assert field(drop, "pred_fwd") == field(null, "pred_fwd")
    assert field(drop, "learned_clock") == field(null, "learned_clock")
    assert drop.height == df.height
    # Under "keep" the cut rows are learned from what their windows saw, so
    # the predictions after the gap differ (the clock of the newest row
    # released to the models does not: a null target is released too).
    keep = native(df, spec(fwd(partial="keep")))
    assert field(keep, "pred_fwd") != field(null, "pred_fwd")


def test_a_row_exactly_one_window_later_waits_for_the_window_to_close() -> None:
    """With the embargo equal to the window under ``closed="right"``, a row
    exactly one window later is a member of the window, so the native path
    learns the row only at the next distinct stamp, where the column form
    -- the target already in the column -- learns it as the embargo runs
    out. The one place the two part, and why resolutions are applied in
    row order (the review under task 104)."""
    df = pl.DataFrame(
        {
            "t": [0.0, 2.0, 4.0, 10.0, 12.0, 14.0, 20.0, 22.0, 24.0, 30.0, 32.0],
            "x": [0.5, -0.2, 0.1, 0.3, -0.4, 0.2, 0.1, -0.3, 0.4, 0.0, 0.2],
            "mid": [1.0, 1.5, 1.2, 1.8, 1.6, 1.9, 2.1, 2.0, 2.4, 2.2, 2.5],
        }
    )
    s = spec(fwd(), embargo=W, min_weight=0.0)
    got = native(df, s)
    want = column_form(df, s, fwd())
    # Row 0's window is (0, 10]: t = 10 is in it. The column form learns row
    # 0 at t = 10 (its wait ran out); the native path at t = 12, when the
    # window closed.
    assert field(want, "learned_clock")[3] == 0.0
    assert field(got, "learned_clock")[3] is None
    assert field(got, "learned_clock")[4] == 0.0
    # From then on every row is learned where its window closes: row 1's
    # (2, 12] at t = 14, row 2's (4, 14] at t = 20, and so on.
    assert field(got, "learned_clock")[5:] == [2.0, 4.0, 10.0, 12.0, 14.0, 20.0]
    # No embargo at all, under `fit`: the same rows learned at the same
    # time, since the window closing is what releases them. Held to the
    # `embargo = W` run's count of rows learned and to its Gram, bit for bit;
    # `rows_seen() == height` alone was true of any fit (review 2026-10-05,
    # TA8). A longer embargo, W + 2.5, learns six rows here, not seven.
    fitted = {}
    for embargo in (None, W):
        bank = po.ModelBank([spec(fwd(), embargo=embargo, min_weight=0.0)])
        bank.fit(df)
        fitted[embargo] = bank
    plain, embargoed = fitted[None], fitted[W]
    assert plain.summary()["rows_learned"].to_list() == [7]
    assert embargoed.summary()["rows_learned"].to_list() == [7]
    [a], [b] = plain.gram("m"), embargoed.gram("m")
    assert a["weight_sum"] == b["weight_sum"] > 0.0
    for key in ("means", "comoments", "cross_centred", "target_weights", "target_means"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


# --------------------------------------------------------------------------
# What fit_predict refuses, fit takes, predict ignores


def test_fit_predict_refuses_an_embargo_below_the_window_and_fit_takes_it() -> None:
    df = stream(300, 2)
    for short in [None, 4.0]:
        bank = po.ModelBank([spec(fwd(), embargo=short)])
        with pytest.raises(ValueError, match="fit_predict needs an embargo of at least 10"):
            bank.fit_predict(df)
        with pytest.raises(ValueError, match="takes any embargo"):
            list(bank.fit_predict_batches(df, chunk_rows=10))
        assert bank.rows_seen() == 0
    later = df.slice(250, 50)
    covered = po.ModelBank([spec(fwd(), embargo=W)])
    covered.fit(df.slice(0, 250), chunk_rows=40)
    want = covered.predict(later)
    for short in [None, 4.0]:
        bank = po.ModelBank([spec(fwd(), embargo=short)])
        bank.fit(df.slice(0, 250), chunk_rows=40)
        got = bank.predict(later)
        assert field(got, "pred_fwd") == field(want, "pred_fwd"), short
        # The flag is the run's: fit_predict refuses the same bank after.
        with pytest.raises(ValueError, match="fit_predict needs an embargo"):
            bank.fit_predict(later)
    # The plan form refuses while the plan is built, before a row moves.
    with pytest.raises(ValueError, match="fit_predict needs an embargo"):
        df.lazy().online.fit_predict([spec(fwd(), embargo=None)])


def test_groups_keep_their_own_clocks_and_cores() -> None:
    """Each group has its own window core, as it has its own stream: groups
    interleaved on clocks of their own run, and a group's predictions are
    those of its rows run alone."""
    df = stream(400, 21, groups=2)
    own = df["t"].to_numpy().copy()
    steps = np.diff(own, prepend=0.0)
    clocks = {"g0": 0.0, "g1": 0.5}
    for i, g in enumerate(df["g"].to_list()):
        clocks[g] += steps[i]
        own[i] = clocks[g]
    df = df.with_columns(t=pl.Series(own))
    s = spec(fwd(), group="g")
    got = native(df, s, chunk_rows=50)
    for g in ["g0", "g1"]:
        alone = native(df.filter(pl.col("g") == g), s, chunk_rows=30)
        picked = got.filter(pl.col("g") == g)
        assert field(picked, "pred_fwd") == field(alone, "pred_fwd"), g
        assert sum(p is not None for p in field(alone, "pred_fwd")) > 100


def test_predict_scores_without_the_target() -> None:
    df = stream(200, 6)
    bank = po.ModelBank([spec(fwd())])
    bank.fit_predict(df.slice(0, 150))
    scored = bank.predict(df.slice(150, 50).drop("mid"))
    assert sum(p is not None for p in field(scored, "pred_fwd")) > 40
    assert bank.rows_seen() == 150
    # The summary counts a row handed a formula target as learned from, as
    # it counts a plain target held under an embargo.
    assert bank.summary()["rows_learned"][0] > 100


# --------------------------------------------------------------------------
# Refusals, by name


def test_what_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no operator looking ahead"):
        spec((po.ewm_mean("mid", half_life=H) - pl.col("mid")).alias("back"))


def test_a_target_known_at_its_own_row_is_pointed_to_the_call_that_makes_its_column() -> None:
    """A formula that only looks back is a column `with_windows` makes; one of
    the row alone is Polars' own `with_columns`, which `with_windows` refuses.
    The refusal used to send both to `with_windows` (docs/README-ITERATIONS.md,
    E11), so following it failed for the second."""
    with pytest.raises(ValueError, match="po.stream.with_windows"):
        spec((po.ewm_mean("mid", half_life=H) - pl.col("mid")).alias("back"))
    with pytest.raises(ValueError, match="with_columns") as plain:
        spec((pl.col("price") - pl.col("mid")).alias("rel"))
    assert "with_windows" not in str(plain.value)
    # And the advice works: with_windows refuses the plain formula, as the
    # message now says, and makes the column from the one that looks back.
    frame = pl.DataFrame({"t": [0.0, 1.0, 2.0], "mid": [1.0, 2.0, 3.0], "price": [1.5, 2.5, 3.5]})
    with pytest.raises(ValueError, match="with_columns"):
        po.stream.with_windows(frame, rel=pl.col("price") - pl.col("mid"), clock="t", gap_cap=10.0)
    made = po.stream.with_windows(
        frame, back=po.ewm_mean("mid", half_life=H) - pl.col("mid"), clock="t", gap_cap=10.0
    )
    assert made.columns == ["t", "mid", "price", "back"]
    with pytest.raises(ValueError, match="needs a name"):
        spec(po.rewm_mean("mid", half_life=H, window_size=W))
    with pytest.raises(ValueError, match="group_close does not work with a formula target"):
        spec(fwd(), group="g", group_close="monotone", embargo=None)
    with pytest.raises(ValueError, match="a formula target .* does not apply"):
        po.spec.ftrl("f", targets=[fwd()], features=["x"])  # a probability to a 0/1 target
    with pytest.raises(TypeError, match="window expressions looking ahead"):
        spec(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="temporal column can only be a clock"):
        po.ModelBank([spec(fwd())]).fit_predict(
            stream(20, 1).with_columns(mid=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s"))
        )


# --------------------------------------------------------------------------
# The clocks of task 152, resuming, the plan, the TOML


def test_the_embargo_and_the_window_are_visible_through_the_clocks() -> None:
    df = stream(400, 8)
    out = native(df, spec(fwd(), embargo=W + 2.5))
    scored, learned = field(out, "scored_clock"), field(out, "learned_clock")
    assert all(s - w >= W + 2.5 for s, w in zip(scored, learned, strict=True) if w is not None)
    # And the row learned is always one whose window had closed: more than
    # W before the scored row.
    assert all(s - w > W for s, w in zip(scored, learned, strict=True) if w is not None)


def test_a_state_saved_mid_window_resumes_as_one_run(tmp_path: Any) -> None:
    df = stream(400, 12, gap_at=200)
    whole = native(df, spec(fwd()))
    bank = po.ModelBank([spec(fwd())])
    first = bank.fit_predict(df.slice(0, 203))
    bank.save(tmp_path / "bank.state")
    resumed = po.ModelBank.load(tmp_path / "bank.state")
    rest = resumed.fit_predict(df.slice(203, 197))
    out = pl.concat([first, rest])
    assert field(out, "pred_fwd") == field(whole, "pred_fwd")
    assert field(out, "learned_clock") == field(whole, "learned_clock")
    # The state is self-describing: the spec carries the formula.
    assert resumed.specs[0]["targets"][0]["name"] == "fwd"
    assert resumed.specs[0]["targets"][0]["formula"][0] == "-"


def test_a_state_saved_inside_a_tied_stamp_resumes_as_one_run(tmp_path: Any) -> None:
    """Task 159 (B1): the same-stamp refusal task 158's E13 put on every
    resumed window run fired inside the bank's formula-target resolver, so a
    bank saved between two rows at one stamp refused to resume, where the
    column form resumed. The bank's own clock has refused any real step back
    before a core is fed, so a chunk boundary inside a tied stamp is the
    stream going on. Three rows share each stamp here, and the cut falls
    between two of them; on the Date clock, four rows share each day."""
    rng = np.random.default_rng(5)
    n = 300
    df = pl.DataFrame(
        {
            "t": np.repeat(np.arange(n // 3, dtype=float), 3),
            "x": rng.standard_normal(n),
            "mid": 100 + np.cumsum(rng.standard_normal(n) * 0.1),
        }
    )
    cut = 100
    assert df["t"][cut - 1] == df["t"][cut] == df["t"][cut + 1], "the cut is inside a stamp"
    whole = native(df, spec(fwd()))
    bank = po.ModelBank([spec(fwd())])
    first = bank.fit_predict(df.slice(0, cut))
    bank.save(tmp_path / "tied.state")
    rest = po.ModelBank.load(tmp_path / "tied.state").fit_predict(df.slice(cut))
    out = pl.concat([first, rest])
    assert field(out, "pred_fwd") == field(whole, "pred_fwd")
    assert field(out, "learned_clock") == field(whole, "learned_clock")
    # A Date clock, grouped: the cut falls inside a day.
    n = 240
    dd = pl.DataFrame(
        {
            "t": pl.Series(np.repeat(np.arange(n // 4), 4)).cast(pl.Int32).cast(pl.Date),
            "x": rng.standard_normal(n),
            "mid": 100 + np.cumsum(rng.standard_normal(n) * 0.1),
            "g": rng.choice(["a", "b"], n),
        }
    )
    target = (po.rewm_mean("mid", half_life="5d", window_size="10d") - pl.col("mid")).alias("fwd")
    s = spec(target, gap_cap="100d", half_life="50d", embargo="13d", group="g")
    whole = native(dd, s)
    bank = po.ModelBank([s])
    first = bank.fit_predict(dd.slice(0, 50))
    bank.save(tmp_path / "days.state")
    rest = po.ModelBank.load(tmp_path / "days.state").fit_predict(dd.slice(50))
    assert field(pl.concat([first, rest]), "pred_fwd") == field(whole, "pred_fwd")


def test_the_plan_keeps_the_formulas_columns() -> None:
    """A projection after the bank that drops ``mid`` still reads it for the
    target; the bank's output is the same."""
    df = stream(300, 13)
    want = native(df, spec(fwd()))
    got = df.lazy().online.fit_predict([spec(fwd())], chunk_rows=50).select("t", "m").collect()
    assert field(got, "pred_fwd") == field(want, "pred_fwd")
    assert got.columns == ["t", "m"]


def test_the_toml_form_runs_the_same(online_cli: Any, tmp_path: Any) -> None:
    df = stream(300, 14)
    s = spec(fwd())
    assert s["targets"][0]["formula"][0] == "-"
    df.write_parquet(tmp_path / "in.parquet")
    run_online(
        online_cli, tmp_path, [s], input=tmp_path / "in.parquet", output=tmp_path / "out.parquet"
    )
    got = pl.read_parquet(tmp_path / "out.parquet")
    want = native(df, s)
    assert field(got, "pred_fwd") == field(want, "pred_fwd")
    short = spec(fwd(), embargo=2.0)
    res = run_online(
        online_cli,
        tmp_path,
        [short],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out2.parquet",
        check=False,
    )
    assert res.returncode != 0 and "fit_predict needs an embargo" in res.stderr


def test_the_dry_run_refuses_a_short_embargo_as_the_run_does(
    online_cli: Any, tmp_path: Any
) -> None:
    """Task 154: the dry run built the bank but never asked it, so it passed
    a config the run then refused at its first chunk. A run that keeps no
    prediction takes any embargo, in the dry run as in the run."""
    stream(50, 16).write_parquet(tmp_path / "in.parquet")
    short = spec(fwd(), embargo=2.0)
    res = run_online(
        online_cli,
        tmp_path,
        [short],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        args=["--dry-run"],
        check=False,
    )
    assert res.returncode != 0 and "fit_predict needs an embargo" in res.stderr
    run_online(
        online_cli,
        tmp_path,
        [short],
        input=tmp_path / "in.parquet",
        save_state=tmp_path / "bank.state",
        args=["--dry-run", "--no-output"],
    )


def test_a_formula_target_beside_a_plain_one() -> None:
    """One spec, one `X'X`, a plain column and a formula: the plain target is
    learned at the embargo, the formula once its window has closed too."""
    df = stream(300, 15).with_columns(y=pl.col("mid") * 0.5 + pl.col("x").fill_null(0.0))
    s = po.spec.ewridge(
        "m",
        targets=["y", fwd()],
        features=["x"],
        clock="t",
        gap_cap=100.0,
        half_life=50.0,
        min_weight=3.0,
        max_rows_between_solves=1,
        embargo=W + 2.5,
        emit_clocks=True,
    )
    got = native(df, s, chunk_rows=37)
    plain = dict(s)
    plain["targets"] = ["y", "fwd"]
    want = po.ModelBank([plain]).fit_predict(po.stream.with_windows(df, fwd(), like=s))
    for name in ["y", "fwd"]:
        assert field(got, f"pred_{name}") == field(want, f"pred_{name}"), name
    assert {"pred_y", "pred_fwd", "resid_y", "resid_fwd"} <= set(po.spec.output_fields(s))
    assert math.isfinite(field(got, "pred_y")[-1])


# --------------------------------------------------------------------------
# Review round R1 (2026-10-03)


def test_a_dropped_group_starts_cold_in_the_window_core_too() -> None:
    """R1-D1: ``drop_groups`` left the group's window core behind, so the
    same rows fed again met a stale clock (refused) or stale held rows."""
    df = stream(200, 31).with_columns(g=pl.lit("a"))
    s = spec(fwd(), group="g")
    fresh = native(df, s)
    bank = po.ModelBank([s])
    bank.fit_predict(df)
    assert bank.drop_groups(["a"]) == 1
    again = bank.fit_predict(df)
    assert field(again, "pred_fwd") == field(fresh, "pred_fwd")


def test_a_boolean_column_reaches_a_formula_target() -> None:
    """R1-D2: a boolean was cast to a number on its way into the bank and
    the resolver's dtype check refused it, where ``with_windows`` took it."""
    df = stream(300, 32).with_columns(
        is_buy=pl.Series(np.random.default_rng(1).random(300) < 0.5), qty=pl.lit(2.0)
    )
    expr = (
        po.rewm_sum(pl.when(pl.col("is_buy")).then("qty"), half_life=H, window_size=W)
        - pl.col("qty")
    ).alias("fwd")
    s = spec(expr)
    got = native(df, s)
    want = column_form(df, s, expr)
    assert field(got, "pred_fwd") == field(want, "pred_fwd")
    assert sum(p is not None for p in field(got, "pred_fwd")) > 100


def test_rows_learned_counts_a_formula_row_once_its_target_resolved() -> None:
    """R1-D3: a row whose window a gap cut null was counted as learned from
    at its arrival; it counts when it is released with a value, and the
    count is the same whatever the chunking."""
    df = stream(240, 33, gap_at=120)
    s = spec(fwd(partial="null"))
    counts = []
    for chunk_rows in (None, 7):
        bank = po.ModelBank([s])
        if chunk_rows is None:
            bank.fit_predict(df)
        else:
            list(bank.fit_predict_batches(df, chunk_rows=chunk_rows))
        counts.append(bank.summary()["rows_learned"][0])
    assert counts[0] == counts[1]
    # The column form says which rows had a target: those the spec accepts
    # whose window closed with a value. The rows still held under the embargo
    # when the input ends are not learned from yet.
    col = po.stream.with_windows(df, fwd(partial="null"), like=s)["fwd"]
    known = col.is_not_null().sum()
    tail = int((df["t"] > df["t"].max() - (W + 2.5)).sum())
    assert known - tail <= counts[0] <= known < df.height


def test_drop_on_one_target_leaves_the_rows_other_targets() -> None:
    """R1-D4: ``partial="drop"`` on one operator nulled every formula of the
    row; it nulls the formulas over that operator."""
    df = stream(240, 34, gap_at=120)
    outs = []
    for pa in ("null", "drop"):
        s = po.spec.ewridge(
            "m",
            targets=[
                (
                    po.rewm_mean("mid", half_life=H, window_size=20.0, partial=pa) - pl.col("mid")
                ).alias("a"),
                (po.rewm_mean("mid", half_life=H, window_size=5.0) - pl.col("mid")).alias("b"),
            ],
            features=["x"],
            clock="t",
            gap_cap=100.0,
            half_life=50.0,
            min_weight=3.0,
            max_rows_between_solves=1,
            embargo=22.5,
            emit_clocks=True,
        )
        outs.append(native(df, s))
    assert field(outs[0], "pred_b") == field(outs[1], "pred_b")
    assert field(outs[0], "pred_a") == field(outs[1], "pred_a")


def test_a_formulas_spans_are_the_specs_clock_spans() -> None:
    """R1-D7: a formula's ``window_size`` beside the spec's durations is
    refused at the spec, as any other mixture is, rather than compared as
    unlike numbers by the embargo check."""
    with pytest.raises(ValueError, match="window_size"):
        po.spec.ewridge(
            "m",
            targets=[
                (po.rewm_mean("mid", half_life="5s", window_size=600.0) - pl.col("mid")).alias("f")
            ],
            features=["x"],
            clock="t",
            gap_cap="5m",
            half_life="1h",
            embargo="2m",
        )


def test_a_raw_spec_dict_takes_a_window_expression() -> None:
    """R1-D8: a hand-written dict with a ``pl.Expr`` in ``targets`` died in
    ``json.dumps``; the bank normalises it as the builders do."""
    s = dict(spec(fwd()))
    s["targets"] = [fwd()]
    bank = po.ModelBank([s])
    assert bank.specs[0]["targets"][0]["name"] == "fwd"
    assert field(bank.fit_predict(stream(120, 35)), "pred_fwd")[-1] is not None


# --------------------------------------------------------------------------
# Review round R2 (2026-10-03)


def test_learn_only_is_the_calls_not_the_banks() -> None:
    """R2-P1: a learn-only flag on the bank could be left set when a
    ``predict`` on another thread held the bank, and ``fit_predict`` then ran
    in sample; the run says so per call, and the bank keeps no such flag."""
    df = stream(120, 41)
    bank = po.ModelBank([spec(fwd(), embargo=None)])
    assert not hasattr(bank._native, "set_learn_only")
    bank._native.fit_predict(df.slice(0, 60), 0, True)
    with pytest.raises(ValueError, match="fit_predict needs an embargo"):
        bank._native.fit_predict(df.slice(60, 60), 60)
    with pytest.raises(ValueError, match="keeps no prediction"):
        bank.fit_predict(df.slice(60, 60))


def test_a_formula_target_is_never_order_free() -> None:
    """R2-P2: ``fit`` over accumulator-only specs skips the row-order warning,
    since their sums commute; a target that reads the rows ahead does not."""
    from polars_online._frame import _order_free

    plain = po.spec.ewridge("m", targets=["y"], features=["x"], lam=1.0)
    ahead = po.spec.ewridge(
        "m",
        targets=[(po.rewm_mean("mid", half_life=5.0, window_size=10.0) - pl.col("mid")).alias("f")],
        features=["x"],
        lam=1.0,
    )
    assert _order_free([plain]) and not _order_free([ahead])
    df = stream(200, 42).with_columns(y=pl.col("mid"))
    other = pl.DataFrame({"t": df["t"], "z": range(200)})
    plan = df.lazy().join(other.lazy(), on="t")
    with pytest.warns(po.OrderNotGuaranteedWarning):
        po.ModelBank([ahead]).fit(plan)


def test_a_null_literal_has_a_form_toml_can_carry(online_cli: Any, tmp_path: Any) -> None:
    """R2-P3: ``when/then`` with no ``otherwise`` carries a null literal, which
    TOML has no value for; the tree spells it ``["lit"]``."""
    from polars_online._formula import from_tree, to_tree

    tree = to_tree(pl.when(pl.col("x") > 0).then("mid"))
    assert tree[-1] == ["lit"]
    assert from_tree(tree).meta.eq(pl.when(pl.col("x") > 0).then("mid"))
    assert to_tree(pl.col("x").clip(upper_bound=1.0))[2] == ["lit"]
    assert from_tree(to_tree(pl.col("x").clip(upper_bound=1.0))).meta.eq(
        pl.col("x").clip(upper_bound=1.0)
    )
    df = stream(240, 43)
    target = (
        po.rewm_sum(pl.when(pl.col("x") > 0).then("mid"), half_life=H, window_size=W)
        - pl.col("mid")
    ).alias("fwd")
    s = spec(target)
    df.write_parquet(tmp_path / "in.parquet")
    run_online(
        online_cli, tmp_path, [s], input=tmp_path / "in.parquet", output=tmp_path / "out.parquet"
    )
    got = pl.read_parquet(tmp_path / "out.parquet")
    assert field(got, "pred_fwd") == field(native(df, s), "pred_fwd")


def test_no_output_is_the_command_lines_fit(online_cli: Any, tmp_path: Any) -> None:
    """R2-P4: a run with no output keeps no prediction, so it takes any
    embargo, as ``ModelBank.fit`` does."""
    df = stream(120, 44)
    df.write_parquet(tmp_path / "in.parquet")
    run_online(
        online_cli,
        tmp_path,
        [spec(fwd(), embargo=None)],
        input=tmp_path / "in.parquet",
        save_state=tmp_path / "bank.state",
        args=["--no-output"],
    )
    assert po.ModelBank.load(tmp_path / "bank.state").rows_seen() == 120


def test_every_surface_takes_a_raw_dict_with_a_window_expression() -> None:
    """R2-P5/F5: not only the constructor -- ``output_fields``, ``load_bytes``
    and a tuple of targets."""
    raw = dict(spec(fwd()))
    raw["targets"] = (fwd(),)
    assert "pred_fwd" in po.spec.output_fields(raw)
    bank = po.ModelBank([raw])
    bank.fit_predict(stream(80, 45))
    again = po.ModelBank.load_bytes(bank.save_bytes(), [raw])
    assert again.rows_seen() == 80


def test_a_raw_dict_takes_a_duration_under_a_clock_parameter() -> None:
    """Review 2026-10-05 (YA11): an expression under any key of a raw dict
    was read as a formula target, so ``half_life=pl.duration(...)``, the form
    a builder takes, was refused as a formula literal that named no key. An
    expression is a formula under ``targets`` alone; under a clock parameter
    it is a duration, written as the builders write it."""
    df = stream(80, 45).with_columns(
        ts=pl.from_epoch((pl.col("t") * 1e6).cast(pl.Int64), time_unit="us")
    )
    built = po.spec.ewridge(
        "m", targets=["mid"], features=["x"], clock="ts", gap_cap="100s", half_life="50s"
    )
    raw = dict(built, half_life=pl.duration(seconds=50), gap_cap=timedelta(seconds=100))
    assert po.spec.output_fields(raw) == po.spec.output_fields(built)
    assert po.ModelBank([raw]).fit_predict(df).equals(po.ModelBank([built]).fit_predict(df))
    with pytest.raises(TypeError, match='^spec "m": half_life must be a duration that reads no'):
        po.ModelBank([dict(built, half_life=pl.col("x"))])


def test_a_refused_chunk_leaves_every_specs_core_as_it_was() -> None:
    """R2-F1: with two formula specs, a refusal raised by the second after
    the first was fed left the first's core holding the chunk."""
    df = stream(300, 46).with_columns(ask=pl.col("mid") + 0.01)
    a = spec(fwd())
    b = spec((po.rewm_mean("ask", half_life=H, window_size=W) - pl.col("ask")).alias("fwd"))
    b["name"] = "b"
    fresh = po.ModelBank([a, b]).fit_predict(df)
    bank = po.ModelBank([a, b])
    with pytest.raises(ValueError, match="ask"):
        bank.fit_predict(df.drop("ask"))
    assert bank.rows_seen() == 0
    again = bank.fit_predict(df)
    assert field(again, "pred_fwd") == field(fresh, "pred_fwd")


def test_a_non_strict_cast_survives_the_specs_round_trip() -> None:
    """R2-F2: the tree's serializer dropped ``"non_strict"``, so a saved spec
    ran a strict cast."""
    expr = (
        po.rewm_mean(pl.col("mid").cast(pl.Float32, strict=False), half_life=H, window_size=W)
        - pl.col("mid")
    ).alias("fwd")
    bank = po.ModelBank([spec(expr)])
    formula = bank.specs[0]["targets"][0]["formula"]
    assert "non_strict" in str(formula)
    bank.fit_predict(stream(60, 47))
    again = po.ModelBank.load_bytes(bank.save_bytes())
    assert again.specs[0]["targets"][0]["formula"] == formula


def test_a_boolean_also_read_as_a_number_reaches_the_formula_as_a_boolean() -> None:
    """R2-F3: a boolean a feature also reads is held in two forms, and the
    formula took the first found -- the number."""
    df = stream(300, 48).with_columns(
        is_buy=pl.Series(np.random.default_rng(2).random(300) < 0.5), qty=pl.lit(2.0)
    )
    expr = (
        po.rewm_sum(pl.when(pl.col("is_buy")).then("qty"), half_life=H, window_size=W)
        - pl.col("qty")
    ).alias("fwd")
    s = spec(expr, features=["x", "is_buy"])
    got = native(df, s)
    want = column_form(df, s, expr)
    assert field(got, "pred_fwd") == field(want, "pred_fwd")


# --------------------------------------------------------------------------
# Review round R4 (2026-10-03)


def test_a_bank_state_from_before_the_windows_state_changed_is_refused_by_number() -> None:
    """R4-A2, then R6-D5: a bank file carries each formula target's window
    core in the windows state's own form, which went to version 3 in round
    three, 4 in round five and 5 in round six; the bank's schema moved with
    the first and the last (25 since round six), so a file before it is
    refused at load by its number, not at a group's first chunk by the
    windows message. Round five moved the windows version alone, and a 24
    file holding a version-4 core failed late. 28 since task 161, for
    `ew_cov`, 29 since task 163, for `micro`, and 30 since task 162, for
    the models' window cadence, the windows state unchanged each time."""
    bank = po.ModelBank([spec(fwd())])
    bank.fit_predict(stream(60, 50))
    state = bank.save_bytes()
    key = b"\xaeschema_version"
    i = state.index(key) + len(key)
    assert state[i] == 30, state[i]
    for before in (25, 26, 27, 28, 29):
        old = state[:i] + bytes([before]) + state[i + 1 :]
        with pytest.raises(
            ValueError, match=rf"schema version {before} not supported \(this build loads 30"
        ):
            po.ModelBank.load_bytes(old)


def test_a_target_named_after_its_own_column_is_refused_at_the_builder(
    online_cli: Any, tmp_path: Any
) -> None:
    """Task 159 (P1): a window target without ``.alias()`` is named after its
    leftmost column; where that is a column the formula reads, the bank could
    never add the target beside it, yet the builder accepted it and the bank
    refused it on the first chunk in ``with_windows``' words. The builder
    refuses it, naming ``.alias``; and a bank's refusal of a formula target
    names the spec, never ``with_windows``."""
    with pytest.raises(ValueError, match=r"named after a column its formula reads.*\.alias"):
        spec(pl.col("mid") - po.rewm_mean("mid", half_life=H, window_size=W))
    # Task 160 (PB5): the same spec as a table, the form a dict or a TOML
    # file writes, passed the Rust validation and was refused only at the
    # first chunk. It is refused where every spec is checked: by
    # `validate_spec`, by the bank as it is built, and by the CLI's dry run.
    table = spec(fwd())
    table["targets"][0]["name"] = "mid"
    with pytest.raises(ValueError, match='spec "m": target "mid" is named after a column its'):
        validate_spec(json.dumps(table))
    with pytest.raises(ValueError, match="named after a column its formula reads"):
        po.ModelBank([table])
    stream(50, 17).write_parquet(tmp_path / "in.parquet")
    res = run_online(
        online_cli,
        tmp_path,
        [table],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        args=["--dry-run"],
        check=False,
    )
    assert res.returncode != 0, res.stdout
    assert "named after a column its formula reads" in res.stderr, res.stderr
    # A refusal the core raises while the bank runs: a strict cast that
    # overflows. The bank's words, not the operator layer's.
    df = stream(60, 3).with_columns(mid=pl.col("mid") * 1e6)
    s = spec((po.rewm_mean("mid", half_life=H, window_size=W).cast(pl.Int8) * 1.0).alias("fwd"))
    with pytest.raises(ValueError, match="conversion from") as err:
        po.ModelBank([s]).fit_predict(df)
    assert "with_windows" not in str(err.value)


def test_a_clip_bound_left_out_is_the_null_literal_in_a_saved_spec() -> None:
    """Task 159 (F5): the Rust tree wrote a clip's absent bound as a bare
    ``null``, where Python writes ``["lit"]``, the form TOML can carry; a
    saved spec holds no bare null in a formula."""
    target = (
        po.rewm_mean("mid", half_life=H, window_size=W).clip(lower_bound=0) - pl.col("mid")
    ).alias("fwd")
    saved = json.loads(po.ModelBank([spec(target)]).to_json())
    formula = saved["specs"][0]["targets"][0]["formula"]

    def nodes(tree: Any) -> Any:
        yield tree
        if isinstance(tree, list):
            for item in tree:
                yield from nodes(item)

    clips = [n for n in nodes(formula) if isinstance(n, list) and n and n[0] == "clip"]
    assert clips and clips[0][3] == ["lit"], clips
    assert None not in list(nodes(formula))
