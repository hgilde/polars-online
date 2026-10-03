"""`ModelBank.skip_learned`: resuming a saved bank on input that overlaps it
(docs/PLAN.md task 120, decided by the user on 2026-09-28: "Resume should get
a helper").

A bank resumes at the next row. Input that starts before the save steps every
group's clock back to rows the state has learned, which is refused unless
`restart_after_step_back` reads it as a new start. The helper keeps each row after its
group's last clock, in every spec with a clock, and every row of a group the
bank has not seen, so the rerun learns each row once.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po


def frame(n=400, seed=0, groups=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "g": [groups[i % len(groups)] for i in range(n)],
            "t": np.arange(float(n)),
            "x0": x,
            "y": 2.0 * x + 0.1 * rng.standard_normal(n),
        }
    )


def spec(name="m", **kw):
    d = dict(targets=["y"], features=["x0"], clock="t", half_life=50.0, gap_cap=10.0)
    d.update(kw)
    return po.spec.ewridge(name, **d)


def test_a_rerun_learns_each_row_once():
    df = frame()
    specs = [spec(group="g")]
    whole = po.ModelBank(specs).fit_predict(df)
    bank = po.ModelBank(specs)
    bank.fit_predict(df.head(250))
    saved = bank.save_bytes()
    # The rerun starts 100 rows before the save.
    rerun = df.slice(150)
    with pytest.raises(ValueError, match="skip_learned"):
        po.ModelBank.load_bytes(saved, specs).fit_predict(rerun)
    resumed = po.ModelBank.load_bytes(saved, specs)
    kept = resumed.skip_learned(rerun)
    assert kept.equals(df.slice(250))
    out = resumed.fit_predict(kept)
    assert out["m"].equals(whole["m"].slice(250))


def test_a_lazy_frame_stays_lazy_and_keeps_its_order():
    df = frame()
    bank = po.ModelBank([spec(group="g")])
    bank.fit_predict(df.head(200))
    lazy = bank.skip_learned(df.lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(df.slice(200))


def test_a_group_the_bank_has_not_seen_keeps_every_row_and_a_null_group_is_one():
    df = frame().with_columns(
        g=pl.when(pl.col("g") == "c").then(None).otherwise(pl.col("g")).alias("g")
    )
    bank = po.ModelBank([spec(group="g")])
    bank.fit_predict(df.head(90))
    later = pl.concat([df.slice(60, 60), frame(9, 1, ("new",)).with_columns(t=pl.lit(70.0))])
    kept = bank.skip_learned(later)
    last = dict(df.head(90).group_by("g").agg(pl.col("t").max()).iter_rows())
    assert None in last
    want = [r for r in later.iter_rows(named=True) if r["g"] == "new" or r["t"] > last[r["g"]]]
    assert kept.to_dicts() == want
    assert kept.filter(pl.col("g") == "new").height == 9
    assert kept.filter(pl.col("g").is_null()).height > 0


def test_a_row_at_the_last_clock_counts_as_learned_and_a_null_clock_is_kept():
    df = frame(groups=("a",))
    bank = po.ModelBank([spec(group="g")])
    bank.fit_predict(df.head(50))
    last = df["t"][49]
    probe = pl.DataFrame(
        {
            "g": ["a"] * 4,
            "t": [last - 1.0, last, last + 1e-9, None],
            "x0": [1.0] * 4,
            "y": [1.0] * 4,
        }
    )
    assert bank.skip_learned(probe)["t"].to_list() == [last + 1e-9, None]


@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_a_temporal_clock_is_compared_in_exact_nanoseconds(tz):
    """A double of epoch seconds resolves about 0.24 us today; the state keeps
    integer nanoseconds, and so does the comparison, whatever the zone."""
    start = datetime(2024, 3, 10, 6, 59, 59, tzinfo=UTC)
    ns = [int(start.timestamp()) * 10**9 + i for i in range(6)]
    t = pl.Series("t", ns).cast(pl.Datetime("ns", "UTC"))
    if tz is not None:
        t = t.dt.convert_time_zone(tz)
    df = pl.DataFrame({"t": t, "x0": np.arange(6.0), "y": np.arange(6.0)})
    bank = po.ModelBank([spec(half_life="1s", gap_cap="1s")])
    bank.fit_predict(df.head(3))
    assert bank.skip_learned(df).equals(df.slice(3))


def test_a_date_clock():
    days = pl.date_range(datetime(2024, 1, 1), datetime(2024, 1, 20), eager=True)
    df = pl.DataFrame({"d": days, "x0": np.arange(20.0), "y": np.arange(20.0)})
    bank = po.ModelBank([spec(clock="d", half_life="5d", gap_cap="3d")])
    bank.fit_predict(df.head(12))
    assert bank.skip_learned(df.slice(5)).equals(df.slice(12))
    shifted = df.with_columns(d=pl.col("d") - timedelta(days=1))
    assert bank.skip_learned(shifted).equals(shifted.slice(13))


def test_a_row_is_kept_only_where_every_spec_has_not_learned_it():
    """Two specs, one grouped and one not: a row is new only after the last
    clock of its group in the first and of the whole stream in the second."""
    df = frame()
    bank = po.ModelBank([spec("grouped", group="g"), spec("whole")])
    bank.fit_predict(df.head(100))
    extra = pl.DataFrame({"g": ["a"], "t": [98.5], "x0": [0.0], "y": [0.0]})
    # After group a's last clock (97) but before the stream's (99): not new.
    assert bank.skip_learned(extra).height == 0
    assert bank.skip_learned(df).equals(df.slice(100))


def test_what_it_refuses():
    counted = po.ModelBank([po.spec.ewridge("n", targets=["y"], features=["x0"], half_life=5.0)])
    with pytest.raises(ValueError, match="no spec reads a clock"):
        counted.skip_learned(frame())
    bank = po.ModelBank([spec(group="g")])
    bank.fit_predict(frame().head(20))
    with pytest.raises(ValueError, match="clock column 't' is not in the frame"):
        bank.skip_learned(frame().drop("t"))
    with pytest.raises(ValueError, match="group column 'g' is not in the frame"):
        bank.skip_learned(frame().drop("g"))
    temporal = frame().with_columns(t=pl.from_epoch(pl.col("t").cast(pl.Int64), time_unit="s"))
    with pytest.raises(ValueError, match="temporal in the frame and was numeric in the bank"):
        bank.skip_learned(temporal)


def test_an_empty_bank_keeps_everything():
    df = frame()
    assert po.ModelBank([spec(group="g")]).skip_learned(df).equals(df)


def test_an_instant_nanoseconds_cannot_hold_is_kept_for_the_bank_to_refuse():
    """Review 2026-09-28: the frame's clock was scaled to nanoseconds in polars
    integer arithmetic, which wraps, so a Datetime("ms") past 2262 fell below
    every saved clock and was dropped silently, where fit_predict refuses it
    by name. The comparison is in the column's own unit now."""
    t = pl.Series("t", [1_700_000_000_000 + i for i in range(4)]).cast(pl.Datetime("ms"))
    df = pl.DataFrame({"t": t, "x0": np.arange(4.0), "y": np.arange(4.0)})
    bank = po.ModelBank([spec(half_life="1s", gap_cap="1s")])
    bank.fit_predict(df.head(2))
    far = pl.DataFrame(
        {
            "t": pl.Series([datetime(3000, 1, 1, tzinfo=UTC)]).cast(pl.Datetime("ms")),
            "x0": [0.0],
            "y": [0.0],
        }
    )
    kept = bank.skip_learned(pl.concat([df.slice(2), far]))
    assert kept.height == 3
    with pytest.raises(ValueError, match="nanoseconds cannot hold"):
        bank.fit_predict(kept)
