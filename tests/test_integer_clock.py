"""Task 200: an integer clock column is held as an integer, exactly.

An ``Int64`` column of epoch nanoseconds stands near 1.79e18 in 2026, where a
double resolves 256 nanoseconds. Read as a ``Float64``, as every number clock
was, ticks 1, 7 and 100 apart became steps of 0 and 256, and every decision
taken on the clock -- a step back, a gap past ``gap_cap``, a cadence, a
window's edge, an embargo's release -- inherited the rounding. A difference
is now taken in the column's own type, then converted.

Every expected value is computed from the raw integers, in Python's exact
integers. Where a whole run is the oracle, it is the same stream shifted to
start at 0: a ``Float64`` clock whose every value and step a double holds
exactly, which must give the same output to the bit.
"""

import math
import re

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online

#: Epoch nanoseconds in 2026, a multiple of 256: ``T0 + 1`` to ``T0 + 127``
#: are all ``T0`` as a double.
T0 = 1_790_000_000_000_000_000
assert float(T0) == T0 and float(T0 + 127) == T0


def offsets(n: int, seed: int = 0, steps=(1, 7, 100)) -> list[int]:
    rng = np.random.default_rng(seed)
    return [0, *(int(v) for v in np.cumsum(rng.choice(steps, n - 1)))]


def frame(n: int = 90, seed: int = 0, steps=(1, 7, 100), dtype=pl.Int64, base=T0) -> pl.DataFrame:
    off = offsets(n, seed, steps)
    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal(n)
    return pl.DataFrame(
        {
            "t": pl.Series([base + o for o in off], dtype=dtype),
            "x": x,
            "y": 2.0 * x + 0.1 * rng.standard_normal(n),
        }
    )


def shifted(df: pl.DataFrame) -> pl.DataFrame:
    """The same stream at 0: a float clock a double holds exactly."""
    return df.with_columns((pl.col("t") - T0).cast(pl.Float64))


#: Long enough for the fit to carry its two coefficients over the streams.
HALF_LIFE = 2000.0


def ridge(**kw):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=1e6,
        half_life=HALF_LIFE,
        min_weight=0.0,
        emit_clocks=True,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


def run(spec, df: pl.DataFrame, chunks: int = 1) -> pl.DataFrame:
    bank = po.ModelBank([spec])
    size = max(1, math.ceil(df.height / chunks))
    return pl.concat([bank.fit_predict(p) for p in df.iter_slices(size)]).unnest("m")


def same_but_the_clock(a: pl.DataFrame, b: pl.DataFrame) -> None:
    """Every field but the clocks, equal to the bit."""
    clocks = {"t", "scored_clock", "learned_clock"}
    cols = [c for c in a.columns if c not in clocks]
    assert a.select(cols).equals(b.select(cols)), cols


def integers(s: pl.Series) -> list[int | None]:
    return [None if v is None else int(v) for v in s.to_list()]


# --------------------------------------------------------------------------
# The steps, the stamps and the decisions on them


class TestTheSteps:
    def test_a_model_steps_by_the_integers(self):
        """``d_clock`` is the integers' difference: the model, its decay and
        ``settled_frac`` are the shifted stream's to the bit, and
        ``settled_frac`` is ``1 - 2^(-T/h)`` with ``T`` the raw clock
        covered."""
        df = frame()
        got = run(ridge(), df)
        same_but_the_clock(got, run(ridge(), shifted(df)))
        off = offsets(df.height)
        # Before the row: the clock covered up to the row before it.
        want = [0.0] + [1.0 - 2.0 ** (-(o - off[0]) / HALF_LIFE) for o in off[:-1]]
        assert got["settled_frac"].to_list() == pytest.approx(want, abs=1e-15)
        assert got["pred_y"].drop_nulls().len() > 60, "the fit predicts"
        assert any(b - a == 1 for a, b in zip(off, off[1:], strict=False)), "steps of 1 ns"

    def test_a_coef_cadence_counts_the_integers(self):
        """``coef_every = 150``: a ``coef`` row once the clock has moved 150
        since the last, counted on the raw integers, inclusive."""
        df = frame(steps=(1, 7, 50, 100))
        got = run(ridge(coef_every=150), df)
        every_row = run(ridge(coef_every=0), df)
        off = offsets(df.height, steps=(1, 7, 50, 100))
        want, last = [], off[0]
        for i, o in enumerate(off):
            if o - last >= 150:
                want.append(i)
                last = o
        fitted = {i for i, c in enumerate(every_row["coef"].to_list()) if c is not None}
        observed = [i for i, c in enumerate(got["coef"].to_list()) if c is not None]
        assert observed == [i for i in want if i in fitted]
        assert any(off[i] - off[j] == 150 for i, j in zip(want[1:], want, strict=False)), (
            "an edge exactly at the cadence"
        )
        same_but_the_clock(got, run(ridge(coef_every=150), shifted(df)))

    @pytest.mark.parametrize("closed", ["right", "both"])
    def test_a_model_window_holds_what_the_integers_put_in_it(self, closed):
        """``window_size = 100`` on ``ew_cov`` with no decay: ``weight_sum``
        counts the rows less than 100 behind the previous row's clock, from
        the raw integers; a row exactly 100 behind is left out under
        ``closed="right"``, the default, and kept under ``"both"`` (task
        196)."""
        df = frame(steps=(1, 7, 100, 50))
        spec = po.spec.ew_cov(
            "m",
            features=["x"],
            clock="t",
            gap_cap=1e6,
            half_life=math.inf,
            window_size=100.0,
            closed=closed,
            min_weight=0.0,
            stats=["mean"],
        )
        got = run(spec, df)
        off = offsets(df.height, steps=(1, 7, 100, 50))
        reach = 100 if closed == "both" else 99
        want = [
            float(sum(1 for j in range(i) if off[i - 1] - off[j] <= reach))
            for i in range(1, len(off))
        ]
        assert got["weight_sum"][1:].to_list() == want
        edges = sum(1 for i in range(1, len(off)) for j in range(i) if off[i - 1] - off[j] == 100)
        assert edges > 3, "rows exactly one window back"
        same_but_the_clock(got, run(spec, shifted(df)))

    def test_an_embargo_releases_on_the_integers(self):
        """``embargo = 150``: the newest row learned at row ``i`` is the
        newest earlier row at least 150 behind it, inclusive, and the clock
        fields are the column's own ``Int64``, exact."""
        df = frame(steps=(1, 7, 50, 100))
        got = run(ridge(embargo=150), df)
        off = offsets(df.height, steps=(1, 7, 50, 100))
        want = []
        for i in range(len(off)):
            cands = [j for j in range(i) if off[i] - off[j] >= 150]
            want.append(T0 + off[cands[-1]] if cands else None)
        assert got["learned_clock"].dtype == pl.Int64
        assert got["scored_clock"].dtype == pl.Int64
        assert integers(got["learned_clock"]) == want
        assert integers(got["scored_clock"]) == [T0 + o for o in off]
        assert sum(1 for i in range(len(off)) for j in range(i) if off[i] - off[j] == 150) > 3
        same_but_the_clock(got, run(ridge(embargo=150), shifted(df)))

    def test_a_step_back_of_one_is_refused(self):
        df = pl.DataFrame(
            {"t": [T0 + 5, T0 + 6, T0 + 5], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(ValueError, match="goes backwards by 1 at row 2"):
            po.ModelBank([ridge()]).fit_predict(df)

    def test_a_step_back_of_exactly_restart_after_step_back_is_a_late_row(self):
        """``restart_after_step_back = 100`` is inclusive: a step back of 100
        is a late row, refused; one of 101 starts the stream over. As
        doubles the two clocks were 256 apart, past it."""
        spec = ridge(restart_after_step_back=100)
        late = pl.DataFrame({"t": [T0 + 227, T0 + 127], "x": [1.0, 2.0], "y": [1.0, 2.0]})
        with pytest.raises(ValueError, match="goes backwards by 100 at row 1, no more than"):
            po.ModelBank([spec]).fit_predict(late)
        bank = po.ModelBank([spec])
        bank.fit_predict(late.with_columns(t=pl.Series([T0 + 228, T0 + 127])))
        assert bank.summary()["resets"].to_list() == [1]

    def test_a_uint64_past_the_largest_int64_is_refused_by_row(self):
        df = pl.DataFrame(
            {
                "t": pl.Series([5, 2**63], dtype=pl.UInt64),
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
            }
        )
        with pytest.raises(Exception, match=r'clock column "t" has 9223372036854775808 at row 1'):
            po.ModelBank([ridge()]).fit_predict(df)
        ok = df.with_columns(t=pl.Series([5, 2**63 - 1], dtype=pl.UInt64))
        out = po.ModelBank([ridge()]).fit_predict(ok).unnest("m")
        assert out["scored_clock"].dtype == pl.UInt64
        assert out["scored_clock"].to_list() == [5, 2**63 - 1]


# --------------------------------------------------------------------------
# The frames


@pytest.mark.parametrize(
    ("dtype", "base"),
    [(pl.Int32, 2_000_000_000), (pl.Int64, T0), (pl.UInt32, 4_000_000_000)],
)
def test_the_frames_give_an_integer_clock_in_its_own_dtype(dtype, base):
    df = frame(n=40, dtype=dtype, base=base).with_columns(g=pl.lit("a"))
    off = offsets(40)
    lo, hi = base + off[0], base + off[-1]
    bank = po.ModelBank([ridge(group="g")])
    bank.fit_predict(df)
    s = bank.summary()
    for col, want in (("clock_min", lo), ("clock_max", hi), ("last_clock", hi)):
        assert s[col].dtype == dtype, col
        assert s[col].to_list() == [want], col
    g = bank.groups()
    assert g["last_clock"].dtype == dtype
    assert g["last_clock"].to_list() == [hi]

    cov = po.spec.ew_cov(
        "c",
        features=["x", "y"],
        clock="t",
        gap_cap=1e6,
        half_life=50.0,
        group="g",
        group_close="monotone",
    )
    closing = po.ModelBank([cov])
    two = pl.concat([df, df.with_columns(g=pl.lit("b"), t=pl.col("t") + 1000)])
    closing.fit_predict(two)
    closed = closing.closed_groups()
    assert closed["clock_min"].dtype == dtype and closed["clock_max"].dtype == dtype
    assert closed["clock_min"].to_list() == [lo] and closed["clock_max"].to_list() == [hi]


def test_a_float_clock_stays_a_float():
    df = shifted(frame(n=20))
    bank = po.ModelBank([ridge()])
    out = bank.fit_predict(df).unnest("m")
    assert out["scored_clock"].dtype == pl.Float64
    assert bank.summary()["clock_max"].dtype == pl.Float64
    assert bank.groups()["last_clock"].dtype == pl.Float64


def test_another_number_dtype_than_the_banks_is_refused():
    """The bank keeps the clock's dtype from its first chunk: an integer
    clock and a float one are two ways of reading every step, so a chunk of
    the other is refused by name, as a key column of another form is."""
    df = frame(n=20)
    bank = po.ModelBank([ridge()])
    bank.fit_predict(df[:10])
    with pytest.raises(Exception, match=r'clock column "t" was i64 and is now f64'):
        bank.fit_predict(df[10:].with_columns(pl.col("t").cast(pl.Float64)))
    with pytest.raises(Exception, match=r'clock column "t" was i64 and is now i32'):
        bank.fit_predict(
            df[10:].with_columns(pl.col("t") - T0).with_columns(pl.col("t").cast(pl.Int32))
        )
    # Refused before anything moved: the right dtype goes on.
    bank.fit_predict(df[10:])


# --------------------------------------------------------------------------
# Chunks and states


class TestChunksAndStates:
    def spec(self):
        return ridge(coef_every=150, embargo=120, window_size=300.0)

    @pytest.mark.parametrize("chunks", [7, 37])
    def test_any_chunking_gives_one_output(self, chunks):
        df = frame(n=120)
        one = run(self.spec(), df)
        many = run(self.spec(), df, chunks=chunks)
        floats = [c for c in one.columns if c != "coef"]
        assert one.select(floats).equals(many.select(floats))

    def test_a_state_saved_between_two_events_resumes_on_the_integers(self, tmp_path):
        df = frame(n=120)
        whole = run(self.spec(), df)
        bank = po.ModelBank([self.spec()])
        first = bank.fit_predict(df[:61]).unnest("m")
        path = tmp_path / "int.state"
        bank.save(path)
        second = po.ModelBank.load(path).fit_predict(df[61:]).unnest("m")
        floats = [c for c in whole.columns if c != "coef"]
        assert pl.concat([first, second]).select(floats).equals(whole.select(floats))
        assert second["learned_clock"].dtype == pl.Int64

    def test_skip_learned_compares_the_integers(self):
        """A row one nanosecond after the last learned clock is unlearned;
        as doubles the two were one value, and the row was dropped."""
        df = pl.DataFrame({"t": [T0 + 1, T0 + 5], "x": [1.0, 2.0], "y": [1.0, 2.0]})
        bank = po.ModelBank([ridge()])
        bank.fit_predict(df)
        rerun = pl.DataFrame({"t": [T0 + 5, T0 + 6], "x": [2.0, 3.0], "y": [2.0, 3.0]})
        assert bank.skip_learned(rerun)["t"].to_list() == [T0 + 6]


# --------------------------------------------------------------------------
# The other readers of a clock column


CLOCK = {"clock": "t", "gap_cap": 1e9}


@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_a_window_holds_what_polars_rolling_holds_on_the_integers(closed):
    """A window operator on an integer clock of epoch nanoseconds holds the
    rows Polars' ``rolling_sum_by`` holds, which computes on the integers:
    with steps of 0, 1, 7 and 100 and a window of 100, rows exactly one
    window apart land on Polars' side of every edge."""
    df = frame(n=150, steps=(0, 1, 7, 100))
    out = po.stream.with_windows(
        df, s=po.ewm_sum("x", half_life=math.inf, window_size=100.0, closed=closed), **CLOCK
    )
    ref = df.select(pl.col("x").rolling_sum_by("t", window_size="100i", closed=closed))["x"]
    # An empty window sums to 0 from Polars 1.41.1 and to null before it
    # (`test_windows.py`, measured 2026-10-07; the floor canary of
    # 2026-10-08 met it here): both sides read an empty window as 0.
    want = [0.0 if v is None else v for v in ref.to_list()]
    got = [0.0 if v is None else v for v in out["s"].to_list()]
    assert got == pytest.approx(want, abs=1e-12), closed
    off = offsets(150, steps=(0, 1, 7, 100))
    assert sum(1 for i in range(len(off)) for j in range(i) if off[i] - off[j] == 100) > 3


def test_a_gap_of_exactly_gap_cap_does_not_cut_a_window():
    """``gap_cap = 100``: a step of exactly 100 keeps the window open, one of
    101 cuts it. As doubles the first was a step of 256."""
    sums = po.ewm_sum("x", half_life=math.inf, window_size=1000.0)

    def run_on(t):
        df = pl.DataFrame({"t": t, "x": [1.0, 10.0, 100.0]})
        return po.stream.with_windows(df, s=sums, clock="t", gap_cap=100.0)["s"].to_list()

    assert run_on([T0 + 27, T0 + 127, T0 + 227]) == [1.0, 11.0, 111.0]
    assert run_on([T0 + 27, T0 + 127, T0 + 228]) == [1.0, 11.0, 100.0]


def test_an_increment_of_an_integer_input_is_the_integers_step():
    """``po.increment`` of an ``Int64`` column near 1.79e18 is its step in
    integers, then converted: 1, 7 and 100, where the doubles of the two
    values gave 0."""
    c = [T0, T0 + 1, T0 + 8, T0 + 108, T0 + 108, T0 + 5]
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "c": c})
    out = po.stream.with_windows(df, d=po.increment("c"), **CLOCK)
    assert out["d"].to_list() == [None, 1.0, 7.0, 100.0, 0.0, -103.0]
    unsigned = df.with_columns(pl.col("c").cast(pl.UInt64))
    out = po.stream.with_windows(unsigned, d=po.increment("c"), **CLOCK)
    assert out["d"].to_list() == [None, 1.0, 7.0, 100.0, 0.0, -103.0]


def test_a_window_state_resumes_on_the_integers(tmp_path):
    df = frame(n=150, steps=(0, 1, 7, 100))
    s = po.ewm_sum("x", half_life=math.inf, window_size=100.0)
    whole = po.stream.with_windows(df, s=s, **CLOCK)["s"]
    state = tmp_path / "w.state"
    a = po.stream.with_windows(df[:70], s=s, save_state=state, **CLOCK)
    b = po.stream.with_windows(df[70:], s=s, load_state=state, **CLOCK)
    assert pl.concat([a, b])["s"].equals(whole)


def test_refresh_time_orders_an_integer_clock_exactly():
    """A step back of one nanosecond is refused, by row; as doubles the two
    clocks were one."""
    df = pl.DataFrame(
        {"series": ["a", "b", "a"], "t": [T0 + 5, T0 + 6, T0 + 5], "v": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(Exception, match="row 2.*clock order"):
        po.stream.refresh_time(df, series="series", names=["a", "b"], clock="t", value="v")
    fine = df.with_columns(t=pl.Series([T0 + 5, T0 + 6, T0 + 7]))
    grid = po.stream.refresh_time(fine, series="series", names=["a", "b"], clock="t", value="v")
    assert grid["time_refresh"].dtype == pl.Int64
    assert grid["time_refresh"].to_list() == [T0 + 6]


def test_the_command_line_reads_an_integer_clock_as_the_bank_does(online_cli, tmp_path):
    """The CLI runs the bank's own reader: the same output, the clock fields
    in ``Int64``, and a step back of one nanosecond refused by row."""
    df = frame(steps=(1, 7, 50, 100))
    spec = ridge(embargo=150)
    df.write_parquet(tmp_path / "in.parquet")
    run_online(
        online_cli, tmp_path, [spec], input=tmp_path / "in.parquet", output=tmp_path / "out.parquet"
    )
    got = pl.read_parquet(tmp_path / "out.parquet").unnest("m")
    want = po.ModelBank([spec]).fit_predict(df).unnest("m")
    assert got["learned_clock"].dtype == pl.Int64
    assert got.select(want.columns).equals(want)
    back = pl.DataFrame({"t": [T0 + 5, T0 + 6, T0 + 5], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    back.write_parquet(tmp_path / "back.parquet")
    res = run_online(
        online_cli,
        tmp_path,
        [ridge()],
        input=tmp_path / "back.parquet",
        output=tmp_path / "back_out.parquet",
        check=False,
    )
    assert res.returncode != 0 and "goes backwards by 1 at row 2" in res.stderr, res.stderr


def test_embargo_moves_an_integer_clock_in_integers():
    """``po.stream.embargo`` adds a whole delay in the clock's own dtype,
    which Polars computes on the integers: exact before this task too."""
    df = frame(n=30)
    out = po.stream.embargo(df, clock="t", delay=7)
    learn = out.filter(pl.col("_online_role") == "learn")
    assert learn["t"].dtype == pl.Int64
    assert sorted(learn["t"].to_list()) == sorted(T0 + o + 7 for o in offsets(30))


# --------------------------------------------------------------------------
# Wider than 64 bits (review round 5, B3)

#: Past the largest ``Int64``: where an ``Int128`` running count goes on.
BIG = 2**63

#: What every surface says of an ``Int128`` clock, the dtype named. The
#: quote around the column is Rust's on the bank's side and Python's on
#: ``embargo``'s.
WIDE = (
    r"clock column [\"']t[\"'] is Int128, wider than the Int64 an integer clock is held "
    r"in; cast it to Int64 if its values fit, or after subtracting an origin"
)


def _wide(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("t").cast(pl.Int128))


@pytest.mark.parametrize("surface", ["bank", "with_windows", "refresh_time", "embargo"])
def test_an_int128_clock_is_refused_by_name(surface):
    """An integer clock is held as an ``Int64`` (task 200); an ``Int128`` was
    read as a double instead, so its steps of 1 near ``1.79e18`` were 0, and
    nothing said so: ``settled_frac`` stayed 0 and a five-unit window held a
    row seven units back (review round 5, B3). Every surface that reads a
    clock refuses it by name, with the advice a ``UInt64`` past the largest
    ``Int64`` gets."""
    df = _wide(frame(n=5, steps=(1, 7)))
    if surface == "bank":
        call = lambda: po.ModelBank([ridge()]).fit_predict(df)  # noqa: E731
    elif surface == "with_windows":
        s = po.ewm_sum("x", half_life=math.inf, window_size=5.0)
        call = lambda: po.stream.with_windows(df, s=s, **CLOCK)  # noqa: E731
    elif surface == "refresh_time":
        ticks = df.with_columns(series=pl.Series(["a", "b", "a", "b", "a"]))
        call = lambda: po.stream.refresh_time(  # noqa: E731
            ticks, series="series", names=["a", "b"], clock="t", value="x"
        )
    else:
        call = lambda: po.stream.embargo(df, clock="t", delay=7)  # noqa: E731
    with pytest.raises((ValueError, TypeError, pl.exceptions.ComputeError), match=WIDE):
        call()


def test_the_command_line_refuses_an_int128_clock_by_name(online_cli, tmp_path):
    """The CLI runs the bank's reader, so a parquet file's ``Int128`` clock is
    refused there too, with the same words."""
    _wide(frame(n=5, steps=(1, 7))).write_parquet(tmp_path / "wide.parquet")
    res = run_online(
        online_cli,
        tmp_path,
        [ridge()],
        input=tmp_path / "wide.parquet",
        output=tmp_path / "wide_out.parquet",
        check=False,
    )
    assert res.returncode != 0, res.stderr
    assert re.search(WIDE, res.stderr), res.stderr


def test_an_increment_of_an_int128_input_is_the_integers_step():
    """``po.increment`` of an ``Int128`` running count past the largest
    ``Int64`` -- ``2**63 + k`` -- is its step in integers, then made a float,
    as a 64-bit one's is: 1, 7, 100, 0 and -103. It was read as doubles,
    which resolve 2048 there, and gave 0.0 for every step (review round 5,
    B3). At the type's two ends the step is ``2**128 - 1``, which no ``i128``
    holds: it is taken whole and rounded once, to ``2**128``."""
    c = [BIG + k for k in (0, 1, 8, 108, 108, 5)]
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "c": pl.Series(c, dtype=pl.Int128)})
    out = po.stream.with_windows(df, d=po.increment("c"), **CLOCK)
    assert out["d"].to_list() == [None, 1.0, 7.0, 100.0, 0.0, -103.0]
    ends = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0], "c": pl.Series([-(2**127), 2**127 - 1, -(2**127)], dtype=pl.Int128)}
    )
    out = po.stream.with_windows(ends, d=po.increment("c"), **CLOCK)
    assert out["d"].to_list() == [None, 2.0**128, -(2.0**128)]


def test_an_int128_increment_resumes_on_the_integers(tmp_path):
    """The previous value a saved state keeps for an ``Int128`` input past
    the largest ``Int64`` is the integer, so a resumed stream's first step is
    exact too."""
    c = [BIG + k for k in (0, 1, 8, 108, 108, 5)]
    df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "c": pl.Series(c, dtype=pl.Int128)})
    d = po.increment("c")
    state = tmp_path / "w.state"
    a = po.stream.with_windows(df[:3], d=d, save_state=state, **CLOCK)
    b = po.stream.with_windows(df[3:], d=d, load_state=state, **CLOCK)
    assert pl.concat([a, b])["d"].to_list() == [None, 1.0, 7.0, 100.0, 0.0, -103.0]


# --------------------------------------------------------------------------
# UInt128: no surface can take one in (task 208's worker)

#: What a ``UInt128`` clock gets: the ``Int128`` clock's words (B3).
WIDE_UNSIGNED = (
    r"clock column [\"']t[\"'] is UInt128, wider than the Int64 an integer clock is held "
    r"in; cast it to Int64 if its values fit, or after subtracting an origin"
)

#: What any other ``UInt128`` column gets: the column named, and a cast.
UNSIGNED_128 = (
    r"column 'c' is UInt128, which polars-online cannot take in; cast it to Int64 if its "
    r"values fit, or to Float64, or leave it out of the frame"
)

#: Every surface a frame crosses to the extension on.
SURFACES = [
    "fit_predict",
    "predict",
    "fit",
    "fit_predict_batches",
    "fit_predict_arrow",
    "predict_arrow",
    "lf.online.fit_predict",
    "lf.online.predict",
    "with_windows",
    "refresh_time",
]


def _call(surface: str, df: pl.DataFrame):
    """``surface`` run on ``df``, its clock ``t``, as a callable."""
    if surface == "with_windows":
        s = po.ewm_sum("x", half_life=math.inf, window_size=5.0)
        return lambda: po.stream.with_windows(df, s=s, **CLOCK)
    if surface == "refresh_time":
        ticks = df.with_columns(series=pl.Series(["a", "b"] * (df.height // 2) + ["a"]))
        return lambda: po.stream.refresh_time(
            ticks, series="series", names=["a", "b"], clock="t", value="x"
        )
    if surface == "lf.online.fit_predict":
        return lambda: df.lazy().online.fit_predict([ridge()]).collect()
    if surface == "lf.online.predict":
        return lambda: df.lazy().online.predict(po.ModelBank([ridge()])).collect()
    if surface == "fit_predict_batches":
        return lambda: list(po.ModelBank([ridge()]).fit_predict_batches(df))
    bank = po.ModelBank([ridge()])
    return lambda: getattr(bank, surface)(df)


@pytest.mark.filterwarnings("ignore::polars_online.UnstableWarning")
@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("role", ["clock", "other"])
def test_a_uint128_column_is_refused_by_name(surface, role):
    """A frame with a ``UInt128`` column panicked in Polars' own conversion as
    it crossed to the extension (``PanicException: activate 'dtype-u128'
    feature``), before any of this library's code ran, whether the column
    was a clock, a feature or one no spec reads (task 208's worker). Every
    surface a frame crosses on refuses it first, by name, with a cast to
    make: a clock in the words an ``Int128`` clock gets, any other column in
    words of its own."""
    df = frame(n=5, steps=(1, 7))
    if role == "clock":
        df, words = df.with_columns(pl.col("t").cast(pl.UInt128)), WIDE_UNSIGNED
    else:
        df, words = df.with_columns(c=pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt128)), UNSIGNED_128
    with pytest.raises(ValueError, match=words):
        _call(surface, df)()
