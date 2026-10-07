"""Task 152: ``emit_clocks`` -- the clock a row was scored at, and the clock
of the newest row the models had learned from when it was scored.

Every expected value is counted from the frame: the learned row at row *i*
is the newest earlier row that was accepted at a positive weight and, under
``embargo``, whose delay had passed on the clock column by row *i*. A
``Datetime`` clock comes back in its own unit and zone, exact to the
nanosecond; with no clock, the row's index in its group, every row counted.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po


def frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.5, 2.0, n))
    x = rng.standard_normal(n)
    return pl.DataFrame({"t": t, "x": x, "y": 2 * x + rng.standard_normal(n)})


def spec(**kw):
    d = dict(
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=5.0,
        half_life=20.0,
        min_weight=0.0,
        emit_clocks=True,
    )
    d.update(kw)
    return po.spec.ewridge("m", **d)


def expected(t, accept, learned_at, delay=0.0):
    """Per row: (scored, learned) as the plan defines them, from the frame.
    ``learned_at[j]`` is whether row *j* teaches at a positive weight."""
    n = len(t)
    scored = [t[i] if accept[i] else None for i in range(n)]
    learned = []
    for i in range(n):
        if not accept[i]:
            learned.append(None)
            continue
        cands = [j for j in range(i) if learned_at[j] and t[i] - t[j] >= delay]
        learned.append(t[cands[-1]] if cands else None)
    return scored, learned


def fields(out):
    st = out["m"].struct
    return st.field("scored_clock").to_list(), st.field("learned_clock").to_list()


class TestWithoutADelay:
    def test_learned_is_the_previous_row_taught_at_a_positive_weight(self):
        df = frame()
        n = df.height
        w = np.ones(n)
        w[[10, 11, 40]] = 0.0  # a zero-weight row teaches nothing
        df = df.with_columns(w=pl.Series(w))
        df = df.with_columns(
            x=pl.when(pl.int_range(pl.len()).is_in([5, 6, 50])).then(None).otherwise(pl.col("x"))
        )
        out = po.ModelBank([spec(weight="w")]).fit_predict(df)
        t = df["t"].to_numpy()
        accept = df["x"].is_not_null().to_numpy()
        learned_at = accept & (w > 0.0)
        s_want, l_want = expected(t, accept, learned_at)
        s_got, l_got = fields(out)
        assert s_got == pytest.approx(s_want, abs=0.0)
        assert l_got == pytest.approx(l_want, abs=0.0)
        assert l_got[0] is None
        assert l_got[11] == t[9]  # the zero-weight row 10 was never learned

    def test_the_fields_are_last_and_typed_clock(self):
        s = spec()
        names = po.spec.output_fields(s)
        assert names[-2:] == ["scored_clock", "learned_clock"]
        idx = po.spec.output_index(s)
        assert idx.filter(pl.col("field").is_in(["scored_clock", "learned_clock"]))[
            "dtype"
        ].to_list() == ["clock", "clock"]
        assert "scored_clock" not in po.spec.output_fields(spec(emit_clocks=False))


class TestUnderADelay:
    DELAY = 7.0

    def test_learned_is_the_newest_row_whose_delay_had_passed(self):
        df = frame(n=120, seed=1)
        t = df["t"].to_numpy().copy()
        t[60:] += 3.0  # a capped gap inside the delay holds the rows (task 153)
        df = df.with_columns(t=pl.Series(t))
        out = po.ModelBank([spec(embargo=self.DELAY)]).fit_predict(df)
        accept = np.ones(df.height, dtype=bool)
        s_want, l_want = expected(t, accept, accept, self.DELAY)
        s_got, l_got = fields(out)
        assert s_got == pytest.approx(s_want, abs=0.0)
        assert l_got == pytest.approx(l_want, abs=0.0)

    def test_the_embargo_is_visible_on_every_row(self):
        df = frame(n=150, seed=2)
        t = df["t"].to_numpy()
        out = po.ModelBank([spec(embargo=self.DELAY)]).fit_predict(df)
        s_got, l_got = fields(out)
        for i, (s, lrn) in enumerate(zip(s_got, l_got, strict=True)):
            if lrn is None:
                continue
            assert s - lrn >= self.DELAY, i
            # The next row the bank still holds is less than one delay back.
            held = [j for j in range(i) if t[j] > lrn]
            if held:
                assert t[i] - t[held[0]] < self.DELAY, i


class TestTypes:
    @pytest.mark.parametrize("unit", ["ms", "us", "ns"])
    @pytest.mark.parametrize("tz", [None, "UTC", "Europe/London"])
    def test_a_datetime_clock_in_its_own_unit_and_zone_exact_to_the_nanosecond(self, unit, tz):
        n = 40
        rng = np.random.default_rng(3)
        per = {"ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
        # Steps of 0.5 to 2 s with sub-unit detail: every tick of the unit matters.
        steps = (rng.uniform(0.5, 2.0, n) * 1e9).astype(np.int64) // per
        ticks = np.cumsum(steps) + 1_700_000_000 * (1_000_000_000 // per)
        clock = pl.Series(ticks).cast(pl.Datetime(unit))
        if tz:
            clock = clock.dt.replace_time_zone(tz)
        x = rng.standard_normal(n)
        df = pl.DataFrame({"t": clock, "x": x, "y": 2 * x + rng.standard_normal(n)})
        out = po.ModelBank([spec(gap_cap="10s", half_life="20s")]).fit_predict(df)
        st = out["m"].struct
        assert st.field("scored_clock").dtype == df["t"].dtype
        assert st.field("learned_clock").dtype == df["t"].dtype
        assert st.field("scored_clock").to_list() == df["t"].to_list()
        assert st.field("learned_clock").to_list() == [None] + df["t"].to_list()[:-1]

    def test_a_date_clock(self):
        n = 30
        rng = np.random.default_rng(4)
        start = date(2024, 1, 1)
        days = [start + timedelta(days=int(i)) for i in range(n)]
        x = rng.standard_normal(n)
        df = pl.DataFrame({"t": days, "x": x, "y": x + rng.standard_normal(n)})
        out = po.ModelBank([spec(gap_cap="5d", half_life="20d")]).fit_predict(df)
        st = out["m"].struct
        assert st.field("scored_clock").dtype == pl.Date
        assert st.field("scored_clock").to_list() == days
        assert st.field("learned_clock").to_list() == [None] + days[:-1]

    def test_a_duration_clock(self):
        n = 30
        rng = np.random.default_rng(5)
        ms = np.cumsum(rng.integers(500, 2000, n))
        x = rng.standard_normal(n)
        df = pl.DataFrame(
            {"t": pl.Series(ms).cast(pl.Duration("ms")), "x": x, "y": x + rng.standard_normal(n)}
        )
        out = po.ModelBank([spec(gap_cap="10s", half_life="20s")]).fit_predict(df)
        st = out["m"].struct
        assert st.field("scored_clock").dtype == pl.Duration("ms")
        assert st.field("scored_clock").to_list() == df["t"].to_list()

    def test_an_integer_clock_comes_back_as_its_integers(self):
        """Task 200: an integer clock is held as an integer and comes back
        in its own dtype, where it came back as a float."""
        n = 30
        rng = np.random.default_rng(6)
        x = rng.standard_normal(n)
        df = pl.DataFrame({"t": np.arange(n), "x": x, "y": x + rng.standard_normal(n)})
        out = po.ModelBank([spec()]).fit_predict(df)
        st = out["m"].struct
        assert st.field("scored_clock").dtype == pl.Int64
        assert st.field("scored_clock").to_list() == list(range(n))
        df = df.with_columns(pl.col("t").cast(pl.UInt16))
        out = po.ModelBank([spec()]).fit_predict(df)
        assert out["m"].struct.field("scored_clock").dtype == pl.UInt16

    def test_no_clock_is_the_rows_index_in_its_group(self):
        n = 60
        rng = np.random.default_rng(7)
        x = rng.standard_normal(n)
        x[[4, 5, 9]] = np.nan
        g = np.where(np.arange(n) % 3 == 0, "a", "b")
        df = pl.DataFrame(
            {"x": x, "y": np.nan_to_num(x) + rng.standard_normal(n), "g": g}
        ).with_columns(x=pl.col("x").fill_nan(None))
        out = po.ModelBank([spec(clock=None, gap_cap=None, group="g")]).fit_predict(df)
        st = out["m"].struct
        assert st.field("scored_clock").dtype == pl.Int64
        s_got, l_got = fields(out)
        index = {}
        for i in range(n):
            index[i] = sum(1 for j in range(i) if g[j] == g[i])
        for i in range(n):
            if np.isnan(x[i]):
                assert s_got[i] is None and l_got[i] is None
                continue
            assert s_got[i] == index[i], i
            prev = [j for j in range(i) if g[j] == g[i] and not np.isnan(x[j])]
            assert l_got[i] == (index[prev[-1]] if prev else None), i


class TestEvents:
    def test_a_reset_clears_the_learned_clock(self):
        df = frame(n=60, seed=8)
        t = df["t"].to_numpy().copy()
        t[30:] -= t[30] - 1.0  # the clock jumps back at row 30
        df = df.with_columns(t=pl.Series(t))
        s = spec(restart_after_step_back=0.0)
        out = po.ModelBank([s]).fit_predict(df)
        _, l_got = fields(out)
        assert l_got[30] is None
        assert l_got[31] == t[30]

    def test_each_groups_first_row_has_no_learned_clock(self):
        df = frame(n=60, seed=9).with_columns(
            g=pl.Series(np.where(np.arange(60) % 2 == 0, "a", "b"))
        )
        out = po.ModelBank([spec(group="g")]).fit_predict(df)
        _, l_got = fields(out)
        assert l_got[0] is None and l_got[1] is None
        assert l_got[2] == df["t"][0] and l_got[3] == df["t"][1]


class TestPlumbing:
    def _df(self):
        return frame(n=200, seed=10).with_columns(
            w=pl.Series(np.where(np.arange(200) % 7 == 3, 0.0, 1.0))
        )

    @pytest.mark.parametrize("size", [1, 3, 7, 50, 199])
    def test_chunk_invariance(self, size):
        df = self._df()
        s = spec(weight="w", embargo=4.0)
        one = po.ModelBank([s]).fit_predict(df)["m"].struct
        bank = po.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])[
            "m"
        ].struct
        for f in ("scored_clock", "learned_clock", "pred_y", "weight_sum"):
            assert one.field(f).to_list() == many.field(f).to_list(), (f, size)

    def test_save_and_load(self, tmp_path):
        df = self._df()
        s = spec(weight="w", embargo=4.0)
        whole = po.ModelBank([s]).fit_predict(df)["m"].struct
        a = po.ModelBank([s])
        a.fit_predict(df.head(100))
        a.save(tmp_path / "c.state")
        b = po.ModelBank.load(tmp_path / "c.state")
        rest = b.fit_predict(df.tail(100))["m"].struct
        for f in ("scored_clock", "learned_clock"):
            assert rest.field(f).to_list() == whole.field(f).to_list()[100:], f

    def test_predict_shows_the_fits_last_learned_row(self):
        df = self._df()
        bank = po.ModelBank([spec(weight="w")])
        bank.fit_predict(df.head(150))
        t = df["t"].to_numpy()
        w = df["w"].to_numpy()
        last = max(t[j] for j in range(150) if w[j] > 0.0)
        out = bank.predict(df.tail(50))["m"].struct
        assert out.field("scored_clock").to_list() == df["t"].to_list()[150:]
        assert out.field("learned_clock").to_list() == [last] * 50

    def test_last_row_carries_the_clocks(self):
        df = self._df()
        bank = po.ModelBank([spec(weight="w", embargo=4.0)])
        out = bank.fit_predict(df)["m"].struct
        last = bank.last_row()
        # The last row processed is the last row of the frame, scored where it sat.
        assert last["scored_clock"][0] == out.field("scored_clock").to_list()[-1]
        assert last["learned_clock"][0] == out.field("learned_clock").to_list()[-1]


def test_predict_without_a_clock_counts_each_row_in_its_group() -> None:
    """Task 159 (B2): with no clock column ``predict`` wrote the stream's count
    of rows fed as ``scored_clock`` on every row, where the docs say the row's
    index in its group, which is what ``fit_predict`` writes. It counts on from
    the rows learned, as ``fit_predict`` would."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(6), "y": rng.standard_normal(6)})
    spec = po.spec.ewridge("m", targets=["y"], features=["x"], half_life=10.0, emit_clocks=True)
    bank = po.ModelBank([spec])
    fitted = bank.fit_predict(df)
    assert fitted["m"].struct.field("scored_clock").to_list() == [0, 1, 2, 3, 4, 5]
    scored = bank.predict(df)
    assert scored["m"].struct.field("scored_clock").to_list() == [6, 7, 8, 9, 10, 11]
    assert scored["m"].struct.field("learned_clock").to_list() == [5] * 6
