"""A temporal clock and its durations (docs/PLAN.md task 88).

A clock column that is a ``Datetime``, ``Date`` or ``Duration`` carries its
own unit, so the parameters measured against it are durations:
``halflife=pl.duration(minutes=10)``, ``timedelta(minutes=10)`` or ``"10m"``.
The bank reads such a clock in seconds since the Unix epoch, and every
duration in seconds, so the column's own unit never reaches the fit. A
numeric clock keeps plain numbers of its own units. Either mixture is
refused, naming the column, the parameter and the fix.

This is T-E10 turned from a refusal into a pass: the same wall-clock data
as ``Datetime(ms)``, ``Datetime(us)`` and ``Datetime(ns)`` give the same
numbers, to the bit, as a float clock in seconds with the same parameters
as plain numbers.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online
from polars_online._polars_online import spec_clock_fields
from polars_online._spec import _takes_duration
from test_model_registry import MINIMAL

# 2024-01-02 09:30:00 UTC, in epoch seconds.
START = 1_704_187_800


def _frame(n: int = 400, seed: int = 3) -> pl.DataFrame:
    """Irregular whole-second ticks, with a few gaps past any cap below, a
    session change halfway, a linear target and a label for ``ew_class``."""
    rng = np.random.default_rng(seed)
    gaps = rng.integers(1, 120, n)
    gaps[rng.random(n) < 0.03] = 7_200
    secs = START + np.cumsum(gaps)
    x = rng.normal(size=(n, 2))
    y = 0.5 + x @ np.array([1.0, -2.0]) + rng.normal(0.0, 0.3, n)
    return pl.DataFrame(
        {
            "t_s": secs.astype(np.float64),
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": y,
            "s": np.where(np.arange(n) < n // 2, "am", "pm"),
            "lab": np.where(y > 0.5, "a", "b"),
        }
    )


def _temporal(df: pl.DataFrame, unit: str = "us") -> pl.DataFrame:
    """The same instants as a ``Datetime`` column ``t``."""
    return df.with_columns(
        t=pl.from_epoch(pl.col("t_s").cast(pl.Int64), time_unit="s").cast(pl.Datetime(unit))
    )


def _ridge(clock: str, **kw) -> dict:
    # The noise gate is off: a window emptied by a long gap would withhold
    # every prediction after it and warn, which is not what these test.
    return po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        clock=clock,
        ridge=1e-6,
        max_error_inflation=float("inf"),
        **kw,
    )


#: One spec's clock parameters as plain seconds, and the same as durations,
#: in each of the three forms a duration can be written in.
NUMBERS = dict(
    halflife=600.0,
    max_dclock=1_800.0,
    label_delay=60.0,
    min_backwards_jump=30.0,
    session_gap=600.0,
    window=3_600.0,
    solve_every=120.0,
)
DURATIONS = {
    "timedelta": dict(
        halflife=timedelta(minutes=10),
        max_dclock=timedelta(minutes=30),
        label_delay=timedelta(minutes=1),
        min_backwards_jump=timedelta(seconds=30),
        session_gap=timedelta(minutes=10),
        window=timedelta(hours=1),
        solve_every=timedelta(minutes=2),
    ),
    "pl.duration": dict(
        halflife=pl.duration(minutes=10),
        max_dclock=pl.duration(minutes=30),
        label_delay=pl.duration(minutes=1),
        min_backwards_jump=pl.duration(seconds=30),
        session_gap=pl.duration(minutes=10),
        window=pl.duration(hours=1),
        solve_every=pl.duration(minutes=2),
    ),
    "text": dict(
        halflife="10m",
        max_dclock="30m",
        label_delay="1m",
        min_backwards_jump="30s",
        session_gap="10m",
        window="1h",
        solve_every="2m",
    ),
}


def _fit(df: pl.DataFrame, spec: dict) -> pl.Series:
    return po.ModelBank([spec]).fit_predict(df)["m"]


class TestTheUnitNeverReachesTheFit:
    @pytest.mark.parametrize("unit", ["ms", "us", "ns"])
    @pytest.mark.parametrize("form", sorted(DURATIONS))
    def test_a_datetime_clock_gives_the_numbers_of_its_seconds(self, unit, form):
        df = _frame()
        ref = _fit(df, _ridge("t_s", session="s", **NUMBERS))
        got = _fit(_temporal(df, unit), _ridge("t", session="s", **DURATIONS[form]))
        assert got.equals(ref, null_equal=True), (unit, form)
        # The parameters acted: decay, the cap and the window all bound
        # n_eff well below the row count, and predictions were made.
        n_eff = ref.struct.field("n_eff")
        assert n_eff.max() < 50 and ref.struct.field("pred_y").drop_nulls().len() > 300

    def test_a_date_clock_reads_day_durations(self):
        """Weekdays only, so the clock has two-day and three-day gaps; a
        numeric clock in days with the same parameters is the reference."""
        days = [d for d in pl.date_range(date(2024, 1, 1), date(2024, 12, 31), eager=True)]
        days = [d for d in days if d.weekday() < 5]
        rng = np.random.default_rng(5)
        x = rng.normal(size=len(days))
        df = pl.DataFrame({"d": days, "x0": x, "y": 2.0 * x + rng.normal(0.0, 0.1, len(days))})
        df = df.with_columns(n=pl.col("d").cast(pl.Int32).cast(pl.Float64))
        spec = dict(targets=["y"], features=["x0"], ridge=1e-6, max_error_inflation=float("inf"))
        ref = _fit(df, po.spec.ewridge("m", clock="n", halflife=5.0, max_dclock=10.0, **spec))
        got = _fit(df, po.spec.ewridge("m", clock="d", halflife="5d", max_dclock="10d", **spec))
        assert got.equals(ref, null_equal=True)
        assert ref.struct.field("n_eff").max() < 10

    def test_a_duration_column_is_a_clock_too(self):
        df = _temporal(_frame()).with_columns(elapsed=pl.col("t") - pl.col("t").first())
        want = _fit(df.with_columns(e=pl.col("t_s") - START), _ridge("e", session="s", **NUMBERS))
        got = _fit(df, _ridge("elapsed", session="s", **DURATIONS["text"]))
        assert got.equals(want, null_equal=True)

    def test_summer_time_neither_stretches_nor_folds_the_clock(self):
        """Hourly UTC instants shown in New York time across the spring
        change: the wall clock jumps two hours at 02:00, the instants one.
        A zone-aware column is read as its UTC instants."""
        utc = pl.datetime_range(
            datetime(2024, 3, 9, 20), datetime(2024, 3, 11, 4), "1h", time_zone="UTC", eager=True
        )
        rng = np.random.default_rng(9)
        x = rng.normal(size=len(utc))
        base = pl.DataFrame({"x0": x, "y": x + rng.normal(0.0, 0.1, len(utc))})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            halflife="3h",
            max_dclock="12h",
            max_error_inflation=float("inf"),
        )
        zoned = _fit(base.with_columns(t=utc.dt.convert_time_zone("America/New_York")), spec)
        naive_utc = _fit(base.with_columns(t=utc.dt.replace_time_zone(None)), spec)
        wall = _fit(
            base.with_columns(
                t=utc.dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
            ),
            spec,
        )
        assert zoned.equals(naive_utc, null_equal=True)
        # The wall clock does jump: read naively, it would give other numbers.
        assert not zoned.equals(wall, null_equal=True)

    def test_the_mean_is_pandas_with_times_on_a_datetime_clock(self):
        """T-S9's irregular-clock oracle, on a nanosecond Datetime clock with a
        ``timedelta`` halflife, which is the form pandas takes too. The bank
        reads the clock as seconds from its first instant, so it agrees with
        the EW mean recursed on the exact nanosecond gaps to 1e-12 (measured
        2e-16). pandas is the looser side: its own arithmetic on ``times``
        sits 1.8e-9 from that recursion, which is the 5e-9 below."""
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(8)
        n = 300
        ns = (np.cumsum(rng.uniform(0.2, 3.0, n)) * 1e9).astype(np.int64) + START * 10**9
        x = rng.normal(0.0, 1.0, n)
        t = pl.Series("t", ns).cast(pl.Datetime("ns"))
        spec = po.spec.ew_cov(
            "m",
            features=["x0"],
            halflife=timedelta(seconds=25),
            clock="t",
            max_dclock=float("inf"),
            stats=["mean"],
            min_periods=0.0,
        )
        out = po.ModelBank([spec]).fit_predict(pl.DataFrame({"t": t, "x0": x}))
        got = out["m"].struct.field("mean_x0").to_numpy().astype(float)
        # The recursion on the exact gaps: the mean before each row.
        exact, w, s = [], 0.0, 0.0
        for i in range(n):
            exact.append(s / w if w > 0 else np.nan)
            lam = 2.0 ** (-(float(ns[i] - ns[i - 1]) if i else 0.0) / 25e9)
            w, s = w * lam + 1.0, s * lam + x[i]
        exact = np.array(exact)
        live = np.isfinite(got) & np.isfinite(exact)
        assert live.sum() > n - 10
        assert np.max(np.abs(got[live] - exact[live])) <= 1e-12
        times = pd.to_datetime(ns, unit="ns")
        ref = pd.Series(x).ewm(halflife=pd.Timedelta(seconds=25), times=times).mean()
        ref = ref.to_numpy()[:-1]
        assert np.max(np.abs(got[1:][live[1:]] - ref[live[1:]])) <= 5e-9


class TestHowADurationIsWritten:
    def test_three_forms_are_one_text(self):
        for form in (timedelta(minutes=90), pl.duration(hours=1, minutes=30), "1h30m"):
            assert _ridge("t", halflife=form, max_dclock="inf")["halflife"] == "1h30m"
        # Text is kept as written; it means the same thing.
        assert _ridge("t", halflife="90m", max_dclock="inf")["halflife"] == "90m"

    def test_a_grid_is_named_by_its_durations(self):
        spec = _ridge("t", halflife=["5m", timedelta(hours=1)], max_dclock="inf")
        assert spec["halflife"] == ["5m", "1h"]
        fields = po.spec.output_fields(spec)
        assert any("@h5m" in f for f in fields) and any("@h1h" in f for f in fields), fields

    def test_zero_and_infinity_mean_the_same_in_every_unit(self):
        df = _temporal(_frame())
        free = _fit(df, _ridge("t", halflife=float("inf"), max_dclock=float("inf")))
        # n_eff is the weight before the row's own update (hard rule 8).
        assert free.struct.field("n_eff")[-1] == pytest.approx(df.height - 1)

    @pytest.mark.parametrize(
        ("value", "exc", "says"),
        [
            ("10", ValueError, 'halflife "10" is not a duration: 10 has no unit'),
            ("1mo", ValueError, "a month has no fixed length"),
            ("1.5h", ValueError, "1h30m"),
            ("3i", ValueError, "counts rows"),
            (pl.col("x0"), TypeError, "must be a duration that reads no column"),
            (pl.lit(5), TypeError, "must be one duration"),
        ],
    )
    def test_what_is_not_a_duration_is_refused_by_name(self, value, exc, says):
        with pytest.raises(exc, match='spec "m": halflife') as e:
            _ridge("t", halflife=value, max_dclock="inf")
        assert says in str(e.value), str(e.value)


class TestEachMixtureIsRefused:
    def test_a_temporal_clock_refuses_a_plain_number(self):
        with pytest.raises(ValueError) as e:
            _fit(_temporal(_frame()), _ridge("t", halflife=600.0, max_dclock=1_800.0))
        msg = str(e.value)
        for part in (
            '"t"',
            "temporal clock",
            "halflife is a plain number",
            "pl.duration(",
            "dt.epoch",
        ):
            assert part in msg, (part, msg)

    def test_a_numeric_clock_refuses_a_duration(self):
        with pytest.raises(ValueError) as e:
            _fit(_frame(), _ridge("t_s", halflife="10m", max_dclock="30m"))
        msg = str(e.value)
        for part in ('"t_s"', "halflife is a duration", "Datetime", "from_epoch"):
            assert part in msg, (part, msg)

    def test_one_spec_cannot_mix_the_two(self):
        with pytest.raises(
            ValueError, match="halflife is a duration but max_dclock is a plain number"
        ):
            _ridge("t", halflife="10m", max_dclock=1_800.0)

    def test_a_rate_per_clock_unit_has_no_duration_form(self):
        with pytest.raises(ValueError, match="lam is a decay per clock unit"):
            po.spec.ewridge(
                "m", targets=["y"], features=["x0"], clock="t", lam=0.99, max_dclock="30m"
            )
        with pytest.raises(ValueError, match="q is a variance per clock unit"):
            po.spec.kalman(
                "m",
                targets=["y"],
                features=["x0"],
                clock="t",
                halflife="10m",
                max_dclock="30m",
                coef_halflife="1h",
                q=[0.0, 0.1],
            )
        # And on a temporal clock without any duration, lam is the number refused.
        with pytest.raises(ValueError, match="lam is a rate per clock unit") as e:
            _fit(
                _temporal(_frame()),
                po.spec.ewridge(
                    "m", targets=["y"], features=["x0"], clock="t", lam=0.99, max_dclock="inf"
                ),
            )
        assert "give halflife as a duration" in str(e.value)

    def test_a_duration_needs_a_clock(self):
        with pytest.raises(ValueError, match="needs a clock column"):
            po.spec.ewridge("m", targets=["y"], features=["x0"], halflife="10m")

    def test_a_time_of_day_is_no_clock(self):
        df = _temporal(_frame()).with_columns(tod=pl.col("t").dt.time())
        with pytest.raises(ValueError, match="time of day"):
            _fit(df, _ridge("tod", halflife="10m", max_dclock="30m"))

    def test_a_temporal_clock_is_not_also_a_feature(self):
        df = _temporal(_frame())
        other = po.spec.ewridge("f", targets=["y"], features=["t"], halflife=50.0)
        with pytest.raises(ValueError, match="can only be a clock"):
            po.ModelBank([_ridge("t", halflife="10m", max_dclock="30m"), other]).fit_predict(df)


class TestADurationSurvives:
    def test_save_load_and_the_specs_it_was_built_from(self, tmp_path):
        df = _temporal(_frame())
        spec = _ridge("t", session="s", **DURATIONS["pl.duration"])
        straight = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        first = bank.fit_predict(df.slice(0, 150))
        bank.save(tmp_path / "b.state")
        loaded = po.ModelBank.load(tmp_path / "b.state")
        assert loaded.specs == [spec]
        rest = loaded.fit_predict(df.slice(150))
        assert (
            pl.concat([first, rest])["m"]
            .struct.field("pred_y")
            .equals(straight["m"].struct.field("pred_y"), null_equal=True)
        )

    def test_the_command_line_reads_durations_from_its_config(self, tmp_path, online_cli):
        df = _temporal(_frame(), "ns").drop("t_s", "lab")
        df.write_parquet(tmp_path / "in.parquet")
        spec = _ridge("t", session="s", **DURATIONS["text"])
        run_online(
            online_cli,
            tmp_path,
            [spec],
            input=tmp_path / "in.parquet",
            output=tmp_path / "out.parquet",
        )
        got = pl.read_parquet(tmp_path / "out.parquet")["m"]
        assert got.equals(_fit(df, spec), null_equal=True)


def _tables() -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    fields, rates = spec_clock_fields()
    return dict(fields), [(o, f) for o, fs in rates for f in fs]


def _kind(builder: str) -> str:
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "halflife": 50.0}
    kw.update(MINIMAL[builder])
    spec = getattr(po.spec, builder)("m", **{k: v for k, v in kw.items() if v is not None})
    return spec["model"]["type"]


class TestEveryClockParameterTakesADuration:
    """The completeness test: every parameter measured in clock units takes a
    duration, so one added later cannot be missed. The table comes from the
    Rust side (``CLOCK_FIELDS``), which its own test holds to the types; here
    it is held to the builders' annotations and docstrings, and each entry
    is fitted, both ways, on a temporal clock."""

    def test_the_table_is_the_builders_annotations(self):
        fields, rates = _tables()
        rate_names = {f for _, f in rates}
        import typing

        from polars_online import _spec

        shared = {k for k, v in typing.get_type_hints(_spec._common).items() if _takes_duration(v)}
        assert shared == set(fields["*"])
        found = 0
        for builder in MINIMAL:
            fn = getattr(po.spec, builder)
            hints = typing.get_type_hints(fn.__wrapped__)
            own = {k for k, v in hints.items() if _takes_duration(v)} - shared
            assert own == set(fields.get(_kind(builder), [])), builder
            # The docstrings are the second opinion: a parameter documented
            # in clock units, or named as a halflife, takes a duration or
            # is a rate.
            doc = fn.__doc__ or ""
            for name in hints:
                # A parameter's own entry: from its header to the next one.
                head = f"\n    ``{name}``"
                entry = doc.split(head, 1)[1].split("\n    ``", 1)[0] if head in doc else ""
                # "counted in learned rows, not clock units" is not one.
                clocky = name.endswith("halflife") or bool(
                    re.search(r"(?<!not )(?<!not in )\bclock units?\b", entry)
                )
                if clocky and name not in rate_names:
                    assert name in own | shared, f"{builder}.{name} is in clock units"
                    found += 1
        # 12 of the 15 model parameters say so in their own entry; the
        # other three, `solve_every` in lasso, huber and quantile, defer to
        # ewridge's.
        assert found >= 12, f"the docstrings named only {found} clock parameters"

    def _base(self, builder: str) -> dict:
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "halflife": "50s"}
        kw.update(MINIMAL[builder])
        if builder == "kalman":
            kw["coef_halflife"] = "1m"
        if builder == "ew_class":
            kw["label"] = "lab"
        kw.update(clock="t", max_dclock="1h")
        return kw

    #: A duration each parameter can take on the frame above, and whatever
    #: else the parameter needs beside it.
    VALUES = {
        "halflife": ("5m", {}),
        "max_dclock": ("30m", {}),
        "min_backwards_jump": ("10s", {}),
        "session_gap": ("10m", {"session": "s"}),
        "label_delay": ("30s", {}),
        "long_halflife": ("1h", {"session": "s", "session_gap": "10m", "session_shrink": 0.5}),
        "solve_every": ("10s", {}),
        "window": ("30m", {}),
        "select_halflife": ("1h", {}),
        "coef_halflife": ("2m", {}),
        "revert_halflife": ("1h", {}),
        "level_halflife": ("5m", {"halflife": None}),
        "trend_halflife": ("20m", {}),
    }

    def _cases(self):
        fields, _ = _tables()
        kinds = {builder: _kind(builder) for builder in MINIMAL}
        for owner, names in fields.items():
            builders = ["ewridge"] if owner == "*" else [b for b, k in kinds.items() if k == owner]
            assert builders, owner
            for builder in builders:
                for name in names:
                    yield builder, name

    #: How each clock parameter is made unit-free when another is the one
    #: under test: ``inf`` where that means something, left out otherwise.
    UNIT_FREE = {
        "halflife": float("inf"),
        "max_dclock": float("inf"),
        "coef_halflife": float("inf"),
        "level_halflife": float("inf"),
        "trend_halflife": float("inf"),
        "long_halflife": float("inf"),
        "select_halflife": float("inf"),
        "revert_halflife": float("inf"),
        "session_gap": float("inf"),
        "window": None,
        "solve_every": None,
        "label_delay": None,
        "min_backwards_jump": None,
    }

    # The noise gate's notice is about the tiny frame, not about durations.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_each_takes_a_duration_and_refuses_a_number_on_a_temporal_clock(self):
        df = _temporal(_frame())
        seen = 0
        for builder, name in self._cases():
            assert name in self.VALUES, f"{builder}.{name} needs a value in VALUES"
            value, extra = self.VALUES[name]
            kw = {**self._base(builder), **extra, name: value}
            fn = getattr(po.spec, builder)
            spec = fn("m", **{k: v for k, v in kw.items() if v is not None})
            po.ModelBank([spec]).fit_predict(df)
            # The same parameter as a plain number, with every other clock
            # parameter unit-free, is refused by name at the clock.
            plain = {
                k: (self.UNIT_FREE[k] if k in self.UNIT_FREE and isinstance(v, str) else v)
                for k, v in kw.items()
                if k != name
            }
            plain = {k: v for k, v in plain.items() if v is not None}
            plain[name] = 600.0
            with pytest.raises(ValueError, match=f"{name} is a plain number"):
                po.ModelBank([fn("m", **plain)]).fit_predict(df)
            seen += 1
        assert seen >= 20, seen


class TestADurationTheDataCannotHold:
    """A duration can be well formed and still mean nothing on the clock it
    is given: finer than the column's own step, where it cannot act."""

    def _daily(self) -> pl.DataFrame:
        days = pl.date_range(date(2024, 1, 1), date(2024, 3, 31), eager=True)
        rng = np.random.default_rng(2)
        x = rng.normal(size=len(days))
        return pl.DataFrame({"d": days, "x0": x, "y": x + rng.normal(0.0, 0.1, len(days))})

    def _daily_spec(self, **kw) -> dict:
        return po.spec.ewridge("m", targets=["y"], features=["x0"], clock="d", halflife="5d", **kw)

    def test_a_cap_finer_than_a_day_is_refused_on_a_date_clock(self):
        with pytest.raises(ValueError, match="max_dclock is 12h, less than a day") as e:
            _fit(self._daily(), self._daily_spec(max_dclock="12h"))
        assert "clock would count rows" in str(e.value)
        _fit(self._daily(), self._daily_spec(max_dclock="1d"))

    def test_a_threshold_no_jump_can_undercut_is_refused(self):
        with pytest.raises(ValueError, match="min_backwards_jump is 12h") as e:
            _fit(self._daily(), self._daily_spec(max_dclock="3d", min_backwards_jump="12h"))
        assert "never fire" in str(e.value)

    # A 500 us halflife forgets every row at once, which the readiness notice
    # says; that it is allowed is the point here.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_the_step_is_the_column_unit_for_a_datetime(self):
        df = _temporal(_frame(), "ms")
        with pytest.raises(ValueError, match="less than a millisecond"):
            _fit(df, _ridge("t", halflife="10m", max_dclock="500us"))
        # A halflife finer than the step is not refused: it forgets fast, and means it.
        _fit(df, _ridge("t", halflife="500us", max_dclock="30m"))


class TestTheHelpersMeasureTheClockTheSameWay:
    def test_embargo_takes_a_duration_on_a_temporal_clock(self):
        df = _frame()
        spec = dict(weight="_online_role_weight")
        numbers = po.prep.embargo(df.lazy(), clock="t_s", delay=300.0).collect()
        numbers = _fit(numbers, _ridge("t_s", halflife=600.0, max_dclock=1_800.0, **spec))
        for form in ("5m", timedelta(minutes=5), pl.duration(minutes=5)):
            doubled = po.prep.embargo(_temporal(df).lazy(), clock="t", delay=form).collect()
            assert doubled["t"].dtype == pl.Datetime("us")
            got = _fit(doubled, _ridge("t", halflife="10m", max_dclock="30m", **spec))
            assert got.equals(numbers, null_equal=True), form

    @pytest.mark.parametrize(
        ("frame", "clock", "delay", "says"),
        [
            ("temporal", "t", 300.0, "must be a duration"),
            ("numeric", "t_s", "5m", "has no unit to measure it"),
            ("temporal_ms", "t", "1500us", "not a whole number of steps"),
            ("date", "d", "12h", "not a whole number of steps"),
        ],
    )
    def test_embargo_refuses_a_delay_the_clock_cannot_take(self, frame, clock, delay, says):
        df = _frame()
        frames = {
            "temporal": _temporal(df),
            "numeric": df,
            "temporal_ms": _temporal(df, "ms"),
            "date": _temporal(df).with_columns(d=pl.col("t").dt.date()),
        }
        with pytest.raises(ValueError, match=says):
            po.prep.embargo(frames[frame].lazy(), clock=clock, delay=delay)

    def test_rolling_metrics_buckets_a_temporal_clock_by_a_duration(self):
        df = _frame()
        numbers = df.hstack(_fit(df, _ridge("t_s", **NUMBERS | {"session_gap": None})).to_frame())
        durations = _temporal(df)
        durations = durations.hstack(
            _fit(durations, _ridge("t", **DURATIONS["text"] | {"session_gap": None})).to_frame()
        )
        want = po.eval.rolling_metrics(numbers, "m", clock="t_s", window=3_600.0, min_obs=5)
        got = po.eval.rolling_metrics(durations, "m", clock="t", window="1h", min_obs=5)
        assert got["window_start"].dtype == pl.Datetime("us")
        assert want.height > 3
        seconds = got["window_start"].dt.epoch("s").cast(pl.Float64)
        assert seconds.equals(want["window_start"], check_names=False)
        assert got.drop("window_start").equals(want.drop("window_start"))


class TestNanosecondsAreKept:
    """A temporal clock is read as seconds from its first instant, kept in
    the bank's state, so the double it becomes resolves about a nanosecond
    over a year of stream. Read as seconds since 1970 it would resolve a
    quarter of a microsecond at 2024's 1.7e9 seconds, and ticks closer than
    that would read as simultaneous."""

    def _ticks(self, n: int = 300, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        gaps = rng.integers(1, 5_000, n)  # nanoseconds apart
        ns = START * 10**9 + np.cumsum(gaps)
        return ns, gaps

    def _n_eff(self, gaps: np.ndarray, halflife_ns: float) -> np.ndarray:
        """The exact recursion: weights of 1, decayed by the exact gap."""
        out, w = [], 0.0
        for g in gaps:
            out.append(w)
            w = w * 2.0 ** (-float(g) / halflife_ns) + 1.0
        return np.array(out)

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_ticks_nanoseconds_apart_decay_by_their_exact_gaps(self):
        ns, _ = self._ticks()
        # n_eff at a row is the weight before that row's own decay, which
        # is by the gap from the row before it: no gap ages row 0.
        want = self._n_eff(np.concatenate([[0], np.diff(ns)]), 1_000.0)
        rng = np.random.default_rng(1)
        x = rng.normal(size=len(ns))
        df = pl.DataFrame({"t": pl.Series(ns).cast(pl.Datetime("ns")), "x0": x, "y": x})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            halflife="1us",
            max_dclock="inf",
            max_error_inflation=float("inf"),
        )
        got = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("n_eff").to_numpy()
        assert np.max(np.abs(got - want)) < 1e-12 * np.max(want)
        # The same instants as a float clock in epoch nanoseconds cannot
        # say this: a double at 1.7e18 resolves 256 ns, so the gaps are
        # rounded and the decay drifts from the exact recursion.
        as_float = df.with_columns(t=pl.col("t").cast(pl.Int64).cast(pl.Float64))
        loose = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            halflife=1_000.0,
            max_dclock=float("inf"),
            max_error_inflation=float("inf"),
        )
        drift = po.ModelBank([loose]).fit_predict(as_float)["m"].struct.field("n_eff").to_numpy()
        assert np.max(np.abs(drift - want)) > 1e-3 * np.max(want)

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @pytest.mark.parametrize(
        "age_s", [86_400, 31_557_600, 315_576_000], ids=["a day", "a year", "a decade"]
    )
    def test_the_rounding_is_the_clocks_resolution_at_the_streams_age(self, age_s):
        """The double that holds seconds from the origin resolves one part in
        2**52 of the time since it, so nanosecond ticks late in a stream are
        read to that resolution, and the error a model sees is at most that
        over the halflife. Measured under a 1 us halflife: 3.9e-6, 9.6e-4
        and 1.1e-2 against bounds of 1.5e-5, 3.7e-3 and 6e-2."""
        ns, _ = self._ticks()
        ns = np.concatenate([[ns[0]], ns + age_s * 10**9])
        rng = np.random.default_rng(3)
        x = rng.normal(size=len(ns))
        df = pl.DataFrame({"t": pl.Series(ns).cast(pl.Datetime("ns")), "x0": x, "y": x})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            halflife="1us",
            max_dclock="inf",
            max_error_inflation=float("inf"),
        )
        got = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("n_eff").to_numpy()
        want = self._n_eff(np.concatenate([[0], np.diff(ns)]), 1_000.0)
        err = np.max(np.abs(got - want)) / np.max(want)
        assert err <= np.spacing(float(age_s)) / 1e-6

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_fed_in_chunks_the_origin_is_the_first_chunks_first_instant(self):
        """Hard rule 3 on a temporal clock: only the first accepted chunk
        sets the origin, and every later chunk is read from it, so the
        nanosecond gaps across a chunk boundary are the one-chunk run's."""
        ns, _ = self._ticks()
        rng = np.random.default_rng(2)
        x = rng.normal(size=len(ns))
        df = pl.DataFrame({"t": pl.Series(ns).cast(pl.Datetime("ns")), "x0": x, "y": x})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            halflife="1us",
            max_dclock="inf",
            max_error_inflation=float("inf"),
        )
        whole = po.ModelBank([spec]).fit_predict(df)["m"]
        bank = po.ModelBank([spec])
        chunked = pl.concat([bank.fit_predict(df.slice(i, 37))["m"] for i in range(0, len(df), 37)])
        for field in ("pred_y", "n_eff"):
            assert whole.struct.field(field).equals(chunked.struct.field(field), null_equal=True)

    def test_a_refused_chunk_leaves_no_origin_behind(self):
        df = _temporal(_frame())
        # A minute apart throughout, so a row fed late is a jump back of
        # less than max_dclock, which the disorder check refuses.
        df = df.with_columns(
            t_s=(START + 60.0 * pl.int_range(pl.len())).cast(pl.Float64)
        ).with_columns(t=pl.from_epoch(pl.col("t_s").cast(pl.Int64), time_unit="s"))
        spec = _ridge("t", halflife="10m", max_dclock="30m")
        untouched = po.ModelBank([spec]).save_bytes()
        bank = po.ModelBank([spec])
        late = pl.concat([df.slice(0, 5), df.slice(6, 5), df.slice(5, 1)])
        with pytest.raises(ValueError, match="backwards"):
            bank.fit_predict(late)
        assert bank.save_bytes() == untouched
        # Accepted, the first row's instant is the origin, and the clock
        # range comes back as seconds since 1970 all the same.
        bank.fit_predict(df.slice(0, 50))
        first = df["t_s"][0]
        assert bank.summary()["clock_min"][0] == pytest.approx(first, abs=1e-6)
        assert bank.summary()["clock_max"][0] == pytest.approx(df["t_s"][49], abs=1e-6)
        assert bank.groups()["last_clock"][0] == pytest.approx(df["t_s"][49], abs=1e-6)

    def test_scoring_first_then_learning_changes_nothing(self):
        df = _temporal(_frame())
        spec = _ridge("t", session="s", **DURATIONS["text"])
        scored_first = po.ModelBank([spec])
        assert scored_first.predict(df.slice(0, 20))["m"].struct.field("pred_y").is_null().all()
        a = scored_first.fit_predict(df)
        b = po.ModelBank([spec]).fit_predict(df)
        assert a["m"].equals(b["m"], null_equal=True)

    def test_the_query_form_agrees_with_the_bank(self):
        df = _temporal(_frame())
        spec = _ridge("t", session="s", **DURATIONS["pl.duration"])
        plan = df.lazy().online.fit_predict([spec]).collect()
        assert plan["m"].equals(po.ModelBank([spec]).fit_predict(df)["m"], null_equal=True)

    def test_the_origin_survives_a_save_and_load(self, tmp_path):
        """The origin is in the state: loaded, a bank keeps measuring from the
        same instant, which the clock range reports as seconds since 1970."""
        df = _temporal(_frame())
        bank = po.ModelBank([_ridge("t", halflife="10m", max_dclock="30m")])
        bank.fit_predict(df.slice(0, 100))
        bank.save(tmp_path / "o.state")
        loaded = po.ModelBank.load(tmp_path / "o.state")
        loaded.fit_predict(df.slice(100))
        assert loaded.summary()["clock_min"][0] == pytest.approx(df["t_s"][0], abs=1e-6)
        assert loaded.summary()["clock_max"][0] == pytest.approx(df["t_s"][-1], abs=1e-6)


class TestTheClockColumnInOtherRoles:
    def test_a_temporal_column_may_be_the_clock_and_the_session(self):
        df = _temporal(_frame()).with_columns(day=pl.col("t").dt.date())
        df = df.with_columns(day_s=pl.col("day").cast(pl.Int32).cast(pl.Float64))
        want = _fit(df, _ridge("t_s", session="day_s", **NUMBERS))
        got = _fit(df, _ridge("t", session="day", **DURATIONS["text"]))
        assert got.equals(want, null_equal=True)

    def test_embargo_on_a_date_clock_moves_whole_days(self):
        days = pl.date_range(date(2024, 1, 1), date(2024, 1, 20), eager=True)
        rng = np.random.default_rng(6)
        x = rng.normal(size=len(days))
        df = pl.DataFrame({"d": days, "x0": x, "y": 2.0 * x + rng.normal(0.0, 0.1, len(days))})
        doubled = po.prep.embargo(df.lazy(), clock="d", delay="2d").collect()
        assert doubled["d"].dtype == pl.Date
        learn = doubled.filter(pl.col("_online_role") == "learn")
        assert (learn["d"] - df["d"]).unique().to_list() == [timedelta(days=2)]

    def test_rolling_metrics_buckets_a_duration_clock(self):
        df = _temporal(_frame()).with_columns(elapsed=pl.col("t") - pl.col("t").first())
        spec = _ridge("elapsed", halflife="10m", max_dclock="30m")
        out = df.hstack(_fit(df, spec).to_frame())
        got = po.eval.rolling_metrics(out, "m", clock="elapsed", window="1h", min_obs=5)
        assert got["window_start"].dtype == pl.Duration("us")
        assert got.height > 3
        starts = got["window_start"].dt.total_seconds().to_list()
        assert all(s % 3_600 == 0 for s in starts)

    def test_padding_and_spaces_in_duration_text(self):
        spec = _ridge("t", halflife=[" 5m ", "1h"], max_dclock="inf")
        assert spec["halflife"] == ["5m", "1h"]
        assert any("@h5m" in f for f in po.spec.output_fields(spec))
        with pytest.raises(ValueError, match="has a space in it"):
            _ridge("t", halflife="1h 30m", max_dclock="inf")
