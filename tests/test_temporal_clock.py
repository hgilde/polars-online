"""A temporal clock and its durations (docs/PLAN.md task 88).

A clock column that is a ``Datetime``, ``Date`` or ``Duration`` carries its
own unit, so the parameters measured against it are durations:
``half_life=pl.duration(minutes=10)``, ``timedelta(minutes=10)`` or ``"10m"``.
The bank reads such a clock in its own integer nanoseconds, takes the gap
between two rows in integers, and reads every duration in seconds, so the
column's own unit never reaches the fit and a nanosecond timestamp keeps its
nanoseconds whatever the stream's age. A
numeric clock keeps plain numbers of its own units. Either mixture is
refused, naming the column, the parameter and the fix.

This is T-E10 turned from a refusal into a pass: the same wall-clock data
as ``Datetime(ms)``, ``Datetime(us)`` and ``Datetime(ns)`` give the same
numbers, to the bit, as a float clock in seconds with the same parameters
as plain numbers.
"""

from __future__ import annotations

import inspect
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
#: in each of the three forms a duration can be written in. A minimum for a
#: step back is `"reset_state"`'s alone (task 120).
NUMBERS = dict(
    half_life=600.0,
    gap_cap=1_800.0,
    embargo=60.0,
    restart_after_step_back=30.0,
    session_gap=600.0,
    window_size=3_600.0,
    solve_every=120.0,
)
DURATIONS = {
    "timedelta": dict(
        half_life=timedelta(minutes=10),
        gap_cap=timedelta(minutes=30),
        embargo=timedelta(minutes=1),
        restart_after_step_back=timedelta(seconds=30),
        session_gap=timedelta(minutes=10),
        window_size=timedelta(hours=1),
        solve_every=timedelta(minutes=2),
    ),
    "pl.duration": dict(
        half_life=pl.duration(minutes=10),
        gap_cap=pl.duration(minutes=30),
        embargo=pl.duration(minutes=1),
        restart_after_step_back=pl.duration(seconds=30),
        session_gap=pl.duration(minutes=10),
        window_size=pl.duration(hours=1),
        solve_every=pl.duration(minutes=2),
    ),
    "text": dict(
        half_life="10m",
        gap_cap="30m",
        embargo="1m",
        restart_after_step_back="30s",
        session_gap="10m",
        window_size="1h",
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
        # weight_sum well below the row count, and predictions were made.
        weight_sum = ref.struct.field("weight_sum")
        assert weight_sum.max() < 50 and ref.struct.field("pred_y").drop_nulls().len() > 300

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
        ref = _fit(df, po.spec.ewridge("m", clock="n", half_life=5.0, gap_cap=10.0, **spec))
        got = _fit(df, po.spec.ewridge("m", clock="d", half_life="5d", gap_cap="10d", **spec))
        assert got.equals(ref, null_equal=True)
        assert ref.struct.field("weight_sum").max() < 10

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
            half_life="3h",
            gap_cap="12h",
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
        ``timedelta`` half-life, which is the form pandas takes too. The bank
        takes the gaps in integer nanoseconds, so it agrees with the EW mean
        recursed on the exact nanosecond gaps to 1e-12 (measured 2e-16).
        pandas is the looser side: its own arithmetic on ``times`` sits
        1.8e-9 from that recursion, which is the 5e-9 below."""
        import pandas as pd

        rng = np.random.default_rng(8)
        n = 300
        ns = (np.cumsum(rng.uniform(0.2, 3.0, n)) * 1e9).astype(np.int64) + START * 10**9
        x = rng.normal(0.0, 1.0, n)
        t = pl.Series("t", ns).cast(pl.Datetime("ns"))
        spec = po.spec.ew_cov(
            "m",
            features=["x0"],
            half_life=timedelta(seconds=25),
            clock="t",
            # A cap no gap here reaches (they are under 3 s): pandas has none.
            gap_cap="1d",
            stats=["mean"],
            min_weight=0.0,
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
            assert _ridge("t", half_life=form, gap_cap="365d")["half_life"] == "1h30m"
        # Text is kept as written; it means the same thing.
        assert _ridge("t", half_life="90m", gap_cap="365d")["half_life"] == "90m"

    def test_a_grid_is_named_by_its_durations(self):
        spec = _ridge("t", half_life=["5m", timedelta(hours=1)], gap_cap="365d")
        assert spec["half_life"] == ["5m", "1h"]
        fields = po.spec.output_fields(spec)
        assert any("@h5m" in f for f in fields) and any("@h1h" in f for f in fields), fields

    def test_zero_and_infinity_mean_the_same_in_every_unit(self):
        df = _temporal(_frame())
        free = _fit(df, _ridge("t", half_life=float("inf"), gap_cap="365d"))
        # weight_sum is the weight before the row's own update (hard rule 8).
        assert free.struct.field("weight_sum")[-1] == pytest.approx(df.height - 1)

    @pytest.mark.parametrize(
        ("value", "exc", "says"),
        [
            ("10", ValueError, 'half_life "10" is not a duration: 10 has no unit'),
            ("1mo", ValueError, "a month has no fixed length"),
            ("1.5h", ValueError, "1h30m"),
            ("3i", ValueError, "counts rows"),
            (pl.col("x0"), TypeError, "must be a duration that reads no column"),
            (pl.lit(5), TypeError, "must be one duration"),
        ],
    )
    def test_what_is_not_a_duration_is_refused_by_name(self, value, exc, says):
        with pytest.raises(exc, match='spec "m": half_life') as e:
            _ridge("t", half_life=value, gap_cap="365d")
        assert says in str(e.value), str(e.value)

    @pytest.mark.parametrize("value", ["inf", "+INF", "nan", "reset", "10"])
    @pytest.mark.parametrize(
        ("call", "who"),
        [
            (
                lambda df, v: po.eval.rolling_metrics(df, "m", clock="t", window_size=v),
                "rolling_metrics: window_size",
            ),
            (lambda df, v: po.stream.embargo(df.lazy(), clock="t", delay=v), "embargo: delay"),
        ],
        ids=["rolling_metrics", "embargo"],
    )
    def test_what_is_not_a_duration_is_refused_by_name_by_a_helper(self, value, call, who):
        """Review 2026-10-05 (YA6): the words a spec's clock parameter takes
        in place of a number passed the text check, and the parse after it
        said '"inf" is not a duration' without naming the helper or the
        parameter. A helper's duration has no word form, and is named."""
        with pytest.raises(ValueError, match=f"^{who} ") as e:
            call(_temporal(_frame()), value)
        assert "is not a duration" in str(e.value), str(e.value)


class TestEachMixtureIsRefused:
    def test_a_temporal_clock_refuses_a_plain_number(self):
        with pytest.raises(ValueError) as e:
            _fit(_temporal(_frame()), _ridge("t", half_life=600.0, gap_cap=1_800.0))
        msg = str(e.value)
        for part in (
            '"t"',
            "temporal clock",
            "half_life is a plain number",
            "pl.duration(",
            "dt.epoch",
        ):
            assert part in msg, (part, msg)

    def test_a_numeric_clock_refuses_a_duration(self):
        with pytest.raises(ValueError) as e:
            _fit(_frame(), _ridge("t_s", half_life="10m", gap_cap="30m"))
        msg = str(e.value)
        for part in ('"t_s"', "half_life is a duration", "Datetime", "from_epoch"):
            assert part in msg, (part, msg)

    def test_one_spec_cannot_mix_the_two(self):
        with pytest.raises(
            ValueError, match="half_life is a duration but gap_cap is a plain number"
        ):
            _ridge("t", half_life="10m", gap_cap=1_800.0)

    def test_a_rate_per_clock_unit_has_no_duration_form(self):
        with pytest.raises(ValueError, match="lam is a decay per clock unit"):
            po.spec.ewridge("m", targets=["y"], features=["x0"], clock="t", lam=0.99, gap_cap="30m")
        with pytest.raises(
            ValueError, match="q is the noise a row one clock unit after the last adds"
        ):
            po.spec.kalman(
                "m",
                targets=["y"],
                features=["x0"],
                clock="t",
                half_life="10m",
                gap_cap="30m",
                q=[0.0, 0.1],
            )
        # And on a temporal clock without any duration, lam is the number refused.
        with pytest.raises(ValueError, match="lam is a number in the clock's own units") as e:
            _fit(
                _temporal(_frame()),
                po.spec.ewridge(
                    "m", targets=["y"], features=["x0"], clock="t", lam=0.99, gap_cap=300.0
                ),
            )
        assert "give half_life as a duration" in str(e.value)

    def test_a_duration_needs_a_clock(self):
        with pytest.raises(ValueError, match="needs a clock column"):
            po.spec.ewridge("m", targets=["y"], features=["x0"], half_life="10m")

    def test_a_time_of_day_is_no_clock(self):
        df = _temporal(_frame()).with_columns(tod=pl.col("t").dt.time())
        with pytest.raises(ValueError, match="time of day"):
            _fit(df, _ridge("tod", half_life="10m", gap_cap="30m"))

    def test_a_temporal_clock_is_not_also_a_feature(self):
        df = _temporal(_frame())
        other = po.spec.ewridge("f", targets=["y"], features=["t"], half_life=50.0)
        with pytest.raises(ValueError, match="can only be a clock"):
            po.ModelBank([_ridge("t", half_life="10m", gap_cap="30m"), other]).fit_predict(df)

    def test_a_temporal_clock_is_not_a_target_or_a_reference_either(self):
        """A table target's column, and the column it is taken against, are
        numeric roles too (review 2026-09-26, D6: the check read the names
        and knew no reference, so the refusal came later, with another
        message)."""
        df = _temporal(_frame())
        for target, role in [
            (po.target("t", relative_to="y", name="tt"), "a target"),
            (po.target("y", relative_to="t"), "a relative_to reference"),
        ]:
            other = po.spec.ewridge("f", targets=[target], features=["x0"], half_life=50.0)
            with pytest.raises(ValueError, match="can only be a clock") as exc:
                po.ModelBank([_ridge("t", half_life="10m", gap_cap="30m"), other]).fit_predict(df)
            assert role in str(exc.value), str(exc.value)


class TestADurationSurvives:
    # An infinity word is a number, not text: the spec holds the float, which
    # a loaded bank's `specs` decode the Rust side's "inf" to, and the two
    # compared unequal for `half_life="inf"` (review 2026-10-05, YA7).
    @pytest.mark.parametrize("half_life", [None, "inf", "+INF", "infinity"])
    def test_save_load_and_the_specs_it_was_built_from(self, tmp_path, half_life):
        df = _temporal(_frame())
        durations = DURATIONS["pl.duration"]
        if half_life is not None:
            durations = {**durations, "half_life": half_life}
        spec = _ridge("t", session="s", **durations)
        assert po.ModelBank([spec]).specs == [spec]
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


def _tables() -> tuple[dict[str, list[str]], list[tuple[str, str]], dict[str, list[str]]]:
    """The clock parameters, the rates per clock unit, and the parameters
    that take a duration or a number bound to no clock unit (task 179)."""
    fields, rates, either = spec_clock_fields()
    return dict(fields), [(o, f) for o, fs in rates for f in fs], dict(either)


def _kind(builder: str) -> str:
    kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
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
        fields, rates, either = _tables()
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
            kind = _kind(builder)
            # A parameter of the second table takes a duration too, and a
            # number that binds to no clock unit (task 179).
            assert own == set(fields.get(kind, [])) | set(either.get(kind, [])), builder
            # The docstrings are the second opinion: a parameter documented
            # in clock units, or named as a half-life, takes a duration or
            # is a rate.
            # Python 3.13 strips a docstring's common indentation when it
            # compiles it, and 3.12 does not, so the text is cleaned to one
            # form first: headers at column 0, their entries indented under
            # them. Read raw on 3.14, a four-space header pattern matched an
            # indented line inside another entry and read that entry's text
            # (the v0.10.0 macOS CI leg, CPython 3.14.7).
            doc = inspect.cleandoc(fn.__doc__ or "")
            for name in hints:
                # A parameter's own entry: from its header to the next one.
                head = f"\n``{name}``"
                entry = doc.split(head, 1)[1].split("\n``", 1)[0] if head in doc else ""
                # "counted in learned rows, not clock units" is not one.
                clocky = name.endswith("half_life") or bool(
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
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": "50s"}
        kw.update(MINIMAL[builder])
        if builder == "kalman":
            kw["coef_half_life"] = "1m"
        if builder == "ew_class":
            kw["label"] = "lab"
        kw.update(clock="t", gap_cap="1h")
        return kw

    #: A duration each parameter can take on the frame above, and whatever
    #: else the parameter needs beside it.
    VALUES = {
        "half_life": ("5m", {}),
        "gap_cap": ("30m", {}),
        "restart_after_step_back": ("10s", {}),
        "session_gap": ("10m", {"session": "s"}),
        "embargo": ("30s", {}),
        "long_half_life": ("1h", {"session": "s", "session_gap": "10m", "session_shrink": 0.5}),
        "solve_every": ("10s", {}),
        "window_size": ("30m", {}),
        "select_half_life": ("1h", {}),
        "coef_half_life": ("2m", {}),
        "revert_half_life": ("1h", {}),
        "level_half_life": ("5m", {"half_life": None}),
        "trend_half_life": ("20m", {}),
        "pca_every": ("10m", {"pca": 1}),
        "prune_every": ("10m", {}),
        "window_every": ("1m", {"window_size": "30m"}),
        "drift_threshold": ("20m", {"emit_drift": True}),
        "coef_every": ("10m", {}),
    }

    def _cases(self):
        fields, _, _ = _tables()
        kinds = {builder: _kind(builder) for builder in MINIMAL}
        for owner, names in fields.items():
            builders = ["ewridge"] if owner == "*" else [b for b, k in kinds.items() if k == owner]
            assert builders, owner
            for builder in builders:
                for name in names:
                    yield builder, name

    #: How each clock parameter is made unit-free when another is the one
    #: under test: ``inf`` where that means something, left out where it can
    #: be. ``gap_cap`` and ``session_gap`` take no ``inf`` (task 120) and
    #: stay durations: the parameter under test, a plain number beside them,
    #: is then refused as a mixture, which names it the same way.
    UNIT_FREE = {
        "half_life": float("inf"),
        "coef_half_life": float("inf"),
        "level_half_life": float("inf"),
        "trend_half_life": float("inf"),
        "long_half_life": float("inf"),
        "select_half_life": float("inf"),
        "revert_half_life": float("inf"),
        "window_size": None,
        "solve_every": None,
        "embargo": None,
        "restart_after_step_back": None,
        "pca_every": None,
        "prune_every": None,
        "window_every": None,
        "coef_every": None,
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

    def test_the_second_table_takes_a_duration_and_a_number_of_rows(self):
        """The parameters that take a duration or a number bound to no clock
        unit -- ``bocpd``'s ``hazard``, task 179 -- take both on a temporal
        clock: the number counts rows there as anywhere, so it is no
        mixture. A duration with no clock is refused by name, as a clock
        parameter's is."""
        _, _, either = _tables()
        df = _temporal(_frame())
        seen = 0
        for owner, names in either.items():
            builders = [b for b in MINIMAL if _kind(b) == owner]
            assert builders, owner
            for builder in builders:
                fn = getattr(po.spec, builder)
                kw = {k: v for k, v in self._base(builder).items() if v is not None}
                for name in names:
                    po.ModelBank([fn("m", **{**kw, name: "10m"})]).fit_predict(df)
                    po.ModelBank([fn("m", **{**kw, name: 600.0})]).fit_predict(df)
                    clockless = {k: v for k, v in kw.items() if k not in ("clock", "gap_cap")}
                    with pytest.raises(ValueError, match=f"{name} is a duration, which needs"):
                        fn("m", **{**clockless, name: "10m"})
                    seen += 1
        assert seen == 1, seen


class TestADurationTheDataCannotHold:
    """A duration can be well formed and still mean nothing on the clock it
    is given: finer than the column's own step, where it cannot act."""

    def _daily(self) -> pl.DataFrame:
        days = pl.date_range(date(2024, 1, 1), date(2024, 3, 31), eager=True)
        rng = np.random.default_rng(2)
        x = rng.normal(size=len(days))
        return pl.DataFrame({"d": days, "x0": x, "y": x + rng.normal(0.0, 0.1, len(days))})

    def _daily_spec(self, **kw) -> dict:
        return po.spec.ewridge("m", targets=["y"], features=["x0"], clock="d", half_life="5d", **kw)

    def test_a_cap_finer_than_a_day_is_refused_on_a_date_clock(self):
        with pytest.raises(ValueError, match="gap_cap is 12h, less than a day") as e:
            _fit(self._daily(), self._daily_spec(gap_cap="12h"))
        assert "clock would count rows" in str(e.value)
        _fit(self._daily(), self._daily_spec(gap_cap="1d"))

    def test_a_threshold_no_step_back_can_meet_is_refused(self):
        """Under a minimum finer than a day no step back could be as small, so
        every one would start the model over, which `0` says directly. A
        minimum of exactly a day acts: the comparison is inclusive, so a day's
        step back is a late row (task 120: it passed under a strict `<`)."""
        restart = dict(gap_cap="3d")
        with pytest.raises(ValueError, match="restart_after_step_back is 12h") as e:
            _fit(self._daily(), self._daily_spec(restart_after_step_back="12h", **restart))
        assert "every one would start the model over" in str(e.value)
        df = self._daily()
        late = pl.concat([df.slice(0, 3), df.slice(4, 1), df.slice(3, 1), df.slice(5)])
        with pytest.raises(ValueError, match="goes backwards by 1d at row 4") as e:
            _fit(late, self._daily_spec(restart_after_step_back="1d", **restart))
        assert "restart_after_step_back = 1d" in str(e.value)

    # A 500 us half-life forgets every row at once, which the readiness notice
    # says; that it is allowed is the point here.
    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_the_step_is_the_column_unit_for_a_datetime(self):
        df = _temporal(_frame(), "ms")
        with pytest.raises(ValueError, match="less than a millisecond"):
            _fit(df, _ridge("t", half_life="10m", gap_cap="500us"))
        # A half-life finer than the step is not refused: it forgets fast, and means it.
        _fit(df, _ridge("t", half_life="500us", gap_cap="30m"))


class TestTheHelpersMeasureTheClockTheSameWay:
    def test_embargo_takes_a_duration_on_a_temporal_clock(self):
        df = _frame()
        spec = dict(weight="_online_role_weight")
        numbers = po.stream.embargo(df.lazy(), clock="t_s", delay=300.0).collect()
        numbers = _fit(numbers, _ridge("t_s", half_life=600.0, gap_cap=1_800.0, **spec))
        for form in ("5m", timedelta(minutes=5), pl.duration(minutes=5)):
            doubled = po.stream.embargo(_temporal(df).lazy(), clock="t", delay=form).collect()
            assert doubled["t"].dtype == pl.Datetime("us")
            got = _fit(doubled, _ridge("t", half_life="10m", gap_cap="30m", **spec))
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
            po.stream.embargo(frames[frame].lazy(), clock=clock, delay=delay)

    def test_rolling_metrics_buckets_a_temporal_clock_by_a_duration(self):
        df = _frame()
        numbers = df.hstack(_fit(df, _ridge("t_s", **NUMBERS | {"session_gap": None})).to_frame())
        durations = _temporal(df)
        durations = durations.hstack(
            _fit(durations, _ridge("t", **DURATIONS["text"] | {"session_gap": None})).to_frame()
        )
        want = po.eval.rolling_metrics(numbers, "m", clock="t_s", window_size=3_600.0, min_obs=5)
        got = po.eval.rolling_metrics(durations, "m", clock="t", window_size="1h", min_obs=5)
        assert got["window_start"].dtype == pl.Datetime("us")
        assert want.height > 3
        seconds = got["window_start"].dt.epoch("s").cast(pl.Float64)
        assert seconds.equals(want["window_start"], check_names=False)
        assert got.drop("window_start").equals(want.drop("window_start"))


class TestNanosecondsAreKept:
    """A temporal clock is read in its own integer nanoseconds, and the gap
    between two rows is taken in integers before it becomes seconds, so the
    decay uses the exact gap whatever the stream's age. Read as a double of
    seconds -- since 1970, or from the stream's first instant -- a clock
    resolves 2**-52 of the time since that origin, and ticks closer than
    that read as simultaneous."""

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
        # weight_sum at a row is the weight before that row's own decay, which
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
            half_life="1us",
            gap_cap="365d",
            max_error_inflation=float("inf"),
        )
        got = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("weight_sum").to_numpy()
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
            half_life=1_000.0,
            gap_cap=1e18,
            max_error_inflation=float("inf"),
        )
        drift = (
            po.ModelBank([loose]).fit_predict(as_float)["m"].struct.field("weight_sum").to_numpy()
        )
        assert np.max(np.abs(drift - want)) > 1e-3 * np.max(want)

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @pytest.mark.parametrize(
        "age_s", [86_400, 31_557_600, 315_576_000], ids=["a day", "a year", "a decade"]
    )
    def test_the_gaps_are_exact_at_any_age_of_the_stream(self, age_s):
        """The gap is taken in integer nanoseconds, so ticks a decade into a
        stream decay by their exact gaps as they do at its start. Read as a
        double of seconds from the first instant, they drifted by the
        double's resolution at that age over the half-life: 3.9e-6 at a day,
        9.6e-4 at a year and 1.1e-2 at a decade under this 1 us half-life."""
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
            half_life="1us",
            gap_cap="365d",
            max_error_inflation=float("inf"),
        )
        got = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("weight_sum").to_numpy()
        want = self._n_eff(np.concatenate([[0], np.diff(ns)]), 1_000.0)
        assert np.max(np.abs(got - want)) < 1e-12 * np.max(want)

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    def test_fed_in_chunks_the_gaps_are_the_one_chunk_runs(self):
        """Hard rule 3 on a temporal clock: the previous row's instant
        crosses a chunk boundary in the state, so the nanosecond gaps across
        it are the one-chunk run's."""
        ns, _ = self._ticks()
        rng = np.random.default_rng(2)
        x = rng.normal(size=len(ns))
        df = pl.DataFrame({"t": pl.Series(ns).cast(pl.Datetime("ns")), "x0": x, "y": x})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            half_life="1us",
            gap_cap="365d",
            max_error_inflation=float("inf"),
        )
        whole = po.ModelBank([spec]).fit_predict(df)["m"]
        bank = po.ModelBank([spec])
        chunked = pl.concat([bank.fit_predict(df.slice(i, 37))["m"] for i in range(0, len(df), 37)])
        for field in ("pred_y", "weight_sum"):
            assert whole.struct.field(field).equals(chunked.struct.field(field), null_equal=True)

    def test_a_refused_chunk_leaves_the_state_untouched(self):
        df = _temporal(_frame())
        # A minute apart throughout, so a row fed late is a step back, which
        # the default policy refuses.
        df = df.with_columns(
            t_s=(START + 60.0 * pl.int_range(pl.len())).cast(pl.Float64)
        ).with_columns(t=pl.from_epoch(pl.col("t_s").cast(pl.Int64), time_unit="s"))
        spec = _ridge("t", half_life="10m", gap_cap="30m")
        untouched = po.ModelBank([spec]).save_bytes()
        bank = po.ModelBank([spec])
        late = pl.concat([df.slice(0, 5), df.slice(6, 5), df.slice(5, 1)])
        with pytest.raises(ValueError, match="backwards"):
            bank.fit_predict(late)
        assert bank.save_bytes() == untouched
        # Accepted, the clock range and the last clock come back as seconds
        # since 1970.
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

    def test_the_previous_instant_survives_a_save_and_load(self, tmp_path):
        """The previous row's instant is in the state, so a loaded bank takes
        the next row's gap from it exactly: the numbers go on as the
        unbroken run's, and the clock range reports seconds since 1970."""
        df = _temporal(_frame())
        spec = _ridge("t", half_life="10m", gap_cap="30m")
        whole = po.ModelBank([spec]).fit_predict(df)["m"]
        bank = po.ModelBank([spec])
        bank.fit_predict(df.slice(0, 100))
        bank.save(tmp_path / "o.state")
        loaded = po.ModelBank.load(tmp_path / "o.state")
        tail = loaded.fit_predict(df.slice(100))["m"]
        for field in ("pred_y", "weight_sum"):
            assert (
                whole.struct.field(field)
                .slice(100)
                .equals(tail.struct.field(field), null_equal=True)
            )
        assert loaded.summary()["clock_min"][0] == pytest.approx(df["t_s"][0], abs=1e-6)
        assert loaded.summary()["clock_max"][0] == pytest.approx(df["t_s"][-1], abs=1e-6)

    @pytest.mark.parametrize("dtype", [pl.Datetime("ms"), pl.Date], ids=["Datetime(ms)", "Date"])
    def test_an_instant_past_2262_is_refused_by_row(self, dtype):
        """A coarse temporal column reaches further than nanoseconds in an
        i64 do; a value past that is refused, naming its row, rather than
        wrapped."""
        df = _temporal(_frame(20)).with_columns(t=pl.col("t").cast(dtype))
        far = pl.datetime(2300, 1, 1).cast(dtype)
        late = df.with_columns(
            t=pl.when(pl.int_range(pl.len()) == 7).then(far).otherwise(pl.col("t"))
        )
        step = "2d" if dtype == pl.Date else "10m"
        spec = _ridge("t", half_life=step, gap_cap=step)
        with pytest.raises(ValueError, match="row 7.*nanoseconds cannot hold"):
            po.ModelBank([spec]).fit_predict(late)


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
        doubled = po.stream.embargo(df.lazy(), clock="d", delay="2d").collect()
        assert doubled["d"].dtype == pl.Date
        learn = doubled.filter(pl.col("_online_role") == "learn")
        assert (learn["d"] - df["d"]).unique().to_list() == [timedelta(days=2)]

    def test_rolling_metrics_buckets_a_duration_clock(self):
        df = _temporal(_frame()).with_columns(elapsed=pl.col("t") - pl.col("t").first())
        spec = _ridge("elapsed", half_life="10m", gap_cap="30m")
        out = df.hstack(_fit(df, spec).to_frame())
        got = po.eval.rolling_metrics(out, "m", clock="elapsed", window_size="1h", min_obs=5)
        assert got["window_start"].dtype == pl.Duration("us")
        assert got.height > 3
        starts = got["window_start"].dt.total_seconds().to_list()
        assert all(s % 3_600 == 0 for s in starts)

    def test_padding_and_spaces_in_duration_text(self):
        spec = _ridge("t", half_life=[" 5m ", "1h"], gap_cap="365d")
        assert spec["half_life"] == ["5m", "1h"]
        assert any("@h5m" in f for f in po.spec.output_fields(spec))
        with pytest.raises(ValueError, match="has a space in it"):
            _ridge("t", half_life="1h 30m", gap_cap="365d")


UNIT_NS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "ms": 1_000_000,
    "s": 10**9,
    "m": 60 * 10**9,
    "h": 3_600 * 10**9,
    "d": 86_400 * 10**9,
    "w": 7 * 86_400 * 10**9,
}
#: The keyword ``pl.duration`` and ``timedelta`` take for each unit.
UNIT_KEYWORD = {
    "ns": "nanoseconds",
    "us": "microseconds",
    "ms": "milliseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}
FORMS = ["text", "pl.duration", "timedelta"]
COLUMNS = {
    "Datetime(ms)": (pl.Datetime("ms"), 1_000_000),
    "Datetime(us)": (pl.Datetime("us"), 1_000),
    "Datetime(ns)": (pl.Datetime("ns"), 1),
    "Datetime(us, New York)": (pl.Datetime("us", "America/New_York"), 1_000),
    "Date": (pl.Date, 86_400 * 10**9),
    "Duration(ms)": (pl.Duration("ms"), 1_000_000),
    "Duration(us)": (pl.Duration("us"), 1_000),
    "Duration(ns)": (pl.Duration("ns"), 1),
}


def _exact_n_eff(gaps_ns: np.ndarray, halflife_ns: float) -> np.ndarray:
    """The exact recursion: weights of 1, each decayed by its exact gap."""
    out, w = [], 0.0
    for g in gaps_ns:
        out.append(w)
        w = w * 2.0 ** (-float(g) / halflife_ns) + 1.0
    return np.array(out)


def _column(dtype, ns: np.ndarray) -> pl.Series:
    """Instants given in nanoseconds, as a column of ``dtype`` in its own
    unit; a zone-aware column holds the same UTC instants."""
    if dtype == pl.Date:
        return pl.Series(ns // UNIT_NS["d"]).cast(pl.Int32).cast(pl.Date)
    if isinstance(dtype, pl.Datetime):
        s = pl.Series(ns).cast(pl.Datetime("ns"))
        if dtype.time_zone is not None:
            s = s.dt.replace_time_zone("UTC").dt.convert_time_zone(dtype.time_zone)
        return s.dt.cast_time_unit(dtype.time_unit)
    return pl.Series(ns).cast(pl.Duration("ns")).cast(dtype)


def _writes(form: str, unit: str) -> bool:
    """Whether ``form`` can write ``unit``. Text writes all nine spellings;
    ``µs`` is text's other spelling of ``us``, which the two keyword forms
    write as ``microseconds``; and a ``timedelta`` holds nothing finer than
    a microsecond, so it cannot write nanoseconds."""
    if form == "text":
        return True
    return unit != "µs" and not (form == "timedelta" and unit == "ns")


def _spell(form: str, n: int, unit: str):
    """``n`` of ``unit``, written in ``form``."""
    if form == "text":
        return f"{n}{unit}"
    if form == "pl.duration":
        return pl.duration(**{UNIT_KEYWORD[unit]: n})
    return timedelta(**{UNIT_KEYWORD[unit]: n})


def _cases(finer: bool) -> list:
    """Every ``(column, unit, form)`` the form can write, with the unit
    finer than the column's step or not. The two lists partition the
    matrix, so each case runs in exactly one of the two tests below."""
    return [
        pytest.param(column, unit, form, id=f"{column}-{unit}-{form}")
        for column, (_, step) in COLUMNS.items()
        for unit, u in UNIT_NS.items()
        if (u < step) == finer
        for form in FORMS
        if _writes(form, unit)
    ]


class TestEveryUnitAgainstEveryColumn:
    """Every duration unit -- ``ns``, ``us``/``µs``, ``ms``, ``s``, ``m``,
    ``h``, ``d``, ``w`` -- in each form a duration is written in (polars'
    text, ``pl.duration(...)`` with that unit's keyword, and ``timedelta``),
    against every temporal column a clock can be, in each of its units, a
    zone-aware one among them. A unit the column can express gives the
    exact recursion on gaps of that unit; a unit finer than the column's
    step is refused by name as a cap or a disorder threshold, and a
    half-life in it is read at its own scale all the same."""

    def test_the_two_tests_cover_the_matrix_once(self):
        """Every unit each form can write, against every column, in exactly
        one of the two tests: text writes nine, ``pl.duration`` eight and
        ``timedelta`` seven, so 24 against each of the eight columns."""
        coarse = {c.id for c in _cases(finer=False)}
        fine = {c.id for c in _cases(finer=True)}
        assert not coarse & fine
        writable = [(f, u) for f in FORMS for u in UNIT_NS if _writes(f, u)]
        assert len(writable) == 9 + 8 + 7
        assert len(coarse | fine) == len(COLUMNS) * len(writable) == 192

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @pytest.mark.parametrize(("column", "unit", "form"), _cases(finer=False))
    def test_a_unit_the_column_can_express_gives_the_exact_recursion(self, column, unit, form):
        dtype, step = COLUMNS[column]
        u = UNIT_NS[unit]
        half_life, cap = _spell(form, 7, unit), _spell(form, 20, unit)
        rng = np.random.default_rng(7)
        gaps = rng.integers(1, 4, 200) * u  # one to three units apart
        ns = START * 10**9 + np.cumsum(gaps)
        x = rng.normal(size=len(ns))
        df = pl.DataFrame({"t": _column(dtype, ns), "x0": x, "y": x})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            half_life=half_life,
            gap_cap=cap,
            max_error_inflation=float("inf"),
        )
        got = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("weight_sum").to_numpy()
        want = _exact_n_eff(np.concatenate([[0], np.diff(ns)]), 7.0 * u)
        assert np.max(np.abs(got - want)) < 1e-12 * np.max(want)
        # The decay is real at this scale, so a unit read at another scale
        # could not have passed: no row's weight is trivially 1 or the sum.
        assert 1.5 < want[-1] < len(ns) - 1

    @pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
    @pytest.mark.parametrize(("column", "unit", "form"), _cases(finer=True))
    def test_a_unit_finer_than_the_columns_step_cannot_cap_it(self, column, unit, form):
        dtype, step = COLUMNS[column]
        u = UNIT_NS[unit]
        seven = _spell(form, 7, unit)
        ns = START * 10**9 + np.arange(20) * 3 * step
        x = np.arange(20, dtype=float)
        df = pl.DataFrame({"t": _column(dtype, ns), "x0": x, "y": x})
        coarse = {
            "half_life": seven,
            "gap_cap": "3w",
            "restart_after_step_back": "3w",
        }
        for param in ("gap_cap", "restart_after_step_back"):
            spec = po.spec.ewridge(
                "m", targets=["y"], features=["x0"], clock="t", **{**coarse, param: seven}
            )
            with pytest.raises(ValueError, match=f"{param} is 7{unit}, less than .*smallest step"):
                po.ModelBank([spec]).fit_predict(df)
        # A half-life finer than the step is a setting, not a contradiction:
        # every gap is many half-lives, so each row is fitted on itself, as
        # the exact recursion says.
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0"],
            clock="t",
            max_error_inflation=float("inf"),
            **coarse,
        )
        got = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("weight_sum").to_numpy()
        want = _exact_n_eff(np.concatenate([[0], np.diff(ns)]), 7.0 * u)
        assert np.array_equal(got, want)
