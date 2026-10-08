"""``drift_threshold`` is a clock parameter (docs/PLAN.md task 168; review
2026-10-05, CC1).

The drift detector sums each row's excess, in ``sigma``, times the row's clock
step, so its threshold is ``sigma`` times clock time: a number of the clock
column's units on a numeric clock, a duration on a temporal one, required with
a clock as ``gap_cap`` is, and 20 without one, where a row is one unit. Before,
the threshold was a plain number with 20 as its default everywhere, and every
temporal clock is read on one scale, seconds, whatever the column's own unit.
So on a temporal clock the default was 20 ``sigma``-seconds, a unit no one
chose, and at a row a minute it flagged noise.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

TEMPORAL = dict(clock="ts", half_life="1d", gap_cap="10m")
SECONDS = dict(clock="t", half_life=86400.0, gap_cap=600.0)


def frame(n: int = 3000, burst: tuple[int, int] | None = None, seed: int = 3) -> pl.DataFrame:
    """Rows a minute apart, on a temporal clock and the same clock in seconds,
    with a target whose noise is four times larger over ``burst``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 60
    x = rng.standard_normal(n)
    size = np.ones(n)
    if burst is not None:
        size[burst[0] : burst[1]] = 4.0
    ts = [datetime(2024, 1, 2) + timedelta(seconds=int(s)) for s in t]
    return pl.DataFrame(
        {"ts": ts, "t": t.astype(float), "x": x, "y": 2 * x + size * rng.standard_normal(n)}
    )


def spec(**kw):
    return po.spec.ewridge("m", targets=["y"], features=["x"], emit_drift=True, **kw)


def flags(df: pl.DataFrame, **kw) -> np.ndarray:
    out = po.ModelBank([spec(**kw)]).fit_predict(df)
    return out["m"].struct.field("drift_y").fill_null(False).to_numpy()


def test_a_duration_is_its_length_on_the_clock_whatever_the_column_unit():
    """``"20m"`` flags the rows 1200 does on the same clock in seconds, and so
    does every way of writing twenty minutes, on a column in milliseconds,
    microseconds or nanoseconds."""
    df = frame(burst=(2000, 2100))
    want = flags(df, drift_threshold=1200.0, **SECONDS)
    assert want.any(), "the case needs a break"
    for threshold in ["20m", "1200s", timedelta(minutes=20), pl.duration(minutes=20)]:
        got = flags(df, drift_threshold=threshold, **TEMPORAL)
        assert np.array_equal(got, want), threshold
    for unit in ["ms", "us", "ns"]:
        cast = df.with_columns(pl.col("ts").cast(pl.Datetime(unit)))
        assert np.array_equal(flags(cast, drift_threshold="20m", **TEMPORAL), want), unit


def test_the_old_default_flagged_noise_and_a_threshold_in_minutes_does_not():
    """The old default on a temporal clock, 20 ``sigma``-seconds, flags 244 of
    3,000 rows of noise a minute apart (the review measured 237 on its own
    stream). Twenty minutes is twenty per row here, the classic test's
    default: it flags no noise, and finds a burst 18 rows in."""
    noise = flags(frame(), drift_threshold="20s", **TEMPORAL)
    assert noise.sum() > 200, noise.sum()
    assert not flags(frame(), drift_threshold="20m", **TEMPORAL).any()
    hits = np.flatnonzero(flags(frame(burst=(2000, 2100)), drift_threshold="20m", **TEMPORAL))
    assert hits.size and 2000 <= hits[0] < 2100, hits


@pytest.mark.parametrize("half_life", [float("inf"), 1000.0, 100.0])
def test_a_stationary_stream_at_the_row_default_flags_nothing(half_life):
    """The defaults without a clock, ``drift_delta`` 0.5 and a threshold of
    20 rows, on 200,000 rows of Gaussian residuals: no flag at all. The only
    stationary test was 3,000 rows under ``"20m"`` (review 2026-10-06, CE6).
    The rate rests on the residual's tail: on ``t`` residuals with 3 degrees
    of freedom the same defaults flagged 3 to 11 times per 200,000 rows,
    about 4e-5 a row, at each of these half-lives (the review's three seeds,
    and this stream's ``x`` with a ``t₃`` noise: 10, 11 and 9)."""
    rng = np.random.default_rng(17)
    n = 200_000
    x = rng.standard_normal(n)
    df = pl.DataFrame({"x": x, "y": 2 * x + rng.standard_normal(n)})
    got = flags(df, half_life=half_life)
    assert got.size == n
    assert not got.any(), np.flatnonzero(got)[:10]


def test_without_a_clock_the_default_is_20_rows():
    """A row is one unit without a clock column, so 20 is the classic test,
    and the default stays."""
    df = frame(n=800, burst=(500, 560))
    default = flags(df, half_life=1000.0)
    assert np.array_equal(default, flags(df, half_life=1000.0, drift_threshold=20.0))
    assert default.any(), "the case needs a break"


class TestRefusals:
    def test_a_clock_needs_a_threshold(self):
        for clock in (TEMPORAL, SECONDS):
            with pytest.raises(ValueError, match="drift_threshold is required when clock is given"):
                spec(**clock)

    def test_a_temporal_clock_refuses_a_plain_number(self):
        with pytest.raises(ValueError, match="drift_threshold is a plain number"):
            spec(drift_threshold=20.0, **TEMPORAL)

    def test_a_numeric_clock_refuses_a_duration(self):
        with pytest.raises(ValueError, match="drift_threshold is a duration"):
            spec(drift_threshold="20m", **SECONDS)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
    def test_a_threshold_is_finite_and_above_zero(self, bad):
        # The builder refuses inf itself, as "must be finite, got float inf".
        with pytest.raises(ValueError, match="drift_threshold must be finite"):
            spec(drift_threshold=bad, **SECONDS)


def test_a_bank_resumes_with_its_duration_threshold():
    """The spec keeps the duration as written, in its JSON and in a saved
    state, and a bank loaded mid-stream flags what the unbroken run does."""
    df = frame(burst=(2000, 2100))
    s = spec(drift_threshold="20m", **TEMPORAL)
    whole = po.ModelBank([s]).fit_predict(df)["m"].struct.field("drift_y")
    first = po.ModelBank([s])
    first.fit_predict(df.head(1500))
    assert '"drift_threshold": "20m"' in first.to_json()
    resumed = po.ModelBank.load_bytes(first.save_bytes(), specs=[s])
    tail = resumed.fit_predict(df.slice(1500))["m"].struct.field("drift_y")
    assert tail.equals(whole.slice(1500), null_equal=True)
