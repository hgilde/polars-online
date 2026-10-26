"""E47: `embargo`, and the doubled stream it replaces.

The claim is that a row is *scored* where it sits and *learned from* only
once its label would really have been known. Two things have to be earned:

1. It is the doubled stream, not something like it. `po.stream.embargo` builds
   the recipe E47 names -- every row twice, a zero-weight prediction at `t`
   and a lesson at `t + delay` -- and the native path is held against it
   **bit for bit** on everything it predicts, not approximately. Not on what
   it *folds*: since the code review of 2026-09-12 (C21) the residual
   diagnostics fold the prediction a row was scored with, where the doubled
   stream's lesson forms its residual from a model that has learned every
   row before it. That difference is pinned below, and which side is right
   is held against river in `tests/test_second_opinion.py`.
2. It removes a leak that is otherwise there. With an autocorrelated feature
   and a forward-looking target, learning the label where it sits makes a
   pure noise column look predictive. The test measures that, and measures it
   gone.
"""

import math
import subprocess
from datetime import time

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online
from polars_online import stream
from test_model_registry import MINIMAL

TIER = "mixed"

HALFLIFE = 50.0


def frame(n=400, seed=0, step=1.0, ar=0.0, horizon=0):
    """A stream with an optional autocorrelated feature and an optional
    forward-looking target.

    ``horizon = h`` makes ``y`` the sum of the next ``h`` innovations -- an
    overlapping forward return, the shape of target this whole feature is
    for. Consecutive values then share ``h - 1`` terms, so knowing one is
    most of knowing the next; that is the future a stream leaks when it
    learns the label where it sits.
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = ar * noise[i - 1] + rng.standard_normal()
    x = rng.standard_normal(n)
    innov = rng.standard_normal(n + max(horizon, 1))
    forward = np.array([innov[i : i + horizon].sum() for i in range(n)])
    y = innov[:n] if horizon == 0 else forward
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float) * step,
            "x": x,
            "noise": noise,
            "y": y,
        }
    )


def spec(name="m", *, features=("x",), **kw):
    kw.setdefault("clock", "t")
    kw.setdefault("half_life", HALFLIFE)
    kw.setdefault("gap_cap", 1e9)
    kw.setdefault("min_weight", 3.0)
    kw.setdefault("standardize", False)
    kw.setdefault("max_rows_between_solves", 1)
    return po.spec.ewridge(name, targets=["y"], features=list(features), **kw)


def doubled(df, delay, **kw):
    """The same fit through `po.stream.embargo`: the oracle E47 names."""
    frame_ = stream.embargo(df, clock="t", delay=delay)
    bank = po.ModelBank([spec(weight=stream.ROLE + "_weight", **kw)])
    out = bank.fit_predict(frame_)
    return out.filter(pl.col(stream.ROLE) == "predict"), bank


class TestItIsTheDoubledStream:
    @pytest.mark.parametrize("delay", [1.0, 5.0, 40.0])
    def test_every_field_matches_the_oracle_to_the_bit(self, delay):
        df = frame()
        native = po.ModelBank([spec(embargo=delay)]).fit_predict(df)
        oracle, _ = doubled(df, delay)
        a, b = native["m"].struct, oracle["m"].struct
        for field in po.spec.output_fields(spec(embargo=delay)):
            # The lists ride a cadence; the reason is an enum, not a number;
            # and `settled_frac` reads the clock *before* the row, where the
            # embargo frame puts a maturing row's learn copy at the same clock
            # as, and ahead of, its predict copy -- so once releases begin the
            # oracle counts the row's own delta and the stream does not.
            if field in ("coef", "support_coef", "withheld_reason", "settled_frac"):
                continue
            x, y = a.field(field).to_numpy(), b.field(field).to_numpy()
            assert (np.isnan(x) == np.isnan(y)).all(), field
            fin = np.isfinite(x)
            assert np.array_equal(x[fin], y[fin]), (field, np.max(np.abs(x[fin] - y[fin])))

    @pytest.mark.parametrize(
        "kw",
        [
            dict(emit_sigma=True, emit_zscore=True, emit_metrics=True, conformal=0.9),
            dict(emit_drift=True, drift_threshold=20.0, emit_autocorr=True, resid_quantiles=[0.5]),
        ],
        ids=["weighted", "weight-free"],
    )
    def test_the_predictions_match_and_the_diagnostics_do_not(self, kw):
        """The doubled stream is the oracle for what a delayed bank *predicts*
        -- `pred`, `resid` and `weight_sum` match it to the bit -- and not for what
        the bank *folds*. Its lesson at `t + delay` forms its residual from the
        model as it then stands, which has learned every row before it: the
        prediction that peeks, and the one the diagnostics folded until the
        code review of 2026-09-12 (C21). They now fold the prediction the row
        was scored with, so every residual diagnostic -- sigma, zscore, the
        metrics, the conformal interval, the quantiles, the autocorrelation,
        drift -- parts from the oracle's, and must. Which is right is held
        against river's delayed progressive validation in
        `tests/test_second_opinion.py::TestLabelDelayFoldsWhatWasScored`; this
        pins that the difference is there and only there.

        This test was two until then, both asserting agreement, and the
        weight-free half once asserted *disagreement*, for a reason that was
        the diagnostics' defect rather than the oracle's (S28)."""
        df = frame(n=600, seed=1)
        native = po.ModelBank([spec(embargo=7.0, **kw)]).fit_predict(df)
        oracle, _ = doubled(df, 7.0, **kw)
        a, b = native["m"].struct, oracle["m"].struct
        predicted = {"pred_y", "resid_y", "weight_sum"}
        parted = []
        for field in po.spec.output_fields(spec(embargo=7.0, **kw)):
            # The lists ride a cadence; the reason is an enum, not a number.
            if field in ("coef", "support_coef", "withheld_reason"):
                continue
            x = a.field(field).cast(pl.Float64).to_numpy()
            y = b.field(field).cast(pl.Float64).to_numpy()
            same = (np.isnan(x) == np.isnan(y)).all() and np.array_equal(
                x[np.isfinite(x)], y[np.isfinite(y)]
            )
            if field in predicted:
                assert same, field
            elif not same:
                parted.append(field)
        assert parted, "no residual diagnostic parted from the doubled stream"

    def test_without_a_clock_a_skipped_row_counts(self):
        """Task 148: with no ``clock`` one unit is one row of the group, a
        skipped row included, as the docs now say: the row two before is
        released across a row skipped for a null feature, where counting
        accepted rows would hold it one row longer."""
        x = np.arange(12, dtype=float)
        x[3] = np.nan
        df = pl.DataFrame({"x": x, "y": 2 * np.nan_to_num(x) + 1})
        s = po.spec.ewridge(
            "m", targets=["y"], features=["x"], half_life=1e9, min_weight=0.0, embargo=2
        )
        weight_sum = po.ModelBank([s]).fit_predict(df)["m"].struct.field("weight_sum").to_list()
        # Row 2 is released at row 4, two rows on, across the skipped row 3.
        assert weight_sum[4] == pytest.approx(3.0, abs=1e-6)
        assert weight_sum[3] is None

    def test_the_state_is_the_state_of_the_matured_rows(self):
        """A delayed bank at the end of a stream is bit-for-bit the bank a
        plain one would be, fed only the rows whose labels had matured. Not
        the doubled stream's bank: that frame runs `delay` clock units past
        the input, so its tail lessons have landed and the native one's have
        not."""
        n, delay = 500, 9
        df = frame(n=n, seed=2)
        native = po.ModelBank([spec(embargo=float(delay))])
        native.fit_predict(df)
        matured = po.ModelBank([spec()])
        matured.fit_predict(df.head(n - delay))
        assert native.coef("m")["coef"].to_list() == matured.coef("m")["coef"].to_list()
        assert native.gram("m")[0]["weight_sum"] == matured.gram("m")[0]["weight_sum"]


class TestItClosesTheLeak:
    def test_a_noise_feature_looks_predictive_without_the_delay(self):
        """The reason E47 exists. `y` is an overlapping 20-row forward sum,
        `noise` is autocorrelated and independent of it. Learning the label
        where it sits fits the model to a `y` that overlaps the next one by
        19 terms, and the autocorrelated feature carries that fit forward:
        a column with no relationship to the target scores a positive
        out-of-sample R^2. With the delay it scores nothing, which is the
        truth."""
        h = 20
        df = frame(n=20_000, seed=3, ar=0.98, horizon=h)
        common = dict(features=["noise"], half_life=200.0)
        leak = _oos_r2(df, spec(**common))
        clean = _oos_r2(df, spec(embargo=float(h), **common))
        # A noise column scoring +5% "out-of-sample" is the whole problem.
        assert leak > 0.03, f"the fixture did not leak: {leak}"
        # With the delay it scores below zero, which is what fitting noise
        # actually costs -- the honest answer, not merely a smaller one.
        assert clean < 0.0, f"a delayed noise feature should predict nothing: {clean}"

    def test_a_real_feature_still_predicts_with_the_delay(self):
        """The delay must not simply break the model: a feature that really
        does explain the target still does, delay or no delay."""
        n, h = 3000, 10
        rng = np.random.default_rng(4)
        x = rng.standard_normal(n)
        df = pl.DataFrame(
            {
                "t": np.arange(n, dtype=float),
                "x": x,
                "y": 2.0 * x + 0.3 * rng.standard_normal(n),
            }
        )
        delayed = _oos_r2(df, spec(half_life=200.0, embargo=float(h)))
        assert delayed > 0.9, delayed


def _oos_r2(df, s):
    out = po.ModelBank([s]).fit_predict(df)
    pred = out[s["name"]].struct.field("pred_y").to_numpy()
    y = df["y"].to_numpy()
    ok = np.isfinite(pred)
    resid = y[ok] - pred[ok]
    return float(1.0 - resid @ resid / ((y[ok] - y[ok].mean()) @ (y[ok] - y[ok].mean())))


class TestTheStreamContract:
    @pytest.mark.parametrize("size", [1, 7, 64, 1000])
    def test_chunking_cannot_move_a_row(self, size):
        """Release depends on the clock alone, so a chunk boundary cannot
        change which rows have matured."""
        df = frame(n=500, seed=5)
        s = spec(embargo=11.0, emit_sigma=True, emit_metrics=True)

        # `coef` is emitted on each chunk's last row, so chunking moves the
        # cadence it is reported at; every value is compared.
        def fields(frame_):
            return frame_.select("m").unnest("m").drop("coef", "support_coef")

        one = fields(po.ModelBank([s]).fit_predict(df))
        bank = po.ModelBank([s])
        many = fields(
            pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
        )
        assert many.equals(one, null_equal=True)

    def test_the_buffer_survives_a_save_and_load(self, tmp_path):
        df = frame(n=400, seed=6)
        s = spec(embargo=13.0)
        whole = po.ModelBank([s]).fit_predict(df)
        part = po.ModelBank([s])
        part.fit_predict(df.head(200))
        part.save(tmp_path / "b.state")
        # The buffer is not empty at the cut: 13 clock units of rows.
        resumed = po.ModelBank.load(tmp_path / "b.state")
        rest = resumed.fit_predict(df.tail(200))
        assert (
            rest.select("m")
            .unnest("m")
            .equals(whole.tail(200).select("m").unnest("m"), null_equal=True)
        )

    def test_only_a_delayed_state_carries_a_buffer(self):
        """The stream's `pending` is skipped when empty, so a spec without a
        delay writes what it always did. (The clock has a `pending` of its
        own -- the time of skipped rows -- so the two are told apart by
        counting, not by the name appearing at all. Since schema 15 a delayed
        stream also keeps `pending_clock`, the clock its held rows cover.)"""
        df = frame(n=100)
        plain = po.ModelBank([spec()])
        plain.fit_predict(df)
        delayed = po.ModelBank([spec(embargo=5.0)])
        delayed.fit_predict(df)
        assert plain.save_bytes().count(b"pending") == 1, "the clock's, and no other"
        assert delayed.save_bytes().count(b"pending") == 3, (
            "the clock's, the stream's, and the held rows' clock"
        )
        assert len(delayed.save_bytes()) > len(plain.save_bytes())

    def test_rows_still_waiting_are_never_learned_from(self):
        """At the end of a stream the buffer is simply unlearned: those
        labels have not matured, and inventing a deadline would be the leak
        this exists to prevent."""
        df = frame(n=200)
        plain = po.ModelBank([spec()])
        plain.fit_predict(df)
        delayed = po.ModelBank([spec(embargo=30.0)])
        delayed.fit_predict(df)
        # 30 clock units at one row per unit: the last 30 rows never landed.
        assert delayed.gram("m")[0]["weight_sum"] < plain.gram("m")[0]["weight_sum"]
        head = po.ModelBank([spec()])
        head.fit_predict(df.head(200 - 30))
        assert delayed.gram("m")[0]["weight_sum"] == pytest.approx(
            head.gram("m")[0]["weight_sum"], rel=1e-12
        )

    def test_a_skipped_row_waits_for_nothing(self):
        """A null feature skips the row entirely, so it never enters the
        buffer; its clock time still counts the buffer down, because it is
        folded into the next accepted row's delta as it always was."""
        df = frame(n=300, seed=7)
        holes = df.with_columns(
            x=pl.when(pl.int_range(pl.len()) % 17 == 3).then(None).otherwise(pl.col("x"))
        )
        native = po.ModelBank([spec(embargo=8.0)]).fit_predict(holes)
        oracle_frame = stream.embargo(holes, clock="t", delay=8.0)
        oracle = (
            po.ModelBank([spec(weight=stream.ROLE + "_weight")])
            .fit_predict(oracle_frame)
            .filter(pl.col(stream.ROLE) == "predict")
        )
        a = native["m"].struct.field("pred_y").to_numpy()
        b = oracle["m"].struct.field("pred_y").to_numpy()
        assert (np.isnan(a) == np.isnan(b)).all()
        # A ulp, not a bit: a skipped row's clock time is folded into the
        # next *accepted* row, and the doubled stream's accepted rows are not
        # the same rows, so the two partition the same total elapsed time
        # differently. The decay factors multiply to the same number to
        # within rounding, which is all that can be asked of them.
        fin = np.isfinite(a)
        assert a[fin] == pytest.approx(b[fin], rel=1e-12)

    def test_groups_each_have_their_own_buffer(self):
        df = frame(n=600, seed=8).with_columns(g=pl.Series([f"g{i % 3}" for i in range(600)]))
        s = spec(embargo=12.0, group="g")
        together = po.ModelBank([s]).fit_predict(df)
        for key in ("g0", "g1", "g2"):
            part = df.filter(pl.col("g") == key)
            alone = po.ModelBank([spec(embargo=12.0)]).fit_predict(part)
            assert (
                together.filter(pl.col("g") == key)
                .select("m")
                .unnest("m")
                .equals(alone.select("m").unnest("m"), null_equal=True)
            )

    def test_a_reset_drops_the_buffer(self):
        """A restart at a step back throws the models away; the rows
        waiting to teach them go too."""
        n = 200
        clock = np.concatenate([np.arange(100.0), np.arange(100.0)])
        df = frame(n=n, seed=9).with_columns(t=pl.Series(clock))
        # The jump back is what `reset_state` is being asked about: no step
        # back is a late row here.
        s = spec(embargo=10.0, restart_after_step_back=0.0)
        bank = po.ModelBank([s])
        bank.fit_predict(df)
        # After the reset the stream is the second half alone, minus the
        # rows still waiting at its end.
        fresh = po.ModelBank([spec(embargo=10.0)])
        fresh.fit_predict(df.tail(100).with_columns(t=pl.Series(np.arange(100.0))))
        assert bank.gram("m")[0]["weight_sum"] == pytest.approx(
            fresh.gram("m")[0]["weight_sum"], rel=1e-12
        )

    def test_a_session_change_after_a_long_gap_releases_the_buffer(self):
        """The delay counts the time that passed (task 153), and the night
        between these sessions is longer than it, so every row still waiting
        at the boundary has matured there: the first session is learned
        whole, and the second bar its last ten."""
        n = 200
        df = frame(n=n, seed=10).with_columns(
            s=pl.Series(["a"] * 100 + ["b"] * 100),
            t=pl.Series(np.concatenate([np.arange(100.0), np.arange(1000.0, 1100.0)])),
        )
        s = spec(embargo=10.0, session="s", session_gap=1.0)
        bank = po.ModelBank([s])
        bank.fit_predict(df)
        # Every row of the first session was learned from, and every row of
        # the second bar the last ten.
        no_delay = po.ModelBank([spec(session="s", session_gap=1.0)])
        no_delay.fit_predict(df.head(190))
        assert bank.gram("m")[0]["weight_sum"] == pytest.approx(
            no_delay.gram("m")[0]["weight_sum"], rel=1e-6
        )

    def test_a_capped_gap_clears_the_lag_ring_between_the_rows_it_parts(self):
        """A gap over ``gap_cap`` says the rows behind it are no longer
        adjacent, and the models drop what is indexed by rows back. The clear
        waits with the row after the gap and runs when that row is learned,
        after every row before it (task 153), so no ring pairs rows across
        the break (docs/REVIEW-E54-E64.md L2). The gap here is longer than
        the delay, so every held row matures at it.

        ``ew_cov`` with a lag is the model that shows it: the lagged
        co-moments are exactly a pairing of adjacent rows.
        """
        n = 60
        delay = 5.0
        rng = np.random.default_rng(4)
        t = np.arange(float(n))
        t[30:] += 500.0  # one gap, far over the cap
        df = pl.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n), "t": t})

        def cov(**kw):
            # `gap_cap` below the delay: the capped row's own delta is
            # clipped to 2, while the 500 units that passed mature the
            # whole buffer.
            return po.spec.ew_cov(
                "c",
                features=["x0", "x1"],
                lags=[1, 2],
                half_life=1e9,
                clock="t",
                gap_cap=2.0,
                min_weight=3.0,
                **kw,
            )

        # The delayed run learns the same rows in the same order as the plain
        # one -- the delay only moves *when* -- so once every row has matured
        # the lagged matrices must agree.
        plain = po.ModelBank([cov()])
        plain.fit_predict(df)
        delayed = po.ModelBank([cov(embargo=delay)])
        delayed.fit_predict(df)
        a = plain.gram("c")[0]
        b = delayed.gram("c")[0]
        # The last few rows of the delayed run are still waiting, so compare
        # the run that stops where its buffer does.
        matured = int(np.searchsorted(t, t[-1] - delay, side="right"))
        short = po.ModelBank([cov()])
        short.fit_predict(df.head(matured))
        assert np.allclose(b["lag_comoments"], short.gram("c")[0]["lag_comoments"])
        assert not np.allclose(a["lag_comoments"], np.zeros_like(a["lag_comoments"]))


class TestBreaksCountElapsedTime:
    """docs/PLAN.md task 153: a row is learned once its delay has passed in
    elapsed time -- the clock column's own step, skipped rows included, and
    ``session_gap`` where a session change restarts the clock -- and a break
    releases nothing early. A gap past ``gap_cap`` and a session change
    released every held row at once, so a plain forward label was learned
    before it was known wherever a break was shorter than the delay.

    With no decay and unit weights ``weight_sum`` is the number of rows learned,
    so the oracle is a count: at row *s*, the rows *u* before it whose
    elapsed time to *s* is at least the delay."""

    DELAY = 10.0

    @staticmethod
    def _expected(steps, delay):
        """Rows learned before each row is scored, from the elapsed steps."""
        since = np.concatenate([[0.0], np.cumsum(steps[1:])])
        return np.array([np.sum(since[s] - since[:s] >= delay) for s in range(len(steps))])

    def _learned(self, df, **kw):
        s = spec(embargo=self.DELAY, half_life=1e9, min_weight=0.0, **kw)
        bank = po.ModelBank([s])
        out = bank.fit_predict(df)
        return np.array(out["m"].struct.field("weight_sum").to_list(), dtype=float), bank

    def test_a_capped_gap_shorter_than_the_delay_holds_the_rows(self):
        n = 80
        t = np.arange(float(n))
        t[40:] += 4.0  # a 5-unit step past a 2-unit cap, short of the delay
        df = frame(n=n, seed=21).with_columns(t=pl.Series(t))
        got, _ = self._learned(df, gap_cap=2.0)
        want = self._expected(np.diff(t, prepend=t[0]), self.DELAY)
        assert got == pytest.approx(want, rel=1e-6), np.flatnonzero(np.abs(got - want) > 1e-3)

    def test_a_session_change_with_no_gap_holds_the_rows(self):
        n = 80
        t = np.arange(float(n))
        df = frame(n=n, seed=22).with_columns(t=pl.Series(t), s=pl.Series(["a"] * 40 + ["b"] * 40))
        got, _ = self._learned(df, session="s", session_gap=30.0)
        # A session gap longer than the step is what the model forgets, not
        # time that passed: the rows wait for the clock.
        want = self._expected(np.diff(t, prepend=t[0]), self.DELAY)
        assert got == pytest.approx(want, rel=1e-6), np.flatnonzero(np.abs(got - want) > 1e-3)

    def test_a_clock_that_restarts_at_a_session_counts_the_session_gap(self):
        n = 80
        t = np.concatenate([np.arange(40.0), np.arange(40.0)])
        df = frame(n=n, seed=23).with_columns(t=pl.Series(t), s=pl.Series(["a"] * 40 + ["b"] * 40))
        got, _ = self._learned(df, session="s", session_gap=3.0)
        steps = np.diff(t, prepend=t[0])
        steps[40] = 3.0  # the column restarts, so the gap is the session's
        want = self._expected(steps, self.DELAY)
        assert got == pytest.approx(want, rel=1e-6), np.flatnonzero(np.abs(got - want) > 1e-3)

    def test_the_state_is_the_matured_rows_across_breaks(self):
        """With a capped gap and a session change inside the last delay of
        the stream, the delayed bank ends as a plain bank fed only the rows
        whose delay had passed: the same rows, in order, with every break's
        events -- the lag rings' clear, the session's gap -- where they
        fall. The early release learned rows a plain bank never saw."""
        n = 120
        t = np.arange(float(n))
        t[110:] += 3.0
        sess = ["a"] * 114 + ["b"] * 6
        df = frame(n=n, seed=24).with_columns(t=pl.Series(t), s=pl.Series(sess))
        kw = dict(half_life=40.0, gap_cap=2.0, session="s", session_gap=1.0, min_weight=0.0)
        delayed = po.ModelBank([spec(embargo=self.DELAY, **kw)])
        delayed.fit_predict(df)
        since = t[-1] - t
        matured = int(np.sum(since >= self.DELAY))
        plain = po.ModelBank([spec(**kw)])
        plain.fit_predict(df.head(matured))
        assert delayed.coef("m")["coef"].to_list() == plain.coef("m")["coef"].to_list()
        assert delayed.gram("m")[0]["weight_sum"] == plain.gram("m")[0]["weight_sum"]

    def test_the_lag_rings_never_pair_across_a_break_while_rows_are_held(self):
        """The break's clear waits with its row and runs when that row is
        learned, after every row before the break: the lagged co-moments
        are the plain run's over the matured rows, with a gap and a session
        change each shorter than the delay (docs/REVIEW-E54-E64.md L2)."""
        n = 90
        delay = 6.0
        rng = np.random.default_rng(25)
        t = np.arange(float(n))
        t[30:] += 3.0  # past the cap, short of the delay
        sess = ["a"] * 60 + ["b"] * 30
        df = pl.DataFrame(
            {"x0": rng.standard_normal(n), "x1": rng.standard_normal(n), "t": t, "s": sess}
        )

        def cov(**kw):
            return po.spec.ew_cov(
                "c",
                features=["x0", "x1"],
                lags=[1, 2],
                half_life=1e9,
                clock="t",
                gap_cap=2.0,
                session="s",
                session_gap=1.0,
                min_weight=3.0,
                **kw,
            )

        delayed = po.ModelBank([cov(embargo=delay)])
        delayed.fit_predict(df)
        matured = int(np.sum(t[-1] - t >= delay))
        short = po.ModelBank([cov()])
        short.fit_predict(df.head(matured))
        a, b = delayed.gram("c")[0], short.gram("c")[0]
        assert np.allclose(a["lag_comoments"], b["lag_comoments"], rtol=1e-12, atol=1e-15)
        assert a["weight_sum"] == pytest.approx(b["weight_sum"], rel=1e-12)


class TestBreaksOnSkippedRows:
    """A break raised on a row the spec skips -- a null feature on the row of
    a session change or of a capped gap -- waits with the next accepted row
    (task 153), across a chunk boundary too."""

    @staticmethod
    def _df():
        n = 120
        t = np.arange(float(n))
        t[50:] += 3.0  # a capped gap on row 50
        sess = ["a"] * 80 + ["b"] * 40  # a session change on row 80
        df = frame(n=n, seed=26).with_columns(t=pl.Series(t), s=pl.Series(sess))
        holes = [50, 51, 80]
        return df.with_columns(
            x=pl.when(pl.int_range(pl.len()).is_in(holes)).then(None).otherwise(pl.col("x"))
        )

    # The events have to matter to be seen: the session's blend moves the
    # fit, and both breaks clear the residual autocorrelation's ring.
    EVENTS = dict(session_shrink=0.5, long_half_life=500.0, emit_autocorr=True)

    @classmethod
    def _spec(cls, **kw):
        return spec(gap_cap=2.0, session="s", session_gap=1.0, embargo=6.0, **cls.EVENTS, **kw)

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 50, 51, 81])
    def test_chunking_cannot_move_a_break(self, size):
        df = self._df()
        whole = po.ModelBank([self._spec()]).fit_predict(df)["m"].struct
        bank = po.ModelBank([self._spec()])
        parts = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])[
            "m"
        ].struct
        for field in ("pred_y", "resid_y", "weight_sum", "autocorr_y"):
            a, b = whole.field(field).to_numpy(), parts.field(field).to_numpy()
            assert (np.isnan(a) == np.isnan(b)).all(), field
            fin = np.isfinite(a)
            assert np.array_equal(a[fin], b[fin]), field

    def test_the_state_is_the_matured_rows(self):
        df = self._df()
        t = df["t"].to_numpy()
        delayed = po.ModelBank([self._spec(min_weight=0.0)])
        delayed.fit_predict(df)
        matured = int(np.sum(t[-1] - t >= 6.0))
        plain = po.ModelBank(
            [spec(gap_cap=2.0, session="s", session_gap=1.0, min_weight=0.0, **self.EVENTS)]
        )
        plain.fit_predict(df.head(matured))
        assert delayed.coef("m")["coef"].to_list() == plain.coef("m")["coef"].to_list()
        assert delayed.gram("m")[0]["weight_sum"] == plain.gram("m")[0]["weight_sum"]


T0_NS = 1_704_067_200_000_000_000


def _first_differences(got, want):
    bad = [i for i, (a, b) in enumerate(zip(got, want, strict=True)) if a != b]
    return len(bad), [(i, got[i], want[i]) for i in bad[:3]]


class TestTheReleaseIsExact:
    """docs/PLAN.md task 176: a row is released on the elapsed clock held
    exactly, as a window's edge is (task 175): integer nanoseconds on a
    temporal clock, one subtraction of the raw values on a number clock. The
    countdown in doubles drifted: on rows 1 ms apart under ``embargo="2s"``
    every row was learned 2.001 s after it arrived, a row late. The oracles
    are the raw integer nanoseconds and the raw numbers."""

    @pytest.mark.parametrize("unit", ["ms", "us", "ns"])
    def test_a_row_is_learned_exactly_one_embargo_later(self, unit):
        n = 2_600
        ns = T0_NS + np.arange(n, dtype=np.int64) * 1_000_000
        rng = np.random.default_rng(12)
        df = pl.DataFrame(
            {"t": ns, "x": rng.standard_normal(n), "y": rng.standard_normal(n)}
        ).with_columns(pl.col("t").cast(pl.Datetime("ns")).cast(pl.Datetime(unit)))
        s = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            clock="t",
            gap_cap="1d",
            half_life="1h",
            embargo="2s",
            emit_clocks=True,
        )
        out = po.ModelBank([s]).fit_predict(df)["m"].struct
        learned = out.field("learned_clock").cast(pl.Datetime("ns")).cast(pl.Int64).to_list()
        # The newest row at least 2 s before each row, from the integers.
        newest = np.searchsorted(ns, ns - 2_000_000_000, side="right") - 1
        want = [int(ns[u]) if u >= 0 else None for u in newest]
        assert want[1_999] is None and want[2_000] == int(ns[0]), "2,000 rows back"
        count, first = _first_differences(learned, want)
        assert count == 0, (count, first)

    @pytest.mark.parametrize("embargo", [0.3, 0.5, 1.0, 2.0])
    def test_a_number_clocks_release_is_one_subtraction(self, embargo):
        """Under 0.3 the countdown happened to agree with the subtraction on
        every row of this clock; under 0.5 and 1.0 it learned two rows a row
        late, and under 2.0 two rows a row early (2.3 - 0.3 is
        1.9999999999999998)."""
        n = 400
        t = np.arange(n) / 10.0
        df = frame(n=n, seed=13).with_columns(t=pl.Series(t))
        s = spec(embargo=embargo, gap_cap=1.0, emit_clocks=True)
        learned = po.ModelBank([s]).fit_predict(df)["m"].struct.field("learned_clock").to_list()
        want = []
        for row in range(n):
            back = [u for u in range(row) if t[row] - t[u] >= embargo]
            want.append(float(t[back[-1]]) if back else None)
        if embargo == 0.3:
            # 0.7 - 0.4 is 0.29999999999999993, so the row at 0.4 waits for 0.8.
            assert want[7] == 0.3 and want[8] == 0.5
        count, first = _first_differences(learned, want)
        assert count == 0, (count, first)


class TestTheSurfaces:
    def test_the_lazy_plan_and_the_bank_agree(self):
        df = frame(n=300, seed=11)
        s = spec(embargo=6.0)
        bank = po.ModelBank([s]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([s]).collect()
        assert lazy.select("m").unnest("m").equals(bank.select("m").unnest("m"), null_equal=True)

    def test_the_cli(self, tmp_path, online_cli):
        df = frame(n=300, seed=12)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        s = spec(embargo=6.0)
        out = tmp_path / "out.parquet"
        run_online(online_cli, tmp_path, [s], input=src, output=out, chunk_size=64)

        # `coef` rides the chunk cadence, as everywhere; every value is
        # compared.
        def fields(frame_):
            return frame_.select("m").unnest("m").drop("coef", "support_coef")

        want = fields(po.ModelBank([s]).fit_predict(df))
        assert fields(pl.read_parquet(out)).equals(want, null_equal=True)

        cli_out = tmp_path / "cli.parquet"
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
output = "{cli_out.as_posix()}"
chunk_size = 100

[[specs]]
name = "m"
targets = ["y"]
features = ["x"]
clock = "t"
half_life = {HALFLIFE}
gap_cap = 1e9
min_weight = 3.0
embargo = 6.0
[specs.model]
type = "ewridge"
standardize = false
max_rows_between_solves = 1
""",
            encoding="utf-8",
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        assert fields(pl.read_parquet(cli_out)).equals(want, null_equal=True)


class TestRefusals:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_delay_must_be_finite_and_positive(self, bad):
        with pytest.raises(ValueError, match="embargo"):
            spec(embargo=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
    def test_embargo_refuses_the_same_delays(self, bad):
        df = frame(n=10)
        with pytest.raises(ValueError, match="delay must be finite and > 0"):
            stream.embargo(df, clock="t", delay=bad)

    def test_embargo_needs_the_columns_it_names(self):
        df = frame(n=10)
        with pytest.raises(ValueError, match="no clock column 'nope'"):
            stream.embargo(df, clock="nope", delay=1.0)
        with pytest.raises(ValueError, match="no weight column 'nope'"):
            stream.embargo(df, clock="t", delay=1.0, weight="nope")
        with pytest.raises(ValueError, match="already has a column named 'x'"):
            stream.embargo(df, clock="t", delay=1.0, role="x")

    def test_embargo_names_the_weight_column_it_would_add(self):
        """Review 2026-10-05 (YB10): the role column was checked against the
        frame and the weight column added beside it was not, so a frame that
        already held it died in polars' DuplicateError."""
        df = frame(n=10).with_columns(_online_role_weight=pl.lit(2.0))
        with pytest.raises(ValueError, match="already has a column named '_online_role_weight'"):
            stream.embargo(df, clock="t", delay=1.0)
        # With a weight of its own the column is not added, so it is no clash.
        out = stream.embargo(df.with_columns(w=pl.lit(1.0)), clock="t", delay=1.0, weight="w")
        assert out["_online_role_weight"].to_list() == [2.0] * 20

    @pytest.mark.parametrize("dtype", [pl.Int64, pl.Int32, pl.UInt16])
    def test_a_whole_delay_keeps_an_integer_clocks_dtype(self, dtype):
        """Review 2026-10-05 (YB3): the docstring's own ``delay=5.0`` on an
        integer clock made the learn copy's clock a float, and the merge of
        the two copies died in a SchemaError. A whole delay is the clock's
        own number."""
        df = pl.DataFrame({"t": [0, 1, 2], "x": [1.0, 2.0, 3.0]}, schema_overrides={"t": dtype})
        floats = stream.embargo(df.with_columns(pl.col("t").cast(pl.Float64)), clock="t", delay=2.0)
        for delay in (2, 2.0, np.float32(2.0)):
            out = stream.embargo(df, clock="t", delay=delay)
            assert out["t"].dtype == dtype, delay
            assert out.with_columns(pl.col("t").cast(pl.Float64)).equals(floats), delay

    def test_a_delay_an_integer_clock_cannot_hold_is_refused_by_name(self):
        df = pl.DataFrame({"t": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="^embargo: delay 2.5 is not a whole number") as e:
            stream.embargo(df, clock="t", delay=2.5)
        assert "clock column 't' is Int64" in str(e.value), str(e.value)

    @pytest.mark.parametrize(
        ("dtype", "top"),
        [
            (pl.Int8, 2**7 - 1),
            (pl.UInt8, 2**8 - 1),
            (pl.Int16, 2**15 - 1),
            (pl.Int32, 2**31 - 1),
            (pl.Int64, 2**63 - 1),
            (pl.UInt64, 2**64 - 1),
        ],
    )
    def test_a_clock_within_delay_of_its_dtypes_top_is_refused_not_wrapped(self, dtype, top):
        """Review round 5 (B2): Polars adds integers in the column's width
        and wraps at its top, so an Int8 clock of ``[5, 6, 120]`` with a
        delay of 100 put row 120's learn copy at -36, first in the stream,
        where a bank learned the row before scoring it -- a silent
        look-ahead. The add is widened (Int64, and Int128 past 64 bits) and
        the cast back is strict, so Polars refuses the value when the plan
        runs, naming the column. Measured the same on polars 1.34.0, the
        floor, and 1.44.2."""
        delay = 100
        fits = pl.DataFrame(
            {"t": [top - 115, top - 114, top - delay], "x": [1.0, 2.0, 3.0]},
            schema_overrides={"t": dtype},
        )
        out = stream.embargo(fits, clock="t", delay=delay)
        assert out["t"].dtype == dtype
        assert out["t"].to_list() == sorted(
            [top - 115, top - 114, top - delay, top - 15, top - 14, top]
        )
        wraps = pl.DataFrame(
            {"t": [top - 114, top - delay, top], "x": [1.0, 2.0, 3.0]},
            schema_overrides={"t": dtype},
        )
        with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion from") as e:
            stream.embargo(wraps, clock="t", delay=delay)
        assert "column 't'" in str(e.value), str(e.value)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="conversion from"):
            stream.embargo(wraps.lazy(), clock="t", delay=delay).collect()

    def test_a_zero_delay_is_told_to_leave_it_out(self):
        """Task 160, PB9: the refusal called 0 "the default", where the
        default is no embargo at all."""
        with pytest.raises(ValueError) as exc:
            spec(embargo=0.0)
        assert "embargo must be finite and > 0 (got 0); leave it out for no delay" in str(
            exc.value
        ), str(exc.value)
        assert "default" not in str(exc.value), str(exc.value)


def _break_frame(n=600, at=300):
    """A regime break at row ``at``: the slope flips and the level jumps, on
    a clock one unit a row."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = np.where(np.arange(n) < at, 2.0 * x, -2.0 * x + 8.0) + 0.1 * rng.standard_normal(n)
    return pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y})


class TestDriftUnderAnEmbargo:
    """Task 160, PB1: under ``embargo`` a row's residual reaches the drift
    detector when its label is released, on a replay that writes no output.
    The flag was written only where the row is output, so ``drift_<t>`` was
    never true, and ``drift_action="reset"`` restarted the model with no flag
    at all. The flag goes on the row whose clock released the label that
    tripped the detector; that label's own row is already out."""

    @pytest.mark.parametrize("grid", [False, True], ids=["one instance", "a grid"])
    @pytest.mark.parametrize("action", ["flag", "reset"])
    @pytest.mark.parametrize("delay", [1.0, 5.0])
    def test_a_break_is_flagged_on_the_row_that_released_its_label(self, delay, action, grid):
        df = _break_frame()
        kw = dict(
            emit_drift=True,
            drift_delta=0.5,
            drift_threshold=5.0,
            drift_action=action,
            half_life=[50.0, 100.0] if grid else 50.0,
        )
        fields = po.spec.output_fields(spec(**kw))
        drift = [f for f in fields if f.startswith("drift_")]
        weight = [f for f in fields if f.startswith("weight_sum")]
        assert len(drift) == (2 if grid else 1) and len(weight) == len(drift), fields

        def first_flag(out):
            flags = out.select(pl.any_horizontal(pl.col(drift).fill_null(False))).to_series()
            rows = flags.arg_true().to_list()
            assert rows, "the detector never fired"
            return rows[0]

        plain = po.ModelBank([spec(**kw)]).fit_predict(df)["m"].struct.unnest()
        held = po.ModelBank([spec(embargo=delay, **kw)]).fit_predict(df)["m"].struct.unnest()
        assert first_flag(plain) == 300, "the break is where the data put it"
        # Row 300's label is released by the row `delay` clock units later.
        released_at = 300 + int(delay)
        assert first_flag(held) == released_at
        # The restart takes effect at the replay, before the releasing row is
        # scored: that row reads a fresh model, and the row before it the old.
        for w in weight:
            before, at = held[w][released_at - 1], held[w][released_at]
            assert before > 10.0, (w, before)
            if action == "reset":
                assert at == 0.0, (w, at)
            else:
                assert at > 10.0, (w, at)


class TestEmbargoItself:
    def test_the_shape_and_the_order(self):
        df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
        out = stream.embargo(df, clock="t", delay=2.0)
        assert out.height == 6
        assert out["t"].to_list() == [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
        assert out[stream.ROLE].to_list() == [
            "predict",
            "predict",
            "learn",
            "predict",
            "learn",
            "learn",
        ]
        assert out["_online_role_weight"].to_list() == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        assert out.columns == ["t", "x", "_online_role_weight", stream.ROLE]

    def test_a_role_of_its_own_names_both_columns(self):
        """``role`` names the role column and, with ``_weight`` after it, the
        weight column added beside it (task 278, batch 15: no test passed
        it but to see it refused). The frame is the default one with those
        two columns renamed, and a bank reading them predicts the same."""
        df = frame(n=60, seed=14)
        default = stream.embargo(df, clock="t", delay=3.0)
        mine = stream.embargo(df, clock="t", delay=3.0, role="r")
        assert mine.columns == [*df.columns, "r_weight", "r"]
        assert stream.ROLE not in mine.columns and f"{stream.ROLE}_weight" not in mine.columns
        renamed = default.rename({stream.ROLE: "r", f"{stream.ROLE}_weight": "r_weight"})
        assert mine.equals(renamed)
        a = po.ModelBank([spec(weight=f"{stream.ROLE}_weight")]).fit_predict(default)
        b = po.ModelBank([spec(weight="r_weight")]).fit_predict(mine)
        assert a["m"].equals(b["m"], null_equal=True)
        assert a["m"].struct.field("pred_y").drop_nulls().len() > 40
        # Lazily too.
        assert stream.embargo(df.lazy(), clock="t", delay=3.0, role="r").collect().equals(mine)

    def test_an_existing_weight_is_zeroed_not_replaced(self):
        df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0], "w": [3.0, 4.0]})
        out = stream.embargo(df, clock="t", delay=1.0, weight="w")
        assert out["w"].to_list() == [0.0, 3.0, 0.0, 4.0]
        assert out.columns == ["t", "x", "w", stream.ROLE]

    def test_it_needs_the_clock_order_across_groups(self):
        """Task 120, measured first: the copies are merged by the clock alone,
        so a frame sorted within its groups but not across them comes back
        with a group out of order, which a bank refuses by default; sorted by
        the clock first, every group is in order."""
        by_group = pl.DataFrame(
            {
                "g": ["a"] * 10 + ["b"] * 10,
                "t": [100.0 + i for i in range(10)] + [float(i) for i in range(10)],
                "x": [float(i % 10) for i in range(20)],
                "y": [1.0] * 10 + [2.0] * 10,
            }
        )
        s = spec(group="g", weight="_online_role_weight", gap_cap=5.0)
        out = stream.embargo(by_group, clock="t", delay=3.0)
        assert out.filter(pl.col("g") == "b")["t"].is_sorted() is False
        with pytest.raises(ValueError, match="goes backwards"):
            po.ModelBank([s]).fit_predict(out)
        fixed = stream.embargo(by_group.sort("t", maintain_order=True), clock="t", delay=3.0)
        for g in ("a", "b"):
            assert fixed.filter(pl.col("g") == g)["t"].is_sorted()
        assert po.ModelBank([s]).fit_predict(fixed).height == 40

    def test_it_gives_back_the_kind_of_frame_it_was_given(self):
        df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
        # Task 105, rule 1: a LazyFrame stays lazy, a DataFrame is collected.
        assert isinstance(stream.embargo(df.lazy(), clock="t", delay=1.0), pl.LazyFrame)
        assert isinstance(stream.embargo(df, clock="t", delay=1.0), pl.DataFrame)

    @pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt8, pl.Float32, pl.Boolean])
    def test_a_weight_column_of_any_numeric_dtype_is_zeroed(self, dtype):
        """Review round 4 (YB2): the predict copy's weight was ``w * 0.0``, a
        float, and the learn copy's ``w`` as it was, so an integer weight
        column died in the merge of the two (``SchemaError``: ``('w': f64) !=
        ('w': i64)``), where the bank takes the same column. Both copies
        carry it as ``Float64``, the number the bank reads it as."""
        df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0], "w": [1, 2, 0]})
        df = df.with_columns(pl.col("w").cast(dtype))
        out = stream.embargo(df, clock="t", delay=1.0, weight="w")
        assert out.schema["w"] == pl.Float64
        assert out[stream.ROLE].to_list() == [
            "predict",
            "learn",
            "predict",
            "learn",
            "predict",
            "learn",
        ]
        want = [float(v) for v in df["w"].cast(pl.Float64)]
        assert out["w"].to_list() == [0.0, want[0], 0.0, want[1], 0.0, want[2]]
        s = spec(weight="w", gap_cap=5.0)
        assert po.ModelBank([s]).fit_predict(out.with_columns(y=pl.col("x"))).height == 6

    def test_a_clock_that_is_neither_a_number_nor_a_time_is_refused_by_name(self):
        """Review round 4 (YB15): a String clock failed inside polars
        (``InvalidOperationError: arithmetic on dtypes str and dyn float``)
        while ``po.eval.window_metrics`` refuses the same clock by name, as
        this now does, before any plan is built."""
        df = pl.DataFrame({"t": ["a", "b"], "x": [1.0, 2.0], "b": [True, False]})
        for clock, delay in (("t", 1.0), ("t", "1s"), ("b", 1.0)):
            with pytest.raises(
                TypeError, match=f"^embargo: clock column '{clock}' must be numeric"
            ):
                stream.embargo(df.lazy(), clock=clock, delay=delay)

    def test_a_time_of_day_clock_is_refused_as_a_spec_refuses_it(self):
        """The docstring's refusal (task 239): a ``Time`` column is temporal,
        so it passes the numeric-or-temporal test, and is refused by name as
        a clock that starts again at midnight, a ``ValueError``."""
        df = pl.DataFrame({"t": [time(9), time(10)], "x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="is a time of day, which starts again at midnight"):
            stream.embargo(df, clock="t", delay="1m")

    def test_a_single_use_source_is_warned_about(self):
        """Review round 4 (YB3): the predict and the learn copies are two
        reads of the input, so over a source spent after one read -- what
        ``pl.scan_arrow_c_stream`` builds -- the second read gave nothing and
        half the stream was missing, with no warning, where the bank over the
        same spent scan warns ``ConsumedSourceWarning``. Whether the source
        is spent cannot be seen until the plan runs, so ``embargo`` warns as
        it builds the plan, whenever its input's source is a Python scan, and
        its docstring says the input is read twice."""
        from polars.io.plugins import register_io_source

        spent: list[bool] = []
        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})

        def source(with_columns, predicate, n_rows, batch_size):  # noqa: ANN001, ANN202
            if spent:
                return
            spent.append(True)
            yield rows

        lf = register_io_source(io_source=source, schema=rows.schema)
        with pytest.warns(po.ConsumedSourceWarning, match="^embargo: .*read twice"):
            plan = stream.embargo(lf, clock="t", delay=1.0)
        # What the warning is about: half the doubled stream is missing.
        assert plan.collect().height == 3
        # An input that is not a Python scan is read twice safely, and quietly.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ConsumedSourceWarning)
            assert stream.embargo(rows.lazy(), clock="t", delay=1.0).collect().height == 6
        doc = " ".join((stream.embargo.__doc__ or "").split())
        assert "reads its input twice" in doc


# ---------------------------------------------------------------------------
# Every kind that takes an embargo (task 278, batch 15)
#
# The relations above were held on `ewridge` and `ew_cov`. Here they run over
# every kind that takes an `embargo`, each kind built as
# `test_model_registry.MINIMAL` builds it:
#
# - on a row clock a delay of E rows is the undelayed run E rows behind;
# - on any clock a row is scored as a bank fed only the rows matured by then
#   scores it -- the definition of the delay, with no doubled stream between;
# - the doubled stream, where it is the reference by construction; each kind
#   it is not the reference for is named with its reason, and the reason shown;
# - a delay shorter than every clock step is the undelayed run.
#
# Essentials: `ewridge` and one other kind per relation. Extended: the rest.

EVERY_KIND = pytest.mark.extended(reason="the sweep over every kind that takes an embargo")

#: The kinds that refuse an embargo, and the words each refuses it in:
#: `audit` counts what it is given and learns nothing a delay could hold
#: back; `rcov` needs `group_close`, and a closed group cannot release the
#: rows it still holds.
REFUSE_EMBARGO = {"audit": "embargo", "rcov": "group_close does not work with embargo"}

#: A sweep name that is not a kind: `sgd`'s Huber loss, with the refit of
#: its first rows on as `huber`'s is (task 277).
VARIANT_OF: dict[str, tuple[str, dict]] = {
    "sgd_huber": ("sgd", {"learning_rate": 0.01, "loss": "huber"}),
}
TAKE_EMBARGO = sorted((set(MINIMAL) - set(REFUSE_EMBARGO)) | set(VARIANT_OF))
#: The kinds with no half-life (MINIMAL sets it to None).
NO_HALF_LIFE = frozenset(
    k for k, v in MINIMAL.items() if "half_life" in v and v["half_life"] is None
)
#: `seqtest` takes no weight: every row it learns is one trial.
NO_WEIGHT = frozenset({"seqtest"})
#: The fields laid out like `coef`, written after the row's update.
AFTER_THE_UPDATE = frozenset({"coef", "support_coef", "se_coef", "se_coef_hc0", "se_coef_hac"})


def kind_frame(kind: str, *, n: int = 400, steps=None, seed: int = 5) -> pl.DataFrame:
    """Two features, a target with a level shift halfway, uneven weights;
    the target binary for `ftrl` and a label for `ew_class`. ``steps`` puts a
    clock ``t`` under it (the first step is the first row's clock)."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    y = (
        1.0
        + 2.0 * x0
        - x1
        + 0.3 * rng.standard_normal(n)
        + np.where(np.arange(n) > n // 2, 3.0, 0.0)
    )
    df = pl.DataFrame({"x0": x0, "x1": x1, "y": y, "w": rng.uniform(0.2, 3.0, n)})
    if steps is not None:
        df = df.with_columns(t=pl.Series(np.cumsum(steps), dtype=pl.Float64))
    if kind == "ftrl":
        df = df.with_columns(y=(pl.col("y") > 1).cast(pl.Float64))
    if kind == "ew_class":
        df = df.with_columns(y=pl.when(pl.col("y") > 1).then(pl.lit("a")).otherwise(pl.lit("b")))
    return df


def irregular_steps(n: int, top: int, seed: int = 3) -> np.ndarray:
    """Whole clock steps of 1 to ``top``, the first row's 0: sums exact."""
    steps = np.random.default_rng(seed).integers(1, top + 1, n).astype(float)
    steps[0] = 0.0
    return steps


def kind_spec(kind: str, **kw) -> dict:
    """``kind`` as MINIMAL builds it on these frames, ``kw`` on top; a key
    given None is left out."""
    builder, own = VARIANT_OF.get(kind, (kind, {}))
    a: dict[str, object] = {"targets": ["y"], "features": ["x0", "x1"], "half_life": 50.0}
    a.update(MINIMAL[builder])
    a.update(own)
    if builder in ("kmeans", "hmm"):
        # Seeded inside the stream's first rows, so most rows are scored.
        a["warm_rows"] = 20
    a.update(kw)
    return getattr(po.spec, builder)("m", **{k: v for k, v in a.items() if v is not None})


def has_coef(kind: str) -> bool:
    return "coef" in po.spec.output_fields(kind_spec(kind))


def unnested(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return po.ModelBank([spec], threads=1).fit_predict(df).select("m").unnest("m")


def kind_params(names, essential):
    """One param per name, extended unless the name is ``essential``."""
    return [pytest.param(n, id=n, marks=() if n in essential else (EVERY_KIND,)) for n in names]


def same(a: pl.Series, b: pl.Series) -> bool:
    return a.equals(b, null_equal=True)


def test_the_kinds_left_out_refuse_an_embargo():
    for kind, words in REFUSE_EMBARGO.items():
        with pytest.raises(ValueError, match=words):
            kind_spec(kind, embargo=5.0)
    # And every other kind takes one.
    for kind in TAKE_EMBARGO:
        kind_spec(kind, embargo=5.0)


# -- the shift on a row clock ------------------------------------------------

#: Each kind's fields computed from its state alone, before the row: on a
#: row clock the delayed run's row ``i`` reads the state the undelayed run's
#: row ``i - E + 1`` read, since both have learned rows ``0 .. i - E``. Every
#: kind's ``weight_sum`` and ``withheld_reason`` are such fields too. The
#: rest read the row itself (a prediction, a distance, a statistic of the
#: row), and ``settled_frac`` reads the clock, which held rows advance as
#: learned ones do.
STATE_READS: dict[str, frozenset[str]] = {
    "bocpd": frozenset({"run_mode", "run_mean", "pred_x0", "pred_x1"}),
    "corrchange": frozenset({"crit", "since_flag"}),
    "deco": frozenset({"rho"}),
    "ew_cov": frozenset({"mean_x0", "mean_x1", "std_x0", "std_x1", "corr_x0_x1"}),
    "hmm": frozenset(
        {"filtered_state0", "filtered_state1", "predicted_state0", "predicted_state1", "state"}
    ),
    # Holt reads no features: its forecast is the state's.
    "holt": frozenset({"pred_y"}),
    "lasso": frozenset({"penalty_selected_y"}),
    "micro": frozenset({"n_clusters", "n_micro"}),
    "seqtest": frozenset({"log_e_pos_y", "log_e_neg_y", "n_pos_y", "n_neg_y"}),
}
ALWAYS_STATE = frozenset({"weight_sum", "withheld_reason"})
#: The state reads that also read the clock from the last learned row to
#: the row (task 278, D11): a held row sits ``E`` rows after it, where the
#: undelayed run's row sits one, so these part from the shift by
#: construction. `holt` extrapolates its trend over that clock; `hmm` ages
#: its transition counts over it, which moves nothing with no decay.
#: `TestScoredAsTheMaturedRowsWouldScoreIt` holds them to their definition.
STEP_READS: dict[str, frozenset[str]] = {
    "holt": frozenset({"pred_y"}),
    "hmm": frozenset({"predicted_state0", "predicted_state1"}),
}
SHIFT_ROWS = 5
SHIFT_CASES = [
    pytest.param(
        kind,
        h,
        id=f"{kind}-h{h}",
        marks=() if kind in ("ewridge", "kmeans") and h == 50.0 else (EVERY_KIND,),
    )
    for kind in TAKE_EMBARGO
    for h in ((None,) if kind in NO_HALF_LIFE else (math.inf, 50.0))
]


class TestTheShiftOnARowClock:
    """Without a clock column a unit of delay is a row, so `embargo=E` learns
    row ``i - E`` at row ``i``: the same rows, in the same order, at the same
    steps as the undelayed run, ``E`` rows later. So the fit written after
    row ``i`` (``coef`` and the fields laid out like it) is the undelayed
    run's after row ``i - E``, and a field read from the state before row
    ``i`` is the undelayed run's before row ``i - E + 1``, but for a field
    that also reads the clock since the last learned row, ``E`` rows for the
    held row and one for the undelayed run's (`STEP_READS`). The probe of
    2026-10-25 found this to the bit on fourteen kinds; here it is held on
    every kind that takes an embargo, at no decay and at a half-life, with
    uneven weights."""

    @pytest.mark.parametrize(("kind", "half_life"), SHIFT_CASES)
    def test_a_delay_of_e_rows_is_the_undelayed_run_e_rows_behind(self, kind, half_life):
        e = SHIFT_ROWS
        df = kind_frame(kind)
        kw: dict[str, object] = {"weight": None if kind in NO_WEIGHT else "w"}
        if kind not in NO_HALF_LIFE:
            kw["half_life"] = half_life
        if has_coef(kind):
            kw["coef_every"] = 0
        plain = unnested(kind_spec(kind, **kw), df)
        held = unnested(kind_spec(kind, embargo=float(e), **kw), df)
        n = df.height
        # `marginal` writes `weight_sum` alone; its table is read from the
        # state.
        state = (ALWAYS_STATE & set(held.columns)) | STATE_READS.get(kind, frozenset())
        assert "weight_sum" in state and state <= set(held.columns), state - set(held.columns)
        for c in held.columns:
            if c in AFTER_THE_UPDATE:
                assert same(held[c][e:], plain[c][: n - e]), c
                # Before the first release the fit is the unlearned one.
                assert all(same(held[c][:1], held[c][r : r + 1]) for r in range(e)), c
            elif c == "settled_frac":
                assert same(held[c], plain[c]), c
            elif c in STEP_READS.get(kind, frozenset()):
                # Read over E rows where the undelayed row reads one: parts,
                # unless the step moves nothing (`hmm` with no decay).
                ages_only = kind == "hmm" and half_life == math.inf
                assert same(held[c][e - 1 :], plain[c][: n - e + 1]) == ages_only, c
            elif c in state:
                assert same(held[c][e - 1 :], plain[c][: n - e + 1]), c
            else:
                # The row's own reading: a field that matched here would be
                # a state read this table leaves out.
                assert not same(held[c][e - 1 :], plain[c][: n - e + 1]), (
                    f"{kind}: {c} reads only the state; add it to STATE_READS"
                )
        # The delay took effect.
        assert not same(held["weight_sum"], plain["weight_sum"])
        if has_coef(kind):
            assert held["coef"].null_count() < n // 2


# -- scored as the matured rows would score it -------------------------------

#: The delay on the clock of these tests, and the stream: steps of 1 to 5
#: clock units, so a row waits for 3 to 15 others.
MATURED_DELAY, MATURED_ROWS = 15.0, 300

#: The cases: each kind as MINIMAL builds it, and a few with a part of the
#: model the default leaves off or on (the error inflation, off by default;
#: ``sgd``'s and ``pa``'s scaler, on by default).
MATURED_CASES: dict[str, tuple[str, dict]] = {
    **{k: (k, {}) for k in TAKE_EMBARGO},
    "kalman-error-inflation": ("kalman", {"emit_error_inflation": True}),
    "ewridge-error-inflation": ("ewridge", {"emit_error_inflation": True}),
    "rls-error-inflation": ("rls", {"emit_error_inflation": True}),
    "sgd-unscaled": ("sgd", {"standardize": False}),
    "pa-unscaled": ("pa", {"standardize": False}),
}

#: The cases whose score reads the clock since the last row the model
#: learned, and the fields it moves. Under an embargo that clock is the held
#: rows' steps and the row's own; until task 278 (D11) a held row was scored
#: over its own step alone, as if it sat one step after the last learned
#: row: `holt` extrapolated its trend one step where the row was several
#: ahead; `hmm` aged its transition counts by one step; `micro` decayed its
#: summaries' weights by one step before deciding where the row goes; `sgd`
#: and `pa` folded the row into their scaler's moments at that step while
#: the scaler warmed up; `kalman` added one step's drift to the row's
#: estimation variance. Every other kind's score reads no step. A delay
#: shorter than every step (below) cannot tell the two apart: there the
#: row's own step is the clock since the last learned row.
READS_THE_STEP: dict[str, tuple[str, ...]] = {
    "holt": ("pred_y", "resid_y"),
    "hmm": ("predicted_state0", "predicted_state1", "loglik"),
    "micro": ("micro_id", "outlier"),
    "sgd": ("pred_y", "resid_y"),
    # sgd's scaler, under its Huber loss (task 277).
    "sgd_huber": ("pred_y", "resid_y"),
    "pa": ("pred_y", "resid_y"),
    "kalman-error-inflation": ("error_inflation_y",),
}


def matured_case(case: str, **kw) -> dict:
    kind, extra = MATURED_CASES[case]
    return kind_spec(kind, **{"clock": "t", "gap_cap": 1e9, **extra, **kw})


def matured_frame(case: str) -> pl.DataFrame:
    kind = MATURED_CASES[case][0]
    return kind_frame(kind, n=MATURED_ROWS, steps=irregular_steps(MATURED_ROWS, 5))


def matured_rows(t: np.ndarray, i: int, delay: float) -> int:
    """How many rows have matured when row ``i`` is scored: those at least
    ``delay`` behind it on the clock."""
    return int(np.searchsorted(t, t[i] - delay, side="right"))


def score_as_matured(
    case: str, df: pl.DataFrame, *, own_step: bool = False, **kw
) -> dict[int, dict]:
    """Each row as a plain bank fed only the rows matured by then scores it
    (`ModelBank.predict`, which learns nothing), walking forward, by row; a
    row with no matured row yet has nothing to be scored by and is left
    out. With ``own_step`` the row is scored at the clock one of its own
    steps after the last matured row instead. ``kw`` goes to the spec."""
    t = df["t"].to_numpy()
    bank = po.ModelBank([matured_case(case, **kw)], threads=1)
    fed, got = 0, {}
    for i in range(df.height):
        k = matured_rows(t, i, MATURED_DELAY)
        if k > fed:
            bank.fit_predict(df.slice(fed, k - fed))
            fed = k
        if k == 0:
            continue
        row = df.slice(i, 1)
        if own_step:
            row = row.with_columns(t=pl.lit(t[k - 1] + (t[i] - t[i - 1])))
        got[i] = bank.predict(row).select("m").unnest("m").row(0, named=True)
    return got


def parted(held: pl.DataFrame, oracle: dict[int, dict]) -> dict[str, list[int]]:
    """The rows on which each field read before the row differs from the
    oracle's, over the rows the oracle scored. NaN equals NaN."""
    out: dict[str, list[int]] = {}
    for i, want in oracle.items():
        for c, v in want.items():
            if c in NOT_SCORED:
                continue
            got = held[c][i]
            nan = isinstance(got, float) and isinstance(v, float) and got != got and v != v
            if got != v and not nan:
                out.setdefault(c, []).append(i)
    return out


#: Read before the row, the fields a bank's `predict` writes and the
#: delayed row is compared on; `settled_frac` counts the held rows' clock
#: (`settled_frac`'s own tests), and the rest are written after the update.
NOT_SCORED = AFTER_THE_UPDATE | {"settled_frac"}


def _matured_params():
    out = []
    for case in MATURED_CASES:
        marks: list = [] if case in ("ewridge", "ew_cov", "holt") else [EVERY_KIND]
        out.append(pytest.param(case, id=case, marks=marks))
    return out


class TestScoredAsTheMaturedRowsWouldScoreIt:
    """What an embargo means, with nothing between: a row is scored as a
    bank fed only the rows whose delay had passed would score it. Each row
    of the delayed run against `ModelBank.predict` on a plain bank fed
    exactly the matured rows, every field read before the row, to the bit,
    on an irregular clock where a row waits for 3 to 15 others. No doubled
    stream: its lessons sit `delay` after their rows, which is not the
    clock the rows were learned at (below)."""

    @pytest.mark.parametrize("case", _matured_params())
    def test_a_held_row_is_scored_as_the_matured_rows_would_score_it(self, case):
        df = matured_frame(case)
        held = unnested(matured_case(case, embargo=MATURED_DELAY), df)
        oracle = score_as_matured(case, df)
        assert len(oracle) > MATURED_ROWS * 0.9, len(oracle)
        assert set(oracle[MATURED_ROWS - 1]) == set(held.columns)
        bad = parted(held, oracle)
        assert not bad, {c: (len(r), r[:4]) for c, r in bad.items()}

    @pytest.mark.parametrize(
        "case",
        [pytest.param(c, id=c, marks=() if c == "holt" else (EVERY_KIND,)) for c in READS_THE_STEP],
    )
    def test_the_rows_own_step_would_move_these_fields(self, case):
        """The relation above is not met by a score that reads no step: scored
        as the matured bank scores it at the clock one of the row's own steps
        after the last matured row -- the step a held row was read over until
        task 278 (D11) -- each case parts from the delayed run on the fields
        `READS_THE_STEP` names, and only those."""
        df = matured_frame(case)
        held = unnested(matured_case(case, embargo=MATURED_DELAY), df)
        moved = parted(held, score_as_matured(case, df, own_step=True))
        assert set(moved) == set(READS_THE_STEP[case]), moved

    @pytest.mark.parametrize("case", kind_params(READS_THE_STEP, essential={"holt"}))
    def test_a_capped_gap_is_scored_as_the_matured_rows_would_score_it(self, case):
        """Under a `gap_cap` the clock since the last learned row is a gap
        like any other, capped whole: the held rows' steps, each capped, and
        the row's own, summed and capped, are the cap of the raw sum the
        matured bank reads, since a step capped alone is the cap already.
        Steps of 1 to 5 under a cap of 3: single steps are capped, and so is
        nearly every held row's gap."""
        df = matured_frame(case)
        held = unnested(matured_case(case, embargo=MATURED_DELAY, gap_cap=3.0), df)
        bad = parted(held, score_as_matured(case, df, gap_cap=3.0))
        assert not bad, {c: (len(r), r[:4]) for c, r in bad.items()}
        # The cap acts: summed uncapped, the score parts.
        loose = parted(held, score_as_matured(case, df, gap_cap=1e9))
        assert set(READS_THE_STEP[case]) & set(loose), loose

    @pytest.mark.parametrize("case", kind_params(READS_THE_STEP, essential={"holt"}))
    def test_predict_against_a_bank_holding_rows_is_the_matured_banks(self, case):
        """`ModelBank.predict` against a delayed bank holding rows reads the
        clock since the last row it learned, as its own held rows are scored:
        the matured bank's `predict`, to the bit."""
        df = matured_frame(case)
        t = df["t"].to_numpy()
        for m in (60, 150, 240):
            bank = po.ModelBank([matured_case(case, embargo=MATURED_DELAY)], threads=1)
            bank.fit_predict(df.head(m))
            k = matured_rows(t, m - 1, MATURED_DELAY)
            assert m - k >= 3, (m, k)
            plain = po.ModelBank([matured_case(case)], threads=1)
            plain.fit_predict(df.head(k))
            for i in range(m, m + 10):
                row = df.slice(i, 1)
                a = bank.predict(row).select("m").unnest("m").row(0, named=True)
                b = plain.predict(row).select("m").unnest("m").row(0, named=True)
                bad = [
                    c
                    for c in a
                    if c not in NOT_SCORED and a[c] != b[c] and not (a[c] != a[c] and b[c] != b[c])
                ]
                assert not bad, (m, i, bad)

    @pytest.mark.parametrize("case", kind_params(READS_THE_STEP, essential={"holt"}))
    def test_the_held_clock_crosses_chunks_and_a_save(self, case, tmp_path):
        """The step reads the held rows' clock, which the stream keeps
        across a chunk boundary and a save (hard rules 3 and 5): chunked into
        1, 7 and 64 rows, and saved and loaded at each 64-row boundary, the
        delayed run is the one-chunk run, every field to the bit."""
        df = matured_frame(case)
        spec_ = matured_case(case, embargo=MATURED_DELAY)
        if has_coef(MATURED_CASES[case][0]):
            spec_ = matured_case(case, embargo=MATURED_DELAY, coef_every=0)
        whole = po.ModelBank([spec_], threads=1).fit_predict(df)
        for size in (1, 7, 64):
            bank = po.ModelBank([spec_], threads=1)
            parts = []
            for start in range(0, df.height, size):
                parts.append(bank.fit_predict(df.slice(start, size)))
                if size == 64:
                    bank.save(tmp_path / "b.state")
                    bank = po.ModelBank.load(tmp_path / "b.state")
            assert pl.concat(parts).equals(whole, null_equal=True), size

    def test_a_fractional_clock_is_scored_within_rounding(self):
        """On a clock whose steps are not whole, the held rows' steps summed
        and the matured bank's one subtraction round differently, so the
        relation holds to the last bits rather than to the bit."""
        df = matured_frame("holt").with_columns(pl.col("t") * 0.1)
        t = df["t"].to_numpy()
        delay = MATURED_DELAY * 0.1
        held = unnested(matured_case("holt", embargo=delay), df)["pred_y"].to_numpy()
        plain = po.ModelBank([matured_case("holt")], threads=1)
        fed, want = 0, np.full(df.height, np.nan)
        for i in range(df.height):
            k = matured_rows(t, i, delay)
            if k > fed:
                plain.fit_predict(df.slice(fed, k - fed))
                fed = k
            if k:
                want[i] = plain.predict(df.slice(i, 1))["m"].struct.field("pred_y")[0]
        ok = np.isfinite(want)
        assert ok.sum() > df.height * 0.9
        np.testing.assert_allclose(held[ok], want[ok], rtol=1e-12, atol=0)


# -- the doubled stream on every kind ----------------------------------------

#: The kinds the doubled stream is the reference for, each with what it
#: needs: every field but `settled_frac` (see `TestItIsTheDoubledStream`)
#: to the bit, on a clock of one unit a row. Their state moves only on the
#: rows they learn, by the steps between them, and their score reads no
#: clock past the last learned row; the doubled stream's lessons sit `delay`
#: after their rows, all by the same amount, which such a model cannot see.
DOUBLED: dict[str, dict] = {
    "bocpd": {},
    "corrchange": {},
    "deco": {},
    "ew_class": {},
    "ew_cov": {},
    "ewridge": {},
    "ftrl": {},
    "huber": {},
    "kmeans": {},
    "lasso": {},
    "marginal": {},
    "quantile": {},
    # With no decay; at a half-life, `NOT_DOUBLED` says why not.
    "rls": {"half_life": math.inf},
    # With the scaler off; on, `NOT_DOUBLED` says why not.
    "sgd": {"standardize": False},
    "pa": {"standardize": False},
    "sgd_huber": {"standardize": False},
}

#: The kinds it is not the reference for, why, and how each is shown to
#: part from it (`test_each_kind_left_out_parts_for_its_reason`): the
#: spec's extra keywords, the delay, and a field that parts. The first four
#: reasons are one: the score reads the clock since the last learned row,
#: and the doubled stream's lesson sits `delay` later than its row.
NOT_DOUBLED: dict[str, tuple[str, dict, float, str]] = {
    "holt": (
        "forecasts over the clock since its last observation, so a lesson that "
        "sits `delay` late forecasts `delay` short",
        {},
        5.0,
        "pred_y",
    ),
    "hmm": (
        "ages its transition counts over the clock from the last learned row",
        {},
        5.0,
        "predicted_state0",
    ),
    "micro": (
        "decays its summaries over the clock from the last learned row before it places the row",
        {},
        1.0,
        "micro_id",
    ),
    "sgd-scaled": (
        "folds the row into its scaler's moments over that clock while the scaler warms up",
        {},
        5.0,
        "pred_y",
    ),
    "pa-scaled": (
        "folds the row into its scaler's moments over that clock while the scaler warms up",
        {},
        5.0,
        "pred_y",
    ),
    "rls-decayed": (
        "its prior `delta I` decays with the clock from the stream's first row, and "
        "the doubled stream's first lesson comes `delay` after that row",
        {"half_life": 50.0},
        5.0,
        "pred_y",
    ),
}
#: Left out for a reason no stream shows: `seqtest` takes no weight, so the
#: doubled stream's predict rows, a weight of 0, cannot be written for it;
#: and `kalman` until its fix on another branch (task 278's kalman batch: P
#: in units of the noise, a diffuse intercept on the first row that observes
#: a target) merges, since that fix changes how its first rows are read, the
#: doubled stream's leading predict rows among them. At this commit kalman
#: matches the doubled stream to the bit (probe 2026-10-25).
NOT_DOUBLED_AT_ALL = {"seqtest": "weight does not apply to seqtest", "kalman": None}


def doubled_against_native(kind: str, delay: float, **kw) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The native run and the bank over the doubled stream's predict rows,
    on a clock of one unit a row, ``coef`` on every row where there is one."""
    df = kind_frame(kind, steps=np.r_[0.0, np.ones(399)])
    if has_coef(kind):
        kw.setdefault("coef_every", 0)
    native = unnested(kind_spec(kind, clock="t", gap_cap=1e9, embargo=delay, **kw), df)
    frame_ = stream.embargo(df, clock="t", delay=delay)
    spec_ = kind_spec(kind, clock="t", gap_cap=1e9, weight=stream.ROLE + "_weight", **kw)
    oracle = (
        po.ModelBank([spec_], threads=1)
        .fit_predict(frame_)
        .filter(pl.col(stream.ROLE) == "predict")
        .select("m")
        .unnest("m")
    )
    return native, oracle


class TestTheDoubledStreamOnEveryKind:
    @pytest.mark.parametrize("delay", [1.0, 5.0])
    @pytest.mark.parametrize("kind", kind_params(DOUBLED, essential={"kmeans"}))
    def test_every_field_matches_the_doubled_stream(self, kind, delay):
        native, oracle = doubled_against_native(kind, delay, **DOUBLED[kind])
        assert native.columns == oracle.columns
        bad = [c for c in native.columns if c != "settled_frac" and not same(native[c], oracle[c])]
        assert not bad, bad
        assert native["weight_sum"].null_count() < native.height

    @pytest.mark.parametrize("case", kind_params(NOT_DOUBLED, essential={"holt"}))
    def test_each_kind_left_out_parts_for_its_reason(self, case):
        """The reason a kind is left out is real: run as `NOT_DOUBLED` says,
        the field it names parts from the doubled stream."""
        _, extra, delay, field = NOT_DOUBLED[case]
        kind = case.split("-")[0]
        native, oracle = doubled_against_native(kind, delay, **extra)
        assert not same(native[field], oracle[field]), case
        # The weight is read from the learned rows alone, and does not part.
        assert same(native["weight_sum"], oracle["weight_sum"]), case

    def test_seqtest_takes_no_weight_to_zero(self):
        with pytest.raises(ValueError, match=NOT_DOUBLED_AT_ALL["seqtest"]):
            kind_spec("seqtest", weight="w")

    def test_every_kind_is_held_or_left_out_with_its_reason(self):
        named = set(DOUBLED) | {c.split("-")[0] for c in NOT_DOUBLED} | set(NOT_DOUBLED_AT_ALL)
        assert named == set(TAKE_EMBARGO), named ^ set(TAKE_EMBARGO)


# -- a delay shorter than every clock step -----------------------------------

#: The residual diagnostics, on wherever a kind scores a target, so the
#: residual path is in the relation too.
DIAGNOSTICS = dict(
    emit_sigma=True,
    emit_metrics=True,
    conformal=0.9,
    emit_autocorr=True,
    emit_drift=True,
    drift_threshold=20.0,
    emit_calibration=True,
    emit_breaks=True,
    # The target looks no row ahead (the breaks' long-run variance needs to
    # be told, on a clock where the embargo is not in rows).
    horizon_rows=0,
)
SHORT = 0.5
#: The drift flags' prefix.
DRIFT = "drift_"
#: The kinds the level shift halfway does not trip the detector on: `holt`
#: follows a level, and `ftrl` scores a probability against a 0/1 label.
QUIET = frozenset({"holt", "ftrl"})


def scores_a_target(kind: str) -> bool:
    return any(f.startswith("pred_y") for f in po.spec.output_fields(kind_spec(kind)))


def short_frame(kind: str) -> pl.DataFrame:
    """Steps of 1 to 3 clock units: every one longer than ``SHORT``."""
    return kind_frame(kind, steps=irregular_steps(400, 3))


class TestADelayShorterThanEveryStep:
    """A delay shorter than every clock step releases each row at the next,
    before that row is scored: the model has learned every row before the
    one it scores, as without a delay. So the run is the undelayed run, every
    field to the bit, but for the two things a release does where it lands:
    the fit written after row ``i`` has not learned row ``i`` yet, so `coef`
    is the undelayed run's one row back; and a drift flag goes on the row
    whose clock released the label that tripped it (`TestDriftUnderAnEmbargo`),
    one row on. The doubled stream with such a delay is the undelayed run too
    (with no decay, below). Nothing shorter can be asked: a delay of 0 is
    refused, with leave-it-out as the reason."""

    @pytest.mark.parametrize("kind", kind_params(TAKE_EMBARGO, essential={"ewridge", "holt"}))
    def test_it_is_the_undelayed_run(self, kind):
        df = short_frame(kind)
        kw: dict[str, object] = {"clock": "t", "gap_cap": 1e9}
        if scores_a_target(kind):
            kw.update(DIAGNOSTICS)
        if has_coef(kind):
            kw["coef_every"] = 0
        plain = unnested(kind_spec(kind, **kw), df)
        held = unnested(kind_spec(kind, embargo=SHORT, **kw), df)
        assert held.columns == plain.columns
        flags = 0
        for c in held.columns:
            if c in AFTER_THE_UPDATE:
                assert same(held[c][1:], plain[c][:-1]), c
            elif c.startswith(DRIFT):
                assert held[c][0] in (None, False), c
                assert same(held[c][1:], plain[c][:-1]), c
                flags += int(plain[c].sum() or 0)
            else:
                assert same(held[c], plain[c]), c
        if scores_a_target(kind) and kind not in QUIET:
            # The level shift halfway trips the detector: the flag's move is
            # seen, not assumed.
            assert flags > 0, kind

    @pytest.mark.parametrize(
        "kind",
        kind_params(
            [k for k in TAKE_EMBARGO if k not in ("holt", "seqtest", "kalman")],
            essential={"ewridge", "hmm"},
        ),
    )
    def test_the_doubled_stream_is_the_undelayed_run_with_no_decay(self, kind):
        """Each lesson sits ``SHORT`` after its row and before the next row,
        so the bank over the doubled stream's predict rows is the undelayed
        bank, every field but `settled_frac`, `coef` (the predict row's
        `coef` has not learned its own row) and the drift flags, with no
        decay. The detector sums each residual's excess times the clock step
        of the row that carries it, and a lesson's step is the ``SHORT``
        since its own predict row, not the row's: its flag comes later, on a
        lesson row. Under a decay the
        step split at the lesson multiplies two decay factors where the
        undelayed run multiplies one, which agree to rounding only, and the
        kinds `NOT_DOUBLED` names read the clock the lesson moved. Holt is the
        one kind that parts with no decay: its lesson ``SHORT`` late forecasts
        the next row ``SHORT`` short. `seqtest` takes no weight, and `kalman`
        waits for its fix (`NOT_DOUBLED_AT_ALL`)."""
        df = short_frame(kind)
        kw: dict[str, object] = {"clock": "t", "gap_cap": 1e9}
        if kind not in NO_HALF_LIFE:
            kw["half_life"] = math.inf
        if scores_a_target(kind):
            kw.update(DIAGNOSTICS)
        plain = unnested(kind_spec(kind, weight="w", **kw), df)
        doubled_ = stream.embargo(df, clock="t", delay=SHORT, weight="w")
        oracle = (
            po.ModelBank([kind_spec(kind, weight="w", **kw)], threads=1)
            .fit_predict(doubled_)
            .filter(pl.col(stream.ROLE) == "predict")
            .select("m")
            .unnest("m")
        )
        skip = AFTER_THE_UPDATE | {"settled_frac"}
        bad = [
            c
            for c in plain.columns
            if c not in skip and not c.startswith(DRIFT) and not same(plain[c], oracle[c])
        ]
        assert not bad, bad

    def test_holt_parts_from_the_doubled_stream_even_with_no_decay(self):
        df = short_frame("holt")
        kw = dict(clock="t", gap_cap=1e9, half_life=math.inf, weight="w")
        plain = unnested(kind_spec("holt", **kw), df)
        oracle = (
            po.ModelBank([kind_spec("holt", **kw)], threads=1)
            .fit_predict(stream.embargo(df, clock="t", delay=SHORT, weight="w"))
            .filter(pl.col(stream.ROLE) == "predict")
            .select("m")
            .unnest("m")
        )
        assert same(plain["weight_sum"], oracle["weight_sum"])
        assert not same(plain["pred_y"], oracle["pred_y"])
