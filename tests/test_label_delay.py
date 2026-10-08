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

import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online
from polars_online import stream

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
"""
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
