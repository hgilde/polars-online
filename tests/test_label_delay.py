"""E47: `label_delay`, and the doubled stream it replaces.

The claim is that a row is *scored* where it sits and *learned from* only
once its label would really have been known. Two things have to be earned:

1. It is the doubled stream, not something like it. `po.prep.embargo` builds
   the recipe E47 names -- every row twice, a zero-weight prediction at `t`
   and a lesson at `t + delay` -- and the native path is held against it
   **bit for bit**, not approximately.
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
from polars_online import prep

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
    kw.setdefault("halflife", HALFLIFE)
    kw.setdefault("max_dclock", 1e9)
    kw.setdefault("min_periods", 3.0)
    kw.setdefault("standardize", False)
    kw.setdefault("max_rows_between_solves", 1)
    return po.spec.ewridge(name, targets=["y"], features=list(features), **kw)


def doubled(df, delay, **kw):
    """The same fit through `po.prep.embargo`: the oracle E47 names."""
    frame_ = prep.embargo(df, clock="t", delay=delay).collect()
    bank = po.ModelBank([spec(weight=prep.ROLE + "_weight", **kw)])
    out = bank.fit_predict(frame_)
    return out.filter(pl.col(prep.ROLE) == "predict"), bank


class TestItIsTheDoubledStream:
    @pytest.mark.parametrize("delay", [1.0, 5.0, 40.0])
    def test_every_field_matches_the_oracle_to_the_bit(self, delay):
        df = frame()
        native = po.ModelBank([spec(label_delay=delay)]).fit_predict(df)
        oracle, _ = doubled(df, delay)
        a, b = native["m"].struct, oracle["m"].struct
        for field in po.spec.output_fields(spec(label_delay=delay)):
            if field == "coef":
                continue
            x, y = a.field(field).to_numpy(), b.field(field).to_numpy()
            assert (np.isnan(x) == np.isnan(y)).all(), field
            fin = np.isfinite(x)
            assert np.array_equal(x[fin], y[fin]), (field, np.max(np.abs(x[fin] - y[fin])))

    def test_the_weighted_diagnostics_match_too(self):
        """sigma, resid_z, the metrics and the conformal interval are all fed
        from the residual and all respect the row weight, so a delay that fed
        them early would show here."""
        df = frame(n=600, seed=1)
        kw = dict(
            emit_sigma=True,
            emit_resid_z=True,
            emit_metrics=True,
            conformal=0.9,
        )
        native = po.ModelBank([spec(label_delay=7.0, **kw)]).fit_predict(df)
        oracle, _ = doubled(df, 7.0, **kw)
        a, b = native["m"].struct, oracle["m"].struct
        checked = 0
        for field in po.spec.output_fields(spec(label_delay=7.0, **kw)):
            if field == "coef":
                continue
            x, y = a.field(field).to_numpy(), b.field(field).to_numpy()
            assert (np.isnan(x) == np.isnan(y)).all(), field
            fin = np.isfinite(x)
            assert np.array_equal(x[fin], y[fin]), field
            checked += 1
        assert checked >= 9, f"only {checked} fields compared"

    def test_the_weight_free_diagnostics_are_where_the_two_differ(self):
        """`resid_quantiles`, `emit_autocorr` and `emit_drift` do not take a
        row weight -- a P2 estimator counts samples, not weight -- so in the
        doubled stream a zero-weight predict row feeds them as much as its
        learn copy does, and every residual lands twice. `label_delay` feeds
        them once, when the label matures. The native path is the right one,
        and this pins the difference so it cannot be discovered by surprise."""
        df = frame(n=600, seed=1)
        kw = dict(emit_drift=True, emit_autocorr=True, resid_quantiles=[0.5])
        native = po.ModelBank([spec(label_delay=7.0, **kw)]).fit_predict(df)
        oracle, _ = doubled(df, 7.0, **kw)
        a = native["m"].struct.field("absresid_q0.5_y").to_numpy()
        b = oracle["m"].struct.field("absresid_q0.5_y").to_numpy()
        assert not np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
        # And the oracle warms up sooner, because a P2 estimator needs five
        # samples and the doubled stream hands it two per row.
        assert np.isfinite(a).sum() < np.isfinite(b).sum()

    def test_the_state_is_the_state_of_the_matured_rows(self):
        """A delayed bank at the end of a stream is bit-for-bit the bank a
        plain one would be, fed only the rows whose labels had matured. Not
        the doubled stream's bank: that frame runs `delay` clock units past
        the input, so its tail lessons have landed and the native one's have
        not."""
        n, delay = 500, 9
        df = frame(n=n, seed=2)
        native = po.ModelBank([spec(label_delay=float(delay))])
        native.fit_predict(df)
        matured = po.ModelBank([spec()])
        matured.fit_predict(df.head(n - delay))
        assert native.coef("m")["coef"].to_list() == matured.coef("m")["coef"].to_list()
        assert native.gram("m")[0]["n_eff"] == matured.gram("m")[0]["n_eff"]


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
        common = dict(features=["noise"], halflife=200.0)
        leak = _oos_r2(df, spec(**common))
        clean = _oos_r2(df, spec(label_delay=float(h), **common))
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
        delayed = _oos_r2(df, spec(halflife=200.0, label_delay=float(h)))
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
        s = spec(label_delay=11.0, emit_sigma=True, emit_metrics=True)

        # `coef` is emitted on each chunk's last row, so chunking moves the
        # cadence it is reported at; every value is compared.
        def fields(frame_):
            return frame_.select("m").unnest("m").drop("coef")

        one = fields(po.ModelBank([s]).fit_predict(df))
        bank = po.ModelBank([s])
        many = fields(
            pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
        )
        assert many.equals(one, null_equal=True)

    def test_the_buffer_survives_a_save_and_load(self, tmp_path):
        df = frame(n=400, seed=6)
        s = spec(label_delay=13.0)
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
        counting, not by the name appearing at all.)"""
        df = frame(n=100)
        plain = po.ModelBank([spec()])
        plain.fit_predict(df)
        delayed = po.ModelBank([spec(label_delay=5.0)])
        delayed.fit_predict(df)
        assert plain.save_bytes().count(b"pending") == 1, "the clock's, and no other"
        assert delayed.save_bytes().count(b"pending") == 2, "the clock's and the stream's"
        assert len(delayed.save_bytes()) > len(plain.save_bytes())

    def test_rows_still_waiting_are_never_learned_from(self):
        """At the end of a stream the buffer is simply unlearned: those
        labels have not matured, and inventing a deadline would be the leak
        this exists to prevent."""
        df = frame(n=200)
        plain = po.ModelBank([spec()])
        plain.fit_predict(df)
        delayed = po.ModelBank([spec(label_delay=30.0)])
        delayed.fit_predict(df)
        # 30 clock units at one row per unit: the last 30 rows never landed.
        assert delayed.gram("m")[0]["n_eff"] < plain.gram("m")[0]["n_eff"]
        head = po.ModelBank([spec()])
        head.fit_predict(df.head(200 - 30))
        assert delayed.gram("m")[0]["n_eff"] == pytest.approx(head.gram("m")[0]["n_eff"], rel=1e-12)

    def test_a_skipped_row_waits_for_nothing(self):
        """A null feature skips the row entirely, so it never enters the
        buffer; its clock time still counts the buffer down, because it is
        folded into the next accepted row's delta as it always was."""
        df = frame(n=300, seed=7)
        holes = df.with_columns(
            x=pl.when(pl.int_range(pl.len()) % 17 == 3).then(None).otherwise(pl.col("x"))
        )
        native = po.ModelBank([spec(label_delay=8.0)]).fit_predict(holes)
        oracle_frame = prep.embargo(holes, clock="t", delay=8.0).collect()
        oracle = (
            po.ModelBank([spec(weight=prep.ROLE + "_weight")])
            .fit_predict(oracle_frame)
            .filter(pl.col(prep.ROLE) == "predict")
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
        s = spec(label_delay=12.0, group="g")
        together = po.ModelBank([s]).fit_predict(df)
        for key in ("g0", "g1", "g2"):
            part = df.filter(pl.col("g") == key)
            alone = po.ModelBank([spec(label_delay=12.0)]).fit_predict(part)
            assert (
                together.filter(pl.col("g") == key)
                .select("m")
                .unnest("m")
                .equals(alone.select("m").unnest("m"), null_equal=True)
            )

    def test_a_reset_drops_the_buffer(self):
        """`on_clock_reset="reset_state"` throws the models away; the rows
        waiting to teach them go too."""
        n = 200
        clock = np.concatenate([np.arange(100.0), np.arange(100.0)])
        df = frame(n=n, seed=9).with_columns(t=pl.Series(clock))
        s = spec(label_delay=10.0, on_clock_reset="reset_state")
        bank = po.ModelBank([s])
        bank.fit_predict(df)
        # After the reset the stream is the second half alone, minus the
        # rows still waiting at its end.
        fresh = po.ModelBank([spec(label_delay=10.0)])
        fresh.fit_predict(df.tail(100).with_columns(t=pl.Series(np.arange(100.0))))
        assert bank.gram("m")[0]["n_eff"] == pytest.approx(fresh.gram("m")[0]["n_eff"], rel=1e-12)

    def test_a_session_change_releases_the_buffer(self):
        """One session's clock does not measure time in the next, so a row
        still waiting at the boundary is released rather than left to wait
        for a deadline that never comes."""
        n = 200
        df = frame(n=n, seed=10).with_columns(
            s=pl.Series(["a"] * 100 + ["b"] * 100),
            t=pl.Series(np.concatenate([np.arange(100.0), np.arange(1000.0, 1100.0)])),
        )
        s = spec(label_delay=10.0, session="s", session_gap=1.0)
        bank = po.ModelBank([s])
        bank.fit_predict(df)
        # Every row of the first session was learned from, and every row of
        # the second bar the last ten.
        no_delay = po.ModelBank([spec(session="s", session_gap=1.0)])
        no_delay.fit_predict(df.head(190))
        assert bank.gram("m")[0]["n_eff"] == pytest.approx(no_delay.gram("m")[0]["n_eff"], rel=1e-6)


class TestTheSurfaces:
    def test_the_lazy_plan_and_the_bank_agree(self):
        df = frame(n=300, seed=11)
        s = spec(label_delay=6.0)
        bank = po.ModelBank([s]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([s]).collect()
        assert lazy.select("m").unnest("m").equals(bank.select("m").unnest("m"), null_equal=True)

    def test_the_runner_and_the_cli(self, tmp_path, online_cli):
        df = frame(n=300, seed=12)
        src = tmp_path / "in.parquet"
        df.write_parquet(src)
        s = spec(label_delay=6.0)
        out = tmp_path / "out.parquet"
        po.run(specs=[s], input=src, output=out, chunk_rows=64)

        # `coef` rides the chunk cadence, as everywhere; every value is
        # compared.
        def fields(frame_):
            return frame_.select("m").unnest("m").drop("coef")

        want = fields(po.ModelBank([s]).fit_predict(df))
        assert fields(pl.read_parquet(out)).equals(want, null_equal=True)

        cli_out = tmp_path / "cli.parquet"
        cfg = tmp_path / "c.toml"
        cfg.write_text(
            f"""
input = "{src.as_posix()}"
output = "{cli_out.as_posix()}"
chunk_rows = 100

[[specs]]
name = "m"
targets = ["y"]
features = ["x"]
clock = "t"
halflife = {HALFLIFE}
max_dclock = 1e9
min_periods = 3.0
label_delay = 6.0
[specs.model]
type = "ew_ridge"
standardize = false
max_rows_between_solves = 1
"""
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        assert fields(pl.read_parquet(cli_out)).equals(want, null_equal=True)

    def test_the_expression_form(self):
        df = frame(n=200, seed=13)
        s = spec(label_delay=4.0)
        want = po.ModelBank([s]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        with pytest.warns(po.InMemoryExpressionWarning):
            got = df.select(
                po.online(pl.col("y")).ewridge(
                    ["x"],
                    clock="t",
                    halflife=HALFLIFE,
                    max_dclock=1e9,
                    min_periods=3.0,
                    standardize=False,
                    max_rows_between_solves=1,
                    label_delay=4.0,
                )
            )
        out = got.to_series().struct.field("pred_y").to_numpy()
        assert (np.isnan(out) == np.isnan(want)).all()
        assert np.array_equal(out[np.isfinite(out)], want[np.isfinite(want)])


class TestRefusals:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_delay_must_be_finite_and_positive(self, bad):
        with pytest.raises(ValueError, match="label_delay"):
            spec(label_delay=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
    def test_embargo_refuses_the_same_delays(self, bad):
        df = frame(n=10)
        with pytest.raises(ValueError, match="delay must be finite and > 0"):
            prep.embargo(df, clock="t", delay=bad)

    def test_embargo_needs_the_columns_it_names(self):
        df = frame(n=10)
        with pytest.raises(ValueError, match="no clock column 'nope'"):
            prep.embargo(df, clock="nope", delay=1.0)
        with pytest.raises(ValueError, match="no weight column 'nope'"):
            prep.embargo(df, clock="t", delay=1.0, weight="nope")
        with pytest.raises(ValueError, match="already has a column named 'x'"):
            prep.embargo(df, clock="t", delay=1.0, role="x")


class TestEmbargoItself:
    def test_the_shape_and_the_order(self):
        df = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
        out = prep.embargo(df, clock="t", delay=2.0).collect()
        assert out.height == 6
        assert out["t"].to_list() == [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
        assert out[prep.ROLE].to_list() == [
            "predict",
            "predict",
            "learn",
            "predict",
            "learn",
            "learn",
        ]
        assert out["_online_role_weight"].to_list() == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        assert out.columns == ["t", "x", "_online_role_weight", prep.ROLE]

    def test_an_existing_weight_is_zeroed_not_replaced(self):
        df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0], "w": [3.0, 4.0]})
        out = prep.embargo(df, clock="t", delay=1.0, weight="w").collect()
        assert out["w"].to_list() == [0.0, 3.0, 0.0, 4.0]
        assert out.columns == ["t", "x", "w", prep.ROLE]

    def test_it_stays_lazy(self):
        df = pl.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
        assert isinstance(prep.embargo(df.lazy(), clock="t", delay=1.0), pl.LazyFrame)
        assert isinstance(prep.embargo(df, clock="t", delay=1.0), pl.LazyFrame)
