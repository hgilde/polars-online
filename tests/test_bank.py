"""Task 7: the bank's own semantics -- chunk invariance, out-of-sample
predictions, the null policy and warmup, the clock, per-target `min_periods`
(docs/PLAN.md section 9, classes 2-5). Exercised through `ewridge`; the
oracle agreement of each model is in its own `test_<model>.py`."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic

HL = 300.0
MAXD = 50.0
GAP = 25.0


def _spec(k=3, targets=("y0",), **kw):
    defaults = dict(
        targets=list(targets),
        features=[f"x{j}" for j in range(k)],
        clock="t",
        halflife=HL,
        max_dclock=MAXD,
        session="session",
        session_gap=GAP,
        weight="w",
        group="group",
        ridge=1e-6,
        max_rows_between_solves=1,
        min_periods=5.0,
    )
    defaults.update(kw)
    return po.spec.ewridge("m", **defaults)


def _np(df, col):
    return df["m"].struct.field(col).to_numpy().astype(float)


class TestChunkInvariance:
    @pytest.mark.parametrize("n_chunks", [7, 100])
    def test_chunked_equals_single(self, n_chunks):
        df, _ = synthetic(seed=7, n_groups=3, n_rows=200, k=3)
        one = po.ModelBank([_spec()]).fit_predict(df)

        bank = po.ModelBank([_spec()])
        step = -(-df.height // n_chunks)
        outs = [bank.fit_predict(df.slice(i, step)) for i in range(0, df.height, step)]
        many = pl.concat(outs)
        # coef legitimately differs (emitted on each chunk's last row)
        a = one.unnest("m").drop("coef")
        b = many.unnest("m").drop("coef")
        assert a.equals(b, null_equal=True)

    def test_save_load_mid_stream(self, tmp_path):
        df, _ = synthetic(seed=8, n_groups=2, n_rows=300, k=3)
        first, second = df.slice(0, 150), df.slice(150, 150)

        b1 = po.ModelBank([_spec()])
        b1.fit_predict(first)
        p = tmp_path / "bank.state"
        b1.save(p)
        b2 = po.ModelBank.load(p, specs=[_spec()])
        a = b1.fit_predict(second).unnest("m").drop("coef")
        b = b2.fit_predict(second).unnest("m").drop("coef")
        assert a.equals(b, null_equal=True)

    def test_load_rejects_wrong_specs(self, tmp_path):
        b1 = po.ModelBank([_spec()])
        p = tmp_path / "bank.state"
        b1.save(p)
        with pytest.raises(OSError, match="do not match"):
            po.ModelBank.load(p, specs=[_spec(ridge=0.5)])


class TestOutOfSample:
    def test_noise_target_has_no_ic(self):
        # docs/PLAN.md section 9 class 3: pure-noise target => IC ~ 0. A model
        # that leaks the current row's y fails this loudly.
        rng = np.random.default_rng(0)
        n = 4000
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "y0": rng.standard_normal(n),
            }
        )
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            halflife=200.0,
            max_rows_between_solves=1,
            min_periods=10.0,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        pred = _np(out, "pred_y0")
        m = np.isfinite(pred)
        ic = np.corrcoef(pred[m], df["y0"].to_numpy()[m])[0, 1]
        assert abs(ic) < 0.06, f"IC {ic}: predictions are not out-of-sample"


class TestNullPolicyAndWarmup:
    def _df(self):
        return pl.DataFrame(
            {
                "x0": [1.0, 2.0, None, 0.5, 1.5, 2.5, 1.0],
                "x1": [0.5, 1.0, 1.5, 0.25, 0.75, 1.25, 0.5],
                "y0": [2.0, 4.0, 3.0, 1.0, None, 5.0, 2.0],
            }
        )

    def _out(self, min_periods=3.0):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            halflife=100.0,
            max_rows_between_solves=1,
            min_periods=min_periods,
        )
        return po.ModelBank([spec]).fit_predict(self._df())

    def test_feature_null_skips_row_entirely(self):
        out = self._out()
        row = out.row(2, named=True)["m"]
        assert row["pred_y0"] is None
        assert row["resid_y0"] is None
        assert row["n_eff"] is None

    def test_target_null_pred_only(self):
        out = self._out(min_periods=2.0)
        row = out.row(4, named=True)["m"]
        assert row["pred_y0"] is not None
        assert row["resid_y0"] is None

    def test_warmup_nulls_until_min_periods(self):
        out = self._out(min_periods=3.0)
        s = out["m"].struct
        preds = s.field("pred_y0").to_list()
        neff = s.field("n_eff").to_list()
        # rows 0-2: n_eff before update < 3 (row 2 skipped); row 3: n_eff ~ 2 -> null
        assert preds[0] is None and preds[1] is None
        assert neff[0] == 0.0
        # first non-null pred appears once n_eff >= 3
        first = next(i for i, p in enumerate(preds) if p is not None)
        assert neff[first] >= 3.0

    def test_null_weight_skips_row(self):
        df = self._df().with_columns(w=pl.Series([1.0, 1.0, 1.0, None, 1.0, 1.0, 1.0]))
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            halflife=100.0,
            weight="w",
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        assert out.row(3, named=True)["m"]["n_eff"] is None


class TestClockSemantics:
    def _run(self, t, on_clock_reset="max", session=None, session_gap=None, **kw):
        n = len(t)
        data = {
            "t": t,
            "x0": [1.0] * n,
            "y0": [1.0] * n,
        }
        if session is not None:
            data["session"] = session
        df = pl.DataFrame(data)
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            clock="t",
            halflife=10.0,
            max_dclock=50.0,
            on_clock_reset=on_clock_reset,
            session="session" if session is not None else None,
            session_gap=session_gap,
            max_rows_between_solves=1,
            min_periods=1.0,
            **kw,
        )
        return po.ModelBank([spec]).fit_predict(df)

    @staticmethod
    def _neff(out):
        return out["m"].struct.field("n_eff").to_numpy()

    def test_gap_cap(self):
        # Delta of 1e6 capped at 50 => decay 0.5^(50/10) = 1/32, not ~0.
        # n_eff is reported BEFORE the row's update, so the effect of row 1's
        # decay shows in row 2's n_eff.
        out = self._run([0.0, 1e6, 1e6 + 10])
        n = self._neff(out)
        assert abs(n[2] - (0.5 ** (50.0 / 10.0) + 1.0)) < 1e-12

    def test_negative_delta_max(self):
        # Row 2's clock runs backwards (100 -> 50): treated as max_dclock.
        out = self._run([0.0, 100.0, 50.0, 51.0])
        n = self._neff(out)
        w1 = 1.0 * 0.5 ** (50.0 / 10.0) + 1.0  # after row 1 (delta 100 capped to 50)
        w2 = w1 * 0.5 ** (50.0 / 10.0) + 1.0  # after row 2 (negative -> max)
        assert abs(n[2] - w1) < 1e-12
        assert abs(n[3] - w2) < 1e-12

    def test_reset_state(self):
        out = self._run([0.0, 10.0, 5.0], on_clock_reset="reset_state")
        n = self._neff(out)
        assert n[2] == 0.0  # state was reset before row 2

    def test_session_reset(self):
        out = self._run(
            [0.0, 10.0, 20.0],
            session=["a", "a", "b"],
            session_gap="reset",
        )
        n = self._neff(out)
        assert n[2] == 0.0

    def test_session_gap_value(self):
        out = self._run(
            [0.0, 10.0, 10.5, 11.5],
            session=["a", "a", "b", "b"],
            session_gap=20.0,
        )
        n = self._neff(out)
        # row 2's update uses the 20-unit session gap, not the raw 0.5 delta;
        # visible in row 3's (before-update) n_eff.
        w1 = 0.5 ** (10.0 / 10.0) * 1.0 + 1.0
        w2 = w1 * 0.5 ** (20.0 / 10.0) + 1.0
        assert abs(n[3] - w2) < 1e-12


class TestPerTargetMinPeriods:
    """E7: `min_periods` accepts one threshold per target.

    A 5-minute-ahead target and a 1-day-ahead target rarely deserve the same
    warmup. Warmup gates *output*, not learning.
    """

    def _out(self, min_periods, n=120):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x, "y1": -x})
        spec = po.spec.ewridge(
            "m",
            targets=["y0", "y1"],
            features=["x0"],
            halflife=1e9,
            min_periods=min_periods,
            max_rows_between_solves=1,
        )
        return po.ModelBank([spec]).fit_predict(df)

    @staticmethod
    def _first(out, field):
        vals = out["m"].struct.field(field).to_list()
        return next((i for i, v in enumerate(vals) if v is not None), None)

    def test_each_target_waits_for_its_own_threshold(self):
        out = self._out([5.0, 50.0])
        assert self._first(out, "pred_y0") == 6
        assert self._first(out, "pred_y1") == 51

    def test_a_scalar_still_applies_to_every_target(self):
        out = self._out(20.0)
        assert self._first(out, "pred_y0") == self._first(out, "pred_y1") == 21

    def test_a_late_target_has_no_residual_or_sigma_either(self):
        rng = np.random.default_rng(1)
        n = 120
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x, "y1": -x})
        spec = po.spec.ewridge(
            "m",
            targets=["y0", "y1"],
            features=["x0"],
            halflife=1e9,
            min_periods=[5.0, 60.0],
            max_rows_between_solves=1,
            emit_sigma=True,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for field in ("resid_y1", "sigma_y1"):
            vals = out["m"].struct.field(field).to_list()
            assert all(v is None for v in vals[:55]), f"{field} leaked during warmup"

    def test_learning_is_not_gated_only_output(self):
        # The late target's fit must be as good as if it had reported all along:
        # the model still learned from the withheld rows.
        gated = self._out([5.0, 60.0], n=400)
        eager = self._out([5.0, 5.0], n=400)
        assert gated["m"].struct.field("pred_y1").to_list()[-1] == pytest.approx(
            eager["m"].struct.field("pred_y1").to_list()[-1]
        )

    def test_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="min_periods list has 3 entries"):
            po.spec.ewridge(
                "m",
                targets=["y0", "y1"],
                features=["x0"],
                halflife=100.0,
                min_periods=[1.0, 2.0, 3.0],
            )

    def test_negative_is_rejected(self):
        with pytest.raises(ValueError, match="min_periods must be >= 0"):
            po.spec.ewridge("m", targets=["y0"], features=["x0"], halflife=100.0, min_periods=-1.0)
