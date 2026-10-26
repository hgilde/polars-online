"""E21 / E12: `emit_sigma` and `emit_zscore`.

`sigma_<slot>` is the EW standard deviation of that slot's out-of-sample
residuals, read from the state *before* each row; `zscore_<slot>` is
`resid / sigma`. Both are computed in the stream layer rather than per model,
so every model gets the same definition (the models' own internal `sigma2`
fields serve different purposes and are not all present or comparable).
"""

import math

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("rls", {"delta": 1.0}),
    ("kalman", {"coef_half_life": 100.0}),
    ("lasso", {"penalties": [0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {}),
]
IDS = [m[0] for m in MODELS]


def _ew_quantile(values, weights, p):
    """The exponentially weighted quantile by definition: the smallest value
    whose cumulative weight, ascending, reaches ``p`` of the total."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, p * cum[-1])])


ALPHA = math.tanh(1 / 128)  # the sketch's relative accuracy (task 146)


def _df(n=400, seed=0, shock_at=None, noise=0.5, binary=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (rng.random(n) < 0.5).astype(float) if binary else 2 * x + noise * rng.standard_normal(n)
    if shock_at is not None and not binary:
        y[shock_at] += 20.0
    return pl.DataFrame({"x0": x, "y0": y, "t": np.arange(float(n))})


def _spec(model="ewridge", extra=None, **kw):
    d = dict(
        targets=["y0"],
        features=["x0"],
        half_life=100.0,
        min_weight=10.0,
        emit_sigma=True,
        emit_zscore=True,
    )
    # `extra is None`, not `extra or ...`: ftrl's entry is an empty dict, and
    # `{} or default` would silently substitute the default.
    d.update({"max_rows_between_solves": 1} if extra is None else extra)
    d.update(kw)
    return getattr(po.spec, model)("m", **d)


def _slot(out, prefix):
    """First field with this prefix (lasso suffixes its slots by path point)."""
    name = next(f.name for f in out.schema["m"].fields if f.name.startswith(prefix))
    return out["m"].struct.field(name).to_numpy().astype(float)


def _f(out, name):
    return out["m"].struct.field(name).to_numpy().astype(float)


class TestFieldsAreOptional:
    def test_absent_by_default(self):
        spec = po.spec.ewridge("m", targets=["y0"], features=["x0"], half_life=100.0)
        fields = po.spec.output_fields(spec)
        assert not any(f.startswith(("sigma_", "zscore_")) for f in fields)

    def test_present_when_requested(self):
        assert po.spec.output_fields(_spec()) == [
            "pred_y0",
            "resid_y0",
            "sigma_y0",
            "zscore_y0",
            "weight_sum",
            "settled_frac",
            "withheld_reason",
            "coef",
            "support_coef",
        ]

    def test_each_flag_is_independent(self):
        only_sigma = _spec(emit_zscore=False)
        only_z = _spec(emit_sigma=False)
        assert "sigma_y0" in po.spec.output_fields(only_sigma)
        assert "zscore_y0" not in po.spec.output_fields(only_sigma)
        assert "zscore_y0" in po.spec.output_fields(only_z)
        assert "sigma_y0" not in po.spec.output_fields(only_z)

    def test_one_field_per_slot_in_a_grid(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y0", "y1"],
            features=["x0"],
            ridge=[1e-6, 1.0],
            half_life=100.0,
            emit_sigma=True,
            emit_zscore=True,
        )
        fields = po.spec.output_fields(spec)
        assert [f for f in fields if f.startswith("sigma_")] == [
            "sigma_y0__r0.000001",
            "sigma_y0__r1",
            "sigma_y1__r0.000001",
            "sigma_y1__r1",
        ]


class TestValues:
    def test_sigma_recovers_the_noise_level(self):
        out = po.ModelBank([_spec()]).fit_predict(_df(n=3000, noise=0.5))
        # after warmup the EW residual sd should sit near the true noise sd
        assert np.nanmedian(_f(out, "sigma_y0")[500:]) == pytest.approx(0.5, abs=0.1)

    def test_resid_z_flags_a_shock(self):
        out = po.ModelBank([_spec()]).fit_predict(_df(shock_at=300))
        z = _f(out, "zscore_y0")
        assert np.nanmedian(np.abs(z)) < 1.5
        assert abs(z[300]) > 10.0
        assert int(np.nanargmax(np.abs(z))) == 300

    def test_resid_z_equals_resid_over_sigma(self):
        out = po.ModelBank([_spec()]).fit_predict(_df())
        r, s, z = (_f(out, f"{k}_y0") for k in ("resid", "sigma", "zscore"))
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        np.testing.assert_allclose(z[m], r[m] / s[m], rtol=1e-12)

    def test_sigma_is_out_of_sample(self):
        # sigma at row i must not include row i's own residual: a single huge
        # shock must not inflate the sigma reported on that same row.
        out = po.ModelBank([_spec()]).fit_predict(_df(shock_at=300))
        s = _f(out, "sigma_y0")
        assert s[300] == pytest.approx(s[299], rel=0.05), (
            "the shock leaked into the sigma reported for its own row"
        )
        assert s[301] > s[300] * 2, "the shock should raise sigma from the next row on"

    def test_sigma_is_null_before_the_first_residual(self):
        out = po.ModelBank([_spec(min_weight=20.0)]).fit_predict(_df())
        s = out["m"].struct.field("sigma_y0").to_list()
        first_pred = next(
            i for i, v in enumerate(out["m"].struct.field("pred_y0").to_list()) if v is not None
        )
        assert s[first_pred] is None, "no residual has been seen yet"
        assert s[first_pred + 2] is not None


@pytest.mark.parametrize(("model", "extra"), MODELS, ids=IDS)
class TestAllModels:
    def test_fields_are_emitted_and_consistent(self, model, extra):
        df = _df(binary=model == "ftrl")
        out = po.ModelBank([_spec(model, extra)]).fit_predict(df)
        r, s, z = (_slot(out, f"{k}_y0") for k in ("resid", "sigma", "zscore"))
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        assert m.sum() > 50, f"{model}: almost nothing was emitted"
        np.testing.assert_allclose(z[m], r[m] / s[m], rtol=1e-12)

    def test_chunk_invariance_of_the_new_fields(self, model, extra):
        df = _df(n=200, binary=model == "ftrl")
        spec = _spec(model, extra)
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 17)) for i in range(0, df.height, 17)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    def test_survives_save_load(self, model, extra, tmp_path):
        df = _df(n=200, binary=model == "ftrl")
        spec = _spec(model, extra)
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 100))
        p = tmp_path / "s.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(100, 100)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)


class TestOnlineSelection:
    """E13: `emit_selected` — online model selection across grid slots.

    Generalizes the lasso's `lam_selected` to ridge values, feature sets and
    half-lives: for each target, pick the slot with the lowest EW out-of-sample
    error so far (the `sigma` already tracked for E12) and emit that slot's
    prediction plus its label.
    """

    def _grid_spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            ridge=[1e-8, 1.0, 1000.0],
            half_life=300.0,
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_selected=True,
        )
        d.update(kw)
        return po.spec.ewridge("m", **d)

    def _selection(self, y, x, **kw):
        out = po.ModelBank([self._grid_spec(**kw)]).fit_predict(pl.DataFrame({"x0": x, "y0": y}))
        return out

    def test_fields_are_appended_once_per_target(self):
        spec = self._grid_spec(targets=["y0", "y1"])
        fields = po.spec.output_fields(spec)
        assert fields[-4:] == [
            "pred_y0__selected",
            "selected_y0",
            "pred_y1__selected",
            "selected_y1",
        ]

    def test_picks_the_light_ridge_on_a_strong_signal(self):
        rng = np.random.default_rng(1)
        n = 4000
        x = rng.standard_normal(n)
        out = self._selection(2 * x + 0.3 * rng.standard_normal(n), x)
        sel = [s for s in out["m"].struct.field("selected_y0").to_list()[-1500:] if s]
        assert sel.count("r1e-8") / len(sel) > 0.9

    def test_picks_shrinkage_on_pure_noise(self):
        rng = np.random.default_rng(1)
        n = 4000
        x = rng.standard_normal(n)
        out = self._selection(rng.standard_normal(n), x)
        sel = [s for s in out["m"].struct.field("selected_y0").to_list()[-1500:] if s]
        # the near-zero ridge fits the noise and must not dominate
        assert sel.count("r1e-8") / len(sel) < 0.2

    def test_selected_prediction_equals_the_named_slot(self):
        rng = np.random.default_rng(2)
        n = 1000
        x = rng.standard_normal(n)
        out = self._selection(2 * x + rng.standard_normal(n), x)
        st = out["m"].struct
        names = st.field("selected_y0").to_list()
        chosen = st.field("pred_y0__selected").to_list()
        for i in (300, 600, 900):
            assert names[i] is not None
            assert chosen[i] == st.field(f"pred_y0__{names[i]}").to_list()[i]

    def test_selects_across_halflives_too(self):
        rng = np.random.default_rng(3)
        n = 2000
        x = rng.standard_normal(n)
        spec = self._grid_spec(ridge=1e-6, half_life=[20.0, 2000.0])
        out = po.ModelBank([spec]).fit_predict(
            pl.DataFrame({"x0": x, "y0": 2 * x + 0.2 * rng.standard_normal(n)})
        )
        sel = {s for s in out["m"].struct.field("selected_y0").to_list() if s}
        assert sel <= {"@h20", "@h2000"}
        assert sel, "nothing was ever selected"

    def test_a_halflife_grid_ranks_each_slot_at_its_own_halflife(self):
        """Task 148: each slot's error is the EW mean of its own squared
        residuals at its own instance's half-life, so across a half-life grid
        the slots are ranked over windows of different lengths, as the docs
        now say. Held against that recursion from the residuals."""
        rng = np.random.default_rng(5)
        n = 600
        x = rng.standard_normal(n)
        hs = (20.0, 2000.0)
        spec = self._grid_spec(ridge=1e-6, half_life=list(hs), emit_sigma=True)
        out = po.ModelBank([spec]).fit_predict(
            pl.DataFrame({"x0": x, "y0": 2 * x + 0.5 * rng.standard_normal(n)})
        )
        st = out["m"].struct
        sigma = {}
        for h in hs:
            tag = f"@h{h:g}"
            r = st.field(f"resid_y0{tag}").to_numpy()
            got = st.field(f"sigma_y0{tag}").to_numpy()
            lam, num, den, want = 0.5 ** (1 / h), 0.0, 0.0, np.full(n, np.nan)
            for i in range(n):
                if den > 0.0:
                    want[i] = np.sqrt(num / den)
                num, den = lam * num, lam * den
                if np.isfinite(r[i]):
                    num, den = num + r[i] ** 2, den + 1.0
            ok = np.isfinite(want)
            assert np.isfinite(got).tolist() == ok.tolist(), tag
            assert got[ok] == pytest.approx(want[ok], rel=1e-9), tag
            sigma[tag] = got
        names = st.field("selected_y0").to_list()
        tags = list(sigma)
        for i in range(100, n):
            best = tags[int(np.argmin([sigma[t][i] for t in tags]))]
            assert names[i] == best, i

    def test_requires_more_than_one_slot(self):
        with pytest.raises(ValueError, match="more than one slot"):
            po.spec.ewridge(
                "m", targets=["y0"], features=["x0"], half_life=100.0, emit_selected=True
            )

    def test_rejected_for_ew_cov(self):
        with pytest.raises(ValueError, match="does not apply to ew_cov"):
            po.spec.ew_cov("c", features=["x0", "x1"], half_life=100.0, emit_selected=True)

    def test_chunk_invariance(self):
        rng = np.random.default_rng(4)
        n = 300
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + rng.standard_normal(n)})
        spec = self._grid_spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 29)) for i in range(0, df.height, 29)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)


class TestDriftDetection:
    """E20: `emit_drift` — Page-Hinkley on each slot's absolute out-of-sample
    residual, scaled by that slot's own EW residual std.

    Decay and drift detection answer different questions: a half-life forgets
    smoothly and always, a detector notices a *break* and says so.
    """

    def _df(self, n=6000, flip_at=3000, seed=0, noise=0.2):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        sign = np.where(np.arange(n) < flip_at, 1.0, -1.0)
        return pl.DataFrame({"x0": x, "y0": sign * 2 * x + noise * rng.standard_normal(n)})

    def _spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            half_life=1e5,
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_drift=True,
        )
        d.update(kw)
        return po.spec.ewridge("m", **d)

    def _flags(self, df, **kw):
        out = po.ModelBank([self._spec(**kw)]).fit_predict(df)
        vals = out["m"].struct.field("drift_y0").fill_null(False).to_list()  # noqa: FBT003
        return np.array([bool(v) for v in vals]), out

    def test_field_is_opt_in(self):
        plain = po.spec.ewridge("m", targets=["y0"], features=["x0"], half_life=100.0)
        assert not any(f.startswith("drift_") for f in po.spec.output_fields(plain))
        assert "drift_y0" in po.spec.output_fields(self._spec())

    def test_detects_a_regime_flip_quickly(self):
        flags, _ = self._flags(self._df(flip_at=3000))
        hits = np.flatnonzero(flags)
        assert len(hits) >= 1, "no drift detected across a sign flip"
        after = hits[hits >= 3000]
        assert len(after) >= 1
        assert after[0] - 3000 < 100, f"took {after[0] - 3000} rows to notice"

    def test_no_false_positives_on_a_stationary_stream(self):
        rng = np.random.default_rng(1)
        n = 8000
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + 0.2 * rng.standard_normal(n)})
        flags, _ = self._flags(df)
        assert flags.sum() == 0, f"{flags.sum()} false positives on a stationary stream"

    def test_a_burst_counts_the_same_at_any_row_density(self):
        """Task 146: the excess is integrated over the clock, so the same
        30-unit burst of error is flagged at the same clock whether the rows
        during it are one or four to a unit; counting rows flagged it about
        four times sooner at four. The noise's size is fixed and only its
        sign drawn, so both densities see the same error, and the half-life
        is long, so ``sigma`` -- which each row reads from before it, and
        which lags further behind a moving scale where rows are sparser --
        barely moves inside the burst. Measured: 211.0 at both; 0.12.0's
        detector, counting rows, flagged at 211.0 and 202.0."""
        first = {}
        for per_unit in (1, 4):
            rng = np.random.default_rng(7)
            t = np.arange(0.0, 400.0, 1.0 / per_unit)
            x = rng.standard_normal(t.size)
            size = np.where((t >= 200.0) & (t < 230.0), 0.8, 0.2)
            noise = size * rng.choice([-1.0, 1.0], t.size)
            df = pl.DataFrame({"t": t, "x0": x, "y0": 2 * x + noise})
            flags, _ = self._flags(
                df, clock="t", gap_cap=5.0, half_life=1000.0, drift_threshold=20.0
            )
            hits = t[flags]
            first[per_unit] = float(hits[0]) if hits.size else None
        assert first[1] is not None and 205.0 <= first[1] < 230.0, first
        assert first[4] is not None and abs(first[4] - first[1]) <= 1.0, first

    def test_threshold_trades_sensitivity_for_noise(self):
        df = self._df()
        eager, _ = self._flags(df, drift_threshold=2.0)
        patient, _ = self._flags(df, drift_threshold=200.0)
        # Strictly: an equality let a detector that never fires pass (116
        # flags against 1 measured; review 2026-10-06, TB10).
        assert eager.sum() > patient.sum()

    def test_delta_absorbs_small_shifts(self):
        df = self._df()
        tolerant, _ = self._flags(df, drift_delta=50.0)
        assert tolerant.sum() == 0, "a huge delta should absorb everything"

    def test_reset_action_restarts_the_model(self):
        df = self._df(flip_at=3000)
        flags, out = self._flags(df, drift_action="reset")
        hits = np.flatnonzero(flags)
        assert len(hits) >= 1
        weight_sum = out["m"].struct.field("weight_sum").to_numpy().astype(float)
        # weight_sum climbs monotonically unless something resets it
        assert weight_sum[hits[0] + 1] < weight_sum[hits[0]], "the reset did not take effect"

    def test_flag_action_leaves_the_model_running(self):
        df = self._df(flip_at=3000)
        flags, out = self._flags(df, drift_action="flag")
        weight_sum = out["m"].struct.field("weight_sum").to_numpy().astype(float)
        hits = np.flatnonzero(flags)
        assert len(hits) >= 1
        assert weight_sum[hits[0] + 1] > weight_sum[hits[0]], "flag-only should not reset"

    def test_chunk_invariance(self):
        df = self._df(n=600, flip_at=300)
        spec = self._spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 47)) for i in range(0, df.height, 47)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    def test_survives_save_load(self, tmp_path):
        df = self._df(n=600, flip_at=300)
        spec = self._spec()
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 300))
        p = tmp_path / "d.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(300, 300)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_bad_config_rejected(self):
        with pytest.raises(ValueError, match="drift_action"):
            self._spec(drift_action="explode")
        with pytest.raises(ValueError, match="drift_threshold"):
            self._spec(drift_threshold=0.0)
        with pytest.raises(ValueError, match="drift_delta"):
            self._spec(drift_delta=-1.0)


class TestModelAveraging:
    """E14: `emit_averaged` — exponentially weighted forecaster over grid slots.

    The soft counterpart of `emit_selected`: weights are
    `softmax(-eta * EW squared error / the best slot's)`, so averaging hedges
    where selection commits, in any target's units (review 2026-09-12, S23).
    """

    SLOTS = ["pred_y0__r1e-8", "pred_y0__r1", "pred_y0__r100"]

    def _run(self, n=4000, seed=1, noise=3.0, scale=1.0, **kw):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": scale * (2 * x + noise * rng.standard_normal(n))})
        d = dict(
            targets=["y0"],
            features=["x0"],
            ridge=[1e-8, 1.0, 100.0],
            half_life=500.0,
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_selected=True,
            emit_averaged=True,
        )
        d.update(kw)
        out = po.ModelBank([po.spec.ewridge("m", **d)]).fit_predict(df)
        return out, df

    @staticmethod
    def _f(out, name):
        return out["m"].struct.field(name).to_numpy().astype(float)

    def test_field_is_opt_in(self):
        plain = po.spec.ewridge(
            "m", targets=["y0"], features=["x0"], ridge=[1e-6, 1.0], half_life=100.0
        )
        assert "pred_y0__averaged" not in po.spec.output_fields(plain)

    def test_is_a_convex_combination_of_the_slots(self):
        out, _ = self._run()
        avg = self._f(out, "pred_y0__averaged")
        slots = np.array([self._f(out, s) for s in self.SLOTS])
        m = np.isfinite(avg) & np.isfinite(slots).all(axis=0)
        assert m.sum() > 1000
        assert (avg[m] >= slots[:, m].min(axis=0) - 1e-9).all()
        assert (avg[m] <= slots[:, m].max(axis=0) + 1e-9).all()

    def test_large_eta_reproduces_selection(self):
        out, _ = self._run(average_eta=1e6)
        avg, sel = self._f(out, "pred_y0__averaged"), self._f(out, "pred_y0__selected")
        m = np.isfinite(avg) & np.isfinite(sel)
        assert np.max(np.abs(avg[m] - sel[m])) < 1e-9

    def test_an_infinite_eta_is_the_selection(self):
        """``average_eta = inf`` is ``emit_selected``'s argmin: the best slot
        weighs ``exp(0) = 1`` and every other ``exp(-inf) = 0``, a tie shared
        (review 2026-09-12, S27). The builder refused it, and at the best slot
        the formula's ``-inf * 0`` is NaN."""
        out, _ = self._run(average_eta=float("inf"), emit_sigma=True)
        preds = np.array([self._f(out, s) for s in self.SLOTS])
        sig = np.array([self._f(out, s.replace("pred_", "sigma_", 1)) for s in self.SLOTS])
        avg, sel = self._f(out, "pred_y0__averaged"), self._f(out, "pred_y0__selected")
        ok = np.isfinite(preds) & np.isfinite(sig)
        rows = np.flatnonzero(ok.any(axis=0))
        assert rows.size > 1000 and np.isfinite(avg[rows]).all()
        unique = 0
        for i in rows:
            s2, p = sig[ok[:, i], i] ** 2, preds[ok[:, i], i]
            best = s2 == s2.min()
            assert avg[i] == pytest.approx(p[best].mean(), rel=1e-15), i
            if best.sum() == 1:
                assert avg[i] == sel[i], i
                unique += 1
        assert unique > 1000

    def test_small_eta_is_the_equal_weight_mean(self):
        out, _ = self._run(average_eta=1e-6)
        avg = self._f(out, "pred_y0__averaged")
        mean = np.mean([self._f(out, s) for s in self.SLOTS], axis=0)
        m = np.isfinite(avg) & np.isfinite(mean)
        assert np.max(np.abs(avg[m] - mean[m])) < 1e-4

    def test_is_the_same_average_in_any_units(self):
        """``eta`` weighs each slot's error against the best slot's, as a
        ratio, so the weights do not depend on the target's units: the same
        stream with ``y`` ten thousand times larger gives an average ten
        thousand times larger. The weights were ``exp(-eta * (sigma² -
        sigma²_best))``, so ``eta = 1`` was an equal-weight mean for a return
        and the argmin for a price (review 2026-09-12, S23)."""
        base, _ = self._run()
        big, _ = self._run(scale=1e4)
        a, b = self._f(base, "pred_y0__averaged"), self._f(big, "pred_y0__averaged")
        m = np.isfinite(a) & np.isfinite(b)
        assert m.sum() > 1000
        np.testing.assert_allclose(b[m], 1e4 * a[m], rtol=1e-9)

    def test_the_weights_are_each_error_over_the_best(self):
        """The formula, from the bank's own columns: each slot with a
        prediction and a sigma on the row weighs ``exp(-eta * (sigma² /
        sigma²_best - 1))``, normalized."""
        eta = 0.7
        out, _ = self._run(average_eta=eta, emit_sigma=True)
        preds = np.array([self._f(out, s) for s in self.SLOTS])
        sig = np.array([self._f(out, s.replace("pred_", "sigma_", 1)) for s in self.SLOTS])
        avg = self._f(out, "pred_y0__averaged")
        ok = np.isfinite(preds) & np.isfinite(sig)
        rows = np.flatnonzero(ok.any(axis=0) & np.isfinite(avg))
        assert rows.size > 1000
        for i in rows:
            s2, p = sig[ok[:, i], i] ** 2, preds[ok[:, i], i]
            w = np.exp(-eta * (s2 / s2.min() - 1.0))
            assert avg[i] == pytest.approx(np.dot(w, p) / w.sum(), rel=1e-12)

    def test_beats_the_worst_slot(self):
        out, df = self._run()
        y = df["y0"].to_numpy()

        def mse(name):
            p = self._f(out, name)
            m = np.isfinite(p)
            return float(np.mean((p[m] - y[m]) ** 2))

        worst = max(mse(s) for s in self.SLOTS)
        assert mse("pred_y0__averaged") < worst

    def test_hedges_when_the_best_slot_changes(self):
        # A regime switch flips which ridge is best. Selection whipsaws;
        # averaging should not be far behind, and must beat the slot that is
        # wrong for half the stream.
        rng = np.random.default_rng(7)
        n = 8000
        x = rng.standard_normal(n)
        noise = np.where(np.arange(n) < n // 2, 0.1, 6.0)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + noise * rng.standard_normal(n)})
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            ridge=[1e-8, 100.0],
            half_life=300.0,
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_averaged=True,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        y = df["y0"].to_numpy()

        def mse(name):
            p = out["m"].struct.field(name).to_numpy().astype(float)
            m = np.isfinite(p)
            return float(np.mean((p[m] - y[m]) ** 2))

        assert mse("pred_y0__averaged") < mse("pred_y0__r100")

    def test_rejected_for_ew_cov(self):
        with pytest.raises(ValueError, match="does not apply to ew_cov"):
            po.spec.ew_cov("c", features=["x0", "x1"], half_life=100.0, emit_averaged=True)

    def test_bad_eta_rejected(self):
        with pytest.raises(ValueError, match="average_eta"):
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                ridge=[1e-6, 1.0],
                half_life=100.0,
                emit_averaged=True,
                average_eta=0.0,
            )

    def test_chunk_invariance(self):
        out, df = self._run(n=300)
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            ridge=[1e-8, 1.0, 100.0],
            half_life=500.0,
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_averaged=True,
        )
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 29)) for i in range(0, df.height, 29)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)


class TestResidualDistribution:
    """E23: P² quantiles of |resid| and residual autocorrelation.

    `sigma` describes a Gaussian; these describe the distribution that is
    actually there, and whether the residual stream still looks like noise.
    """

    def _fit(self, df, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            half_life=500.0,
            min_weight=20.0,
            max_rows_between_solves=1,
        )
        d.update(kw)
        return po.ModelBank([po.spec.ewridge("m", **d)]).fit_predict(df)

    @staticmethod
    def _last(out, name):
        return out["m"].struct.field(name).to_list()[-1]

    def _fat_tailed(self, n=8000, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        noise = np.where(
            rng.random(n) < 0.01, 50 * rng.standard_normal(n), 0.3 * rng.standard_normal(n)
        )
        return pl.DataFrame({"x0": x, "y0": 2 * x + noise})

    def test_fields_are_opt_in_and_named_by_level(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=100.0,
            resid_quantiles=[0.5, 0.99],
            emit_autocorr=True,
        )
        fields = po.spec.output_fields(spec)
        assert "abs_resid_q0.5_y0" in fields
        assert "abs_resid_q0.99_y0" in fields
        assert "autocorr_y0" in fields
        plain = po.spec.ewridge("m", targets=["y0"], features=["x0"], half_life=100.0)
        assert not any(
            f.startswith(("absresid_", "autocorr_")) for f in po.spec.output_fields(plain)
        )

    def test_quantiles_describe_the_bulk_where_sigma_cannot(self):
        out = self._fit(self._fat_tailed(), emit_sigma=True, resid_quantiles=[0.5, 0.99])
        sigma = self._last(out, "sigma_y0")
        med = self._last(out, "abs_resid_q0.5_y0")
        # 1% gross outliers inflate sigma far above a typical residual, which is
        # exactly why a quantile is worth tracking separately.
        assert med < sigma / 3, f"median |resid| {med} vs sigma {sigma}"

    def test_quantiles_are_ordered(self):
        out = self._fit(self._fat_tailed(), resid_quantiles=[0.25, 0.5, 0.9])
        vals = [self._last(out, f"abs_resid_q{q}_y0") for q in (0.25, 0.5, 0.9)]
        assert vals == sorted(vals), vals

    def test_quantile_matches_the_empirical_one(self):
        rng = np.random.default_rng(3)
        n = 20000
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + rng.standard_normal(n)})
        out = self._fit(df, half_life=1e9, resid_quantiles=[0.9])
        r = np.abs(np.array(out["m"].struct.field("resid_y0").to_list(), dtype=float))
        truth = np.nanquantile(r[np.isfinite(r)], 0.9)
        # 1.4e-3 measured, inside the sketch's own accuracy tanh(1/128) =
        # 7.8e-3; 0.15 admitted a quantile at another level.
        assert self._last(out, "abs_resid_q0.9_y0") == pytest.approx(truth, rel=0.03)

    def test_the_quantile_forgets_on_the_clock(self):
        """Task 146: each level is the exponentially weighted quantile of
        the residuals so far, at the model's half-life, to the sketch's
        relative accuracy ``tanh(1/128)``. P² never forgot: 3,000 rows after
        the noise fell tenfold at a half-life of 10, its 0.9 quantile read
        1.55 where the recent one is 0.166."""
        rng = np.random.default_rng(8)
        n, h = 4000, 10.0
        x = rng.standard_normal(n)
        noise = np.where(np.arange(n) < 1000, 1.0, 0.1)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + noise * rng.standard_normal(n)})
        levels = [0.1, 0.5, 0.9, 0.99]
        # `min_weight` under the weight's ceiling at this half-life, ~14.9.
        out = self._fit(df, half_life=h, min_weight=5.0, resid_quantiles=levels)
        r = np.abs(np.array(out["m"].struct.field("resid_y0").to_list(), dtype=float))
        lam = 0.5 ** (1 / h)
        for i in (300, 1100, 1500, 3999):
            seen = np.flatnonzero(np.isfinite(r[:i]))
            assert seen.size > 200, (i, seen.size)
            w = lam ** (i - 1 - seen.astype(float))
            for p in levels:
                want = _ew_quantile(r[seen], w, p)
                got = out["m"].struct.field(f"abs_resid_q{p}_y0").to_list()[i]
                assert abs(got - want) <= ALPHA * want * (1 + 1e-9), (i, p, got, want)

    def test_autocorr_pairs_nothing_across_a_break(self):
        """Task 146: a residual pairs only with the one ``lag`` back inside
        the same run of adjacent rows. A gap capped by ``gap_cap`` and a
        session change both end the run, as they clear the models' rings, so
        with a break before every row nothing ever pairs."""
        rng = np.random.default_rng(9)
        n = 300
        x = rng.standard_normal(n)
        y = 2 * x + rng.standard_normal(n)
        for kw, extra in (
            (dict(clock="t", gap_cap=5.0), {"t": 10.0 * np.arange(n)}),
            (dict(session="s", session_gap=1.0), {"s": np.arange(n) % 2}),
        ):
            df = pl.DataFrame({"x0": x, "y0": y} | extra)
            out = self._fit(df, emit_autocorr=True, **kw)
            got = out["m"].struct.field("autocorr_y0").to_list()
            assert all(v is None for v in got), kw

    def test_autocorr_is_near_zero_for_a_well_specified_model(self):
        rng = np.random.default_rng(4)
        n = 20000
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + rng.standard_normal(n)})
        out = self._fit(df, emit_autocorr=True)
        assert abs(self._last(out, "autocorr_y0")) < 0.1

    def test_autocorr_flags_a_missing_feature(self):
        # A slow-moving omitted driver leaves autocorrelated residuals, which is
        # the classic sign of mis-specification.
        rng = np.random.default_rng(5)
        n = 20000
        x = rng.standard_normal(n)
        omitted = np.cumsum(rng.standard_normal(n)) * 0.05
        df = pl.DataFrame({"x0": x, "y0": 2 * x + omitted + 0.1 * rng.standard_normal(n)})
        out = self._fit(df, emit_autocorr=True, half_life=2000.0)
        assert self._last(out, "autocorr_y0") > 0.3, "an omitted driver should show up"

    def test_chunk_invariance_and_save_load(self, tmp_path):
        df = self._fat_tailed(n=600, seed=6)
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=500.0,
            min_weight=20.0,
            max_rows_between_solves=1,
            resid_quantiles=[0.5, 0.9],
            emit_autocorr=True,
        )
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 61)) for i in range(0, df.height, 61)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 300))
        p = tmp_path / "q.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(300, 300)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_bad_config_rejected(self):
        base = dict(targets=["y0"], features=["x0"], half_life=100.0)
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            po.spec.ewridge("m", resid_quantiles=[0.0], **base)
        with pytest.raises(ValueError, match="non-empty"):
            po.spec.ewridge("m", resid_quantiles=[], **base)
        with pytest.raises(ValueError, match="resid_autocorr_lag"):
            po.spec.ewridge("m", emit_autocorr=True, resid_autocorr_lag=0, **base)


class TestStreamingMetrics:
    """E22: `emit_metrics` — IC, R² and hit rate kept beside the model.

    `polars_online.eval` computes the same things in Polars over collected
    output, which needs the whole frame. This is the O(state) version, for a
    long-running stream or the CLI.
    """

    def _out(self, df, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            half_life=float("inf"),
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_metrics=True,
        )
        d.update(kw)
        return po.ModelBank([po.spec.ewridge("m", **d)]).fit_predict(df)

    @staticmethod
    def _last(out, name):
        return out["m"].struct.field(name).to_list()[-1]

    def _learnable(self, n=8000, seed=0, noise=1.0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        return pl.DataFrame({"x0": x, "y0": 2 * x + noise * rng.standard_normal(n)})

    def test_fields_are_opt_in(self):
        plain = po.spec.ewridge("m", targets=["y0"], features=["x0"], half_life=100.0)
        assert not any(
            f.startswith(("ic_", "r2_", "hit_rate_")) for f in po.spec.output_fields(plain)
        )
        spec = po.spec.ewridge(
            "m", targets=["y0"], features=["x0"], half_life=100.0, emit_metrics=True
        )
        for f in ("ic_y0", "r2_y0", "hit_rate_y0"):
            assert f in po.spec.output_fields(spec)

    def test_agrees_with_the_batch_evaluator(self):
        # With no decay the streaming metrics and eval.py measure the same
        # thing, so they must land in the same place.
        out = self._out(self._learnable())
        batch = po.eval.metrics(out, "m", targets=["y0"]).row(0, named=True)
        assert self._last(out, "ic_y0") == pytest.approx(batch["ic"], abs=0.02)
        assert self._last(out, "r2_y0") == pytest.approx(batch["r2"], abs=0.02)
        assert self._last(out, "hit_rate_y0") == pytest.approx(batch["hit_rate"], abs=0.02)

    def test_reports_a_good_fit_as_good(self):
        out = self._out(self._learnable(noise=0.1))
        assert self._last(out, "ic_y0") > 0.95
        assert self._last(out, "r2_y0") > 0.9
        assert self._last(out, "hit_rate_y0") > 0.9

    def test_reports_a_useless_fit_as_useless(self):
        rng = np.random.default_rng(2)
        n = 8000
        df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n)})
        out = self._out(df)
        assert abs(self._last(out, "ic_y0")) < 0.1
        assert self._last(out, "r2_y0") < 0.05
        assert abs(self._last(out, "hit_rate_y0") - 0.5) < 0.08

    @pytest.mark.parametrize("row", [25, 100, 777, 1999])
    def test_metrics_are_out_of_sample(self, row):
        """The metrics and the autocorrelation reported on a row do not
        include that row's own outcome: move ``y`` on the row by 50, across
        zero, and every one of them on that row is the same to the bit, as
        ``sigma`` is held above, while the next row's moves (the hit rate
        with it, since the sign flips). It asserted only that the first
        reported row was not row 0, which an in-sample metric passes too
        (review 2026-10-06, TB4)."""
        df = self._learnable(n=2000)
        fields = ("ic_y0", "r2_y0", "hit_rate_y0", "autocorr_y0")

        def fit(frame):
            return self._out(frame, emit_autocorr=True)["m"].struct.unnest().select(fields)

        y = df["y0"].to_numpy()
        moved = y - 50.0 * np.sign(y) * (np.arange(df.height) == row)
        base = fit(df)
        bumped = fit(df.with_columns(pl.Series("y0", moved)))
        for name in fields:
            here, there = base[name][row], bumped[name][row]
            assert here is not None and here == there, (name, here, there)
            if row + 1 < df.height:
                assert base[name][row + 1] != bumped[name][row + 1], (
                    f"{name}: the bump never arrived"
                )

    @pytest.mark.parametrize("model", ["sgd", "ftrl"])
    def test_a_logistic_fit_reads_the_metrics_as_a_classifiers(self, model):
        """The README's table for a logistic fit: ``hit_rate`` is the accuracy
        at a 0.5 threshold, ``r2`` the Brier skill score against the running
        base rate, ``ic`` the point-biserial correlation of probability and
        label, each over the rows scored before the row. With no decay the EW
        means are plain ones, so scikit-learn's ``accuracy_score`` and
        ``brier_score_loss`` and scipy's ``pointbiserialr`` compute them
        (review 2026-10-06, TB4: the table had no test)."""
        from scipy.stats import pointbiserialr
        from sklearn.metrics import accuracy_score, brier_score_loss

        rng = np.random.default_rng(12)
        n = 400
        x = rng.standard_normal(n)
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * x))).astype(float)
        df = pl.DataFrame({"x0": x, "y0": y})
        loss = {"loss": "logistic"} if model == "sgd" else {}
        spec = getattr(po.spec, model)(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=float("inf"),
            min_weight=5.0,
            emit_metrics=True,
            **loss,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        p = out["pred_y0"].to_numpy().astype(float)
        scored = np.isfinite(p)
        checked = 0
        for i in range(n):
            seen = scored[:i]
            ps, ys = p[:i][seen], y[:i][seen]
            if ys.size < 3 or ys.min() == ys.max():
                continue
            base = brier_score_loss(ys, np.full(ys.size, ys.mean()))
            assert out["hit_rate_y0"][i] == pytest.approx(
                accuracy_score(ys > 0.5, ps > 0.5), abs=1e-12
            )
            assert out["r2_y0"][i] == pytest.approx(1.0 - brier_score_loss(ys, ps) / base, abs=1e-9)
            assert out["ic_y0"][i] == pytest.approx(pointbiserialr(ys, ps).statistic, abs=1e-9)
            checked += 1
        assert checked > 350, checked

    def test_decay_lets_metrics_forget(self):
        # A model that was good and then breaks should report a falling IC.
        rng = np.random.default_rng(3)
        n = 12000
        x = rng.standard_normal(n)
        sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)
        df = pl.DataFrame({"x0": x, "y0": sign * 2 * x + 0.1 * rng.standard_normal(n)})
        out = self._out(df, half_life=300.0)
        ic = out["m"].struct.field("ic_y0").to_list()
        assert ic[n // 2 - 10] > 0.9, "should be good before the flip"
        assert ic[n // 2 + 200] < ic[n // 2 - 10], "should degrade after it"

    def test_chunk_invariance_and_save_load(self, tmp_path):
        df = self._learnable(n=600, seed=4)
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=float("inf"),
            min_weight=20.0,
            max_rows_between_solves=1,
            emit_metrics=True,
        )
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 61)) for i in range(0, df.height, 61)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 300))
        p = tmp_path / "m.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(300, 300)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)


def test_quantiles_are_null_until_a_residual_and_span_extreme_levels():
    """A quantile is null until the first residual has been folded in, and
    is that residual's bucket from the next row on (task 146: P² needed
    five); and it accepts levels out at 0.001 / 0.999, ordered (review
    2026-09-18, phase 4)."""
    out = po.ModelBank([_spec(min_weight=3.0, resid_quantiles=[0.5])]).fit_predict(_df(n=8))
    st = out["m"].struct
    name = next(f.name for f in out.schema["m"].fields if f.name.startswith("abs_resid_q"))
    got, r = st.field(name).to_list(), st.field("resid_y0").to_list()
    first = next(i for i, v in enumerate(r) if v is not None)
    assert got[: first + 1] == [None] * (first + 1)
    assert got[first + 1] == pytest.approx(abs(r[first]), rel=ALPHA)
    out2 = po.ModelBank([_spec(resid_quantiles=[0.001, 0.5, 0.999])]).fit_predict(_df(n=500))
    names = [f.name for f in out2.schema["m"].fields if f.name.startswith("abs_resid_q")]
    last = [out2["m"].struct.field(nm).to_list()[-1] for nm in names]
    assert all(np.isfinite(last)), last
    assert last[0] <= last[1] <= last[2], last


# --- The residual layer held to its definitions across kinds (task 278, batch 23) ---
#
# Each field below is read from the bank's own residuals and predictions, so
# what is held is the layer's arithmetic, not any model's fit: the
# definition is written here in closed form over the rows before each row
# (sums over explicit decayed weights), where the stream keeps a recursion.

GAP_CAP = 6.0
SESSION_GAP = 2.0


def _layer_frame(n=600, seed=0, *, zero_targets=False):
    """An irregular clock with capped gaps and session changes, uneven
    weights with zeros, null targets now and then, and (with
    ``zero_targets``) targets of exactly 0, which the hit rate leaves out."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.5, 1.0, 1.5], size=n)
    dt[0] = 0.0
    dt[rng.choice(np.arange(1, n), 6, replace=False)] = 20.0  # past gap_cap
    session = np.cumsum(np.isin(np.arange(n), rng.choice(np.arange(1, n), 5, replace=False)))
    x0 = rng.standard_normal(n)
    x1 = 1.0 + rng.standard_normal(n)
    # AR(1) noise, so the autocorrelation has something to read.
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = 0.5 * u[i - 1] + rng.standard_normal()
    y = 0.5 + 2.0 * x0 - x1 + 0.6 * u
    if zero_targets:
        y[rng.choice(n, n // 10, replace=False)] = 0.0
    y[rng.choice(n, 12, replace=False)] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[rng.choice(np.arange(1, n), n // 20, replace=False)] = 0.0
    frame = {"t": np.cumsum(dt), "s": session, "x0": x0, "x1": x1, "y0": y, "w": w}
    return pl.DataFrame(frame).with_columns(pl.col("y0").fill_nan(None))


def _clock_and_runs(df):
    """The clock the decay reads (each step capped at ``GAP_CAP``, a session
    change's step ``SESSION_GAP``) and the run each row is in: a capped gap
    or a session change starts a new one. Every row is accepted (no
    feature is ever null here)."""
    t, s = df["t"].to_numpy(), df["s"].to_numpy()
    clock, run = np.zeros(len(t)), np.zeros(len(t), dtype=int)
    for i in range(1, len(t)):
        changed = s[i] != s[i - 1]
        step = SESSION_GAP if changed else t[i] - t[i - 1]
        cut = changed or step > GAP_CAP
        clock[i] = clock[i - 1] + min(step, GAP_CAP)
        run[i] = run[i - 1] + int(cut)
    return clock, run


def _layer_spec(kind, half_life=40.0, **kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=GAP_CAP,
        session="s",
        session_gap=SESSION_GAP,
        weight="w",
        half_life=half_life,
        min_weight=5.0,
    )
    d.update(kw)
    if kind == "holt":
        d.pop("features")
    if kind == "kalman":
        # Its `min_weight` meets its learning in a way being changed
        # elsewhere; nothing here reads the fit, so it keeps the default.
        d.pop("min_weight")
        d.setdefault("coef_half_life", 100.0)
    if kind == "rls":
        d.setdefault("delta", 1.0)
    if kind == "sgd":
        d.setdefault("learning_rate", 0.02)
    if kind == "ewridge":
        d.setdefault("max_rows_between_solves", 1)
    return getattr(po.spec, kind)("m", **d)


def _ew_autocorr(resid, w, clock, run, lag, half_life):
    """The EW lag-``lag`` autocorrelation each row reads, by definition from
    the residuals before it.

    A *value* is a finite residual on a row of positive weight, each at unit
    weight decayed on the clock, ``u_j = 2^(-(c - c_j) / h)``. Its mean
    ``m`` and spread ``var = Σ u (x - m)² / Σ u`` are the values' weighted
    ones. A value pairs with the value ``lag`` values back in its run; the
    pair's product is ``(x_j - m_j)(x_p - m_{j-})``, ``m_j`` the mean with
    ``x_j`` and ``m_{j-}`` the one before it (the variance's own Welford
    centring), and ``cross`` is the pairs' mean of it at the same decayed
    unit weights. The field is ``cross / var``, clamped to [-1, 1]; null
    before the first pair or while ``var`` is 0."""
    n = len(resid)
    vals = np.flatnonzero(np.isfinite(resid) & (w > 0))
    x, c = resid[vals], clock[vals]
    m_after = np.array(
        [
            np.average(x[: k + 1], weights=0.5 ** ((c[k] - c[: k + 1]) / half_life))
            for k in range(len(vals))
        ]
    )
    m_before = np.r_[0.0, m_after[:-1]]
    partner = np.full(len(vals), -1)
    for k in range(lag, len(vals)):
        # Runs only grow, so the value `lag` back shares the run exactly
        # when every value between does.
        if run[vals[k]] == run[vals[k - lag]]:
            partner[k] = k - lag
    out = np.full(n, np.nan)
    for i in range(n):
        have = np.flatnonzero(vals < i)
        if have.size == 0:
            continue
        u = 0.5 ** ((clock[i] - c[have]) / half_life)
        m = np.average(x[have], weights=u)
        var = np.average((x[have] - m) ** 2, weights=u)
        pairs = have[partner[have] >= 0]
        if pairs.size == 0 or var <= 0.0:
            continue
        v = 0.5 ** ((clock[i] - c[pairs]) / half_life)
        prod = (x[pairs] - m_after[pairs]) * (x[partner[pairs]] - m_before[pairs])
        out[i] = np.clip(np.average(prod, weights=v) / var, -1.0, 1.0)
    return out


class TestAutocorrByDefinition:
    """``autocorr_<slot>`` against :func:`_ew_autocorr` on every row, at three
    lags, through capped gaps, session changes, zero weights and null
    targets. Measured 2026-10-26: 1.0e-15 at most; the tolerance is 1e-12.

    The row's weight does not enter: each value counts once, decayed, where
    ``sigma`` and the quantiles weigh it by the row's weight (an observation
    reported with this batch; the docs do not say either way)."""

    @pytest.mark.parametrize("lag", [1, 2, 5])
    def test_each_row_reads_the_pairs_before_it(self, lag):
        df = _layer_frame(seed=21)
        spec = _layer_spec("ewridge", emit_autocorr=True, resid_autocorr_lag=lag)
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        clock, run = _clock_and_runs(df)
        resid = out["resid_y0"].to_numpy().astype(float)
        got = out["autocorr_y0"].to_numpy().astype(float)
        want = _ew_autocorr(resid, df["w"].to_numpy(), clock, run, lag, 40.0)
        assert (np.isnan(got) == np.isnan(want)).all(), "null pattern"
        ok = np.isfinite(want)
        assert ok.sum() > 500, ok.sum()
        # The frame's runs: the pairs restart after each cut.
        assert run.max() >= 8, run.max()
        np.testing.assert_allclose(got[ok], want[ok], rtol=0, atol=1e-12)

    def test_the_lag_reads_its_own_structure(self):
        """Residuals of an AR(2) with no lag-1 term: lag 2 reads its 0.7,
        lag 1 near 0 (activation: the lag takes effect). Measured
        2026-10-26 at the last row: lag 1 -0.039, lag 2 0.690."""
        rng = np.random.default_rng(5)
        n = 6000
        x = rng.standard_normal(n)
        u = np.zeros(n)
        for i in range(2, n):
            u[i] = 0.7 * u[i - 2] + rng.standard_normal()
        df = pl.DataFrame({"x0": x, "y0": 2.0 * x + u})
        last = {}
        for lag in (1, 2):
            spec = po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                half_life=1000.0,
                min_weight=20.0,
                emit_autocorr=True,
                resid_autocorr_lag=lag,
            )
            out = po.ModelBank([spec]).fit_predict(df)
            last[lag] = out["m"].struct.field("autocorr_y0").to_list()[-1]
        assert abs(last[1]) < 0.15, last
        assert 0.55 < last[2] < 0.85, last

    def test_a_lag_longer_than_every_run_reads_null(self):
        """Sessions of three rows: lag 2 pairs inside each, lag 5 never."""
        rng = np.random.default_rng(6)
        n = 300
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 2 * x + rng.standard_normal(n), "s": np.arange(n) // 3})
        got = {}
        for lag in (2, 5):
            spec = po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                half_life=100.0,
                min_weight=5.0,
                session="s",
                session_gap=1.0,
                emit_autocorr=True,
                resid_autocorr_lag=lag,
            )
            got[lag] = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("autocorr_y0")
        assert got[2].is_not_null().sum() > 250
        assert got[5].is_null().all()


def _ew_quantile_before(absr, w, clock, i, half_life, p):
    seen = np.flatnonzero(np.isfinite(absr[:i]) & (w[:i] > 0))
    if seen.size == 0:
        return np.nan
    weight = w[seen] * 0.5 ** ((clock[i] - clock[seen]) / half_life)
    return _ew_quantile(absr[seen], weight, p)


class TestResidQuantilesAcrossKinds:
    """``abs_resid_q<p>_<slot>`` within the sketch's ``tanh(1/128)`` of the
    exponentially weighted quantile of ``|resid|``, each residual at its
    row's weight decayed on the capped clock at the model's half-life, on
    four kinds beyond ``ewridge``; and the levels ordered on every row."""

    LEVELS = [0.1, 0.5, 0.9]

    @pytest.mark.parametrize("kind", ["rls", "kalman", "sgd", "holt"])
    def test_each_level_is_the_weighted_ew_quantile(self, kind):
        df = _layer_frame(seed=22)
        spec = _layer_spec(kind, resid_quantiles=self.LEVELS)
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        clock, _ = _clock_and_runs(df)
        w = df["w"].to_numpy()
        absr = np.abs(out["resid_y0"].to_numpy().astype(float))
        qs = np.array([out[f"abs_resid_q{p}_y0"].to_numpy().astype(float) for p in self.LEVELS])
        rows = np.flatnonzero(np.isfinite(qs[0]))
        assert rows.size > 500, rows.size
        for i in rows:
            for li, p in enumerate(self.LEVELS):
                want = _ew_quantile_before(absr, w, clock, i, 40.0, p)
                assert abs(qs[li, i] - want) <= ALPHA * want * (1 + 1e-9), (kind, i, p)
        # Ordered on every row, not only the last.
        assert (np.diff(qs[:, rows], axis=0) >= 0).all()


def _ew_metrics_before(pred, y, w, clock, i, half_life):
    """``ic``, ``r2`` and ``hit_rate`` from the scored rows before row ``i``,
    each at its weight decayed on the clock: the weighted correlation of
    ``pred`` and ``y``; one less the weighted mean squared error over ``y``'s
    weighted variance about its weighted mean; the weighted share of signs
    hit among the rows with ``y != 0`` and ``pred != 0``."""
    seen = np.flatnonzero(np.isfinite(pred[:i]) & np.isfinite(y[:i]) & (w[:i] > 0))
    if seen.size < 2:
        return np.nan, np.nan, np.nan
    v = w[seen] * 0.5 ** ((clock[i] - clock[seen]) / half_life)
    p, t = pred[seen], y[seen]
    mp, mt = np.average(p, weights=v), np.average(t, weights=v)
    cov = np.average((p - mp) * (t - mt), weights=v)
    vp, vt = np.average((p - mp) ** 2, weights=v), np.average((t - mt) ** 2, weights=v)
    ic = cov / np.sqrt(vp * vt)
    r2 = 1.0 - np.average((t - p) ** 2, weights=v) / vt
    signed = (t != 0) & (p != 0)
    hit = np.average(np.sign(p[signed]) == np.sign(t[signed]), weights=v[signed])
    return ic, r2, hit


class TestMetricsByDefinition:
    @pytest.mark.parametrize("kind", ["ewridge", "rls"])
    def test_at_a_finite_half_life_each_metric_is_its_weighted_definition(self, kind):
        """With weights, zeros among them, targets of exactly 0 and capped
        gaps, at a half-life of 40 clock units. Measured 2026-10-26: 1.3e-15
        at most on the three; the tolerance is 1e-12."""
        df = _layer_frame(seed=23, zero_targets=True)
        spec = _layer_spec(kind, emit_metrics=True)
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        clock, _ = _clock_and_runs(df)
        pred, y = out["pred_y0"].to_numpy().astype(float), df["y0"].to_numpy().astype(float)
        w = df["w"].to_numpy()
        got = {k: out[f"{k}_y0"].to_numpy().astype(float) for k in ("ic", "r2", "hit_rate")}
        assert (y == 0).sum() > 50
        checked = 0
        for i in range(df.height):
            want = _ew_metrics_before(pred, y, w, clock, i, 40.0)
            if not np.isfinite(want[0]):
                continue
            for k, v in zip(("ic", "r2", "hit_rate"), want, strict=True):
                assert got[k][i] == pytest.approx(v, abs=1e-12), (k, i)
            checked += 1
        assert checked > 500, checked

    def test_run_once_the_last_row_is_the_batch_evaluator_exactly(self):
        """No decay, unit weights: the last row's metrics read every scored
        row before it, which is ``po.eval.metrics`` over those rows. Measured
        2026-10-26: 4.4e-16 at most."""
        rng = np.random.default_rng(24)
        n = 3000
        x = rng.standard_normal(n)
        y = 2 * x + rng.standard_normal(n)
        y[rng.choice(n, 200, replace=False)] = 0.0
        df = pl.DataFrame({"x0": x, "y0": y})
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=float("inf"),
            min_weight=20.0,
            emit_metrics=True,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        last = out["m"].struct.unnest().row(n - 1, named=True)
        batch = po.eval.metrics(out.head(n - 1), "m", targets=["y0"]).row(0, named=True)
        for k in ("ic", "r2", "hit_rate"):
            assert last[f"{k}_y0"] == pytest.approx(batch[k], rel=1e-12, abs=1e-14), k


GRIDS = {
    "rls": {"half_life": [15.0, 60.0, 400.0]},
    "kalman": {"half_life": [15.0, 60.0, 400.0], "coef_half_life": 50.0},
    "huber": {"half_life": [15.0, 60.0, 400.0], "max_rows_between_solves": 1},
    "sgd": {"half_life": [15.0, 60.0, 400.0]},
    "holt": {"half_life": [15.0, 60.0, 400.0]},
}


class TestSelectionAndAveragingAcrossKinds:
    """``emit_selected`` and ``emit_averaged`` on half-life grids of five
    kinds beyond ``ewridge``: the selected slot is the argmin of the slots'
    ``sigma`` on the row, its prediction that slot's, and the average the
    documented weights ``exp(-eta (sigma² / sigma²_best - 1))`` over the
    slots with a prediction and a ``sigma``. Read from the bank's own
    columns, so no model's fit is pinned."""

    @pytest.mark.parametrize("kind", list(GRIDS))
    def test_the_selection_is_the_argmin_and_the_average_its_formula(self, kind):
        eta = 0.7
        df = _layer_frame(seed=25)
        spec = _layer_spec(
            kind,
            emit_sigma=True,
            emit_selected=True,
            emit_averaged=True,
            average_eta=eta,
            **GRIDS[kind],
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        tags = [f"@h{h:g}" for h in GRIDS[kind]["half_life"]]
        preds = np.array([out[f"pred_y0{t}"].to_numpy().astype(float) for t in tags])
        sig = np.array([out[f"sigma_y0{t}"].to_numpy().astype(float) for t in tags])
        names = out["selected_y0"].cast(pl.String).to_list()
        chosen = out["pred_y0__selected"].to_numpy().astype(float)
        avg = out["pred_y0__averaged"].to_numpy().astype(float)
        full = np.isfinite(preds).all(axis=0) & np.isfinite(sig).all(axis=0)
        assert full.sum() > 400, full.sum()
        switched = set()
        for i in np.flatnonzero(full):
            best = tags[int(np.argmin(sig[:, i]))]
            assert names[i] == best, (i, names[i], best)
            assert chosen[i] == preds[tags.index(best), i], i
            switched.add(best)
            s2 = sig[:, i] ** 2
            wts = np.exp(-eta * (s2 / s2.min() - 1.0))
            assert avg[i] == pytest.approx(np.dot(wts, preds[:, i]) / wts.sum(), rel=1e-12), i
        # More than one slot was best somewhere, so the argmin was tested.
        assert len(switched) >= 2, switched

    def test_the_average_moves_toward_the_selection_as_eta_grows(self):
        """Two slots: ``averaged - selected = w_other (p_other - p_best)``
        with ``w_other = 1 / (1 + exp(eta (r - 1)))``, ``r >= 1`` the other's
        error over the best's, which cannot rise with ``eta``. So on every
        row ``|averaged - selected|`` is non-increasing in ``eta``
        (derived), here on a ``kalman`` grid of two half-lives."""
        df = _layer_frame(seed=26)
        gaps = []
        for eta in (1e-3, 0.1, 1.0, 10.0, 100.0):
            spec = _layer_spec(
                "kalman",
                half_life=[15.0, 400.0],
                coef_half_life=50.0,
                emit_selected=True,
                emit_averaged=True,
                average_eta=eta,
            )
            out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
            a = out["pred_y0__averaged"].to_numpy().astype(float)
            s = out["pred_y0__selected"].to_numpy().astype(float)
            gaps.append(np.abs(a - s))
        gaps = np.array(gaps)
        ok = np.isfinite(gaps).all(axis=0)
        assert ok.sum() > 500
        # Rounding of a sum of two terms: 1e-12 of the predictions' size.
        assert (np.diff(gaps[:, ok], axis=0) <= 1e-12).all()
        assert gaps[0, ok].mean() > 10 * gaps[-1, ok].mean()

    def test_on_a_three_slot_grid_the_mean_gap_falls_with_eta(self):
        """Three slots: per row the gap need not fall (the average can pass
        the best on its way), but over the stream it does. Measured
        2026-10-26 on this ``rls`` grid: 0.0708, 0.0703, 0.0657, 0.0361 and
        0.0057 at eta 1e-3, 0.1, 1, 10 and 100."""
        df = _layer_frame(seed=27)
        means = []
        for eta in (1e-3, 0.1, 1.0, 10.0, 100.0):
            spec = _layer_spec(
                "rls",
                half_life=[15.0, 60.0, 400.0],
                emit_selected=True,
                emit_averaged=True,
                average_eta=eta,
            )
            out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
            a = out["pred_y0__averaged"].to_numpy().astype(float)
            s = out["pred_y0__selected"].to_numpy().astype(float)
            means.append(np.nanmean(np.abs(a - s)))
        assert all(b < a for a, b in zip(means, means[1:], strict=False)), means

    @pytest.mark.parametrize("kind", ["rls", "sgd"])
    def test_selection_and_average_survive_a_save_and_load(self, kind, tmp_path):
        df = _layer_frame(seed=28)
        spec = _layer_spec(kind, emit_selected=True, emit_averaged=True, **GRIDS[kind])
        whole = po.ModelBank([spec]).fit_predict(df)
        bank = po.ModelBank([spec])
        head = bank.fit_predict(df.slice(0, 250))
        bank.save(tmp_path / "s.state")
        tail = po.ModelBank.load(tmp_path / "s.state", specs=[spec]).fit_predict(df.slice(250))
        both = pl.concat([head, tail])["m"].struct.unnest()
        one = whole["m"].struct.unnest()
        for col in ("selected_y0", "pred_y0__selected", "pred_y0__averaged"):
            assert both[col].is_not_null().sum() > 400, col
            assert both[col].equals(one[col], null_equal=True), col
