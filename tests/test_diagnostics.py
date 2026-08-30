"""E21 / E12: `emit_sigma` and `emit_resid_z`.

`sigma_<slot>` is the EW standard deviation of that slot's out-of-sample
residuals, read from the state *before* each row; `resid_z_<slot>` is
`resid / sigma`. Both are computed in the stream layer rather than per model,
so every model gets the same definition (the models' own internal `sigma2`
fields serve different purposes and are not all present or comparable).
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("rls", {"ridge": 1.0}),
    ("kalman", {"coef_halflife": 100.0}),
    ("lasso", {"lasso_path": [0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {}),
]
IDS = [m[0] for m in MODELS]


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
        halflife=100.0,
        min_periods=10.0,
        emit_sigma=True,
        emit_resid_z=True,
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
        spec = po.spec.ewridge("m", targets=["y0"], features=["x0"], halflife=100.0)
        fields = po.spec.output_fields(spec)
        assert not any(f.startswith(("sigma_", "resid_z_")) for f in fields)

    def test_present_when_requested(self):
        assert po.spec.output_fields(_spec()) == [
            "pred_y0",
            "resid_y0",
            "sigma_y0",
            "resid_z_y0",
            "n_eff",
            "coef",
        ]

    def test_each_flag_is_independent(self):
        only_sigma = _spec(emit_resid_z=False)
        only_z = _spec(emit_sigma=False)
        assert "sigma_y0" in po.spec.output_fields(only_sigma)
        assert "resid_z_y0" not in po.spec.output_fields(only_sigma)
        assert "resid_z_y0" in po.spec.output_fields(only_z)
        assert "sigma_y0" not in po.spec.output_fields(only_z)

    def test_one_field_per_slot_in_a_grid(self):
        spec = po.spec.ewridge(
            "m",
            targets=["y0", "y1"],
            features=["x0"],
            ridge=[1e-6, 1.0],
            halflife=100.0,
            emit_sigma=True,
            emit_resid_z=True,
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
        z = _f(out, "resid_z_y0")
        assert np.nanmedian(np.abs(z)) < 1.5
        assert abs(z[300]) > 10.0
        assert int(np.nanargmax(np.abs(z))) == 300

    def test_resid_z_equals_resid_over_sigma(self):
        out = po.ModelBank([_spec()]).fit_predict(_df())
        r, s, z = (_f(out, f"{k}_y0") for k in ("resid", "sigma", "resid_z"))
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
        out = po.ModelBank([_spec(min_periods=20.0)]).fit_predict(_df())
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
        r, s, z = (_slot(out, f"{k}_y0") for k in ("resid", "sigma", "resid_z"))
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
        keep = [c for c in one.columns if not c.startswith("coef")]
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
    halflives: for each target, pick the slot with the lowest EW out-of-sample
    error so far (the `sigma` already tracked for E12) and emit that slot's
    prediction plus its label.
    """

    def _grid_spec(self, **kw):
        d = dict(
            targets=["y0"],
            features=["x0"],
            ridge=[1e-8, 1.0, 1000.0],
            halflife=300.0,
            min_periods=20.0,
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
        assert sel.count("r0.00000001") / len(sel) > 0.9

    def test_picks_shrinkage_on_pure_noise(self):
        rng = np.random.default_rng(1)
        n = 4000
        x = rng.standard_normal(n)
        out = self._selection(rng.standard_normal(n), x)
        sel = [s for s in out["m"].struct.field("selected_y0").to_list()[-1500:] if s]
        # the near-zero ridge fits the noise and must not dominate
        assert sel.count("r0.00000001") / len(sel) < 0.2

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
        spec = self._grid_spec(ridge=1e-6, halflife=[20.0, 2000.0])
        out = po.ModelBank([spec]).fit_predict(
            pl.DataFrame({"x0": x, "y0": 2 * x + 0.2 * rng.standard_normal(n)})
        )
        sel = {s for s in out["m"].struct.field("selected_y0").to_list() if s}
        assert sel <= {"@h20", "@h2000"}
        assert sel, "nothing was ever selected"

    def test_requires_more_than_one_slot(self):
        with pytest.raises(ValueError, match="more than one slot"):
            po.spec.ewridge(
                "m", targets=["y0"], features=["x0"], halflife=100.0, emit_selected=True
            )

    def test_rejected_for_ew_cov(self):
        with pytest.raises(ValueError, match="does not apply to ew_cov"):
            po.spec.ew_cov("c", features=["x0", "x1"], halflife=100.0, emit_selected=True)

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
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)


class TestDriftDetection:
    """E20: `emit_drift` — Page-Hinkley on each slot's absolute out-of-sample
    residual, scaled by that slot's own EW residual std.

    Decay and drift detection answer different questions: a halflife forgets
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
            halflife=1e5,
            min_periods=20.0,
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
        plain = po.spec.ewridge("m", targets=["y0"], features=["x0"], halflife=100.0)
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

    def test_threshold_trades_sensitivity_for_noise(self):
        df = self._df()
        eager, _ = self._flags(df, drift_threshold=2.0)
        patient, _ = self._flags(df, drift_threshold=200.0)
        assert eager.sum() >= patient.sum()

    def test_delta_absorbs_small_shifts(self):
        df = self._df()
        tolerant, _ = self._flags(df, drift_delta=50.0)
        assert tolerant.sum() == 0, "a huge delta should absorb everything"

    def test_reset_action_restarts_the_model(self):
        df = self._df(flip_at=3000)
        flags, out = self._flags(df, drift_action="reset")
        hits = np.flatnonzero(flags)
        assert len(hits) >= 1
        n_eff = out["m"].struct.field("n_eff").to_numpy().astype(float)
        # n_eff climbs monotonically unless something resets it
        assert n_eff[hits[0] + 1] < n_eff[hits[0]], "the reset did not take effect"

    def test_flag_action_leaves_the_model_running(self):
        df = self._df(flip_at=3000)
        flags, out = self._flags(df, drift_action="flag")
        n_eff = out["m"].struct.field("n_eff").to_numpy().astype(float)
        hits = np.flatnonzero(flags)
        assert len(hits) >= 1
        assert n_eff[hits[0] + 1] > n_eff[hits[0]], "flag-only should not reset"

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
        keep = [c for c in one.columns if not c.startswith("coef")]
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
