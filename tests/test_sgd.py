"""E16: SGD with pluggable losses (ENHANCEMENTS E16).

The cheap baseline — one gradient step per row, no solves — and the only model
here that takes count targets, via the Poisson loss with a log link.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from conftest import run_online

TIER = "mixed"


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        half_life=float("inf"),
        min_weight=10.0,
        learning_rate=0.05,
    )
    d.update(kw)
    return po.spec.sgd("m", **d)


def _fit(df, **kw):
    out = po.ModelBank([_spec(coef_every=1, **kw)]).fit_predict(df)
    return np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float), out


def _linear(n=20000, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    return pl.DataFrame(
        {"x0": x0, "x1": x1, "y0": 1.5 * x0 - 0.5 * x1 + 0.25 + noise * rng.standard_normal(n)}
    )


class TestLosses:
    def test_squared_recovers_the_coefficients(self):
        c, _ = _fit(_linear())
        assert c[0] == pytest.approx(0.25, abs=0.05)
        assert c[1] == pytest.approx(1.5, abs=0.05)
        assert c[2] == pytest.approx(-0.5, abs=0.05)

    def test_poisson_recovers_a_log_rate(self):
        rng = np.random.default_rng(0)
        n = 30000
        x = rng.standard_normal(n)
        y = rng.poisson(np.exp(0.4 + 0.8 * x)).astype(float)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        c, out = _fit(df, loss="poisson", learning_rate=0.02)
        assert c[0] == pytest.approx(0.4, abs=0.15), f"log-intercept {c[0]}"
        assert c[1] == pytest.approx(0.8, abs=0.15), f"log-slope {c[1]}"
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        assert np.nanmin(p) >= 0.0, "a Poisson rate cannot be negative"
        # Compare the *converged* predictions: the mean over the whole stream is
        # dominated by the early rows, where the fit is still finding the scale.
        assert np.nanmean(p[-10000:]) == pytest.approx(y[-10000:].mean(), rel=0.2)

    def test_poisson_needs_the_default_gradient_clip(self):
        # The default clip_gradient is finite precisely because of this: with a
        # log link one large count makes the next gradient exponentially bigger.
        rng = np.random.default_rng(0)
        n = 30000
        x = rng.standard_normal(n)
        y = rng.poisson(np.exp(0.4 + 0.8 * x)).astype(float)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        default, _ = _fit(df, loss="poisson", learning_rate=0.02)
        unclipped, _ = _fit(df, loss="poisson", learning_rate=0.02, clip_gradient=1e12)
        assert abs(default[0]) < 1.0, "the default should be stable"
        assert abs(unclipped[0]) > 1e3, "expected the unclipped fit to diverge"

    def test_clip_does_not_bind_for_squared_loss(self):
        df = _linear(n=5000)
        with_clip, _ = _fit(df)
        without, _ = _fit(df, clip_gradient=1e12)
        np.testing.assert_allclose(with_clip, without, rtol=0, atol=0)

    @pytest.mark.parametrize(("tau", "other"), [(0.1, 0.9)])
    def test_quantile_levels_are_ordered(self, tau, other):
        rng = np.random.default_rng(3)
        n = 20000
        df = pl.DataFrame({"x0": np.zeros(n), "x1": np.zeros(n), "y0": 1.0 + 2.0 * rng.random(n)})
        lo, _ = _fit(df, loss="quantile", quantile=tau)
        hi, _ = _fit(df, loss="quantile", quantile=other)
        assert lo[0] < hi[0]

    def test_an_infinite_huber_delta_is_the_squared_loss(self):
        """Huber's gradient is the residual clipped at ``delta``, and at
        ``inf`` nothing is clipped: the squared loss's fit, to the bit (review
        2026-09-12, S27). The builder refused it."""
        df = _linear(n=5000, noise=1.0)
        _, hub = _fit(df, loss="huber", huber_delta=float("inf"))
        _, sq = _fit(df, loss="squared")
        assert hub.equals(sq, null_equal=True)

    def test_huber_beats_squared_under_contamination(self):
        rng = np.random.default_rng(4)
        n = 20000
        x = rng.standard_normal(n)
        y = 2.0 * x
        bad = rng.random(n) < 0.05
        y[bad] = 500.0 * rng.standard_normal(bad.sum())
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        hub, _ = _fit(df, loss="huber", huber_delta=1.0)
        sq, _ = _fit(df, loss="squared")
        assert abs(hub[1] - 2.0) < abs(sq[1] - 2.0)

    def test_logistic_predicts_probabilities(self):
        rng = np.random.default_rng(5)
        n = 10000
        x = rng.standard_normal(n)
        y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(float)
        df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})
        _, out = _fit(df, loss="logistic")
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        finite = p[np.isfinite(p)]
        assert ((finite >= 0) & (finite <= 1)).all()

    def test_epsilon_insensitive_fits_with_either_schedule(self):
        # Both reach the slope on this data. Which one wins depends on the
        # hyperparameters, so no ordering is asserted here; the controlled
        # demonstration that the sign-valued subgradient needs annealing to
        # *settle* lives in the Rust unit tests.
        df = _linear(n=30000, noise=0.1)
        const, _ = _fit(df, loss="epsilon_insensitive", eps=0.2, learning_rate=0.01)
        anneal, _ = _fit(
            df,
            loss="epsilon_insensitive",
            eps=0.2,
            learning_rate=0.5,
            schedule="inv_scaling",
            power=0.5,
        )
        for name, c in (("constant", const), ("annealed", anneal)):
            assert c[1] == pytest.approx(1.5, abs=0.1), f"{name}: {c}"
            assert c[2] == pytest.approx(-0.5, abs=0.1), f"{name}: {c}"


class TestSchedules:
    @pytest.mark.parametrize(
        ("schedule", "lr"), [("constant", 0.05), ("adagrad", 0.5), ("inv_scaling", 0.5)]
    )
    def test_all_schedules_converge(self, schedule, lr):
        # `power` is inv_scaling's alone, and refused beside another schedule.
        power = {"power": 0.25} if schedule == "inv_scaling" else {}
        c, _ = _fit(_linear(n=30000), schedule=schedule, learning_rate=lr, **power)
        assert c[1] == pytest.approx(1.5, abs=0.15), f"{schedule}: {c}"
        assert c[2] == pytest.approx(-0.5, abs=0.15), f"{schedule}: {c}"

    def test_l2_shrinks(self):
        df = _linear(n=5000)
        plain, _ = _fit(df)
        shrunk, _ = _fit(df, l2=1.0)
        assert abs(shrunk[1]) < abs(plain[1])


class TestPlumbing:
    def test_chunk_invariance(self):
        df = _linear(n=400, seed=9)
        spec = _spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    def test_save_load(self, tmp_path):
        df = _linear(n=400, seed=10)
        spec = _spec(schedule="adagrad", learning_rate=0.5)
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 200))
        p = tmp_path / "s.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(200, 200)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_out_of_sample_on_noise(self):
        rng = np.random.default_rng(11)
        n = 5000
        df = pl.DataFrame(
            {
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "y0": rng.standard_normal(n),
            }
        )
        _, out = _fit(df, half_life=2000.0)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(p)
        assert abs(np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]) < 0.06

    def test_bad_config_rejected(self):
        with pytest.raises(ValueError, match="unknown sgd loss"):
            _spec(loss="hinge")
        with pytest.raises(ValueError, match="unknown sgd schedule"):
            _spec(schedule="cosine")
        with pytest.raises(ValueError, match="learning_rate"):
            _spec(learning_rate=0.0)
        with pytest.raises(ValueError, match="needs a .quantile. level"):
            _spec(loss="quantile")

    @pytest.mark.parametrize(
        ("kw", "says"),
        [
            (dict(huber_delta=0.5), 'sgd huber_delta is for loss "huber"; loss "squared"'),
            (dict(huber_delta=0.5, loss="logistic"), 'huber_delta is for loss "huber"; loss "log'),
            (dict(quantile=0.9), 'sgd quantile is for loss "quantile"; loss "squared"'),
            (dict(quantile=0.9, loss="huber"), 'quantile is for loss "quantile"; loss "huber"'),
            (dict(eps=5.0), 'sgd eps is for loss "epsilon_insensitive"; loss "squared"'),
            (dict(eps=5.0, loss="poisson"), 'eps is for loss "epsilon_insensitive"; loss "poi'),
            (dict(power=0.9), 'sgd power is for schedule "inv_scaling"; schedule "constant"'),
            (dict(power=0.9, schedule="adagrad"), 'is for schedule "inv_scaling"; schedule "ada'),
        ],
    )
    def test_a_parameter_of_another_loss_or_schedule_is_refused_by_name(self, kw, says):
        """Review 2026-10-05 (YA8): each of these went through and was ignored,
        the output equal to the spec without it, where a parameter whose
        switch is off is refused (:mod:`polars_online.spec`)."""
        with pytest.raises(ValueError, match='^spec "m": sgd ') as e:
            _spec(**kw)
        assert says in str(e.value), str(e.value)
        assert "does not use it" in str(e.value), str(e.value)

    @pytest.mark.parametrize(
        "kw",
        [
            dict(loss="huber", huber_delta=0.5),
            dict(loss="quantile", quantile=0.9),
            dict(loss="epsilon_insensitive", eps=0.2),
            dict(schedule="inv_scaling", power=0.25),
        ],
    )
    def test_each_parameter_is_taken_by_the_loss_or_schedule_it_belongs_to(self, kw):
        assert {k: _spec(**kw)["model"][k] for k in kw} == kw


class TestFeatureScaling:
    """E24: `standardize` standardizes inputs against their running moments.

    Gradient methods are the ones that need it: a single learning rate has to
    suit every coordinate, so a feature in thousands and one in thousandths
    cannot both converge. The exact solvers do not care.
    """

    def _fit(self, scale):
        rng = np.random.default_rng(0)
        n = 20000
        x0 = 1000.0 * rng.standard_normal(n)
        x1 = 0.001 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x0, "x1": x1, "y0": 0.002 * x0 + 900.0 * x1})
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            learning_rate=0.01,
            half_life=float("inf"),
            min_weight=0.0,
            standardize=scale,
            coef_every=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        return np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float)

    @staticmethod
    def _rel_err(c):
        return abs(c[1] - 0.002) / 0.002 + abs(c[2] - 900.0) / 900.0

    def test_rescues_badly_scaled_features(self):
        plain, scaled = self._fit(False), self._fit(True)
        assert self._rel_err(scaled) < self._rel_err(plain)
        assert self._rel_err(scaled) < 0.2, f"scaled fit still poor: {scaled}"

    def test_coefficients_come_back_in_original_units(self):
        c = self._fit(True)
        assert c[1] == pytest.approx(0.002, rel=0.15)
        assert c[2] == pytest.approx(900.0, rel=0.15)

    def test_on_by_default(self):
        """As for ``kalman`` (docs/PLAN.md task 195, U2): one learning rate has
        to suit every feature, which only standardized features let it do."""
        spec = po.spec.sgd(
            "m", targets=["y0"], features=["x0"], half_life=100.0, learning_rate=0.01
        )
        assert spec["model"]["standardize"] is True

    def test_chunk_invariance(self):
        rng = np.random.default_rng(3)
        n = 400
        x0 = 100.0 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x0, "x1": rng.standard_normal(n), "y0": 0.01 * x0})
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            learning_rate=0.05,
            half_life=float("inf"),
            min_weight=5.0,
            standardize=True,
        )
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    # The moments a row is standardized against include the row (PLAN task 74,
    # sklearn's `partial_fit` then `transform`). Against the moments from
    # *before* it, a two-row variance can be tiny by chance, the standardized
    # value huge, and one step throws a coefficient the rest of the group never
    # brings back. The condition is few rows per feature: the start of every
    # group, and every row of a wide fit.

    @staticmethod
    def _groups(n_groups, rows_per, k, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_groups * rows_per, k))
        beta = np.repeat(rng.standard_normal((n_groups, k)) / np.sqrt(k), rows_per, axis=0)
        y = (x * beta).sum(axis=1) + 0.1 * rng.standard_normal(len(x))
        df = pl.DataFrame({f"x{j}": x[:, j] for j in range(k)}).with_columns(
            y0=pl.Series(y), g=pl.Series(np.repeat(np.arange(n_groups), rows_per))
        )
        return x, y, df

    @staticmethod
    def _r2(pred, y, ok):
        err = y[ok] - pred[ok]
        return 1.0 - float(err @ err) / float(((y[ok] - y[ok].mean()) ** 2).sum())

    def test_learns_from_a_short_history(self):
        """`scripts/sklearn_comparison.py short` as a test: 500 groups of 200
        rows, `k = 20`, scored by position in the group. Before task 74 this
        read R² −6.9 at rows 25–50 and 0.11 at rows 100–200; now 0.43 and
        0.91. `test_learns_a_short_history_as_sgdregressor_does` holds it to
        sklearn's `SGDRegressor` live, where this docstring quoted 0.45 and
        0.91."""
        n_groups, rows_per, k = 500, 200, 20
        x, y, df = self._groups(n_groups, rows_per, k)
        pos = np.tile(np.arange(rows_per), n_groups)
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=[f"x{j}" for j in range(k)],
            group="g",
            learning_rate=0.01,
            half_life=float("inf"),
            min_weight=25.0,
            standardize=True,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        early = self._r2(pred, y, (pos >= 25) & (pos < 50))
        late = self._r2(pred, y, (pos >= 100) & (pos < 200))
        assert early > 0.4, f"rows 25-50: R2 {early}"
        assert late > 0.85, f"rows 100-200: R2 {late}"

    @pytest.mark.extended(
        reason="a second or more: scikit-learn's SGDRegressor, refitted per row (2.6 s)"
    )
    def test_learns_a_short_history_as_sgdregressor_does(self):
        """The same groups, with scikit-learn's `SGDRegressor` beside it at the
        same constant rate, one estimator and one `StandardScaler` per group
        fed row by row, each row predicted before it is learned (task 121).
        Statistical tier: sklearn standardizes a row against the moments
        from before it and ours with the row in, so the two agree to a
        tolerance (its default `l2` penalty is off, so the scaler is the
        whole of the gap; review 2026-09-26, F9). Measured 0.018 apart at
        rows 25-50 and 0.005 at rows 100-200, over three seeds and sizes
        from 60 to 100 groups; fewer groups here, since sklearn's row-by-row
        loop is the slow half."""
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        n_groups, rows_per, k = 40, 200, 20
        x, y, df = self._groups(n_groups, rows_per, k)
        pos = np.tile(np.arange(rows_per), n_groups)
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=[f"x{j}" for j in range(k)],
            group="g",
            learning_rate=0.01,
            half_life=float("inf"),
            min_weight=25.0,
            standardize=True,
        )
        ours = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y0").to_numpy()
        theirs = np.full(len(y), np.nan)
        for gi in range(n_groups):
            lo = gi * rows_per
            model = SGDRegressor(random_state=0, learning_rate="constant", eta0=0.01, penalty=None)
            scaler = StandardScaler()
            for i in range(lo, lo + rows_per):
                xi = x[i : i + 1]
                if i > lo:
                    theirs[i] = model.predict(scaler.transform(xi))[0]
                scaler.partial_fit(xi)
                model.partial_fit(scaler.transform(xi), y[i : i + 1])
        for lo, hi in ((25, 50), (100, 200)):
            ok = (pos >= lo) & (pos < hi)
            a, b = self._r2(ours.astype(float), y, ok), self._r2(theirs, y, ok)
            assert abs(a - b) < 0.03, f"rows {lo}-{hi}: R2 {a} against sklearn's {b}"

    def test_matches_a_numpy_replica(self):
        """The model against a numpy LMS that standardizes each row against
        Welford moments updated with the row first, the way sklearn's recipe
        reads (`scaler.partial_fit(x)`, then `transform(x)`). While the scaler
        warms up, its first 22 rows (docs/PLAN.md task 206), one `z` serves
        the prediction and the gradient, `beta -= lr * (z . beta - y) * z`; on
        the 22nd row `beta` is read out with the moments as they stand, `b_i =
        beta_i / s_i` and `b_0 = beta_0 - sum_i b_i m_i`, and from then on the
        prediction is `[1, x] . b` and the step in `z` is mapped back by the
        row's means and scales. Agreement to 1e-12 relative on every
        prediction of a few groups."""
        n_groups, rows_per, k = 3, 200, 5
        lr, min_weight = 0.03, 10
        x, y, df = self._groups(n_groups, rows_per, k, seed=1)
        spec = po.spec.sgd(
            "m",
            targets=["y0"],
            features=[f"x{j}" for j in range(k)],
            group="g",
            learning_rate=lr,
            half_life=float("inf"),
            min_weight=float(min_weight),
            standardize=True,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)

        want = np.full(len(y), np.nan)
        for gi in range(n_groups):
            mean, m2, beta = np.zeros(k), np.zeros(k), np.zeros(k + 1)
            for n, i in enumerate(range(gi * rows_per, (gi + 1) * rows_per), start=1):
                xi = x[i]
                delta = xi - mean
                mean = mean + delta / n
                m2 = m2 + delta * (xi - mean)
                var = m2 / n
                scale = np.where(var > 0.0, np.sqrt(var), 1.0)
                z = np.concatenate(([1.0], (xi - mean) / scale))
                held = n > 22
                p = beta[0] + xi @ beta[1:] if held else z @ beta
                if n - 1 >= min_weight:
                    want[i] = p
                step = -lr * np.clip((p - y[i]) * z, -1e3, 1e3)
                if held:
                    step[1:] /= scale
                    step[0] -= mean @ step[1:]
                beta = beta + step
                if n == 22:
                    beta[1:] = beta[1:] / scale
                    beta[0] -= beta[1:] @ mean
        ok = np.isfinite(want)
        assert np.array_equal(ok, np.isfinite(pred))
        np.testing.assert_allclose(pred[ok], want[ok], rtol=1e-12, atol=1e-12)

    def test_wide_row_is_the_same_fit_scaled_or_not(self):
        """A wide fit has few rows per feature on *every* row, which is the
        short-history condition again. At `k = 1,000` over 20,000 rows of
        unit-variance features, `standardize=True` and `False` are now
        the same fit to within 0.001 R² (0.8512 against 0.8516 in
        `scripts/sklearn_comparison.py wide`); before task 74 the scaled
        fit's predictions correlated 0.978 with sklearn's at this width and
        0.52 at `k = 10,000`. Smaller here so the suite stays quick."""
        n, k = 4000, 200
        rng = np.random.default_rng(2)
        x = rng.standard_normal((n, k))
        beta = rng.standard_normal(k) / np.sqrt(k)
        y = x @ beta + 0.3 * rng.standard_normal(n)
        df = pl.DataFrame({f"x{j}": x[:, j] for j in range(k)}).with_columns(y0=pl.Series(y))
        preds = {}
        for scale in (False, True):
            spec = po.spec.sgd(
                "m",
                targets=["y0"],
                features=[f"x{j}" for j in range(k)],
                learning_rate=0.2 / k,
                half_life=float("inf"),
                min_weight=50.0,
                standardize=scale,
            )
            out = po.ModelBank([spec]).fit_predict(df)
            preds[scale] = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        ok = np.arange(n) >= 50
        unscaled, scaled = self._r2(preds[False], y, ok), self._r2(preds[True], y, ok)
        assert scaled > 0.5, f"scaled fit did not learn: R2 {scaled}"
        assert abs(scaled - unscaled) < 0.01, f"scaled {scaled} against unscaled {unscaled}"
        assert np.corrcoef(preds[False][ok], preds[True][ok])[0, 1] > 0.999


@pytest.mark.parametrize(
    ("extra", "msg"),
    [
        ({"huber_delta": 0.5}, 'sgd huber_delta is for loss "huber"; loss "squared"'),
        ({"quantile": 0.3}, 'sgd quantile is for loss "quantile"; loss "squared"'),
        ({"eps": 0.2}, 'sgd eps is for loss "epsilon_insensitive"; loss "squared"'),
        ({"power": 0.9}, 'sgd power is for schedule "inv_scaling"; schedule "constant"'),
    ],
    ids=["huber_delta", "quantile", "eps", "power"],
)
def test_a_raw_spec_with_a_parameter_nothing_reads_is_refused(extra, msg, tmp_path, online_cli):
    """Task 160, YA8b: a parameter of a loss or a schedule the spec does not
    use was taken and ignored. The builder refuses it (YA8); a raw dict, the
    form a JSON or TOML spec takes, skipped the builder and built, and is
    refused now by the validation every spec meets, as a TOML file is by the
    CLI's dry run."""
    spec = _spec()
    raw = dict(spec, model={**spec["model"], **extra})
    with pytest.raises(ValueError, match=f'spec "m": {msg} does not use it'):
        po.ModelBank([raw])
    _linear(n=50).write_parquet(tmp_path / "in.parquet")
    res = run_online(
        online_cli,
        tmp_path,
        [raw],
        input=tmp_path / "in.parquet",
        output=tmp_path / "out.parquet",
        args=["--dry-run"],
        check=False,
    )
    assert res.returncode != 0, res.stdout
    assert f"{msg} does not use it" in res.stderr, res.stderr
    # The builder's own dict carries each as a null, which is the default,
    # not a value given.
    assert all(spec["model"][k] is None for k in extra), spec["model"]
    po.ModelBank([spec])


def _cc4(scale_y=1.0, scale_x=1.0, n=3000):
    """Review round 4's CC4 stream: `y = 0.5 + 2x + 0.3·noise`, the target
    and the feature each scaled."""
    rng = np.random.default_rng(2)
    x, noise = rng.normal(size=n), rng.normal(size=n)
    return pl.DataFrame({"x": scale_x * x, "y": scale_y * (0.5 + 2.0 * x + 0.3 * noise)})


def _oos_r2(df, **kw):
    spec = po.spec.sgd("s", targets=["y"], features=["x"], half_life=1e9, min_weight=5.0, **kw)
    p = po.ModelBank([spec]).fit_predict(df)["s"].struct.field("pred_y").to_numpy()
    y = df["y"].to_numpy()
    ok = np.isfinite(p)
    return 1.0 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


class TestUnitFreeDefaults:
    """docs/PLAN.md task 195 (U1, U2; review round 4, CC4 and CC6): `sgd`
    standardizes by default, as `kalman` does, and its `huber_delta` is in
    units of the target's EW residual standard deviation, as `huber`'s is;
    its `eps` is in units of the target's own EW standard deviation (task
    202). A default in the data's units fitted one scale and failed the
    others."""

    def test_standardize_is_on_by_default(self):
        """Features times 100 at the defaults: R² -71847 with a raw step."""
        assert _oos_r2(_cc4(scale_x=100.0)) > 0.95
        assert _oos_r2(_cc4(scale_x=0.01)) > 0.95

    def test_the_huber_cut_binds_at_every_scale_of_the_target(self):
        """In the target's units the cut never bound on a target in
        thousandths (huber equal to squared on 2994 of 2994 rows) and bound
        on every row of one in thousands. In σ it binds on the tail rows at
        every scale, and the fit is the same fit scaled. The gradient's clip
        is a box in the target's units, so it is lifted here."""
        preds = {}
        for c in (2.0**-10, 1.0, 2.0**10):
            df = _cc4(scale_y=c)
            for loss in ("huber", "squared"):
                spec = po.spec.sgd(
                    "s",
                    targets=["y"],
                    features=["x"],
                    half_life=1e9,
                    min_weight=5.0,
                    loss=loss,
                    clip_gradient=float("inf"),
                )
                out = po.ModelBank([spec]).fit_predict(df)["s"].struct.field("pred_y")
                preds[c, loss] = out.to_numpy() / c
            differ = np.sum(preds[c, "huber"][5:] != preds[c, "squared"][5:])
            assert differ > 2000, f"scale {c}: the cut bound on {differ} rows"
        for c in (2.0**-10, 2.0**10):
            np.testing.assert_array_equal(preds[c, "huber"], preds[1.0, "huber"])


class TestLogisticLabels:
    """docs/PLAN.md task 195 (S4; review round 4, CC9): a label outside {0, 1}
    under the logistic loss takes `ftrl`'s rule. It is clamped into [0, 1] by
    default, and `strict_binary=True` refuses the chunk naming the row."""

    @staticmethod
    def _frame(hi):
        rng = np.random.default_rng(4)
        x = rng.normal(size=600)
        return pl.DataFrame({"x": x, "y": np.where(x + 0.5 * rng.normal(size=600) > 0, hi, 0.0)})

    def test_a_label_above_one_is_clamped_to_one(self):
        spec = po.spec.sgd(
            "m", targets=["y"], features=["x"], half_life=1e9, min_weight=5.0, loss="logistic"
        )
        five = po.ModelBank([spec]).fit_predict(self._frame(5.0))
        one = po.ModelBank([spec]).fit_predict(self._frame(1.0))
        assert five["m"].struct.field("pred_y").equals(one["m"].struct.field("pred_y"))

    def test_strict_binary_refuses_the_chunk_naming_the_row(self):
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x"],
            half_life=1e9,
            loss="logistic",
            strict_binary=True,
        )
        bank = po.ModelBank([spec])
        with pytest.raises(ValueError, match=r"strict_binary.*neither 0 nor 1"):
            bank.fit_predict(self._frame(5.0))
        po.ModelBank([spec]).fit_predict(self._frame(1.0))

    def test_strict_binary_is_for_the_logistic_loss(self):
        with pytest.raises(ValueError, match="strict_binary"):
            po.spec.sgd("m", targets=["y"], features=["x"], half_life=1e9, strict_binary=True)


class TestATargetBelowZero:
    """docs/PLAN.md task 209 (c): every loss on a target at a level below
    zero, where the only negative-target test compared ``s`` to its own
    residuals. Each step's derivative is odd in the residual -- ``p - y``,
    its clamp at ``±delta * s``, its sign outside a tube of ``eps * s_y``,
    and the check's ``1{y < p} - tau``, which mirrors to ``1 - tau`` -- and
    ``s`` and ``s_y`` are spreads, so the fit of ``-y`` is the fit of ``y``
    mirrored, the quantile's at ``1 - tau``: at -1,000 as at 1,000. The
    logistic loss's target is a label, and a Poisson fit refuses a negative
    count (`TestNegativeCounts`); scikit-learn holds the squared and
    epsilon-insensitive losses row for row at -1,000
    (`test_second_opinion.TestSgdIsScikitLearnsSgd`)."""

    @staticmethod
    def _frame(level, n=6000, seed=8):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        y = level + 1.5 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * rng.normal(size=n)
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y})

    @staticmethod
    def _pred(df, **kw):
        spec = po.spec.sgd(
            "m", targets=["y0"], features=["x0", "x1"], half_life=500.0, min_weight=10.0, **kw
        )
        return po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y0").to_numpy()

    @pytest.mark.parametrize(
        ("kw", "mirrored"),
        [
            ({"loss": "squared"}, {"loss": "squared"}),
            ({"loss": "huber"}, {"loss": "huber"}),
            ({"loss": "epsilon_insensitive"}, {"loss": "epsilon_insensitive"}),
            ({"loss": "quantile", "quantile": 0.25}, {"loss": "quantile", "quantile": 0.75}),
        ],
        ids=["squared", "huber", "epsilon_insensitive", "quantile"],
    )
    def test_a_target_below_zero_is_the_mirror_of_one_above(self, kw, mirrored):
        up = self._frame(1000.0)
        down = up.with_columns(-pl.col("y0"))
        assert (down["y0"] < 0).all(), "the case: every target below zero"
        above, below = self._pred(up, **kw), self._pred(down, **mirrored)
        assert np.isfinite(below[20:]).all()
        np.testing.assert_array_equal(below, -above)

    @pytest.mark.parametrize("loss", ["squared", "huber"])
    def test_a_target_at_minus_a_thousand_fits_as_one_at_zero(self, loss):
        """The losses whose step is the residual's size travel to the
        level: from row 3,000 their error is the fit at 0's to 1% (measured,
        0.3009 at both under the squared loss, 0.3009 against 0.3006 under
        Huber's). The sign-valued ones step by the rate and are a level's
        worth of rows away (the README's `sgd` section): 955 off at row
        3,000 here."""
        at_zero, below = self._frame(0.0), self._frame(-1000.0)
        p0, p = self._pred(at_zero, loss=loss), self._pred(below, loss=loss)

        def rmse(p, df):
            return float(np.sqrt(np.mean((p[3000:] - df["y0"].to_numpy()[3000:]) ** 2)))

        assert rmse(p0, at_zero) < 0.4, "the case: the fit at 0 fits"
        assert rmse(p, below) <= 1.01 * rmse(p0, at_zero), (rmse(p, below), rmse(p0, at_zero))


class TestNegativeCounts:
    """docs/PLAN.md task 209 (e), the user's decision of 2026-10-08: a
    Poisson fit takes counts, and a negative target refuses the chunk,
    naming the row, before any stream is touched, as scikit-learn's
    `PoissonRegressor` refuses one and as `strict_binary` refuses a
    logistic label. Taken as it stood, `p - y` with `y < 0` drove the
    prediction down to the link's clamp, `e ** -30`, and held it there."""

    @staticmethod
    def _frame(y):
        n = len(y)
        rng = np.random.default_rng(5)
        return pl.DataFrame({"x": rng.normal(size=n), "y": np.asarray(y, dtype=float)})

    @staticmethod
    def _spec():
        return po.spec.sgd(
            "m", targets=["y"], features=["x"], half_life=1e9, min_weight=2.0, loss="poisson"
        )

    def test_a_negative_count_refuses_the_chunk_naming_the_row(self):
        counts = [1.0, 0.0, 3.0, 2.0, -1.0, 4.0]
        bank = po.ModelBank([self._spec()])
        with pytest.raises(ValueError, match=r'"y" has -1 at row 4.*poisson.*never negative'):
            bank.fit_predict(self._frame(counts))
        # Before any stream is touched: the bank goes on as a new one does.
        good = self._frame([2.0, 0.0, 1.0, 5.0])
        assert bank.fit_predict(good).equals(po.ModelBank([self._spec()]).fit_predict(good))

    def test_the_row_is_counted_across_one_inputs_chunks(self):
        chunks = [self._frame([1.0, 2.0, 3.0]), self._frame([0.0, -0.5, 2.0])]
        with pytest.raises(ValueError, match=r"has -0\.5 at row 4\b"):
            list(po.ModelBank([self._spec()]).fit_predict_batches(iter(chunks)))

    def test_negative_zero_is_the_count_zero_and_a_null_is_scored(self):
        """`-0.0` is not below zero: it fits as `0.0` does. A null target is
        scored and not learned, as everywhere."""
        y = [1.0, 0.0, 2.0, -0.0, None, 3.0, 1.0]
        df = self._frame([0.0] * len(y)).with_columns(y=pl.Series(y, dtype=pl.Float64))
        out = po.ModelBank([self._spec()]).fit_predict(df)["m"].struct.field("pred_y")
        zero = df.with_columns(y=pl.col("y").abs())
        assert out.equals(
            po.ModelBank([self._spec()]).fit_predict(zero)["m"].struct.field("pred_y")
        )
        assert out[4] is not None, "the null target's row is scored"

    def test_another_loss_takes_a_negative_target(self):
        spec = po.spec.sgd("m", targets=["y"], features=["x"], half_life=1e9, min_weight=2.0)
        po.ModelBank([spec]).fit_predict(self._frame([1.0, -2.0, -3.0, 4.0]))


def test_a_poisson_fits_hit_rate_is_null():
    """docs/PLAN.md task 195 (S5; review round 4, CC5): a rate is positive
    and a count is not negative, so the sign test about zero agreed on every
    row and `hit_rate` read 1.0 whatever the fit. There is no sign to hit, so
    it is null, as `po.eval` nulls a metric that is not defined; `ic` and
    `r2` are still read."""
    rng = np.random.default_rng(3)
    n = 2000
    df = pl.DataFrame({"x": rng.normal(size=n), "y": rng.poisson(2.0, size=n).astype(float)})
    spec = po.spec.sgd(
        "s",
        targets=["y"],
        features=["x"],
        half_life=200.0,
        min_weight=5.0,
        loss="poisson",
        emit_metrics=True,
    )
    out = po.ModelBank([spec]).fit_predict(df)["s"].struct.unnest()
    assert out["hit_rate_y"].null_count() == n
    assert out["r2_y"].drop_nulls().len() > n - 20
