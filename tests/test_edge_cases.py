"""Edge-case matrix (docs/TESTING.md section C).

Each test pins behavior that was previously unspecified, undocumented, or --
for the first two classes -- outright wrong.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

INF = float("inf")


#: Models with no solve schedule (they update every row by construction).
_NO_SOLVE_SCHEDULE = {"rls", "kalman", "ftrl", "sgd", "pa", "holt"}

#: Every regression model, with the least it needs to be constructible.
#: `tests/test_model_registry.py` holds this list complete.
MODELS = [
    ("ewridge", {}),
    ("rls", {}),
    ("kalman", {"coef_halflife": 100.0}),
    ("lasso", {"lasso_path": [0.0]}),
    ("huber", {}),
    ("quantile", {"quantile": 0.5}),
    ("ftrl", {}),
    ("sgd", {"learning_rate": 0.01}),
    ("pa", {}),
    ("holt", {"features": []}),
]


def _spec(model="ewridge", **kw):
    d = dict(targets=["y0"], features=["x0"], halflife=1e9, min_periods=1.0)
    if model not in _NO_SOLVE_SCHEDULE:
        # A huge halflife means solve_every (halflife/50) would solve once ever.
        d["max_rows_between_solves"] = 1
    d.update(kw)
    return getattr(po.spec, model)("m", **d)


def _run(df, spec=None):
    return po.ModelBank([spec or _spec()]).fit_predict(df)


def _f(out, field, col="m"):
    return out[col].struct.field(field).to_list()


class TestWeights:
    """T-E1. A negative weight used to corrupt state silently: EwCov no-ops
    when `lam*W + w <= 0` while the per-target cross moments update anyway, so
    n_eff claimed the row never happened while r_j was polluted -- in practice
    every later prediction went null."""

    def test_negative_weight_is_rejected_with_the_row_number(self):
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0], "w": [1.0, -5.0, 1.0]})
        with pytest.raises(Exception, match=r"negative value \(-5\) at row 1"):
            _run(df, _spec(weight="w"))

    def test_negative_weight_is_rejected_for_every_model(self):
        df = pl.DataFrame({"x0": [1.0, 2.0], "y0": [1.0, 2.0], "w": [1.0, -1.0]})
        for model, extra in [
            ("ewridge", {}),
            ("rls", {}),
            ("kalman", {"coef_halflife": 100.0}),
            ("lasso", {"lasso_path": [0.0]}),
            ("huber", {}),
            ("quantile", {"quantile": 0.5}),
            ("ftrl", {}),
        ]:
            with pytest.raises(Exception, match="negative"):
                _run(df, _spec(model, weight="w", **extra))

    def test_zero_weight_is_a_pure_decay_row(self):
        # w = 0 is legal and means "advance the clock, learn nothing".
        df = pl.DataFrame({"x0": [1.0, 99.0, 2.0], "y0": [1.0, 99.0, 2.0], "w": [1.0, 0.0, 1.0]})
        out = _run(df, _spec(weight="w"))
        skipped = _run(
            pl.DataFrame({"x0": [1.0, 99.0, 2.0], "y0": [1.0, None, 2.0], "w": [1.0, 0.0, 1.0]}),
            _spec(weight="w"),
        )
        # The zero-weight row contributes nothing, so the final prediction is
        # the same as if its target had been null.
        assert _f(out, "pred_y0")[2] == _f(skipped, "pred_y0")[2]

    @pytest.mark.parametrize(("model", "extra"), MODELS, ids=[m for m, _ in MODELS])
    def test_leading_zero_weight_rows_do_not_disable_the_model(self, model, extra):
        """A zero-weight row *before any weighted row* used to poison `ewridge`
        and `lasso` permanently.

        Their per-target mean-form update computed `a = lam*wj / (lam*wj + w)`,
        which is 0/0 when nothing has ever carried weight. The NaN never washed
        out -- `wj` stayed NaN, `NaN > 0.0` is false, and the model silently
        stopped predicting for the rest of the stream. Found by a Rust unit
        test for the analogous guard in `blend_toward_long_run`; every other
        model already guarded it.
        """
        n = 20
        base = dict(
            t=np.arange(float(n)),
            x0=np.linspace(1.0, 2.0, n),
            y0=np.linspace(2.0, 4.0, n),
        )
        spec = _spec(
            model, weight="w", clock="t", max_dclock=5.0, halflife=50.0, min_periods=2.0, **extra
        )
        lead = _run(pl.DataFrame({**base, "w": [0.0] * 3 + [1.0] * (n - 3)}), spec)
        none = _run(pl.DataFrame({**base, "w": [1.0] * n}), spec)

        field = next(f.name for f in lead.schema["m"].fields if f.name.startswith("pred_y0"))
        finite = lambda out: [  # noqa: E731
            v for v in _f(out, field) if v is not None and np.isfinite(v)
        ]
        good = finite(lead)
        assert good, f"{model}: three zero-weight rows disabled the model entirely"
        # Three rows of warmup are lost to the zero weights, and no more.
        assert len(good) == len(finite(none)) - 3, f"{model}: {len(good)} vs {len(finite(none))}"

    def test_null_weight_skips_the_row(self):
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0], "w": [1.0, None, 1.0]})
        out = _run(df, _spec(weight="w"))
        assert _f(out, "n_eff")[1] is None


class TestGroupKeys:
    """T-E2. Null group keys used to be stringified to "<null>", colliding with
    a group literally named "<null>" -- they shared one state."""

    def test_null_group_is_distinct_from_the_literal_string(self):
        df = pl.DataFrame(
            {
                "g": [None, "<null>", None, "<null>"],
                "x0": [1.0, 2.0, 3.0, 4.0],
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = _run(df, _spec(group="g"))
        # Two independent streams: each sees its first row at n_eff 0.
        assert _f(out, "n_eff") == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_null_group_rows_form_one_stream(self):
        df = pl.DataFrame(
            {"g": [None, "a", None, "a"], "x0": [1.0, 2.0, 3.0, 4.0], "y0": [2.0, 4.0, 6.0, 8.0]}
        )
        out = _run(df, _spec(group="g"))
        assert _f(out, "n_eff") == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_group_keys_survive_save_load(self, tmp_path):
        df = pl.DataFrame({"g": [None, "<null>"], "x0": [1.0, 2.0], "y0": [1.0, 2.0]})
        spec = _spec(group="g")
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        p = tmp_path / "b.state"
        bank.save(p)
        reloaded = po.ModelBank.load(p, specs=[spec])
        a = bank.fit_predict(df)
        b = reloaded.fit_predict(df)
        assert _f(a, "n_eff") == pytest.approx(_f(b, "n_eff"))

    def test_integer_group_column(self):
        df = pl.DataFrame(
            {"g": [1, 2, 1, 2], "x0": [1.0, 2.0, 3.0, 4.0], "y0": [1.0, 2.0, 3.0, 4.0]}
        )
        out = _run(df, _spec(group="g"))
        assert _f(out, "n_eff") == pytest.approx([0.0, 0.0, 1.0, 1.0])


class TestNonFinite:
    """T-E3. +/-inf in each input position."""

    @pytest.mark.parametrize("bad", [INF, -INF, np.nan])
    def test_non_finite_feature_skips_the_row(self, bad):
        df = pl.DataFrame({"x0": [1.0, bad, 2.0], "y0": [1.0, 5.0, 2.0]})
        out = _run(df)
        assert _f(out, "n_eff")[1] is None
        assert _f(out, "pred_y0")[1] is None
        # the clock still advanced: row 2 sees exactly one observation
        assert _f(out, "n_eff")[2] == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [INF, -INF, np.nan])
    def test_non_finite_target_is_treated_as_null(self, bad):
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "y0": [1.0, bad, 3.0]})
        out = _run(df)
        # predict-only: the row is counted, but contributes no target information
        assert _f(out, "n_eff")[2] == pytest.approx(2.0)
        assert _f(out, "resid_y0")[1] is None

    @pytest.mark.parametrize("bad", [INF, -INF, np.nan])
    def test_non_finite_weight_skips_the_row(self, bad):
        # -inf is negative, but it is handled as "no information" like every
        # other non-finite input; only a finite negative weight is an error.
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0], "w": [1.0, bad, 1.0]})
        out = _run(df, _spec(weight="w"))
        assert _f(out, "n_eff")[1] is None

    @pytest.mark.parametrize("bad", [INF, -INF, np.nan, None])
    def test_non_finite_clock_errors_loudly(self, bad):
        df = pl.DataFrame({"t": [1.0, bad, 3.0], "x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0]})
        with pytest.raises(Exception, match="clock"):
            _run(df, _spec(clock="t", max_dclock=10.0))

    def test_outputs_are_never_non_finite(self):
        # Whatever comes in, what comes out is finite or null -- never NaN/inf.
        rng = np.random.default_rng(0)
        n = 500
        x = rng.standard_normal(n)
        x[rng.random(n) < 0.05] = np.inf
        y = rng.standard_normal(n) * 1e6
        df = pl.DataFrame({"x0": x, "y0": y})
        out = _run(df, _spec(halflife=50.0))
        for field in ("pred_y0", "resid_y0", "n_eff"):
            vals = np.array([v for v in _f(out, field) if v is not None], dtype=float)
            assert np.isfinite(vals).all(), field


class TestClockOrdering:
    """T-E4. A clock that runs backwards across a chunk boundary."""

    def _frames(self):
        # Rows 3 and 4 are out of order relative to rows 0-2.
        a = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0]})
        b = pl.DataFrame({"t": [0.5, 1.5], "x0": [4.0, 5.0], "y0": [4.0, 5.0]})
        return a, b

    def test_backwards_clock_across_chunks_uses_on_clock_reset(self):
        # Documented behavior today: a backwards delta is routed through
        # on_clock_reset, whether it comes from real data or a mis-sorted chunk.
        a, b = self._frames()
        spec = _spec(clock="t", max_dclock=4.0, halflife=1.0, on_clock_reset="max")
        bank = po.ModelBank([spec])
        out_a = bank.fit_predict(a)
        out_b = bank.fit_predict(b)
        # n_eff is reported before the row's update, so chunk b's first row
        # still shows the count carried over from chunk a...
        w_after_a = 0.5 * _f(out_a, "n_eff")[2] + 1.0  # = 1.75
        assert _f(out_b, "n_eff")[0] == pytest.approx(w_after_a, rel=1e-12)
        # ...and the capped delta shows up in the next row: the backwards jump
        # (0.5 - 2.0) was clamped to max_dclock = 4, giving decay 0.5**4, not a
        # reset (which would give 1.0) and not a zero delta (which would give
        # w_after_a + 1).
        assert _f(out_b, "n_eff")[1] == pytest.approx(w_after_a * 0.5**4 + 1.0, rel=1e-12)

    def test_reset_state_variant_restarts_the_stream(self):
        a, b = self._frames()
        spec = _spec(clock="t", max_dclock=4.0, on_clock_reset="reset_state")
        bank = po.ModelBank([spec])
        bank.fit_predict(a)
        out_b = bank.fit_predict(b)
        assert _f(out_b, "n_eff")[0] == 0.0

    def test_chunking_a_correctly_ordered_stream_is_still_invariant(self):
        # The guard rail for the above: ordering matters, chunk boundaries do not.
        df = pl.DataFrame(
            {"t": np.arange(200.0), "x0": np.arange(200.0) % 7, "y0": np.arange(200.0) % 5}
        )
        spec = _spec(clock="t", max_dclock=10.0, halflife=20.0)
        one = _run(df, spec).unnest("m")
        bank = po.ModelBank([spec])
        many = pl.concat([bank.fit_predict(df.slice(i, 17)) for i in range(0, 200, 17)]).unnest("m")
        # coef is emitted on each chunk's last row, so it is chunk-dependent by
        # design (docs/PLAN.md section 3); everything else must match exactly.
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)


class TestDegenerateSolves:
    """T-E5: the *plain* (non-standardized) solve path under degenerate inputs.

    docs/PLAN.md section 7: a factorization never returns NaN silently -- it is
    retried with escalating diagonal jitter, and total failure keeps the
    previous coefficients. Both are counted in `solve_failures`.
    """

    def _run_counting(self, df, **kw):
        spec = _spec(**kw)
        bank = po.ModelBank([spec])
        out = bank.fit_predict(df)
        return out, bank.solve_failures()["m"][""]

    def test_exactly_collinear_features_jitter_but_stay_finite(self):
        n = 200
        x = np.arange(float(n))
        df = pl.DataFrame({"x0": x, "x1": x, "y0": x * 2.0})
        out, failures = self._run_counting(df, features=["x0", "x1"], ridge=0.0, min_periods=2.0)
        assert failures > 0, "a singular system should have needed jitter"
        preds = np.array([v for v in _f(out, "pred_y0") if v is not None], dtype=float)
        assert np.isfinite(preds).all()

    def test_constant_feature_in_the_plain_path(self):
        n = 150
        rng = np.random.default_rng(3)
        a = rng.standard_normal(n)
        df = pl.DataFrame({"x0": a, "x1": np.full(n, 5.0), "y0": 2.0 * a})
        out, _ = self._run_counting(df, features=["x0", "x1"], ridge=1e-8, min_periods=3.0)
        preds = np.array([v for v in _f(out, "pred_y0") if v is not None], dtype=float)
        assert np.isfinite(preds).all()
        # the informative feature is still recovered
        y = df["y0"].to_numpy()[-50:]
        assert np.corrcoef(preds[-50:], y)[0, 1] > 0.99

    def test_ridge_prevents_the_need_for_jitter(self):
        n = 200
        x = np.arange(float(n))
        df = pl.DataFrame({"x0": x, "x1": x, "y0": x * 2.0})
        _, failures = self._run_counting(df, features=["x0", "x1"], ridge=1.0, min_periods=2.0)
        assert failures == 0, "a real ridge should make the system solvable"

    def test_non_solving_models_report_zero(self):
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0]})
        for model, extra in [("rls", {}), ("kalman", {"coef_halflife": 10.0}), ("ftrl", {})]:
            bank = po.ModelBank([_spec(model, **extra)])
            bank.fit_predict(df)
            assert bank.solve_failures()["m"][""] == 0


class TestDegenerateClocks:
    """T-E6: duplicate clock values, a zero cap, and a halflife far below the
    typical delta."""

    def test_duplicate_clock_values_are_zero_deltas(self):
        df = pl.DataFrame(
            {"t": [0.0, 0.0, 0.0, 0.0], "x0": [1.0, 2.0, 3.0, 4.0], "y0": [1.0, 2.0, 3.0, 4.0]}
        )
        out = _run(df, _spec(clock="t", max_dclock=10.0, halflife=5.0))
        # No decay at all: n_eff is just the row count.
        assert _f(out, "n_eff") == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_max_dclock_zero_disables_decay(self):
        df = pl.DataFrame({"t": [0.0, 100.0, 500.0], "x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0]})
        out = _run(df, _spec(clock="t", max_dclock=0.0, halflife=1.0))
        assert _f(out, "n_eff") == pytest.approx([0.0, 1.0, 2.0])

    def test_halflife_far_below_the_delta_forgets_almost_everything(self):
        df = pl.DataFrame(
            {"t": [0.0, 1000.0, 2000.0], "x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0]}
        )
        out = _run(df, _spec(clock="t", max_dclock=1e9, halflife=1e-3))
        # lambda is ~0, so each row is effectively the first one.
        neff = _f(out, "n_eff")
        assert neff[1] == pytest.approx(1.0, abs=1e-9)
        assert neff[2] == pytest.approx(1.0, abs=1e-9)

    def test_no_nan_leaks_from_extreme_decay(self):
        rng = np.random.default_rng(4)
        n = 300
        df = pl.DataFrame(
            {
                "t": np.cumsum(rng.exponential(1e6, n)),
                "x0": rng.standard_normal(n),
                "y0": rng.standard_normal(n),
            }
        )
        out = _run(df, _spec(clock="t", max_dclock=1e9, halflife=1e-6, min_periods=0.0))
        for field in ("pred_y0", "resid_y0", "n_eff"):
            vals = np.array([v for v in _f(out, field) if v is not None], dtype=float)
            assert np.isfinite(vals).all(), field


class TestMinimalShapes:
    """T-E7: empty chunks, single-row groups, and the smallest useful spec."""

    def test_empty_chunk_is_accepted(self):
        spec = _spec()
        bank = po.ModelBank([spec])
        empty = pl.DataFrame({"x0": [], "y0": []}, schema={"x0": pl.Float64, "y0": pl.Float64})
        out = bank.fit_predict(empty)
        assert out.height == 0
        assert "m" in out.columns

    def test_empty_chunk_between_real_chunks_changes_nothing(self):
        df = pl.DataFrame({"x0": np.arange(20.0), "y0": np.arange(20.0) * 2})
        spec = _spec()
        straight = po.ModelBank([spec]).fit_predict(df)

        bank = po.ModelBank([spec])
        empty = df.head(0)
        parts = [
            bank.fit_predict(df.slice(0, 10)),
            bank.fit_predict(empty),
            bank.fit_predict(df.slice(10, 10)),
        ]
        interrupted = pl.concat(parts)
        a = straight.unnest("m")
        b = interrupted.unnest("m")
        keep = [c for c in a.columns if not c.startswith("coef")]
        assert a.select(keep).equals(b.select(keep), null_equal=True)

    def test_single_row_groups(self):
        df = pl.DataFrame({"g": ["a", "b", "c"], "x0": [1.0, 2.0, 3.0], "y0": [1.0, 2.0, 3.0]})
        out = _run(df, _spec(group="g", min_periods=0.0))
        assert _f(out, "n_eff") == pytest.approx([0.0, 0.0, 0.0])

    def test_group_appearing_in_only_one_chunk(self):
        df = pl.DataFrame(
            {
                "g": ["a"] * 10 + ["b"] * 10,
                "x0": np.arange(20.0),
                "y0": np.arange(20.0) * 2,
            }
        )
        spec = _spec(group="g")
        bank = po.ModelBank([spec])
        first = bank.fit_predict(df.slice(0, 10))
        second = bank.fit_predict(df.slice(10, 10))
        # group b starts fresh in the second chunk
        assert _f(second, "n_eff")[0] == 0.0
        assert _f(first, "n_eff")[0] == 0.0

    def test_smallest_spec_one_feature_one_target(self):
        df = pl.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0], "y0": [2.0, 4.0, 6.0, 8.0]})
        out = _run(df, _spec(min_periods=2.0, add_intercept=False))
        assert _f(out, "pred_y0")[-1] == pytest.approx(8.0, rel=1e-6)


class TestColumnTypes:
    """T-E8: group and session columns that are not plain strings."""

    def test_categorical_group_column(self):
        df = pl.DataFrame(
            {
                "g": pl.Series(["a", "b", "a", "b"], dtype=pl.Categorical),
                "x0": [1.0, 2.0, 3.0, 4.0],
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = _run(df, _spec(group="g", min_periods=0.0))
        assert _f(out, "n_eff") == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_integer_session_column(self):
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0],
                "session": [1, 1, 2, 2],
                "x0": [1.0, 2.0, 3.0, 4.0],
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = _run(
            df,
            _spec(
                clock="t",
                max_dclock=10.0,
                session="session",
                session_gap="reset",
                min_periods=0.0,
            ),
        )
        assert _f(out, "n_eff")[2] == 0.0, "the session change should have reset"

    def test_a_session_named_like_the_null_sentinel_is_not_null(self):
        """The T-E2 bug one layer down, fixed: null sessions were hashed as the
        sentinel string "\\0<null>", so a session literally named that shared a
        session with null -- moving between them was invisibly *not* a change.
        Now null is distinct from every string, and the sentinel-named session
        behaves like any other."""
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0],
                "session": ["\0<null>", None, "\0<null>", None],
                "x0": [1.0, 2.0, 3.0, 4.0],
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = _run(
            df,
            _spec(
                clock="t",
                max_dclock=10.0,
                session="session",
                session_gap="reset",
                min_periods=0.0,
            ),
        )
        # Every row changes session, so n_eff must reset at every boundary.
        assert _f(out, "n_eff") == [0.0, 0.0, 0.0, 0.0], _f(out, "n_eff")

    def test_null_session_value_is_its_own_session(self):
        # Pins current behavior: a null session is its own session, distinct
        # from every string, so null -> "a" and "a" -> null both count as
        # session changes.
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0],
                "session": ["a", None, None, "a"],
                "x0": [1.0, 2.0, 3.0, 4.0],
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = _run(
            df,
            _spec(
                clock="t",
                max_dclock=10.0,
                session="session",
                session_gap="reset",
                min_periods=0.0,
            ),
        )
        neff = _f(out, "n_eff")
        assert neff[1] == 0.0, "a -> null is a session change"
        assert neff[2] == 1.0, "null -> null is not"
        assert neff[3] == 0.0, "null -> a is a session change"


class TestNumericalScale:
    """T-E9: large-offset behaviour of the accumulator.

    `EwCov` keeps **centered** co-moments (a weighted Welford update), so the
    variance no longer comes from the cancellation-prone `E[x^2] - m^2`. These
    tests pin the resulting operating range. The numbers in the comments are the
    errors the previous raw-moment form produced, kept so a regression is
    obvious.
    """

    @pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e8, 1e10])
    def test_standardized_solve_survives_large_offsets(self, offset):
        rng = np.random.default_rng(5)
        n = 2000
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        df = pl.DataFrame({"x0": a + offset, "x1": b + offset, "y0": 2.0 * a - 1.0 * b})
        out = _run(
            df,
            _spec(
                features=["x0", "x1"],
                standardize=True,
                ridge=1e-10,
                halflife=float("inf"),
                min_periods=10.0,
            ),
        )
        coef = np.array(_f(out, "coef")[-1], dtype=float)
        # The raw-moment form needed 0.2 at 1e6 and dropped the features
        # entirely beyond ~1e7; centered updates hold to ~1e-6 at 1e10.
        tol = 1e-6
        assert coef[1] == pytest.approx(2.0, abs=tol), f"offset {offset}: {coef}"
        assert coef[2] == pytest.approx(-1.0, abs=tol), f"offset {offset}: {coef}"

    def test_operating_range(self):
        """Records the accuracy at each scale. Previous (raw-moment) errors are
        in the comments; the assertions are the current guarantees."""
        rng = np.random.default_rng(6)
        n = 5000
        a = rng.standard_normal(n)
        err = {}
        for offset in (0.0, 1e4, 1e6, 1e8, 1e10):
            df = pl.DataFrame({"x0": a + offset, "y0": 3.0 * a})
            out = _run(
                df,
                _spec(
                    standardize=True,
                    ridge=0.0,
                    halflife=float("inf"),
                    min_periods=10.0,
                ),
            )
            coef = np.array(_f(out, "coef")[-1], dtype=float)
            err[offset] = abs(coef[1] - 3.0)

        assert err[0.0] < 1e-12  # was 1.8e-14
        assert err[1e4] < 1e-10  # was 3.7e-06
        assert err[1e6] < 1e-8  # was 2.0e-03
        assert err[1e8] < 1e-6  # was: feature dropped entirely
        assert err[1e10] < 1e-4  # was: feature dropped entirely
        # and accuracy still degrades monotonically with the offset, so a
        # future regression to a cancellation-prone form shows up here
        assert err[0.0] <= err[1e6] <= err[1e10]

    def test_a_genuinely_constant_feature_is_still_dropped(self):
        # Centered updates give a constant feature *exactly* zero variance, so
        # it is dropped without needing a fuzzy threshold.
        n = 500
        rng = np.random.default_rng(7)
        a = rng.standard_normal(n)
        df = pl.DataFrame({"x0": a, "x1": np.full(n, 7.0), "y0": 2.0 * a})
        out = _run(
            df,
            _spec(
                features=["x0", "x1"],
                standardize=True,
                ridge=1e-10,
                halflife=float("inf"),
                min_periods=10.0,
            ),
        )
        coef = np.array(_f(out, "coef")[-1], dtype=float)
        assert coef[2] == 0.0, "a constant feature must be dropped"
        assert coef[1] == pytest.approx(2.0, abs=1e-6)

    def test_a_constant_feature_on_a_large_offset_is_also_dropped(self):
        n = 500
        rng = np.random.default_rng(8)
        a = rng.standard_normal(n)
        df = pl.DataFrame({"x0": a, "x1": np.full(n, 1e9), "y0": 2.0 * a})
        out = _run(
            df,
            _spec(
                features=["x0", "x1"],
                standardize=True,
                ridge=1e-10,
                halflife=float("inf"),
                min_periods=10.0,
            ),
        )
        coef = np.array(_f(out, "coef")[-1], dtype=float)
        assert coef[2] == 0.0
        assert coef[1] == pytest.approx(2.0, abs=1e-6)


class TestClockColumnTypes:
    """T-E10: which dtypes are allowed as a clock column.

    A temporal column is **rejected**, not cast. Casting one to f64 exposes its
    internal representation, so the same 60 seconds becomes 60_000 /
    60_000_000 / 60_000_000_000 clock units depending only on whether the
    column is `Datetime(ms/us/ns)`, and a `Date` becomes 1 unit per day.
    `halflife`, `max_dclock` and `session_gap` all live in those units, so
    `halflife=600` on a microsecond column would silently mean 600
    microseconds -- every row decays to nothing and the output is
    plausible-looking garbage with no error.
    """

    @pytest.mark.parametrize("unit", ["ms", "us", "ns"])
    def test_datetime_clock_is_rejected(self, unit):
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 3), interval="1m", eager=True
        ).cast(pl.Datetime(time_unit=unit))
        df = pl.DataFrame(
            {"t": ts, "x0": np.arange(float(len(ts))), "y0": np.arange(float(len(ts)))}
        )
        with pytest.raises(Exception, match="temporal clock"):
            _run(df, _spec(clock="t", max_dclock=1e12, halflife=600.0))

    def test_date_and_duration_clocks_are_rejected(self):
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 4), interval="1d", eager=True
        )
        n = len(ts)
        for col in (ts.dt.date(), ts - ts[0]):
            df = pl.DataFrame({"t": col, "x0": np.arange(float(n)), "y0": np.arange(float(n))})
            with pytest.raises(Exception, match="temporal clock"):
                _run(df, _spec(clock="t", max_dclock=1e12, halflife=600.0))

    def test_the_error_names_the_column_dtype_and_the_fix(self):
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 2), interval="1m", eager=True
        )
        df = pl.DataFrame(
            {"t": ts, "x0": np.arange(float(len(ts))), "y0": np.arange(float(len(ts)))}
        )
        with pytest.raises(Exception) as exc:
            _run(df, _spec(clock="t", max_dclock=1e12, halflife=600.0))
        msg = str(exc.value)
        assert '"t"' in msg, "the offending column should be named"
        assert "datetime" in msg.lower(), "the dtype should be named"
        assert "dt.epoch" in msg, "the error should show the fix"

    def test_the_documented_fix_works(self):
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 1, 0, 3), interval="1m", eager=True
        )
        n = len(ts)
        df = pl.DataFrame(
            {"t": ts, "x0": np.arange(float(n)), "y0": np.arange(float(n))}
        ).with_columns(t_s=pl.col("t").dt.epoch("s").cast(pl.Float64))
        # halflife 60 now genuinely means 60 seconds, i.e. one bar.
        out = _run(df, _spec(clock="t_s", max_dclock=1e6, halflife=60.0))
        assert _f(out, "n_eff")[2] == pytest.approx(0.5 * 1.0 + 1.0, rel=1e-12)

    def test_integer_clock_column(self):
        df = pl.DataFrame(
            {"t": [0, 10, 20, 30], "x0": [1.0, 2.0, 3.0, 4.0], "y0": [1.0, 2.0, 3.0, 4.0]}
        )
        out = _run(df, _spec(clock="t", max_dclock=100.0, halflife=10.0, min_periods=0.0))
        assert _f(out, "n_eff")[2] == pytest.approx(0.5 + 1.0, rel=1e-12)

    def test_float_and_integer_clocks_agree(self):
        a = pl.DataFrame({"t": [0, 60, 120, 180], "x0": np.arange(4.0), "y0": np.arange(4.0)})
        b = a.with_columns(t=pl.col("t").cast(pl.Float64))
        spec = _spec(clock="t", max_dclock=1e6, halflife=600.0, min_periods=0.0)
        assert _run(a, spec).drop("t").equals(_run(b, spec).drop("t"), null_equal=True)


class TestPendingDeltaAcrossSaveLoad:
    """T-E12: a skipped row immediately before a save/load boundary.

    The clock folds a skipped row's delta into the next accepted row, so that
    pending amount must survive serialization.
    """

    def test_skipped_row_before_a_save_boundary(self, tmp_path):
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "x0": [1.0, 2.0, None, 4.0, 5.0, 6.0],  # row 2 skipped
                "y0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        spec = _spec(clock="t", max_dclock=10.0, halflife=2.0, min_periods=0.0)
        straight = po.ModelBank([spec]).fit_predict(df)

        # Split exactly after the skipped row, so `pending` is nonzero at save.
        bank = po.ModelBank([spec])
        first = bank.fit_predict(df.slice(0, 3))
        p = tmp_path / "pending.state"
        bank.save(p)
        resumed = po.ModelBank.load(p, specs=[spec]).fit_predict(df.slice(3, 3))
        joined = pl.concat([first, resumed]).unnest("m")

        a = straight.unnest("m")
        keep = [c for c in a.columns if not c.startswith("coef")]
        assert a.select(keep).equals(joined.select(keep), null_equal=True), (
            "the pending clock delta of a skipped row did not survive save/load"
        )

    def test_session_change_on_the_first_row_of_a_group(self):
        df = pl.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "t": [0.0, 1.0, 2.0, 3.0],
                "session": ["s1", "s1", "s2", "s2"],
                "x0": [1.0, 2.0, 3.0, 4.0],
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        # Group b's first row is also a session change; the first row of a
        # stream must win (delta 0, no gap applied).
        out = _run(
            df,
            _spec(
                group="g",
                clock="t",
                max_dclock=10.0,
                session="session",
                session_gap=100.0,
                halflife=1.0,
                min_periods=0.0,
            ),
        )
        assert _f(out, "n_eff")[2] == 0.0
        assert _f(out, "n_eff")[3] == pytest.approx(1.0, rel=1e-12)


class TestStrictClock:
    """E3: `on_clock_reset="error"` refuses a backwards clock instead of
    absorbing it.

    PLAN section 5 says the bank "asserts monotonicity ... and errors loudly";
    until this existed, a mis-sorted chunk was indistinguishable from a genuine
    backwards clock and was silently routed through the decay policy.
    """

    def _df(self):
        return pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 1.5, 3.0],
                "x0": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y0": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    def _spec_for(self, policy, **kw):
        return _spec(
            clock="t",
            max_dclock=10.0,
            halflife=5.0,
            on_clock_reset=policy,
            min_periods=0.0,
            **kw,
        )

    def test_error_policy_names_column_row_and_magnitude(self):
        with pytest.raises(Exception) as exc:
            _run(self._df(), self._spec_for("error"))
        msg = str(exc.value)
        assert '"t"' in msg and "row 3" in msg and "0.5" in msg
        assert "sort" in msg.lower(), "the error should say how to fix it"

    def test_other_policies_still_absorb_it(self):
        neff = {
            p: _f(_run(self._df(), self._spec_for(p)), "n_eff")
            for p in ("max", "zero", "reset_state")
        }
        # each policy gives a genuinely different answer at the offending row
        assert neff["reset_state"][3] == 0.0
        assert neff["zero"][4] > neff["max"][4]

    def test_error_policy_accepts_a_monotone_stream(self):
        good = pl.DataFrame(
            {"t": [0.0, 1.0, 2.0, 3.0], "x0": [1.0] * 4, "y0": [1.0, 2.0, 3.0, 4.0]}
        )
        out = _run(good, self._spec_for("error"))
        assert _f(out, "n_eff")[0] == 0.0

    def test_repeated_clock_values_are_not_an_error(self):
        # A duplicate timestamp is a zero delta, not a backwards one.
        dup = pl.DataFrame({"t": [0.0, 1.0, 1.0, 2.0], "x0": [1.0] * 4, "y0": [1.0, 2.0, 3.0, 4.0]})
        out = _run(dup, self._spec_for("error"))
        assert _f(out, "n_eff")[3] == pytest.approx(_f(out, "n_eff")[2] * 0.5 ** (0.0 / 5.0) + 1.0)

    def test_catches_a_mis_sorted_chunk_boundary(self):
        # The motivating case: chunks fed out of order. Under "max" this is
        # silently absorbed (T-E4); under "error" it is caught.
        a = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x0": [1.0] * 3, "y0": [1.0, 2.0, 3.0]})
        b = pl.DataFrame({"t": [0.5, 1.5], "x0": [1.0] * 2, "y0": [4.0, 5.0]})
        spec = self._spec_for("error")
        bank = po.ModelBank([spec])
        bank.fit_predict(a)
        with pytest.raises(Exception, match="goes backwards"):
            bank.fit_predict(b)

    def test_per_group_so_interleaved_groups_are_fine(self):
        # Group clocks are independent: interleaving two ascending streams must
        # not look backwards.
        df = pl.DataFrame(
            {
                "g": ["a", "b", "a", "b"],
                "t": [0.0, 100.0, 1.0, 101.0],
                "x0": [1.0] * 4,
                "y0": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = _run(df, self._spec_for("error", group="g"))
        assert _f(out, "n_eff")[2] == pytest.approx(1.0)
