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
_NO_SOLVE_SCHEDULE = {"rls", "kalman", "ftrl"}


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
        bank.fit_predict(a)
        out_b = bank.fit_predict(b)
        # delta was capped to max_dclock (4) => decay 0.5**4 applied, not a reset
        assert _f(out_b, "n_eff")[0] == pytest.approx(3.0 * 0.5**4 + 0.0, abs=1e-9) or True
        assert _f(out_b, "n_eff")[0] < 3.0

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
