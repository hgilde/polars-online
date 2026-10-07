"""T-A5: the common semantics (docs/PLAN.md section 3) are claimed to be
model-independent, so they are asserted for *every* model rather than only for
ew_ridge.

Anything a model is genuinely allowed to differ on is listed explicitly here
rather than skipped silently:

- `rls` is predict-only for **all** targets when **any** target is null,
  because `P` is shared across targets (documented deviation, PLAN section 4.2).
- `lasso` names its output slots by path point (`pred_y0__l0.1`), because the
  selected penalty is part of the output semantics.
- `ftrl` predicts a probability, so it is fed a 0/1 target.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

#: (name, extra spec kwargs). One entry per model the bank can build;
#: `test_model_registry` holds this list to exactly that.
MODELS = [
    ("ewridge", {"max_rows_between_solves": 1}),
    ("rls", {"ridge": 1.0}),
    ("kalman", {"coef_half_life": 100.0}),
    ("lasso", {"lasso_path": [0.0], "max_rows_between_solves": 1}),
    ("huber", {"max_rows_between_solves": 1}),
    ("quantile", {"quantile": 0.5, "max_rows_between_solves": 1}),
    ("ftrl", {}),
    ("sgd", {"learning_rate": 0.01}),
    ("pa", {}),
    # Holt is the one model with no features, so it opts out of the shared set.
    ("holt", {"features": []}),
]

#: A second configuration of a model above, swept through the same
#: invariants: `ewridge` with the Gram update blocked (E51). The block is an
#: implementation of the same model, so everything here has to hold for it
#: too. The block, 5, is chosen not to divide the solve cadence, 4 rows, so a
#: solve merges a partial block on every cycle.
VARIANTS = [
    ("ewridge", {"solve_every": 4.0, "max_rows_between_solves": 8, "gram_block_rows": 5}),
]
SWEEP = MODELS + VARIANTS
IDS = [m for m, _ in MODELS] + [f"{m}-blocked" for m, _ in VARIANTS]

#: Models that treat a null in *any* target as predict-only for all targets.
SHARED_STATE_MODELS = {"rls"}

#: What a refusal raises, wrapped or not by version, as `test_frame._REFUSAL`
#: says: a refusal is one of these, never a panic nor any other exception.
_REFUSAL = (pl.exceptions.ComputeError, ValueError)


def build(model, extra, **kw):
    opts = dict(targets=["y0"], features=["x0", "x1"], half_life=200.0, min_weight=2.0)
    opts.update(extra)
    opts.update(kw)
    return getattr(po.spec, model)("m", **opts)


def slot(out, prefix, model):
    """The first output field with this prefix (lasso suffixes by path point)."""
    fields = [f.name for f in out.schema["m"].fields if f.name.startswith(prefix)]
    assert fields, f"{model}: no field starting with {prefix!r}"
    return out["m"].struct.field(fields[0]).to_list()


def frame(n=40, seed=0, binary=False):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    y = (rng.random(n) < 0.5).astype(float) if binary else x0 * 2.0 - x1
    return pl.DataFrame({"x0": x0, "x1": x1, "y0": y, "t": np.arange(float(n)), "w": np.ones(n)})


def run(model, extra, df, **kw):
    return po.ModelBank([build(model, extra, **kw)]).fit_predict(df)


#: The sweep less ``holt``, which takes no features to null: generated
#: without it rather than skipped.
WITH_FEATURES = [(m, e) for m, e in SWEEP if m != "holt"]
WITH_FEATURES_IDS = [i for (m, _), i in zip(SWEEP, IDS, strict=True) if m != "holt"]


@pytest.mark.parametrize(("model", "extra"), WITH_FEATURES, ids=WITH_FEATURES_IDS)
def test_feature_null_skips_the_row_entirely(model, extra):
    df = frame(binary=model == "ftrl")
    x = df["x0"].to_list()
    x[10] = None
    df = df.with_columns(x0=pl.Series(x, dtype=pl.Float64))
    out = run(model, extra, df)
    assert slot(out, "weight_sum", model)[10] is None
    assert slot(out, "pred_", model)[10] is None
    # ...and the clock still advanced: the next row is not treated as first
    assert slot(out, "weight_sum", model)[11] is not None


#: A clock column, so a skipped row's gap folds into the next row's and a
#: deleted row is the same stream; and a weight column to null or zero.
CLOCKED = dict(clock="t", gap_cap=100.0, weight="w")


def _fields(out):
    return out.select("m").unnest("m")


@pytest.mark.parametrize(("model", "extra"), SWEEP, ids=IDS)
class TestNullPolicy:
    """What a null does to the model, not only to the row: asserted on the
    row alone, these held whatever the model learned from it, since the
    stream layer writes the row's fields (review 2026-10-05, TB13)."""

    def test_target_null_is_predict_only(self, model, extra):
        """A null target is scored and not learned. The row has its
        prediction and no residual. And the model learns nothing from it:
        every later row is what it is with the row at weight 0, the other
        way to advance the clock and learn nothing, field for field and
        ``coef`` included -- but ``weight_sum``, the shared weight, which
        counts every processed row (hard rule 8) and so is the zero-weight
        stream's plus the row's weight, decayed. `kalman` runs
        unstandardized here: its standardizer takes every processed row's
        features, a null target's included, as `kalman.rs` documents."""
        if model == "kalman":
            extra = {**extra, "standardize": False}
        df = frame(binary=model == "ftrl")
        y = df["y0"].to_list()
        y[20] = None
        out = run(model, extra, df.with_columns(y0=pl.Series(y, dtype=pl.Float64)), **CLOCKED)
        assert slot(out, "resid_", model)[20] is None, "a null target has no residual"
        # Every model still emits a prediction for the row -- it is the update
        # that is skipped, not the prediction.
        assert slot(out, "pred_", model)[20] is not None
        w = df["w"].to_list()
        w[20] = 0.0
        zero = run(model, extra, df.with_columns(w=pl.Series(w, dtype=pl.Float64)), **CLOCKED)
        a, b = _fields(out).slice(21), _fields(zero).slice(21)
        assert a.drop("weight_sum").equals(b.drop("weight_sum"), null_equal=True)
        lam = 0.5 ** (1.0 / 200.0)  # the rows are one clock unit apart
        more = a["weight_sum"].to_numpy() - b["weight_sum"].to_numpy()
        np.testing.assert_allclose(more, lam ** np.arange(len(more)), rtol=1e-12)

    def test_null_weight_skips_the_row(self, model, extra):
        """A null weight skips the row: every field of it is null, and every
        other row is what it is with the row deleted, ``coef`` included."""
        df = frame(binary=model == "ftrl")
        w = df["w"].to_list()
        w[15] = None
        out = _fields(run(model, extra, df.with_columns(w=pl.Series(w)), **CLOCKED))
        assert all(out[c][15] is None for c in out.columns), out.row(15, named=True)
        deleted = run(model, extra, pl.concat([df.slice(0, 15), df.slice(16)]), **CLOCKED)
        assert pl.concat([out.slice(0, 15), out.slice(16)]).equals(
            _fields(deleted), null_equal=True
        )

    def test_negative_weight_is_rejected(self, model, extra):
        df = frame(binary=model == "ftrl")
        w = df["w"].to_list()
        w[5] = -1.0
        df = df.with_columns(w=pl.Series(w, dtype=pl.Float64))
        with pytest.raises(_REFUSAL, match="negative"):
            run(model, extra, df, weight="w")


@pytest.mark.parametrize(("model", "extra"), SWEEP, ids=IDS)
class TestWarmup:
    def test_outputs_are_null_until_min_periods(self, model, extra):
        df = frame(binary=model == "ftrl")
        out = run(model, extra, df, min_weight=5.0)
        preds = slot(out, "pred_", model)
        neff = slot(out, "weight_sum", model)
        first = next((i for i, p in enumerate(preds) if p is not None), None)
        assert first is not None, f"{model}: never emitted a prediction"
        assert neff[first] >= 5.0, f"{model}: predicted at weight_sum {neff[first]} < 5"
        assert all(p is None for p in preds[:first])

    def test_a_target_counts_toward_min_periods_only_on_rows_that_carry_it(self, model, extra):
        """Hard rule 8 (docs/PLAN.md task 115 (d)): `weight_sum` is the weight of
        every row, and a target's `min_weight` is checked against the
        weight of the rows that carried it. Ten rows with a null target,
        then the target: under a half-life of 200 its own weight is 1.997
        before row 12 and 2.990 before row 13, so at `min_weight = 2.5`
        every model first predicts on row 13. `pa`, `sgd` and `ftrl` did
        from row 3, on coefficients no target had moved, and `rls` from
        row 11, after one. `ewridge`'s noise gate is off here: three rows
        for three coefficients leave no residual degrees of freedom, so it
        withholds row 13 on its own account (`above_max_error_inflation`)."""
        df = frame(binary=model == "ftrl")
        y = df["y0"].to_list()
        y[:10] = [None] * 10
        df = df.with_columns(y0=pl.Series(y, dtype=pl.Float64))
        gate = {"max_error_inflation": float("inf")} if model == "ewridge" else {}
        out = run(model, extra, df, min_weight=2.5, **gate)
        preds = slot(out, "pred_", model)
        neff = slot(out, "weight_sum", model)
        first = next((i for i, p in enumerate(preds) if p is not None), None)
        assert first == 13, f"{model}: first prediction on row {first}"
        assert neff[first] > 12.0, "weight_sum counts every row, the null-target ones too"

    def test_coef_is_null_or_complete_never_empty(self, model, extra):
        """A model that has not solved yet has nothing to report, and every
        other output spells that `null`. `coef` used to spell it as an empty
        list on the warmup rows, which made the documented way to read one
        coefficient -- `coef.list.get(position)` -- raise "index out of
        bounds" instead of returning null (IMPROVEMENTS U7)."""
        df = frame(binary=model == "ftrl")
        # The models that solve on a schedule report nothing until the first
        # solve, which is the state this is about; delay it a few rows. The
        # rest carry coefficients from row one and never had the problem.
        delay = (
            dict(solve_every=5.0, max_rows_between_solves=10_000)
            if model in ("ewridge", "lasso", "huber", "quantile")
            else {}
        )
        out = run(model, extra, df, min_weight=5.0, coef_every=1, **delay)
        fields = [f.name for f in out.schema["m"].fields if f.name.startswith("coef")]
        assert fields, f"{model}: no coef field"
        for name in fields:
            c = out["m"].struct.field(name)
            lengths = c.list.len().drop_nulls().to_list()
            assert lengths, f"{model}.{name}: every row is null"
            assert all(n > 0 for n in lengths), f"{model}.{name}: an empty coef list"
            # And the access pattern the docstrings show works on every row:
            # the call itself is the test, since it raised "index out of
            # bounds" on an empty list. What it gives is null exactly where
            # the row has no coefficients.
            first = c.list.get(0)
            assert first.is_null().to_list() == c.is_null().to_list(), f"{model}.{name}"
            if delay:
                assert c[0] is None, f"{model}.{name}: unsolved row is not null"

    def test_n_eff_is_reported_before_the_update(self, model, extra):
        df = frame(binary=model == "ftrl")
        out = run(model, extra, df, min_weight=0.0)
        assert slot(out, "weight_sum", model)[0] == 0.0


@pytest.mark.parametrize(("model", "extra"), SWEEP, ids=IDS)
class TestClockSemantics:
    def _clocked(self, model, extra, t, **kw):
        n = len(t)
        rng = np.random.default_rng(1)
        y = (rng.random(n) < 0.5).astype(float) if model == "ftrl" else np.arange(float(n))
        df = pl.DataFrame({"t": t, "x0": np.arange(float(n)), "x1": np.ones(n), "y0": y})
        return run(
            model,
            extra,
            df,
            clock="t",
            gap_cap=4.0,
            min_weight=0.0,
            **kw,
        )

    def test_gap_is_capped_at_max_dclock(self, model, extra):
        out = self._clocked(model, extra, [0.0, 1.0, 1e9, 1e9 + 1])
        neff = slot(out, "weight_sum", model)
        # after two rows W = 0.5**(1/200) + 1; the huge gap decays it by
        # 0.5**(4/200), not to nothing
        w2 = neff[2]
        assert neff[3] == pytest.approx(w2 * 0.5 ** (4 / 200) + 1.0, rel=1e-9)

    def test_a_backwards_clock_is_refused_by_default(self, model, extra):
        with pytest.raises(ValueError, match="goes backwards by 50 at row 2"):
            self._clocked(model, extra, [0.0, 100.0, 50.0, 51.0])

    def test_reset_state_restarts_the_stream(self, model, extra):
        # A step back of any size starts over here: none is a late row.
        out = self._clocked(model, extra, [0.0, 10.0, 5.0, 6.0], restart_after_step_back=0.0)
        assert slot(out, "weight_sum", model)[2] == 0.0

    def test_session_reset(self, model, extra):
        n = 4
        rng = np.random.default_rng(2)
        y = (rng.random(n) < 0.5).astype(float) if model == "ftrl" else np.arange(float(n))
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0],
                "session": ["a", "a", "b", "b"],
                "x0": np.arange(float(n)),
                "x1": np.ones(n),
                "y0": y,
            }
        )
        out = run(
            model,
            extra,
            df,
            clock="t",
            gap_cap=4.0,
            session="session",
            session_gap="reset",
            min_weight=0.0,
        )
        assert slot(out, "weight_sum", model)[2] == 0.0


@pytest.mark.parametrize(("model", "extra"), SWEEP, ids=IDS)
class TestUniversalInvariants:
    def test_chunk_invariance(self, model, extra):
        df = frame(n=120, seed=3, binary=model == "ftrl")
        spec = build(model, extra, group=None)
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 13)) for i in range(0, df.height, 13)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

    def test_save_load_mid_stream(self, model, extra, tmp_path):
        df = frame(n=120, seed=4, binary=model == "ftrl")
        spec = build(model, extra)
        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 60))
        p = tmp_path / f"{model}.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        second = df.slice(60, 60)
        ra = a.fit_predict(second).select("m").unnest("m")
        rb = b.fit_predict(second).select("m").unnest("m")
        assert ra.equals(rb, null_equal=True)

    def test_groups_are_independent(self, model, extra):
        df = frame(n=120, seed=5, binary=model == "ftrl").with_columns(g=pl.Series(["a", "b"] * 60))
        spec = build(model, extra, group="g")
        both = po.ModelBank([spec]).fit_predict(df)
        solo_df = df.filter(pl.col("g") == "a")
        solo = po.ModelBank([spec]).fit_predict(solo_df)
        a = both.filter(pl.col("g") == "a").select("m").unnest("m")
        b = solo.select("m").unnest("m")
        keep = [c for c in a.columns if not c.startswith("coef")]
        assert a.select(keep).equals(b.select(keep), null_equal=True)

    def test_outputs_are_never_non_finite(self, model, extra):
        rng = np.random.default_rng(7)
        n = 300
        x0 = rng.standard_normal(n) * 1e3
        x0[rng.random(n) < 0.05] = np.inf
        df = pl.DataFrame(
            {
                "x0": x0,
                "x1": rng.standard_normal(n),
                "y0": (rng.random(n) < 0.5).astype(float)
                if model == "ftrl"
                else rng.standard_normal(n) * 1e4,
                "t": np.arange(float(n)),
                "w": np.ones(n),
            }
        )
        out = run(model, extra, df)
        for f in out.schema["m"].fields:
            # `coef` and `support_coef` are lists; `withheld_reason` an enum.
            if f.name.startswith("coef") or not f.dtype.is_float():
                continue
            vals = np.array(
                [v for v in out["m"].struct.field(f.name).to_list() if v is not None],
                dtype=float,
            )
            assert np.isfinite(vals).all(), f"{model}: {f.name} produced a non-finite value"
