"""Task 221: the diagnostics that keep a memory of their own.

Each is one accumulator per slot with its own half-life, the model
instance's unless set, and ``inf`` the run-once form (docs/PLAN.md task
221). The oracles -- statsmodels on the same rows -- are in
``tests/test_second_opinion.py``; this file holds each accumulator to the
hard rules: a field never reads its own row (rule 2), one chunk or many give
the same numbers (rule 3), a weight of zero only advances the clock, as the
first row too (rule 9), the state goes on to the bit across a save, and a
knob without its switch is refused.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

CALIBRATION = ["calibration_slope_y", "calibration_intercept_y", "calibration_wald_y"]


def _frame(n: int = 900, seed: int = 221) -> pl.DataFrame:
    """Two features, a target, weights with zeros among them -- the first
    two rows included -- and a null target now and then."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 0.3 + x @ np.array([1.5, -0.7]) + rng.normal(size=n) * (1.0 + 0.5 * (x[:, 0] > 0))
    w = rng.uniform(0.5, 1.5, size=n)
    w[[0, 1]] = 0.0
    w[rng.random(n) < 0.05] = 0.0
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": w})
    return df.with_columns(
        pl.when(pl.int_range(pl.len()) % 31 == 17).then(None).otherwise(pl.col("y")).alias("y")
    )


def _spec(**kw):
    base = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=60.0,
        weight="w",
        min_weight=10.0,
        emit_calibration=True,
    )
    base.update(kw)
    return po.spec.ewridge("m", **base)


def _run(df: pl.DataFrame, spec, chunk: int | None = None) -> pl.DataFrame:
    bank = po.ModelBank([spec])
    if chunk is None:
        return bank.fit_predict(df).unnest("m")
    parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
    return pl.concat(parts).unnest("m")


@pytest.mark.parametrize("memory", [None, 150.0, float("inf")])
def test_calibration_is_chunk_invariant(memory):
    df = _frame()
    spec = _spec(calibration_half_life=memory) if memory is not None else _spec()
    whole = _run(df, spec).select(CALIBRATION)
    assert whole["calibration_wald_y"].drop_nulls().len() > 500, "the test reads values"
    for chunk in (7, 600):
        got = _run(df, spec, chunk).select(CALIBRATION)
        assert got.equals(whole, null_equal=True), f"{chunk}-row chunks"


def test_calibration_never_reads_its_own_row():
    """Rule 2: a row's outcome moves the fields of the rows after it, never
    its own."""
    df = _frame()
    at = 400
    moved = df.with_columns(
        pl.when(pl.int_range(pl.len()) == at)
        .then(pl.col("y") + 50.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    a, b = _run(df, _spec()), _run(moved, _spec())
    for f in CALIBRATION:
        assert a[f][: at + 1].equals(b[f][: at + 1], null_equal=True), f
        assert a[f][at + 1] != b[f][at + 1], f


def test_zero_weight_rows_only_advance_the_clock():
    """Rule 9: the stream opens on two rows of weight zero, which leave no
    trace -- the fields are those of the stream without them, read on a
    clock that the two rows advanced -- and no field is NaN."""
    df = _frame().with_columns(pl.int_range(pl.len()).cast(pl.Float64).alias("t"))
    spec = _spec(clock="t", gap_cap=10.0)
    out = _run(df, spec)
    for f in CALIBRATION:
        assert out[f].is_nan().sum() == 0, f
    # The same rows without the two heads: their clock steps no further,
    # since nothing was learned before them, so the fields agree.
    rest = _run(df.slice(2), spec)
    for f in CALIBRATION:
        np.testing.assert_allclose(
            out[f][2:].to_numpy(), rest[f].to_numpy(), rtol=1e-12, atol=0, equal_nan=True
        )


def test_calibration_goes_on_across_a_save():
    df = _frame()
    whole = _run(df, _spec()).select(CALIBRATION)
    bank = po.ModelBank([_spec()])
    first = bank.fit_predict(df.slice(0, 500))
    resumed = po.ModelBank.load_bytes(bank.save_bytes())
    rest = resumed.fit_predict(df.slice(500))
    got = pl.concat([first, rest]).unnest("m").select(CALIBRATION)
    assert got.equals(whole, null_equal=True)


def test_a_reset_restarts_the_calibration():
    """``drift_action = "reset"`` restarts the diagnostics with the model."""
    rng = np.random.default_rng(3)
    n = 600
    x = rng.normal(size=n)
    y = np.where(np.arange(n) < 300, 2.0 * x, -2.0 * x + 8.0) + 0.1 * rng.normal(size=n)
    df = pl.DataFrame({"x0": x, "y": y})
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        half_life=50.0,
        min_weight=5.0,
        emit_drift=True,
        drift_action="reset",
        emit_calibration=True,
    )
    out = _run(df, spec)
    fired = out["drift_y"].fill_null(False).arg_true()
    assert fired.len() > 0, "the break fires the detector"
    after = int(fired[0]) + 1
    # The row after a reset reads a fresh calibration: nothing scored yet.
    assert out["calibration_slope_y"][after] is None


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"emit_calibration": False, "calibration_half_life": 10.0}, "needs emit_calibration"),
        ({"calibration_half_life": 0.0}, "calibration_half_life must be > 0"),
        ({"calibration_half_life": -1.0}, "calibration_half_life must be > 0"),
    ],
)
def test_a_memory_is_refused_without_its_switch_or_out_of_range(kw, message):
    with pytest.raises(ValueError, match=message):
        _spec(**kw)


def test_a_model_with_no_prediction_refuses_the_switch():
    with pytest.raises(ValueError, match="emit_calibration does not apply to ew_cov"):
        po.spec.ew_cov("c", features=["x0", "x1"], half_life=10.0, emit_calibration=True)


@pytest.mark.parametrize(
    ("build", "extra"),
    [
        (po.spec.ewridge, {}),
        (po.spec.rls, {}),
        (po.spec.lasso, {"lasso_path": [0.01]}),
        (po.spec.kalman, {"coef_half_life": 100.0}),
        (po.spec.huber, {}),
        (po.spec.quantile, {"quantile": 0.5}),
        (po.spec.sgd, {}),
        (po.spec.pa, {}),
        (po.spec.ftrl, {"loss": "squared"}),
    ],
)
def test_every_linear_model_takes_the_calibration(build, extra):
    """The ten linear models (``holt`` below, with no features)."""
    df = _frame()
    spec = build(
        "m", targets=["y"], features=["x0", "x1"], half_life=60.0, emit_calibration=True, **extra
    )
    out = _run(df, spec)
    # `lasso` names its slot by the path point.
    (slope,) = [c for c in out.columns if c.startswith("calibration_slope_y")]
    assert out[slope].drop_nulls().len() > 400


def test_holt_takes_the_calibration():
    df = _frame()
    spec = po.spec.holt("m", targets=["y"], half_life=60.0, emit_calibration=True)
    out = _run(df, spec)
    assert out["calibration_slope_y"].drop_nulls().len() > 400
