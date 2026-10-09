"""Task 221: the diagnostics that keep a memory of their own.

Each is one accumulator per slot with its own half-life, the model
instance's unless set, and ``inf`` the run-once form (docs/PLAN.md task
221). The oracles -- statsmodels on the same rows -- are in
``tests/test_second_opinion.py``; this file holds each accumulator to the
hard rules: a field never reads its own row (rule 2), one chunk or many give
the same numbers (rule 3), a weight of zero only advances the clock, as the
first row too (rule 9), the state goes on to the bit across a save, a reset
starts it over, and a knob without its switch is refused.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

#: Each switch, its memory's key, and the fields it writes for target `y`.
#: `studentized` is the row's own, as `zscore` is: its scale is read before
#: the row, its residual is the row's.
SWITCHES = {
    "emit_calibration": (
        "calibration_half_life",
        ["calibration_slope_y", "calibration_intercept_y", "calibration_wald_y"],
    ),
    "emit_breaks": (
        "breaks_half_life",
        ["studentized_y", "cusum_y", "cusum_sq_y", "break_wald_y"],
    ),
}
OWN_ROW = {"studentized_y"}


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


def _spec(switch: str, **kw):
    base = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=60.0,
        weight="w",
        min_weight=10.0,
    )
    base[switch] = True
    base.update(kw)
    return po.spec.ewridge("m", **base)


def _run(df: pl.DataFrame, spec, chunk: int | None = None) -> pl.DataFrame:
    bank = po.ModelBank([spec])
    if chunk is None:
        return bank.fit_predict(df).unnest("m")
    parts = [bank.fit_predict(df.slice(i, chunk)) for i in range(0, df.height, chunk)]
    return pl.concat(parts).unnest("m")


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("memory", [None, 150.0, float("inf")])
def test_chunk_invariant(switch, memory):
    key, fields = SWITCHES[switch]
    df = _frame()
    spec = _spec(switch, **({key: memory} if memory is not None else {}))
    whole = _run(df, spec).select(fields)
    for f in fields:
        if f == "break_wald_y" and memory == float("inf"):
            assert whole[f].null_count() == whole.height, "run once has no slow fit"
        else:
            assert whole[f].drop_nulls().len() > 400, f"{f}: the test reads values"
    for chunk in (7, 600):
        got = _run(df, spec, chunk).select(fields)
        assert got.equals(whole, null_equal=True), f"{chunk}-row chunks"


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_field_never_reads_its_own_row(switch):
    """Rule 2: a row's outcome moves the fields of the rows after it, never
    its own -- but for the studentized residual, which is the row's own
    residual over a scale read before it."""
    _, fields = SWITCHES[switch]
    df = _frame()
    at = 400
    moved = df.with_columns(
        pl.when(pl.int_range(pl.len()) == at)
        .then(pl.col("y") + 50.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    a, b = _run(df, _spec(switch)), _run(moved, _spec(switch))
    for f in fields:
        if f in OWN_ROW:
            # The scale is the row's before: the ratio to the residual holds.
            ra = a[f][at] / a["resid_y"][at]
            rb = b[f][at] / b["resid_y"][at]
            assert abs(ra - rb) < 1e-12 * abs(ra), f
            continue
        assert a[f][: at + 1].equals(b[f][: at + 1], null_equal=True), f
        assert a[f][at + 1] != b[f][at + 1], f


@pytest.mark.parametrize("switch", SWITCHES)
def test_zero_weight_rows_only_advance_the_clock(switch):
    """Rule 9: the stream opens on two rows of weight zero, which leave no
    trace -- the fields are those of the stream without them -- and no
    field is NaN."""
    _, fields = SWITCHES[switch]
    df = _frame().with_columns(pl.int_range(pl.len()).cast(pl.Float64).alias("t"))
    spec = _spec(switch, clock="t", gap_cap=10.0)
    out = _run(df, spec)
    for f in fields:
        assert out[f].is_nan().sum() == 0, f
    # Nothing was learned before the third row, so the two heads' clock
    # steps reach no weight, and the fields agree.
    rest = _run(df.slice(2), spec)
    for f in fields:
        np.testing.assert_allclose(
            out[f][2:].to_numpy(), rest[f].to_numpy(), rtol=1e-12, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("switch", SWITCHES)
def test_the_state_goes_on_across_a_save(switch):
    _, fields = SWITCHES[switch]
    df = _frame()
    whole = _run(df, _spec(switch)).select(fields)
    bank = po.ModelBank([_spec(switch)])
    first = bank.fit_predict(df.slice(0, 500))
    resumed = po.ModelBank.load_bytes(bank.save_bytes())
    rest = resumed.fit_predict(df.slice(500))
    got = pl.concat([first, rest]).unnest("m").select(fields)
    assert got.equals(whole, null_equal=True)


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_reset_starts_the_diagnostic_over(switch):
    """``drift_action = "reset"`` restarts the diagnostics with the model:
    the row after it reads accumulators with nothing in them."""
    _, fields = SWITCHES[switch]
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
        **{switch: True},
    )
    out = _run(df, spec)
    fired = out["drift_y"].fill_null(False).arg_true()
    assert fired.len() > 0, "the break fires the detector"
    after = int(fired[0]) + 1
    assert out[fields[1]][after - 2] is not None
    assert out[fields[1]][after] is None


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_memory_is_refused_without_its_switch_or_out_of_range(switch):
    key, _ = SWITCHES[switch]
    with pytest.raises(ValueError, match=f"{key} needs {switch}"):
        _spec(switch, **{switch: False, key: 10.0})
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match=f"{key} must be > 0"):
            _spec(switch, **{key: bad})


@pytest.mark.parametrize("switch", SWITCHES)
def test_a_model_with_no_prediction_refuses_the_switch(switch):
    with pytest.raises(ValueError, match=f"{switch} does not apply to ew_cov"):
        po.spec.ew_cov("c", features=["x0", "x1"], half_life=10.0, **{switch: True})


@pytest.mark.parametrize("switch", SWITCHES)
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
def test_every_linear_model_takes_it(switch, build, extra):
    """The ten linear models (``holt`` below, with no features)."""
    _, fields = SWITCHES[switch]
    df = _frame()
    spec = build(
        "m", targets=["y"], features=["x0", "x1"], half_life=60.0, **extra, **{switch: True}
    )
    out = _run(df, spec)
    for f in fields:
        # `lasso` names its slot by the path point.
        (name,) = [c for c in out.columns if c.startswith(f)]
        assert out[name].drop_nulls().len() > 400, (f, name)


@pytest.mark.parametrize("switch", SWITCHES)
def test_holt_takes_it(switch):
    """`holt` has no features: the twin fits compare its level alone."""
    _, fields = SWITCHES[switch]
    df = _frame()
    out = _run(df, po.spec.holt("m", targets=["y"], half_life=60.0, **{switch: True}))
    for f in fields:
        assert out[f].drop_nulls().len() > 400, f


def test_a_feature_set_compares_its_own_coefficients():
    """``break_wald`` reads each slot's own features: a set without the
    feature whose slope breaks barely moves, the set with it passes chi2's
    0.01% value."""
    rng = np.random.default_rng(8)
    n = 1500
    x = rng.normal(size=(n, 2))
    slope = np.where(np.arange(n) < 900, 1.0, 2.5)
    y = slope * x[:, 0] - x[:, 1] + rng.normal(size=n)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        feature_sets={"both": ["x0", "x1"], "other": ["x1"]},
        half_life=100.0,
        emit_breaks=True,
    )
    out = _run(df, spec)
    after = out.slice(950, 100)
    assert after["break_wald_y__both"].max() > 21.1
    assert after["break_wald_y__other"].max() < after["break_wald_y__both"].max() / 4
