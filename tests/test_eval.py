"""Task 12: the evaluation harness (docs/PLAN.md section 8)."""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


def _fitted(n_groups=2, n_rows=300, **kw):
    df, _ = synthetic(seed=51, n_groups=n_groups, n_rows=n_rows, k=3, null_frac=0.0)
    opts = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        half_life=200.0,
        clock="t",
        gap_cap=50.0,
        group="group",
        min_weight=10.0,
    )
    opts.update(kw)
    return po.ModelBank([po.spec.ewridge("m", **opts)]).fit_predict(df)


def test_metrics_shape_and_values():
    out = _fitted()
    m = po.eval.metrics(out, "m", by=["group"], targets=["y0"])
    assert set(m["group"]) == {"g0", "g1"}
    assert m["slot"].unique().to_list() == ["pred_y0"]
    # synthetic data is genuinely predictable, so R^2 and IC must be positive
    assert (m["r2"] > 0.3).all()
    assert (m["ic"] > 0.5).all()
    assert ((m["hit_rate"] >= 0) & (m["hit_rate"] <= 1)).all()


def test_metrics_drops_nulls_not_rows():
    # Warmup rows have null predictions; they must not enter the counts.
    # (min_weight must stay below the saturation level of weight_sum, which for
    # half-life 200 and ~10-unit clock deltas is around 29.)
    out = _fitted(min_weight=20.0)
    m = po.eval.metrics(out, "m", targets=["y0"])
    n_finite = out["m"].struct.field("pred_y0").is_not_null().sum()
    assert m["n"][0] == n_finite


def test_r2_matches_a_manual_computation():
    out = _fitted(n_groups=1)
    m = po.eval.metrics(out, "m", targets=["y0"])
    pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
    y = out["y0"].to_numpy().astype(float)
    ok = np.isfinite(pred) & np.isfinite(y)
    r2 = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    assert abs(m["r2"][0] - r2) < 1e-12


def test_rolling_windows_partition_the_clock():
    out = _fitted(n_groups=1, n_rows=600)
    r = po.eval.rolling_metrics(out, "m", clock="t", window_size=800.0, targets=["y0"], min_obs=5)
    assert r.height > 1
    starts = r["window_start"].to_numpy()
    assert (np.diff(starts) == 800.0).all()
    assert (starts % 800.0 == 0).all()


def test_compare_specs_stacks_with_a_spec_column():
    df, _ = synthetic(seed=52, n_groups=1, n_rows=250, k=3, null_frac=0.0)
    common = dict(targets=["y0"], features=["x0", "x1", "x2"], half_life=200.0, min_weight=10.0)
    out = po.ModelBank(
        [
            po.spec.ewridge("a", ridge=1e-6, **common),
            po.spec.ewridge("b", ridge=10.0, **common),
        ]
    ).fit_predict(df)
    cmp = po.eval.compare_specs(out, ["a", "b"], targets=["y0"])
    assert set(cmp["spec"]) == {"a", "b"}
    # heavy ridge must fit worse on data with a real signal
    mse = dict(zip(cmp["spec"], cmp["mse"], strict=True))
    assert mse["a"] < mse["b"]


def test_grid_slots_resolve_to_their_target():
    df, _ = synthetic(seed=53, n_groups=1, n_rows=250, k=2, n_targets=2, null_frac=0.0)
    spec = po.spec.ewridge(
        "m",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        ridge=[1e-6, 1.0],
        half_life=200.0,
        min_weight=10.0,
    )
    out = po.ModelBank([spec]).fit_predict(df)
    m = po.eval.metrics(out, "m", targets=["y0", "y1"])
    assert m.height == 4  # 2 targets x 2 ridge values
    assert set(m["target"]) == {"y0", "y1"}


def test_unpack_is_long_form():
    out = _fitted(n_groups=1, n_rows=100)
    long = po.eval.unpack(out, "m", targets=["y0"])
    assert long.height == out.height  # one slot
    assert {"slot", "target", "pred", "y"} <= set(long.columns)


def test_rejects_non_struct_column():
    out = _fitted(n_groups=1, n_rows=50)
    with pytest.raises(TypeError, match="not a model-output struct"):
        po.eval.metrics(out, "x0")


def test_mistakes_are_named():
    # The failure modes eval's docstrings promise, each as the exception it
    # says: a missing column, a struct with nothing to unpack (ew_cov), a
    # target the frame has lost, a window that would divide by zero, and a
    # clock that is not a number.
    out = _fitted(n_groups=1, n_rows=50)
    with pytest.raises(KeyError, match="nope"):
        po.eval.metrics(out, "nope")
    cov = po.ModelBank([po.spec.ew_cov("c", features=["x0", "x1"], half_life=20.0)]).fit_predict(
        out.drop("m")
    )
    with pytest.raises(TypeError, match="no prediction fields"):
        po.eval.unpack(cov, "c")
    with pytest.raises(ValueError, match="cannot infer the target column for slot 'pred_y0'"):
        po.eval.unpack(out.drop("y0"), "m")
    with pytest.raises(ValueError, match="window_size must be > 0, got 0"):
        po.eval.rolling_metrics(out, "m", clock="t", window_size=0)
    with pytest.raises(TypeError, match="clock column 'group' must be numeric"):
        po.eval.rolling_metrics(out, "m", clock="group", window_size=10.0)
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        po.eval.metrics(out, "m", by=["zz"])


def test_noise_target_gives_no_edge():
    rng = np.random.default_rng(3)
    n = 3000
    df = pl.DataFrame(
        {"x0": rng.standard_normal(n), "x1": rng.standard_normal(n), "y0": rng.standard_normal(n)}
    )
    spec = po.spec.ewridge(
        "m", targets=["y0"], features=["x0", "x1"], half_life=300.0, min_weight=20.0
    )
    out = po.ModelBank([spec]).fit_predict(df)
    m = po.eval.metrics(out, "m", targets=["y0"])
    assert abs(m["ic"][0]) < 0.06
    assert abs(m["hit_rate"][0] - 0.5) < 0.05


def _logistic_fit(n=20000, seed=0, informative=True, emit_metrics=False):
    """A `sgd(loss="logistic")` fit: `pred` a probability, `y0` a 0/1 label.

    `informative=False` gives the features no relationship to the label at
    all, the fixture PLAN task 76 measured `hit_rate` at exactly 1.0 on."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    if informative:
        p = 1.0 / (1.0 + np.exp(-(1.2 * x0 - 0.8 * x1)))
        y = (rng.random(n) < p).astype(float)
    else:
        y = (rng.random(n) < 0.5).astype(float)
    df = pl.DataFrame({"x0": x0, "x1": x1, "y0": y})
    spec = po.spec.sgd(
        "m",
        targets=["y0"],
        features=["x0", "x1"],
        loss="logistic",
        learning_rate=0.05,
        half_life=float("inf"),
        min_weight=50.0,
        emit_metrics=emit_metrics,
    )
    return po.ModelBank([spec]).fit_predict(df), y


class TestBinary:
    """PLAN task 76: `hit_rate` on a probability against a 0/1 label. Before
    this, `pred.signum() == y.signum()` agreed on every row (both positive by
    construction) and `hit_rate` read 1.0 whatever the fit did."""

    def test_binary_hit_rate_is_not_a_constant_one(self):
        out, _ = _logistic_fit(informative=False)
        m = po.eval.metrics(out, "m", targets=["y0"], binary=True)
        assert m["hit_rate"][0] < 0.6, f"a fit that knows nothing scored {m['hit_rate'][0]}"

    def test_binary_hit_rate_matches_a_numpy_replica(self):
        out, y = _logistic_fit()
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        ok = np.isfinite(pred)
        want = float(((pred[ok] > 0.5) == (y[ok] > 0.5)).mean())
        m = po.eval.metrics(out, "m", targets=["y0"], binary=True)
        assert abs(m["hit_rate"][0] - want) < 1e-12
        assert want > 0.6, f"the fit should have learned something: {want}"

    def test_sign_reading_still_reads_one_on_the_same_fit(self):
        # binary=False (the default) is unchanged: two positive numbers
        # always agree in sign, which is exactly the defect -- reading it
        # without `binary=True` still shows the old number, undisturbed.
        out, _ = _logistic_fit()
        m = po.eval.metrics(out, "m", targets=["y0"])
        assert m["hit_rate"][0] == 1.0

    def test_binary_adds_log_loss(self):
        out, y = _logistic_fit()
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        ok = np.isfinite(pred)
        p = np.clip(pred[ok], 1e-15, 1 - 1e-15)
        want = float(-(y[ok] * np.log(p) + (1 - y[ok]) * np.log(1 - p)).mean())
        m = po.eval.metrics(out, "m", targets=["y0"], binary=True)
        assert abs(m["log_loss"][0] - want) < 1e-9
        assert "log_loss" not in po.eval.metrics(out, "m", targets=["y0"]).columns

    def test_streaming_metric_agrees_with_the_frame(self):
        # The same invariant E22 already claims for the regression reading:
        # `emit_metrics` (O(state), read before each row) and `po.eval.metrics`
        # (over the collected frame) must land on the same number.
        out, _ = _logistic_fit(emit_metrics=True)
        last = out["m"].struct.field("hit_rate_y0")[-1]
        m = po.eval.metrics(out, "m", targets=["y0"], binary=True)
        # emit_metrics is exponentially weighted (half_life=inf here, so it is
        # a plain running mean) and read before the last row; po.eval scores
        # every row including the last, so compare to that same window.
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        assert abs(last - m["hit_rate"][0]) < 0.01, (
            last,
            m["hit_rate"][0],
            (~np.isnan(pred)).sum(),
        )

    def test_sums_binary_matches_metrics(self):
        out, _ = _logistic_fit()
        want = po.eval.metrics(out, "m", targets=["y0"], binary=True)
        s = po.eval.sums(out, "m", targets=["y0"], binary=True)
        got = po.eval.from_sums(s)
        assert abs(got["hit_rate"][0] - want["hit_rate"][0]) < 1e-12
        assert abs(got["r2"][0] - want["r2"][0]) < 1e-9
        assert abs(got["ic"][0] - want["ic"][0]) < 1e-9


def test_target_named_like_an_output_column_does_not_collide():
    # A target column literally called "y" collided with unpack()'s own "y"
    # output; reserved names are dropped from the passthrough columns instead.
    df, _ = synthetic(seed=54, n_groups=1, n_rows=150, k=2, null_frac=0.0)
    df = df.rename({"y0": "y"})
    spec = po.spec.ewridge(
        "m", targets=["y"], features=["x0", "x1"], half_life=200.0, min_weight=10.0
    )
    out = po.ModelBank([spec]).fit_predict(df)
    m = po.eval.metrics(out, "m", targets=["y"])
    assert m.height == 1
    assert m["target"][0] == "y"
    long = po.eval.unpack(out, "m", targets=["y"])
    assert long.columns.count("y") == 1


def test_rolling_metrics_names_a_missing_clock_before_reading_the_window():
    """R2-P9: a duration ``window_size`` beside a clock the frame lacks was
    compared with the missing dtype first and raised a bare ``TypeError``."""
    df = pl.DataFrame({"t": [0.0, 1.0], "m": [{"pred_y": 1.0, "resid_y": 0.5}] * 2})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        po.eval.rolling_metrics(df, "m", clock="nope", window_size="1h")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1e101], ids=["nan", "inf", "bound"])
def test_what_the_bank_reads_as_missing_is_missing_here_too(bad):
    """Review 2026-10-05 (YB4): the bank reads a NaN target as missing, as it
    does an infinity and a magnitude past its input bound, and still predicts
    the row; the functions here dropped nulls alone, so ``r2``, ``ic`` and
    ``mse`` came out NaN and ``hit_rate`` was quietly lowered. A target or a
    prediction the bank would not learn from is missing here too: each
    function gives what it gives on the frame without those rows."""
    df, _ = synthetic(seed=51, n_groups=1, n_rows=300, k=3, null_frac=0.0)
    bad_y, bad_pred = [40, 41, 150], [200]
    marked = pl.int_range(pl.len())
    df = df.with_columns(pl.when(marked.is_in(bad_y)).then(bad).otherwise(pl.col("y0")).alias("y0"))
    spec = po.spec.ewridge(
        "m", targets=["y0"], features=["x0", "x1", "x2"], half_life=200.0, min_weight=10.0
    )
    out = po.ModelBank([spec]).fit_predict(df)
    assert out["m"].struct.field("pred_y0").gather(bad_y).is_not_null().all(), (
        "the bank scored them"
    )
    # And a prediction the bank would not learn from, which a frame that did
    # not come from a bank can hold.
    pred = pl.col("m").struct.field("pred_y0")
    bad_field = pl.when(marked.is_in(bad_pred)).then(bad).otherwise(pred).alias("pred_y0")
    fields = out.schema["m"].fields
    out = out.with_columns(
        pl.struct(
            [bad_field if f.name == "pred_y0" else pl.col("m").struct.field(f.name) for f in fields]
        ).alias("m")
    )
    assert out["m"].struct.fields == [f.name for f in fields]
    kept = out.filter(~marked.is_in(bad_y + bad_pred))
    for run in (
        lambda f: po.eval.metrics(f, "m", min_obs=1),
        lambda f: po.eval.rolling_metrics(f, "m", clock="t", window_size=800.0, min_obs=1),
        lambda f: po.eval.sums(f, "m"),
    ):
        got, want = run(out), run(kept)
        assert got.equals(want), (got, want)
        assert got.height > 0


def test_rolling_metrics_refuses_a_window_that_is_not_finite():
    """Review 2026-10-05 (YB11): ``window_size=inf`` gave one bucket, whose
    ``window_start`` was NaN."""
    out = _fitted(n_groups=1, n_rows=50)
    with pytest.raises(ValueError, match="^rolling_metrics: window_size must be finite"):
        po.eval.rolling_metrics(out, "m", clock="t", window_size=float("inf"))
