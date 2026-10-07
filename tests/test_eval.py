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


# --- A relative target (review round 4, YB1) ---------------------------------

HOW = {
    "difference": pl.col("p") - pl.col("mid"),
    "ratio": pl.col("p") / pl.col("mid"),
    "log_ratio": (pl.col("p") / pl.col("mid")).log(),
}


def _relative_frame(n: int = 600, seed: int = 3) -> pl.DataFrame:
    """A price ``p`` about a drifting ``mid`` (``test_relative_targets.py``'s
    frame), with what the bank cannot use on either side: a null, a value past
    its input bound on both sides at once -- two values that must not subtract
    into a usable zero -- and a ``mid`` that is not positive, which a ratio
    cannot take. The last row's ``p`` is null: the bank reads its metrics
    before each row, so the last row's are over every row before it, which
    are the rows ``po.eval`` scores."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    mid = 100.0 + np.cumsum(0.2 * rng.standard_normal(n))
    p = mid * np.exp(0.01 * (0.5 * x[:, 0] - 0.3 * x[:, 1]) + 0.002 * rng.standard_normal(n))
    i = pl.int_range(pl.len())
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "mid": mid, "p": p}).with_columns(
        p=pl.when((i % 17 == 3) | (i == n - 1))
        .then(None)
        .when(i == 50)
        .then(1e200)
        .otherwise(pl.col("p")),
        mid=pl.when(i % 23 == 5)
        .then(None)
        .when(i == 50)
        .then(1e200)
        .when(i % 29 == 7)
        .then(-1.0)
        .otherwise(pl.col("mid")),
    )


def _relative_fit(relative: str, name: str | None = None):
    target = po.target("p", relative_to="mid", relative=relative, name=name)
    spec = po.spec.ewridge(
        "m", targets=[target], features=["x0", "x1"], half_life=float("inf"), emit_metrics=True
    )
    return spec, po.ModelBank([spec]).fit_predict(_relative_frame())


@pytest.mark.parametrize("name", [None, "ret"], ids=["named after its column", "named"])
@pytest.mark.parametrize("relative", list(HOW))
def test_a_relative_target_is_scored_as_the_bank_scores_it(relative, name):
    """Review round 4 (YB1): ``po.eval`` read a relative target's raw column
    as ``y`` -- the price, about 100, against a prediction of the price less
    the mid, about 0 -- and reported ``r2 = -1723`` beside the bank's own
    ``hit_rate`` of 0.86; a target with a name of its own raised polars'
    ``ColumnNotFoundError``. With the spec in hand, ``y`` is the bank's: the
    difference, the ratio or the log ratio, null where the bank cannot use a
    side, and a ratio's hit test about 1. Held to the bank's own
    ``emit_metrics`` without decay, which over the same rows are the same
    three numbers."""
    spec, out = _relative_fit(relative, name)
    t = name or "p"
    bank = out["m"].struct.unnest().tail(1)
    centre = 1.0 if relative == "ratio" else 0.0
    pred = out["m"].struct.field(f"pred_{t}")
    # A prediction exactly on the centre is a hit in the bank and a miss
    # here (S6, not this test's): none may sit there.
    assert pred.is_not_null().sum() > 500 and not (pred == centre).any()

    def held(frame: pl.DataFrame) -> None:
        assert frame["target"].to_list() == [t]
        for metric in ("r2", "ic", "hit_rate"):
            want = bank[f"{metric}_{t}"][0]
            assert frame[metric][0] == pytest.approx(want, rel=1e-9, abs=1e-12), (metric, frame)

    held(po.eval.from_sums(po.eval.sums(out, "m", spec=spec), min_obs=1))
    got = po.eval.metrics(out, "m", spec=spec, min_obs=1)
    held(got)
    assert 0.3 < got["hit_rate"][0] < 0.95, "a rate, not the sign test's 1.0 or 0.0"
    # And the windowed and stacked forms take the spec the same way.
    windows = po.eval.rolling_metrics(
        out.with_row_index("i"), "m", clock="i", window_size=1000, spec=spec, min_obs=1
    )
    assert windows["hit_rate"].to_list() == pytest.approx(got["hit_rate"].to_list(), rel=1e-12)
    stacked = po.eval.compare_specs(out, ["m"], specs=[spec], min_obs=1)
    assert stacked.drop("spec").equals(got)


def test_unpack_takes_a_target_as_its_spec_writes_it():
    """``y`` is each target's own definition (``po.target``'s table), computed
    in polars here as ``test_relative_targets.py`` computes it, and
    ``target`` is the name the output fields carry. A column renamed with
    ``name=`` is the column."""
    df = _relative_frame()
    targets = [
        po.target("p", relative_to="mid", name="diff"),
        po.target("p", relative_to="mid", relative="ratio", name="ratio"),
        po.target("p", relative_to="mid", relative="log_ratio", name="log"),
        po.target("p", name="price"),
    ]
    spec = po.spec.ewridge("m", targets=targets, features=["x0", "x1"], half_life=float("inf"))
    long = po.eval.unpack(po.ModelBank([spec]).fit_predict(df), "m", spec=spec)
    usable = (pl.col("p").abs() <= 1e100) & (pl.col("mid").abs() <= 1e100)
    positive = usable & (pl.col("p") > 0) & (pl.col("mid") > 0)
    want = {
        "diff": pl.when(usable).then(HOW["difference"]),
        "ratio": pl.when(positive).then(HOW["ratio"]),
        "log": pl.when(positive).then(HOW["log_ratio"]),
        "price": pl.col("p"),
    }
    assert long["target"].unique(maintain_order=True).to_list() == list(want)
    for name, expr in want.items():
        got = long.filter(pl.col("target") == name)["y"]
        assert got.equals(df.select(expr.alias("y"))["y"]), name
        assert got.is_null().sum() >= 50 or name == "price", name


def test_without_the_spec_a_slot_named_after_no_column_says_to_pass_it():
    """A target with a name of its own writes ``pred_<name>``, which names no
    column: without the spec there is no way to know what ``y`` is, and the
    refusal says where it is written down."""
    _, out = _relative_fit("log_ratio", "ret")
    with pytest.raises(ValueError, match=r"slot 'pred_ret'.*pass spec="):
        po.eval.metrics(out, "m", min_obs=1)
    with pytest.raises(ValueError, match=r"slot 'pred_ret'.*pass spec="):
        po.eval.unpack(out, "m")


def test_a_formula_target_the_frame_has_no_column_for_is_refused_by_name():
    """A formula target's value is resolved inside the bank and not written
    out; with the spec, a frame without the column is the documented
    ``ValueError``, not polars' ``ColumnNotFoundError``."""
    df = pl.DataFrame(
        {"t": np.arange(60, dtype=float), "x": np.sin(np.arange(60.0)), "mid": np.arange(60.0)}
    )
    fwd = (po.rewm_mean("mid", half_life=5.0, window_size=5.0) - pl.col("mid")).alias("fwd")
    spec = po.spec.ewridge(
        "m",
        targets=[fwd],
        features=["x"],
        clock="t",
        gap_cap=100.0,
        half_life=50.0,
        embargo=5.0,
        min_weight=3.0,
    )
    out = po.ModelBank([spec]).fit_predict(df)
    with pytest.raises(ValueError, match=r"slot 'pred_fwd'.*'fwd'"):
        po.eval.unpack(out, "m", spec=spec)
    # With the column there, as `with_windows` writes it, it is read.
    with_column = out.with_columns(fwd=pl.lit(0.5))
    assert po.eval.unpack(with_column, "m", spec=spec)["y"].unique().to_list() == [0.5]


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Int32, pl.UInt32])
def test_rolling_metrics_keeps_an_integer_clocks_dtype(dtype):
    """Review round 4 (YB14): on an ``Int64`` clock ``window_start`` came back
    ``Float64`` (``[0.0, 500.0, ...]``), where a temporal clock keeps its own
    dtype. A window of a whole number of the clock's units keeps it too, and
    one that is not whole is refused, as ``embargo`` refuses a delay the
    clock cannot hold: its bucket edges are not values of the column."""
    out = _fitted(n_groups=1, n_rows=600)
    ints = out.with_columns(pl.col("t").round().cast(dtype))
    floats = ints.with_columns(pl.col("t").cast(pl.Float64))
    want = po.eval.rolling_metrics(
        floats, "m", clock="t", window_size=800.0, targets=["y0"], min_obs=5
    )
    assert want.height > 1
    for window in (800, 800.0):
        got = po.eval.rolling_metrics(
            ints, "m", clock="t", window_size=window, targets=["y0"], min_obs=5
        )
        assert got["window_start"].dtype == dtype, window
        assert got.with_columns(pl.col("window_start").cast(pl.Float64)).equals(want), window
    with pytest.raises(ValueError, match="^rolling_metrics: window_size 2.5 is not a whole number"):
        po.eval.rolling_metrics(ints, "m", clock="t", window_size=2.5)


@pytest.mark.parametrize(
    "bad",
    [None, float("nan"), float("inf"), -1.0, 1e101],
    ids=["null", "nan", "inf", "-1", "bound"],
)
def test_sums_drops_a_row_whose_weight_the_bank_would_not_learn_from(bad):
    """Review round 4 (YB21): a null weight counted in ``n`` but not in ``w``
    or the means, and a NaN, infinite, negative or out-of-bound weight went
    into the sums as it was. The bank skips a row whose weight is not a
    number it can use and refuses a negative one; here each is dropped, as a
    missing prediction or target is. A zero weight is legal and stays: in
    ``n``, and not in ``w``."""
    out = _fitted(n_groups=1, n_rows=200)
    marked = pl.int_range(pl.len())
    out = out.with_columns(
        pl.when(marked == 99).then(0.0).otherwise(1.0 + 0.01 * marked).alias("w")
    )
    rows = [30, 31, 120]
    bad_rows = out.with_columns(
        pl.when(marked.is_in(rows)).then(pl.lit(bad, dtype=pl.Float64)).otherwise("w").alias("w")
    )
    got = po.eval.sums(bad_rows, "m", weight="w")
    want = po.eval.sums(out.filter(~marked.is_in(rows)), "m", weight="w")
    assert got.equals(want), (got, want)
    scored = out["m"].struct.field("pred_y0").is_not_null().sum()
    assert po.eval.sums(out, "m", weight="w")["n"][0] == scored, "the zero-weight row is in n"
