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

TIER = "mixed"

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
    "emit_specification": (
        "specification_half_life",
        ["ljung_box_y", "breusch_pagan_y", "ramsey_reset_y"],
    ),
    "emit_tails": ("tails_half_life", ["skew_y", "kurtosis_y", "jarque_bera_y"]),
    "emit_influence": ("influence_half_life", ["influence_y"]),
}
#: The switches that read a row's leverage, and so only `ewridge`'s, `rls`'s
#: and `kalman`'s.
LEVERAGE = {"emit_influence"}
OWN_ROW = {"studentized_y", "influence_y"}


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


#: A clock in rows whose first two rows -- of weight zero in `_frame` -- sit
#: at row 2's instant.
_HEADS_AT_ROW_2 = pl.max_horizontal(pl.int_range(pl.len()), pl.lit(2)).cast(pl.Float64).alias("t")


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
    field is NaN. They share row 2's clock: a diagnostic's readiness reads
    how far the stream has settled, and the clock a row of weight zero
    covers counts toward it (task 232 (1))."""
    _, fields = SWITCHES[switch]
    df = _frame().with_columns(_HEADS_AT_ROW_2)
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
    f = fields[min(1, len(fields) - 1)]
    assert out[f][after - 2] is not None
    assert out[f][after] is None


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
    # A switch that reads a row's leverage is refused for that first.
    with pytest.raises(ValueError, match=f"{switch} (does not apply to|needs a model).*ew_cov"):
        po.spec.ew_cov("c", features=["x0", "x1"], half_life=10.0, **{switch: True})


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize(
    ("build", "extra"),
    [
        (po.spec.ewridge, {}),
        (po.spec.rls, {}),
        (po.spec.lasso, {"penalties": [0.01]}),
        (po.spec.kalman, {"coef_half_life": 100.0}),
        (po.spec.huber, {}),
        (po.spec.quantile, {"quantile": 0.5}),
        (po.spec.sgd, {}),
        (po.spec.pa, {}),
        (po.spec.ftrl, {"loss": "squared"}),
    ],
)
def test_every_linear_model_takes_it(switch, build, extra):
    """The ten linear models (``holt`` below, with no features); a switch
    that reads a row's leverage, the three that have one."""
    _, fields = SWITCHES[switch]
    df = _frame()
    kw = dict(targets=["y"], features=["x0", "x1"], half_life=60.0, **extra, **{switch: True})
    if switch in LEVERAGE and build not in (po.spec.ewridge, po.spec.rls, po.spec.kalman):
        with pytest.raises(ValueError, match=f"{switch} needs a model that reads one row"):
            build("m", **kw)
        return
    spec = build("m", **kw)
    out = _run(df, spec)
    if build is po.spec.quantile and switch == "emit_calibration":
        # A quantile fit's calibration is its coverage (task 232 (6)).
        fields = ["calibration_coverage_y", "calibration_wald_y"]
    for f in fields:
        # `lasso` names its slot by the path point.
        (name,) = [c for c in out.columns if c.startswith(f)]
        assert out[name].drop_nulls().len() > 400, (f, name)


@pytest.mark.parametrize("switch", SWITCHES)
def test_holt_takes_it(switch):
    """`holt` has no features: the twin fits compare its level alone. Its
    trend at the level's half-life keeps readiness where the level's is:
    at the default, four times it, the 900 rows would end before it."""
    _, fields = SWITCHES[switch]
    df = _frame()
    kw = dict(targets=["y"], half_life=60.0, trend_half_life=60.0, **{switch: True})
    if switch in LEVERAGE:
        with pytest.raises(ValueError, match="needs a model that reads one row"):
            po.spec.holt("m", **kw)
        return
    out = _run(df, po.spec.holt("m", **kw))
    for f in fields:
        if f == "breusch_pagan_y":
            # No feature for a spread to move with.
            assert out[f].null_count() == out.height
            continue
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


# --- robust standard errors (task 221 (c)) ----------------------------------

ROBUST = ["se_coef_hc0", "se_coef_hac"]


def _robust_spec(**kw):
    base = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=60.0,
        weight="w",
        min_weight=10.0,
        emit_robust_se=True,
        robust_se_lags=3,
        coef_every=0,
    )
    base.update(kw)
    return po.spec.ewridge("m", **base)


@pytest.mark.parametrize("memory", [None, 150.0, float("inf")])
def test_robust_se_is_chunk_invariant(memory):
    df = _frame()
    spec = _robust_spec(**({"robust_se_half_life": memory} if memory is not None else {}))
    whole = _run(df, spec).select(ROBUST)
    assert whole["se_coef_hac"].drop_nulls().len() > 600, "the test reads values"
    for chunk in (7, 600):
        got = _run(df, spec, chunk).select(ROBUST)
        assert got.equals(whole, null_equal=True), f"{chunk}-row chunks"


def test_robust_se_zero_weight_rows_only_advance_the_clock():
    df = _frame().with_columns(_HEADS_AT_ROW_2)
    spec = _robust_spec(clock="t", gap_cap=10.0)
    out = _run(df, spec)
    rest = _run(df.slice(2), spec)
    for f in ROBUST:
        a = np.array([np.nan if v is None else v.to_list() for v in out[f][2:]], dtype=object)
        b = np.array([np.nan if v is None else v.to_list() for v in rest[f]], dtype=object)
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_allclose(
                np.asarray(x, float), np.asarray(y, float), rtol=1e-12, equal_nan=True
            )
        assert not any(np.isnan(np.asarray(v.to_list(), float)).any() for v in out[f].drop_nulls())


def test_robust_se_goes_on_across_a_save():
    df = _frame()
    whole = _run(df, _robust_spec()).select(ROBUST)
    bank = po.ModelBank([_robust_spec()])
    first = bank.fit_predict(df.slice(0, 500))
    resumed = po.ModelBank.load_bytes(bank.save_bytes())
    rest = resumed.fit_predict(df.slice(500))
    got = pl.concat([first, rest]).unnest("m").select(ROBUST)
    assert got.equals(whole, null_equal=True)


def test_a_session_change_breaks_the_lags():
    """Newey and West's lag products stay within a run of adjacent rows: a
    session change starts a new one, as it clears the models' lags. Run
    once, the meat is ``statsmodels``' kernel summed over the sessions."""
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import S_hac_simple

    df = _frame().drop("w").with_columns((pl.int_range(pl.len()) // 150).alias("s"))
    spec = _robust_spec(weight=None, half_life=float("inf"), session="s", session_gap=1.0)
    out = _run(df, spec)
    resid = out["resid_y"].to_numpy()
    X = sm.add_constant(df.select("x0", "x1").to_numpy())
    s = df["s"].to_numpy()
    ok = np.isfinite(resid)
    bi = np.linalg.inv(X[ok].T @ X[ok])
    meat = sum(
        S_hac_simple(X[ok & (s == v)] * resid[ok & (s == v)][:, None], nlags=3)
        for v in np.unique(s)
    )
    want = np.sqrt(np.diag(bi @ meat @ bi))
    np.testing.assert_allclose(out["se_coef_hac"][-1].to_numpy(), want, rtol=1e-8)


def test_robust_se_lags_default_to_twice_the_embargo_without_a_clock():
    fields = po.spec.output_fields(_robust_spec(robust_se_lags=None, embargo=5.0))
    assert "se_coef_hac" in fields
    assert "se_coef_hac" not in po.spec.output_fields(_robust_spec(robust_se_lags=None))
    from polars_online import _polars_online as native
    from polars_online import _spec

    resolved = native.resolved_defaults(_spec._json(_robust_spec(robust_se_lags=None, embargo=5.0)))
    assert '"robust_se_half_life":60.0' in resolved.replace(" ", "")


def test_robust_se_lags_default_to_the_horizons_twice_its_rows():
    """The default lags are twice the horizon, the rule the calibration,
    Breusch and Pagan and RESET read beside them: an embargo of 2.5 rows is
    a horizon of 3, and 6 lags, where ``ceil(2 * 2.5)`` read 5."""
    df = _frame()
    default = _run(df, _robust_spec(robust_se_lags=None, embargo=2.5))["se_coef_hac"]
    assert default.drop_nulls().len() > 600, "the test reads values"
    six = _run(df, _robust_spec(robust_se_lags=6, embargo=2.5))["se_coef_hac"]
    five = _run(df, _robust_spec(robust_se_lags=5, embargo=2.5))["se_coef_hac"]
    assert default.equals(six, null_equal=True)
    assert not default.equals(five, null_equal=True)


@pytest.mark.parametrize(
    ("build", "extra", "why"),
    [
        (po.spec.kalman, {"coef_half_life": 100.0}, "random walk"),
        (po.spec.lasso, {"penalties": [0.01]}, "post-selection"),
        (po.spec.huber, {}, "M-estimator"),
        (po.spec.sgd, {}, "gradient fit"),
    ],
)
def test_robust_se_is_the_least_squares_fits(build, extra, why):
    with pytest.raises(ValueError, match=f"emit_robust_se needs a least-squares fit.*{why}"):
        build("m", targets=["y"], features=["x0"], half_life=60.0, emit_robust_se=True, **extra)
    for ok in (po.spec.ewridge, po.spec.rls):
        ok("m", targets=["y"], features=["x0"], half_life=60.0, emit_robust_se=True)


def test_robust_se_knobs_need_their_switch():
    with pytest.raises(ValueError, match="robust_se_lags needs emit_robust_se"):
        _robust_spec(emit_robust_se=False)
    with pytest.raises(ValueError, match="robust_se_half_life needs emit_robust_se"):
        _robust_spec(emit_robust_se=False, robust_se_lags=None, robust_se_half_life=10.0)


def test_the_calibrations_memory_defaults_to_four_half_lives():
    """Left out, the calibration reads at four times the model's half-life
    (a `lam` decay's fourth root, by two square roots) and the others at
    the model's; `inf` stays `inf`."""
    import json
    import math

    from polars_online import _polars_online as native
    from polars_online import _spec

    def stream(**kw):
        spec = po.spec.ewridge("m", targets=["y"], features=["x0"], **kw)
        return json.loads(native.resolved_defaults(_spec._json(spec)))["stream"]

    s = stream(half_life=60.0)
    assert s["calibration_half_life"] == 240.0
    assert s["breaks_half_life"] == s["robust_se_half_life"] == 60.0
    assert stream(half_life=float("inf"))["calibration_half_life"] == "inf"
    assert stream(lam=0.99)["calibration_half_life"] == {"lam": math.sqrt(math.sqrt(0.99))}
    assert (
        stream(half_life=60.0, emit_calibration=True, calibration_half_life=30.0)[
            "calibration_half_life"
        ]
        == 30.0
    )


# --- feature health (task 221 (g)) ------------------------------------------

HEALTH = ["spread_ratio_x0", "spread_ratio_x1", "mean_shift_x0", "mean_shift_x1"]


@pytest.mark.parametrize("memory", [None, 150.0])
def test_feature_health_is_chunk_invariant(memory):
    df = _frame()
    kw = {} if memory is None else {"feature_health_half_life": memory}
    spec = _spec("emit_feature_health", **kw)
    whole = _run(df, spec).select(HEALTH)
    assert whole["spread_ratio_x0"].drop_nulls().len() > 800
    for chunk in (7, 600):
        assert _run(df, spec, chunk).select(HEALTH).equals(whole, null_equal=True), chunk


def test_feature_health_reads_the_rows_before_its_own():
    """A row's features move the health of the rows after it, never its
    own."""
    df = _frame()
    at = 400
    moved = df.with_columns(
        pl.when(pl.int_range(pl.len()) == at)
        .then(pl.col("x0") + 9.0)
        .otherwise(pl.col("x0"))
        .alias("x0")
    )
    a = _run(df, _spec("emit_feature_health"))
    b = _run(moved, _spec("emit_feature_health"))
    for f in HEALTH:
        assert a[f][: at + 1].equals(b[f][: at + 1], null_equal=True), f
    assert a["spread_ratio_x0"][at + 1] != b["spread_ratio_x0"][at + 1]


def test_feature_health_is_refused_where_it_would_be_null():
    """Run once the fast and slow memories are both the whole stream and
    every field would be null, so the switch is refused, naming the way out
    (task 232 (2); review round 6, B-4). Beside a window it takes the
    window's memory and reads."""
    for kw in ({"half_life": float("inf")}, {"feature_health_half_life": float("inf")}):
        with pytest.raises(ValueError, match="emit_feature_health.*feature_health_half_life"):
            _spec("emit_feature_health", **kw)
    windowed = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=float("inf"),
        window_size=100.0,
        emit_feature_health=True,
    )
    out = _run(_frame(), windowed)
    assert out["spread_ratio_x0"].drop_nulls().len() > 600


def test_feature_health_zero_weights_and_a_save():
    df = _frame().with_columns(pl.int_range(pl.len()).cast(pl.Float64).alias("t"))
    spec = _spec("emit_feature_health", clock="t", gap_cap=10.0)
    out = _run(df, spec)
    rest = _run(df.slice(2), spec)
    for f in HEALTH:
        assert out[f].is_nan().sum() == 0, f
        np.testing.assert_allclose(
            out[f][2:].to_numpy(), rest[f].to_numpy(), rtol=1e-12, atol=0, equal_nan=True
        )
    bank = po.ModelBank([spec])
    first = bank.fit_predict(df.slice(0, 500))
    resumed = po.ModelBank.load_bytes(bank.save_bytes())
    got = pl.concat([first, resumed.fit_predict(df.slice(500))]).unnest("m").select(HEALTH)
    assert got.equals(out.select(HEALTH), null_equal=True)


def test_a_held_feature_goes_quiet_long_before_rls_winds_up():
    """docs/DATA-QUALITY.md's frozen feed: ``x1`` held from row 300 at a
    half-life of 20. ``spread_ratio_x1`` falls below 0.5 within three
    half-lives; ``rls``'s slope passes 1 after about fifty."""
    rng = np.random.default_rng(3)
    n = 3_500
    t = np.arange(n, dtype=float)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    y = x0 + 0.5 * x1 + 0.2 * rng.standard_normal(n)
    held = (t >= 300) & (t < 3_300)
    df = pl.DataFrame({"t": t, "x0": x0, "x1": np.where(held, x1[299], x1), "y": y})
    spec = po.spec.rls(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=5.0,
        half_life=20.0,
        coef_every=1,
        emit_feature_health=True,
    )
    out = _run(df, spec)
    quiet = int(np.flatnonzero((t >= 300) & (out["spread_ratio_x1"].to_numpy() < 0.5))[0])
    wound = int(np.flatnonzero((t >= 300) & (out["coef"].list.get(2).abs().to_numpy() > 1.0))[0])
    assert quiet - 300 < 3 * 20 < 40 * 20 < wound - 300, (quiet, wound)


def test_feature_health_fields_name_their_feature():
    """``output_index`` tells the health fields apart by the feature each
    reads (review round 6, F-9): ``columns`` holds it."""
    idx = po.spec.output_index(_spec("emit_feature_health"))
    rows = idx.filter(pl.col("kind").is_in(["spread_ratio", "mean_shift"]))
    assert rows.height == 4
    for field, columns in rows.select("field", "columns").iter_rows():
        assert columns == [field.split("_")[-1]], (field, columns)


def test_feature_health_is_refused_on_a_model_with_no_features():
    """``holt`` has no features, so the switch would write no field: it is
    refused by name rather than ignored (review round 6, F6)."""
    with pytest.raises(ValueError, match="emit_feature_health.*holt"):
        po.spec.holt("m", targets=["y"], half_life=60.0, emit_feature_health=True)


def test_unnest_takes_the_robust_standard_errors_apart():
    """``se_coef_hc0`` and ``se_coef_hac`` are laid out like ``coef``, so
    ``online.unnest`` takes them apart per coefficient as it does
    ``se_coef`` (review round 6, F4)."""
    df = _frame()
    spec = _robust_spec(emit_se_coef=True, coef_every=1)
    out = _run(df, spec)
    flat = df.lazy().online.fit_predict([spec]).online.unnest([spec]).collect()
    for field in ("se_coef", *ROBUST):
        names = [f"{field}_y_{term}" for term in ("intercept", "x0", "x1")]
        assert set(names) <= set(flat.columns), (field, flat.columns)
        assert field not in flat.columns
        for position, name in enumerate(names):
            assert flat.schema[name] == pl.Float64
            want = out[field].list.get(position, null_on_oob=True)
            assert flat[name].equals(want, null_equal=True), name
        assert flat[names[1]].drop_nulls().len() > 400, field


def _ew_box_pierce(resid: np.ndarray, lam: float, lags: int, ready: np.ndarray) -> float:
    """Box and Pierce's ``n sum(rho_l**2)`` over the scored residuals before
    the last row of the rows ``ready`` holds, each at its present weight
    ``lam**age``, at Kish's ``n``: a pair is weighed by its later row, and
    its partner is the ``l``-th scored residual before it."""
    age = len(resid) - 1 - np.arange(len(resid))
    ok = np.isfinite(resid) & ready
    ok[-1] = False  # the last row is read before it folds
    e, w = resid[ok], lam ** age[ok].astype(float)
    n = w.sum() ** 2 / (w**2).sum()
    d = e - (w * e).sum() / w.sum()
    den = (w * d * d).sum()
    rho = np.array([(w[lag:] * d[lag:] * d[:-lag]).sum() / den for lag in range(1, lags + 1)])
    return float(n * (rho**2).sum())


def test_ljung_box_under_a_memory_is_box_pierce_at_kishs_size():
    """Under a finite memory the statistic is ``n sum(rho_l**2)`` at Kish's
    ``n``. Ljung and Box's ``(n + 2)/(n - l)`` corrects a count of rows; at
    Kish's size it read 15.8% of iid streams past the 5% value at a
    half-life of 10, where this form reads 4.8% (review round 6, B-3). Run
    once it stays Ljung and Box's, statsmodels' ``acorr_ljungbox``
    (``tests/test_second_opinion.py``)."""
    rng = np.random.default_rng(6)
    n = 2_000
    x = rng.normal(size=(n, 2))
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": x @ [0.5, -0.3] + rng.normal(size=n)})
    for half_life in (10.0, 20.0, 200.0):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=half_life,
            emit_specification=True,
        )
        out = _run(df, spec)
        # Folded from the row 95% settled (task 232 (1)).
        ready = out["settled_frac"].to_numpy() >= 0.95
        want = _ew_box_pierce(out["resid_y"].to_numpy(), 0.5 ** (1 / half_life), 10, ready)
        got = out["ljung_box_y"][-1]
        assert got == pytest.approx(want, rel=1e-9), half_life


def test_an_embargo_reads_the_scored_fits_inflation():
    """Under an embargo a held row is folded at its release with the
    prediction it was scored with, and with the error inflation of the fit
    that scored it (review round 6, A-6). Divided by the release-time fit's
    inflation -- a fit that has since learned the rows released before it
    -- the recursive residuals read too small: on iid labels, where an
    embargo of 50 rows only delays learning, the CUSUM of squares at each
    group's last row sat at a mean of -0.70 where the stream without the
    embargo sat at +0.09, and passed 1.96 on 14.7% of 300 groups where it
    passed on 4.3%."""
    rng = np.random.default_rng(6)
    groups, n, k = 300, 1_500, 5
    x = rng.standard_normal((groups * n, k))
    y = 0.5 + x.sum(1) + rng.standard_normal(groups * n)
    cols = {f"x{j}": x[:, j] for j in range(k)}
    df = pl.DataFrame({"g": np.repeat(np.arange(groups), n), "y": y} | cols)
    read = []
    for embargo in (None, 50.0):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=list(cols),
            half_life=float("inf"),
            group="g",
            emit_breaks=True,
            min_weight=12.0,
            ridge=1e-12,
            **({"embargo": embargo} if embargo else {}),
        )
        last = _run(df, spec).group_by("g").last()
        cusum_sq = last["cusum_sq_y"]
        read.append((cusum_sq.mean(), (cusum_sq.abs() > 1.96).mean()))
    (mean_0, rate_0), (mean_50, rate_50) = read
    assert abs(mean_50 - mean_0) < 0.2, read
    assert rate_50 < rate_0 + 0.03, read


# --- readiness (task 232 (1); review round 6, F-1) ---------------------------


def _at_a_level(groups: int, n: int, level: float = 5_000.0, seed: int = 232) -> pl.DataFrame:
    """A target at a level far from 0 -- the prior's mean -- on two
    centred features: `rls` and `kalman` learn the intercept from a prior
    at 0, so their first predictions are thousands of noise sds off."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((groups * n, 2))
    y = level + x @ np.array([0.5, 0.5]) + rng.standard_normal(groups * n)
    g = np.repeat(np.arange(groups), n)
    return pl.DataFrame({"g": g, "x0": x[:, 0], "x1": x[:, 1], "y": y})


@pytest.mark.parametrize("build", [po.spec.rls, po.spec.kalman])
def test_a_diagnostic_folds_from_readiness(build):
    """A diagnostic folds a row only once its instance is ready: its
    prediction past every gate and the stream 95% settled on the fit's
    memory (task 232 (1)). Folded from the first prediction the gates let
    through, the warm-up's residuals -- the prior's bias, thousands of
    noise sds early on -- held Jarque and Bera's statistic past its 5%
    value on 81% of rows 1,000-2,000 beside `rls` at a half-life of 50
    (20-40 half-lives in). The first row a diagnostic reads is the one
    after the first row past 95% settled."""
    n = 2_000
    df = _at_a_level(4, n)
    kw = dict(targets=["y"], features=["x0", "x1"], half_life=50.0, group="g")
    if build is po.spec.kalman:
        kw["coef_half_life"] = 50.0
    spec = build("m", emit_tails=True, emit_breaks=True, emit_specification=True, **kw)
    out = _run(df, spec).with_columns(r=pl.int_range(pl.len()).over("g"))
    late = out.filter(pl.col("r") >= 1_000)
    assert (late["jarque_bera_y"].drop_nulls() > 5.991).mean() < 0.12
    assert (late["ramsey_reset_y"].drop_nulls() > 5.991).mean() < 0.12
    assert (late["cusum_sq_y"].drop_nulls().abs() > 1.96).mean() < 0.06
    first = out.filter(pl.col("g") == 0)
    settled = first["settled_frac"].to_numpy()
    folded = first["kurtosis_y"].is_not_null().arg_true()
    assert folded.len() > 0
    # The first row that reads a value reads the row before it, which was
    # past 95% settled.
    assert settled[int(folded[0]) - 1] >= 0.95


_LEVEL = dict(targets=["y"], half_life=20.0)
_FEATURES = dict(_LEVEL, features=["x0", "x1"])


@pytest.mark.parametrize(
    ("build", "kw", "longest"),
    [
        (po.spec.holt, _LEVEL, 80.0),
        (po.spec.holt, dict(_LEVEL, trend_half_life=50.0), 50.0),
        (po.spec.holt, dict(_LEVEL, trend_half_life=5.0), 20.0),
        (po.spec.holt, dict(_LEVEL, trend=False), 20.0),
        (po.spec.kalman, dict(_FEATURES, coef_half_life=[10.0, 60.0, 10.0]), 60.0),
        (po.spec.kalman, dict(_FEATURES, coef_half_life=[10.0, float("inf"), 10.0]), 20.0),
        (
            po.spec.ewridge,
            dict(
                _FEATURES, session="s", session_gap=1000.0, session_shrink=0.5, long_half_life=70.0
            ),
            70.0,
        ),
        (po.spec.lasso, dict(_FEATURES, penalties=[0.01], select_half_life=90.0), 90.0),
        (po.spec.ewridge, _FEATURES, 20.0),
    ],
)
def test_readiness_reads_the_longest_memory_the_model_keeps(build, kw, longest):
    """A fit with several memories is settled only once the longest finite
    one has, so a diagnostic folds from the row 95% settled on it (task
    234): `holt`'s trend, four times its level's half-life by default;
    `kalman`'s longest of its `coef_half_life`s and its `half_life`;
    `ewridge`'s `long_half_life` and `lasso`'s `select_half_life`. On rows
    one clock unit apart the row past 95% settled is the first `t` with `1
    - 2^(-t / h) >= 0.95`, and the first row read is the one after it."""
    df = _at_a_level(1, 600, level=0.5).with_columns(s=pl.lit(0))
    spec = build("m", emit_tails=True, **kw)
    out = _run(df, spec)
    kurtosis = next(c for c in out.columns if c.startswith("kurtosis_y"))
    first = int(out[kurtosis].is_not_null().arg_true()[0])
    settled = 1.0 - 2.0 ** (-np.arange(len(out)) / longest)
    want = int(np.argmax(settled >= 0.95))
    # Two rows with a spread before a kurtosis: the first read is past the
    # settled row by a row or two.
    assert want + 1 <= first <= want + 3, (first, want)


def test_holts_cusum_waits_for_its_trend():
    """`holt`'s trend forgets at four times its level's half-life. Ready
    on the level alone, its CUSUM spread 1.56 and passed 3 on 6.7% of the
    500 rows after its diagnostics started, on 20 clean streams at a
    half-life of 50 (28% at 200); ready on the trend, the CUSUM near a
    normal's tails from its first row (task 234)."""
    groups, n = 20, 2_000
    rng = np.random.default_rng(234)
    df = pl.DataFrame(
        {"g": np.repeat(np.arange(groups), n), "y": 0.5 + rng.standard_normal(groups * n)}
    )
    spec = po.spec.holt("m", targets=["y"], half_life=50.0, group="g", emit_breaks=True)
    out = _run(df, spec).with_columns(r=pl.int_range(pl.len()).over("g"))
    start = out.filter(pl.col("cusum_y").is_not_null())["r"].min()
    early = out.filter((pl.col("r") >= start) & (pl.col("r") < start + 500))["cusum_y"].drop_nulls()
    assert early.len() > 9_000
    assert early.std() < 1.1, early.std()
    assert (early.abs() > 3).mean() < 0.005


def test_a_standardizing_fit_folds_once_its_scaler_has_warmed_up():
    """`sgd` and `pa` under `standardize` run another fit while their
    scaler warms up (22 rows of Kish's size, `online_core::WARMUP_ROWS`):
    a diagnostic folds from the row after the switch, not from the first
    prediction (task 232 (1))."""
    df = _frame().with_columns(pl.lit(1.0).alias("w"))
    spec = po.spec.sgd(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=float("inf"),
        standardize=True,
        emit_calibration=True,
        calibration_half_life=float("inf"),
    )
    out = _run(df, spec)
    first_pred = int(out["pred_y"].is_not_null().arg_true()[0])
    first_read = int(out["calibration_intercept_y"].is_not_null().arg_true()[0])
    # Two rows with a spread make a slope: without the wait, two rows past
    # the first prediction.
    assert first_read >= 22 > first_pred + 2


# --- the memory from the fit's (task 232 (2)) --------------------------------


def _resolved(spec) -> dict:
    import json

    from polars_online import _polars_online as native
    from polars_online import _spec

    return json.loads(native.resolved_defaults(_spec._json(spec)))["stream"]


def test_a_diagnostics_memory_follows_the_fits():
    """Left out, a diagnostic's memory is a multiple of the fit's own
    (task 232 (2); review round 6, A-3, B-4, G-4, F-5): under `window_size`
    the half-life whose weights have the window's Kish size, `ln 2 / ln((W
    + 1) / (W - 1))` rows, about `W / 2.885`, and `W ln 2 / 2` clock units
    on a clock; beside a decay as well, the truncated weights' Kish size;
    for `kalman`, its `coef_half_life`, the shortest finite one."""
    import math

    common = dict(targets=["y"], features=["x0"], emit_breaks=True, emit_calibration=True)
    w = 500.0
    s = _resolved(po.spec.ewridge("m", half_life=float("inf"), window_size=w, **common))
    want = math.log(2) / math.log((w + 1) / (w - 1))
    assert s["breaks_half_life"] == pytest.approx(want, rel=1e-12)
    assert s["calibration_half_life"] == pytest.approx(4 * want, rel=1e-12)
    assert want == pytest.approx(w / 2.885, rel=2e-3)
    clocked = _resolved(
        po.spec.ewridge(
            "m", half_life=float("inf"), window_size=w, clock="t", gap_cap=5.0, **common
        )
    )
    assert clocked["breaks_half_life"] == pytest.approx(w * math.log(2) / 2, rel=1e-12)
    # A window and a decay: the half-life whose weights have the Kish size
    # of the decay's weights cut at the window.
    h = 200.0
    lam = 0.5 ** (1 / h)
    both = _resolved(po.spec.ewridge("m", half_life=h, window_size=w, **common))
    i = np.arange(w)
    n = (lam**i).sum() ** 2 / (lam ** (2 * i)).sum()
    mu = (n - 1) / (n + 1)
    assert both["breaks_half_life"] == pytest.approx(-math.log(2) / math.log(mu), rel=1e-9)
    assert both["breaks_half_life"] < min(h, want)
    k = _resolved(
        po.spec.kalman(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            coef_half_life=[float("inf"), 30.0, 80.0],
            emit_breaks=True,
            emit_calibration=True,
        )
    )
    assert k["breaks_half_life"] == 30.0
    assert k["calibration_half_life"] == 120.0
    plain = _resolved(po.spec.ewridge("m", half_life=60.0, **common))
    assert plain["breaks_half_life"] == 60.0


def test_beside_a_window_the_robust_errors_read_the_windows_spread():
    """`se_coef_hc0` beside `window_size=400` with no decay read the window
    slope's true spread, 0.0498 over 200 streams (`1 / sqrt(400)` is 0.05),
    where at the run-once memory it read a third of it (review round 6,
    A-2; task 232 (2))."""
    rng = np.random.default_rng(8)
    groups, n = 40, 2_000
    x = rng.standard_normal(groups * n)
    y = 0.3 * x + rng.standard_normal(groups * n)
    df = pl.DataFrame({"g": np.repeat(np.arange(groups), n), "x0": x, "y": y})
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0"],
        window_size=400.0,
        half_life=float("inf"),
        group="g",
        emit_robust_se=True,
        min_weight=10.0,
        coef_every=0,
    )
    last = _run(df, spec).group_by("g").last()
    se = np.median([v[1] for v in last["se_coef_hc0"].to_list()])
    assert se == pytest.approx(0.05, rel=0.15)


def test_kalmans_diagnostics_forget_at_its_coefficients_memory():
    """`kalman` at `half_life=inf` with `coef_half_life=50` forgets its
    coefficients: `break_wald` reads at that memory, where at the run-once
    memory it had no slow fit and was null on every row (review round 6,
    A-3)."""
    rng = np.random.default_rng(11)
    n = 3_000
    x = rng.standard_normal((n, 2))
    b1 = np.where(np.arange(n) >= 1_500, -0.5, 0.5)
    df = pl.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "y": x[:, 0] + b1 * x[:, 1] + rng.standard_normal(n)}
    )
    spec = po.spec.kalman(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        coef_half_life=50.0,
        half_life=float("inf"),
        emit_breaks=True,
    )
    out = _run(df, spec)
    wald = out["break_wald_y"]
    assert wald.drop_nulls().len() > 2_000
    assert (wald[1_500:1_800].drop_nulls() > 21.1).any()


# --- one horizon (task 232 (3)) ----------------------------------------------


def _look_ahead(groups: int, n: int, h: int, phi: float, seed: int, clock: bool = False):
    """A target that sums the next `h` rows' shocks against two AR(1)
    features at `phi`: residuals that overlap by construction, with nothing
    missing."""
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(groups):
        e = rng.standard_normal((n + 200, 2))
        x = np.zeros_like(e)
        for t in range(1, n + 200):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * e[t]
        x = x[200:]
        u = rng.standard_normal(n + h)
        fwd = np.convolve(u, np.ones(h), "valid")[1 : n + 1] / np.sqrt(h)
        cols = {"g": g, "x0": x[:, 0], "x1": x[:, 1], "y": 0.5 + x @ [1.0, -0.5] + fwd}
        if clock:
            cols["t"] = np.arange(n, dtype=float)
        frames.append(pl.DataFrame(cols))
    return pl.concat(frames)


def test_horizon_rows_gives_a_clocked_spec_its_horizon():
    """On a clock column `embargo` is in clock units and says nothing of the
    rows it spans, so the horizon was 0: Ljung-Box tested lags 1-10 of a
    five-row look-ahead target and passed its 5% value on 100% of rows
    (review round 6, B-2). `horizon_rows` gives it, as `embargo` does
    without a clock, and `se_coef_hac` is written."""
    df = _look_ahead(8, 3_000, 5, 0.0, 2)
    df = df.with_columns(pl.int_range(pl.len()).over("g").cast(pl.Float64).alias("t"))
    kw = dict(targets=["y"], features=["x0", "x1"], half_life=200.0, group="g", clock="t")
    kw |= dict(gap_cap=10.0, embargo=5.0, emit_specification=True, emit_robust_se=True)
    clocked = po.spec.ewridge("m", horizon_rows=5, **kw)
    plain = po.spec.ewridge("m", **{k: v for k, v in kw.items() if k not in ("clock", "gap_cap")})
    a, b = _run(df, clocked), _run(df.drop("t"), plain)
    assert a["ljung_box_y"].equals(b["ljung_box_y"], null_equal=True)
    late = a.filter(pl.int_range(pl.len()).over("g") >= 1_000)
    assert (late["ljung_box_y"].drop_nulls() > 18.307).mean() < 0.1
    assert "se_coef_hac" in a.columns


def test_a_clocked_embargo_without_a_horizon_says_so():
    """A spec with a clock column, an `embargo` and no `horizon_rows` has a
    horizon of 0: each diagnostic that reads one says so, once, naming
    `horizon_rows` (review round 6, B-2, A-8)."""
    df = pl.DataFrame({"t": np.arange(50.0), "x0": np.arange(50.0) % 7, "y": np.arange(50.0) % 5})
    kw = dict(targets=["y"], features=["x0"], half_life=20.0, clock="t", gap_cap=5.0, embargo=3.0)
    spec = po.spec.ewridge("m", emit_specification=True, emit_breaks=True, **kw)
    with pytest.warns(po.ReadinessWarning) as caught:
        po.ModelBank([spec]).fit_predict(df)
    said = [str(w.message) for w in caught if "horizon_rows" in str(w.message)]
    assert len(said) == 2, said
    assert any("emit_specification" in m for m in said) and any("emit_breaks" in m for m in said)
    import warnings

    for quiet in (dict(horizon_rows=3), dict(embargo=None), dict(clock=None, gap_cap=None)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ReadinessWarning)
            po.ModelBank([po.spec.ewridge("m", emit_breaks=True, **(kw | quiet))]).fit_predict(df)


def test_horizon_rows_is_refused_without_a_diagnostic_that_reads_it():
    with pytest.raises(ValueError, match="horizon_rows needs a diagnostic that reads it"):
        po.spec.ewridge("m", targets=["y"], features=["x0"], half_life=20.0, horizon_rows=5)
    with pytest.raises(ValueError, match="horizon_rows must be"):
        _spec("emit_breaks", horizon_rows=2**19 + 1)
    assert _resolved(_spec("emit_breaks", horizon_rows=4))["horizon_rows"] == 4
    assert _resolved(_spec("emit_breaks", embargo=2.5))["horizon_rows"] == 3


@pytest.mark.extended(reason="40 streams of 3,000 rows per case, eight cases")
@pytest.mark.parametrize("phi", [0.5, 0.95])
def test_under_a_horizon_each_test_is_near_its_size(phi):
    """On a five-row look-ahead target against a persistent feature, with
    nothing missing, Breusch-Pagan's and RESET's `n R²` passed their 5%
    values on 12-36% of rows windowed and 27-75% run once (review round 6,
    B-1); under the horizon each is Wald's with Newey and West's variance,
    and the CUSUMs and `break_wald` read the long-run variance (task 232
    (3)). Windowed each passes on about its size here, under 10%; run
    once, under 20%: Newey and West's estimate errs small on a few thousand
    rows of a persistent feature, and a run-once statistic keeps it."""
    df = _look_ahead(30, 3_000, 5, phi, 21)
    for half_life in (200.0, float("inf")):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=half_life,
            group="g",
            embargo=5.0,
            min_weight=10.0,
            emit_calibration=True,
            emit_specification=True,
            emit_breaks=True,
        )
        out = _run(df, spec).filter(pl.int_range(pl.len()).over("g") >= 1_500)
        for field, crit in (
            ("breusch_pagan_y", 5.991),
            ("ramsey_reset_y", 5.991),
            ("calibration_wald_y", 5.991),
        ):
            rate = (out[field].drop_nulls() > crit).mean()
            assert rate < (0.10 if half_life < float("inf") else 0.20), (half_life, field, rate)
        if half_life < float("inf"):
            assert (out["break_wald_y"].drop_nulls() > 7.815).mean() < 0.10


def test_a_quantile_fits_break_wald_under_a_horizon_is_near_its_size():
    """Under a horizon `break_wald`'s noise variance was scaled by the
    long-run factor of the CUSUM's terms, on a quantile fit the
    indicators', whose autocorrelation under an overlap is smaller than
    the residuals' (the arcsine law), so the statistic ran large: on 30
    quiet five-row look-ahead streams at a half-life of 200 it passed
    `chi2(3)`'s 5% value on 10.8% of rows 1,500+ with a mean of 3.88, where
    `ewridge` on the same streams read 5.9% and 3.05 (review round 7,
    A-2). Scaled by the residuals' factor, the quantile fit reads about
    the same rate and mean as `ewridge`."""
    df = _look_ahead(30, 3_000, 5, 0.8, 21)
    common = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        group="g",
        embargo=5.0,
        min_weight=10.0,
        emit_breaks=True,
    )
    rates = {}
    for name, spec in (
        ("ewridge", po.spec.ewridge("m", **common)),
        ("quantile", po.spec.quantile("m", quantile=0.5, **common)),
    ):
        out = _run(df, spec).filter(pl.int_range(pl.len()).over("g") >= 1_500)
        bw = out["break_wald_y"].drop_nulls()
        rates[name] = (float((bw > 7.815).mean()), float(bw.mean()))
    assert rates["ewridge"][0] < 0.08, rates
    assert rates["quantile"][0] < rates["ewridge"][0] + 0.02, rates
    assert rates["quantile"][1] < 3.4, rates


# --- the nulls standardized (task 232 (4)) -----------------------------------


def _clean(groups: int, n: int, seed: int, df: int | None = None) -> pl.DataFrame:
    """A stable relation, its noise Gaussian or Student's t with `df` degrees
    of freedom at unit variance."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((groups * n, 2))
    if df is None:
        e = rng.standard_normal(groups * n)
    else:
        e = rng.standard_t(df, groups * n) / np.sqrt(df / (df - 2))
    y = 0.5 + x @ np.array([1.0, -1.0]) + e
    return pl.DataFrame(
        {"g": np.repeat(np.arange(groups), n), "x0": x[:, 0], "x1": x[:, 1], "y": y}
    )


def test_a_windowed_cusum_reads_its_own_null():
    """Windowed, the CUSUM's spread was 0.75 beside a fit at the same memory
    and 0.48 at four times it -- the fit absorbs part of every level it
    sums -- and the CUSUM of squares' 0.72, its scale at the same memory
    absorbing the squares' level (review round 6, A-5, G-3). Each is now
    over its null's share of the plain variance, `h_fit / (h_fit + h)`
    and 1/2, and reads a spread of about 1 (task 232 (4))."""
    df = _clean(20, 5_000, 4)
    for memory in (None, 800.0):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=200.0,
            group="g",
            emit_breaks=True,
            **({"breaks_half_life": memory} if memory else {}),
        )
        out = _run(df, spec).filter(pl.int_range(pl.len()).over("g") >= 2_000)
        for field in ("cusum_y", "cusum_sq_y"):
            sd = out[field].drop_nulls().std()
            assert 0.9 < sd < 1.2, (memory, field, sd)


@pytest.mark.parametrize(
    ("coef_half_life", "before"),
    [
        ([30.0, 30.0, 30.0], 1.03),
        ([float("inf"), 30.0, 30.0], 1.40),
        ([30.0, 10.0, 10.0], 1.29),
    ],
)
def test_kalmans_cusum_null_share_reads_the_intercepts_memory(coef_half_life, before):
    """The windowed CUSUM's null share, `h_fit / (h_fit + h)`, read `h_fit`
    as the diagnostics' memory beside `kalman`, the shortest finite
    `coef_half_life`; a level is absorbed by the intercept's coefficient
    at its own entry, and `inf` there pins it, absorbing nothing (review
    round 7, A-1). On 20 clean streams, over rows 3,000+, the CUSUM beside
    `[inf, 30, 30]` spread 1.40 and passed 3 on 3.3% of rows, and beside
    `[30, 10, 10]` 1.29 and 1.8%, where `[30, 30, 30]` read 1.03 and 0.35%
    (`before` is each spread); at the intercept's entry each reads about
    1.03. The CUSUM of squares reads the scale's memory, and does not
    move."""
    df = _clean(20, 6_000, 7)
    spec = po.spec.kalman(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=30.0,
        coef_half_life=coef_half_life,
        group="g",
        emit_breaks=True,
    )
    out = _run(df, spec).filter(pl.int_range(pl.len()).over("g") >= 3_000)
    late = out["cusum_y"].drop_nulls()
    sd, past = late.std(), (late.abs() > 3).mean()
    assert 0.95 < sd < 1.1, (sd, past, before)
    assert past < 0.01, (sd, past, before)
    sq = out["cusum_sq_y"].drop_nulls().std()
    assert 0.95 < sq < 1.1, sq


@pytest.mark.parametrize(
    ("window_size", "half_life", "clock", "before"),
    [
        (100.0, float("inf"), False, 1.19),
        (400.0, float("inf"), False, 1.17),
        (50.0, float("inf"), True, 1.19),
        (100.0, 200.0, False, 1.16),
        (400.0, 200.0, False, 1.06),
    ],
)
def test_a_windows_cusum_null_share_reads_the_window(window_size, half_life, clock, before):
    """Under `window_size` the CUSUM's null share read the window as an
    exponential at its Kish half-life, `h_fit / (h_fit + h)`, 1/2 at the
    default memory; a window absorbs a level as a rolling mean, which takes
    less of every sum than an exponential of the same Kish size (review
    round 7, A-3). On 20 clean streams, over rows 3,000+, the CUSUM spread
    1.19 and passed 3 on 1.2% of rows beside a window of 100 rows and 1.17
    and 1.3% beside 400 (`before` is each spread), and 1.16 and 1.06 beside
    those windows cut from a half-life of 200. Over the window's share,
    0.703 with no decay, each reads about 1. On a clock at two rows a unit
    a window of 50 units is the one of 100 rows. The CUSUM of squares
    reads its scale's memory, and does not move."""
    df = _clean(20, 6_000, 3)
    extra = {}
    if clock:
        df = df.with_columns(t=pl.int_range(pl.len()).over("g") / 2.0)
        extra = {"clock": "t", "gap_cap": 1.0}
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=half_life,
        window_size=window_size,
        group="g",
        emit_breaks=True,
        **extra,
    )
    out = _run(df, spec).filter(pl.int_range(pl.len()).over("g") >= 3_000)
    late = out["cusum_y"].drop_nulls()
    assert late.len() == 60_000
    sd, past = late.std(), (late.abs() > 3).mean()
    assert 0.95 < sd < 1.05, (sd, past, before)
    assert past < 0.005, (sd, past, before)
    sq = out["cusum_sq_y"].drop_nulls().std()
    assert 0.95 < sq < 1.05, sq


def test_the_cusum_of_squares_reads_the_measured_fourth_moment():
    """The CUSUM of squares divided by 2, a Gaussian's `E[z⁴] − 1`: on
    Student's t with 4 degrees of freedom it passed 1.96 on 18.5% of rows
    windowed and on 32.5% of streams' last rows run once (task 222's F2).
    Over the measured `E[z⁴] − 1` it passes on about 5% (task 232 (4))."""
    df = _clean(40, 4_000, 3, df=4)
    common = dict(targets=["y"], features=["x0", "x1"], group="g", emit_breaks=True)
    windowed = _run(df, po.spec.ewridge("m", half_life=200.0, **common))
    late = windowed.filter(pl.int_range(pl.len()).over("g") >= 2_000)
    assert (late["cusum_sq_y"].drop_nulls().abs() > 1.96).mean() < 0.09
    once = po.spec.ewridge("m", half_life=float("inf"), min_weight=10.0, **common)
    last = _run(df, once).group_by("g").last()
    assert (last["cusum_sq_y"].abs() > 1.96).mean() < 0.15


# --- break_wald's exact sandwich (task 232 (5)) ------------------------------


def test_a_feature_gone_quiet_is_no_break():
    """With nothing broken, `x1`'s spread falls tenfold at row 1,500. The
    slow fit's Gram stood in for both fits' in `break_wald`'s variance, so
    the fast fit's quieter design read as a moved coefficient: past
    `chi2(3)`'s 0.01% value, 21.1, after the change on 85% of 100 streams
    (review round 6, A-1). With the exact variance of the difference, on
    few (task 232 (5))."""
    rng = np.random.default_rng(4)
    groups, n, at = 40, 3_000, 1_500
    x = rng.standard_normal((groups * n, 2))
    r = np.tile(np.arange(n), groups)
    x[r >= at, 1] *= 0.1
    y = 0.5 + x @ np.array([1.0, -1.0]) + rng.standard_normal(groups * n)
    df = pl.DataFrame({"g": np.repeat(np.arange(groups), n), "x0": x[:, 0], "x1": x[:, 1], "y": y})
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        group="g",
        emit_breaks=True,
        min_weight=10.0,
    )
    out = _run(df, spec).with_columns(r=pl.int_range(pl.len()).over("g"))
    hit = (
        out.filter(pl.col("r") >= at)
        .group_by("g")
        .agg((pl.col("break_wald_y") > 21.1).any().alias("hit"))["hit"]
        .mean()
    )
    assert hit < 0.1, hit
    after = out.filter(pl.col("r") >= at + 600)["break_wald_y"].drop_nulls()
    assert after.mean() < 4.0, after.mean()


# --- quantile forms (task 232 (6)) -------------------------------------------


@pytest.mark.parametrize("q", [0.5, 0.9])
@pytest.mark.parametrize("build", ["quantile", "sgd"])
def test_a_quantile_fits_calibration_is_its_coverage(q, build):
    """A prediction of the `q`-th quantile is calibrated when the outcome
    falls below it on a share `q` of rows. Mincer and Zarnowitz's test of a
    mean, and a CUSUM of residuals whose mean is not 0, flagged 100% of the
    rows of a stable fit at `q = 0.9` (review round 6, G-1, F-2). The
    calibration is now `calibration_coverage`, the EW share below the
    prediction, with its binomial Wald test at Kish's size, and the CUSUM
    sums `1{y < pred} − q` over its spread (task 232 (6))."""
    df = _clean(10, 4_000, 7)
    common = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        group="g",
        emit_calibration=True,
        emit_breaks=True,
    )
    if build == "quantile":
        spec = po.spec.quantile("m", quantile=q, **common)
    else:
        spec = po.spec.sgd("m", loss="quantile", quantile=q, **common)
    out = _run(df, spec)
    assert "calibration_slope_y" not in out.columns
    late = out.filter(pl.int_range(pl.len()).over("g") >= 2_000)
    assert late["calibration_coverage_y"].median() == pytest.approx(q, abs=0.03)
    assert (late["calibration_wald_y"].drop_nulls() > 3.841).mean() < 0.1
    assert (late["cusum_y"].drop_nulls().abs() > 1.96).mean() < 0.1


# --- the rest (task 232 (14)) -------------------------------------------------


def test_lags_read_from_an_embargo_are_capped():
    """An embargo of 100 rows asked Newey and West's 200 lags, and each lag
    costs a slot a sum of `(k + 1)²` numbers: an embargo of 5,000 rows
    asked 40 MB a slot (review round 6, F-7). Lags read from `embargo` stop
    at 64, with a notice that names the way to more; `horizon_rows`, given,
    takes them all, and `robust_se_lags` sets `se_coef_hac`'s alone."""
    kw = dict(targets=["y"], features=["x0", "x1"], half_life=60.0, emit_robust_se=True)
    capped = po.spec.ewridge("m", embargo=100.0, emit_breaks=True, **kw)
    assert _resolved(capped)["robust_se_lags"] == 64
    with pytest.warns(po.ReadinessWarning, match="horizon_rows to take them all"):
        po.ModelBank([capped]).fit_predict(_frame().drop("w"))
    full = po.spec.ewridge("m", embargo=100.0, horizon_rows=100, **kw)
    assert _resolved(full)["robust_se_lags"] == 200
    import warnings

    for quiet in (full, po.spec.ewridge("m", embargo=30.0, **kw)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", po.ReadinessWarning)
            po.ModelBank([quiet]).fit_predict(_frame().drop("w"))
    own = po.spec.ewridge("m", embargo=100.0, robust_se_lags=150, **kw)
    assert _resolved(own)["robust_se_lags"] == 150


def test_feature_health_is_refused_for_what_it_reads():
    """Feature health reads no residual, so its refusal on a model with no
    prediction says what it does read, where it said "no residuals"
    (review round 6, F-8)."""
    with pytest.raises(ValueError, match="watches the features a fit's coefficients lean on"):
        po.spec.ew_cov("c", features=["x0", "x1"], half_life=10.0, emit_feature_health=True)


#: Task 232's new forms, each with the fields it writes: the Newey-West
#: sums under a horizon, and a quantile fit's coverage and indicator CUSUM.
FORMS = {
    "horizon": (
        lambda **kw: _spec(
            "emit_calibration",
            embargo=3.0,
            emit_breaks=True,
            emit_specification=True,
            **kw,
        ),
        [
            "calibration_wald_y",
            "cusum_y",
            "cusum_sq_y",
            "break_wald_y",
            "breusch_pagan_y",
            "ramsey_reset_y",
        ],
    ),
    "quantile": (
        lambda **kw: po.spec.quantile(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=60.0,
            weight="w",
            min_weight=10.0,
            quantile=0.8,
            emit_calibration=True,
            emit_breaks=True,
            **kw,
        ),
        ["calibration_coverage_y", "calibration_wald_y", "cusum_y"],
    ),
}


@pytest.mark.parametrize("form", FORMS)
def test_the_new_forms_keep_the_hard_rules(form):
    """Rules 2, 3 and 9 for task 232's new sums: one chunk or 7 or 600 give
    the same fields, the state goes on across a save, a row's outcome moves
    only the rows after it, and two heads of weight zero leave no trace."""
    build, fields = FORMS[form]
    df = _frame()
    whole = _run(df, build()).select(fields)
    for f in fields:
        assert whole[f].drop_nulls().len() > 300, (form, f)
    if form == "horizon":
        # The forms are on: with the horizon set to 0 every field moves.
        plain = _run(df, build(horizon_rows=0)).select(fields)
        for f in fields:
            assert not plain[f].equals(whole[f], null_equal=True), f"{f} did not move"
    for chunk in (7, 600):
        assert _run(df, build(), chunk).select(fields).equals(whole, null_equal=True), chunk
    bank = po.ModelBank([build()])
    first = bank.fit_predict(df.slice(0, 500))
    rest = po.ModelBank.load_bytes(bank.save_bytes()).fit_predict(df.slice(500))
    assert pl.concat([first, rest]).unnest("m").select(fields).equals(whole, null_equal=True)
    at = 400
    moved = df.with_columns(
        pl.when(pl.int_range(pl.len()) == at).then(pl.col("y") + 50.0).otherwise(pl.col("y"))
    )
    b = _run(moved, build()).select(fields)
    for f in fields:
        assert whole[f][: at + 1].equals(b[f][: at + 1], null_equal=True), (form, f)
    clocked = df.with_columns(_HEADS_AT_ROW_2)
    out = _run(
        clocked,
        build(clock="t", gap_cap=10.0, horizon_rows=3)
        if form == "horizon"
        else build(clock="t", gap_cap=10.0),
    )
    rest = _run(
        clocked.slice(2),
        build(clock="t", gap_cap=10.0, horizon_rows=3)
        if form == "horizon"
        else build(clock="t", gap_cap=10.0),
    )
    for f in fields:
        assert out[f].is_nan().sum() == 0, (form, f)
        np.testing.assert_allclose(
            out[f][2:].to_numpy(), rest[f].to_numpy(), rtol=1e-12, atol=0, equal_nan=True
        )


# --- each memory against the others (task 278, batch 5) ---------------------

#: The seven diagnostics with a memory of their own: each memory's key, its
#: switch, and the fields the diagnostic writes for target `y` and features
#: `x0`, `x1`.
MEMORIES = {
    "calibration_half_life": ("emit_calibration", SWITCHES["emit_calibration"][1]),
    "breaks_half_life": ("emit_breaks", SWITCHES["emit_breaks"][1]),
    "robust_se_half_life": ("emit_robust_se", ROBUST),
    "specification_half_life": ("emit_specification", SWITCHES["emit_specification"][1]),
    "tails_half_life": ("emit_tails", SWITCHES["emit_tails"][1]),
    "influence_half_life": ("emit_influence", SWITCHES["emit_influence"][1]),
    "feature_health_half_life": ("emit_feature_health", HEALTH),
}
#: Seven distinct memories, none the fit's (120) or a multiple of another's.
DISTINCT = dict(zip(MEMORIES, [33.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0], strict=True))
#: The memories each model takes: the sandwich is of a least-squares fit's
#: coefficients, the influence reads a row's leverage, and `holt` has no
#: features for the health to read.
TAKES = {
    "ewridge": list(MEMORIES),
    "rls": list(MEMORIES),
    "sgd": [k for k in MEMORIES if k not in ("robust_se_half_life", "influence_half_life")],
    "holt": [
        "calibration_half_life",
        "breaks_half_life",
        "specification_half_life",
        "tails_half_life",
    ],
}


def _crossed(n: int = 2_500, seed: int = 7) -> pl.DataFrame:
    """A residual every diagnostic has something to read in: serially
    correlated (AR(2)), heavy-tailed, wider with the first feature, and an
    intercept that moves by 3 halfway; uneven weights."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    e = rng.standard_t(5, size=n)
    u = np.zeros(n)
    for i in range(2, n):
        u[i] = 0.1 * u[i - 1] + 0.5 * u[i - 2] + e[i]
    y = 0.3 + x @ np.array([1.2, -0.7]) + u * (1 + 0.3 * np.abs(x[:, 0]))
    y[n // 2 :] += 3.0
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "w": rng.uniform(0.5, 1.5, n)})


def _every_switch(model: str, **kw) -> pl.DataFrame:
    base: dict = dict(targets=["y"], half_life=120.0, weight="w", min_weight=10.0)
    if model != "holt":
        base.update(features=["x0", "x1"], coef_every=0)
    for key in TAKES[model]:
        base[MEMORIES[key][0]] = True
    if "robust_se_half_life" in TAKES[model]:
        base["robust_se_lags"] = 2
    base.update(kw)
    spec = getattr(po.spec, model)("m", **base)
    return po.ModelBank([spec], threads=1).fit_predict(_crossed()).unnest("m")


@pytest.mark.parametrize(
    "model",
    [
        "ewridge",
        *(
            pytest.param(m, marks=pytest.mark.extended(reason="a grid over the models"))
            for m in ("rls", "sgd", "holt")
        ),
    ],
)
def test_each_memory_moves_its_own_fields_and_nothing_else(model):
    """With every diagnostic on and each memory distinct, changing one
    memory moves its diagnostic's fields and leaves every other column --
    the fit's, the other diagnostics' -- as it was. At the defaults every
    memory but the calibration's is the fit's own, so two memories wired to
    each other's accumulators would pass every test that leaves them out.
    The Rust twin, which `cargo mutants` sees, is
    `crates/online-polars/tests/diagnostic_memories.rs`."""
    memories = {k: DISTINCT[k] for k in TAKES[model]}
    base = _every_switch(model, **memories)
    for key in TAKES[model]:
        _, fields = MEMORIES[key]
        # The fields that read anything: `holt`'s Breusch-Pagan is null, as
        # it has no features to regress the squares on.
        want = [f for f in fields if base[f].drop_nulls().len() > 0]
        assert len(want) >= len(fields) - 1, (key, want)
        for f in want:
            assert base[f].drop_nulls().len() > 300, (model, f)
        got = _every_switch(model, **{**memories, key: 71.0})
        moved = [c for c in base.columns if not base[c].equals(got[c], null_equal=True)]
        assert sorted(moved) == sorted(want), (model, key, moved)


@pytest.mark.parametrize(
    ("knob", "value", "fields"),
    [
        ("ljung_box_lags", 4, ["ljung_box_y"]),
        ("robust_se_lags", 5, ["se_coef_hac"]),
    ],
)
def test_each_lag_moves_its_own_field_and_nothing_else(knob, value, fields):
    """Ljung and Box's lags move `ljung_box` alone, and Newey and West's
    `se_coef_hac` alone, with every other diagnostic on beside them."""
    memories = {k: DISTINCT[k] for k in MEMORIES}
    base = _every_switch("ewridge", **memories)
    got = _every_switch("ewridge", **memories, **{knob: value})
    moved = [c for c in base.columns if not base[c].equals(got[c], null_equal=True)]
    assert moved == fields, moved


def test_every_memory_and_lag_is_refused_out_of_range_or_without_its_switch():
    """Each of the seven memories at 0 and below, and without its switch;
    the lags without theirs, below 0 and past the ring's 2^20; the horizon
    below 0. Each refusal names the key."""
    base = dict(targets=["y"], features=["x0", "x1"], half_life=60.0)
    for key, (switch, _) in MEMORIES.items():
        with pytest.raises(ValueError, match=f"{key} needs {switch}"):
            po.spec.ewridge("m", **base, **{key: 30.0})
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match=f"{key} must be > 0"):
                po.spec.ewridge("m", **base, **{switch: True, key: bad})
    for kw, says in (
        (dict(ljung_box_lags=3), "ljung_box_lags needs emit_specification"),
        (
            dict(emit_specification=True, ljung_box_lags=2**20 + 1),
            "ljung_box_lags must be <= 1048576, got 1048577",
        ),
        (dict(emit_robust_se=True, robust_se_lags=-1), "robust_se_lags must be >= 0, got -1"),
        (
            dict(emit_robust_se=True, robust_se_lags=2**20 + 1),
            "robust_se_lags must be <= 1048576, got 1048577",
        ),
        (dict(emit_breaks=True, horizon_rows=-1), "horizon_rows must be >= 0, got -1"),
    ):
        with pytest.raises(ValueError, match=says):
            po.spec.ewridge("m", **base, **kw)
    # The edges themselves are taken.
    for kw in (
        dict(emit_robust_se=True, robust_se_lags=0),
        dict(emit_robust_se=True, robust_se_lags=2**20),
        dict(emit_breaks=True, horizon_rows=0),
    ):
        po.ModelBank([po.spec.ewridge("m", **base, **kw)])


def test_the_influence_scale_reads_its_own_memory():
    """`influence_half_life=30` beside a fit at 120: `influence = (v / s)
    sqrt(h)` with `v = resid / error_inflation`, `h = error_inflation² - 1`
    and `s²` the mean of the folded rows' `v²` before the row, each at its
    weight aged at 30. The rows folded are the fit's: 95% settled at 120."""
    n = 1_500
    df = _crossed(n, 31)
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=120.0,
        weight="w",
        min_weight=10.0,
        emit_error_inflation=True,
        emit_influence=True,
        influence_half_life=30.0,
    )
    out = _run(df, spec)
    infl = out["error_inflation_y"].to_numpy()
    v = out["resid_y"].to_numpy() / infl
    w = df["w"].to_numpy()
    ready = out["settled_frac"].to_numpy() >= 0.95
    assert 500 < int(np.argmax(ready)) < 560
    got = out["influence_y"].to_numpy().astype(float)
    checked = 0
    for t in range(560, n):
        ok = np.isfinite(v[:t]) & ready[:t]
        om = (w[:t] * 0.5 ** (((t - 1) - np.arange(t)) / 30.0))[ok]
        s = np.sqrt((om * v[:t][ok] ** 2).sum() / om.sum())
        # Measured 2026-10-25: 9e-16 relative at most.
        np.testing.assert_allclose(got[t], v[t] / s * np.sqrt(infl[t] ** 2 - 1.0), rtol=1e-12)
        checked += 1
    assert checked == n - 560


def test_the_studentized_scale_reads_the_breaks_memory():
    """`breaks_half_life=40` beside a fit at 120: `studentized = v / s`, `s²`
    the folded rows' mean `v²` before the row at their weights aged at 40,
    once those weights have a Kish size of 10."""
    n, h = 1_500, 40.0
    df = _crossed(n, 32)
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=120.0,
        weight="w",
        min_weight=10.0,
        emit_error_inflation=True,
        emit_breaks=True,
        breaks_half_life=h,
    )
    out = _run(df, spec)
    v = (out["resid_y"] / out["error_inflation_y"]).to_numpy()
    w = df["w"].to_numpy()
    ready = out["settled_frac"].to_numpy() >= 0.95
    z = out["studentized_y"].to_numpy().astype(float)
    checked = 0
    for t in range(1, n):
        ok = np.isfinite(v[:t]) & ready[:t]
        om = (w[:t] * 0.5 ** (((t - 1) - np.arange(t)) / h))[ok]
        if om.size == 0 or om.sum() ** 2 / (om**2).sum() < 10:
            assert np.isnan(z[t]), t
            continue
        want = v[t] / np.sqrt((om * v[:t][ok] ** 2).sum() / om.sum())
        np.testing.assert_allclose(z[t], want, rtol=1e-12)
        checked += 1
    assert checked > 900, checked


def test_box_pierce_at_a_memory_of_its_own():
    """`specification_half_life=30` beside a fit at 200: Box and Pierce's
    statistic at Kish's size with each folded row aged at 30, the rows folded
    the fit's (95% settled at 200)."""
    rng = np.random.default_rng(6)
    n = 2_000
    x = rng.normal(size=(n, 2))
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": x @ [0.5, -0.3] + rng.normal(size=n)})
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        emit_specification=True,
        specification_half_life=30.0,
    )
    out = _run(df, spec)
    ready = out["settled_frac"].to_numpy() >= 0.95
    assert 850 < int(np.argmax(ready)) < 880
    resid = out["resid_y"].to_numpy()
    for t in (1_100, 1_500, n - 1):
        want = _ew_box_pierce(resid[: t + 1], 0.5 ** (1 / 30.0), 10, ready[: t + 1])
        assert out["ljung_box_y"][t] == pytest.approx(want, rel=1e-9), t


@pytest.mark.oracle_grid(reason="200 streams at six memories, read by their tails")
def test_a_longer_breaks_memory_finds_an_intercept_break_later():
    """An intercept break of one noise sd, 2,100 rows into the CUSUM's
    readiness, on 200 streams beside a fit at 200: the delay to `|cusum| >
    3` grows with the breaks' memory at every tail -- the 90th and 95th
    percentiles and the worst stream -- from 25 to run once, and no stream
    goes undetected. Measured 2026-10-25 (90th / 95th / worst, rows): 26 /
    29 / 41 at 25, 32 / 33 / 46 at 50, 36 / 40 / 47 at 100, 42 / 48 / 54
    at 200, 50 / 52 / 63 at 400, 261 / 277 / 333 run once."""
    groups, n, at = 200, 3_600, 3_000
    rng = np.random.default_rng(2780)
    x = rng.normal(size=(groups * n, 2))
    level = 0.5 + 1.0 * (np.tile(np.arange(n), groups) >= at)
    y = level + x @ np.array([1.0, -1.0]) + rng.normal(size=groups * n)
    df = pl.DataFrame({"g": np.repeat(np.arange(groups), n), "x0": x[:, 0], "x1": x[:, 1], "y": y})
    tails = []
    for memory in (25.0, 50.0, 100.0, 200.0, 400.0, float("inf")):
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=200.0,
            group="g",
            min_weight=10.0,
            emit_breaks=True,
            breaks_half_life=memory,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        c = out["m"].struct.field("cusum_y").fill_null(0.0).fill_nan(0.0).to_numpy()
        hit = np.abs(c.reshape(groups, n)[:, at:]) > 3.0
        assert hit.any(axis=1).all(), (memory, (~hit.any(axis=1)).sum())
        delay = hit.argmax(axis=1)
        tails.append(np.quantile(delay, [0.9, 0.95, 1.0]))
        print(memory, tails[-1])
    tails = np.array(tails)
    assert (np.diff(tails, axis=0) > 0).all(), tails


@pytest.mark.parametrize("field", ["cusum_y", "cusum_sq_y"])
def test_a_windowed_cusum_is_near_its_size_from_its_first_rows(field):
    """At the default memory (the fit's 200), on 100 clean streams, the
    CUSUM and the CUSUM of squares pass 1.96 on about 5% of rows once their
    sums are steady (`test_a_windowed_cusum_reads_its_own_null`). Divided
    by their null's filled share, `h_fit / (h_fit + h)` and 1/2, from the
    first row, they passed it on 16.5% and 14.6% of the 150 rows after the
    diagnostics' readiness, with spreads of 1.40 and 2.77 (8.1% and 8.3%
    still 300 to 600 rows in): a young sum is mostly its plain variance.
    Read as far as the sums have filled (task 278, D-B5), 6.4% and 7.0%,
    spreads 1.05 and 2.01, and 6.1% and 4.3% 300 to 600 rows in (measured
    2026-10-25). The bound is the steady state's with room: at most 8% of
    the first 150 rows."""
    groups, n = 100, 1_500
    df = _clean(groups, n, 5)
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        group="g",
        min_weight=10.0,
        emit_breaks=True,
    )
    c = _run(df, spec)[field].to_numpy().astype(float).reshape(groups, n)
    first = int(np.argmax(np.isfinite(c[0])))
    assert 860 < first < 900, first
    young = c[:, first : first + 150]
    young = young[np.isfinite(young)]
    assert young.size > 100 * 140
    assert np.mean(np.abs(young) > 1.96) <= 0.08


def _young_cusum(
    z: np.ndarray, ahead: int, h: float, h_fit: float
) -> tuple[np.ndarray, np.ndarray]:
    """The CUSUM and CUSUM of squares rebuilt from the studentized
    residuals `z` of one stream with unit weights, each over its null's
    share as far as its sums have filled. The sums' age `x = κ T` is read
    from `S1 / Σ ω_s = (1 + p)(1 + p²) / 4`, `p = e^(−x/4)`, filled below
    `p = 1.5 / K`; beside an exponential fit at `h_fit` the CUSUM's share
    is `h_fit / (h_fit + h) + (2h / (h_fit + h)) x e^(−(1 + min(1, β)) x)
    F(|1 − β| x) / (1 − e^(−2x))`, `β = h / h_fit`, `F(a) = (1 − e^(−a)) /
    a`; the CUSUM of squares' `1/2 + (q − D) / (1 + q) + D² ln(N / D) / (1
    − q²)`, `q = e^(−x)`, `D` the scale's lead, `N = D + 1 − q`. The scale
    folded `ahead` rows before the first `z`. Written here from the
    derivation, apart from the library's arithmetic."""
    lam = 0.5 ** (1.0 / h)
    slow = np.sqrt(np.sqrt(lam))
    n = z.size
    ok = np.isfinite(z)
    first = int(np.argmax(ok))
    cusum, cusum_sq = np.full(n, np.nan), np.full(n, np.nan)
    s1 = s2 = z1 = z2 = z4 = w4 = v1 = 0.0
    for t in range(first - ahead, n):
        if t > first:
            target = 4.0 * s1 / w4
            p = 1.0
            for _ in range(100):
                nxt = p - (((p + 1.0) * p + 1.0) * p + 1.0 - target) / ((3.0 * p + 2.0) * p + 1.0)
                if not nxt < p:
                    break
                p = nxt
            if target >= 4.0:
                x = 0.0
            elif target <= 1.0 or p <= 1.5 * s2 / s1**2:
                x = np.inf
            else:
                x = -4.0 * np.log(p)
            beta = h / h_fit
            if x == 0.0:
                r = r2 = 1.0
            elif np.isinf(x):
                r, r2 = h_fit / (h_fit + h), 0.5
            else:
                a = abs(1.0 - beta) * x
                f = 1.0 if a == 0.0 else -np.expm1(-a) / a
                big = 2.0 * h / (h_fit + h) * x * np.exp(-(1.0 + min(1.0, beta)) * x) * f
                r = h_fit / (h_fit + h) + big / -np.expm1(-2.0 * x)
                q, m = np.exp(-x), -np.expm1(-x)
                d = min(max(m * v1 / s1 - m, 0.0), q)
                tail = d * d * np.log1p(m / d) / (m * (1.0 + q)) if d > 0.0 else 0.0
                r2 = 0.5 + (q - d) / (1.0 + q) + tail
            cusum[t] = z1 / np.sqrt(r * s2)
            excess = z4 / w4 - 1.0
            # No CUSUM of squares while the fourth moment is at most 1.
            cusum_sq[t] = (z2 - s1) / np.sqrt(r2 * excess * s2) if excess > 0.0 else np.nan
        v1 = v1 * lam + 1.0
        if t >= first:
            zt = z[t]
            s1, s2 = s1 * lam + 1.0, s2 * lam * lam + 1.0
            z1, z2 = z1 * lam + zt, z2 * lam + zt * zt
            z4, w4 = z4 * slow + zt**4, w4 * slow + 1.0
    return cusum, cusum_sq


@pytest.mark.parametrize("memory", [120.0, 400.0])
def test_a_young_windowed_cusum_reads_its_shares_definition(memory):
    """The CUSUM and the CUSUM of squares beside `ewridge` at a half-life
    of 200, rebuilt from `studentized` with the young shares written from
    their derivation (`_young_cusum`), at memories below and above the
    fit's: to 1e-10 relative from the first row after readiness, where the
    shares are near 1, to the last, where they are their steady forms.
    The scale folds the rows from readiness until its weights reach a Kish
    size of 10, the count read off unit weights."""
    rng = np.random.default_rng(11)
    n = 4_000
    x = rng.standard_normal((n, 2))
    df = pl.DataFrame(
        {
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": 0.5 + x @ np.array([1.0, -1.0]) + rng.standard_normal(n),
        }
    )
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=200.0,
        emit_breaks=True,
        breaks_half_life=memory,
    )
    out = _run(df, spec)
    z = out["studentized_y"].to_numpy().astype(float)
    lam = 0.5 ** (1.0 / memory)
    ahead = 1
    while sum(lam**j for j in range(ahead)) ** 2 < 10.0 * sum(lam ** (2 * j) for j in range(ahead)):
        ahead += 1
    cusum, cusum_sq = _young_cusum(z, ahead, memory, 200.0)
    for field, want in (("cusum_y", cusum), ("cusum_sq_y", cusum_sq)):
        got = out[field].to_numpy().astype(float)
        both = np.isfinite(got) & np.isfinite(want)
        assert both.sum() > n - 1_000, field
        assert (np.isfinite(got) == np.isfinite(want)).all(), field
        np.testing.assert_allclose(got[both], want[both], rtol=1e-10, atol=1e-12, err_msg=field)
    # The evidence the young path ran: the first rows' share, read back off
    # the field, is near 1, where the filled one is 0.625 or 0.333.
    first = int(np.argmax(np.isfinite(z)))
    got = out["cusum_y"].to_numpy().astype(float)
    for k in range(1, 6):
        w = lam ** np.arange(k)[::-1]
        z1, s2 = (w * z[first : first + k]).sum(), (w * w).sum()
        assert (z1 / got[first + k]) ** 2 / s2 > 0.99, k


@pytest.mark.oracle_grid(reason="300 streams x 3 seeds at four memories beside seven fits")
@pytest.mark.parametrize(
    "model",
    ["ewridge", "rls", "lasso", "kalman", "quantile", "window", "sgd"],
)
def test_a_windowed_cusum_is_near_its_size_at_every_age(model):
    """On 3 seeds of 300 clean streams of 7,200 rows beside a fit at a
    half-life of 200, at breaks memories of 50, 200, 400 and 800, the worst
    seed's share of rows past 1.96 over the 150 rows after the diagnostics'
    readiness, rows 300 to 600 after it, and 4,700 to 6,300 after it
    (steady). Divided by the filled shares from the first row, the first
    150 rows passed on up to 35% (CUSUM, memory 800) and 18% (CUSUM of
    squares); read as far as the sums have filled (task 278, D-B5), the
    worst seed read 5.2% to 6.1% (CUSUM) and 5.9% to 6.6% (CUSUM of
    squares) there, at most 6.7% 300 to 600 rows in and 4.9% to 6.1%
    steady (measured 2026-10-25; the rest of the excess over 5% early is
    the young scale's: its `z` are read against few rows and spread wider
    than 1). `quantile`'s steady CUSUM at a memory of 800 read 8.4%, the
    spread of 1.10 to 1.25 its indicators' CUSUM has there, which the young
    share does not touch; it is held to 10%. `sgd`'s CUSUM keeps the plain
    sum, which its learning rate leaves conservative (`|cusum| > 3` finds
    fewer of its breaks), and is held only to its CUSUM of squares;
    `quantile` at its median."""
    kw: dict = {
        "lasso": {"penalties": [0.001]},
        "kalman": {"coef_half_life": 200.0},
        "quantile": {"quantile": 0.5},
        "window": {"window_size": 400.0},
    }.get(model, {})
    builder = getattr(po.spec, "ewridge" if model == "window" else model)
    groups, n = 300, 7_200
    segments = {"young": (0, 150), "300-600": (300, 600), "steady": (4_700, 6_300)}
    fields = ["cusum_sq_y"] if model == "sgd" else ["cusum_y", "cusum_sq_y"]
    worst: dict = {}
    for seed in range(3):
        df = _clean(groups, n, 2_780 + seed)
        for memory in (50.0, 200.0, 400.0, 800.0):
            spec = builder(
                "m",
                targets=["y"],
                features=["x0", "x1"],
                half_life=200.0,
                group="g",
                min_weight=10.0,
                emit_breaks=True,
                breaks_half_life=memory,
                **kw,
            )
            out = _run(df, spec)
            for field in fields:
                c = out[field].to_numpy().astype(float).reshape(groups, n)
                first = np.isfinite(c).argmax(axis=1)
                for name, (a, b) in segments.items():
                    v = np.concatenate([c[g, first[g] + a : first[g] + b] for g in range(groups)])
                    v = v[np.isfinite(v)]
                    assert v.size > groups * (b - a) * 0.9, (model, memory, field, name)
                    size = float(np.mean(np.abs(v) > 1.96))
                    key = (memory, field, name)
                    worst[key] = max(worst.get(key, 0.0), size)
    for (memory, field, name), size in worst.items():
        bound = 0.10 if (model, memory, field) == ("quantile", 800.0, "cusum_y") else 0.085
        assert size <= bound, (model, memory, field, name, size)
        assert size >= 0.035, (model, memory, field, name, size)
