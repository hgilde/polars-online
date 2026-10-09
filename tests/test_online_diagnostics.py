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
        ["ljung_box_y", "breusch_pagan_y", "reset_y"],
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
    for f in fields:
        # `lasso` names its slot by the path point.
        (name,) = [c for c in out.columns if c.startswith(f)]
        assert out[name].drop_nulls().len() > 400, (f, name)


@pytest.mark.parametrize("switch", SWITCHES)
def test_holt_takes_it(switch):
    """`holt` has no features: the twin fits compare its level alone."""
    _, fields = SWITCHES[switch]
    df = _frame()
    if switch in LEVERAGE:
        with pytest.raises(ValueError, match="needs a model that reads one row"):
            po.spec.holt("m", targets=["y"], half_life=60.0, **{switch: True})
        return
    out = _run(df, po.spec.holt("m", targets=["y"], half_life=60.0, **{switch: True}))
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


@pytest.mark.parametrize(
    ("build", "extra", "why"),
    [
        (po.spec.kalman, {"coef_half_life": 100.0}, "random walk"),
        (po.spec.lasso, {"lasso_path": [0.01]}, "post-selection"),
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
    """docs/DATA-ISSUES.md's frozen feed: ``x1`` held from row 300 at a
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
    assert (late["reset_y"].drop_nulls() > 5.991).mean() < 0.12
    assert (late["cusum_sq_y"].drop_nulls().abs() > 1.96).mean() < 0.06
    first = out.filter(pl.col("g") == 0)
    settled = first["settled_frac"].to_numpy()
    folded = first["kurtosis_y"].is_not_null().arg_true()
    assert folded.len() > 0
    # The first row that reads a value reads the row before it, which was
    # past 95% settled.
    assert settled[int(folded[0]) - 1] >= 0.95


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
            ("reset_y", 5.991),
            ("calibration_wald_y", 5.991),
        ):
            rate = (out[field].drop_nulls() > crit).mean()
            assert rate < (0.10 if half_life < float("inf") else 0.20), (half_life, field, rate)
        if half_life < float("inf"):
            assert (out["break_wald_y"].drop_nulls() > 7.815).mean() < 0.10


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
