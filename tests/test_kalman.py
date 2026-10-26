"""Task 11: Kalman / random-walk-beta (docs/PLAN.md section 4.4)."""

import math
import warnings

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

import polars_online as po
from data import synthetic

TIER = "essential"


def _spec(**kw):
    defaults = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        # `q` in place of the half-life, never beside it (review 2026-10-06,
        # PC6).
        **({} if "q" in kw else {"coef_half_life": 100.0}),
        half_life=500.0,
        min_weight=20.0,
    )
    defaults.update(kw)
    return po.spec.kalman("m", **defaults)


def _pred(out, col="pred_y0"):
    return out["m"].struct.field(col).to_numpy().astype(float)


def test_tracks_time_varying_beta_better_than_a_pinned_filter():
    # The synthetic generator's beta is a random walk, which is exactly the
    # Kalman model's assumption; a responsive filter must beat a pinned one.
    df, _ = synthetic(seed=41, n_groups=1, n_rows=1500, k=3, null_frac=0.0, beta_sigma=0.03)
    fast = po.ModelBank([_spec(coef_half_life=50.0)]).fit_predict(df)
    pinned = po.ModelBank([_spec(coef_half_life=float("inf"))]).fit_predict(df)
    y = df["y0"].to_numpy()
    for out, name in ((fast, "fast"), (pinned, "pinned")):
        assert np.isfinite(_pred(out)).sum() > 1000, name
    e_fast = np.nanmean((y - _pred(fast)) ** 2)
    e_pin = np.nanmean((y - _pred(pinned)) ** 2)
    assert e_fast < e_pin, f"fast {e_fast} vs pinned {e_pin}"


def _differ(a, b) -> float:
    pa, pb = _pred(a), _pred(b)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.sum() > 100, "too few rows scored to compare"
    return float(np.abs(pa[m] - pb[m]).max())


def test_per_factor_halflife_and_pinning():
    """Each slot takes its own half-life: the per-factor filter is neither
    the scalar one nor the same one with x1's half-life changed. It once
    checked only that a prediction came out (review 2026-10-05, TB6)."""
    df, _ = synthetic(seed=42, n_groups=1, n_rows=400, k=3, null_frac=0.0)
    # intercept pinned, x0 slow, x1 fast, x2 pinned
    spec = _spec(coef_half_life=[float("inf"), 500.0, 30.0, float("inf")])
    out = po.ModelBank([spec]).fit_predict(df)
    scalar = po.ModelBank([_spec(coef_half_life=100.0)]).fit_predict(df)
    x1_slow = _spec(coef_half_life=[float("inf"), 500.0, 300.0, float("inf")])
    assert _differ(out, scalar) > 1e-6
    assert _differ(out, po.ModelBank([x1_slow]).fit_predict(df)) > 1e-6


def test_explicit_q_overrides_halflife():
    df, _ = synthetic(seed=43, n_groups=1, n_rows=200, k=3, null_frac=0.0)
    a = po.ModelBank([_spec(q=[0.0, 0.0, 0.0, 0.0])]).fit_predict(df)
    b = po.ModelBank([_spec(q=[0.01, 0.01, 0.01, 0.01])]).fit_predict(df)
    # zero process noise converges; nonzero keeps moving, so they differ
    pa, pb = _pred(a), _pred(b)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert np.abs(pa[m] - pb[m]).max() > 1e-6


def test_share_p_is_each_targets_own_filter_when_every_target_is_on_every_row():
    """``P`` is in units of the noise, so its recursion reads no target and
    no noise (task 278): with every target on every row, one shared ``P`` is
    each target's own, and the predictions are the per-target filters' to
    the bit. With a target null on some rows the shared ``P`` takes rows
    that target never saw, and its predictions part. (Before, one ``P`` driven by
    the mean ``sigma^2`` parted from two driven by their own on every row.)"""
    df, _ = synthetic(seed=44, n_groups=1, n_rows=300, k=3, n_targets=2, null_frac=0.0)
    kw = dict(targets=["y0", "y1"])
    a = po.ModelBank([_spec(share_p=False, **kw)]).fit_predict(df)
    b = po.ModelBank([_spec(share_p=True, **kw)]).fit_predict(df)
    for col in ("pred_y0", "pred_y1"):
        pa, pb = _pred(a, col), _pred(b, col)
        assert np.isfinite(pa).sum() > 100, col
        np.testing.assert_array_equal(pa, pb, err_msg=col)
    y1 = df["y1"].to_numpy().copy()
    y1[::4] = np.nan
    gappy = df.with_columns(pl.Series("y1", y1))
    a = po.ModelBank([_spec(share_p=False, **kw)]).fit_predict(gappy)
    b = po.ModelBank([_spec(share_p=True, **kw)]).fit_predict(gappy)
    # The target on every row is its own filter still; the gappy one parts.
    np.testing.assert_array_equal(_pred(a, "pred_y0"), _pred(b, "pred_y0"))
    pa, pb = _pred(a, "pred_y1"), _pred(b, "pred_y1")
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.sum() > 100 and np.abs(pa[m] - pb[m]).max() > 1e-6


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(9)
    n = 3000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )
    out = po.ModelBank([_spec(coef_half_life=200.0, half_life=500.0)]).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
    assert abs(ic) < 0.06, f"IC {ic}: predictions are not out-of-sample"


def test_chunk_invariance():
    df, _ = synthetic(seed=45, n_groups=2, n_rows=200, k=3, null_frac=0.0)
    spec = _spec(group="group", clock="t", gap_cap=50.0, weight="w")
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 30)) for i in range(0, df.height, 30)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


@pytest.mark.parametrize("p0", [None, 0.25])
@pytest.mark.parametrize("share_p", [False, True])
def test_the_targets_units_do_not_matter(share_p, p0):
    """Review 2026-10-05, CC4. Scaling every target by ``c`` scales every
    prediction by ``c``, at the default ``p0`` and at another. Powers of two
    keep each operation exact, so the comparison is to the bit, nulls in the
    same places. Before a target's first residual its noise was the literal
    1.0, in the target's units, and its prior variance was ``p0`` in the
    same units, so the warm-up's gains, and every prediction after them,
    moved with them. The prior is now ``p0`` times the first noise
    estimate."""
    df, _ = synthetic(seed=46, n_groups=2, n_rows=200, k=3, n_targets=2, null_frac=0.05)

    def preds(c: float) -> dict[str, np.ndarray]:
        spec = _spec(
            targets=["y0", "y1"],
            group="group",
            clock="t",
            gap_cap=50.0,
            weight="w",
            p0=p0,
            share_p=share_p,
        )
        out = po.ModelBank([spec]).fit_predict(df.with_columns(pl.col("y0", "y1") * c))
        return {t: _pred(out, f"pred_{t}") for t in ("y0", "y1")}

    base = preds(1.0)
    assert all(np.isfinite(p).sum() > 250 for p in base.values())
    for c in (2.0**20, 2.0**-20):
        for t, got in preds(c).items():
            np.testing.assert_array_equal(got, base[t] * c, err_msg=f"c = {c}, {t}")


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="coef_half_life"):
        _spec(coef_half_life=[1.0, 2.0])  # wrong length for k=3 + intercept
    with pytest.raises(ValueError, match="obs_var"):
        _spec(obs_var=0.0)


def _gappy(n: int = 400, seed: int = 0) -> pl.DataFrame:
    """Two features and a target on them, clock steps of 1 with a step of 9
    before every 17th row."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = 1 + x @ [2.0, -1.0] + 0.3 * rng.standard_normal(n)
    t = np.cumsum(np.where(np.arange(n) % 17 == 0, 9.0, 1.0))
    return pl.DataFrame({"t": t, "x0": x[:, 0], "x1": x[:, 1], "y": y, "w": 1.0})


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda c: po.spec.kalman("m", coef_half_life=20.0, **c), id="kalman"),
        pytest.param(
            lambda c: po.spec.kalman("m", coef_half_life=20.0, standardize=False, **c),
            id="kalman unstandardized",
        ),
        pytest.param(lambda c: po.spec.rls("m", **c), id="rls"),
        pytest.param(lambda c: po.spec.ewridge("m", **c), id="ewridge"),
        pytest.param(lambda c: po.spec.sgd("m", **c), id="sgd"),
    ],
)
def test_a_zero_weight_row_inside_a_gap_moves_no_prediction(build):
    """Hard rule 9 through the bank: a row of weight 0 advances the clock and
    learns nothing, so one inserted half-way through every clock gap leaves
    every real row's prediction where it was, to rounding. `kalman` charged
    its process noise per row, ``Q d**2``, which is not additive over a split
    gap, and moved them by 0.197 at ``coef_half_life = 20``; `rls`,
    `ewridge` and `sgd` by rounding (the review of 2026-10-08; docs/PLAN.md
    task 211). Now `kalman` charges ``Q D**2`` once, on the next row that
    observes the target, for the whole clock since the last."""
    base = _gappy().with_columns(real=pl.lit(True))
    mid = base.with_columns(t=pl.col("t") - 0.5, w=pl.lit(0.0), real=pl.lit(False))
    doubled = pl.concat([base, mid.slice(1)]).sort("t", maintain_order=True)
    common = dict(
        targets=["y"], features=["x0", "x1"], clock="t", weight="w", half_life=50.0, gap_cap=1e9
    )
    spec = build(common)
    a = po.ModelBank([spec]).fit_predict(base)["m"].struct.field("pred_y").to_numpy()
    out = po.ModelBank([spec]).fit_predict(doubled).filter(pl.col("real"))
    b = out["m"].struct.field("pred_y").to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 350
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
    np.testing.assert_allclose(b[ok], a[ok], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("coef_half_life", [20.0, float("inf")])
def test_the_doubled_stream_is_the_native_embargo_with_gaps(coef_half_life):
    """`po.stream.embargo`'s doubled stream -- each row predicted at its clock
    at weight 0 and learned ``delay`` later -- agrees with the native
    ``embargo`` to rounding on a stream with clock gaps and no ``gap_cap``:
    its predict rows are rows that observe nothing, and charge no process
    noise of their own (docs/PLAN.md task 211). It parted by 0.28 at
    ``coef_half_life = 20`` (the review's probe). With a ``gap_cap``, the
    doubled stream's sub-gaps escape a cap the gap whole would meet: a limit
    of the oracle, for every model."""
    rng = np.random.default_rng(11)
    n = 300
    x0 = rng.standard_normal(n)
    x1 = 3 * rng.standard_normal(n) + 2
    y = 1.5 * x0 - 0.75 * x1 + 0.25 + 0.3 * rng.standard_normal(n)
    t = np.cumsum(np.where(np.arange(n) % 17 == 0, 9.0, 1.0))
    df = pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y, "w": np.ones(n)})
    common = dict(
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1e9,
        half_life=25.0,
        weight="w",
        min_weight=4.0,
        coef_half_life=coef_half_life,
    )
    native = po.ModelBank([po.spec.kalman("m", embargo=3.0, **common)]).fit_predict(df)
    nat = native["m"].struct.field("pred_y").to_numpy()
    out = po.ModelBank([po.spec.kalman("m", **common)]).fit_predict(
        po.stream.embargo(df, clock="t", delay=3.0, weight="w")
    )
    dbl = out.filter(pl.col(po.stream.ROLE) == "predict")["m"].struct.field("pred_y").to_numpy()
    np.testing.assert_array_equal(np.isfinite(nat), np.isfinite(dbl))
    ok = np.isfinite(nat)
    assert ok.sum() > 250
    np.testing.assert_allclose(dbl[ok], nat[ok], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("half_life", "floor"), [(50.0, 0.93), (float("inf"), 0.9)])
def test_a_short_history_fits_from_its_first_rows(half_life, floor):
    """Task 74's short histories: 500 groups of 200 rows, 20 features, a
    slope per group. Each coefficient's prior waits for its feature's scale
    and rests on three rows' squared innovations, and the state follows the
    moments from the first row (docs/PLAN.md task 211): R² on rows 25 to 50
    went from 0.722 to 0.950 at a half-life of 50 (0.689 to 0.925 without
    decay). The prior was sized on row 0, from one innovation against
    features read raw, and a 22-row warm-up then read the state in the
    coordinates of whatever the moments had become."""
    n_groups, rows, k = 500, 200, 20
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_groups * rows, k))
    beta = np.repeat(rng.standard_normal((n_groups, k)) / np.sqrt(k), rows, axis=0)
    y = (x * beta).sum(axis=1) + 0.1 * rng.standard_normal(len(x))
    feats = [f"x{i}" for i in range(k)]
    df = pl.DataFrame({**{f: x[:, i] for i, f in enumerate(feats)}, "y": y}).with_columns(
        g=pl.int_range(pl.len()) // rows
    )
    spec = po.spec.kalman(
        "m",
        targets=["y"],
        features=feats,
        group="g",
        coef_half_life=50.0,
        half_life=half_life,
        min_weight=25.0,
    )
    pred = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
    pos = np.arange(len(y)) % rows
    sel = (pos >= 25) & (pos < 50) & np.isfinite(pred)
    # Under a half-life of 50 the decayed weight reaches 25 on row 31.
    assert sel.sum() > 0.7 * 25 * n_groups
    r2 = 1.0 - np.sum((y[sel] - pred[sel]) ** 2) / np.sum((y[sel] - y[sel].mean()) ** 2)
    assert r2 > floor, r2


def test_the_first_row_takes_the_level_and_no_prediction_is_zero():
    """Standardized without ``obs_var`` a target's prior was sized from
    its first three observed rows, and a row past ``min_weight`` before then
    -- a weight of 3 a row meets a floor of 2 after one -- was predicted as
    ``0.0`` whatever the target's level, and that 0 seeded the residual
    variance with the level squared (task 270, review round 9 B-1). Since
    task 278 the intercept's prior is diffuse and needs no noise: the first
    row sizes it, taking the level whole, so the second row is predicted at
    the first row's target (the slope waits for its feature's scale), no
    prediction is 0.0 on a target at 100, and the residual variance starts
    at the first prediction: ``sigma`` on the row after it is that row's
    residual."""
    rng = np.random.default_rng(3)
    n = 300
    x = rng.standard_normal(n)
    y = 100.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
    frame = pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y, "w": np.full(n, 3.0)})
    spec = po.spec.kalman(
        "m",
        targets=["y"],
        features=["x"],
        clock="t",
        gap_cap=10.0,
        weight="w",
        half_life=200.0,
        coef_half_life=100.0,
        emit_sigma=True,
    )
    out = po.ModelBank([spec]).fit_predict(frame)["m"].struct.unnest()
    pred = out["pred_y"].to_numpy().astype(float)
    reason = out["withheld_reason"].cast(pl.String).to_list()
    assert reason[:2] == ["below_min_weight", None]
    assert "below_warm_rows" not in reason
    assert np.isnan(pred[:1]).all() and np.isfinite(pred[1:]).all()
    assert pred[1] == y[0]
    assert not (pred == 0.0).any()
    sigma = out["sigma_y"].to_numpy().astype(float)
    assert np.isnan(sigma[:2]).all()
    assert sigma[2] == pytest.approx(abs(y[1] - pred[1]), rel=1e-12)


# ---- the parameters' spellings and limits (task 278, batch 27) ----


def _relations_frame(seed: int = 278, n: int = 2000) -> pl.DataFrame:
    """Three features of unlike scales and means, a slope that changes half
    way, a second target null where ``x0`` is high, uneven weights with a
    zero on the first row and every 37th, and a clock that is the row
    number."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = rng.standard_normal((n, 3)) * [1.0, 2.0, 0.5] + [0.5, -1.0, 3.0]
    beta = np.array([2.0, -1.0, 0.5])
    y = 1.0 + x @ beta + np.where(t < n / 2, 0.0, 1.5) * x[:, 0] + rng.standard_normal(n)
    z = 0.5 - x @ beta[::-1] + rng.standard_normal(n)
    z[x[:, 0] > 1.2] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[::37] = 0.0
    return pl.DataFrame(
        {"t": t, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y, "w": w}
        | {"z": pl.Series(z).fill_nan(None)}
    )


_RF = ["x0", "x1", "x2"]


def _run(df: pl.DataFrame, **kw) -> tuple[pl.DataFrame, po.ModelBank]:
    base = dict(
        targets=["y"],
        features=_RF,
        clock="t",
        gap_cap=1e9,
        weight="w",
        half_life=200.0,
        min_weight=10.0,
        coef_every=0,
    )
    bank = po.ModelBank([po.spec.kalman("m", **{**base, **kw})], threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", po.ReadinessWarning)
        return bank.fit_predict(df).unnest("m"), bank


def _last_coef(bank: po.ModelBank) -> np.ndarray:
    return bank.coef("m")["coef"].to_numpy()


@pytest.mark.parametrize("standardize", [True, False])
def test_coef_half_life_is_q_spelled_against_a_fixed_noise(standardize):
    """With ``obs_var=v`` the covariance is absolute, and ``coef_half_life
    = h`` adds ``v (ln 2 D / h)^2`` a row: the ``q`` of ``v (ln 2 / h)^2`` on
    every slot. The two spellings are one filter, to the bit. With the noise
    learned they are not: ``q`` is then measured against the noise estimate,
    which the half-life never reads (0.12 apart here, measured 2026-10-26)."""
    df = _relations_frame()
    h, v = 100.0, 2.0
    by_h, _ = _run(df, coef_half_life=h, obs_var=v, standardize=standardize)
    by_q, _ = _run(df, q=[v * (math.log(2) / h) ** 2] * 4, obs_var=v, standardize=standardize)
    assert by_h["pred_y"].is_not_null().sum() > 1900
    assert by_h.equals(by_q, null_equal=True)
    learned_h, _ = _run(df, coef_half_life=h, standardize=standardize)
    learned_q, _ = _run(df, q=[(math.log(2) / h) ** 2] * 4, standardize=standardize)
    assert not learned_h["pred_y"].equals(learned_q["pred_y"])


@pytest.mark.parametrize(
    "extra",
    [{}, {"revert_half_life": 300.0}, {"targets": ["y", "z"], "share_p": True}],
    ids=["plain", "revert", "share_p"],
)
def test_a_scalar_coef_half_life_is_the_list_that_spells_it(extra):
    """A scalar is every slot's half-life, the intercept's included: the
    same filter as the list of four, to the bit, beside a reversion and on
    a shared ``P`` over two targets with gaps."""
    df = _relations_frame()
    scalar, _ = _run(df, coef_half_life=100.0, **extra)
    listed, _ = _run(df, coef_half_life=[100.0] * 4, **extra)
    assert scalar.equals(listed, null_equal=True)
    # A list that differs in one slot is another filter.
    other, _ = _run(df, coef_half_life=[100.0, 100.0, 30.0, 100.0], **extra)
    assert not other["pred_y"].equals(scalar["pred_y"])


@pytest.mark.parametrize("p0", [0.5, 4.0])
def test_with_no_drift_and_a_fixed_noise_the_filter_is_rls(p0):
    """``q = 0``, ``obs_var`` fixed and ``standardize=False``: the filter is
    Bayesian linear regression from ``P_0 = p0 obs_var I``, the intercept
    included, so its mean is the ridge fit with penalty ``1 / p0`` on every
    coefficient -- ``rls(delta=1/p0)`` without decay. The two compute it
    apart (a covariance update against a solve), so they agree to rounding:
    5.2e-14 in a prediction and 2.5e-14 in a coefficient, measured
    2026-10-26. The noise only scales ``P``, so another ``obs_var`` moves no
    prediction beyond rounding either (1.4e-14). Both at ``min_weight=0``,
    so every row from the second learned one is compared."""
    df = _relations_frame()
    kw = dict(half_life=float("inf"), min_weight=0.0)
    filt, _ = _run(df, q=[0.0] * 4, obs_var=1.0, p0=p0, standardize=False, **kw)
    rls = (
        po.ModelBank(
            [
                po.spec.rls(
                    "m",
                    targets=["y"],
                    features=_RF,
                    clock="t",
                    gap_cap=1e9,
                    weight="w",
                    coef_every=0,
                    delta=1.0 / p0,
                    **kw,
                )
            ],
            threads=1,
        )
        .fit_predict(df)
        .unnest("m")
    )
    a, b = filt["pred_y"].to_numpy().astype(float), rls["pred_y"].to_numpy().astype(float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    assert np.isfinite(a).sum() >= df.height - 2
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)
    assert filt["weight_sum"].equals(rls["weight_sum"])
    ca = np.array(filt["coef"].to_list(), dtype=float)
    cb = np.array(rls["coef"].to_list(), dtype=float)
    np.testing.assert_allclose(ca, cb, rtol=0.0, atol=1e-12)
    noisier, _ = _run(df, q=[0.0] * 4, obs_var=7.0, p0=p0, standardize=False, **kw)
    np.testing.assert_allclose(noisier["pred_y"].to_numpy().astype(float), a, atol=1e-12)


@pytest.mark.parametrize("noise", [{"obs_var": 1.0}, {}], ids=["fixed noise", "learned noise"])
def test_with_no_drift_a_vague_prior_is_least_squares(noise):
    """``q = 0``, ``standardize=False``: the prior is a ridge of ``1 / p0``
    (in units of the noise), so as ``p0`` grows the filter's last fit goes
    to weighted least squares on every row -- statsmodels' ``WLS`` -- at the
    rate ``1 / p0``. Measured 2026-10-26: 1.8e-6 off at ``p0 = 1e4`` and
    1.8e-8 at ``1e6`` with the noise fixed, 4.0e-7 and 4.2e-9 learned. A
    covariance-form filter loses digits to a huge prior (4.2e-6 off at
    ``p0 = 1e12``), so the limit is read where the ridge still dominates
    the rounding."""
    df = _relations_frame()
    x = sm.add_constant(df.select(_RF).to_numpy())
    wls = sm.WLS(df["y"].to_numpy(), x, weights=df["w"].to_numpy()).fit().params
    errs = {}
    for p0 in (1e2, 1e4, 1e6):
        _, bank = _run(df, q=[0.0] * 4, p0=p0, standardize=False, half_life=float("inf"), **noise)
        errs[p0] = float(np.max(np.abs(_last_coef(bank) - wls)))
    print({k: f"{v:.3g}" for k, v in errs.items()})
    assert errs[1e6] < 1e-6
    # The rate: each factor of 100 in p0 takes a factor of about 100 off.
    for lo, hi in ((1e2, 1e4), (1e4, 1e6)):
        assert 50.0 < errs[lo] / errs[hi] < 200.0, errs


def test_with_no_drift_a_tight_prior_holds_the_coefficients():
    """``q = 0``, ``standardize=False``, ``p0 -> 0``. With ``obs_var`` the
    prior covers the intercept too, so every coefficient is held at 0, each
    in proportion to ``p0``. With the noise learned the intercept's prior is
    diffuse: the slopes are held at 0 and the intercept is least squares on
    a constant, the weighted mean of ``y``. Measured 2026-10-26 at ``p0 =
    1e-8`` and ``1e-12``: coefficients at most 2.8e-4 and 2.8e-8 (fixed
    noise); slopes 7.6e-5 and 7.6e-9, the intercept 9.9e-5 and 1e-8 from the
    mean (learned)."""
    df = _relations_frame()
    ybar = float(np.average(df["y"].to_numpy(), weights=df["w"].to_numpy()))
    kw = dict(q=[0.0] * 4, standardize=False, half_life=float("inf"))
    fixed = {}
    learned = {}
    for p0 in (1e-8, 1e-12):
        _, bank = _run(df, p0=p0, obs_var=1.0, **kw)
        fixed[p0] = float(np.max(np.abs(_last_coef(bank))))
        _, bank = _run(df, p0=p0, **kw)
        c = _last_coef(bank)
        learned[p0] = (float(np.max(np.abs(c[1:]))), abs(float(c[0]) - ybar))
    print("fixed:", fixed, "learned:", learned)
    assert fixed[1e-12] < 1e-6 and 5e3 < fixed[1e-8] / fixed[1e-12] < 2e4
    (s8, i8), (s12, i12) = learned[1e-8], learned[1e-12]
    assert s12 < 1e-6 and 5e3 < s8 / s12 < 2e4
    assert i12 < 1e-6 and 5e3 < i8 / i12 < 2e4
