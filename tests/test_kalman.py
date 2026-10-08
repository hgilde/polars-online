"""Task 11: Kalman / random-walk-beta (docs/PLAN.md section 4.4)."""

import numpy as np
import polars as pl
import pytest

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


def test_share_p_runs_and_differs_from_per_target():
    df, _ = synthetic(seed=44, n_groups=1, n_rows=300, k=3, n_targets=2, null_frac=0.0)
    kw = dict(targets=["y0", "y1"])
    a = po.ModelBank([_spec(share_p=False, **kw)]).fit_predict(df)
    b = po.ModelBank([_spec(share_p=True, **kw)]).fit_predict(df)
    # One P driven by the mean sigma^2 is not two driven by their own.
    for col in ("pred_y0", "pred_y1"):
        pa, pb = _pred(a, col), _pred(b, col)
        m = np.isfinite(pa) & np.isfinite(pb)
        assert m.sum() > 100 and np.abs(pa[m] - pb[m]).max() > 1e-6, col


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
