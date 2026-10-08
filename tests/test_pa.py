"""E17: passive-aggressive regression."""

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"


def _spec(**kw):
    d = dict(
        targets=["y0"],
        features=["x0", "x1"],
        half_life=float("inf"),
        min_weight=10.0,
    )
    d.update(kw)
    return po.spec.pa("m", **d)


def _fit(df, **kw):
    out = po.ModelBank([_spec(coef_every=1, **kw)]).fit_predict(df)
    return np.array(out["m"].struct.field("coef").to_list()[-1], dtype=float), out


def _linear(n=5000, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    return pl.DataFrame(
        {"x0": x0, "x1": x1, "y0": 1.5 * x0 - 0.5 * x1 + 0.25 + noise * rng.standard_normal(n)}
    )


@pytest.mark.parametrize("mode", ["pa", "pa1", "pa2"])
def test_recovers_a_noiseless_relationship(mode):
    c, _ = _fit(_linear(), mode=mode, eps=0.01, c=1.0)
    assert c[0] == pytest.approx(0.25, abs=0.05), f"{mode}: {c}"
    assert c[1] == pytest.approx(1.5, abs=0.05), f"{mode}: {c}"
    assert c[2] == pytest.approx(-0.5, abs=0.05), f"{mode}: {c}"


@pytest.mark.parametrize("mode", ["pa1", "pa2"])
def test_an_infinite_c_is_the_unbounded_mode(mode):
    """``c = inf`` caps nothing: ``min(l/s, inf) = l/s`` and ``l/(s + 0.5/inf)
    = l/s``, so either bounded mode is mode ``"pa"`` to the bit (review
    2026-09-12, S27). The builder refused it."""
    df = _linear(noise=0.3)
    _, bounded = _fit(df, mode=mode, c=float("inf"))
    _, unbounded = _fit(df, mode="pa")
    assert bounded.equals(unbounded, null_equal=True)


def test_no_learning_rate_is_needed():
    # The point of PA: it reaches the answer with no rate to tune, where SGD at
    # a badly chosen rate does not.
    df = _linear(n=3000)
    pa_c, _ = _fit(df, eps=0.01)
    sgd_out = po.ModelBank(
        [
            po.spec.sgd(
                "m",
                targets=["y0"],
                features=["x0", "x1"],
                learning_rate=1e-4,
                half_life=float("inf"),
                min_weight=10.0,
                coef_every=1,
            )
        ]
    ).fit_predict(df)
    sgd_c = np.array(sgd_out["m"].struct.field("coef").to_list()[-1], dtype=float)
    assert abs(pa_c[1] - 1.5) < abs(sgd_c[1] - 1.5)


def test_bounded_variants_resist_outliers():
    """Measured over the stream, not at the last row.

    Plain PA satisfies each row's constraint *exactly*, so after an outlier the
    next clean row pulls it straight back and the final coefficient looks fine.
    The cost is paid in between: the predictions it makes right after each
    outlier are wild. Averaging the out-of-sample error over the whole stream is
    what shows the difference.
    """
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.standard_normal(n)
    clean = 2.0 * x
    y = clean.copy()
    bad = rng.random(n) < 0.04
    y[bad] = 500.0 * rng.standard_normal(bad.sum())
    df = pl.DataFrame({"x0": x, "x1": np.zeros(n), "y0": y})

    def mean_abs_err(**kw):
        _, out = _fit(df, eps=0.01, **kw)
        p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        m = np.isfinite(p) & ~bad
        return float(np.mean(np.abs(p[m] - clean[m])))

    capped = mean_abs_err(mode="pa1", c=0.05)
    unbounded = mean_abs_err(mode="pa")
    assert capped < unbounded, f"pa1 {capped} should beat pa {unbounded}"
    # and the capped variant is genuinely close to the clean signal
    assert capped < 0.5


def test_wide_tube_is_passive():
    """Once the target has a spread to draw it in, a tube of a million
    spreads holds every row and nothing moves. Before that there is no tube:
    the target's spread reads `y` alone, whatever the fit and `min_weight`,
    so the first two rows teach and the third is inside (docs/PLAN.md task
    202)."""
    df = _linear(n=500)
    # Raw features, so a coefficient moves only where the fit does: under
    # `standardize` it is read through the scaler as it stands while the
    # scaler warms up, over the first 22 rows (docs/PLAN.md task 206).
    c, out = _fit(df, eps=1e6, standardize=False)
    coef = out["m"].struct.field("coef").to_list()
    assert coef[1] == list(c), "nothing should move inside a huge tube"
    assert coef[0] != coef[1], "the second row, before the spread, taught"
    assert np.abs(c).max() > 0.0, "the rows before the spread taught"


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(5)
    n = 5000
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y0": rng.standard_normal(n),
        }
    )
    _, out = _fit(df, mode="pa1", c=0.05)
    p = out["m"].struct.field("pred_y0").to_numpy().astype(float)
    m = np.isfinite(p)
    assert abs(np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]) < 0.06


def test_chunk_invariance():
    df = _linear(n=400, seed=9, noise=0.1)
    spec = _spec()
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)])
        .select("m")
        .unnest("m")
    )
    keep = [c for c in one.columns if not c.startswith("coef")]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_save_load(tmp_path):
    df = _linear(n=400, seed=10, noise=0.1)
    spec = _spec()
    a = po.ModelBank([spec])
    a.fit_predict(df.slice(0, 200))
    p = tmp_path / "pa.state"
    a.save(p)
    b = po.ModelBank.load(p, specs=[spec])
    rest = df.slice(200, 200)
    assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="unknown pa mode"):
        _spec(mode="pa3")
    with pytest.raises(ValueError, match="pa c must be > 0"):
        _spec(c=0.0)
    with pytest.raises(ValueError, match="pa eps must be finite and >= 0"):
        _spec(eps=-1.0)


def _cc4(scale_y=1.0, scale_x=1.0, n=3000):
    """Review round 4's CC4 stream: `y = 0.5 + 2x + 0.3·noise`, the target
    and the feature each scaled."""
    rng = np.random.default_rng(2)
    x, noise = rng.normal(size=n), rng.normal(size=n)
    return pl.DataFrame({"x": scale_x * x, "y": scale_y * (0.5 + 2.0 * x + 0.3 * noise)})


def _oos(df, **kw):
    spec = po.spec.pa("p", targets=["y"], features=["x"], half_life=1e9, min_weight=5.0, **kw)
    p = po.ModelBank([spec]).fit_predict(df)["p"].struct.field("pred_y").to_numpy()
    y = df["y"].to_numpy()
    ok = np.isfinite(p)
    return 1.0 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2), p


def test_a_target_in_hundredths_fits_as_the_unscaled_one_does():
    """docs/PLAN.md task 195 (U1; review round 4, CC4) and task 202: `eps` is
    in units of the target's own EW standard deviation. In the target's units,
    at the defaults, a target scaled by 0.01 sat inside the tube on every row:
    passive for ever, every prediction 0.0 and R² -0.051, against +0.961
    unscaled."""
    unscaled, _ = _oos(_cc4())
    scaled, p = _oos(_cc4(scale_y=0.01))
    assert unscaled > 0.95
    assert scaled > 0.95, scaled
    assert abs(scaled - unscaled) < 0.01
    assert np.sum(p == 0.0) == 0


def _r2_from(p, y, start):
    p, y = p[start:], y[start:]
    ok = np.isfinite(p)
    return 1.0 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


@pytest.mark.parametrize("half_life", [1e9, 500.0])
def test_a_target_far_from_zero_is_learned_with_or_without_decay(half_life):
    """docs/PLAN.md task 202: at the defaults, a target at a level of 1,000 in
    a spread of about 2 fits rows 10,000 to 20,000 as the same target at 0
    does, with or without decay. The fit starts from zero coefficients, so its
    first residuals are the whole level. A band in units of the residual's
    spread, learned from them, was about 100 wide and held every later row;
    without decay nothing narrowed it: R² -52 at a half-life of 1e9, +0.954 at
    500 (task 195's report). The target's own spread is about 2 whatever the
    fit."""
    r2 = {}
    for level in (0.0, 1000.0):
        df = _cc4(n=20_000).with_columns(pl.col("y") + level)
        spec = po.spec.pa("p", targets=["y"], features=["x"], half_life=half_life)
        p = po.ModelBank([spec]).fit_predict(df)["p"].struct.field("pred_y").to_numpy()
        r2[level] = _r2_from(p, df["y"].to_numpy(), 10_000)
    assert r2[1000.0] > 0.9, r2
    assert abs(r2[1000.0] - r2[0.0]) < 0.01, r2


@pytest.mark.parametrize(
    "kind",
    [
        dict(mode="pa", eps=0.1),
        dict(loss="epsilon_insensitive", eps=0.1, schedule="inv_scaling"),
    ],
    ids=["pa", "sgd"],
)
def test_the_band_scales_with_the_target(kind):
    """docs/PLAN.md task 202: the band is in units of the target's own spread,
    which scales with the target, so a target at a level of 1,000 scaled by
    2**-10 or 2**10 fits as the unscaled one does, scaled by it, to the bit:
    `pa` under the unbounded step (a finite `c` caps the step in the target's
    units), `sgd` with its rate scaled too (a sign-valued gradient's rate is
    in the target's units) and its clip lifted."""
    df = _cc4(n=2000).with_columns(pl.col("y") + 1000.0)
    preds = {}
    for c in (1.0, 2.0**-10, 2.0**10):
        scaled = df.with_columns(pl.col("y") * c)
        common = dict(targets=["y"], features=["x"], half_life=200.0)
        if "loss" in kind:
            spec = po.spec.sgd(
                "m", learning_rate=5.0 * c, clip_gradient=float("inf"), **kind, **common
            )
        else:
            spec = po.spec.pa("m", **kind, **common)
        out = po.ModelBank([spec]).fit_predict(scaled)["m"].struct.field("pred_y")
        preds[c] = out.to_numpy() / c
    for c in (2.0**-10, 2.0**10):
        np.testing.assert_array_equal(preds[c], preds[1.0])


@pytest.mark.parametrize(
    "kind",
    [
        dict(),
        dict(loss="epsilon_insensitive", schedule="inv_scaling", learning_rate=0.5),
    ],
    ids=["pa", "sgd"],
)
def test_a_well_predicted_target_reaches_its_slope_at_the_default_band(kind):
    """docs/PLAN.md task 203: the band does not shrink as the fit improves,
    so a fit stops wherever every residual is inside it. Here `y = 2x` plus
    noise of up to 0.01, with `x` in [-1, 1]: the target's spread is about
    1.155, and a fit with every residual inside `eps` of it is off by
    `|b0| + |b1 - 2| <= 1.155 eps - 0.01` at most. At the default of 0.01
    that is 0.0016, and the fit ends within 0.005 (the scaler still moves
    the coefficients a little). At the old default of 0.1 it is 0.105, and
    the fits stopped 0.085 (`pa`) and 0.077 (`sgd`) from the truth. `sgd`
    runs under `inv_scaling`, as its docstring advises for this loss."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, 20_000)
    df = pl.DataFrame({"x": x, "y": 2.0 * x + 0.01 * rng.uniform(-1.0, 1.0, 20_000)})
    common = dict(targets=["y"], features=["x"], half_life=float("inf"))
    spec = po.spec.sgd("m", **kind, **common) if "loss" in kind else po.spec.pa("m", **common)
    coef = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("coef").drop_nulls()[-1]
    b0, b1 = coef.to_list()
    assert abs(b0) + abs(b1 - 2.0) < 0.005, (b0, b1)


def _excess_over_the_noise(r2, **kw):
    """`y = 2x + e`, `x ~ N(0, 1)`, `e` of the variance that makes R² `r2`;
    `pa` without decay; the out-of-sample MSE over rows 10k-30k above the
    noise's, as a share of it (task 207's sweep, one seed)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30_000)
    e = np.sqrt(4.0 * (1.0 - r2) / r2) * rng.standard_normal(30_000)
    df = pl.DataFrame({"x": x, "y": 2.0 * x + e})
    spec = po.spec.pa("m", targets=["y"], features=["x"], half_life=float("inf"), **kw)
    p = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
    p = p.astype(float)[10_000:]
    noise = np.mean(e[10_000:] ** 2)
    return (np.mean((df["y"].to_numpy()[10_000:] - p) ** 2) - noise) / noise


def test_the_trade_off_the_docstring_states():
    """docs/PLAN.md task 207 (review round 5, G3), the trade-off the `pa`
    docstring states at the shipped defaults (`eps = 0.01`, `c = 1` in the
    target's units), on its two targets, each of spread about 2. At R² 0.978
    the band of 0.01 of the target's spread is 0.067 noise stds wide and
    damps nothing, and a cap of 1 binds on few rows: the out-of-sample MSE
    above the noise is about 1.15 of the noise variance at the defaults,
    0.33 at `c = 0.1` and 0.57 at `eps = 0.1`, whose band is 0.67 noise
    stds. At R² 0.99998 the default's band is 2.2 noise stds and no cap
    binds: 0.14 at the defaults and at `c = 0.1` alike, where `eps = 0.1`'s
    band, 22 noise stds, holds the fit wherever it first lands inside it
    (task 207: a median of 65, from 25 to 347 over 20 seeds). The values are
    held with room."""
    default = _excess_over_the_noise(0.978)
    capped = _excess_over_the_noise(0.978, c=0.1)
    wide = _excess_over_the_noise(0.978, eps=0.1)
    assert 1.0 < default < 1.3, default
    assert 0.25 < capped < 0.45, capped
    assert 0.45 < wide < 0.7, wide
    near = _excess_over_the_noise(0.99998)
    assert 0.1 < near < 0.2, near
    assert _excess_over_the_noise(0.99998, c=0.1) == pytest.approx(near, abs=0.02)
    assert _excess_over_the_noise(0.99998, eps=0.1) > 2.0


def test_pa2s_c_is_free_of_the_targets_units():
    """docs/PLAN.md task 207 (review round 5, G5): `pa2`'s `c` enters as
    `1 / (2c)` beside `‖z‖²`, so it is free of the target's units, and a
    target scaled by 2**7 or 2**-7 fits as the unscaled one, scaled, to the
    bit, at the default `c` and at 0.1, and one scaled by 100 or 0.01 to the
    rounding of the scaled values. `pa1`'s `c` is in the target's units and
    has no such property. The stream is review round 4's CC4 at a level of
    1,000."""
    df = _cc4().with_columns(pl.col("y") + 1000.0)
    for c in (None, 0.1):
        kw = {} if c is None else {"c": c}

        def fit(scale, kw=kw):
            p = _oos(df.with_columns(pl.col("y") * scale), mode="pa2", **kw)[1]
            return p / scale

        base = fit(1.0)
        for scale in (2.0**7, 2.0**-7):
            np.testing.assert_array_equal(fit(scale), base, err_msg=f"c={c} x{scale}")
        for scale in (100.0, 0.01):
            np.testing.assert_allclose(fit(scale), base, rtol=1e-12, err_msg=f"x{scale}")


@pytest.mark.parametrize("schedule", ["constant", "inv_scaling"])
def test_sgds_band_holds_no_target_far_from_zero(schedule):
    """docs/PLAN.md task 207: the trap the band's unit exists to keep out
    (task 202), for `sgd` under `epsilon_insensitive` at the default band:
    a target at a level of 1,000 in a spread of about 2 fits rows 10,000 to
    20,000 as the same target at 0 does, without decay. The sign-valued
    gradient's rate is in the target's units, so it is one that travels
    1,000 (`inv_scaling` at 20 covers about `40 sqrt(n)`; a constant 0.2
    covers `0.2 n`): at the default 0.01, or 0.5 under `inv_scaling`, the fit
    never gets there whatever the band, R² -1.9e5 (task 207's report)."""
    lr = {"constant": 0.2, "inv_scaling": 20.0}[schedule]
    r2 = {}
    for level in (0.0, 1000.0):
        df = _cc4(n=20_000).with_columns(pl.col("y") + level)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x"],
            half_life=1e9,
            loss="epsilon_insensitive",
            schedule=schedule,
            learning_rate=lr,
        )
        p = po.ModelBank([spec]).fit_predict(df)["m"].struct.field("pred_y").to_numpy()
        r2[level] = _r2_from(p, df["y"].to_numpy(), 10_000)
    assert r2[1000.0] > 0.9, r2
    assert abs(r2[1000.0] - r2[0.0]) < 0.01, r2


def test_pa1_keeps_the_hard_rules_where_its_cap_binds():
    """docs/PLAN.md task 207: `pa1` at the defaults, on a target at a level
    of 1,000 whose first row has weight 0 and whose half-life is 50, where
    the cap binds (`c = inf` predicts otherwise): one chunk, 7 and 600 give
    the same numbers (hard rule 3); a row's prediction does not move when
    its own target does (rule 2); `weight_sum` is the decayed weight before
    the row (rule 8); and the zero-weight first row advances the clock and
    teaches nothing, so the rows after it predict as the stream without it
    does (rule 9)."""
    df = _cc4(n=1200).with_columns(pl.col("y") + 1000.0)
    w = np.r_[0.0, np.ones(df.height - 1)]
    df = df.with_columns(pl.Series("w", w))
    spec = po.spec.pa("p", targets=["y"], features=["x"], half_life=50.0, weight="w")

    def run(frame, chunks=1, **kw):
        bank = po.ModelBank([{**spec, **kw}] if kw else [spec])
        size = -(-frame.height // chunks)
        parts = [bank.fit_predict(frame.slice(i, size)) for i in range(0, frame.height, size)]
        return pl.concat(parts)["p"].struct.unnest().select("pred_y", "weight_sum")

    one = run(df)
    for chunks in (7, 600):
        assert one.equals(run(df, chunks), null_equal=True), chunks
    uncapped = po.spec.pa("p", targets=["y"], features=["x"], half_life=50.0, weight="w", c=1e9)
    other = po.ModelBank([uncapped]).fit_predict(df)["p"].struct.field("pred_y")
    assert not one["pred_y"].equals(other, null_equal=True), "the cap binds"
    np.testing.assert_allclose(
        one["weight_sum"].to_numpy(), _decayed_sums(w, 0.5 ** (1 / 50)), rtol=1e-12
    )
    i = 700
    row = pl.int_range(pl.len())
    moved = df.with_columns(pl.when(row == i).then(-5e3).otherwise(pl.col("y")).alias("y"))
    assert run(moved)["pred_y"][: i + 1].equals(one["pred_y"][: i + 1])
    assert not run(moved)["pred_y"][i + 1 :].equals(one["pred_y"][i + 1 :])
    rest = run(df.slice(1))["pred_y"]
    np.testing.assert_array_equal(one["pred_y"].to_numpy()[1:], rest.to_numpy())


def _decayed_sums(w, lam):
    """`weight_sum` on each row: `sum_{j < i} lam**(i - j) * w_j`, before
    the row's own update and its decay (hard rule 8)."""
    out = np.zeros(len(w))
    for i in range(1, len(w)):
        out[i] = lam * out[i - 1] + w[i - 1]
    return out


def test_standardize_is_offered_and_on_by_default():
    """docs/PLAN.md task 195 (U2; review round 4, CC6): `pa` standardizes its
    features against the EW scaler `sgd` uses, so `tau = loss / |z|²` and
    `c` stop being in the features' units: features times 128 give the same
    predictions as features times 1, to the bit, at the default and under
    `standardize=True` alike."""
    base = _oos(_cc4())[1]
    for kw in ({}, {"standardize": True}):
        np.testing.assert_array_equal(_oos(_cc4(scale_x=128.0), **kw)[1], base)
    raw = _oos(_cc4(scale_x=128.0), standardize=False)[1]
    assert not np.array_equal(raw, base, equal_nan=True)
