"""docs/PLAN.md task 206 (review round 5, G1): a standardizing ``sgd`` or
``pa`` warms its scaler up over 22 rows (Kish's count of the weights) as it
did before, and from then on holds its fit so that the scaler's moving moves
no prediction, in the caller's units, the step mapped into them by the row's
own moments. ``kalman`` holds its coefficients and their covariance at an
anchor and re-maps them when the moments drift from it, from its first row:
since task 211 it takes no warm-up, each coefficient's prior waiting for its
feature's scale instead.

Read through the moments as they stood, as before the change, every
prediction moved with the scaler's own wander under a finite half-life --
the EW mean of a unit-variance feature has standard deviation ``sqrt((1 -
lam) / (1 + lam))``, 0.083 at a half-life of 50 -- though no step was taken.
The cells here are the review's: ``y = 2 sum(x) + e`` with ``x ~ N(0, 1)``,
30,000 rows, the excess out-of-sample MSE over rows 10,000+ as a share of the
noise variance, standardized against not. Before the change, at R² 0.99998
and a half-life of 50, ``pa`` paid 31.8 noise variances, ``sgd`` 248 and
``kalman`` 223 where unstandardized each paid 0.18, 0.010 and 0.014; at R²
0.978 and a half-life of 10 ``sgd`` paid 1.93 and ended at ``coef`` [0.83,
3.03] for a truth of [0, 2] (median of five seeds; the research behind task
206, TABLES.md section 2). The bounds leave room over the new numbers, not
the old ones; each test's docstring gives what it measured.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po

N = 30_000


def _frame(k: int, r: float, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """The review's stream: ``k`` features ``N(0, 1)``, noise ``r * 2 * sqrt(k)``
    times ``N(0, 1)``, so R² is ``1 / (1 + r²)`` at every ``k``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, k))
    e = r * 2.0 * np.sqrt(k) * rng.standard_normal(N)
    cols = {f"x{i}": x[:, i] for i in range(k)}
    return pl.DataFrame({**cols, "y": 2.0 * x.sum(axis=1) + e}), e


def _spec(kind: str, standardize: bool, features: list[str], **kw) -> dict:
    common = dict(targets=["y"], features=features, standardize=standardize, **kw)
    if kind == "sgd":
        return po.spec.sgd("m", **common)
    if kind == "pa":
        return po.spec.pa("m", **common)
    return po.spec.kalman("m", coef_half_life=50.0, **common)


def _excess(spec: dict, df: pl.DataFrame, e: np.ndarray, lo: int = 10_000, hi: int = N):
    out = po.ModelBank([spec]).fit_predict(df)["m"]
    p = out.struct.field("pred_y").to_numpy().astype(float)[lo:hi]
    noise = float(np.mean(e[lo:hi] ** 2))
    y = df["y"].to_numpy()[lo:hi]
    return (float(np.mean((y - p) ** 2)) - noise) / noise, out


CELLS = [(k, r, hl) for k in (1, 5) for r in (0.15, 0.005) for hl in (10.0, 50.0)]


@pytest.mark.parametrize("kind", ["sgd", "pa", "kalman"])
@pytest.mark.parametrize(("k", "r", "half_life"), CELLS)
def test_a_half_life_costs_no_more_than_the_raw_fit(kind, k, r, half_life):
    """G1's cells: standardized, the excess is within twice the
    unstandardized fit's. Measured on this build (seed 0): at most 1.05
    times it, ``kalman`` at ``k = 5``, half-life 10 (0.0484 against 0.0462);
    before the change, up to 1.8e5 times it (``sgd``, ``k = 1``, R² 0.99998,
    half-life 10: 1,713 against 0.0096)."""
    df, e = _frame(k, r)
    features = [f"x{i}" for i in range(k)]
    std, _ = _excess(_spec(kind, True, features, half_life=half_life), df, e)
    raw, _ = _excess(_spec(kind, False, features, half_life=half_life), df, e)
    assert std <= 2.0 * raw, (std, raw)


def test_sgd_at_a_half_life_of_10_fits_the_slope():
    """R² 0.978, half-life 10: the last ``coef`` is within 0.15 of [0, 2].
    Measured [0.025, 2.018] on this build, [0.83, 3.03] before the change."""
    df, e = _frame(1, 0.15)
    _, out = _excess(_spec("sgd", True, ["x0"], half_life=10.0), df, e)
    coef = out.struct.field("coef").drop_nulls()[-1].to_list()
    assert abs(coef[0]) < 0.15 and abs(coef[1] - 2.0) < 0.15, coef


def _groups(n_groups=500, rows_per=200, k=20, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_groups * rows_per, k))
    beta = np.repeat(rng.standard_normal((n_groups, k)) / np.sqrt(k), rows_per, axis=0)
    y = (x * beta).sum(axis=1) + 0.1 * rng.standard_normal(len(x))
    df = pl.DataFrame({f"x{j}": x[:, j] for j in range(k)}).with_columns(
        y=pl.Series(y), g=pl.Series(np.repeat(np.arange(n_groups), rows_per))
    )
    return y, df


#: R² at rows 25-50 of task 74's 500 groups of 200 rows, ``k = 20``, before
#: task 206 (ed76944, through the bank), and the tolerance it is held to.
SHORT_BEFORE = {"sgd": 0.433, "pa": 0.712, "kalman": 0.675}
SHORT_TOL = 0.01


@pytest.mark.parametrize("kind", ["sgd", "pa", "kalman"])
def test_a_short_history_learns_as_it_did(kind):
    """Task 74's regime, where a fit held in the caller's units from the
    first row lost everything (``sgd`` R² -207, ``pa`` -20206 at rows 25-50,
    the research behind task 206): the warm-up keeps it at least where it
    was, less 0.01. Measured on task 206's build 0.4488 / 0.7075 / 0.6886
    against 0.4328 / 0.7120 / 0.6749 (``sgd`` / ``pa`` / ``kalman``): the
    warm-up ends on row 21, before the window, so its rows are the new
    design's. ``kalman`` has no warm-up since task 211 and measured 0.925
    here (``tests/test_kalman.py`` holds it to 0.9)."""
    y, df = _groups()
    pos = np.tile(np.arange(200), 500)
    features = [f"x{j}" for j in range(20)]
    kw = dict(group="g", half_life=float("inf"), min_weight=25.0)
    if kind == "sgd":
        kw["learning_rate"] = 0.01
    out = po.ModelBank([_spec(kind, True, features, **kw)]).fit_predict(df)
    pred = out["m"].struct.field("pred_y").to_numpy().astype(float)
    ok = (pos >= 25) & (pos < 50)
    err = y[ok] - pred[ok]
    r2 = 1.0 - float(err @ err) / float(((y[ok] - y[ok].mean()) ** 2).sum())
    assert r2 >= SHORT_BEFORE[kind] - SHORT_TOL, r2


def _shift_frame(level: float) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, (3000, 2))
    y = 0.5 + 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.uniform(-1, 1, 3000)
    return pl.DataFrame({"x0": x[:, 0] + level, "x1": x[:, 1] + level, "y": y}), y


@pytest.mark.parametrize("kind", ["sgd", "pa", "kalman"])
def test_a_feature_level_of_1e8_predicts_as_level_0(kind):
    """Both features shifted by 1e8, unit spread, half-life 20: the
    intercept absorbs the level, and the tail (rows 2,000+) predicts as
    level 0 does to within 1e-6 RMS. Measured 1.1e-7 / 2.9e-8 / 1.3e-8 on
    this build (``sgd`` / ``pa`` / ``kalman``), against 9.6e-9 / 1.4e-8 /
    1.8e-6 before the change: the coefficients in the caller's units carry
    the level times the slope, which rounds at that size, and ``kalman``'s
    first rows are read against moments with no row in them."""

    def preds(level):
        df, _ = _shift_frame(level)
        kw = dict(half_life=20.0, min_weight=3.0)
        spec = (
            po.spec.kalman("m", targets=["y"], features=["x0", "x1"], coef_half_life=100.0, **kw)
            if kind == "kalman"
            else _spec(kind, True, ["x0", "x1"], **kw)
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        return out.struct.field("pred_y").to_numpy().astype(float)

    base, shifted = preds(0.0), preds(1e8)
    rms = float(np.sqrt(np.nanmean((shifted[2000:] - base[2000:]) ** 2)))
    assert rms <= 1e-6, rms


#: A x10 change of every feature's scale at row 15,000 (``y = 2 sum(x) +
#: e`` throughout), ``k = 5``, R² 0.99998, half-life 50: the excess over rows
#: 15,000-16,000 and 16,000+ on this build, before it, and the bounds held.
SCALE_MEASURED = {"sgd": (0.118, 0.033), "pa": (2.05, 2.14), "kalman": (0.105, 0.044)}
SCALE_BEFORE = {"sgd": (7.5e4, 2.3e4), "pa": (2.0e4, 6.4e3), "kalman": (2.1e5, 2.1e4)}
SCALE_BOUND = {"sgd": (1.0, 0.3), "pa": (20.0, 20.0), "kalman": (1.0, 0.5)}


@pytest.mark.parametrize("kind", ["sgd", "pa", "kalman"])
def test_a_change_of_scale_does_not_blow_the_fit_up(kind):
    """Every feature times 10 from row 15,000 at a half-life of 50: the fit
    follows the new scale, within ten times what this build measured
    (``SCALE_MEASURED``); before the change the moments' move threw it
    (``SCALE_BEFORE``). ``pa``'s 2.1 after the change is its band, drawn in
    ``y``'s spread, which the change widened tenfold in units of the noise
    (review round 5, G3), not the scaler."""
    rng = np.random.default_rng(0)
    k, r = 5, 0.005
    x = rng.standard_normal((N, k))
    e = r * 2.0 * np.sqrt(k) * rng.standard_normal(N)
    x[15_000:] *= 10.0
    df = pl.DataFrame({**{f"x{i}": x[:, i] for i in range(k)}, "y": 2.0 * x.sum(axis=1) + e})
    spec = _spec(kind, True, [f"x{i}" for i in range(k)], half_life=50.0, min_weight=6.0)
    jump, _ = _excess(spec, df, e, 15_000, 16_000)
    after, _ = _excess(spec, df, e, 16_000, N)
    lo, hi = SCALE_BOUND[kind]
    assert jump <= lo and after <= hi, (jump, after)


#: U2's case (docs/PLAN.md task 195): one feature in thousands, one in basis
#: points, R² 0.978 and 0.99998, no decay: the excess before task 206
#: (ed76944, seed 0, through the bank).
MIXED_BEFORE = {
    ("sgd", 0.15): 0.0150,
    ("sgd", 0.005): 0.0240,
    ("pa", 0.15): 1.2268,
    ("pa", 0.005): 0.2094,
    ("kalman", 0.15): 0.0202,
    ("kalman", 0.005): 0.0264,
}


@pytest.mark.parametrize("r", [0.15, 0.005])
@pytest.mark.parametrize("kind", ["sgd", "pa", "kalman"])
def test_features_of_mixed_scales_fit_as_before(kind, r):
    """One feature times 1,000 and one times 1e-4, no decay: the fit holds
    U2's numbers -- the reason ``sgd`` and ``pa`` standardize by default --
    to within 10% or 0.01 of the noise, whichever is more. Measured on this
    build 0.0151 / 0.0151, 1.2268 / 0.2034 and 0.0202 / 0.0203 (``sgd``,
    ``pa``, ``kalman`` at R² 0.978 / 0.99998); raw, ``sgd`` paid 1.8e8."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, 2))
    e = r * 2.0 * np.sqrt(2.0) * rng.standard_normal(N)
    y = 2.0 * x.sum(axis=1) + e
    df = pl.DataFrame({"x0": 1e3 * x[:, 0], "x1": 1e-4 * x[:, 1], "y": y})
    got, _ = _excess(_spec(kind, True, ["x0", "x1"], half_life=float("inf"), min_weight=3.0), df, e)
    before = MIXED_BEFORE[(kind, r)]
    assert got <= max(1.1 * before, before + 0.01), (got, before)


def _chunk_stream(n=600):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((n, 2)) * np.array([3.0, 0.01]) + np.array([5.0, 0.0])
    y = 1.0 + 2.0 * x[:, 0] - 50.0 * x[:, 1] + 0.2 * rng.standard_normal(n)
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


def _specs():
    common = dict(targets=["y"], features=["x0", "x1"], half_life=20.0, min_weight=3.0)
    return [
        po.spec.sgd("sgd", **common),
        po.spec.pa("pa", **common),
        po.spec.kalman("kalman", coef_half_life=50.0, **common),
    ]


def _floats(df: pl.DataFrame) -> pl.DataFrame:
    """Every float field of every model's struct: chunk invariance is about
    the numbers (``coef`` is emitted on each chunk's last row by design)."""
    cols = []
    for name in ("sgd", "pa", "kalman"):
        s = df[name].struct.unnest()
        cols += [
            s[c].alias(f"{name}.{c}") for c in s.columns if s[c].dtype in (pl.Float64, pl.Float32)
        ]
    return pl.DataFrame(cols)


@pytest.mark.parametrize(
    "cuts",
    [[21], [22], [23], "7", "600"],
    ids=["before-the-switch-row", "at-it", "after-it", "7-chunks", "1-row-chunks"],
)
def test_a_chunk_boundary_around_the_switch_moves_nothing(cuts):
    """Hard rule 3 around the warm-up's end: the switch comes after row 21
    (the 22nd row, unit weights), and the 600-row stream cut there or a row
    either side, into 7 chunks (three of them those cuts) or into 600 chunks
    of one row gives every float field of the one-chunk run, to the bit."""
    df = _chunk_stream()
    whole = _floats(po.ModelBank(_specs()).fit_predict(df))
    if cuts == "7":
        cuts = [21, 22, 23, 150, 300, 450]
    elif cuts == "600":
        cuts = list(range(1, 600))
    bank = po.ModelBank(_specs())
    bounds = [0, *cuts, df.height]
    parts = [bank.fit_predict(df[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]
    assert _floats(pl.concat(parts)).equals(whole)


@pytest.mark.parametrize("cut", [10, 21, 22, 40], ids=["warming", "switch-1", "switch", "past"])
def test_a_bank_saved_around_the_switch_goes_on_to_the_bit(cut):
    """A bank saved inside the warm-up, just before and at the switch and
    past it, and loaded from its bytes, goes on as the bank that never
    stopped: every float field to the bit."""
    df = _chunk_stream()
    whole = _floats(po.ModelBank(_specs()).fit_predict(df))
    first = po.ModelBank(_specs())
    head = first.fit_predict(df[:cut])
    back = po.ModelBank.load_bytes(first.save_bytes())
    tail = back.fit_predict(df[cut:])
    assert _floats(pl.concat([head, tail])).equals(whole)
