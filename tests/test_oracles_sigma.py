"""``emit_sigma`` without a window, held to its definition on every row.

``sigma_<slot>`` is the EW root mean square of the slot's out-of-sample
residuals: each residual the bank emitted before the row at ``w * 0.5 **
(age / halflife)``, the age on the capped clock the decay uses and counted
from the last accepted row, so a row with no residual -- warm-up, a null
target, a skipped row's step folded into the next -- still ages the rest
(the ``polars_online.spec`` diagnostics table). ``resid_z`` is ``resid /
sigma``. Every other diagnostic that reads the spread -- ``resid_z``,
drift's scale, the conformal band, ``emit_selected`` and ``emit_averaged``
-- takes this number as its input in its own tests, and until now only the
windowed spread (``tests/test_second_opinion.py::TestWindowedSpread``, unit
weights, a row-count clock) was held to anything.

The residuals are the bank's own: the fits behind them are held by the
oracles of each model. What is held here is the spread's recursion.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from test_oracles_lasso_paths import FEATURES, TARGETS, _stream

MAX_DCLOCK = 6.0

# Measured over the cases below: sigma 8.1e-16 (|got - expected| / expected)
# and resid_z 4.5e-16 (over 1 + |resid_z|). The tolerance is 100x the
# largest, rounded up to a power of ten.
#
# Seeded into a copy of the definition, each of these fails a case here by
# far more: unit row weights 0.21-0.56; the ages counted in scored rows
# rather than on the clock 0.28; twice the halflife 0.18; the row's own
# residual in its sigma 0.76; the spread centred at the residuals' EW mean
# (a single residual gives 0).
TOL = 1e-13


def _capped_clock(df: pl.DataFrame) -> np.ndarray:
    """The clock the decay reads, NaN on a skipped row: each step capped at
    ``max_dclock``, a skipped row's folded into the next accepted row's."""
    t, skipped = df["t"].to_numpy(), df["x1"].is_null().to_numpy()
    clock = np.full(len(t), np.nan)
    at, pending, started = 0.0, 0.0, False
    for i in range(len(t)):
        step = 0.0 if i == 0 else t[i] - t[i - 1]
        if skipped[i]:
            pending += step
            continue
        at += min(step + pending, MAX_DCLOCK) if started else 0.0
        pending, started = 0.0, True
        clock[i] = at
    return clock


def _sigma(resid: np.ndarray, w: np.ndarray, clock: np.ndarray, halflife: float) -> np.ndarray:
    """Each accepted row's sigma from the residuals emitted before it."""
    out = np.full(len(resid), np.nan)
    for i in np.flatnonzero(~np.isnan(clock)):
        before = np.flatnonzero(~np.isnan(clock[:i]))
        have = np.flatnonzero(np.isfinite(resid[:i]))
        if before.size == 0 or have.size == 0:
            continue
        weight = w[have] * 0.5 ** ((clock[before[-1]] - clock[have]) / halflife)
        if weight.sum() > 0.0:
            out[i] = np.sqrt(np.sum(weight * resid[have] ** 2) / weight.sum())
    return out


def _spec(model, halflife):
    kw = dict(
        targets=TARGETS,
        features=FEATURES,
        clock="t",
        max_dclock=MAX_DCLOCK,
        weight="w",
        halflife=halflife,
        emit_sigma=True,
        emit_resid_z=True,
        min_periods=[15.0, 10.0, 20.0],
    )
    if model == "ewridge":
        return po.spec.ewridge("m", ridge=[1e-6, 0.5], max_error_inflation=float("inf"), **kw)
    if model == "lasso":
        return po.spec.lasso("m", lasso_path=[0.1, 0.0], **kw)
    if model == "huber":
        return po.spec.huber("m", **kw)
    return po.spec.sgd("m", learning_rate=0.02, **kw)


# The 0.5 slot is ridge-dominated on purpose, which ReadinessWarning says.
@pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")
@pytest.mark.parametrize(
    ("model", "halflife"),
    [("ewridge", 30.0), ("ewridge", [20.0, 60.0]), ("lasso", 30.0), ("huber", 30.0), ("sgd", 30.0)],
    ids=["ewridge", "ewridge-halflife-grid", "lasso", "huber", "sgd"],
)
def test_sigma_is_the_ew_rms_of_the_residuals_before_the_row(model, halflife):
    df = _stream(61)
    spec = _spec(model, halflife)
    out = po.ModelBank([spec]).fit_predict(df)["m"]
    index = po.spec.output_index(spec)
    clock, w = _capped_clock(df), df["w"].to_numpy()
    slots = index.filter(pl.col("kind") == "sigma")
    assert slots.height >= 3
    for row in slots.iter_rows(named=True):
        sigma = out.struct.field(row["field"]).to_numpy().astype(float)
        resid = out.struct.field(row["field"].replace("sigma_", "resid_", 1)).to_numpy()
        want = _sigma(resid.astype(float), w, clock, row["halflife"])
        assert (np.isnan(sigma) == np.isnan(want)).all(), f"{row['field']}: null pattern"
        ok = ~np.isnan(want)
        assert ok.sum() > 150, row["field"]
        err = np.abs(sigma[ok] - want[ok]) / want[ok]
        assert err.max() <= TOL, f"{row['field']}: max rel diff {err.max():.3e}"
        z = out.struct.field(row["field"].replace("sigma_", "resid_z_", 1)).to_numpy()
        z = z.astype(float)
        has = ok & np.isfinite(resid.astype(float))
        assert (np.isfinite(z) == has).all(), f"{row['field']}: resid_z null pattern"
        err = np.abs(z[has] - resid[has] / want[has]) / (1.0 + np.abs(z[has]))
        assert err.max() <= TOL, f"{row['field']}: resid_z max rel diff {err.max():.3e}"
