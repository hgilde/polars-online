"""A restart is a fresh fit: after a clock step back under
``restart_after_step_back`` or a ``drift`` flag under ``drift_action="reset"``,
each prediction is scikit-learn's weighted least squares over the rows since
the restart (task 320, rank 12).

The rules for *when* a model restarts are this library's own, and are held
elsewhere; what a third party can hold is the *effect*: from the restart's
row on, ``ewridge``'s out-of-sample prediction for a row is
``LinearRegression`` fitted on the rows from the restart up to the one
before it, each at its weight decayed on the clock, and before the restart
it is the same over the rows from the stream's start. A model that kept a
trace of the rows before the restart would read them here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

import polars_online as po

TIER = "essential"

HALF_LIFE = 80.0


def stream(n: int = 900, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two features, a slope that changes at row 450 and an intercept that
    jumps by 8 there; an irregular clock."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    beta = np.where(np.arange(n)[:, None] < 450, [1.0, -0.5], [-1.0, 2.0])
    y = 0.5 + (x * beta).sum(axis=1) + 0.3 * rng.normal(size=n)
    y[450:] += 8.0
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    return x, y, t


def spec(**kw) -> dict:
    return po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=HALF_LIFE,
        ridge=1e-12,
        solve_every=0.0,
        min_weight=5.0,
        clock="t",
        gap_cap=1e9,
        **kw,
    )


def hold_to_sklearn(out: pl.DataFrame, x, y, t, restarts: list[int]) -> int:
    """Every prediction against scikit-learn's fit over the rows since the
    last restart before it (or the start), at weights ``0.5^(age/h)`` on
    the clock. Returns how many rows were compared."""
    pred = out["pred_y"].to_numpy().astype(float)
    starts = [0, *restarts]
    compared, worst = 0, 0.0
    for i in range(len(y)):
        r = max(s for s in starts if s <= i)
        if not np.isfinite(pred[i]) or i - r < 3:
            continue
        rows = np.arange(r, i)
        fit = LinearRegression().fit(
            x[rows], y[rows], sample_weight=0.5 ** ((t[i - 1] - t[rows]) / HALF_LIFE)
        )
        want = float(fit.predict(x[i : i + 1])[0])
        # Measured 2026-11-04: 8.1e-12 at most, against values of order 10:
        # a mean-form running fit against a batch one.
        assert abs(pred[i] - want) <= 5e-11, (i, r, pred[i], want)
        worst = max(worst, abs(pred[i] - want))
        compared += 1
    print(f"worst {worst:.1e} over {compared} rows")
    return compared


def test_after_a_step_back_the_fit_is_sklearns_on_the_rows_since():
    """A clock that steps back by 500 at row 600 (larger than
    ``restart_after_step_back=10``): the model restarts there, its weight
    before row 600 reads 0, and from row 605 on (after ``min_weight``) every
    prediction is the fit over rows 600 on alone."""
    x, y, t = stream()
    t = t.copy()
    t[600:] -= 500.0
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "t": t})
    out = po.ModelBank([spec(restart_after_step_back=10.0)]).fit_predict(df).unnest("m")
    w = out["weight_sum"].to_numpy()
    assert np.flatnonzero(w == 0.0).tolist() == [0, 600]
    compared = hold_to_sklearn(out, x, y, t, [600])
    assert compared > 850
    # The trace of the rows before it would read here: the fit over every
    # row is far from the prediction just after the restart.
    rows = np.arange(0, 620)
    whole = LinearRegression().fit(
        x[rows], y[rows], sample_weight=0.5 ** ((t[619] - t[rows]).clip(0) / HALF_LIFE)
    )
    assert abs(out["pred_y"][620] - whole.predict(x[620:621])[0]) > 0.1


def test_after_a_drift_reset_the_fit_is_sklearns_on_the_rows_since():
    """``drift_action="reset"``: the intercept's jump of 8 at row 450 makes
    ``drift`` flag, and the model restarts from the row after the flag (its
    weight reads 0 there). From there every prediction is the fit over the
    rows since alone."""
    x, y, t = stream()
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y, "t": t})
    out = (
        po.ModelBank([spec(emit_drift=True, drift_threshold=5.0, drift_action="reset")])
        .fit_predict(df)
        .unnest("m")
    )
    flags = np.flatnonzero(out["drift_y"].fill_null(False).to_numpy())
    w = out["weight_sum"].to_numpy()
    restarts = [int(r) for r in np.flatnonzero(w == 0.0) if r > 0]
    print("flags", flags.tolist(), "restarts", restarts)
    assert len(flags) >= 1 and 450 <= flags[0] < 470, flags
    assert restarts == [int(f) + 1 for f in flags], (flags, restarts)
    compared = hold_to_sklearn(out, x, y, t, restarts)
    assert compared > 800
