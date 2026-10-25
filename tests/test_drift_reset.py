"""``drift_action="reset"`` starts the model over (docs/PLAN.md task 278,
batch 3).

The definition is oracle-free: a stream whose drift detector fires on row
``t`` and resets is, from row ``t + 1`` on, the stream a fresh bank fed rows
``t + 1, ...`` produces -- every field, to the bit. A fresh bank's first row
has a clock step of 0, so nothing it holds decays on that row; the reset
stream's row ``t + 1`` has the ordinary step, which the rebuilt model must
not decay by either. ``rls``'s prior ``A_0 = delta I`` decayed by it, and
its predictions after a reset differed from a fresh bank's by up to 0.017
(``half_life`` 200, ``delta`` 1, measured 2026-10-25), ``kalman``'s did too
under a fixed process noise, and every kind's ``settled_frac`` counted the
step a fresh bank never saw. And ``learned_clock`` on ``t + 1`` showed row
``t``'s clock, which the rebuilt model never learned from, where the
documentation says it is null after a reset.

Each case:

- has a break strong enough to fire (a sign flip of both slopes and a level
  shift of 30 noise sd at the middle row); the fire is asserted and printed,
  never assumed (a reset test whose detector never fires compares two
  identical runs);
- holds every field from ``t + 1`` to a fresh bank's, every row a ``coef``
  row (``coef_every=0``) so the coefficients are compared too;
- holds every residual diagnostic null on ``t + 1``;
- holds chunk invariance across the reset (1, 7 and 600 chunks, and a chunk
  that ends on the row that fired) and a save and load between ``t`` and
  ``t + 1``.

The clock is irregular (steps uniform on ``[0.5, 1.5]``), so the step the
rebuilt model must not decay by is not one row's.

Essentials: ``ewridge``, ``rls``, ``sgd`` and an ``rls`` half-life grid at
3,000 rows. Extended: the other
seven linear kinds and the variants whose state holds a prior or a clock of
its own (``rls`` at a large ``delta``, ``kalman`` with a fixed process noise
and with a mean-reverting slot, ``ewridge`` under ``ridge_scale="sum"``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_online as po

TIER = "mixed"

N = 3000
#: Every residual switch the ten linear kinds take (docs/DIAGNOSTICS.md,
#: "What each costs, and which models take it").
DIAGNOSTICS: dict[str, Any] = dict(
    emit_sigma=True,
    emit_zscore=True,
    emit_metrics=True,
    conformal=0.9,
    resid_quantiles=[0.5],
    emit_autocorr=True,
    emit_calibration=True,
    emit_breaks=True,
    emit_specification=True,
    emit_tails=True,
    emit_clocks=True,
)
#: The nine with features take feature health; ewridge, rls and kalman
#: take the switches that read their solve.
WITH_FEATURES: dict[str, Any] = dict(emit_feature_health=True)
SOLVED: dict[str, Any] = dict(emit_se_coef=True, emit_error_inflation=True, emit_influence=True)

#: name -> (builder, its own keywords, whether it takes the solved switches)
CASES: dict[str, tuple[Any, dict[str, Any], bool]] = {
    "ewridge": (po.spec.ewridge, {}, True),
    "rls": (po.spec.rls, {}, True),
    "sgd": (po.spec.sgd, {"learning_rate": 0.02}, False),
    "kalman": (po.spec.kalman, {"coef_half_life": 100.0}, True),
    "lasso": (po.spec.lasso, {"penalties": [0.01]}, False),
    "huber": (po.spec.huber, {}, False),
    # sgd's Huber loss: its refit of the first rows (default 200) runs again
    # on the rebuilt model, 200 learned rows after the reset.
    "sgd huber": (po.spec.sgd, {"loss": "huber", "learning_rate": 0.02}, False),
    "quantile": (po.spec.quantile, {"quantile": 0.5}, False),
    "pa": (po.spec.pa, {}, False),
    "ftrl": (po.spec.ftrl, {"loss": "squared"}, False),
    "holt": (po.spec.holt, {}, False),
    # A prior that dominates the first rows: the decay a reset must not add
    # is a factor of 100 on it.
    "rls delta=100": (po.spec.rls, {"delta": 100.0}, True),
    # A fixed process noise and observation noise: `P` grows by `q d` on
    # every step, the prior's included.
    "kalman q": (po.spec.kalman, {"q": [0.01, 0.01, 0.01], "obs_var": 0.04}, True),
    # A mean-reverting slot: its transition reads the step too.
    "kalman revert": (po.spec.kalman, {"coef_half_life": 100.0, "revert_half_life": 50.0}, True),
    # The ridge as a prior in the sums: `rls` by another road.
    "ewridge sum": (po.spec.ewridge, {"ridge_scale": "sum", "ridge": 1.0}, True),
    # A half-life grid: a detection in any instance restarts every one, after
    # all have read the row (the coupled path), and each must read the next
    # row at a step of 0.
    "rls grid": (po.spec.rls, {"half_life": [100.0, 200.0]}, True),
}
ESSENTIAL = ("ewridge", "rls", "sgd", "rls grid")


def _param(name: str):
    if name in ESSENTIAL:
        return name
    return pytest.param(
        name, marks=pytest.mark.extended(reason="a kind of the reset sweep beyond the three")
    )


KINDS = [_param(name) for name in CASES]

#: The fields that describe the model rather than its residuals: what a
#: rebuilt model reports before it has scored a row.
STATE_FIELDS = (
    "weight_sum",
    "settled_frac",
    "withheld_reason",
    "penalty_selected",
    "drift",
    "coef",
    "support_coef",
    "se_coef",
    "scored_clock",
    "error_inflation",
)


def frame(n: int = N, seed: int = 5) -> pl.DataFrame:
    """Two features; at the middle row both slopes flip sign and the level
    moves by 6 against a noise sd of 0.2. The clock steps are irregular."""
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 2))
    sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    y = 0.3 + sign * (x @ np.array([2.0, -1.5])) + 0.2 * r.normal(size=n)
    y[n // 2 :] += 6.0
    t = np.cumsum(r.uniform(0.5, 1.5, n))
    return pl.DataFrame({"t": t, "x0": x[:, 0], "x1": x[:, 1], "y": y})


def spec(name: str) -> dict[str, Any]:
    build, own, solved = CASES[name]
    kw: dict[str, Any] = dict(
        targets=["y"],
        clock="t",
        gap_cap=10.0,  # above every step: no step is capped
        half_life=200.0,
        min_weight=10.0,
        coef_every=0,
        emit_drift=True,
        drift_threshold=20.0,  # the row clock's default, in sigma x clock units
        drift_action="reset",
        **DIAGNOSTICS,
    )
    if build is not po.spec.holt:
        kw["features"] = ["x0", "x1"]
        kw.update(WITH_FEATURES)
    if solved:
        kw.update(SOLVED)
    kw.update(own)
    return build("m", **kw)


def run(bank: po.ModelBank, df: pl.DataFrame) -> pl.DataFrame:
    with warnings.catch_warnings():
        # The readiness notices of a fresh start are expected here.
        warnings.simplefilter("ignore")
        return bank.fit_predict(df).select("m").unnest("m")


def first_fire(out: pl.DataFrame) -> int:
    """The first row whose detector fired, asserted to be at or after the
    break and printed: the evidence the reset path ran."""
    drift = [c for c in out.columns if c.startswith("drift_")]
    fired = (
        out.select(pl.any_horizontal(pl.col(drift).fill_null(False))).to_series().arg_true()
    ).to_list()
    assert fired, "the detector never fired: the test would compare two identical runs"
    t = fired[0]
    print(f"fired {len(fired)}x, first at row {t} (break at {N // 2})")
    assert t >= N // 2, f"a fire at {t}, before the break at {N // 2}"
    assert t < N - 200, f"the fire at {t} leaves too few rows after it to compare"
    return t


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return frame()


@pytest.mark.parametrize("name", KINDS)
def test_a_reset_is_a_fresh_bank_from_the_next_row(name, df):
    """Every field from ``t + 1`` on equals a fresh bank fed ``t + 1, ...``."""
    s = spec(name)
    out = run(po.ModelBank([s], threads=1), df)
    t = first_fire(out)
    weights = [c for c in out.columns if c.startswith("weight_sum")]
    assert all(out[c][t + 1] == 0.0 for c in weights), "the reset did not take effect"
    fresh = run(po.ModelBank([s], threads=1), df.slice(t + 1))
    assert_frame_equal(out.slice(t + 1), fresh, check_exact=True)


@pytest.mark.parametrize("name", KINDS)
def test_every_diagnostic_restarts_null(name, df):
    """On row ``t + 1`` every residual diagnostic is null: each restarts with
    the model it describes. So is ``learned_clock``: the rebuilt model has
    learned from no row (it showed row ``t``'s clock until task 278)."""
    out = run(po.ModelBank([spec(name)], threads=1), df)
    t = first_fire(out)
    row = out.row(t + 1, named=True)
    state = {c for c in row if c.startswith(STATE_FIELDS)}
    residual = sorted(c for c in row if c not in state)
    assert len(residual) >= 20, residual  # the switches are on
    kept = [c for c in residual if row[c] is not None]
    print(f"{len(residual)} residual fields null on row t+1; state fields: {sorted(state)}")
    assert kept == [], f"residual diagnostics still set on the row after a reset: {kept}"


@pytest.mark.parametrize("name", KINDS)
def test_chunking_and_save_load_across_the_reset(name, df, tmp_path):
    """1, 7 and 600 chunks, a chunk ending on the row that fired, and a save
    and load between ``t`` and ``t + 1``, each equal to one chunk."""
    s = spec(name)
    one = run(po.ModelBank([s], threads=1), df)
    t = first_fire(one)
    for parts in (7, 600):
        bank = po.ModelBank([s], threads=1)
        edges = np.linspace(0, df.height, parts + 1).astype(int)
        many = pl.concat(
            [run(bank, df.slice(a, b - a)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
        )
        assert_frame_equal(many, one, check_exact=True)
    # A chunk that ends on the row that fired, then a save and a load: the
    # rebuilt model crosses both before it sees row t + 1.
    a = po.ModelBank([s], threads=1)
    head = run(a, df.slice(0, t + 1))
    path = tmp_path / "reset.state"
    a.save(path)
    b = po.ModelBank.load(path, specs=[s], threads=1)
    rest = df.slice(t + 1)
    assert_frame_equal(pl.concat([head, run(b, rest)]), one, check_exact=True)
    assert_frame_equal(run(a, rest), one.slice(t + 1), check_exact=True)


@pytest.mark.parametrize("name", ["huber", "sgd huber"])
def test_the_refit_starts_over_after_a_reset(name, df, tmp_path):
    """A reset rebuilds the model as a fresh bank would, its refit buffer
    empty: the refit of the first 200 learned rows runs before the break and
    again 200 rows after the reset, each moving the fit (the run without it,
    ``huber_refit_rows = 0``, parts there and not before). A save and a load
    between the reset and the second refit, and after it, go on to the bit."""
    s = spec(name)
    off = {**s, "model": {**s["model"], "huber_refit_rows": 0}}
    one = run(po.ModelBank([s], threads=1), df)
    t = first_fire(one)
    zero = run(po.ModelBank([off], threads=1), df)
    p, q = (f["pred_y"].to_numpy() for f in (one, zero))
    differ = np.flatnonzero(~((p == q) | (np.isnan(p) & np.isnan(q))))
    assert differ.size and differ[0] > 199, "the first refit is on the 200th row"
    # From the row after the reset both are fresh banks: equal until the
    # second refit, on the 200th row after it, and apart after it.
    after = differ[differ > t]
    assert after.size and after[0] > t + 200, (t, after[:3])
    for cut in (t + 100, t + 250):
        a = po.ModelBank([s], threads=1)
        head = run(a, df.slice(0, cut))
        path = tmp_path / f"refit-{cut}.state"
        a.save(path)
        b = po.ModelBank.load(path, specs=[s], threads=1)
        assert_frame_equal(pl.concat([head, run(b, df.slice(cut))]), one, check_exact=True)
