"""``emit_drift`` against river's ``PageHinkley`` away from the defaults, on
kinds beyond ``ewridge``, and its two knobs' order (task 278, batch 23).

When nothing decays, the bank's detector is river's ``PageHinkley`` with
``alpha = 1`` (a plain running mean) and ``mode = "up"``, fed the bank's own
``|resid| / sigma`` on every row of positive weight with a residual:
``tests/test_second_opinion.py`` holds that flag for flag on ``ewridge`` at
``drift_delta=0.05`` and ``drift_threshold=20``. Here the same comparison
runs at ``drift_threshold=5`` and at ``drift_delta`` 0 and 0.5, on the
residuals of ``rls``, ``sgd`` and ``holt``. The residuals are the bank's;
what is held is the detector.

The knobs' order needs no library. The statistic is ``m_t = Σ (e - ē - δ)
d``, summed over the rows' clock steps ``d``, and it flags when it climbs
``threshold`` above its lowest point. ``ē`` reads neither knob. A larger
``δ`` lowers every climb ``m_t - m_s`` by ``(δ' - δ)`` times the clock
between, and a larger threshold asks more of the same climb. So up to the
first flag, after which the detector re-arms, the first flag comes no
earlier at a larger ``drift_delta`` or a larger ``drift_threshold``
(derived). How many flags follow is measured, not derived: a later re-arm
can line up differently.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_online as po

TIER = "essential"

SHIFTS = (900, 1500)


def _frame(n: int = 2000, seed: int = 51) -> pl.DataFrame:
    """A target whose level steps by ten noise deviations at each of
    ``SHIFTS``, a zero-weight row with a wild target before the first, and
    uneven weights."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    y = 1.0 + 2.0 * x + rng.normal(0.0, 0.5, n)
    for at in SHIFTS:
        y[at:] += 5.0
    w = rng.uniform(0.5, 1.5, n)
    w[700] = 0.0
    y[700] += 100.0
    return pl.DataFrame({"x": x, "y": y, "w": w})


def _spec(kind: str, **kw):
    common = dict(
        targets=["y"],
        half_life=float("inf"),
        weight="w",
        min_weight=5.0,
        emit_sigma=True,
        emit_drift=True,
        **kw,
    )
    if kind == "holt":
        return po.spec.holt("m", **common)
    if kind == "rls":
        return po.spec.rls("m", features=["x"], delta=1.0, **common)
    return po.spec.sgd("m", features=["x"], learning_rate=0.02, **common)


def _fit(kind: str, df: pl.DataFrame, **kw) -> pl.DataFrame:
    return po.ModelBank([_spec(kind, **kw)]).fit_predict(df)["m"].struct.unnest()


@pytest.mark.parametrize("kind", ["rls", "sgd", "holt"])
@pytest.mark.parametrize(
    ("delta", "threshold"),
    [(0.05, 5.0), (0.0, 20.0), (0.5, 20.0), (0.5, 5.0)],
    ids=["threshold-5", "delta-0", "delta-0.5", "both"],
)
def test_the_drift_column_is_rivers_page_hinkley(kind, delta, threshold):
    import river.drift as river_drift

    df = _frame()
    out = _fit(kind, df, drift_delta=delta, drift_threshold=threshold)
    resid, sigma = out["resid_y"].to_numpy(), out["sigma_y"].to_numpy()
    flags = out["drift_y"].fill_null(False).to_numpy()
    w = df["w"].to_numpy()
    ph = river_drift.PageHinkley(
        min_instances=0, delta=delta, threshold=threshold, alpha=1.0, mode="up"
    )
    fed, rivers = 0, []
    for t in range(len(resid)):
        e = abs(resid[t]) / sigma[t] if sigma[t] > 0 else float("nan")
        if w[t] > 0 and np.isfinite(e):
            ph.update(float(e))
            fed += 1
            assert bool(flags[t]) == bool(ph.drift_detected), t
            if ph.drift_detected:
                rivers.append(t)
        else:
            assert not flags[t], t
    assert fed > 1900
    # The detector fired, at a shift, so the comparison read a flag.
    assert rivers == np.flatnonzero(flags).tolist()
    assert any(t >= SHIFTS[0] for t in rivers), rivers


def _flags(kind: str, **kw) -> np.ndarray:
    return np.flatnonzero(_fit(kind, _frame(seed=52), **kw)["drift_y"].fill_null(False).to_numpy())


@pytest.mark.parametrize("kind", ["rls", "sgd", "holt"])
def test_a_larger_threshold_flags_no_sooner_and_no_more(kind):
    """The first flag's order is derived (the module's docstring). The
    counts are measured, 2026-10-26, at thresholds 2, 5, 20 and 200: rls 5,
    2, 1, 1; sgd 3, 2, 2, 0; holt 7, 2, 1, 0. ``sgd``'s count is not held:
    its fit is still moving this release, and a count need not keep its
    order when the residuals move."""
    runs = [_flags(kind, drift_delta=0.5, drift_threshold=th) for th in (2.0, 5.0, 20.0, 200.0)]
    firsts = [r[0] if r.size else np.inf for r in runs]
    assert firsts == sorted(firsts), firsts
    assert firsts[0] < firsts[2] < np.inf, firsts
    if kind != "sgd":
        counts = [r.size for r in runs]
        assert counts == sorted(counts, reverse=True), counts
        assert counts[0] > counts[2] >= 1, counts


@pytest.mark.parametrize("kind", ["rls", "sgd", "holt"])
def test_a_larger_delta_flags_no_sooner_and_no_more(kind):
    """As above; the counts measured 2026-10-26 at ``drift_delta`` 0, 0.05,
    0.5 and 2: rls 3, 2, 1, 1; sgd 3, 3, 2, 2; holt 3, 2, 1, 0."""
    runs = [_flags(kind, drift_delta=d, drift_threshold=20.0) for d in (0.0, 0.05, 0.5, 2.0)]
    firsts = [r[0] if r.size else np.inf for r in runs]
    assert firsts == sorted(firsts), firsts
    assert firsts[0] < firsts[2] < np.inf, firsts
    if kind != "sgd":
        counts = [r.size for r in runs]
        assert counts == sorted(counts, reverse=True), counts
        assert counts[0] > counts[2] >= 1, counts
