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
    )
    common.update(kw)
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
    # The detector reads the residual of each prediction `min_weight`
    # withholds, as `sigma` does (task 278, the gates batch): the run with no
    # floor reports those residuals, and its flags are the gated run's.
    free = _fit(kind, df, drift_delta=delta, drift_threshold=threshold, min_weight=0.0)
    assert out["resid_y"].null_count() > free["resid_y"].null_count(), "the gate withheld rows"
    resid, sigma = free["resid_y"].to_numpy(), free["sigma_y"].to_numpy()
    flags = out["drift_y"].fill_null(False).to_numpy()
    assert (flags == free["drift_y"].fill_null(False).to_numpy()).all()
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


# Under a half-life and on a clock. river's ``alpha`` fades the cumulative
# sums, not the mean, so no drift library computes this detector; pandas
# computes its mean. ``ē`` is the EW mean of the scored errors at the model's
# half-life on the clock, ``ewm(halflife=, times=, adjust=True)``: each error
# weighs ``2^(-(t_now - t_i)/h)``, which is the library's ``W = λW + 1`` with
# ``λ = 2^(-d/h)`` over every row's step, scored or not. The excess is
# numpy's: ``m_t = cumsum(d_t (e_t - ē_t - δ))`` with ``d_t`` the row's clock
# step, its low-water mark ``minimum.accumulate`` from 0, and a flag where
# ``m - M`` first exceeds the threshold. A flag starts the detector over, so
# the oracle splits the scored rows there and starts pandas afresh.

#: (half_life, clock): a row clock, a numeric clock with uneven steps, and a
#: temporal clock in minutes.
CLOCKED = [
    (200.0, None),
    (30.0, None),
    (200.0, "t"),
    (30.0, "t"),
    ("2h", "ts"),
]


def _clocked_frame(n: int = 3000, seed: int = 5) -> pl.DataFrame:
    """Uneven clock steps of 0.5, 1 and 3 units (minutes on ``ts``), two
    level shifts, uneven weights and two zero-weight rows, which age the mean
    and are not scored."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.choice([0.5, 1.0, 3.0], n))
    x = rng.normal(0.0, 1.0, n)
    y = 1.0 + 2.0 * x + rng.normal(0.0, 0.5, n)
    for at in (1000, 2000):
        y[at:] += 4.0
    w = rng.uniform(0.5, 1.5, n)
    w[700:702] = 0.0
    ts = pl.datetime(2024, 1, 2) + pl.duration(milliseconds=pl.lit(t * 60_000.0).cast(pl.Int64))
    return pl.DataFrame({"t": t, "x": x, "y": y, "w": w}).with_columns(ts=ts)


def _pandas_page_hinkley(
    e: np.ndarray, t: np.ndarray, fed: np.ndarray, half_life: float, delta: float, threshold: float
) -> tuple[np.ndarray, list[float]]:
    """The flags the definition gives, and each segment's closest approach
    of the statistic to the threshold (evidence no flag rests on rounding)."""
    import pandas as pd

    d = np.diff(t, prepend=t[0])
    flags = np.zeros(len(e), dtype=bool)
    idx, margins = np.flatnonzero(fed), []
    while idx.size:
        s = pd.Series(e[idx], index=pd.to_datetime(t[idx] * 1e9))
        mean = s.ewm(halflife=pd.Timedelta(seconds=half_life), times=s.index, adjust=True).mean()
        m = np.cumsum(d[idx] * (e[idx] - mean.to_numpy() - delta))
        stat = m - np.minimum.accumulate(np.minimum(m, 0.0))
        hit = np.flatnonzero(stat > threshold)
        end = hit[0] + 1 if hit.size else idx.size
        margins.append(float(np.min(np.abs(stat[:end] - threshold))))
        if not hit.size:
            break
        flags[idx[hit[0]]] = True
        idx = idx[end:]
    return flags, margins


@pytest.mark.parametrize(("half_life", "clock"), CLOCKED, ids=lambda v: str(v))
@pytest.mark.parametrize(
    ("delta", "threshold"), [(0.5, 20.0), (0.05, 5.0), (0.0, 30.0)], ids=["default", "low", "zero"]
)
def test_under_a_half_life_and_on_a_clock_the_mean_is_pandas_ewm(
    half_life, clock, delta, threshold
):
    df = _clocked_frame()
    kw: dict = {}
    if clock == "t":
        kw = dict(clock="t", gap_cap=1e6)
    elif clock == "ts":
        # Minutes on the temporal clock: the oracle reads its seconds.
        kw = dict(clock="ts", gap_cap="100d")
        threshold = f"{threshold * 60:g}s"
    out = _fit(
        "rls",
        df,
        half_life=half_life,
        min_weight=0.0,
        drift_delta=delta,
        drift_threshold=threshold,
        **kw,
    )
    resid, sigma = out["resid_y"].to_numpy(), out["sigma_y"].to_numpy()
    flags = out["drift_y"].fill_null(False).to_numpy()
    if clock is None:
        t = np.arange(len(df), dtype=float)
    elif clock == "t":
        t = df["t"].to_numpy()
    else:
        t = df["t"].to_numpy() * 60.0
    h = 7200.0 if half_life == "2h" else half_life
    th = float(threshold[:-1]) if isinstance(threshold, str) else threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(resid) / sigma
    fed = (df["w"].to_numpy() > 0) & np.isfinite(e) & (sigma > 0)
    want, margins = _pandas_page_hinkley(e, t, fed, h, delta, th)
    assert fed.sum() > 2900 and not fed[700:702].any()
    assert np.array_equal(flags, want), (np.flatnonzero(flags), np.flatnonzero(want))
    # The detector flagged, and no flag stands within rounding of the
    # threshold (the closest approach measured 2026-11-04 is 1.2e-3).
    assert want.any(), "the case needs a flag"
    assert min(margins) > 1e-6, margins
