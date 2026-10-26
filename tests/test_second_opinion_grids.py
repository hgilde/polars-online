"""Seed grids beside five single-seed second opinions (task 280, phase 3).

Each test here is the twin of a test in the essentials that compares this
library with a third-party oracle on one seed, against a bound with room in
it: a change could move the error most of the way to its bound without a
test failing. The twin runs the same design over 50 seeds against the same
oracle and records each seed's error in the oracle ledger
(``tests/ledger.py``), which holds it as measured and only lets it tighten.
The single-seed tests stay where they are, the cheap check in the gate.

- ``marglag_acf``: ``marginal``'s lag correlations against statsmodels'
  ``acf``, the truncated and Bartlett serial factors' pairs and a window's
  rows (three tests of ``tests/test_second_opinion.py``, 0.02), and the
  window's against its definition written out;
- ``kalman_share_p_se``: ``share_p``'s ``se_coef`` against each target's
  own filterpy ``KalmanFilter`` (a ratio within 0.05 and 0.15 of 1);
- ``ewridge_robust_se``: ``ewridge``'s HC0 and HAC run once against
  statsmodels' ``OLS`` (``tests/test_second_opinion_diagnostics.py``, a
  ratio in (1, 1.03));
- ``ewridge_gram_huber``: Huber's algorithm H on the bank's Gram against
  statsmodels' ``RLM`` (1e-6, measured 2.5e-7);
- ``rls_kalman_influence``: ``rls``'s and ``kalman``'s ``influence``
  against statsmodels' DFFITS (``tests/test_second_opinion_rls_errors.py``).

Every grid is ``oracle_grid``: CI runs it on every push, no local run does,
and the gate holds this library's outputs on the first seeds' inputs by a
hash instead.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from filterpy.kalman import KalmanFilter
from statsmodels.stats.outliers_influence import OLSInfluence

import ledger
import polars_online as po

TIER = "oracle_grid"

pytestmark = pytest.mark.oracle_grid(reason="a grid of 50 seeds against a third-party oracle")

SEEDS = range(50)
#: The seeds whose outputs the gate's hash guard holds.
GUARD_SEEDS = range(3)


# --- marginal's lag correlations against statsmodels' acf ----------------------


def ar1(rng: np.random.Generator, phi: float, n: int) -> np.ndarray:
    """An AR(1) series with unit innovations, started at its first draw."""
    e = rng.normal(0.0, 1.0, n)
    out = np.empty(n)
    out[0] = e[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + e[t]
    return out


def serial_pair(seed: int, phi_y: float) -> pl.DataFrame:
    """``x`` AR(1) at 0.8 and ``y`` at ``phi_y``, 5,000 rows: the design of
    ``TestAHopelessSerialFactorSaysSo`` and ``TestTheBartlettSerialFactor``."""
    rng = np.random.default_rng(seed)
    x = ar1(rng, 0.8, 5000)
    return pl.DataFrame({"x": x, "y": ar1(rng, phi_y, 5000)})


def windowed_series(seed: int) -> pl.DataFrame:
    """``TestAWindowedLagIsTheWindows``' design: ``x`` AR(1) at 0.2, then 0.8
    from row 4,500 of 5,000; ``y`` white noise; a row clock."""
    rng = np.random.default_rng(seed)
    n, change = 5000, 4500
    e = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = (0.2 if t < change else 0.8) * x[t - 1] + e[t]
    y = rng.normal(0.0, 1.0, n)
    return pl.DataFrame({"t": np.arange(n, dtype=float), "x": x, "y": y})


#: The three specs, by the name of the test each twins.
LAG_SPECS: dict[str, dict] = {
    "truncated": po.spec.marginal(
        "m",
        targets=["y"],
        features=["x"],
        lags=[1],
        serial_rule="truncated",
        half_life=float("inf"),
    ),
    "bartlett": po.spec.marginal(
        "m",
        targets=["y"],
        features=["x"],
        lags=[1, 2, 3, 4],
        serial_rule="bartlett",
        half_life=float("inf"),
    ),
    "windowed": po.spec.marginal(
        "m",
        targets=["y"],
        features=["x"],
        lags=[1, 2, 3],
        half_life=float("inf"),
        clock="t",
        gap_cap=1.0,
        window_size=500.0,
        window_lags=True,
    ),
}


def lag_row(kind: str, frame: pl.DataFrame) -> dict:
    """``marginal()``'s row after one pass of ``frame`` under ``kind``."""
    bank = po.ModelBank([LAG_SPECS[kind]])
    bank.fit_predict(frame)
    return bank.marginal("m").row(0, named=True)


ledger.guard(
    "marglag_acf",
    lambda: [
        [lag_row(k, serial_pair(s, p)) for k in ("truncated", "bartlett") for p in (0.8, -0.8)]
        + [lag_row("windowed", windowed_series(s))]
        for s in GUARD_SEEDS
    ],
    oracles=["statsmodels"],
)


@pytest.mark.parametrize("kind", ["truncated", "bartlett"])
@pytest.mark.parametrize("phi_y", [0.8, -0.8])
def test_lag_corr_is_acf_over_fifty_seeds(kind, phi_y):
    """Every lag's correlation of each series against ``acf`` of it, the
    largest per seed: within the 0.02 the single-seed tests hold."""
    errors: dict[str, list[float]] = {"xx": [], "yy": []}
    for seed in SEEDS:
        frame = serial_pair(seed, phi_y)
        row = lag_row(kind, frame)
        lags = len(row["lag_corr_xx"])
        for leg, col in (("xx", "x"), ("yy", "y")):
            want = stattools.acf(frame[col].to_numpy(), nlags=lags)[1:]
            errors[leg].append(float(np.max(np.abs(np.asarray(row[f"lag_corr_{leg}"]) - want))))
    tag = f"{kind}_phi_y_{phi_y:g}".replace(".", "_").replace("-", "minus_")
    for leg, values in errors.items():
        ledger.record("marglag_acf", f"{tag}_{leg}", values)
        print(f"{tag} {leg}: worst {max(values):.4f}")
        assert max(values) <= 0.02, (leg, max(values), int(np.argmax(values)))


def window_lag_corr(x: np.ndarray, window: int, lags: list[int]) -> np.ndarray:
    """The windowed lag correlation as the library defines it, by hand: over
    the window's last ``window`` rows, unit weights, each row's product with
    the row ``l`` back -- a row before the window included -- about the
    window's own mean, over the window's population variance."""
    inside = np.arange(len(x) - window, len(x))
    m = x[inside].mean()
    var = np.mean((x[inside] - m) ** 2)
    return np.array([np.mean((x[inside] - m) * (x[inside - lag] - m)) / var for lag in lags])


def test_windowed_lag_corr_is_its_definition_and_near_acf_over_fifty_seeds():
    """The windowed lags against their definition written out, to rounding,
    and against ``acf`` of the window's 500 rows, whose products stop at the
    window's edge where the library's take the rows before it as partners
    (``l`` of 500 products): measured 2026-10-26 over seeds 0-49, at most
    8.3e-14 from the definition and 0.0068 from ``acf``. Before the lag
    moments took an origin they were centred at the running mean as it
    stood, which carried the history's mean into the window: 0.165 from
    ``acf`` (seed 28, a lag-1 correlation of 1.016), 19 of 50 seeds past
    0.02."""
    own, against_acf = [], []
    for seed in SEEDS:
        frame = windowed_series(seed)
        x = frame["x"].to_numpy()
        got = np.asarray(lag_row("windowed", frame)["lag_corr_xx"])
        own.append(float(np.max(np.abs(got - window_lag_corr(x, 500, [1, 2, 3])))))
        inside = stattools.acf(x[-500:], nlags=3)[1:]
        against_acf.append(float(np.max(np.abs(got - inside))))
    ledger.record("marglag_acf", "windowed_xx_definition", own, floor=WINDOW_FLOOR)
    ledger.record("marglag_acf", "windowed_xx", against_acf)
    print(f"windowed: definition {max(own):.2g}, acf {max(against_acf):.4f}")
    assert max(own) <= 1e-12, (max(own), int(np.argmax(own)))
    assert max(against_acf) <= 0.01, (max(against_acf), int(np.argmax(against_acf)))


#: The windowed lags' distance from their definition is rounding: its
#: floor in the ledger, twice the worst measured (8.3e-14).
WINDOW_FLOOR = 1.7e-13


# --- kalman's shared covariance against each target's own filter ---------------

SHARED_NOISE = (0.01, 1.0)
SHARED_H = 50.0


def shared_rows(seed: int, n: int = 6000) -> tuple[np.ndarray, dict[str, np.ndarray], pl.DataFrame]:
    """``TestASharedCovariancesStandardErrorsAreEachTargets``' design: two
    targets on drifting coefficients, of noise 0.01 and 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    Z = np.column_stack([np.ones(n), x])
    ys = {}
    for j, r in enumerate(SHARED_NOISE):
        q = r * (np.log(2.0) / SHARED_H) ** 2
        b = np.array([0.5, -0.3, 0.2]) + np.cumsum(np.sqrt(q) * rng.normal(size=(n, 3)), 0)
        ys[f"y{j}"] = (Z * b).sum(axis=1) + np.sqrt(r) * rng.normal(size=n)
    return Z, ys, pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], **ys})


def shared_se(frame: pl.DataFrame) -> np.ndarray:
    """The bank's ``se_coef`` on every row, both targets' side by side."""
    spec = po.spec.kalman(
        "k",
        targets=["y0", "y1"],
        features=["x0", "x1"],
        coef_half_life=SHARED_H,
        half_life=500.0,
        min_weight=0.0,
        standardize=False,
        share_p=True,
        emit_se_coef=True,
        coef_every=1,
    )
    out = po.ModelBank([spec]).fit_predict(frame)["k"].struct.field("se_coef")
    return np.array(out.to_list(), dtype=float)


ledger.guard(
    "kalman_share_p_se",
    lambda: [shared_se(shared_rows(s)[2]) for s in GUARD_SEEDS],
    oracles=["filterpy"],
)


def test_each_targets_se_is_its_own_filters_over_fifty_seeds():
    """The ratio of the bank's standard errors to filterpy's at each
    target's own noise, past the first 2,000 rows: its median within 0.05
    of 1 and its 1% and 99% quantiles within 0.15, on every seed."""
    stats: dict[str, list[float]] = {}
    for seed in SEEDS:
        Z, ys, frame = shared_rows(seed)
        se = shared_se(frame)
        for j, r in enumerate(SHARED_NOISE):
            kf = KalmanFilter(dim_x=3, dim_z=1)
            kf.x = np.zeros((3, 1))
            kf.P = np.eye(3) * r
            own = np.empty((len(Z), 3))
            for i in range(len(Z)):
                kf.predict(Q=np.eye(3) * r * (np.log(2.0) / SHARED_H) ** 2)
                kf.update(ys[f"y{j}"][i], R=r, H=Z[i][None, :])
                own[i] = np.sqrt(np.diag(kf.P))
            ratio = se[2000:, 3 * j : 3 * j + 3] / own[2000:]
            lo, hi = np.quantile(ratio, [0.01, 0.99])
            for name, v in (("median", np.median(ratio)), ("q01", lo), ("q99", hi)):
                stats.setdefault(f"y{j}_{name}_ratio", []).append(float(v))
    for case, values in stats.items():
        ledger.record("kalman_share_p_se", case, values, ideal=1.0)
        gap = np.abs(np.asarray(values) - 1.0)
        print(f"{case}: worst gap {gap.max():.4f}")
        assert gap.max() <= (0.05 if "median" in case else 0.15), (case, gap.max())


# --- ewridge's robust standard errors run once, against statsmodels' OLS --------


def sandwich_rows(n: int, seed: int) -> pl.DataFrame:
    """Two features far from zero and a residual whose spread grows with the
    first: heteroskedastic, as HC0 is for (the diagnostics tests' design)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([3.0, -1.0])
    e = rng.normal(size=n) * (0.5 + np.abs(x[:, 0] - 3.0))
    y = 0.5 + x @ np.array([1.0, -2.0]) + e
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


def robust_se(frame: pl.DataFrame, lags: int = 3) -> pl.DataFrame:
    spec = po.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        ridge=1e-12,
        min_weight=10.0,
        emit_robust_se=True,
        robust_se_lags=lags,
        coef_every=0,
        half_life=float("inf"),
    )
    return po.ModelBank([spec]).fit_predict(frame).unnest("m")


ledger.guard(
    "ewridge_robust_se",
    lambda: [robust_se(sandwich_rows(1500, s)) for s in GUARD_SEEDS],
    oracles=["statsmodels"],
)


def test_the_sandwich_run_once_is_statsmodels_over_fifty_seeds():
    """Each coefficient's HC0 and HAC standard error over statsmodels' on
    the same rows. Out of sample the residuals carry the estimation error,
    so the ratio is above 1 on most seeds, as the single-seed test holds on
    seed 52 (in (1, 1.03)). Measured 2026-10-26 over seeds 100-149: HC0
    0.990 to 1.034, below 1 for a coefficient on 5 seeds; HAC 0.985 to
    1.059, on 8. Held within (0.97, 1.08)."""
    ratios: dict[str, list[np.ndarray]] = {"hc0": [], "hac": []}
    for seed in SEEDS:
        frame = sandwich_rows(1500, 100 + seed)
        out = robust_se(frame)
        ok = np.isfinite(out["resid_y"].to_numpy())
        X = sm.add_constant(frame.select("x0", "x1").to_numpy())[ok]
        y = frame["y"].to_numpy()[ok]
        for cov, kw, field in (("HC0", {}, "hc0"), ("HAC", {"maxlags": 3}, "hac")):
            fit = sm.OLS(y, X).fit(cov_type=cov, cov_kwds=kw or None)
            ratios[field].append(out[f"se_coef_{field}"][-1].to_numpy() / fit.bse)
    for field, rows in ratios.items():
        a = np.array(rows)
        ledger.record("ewridge_robust_se", f"{field}_ratio_largest", a.max(axis=1), ideal=1.0)
        ledger.record("ewridge_robust_se", f"{field}_ratio_smallest", a.min(axis=1), ideal=1.0)
        print(f"{field}: ratios {a.min():.4f} to {a.max():.4f}")
    for field, rows in ratios.items():
        a = np.array(rows)
        assert ((a > 0.97) & (a < 1.08)).all(), (field, a.min(), a.max())


# --- algorithm H on the bank's Gram, against statsmodels' RLM ---------------------


def gram_rows(seed: int, n: int = 20_000) -> pl.DataFrame:
    """``TestAlgorithmHOnAGramAgainstStatsmodels``' stream: a level of 1,000,
    +50 on a tenth of the rows with ``x > 0``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 1000.0 + 2.0 * x + 0.5 * rng.standard_normal(n)
    y[(rng.random(n) < 0.10) & (x > 0)] += 50.0
    return pl.DataFrame({"x": x, "y": y})


GRAM_SPEC = po.spec.ewridge(
    "g", targets=["y"], features=["x"], half_life=float("inf"), solve_every=1e18
)


def gram_of(frame: Any) -> dict:
    bank = po.ModelBank([GRAM_SPEC])
    bank.fit(frame)
    return bank.gram("g")[0]


def algorithm_h(frame: pl.DataFrame, passes: int = 30) -> np.ndarray:
    """``docs/GRAM.md``'s recipe: the Gram's least-squares solve, then
    ``passes`` of Huber's algorithm H, each a Polars query over the rows."""
    g = gram_of(frame.lazy())
    m = float(np.asarray(g["means"])[1])
    var = float(np.asarray(g["comoments"])[1, 1])
    coef = np.asarray(po.gram.solve(g), dtype=float)
    for _ in range(passes):
        fit = coef[0] + coef[1] * pl.col("x")
        r = pl.col("y") - fit
        s = r.abs().median() / 0.6745
        ystar = fit + s * (r / s).clip(-1.345, 1.345)
        q = frame.select(mean=ystar.mean(), c=((pl.col("x") - m) * ystar).mean())
        slope = q["c"][0] / var
        coef = np.array([q["mean"][0] - m * slope, slope])
    return coef


ledger.guard(
    "ewridge_gram_huber",
    lambda: [
        (gram_of(gram_rows(s).lazy()), algorithm_h(gram_rows(s), passes=3)) for s in GUARD_SEEDS
    ],
    oracles=["statsmodels"],
)


def test_algorithm_h_reaches_the_m_estimate_over_fifty_seeds():
    """The passes' intercept and slope against ``RLM(HuberT(1.345))`` with
    the MAD scale, within the single-seed tests' 1e-6."""
    gaps: dict[str, list[float]] = {"intercept": [], "slope": []}
    for seed in SEEDS:
        frame = gram_rows(seed)
        coef = algorithm_h(frame)
        x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = (
                sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT(1.345))
                .fit(scale_est="mad", tol=1e-12, maxiter=500)
                .params
            )
        gaps["intercept"].append(abs(coef[0] - theirs[0]))
        gaps["slope"].append(abs(coef[1] - theirs[1]))
    for name, values in gaps.items():
        ledger.record("ewridge_gram_huber", f"{name}_gap", values)
        print(f"{name}: worst {max(values):.3g}")
        assert max(values) < 1e-6, (name, max(values), int(np.argmax(values)))


# --- rls's and kalman's influence against statsmodels' DFFITS -----------------------


def leverage_rows(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Two features away from zero, with rows of high leverage (the rls
    errors tests' design)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) + np.array([4.0, -2.0])
    x[::37, 0] += 5.0
    y = 0.5 + x @ np.array([1.0, -1.0]) + rng.normal(size=n)
    return x, y, pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})


def influence_spec(kind: str) -> dict:
    """``TestInfluenceIsDffitsAtTheNewestRow``'s specs: a vanishing prior."""
    common: dict[str, Any] = dict(
        targets=["y"], features=["x0", "x1"], half_life=float("inf"), emit_influence=True
    )
    if kind == "rls":
        return po.spec.rls("m", delta=1e-10, min_weight=3.0, **common)
    return po.spec.kalman(
        "m", coef_half_life=float("inf"), p0=1e7, obs_var=1.0, standardize=False, **common
    )


def influence(kind: str, frame: pl.DataFrame) -> np.ndarray:
    out = po.ModelBank([influence_spec(kind)]).fit_predict(frame).unnest("m")
    return out["influence_y"].to_numpy().astype(float)


ledger.guard(
    "rls_kalman_influence",
    lambda: [
        influence(k, leverage_rows(200 + s)[2]) for k in ("rls", "kalman") for s in GUARD_SEEDS
    ],
    oracles=["statsmodels"],
)


@pytest.mark.parametrize(("kind", "rtol"), [("rls", 6e-6), ("kalman", 6e-3)])
def test_influence_is_dffits_over_fifty_seeds(kind, rtol):
    """Each checked row's influence against DFFITS of its prefix, the
    largest relative gap a seed: the prior's share. The single-seed test's
    tolerances, 1e-8 (``rls``) and 1e-5 (``kalman``), hold on seed 81 and
    on most of these; measured 2026-10-26 over seeds 200-249 the worst is
    3.8e-6 and 3.8e-3 (seed 240, the 95th percentile 3.5e-7 and 3.5e-4),
    on the fifth row, two rows past the three the fit needs, where the
    design is near singular and DFFITS is 28: the prior ``delta`` (or
    ``1/p0``) beside a small eigenvalue of ``X'X`` is its largest share
    there. Held at 1.5 times the worst."""
    worst = []
    for seed in SEEDS:
        x, y, frame = leverage_rows(200 + seed)
        got = influence(kind, frame)
        X = sm.add_constant(x)
        rows = np.flatnonzero(np.isfinite(got))
        gaps = []
        for t in rows[(rows < 30) | (rows % 5 == 0)]:
            want = OLSInfluence(sm.OLS(y[: t + 1], X[: t + 1]).fit()).dffits[0][t]
            gaps.append(abs(got[t] - want) / abs(want))
        worst.append(max(gaps))
    ledger.record("rls_kalman_influence", f"{kind}_relative_gap", worst)
    print(f"{kind}: worst {max(worst):.3g}")
    assert max(worst) <= rtol, (max(worst), int(np.argmax(worst)))
