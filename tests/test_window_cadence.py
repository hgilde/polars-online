"""A model window's snapshots are spaced on the clock (docs/PLAN.md task
162): every ``snapshot_every`` clock units or every
``max_rows_between_snapshots`` rows, whichever comes first, as the
regressions' ``solve_every`` and ``max_rows_between_solves`` schedule a
solve; every row with neither.

Each test holds the model to its window written out from the definition, over
the stream's own clock:

- the clock is the one the model is stepped on, the first row's step 0 and
  each later one capped at ``gap_cap``;
- a snapshot is taken at the first row, then at each row whose clock is
  ``snapshot_every`` past the newest snapshot's or which is
  ``max_rows_between_snapshots`` rows past it, and at any row whose newest
  snapshot has left the window;
- a row reads the window as the row before it left it: the boundary is the
  oldest snapshot whose clock is within ``window_size`` of that row's;
- the window holds the rows from the boundary on, each at its exponential
  weight, so a windowed mean is their weighted mean and a windowed ridge is
  scikit-learn's ``Ridge`` on them (``alpha = ridge * the weight``, as
  ``TestEwRidgeIsSklearnsRidge`` maps it);
- a ridge solves after every row with weight, and a row of weight 0 never
  solves (hard rule 9; docs/PLAN.md task 214), so a row after one reads the
  fit the last row with weight left, on its window.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge

import polars_online as po
from reference_paths import WindowRing, window_starts

TIER = "mixed"

#: The closed x shape x cadence matrix (task 278, batch 10): one case a model
#: is essential, the rest extended.
GRID = pytest.mark.extended(reason="the closed x shape x cadence matrix")

#: The ridge here is large enough to be seen in the fit, as the oracle maps
#: it, so a short window early on reads mostly ridge, which the readiness
#: notice says; that is beside the point of these tests.
pytestmark = pytest.mark.filterwarnings("ignore::polars_online.ReadinessWarning")

FEATURES = ["x0", "x1"]
#: Seconds: the window, the half-life, the cap on a step, the clock spacing.
WINDOW, HALF_LIFE, GAP_CAP, SPACING = 1800.0, 600.0, 600.0, 60.0
RIDGE = 0.5
#: The three cadences: the clock, the rows, and both.
CADENCES = [("1m", None), (None, 25), ("1m", 25)]
IDS = ["clock", "rows", "both"]


def stream_steps(n: int = 700, seed: int = 3) -> np.ndarray:
    """Seconds between rows: ordinary rows 5 to 39 apart, six bursts of
    forty rows at most a second apart, which a clock spacing takes no extra
    snapshot in, and three two-hour gaps, which ``gap_cap`` cuts to ten
    minutes. Whole seconds, so the clock's sums are exact."""
    rng = np.random.default_rng(seed)
    steps = rng.integers(5, 40, n)
    for start in rng.choice(np.arange(20, n - 60, 50), 6, replace=False):
        steps[start : start + 40] = rng.integers(0, 2, 40)
    for gap in rng.choice(np.arange(60, n, 70), 3, replace=False):
        steps[gap] = 7200
    steps[0] = 0
    return steps


def coarse_steps() -> np.ndarray:
    """``stream_steps`` with every ordinary step rounded up to whole half
    minutes and the bursts kept: the window of thirty minutes is sixty of
    them, so rows sit exactly one window back often enough for
    ``closed`` to move a row's window on many rows."""
    steps = stream_steps()
    return np.where(steps >= 5, 30 * np.ceil(steps / 30), steps).astype(int)


def frame(steps: np.ndarray, seed: int = 4) -> pl.DataFrame:
    """Rows at those steps on a ``Datetime`` clock and on a number clock in
    seconds, two features, a target, and weights of which one row in nine is
    zero."""
    rng = np.random.default_rng(seed)
    n = len(steps)
    t0 = datetime(2024, 1, 2, 9, 30)
    seconds = np.cumsum(steps)
    x = rng.standard_normal((n, 2)) * [1.0, 2.0] + [0.5, -1.0]
    w = rng.uniform(0.5, 2.0, n)
    w[rng.random(n) < 1 / 9] = 0.0
    return pl.DataFrame(
        {
            "ts": [t0 + timedelta(seconds=int(s)) for s in seconds],
            "t": seconds.astype(float),
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": 1.0 + 2.0 * x[:, 0] - x[:, 1] + 0.3 * rng.standard_normal(n),
            "w": w,
        }
    )


def model_clock(steps: np.ndarray, gap_cap: float = GAP_CAP) -> np.ndarray:
    """The clock the model is stepped on: the first row's step is 0, and a
    longer step than ``gap_cap`` counts as the cap."""
    d = np.minimum(np.asarray(steps, dtype=float), gap_cap)
    d[0] = 0.0
    return np.cumsum(d)


def snapshot_rows(
    clock: np.ndarray, *, every: float | None, cap: int | None, window: float = WINDOW
) -> list[int]:
    """The rule, written out: the first row; then a row whose clock is
    ``every`` past the newest snapshot's, or ``cap`` rows past it, whichever
    comes first, and whatever the cadence a row whose newest snapshot has
    left the window -- a window or more old, under the default
    ``closed="right"`` (docs/PLAN.md task 196). With neither, every row."""
    if every is None and cap is None:
        cap = 1
    rows: list[int] = []
    for i, c in enumerate(clock):
        if rows:
            last = rows[-1]
            due = (
                (every is not None and c - clock[last] >= every)
                or (cap is not None and i - last >= cap)
                or c - clock[last] >= window
            )
        else:
            due = True
        if due:
            rows.append(i)
    return rows


def boundary(clock: np.ndarray, snaps: list[int], t: int, window: float = WINDOW) -> int:
    """The row of the snapshot row ``t`` subtracts: the oldest one inside the
    window of the row before it -- less than a window old, under the default
    ``closed="right"`` -- which is the ring that row left."""
    ref = clock[t - 1]
    return next(j for j in snaps if j <= t - 1 and ref - clock[j] < window)


def inside(clock: np.ndarray, w: np.ndarray, t: int, b: int) -> np.ndarray:
    """The weights of rows ``b`` to ``t - 1`` as row ``t`` reads them."""
    return w[b:t] * 0.5 ** ((clock[t - 1] - clock[b:t]) / HALF_LIFE)


def cadence(every: str | float | None, cap: int | None) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if every is not None:
        kw["snapshot_every"] = every
    if cap is not None:
        kw["max_rows_between_snapshots"] = cap
    return kw


def spacing_of(every: str | float | None) -> float | None:
    return None if every is None else (SPACING if every == "1m" else float(every))


def ew_cov(**kw):
    d: dict[str, Any] = dict(
        features=FEATURES,
        clock="ts",
        half_life="10m",
        gap_cap="10m",
        stats=["mean"],
        window_size="30m",
        min_weight=2.0,
        weight="w",
    )
    d.update(kw)
    return po.spec.ew_cov("c", **d)


def ridge(**kw):
    d: dict[str, Any] = dict(
        targets=["y"],
        features=FEATURES,
        clock="ts",
        half_life="10m",
        gap_cap="10m",
        window_size="30m",
        solve_every=0,
        ridge=RIDGE,
        min_weight=0.0,
        weight="w",
    )
    d.update(kw)
    return po.spec.ewridge("r", **d)


def lasso(**kw):
    d: dict[str, Any] = dict(
        targets=["y"],
        features=FEATURES,
        clock="ts",
        half_life="10m",
        gap_cap="10m",
        window_size="30m",
        penalties=[0.05],
        min_weight=0.0,
        weight="w",
    )
    d.update(kw)
    return po.spec.lasso("l", **d)


def marginal(**kw):
    d: dict[str, Any] = dict(
        targets=["y"],
        features=FEATURES,
        clock="ts",
        half_life="10m",
        gap_cap="10m",
        window_size="30m",
        weight="w",
    )
    d.update(kw)
    return po.spec.marginal("g", **d)


def label(df: pl.DataFrame) -> pl.DataFrame:
    """A class from the features and the noise in the target, so both
    classes stay in every window and their moments differ."""
    return df.with_columns(
        cls=pl.when(pl.col("x0") - 0.3 * pl.col("x1") + pl.col("y") * 0.2 > 0.8)
        .then(pl.lit("a"))
        .otherwise(pl.lit("b"))
    )


def ew_class(**kw):
    d: dict[str, Any] = dict(
        features=FEATURES,
        label="cls",
        classes=["a", "b"],
        clock="ts",
        half_life="10m",
        gap_cap="10m",
        window_size="30m",
        precision_prior=1.0,
        min_weight=0.0,
        weight="w",
    )
    d.update(kw)
    return po.spec.ew_class("k", **d)


def run(spec, df: pl.DataFrame) -> pl.DataFrame:
    return po.ModelBank([spec]).fit_predict(df).unnest(spec["name"])


def floats(out: pl.DataFrame) -> list[str]:
    return [c for c, t in out.schema.items() if t.is_float()]


class TestTheWindowIsTheDefinitions:
    @pytest.mark.parametrize(("every", "cap"), CADENCES, ids=IDS)
    def test_a_windowed_ew_cov_is_the_ew_mean_from_its_boundary_on(self, every, cap):
        steps = stream_steps()
        df = frame(steps)
        out = run(ew_cov(**cadence(every, cap)), df)
        clock = model_clock(steps)
        snaps = snapshot_rows(clock, every=spacing_of(every), cap=cap)
        w, x = df["w"].to_numpy(), df.select(FEATURES).to_numpy()
        got_w = out["weight_sum"].to_numpy()
        got_m = out.select([f"mean_{f}" for f in FEATURES]).to_numpy()
        means = 0
        for t in range(1, len(steps)):
            b = boundary(clock, snaps, t)
            a = inside(clock, w, t, b)
            want = a.sum()
            assert got_w[t] == pytest.approx(want, rel=1e-9, abs=1e-12), t
            if want > 2.0 * (1 + 1e-9):
                mean = (a[:, None] * x[b:t]).sum(0) / want
                assert np.allclose(got_m[t], mean, rtol=0, atol=1e-9), (t, got_m[t], mean)
                means += 1
            elif want < 2.0 * (1 - 1e-9):
                assert np.isnan(got_m[t]).all(), t
        assert means > 600, means

    @pytest.mark.parametrize(("every", "cap"), CADENCES, ids=IDS)
    def test_a_windowed_ridge_is_sklearns_ridge_on_the_rows_from_its_boundary_on(self, every, cap):
        steps = stream_steps()
        df = frame(steps)
        out = run(ridge(**cadence(every, cap)), df)
        clock = model_clock(steps)
        snaps = snapshot_rows(clock, every=spacing_of(every), cap=cap)
        w, x, y = df["w"].to_numpy(), df.select(FEATURES).to_numpy(), df["y"].to_numpy()
        got_w, got_p = out["weight_sum"].to_numpy(), out["pred_y"].to_numpy()
        checked = 0
        for t in range(1, len(steps)):
            b = boundary(clock, snaps, t)
            a = inside(clock, w, t, b)
            assert got_w[t] == pytest.approx(a.sum(), rel=1e-9, abs=1e-12), t
            # The fit row `t` reads is the last solve's, and a row of weight 0
            # never solves (hard rule 9; docs/PLAN.md task 214): it is the one
            # the last row with weight before `t` left, on its window.
            s = t
            while s > 1 and w[s - 1] <= 0.0:
                s -= 1
            b_s = boundary(clock, snaps, s)
            a_s = inside(clock, w, s, b_s)
            if t % 5 or (a_s > 0).sum() < 10:
                continue
            fit = Ridge(alpha=RIDGE * a_s.sum()).fit(x[b_s:s], y[b_s:s], sample_weight=a_s)
            want = fit.predict(x[t : t + 1])[0]
            assert got_p[t] == pytest.approx(want, rel=1e-8, abs=1e-8), (t, got_p[t], want)
            checked += 1
        assert checked > 100, checked

    @pytest.mark.parametrize(
        ("every", "cap"),
        [
            pytest.param("1m", 25, id="both"),
            pytest.param("1m", None, id="clock", marks=GRID),
            pytest.param(None, 5, id="rows", marks=GRID),
            pytest.param(None, None, id="every-row", marks=GRID),
        ],
    )
    def test_past_a_thinning_budget_the_ridge_is_sklearns_on_the_thinned_rings_rows(
        self, every, cap
    ):
        """A thinning budget keeps the newest snapshot and every second one
        before it, and doubles the cadence, each time the ring would pass it
        (``reference_paths.WindowRing``). The budget counts bytes, so the
        ring's capacity in snapshots is measured by the same budget refusing:
        on a window no row leaves, a ``{"refuse": mib}`` budget refuses at the
        row whose snapshot would not fit. Under ``{"thin": mib}`` every row's
        ``weight_sum`` is then the weight from the thinned ring's front on,
        and its prediction scikit-learn's ``Ridge`` on those rows."""
        steps = stream_steps()
        df = frame(steps)
        mib = 0.003
        cap_rows = capacity(ridge, df, mib)
        out = run(ridge(window_budget={"thin": mib}, **cadence(every, cap)), df)
        clock = model_clock(steps)
        ring = WindowRing(WINDOW, every=spacing_of(every), cap=cap, capacity=cap_rows)
        starts = np.array([ring.learn(i, c) for i, c in enumerate(clock)])
        w, x, y = df["w"].to_numpy(), df.select(FEATURES).to_numpy(), df["y"].to_numpy()
        want = ring_weights(clock, w, starts, HALF_LIFE)
        np.testing.assert_allclose(out["weight_sum"].to_numpy(), want, rtol=1e-9, atol=1e-12)
        unthinned = window_starts(clock, WINDOW, every=spacing_of(every), cap=cap)
        moved = (np.abs(ring_weights(clock, w, unthinned, HALF_LIFE) - want) > 1e-6).sum()
        print(f"capacity {cap_rows}, thinned {ring.thinned} times, {moved} rows moved")
        # Measured 2026-10-25: a capacity of 10 snapshots, thinned 2-5 times,
        # moving the weight of 342-609 of the 700 rows.
        assert ring.thinned >= 1 and moved > 100, (ring.thinned, moved)
        got_p = out["pred_y"].to_numpy()
        checked = 0
        for t in range(2, len(steps), 3):
            s = t
            while s > 1 and w[s - 1] <= 0.0:
                s -= 1
            b = starts[s - 1]
            a = w[b:s] * 0.5 ** ((clock[s - 1] - clock[b:s]) / HALF_LIFE)
            if (a > 0).sum() < 10:
                continue
            fit = Ridge(alpha=RIDGE * a.sum()).fit(x[b:s], y[b:s], sample_weight=a)
            want_p = fit.predict(x[t : t + 1])[0]
            # Measured 2026-10-25: at most 4.7e-15 relative to 1 + |pred| on
            # 230 rows a case; the unthinned test's bound.
            assert got_p[t] == pytest.approx(want_p, rel=1e-8, abs=1e-8), (t, got_p[t], want_p)
            checked += 1
        assert checked > 150, checked

    def test_the_spread_is_cut_where_the_fit_is(self):
        """``sigma`` reads the stream's own ring of the residual spread, which
        takes the model's cadence, so its boundary is the fit's: the EW root
        mean square of the out-of-sample residuals from that boundary on, each
        at its row's weight. On every row's boundary instead it is other
        numbers, so the cadence is what the test reads. The residuals are
        each slot's own, those the noise gate withholds included (a gate
        withholds output only: task 278, batch 17): the same spec's with the
        gate off."""
        steps = stream_steps()
        df = frame(steps)
        kw = dict(snapshot_every="1m", max_rows_between_snapshots=25, emit_sigma=True)
        out = run(ridge(**kw), df)
        own = run(ridge(**kw, max_error_inflation=float("inf")), df)
        clock = model_clock(steps)
        w, r = df["w"].to_numpy(), own["resid_y"].to_numpy()

        def sigma(snaps: list[int]) -> np.ndarray:
            got = np.full(len(steps), np.nan)
            for t in range(2, len(steps)):
                b = boundary(clock, snaps, t)
                a = inside(clock, w, t, b) * np.isfinite(r[b:t])
                if a.sum() > 0:
                    got[t] = np.sqrt((a * np.nan_to_num(r[b:t]) ** 2).sum() / a.sum())
            return got

        want = sigma(snapshot_rows(clock, every=SPACING, cap=25))
        got = out["sigma_y"].to_numpy()
        assert np.isfinite(want[2:]).sum() > 600
        assert np.allclose(got, want, rtol=1e-9, atol=0, equal_nan=True)
        every_row = sigma(snapshot_rows(clock, every=None, cap=None))
        assert np.nanmax(np.abs(every_row - want) / want) > 1e-3

    @pytest.mark.parametrize(("every", "cap"), [CADENCES[0], CADENCES[2]], ids=["clock", "both"])
    @pytest.mark.parametrize(
        "make",
        [ew_cov, ridge, lasso, marginal, ew_class],
        ids=["ew_cov", "ewridge", "lasso", "marginal", "ew_class"],
    )
    def test_the_effective_window_is_inside_the_window_less_one_spacing(self, every, cap, make):
        """Every row at most ``window_size - snapshot_every`` old is inside the
        window and none older than ``window_size`` is, read from the model's
        own ``weight_sum`` against the weights of those two sets of rows."""
        steps = stream_steps()
        df = label(frame(steps))
        got = run(make(**cadence(every, cap)), df)["weight_sum"].to_numpy()
        clock, w = model_clock(steps), df["w"].to_numpy()
        tight = 0
        for t in range(1, len(steps)):
            age = clock[t - 1] - clock[:t]
            a = w[:t] * 0.5 ** (age / HALF_LIFE)
            least, most = a[age <= WINDOW - SPACING].sum(), a[age <= WINDOW].sum()
            assert least * (1 - 1e-9) - 1e-12 <= got[t] <= most * (1 + 1e-9) + 1e-12, t
            tight += got[t] < most * (1 - 1e-6)
        assert tight > 100, "the case needs rows the spacing drops"

    def test_the_case_needs_the_cap_and_the_bursts(self):
        """The stream is one where both matter, so the tests above that hold
        to it are evidence: on the uncapped clock the windows hold other
        rows (a two-hour gap would empty them, where ten minutes leaves two
        thirds), and inside the bursts a clock spacing takes far fewer
        snapshots than a row cap of three does."""
        steps = stream_steps()
        w = frame(steps)["w"].to_numpy()

        def weights(clock: np.ndarray) -> np.ndarray:
            snaps = snapshot_rows(clock, every=SPACING, cap=None)
            return np.array(
                [inside(clock, w, t, boundary(clock, snaps, t)).sum() for t in range(1, len(steps))]
            )

        assert np.abs(weights(model_clock(steps)) - weights(model_clock(steps, np.inf))).max() > 1
        capped = snapshot_rows(model_clock(steps), every=SPACING, cap=None)
        burst = set(np.flatnonzero(steps <= 1))
        by_rows = snapshot_rows(model_clock(steps), every=None, cap=3)
        assert len(set(capped) & burst) * 4 < len(set(by_rows) & burst)


MODELS = {
    "ew_cov": ew_cov,
    "ewridge": ridge,
    "lasso": lasso,
    "marginal": marginal,
    "ew_class": ew_class,
}


def matrix(essential: tuple[Any, ...]) -> list[Any]:
    """Every model x edge x cadence, ``essential`` the one case a model the
    essentials run."""
    cases = []
    for name in MODELS:
        for closed in ("right", "both"):
            for (every, cap), cid in zip(CADENCES, IDS, strict=True):
                case = (name, closed, every, cap)
                marks = () if (closed, every, cap) == essential else (GRID,)
                cases.append(pytest.param(*case, id=f"{name}-{closed}-{cid}", marks=marks))
    return cases


def capacity(make, df: pl.DataFrame, mib: float) -> int:
    """How many of ``make``'s snapshots a budget of ``mib`` holds: the row at
    which ``{"refuse": mib}`` refuses, fed a row at a time with a snapshot on
    every row and a window no row leaves, so each row adds one snapshot."""
    bank = po.ModelBank(
        [make(window_size="1000d", snapshot_every=0, window_budget={"refuse": mib})]
    )
    for i in range(df.height):
        try:
            bank.fit_predict(df.slice(i, 1))
        except ValueError as e:
            assert "window_budget" in str(e), e
            return i
    raise AssertionError(f"{mib} MiB held every row's snapshot")


def ring_weights(clock: np.ndarray, w: np.ndarray, starts: np.ndarray, hl: float) -> np.ndarray:
    """Each row's ``weight_sum``: the weights of the rows the ring kept after
    the row before it, ``starts[t - 1]`` to ``t - 1``, aged from that row."""
    got = np.zeros(len(clock))
    for t in range(1, len(clock)):
        b = starts[t - 1]
        got[t] = (w[b:t] * 0.5 ** ((clock[t - 1] - clock[b:t]) / hl)).sum()
    return got


class TestEveryWindowedModelsRing:
    """Every windowed model under each cadence and each edge reads its window
    from the ring's front snapshot (``reference_paths.WindowRing``, the rule
    the module docstring states, with ``closed="both"`` keeping a snapshot
    exactly one window old): its ``weight_sum`` is the weight of the rows from
    there on, and under ``closed="both"`` and a clock cadence ``ew_class``'s
    posterior and ``marginal``'s table are scipy's and numpy's of those rows.
    The stream is ``coarse_steps``, on which the edge decides many rows.
    Before task 278 (batch 10) only ``ew_cov`` and ``ewridge`` were held to a
    cadence, and ``max_rows_between_snapshots`` was shown taking effect on
    neither ``lasso`` nor ``ew_class``."""

    @pytest.mark.parametrize(("name", "closed", "every", "cap"), matrix(("both", "1m", 25)))
    def test_weight_sum_is_the_weight_from_the_rings_front_on(self, name, closed, every, cap):
        steps = coarse_steps()
        df = label(frame(steps))
        out = run(MODELS[name](closed=closed, **cadence(every, cap)), df)
        clock, w = model_clock(steps), df["w"].to_numpy()
        starts = window_starts(clock, WINDOW, every=spacing_of(every), cap=cap, closed=closed)
        want = ring_weights(clock, w, starts, HALF_LIFE)
        got = out["weight_sum"].to_numpy()
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
        # Evidence the case reads the cadence and the edge: the plain window
        # (every row a snapshot) and the other edge are other weights.
        plain = ring_weights(clock, w, window_starts(clock, WINDOW, closed=closed), HALF_LIFE)
        other = "right" if closed == "both" else "both"
        edge = window_starts(clock, WINDOW, every=spacing_of(every), cap=cap, closed=other)
        moved = (np.abs(plain - want) > 1e-6).sum()
        edged = (np.abs(ring_weights(clock, w, edge, HALF_LIFE) - want) > 1e-6).sum()
        # Measured 2026-10-25 (every model to 1.7e-15 relative): the cadence
        # moves 152-637 of the 700 rows' weights, and the edge 7-122.
        assert moved > 100, moved
        assert edged >= 5, edged

    @pytest.mark.parametrize(
        ("every", "cap"),
        [
            pytest.param("1m", 25, id="both"),
            pytest.param("1m", None, id="clock", marks=GRID),
            pytest.param(None, 25, id="rows", marks=GRID),
        ],
    )
    def test_ew_class_is_the_scipy_posterior_from_the_rings_front_on(self, every, cap):
        """At ``half_life = inf`` and a tiny precision prior the posterior is
        the two weighted Gaussians of the rows the ring keeps, each class's
        prior its share of their weight."""
        steps = coarse_steps()
        df = label(frame(steps))
        prior = 1e-12
        spec = ew_class(
            half_life=float("inf"), precision_prior=prior, closed="both", **cadence(every, cap)
        )
        out = run(spec, df)
        clock = model_clock(steps)
        x, w = df.select(FEATURES).to_numpy(), df["w"].to_numpy()
        lab = df["cls"].to_numpy()
        starts = window_starts(clock, WINDOW, every=spacing_of(every), cap=cap, closed="both")
        plain = window_starts(clock, WINDOW, closed="both")

        def posterior(t: int, b: int) -> float | None:
            rows, wr, lr = x[b:t], w[b:t], lab[b:t]
            logp = []
            for c in ("a", "b"):
                wc = wr[lr == c]
                if (wc > 0).sum() < 5:
                    return None
                mean = np.average(rows[lr == c], axis=0, weights=wc)
                cov = np.cov(rows[lr == c].T, aweights=wc, ddof=0) + prior * np.eye(2)
                logp.append(
                    np.log(wc.sum() / wr.sum()) + multivariate_normal(mean, cov).logpdf(x[t])
                )
            p = np.exp(np.array(logp) - max(logp))
            return float(p[0] / p.sum())

        checked, apart = 0, 0.0
        for t in range(2, len(steps), 2):
            want = posterior(t, starts[t - 1])
            if want is None:
                continue
            # Measured 2026-10-25: at most 2.3e-12 from scipy over the three
            # cadences, the every-row window 0.18-0.99 away; 1e-9 is the bound.
            assert out["p_a"][t] == pytest.approx(want, abs=1e-9), t
            checked += 1
            every_row = posterior(t, plain[t - 1])
            if every_row is not None:
                apart = max(apart, abs(every_row - want))
        print(f"{every} {cap}: {checked} rows; the every-row window moves p_a by {apart:.3g}")
        assert checked > 250, checked
        assert apart > 1e-3, apart

    @pytest.mark.parametrize(
        ("every", "cap"),
        [
            pytest.param("1m", 25, id="both"),
            pytest.param("1m", None, id="clock", marks=GRID),
            pytest.param(None, 25, id="rows", marks=GRID),
        ],
    )
    def test_the_marginal_table_is_numpy_from_the_rings_front_on(self, every, cap):
        """The table read after the last row is numpy's weighted moments of
        the rows from the ring's front on, each at ``w * 0.5 ** (age /
        half_life)``. The stream ends at rows between snapshots, where the
        every-row table is another one (a row that takes a snapshot is read
        the same way under every cadence)."""
        steps = coarse_steps()
        df = label(frame(steps))
        clock = model_clock(steps)
        starts = window_starts(clock, WINDOW, every=spacing_of(every), cap=cap, closed="both")
        plain = window_starts(clock, WINDOW, closed="both")
        x, y, w = df["x0"].to_numpy(), df["y"].to_numpy(), df["w"].to_numpy()

        def table(n: int, b: int) -> dict[str, float]:
            a = w[b:n] * 0.5 ** ((clock[n - 1] - clock[b:n]) / HALF_LIFE)
            cov = np.cov(np.vstack([x[b:n], y[b:n]]), aweights=a, ddof=0)
            return {
                "weight_sum": a.sum(),
                "mean_x": np.average(x[b:n], weights=a),
                "mean_y": np.average(y[b:n], weights=a),
                "var_x": cov[0, 0],
                "cov": cov[0, 1],
                "beta": cov[0, 1] / cov[0, 0],
                "corr": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
            }

        ends = [
            n
            for n in range(150, len(steps), 11)
            if abs(table(n, starts[n - 1])["weight_sum"] - table(n, plain[n - 1])["weight_sum"])
            > 1e-3
        ]
        assert len(ends) >= 5, ends
        for n in ends[:5]:
            bank = po.ModelBank([marginal(closed="both", **cadence(every, cap))])
            bank.fit_predict(df.head(n))
            got = bank.marginal("g").filter(pl.col("feature") == "x0").row(0, named=True)
            want = table(n, starts[n - 1])
            for key, value in want.items():
                # Measured 2026-10-25: at most 1.4e-15 relative; 1e-10 is the bound.
                assert got[key] == pytest.approx(value, rel=1e-10, abs=1e-12), (n, key)


class TestTheCadence:
    @pytest.mark.parametrize(
        "kw",
        [
            {"snapshot_every": 0},
            {"max_rows_between_snapshots": 1},
            {"snapshot_every": "1m", "max_rows_between_snapshots": 1},
            {"snapshot_every": 0, "max_rows_between_snapshots": 25},
        ],
        ids=["every=0", "cap=1", "1m,cap=1", "0,cap=25"],
    )
    def test_every_row_is_the_default(self, kw):
        df = frame(stream_steps())
        want = run(ew_cov(), df)
        got = run(ew_cov(**kw), df)
        for f in floats(want):
            assert np.array_equal(got[f].to_numpy(), want[f].to_numpy(), equal_nan=True), f

    def test_without_a_clock_column_window_every_counts_rows_as_before(self):
        """A spec without a clock reads the row's number as its clock, so a
        number ``snapshot_every`` is that many rows, as it was before task
        162: the same as a row cap of as many, to the bit."""
        steps = stream_steps()
        df = frame(steps)
        base: dict[str, Any] = dict(clock=None, gap_cap=None, half_life=40.0, window_size=120.0)
        by_clock = run(ew_cov(**base, snapshot_every=5), df)
        by_rows = run(ew_cov(**base, max_rows_between_snapshots=5), df)
        for f in floats(by_clock):
            a, b = by_clock[f].to_numpy(), by_rows[f].to_numpy()
            assert np.array_equal(a, b, equal_nan=True), f
        rows = np.arange(len(steps), dtype=float)
        snaps = snapshot_rows(rows, every=5.0, cap=None, window=120.0)
        assert set(np.diff(snaps)) == {5}
        w, got = df["w"].to_numpy(), by_clock["weight_sum"].to_numpy()
        for t in range(1, len(steps)):
            b = boundary(rows, snaps, t, 120.0)
            want = (w[b:t] * 0.5 ** ((rows[t - 1] - rows[b:t]) / 40.0)).sum()
            assert got[t] == pytest.approx(want, rel=1e-9, abs=1e-12), t

    def test_a_number_clock_counts_its_own_units(self):
        """The same stream on a number clock in seconds, every parameter a
        number of them, is the temporal spec to the bit."""
        df = frame(stream_steps())
        s = ew_cov(
            clock="t",
            half_life=HALF_LIFE,
            gap_cap=GAP_CAP,
            window_size=WINDOW,
            snapshot_every=SPACING,
        )
        temporal = run(ew_cov(snapshot_every="1m"), df)
        number = run(s, df)
        for f in floats(temporal):
            a, b = temporal[f].to_numpy(), number[f].to_numpy()
            assert np.array_equal(a, b, equal_nan=True), f


class TestChunkingAndResume:
    @pytest.mark.parametrize("make", [ew_cov, ridge], ids=["ew_cov", "ewridge"])
    def test_the_window_does_not_depend_on_the_chunking(self, make):
        df = frame(stream_steps())
        s = make(snapshot_every="1m", max_rows_between_snapshots=25)
        whole = run(s, df)
        for rows in (1, 7, 37):
            parts = pl.concat(po.ModelBank([s]).fit_predict_batches(df, chunk_size=rows))
            parts = parts.unnest(s["name"])
            for f in floats(whole):
                a, b = whole[f].to_numpy(), parts[f].to_numpy()
                assert np.array_equal(a, b, equal_nan=True), (rows, f)

    @pytest.mark.parametrize("make", [ew_cov, ridge], ids=["ew_cov", "ewridge"])
    def test_a_save_between_snapshots_resumes_where_the_unbroken_run_does(self, make):
        """Cut where the newest snapshot is part of a spacing behind the last
        row learned, so the resumed run must know the spacing and the clock it
        is measured from."""
        steps = stream_steps()
        df = frame(steps)
        s = make(snapshot_every="1m", max_rows_between_snapshots=25)
        whole = run(s, df)
        clock = model_clock(steps)
        snaps = snapshot_rows(clock, every=SPACING, cap=25)

        def between(cut: int) -> bool:
            last = max(j for j in snaps if j <= cut - 1)
            return 0 < clock[cut - 1] - clock[last] < SPACING and cut not in snaps

        cuts = [next(c for c in range(start, len(steps)) if between(c)) for start in (90, 333)]
        for cut in cuts:
            first = po.ModelBank([s])
            first.fit_predict(df.head(cut))
            resumed = po.ModelBank.load_bytes(first.save_bytes(), specs=[s])
            tail = resumed.fit_predict(df.slice(cut)).unnest(s["name"])
            for f in floats(whole):
                a, b = whole[f].to_numpy()[cut:], tail[f].to_numpy()
                assert np.array_equal(a, b, equal_nan=True), (cut, f)


class TestRefusals:
    def test_a_temporal_spec_refuses_a_plain_number(self):
        with pytest.raises(ValueError, match="snapshot_every is a plain number"):
            run(ew_cov(snapshot_every=60.0), frame(stream_steps()))

    def test_a_number_spec_refuses_a_duration(self):
        """The mirror: a spec in plain numbers, read on a number clock, with
        ``snapshot_every`` a duration, is one spec mixing the two."""
        with pytest.raises(ValueError, match="snapshot_every is a duration"):
            ew_cov(
                clock="t",
                half_life=HALF_LIFE,
                gap_cap=GAP_CAP,
                window_size=WINDOW,
                snapshot_every="1m",
            )

    @pytest.mark.parametrize("bad", [-5.0, -1])
    def test_a_negative_spacing_is_refused(self, bad):
        with pytest.raises(ValueError, match="snapshot_every must be finite and >= 0 clock units"):
            ew_cov(clock=None, gap_cap=None, half_life=40.0, window_size=120.0, snapshot_every=bad)

    @pytest.mark.parametrize("bad", [-1, 0])
    def test_a_row_cap_of_no_rows_is_refused(self, bad):
        """A cap of no rows is no schedule; `snapshot_every = 0` is every row
        (docs/PLAN.md task 196, U7)."""
        with pytest.raises(ValueError, match=f"max_rows_between_snapshots must be >= 1, got {bad}"):
            ew_cov(max_rows_between_snapshots=bad)

    @pytest.mark.parametrize(
        "key,value", [("snapshot_every", "1m"), ("max_rows_between_snapshots", 5)]
    )
    @pytest.mark.parametrize("make", [ew_cov, ridge], ids=["ew_cov", "ewridge"])
    def test_each_needs_a_window(self, make, key, value):
        with pytest.raises(ValueError, match=f"{key} needs `window_size`"):
            make(window_size=None, **{key: value})

    @pytest.mark.parametrize(
        "kw,named",
        [
            ({"snapshot_every": "1m"}, ["raise snapshot_every (1m now)"]),
            ({"max_rows_between_snapshots": 25}, ["raise max_rows_between_snapshots (25 now)"]),
            (
                {"snapshot_every": "1m", "max_rows_between_snapshots": 25},
                ["snapshot_every (1m now) and max_rows_between_snapshots (25 now)"],
            ),
            ({}, ["snapshot_every", "max_rows_between_snapshots", "every row now"]),
        ],
        ids=["clock", "rows", "both", "neither"],
    )
    def test_the_refusal_past_a_budget_names_the_cadence_in_force(self, kw, named):
        df = frame(stream_steps())
        bank = po.ModelBank([ridge(window_budget={"refuse": 0.001}, **kw)])
        with pytest.raises(ValueError, match="window_budget") as e:
            bank.fit_predict(df)
        for words in named:
            assert words in str(e.value), (words, str(e.value))


class TestTheJsonExport:
    @pytest.mark.parametrize(
        "kw,spacing",
        [
            ({}, "inf"),
            ({"max_rows_between_snapshots": 25}, "inf"),
            ({"snapshot_every": "1m"}, SPACING),
            ({"snapshot_every": "1m", "max_rows_between_snapshots": 25}, SPACING),
        ],
        ids=["neither", "rows", "clock", "both"],
    )
    @pytest.mark.parametrize("make", [ew_cov, ridge], ids=["ew_cov", "ewridge"])
    def test_the_export_keeps_the_rings_spacing(self, make, kw, spacing):
        """``to_json`` reads its own text back and refuses one that is not the
        state, so a spacing of none, an infinity JSON has no literal for, must
        be written as the word (``humanfloat``): every ring here holds one
        but where a clock spacing is given."""
        bank = po.ModelBank([make(**kw)])
        bank.fit_predict(frame(stream_steps()))
        found: list[Any] = []

        def walk(v: Any) -> None:
            if isinstance(v, dict):
                for k, x in v.items():
                    if k == "spacing":
                        found.append(x)
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(json.loads(bank.write_json()))
        assert found and all(s == spacing for s in found), found
