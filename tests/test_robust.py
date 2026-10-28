"""Task 13: robust models - Huber and quantile (docs/PLAN.md section 4.5)."""

import datetime as dt
import json
import math
import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import EwMedian

TIER = "mixed"


def _pred(out, col="pred_y0"):
    return out["m"].struct.field(col).to_numpy().astype(float)


def test_huber_beats_least_squares_under_outliers():
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.standard_normal(n)
    y = 2.0 * x + 0.1 * rng.standard_normal(n)
    contaminated = rng.random(n) < 0.03
    y[contaminated] = 300.0 * rng.standard_normal(contaminated.sum())
    df = pl.DataFrame({"x0": x, "y0": y})
    clean = 2.0 * x  # what a perfect model would predict

    common = dict(
        targets=["y0"],
        features=["x0"],
        half_life=1e9,
        min_weight=10.0,
        # The oracle below refits every row; the default cadence solves by
        # weight, so ask for every row.
        max_rows_between_solves=1,
    )
    hub = po.ModelBank([po.spec.huber("m", huber_delta=1.5, **common)]).fit_predict(df)
    ols = po.ModelBank([po.spec.ewridge("m", ridge=1e-8, **common)]).fit_predict(df)

    ok = np.isfinite(_pred(hub)) & np.isfinite(_pred(ols)) & ~contaminated
    e_hub = np.mean((_pred(hub)[ok] - clean[ok]) ** 2)
    e_ols = np.mean((_pred(ols)[ok] - clean[ok]) ** 2)
    assert e_hub < e_ols, f"huber {e_hub} should beat ols {e_ols}"


def test_huge_delta_reduces_to_least_squares():
    df, _ = synthetic(seed=61, n_groups=1, n_rows=300, k=2, null_frac=0.0)
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        half_life=1e9,
        min_weight=10.0,
        max_rows_between_solves=1,
    )
    a = po.ModelBank([po.spec.huber("m", huber_delta=1e9, **common)]).fit_predict(df)
    b = po.ModelBank([po.spec.ewridge("m", ridge=1e-6, **common)]).fit_predict(df)
    pa, pb = _pred(a), _pred(b)
    m = np.isfinite(pa) & np.isfinite(pb)
    assert np.max(np.abs(pa[m] - pb[m])) < 1e-6


@pytest.mark.parametrize("standardize", [False, True], ids=["plain", "standardized"])
@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "origin"])
def test_the_targets_units_do_not_matter(fit_intercept, standardize):
    """Task 177. Scaling every target by ``c`` scales every prediction by
    ``c``. Powers of two keep each operation exact, so the comparison is to
    the bit, nulls in the same places. Until a target had a residual scale,
    its Huber cut was ``huber_delta`` times the literal 1, in the target's
    units, so a target in millions had its first predicted rows down-weighted
    as outliers and one in millionths none, and every prediction after them
    moved with the units. Now no row is down-weighted before the scale
    exists. The first target opens on twenty rows a group at exactly zero, so
    its residuals are exactly zero until it moves."""
    df, _ = synthetic(seed=47, n_groups=2, n_rows=200, k=3, n_targets=2, null_frac=0.05)
    opening = pl.int_range(pl.len()).over("group") < 20
    df = df.with_columns(pl.when(opening).then(0.0).otherwise(pl.col("y0")).alias("y0"))

    def preds(c: float) -> dict[str, np.ndarray]:
        spec = po.spec.huber(
            "m",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            group="group",
            clock="t",
            gap_cap=50.0,
            weight="w",
            half_life=300.0,
            fit_intercept=fit_intercept,
            standardize=standardize,
        )
        out = po.ModelBank([spec]).fit_predict(df.with_columns(pl.col("y0", "y1") * c))
        return {t: _pred(out, f"pred_{t}") for t in ("y0", "y1")}

    base = preds(1.0)
    assert all(np.isfinite(p).sum() > 250 for p in base.values())
    for c in (2.0**20, 2.0**-20):
        for t, got in preds(c).items():
            np.testing.assert_array_equal(got, base[t] * c, err_msg=f"c = {c}, {t}")


@pytest.mark.parametrize(("tau", "min_weight"), [(0.9, None), (0.5, 20.0), (0.25, 20.0)])
def test_the_quantile_targets_units_do_not_matter(tau, min_weight):
    """Task 177, for the quantile loss. Its band is ``quantile_eps``, or the
    floor, times ``s``, and ``s`` was the literal 1 until the target had a
    residual scale, so a band drawn before one was in the target's units.
    Such a band is drawn past the warm-up only: on a target's first
    prediction when ``min_weight`` holds it back past three rows per
    coefficient, and while every residual is exactly zero, as on the first
    target's thirty rows a group at zero. Now such a row is a least-squares
    row, as in the warm-up. Scaling every target by ``c`` scales every
    prediction by ``c``, to the bit."""
    df, _ = synthetic(seed=48, n_groups=2, n_rows=200, k=3, n_targets=2, null_frac=0.05)
    opening = pl.int_range(pl.len()).over("group") < 30
    df = df.with_columns(pl.when(opening).then(0.0).otherwise(pl.col("y0")).alias("y0"))

    def preds(c: float) -> dict[str, np.ndarray]:
        spec = po.spec.quantile(
            "m",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            group="group",
            clock="t",
            gap_cap=50.0,
            weight="w",
            half_life=300.0,
            quantile=tau,
            min_weight=min_weight,
        )
        out = po.ModelBank([spec]).fit_predict(df.with_columns(pl.col("y0", "y1") * c))
        return {t: _pred(out, f"pred_{t}") for t in ("y0", "y1")}

    base = preds(1.0)
    assert all(np.isfinite(p).sum() > 250 for p in base.values())
    for c in (2.0**20, 2.0**-20):
        for t, got in preds(c).items():
            np.testing.assert_array_equal(got, base[t] * c, err_msg=f"c = {c}, {t}")


def test_quantile_levels_are_ordered():
    rng = np.random.default_rng(12)
    n = 6000
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": 1.0 + 2.0 * rng.random(n)})
    common = dict(
        targets=["y0"],
        features=["x0"],
        half_life=1e9,
        min_weight=20.0,
        max_rows_between_solves=1,
    )
    preds = {}
    for tau in (0.1, 0.5, 0.9):
        out = po.ModelBank([po.spec.quantile("m", quantile=tau, **common)]).fit_predict(df)
        preds[tau] = np.nanmean(_pred(out))
    assert preds[0.1] < preds[0.5] < preds[0.9], preds


def test_quantile_coverage_is_the_level_asked_for():
    rng = np.random.default_rng(13)
    n = 8000
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n)})
    out = po.ModelBank(
        [
            po.spec.quantile(
                "m",
                quantile=0.9,
                targets=["y0"],
                features=["x0"],
                half_life=2000.0,
                min_weight=50.0,
                max_rows_between_solves=1,
            )
        ]
    ).fit_predict(df)
    p = _pred(out)
    y = df["y0"].to_numpy()
    m = np.isfinite(p)
    below = (y[m] < p[m]).mean()
    settled = m & (np.arange(n) >= n // 2)
    after = (y[settled] < p[settled]).mean()
    # A tail quantile on 8 000 rows is still arriving -- at tau = 0.9 the band
    # holds a ninth of them -- so the whole stream reads a little under the
    # level and its second half reads the level: 0.858 and 0.893 measured,
    # where the frozen IRLS weights this replaced read 0.777 and 0.855 (N9).
    assert 0.84 < below < 0.90, below
    assert 0.87 < after < 0.92, after


def test_quantile_predicts_every_row_under_a_finite_halflife():
    """The per-target ``min_weight`` gate reads the rows a target was present
    on (hard rule 8, S2). From N9 to the second review's F1 the quantile fit
    reported its band's weight instead, which a half-life caps at the band's
    share of the effective sample -- a ninth of it at ``tau = 0.9`` -- so
    a ``min_weight`` above that share closed the gate again after the first
    predictions: at ``half_life = 100`` and ``min_weight = 20`` the rows with
    a prediction ran 178, 233, 2, 88, 155 and 203 per thousand, where
    ``huber`` predicted every one. Once the gate opens, it stays open."""
    rng = np.random.default_rng(13)
    n = 6000
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n)})
    spec = po.spec.quantile(
        "m",
        quantile=0.9,
        targets=["y0"],
        features=["x0"],
        half_life=100.0,
        min_weight=20.0,
        max_rows_between_solves=1,
    )
    p = _pred(po.ModelBank([spec]).fit_predict(df))
    finite = np.isfinite(p)
    # The decayed weight reaches 20 a row or two after row 20; from there the
    # gate stays open.
    first = int(np.argmax(finite))
    assert first <= 25, first
    assert finite[first:].all(), np.flatnonzero(~finite[first:])[:5] + first


@pytest.mark.parametrize("half_life", [30.0, 40.0])
def test_quantile_hits_its_level_under_a_short_halflife(half_life):
    """A short half-life leaves a tail quantile's band few rows: at ``tau =
    0.9`` and ``half_life = 30`` about three rows' weight against a warm-up
    bar of six, so the fit kept falling back into warm-up -- least-squares
    rows aimed at the mean -- and its coverage read 0.825, and 0.864 at 40
    (the second review's F3). The warm-up reads the rows present now, and
    the band widens as the effective sample shrinks, so the level holds."""
    n = 20000
    rng = np.random.default_rng(13)
    df = pl.DataFrame({"x0": rng.standard_normal(n), "y0": rng.standard_normal(n)})
    spec = po.spec.quantile(
        "m",
        quantile=0.9,
        targets=["y0"],
        features=["x0"],
        half_life=half_life,
        min_weight=5.0,
        max_rows_between_solves=1,
    )
    p = _pred(po.ModelBank([spec]).fit_predict(df))
    y = df["y0"].to_numpy()
    half = np.isfinite(p) & (np.arange(n) >= n // 2)
    coverage = (y[half] < p[half]).mean()
    assert 0.87 < coverage < 0.93, coverage


def test_out_of_sample_on_noise():
    rng = np.random.default_rng(14)
    n = 3000
    df = pl.DataFrame(
        {"x0": rng.standard_normal(n), "x1": rng.standard_normal(n), "y0": rng.standard_normal(n)}
    )
    out = po.ModelBank(
        [
            po.spec.huber(
                "m",
                targets=["y0"],
                features=["x0", "x1"],
                half_life=300.0,
                min_weight=20.0,
                max_rows_between_solves=1,
            )
        ]
    ).fit_predict(df)
    p = _pred(out)
    m = np.isfinite(p)
    ic = np.corrcoef(p[m], df["y0"].to_numpy()[m])[0, 1]
    assert abs(ic) < 0.06


def _plumbing_case():
    df, _ = synthetic(seed=62, n_groups=2, n_rows=200, k=2, null_frac=0.0)
    kw = dict(
        targets=["y0"],
        features=["x0", "x1"],
        half_life=300.0,
        clock="t",
        gap_cap=50.0,
        min_weight=10.0,
        huber_delta=1.5,
    )
    return df, kw, po.spec.huber("m", group="group", **kw)


def test_chunk_invariance():
    df, _, spec = _plumbing_case()
    one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
    bank = po.ModelBank([spec])
    many = (
        pl.concat([bank.fit_predict(df.slice(i, 25)) for i in range(0, df.height, 25)])
        .select("m")
        .unnest("m")
    )
    # `coef`, and `support_coef` beside it since task 116, are written on each
    # chunk's last row, so they follow the chunking by design.
    keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
    assert one.select(keep).equals(many.select(keep), null_equal=True)


def test_bad_config_rejected():
    with pytest.raises(ValueError, match="quantile"):
        po.spec.quantile("m", quantile=0.0, targets=["y0"], features=["x0"], half_life=10.0)
    with pytest.raises(ValueError, match="huber_delta"):
        po.spec.huber("m", huber_delta=-1.0, targets=["y0"], features=["x0"], half_life=10.0)


@pytest.mark.parametrize("standardize", [False, True], ids=["plain", "standardized"])
def test_a_level_costs_the_fit_nothing(standardize):
    """Huber at ``huber_delta = 1e9`` is least squares, and least squares is
    invariant to a shift of every column: the same stream at a level of
    ``1e8`` must give the same slopes, and the same predictions less the
    level. ``robust`` kept raw cross-moments, ``E[z·y]``, and solved the raw
    normal equations (or centred them by subtraction), which loses
    ``level²·ε`` -- the whole fit at ``1e8`` -- where ``ewridge`` and
    ``lasso`` were moved to centred cross-moments in the 2026-09-12 round
    (review 2026-09-18, S2)."""
    rng = np.random.default_rng(21)
    n = 2000
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    y = 1.5 * x0 - 0.5 * x1 + 0.3 * rng.standard_normal(n)
    common = dict(
        targets=["y0"],
        features=["x0", "x1"],
        half_life=1e9,
        min_weight=10.0,
        max_rows_between_solves=1,
        coef_every=1,
        huber_delta=1e9,
        standardize=standardize,
    )

    def fit(level):
        df = pl.DataFrame({"x0": x0 + level, "x1": x1 + level, "y0": y + level})
        out = po.ModelBank([po.spec.huber("m", **common)]).fit_predict(df)
        return np.array(out["m"].struct.field("coef").to_list()[-1]), _pred(out)

    coef0, pred0 = fit(0.0)
    assert coef0[1:] == pytest.approx([1.5, -0.5], abs=0.05), "the fixture is not what it claims"
    # Either sign of the level (docs/PLAN.md task 209 (b)).
    for level in (1e8, -1e8):
        coef8, pred8 = fit(level)
        assert coef8[1:] == pytest.approx(coef0[1:], abs=1e-6), (level, coef8, coef0)
        m = np.isfinite(pred0) & np.isfinite(pred8)
        assert m.sum() > n - 20
        assert np.max(np.abs((pred8[m] - level) - pred0[m])) < 1e-5, level


def test_a_level_costs_the_quantile_fit_nothing():
    """The same shift invariance for the quantile loss, whose nudge enters
    the same cross-moment (S2)."""
    rng = np.random.default_rng(22)
    n = 3000
    x0 = rng.standard_normal(n)
    y = 2.0 * x0 + rng.standard_normal(n)
    common = dict(
        targets=["y0"],
        features=["x0"],
        half_life=1e9,
        min_weight=20.0,
        max_rows_between_solves=1,
        coef_every=1,
        quantile=0.75,
    )

    def fit(level):
        df = pl.DataFrame({"x0": x0 + level, "y0": y + level})
        out = po.ModelBank([po.spec.quantile("m", **common)]).fit_predict(df)
        return np.array(out["m"].struct.field("coef").to_list()[-1]), _pred(out)

    coef0, pred0 = fit(0.0)
    assert coef0[1] == pytest.approx(2.0, abs=0.1), "the fixture is not what it claims"
    # Either sign of the level (docs/PLAN.md task 209 (b)).
    for level in (1e8, -1e8):
        coef8, pred8 = fit(level)
        assert coef8[1] == pytest.approx(coef0[1], abs=1e-6), (level, coef8, coef0)
        m = np.isfinite(pred0) & np.isfinite(pred8)
        assert np.max(np.abs((pred8[m] - level) - pred0[m])) < 1e-5, level


class TestTheQuantileFitsDefinition:
    """The quantile fit held to the problem it solves, not to a replay of its
    arithmetic (`tests/reference.py`'s `robust_ref` restates the core; review
    2026-10-05, TC1). `robust.rs`'s module doc defines each row's part in
    the Newton system: under three rows per coefficient, or a band holding
    under one row per coefficient, or before ``s`` is above 0, a
    least-squares row (task 177: a band needs a scale); inside the band of
    half-width ``h = s max(quantile_eps, (k/n)^(2/5))`` a least-squares row
    with target ``y + 2h(tau - 1/2)``; outside it a nudge ``2h psi(r) z``
    into the cross-moment only, with ``psi = tau - 1{r < 0}``, its step
    bounded by ``|r| / (1 + u'A^-1 u)``, the row's full leverage against the
    band's system: ``A`` the band rows' centred covariance plus the ridge,
    ``u`` the row's centred features (TC1b). ``s`` is the MAD of the scored
    residuals before the row, their EW median over ``Phi^-1(3/4)`` on the
    sketch's buckets (``reference.EwMedian``, written from the sketch's
    definition; the RMS until task 278's D-Q), and ``n`` the target's rows,
    both decayed, and ``r`` the residual the row was scored with."""

    EPS, RIDGE = 0.2, 1e-6

    @staticmethod
    def stream(n=3000, rho=0.0, seed=5):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(n)
        b = rho * a + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
        x = np.column_stack([a, b])
        y = 1.0 + x @ np.array([0.8, -0.4]) + rng.standard_exponential(n) - 1.0
        return x, y

    def fit(self, x, y, tau, half_life):
        spec = po.spec.quantile(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            quantile=tau,
            half_life=half_life,
            max_rows_between_solves=1,
            min_weight=0.0,
            quantile_eps=self.EPS,
            ridge=self.RIDGE,
            coef_every=1,
            # The one pass's rules: the refit of the first rows rebuilds the
            # sums at the converged fit, held to its own definition in
            # `tests/test_quantile_refit.py` (task 278, D-Q2).
            quantile_refit_rows=0,
        )
        df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y})
        out = po.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        pred = out["pred_y0"].fill_null(float("nan")).to_numpy()
        coef = np.array([c if c is not None else [np.nan] * 3 for c in out["coef"].to_list()])
        return pred, coef

    def rows(self, x, y, pred, tau, half_life):
        """Each row's part by the module doc's rules, from the rows and the
        bank's own predictions: ``"ols"``, ``"band"`` (with ``h``) or
        ``"nudge"`` (with its step, as the sum form takes it)."""
        n = len(y)
        Z = np.column_stack([np.ones(n), x])
        k = Z.shape[1]
        lam = 1.0 if np.isinf(half_life) else 2.0 ** (-1.0 / half_life)
        kinds, h_at, nudge = [], np.full(n, np.nan), np.zeros((n, k))
        fit_rows: list[int] = []
        mad = EwMedian()
        for t in range(n):
            rows = float((lam ** (t - np.arange(t))).sum())
            if t > 0:
                mad.age(lam)
            scale = mad.scale()
            f = np.array(fit_rows, dtype=int)
            fw = lam ** (t - f)
            band_w = float(fw.sum())
            if not np.isfinite(pred[t]) or rows < 3 * k or band_w < k or scale is None:
                kinds.append("ols")
                fit_rows.append(t)
            else:
                h = scale * max(self.EPS, (k / rows) ** 0.4)
                r = y[t] - pred[t]
                h_at[t] = h
                if abs(r) < h:
                    kinds.append("band")
                    fit_rows.append(t)
                else:
                    kinds.append("nudge")
                    raw = 2.0 * h * (tau if r > 0.0 else tau - 1.0) / band_w
                    m = (fw[:, None] * Z[f]).sum(axis=0) / band_w
                    dev = Z[f][:, 1:] - m[1:]
                    A = (fw[:, None] * dev).T @ dev / band_w + self.RIDGE * np.eye(k - 1)
                    u = Z[t, 1:] - m[1:]
                    leverage = float(u @ np.linalg.solve(A, u))
                    most = abs(r) / (1.0 + leverage)
                    step = np.copysign(most, raw) if abs(raw) > most else raw
                    nudge[t] = band_w * step * Z[t]
            if np.isfinite(pred[t]):
                mad.add(abs(y[t] - pred[t]), 1.0)
        return Z, lam, kinds, h_at, nudge

    @pytest.mark.parametrize(("half_life", "tau"), [(float("inf"), 0.5), (300.0, 0.9)])
    def test_the_fit_is_stationary_for_the_loss_it_smooths(self, half_life, tau):
        """The solved fit makes the score of the smoothed check loss zero,
        ridge included: ``sum_t lam^(T-t) [w z (target - z'b)]`` over the
        Gram's rows plus every nudge equals ``ridge W (0, b_slopes)``. Inside
        the band ``target - z'b`` is ``2h psi_h(y - z'b)``, the smoothed
        score at the fit; outside, the nudge is the score at the residual
        the row was scored with. Checked in the lasso-KKT pattern, on the
        last solve."""
        x, y = self.stream()
        pred, coef = self.fit(x, y, tau, half_life)
        Z, lam, kinds, h_at, nudge = self.rows(x, y, pred, tau, half_life)
        beta = coef[-1]
        T = len(y) - 1
        g, size = np.zeros(3), np.zeros(3)
        for t, kind in enumerate(kinds):
            d = lam ** (T - t)
            if kind == "ols":
                term = d * Z[t] * (y[t] - Z[t] @ beta)
            elif kind == "band":
                term = d * Z[t] * (y[t] + 2.0 * h_at[t] * (tau - 0.5) - Z[t] @ beta)
            else:
                term = d * nudge[t]
            g += term
            size += np.abs(term)
        W = sum(lam ** (T - t) for t, kind in enumerate(kinds) if kind != "nudge")
        g[1:] -= self.RIDGE * W * beta[1:]
        assert np.abs(g).max() <= 1e-10 * size.max(), (g, size)
        assert kinds.count("band") > 100 and kinds.count("nudge") > 100, kinds.count("band")

    @pytest.mark.parametrize("rho", [0.0, 0.999])
    def test_a_nudge_never_moves_its_row_past_its_residual(self, rho):
        """The step's bound has one purpose, which `robust.rs` states: the
        row is brought at most to its target, not thrown past it. So the
        solve after a nudged row moves that row's prediction by no more than
        its residual. Every hundredth row past the warm-up runs against the
        features' correlation, at ``(+1, -1)``, on the true line: under
        ``rho = 0.999`` its leverage is near 2,000, where the diagonal
        reading the bound used until review 2026-10-05 (TC1b) said 2, and
        the row moved 9.2 times its residual."""
        x, y = self.stream(rho=rho)
        glitch = np.arange(1000, len(y), 100)
        x[glitch] = [1.0, -1.0]
        y[glitch] = 1.0 + 0.8 + 0.4
        pred, coef = self.fit(x, y, 0.5, 500.0)
        Z, _, kinds, _, _ = self.rows(x, y, pred, 0.5, 500.0)
        nudged = [t for t in range(len(y)) if kinds[t] == "nudge"]
        worst = max((abs(coef[t] @ Z[t] - pred[t]) / abs(y[t] - pred[t]), t) for t in nudged)
        # Bound (task 273): where the bound binds, at the glitch rows, the
        # step moves the row by exactly its residual, so the ratio is 1 in
        # exact arithmetic and the computed one is 1 plus the solve's
        # rounding: a few 2^-52 times the band system's condition number,
        # (1 + rho) / (1 - rho) for two unit features correlated at rho
        # (1,999 at 0.999). Measured on macOS (Apple's libm): 1 - 1.7e-13 at
        # row 1000. With every exp, exp2, ln and powf the wheel calls moved
        # by 1 or 2 ulp (a scratch build): 1 + 1.4e-13 at +1 ulp, so a
        # bound of exactly 1 failed on a libm one ulp off. `4 * 2^-52 *
        # cond` is 1 + 1.8e-12 at 0.999 and 1 + 8.9e-16 at 0; the defect
        # this pins moved the row 9.2 times its residual.
        cond = (1.0 + rho) / (1.0 - rho)
        assert worst[0] <= 1.0 + 4 * 2.0**-52 * cond, (
            f"row {worst[1]} moved {worst[0]!r} times its residual"
        )
        assert set(glitch) & set(nudged), "no glitch row was nudged"


class TestTheBandFactorIsState:
    """Task 170: a quantile fit at ``ridge = 0`` moves its band factor in
    place at each row inside the band, which holds the band Gram's system
    to rounding where a factor made afresh holds it to the bit. Until the
    factor was part of the state, a bank saved and loaded built it afresh
    where the saved bank read the moved one, and parted from the run that
    never stopped by a rounding (saved after rows 997 to 999 of this
    stream: 1,357 predictions, up to 3.4e-13). The README promises every
    spec the same numbers, to the last bit, with a save and resume in the
    middle; these hold the promise for the quantile fit on TC1b's stream:
    two features correlated at 0.999 and, from row 1000, a row against the
    correlation every hundred rows, whose nudge's bound binds."""

    @staticmethod
    def stream(n=1400):
        rng = np.random.default_rng(5)
        a = rng.standard_normal(n)
        b = 0.999 * a + np.sqrt(1.0 - 0.999**2) * rng.standard_normal(n)
        x = np.column_stack([a, b])
        y = 1.0 + x @ np.array([0.8, -0.4]) + rng.standard_exponential(n) - 1.0
        glitch = np.arange(1000, n, 100)
        x[glitch] = [1.0, -1.0]
        y[glitch] = 1.0 + 0.8 + 0.4
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})

    @staticmethod
    def spec(standardize):
        return po.spec.quantile(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            quantile=0.5,
            half_life=500.0,
            quantile_eps=0.2,
            ridge=0.0,
            standardize=standardize,
            coef_every=1,
        )

    @pytest.mark.parametrize("standardize", [False, True])
    def test_a_bank_saved_at_any_row_resumes_to_the_bit(self, standardize):
        """Saved and loaded at every row from 990 to 1010 -- the band rows
        before the first row against the correlation move the factor, and
        that row's nudge reads it -- a bank goes on as the one that never
        stopped, every field to the bit."""
        df, spec = self.stream(), self.spec(standardize)
        whole = po.ModelBank([spec]).fit_predict(df)
        for cut in range(990, 1011):
            bank = po.ModelBank([spec])
            head = bank.fit_predict(df.slice(0, cut))
            loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
            tail = loaded.fit_predict(df.slice(cut))
            assert pl.concat([head, tail]).equals(whole, null_equal=True), cut

    @pytest.mark.parametrize("standardize", [False, True])
    def test_one_chunk_or_a_thousand_give_one_run(self, standardize):
        """Hard rule 3 with the factor moving: chunks of 1, 7 and 600 rows
        give the one chunk's output to the bit (``coef`` on every row, so
        the rows that carry it do not depend on the chunking)."""
        df, spec = self.stream(), self.spec(standardize)
        whole = po.ModelBank([spec]).fit_predict(df)
        for rows in (1, 7, 600):
            bank = po.ModelBank([spec])
            parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
            assert pl.concat(parts).equals(whole, null_equal=True), rows

    def test_the_json_export_carries_the_moved_factor(self):
        """A state saved after row 999 holds a factor its band rows moved;
        the JSON export carries it -- its order, packed triangle, rung and
        moves -- and reads back to the state's bytes, which ``to_json``
        checks itself."""
        df, spec = self.stream(), self.spec(False)
        bank = po.ModelBank([spec])
        bank.fit_predict(df.slice(0, 1000))
        found: list = []

        def walk(node):
            if isinstance(node, dict):
                if "systems" in node:
                    found.extend(node["systems"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(json.loads(bank.write_json()))
        assert len(found) == 1, found
        factor = found[0]["factor"]
        assert factor["order"] == 2 and len(factor["lower"]) == 3, factor
        assert factor["attempts"] == 0 and factor["moves"] > 0, factor


class TestTheQuantilesOwnKeywords:
    """Task 278 (b07): ``quantile``'s ``ridge``, ``standardize``,
    ``quantile_eps`` and ``solve_every``, each shown taking effect where it
    should and nowhere else, without an oracle. The oracled halves -- a
    huge ridge against ``QuantReg`` on a constant, the standardized solve
    against ``QuantReg`` on features of mixed scales, ``quantile_eps``
    against the batch-smoothed fit at its bandwidth, and ``robust_ref`` at
    two ridges -- are in ``tests/test_oracles.py``'s ``TestRobustOracles``.

    The stream is ``TestQuantileIsQuantReg``'s: two unit normal features,
    ``y = 1 + 2 x0 - x1`` plus exponential noise, so the 0.9 quantile's fit
    sits far from the mean's and its band and its nudges both fire."""

    @staticmethod
    def stream(n=3000, seed=3, scales=(1.0, 1.0)):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.exponential(1.0, n)
        return pl.DataFrame({"x0": x[:, 0] * scales[0], "x1": x[:, 1] * scales[1], "y": y})

    @staticmethod
    def run(df, **kw):
        kw = {
            "quantile": 0.9,
            "half_life": 500.0,
            "max_rows_between_solves": 1,
            "coef_every": 1,
            **kw,
        }
        spec = po.spec.quantile("m", targets=["y"], features=["x0", "x1"], **kw)
        # A large ridge or a feature at 1e-3 draws the readiness notice that
        # the ridge, not the data, sets a coefficient: the point of the case.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            return po.ModelBank([spec]).fit_predict(df).unnest("m")

    @staticmethod
    def coef(out) -> np.ndarray:
        return np.array(out["coef"].to_list()[-1])

    @pytest.mark.parametrize("half_life", [float("inf"), 500.0])
    @pytest.mark.parametrize("seed", [3, 7, 9])
    def test_a_ridge_shrinks_the_slopes_and_moves_the_fit(self, seed, half_life):
        """The ridge is on the mean-form Gram's slopes, so the last fit's
        slope norm falls at each tenfold step of it, and the predictions
        move with it. Not a theorem for this fit: which rows sit in the band
        follows the predictions, so a step too small to matter is noise. At
        0 against 1e-6 the norm rose on 3 seeds of 100 (by 2e-7 of itself,
        seeds 7, 9 and 19 at ``half_life = inf``; measured 2026-10-25), and
        no step of this grid failed on any of the 100 (seeds 0-49, both
        half-lives). Seeds 7 and 9 are two of those three."""
        df = self.stream(seed=seed)
        ridges = [0.0, 0.1, 1.0, 10.0, 100.0, 1e4]
        outs = [self.run(df, half_life=half_life, ridge=r) for r in ridges]
        norms = [float(np.sqrt(np.sum(self.coef(o)[1:] ** 2))) for o in outs]
        assert all(b < a for a, b in zip(norms, norms[1:], strict=False)), norms
        assert norms[0] > 1.5 and norms[-1] < 1e-3, norms
        # The predictions move: by 7.0 at ridge 10 against the default on
        # seed 3 under half_life 500 (measured 2026-10-25).
        default = self.run(df, half_life=half_life)
        moved = np.nanmax(np.abs(_pred_of(outs[3]) - _pred_of(default)))
        assert moved > 1.0, moved

    @pytest.mark.extended(reason="a grid: 50 seeds at two half-lives and six ridges, a few seconds")
    def test_the_ridge_and_the_ridge_0_limit_over_fifty_seeds(self):
        """The two relations above, judged by the worst of seeds 0-49: the
        slope norm falls at every tenfold step of the ridge under both
        half-lives, and at ``ridge = 0`` correlation form is the plain solve
        within the bound below, features of mixed scales included.

        The two forms round differently, and the gap is the rounding of the
        first few hundred rows' solves, whose systems of a handful of rows
        are the worst conditioned; it falls to ``1e-16`` by row 500. Which
        rounding they take follows the decay's last bit, the platform's
        ``exp2``: 9.2e-14 here (macOS, 2026-10-27), and 2.1e-13 to 3.75e-13
        with ``exp2`` moved one to three ulps either way. Windows-2022's
        UCRT gave 3.749852472164254e-13, the +2-ulp figure to the last
        digit. The bound leaves under three times the largest."""
        failures, worst = [], 0.0
        for seed in range(50):
            for half_life in (float("inf"), 500.0):
                df = self.stream(seed=seed)
                norms = [
                    float(
                        np.sqrt(
                            np.sum(self.coef(self.run(df, half_life=half_life, ridge=r))[1:] ** 2)
                        )
                    )
                    for r in (0.0, 0.1, 1.0, 10.0, 100.0, 1e4)
                ]
                if not all(b < a for a, b in zip(norms, norms[1:], strict=False)):
                    failures.append((seed, half_life, norms))
            for scales in ((1.0, 1.0), (1e-3, 1e3)):
                df = self.stream(seed=seed, scales=scales)
                a = _pred_of(self.run(df, ridge=0.0))
                b = _pred_of(self.run(df, ridge=0.0, standardize=True))
                assert np.array_equal(np.isnan(a), np.isnan(b)), (seed, scales)
                worst = max(worst, float(np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a))))
        assert not failures, failures
        print(f"ridge 0, correlation form against plain: worst {worst:.3g} of the largest pred")
        assert worst <= 1e-12, worst

    @pytest.mark.parametrize("scales", [(1.0, 1.0), (1e-3, 1e3)], ids=["unit", "mixed"])
    def test_at_ridge_0_standardizing_is_the_plain_solve(self, scales):
        """Correlation form rescales the Gram by its diagonal and back,
        which at ``ridge = 0`` is the same system, so the fits agree to
        rounding, features at 1e-3 and 1e3 included, and every row is
        predicted or withheld alike. Measured over seeds 0-49 (2026-10-25):
        at most 1.6e-15 of the largest prediction with unit features and
        3.2e-15 with mixed ones; the bound leaves thirty times that."""
        df = self.stream(scales=scales)
        plain = self.run(df, ridge=0.0)
        corr = self.run(df, ridge=0.0, standardize=True)
        a, b = _pred_of(plain), _pred_of(corr)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.nanmax(np.abs(a - b)) <= 1e-13 * np.nanmax(np.abs(a))
        # A quantile fit, its band and its nudges both at work: about a
        # tenth of the residuals above the fit at the 0.9 quantile.
        above = float((plain["resid_y"].drop_nulls() > 0).mean())
        assert 0.07 < above < 0.13, above

    def test_under_the_default_ridge_standardizing_saves_a_feature_in_thousandths(self):
        """The default ridge, ``1e-6``, is in the features' squared units
        under the plain solve: a feature at scale 1e-3 has a variance of
        1e-6, so the ridge is its whole variance and halves its slope, ``2
        v / (v + 1e-6) = 1`` in the feature's own units, where correlation
        form makes the ridge dimensionless and keeps it at 2. Seed 3
        measured 1.03 against 2.01 under ``half_life = 500`` (2026-10-25)."""
        df = self.stream(scales=(1e-3, 1.0))
        plain = self.coef(self.run(df, half_life=float("inf")))[1] * 1e-3
        corr = self.coef(self.run(df, half_life=float("inf"), standardize=True))[1] * 1e-3
        assert abs(corr - 2.0) < 0.15, corr
        assert abs(plain - 1.0) < 0.15, plain

    @pytest.mark.parametrize(
        ("standardize", "ridge"),
        [(True, 1e-6), (True, 1.0), (False, 0.0)],
        ids=["standardized, default ridge", "standardized, ridge 1", "plain, ridge 0"],
    )
    def test_a_feature_scaled_by_a_power_of_two_changes_only_its_coefficient(
        self, standardize, ridge
    ):
        """Scaling ``x0`` by ``2^k`` is exact, and so is every step of the
        fit that reads it under correlation form (the ridge is
        dimensionless there) or with no ridge: every field to the bit, and
        ``x0``'s coefficient divided by ``2^k`` to the bit.

        But one step under the plain solve at ``ridge = 0``: a factorization
        that fails retries with ``eps * trace / k`` on the diagonal, a ridge
        in the features' squared units. The second row's centred system has
        rank 1, so its last pivot is rounding, of either sign; where it is
        not positive the jitter picks one of the fits that pass through
        both rows, and the pick moves with ``x0``'s units, in that row's
        ``coef`` and ``support_coef``. Nothing reads it: the row's
        prediction is withheld and the next solve starts from the sums.
        Which sign the pivot takes follows the decay's last bit, the
        platform's ``exp2``: seed 3 needs no jitter here (macOS, glibc and
        windows-2025) and needed it on windows-2022, where one ulp up of
        ``exp2`` reproduces it; over seeds 0-39 at five half-lives it fired
        on 107 of 200 streams (macOS, 2026-10-27). From the third row the
        system has full rank, far from a failed pivot, and every field is
        exact there whatever the platform's ``exp2``."""
        df = self.stream()
        base = self.run(df, standardize=standardize, ridge=ridge)
        # The rows whose system may have a failed pivot: the first two, under
        # the plain solve (the docstring).
        first = 0 if standardize else 2
        coef = np.array(base["coef"].to_list())[first:]
        for k in (10, -10, 30):
            out = self.run(
                df.with_columns(pl.col("x0") * 2.0**k), standardize=standardize, ridge=ridge
            )
            got = np.array(out["coef"].to_list())[first:]
            assert (
                out[first:]
                .drop("coef", "x0")
                .equals(base[first:].drop("coef", "x0"), null_equal=True)
            ), k
            np.testing.assert_array_equal(got[:, 1] * 2.0**k, coef[:, 1], err_msg=f"k = {k}")
            np.testing.assert_array_equal(got[:, [0, 2]], coef[:, [0, 2]], err_msg=f"k = {k}")
            # The rows left out are withheld either way: nothing is predicted
            # from a fit the jitter picked.
            assert out["pred_y"][:first].is_null().all(), k
            assert base["pred_y"][:first].is_null().all(), k

    def test_a_jittered_plain_solve_has_the_features_units(self):
        """The exception above, pinned: without decay seed 3's second-row
        system needs the jitter on every platform (no ``exp2`` is called),
        and scaling ``x0`` by ``2^10`` moves that row's coefficients and
        data shares, and no other row's field."""
        df = self.stream()
        kw = {"ridge": 0.0, "half_life": float("inf")}
        base = self.run(df, **kw)
        out = self.run(df.with_columns(pl.col("x0") * 2.0**10), **kw)
        assert out[2:].drop("coef", "x0").equals(base[2:].drop("coef", "x0"), null_equal=True)
        assert out[0].drop("x0").equals(base[0].drop("x0"), null_equal=True)
        a, b = base["support_coef"][1].to_list(), out["support_coef"][1].to_list()
        assert abs(a[1] - b[1]) > 0.1, (a, b)
        assert out["pred_y"][:2].is_null().all() and base["pred_y"][:2].is_null().all()

    def test_without_standardizing_a_ridge_has_units(self):
        """The contrast: under the plain solve a ridge of 1 is in squared
        feature units, so scaling ``x0`` by 2^10 moves the fit."""
        df = self.stream()
        base = _pred_of(self.run(df, ridge=1.0))
        scaled = _pred_of(self.run(df.with_columns(pl.col("x0") * 2.0**10), ridge=1.0))
        assert np.nanmax(np.abs(scaled - base)) > 0.1

    def test_quantile_eps_under_its_floor_changes_nothing(self):
        """The band's half-width is ``s max(quantile_eps, (k/n)^(2/5))``,
        with ``n`` the target's rows decayed. Under ``half_life = 200`` on
        the row clock ``n`` never passes ``1 / (1 - 2^(-1/200)) = 289.04``,
        so with ``k = 3`` the floor never falls under ``(3 / 289.04)^0.4 =
        0.1606``: ``quantile_eps`` at 0.05 and at 0.15 draw the same band on
        every row, and the output is the same to the bit. At 0.17 the band
        is wider once the window has filled, and the fit moves (by 0.12 on
        seed 3, measured 2026-10-25; by 0.046 with the refit of the first
        rows, task 278, D-Q2, measured 2026-10-26: the refit draws its band
        at the floor of its 500 rows, decayed, which is above 0.17, so the
        two start from one fit). Without a half-life the floor falls to 0,
        and 0.05 and 0.15 part (0.096)."""
        df = self.stream()
        low, under = (self.run(df, half_life=200.0, quantile_eps=e) for e in (0.05, 0.15))
        assert low.equals(under, null_equal=True)
        above = self.run(df, half_life=200.0, quantile_eps=0.17)
        assert np.nanmax(np.abs(_pred_of(above) - _pred_of(low))) > 0.02
        a, b = (self.run(df, half_life=float("inf"), quantile_eps=e) for e in (0.05, 0.15))
        assert np.nanmax(np.abs(_pred_of(a) - _pred_of(b))) > 0.05


class TestTheQuantileSolveEvery:
    """Task 278 (b07): ``solve_every`` through the bank, on a clock column
    whose steps are quarters (so every elapsed clock is exact, and a solve
    falls on the edge itself 22 times in this stream). Until ``solve_every``
    has passed since the first row, the first period, the fit is solved on
    the default weight-share schedule (task 278, "Default cadence first"),
    whatever ``min_weight`` says; the row it passes on solves, then each
    first row whose clock is at least ``solve_every`` past the last solve's,
    and the coefficients hold in between (the builder's docstring);
    ``coef`` on every row shows each solve. (Until then the fit was first
    solved on the row its target had been present on a row per coefficient,
    the third, and the cadence counted from it.)"""

    N, EVERY, MIN_WEIGHT = 600, 5.0, 3.0

    @classmethod
    def frame(cls):
        rng = np.random.default_rng(4)
        x = rng.normal(0.0, 1.0, (cls.N, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.exponential(1.0, cls.N)
        t = np.cumsum(rng.integers(1, 9, cls.N) * 0.25)
        return pl.DataFrame({"t": t, "x0": x[:, 0], "x1": x[:, 1], "y": y})

    @classmethod
    def spec(cls, **kw):
        kw = {
            "clock": "t",
            "gap_cap": 10.0,
            "half_life": 100.0,
            "min_weight": cls.MIN_WEIGHT,
            "solve_every": cls.EVERY,
            # The refit moves the fit on its own row, off the cadence this
            # class holds (task 278, D-Q2), so it is off here.
            "quantile_refit_rows": 0,
            **kw,
        }
        return po.spec.quantile(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            quantile=0.9,
            max_rows_between_coefs=1,
            **kw,
        )

    def test_the_fit_moves_on_the_scheduled_rows_and_holds_between(self):
        df = self.frame()
        out = po.ModelBank([self.spec()]).fit_predict(df).unnest("m")
        t, ws = df["t"].to_numpy(), out["weight_sum"].to_numpy()
        coef = out["coef"].to_list()
        moved = [
            i for i in range(self.N) if coef[i] is not None and (i == 0 or coef[i] != coef[i - 1])
        ]
        # The first period ends on the first row `EVERY` past the first's,
        # and solves before it on the default weight-share schedule: once
        # the weight learned since the last solve reaches ln 2 / 50 of the
        # weight held, both after the row (every row, this early).
        end = next(i for i in range(self.N) if t[i] - t[0] >= self.EVERY)
        share, held, since = [], 0.0, 0.0
        for i in range(end):
            held = (0.5 ** ((t[i] - t[i - 1]) / 100.0) if i else 1.0) * held + 1.0
            since += 1.0
            if since >= math.log(2) / 50 * held:
                share.append(i)
                since = 0.0
        # `min_weight` moves none of it: the period's first solves come
        # before the weight reaches it.
        assert ws[1] < self.MIN_WEIGHT and len(share) >= 3, (ws[:end], share)
        want, last = [*share, end], end
        for i in range(end + 1, self.N):
            if t[i] - t[last] >= self.EVERY:
                want.append(i)
                last = i
        assert moved == want
        on_the_edge = sum(t[b] - t[a] == self.EVERY for a, b in zip(want, want[1:], strict=False))
        assert on_the_edge > 10, on_the_edge
        # And each prediction is the coefficients the row before reported.
        pred = out["pred_y"].to_numpy()
        z = np.column_stack([np.ones(self.N), df["x0"].to_numpy(), df["x1"].to_numpy()])
        rows = [i for i in range(1, self.N) if coef[i - 1] is not None and np.isfinite(pred[i])]
        assert len(rows) > self.N - 10
        want_pred = np.array([z[i] @ np.array(coef[i - 1]) for i in rows])
        np.testing.assert_allclose(pred[rows], want_pred, rtol=1e-14, atol=1e-14)

    def test_zero_is_every_row(self):
        """``solve_every = 0`` solves every row: the row cap of 1."""
        df = self.frame()
        zero = po.ModelBank([self.spec(solve_every=0.0)]).fit_predict(df)
        cap = po.ModelBank([self.spec(solve_every=1e18, max_rows_between_solves=1)]).fit_predict(df)
        assert zero.equals(cap, null_equal=True)
        assert not zero.equals(po.ModelBank([self.spec()]).fit_predict(df), null_equal=True)

    def test_a_duration_on_a_datetime_clock_is_seconds_on_a_float_one(self):
        df = self.frame()
        start = dt.datetime(2024, 1, 1)
        stamps = pl.lit(start) + pl.duration(microseconds=(pl.col("t") * 1e6).cast(pl.Int64))
        dated = df.with_columns(stamps.alias("t"))
        spec = self.spec(solve_every="5s", gap_cap="10s", half_life="100s")
        got = po.ModelBank([spec]).fit_predict(dated)["m"]
        want = po.ModelBank([self.spec()]).fit_predict(df)["m"]
        assert got.equals(want, null_equal=True)

    def test_chunks_and_a_save_between_solves_keep_the_schedule(self):
        """The clock since the last solve is state: chunks of 1, 7 and 600
        rows, and a bank saved and loaded at rows on either side of solves,
        give the one run to the bit."""
        df, spec = self.frame(), self.spec()
        whole = po.ModelBank([spec]).fit_predict(df)
        coef = whole["m"].struct.field("coef").to_list()
        solves = [i for i in range(200, 212) if coef[i] != coef[i - 1]]
        assert 2 <= len(solves) <= 10, solves
        for rows in (1, 7, 600):
            bank = po.ModelBank([spec])
            parts = [bank.fit_predict(df.slice(i, rows)) for i in range(0, df.height, rows)]
            assert pl.concat(parts).equals(whole, null_equal=True), rows
        for cut in range(200, 212):
            bank = po.ModelBank([spec])
            head = bank.fit_predict(df.slice(0, cut))
            loaded = po.ModelBank.load_bytes(bank.save_bytes(), specs=[spec])
            assert pl.concat([head, loaded.fit_predict(df.slice(cut))]).equals(
                whole, null_equal=True
            ), cut


def _pred_of(out) -> np.ndarray:
    return out["pred_y"].fill_null(float("nan")).to_numpy().astype(float)


@pytest.mark.parametrize("loss", ["huber", "quantile"])
def test_on_the_row_clock_solve_every_is_its_row_cap(loss):
    """Without a clock column the clock is the row's number, the first row
    being 1, so ``solve_every=n`` solves on the rows
    ``max_rows_between_solves=n`` does. Measured from the first row, whose
    step is 0, it solved a row later, the whole stream long, which ``coef``
    showed (task 278, D13). The first period ends on the ``n``-th row,
    whatever ``min_weight`` says (1e9 here, never met), and the model solves
    on the default weight-share schedule before it -- every row at this
    half-life (task 278, "Default cadence first"; until then the cadence
    counted from the first solve, the floor of a row per coefficient, row
    2)."""
    rng = np.random.default_rng(0)
    n = 60
    x = rng.standard_normal(n)
    df = pl.DataFrame({"x0": x, "x1": rng.standard_normal(n), "y0": 2.0 * x})
    extra = {"huber_refit_rows": 0} if loss == "huber" else {"quantile": 0.5}

    def run(**kw):
        spec = getattr(po.spec, loss)(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            half_life=20.0,
            min_weight=1e9,
            max_rows_between_coefs=1,
            **extra,
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            return po.ModelBank([spec]).fit_predict(df).unnest("m")

    clock = run(solve_every=7.0)
    assert clock.equals(run(solve_every=1e9, max_rows_between_solves=7), null_equal=True)
    c = clock["coef"].to_list()
    assert [i for i in range(1, n) if c[i] != c[i - 1]] == [1, 2, 3, 4, 5, *range(6, n, 7)]
