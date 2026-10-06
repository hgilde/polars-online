"""Task 13: robust models - Huber and quantile (docs/PLAN.md section 4.5)."""

import json

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic


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
    # holds a fifteenth of them -- so the whole stream reads a little under the
    # level and its second half reads the level: 0.858 and 0.893 measured,
    # where the frozen IRLS weights this replaced read 0.777 and 0.855 (N9).
    assert 0.84 < below < 0.90, below
    assert 0.87 < after < 0.92, after


def test_quantile_predicts_every_row_under_a_finite_halflife():
    """The per-target ``min_weight`` gate reads the rows a target was present
    on (hard rule 8, S2). From N9 to the second review's F1 the quantile fit
    reported its band's weight instead, which a half-life caps at the band's
    share of the effective sample -- a fifteenth of it at ``tau = 0.9`` -- so
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
    keep = [c for c in one.columns if not c.startswith("coef")]
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
    coef8, pred8 = fit(1e8)
    assert coef0[1:] == pytest.approx([1.5, -0.5], abs=0.05), "the fixture is not what it claims"
    assert coef8[1:] == pytest.approx(coef0[1:], abs=1e-6), (coef8, coef0)
    m = np.isfinite(pred0) & np.isfinite(pred8)
    assert m.sum() > n - 20
    assert np.max(np.abs((pred8[m] - 1e8) - pred0[m])) < 1e-5


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
    coef8, pred8 = fit(1e8)
    assert coef0[1] == pytest.approx(2.0, abs=0.1), "the fixture is not what it claims"
    assert coef8[1] == pytest.approx(coef0[1], abs=1e-6), (coef8, coef0)
    m = np.isfinite(pred0) & np.isfinite(pred8)
    assert np.max(np.abs((pred8[m] - 1e8) - pred0[m])) < 1e-5


class TestTheQuantileFitsDefinition:
    """The quantile fit held to the problem it solves, not to a replay of its
    arithmetic (`tests/reference.py`'s `robust_ref` restates the core; review
    2026-10-05, TC1). `robust.rs`'s module doc defines each row's part in
    the Newton system: under three rows per coefficient, or a band holding
    under one row per coefficient, a least-squares row; inside the band of
    half-width ``h = s max(quantile_eps, (k/n)^(2/5))`` a least-squares row
    with target ``y + 2h(tau - 1/2)``; outside it a nudge ``2h psi(r) z``
    into the cross-moment only, with ``psi = tau - 1{r < 0}``, its step
    bounded by ``|r| / (1 + u'A^-1 u)``, the row's full leverage against the
    band's system: ``A`` the band rows' centred covariance plus the ridge,
    ``u`` the row's centred features (TC1b). ``s`` is the RMS of the scored
    residuals before the row and ``n`` the target's rows, both decayed, and
    ``r`` the residual the row was scored with."""

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
        scored: list[int] = []
        for t in range(n):
            rows = float((lam ** (t - np.arange(t))).sum())
            sig2 = 0.0
            if scored:
                s = np.array(scored)
                d = lam ** (t - s)
                sig2 = float((d * (y[s] - pred[s]) ** 2).sum() / d.sum())
            scale = np.sqrt(sig2) if sig2 > 0.0 else 1.0
            f = np.array(fit_rows, dtype=int)
            fw = lam ** (t - f)
            band_w = float(fw.sum())
            if not np.isfinite(pred[t]) or rows < 3 * k or band_w < k:
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
                scored.append(t)
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
        assert worst[0] <= 1.0, f"row {worst[1]} moved {worst[0]:.2f} times its residual"
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

        walk(json.loads(bank.to_json()))
        assert len(found) == 1, found
        factor = found[0]["factor"]
        assert factor["order"] == 2 and len(factor["lower"]) == 3, factor
        assert factor["attempts"] == 0 and factor["moves"] > 0, factor
