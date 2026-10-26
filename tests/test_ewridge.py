"""`ewridge` against the numpy oracle (`reference.ewridge_ref`), and the
features only it has: warm priors (`coef_prior`, E15) and `session_shrink` (E6).

The oracle covers multi-target, `standardize`, the row-count clock with `lam`,
nulls, weights, groups and sessions; `test_semantics_all_models` and the other
sweeps hold this model to the invariants every model shares.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import compute_dclock, ewridge_ref

TIER = "mixed"

HL = 300.0
MAXD = 50.0
GAP = 25.0


def _spec(k=3, targets=("y0",), **kw):
    defaults = dict(
        targets=list(targets),
        features=[f"x{j}" for j in range(k)],
        clock="t",
        half_life=HL,
        gap_cap=MAXD,
        session="session",
        session_gap=GAP,
        weight="w",
        group="group",
        ridge=1e-6,
        max_rows_between_solves=1,
        min_weight=5.0,
    )
    defaults.update(kw)
    return po.spec.ewridge("m", **defaults)


def _oracle_frame(df: pl.DataFrame, k=3, n_targets=1, **ref_kw):
    """Run ewridge_ref per group and return pred/resid/weight_sum arrays row-aligned."""
    ref_defaults = dict(half_life=HL, ridge=1e-6, min_weight=5.0)
    ref_defaults.update(ref_kw)
    pred = np.full((df.height, n_targets), np.nan)
    resid = np.full((df.height, n_targets), np.nan)
    weight_sum = np.full(df.height, np.nan)
    df = df.with_row_index("_i")
    for _, g in df.group_by("group", maintain_order=True):
        idx = g["_i"].to_numpy()
        x = np.column_stack([g[f"x{j}"].to_numpy() for j in range(k)])
        y = np.column_stack([g[f"y{j}"].to_numpy() for j in range(n_targets)])
        w = g["w"].to_numpy()
        dc, rs = compute_dclock(
            g["t"].to_numpy(),
            g["session"].to_numpy(),
            len(g),
            gap_cap=MAXD,
            session_gap=GAP,
        )
        out = ewridge_ref(x, y, dc, w, rs, gap_cap=MAXD, **ref_defaults)
        pred[idx] = out["pred"]
        resid[idx] = out["resid"]
        weight_sum[idx] = out["weight_sum"]
    return pred, resid, weight_sum


def _close(a: np.ndarray, b: np.ndarray, tol=1e-9):
    both_nan = np.isnan(a) & np.isnan(b)
    ok = both_nan | (np.abs(a - b) < tol * (1.0 + np.abs(a)))
    assert ok.all(), f"max diff {np.nanmax(np.abs(a - b))}, mismatches {np.sum(~ok)}"


def _np(df, col):
    return df["m"].struct.field(col).to_numpy().astype(float)


class TestOracle:
    def test_ewridge_matches_reference(self):
        df, _ = synthetic(seed=3, n_groups=2, n_rows=350, k=3)
        bank = po.ModelBank([_spec()])
        out = bank.fit_predict(df)
        pred, resid, weight_sum = _oracle_frame(df)
        _close(_np(out, "pred_y0"), pred[:, 0])
        _close(_np(out, "resid_y0"), resid[:, 0])
        _close(_np(out, "weight_sum"), weight_sum)

    def test_multi_target(self):
        df, _ = synthetic(seed=4, n_groups=1, n_rows=250, k=3, n_targets=2)
        bank = po.ModelBank([_spec(targets=("y0", "y1"))])
        out = bank.fit_predict(df)
        pred, resid, _ = _oracle_frame(df, n_targets=2)
        _close(_np(out, "pred_y0"), pred[:, 0])
        _close(_np(out, "pred_y1"), pred[:, 1])
        _close(_np(out, "resid_y1"), resid[:, 1])

    def test_standardize(self):
        df, _ = synthetic(seed=5, n_groups=1, n_rows=250, k=3)
        bank = po.ModelBank([_spec(standardize=True)])
        out = bank.fit_predict(df)
        pred, _, _ = _oracle_frame(df, standardize=True)
        _close(_np(out, "pred_y0"), pred[:, 0])

    def test_row_count_clock_and_lam(self):
        df, _ = synthetic(seed=6, n_groups=1, n_rows=200, k=2, null_frac=0.0)
        lam = 0.99
        bank = po.ModelBank(
            [
                _spec(
                    k=2,
                    clock=None,
                    half_life=None,
                    lam=lam,
                    gap_cap=None,
                    session=None,
                    session_gap=None,
                )
            ]
        )
        out = bank.fit_predict(df)
        x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
        y = df["y0"].to_numpy().reshape(-1, 1)
        n = len(df)
        dc = np.ones(n)
        dc[0] = 0.0
        halflife_equiv = -np.log(2.0) / np.log(lam)
        ref = ewridge_ref(
            x, y, dc, df["w"].to_numpy(), half_life=halflife_equiv, ridge=1e-6, min_weight=5.0
        )
        _close(_np(out, "pred_y0"), ref["pred"][:, 0])


class TestWarmPriors:
    """E15: `coef_prior` shrinks toward a stated belief instead of toward zero.

    Whether the prior fades depends on `ridge_scale`, and the distinction is
    the point: `S` is a weighted *mean*, so a plain `ridge` is a fixed
    per-observation penalty whose pull never washes out.
    """

    def _slopes(self, **kw):
        rng = np.random.default_rng(0)
        n = 3000
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 1.5 * x})
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            ridge=10.0,
            half_life=1e9,
            min_weight=0.0,
            max_rows_between_solves=1,
            coef_every=0,
            **kw,
        )
        coefs = np.array(
            po.ModelBank([spec]).fit_predict(df)["m"].struct.field("coef").to_list(),
            dtype=float,
        )
        return coefs[10][1], coefs[-1][1]

    def test_no_prior_shrinks_toward_zero(self):
        early, late = self._slopes()
        assert late < 0.5, f"a heavy ridge with no prior should shrink to ~0, got {late}"

    def test_fixed_ridge_pull_is_permanent(self):
        early, late = self._slopes(coef_prior=[[0.0, 5.0]])
        assert early > 4.0, "should start near the prior"
        assert late > 3.0, f"a fixed ridge keeps pulling toward the prior forever, got {late}"

    def test_ridge_decay_makes_it_a_fading_warm_start(self):
        early, late = self._slopes(coef_prior=[[0.0, 5.0]], ridge_scale="sum")
        assert early > 2.5, f"should start warm near the prior, got {early}"
        assert abs(late - 1.5) < 0.1, f"the prior should fade to the truth, got {late}"

    def test_prior_is_in_original_units_under_standardization(self):
        rng = np.random.default_rng(1)
        n = 2000
        x = 100.0 * rng.standard_normal(n)
        df = pl.DataFrame({"x0": x, "y0": 0.05 * x})
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            ridge=1e6,
            standardize=True,
            coef_prior=[[0.0, 0.02]],
            half_life=1e9,
            min_weight=0.0,
            max_rows_between_solves=1,
            coef_every=0,
        )
        c = np.array(
            po.ModelBank([spec]).fit_predict(df)["m"].struct.field("coef").to_list()[-1],
            dtype=float,
        )
        assert abs(c[1] - 0.02) < 5e-3, f"expected the prior in original units, got {c[1]}"

    def test_shape_is_validated(self):
        with pytest.raises(ValueError, match="coef_prior"):
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0", "x1"],
                half_life=100.0,
                coef_prior=[[0.0, 1.0]],  # too short for 2 features + intercept
            )


class TestSessionShrink:
    """E6: revert partway toward the long run at a session boundary.

    PLAN section 12's first open question. `session_gap` only changes how
    *confident* the model is; this changes what it believes, by mixing the
    accumulators with a slow-moving twin.
    """

    N1, N2, N3 = 4000, 300, 300

    def _df(self):
        rng = np.random.default_rng(0)
        n = self.N1 + self.N2 + self.N3
        x = rng.standard_normal(n)
        # long run slope +1, then a stretch at -1, then a new session
        y = np.concatenate([x[: self.N1], -x[self.N1 :]])
        return pl.DataFrame(
            {
                "t": np.arange(float(n)),
                "x0": x,
                "y0": y,
                "s": ["a"] * (self.N1 + self.N2) + ["b"] * self.N3,
            }
        )

    def _coefs(self, **kw):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=50.0,
            clock="t",
            gap_cap=5.0,
            session="s",
            session_gap=0.0,
            min_weight=0.0,
            max_rows_between_solves=1,
            coef_every=0,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(self._df())
        return np.array(out["m"].struct.field("coef").to_list(), dtype=float)

    def _jump(self, **kw):
        """How far the slope moves across the session boundary, against a
        typical row-to-row move just before it."""
        c = self._coefs(**kw)[:, 1]
        i = self.N1 + self.N2
        typical = float(np.median(np.abs(np.diff(c[i - 100 : i]))))
        return abs(c[i] - c[i - 1]), typical

    def test_zero_shrink_carries_the_recent_fit_through(self):
        jump, typical = self._jump(session_shrink=0.0, long_half_life=1e5)
        assert jump < 10 * typical, (
            f"boundary move {jump} should look like an ordinary row ({typical})"
        )

    def test_shrink_reverts_toward_the_long_run(self):
        c = self._coefs(session_shrink=0.9, long_half_life=1e5)
        before, after = c[self.N1 + self.N2 - 1][1], c[self.N1 + self.N2][1]
        assert before < -0.5, "the recent regime should have taken over first"
        assert after > 0.5, f"the break should revert toward +1, got {after}"

    def test_shrink_is_monotone(self):
        after = {
            f: self._coefs(session_shrink=f, long_half_life=1e5)[self.N1 + self.N2][1]
            for f in (0.0, 0.3, 0.6, 0.9)
        }
        vals = [after[f] for f in (0.0, 0.3, 0.6, 0.9)]
        assert vals == sorted(vals), f"more shrink should mean more reversion: {after}"

    def test_the_share_is_of_the_data(self):
        """Task 145: at ``f`` the slope moves about ``f`` of the way to the long
        run's fit -- exactly, if both had the same feature moments; here the
        features are drawn alike in both regimes."""
        i = self.N1 + self.N2
        today = self._coefs(session_shrink=0.0, long_half_life=1e5)[i][1]
        long_run = self._coefs(session_shrink=1.0, long_half_life=1e5)[i][1]
        assert today < -0.5 < 0.5 < long_run
        for f in (0.25, 0.5):
            after = self._coefs(session_shrink=f, long_half_life=1e5)[i][1]
            assert after == pytest.approx((1 - f) * today + f * long_run, abs=0.05), f

    def test_the_blend_keeps_the_weight_and_the_solve_schedule(self):
        """Task 145: ``weight_sum`` and the Kish size are the same with and without
        the blend, and the next solve after the open comes when it would
        without it: under a weight-mixed blend the slow twin's weight made
        both jump and the schedule wait (32 rows against 3, measured)."""
        df = self._df()
        out = {}
        for f in (0.0, 0.25):
            spec = po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                half_life=50.0,
                clock="t",
                gap_cap=5.0,
                session="s",
                session_gap=0.0,
                min_weight=0.0,
                coef_every=0,
                session_shrink=f,
                long_half_life=1e5,
            )
            bank = po.ModelBank([spec])
            res = bank.fit_predict(df).unnest("m")
            coef = np.array(res["coef"].to_list(), dtype=float)[:, 1]
            open_ = self.N1 + self.N2
            changes = np.flatnonzero(np.diff(coef[open_:]) != 0)
            out[f] = (res["weight_sum"].to_numpy(), int(changes[0]), bank.gram("m")[0]["n_kish"])
        np.testing.assert_array_equal(out[0.25][0], out[0.0][0])
        assert out[0.25][1] == out[0.0][1]
        assert out[0.25][2] == out[0.0][2]

    def test_absent_by_default(self):
        jump, typical = self._jump()
        assert jump < 10 * typical, "no shrink configured should mean no jump"

    def test_shrink_produces_a_visible_jump(self):
        jump, typical = self._jump(session_shrink=0.9, long_half_life=1e5)
        assert jump > 100 * typical, (
            f"a 0.9 shrink should move the fit far more than a row does ({jump} vs {typical})"
        )

    def test_chunk_invariance_and_save_load(self, tmp_path):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            half_life=50.0,
            clock="t",
            gap_cap=5.0,
            session="s",
            session_gap=0.0,
            session_shrink=0.5,
            long_half_life=1e5,
            min_weight=0.0,
            max_rows_between_solves=1,
        )
        df = self._df().slice(0, 800)
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        bank = po.ModelBank([spec])
        many = (
            pl.concat([bank.fit_predict(df.slice(i, 97)) for i in range(0, df.height, 97)])
            .select("m")
            .unnest("m")
        )
        keep = [c for c in one.columns if not c.startswith(("coef", "support_coef"))]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 400))
        p = tmp_path / "ss.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(400, 400)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    # ---- the blend leaves the solve schedule alone (task 278, batch 4) ----
    #
    # A blend used to force a solve at the open and restart the cadence from
    # it, while `f = 0` returned early and forced nothing: at `f = 1e-12`
    # against `0` the predictions parted by 0.48 under the default cadence
    # and 1.43 under `solve_every=50` (3,000 rows, measured 2026-10-25),
    # though the moments differed by 1e-12 of their spread. Now a blend
    # moves the moments and nothing else; the next scheduled solve reads
    # them. These hold the four consequences: 0 is no shrink to the bit,
    # the output is continuous in `f`, the solve rows do not move, and `1`
    # is still the long run's moments at today's weight.

    SCHEDULES = ({}, {"solve_every": 50.0})

    def _sessions_df(self, n: int = 3000):
        """Three sessions of 1,000 rows; the slopes change at each open, so a
        blend moves the fit."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal((n, 2))
        seg = np.arange(n) // 1000
        slopes = np.array([[1.0, -0.5], [-1.0, 0.8], [0.4, 1.2]])[seg]
        y = 0.2 + (x * slopes).sum(axis=1) + 0.1 * rng.standard_normal(n)
        return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y0": y, "s": seg})

    def _session_run(self, schedule, **kw):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            half_life=200.0,
            min_weight=10.0,
            session="s",
            session_gap=1.0,
            coef_every=0,
            **schedule,
            **kw,
        )
        return po.ModelBank([spec], threads=1).fit_predict(self._sessions_df()).unnest("m")

    @staticmethod
    def _solve_rows(out) -> np.ndarray:
        """The rows a solve ran on: where `coef` changed (every row is a
        `coef` row under `coef_every=0`)."""
        coef = np.array(
            [[np.nan] * 3 if c is None else c for c in out["coef"].to_list()], dtype=float
        )
        changed = np.any(coef[1:] != coef[:-1], axis=1) & ~np.isnan(coef[1:]).any(axis=1)
        return np.flatnonzero(changed) + 1

    @pytest.mark.parametrize("schedule", SCHEDULES, ids=["default", "solve_every"])
    def test_zero_shrink_is_no_shrink_to_the_bit(self, schedule):
        plain = self._session_run(schedule)
        zero = self._session_run(schedule, session_shrink=0.0, long_half_life=500.0)
        assert zero.equals(plain, null_equal=True)

    @pytest.mark.parametrize("schedule", SCHEDULES, ids=["default", "solve_every"])
    def test_the_blend_moves_no_solve(self, schedule):
        """The rows a solve runs on are the same with and without a blend, at
        any share. The evidence the schedule is not every row: the open rows
        1000 and 2000 are not solve rows without a blend, and solves are
        several rows apart."""
        plain = self._solve_rows(self._session_run(schedule))
        gaps = np.diff(plain)
        print(f"{plain.size} solves, median spacing {np.median(gaps)} rows")
        assert 1000 not in plain and 2000 not in plain, "an open is a solve row without a blend"
        assert np.median(gaps) >= 3, "the cadence solves too often to show a moved solve"
        for f in (1e-12, 0.5, 1.0):
            got = self._solve_rows(
                self._session_run(schedule, session_shrink=f, long_half_life=500.0)
            )
            np.testing.assert_array_equal(got, plain, err_msg=f"session_shrink={f}")

    @pytest.mark.parametrize("schedule", SCHEDULES, ids=["default", "solve_every"])
    def test_the_output_is_continuous_in_the_share(self, schedule):
        """``pred(f) - pred(0)`` is ``O(f)``. The blend is
        ``m' = m + f (m_slow - m)`` and
        ``C' = C + f (C_slow - C) + f (1 - f) dm dm'``: affine in ``f`` up to a
        term of ``f``'s order, and every row after it -- decay and a rank-one
        addition -- carries that forward linearly, so the moments at any row
        are ``M(0) + f D + O(f^2)``. A solve and a prediction are smooth in
        the moments where the system is regular, so
        ``|pred(f) - pred(0)| = f |g'(0)| + O(f^2)``: the largest difference
        over the stream, divided by ``f``, tends to a constant ``L``. Held as:
        the ratio at ``f = 1e-6`` and ``1e-9`` is within 1% of the ratio at
        ``1e-4`` (the second-order term is ``f`` times smaller again), plus a
        floor of 1e-13 for the double's rounding of a prediction of order 1,
        and ``L`` is finite and nonzero (the blend acts)."""
        base = self._session_run(schedule, session_shrink=0.0, long_half_life=500.0)
        p0 = base["pred_y0"].to_numpy().astype(float)
        ok = np.isfinite(p0)

        def gap(f: float) -> float:
            out = self._session_run(schedule, session_shrink=f, long_half_life=500.0)
            pf = out["pred_y0"].to_numpy().astype(float)
            np.testing.assert_array_equal(np.isfinite(pf), ok)
            return float(np.max(np.abs(pf[ok] - p0[ok])))

        lip = gap(1e-4) / 1e-4
        print(f"L = max |pred(f) - pred(0)| / f at f = 1e-4: {lip:.6g}")
        # Measured 2026-10-25: 1.265 (default cadence), 1.280 (solve_every
        # 50); the ratio agreed at 1e-6 and 1e-9 to five digits. Before the
        # fix the forced solve gave 608 and 7,325 here, and 6.1e4 at 1e-6.
        assert 0.1 < lip < 10.0, lip
        for f in (1e-6, 1e-9):
            d = gap(f)
            print(f"f = {f:g}: max |dpred| = {d:.3g}, / f = {d / f:.6g}")
            assert abs(d - lip * f) <= 0.01 * lip * f + 1e-13, (f, d, lip * f)

    def test_full_shrink_at_the_fits_own_half_life_moves_nothing(self):
        """At ``f = 1`` the moments become the slow twin's, at today's weight.
        With ``long_half_life == half_life`` the twin is the fit itself, so
        the blend changes nothing and the output is the no-blend stream's, to
        the rounding of two accumulators summed apart (under 1e-12 of a
        prediction of order 1). It moved predictions by up to 0.48 while a
        blend forced a solve."""
        for schedule in self.SCHEDULES:
            plain = self._session_run(schedule)
            full = self._session_run(schedule, session_shrink=1.0, long_half_life=200.0)
            a = plain["pred_y0"].to_numpy().astype(float)
            b = full["pred_y0"].to_numpy().astype(float)
            np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
            ok = np.isfinite(a)
            assert np.max(np.abs(a[ok] - b[ok])) < 1e-12, schedule
            np.testing.assert_array_equal(
                full["weight_sum"].to_numpy(), plain["weight_sum"].to_numpy()
            )

    def test_config_is_validated(self):
        base = dict(targets=["y0"], features=["x0"], half_life=50.0, session="s", session_gap=0.0)
        with pytest.raises(ValueError, match="needs long_half_life"):
            po.spec.ewridge("m", session_shrink=0.5, **base)
        with pytest.raises(ValueError, match="must be in .0, 1."):
            po.spec.ewridge("m", session_shrink=1.5, long_half_life=1e5, **base)
        with pytest.raises(ValueError, match="needs a .session. column"):
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                half_life=50.0,
                session_shrink=0.5,
                long_half_life=1e5,
            )


def _batch_local_linear(x, y, half_life, ridge, min_weight):
    """A one-sided exponential-kernel local linear regression, computed from
    scratch at every row: for row `i`, the weighted least squares of `y` on
    `[1, x]` over the rows before it under weight `0.5 ** (dx / half-life)`,
    read at that row's own `x`. The ridge is on the mean scale with the
    intercept unpenalised, which is the solve `ewridge` does."""
    n = len(x)
    out = np.full(n, np.nan)
    pen = np.eye(2)
    pen[0, 0] = 0.0
    for i in range(1, n):
        w = 0.5 ** ((x[i - 1] - x[:i]) / half_life)
        if w.sum() < min_weight:
            continue
        xi = np.column_stack([np.ones(i), x[:i]])
        moments = (xi.T * w) @ xi / w.sum()
        cross = (xi.T * w) @ y[:i] / w.sum()
        beta = np.linalg.solve(moments + ridge * pen, cross)
        out[i] = beta[0] + beta[1] * x[i]
    return out


@pytest.mark.parametrize(("half_life", "ridge"), [(0.25, 1e-9), (0.3, 1e-4), (1.0, 1e-2)])
def test_a_feature_as_the_clock_is_a_local_linear_regression(half_life, ridge):
    """The README's "A local fit along any feature": sorted by a feature and
    clocked on it, the decay is a kernel in that feature and the fit is a
    local linear regression -- LOESS with a one-sided exponential kernel,
    out of a state that does not grow. The oracle here is the batch fit that
    definition describes, recomputed at every row; the model's recursion is
    the only thing under test."""
    rng = np.random.default_rng(1)
    n = 1200
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y = np.sin(x) + 0.1 * rng.standard_normal(n)
    spec = po.spec.ewridge(
        "loc",
        targets=["y"],
        features=["x"],
        clock="x",
        half_life=half_life,
        gap_cap=1.0,
        ridge=ridge,
        max_rows_between_solves=1,
        min_weight=20.0,
    )
    got = (
        po.ModelBank([spec])
        .fit_predict(pl.DataFrame({"x": x, "y": y}))
        .unnest("loc")["pred_y"]
        .to_numpy()
    )
    want = _batch_local_linear(x, y, half_life, ridge, 20.0)

    assert np.array_equal(np.isnan(got), np.isnan(want)), "a different set of rows warmed up"
    fit = ~np.isnan(want)
    assert fit.sum() > n - 50
    assert np.abs(got[fit] - want[fit]).max() < 1e-12


def test_a_bandwidth_in_the_clock_column_follows_a_curve_a_line_cannot():
    """What the local fit buys, and what the one-sided kernel costs. On
    `sin(x)` with a bandwidth of a quarter of a unit the fit sits 0.08 from
    the truth against a straight line's 0.39 -- but the kernel looks only
    backwards, because a row is scored before it is learned from, so a curve
    is followed with the lag that implies. A wider bandwidth is a flatter
    fit: at `half_life=1.0` the same stream gives 0.29, most of the way back
    to the line. The README's *A local fit along any feature* cites these
    three figures, rounded to two places, and this test holds them there."""
    rng = np.random.default_rng(1)
    n = 1200
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y = np.sin(x) + 0.1 * rng.standard_normal(n)
    df = pl.DataFrame({"x": x, "y": y})

    def local(half_life):
        spec = po.spec.ewridge(
            "loc",
            targets=["y"],
            features=["x"],
            clock="x",
            half_life=half_life,
            gap_cap=1.0,
            max_rows_between_solves=1,
            min_weight=20.0,
        )
        return po.ModelBank([spec]).fit_predict(df).unnest("loc")["pred_y"].to_numpy()

    narrow, wide = local(0.25), local(1.0)
    fit = ~np.isnan(narrow)
    line = np.polyval(np.polyfit(x, y, 1), x)
    rms = lambda a: float(np.sqrt(np.mean((a[fit] - np.sin(x[fit])) ** 2)))  # noqa: E731

    assert rms(narrow) < 0.12 < rms(wide) < rms(line)
    assert [round(rms(a), 2) for a in (narrow, line, wide)] == [0.08, 0.39, 0.29]


class TestWarmPriorsThroughTheOrigin:
    """``coef_prior`` on the one solve that dropped it: standardized, no
    intercept. The three other branches move the penalty's target into the
    right-hand side; ``solve_scaled_through_origin`` scaled, solved and
    unscaled the bare ``b`` and never read the prior, so a warm-started,
    standardized, through-origin fit was an un-warmed one with no error
    (review 2026-09-18, B2)."""

    @staticmethod
    def _frame(n=500, seed=9):
        rng = np.random.default_rng(seed)
        x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
        return pl.DataFrame({"x0": x0, "x1": x1, "y0": 3.0 * x0 - 2.0 * x1})

    @staticmethod
    def _coef(df, **kw):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            half_life=float("inf"),
            min_weight=5.0,
            max_rows_between_solves=1,
            coef_every=0,
            fit_intercept=False,
            standardize=True,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        return np.array(out["m"].struct.field("coef").to_list()[-1])

    def test_an_overwhelming_ridge_lands_on_the_prior(self):
        prior = [3.0, -2.0]
        got = self._coef(self._frame(), ridge=1e12, coef_prior=[prior])
        assert got == pytest.approx(prior, abs=1e-3), got

    def test_a_prior_moves_a_moderate_ridge_toward_itself(self):
        """At a ridge that bites, the fit with a prior sits strictly closer to
        the prior than the fit without one -- on every slope -- and the two
        are not the same numbers."""
        df = self._frame()
        prior = np.array([5.0, 1.0])
        with_prior = self._coef(df, ridge=50.0, coef_prior=[prior.tolist()])
        without = self._coef(df, ridge=50.0)
        assert not np.allclose(with_prior, without), "the prior changed nothing"
        assert (np.abs(with_prior - prior) < np.abs(without - prior)).all(), (with_prior, without)


class TestRidgeDecayPenalisesTheIntercept:
    """Task 148: ``ridge_scale``'s system is RLS's, ``(W S + prior_scale *
    ridge * I) b = W r``, so its penalty reaches the intercept, where a plain
    ``ridge`` never does; and a ``coef_prior`` intercept is what it shrinks
    toward. A constant target of 5 shows both."""

    def _intercepts(self, **kw):
        n = 200
        rng = np.random.default_rng(0)
        df = pl.DataFrame({"x": rng.standard_normal(n), "y": np.full(n, 5.0)})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            half_life=50.0,
            ridge=10.0,
            min_weight=0.0,
            coef_every=0,
            max_rows_between_solves=1,
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            out = po.ModelBank([spec]).fit_predict(df).unnest("m")
        return np.array([c[0] for c in out["coef"].to_list()])

    def test_a_plain_ridge_leaves_the_intercept_alone(self):
        assert self._intercepts() == pytest.approx(5.0, abs=1e-9)

    def test_ridge_decay_shrinks_it_until_the_prior_fades(self):
        got = self._intercepts(ridge_scale="sum")
        assert got[20] == pytest.approx(3.504, abs=1e-3)
        assert got[199] == pytest.approx(4.954, abs=1e-3)
        assert np.all(np.diff(got[1:]) > 0.0)

    def test_it_shrinks_toward_the_prior_intercept(self):
        got = self._intercepts(ridge_scale="sum", coef_prior=[[5.0, 0.0]])
        assert got == pytest.approx(5.0, abs=1e-9)


class TestTheRowClockSolveCadence:
    """Without a clock column the clock is the row's number, the first row
    being 1, so the model's start is a row before its first row. Before the
    first solve ``solve_every=n`` solves on the ``n``-th row, the row
    ``max_rows_between_solves=n`` does: the two are one schedule, as
    ``coef_every=n`` and ``max_rows_between_coefs=n`` are. Measured from the
    first row, whose step is 0, the clock solved a row after the cap until
    the first solve, and with a ``min_weight`` never met, which the first
    solve waited for, the whole stream long (task 278, D13), which ``coef``
    and ``support_coef`` showed. Both schedules' first period ends on that
    ``n``-th row, and before it the model solves on the default weight-share
    schedule (task 278, "Default cadence first"): at a half-life of 20 rows,
    every row. (Until then both counted from the first solve, on the row the
    weight reached a row per unknown, the third here.)
    """

    @staticmethod
    def run(**kw) -> pl.DataFrame:
        rng = np.random.default_rng(0)
        n = 60
        x = rng.standard_normal(n)
        df = pl.DataFrame({"x": x, "y": 2.0 * x + rng.standard_normal(n)})
        spec = po.spec.ewridge(
            "m",
            targets=["y"],
            features=["x"],
            half_life=20.0,
            min_weight=1e9,
            max_rows_between_coefs=1,
            **kw,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            return po.ModelBank([spec]).fit_predict(df).unnest("m")

    def test_solve_every_is_its_row_cap(self):
        clock = self.run(solve_every=7.0)
        cap = self.run(solve_every=1e9, max_rows_between_solves=7)
        assert clock.equals(cap, null_equal=True)
        # The solves, seen where the fit the rows report moves: every row of
        # the first period, which ends on the 7th (0-based 6), then every 7th
        # row from it.
        c = clock["support_coef"].to_list()
        moved = [i for i in range(1, len(c)) if c[i] != c[i - 1]]
        assert moved == [1, 2, 3, 4, 5, *range(6, 60, 7)]


def _relation_frame(seed: int, n: int = 2000, gaps: bool = True) -> pl.DataFrame:
    """Three features of unlike scales and means, a slope that changes half
    way, two targets -- ``z`` null where ``x0`` is high -- uneven weights with
    a zero on the first row and every 37th, and a clock that is the row
    number."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = rng.standard_normal((n, 3)) * [1.0, 2.0, 0.5] + [0.5, -1.0, 3.0]
    beta = np.array([2.0, -1.0, 0.5])
    y = 1.0 + x @ beta + np.where(t < n / 2, 0.0, 1.5) * x[:, 0] + rng.standard_normal(n)
    z = 0.5 - x @ beta[::-1] + rng.standard_normal(n)
    if gaps:
        z[x[:, 0] > 1.2] = np.nan
    w = rng.uniform(0.5, 1.5, n)
    w[::37] = 0.0
    return pl.DataFrame(
        {"t": t, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y, "w": w}
        | {"z": pl.Series(z).fill_nan(None)}
    )


_GRID = pytest.mark.extended(reason="a grid: three seeds x two half-lives")
#: (seed, half_life): one stream in the essentials, three seeds at a short and
#: a long half-life in the extended tier.
_CASES = [
    pytest.param(278, 200.0, id="s278-h200"),
    *(
        pytest.param(s, h, id=f"s{s}-h{h:g}", marks=_GRID)
        for s in (1, 2, 3)
        for h in (50.0, 1000.0)
    ),
]
_SEEDS = [
    pytest.param(278, id="s278"),
    *(
        pytest.param(s, id=f"s{s}", marks=pytest.mark.extended(reason="a grid: three seeds"))
        for s in (1, 2, 3)
    ),
]
_F = ["x0", "x1", "x2"]


class TestParameterRelations:
    """``ewridge``'s own keywords against each other and against their limits
    (task 278, batch 27): the spellings that must be one model, the values at
    which a keyword must vanish, and the direction it must move the fit. Each
    relation was measured on today's model (2026-10-26) before it was
    written down; an exact one is held to the bit."""

    @staticmethod
    def run(df: pl.DataFrame, **kw) -> tuple[pl.DataFrame, po.ModelBank]:
        base = dict(
            targets=["y", "z"],
            features=_F,
            clock="t",
            gap_cap=1e9,
            weight="w",
            min_weight=10.0,
            coef_every=0,
            max_rows_between_solves=1,
        )
        bank = po.ModelBank([po.spec.ewridge("m", **{**base, **kw})], threads=1)
        with warnings.catch_warnings():
            # A heavy ridge on a narrow column is mostly ridge: said, not wrong.
            warnings.simplefilter("ignore", po.ReadinessWarning)
            return bank.fit_predict(df).unnest("m"), bank

    @staticmethod
    def last_coef(bank: po.ModelBank) -> np.ndarray:
        """Every target's coefficients, intercept first, in the bank's order."""
        return bank.coef("m")["coef"].to_numpy()

    # ---- spellings of one model ----

    @pytest.mark.parametrize(("seed", "half_life"), _CASES)
    def test_a_ridge_grid_slot_is_the_lone_spec_at_that_ridge(self, seed, half_life):
        df = _relation_frame(seed)
        grid, _ = self.run(df, half_life=half_life, ridge=[1e-6, 2.0])
        lone, _ = self.run(df, half_life=half_life, ridge=2.0)
        assert lone["pred_y"].is_not_null().sum() > 1900
        for t in ("y", "z"):
            for f in ("pred", "resid"):
                assert grid[f"{f}_{t}__r2"].equals(lone[f"{f}_{t}"]), (f, t)
        assert grid["weight_sum"].equals(lone["weight_sum"])
        # The other slot is another fit: the grid is not one model twice.
        assert not grid["pred_y__r0.000001"].equals(grid["pred_y__r2"])

    @pytest.mark.parametrize(("seed", "half_life"), _CASES)
    def test_a_feature_set_is_the_spec_on_its_columns(self, seed, half_life):
        df = _relation_frame(seed)
        sets, _ = self.run(df, half_life=half_life, feature_sets={"a": ["x0"], "b": ["x0", "x2"]})
        for name, cols in (("a", ["x0"]), ("b", ["x0", "x2"])):
            alone, _ = self.run(df, half_life=half_life, features=cols)
            for t in ("y", "z"):
                for f in ("pred", "resid"):
                    assert sets[f"{f}_{t}__{name}"].equals(alone[f"{f}_{t}"]), (name, f, t)

    @pytest.mark.parametrize(("seed", "half_life"), _CASES)
    def test_one_set_of_every_feature_is_the_plain_spec(self, seed, half_life):
        df = _relation_frame(seed)
        one, _ = self.run(df, half_life=half_life, feature_sets={"all": _F})
        plain, _ = self.run(df, half_life=half_life)
        assert one.equals(plain, null_equal=True)

    @pytest.mark.parametrize(("seed", "half_life"), _CASES)
    def test_own_rows_is_pairwise_where_no_target_is_ever_null(self, seed, half_life):
        full = _relation_frame(seed, gaps=False)
        own, _ = self.run(full, half_life=half_life)
        pairwise, _ = self.run(full, half_life=half_life, target_gaps="pairwise")
        assert own.equals(pairwise, null_equal=True)
        # With gaps the two rules read different rows, so they part.
        gappy = _relation_frame(seed)
        a, _ = self.run(gappy, half_life=half_life)
        b, _ = self.run(gappy, half_life=half_life, target_gaps="pairwise")
        assert not a["pred_z"].equals(b["pred_z"])

    @pytest.mark.parametrize(("seed", "half_life"), _CASES)
    def test_closed_both_is_right_where_no_row_sits_on_the_window_edge(self, seed, half_life):
        """On the clock ``0.7 t + 0.05`` no two rows lie exactly 100 apart,
        so no row is ever one window old and the edge rule decides nothing.
        On the integer clock rows land on the edge and the rules part."""
        df = _relation_frame(seed)
        off = df.with_columns(pl.col("t") * 0.7 + 0.05)
        right, _ = self.run(off, half_life=half_life, window_size=100.0)
        both, _ = self.run(off, half_life=half_life, window_size=100.0, closed="both")
        assert right.equals(both, null_equal=True)
        on_r, _ = self.run(df, half_life=half_life, window_size=100.0)
        on_b, _ = self.run(df, half_life=half_life, window_size=100.0, closed="both")
        assert not on_r["weight_sum"].equals(on_b["weight_sum"])

    # ---- the ridge: its direction and its limits ----

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_the_standardized_slopes_shrink_as_the_ridge_grows(self, seed):
        """Under ``standardize`` the ridge is on the slopes in units of each
        feature's spread, so ``|b * s|`` is the norm it penalizes, and it
        cannot grow with the ridge (a ridge path is monotone in the penalized
        norm). Measured at seed 278: 3.38, 3.34, 3.06, 1.65, 0.30, 0.032 for
        ridge 0, 0.01, ... 100."""
        df = _relation_frame(seed)
        spread = df.select(_F).to_numpy().std(axis=0)
        norms = []
        for r in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4):
            _, bank = self.run(df, targets=["y"], half_life=float("inf"), standardize=True, ridge=r)
            norms.append(float(np.linalg.norm(self.last_coef(bank)[1:] * spread)))
        print("standardized slope norms:", [f"{v:.4g}" for v in norms])
        assert np.all(np.diff(norms) < 0), norms
        assert norms[-1] < 1e-2 * norms[0]

    @pytest.mark.parametrize("standardize", [False, True])
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_an_overwhelming_ridge_leaves_the_slopes_at_the_covariance_over_it(
        self, seed, standardize
    ):
        """At ``half_life=inf`` the system is ``(C + r D) b = c``, ``C`` the
        weighted covariance of the features, ``c`` theirs with ``y`` and
        ``D`` the identity (or the variances, under ``standardize``). As
        ``r`` grows, ``r b -> D^-1 c`` with an error of order ``|C| / r``,
        and the unpenalized intercept ``ybar - xbar' b`` goes to the
        weighted mean of ``y``. ``numpy.cov(aweights=)`` computes ``c``.
        Measured 2026-10-26 over the four seeds: ``r b`` within 1e-8 to
        4.1e-8 (relative) of the limit at ``r = 1e8`` and the intercept
        2.7e-8 to 6.1e-8 from the mean, both falling by 1e4 at ``r = 1e12``."""
        df = _relation_frame(seed).filter(pl.col("w") > 0)
        x, y, w = df.select(_F).to_numpy(), df["y"].to_numpy(), df["w"].to_numpy()
        cov = np.cov(np.column_stack([x, y]), rowvar=False, aweights=w, bias=True)
        d = np.diag(cov)[:3] if standardize else np.ones(3)
        limit = cov[:3, 3] / d
        ybar = float(np.average(y, weights=w))
        errs = []
        for r in (1e8, 1e12):
            _, bank = self.run(
                df, targets=["y"], half_life=float("inf"), standardize=standardize, ridge=r
            )
            c = self.last_coef(bank)
            slope_err = float(np.max(np.abs(r * c[1:] - limit) / np.abs(limit)))
            errs.append((slope_err, abs(c[0] - ybar)))
            print(f"ridge {r:g}: r b off the limit by {slope_err:.2g}, intercept {c[0] - ybar:.2g}")
        (s8, i8), (s12, i12) = errs
        # The error's order: |C| / r with |C| about 4, so ~4e-8 relative at
        # 1e8 (bounded at 1e-5 for the seeds' spread) and the next order at
        # 1e12, down to the double's rounding of ``r b``.
        assert s8 < 1e-5 and s12 < max(1e-3 * s8, 1e-9), errs
        assert i8 < 1e-6 and i12 < max(1e-3 * i8, 1e-13), errs

    # ---- coef_prior ----

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_a_prior_fades_from_the_fit_with_the_ridge(self, seed):
        """``coef_prior`` enters the system only through ``r * prior`` on
        its right-hand side, so its effect on a prediction is linear in the
        ridge and none at 0, to the bit. Measured at seed 278: the largest
        move 5.4e-5, 5.4e-9 and 5.4e-13 at ridge 1e-6, 1e-10, 1e-14."""
        df = _relation_frame(seed)
        prior = [[0.0, 7.0, 8.0, 6.0]]
        kw = dict(targets=["y"], half_life=float("inf"))
        moves = []
        for r in (1e-6, 1e-10):
            a, _ = self.run(df, ridge=r, coef_prior=prior, **kw)
            b, _ = self.run(df, ridge=r, **kw)
            pa, pb = a["pred_y"].to_numpy().astype(float), b["pred_y"].to_numpy().astype(float)
            moves.append(float(np.nanmax(np.abs(pa - pb))))
        print("the prior's largest move at ridge 1e-6, 1e-10:", moves)
        assert moves[0] > 1e-7, "the prior moves nothing at a ridge of 1e-6"
        assert moves[1] / 1e-10 == pytest.approx(moves[0] / 1e-6, rel=1e-3)
        a, _ = self.run(df, ridge=0.0, coef_prior=prior, **kw)
        b, _ = self.run(df, ridge=0.0, **kw)
        assert a.equals(b, null_equal=True)

    @pytest.mark.parametrize("standardize", [False, True])
    def test_an_overwhelming_ridge_lands_on_the_prior_with_the_intercept_free(self, standardize):
        """With an intercept the slopes land on the prior and the intercept,
        which a plain ridge leaves free, is the weighted mean of
        ``y - x' prior``, both off by order ``1 / r``. Measured 2026-10-26
        at ``r = 1e12``: the slopes within 3.8e-11 (9.1e-12 standardized)
        and the intercept within 2.8e-11 (1.5e-11), each 1e4 times its
        distance at ``r = 1e8``."""
        df = _relation_frame(278).filter(pl.col("w") > 0)
        x, y, w = df.select(_F).to_numpy(), df["y"].to_numpy(), df["w"].to_numpy()
        slopes = np.array([7.0, 8.0, 6.0])
        intercept = float(np.average(y - x @ slopes, weights=w))
        _, bank = self.run(
            df,
            targets=["y"],
            half_life=float("inf"),
            standardize=standardize,
            ridge=1e12,
            coef_prior=[[0.0, *slopes]],
        )
        c = self.last_coef(bank)
        assert c[1:] == pytest.approx(slopes, abs=1e-9)
        assert c[0] == pytest.approx(intercept, abs=1e-9)

    @pytest.mark.parametrize("standardize", [False, True])
    @pytest.mark.parametrize("half_life", [200.0, float("inf")])
    def test_under_mean_the_priors_intercept_moves_nothing(self, half_life, standardize):
        """Under ``ridge_scale="mean"`` the intercept is not penalized, so
        the prior's intercept slot is never read: any value gives the same
        output, to the bit. Under ``"sum"`` the intercept is penalized and
        the slot acts."""
        df = _relation_frame(278)
        kw = dict(targets=["y"], half_life=half_life, ridge=3.0)
        a, _ = self.run(df, standardize=standardize, coef_prior=[[0.0, 1.0, -2.0, 0.5]], **kw)
        b, _ = self.run(df, standardize=standardize, coef_prior=[[123.0, 1.0, -2.0, 0.5]], **kw)
        assert a.equals(b, null_equal=True)
        if not standardize:
            c, _ = self.run(df, ridge_scale="sum", coef_prior=[[0.0, 1.0, -2.0, 0.5]], **kw)
            d, _ = self.run(df, ridge_scale="sum", coef_prior=[[123.0, 1.0, -2.0, 0.5]], **kw)
            assert not c["pred_y"].equals(d["pred_y"])

    # ---- standardize ----

    @pytest.mark.parametrize(
        "window", [{}, {"fit_intercept": False}, {"window_size": 300.0}], ids=str
    )
    @pytest.mark.parametrize("k", [-20, 5, 30])
    def test_a_feature_scaled_by_a_power_of_two_predicts_the_same_to_the_bit(self, k, window):
        """Standardized, the solve reads each feature only through its
        z-score, and scaling by ``2^k`` changes no bit of a z-score: the
        mean, the spread and the difference all scale exactly. So every
        prediction is the same, to the bit, and the feature's slope is
        scaled by exactly ``2^-k``; the other coefficients do not move."""
        df = _relation_frame(278)
        s = 2.0**k
        kw = dict(half_life=200.0, standardize=True, ridge=0.5, **window)
        a, ba = self.run(df, **kw)
        b, bb = self.run(df.with_columns(pl.col("x1") * s), **kw)
        for c in ("pred_y", "pred_z", "resid_y", "resid_z", "weight_sum"):
            assert a[c].equals(b[c]), c
        ca, cb = self.last_coef(ba), self.last_coef(bb)
        on_x1 = (ba.coef("m")["term"] == "x1").to_numpy()
        assert on_x1.sum() == 2
        np.testing.assert_array_equal(cb[on_x1], ca[on_x1] / s)
        np.testing.assert_array_equal(cb[~on_x1], ca[~on_x1])

    # ---- ridge_scale ----

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_without_decay_sum_at_ridge_r_is_mean_at_r_over_the_weight(self, seed):
        """Through the origin at ``half_life=inf`` the sum form solves
        ``(W S + r I) b = W s`` and the mean form ``(S + r' I) b = s``, ``W``
        the weight learned: one system at ``r' = r / W``. ``W`` grows, so the
        two meet at the last row only. Measured 4.9e-16 apart (relative)."""
        df = _relation_frame(seed)
        kw = dict(targets=["y"], half_life=float("inf"), fit_intercept=False)
        _, by_sum = self.run(df, ridge=0.5, ridge_scale="sum", **kw)
        weight = by_sum.gram("m")[0]["weight_sum"]
        _, by_mean = self.run(df, ridge=0.5 / weight, **kw)
        a, b = self.last_coef(by_sum), self.last_coef(by_mean)
        np.testing.assert_allclose(b, a, rtol=1e-13, atol=0.0)
        # The ridge is not negligible here: at ``r`` itself the mean form is
        # another fit.
        _, other = self.run(df, ridge=0.5 * 1e3, **kw)
        assert np.max(np.abs(self.last_coef(other) - a)) > 1e-3
