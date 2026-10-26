"""Oracle agreement for the models PLAN section 9 class 1 promised but that
were only property-tested: Kalman against a numpy reference (T-A1), and the
lasso against its own optimality conditions and a numpy reference (T-A2).

The lasso's KKT check is deliberately *not* a second copy of coordinate
descent: it verifies the KKT conditions of the penalized objective, which any
correct solver must satisfy, so it cannot agree with a bug the way a ported
implementation could. It sees only the last solve, so the pred path is held to
`reference.lasso_ref` as well. That is a descent written from the objective,
run from zero to a tolerance at which the start cannot matter, on the
documented solve schedule.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import ledger
import polars_online as po
from data import synthetic
from reference import EwMedian, ftrl_ref, kalman_ref, lasso_ref, robust_ref

TIER = "mixed"

MAXD = 50.0


def _arrays(df, k):
    x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(k)])
    n = df.height
    dc = np.zeros(n)
    dc[1:] = np.diff(df["t"].to_numpy())
    return x, np.clip(dc, 0.0, MAXD), df["w"].to_numpy()


def _rowcount_clock(n):
    """Deltas and weights a spec with no `clock`/`weight` column implies."""
    dc = np.ones(n)
    dc[0] = 0.0
    return dc, np.ones(n)


def _close(got, exp, tol=1e-9, what=""):
    both_nan = np.isnan(got) & np.isnan(exp)
    assert (np.isnan(got) == np.isnan(exp)).all(), f"{what}: null patterns differ"
    ok = both_nan | (np.abs(got - exp) <= tol * (1.0 + np.abs(exp)))
    assert ok.all(), f"{what}: max diff {np.nanmax(np.abs(got - exp))}"


class TestKalmanOracle:
    """T-A1: Kalman vs tests/reference.py::kalman_ref."""

    def _compare(self, df, k=3, targets=("y0",), **kw):
        x, dc, w = _arrays(df, k)
        y = np.column_stack([df[t].to_numpy() for t in targets])
        ref = kalman_ref(x, y, dc, w, gap_cap=MAXD, **kw)
        spec = po.spec.kalman(
            "m",
            targets=list(targets),
            features=[f"x{j}" for j in range(k)],
            clock="t",
            gap_cap=MAXD,
            weight="w",
            half_life=kw.get("half_life", 500.0),
            # One or the other (review 2026-10-06, PC6).
            coef_half_life=None if "q" in kw else kw.get("coef_half_life", 100.0),
            q=kw.get("q"),
            obs_var=kw.get("obs_var"),
            p0=kw.get("p0"),
            share_p=kw.get("share_p", False),
            min_weight=kw.get("min_weight", 10.0),
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for j, t in enumerate(targets):
            for field, key in (("pred_", "pred"), ("resid_", "resid")):
                got = out["m"].struct.field(f"{field}{t}").to_numpy().astype(float)
                _close(got, ref[key][:, j], what=f"{field}{t}")
        _close(
            out["m"].struct.field("weight_sum").to_numpy().astype(float),
            ref["weight_sum"],
            what="weight_sum",
        )
        return out

    def test_scalar_coef_halflife(self):
        df, _ = synthetic(seed=71, n_groups=1, n_rows=300, k=3, null_frac=0.0)
        self._compare(df)

    @pytest.mark.parametrize(
        ("targets", "share_p"),
        [(("y0",), False), (("y0", "y1"), True)],
        ids=["one target", "two targets, shared P"],
    )
    def test_a_row_with_no_prediction_ages_sigma2s_weight(self, targets, share_p):
        """sigma2's weight ages on every row the filter sees, a row with a
        target, a weight and no prediction included: one that comes after a
        run of rows of weight 0 has taken the weight under ``min_weight``
        (review 2026-09-12, N6; ``kalman.rs``). No other case withholds a
        prediction once they have begun, and kalman_ref aged the weight on
        such a row by nothing, so its sigma2 forgot less across it than the
        core's: 1.75e-3 from the bank on one target (task 177). The run of
        230 rows of weight 0 holds the predictions back on the rows at its
        end and just after it (asserted)."""
        df, _ = synthetic(
            seed=71, n_groups=1, n_rows=420, k=3, n_targets=len(targets), null_frac=0.0
        )
        w = df["w"].to_numpy().copy()
        w[100:330] = 0.0
        df = df.with_columns(pl.Series("w", w))
        out = self._compare(df, targets=targets, share_p=share_p)
        for t in targets:
            pred = out["m"].struct.field(f"pred_{t}").to_numpy().astype(float)
            withheld = np.isnan(pred[330:])
            assert withheld.any() and not withheld.all(), f"{t}: none withheld after the run"

    def test_a_zero_weight_first_row(self):
        """Hard rule 9 at the head of the stream: a first row of weight 0
        advances the clock and teaches nothing, and the feature moments'
        update after it is 0/0. kalman_ref divided by that weight unguarded,
        so none of its 120 predictions was a number while the bank's 108
        were (review 2026-10-06, TA1)."""
        df, _ = synthetic(seed=71, n_groups=1, n_rows=120, k=3, null_frac=0.0)
        w = df["w"].to_numpy().copy()
        w[0] = 0.0
        out = self._compare(df.with_columns(pl.Series("w", w)))
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        assert np.isfinite(pred).sum() > 100, "the row of weight 0 must not silence the model"

    def test_per_factor_halflife_with_pinning(self):
        df, _ = synthetic(seed=72, n_groups=1, n_rows=300, k=3, null_frac=0.0)
        # intercept pinned, x0 slow, x1 fast, x2 pinned
        self._compare(df, coef_half_life=[float("inf"), 500.0, 30.0, float("inf")])

    def test_explicit_q(self):
        df, _ = synthetic(seed=73, n_groups=1, n_rows=250, k=3, null_frac=0.0)
        self._compare(df, q=[0.0, 0.01, 0.02, 0.0])

    def test_fixed_obs_var_and_p0(self):
        df, _ = synthetic(seed=74, n_groups=1, n_rows=250, k=3, null_frac=0.0)
        self._compare(df, obs_var=0.25, p0=4.0)

    def test_multi_target_per_target_p(self):
        df, _ = synthetic(seed=75, n_groups=1, n_rows=250, k=3, n_targets=2, null_frac=0.0)
        self._compare(df, targets=("y0", "y1"))

    def test_multi_target_shared_p(self):
        df, _ = synthetic(seed=76, n_groups=1, n_rows=250, k=3, n_targets=2, null_frac=0.0)
        self._compare(df, targets=("y0", "y1"), share_p=True)

    def test_null_targets_and_features(self):
        # The null policy is the part most likely to drift between the model and
        # a reference, so it gets its own oracle comparison.
        df, _ = synthetic(seed=77, n_groups=1, n_rows=300, k=3, null_frac=0.05)
        self._compare(df)

    def test_no_intercept(self):
        df, _ = synthetic(seed=78, n_groups=1, n_rows=200, k=2, null_frac=0.0)
        x, dc, w = _arrays(df, 2)
        y = df["y0"].to_numpy().reshape(-1, 1)
        ref = kalman_ref(x, y, dc, w, fit_intercept=False, min_weight=10.0, gap_cap=MAXD)
        spec = po.spec.kalman(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            fit_intercept=False,
            clock="t",
            gap_cap=MAXD,
            weight="w",
            half_life=500.0,
            coef_half_life=100.0,
            min_weight=10.0,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        _close(
            out["m"].struct.field("pred_y0").to_numpy().astype(float),
            ref["pred"][:, 0],
            what="pred (no intercept)",
        )


class TestLassoOptimality:
    """T-A2: the emitted coefficients must satisfy the lasso/elastic-net KKT
    conditions on the model's own standardized statistics."""

    @staticmethod
    def _ew_stats(x, y, dclock, w, half_life):
        """EW mean and raw second moments of z = [1, x] through every row,
        matching the core's mean-form recursion (stats updated after the row's
        prediction, so index i includes row i)."""
        n, k = x.shape
        kt = k + 1
        mean = np.zeros(kt)
        raw = np.zeros((kt, kt))
        ry = np.zeros(kt)
        wj = 0.0
        W = 0.0
        out = []
        for i in range(n):
            z = np.concatenate(([1.0], x[i]))
            lam = 0.5 ** (dclock[i] / half_life)
            W_new = lam * W + w[i]
            a, b = lam * W / W_new, w[i] / W_new
            mean = a * mean + b * z
            raw = a * raw + b * np.outer(z, z)
            W = W_new
            wj_new = lam * wj + w[i]
            ry = (lam * wj * ry + w[i] * z * y[i]) / wj_new
            wj = wj_new
            out.append((mean.copy(), raw.copy(), ry.copy()))
        return out

    def _kkt_residuals(self, x, y, dclock, w, half_life, coefs, lam_value, l1_ratio):
        """Return (b_std, g, l1) at the final row: g_i is the stationarity
        quantity that must equal l1*sign(b_i) where b_i != 0, and satisfy
        |g_i| <= l1 where b_i == 0."""
        mean, raw, ry = self._ew_stats(x, y, dclock, w, half_life)[-1]
        k = x.shape[1]
        cov = raw[1:, 1:] - np.outer(mean[1:], mean[1:])
        s = np.array(
            [
                np.sqrt(cov[i, i]) if cov[i, i] > 1e-10 * abs(raw[i + 1, i + 1]) else 0.0
                for i in range(k)
            ]
        )
        keep = s > 0
        c_mat = np.eye(k)
        for i in range(k):
            for j in range(k):
                if keep[i] and keep[j]:
                    c_mat[i, j] = cov[i, j] / (s[i] * s[j])
        ybar = ry[0]
        c_vec = np.array(
            [(ry[i + 1] - mean[i + 1] * ybar) / s[i] if keep[i] else 0.0 for i in range(k)]
        )
        # model coefficients (original units) -> standardized scale
        b_std = np.array([coefs[i + 1] * s[i] if keep[i] else 0.0 for i in range(k)])
        l1 = lam_value * l1_ratio
        l2 = lam_value * (1.0 - l1_ratio)
        g = c_vec - c_mat @ b_std - l2 * b_std
        return b_std, g, l1

    @pytest.mark.parametrize("lam_value", [0.0, 0.01, 0.1])
    @pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
    def test_kkt_conditions_hold(self, lam_value, l1_ratio):
        df, _ = synthetic(seed=81, n_groups=1, n_rows=400, k=4, null_frac=0.0)
        path = sorted({0.1, 0.01, 0.0} | {lam_value}, reverse=True)
        spec = po.spec.lasso(
            "m",
            targets=["y0"],
            features=[f"x{j}" for j in range(4)],
            penalties=path,
            l1_ratio=l1_ratio,
            half_life=1e9,
            min_weight=10.0,
            max_rows_between_solves=1,
            max_iter=2000,
            tol=1e-14,
        )
        out = po.ModelBank([spec]).fit_predict(df)

        # coef is emitted on the last row of the chunk: flat, (path x k_total)
        flat = out["m"].struct.field("coef").to_list()[-1]
        kt = 5
        idx = path.index(lam_value)
        coefs = flat[idx * kt : (idx + 1) * kt]

        # The spec uses neither a clock column nor a weight column, so the
        # model sees a row-count clock and unit weights; the reference stats
        # must be built the same way.
        x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(4)])
        dc, w = _rowcount_clock(df.height)
        b_std, g, l1 = self._kkt_residuals(
            x, df["y0"].to_numpy(), dc, w, 1e9, coefs, lam_value, l1_ratio
        )
        tol = 1e-6
        for i, (b, gi) in enumerate(zip(b_std, g, strict=True)):
            if abs(b) > 1e-9:
                assert abs(gi - l1 * np.sign(b)) < tol, (
                    f"stationarity violated at active coord {i}: g={gi}, expected {l1 * np.sign(b)}"
                )
            else:
                assert abs(gi) <= l1 + tol, (
                    f"a zero coefficient at coord {i} should have |g| <= {l1}, got {abs(gi)}"
                )

    def test_path_is_monotone_in_sparsity(self):
        # A larger penalty can never produce more non-zero coefficients.
        df, _ = synthetic(seed=82, n_groups=1, n_rows=400, k=4, null_frac=0.0)
        path = [1.0, 0.1, 0.01, 0.0]
        spec = po.spec.lasso(
            "m",
            targets=["y0"],
            features=[f"x{j}" for j in range(4)],
            penalties=path,
            half_life=1e9,
            min_weight=10.0,
            max_rows_between_solves=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        flat = np.array(out["m"].struct.field("coef").to_list()[-1]).reshape(len(path), 5)
        nnz = [(np.abs(row[1:]) > 1e-12).sum() for row in flat]
        assert nnz == sorted(nnz), f"sparsity not monotone along the path: {nnz}"

    def test_zero_penalty_solves_the_normal_equations(self):
        # lambda = 0 must satisfy the *unpenalized* stationarity condition
        # exactly: C b = c on the standardized stats.
        df, _ = synthetic(seed=83, n_groups=1, n_rows=400, k=3, null_frac=0.0)
        spec = po.spec.lasso(
            "m",
            targets=["y0"],
            features=["x0", "x1", "x2"],
            penalties=[0.0],
            half_life=1e9,
            min_weight=10.0,
            max_rows_between_solves=1,
            max_iter=5000,
            tol=1e-15,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        coefs = out["m"].struct.field("coef").to_list()[-1]
        x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(3)])
        dc, w = _rowcount_clock(df.height)
        _, g, _ = self._kkt_residuals(x, df["y0"].to_numpy(), dc, w, 1e9, coefs, 0.0, 1.0)
        assert np.max(np.abs(g)) < 1e-8, f"normal equations not satisfied: {g}"


def test_intercept_matches_the_weighted_means():
    """A lasso intercept must equal ybar - m . beta on the model's own stats,
    for every path point."""
    df, _ = synthetic(seed=84, n_groups=1, n_rows=300, k=3, null_frac=0.0)
    path = [0.05, 0.0]
    spec = po.spec.lasso(
        "m",
        targets=["y0"],
        features=["x0", "x1", "x2"],
        penalties=path,
        half_life=1e9,
        min_weight=10.0,
        max_rows_between_solves=1,
    )
    out = po.ModelBank([spec]).fit_predict(df)
    flat = np.array(out["m"].struct.field("coef").to_list()[-1]).reshape(len(path), 4)

    x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(3)])
    dc, w = _rowcount_clock(df.height)
    stats = TestLassoOptimality._ew_stats(x, df["y0"].to_numpy(), dc, w, 1e9)
    mean, _, ry = stats[-1]
    for row in flat:
        expected = ry[0] - mean[1:] @ row[1:]
        assert abs(row[0] - expected) < 1e-8, f"intercept {row[0]} != {expected}"


class TestLassoPredPath:
    """T-A2, the pred path: every row's ``pred`` and ``resid`` per path point,
    ``weight_sum`` and every emitted ``coef``, against
    ``tests/reference.py::lasso_ref``.

    The KKT check above sees one snapshot, the last solve's. This sees every
    solve through the predictions it made. It checks when each solve ran:
    ``solve_every``, its default, ``max_rows_between_solves`` and the forced
    first solve. It checks what each was fitted from: the decay, a capped gap,
    and skipped and zero-weight rows. And it checks that each reached the
    optimum, not wherever a warm start left the descent. The reference
    descends from zero to 1e-14, so no warm start can change the answer it
    holds the library to.
    """

    FEATURES = ["x0", "x1", "x2", "x3"]
    PATH = [0.2, 0.05, 0.0]
    MAX_DCLOCK = 6.0
    # Measured over the cases below as `_close` reads an error,
    # |got - expected| / (1 + |expected|): pred 1.7e-14, resid 3.3e-14, coef
    # 1.2e-12, weight_sum exact. Each tolerance is 100x the largest it covers,
    # rounded up to a power of ten.
    PRED_TOL = 1e-11
    COEF_TOL = 1e-9
    # At the library's own `tol` (1e-10) and `max_iter` (100) each
    # descent stops short of the optimum: pred and resid measured 1.7e-10
    # from the reference, and the tolerance is set the same way.
    DEFAULT_DESCENT_TOL = 1e-7

    @staticmethod
    def _stream(seed, n=300):
        """Four features and one target. ``x2`` is out of the model but
        correlated with ``x0``, and ``x3`` is small enough for the larger
        penalties to zero. ``x1`` sits at a level of 2, which only the
        centring and the intercept can absorb.

        The clock steps are dyadic, so the clock since a solve sums exactly,
        and a solve that falls due exactly on ``solve_every`` is decided by
        the schedule's ``>=`` rather than by rounding. Two gaps of 40 exceed
        ``gap_cap``. A twentieth of the rows weigh 0, the first row among
        them. Five rows have a null feature and are skipped."""
        rng = np.random.default_rng(seed)
        dt = rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0], size=n)
        dt[0] = 0.0
        dt[[90, 200]] = 40.0
        x0 = rng.standard_normal(n)
        x1 = 2.0 + rng.standard_normal(n)
        x2 = 0.6 * x0 + 0.8 * rng.standard_normal(n)
        x3 = -1.0 + 0.7 * rng.standard_normal(n)
        y = 0.3 + x0 - 0.6 * x1 + 0.25 * x3 + 0.5 * rng.standard_normal(n)
        w = rng.uniform(0.5, 1.5, n)
        w[0] = 0.0
        w[rng.choice(np.arange(1, n), size=n // 20 - 1, replace=False)] = 0.0
        x1[[40, 41, 42, 95, 150]] = np.nan
        frame = {"t": np.cumsum(dt), "x0": x0, "x1": x1, "x2": x2, "x3": x3, "y": y, "w": w}
        return pl.DataFrame(frame).with_columns(pl.col("x1").fill_nan(None))

    def _compare(self, df, default_descent=False, **kw):
        """Fit ``df`` with a spec that ``kw`` completes, and hold the output to
        the reference given the same ``kw``. ``default_descent`` leaves
        ``tol`` and ``max_iter`` at the library's defaults."""
        descent = {} if default_descent else {"tol": 1e-14, "max_iter": 100_000}
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=self.FEATURES,
            penalties=self.PATH,
            clock="t",
            gap_cap=self.MAX_DCLOCK,
            weight="w",
            coef_every=0,
            **descent,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        n, npath, kt = df.height, len(self.PATH), len(self.FEATURES) + 1
        x = df.select(self.FEATURES).to_numpy()
        y = df["y"].to_numpy()
        dc = np.zeros(n)
        dc[1:] = np.diff(df["t"].to_numpy())
        ref = lasso_ref(x, y, dc, df["w"].to_numpy(), self.PATH, gap_cap=self.MAX_DCLOCK, **kw)
        assert ref["solved"].sum() >= 5, "the stream should span several solves"

        # One pred and one resid field per path point, in path order.
        index = po.spec.output_index(spec)
        assert index.filter(pl.col("kind") == "pred")["penalty"].to_list() == self.PATH
        fields = {
            k: index.filter(pl.col("kind") == k)["field"].to_list() for k in ("pred", "resid")
        }
        tol = self.DEFAULT_DESCENT_TOL if default_descent else self.PRED_TOL
        got = np.column_stack(
            [out.struct.field(f).to_numpy().astype(float) for f in fields["pred"]]
        )
        for p, lam in enumerate(self.PATH):
            _close(got[:, p], ref["pred"][:, p], tol=tol, what=f"pred at lambda {lam}")
            resid = out.struct.field(fields["resid"][p]).to_numpy().astype(float)
            _close(resid, y - ref["pred"][:, p], tol=tol, what=f"resid at lambda {lam}")
        weight_sum = out.struct.field("weight_sum").to_numpy().astype(float)
        _close(weight_sum, ref["weight_sum"], tol=self.PRED_TOL, what="weight_sum")

        # The comparison can fail. The same reference read in sample (each
        # row scored with its own update, which hard rule 2 forbids), or with
        # every solve reaching the predictions a row late, is far outside it.
        z = np.column_stack([np.ones(n), x])
        in_sample = np.einsum("ipk,ik->ip", ref["coef"], z)
        late = np.full_like(in_sample, np.nan)
        late[2:] = np.einsum("ipk,ik->ip", ref["coef"][:-2], z[2:])
        for what, probe in (("in sample", in_sample), ("a row late", late)):
            assert np.nanmax(np.abs(got - probe)) > 1e-3, f"pred cannot tell {what} apart"

        if default_descent:
            # A descent here may stop at 100 sweeps on an early, badly
            # conditioned solve (counted in `solve_failures`), which leaves
            # `coef` short of the optimum before any row is scored with it.
            return
        rows = out.struct.field("coef").to_list()
        # `coef` is each row's last solve, and `min_weight` does not gate it:
        # it is null only before the first solve and on a skipped row.
        before_first = np.arange(n) < np.argmax(ref["solved"])
        expect_null = before_first | np.isnan(ref["weight_sum"])
        wrong = np.flatnonzero(np.array([r is None for r in rows]) != expect_null)
        assert wrong.size == 0, f"coef is null on the wrong rows, first {wrong[0]}"
        # A solve with no row of the target and no fit before it has no fit,
        # null inside the list (review round 4, CC1): NaN here, as the
        # reference has it.
        empty = [np.nan] * (npath * kt)
        coef = np.array(
            [empty if r is None else [np.nan if v is None else v for v in r] for r in rows],
            dtype=float,
        ).reshape(n, npath, kt)
        # The stream's first solves are NaN in the reference: from about as
        # few rows as features, their minimiser is not unique or barely is,
        # so there is nothing to hold the library to, and no row is scored
        # with one (`lasso_ref` raises if one is).
        held = ~np.isnan(ref["coef"])
        _close(coef[held], ref["coef"][held], tol=self.COEF_TOL, what="coef")
        # The soft threshold zeroed a coefficient of a fit the stream was
        # scored with, and zeroed exactly what the reference zeroed.
        zero, zero_ref, slopes = coef[..., 1:] == 0, ref["coef"][..., 1:] == 0, held[..., 1:]
        assert zero_ref[~np.isnan(ref["pred"][:, 0])].any(), "the L1 should zero something"
        assert (zero == zero_ref)[slopes].all(), "the L1 zeroed other coefficients"

    def test_a_solve_falls_due_on_the_clock_and_on_a_capped_gap(self):
        """``solve_every`` in clock units. Some steps land on it exactly, and
        a solve is then due (``>=``). Each gap past ``gap_cap`` forces a
        solve with its capped step. The half-life is finite, so the decay and
        the cap both reach the fit."""
        self._compare(
            self._stream(31), l1_ratio=1.0, half_life=40.0, min_weight=20.0, solve_every=2.5
        )

    def test_an_elastic_net_solves_on_a_row_cap_beside_the_clock(self):
        """``max_rows_between_solves`` beside ``solve_every``. A zero-weight
        row counts as a row; a skipped row does not. The capped gap equals
        ``solve_every``, so each gap forces a solve."""
        self._compare(
            self._stream(32),
            l1_ratio=0.5,
            half_life=40.0,
            min_weight=20.0,
            solve_every=6.0,
            max_rows_between_solves=5,
        )

    def test_the_default_cadence_is_by_weight(self):
        """No ``solve_every``: a solve once the weight learned since the last
        reaches ``ln 2 / 50`` of the weight the fit holds (task 115 (b)),
        about every ``half_life / 50`` = 2 clock units in steady state."""
        self._compare(self._stream(33), l1_ratio=1.0, half_life=100.0, min_weight=20.0)

    def test_the_first_solve_is_forced_when_the_floor_is_reached(self):
        """The clock cadence is out of reach and the row cap is 20. So the
        first solve is the forced one on the row the target reaches the
        model's floor, a row per coefficient, whatever ``min_weight`` says
        (task 278, D10), and the cap counts from it: every solve after, and
        so every scored row's fit, falls where that first one puts it. (A
        fit of the floor's five rows is not unique enough for the reference
        to hold, so ``min_weight`` scores no row with it.)"""
        self._compare(
            self._stream(34),
            l1_ratio=1.0,
            half_life=40.0,
            min_weight=25.0,
            solve_every=1e9,
            max_rows_between_solves=20,
        )

    def test_an_infinite_halflife_solves_after_every_row(self):
        """The default cadence without decay is a solve after every row."""
        self._compare(self._stream(35), l1_ratio=0.5, half_life=float("inf"), min_weight=20.0)

    def test_the_default_descent_reaches_the_same_predictions(self):
        """``tol`` and ``max_iter`` left at the library's defaults, the
        settings a user runs. Every scored row still agrees to within the
        descent's own tolerance."""
        self._compare(
            self._stream(33), default_descent=True, l1_ratio=1.0, half_life=100.0, min_weight=20.0
        )


class TestRobustOracles:
    """T-A3: Huber and quantile vs `tests/reference.py::robust_ref`. Huber's
    cases run under each scale its cut can be drawn in (task 271): the MAD,
    the default, and the RMS; ``scale_half_life`` gives the scale a memory of
    its own."""

    def _compare(
        self,
        df,
        k=2,
        targets=("y0",),
        model="huber",
        ref_kw=None,
        scale="mad",
        scale_half_life=None,
        **spec_kw,
    ):
        if model == "huber":
            ref_kw = {**(ref_kw or {}), "huber_scale": scale}
            # The reference is the one-pass recursion: the refit of the
            # first rows (task 277) is held to statsmodels in
            # `test_huber_refit.py`.
            spec_kw = {**spec_kw, "huber_scale": scale, "huber_refit_rows": 0}
            if scale_half_life is not None:
                ref_kw["huber_scale_half_life"] = scale_half_life
                spec_kw["huber_scale_half_life"] = scale_half_life
        x, dc, w = _arrays(df, k)
        y = np.column_stack([df[t].to_numpy() for t in targets])
        ref = robust_ref(
            x,
            y,
            dc,
            w,
            half_life=300.0,
            loss="huber" if model == "huber" else "quantile",
            min_weight=5.0,
            gap_cap=MAXD,
            **(ref_kw or {}),
        )
        spec = getattr(po.spec, model)(
            "m",
            targets=list(targets),
            features=[f"x{j}" for j in range(k)],
            clock="t",
            gap_cap=MAXD,
            weight="w",
            half_life=300.0,
            min_weight=5.0,
            max_rows_between_solves=1,
            **spec_kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for j, t in enumerate(targets):
            _close(
                out["m"].struct.field(f"pred_{t}").to_numpy().astype(float),
                ref["pred"][:, j],
                what=f"pred_{t}",
            )
            _close(
                out["m"].struct.field(f"resid_{t}").to_numpy().astype(float),
                ref["resid"][:, j],
                what=f"resid_{t}",
            )
        _close(
            out["m"].struct.field("weight_sum").to_numpy().astype(float),
            ref["weight_sum"],
            what="weight_sum",
        )
        return out

    @pytest.mark.parametrize("scale", ["mad", "rms"])
    def test_huber(self, scale):
        df, _ = synthetic(seed=91, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, scale=scale)

    @pytest.mark.parametrize("scale", ["mad", "rms"])
    def test_huber_with_outliers_and_a_scale_memory_of_its_own(self, scale):
        """One row in ten 30 noise standard deviations high, so the cut
        binds and the scales part, and the scale aging on a half-life a
        tenth of the fit's."""
        df, _ = synthetic(seed=99, n_groups=1, n_rows=400, k=2, null_frac=0.02)
        y = df["y0"].to_numpy().copy()
        rng = np.random.default_rng(99)
        bad = rng.random(len(y)) < 0.1
        y[bad] += 30.0 * np.nanstd(y)
        df = df.with_columns(pl.Series("y0", y))
        self._compare(df, scale=scale, scale_half_life=30.0)

    @pytest.mark.parametrize("model", ["huber", "quantile"])
    @pytest.mark.parametrize(
        ("n_rows", "zeros"),
        [(250, slice(60, None, 7)), (300, slice(100, 200))],
        ids=["every seventh row", "a hundred rows in a row"],
    )
    @pytest.mark.parametrize("scale", ["mad", "rms"])
    def test_rows_of_weight_zero(self, model, n_rows, zeros, scale):
        """A row of weight 0 advances the clock and teaches nothing (hard rule
        9). The core's residual scale keeps three rules for it: sigma2's
        weight ages on every row the model sees, a row of weight 0 or with no
        prediction included (review 2026-09-12, S13 and N6), and a row of
        weight 0 moves no sigma2, a quantile nudge included (review
        2026-10-05, CC5). No other case holds such a row, and robust_ref aged
        the weight only on the rows it moved sigma2 on, which put it 5.1e-4 to
        4.6e-2 from the bank on these four cases (task 177). A hundred rows of
        weight 0 in a row take the weight under ``min_weight``, so the rows at
        the end of them and just after have a target and no prediction."""
        df, _ = synthetic(seed=91, n_groups=1, n_rows=n_rows, k=2, null_frac=0.0)
        w = df["w"].to_numpy().copy()
        w[zeros] = 0.0
        df = df.with_columns(pl.Series("w", w))
        if model == "huber":
            out = self._compare(df, scale=scale)
        else:
            out = self._compare(df, model="quantile", ref_kw={"quantile": 0.9}, quantile=0.9)
        pred = out["m"].struct.field("pred_y0").to_numpy().astype(float)
        if zeros.step is None:
            withheld = np.isnan(pred[zeros.start :])
            assert withheld.any() and not withheld.all(), "the weight must fall under min_weight"

    @pytest.mark.parametrize("scale", ["mad", "rms"])
    @pytest.mark.parametrize("delta", [0.5, 1.5, 10.0])
    def test_huber_delta_values(self, delta, scale):
        df, _ = synthetic(seed=92, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, ref_kw={"huber_delta": delta}, scale=scale, huber_delta=delta)

    @pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
    def test_quantile_levels(self, tau):
        df, _ = synthetic(seed=93, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, model="quantile", ref_kw={"quantile": tau}, quantile=tau)

    def test_with_nulls(self):
        # The reweighting interacts with the null policy (a null target must not
        # advance that target's accumulator), so it gets an oracle comparison.
        df, _ = synthetic(seed=94, n_groups=1, n_rows=300, k=2, null_frac=0.05)
        self._compare(df)
        self._compare(df, scale="rms")

    def test_multi_target_has_independent_accumulators(self):
        df, _ = synthetic(seed=95, n_groups=1, n_rows=250, k=2, n_targets=2, null_frac=0.0)
        self._compare(df, targets=("y0", "y1"))

    def test_standardized_solve(self):
        df, _ = synthetic(seed=96, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, ref_kw={"standardize": True}, standardize=True)

    def test_huber_down_weights_no_row_before_a_scale_exists(self):
        """Task 177: until a target has a residual scale -- no residual yet,
        or every one so far exactly zero -- Huber down-weights no row. The
        targets are in millions, so their first residuals are far outside
        ``huber_delta`` times 1, the cut the old rule drew there, and the
        first opens on twenty rows at exactly zero, so its residuals are
        exactly zero until it moves. The other cases' first residuals all
        fall inside that cut, where the two rules agree."""
        df, _ = synthetic(seed=97, n_groups=1, n_rows=250, k=2, n_targets=2, null_frac=0.0)
        y0 = pl.when(pl.int_range(pl.len()) < 20).then(0.0).otherwise(pl.col("y0"))
        df = df.with_columns(y0.alias("y0")).with_columns(pl.col("y0", "y1") * 1e6)
        self._compare(df, targets=("y0", "y1"))
        self._compare(df, targets=("y0", "y1"), scale="rms")

    def test_quantile_draws_no_band_before_a_scale_exists(self):
        """Task 177: until a target has a residual scale, a quantile row is
        least squares, as in the warm-up, since a band is a width in units of
        the scale. The first target opens on twenty rows at exactly zero,
        past the warm-up's nine, so its residuals are exactly zero, and the
        band the literal scale of 1 drew took them as band rows aimed above
        the target. The targets are in millions."""
        df, _ = synthetic(seed=98, n_groups=1, n_rows=250, k=2, n_targets=2, null_frac=0.0)
        y0 = pl.when(pl.int_range(pl.len()) < 20).then(0.0).otherwise(pl.col("y0"))
        df = df.with_columns(y0.alias("y0")).with_columns(pl.col("y0", "y1") * 1e6)
        self._compare(
            df, targets=("y0", "y1"), model="quantile", ref_kw={"quantile": 0.9}, quantile=0.9
        )

    # ---- quantile's own keywords at values other than their defaults (task
    # 278 (b07)); the oracle-free halves are test_robust.py's
    # TestTheQuantilesOwnKeywords and TestTheQuantileSolveEvery ----

    @pytest.mark.parametrize("ridge", [0.5, 5.0])
    @pytest.mark.parametrize("tau", [0.5, 0.9])
    def test_quantile_ridge_values(self, tau, ridge):
        """``robust_ref`` at two ridges, a penalty on the band system's
        slopes in the features' squared units; the run is shown to be one
        the ridge moves. At 5 the ridge sets most of a coefficient, which
        the readiness notice reports; the case means it to."""
        df, _ = synthetic(seed=93, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", po.ReadinessWarning)
            out = self._compare(
                df,
                model="quantile",
                ref_kw={"quantile": tau, "ridge": ridge},
                quantile=tau,
                ridge=ridge,
            )
        default = self._compare(df, model="quantile", ref_kw={"quantile": tau}, quantile=tau)
        moved = np.nanmax(
            np.abs(
                out["m"].struct.field("pred_y0").to_numpy().astype(float)
                - default["m"].struct.field("pred_y0").to_numpy().astype(float)
            )
        )
        assert moved > 1e-3, moved

    @pytest.mark.parametrize("ridge", [0.0, 0.5])
    def test_quantile_standardized_solve(self, ridge):
        df, _ = synthetic(seed=96, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(
            df,
            model="quantile",
            ref_kw={"quantile": 0.9, "ridge": ridge, "standardize": True},
            quantile=0.9,
            ridge=ridge,
            standardize=True,
        )

    @pytest.mark.parametrize("eps", [0.1, 0.5])
    def test_quantile_eps_values(self, eps):
        df, _ = synthetic(seed=93, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(
            df,
            model="quantile",
            ref_kw={"quantile": 0.9, "quantile_eps": eps},
            quantile=0.9,
            quantile_eps=eps,
        )

    @staticmethod
    def _quantile_rows(n, seed, scales=(1.0, 1.0)):
        """``TestQuantileIsQuantReg``'s stream: two unit normal features,
        each scaled, and ``y = 1 + 2 x0 - x1`` plus exponential noise, so the
        0.9 quantile's fit is far from the mean's."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, 2))
        y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + rng.exponential(1.0, n)
        return x * np.asarray(scales), y

    @staticmethod
    def _quantile_fit(x, y, tau, **kw):
        """The last fit and the band's scale ``s`` before the last row, with
        no decay and a solve on every row. Without decay ``s`` is the median
        of every scored ``|e|`` over ``Phi^-1(3/4)`` on the sketch's buckets
        (``robust.rs``'s module doc; task 278's D-Q moved it from the RMS),
        which ``reference.EwMedian`` reads from the emitted residuals: the
        bandwidth the stream had settled on. ``min_weight = 0`` reports
        every residual the scale learned from (it withholds the output
        alone)."""
        spec = po.spec.quantile(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            quantile=tau,
            half_life=float("inf"),
            max_rows_between_solves=1,
            **{"min_weight": 0.0, **kw},
        )
        bank = po.ModelBank([spec])
        with warnings.catch_warnings():
            # A ridge of 1e9 makes the ridge, not the data, set the slopes.
            warnings.simplefilter("ignore", po.ReadinessWarning)
            out = bank.fit_predict(pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y}))
        resid = out["m"].struct.field("resid_y").drop_nulls().to_numpy()
        mad = EwMedian()
        for r in resid:
            mad.add(abs(float(r)), 1.0)
        return bank.coef("m")["coef"].to_numpy(), float(mad.scale())

    @staticmethod
    def _smoothed(z, y, tau, h):
        """The batch estimator the stream approximates, written from its
        definition: the minimum of the check loss smoothed over a band of
        half-width ``h`` (``psi`` is ``tau - 1/2 + r / 2h`` inside it and
        ``tau - 1{r < 0}`` outside), by Newton to convergence from least
        squares. The same estimator as ``TestQuantileIsQuantReg``'s."""
        b = np.linalg.lstsq(z, y, rcond=None)[0]
        for _ in range(500):
            r = y - z @ b
            near = np.abs(r) < h
            psi = np.where(near, tau - 0.5 + r / (2.0 * h), tau - (r < 0))
            step = np.linalg.solve((z[near].T @ z[near]) / (2.0 * h), z.T @ psi)
            b = b + step
            if np.max(np.abs(step)) < 1e-13:
                return b
        raise AssertionError("the smoothed fit did not converge")

    def _huge_ridge(self, n, seed, tau):
        """At ``ridge = 1e9`` the slopes are gone and the intercept, which
        no ridge touches, is the tau-quantile of ``y`` alone. Returns the
        slopes, the intercept's distance from ``QuantReg`` on a constant in
        units of its standard error, and from the batch-smoothed constant
        at the stream's bandwidth."""
        import statsmodels.api as sm

        x, y = self._quantile_rows(n, seed)
        coef, s = self._quantile_fit(x, y, tau, ridge=1e9)
        fit = sm.QuantReg(y, np.ones((n, 1))).fit(q=tau)
        smoothed = self._smoothed(np.ones((n, 1)), y, tau, 0.2 * s)[0]
        se = fit.bse[0]
        return (
            coef[1:],
            abs(coef[0] - fit.params[0]) / se,
            abs(coef[0] - smoothed),
            abs(smoothed - fit.params[0]) / se,
            abs(coef[0] - smoothed) / se,
        )

    # Bounds for the huge ridge (task 278 (b07), measured 2026-10-25 over
    # seeds 0-49; re-measured 2026-10-26 with the band in the MAD, D-Q: the
    # slopes 2.2e-9, the smoothed constant 0.0211 / 0.0063 / 0.0015 at tau
    # 0.5 and 0.224 / 0.057 / 0.018 at 0.9, QuantReg 0.60 and 2.74 standard
    # errors at 20,000 rows, every bound below still holding). The slopes: a
    # ridge of 1e9 over a cross-moment of about 2
    # leaves 2e-9 (measured 2.08e-9 at most). Against the batch-smoothed
    # constant, the worst seed at 5,000 / 20,000 / 80,000 rows: 0.0213 /
    # 0.0060 / 0.0015 at tau 0.5, 0.204 / 0.066 / 0.018 at tau 0.9 -- a rate
    # faster than 1/sqrt(n) (the worst fell 14 and 11 times over 16 times the
    # rows); the bounds are 1.5 times those. Against QuantReg the worst was
    # 0.59 and 1.97 standard errors at 20,000 rows; the gap there is the
    # band's smoothing, which does not shrink with n while the standard
    # error does (3.86 at 80,000 rows at tau 0.9), so that check is held at
    # 20,000 rows only.
    #
    # Re-measured 2026-10-26 with the first rows refit to the converged
    # smoothed fit (D-Q2), seeds 0-49: against the batch-smoothed constant
    # 0.0039 / 0.0011 / 0.00037 at tau 0.5 and 0.011 / 0.0059 / 0.0034 at
    # 0.9, lower at every length, so at 0.9 the worst no longer falls four
    # times over 16 times the rows (3.3 times): in QuantReg's standard
    # errors it sits at 0.18 / 0.19 / 0.21, a fifth of one at every length.
    # Against QuantReg the worst is 0.66 / 0.45 / 0.62 and 2.01 / 3.94 /
    # 6.26 standard errors, the batch-smoothed fit's own 0.66 / 0.44 / 0.61
    # and 1.98 / 3.89 / 6.17: the stream is the smoothed estimator, and its
    # band's bias, not the stream, is what grows in standard errors. So the
    # grid holds the stream within 0.3 standard errors of the batch-smoothed
    # fit and of its distance from QuantReg at every length, where it held
    # 3 standard errors from QuantReg at 20,000 rows and the fourfold fall.
    HUGE_RIDGE_SMOOTHED = {0.5: (0.032, 0.009, 0.0023), 0.9: (0.3, 0.1, 0.027)}

    @pytest.mark.parametrize("tau", [0.5, 0.9])
    def test_a_huge_ridge_leaves_quantreg_on_a_constant(self, tau):
        slopes, in_se, smoothed, _, _ = self._huge_ridge(20_000, 3, tau)
        assert np.max(np.abs(slopes)) <= 1e-8, slopes
        assert in_se <= 3.0, in_se
        assert smoothed <= self.HUGE_RIDGE_SMOOTHED[tau][1], smoothed

    @pytest.mark.oracle_grid(reason="50 seeds at three lengths and two levels, against QuantReg")
    @pytest.mark.parametrize("tau", [0.5, 0.9])
    def test_a_huge_ridge_over_fifty_seeds(self, tau):
        for n, bound in zip((5_000, 20_000, 80_000), self.HUGE_RIDGE_SMOOTHED[tau], strict=True):
            runs = [self._huge_ridge(n, seed, tau) for seed in range(50)]
            slopes = max(float(np.max(np.abs(r[0]))) for r in runs)
            in_se = np.array([r[1] for r in runs])
            smoothed = np.array([r[2] for r in runs])
            bias_se = np.array([r[3] for r in runs])
            stream_se = np.array([r[4] for r in runs])
            print(
                f"tau {tau}, n {n}: slopes {slopes:.3g}; QuantReg worst {in_se.max():.3f} se "
                f"(seed {in_se.argmax()}), p95 {np.quantile(in_se, 0.95):.3f}; smoothed worst "
                f"{smoothed.max():.4f} (seed {smoothed.argmax()}), p95 "
                f"{np.quantile(smoothed, 0.95):.4f}"
            )
            t = f"tau_{tau:g}_n_{n}".replace(".", "_")
            ledger.record(
                "quantile_huge_ridge", f"{t}_slopes", [np.max(np.abs(r[0])) for r in runs]
            )
            ledger.record("quantile_huge_ridge", f"{t}_quantreg_se", in_se)
            ledger.record("quantile_huge_ridge", f"{t}_smoothed", smoothed)
            ledger.record("quantile_huge_ridge", f"{t}_smoothed_in_se", stream_se)
            assert slopes <= 1e-8, (n, slopes)
            assert smoothed.max() <= bound, (n, smoothed.max(), smoothed.argmax())
            assert stream_se.max() <= 0.3, (n, stream_se.max(), stream_se.argmax())
            assert (in_se <= bias_se + 0.3).all(), (n, np.flatnonzero(in_se > bias_se + 0.3))

    def _mixed_scales(self, seed, tau):
        """Features at 1e-3 and 1e3 under the default ridge of 1e-6:
        ``QuantReg`` on the raw columns, and both solves' distance from it
        per coefficient, in its standard errors."""
        import statsmodels.api as sm

        scales = np.array([1e-3, 1e3])
        x, y = self._quantile_rows(20_000, seed, scales=scales)
        # QuantReg's own iteration stops at its cap on these raw columns
        # (an IterationLimitWarning on 3 of 100 fits, 2026-10-25). The
        # problem is equivariant in each column's scale, so it is fitted on
        # the unit columns and its coefficients and standard errors carried
        # back to the raw ones, exactly.
        fit = sm.QuantReg(y, sm.add_constant(x / scales)).fit(q=tau)
        back = np.concatenate(([1.0], 1.0 / scales))
        want, se = np.asarray(fit.params) * back, np.asarray(fit.bse) * back
        corr, _ = self._quantile_fit(x, y, tau, standardize=True)
        plain, _ = self._quantile_fit(x, y, tau)
        return np.abs(corr - want) / se, np.abs(plain - want) / se

    # Bounds on mixed scales (task 278 (b07), measured 2026-10-25 over seeds
    # 0-49 at 20,000 rows; re-measured 2026-10-26 with the band in the MAD,
    # D-Q: 1.18 and 1.77 standardized, the plain x0 at least 136 and 43
    # off): standardized, the worst coefficient sat 1.85 of
    # QuantReg's standard errors away at tau 0.5 and 2.57 at 0.9 (the
    # intercept, seeds 1 and 47); bound 4. Plain, the default ridge is the
    # whole variance of the feature at 1e-3 and halves its slope: at least
    # 137 standard errors off at tau 0.5 and 43 at 0.9 on every seed; held
    # at more than 10.
    @pytest.mark.parametrize("tau", [0.5, 0.9])
    def test_standardized_on_mixed_scales_is_quantreg(self, tau):
        corr, plain = self._mixed_scales(3, tau)
        assert corr.max() <= 4.0, corr
        assert plain[1] > 10.0, plain

    @pytest.mark.oracle_grid(reason="50 seeds at two levels, each against QuantReg")
    @pytest.mark.parametrize("tau", [0.5, 0.9])
    def test_standardized_on_mixed_scales_over_fifty_seeds(self, tau):
        runs = [self._mixed_scales(seed, tau) for seed in range(50)]
        corr = np.array([r[0] for r in runs])
        plain = np.array([r[1][1] for r in runs])
        t = f"tau_{tau:g}".replace(".", "_")
        ledger.record("quantile_mixed_scales", f"{t}_standardized_se", corr.max(axis=1))
        print(
            f"tau {tau}: standardized worst {corr.max(axis=0)} se (seed "
            f"{corr.max(axis=1).argmax()}), p95 {np.quantile(corr, 0.95, axis=0)}; plain's x0 "
            f"at least {plain.min():.1f} se off"
        )
        assert corr.max() <= 4.0, (corr.max(), corr.max(axis=1).argmax())
        assert plain.min() > 10.0, (plain.min(), plain.argmin())

    def _eps_gap(self, n, seed, eps):
        """At ``quantile_eps = eps`` and tau 0.9: the stream's distance from
        the batch-smoothed fit at its own bandwidth ``eps * s``, and from the
        one at the other level's (0.5 for 0.1, 0.1 for 0.5)."""
        x, y = self._quantile_rows(n, seed)
        coef, s = self._quantile_fit(x, y, 0.9, quantile_eps=eps)
        z = np.column_stack([np.ones(n), x])
        own = self._smoothed(z, y, 0.9, eps * s)
        other = self._smoothed(z, y, 0.9, (0.6 - eps) * s)
        return float(np.max(np.abs(coef - own))), float(np.max(np.abs(coef - other)))

    # Bounds for quantile_eps (task 278 (b07); re-measured 2026-10-26 over
    # seeds 0-49 at tau 0.9 with the band in the MAD, D-Q; k / n is small
    # enough at these lengths that the floor never binds): the worst seed's
    # distance from the batch-smoothed fit at 5,000 / 20,000 / 80,000 rows
    # was 0.184 / 0.090 / 0.019 at eps 0.1 (seeds 3, 47, 19) and 0.112 /
    # 0.063 / 0.034 at eps 0.5 (seeds 36, 47, 46) -- the early rows' nudges
    # and band rows, which a stream with no decay never forgets. At eps 0.1
    # that shrinks faster than 1/sqrt(n) (9.5 times over 16 times the rows),
    # at eps 0.5 more slowly (3.3 times): the MAD of these residuals, a
    # skewed noise read at its 0.9 quantile, is about 1.5 times their RMS,
    # so the same eps is a band half as wide again as it was, holding more
    # of the early rows at their own bandwidths. In the RMS the worst was
    # 0.236 / 0.122 / 0.028 and 0.103 / 0.040 / 0.016. Bounds 1.5 times the
    # measured, the rate held at a quarter and at 2.5 times less. From 20,000
    # rows every seed sat nearer its own level's fit than the other's (at
    # 5,000, 0 and 2 seeds of 50 did not).
    EPS_BOUNDS = {0.1: (0.28, 0.14, 0.03), 0.5: (0.17, 0.095, 0.051)}
    EPS_RATE = {0.1: 4.0, 0.5: 2.5}

    @pytest.mark.parametrize("eps", [0.1, 0.5])
    def test_quantile_eps_is_the_batch_smoothed_fit_at_its_bandwidth(self, eps):
        own, other = self._eps_gap(20_000, 3, eps)
        assert own <= self.EPS_BOUNDS[eps][1], own
        assert own < other, (own, other)

    @pytest.mark.oracle_grid(reason="50 seeds at three lengths and two levels, each a batch fit")
    @pytest.mark.parametrize("eps", [0.1, 0.5])
    def test_quantile_eps_over_fifty_seeds(self, eps):
        worst = []
        for n, bound in zip((5_000, 20_000, 80_000), self.EPS_BOUNDS[eps], strict=True):
            runs = np.array([self._eps_gap(n, seed, eps) for seed in range(50)])
            own, other = runs[:, 0], runs[:, 1]
            t = f"eps_{eps:g}_n_{n}".replace(".", "_")
            ledger.record("quantile_eps", f"{t}_from_the_smoothed_fit", own)
            print(
                f"eps {eps}, n {n}: worst {own.max():.4f} (seed {own.argmax()}), p95 "
                f"{np.quantile(own, 0.95):.4f}; nearer the other level's fit on "
                f"{int((other <= own).sum())} seeds"
            )
            assert own.max() <= bound, (n, own.max(), own.argmax())
            if n >= 20_000:
                assert (own < other).all(), (n, np.flatnonzero(other <= own))
            worst.append(own.max())
        assert worst[-1] <= worst[0] / self.EPS_RATE[eps], worst


#: The oracle ledger's inputs (tests/ledger.py) for the three quantile grids
#: above: the fits on the first three seeds at 5,000 rows, each grid's
#: settings, every coefficient and the band's scale. The grids' 50 seeds at
#: up to 80,000 rows, a solve a row, take a minute.
def _quantile_guard(scales=(1.0, 1.0), taus=(0.5, 0.9), **kw) -> list:
    rows = [TestRobustOracles._quantile_rows(5_000, seed, scales) for seed in range(3)]
    return [TestRobustOracles._quantile_fit(x, y, tau, **kw) for x, y in rows for tau in taus]


ledger.guard("quantile_huge_ridge", lambda: _quantile_guard(ridge=1e9), oracles=["statsmodels"])
ledger.guard(
    "quantile_mixed_scales",
    lambda: [
        _quantile_guard((1e-3, 1e3), standardize=standardize) for standardize in (True, False)
    ],
    oracles=["statsmodels"],
)
ledger.guard(
    "quantile_eps", lambda: [_quantile_guard(taus=(0.9,), quantile_eps=e) for e in (0.1, 0.5)]
)


class TestFtrlOracle:
    """T-A4: FTRL vs `tests/reference.py::ftrl_ref`."""

    def _binary(self, seed=5, n=400):
        rng = np.random.default_rng(seed)
        x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
        p = 1.0 / (1.0 + np.exp(-(1.5 * x0 - 0.5 * x1)))
        return pl.DataFrame(
            {
                "x0": x0,
                "x1": x1,
                "y0": (rng.random(n) < p).astype(float),
                "w": rng.uniform(0.5, 1.5, n),
                "t": np.cumsum(rng.exponential(5.0, n)),
            }
        )

    def _compare(self, df, **kw):
        n = df.height
        x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
        dc = np.zeros(n)
        dc[1:] = np.clip(np.diff(df["t"].to_numpy()), 0.0, 30.0)
        ref = ftrl_ref(
            x,
            df["y0"].to_numpy().reshape(-1, 1),
            dc,
            df["w"].to_numpy(),
            min_weight=10.0,
            gap_cap=30.0,
            **kw,
        )
        spec = po.spec.ftrl(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=30.0,
            weight="w",
            min_weight=10.0,
            half_life=kw.get("half_life", float("inf")),
            alpha=kw.get("alpha"),
            beta=kw.get("beta"),
            l1=kw.get("l1"),
            l2=kw.get("l2"),
            fit_intercept=kw.get("fit_intercept", True),
            loss=kw.get("loss", "logistic"),
        )
        out = po.ModelBank([spec]).fit_predict(df)
        _close(
            out["m"].struct.field("pred_y0").to_numpy().astype(float),
            ref["pred"][:, 0],
            tol=1e-12,
            what="pred",
        )
        _close(
            out["m"].struct.field("resid_y0").to_numpy().astype(float),
            ref["resid"][:, 0],
            tol=1e-12,
            what="resid",
        )

    def test_no_decay(self):
        self._compare(self._binary())

    def test_with_clock_decay(self):
        self._compare(self._binary(seed=6), half_life=200.0)

    @pytest.mark.parametrize("l1", [0.0, 0.5, 5.0])
    def test_l1_values(self, l1):
        self._compare(self._binary(seed=7), l1=l1)

    def test_alpha_beta_l2(self):
        self._compare(self._binary(seed=8), alpha=0.5, beta=0.1, l2=3.0)

    def test_no_intercept(self):
        self._compare(self._binary(seed=9), fit_intercept=False)

    def test_squared_loss_under_a_halflife(self):
        """The squared loss (E18) had no oracle here (review 2026-09-12,
        D10), and under a half-life the proximal term is a decayed sum of its
        own (C24), which the reference carries."""
        df = self._binary(seed=11).with_columns(y0=1.5 * pl.col("x0") - 0.5 * pl.col("x1"))
        self._compare(df, half_life=200.0, loss="squared", alpha=0.5, l2=0.01)

    def test_null_targets(self):
        df = self._binary(seed=10)
        y = df["y0"].to_list()
        y[50] = None
        y[51] = None
        df = df.with_columns(y0=pl.Series(y, dtype=pl.Float64))
        n = df.height
        x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
        dc = np.zeros(n)
        dc[1:] = np.clip(np.diff(df["t"].to_numpy()), 0.0, 30.0)
        ref = ftrl_ref(
            x,
            df["y0"].to_numpy().astype(float).reshape(-1, 1),
            dc,
            df["w"].to_numpy(),
            min_weight=10.0,
            gap_cap=30.0,
        )
        spec = po.spec.ftrl(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            clock="t",
            gap_cap=30.0,
            weight="w",
            min_weight=10.0,
            half_life=float("inf"),
        )
        out = po.ModelBank([spec]).fit_predict(df)
        _close(
            out["m"].struct.field("pred_y0").to_numpy().astype(float),
            ref["pred"][:, 0],
            tol=1e-12,
            what="pred with nulls",
        )
