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

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import ftrl_ref, kalman_ref, lasso_ref, robust_ref

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
        ref = kalman_ref(x, y, dc, w, max_dclock=MAXD, **kw)
        spec = po.spec.kalman(
            "m",
            targets=list(targets),
            features=[f"x{j}" for j in range(k)],
            clock="t",
            max_dclock=MAXD,
            weight="w",
            halflife=kw.get("halflife", 500.0),
            coef_halflife=kw.get("coef_halflife", 100.0),
            q=kw.get("q"),
            obs_var=kw.get("obs_var"),
            p0=kw.get("p0"),
            share_p=kw.get("share_p", False),
            min_periods=kw.get("min_periods", 10.0),
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for j, t in enumerate(targets):
            for field, key in (("pred_", "pred"), ("resid_", "resid")):
                got = out["m"].struct.field(f"{field}{t}").to_numpy().astype(float)
                _close(got, ref[key][:, j], what=f"{field}{t}")
        _close(
            out["m"].struct.field("n_eff").to_numpy().astype(float),
            ref["n_eff"],
            what="n_eff",
        )

    def test_scalar_coef_halflife(self):
        df, _ = synthetic(seed=71, n_groups=1, n_rows=300, k=3, null_frac=0.0)
        self._compare(df)

    def test_per_factor_halflife_with_pinning(self):
        df, _ = synthetic(seed=72, n_groups=1, n_rows=300, k=3, null_frac=0.0)
        # intercept pinned, x0 slow, x1 fast, x2 pinned
        self._compare(df, coef_halflife=[float("inf"), 500.0, 30.0, float("inf")])

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
        ref = kalman_ref(x, y, dc, w, add_intercept=False, min_periods=10.0, max_dclock=MAXD)
        spec = po.spec.kalman(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            add_intercept=False,
            clock="t",
            max_dclock=MAXD,
            weight="w",
            halflife=500.0,
            coef_halflife=100.0,
            min_periods=10.0,
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
    def _ew_stats(x, y, dclock, w, halflife):
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
            lam = 0.5 ** (dclock[i] / halflife)
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

    def _kkt_residuals(self, x, y, dclock, w, halflife, coefs, lam_value, l1_ratio):
        """Return (b_std, g, l1) at the final row: g_i is the stationarity
        quantity that must equal l1*sign(b_i) where b_i != 0, and satisfy
        |g_i| <= l1 where b_i == 0."""
        mean, raw, ry = self._ew_stats(x, y, dclock, w, halflife)[-1]
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
            lasso_path=path,
            l1_ratio=l1_ratio,
            halflife=1e9,
            min_periods=10.0,
            max_rows_between_solves=1,
            max_cd_iters=2000,
            cd_tol=1e-14,
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
            lasso_path=path,
            halflife=1e9,
            min_periods=10.0,
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
            lasso_path=[0.0],
            halflife=1e9,
            min_periods=10.0,
            max_rows_between_solves=1,
            max_cd_iters=5000,
            cd_tol=1e-15,
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
        lasso_path=path,
        halflife=1e9,
        min_periods=10.0,
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
    ``n_eff`` and every emitted ``coef``, against
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
    # 1.2e-12, n_eff exact. Each tolerance is 100x the largest it covers,
    # rounded up to a power of ten.
    PRED_TOL = 1e-11
    COEF_TOL = 1e-9
    # At the library's own `cd_tol` (1e-10) and `max_cd_iters` (100) each
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
        ``max_dclock``. A twentieth of the rows weigh 0, the first row among
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
        ``cd_tol`` and ``max_cd_iters`` at the library's defaults."""
        descent = {} if default_descent else {"cd_tol": 1e-14, "max_cd_iters": 100_000}
        spec = po.spec.lasso(
            "m",
            targets=["y"],
            features=self.FEATURES,
            lasso_path=self.PATH,
            clock="t",
            max_dclock=self.MAX_DCLOCK,
            weight="w",
            coef_every=1,
            **descent,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        n, npath, kt = df.height, len(self.PATH), len(self.FEATURES) + 1
        x = df.select(self.FEATURES).to_numpy()
        y = df["y"].to_numpy()
        dc = np.zeros(n)
        dc[1:] = np.diff(df["t"].to_numpy())
        ref = lasso_ref(x, y, dc, df["w"].to_numpy(), self.PATH, max_dclock=self.MAX_DCLOCK, **kw)
        assert ref["solved"].sum() >= 5, "the stream should span several solves"

        # One pred and one resid field per path point, in path order.
        index = po.spec.output_index(spec)
        assert index.filter(pl.col("kind") == "pred")["lambda"].to_list() == self.PATH
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
        n_eff = out.struct.field("n_eff").to_numpy().astype(float)
        _close(n_eff, ref["n_eff"], tol=self.PRED_TOL, what="n_eff")

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
        # `coef` is each row's last solve, and `min_periods` does not gate it:
        # it is null only before the first solve and on a skipped row.
        before_first = np.arange(n) < np.argmax(ref["solved"])
        expect_null = before_first | np.isnan(ref["n_eff"])
        wrong = np.flatnonzero(np.array([r is None for r in rows]) != expect_null)
        assert wrong.size == 0, f"coef is null on the wrong rows, first {wrong[0]}"
        empty = [np.nan] * (npath * kt)
        coef = np.array([empty if r is None else r for r in rows]).reshape(n, npath, kt)
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
        a solve is then due (``>=``). Each gap past ``max_dclock`` forces a
        solve with its capped step. The halflife is finite, so the decay and
        the cap both reach the fit."""
        self._compare(
            self._stream(31), l1_ratio=1.0, halflife=40.0, min_periods=20.0, solve_every=2.5
        )

    def test_an_elastic_net_solves_on_a_row_cap_beside_the_clock(self):
        """``max_rows_between_solves`` beside ``solve_every``. A zero-weight
        row counts as a row; a skipped row does not. The capped gap equals
        ``solve_every``, so each gap forces a solve."""
        self._compare(
            self._stream(32),
            l1_ratio=0.5,
            halflife=40.0,
            min_periods=20.0,
            solve_every=6.0,
            max_rows_between_solves=5,
        )

    def test_the_default_cadence_is_a_fiftieth_of_the_halflife(self):
        """No ``solve_every``: a solve every ``halflife / 50`` = 2 clock
        units, which the dyadic steps meet exactly."""
        self._compare(self._stream(33), l1_ratio=1.0, halflife=100.0, min_periods=20.0)

    def test_the_first_solve_is_forced_when_min_periods_is_reached(self):
        """The clock cadence is out of reach and the row cap is 40. So the
        first solve is the forced one at ``min_periods``, and every row up to
        the next solve is scored with it."""
        self._compare(
            self._stream(34),
            l1_ratio=1.0,
            halflife=40.0,
            min_periods=25.0,
            solve_every=1e9,
            max_rows_between_solves=40,
        )

    def test_an_infinite_halflife_solves_after_every_row(self):
        """The default cadence without decay is a solve after every row."""
        self._compare(self._stream(35), l1_ratio=0.5, halflife=float("inf"), min_periods=20.0)

    def test_the_default_descent_reaches_the_same_predictions(self):
        """``cd_tol`` and ``max_cd_iters`` left at the library's defaults, the
        settings a user runs. Every scored row still agrees to within the
        descent's own tolerance."""
        self._compare(
            self._stream(33), default_descent=True, l1_ratio=1.0, halflife=100.0, min_periods=20.0
        )


def test_pl_is_importable():
    assert pl.__version__


class TestRobustOracles:
    """T-A3: Huber and quantile vs `tests/reference.py::robust_ref`."""

    def _compare(self, df, k=2, targets=("y0",), model="huber", ref_kw=None, **spec_kw):
        x, dc, w = _arrays(df, k)
        y = np.column_stack([df[t].to_numpy() for t in targets])
        ref = robust_ref(
            x,
            y,
            dc,
            w,
            halflife=300.0,
            loss="huber" if model == "huber" else "quantile",
            min_periods=5.0,
            max_dclock=MAXD,
            **(ref_kw or {}),
        )
        spec = getattr(po.spec, model)(
            "m",
            targets=list(targets),
            features=[f"x{j}" for j in range(k)],
            clock="t",
            max_dclock=MAXD,
            weight="w",
            halflife=300.0,
            min_periods=5.0,
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
            out["m"].struct.field("n_eff").to_numpy().astype(float),
            ref["n_eff"],
            what="n_eff",
        )

    def test_huber(self):
        df, _ = synthetic(seed=91, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df)

    @pytest.mark.parametrize("delta", [0.5, 1.5, 10.0])
    def test_huber_delta_values(self, delta):
        df, _ = synthetic(seed=92, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, ref_kw={"huber_delta": delta}, huber_delta=delta)

    @pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
    def test_quantile_levels(self, tau):
        df, _ = synthetic(seed=93, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, model="quantile", ref_kw={"quantile": tau}, quantile=tau)

    def test_with_nulls(self):
        # The reweighting interacts with the null policy (a null target must not
        # advance that target's accumulator), so it gets an oracle comparison.
        df, _ = synthetic(seed=94, n_groups=1, n_rows=300, k=2, null_frac=0.05)
        self._compare(df)

    def test_multi_target_has_independent_accumulators(self):
        df, _ = synthetic(seed=95, n_groups=1, n_rows=250, k=2, n_targets=2, null_frac=0.0)
        self._compare(df, targets=("y0", "y1"))

    def test_standardized_solve(self):
        df, _ = synthetic(seed=96, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        self._compare(df, ref_kw={"standardize": True}, standardize=True)


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
            min_periods=10.0,
            max_dclock=30.0,
            **kw,
        )
        spec = po.spec.ftrl(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            clock="t",
            max_dclock=30.0,
            weight="w",
            min_periods=10.0,
            halflife=kw.get("halflife", float("inf")),
            alpha=kw.get("alpha"),
            beta=kw.get("beta"),
            l1=kw.get("l1"),
            l2=kw.get("l2"),
            add_intercept=kw.get("add_intercept", True),
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
        self._compare(self._binary(seed=6), halflife=200.0)

    @pytest.mark.parametrize("l1", [0.0, 0.5, 5.0])
    def test_l1_values(self, l1):
        self._compare(self._binary(seed=7), l1=l1)

    def test_alpha_beta_l2(self):
        self._compare(self._binary(seed=8), alpha=0.5, beta=0.1, l2=3.0)

    def test_no_intercept(self):
        self._compare(self._binary(seed=9), add_intercept=False)

    def test_squared_loss_under_a_halflife(self):
        """The squared loss (E18) had no oracle here (review 2026-09-12,
        D10), and under a halflife the proximal term is a decayed sum of its
        own (C24), which the reference carries."""
        df = self._binary(seed=11).with_columns(y0=1.5 * pl.col("x0") - 0.5 * pl.col("x1"))
        self._compare(df, halflife=200.0, loss="squared", alpha=0.5, l2=0.01)

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
            min_periods=10.0,
            max_dclock=30.0,
        )
        spec = po.spec.ftrl(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            clock="t",
            max_dclock=30.0,
            weight="w",
            min_periods=10.0,
            halflife=float("inf"),
        )
        out = po.ModelBank([spec]).fit_predict(df)
        _close(
            out["m"].struct.field("pred_y0").to_numpy().astype(float),
            ref["pred"][:, 0],
            tol=1e-12,
            what="pred with nulls",
        )
