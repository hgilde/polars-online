"""Oracle agreement for the models PLAN section 9 class 1 promised but that
were only property-tested: Kalman against a numpy reference (T-A1), and the
lasso against its own optimality conditions (T-A2).

The lasso check is deliberately *not* a second copy of coordinate descent: it
verifies the KKT conditions of the penalized objective, which any correct
solver must satisfy, so it cannot agree with a bug the way a ported
implementation could.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import ftrl_ref, kalman_ref, robust_ref

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
        ref = kalman_ref(x, y, dc, w, **kw)
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
        ref = kalman_ref(x, y, dc, w, add_intercept=False, min_periods=10.0)
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
