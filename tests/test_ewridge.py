"""`ewridge` against the numpy oracle (`reference.ewridge_ref`), and the
features only it has: warm priors (`coef0`, E15) and `session_shrink` (E6).

The oracle covers multi-target, `standardize`, the row-count clock with `lam`,
nulls, weights, groups and sessions; `test_semantics_all_models` and the other
sweeps hold this model to the invariants every model shares.
"""

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import compute_dclock, ewridge_ref

HL = 300.0
MAXD = 50.0
GAP = 25.0


def _spec(k=3, targets=("y0",), **kw):
    defaults = dict(
        targets=list(targets),
        features=[f"x{j}" for j in range(k)],
        clock="t",
        halflife=HL,
        max_dclock=MAXD,
        session="session",
        session_gap=GAP,
        weight="w",
        group="group",
        ridge=1e-6,
        max_rows_between_solves=1,
        min_periods=5.0,
    )
    defaults.update(kw)
    return po.spec.ewridge("m", **defaults)


def _oracle_frame(df: pl.DataFrame, k=3, n_targets=1, **ref_kw):
    """Run ewridge_ref per group and return pred/resid/n_eff arrays row-aligned."""
    ref_defaults = dict(halflife=HL, ridge=1e-6, min_periods=5.0)
    ref_defaults.update(ref_kw)
    pred = np.full((df.height, n_targets), np.nan)
    resid = np.full((df.height, n_targets), np.nan)
    n_eff = np.full(df.height, np.nan)
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
            max_dclock=MAXD,
            session_gap=GAP,
        )
        out = ewridge_ref(x, y, dc, w, rs, **ref_defaults)
        pred[idx] = out["pred"]
        resid[idx] = out["resid"]
        n_eff[idx] = out["n_eff"]
    return pred, resid, n_eff


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
        pred, resid, n_eff = _oracle_frame(df)
        _close(_np(out, "pred_y0"), pred[:, 0])
        _close(_np(out, "resid_y0"), resid[:, 0])
        _close(_np(out, "n_eff"), n_eff)

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
                    halflife=None,
                    lam=lam,
                    max_dclock=None,
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
            x, y, dc, df["w"].to_numpy(), halflife=halflife_equiv, ridge=1e-6, min_periods=5.0
        )
        _close(_np(out, "pred_y0"), ref["pred"][:, 0])


class TestWarmPriors:
    """E15: `coef0` shrinks toward a stated belief instead of toward zero.

    Whether the prior fades depends on `ridge_decay`, and the distinction is
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
            halflife=1e9,
            min_periods=0.0,
            max_rows_between_solves=1,
            coef_every=1,
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
        early, late = self._slopes(coef0=[[0.0, 5.0]])
        assert early > 4.0, "should start near the prior"
        assert late > 3.0, f"a fixed ridge keeps pulling toward the prior forever, got {late}"

    def test_ridge_decay_makes_it_a_fading_warm_start(self):
        early, late = self._slopes(coef0=[[0.0, 5.0]], ridge_decay=True)
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
            coef0=[[0.0, 0.02]],
            halflife=1e9,
            min_periods=0.0,
            max_rows_between_solves=1,
            coef_every=1,
        )
        c = np.array(
            po.ModelBank([spec]).fit_predict(df)["m"].struct.field("coef").to_list()[-1],
            dtype=float,
        )
        assert abs(c[1] - 0.02) < 5e-3, f"expected the prior in original units, got {c[1]}"

    def test_shape_is_validated(self):
        with pytest.raises(ValueError, match="coef0"):
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0", "x1"],
                halflife=100.0,
                coef0=[[0.0, 1.0]],  # too short for 2 features + intercept
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
            halflife=50.0,
            clock="t",
            max_dclock=5.0,
            session="s",
            session_gap=0.0,
            min_periods=0.0,
            max_rows_between_solves=1,
            coef_every=1,
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
        jump, typical = self._jump(session_shrink=0.0, long_halflife=1e5)
        assert jump < 10 * typical, (
            f"boundary move {jump} should look like an ordinary row ({typical})"
        )

    def test_shrink_reverts_toward_the_long_run(self):
        c = self._coefs(session_shrink=0.9, long_halflife=1e5)
        before, after = c[self.N1 + self.N2 - 1][1], c[self.N1 + self.N2][1]
        assert before < -0.5, "the recent regime should have taken over first"
        assert after > 0.5, f"the break should revert toward +1, got {after}"

    def test_shrink_is_monotone(self):
        after = {
            f: self._coefs(session_shrink=f, long_halflife=1e5)[self.N1 + self.N2][1]
            for f in (0.0, 0.3, 0.6, 0.9)
        }
        vals = [after[f] for f in (0.0, 0.3, 0.6, 0.9)]
        assert vals == sorted(vals), f"more shrink should mean more reversion: {after}"

    def test_absent_by_default(self):
        jump, typical = self._jump()
        assert jump < 10 * typical, "no shrink configured should mean no jump"

    def test_shrink_produces_a_visible_jump(self):
        jump, typical = self._jump(session_shrink=0.9, long_halflife=1e5)
        assert jump > 100 * typical, (
            f"a 0.9 shrink should move the fit far more than a row does ({jump} vs {typical})"
        )

    def test_chunk_invariance_and_save_load(self, tmp_path):
        spec = po.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0"],
            halflife=50.0,
            clock="t",
            max_dclock=5.0,
            session="s",
            session_gap=0.0,
            session_shrink=0.5,
            long_halflife=1e5,
            min_periods=0.0,
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
        keep = [c for c in one.columns if not c.startswith("coef")]
        assert one.select(keep).equals(many.select(keep), null_equal=True)

        a = po.ModelBank([spec])
        a.fit_predict(df.slice(0, 400))
        p = tmp_path / "ss.state"
        a.save(p)
        b = po.ModelBank.load(p, specs=[spec])
        rest = df.slice(400, 400)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True)

    def test_config_is_validated(self):
        base = dict(targets=["y0"], features=["x0"], halflife=50.0, session="s", session_gap=0.0)
        with pytest.raises(ValueError, match="needs long_halflife"):
            po.spec.ewridge("m", session_shrink=0.5, **base)
        with pytest.raises(ValueError, match="must be in .0, 1."):
            po.spec.ewridge("m", session_shrink=1.5, long_halflife=1e5, **base)
        with pytest.raises(ValueError, match="needs a .session. column"):
            po.spec.ewridge(
                "m",
                targets=["y0"],
                features=["x0"],
                halflife=50.0,
                session_shrink=0.5,
                long_halflife=1e5,
            )
