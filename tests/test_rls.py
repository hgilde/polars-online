"""`rls` against the numpy oracle (`reference.rls_ref`) and against `ewridge`
solved every row, which is the same estimator (docs/PLAN.md section 4.2)."""

import numpy as np
import polars as pl

import polars_online as po
from data import synthetic
from expr_plugin import requires_expr_plugin
from reference import compute_dclock, rls_ref

HL = 300.0
MAXD = 50.0


def _close(a: np.ndarray, b: np.ndarray, tol=1e-9):
    both_nan = np.isnan(a) & np.isnan(b)
    ok = both_nan | (np.abs(a - b) < tol * (1.0 + np.abs(a)))
    assert ok.all(), f"max diff {np.nanmax(np.abs(a - b))}, mismatches {np.sum(~ok)}"


def _np(df, col):
    return df["m"].struct.field(col).to_numpy().astype(float)


class TestRls:
    """Task 9: RLS (docs/PLAN.md section 4.2) and its agreement with EW-ridge."""

    def test_matches_ewridge_solved_every_row(self):
        # ew_ridge with ridge_decay is algebraically the same estimator as RLS
        # with the same prior; solving every row must reproduce it exactly.
        df, _ = synthetic(seed=21, n_groups=2, n_rows=300, k=3, null_frac=0.0)
        common = dict(
            targets=["y0"],
            features=["x0", "x1", "x2"],
            clock="t",
            halflife=HL,
            max_dclock=MAXD,
            weight="w",
            group="group",
            min_periods=5.0,
        )
        a = po.ModelBank([po.spec.rls("m", ridge=0.7, **common)]).fit_predict(df)
        b = po.ModelBank(
            [
                po.spec.ewridge(
                    "m",
                    ridge=0.7,
                    ridge_decay=True,
                    max_rows_between_solves=1,
                    **common,
                )
            ]
        ).fit_predict(df)
        pa, pb = _np(a, "pred_y0"), _np(b, "pred_y0")
        m = np.isfinite(pa) & np.isfinite(pb)
        assert m.sum() > 500
        assert np.max(np.abs(pa[m] - pb[m])) < 1e-9

    def test_matches_numpy_oracle(self):
        df, _ = synthetic(seed=22, n_groups=1, n_rows=250, k=2, null_frac=0.0)
        spec = po.spec.rls(
            "m",
            targets=["y0"],
            features=["x0", "x1"],
            clock="t",
            halflife=HL,
            max_dclock=MAXD,
            weight="w",
            ridge=1.0,
            min_periods=5.0,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        x = np.column_stack([df["x0"].to_numpy(), df["x1"].to_numpy()])
        y = df["y0"].to_numpy().reshape(-1, 1)
        dc, rs = compute_dclock(df["t"].to_numpy(), None, df.height, max_dclock=MAXD)
        ref = rls_ref(x, y, dc, df["w"].to_numpy(), rs, halflife=HL, ridge=1.0, min_periods=5.0)
        _close(_np(out, "pred_y0"), ref["pred"][:, 0])

    def test_null_target_is_predict_only(self):
        df = pl.DataFrame(
            {
                "x0": [1.0, 2.0, 0.5, 1.5, 2.5],
                "y0": [2.0, 4.0, 1.0, None, 5.0],
            }
        )
        spec = po.spec.rls("m", targets=["y0"], features=["x0"], halflife=100.0, min_periods=1.0)
        out = po.ModelBank([spec]).fit_predict(df)
        row = out.row(3, named=True)["m"]
        assert row["pred_y0"] is not None
        assert row["resid_y0"] is None

    @requires_expr_plugin
    def test_expression_equals_bank(self):
        df, _ = synthetic(seed=23, n_groups=2, n_rows=120, k=2, null_frac=0.0)
        kw = dict(clock="t", halflife=HL, max_dclock=MAXD, weight="w", ridge=0.5, min_periods=5.0)
        bank = (
            po.ModelBank(
                [po.spec.rls("m", targets=["y0"], features=["x0", "x1"], group="group", **kw)]
            )
            .fit_predict(df)
            .select("m")
            .unnest("m")
        )
        expr = df.select(pl.col("y0").online.rls(features=["x0", "x1"], **kw).over("group")).unnest(
            "y0"
        )
        for c in bank.columns:
            if c.startswith("coef"):
                continue
            x, y = bank[c].to_numpy().astype(float), expr[c].to_numpy().astype(float)
            nan = np.isnan(x) & np.isnan(y)
            assert (nan | (x == y)).all(), c
