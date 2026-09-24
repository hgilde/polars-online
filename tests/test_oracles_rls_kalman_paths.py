"""``rls`` and ``kalman`` on the paths their oracles leave out.

``rls`` has held only one target with no nulls to ``tests/reference.py``
``rls_ref``, and ``coef_prior``, several targets and ``add_intercept=False``
only through ``ewridge`` (Rust) or not at all. Here it is held row by row to
``tests/reference_paths.py::rls_paths_ref``, which sums the documented
recursion from the raw rows: ``pred``, ``resid``, ``n_eff`` and ``coef``,
across several targets under the rule that a row with any null target is
learned for none, a decaying ``coef_prior``, and no intercept.

``kalman`` has held several targets and a null target to ``kalman_ref``, but
never together; the cases below hold them together, with per-target and
shared ``P`` and without an intercept. ``coef`` is not compared: the core
reports the state in the scales *after* the row, so that ``coef`` applied to
the next row's features is that row's ``pred`` (1e-16), while ``kalman_ref``
unscales with the stats from *before* the row and centres with those after
it -- a mix no fit has, off by up to 0.38 (reported, 2026-09-24).

The streams are ``tests/test_oracles_lasso_paths.py``'s for ``rls`` and
``tests/data.py::synthetic`` for ``kalman``, as ``TestKalmanOracle`` uses it.
"""

import numpy as np
import pytest

import polars_online as po
from data import synthetic
from reference import kalman_ref
from reference_paths import rls_paths_ref
from test_oracles_lasso_paths import FEATURES, _stream

MAX_DCLOCK = 6.0

# Measured over the cases below as |got - expected| / (1 + |expected|): rls
# pred 5.8e-15, resid 2.1e-14, n_eff 2.4e-15, coef 1.1e-14; kalman pred
# 4.9e-15, n_eff exact. The tolerance is 100x the largest, rounded up to a
# power of ten.
#
# Seeded into a copy of rls_paths_ref: each target learned where it is
# present (in place of the any-null rule) moves pred by 2.6-4.0; a prior
# that does not decay, by 7.4e-3 to 0.10; n_eff over the learned rows only,
# by 0.08-2.5. Seeded into a copy of kalman_ref: a null target taken as 0
# moves pred by 0.10-0.31; a null target that does not age its own weight,
# by 1.1e-3 to 1.4e-3.
TOL = 1e-11


def _close(got, exp, tol, what):
    assert (np.isnan(got) == np.isnan(exp)).all(), (
        f"{what}: null patterns differ at rows {np.flatnonzero(np.isnan(got) != np.isnan(exp))[:8]}"
    )
    ok = ~np.isnan(exp)
    err = np.abs(got[ok] - exp[ok]) / (1.0 + np.abs(exp[ok]))
    assert err.size == 0 or err.max() <= tol, f"{what}: max rel diff {err.max():.3e}"


class TestRls:
    @staticmethod
    def _check(df, targets, **kw):
        spec = po.spec.rls(
            "m",
            targets=targets,
            features=FEATURES,
            clock="t",
            max_dclock=MAX_DCLOCK,
            weight="w",
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        n = df.height
        dc = np.zeros(n)
        dc[1:] = np.diff(df["t"].to_numpy())
        y = df.select(targets).to_numpy().astype(float)
        if "coef_prior" in kw:
            kw["coef_prior"] = np.asarray(kw["coef_prior"])
        ref = rls_paths_ref(
            df.select(FEATURES).to_numpy(),
            y,
            dc,
            df["w"].to_numpy(),
            max_dclock=MAX_DCLOCK,
            **kw,
        )
        for j, t in enumerate(targets):
            got = out.struct.field(f"pred_{t}").to_numpy().astype(float)
            _close(got, ref["pred"][:, j], TOL, f"pred_{t}")
            resid = out.struct.field(f"resid_{t}").to_numpy().astype(float)
            _close(resid, y[:, j] - ref["pred"][:, j], TOL, f"resid_{t}")
            assert np.isfinite(ref["pred"][:, j]).sum() > 150
        _close(out.struct.field("n_eff").to_numpy().astype(float), ref["n_eff"], TOL, "n_eff")
        rows = out.struct.field("coef").to_list()
        coef = np.array(
            [[np.nan] * ref["coef"][0].size if r is None else r for r in rows], float
        ).reshape(ref["coef"].shape)
        _close(coef, ref["coef"], TOL, "coef")

    def test_several_targets_learn_only_rows_where_all_are_present(self):
        self._check(_stream(51), ["ya", "yb", "yc"], halflife=60.0, ridge=0.5)

    def test_a_prior_that_fades_as_the_sums_decay(self):
        self._check(
            _stream(51),
            ["ya", "yc"],
            halflife=40.0,
            ridge=2.0,
            coef_prior=[[0.3, 1.0, -0.6, 0.0, 0.25], [-2.0, 0.0, -0.4, 0.0, 1.5]],
        )

    def test_without_an_intercept(self):
        self._check(
            _stream(52, level=3.0),
            ["ya"],
            halflife=80.0,
            ridge=1.0,
            add_intercept=False,
            coef_prior=[[1.0, -0.5, 0.0, 0.3]],
        )


class TestKalmanSeveralTargetsWithNulls:
    @pytest.mark.parametrize(
        "kw",
        [{}, {"share_p": True}, {"add_intercept": False}],
        ids=["per-target-P", "shared-P", "no-intercept"],
    )
    def test_each_target_on_its_own_rows(self, kw):
        df, _ = synthetic(seed=79, n_groups=1, n_rows=300, k=3, n_targets=2, null_frac=0.05)
        maxd = 50.0
        x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(3)])
        dc = np.zeros(df.height)
        dc[1:] = np.diff(df["t"].to_numpy())
        y = np.column_stack([df[t].to_numpy() for t in ("y0", "y1")]).astype(float)
        assert np.isnan(y).any(axis=0).all(), "both targets should have nulls"
        ref = kalman_ref(
            x,
            y,
            np.clip(dc, 0.0, maxd),
            df["w"].to_numpy(),
            halflife=500.0,
            coef_halflife=100.0,
            min_periods=10.0,
            max_dclock=maxd,
            **kw,
        )
        spec = po.spec.kalman(
            "m",
            targets=["y0", "y1"],
            features=["x0", "x1", "x2"],
            clock="t",
            max_dclock=maxd,
            weight="w",
            halflife=500.0,
            coef_halflife=100.0,
            min_periods=10.0,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(df)["m"]
        for j, t in enumerate(("y0", "y1")):
            got = out.struct.field(f"pred_{t}").to_numpy().astype(float)
            _close(got, ref["pred"][:, j], TOL, f"pred_{t}")
            resid = out.struct.field(f"resid_{t}").to_numpy().astype(float)
            _close(resid, ref["resid"][:, j], TOL, f"resid_{t}")
        _close(out.struct.field("n_eff").to_numpy().astype(float), ref["n_eff"], TOL, "n_eff")
