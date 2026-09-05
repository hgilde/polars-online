"""Task 29 / ENHANCEMENTS E41: coefficient reversion on ``kalman``.

``revert_halflife`` gives each slot a reversion halflife ``r_i``. Before a row
predicts, the filter propagates its state over the clock gap ``d`` since the
last row: ``b <- Phi b``, ``P <- Phi P Phi`` with ``Phi = diag(2^(-d/r_i))``,
then adds the process noise ``Q d`` as before. ``inf`` (the default) is
``Phi = I``, the random walk, skipped entirely.

Three kinds of check:

- an oracle (``tests/reference.py::kalman_ref`` with ``revert_halflife``)
  over streams with clock gaps, weights, null targets, skipped rows and
  ``max_dclock``;
- large data: a slope that mean-reverts is tracked better by a reverting
  filter, a slope that random-walks is tracked better by the random walk,
  so the parameter is not a free improvement in either direction;
- exactness: over a run of rows with nothing to learn from, the coefficient
  shrinks by exactly ``2^(-sum(d)/r)`` on an irregular clock, ``inf`` is
  bit-identical to the default, and ``predict`` propagates by the same
  ``Phi`` for the row's clock distance.
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po
from data import synthetic
from reference import compute_dclock, kalman_ref

INF = float("inf")
MAXD = 50.0


def _spec(name="m", **kw):
    defaults = dict(
        targets=["y0"],
        features=["x0", "x1", "x2"],
        coef_halflife=100.0,
        halflife=500.0,
        min_periods=20.0,
    )
    defaults.update(kw)
    return po.spec.kalman(name, **defaults)


def _field(out, field, name="m"):
    return out[name].struct.field(field).to_numpy().astype(float)


def _coef(out, name="m"):
    """The emitted ``coef`` rows as an array with NaN where none was emitted."""
    rows = out[name].struct.field("coef").to_list()
    width = max((len(r) for r in rows if r is not None), default=0)
    arr = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        if r is not None:
            arr[i] = r
    return arr


def _arrays(df, k):
    x = np.column_stack([df[f"x{j}"].to_numpy() for j in range(k)])
    n = df.height
    dc = np.zeros(n)
    dc[1:] = np.diff(df["t"].to_numpy())
    return x, np.clip(dc, 0.0, MAXD), df["w"].to_numpy()


def _close(got, exp, tol=1e-9, what=""):
    both_nan = np.isnan(got) & np.isnan(exp)
    assert (np.isnan(got) == np.isnan(exp)).all(), f"{what}: null patterns differ"
    ok = both_nan | (np.abs(got - exp) <= tol * (1.0 + np.abs(exp)))
    assert ok.all(), f"{what}: max diff {np.nanmax(np.abs(got - exp))}"


# --- the oracle ---------------------------------------------------------------


class TestOracle:
    """``kalman_ref`` with the transition, on the same streams the plain
    filter is held to in ``test_oracles.py``."""

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
            revert_halflife=kw.get("revert_halflife"),
            add_intercept=kw.get("add_intercept", True),
            standardize=kw.get("standardize", True),
        )
        out = po.ModelBank([spec]).fit_predict(df)
        for j, t in enumerate(targets):
            for field, key in (("pred_", "pred"), ("resid_", "resid")):
                _close(_field(out, f"{field}{t}"), ref[key][:, j], what=f"{field}{t}")
        _close(_field(out, "n_eff"), ref["n_eff"], what="n_eff")
        return out, ref

    def test_scalar_revert_halflife(self):
        df, _ = synthetic(seed=91, n_groups=1, n_rows=800, k=3, null_frac=0.0)
        self._compare(df, revert_halflife=60.0)

    def test_per_slot_with_the_intercept_left_alone(self):
        df, _ = synthetic(seed=92, n_groups=1, n_rows=800, k=3, null_frac=0.0)
        self._compare(df, revert_halflife=[INF, 200.0, 25.0, 5.0])

    def test_reversion_with_per_factor_process_noise_and_a_pin(self):
        df, _ = synthetic(seed=93, n_groups=1, n_rows=800, k=3, null_frac=0.0)
        # A pinned coefficient (`q = 0`) that reverts decays to zero and
        # stays there: the transition is independent of the process noise.
        self._compare(
            df,
            coef_halflife=[INF, 500.0, 30.0, INF],
            revert_halflife=[INF, 80.0, 80.0, 40.0],
        )

    def test_explicit_q_fixed_obs_var_and_p0(self):
        df, _ = synthetic(seed=94, n_groups=1, n_rows=600, k=3, null_frac=0.0)
        self._compare(df, q=[0.0, 0.01, 0.02, 0.0], obs_var=0.25, p0=4.0, revert_halflife=30.0)

    def test_multi_target_per_target_p(self):
        df, _ = synthetic(seed=95, n_groups=1, n_rows=600, k=3, n_targets=2, null_frac=0.0)
        self._compare(df, targets=("y0", "y1"), revert_halflife=[INF, 40.0, 40.0, 10.0])

    def test_multi_target_shared_p(self):
        # The shared P is propagated once per row, not once per target.
        df, _ = synthetic(seed=96, n_groups=1, n_rows=600, k=3, n_targets=2, null_frac=0.0)
        self._compare(
            df, targets=("y0", "y1"), share_p=True, revert_halflife=[INF, 40.0, 40.0, 10.0]
        )

    def test_null_targets_and_skipped_features(self):
        # A null target advances the transition without a measurement; a
        # null feature skips the row and folds its clock into the next one.
        df, _ = synthetic(seed=97, n_groups=1, n_rows=800, k=3, null_frac=0.08)
        self._compare(df, revert_halflife=[INF, 30.0, 30.0, 30.0])

    def test_zero_weight_rows(self):
        df, _ = synthetic(seed=98, n_groups=1, n_rows=600, k=3, null_frac=0.0)
        df = df.with_columns(w=pl.when(pl.arange(0, df.height) % 7 == 3).then(0.0).otherwise("w"))
        self._compare(df, revert_halflife=20.0)

    def test_unstandardized(self):
        df, _ = synthetic(seed=99, n_groups=1, n_rows=600, k=3, null_frac=0.0)
        out, ref = self._compare(df, standardize=False, revert_halflife=[INF, 50.0, 50.0, 50.0])
        # Without standardization the state is the coefficient: the oracle's
        # coefficients are checked too, on every row the bank emitted them.
        got = _coef(out)
        emitted = np.isfinite(got[:, 0])
        assert emitted.sum() >= 1
        _close(got[emitted], ref["coef"][emitted, 0, :], tol=1e-9, what="coef")

    def test_no_intercept(self):
        df, _ = synthetic(seed=100, n_groups=1, n_rows=500, k=2, null_frac=0.0)
        self._compare(df, k=2, add_intercept=False, revert_halflife=[15.0, 60.0])

    def test_a_long_gap_is_capped_by_max_dclock(self):
        # The transition sees the same capped delta the decay does.
        df, _ = synthetic(seed=101, n_groups=1, n_rows=500, k=3, null_frac=0.0)
        t = df["t"].to_numpy().copy()
        t[250:] += 1000.0
        df = df.with_columns(t=pl.Series(t))
        self._compare(df, revert_halflife=25.0)


# --- large data -----------------------------------------------------------------


def _ou_stream(n, seed, *, revert_halflife, noise, density=1.0, walk_sigma=0.002):
    """``y = 0.3 + b1(t) x1 + b2(t) x2 + e``: ``b1`` a stationary AR(1)
    around zero with the given halflife and sd 1, ``b2`` a slow random walk.
    ``density < 1`` makes ``x1`` sparse -- zero most rows -- so its slope
    goes unobserved for long stretches, during which the truth reverts."""
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1.0 / revert_halflife)
    eta = rng.normal(0.0, math.sqrt(1.0 - phi * phi), n)
    b1 = np.empty(n)
    b1[0] = rng.normal()
    for i in range(1, n):
        b1[i] = phi * b1[i - 1] + eta[i]
    b2 = np.cumsum(rng.normal(0.0, walk_sigma, n)) + 1.0
    x1 = rng.normal(size=n)
    if density < 1.0:
        x1 *= rng.random(n) < density
    x2 = rng.normal(size=n)
    y = 0.3 + b1 * x1 + b2 * x2 + rng.normal(0.0, noise, n)
    df = pl.DataFrame({"y0": y, "x0": x1, "x1": x2})
    truth = np.column_stack([np.full(n, 0.3), b1, b2])
    # The exact process noise of the truth, so the reverting filter is the
    # Bayes filter for it and the random walk differs only in the transition.
    q = [0.0, 1.0 - phi * phi, walk_sigma * walk_sigma]
    return df, truth, q


def _track_mse(out, truth, tail):
    c = _coef(out)
    ok = np.isfinite(c[:, 0])
    ok[:-tail] = False
    return ((c[ok] - truth[ok]) ** 2).mean(axis=0)


class TestLargeData:
    N = 300_000
    TAIL = 250_000

    def _pair(self, df, q, noise, revert):
        common = dict(
            targets=["y0"],
            features=["x0", "x1"],
            coef_halflife=INF,
            q=q,
            obs_var=noise * noise,
            halflife=2000.0,
            min_periods=50.0,
            standardize=False,
            coef_every=1,
        )
        walk = po.ModelBank([po.spec.kalman("m", **common)]).fit_predict(df)
        rev = po.ModelBank([po.spec.kalman("m", revert_halflife=revert, **common)]).fit_predict(df)
        return walk, rev

    def _pred_mse(self, out, df):
        y = df["y0"].to_numpy()
        return np.nanmean((y - _field(out, "pred_y0"))[-self.TAIL :] ** 2)

    def test_a_sparse_regressor_whose_slope_reverts(self):
        # `x1` is nonzero on 2% of rows. Between those rows nothing says
        # what its slope is: the random walk carries the last estimate, the
        # reverting filter lets it decay as the truth does. Half the
        # tracking error, and better predictions, over the last 250k rows;
        # the always-on random-walk slope is tracked the same either way.
        df, truth, q = _ou_stream(self.N, 7, revert_halflife=40.0, noise=1.0, density=0.02)
        walk, rev = self._pair(df, q, 1.0, [INF, 40.0, INF])
        mse_walk = _track_mse(walk, truth, self.TAIL)
        mse_rev = _track_mse(rev, truth, self.TAIL)
        assert mse_rev[1] < 0.7 * mse_walk[1], (mse_rev, mse_walk)
        assert abs(mse_rev[2] - mse_walk[2]) < 0.02 * mse_walk[2], (mse_rev, mse_walk)
        assert self._pred_mse(rev, df) < self._pred_mse(walk, df)

    def test_a_dense_regressor_whose_slope_reverts(self):
        # Always observed, the gain from knowing the transition is smaller
        # but still there: the reverting filter is the Bayes filter for
        # this truth, the random walk a misspecified one.
        df, truth, q = _ou_stream(self.N, 8, revert_halflife=20.0, noise=2.0)
        walk, rev = self._pair(df, q, 2.0, [INF, 20.0, INF])
        mse_walk = _track_mse(walk, truth, self.TAIL)
        mse_rev = _track_mse(rev, truth, self.TAIL)
        assert mse_rev[1] < 0.95 * mse_walk[1], (mse_rev, mse_walk)
        assert abs(mse_rev[2] - mse_walk[2]) < 0.02 * mse_walk[2], (mse_rev, mse_walk)
        assert self._pred_mse(rev, df) < self._pred_mse(walk, df)

    def test_a_random_walk_slope_is_tracked_better_by_the_random_walk(self):
        # The other direction: when the truth does not revert, pulling it
        # toward zero costs accuracy. Reversion is a prior, not a free lunch.
        rng = np.random.default_rng(9)
        n = self.N
        b1 = np.cumsum(rng.normal(0.0, 0.01, n)) + 2.0
        x1 = rng.normal(size=n)
        y = 0.3 + b1 * x1 + rng.normal(0.0, 1.0, n)
        df = pl.DataFrame({"y0": y, "x0": x1})
        truth = np.column_stack([np.full(n, 0.3), b1])
        common = dict(
            targets=["y0"],
            features=["x0"],
            coef_halflife=INF,
            q=[0.0, 1e-4],
            obs_var=1.0,
            halflife=2000.0,
            min_periods=50.0,
            standardize=False,
            coef_every=1,
        )
        walk = po.ModelBank([po.spec.kalman("m", **common)]).fit_predict(df)
        rev = po.ModelBank(
            [po.spec.kalman("m", revert_halflife=[INF, 30.0], **common)]
        ).fit_predict(df)
        mse_walk = _track_mse(walk, truth, self.TAIL)
        mse_rev = _track_mse(rev, truth, self.TAIL)
        assert mse_walk[1] < 0.5 * mse_rev[1], (mse_walk, mse_rev)

    def test_reversion_bounds_the_covariance(self):
        # A slot nothing identifies: under the random walk `P_ii` grows by
        # `q d` per row without bound; under reversion it settles at
        # `q d / (1 - phi^2)`. Read through the Kalman gain: after 200k
        # rows of null targets, one observation `y = 5` at `z = e_0` moves
        # the coefficient by `5 P / (P + R)` (the mean had decayed to zero
        # under reversion, and to the fitted 1.0 under the walk).
        n = 200_000
        rng = np.random.default_rng(10)
        x = rng.normal(size=(300, 2))
        y = x[:, 0] + rng.normal(0.0, 0.1, 300)
        head = pl.DataFrame({"y0": y, "x0": x[:, 0], "x1": x[:, 1]})
        tail = pl.DataFrame(
            {"y0": np.full(n, np.nan), "x0": rng.normal(size=n), "x1": rng.normal(size=n)}
        )
        probe = pl.DataFrame({"y0": [5.0], "x0": [1.0], "x1": [0.0]})
        q, r_obs = 1e-4, 0.01
        common = dict(
            targets=["y0"],
            features=["x0", "x1"],
            add_intercept=False,
            coef_halflife=INF,
            q=[q, q],
            obs_var=r_obs,
            halflife=INF,
            min_periods=10.0,
            standardize=False,
            coef_every=1,
        )
        got = {}
        for rh in (INF, 100.0):
            bank = po.ModelBank([po.spec.kalman("m", revert_halflife=rh, **common)])
            bank.fit_predict(head)
            before = _coef(bank.fit_predict(tail))[-1]
            after = _coef(bank.fit_predict(probe))[-1]
            got[rh] = (before[0], after[0])
        phi = 0.5 ** (1.0 / 100.0)
        p_settled = q / (1.0 - phi * phi)
        # Reverting: the mean is gone and the gain is the settled one.
        assert abs(got[100.0][0]) < 1e-12
        assert got[100.0][1] == pytest.approx(5.0 * p_settled / (p_settled + r_obs), rel=1e-6)
        # Random walk: the mean is still the fitted slope and the gain ~1.
        p_walk = n * q
        b_fit = got[INF][0]
        assert abs(b_fit - 1.0) < 0.05
        gain = p_walk / (p_walk + r_obs)
        assert got[INF][1] == pytest.approx(b_fit + gain * (5.0 - b_fit), rel=1e-3)


# --- exactness ----------------------------------------------------------------


class TestExactness:
    def test_inf_is_bit_identical_to_the_default(self):
        df, _ = synthetic(seed=102, n_groups=2, n_rows=600, k=3, null_frac=0.03)
        base = _spec(group="group", clock="t", max_dclock=MAXD, weight="w", coef_every=1)
        want = po.ModelBank([base]).fit_predict(df)
        for r in (INF, [INF], [INF, INF, INF, INF]):
            spec = _spec(
                group="group",
                clock="t",
                max_dclock=MAXD,
                weight="w",
                coef_every=1,
                revert_halflife=r,
            )
            got = po.ModelBank([spec]).fit_predict(df)
            assert got.equals(want, null_equal=True), r

    def test_scalar_equals_the_list_that_spells_it(self):
        df, _ = synthetic(seed=103, n_groups=1, n_rows=400, k=3, null_frac=0.0)
        a = po.ModelBank([_spec(revert_halflife=33.0, coef_every=1)]).fit_predict(df)
        b = po.ModelBank([_spec(revert_halflife=[33.0] * 4, coef_every=1)]).fit_predict(df)
        assert a.equals(b, null_equal=True)

    def test_shrinks_exactly_over_a_run_of_null_targets_on_an_irregular_clock(self):
        # With nothing to learn from, only the transition moves the state:
        # after rows with deltas `d_1 .. d_n` the coefficient is
        # `2^(-sum(d)/r)` of what it was, per slot, whatever the spacing --
        # including deltas folded from skipped rows and capped by
        # `max_dclock`. Unstandardized, so `coef` is the state itself.
        rng = np.random.default_rng(104)
        n_fit, n_null = 300, 200
        n = n_fit + n_null
        dt = rng.exponential(2.0, n)
        dt[0] = 0.0
        dt[n_fit + 10] = 500.0  # capped at max_dclock
        t = np.cumsum(dt)
        x = rng.normal(size=(n, 3))
        y = 0.5 + x @ np.array([1.0, -2.0, 0.7]) + rng.normal(0.0, 0.1, n)
        y[n_fit:] = np.nan
        x[n_fit + 20 : n_fit + 25, 1] = np.nan  # skipped rows: clock folds
        df = pl.DataFrame({"t": t, "y0": y, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2]})
        r = [INF, 20.0, 7.0, 3.0]
        spec = _spec(
            clock="t",
            max_dclock=MAXD,
            revert_halflife=r,
            standardize=False,
            q=[0.0] * 4,
            coef_every=1,
        )
        out = po.ModelBank([spec]).fit_predict(df)
        c = _coef(out)
        c_fit = c[n_fit - 1]
        assert np.isfinite(c_fit).all()
        dc, _ = compute_dclock(t, None, n, max_dclock=MAXD)
        accepted = np.isfinite(x).all(axis=1)
        for i in range(n_fit, n):
            if not accepted[i]:
                assert np.isnan(c[i]).all(), i  # a skipped row emits nothing
                continue
            elapsed = dc[n_fit : i + 1].sum()  # skipped rows' deltas included
            want = c_fit * np.array([2.0 ** (-elapsed / h) for h in r])
            assert c[i] == pytest.approx(want, rel=1e-11, abs=1e-14), i
        assert c[-1, 0] == c_fit[0]  # the `inf` slot is untouched to the bit
        assert abs(c[-1, 3]) < 1e-30 * abs(c_fit[3]) + 1e-300

    def test_a_zero_weight_row_advances_the_transition(self):
        # Weight 0 is "advance the clock, learn nothing": the reversion is
        # clock, so it applies, and the measurement does not.
        rng = np.random.default_rng(105)
        n = 200
        x = rng.normal(size=(n, 2))
        y = x @ np.array([1.0, -1.0]) + rng.normal(0.0, 0.1, n)
        w = np.ones(n)
        w[-1] = 0.0
        y[-1] = 1e6  # would move any coefficient if it were learned from
        df = pl.DataFrame({"y0": y, "x0": x[:, 0], "x1": x[:, 1], "w": w})
        spec = _spec(
            features=["x0", "x1"],
            weight="w",
            revert_halflife=[INF, 4.0, 4.0],
            standardize=False,
            q=[0.0] * 3,
            coef_every=1,
        )
        c = _coef(po.ModelBank([spec]).fit_predict(df))
        assert c[-1, 1:] == pytest.approx(c[-2, 1:] * 0.5 ** (1.0 / 4.0), rel=1e-12)
        assert c[-1, 0] == c[-2, 0]

    def test_predict_propagates_over_the_clock_distance(self):
        # `predict(df)` scores each row as the next row of the stream: the
        # coefficients are propagated by `2^(-d/r)` for the row's clock
        # distance from the last learned row (capped by `max_dclock`, and
        # the cap itself for a backwards clock under `on_clock_reset =
        # "max"`), while the emitted `coef` is the frozen state.
        rng = np.random.default_rng(106)
        n = 300
        t = np.cumsum(rng.exponential(1.0, n))
        x = rng.normal(size=(n, 2))
        y = 0.5 + x @ np.array([1.5, -0.5]) + rng.normal(0.0, 0.1, n)
        df = pl.DataFrame({"t": t, "y0": y, "x0": x[:, 0], "x1": x[:, 1]})
        r = [INF, 10.0, 3.0]
        spec = _spec(
            features=["x0", "x1"],
            clock="t",
            max_dclock=20.0,
            revert_halflife=r,
            standardize=False,
            min_periods=10.0,
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df)
        last = bank.groups()["last_clock"][0]
        assert last == t[-1]
        ahead = pl.DataFrame(
            {
                "t": [
                    last,
                    last + 0.5,
                    last + 5.0,
                    last + 19.0,
                    last + 20.0,
                    last + 1e6,
                    last - 3.0,
                ],
                "x0": [0.3, -1.2, 0.8, 2.0, -0.4, 1.0, 0.1],
                "x1": [1.1, 0.4, -0.7, 0.2, 0.9, -1.0, 2.5],
            }
        )
        out = bank.predict(ahead)["m"]
        coef = np.array(out.struct.field("coef").to_list()[-1])
        pred = out.struct.field("pred_y0").to_numpy()
        for ti, x0, x1, p in zip(ahead["t"], ahead["x0"], ahead["x1"], pred, strict=True):
            d = 20.0 if ti < last else min(ti - last, 20.0)
            phi = np.array([2.0 ** (-d / h) for h in r])
            want = np.array([1.0, x0, x1]) @ (phi * coef)
            assert p == pytest.approx(want, rel=1e-12, abs=1e-12), (ti, d)
        # And the bank was not moved: scoring again gives the same numbers.
        assert bank.predict(ahead).equals(bank.predict(ahead), null_equal=True)

    def test_predict_is_fit_predict_of_the_next_row(self):
        # The general E31 contract, row by row, under reversion with a clock.
        df, _ = synthetic(seed=107, n_groups=2, n_rows=120, k=3, null_frac=0.0)
        spec = _spec(
            group="group", clock="t", max_dclock=MAXD, revert_halflife=[INF, 30.0, 30.0, 8.0]
        )
        bank = po.ModelBank([spec])
        bank.fit_predict(df.head(160))
        later = df.tail(40).with_columns(pl.col("t") + 3.0)
        snap = bank.save_bytes()
        got = bank.predict(later)
        for i in range(later.height):
            fresh = po.ModelBank.load_bytes(snap, [spec])
            want = fresh.fit_predict(later.slice(i, 1))
            a, b = got["m"][i], want["m"][0]
            for key in ("pred_y0", "resid_y0", "n_eff"):
                assert (
                    a[key] == b[key]
                    or (a[key] is None and b[key] is None)
                    or (np.isnan(a[key]) and np.isnan(b[key]))
                ), (i, key, a[key], b[key])


# --- plumbing ------------------------------------------------------------------


class TestPlumbing:
    def _spec(self, **kw):
        return _spec(
            group="group",
            clock="t",
            max_dclock=MAXD,
            weight="w",
            revert_halflife=[INF, 40.0, 40.0, 10.0],
            **kw,
        )

    def test_chunk_invariance(self):
        df, _ = synthetic(seed=108, n_groups=2, n_rows=800, k=3, null_frac=0.03)
        spec = self._spec()
        one = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        keep = [c for c in one.columns if not c.startswith("coef")]
        for size in (1, 7, 97, 500):
            bank = po.ModelBank([spec])
            many = (
                pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)])
                .select("m")
                .unnest("m")
            )
            assert one.select(keep).equals(many.select(keep), null_equal=True), size

    def test_save_load_mid_stream(self):
        df, _ = synthetic(seed=109, n_groups=2, n_rows=600, k=3, null_frac=0.03)
        spec = self._spec()
        want = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
        keep = [c for c in want.columns if not c.startswith("coef")]
        for cut in (3, 100, 500):
            bank = po.ModelBank([spec])
            head = bank.fit_predict(df.head(cut))
            bank = po.ModelBank.load_bytes(bank.save_bytes(), [spec])
            tail = bank.fit_predict(df.slice(cut))
            got = pl.concat([head, tail]).select("m").unnest("m")
            assert want.select(keep).equals(got.select(keep), null_equal=True), cut

    def test_groups_revert_independently(self):
        df, _ = synthetic(seed=110, n_groups=3, n_rows=300, k=3, null_frac=0.0)
        spec = self._spec()
        both = po.ModelBank([spec]).fit_predict(df).select("group", "m").unnest("m")
        keep = [c for c in both.columns if not c.startswith("coef")]
        for g in ("g0", "g1", "g2"):
            part = df.filter(pl.col("group") == g)
            alone = po.ModelBank([spec]).fit_predict(part).select("group", "m").unnest("m")
            assert (
                both.filter(pl.col("group") == g)
                .select(keep)
                .equals(alone.select(keep), null_equal=True)
            ), g

    def test_expression_equals_bank(self):
        df, _ = synthetic(seed=111, n_groups=2, n_rows=300, k=3, null_frac=0.0)
        one = po.ModelBank([self._spec()]).fit_predict(df).select("m").unnest("m")
        keep = [c for c in one.columns if not c.startswith("coef")]
        with pytest.warns(po.InMemoryExpressionWarning):
            expr = df.select(
                pl.col("y0")
                .online.kalman(
                    features=["x0", "x1", "x2"],
                    coef_halflife=100.0,
                    halflife=500.0,
                    min_periods=20.0,
                    clock="t",
                    max_dclock=MAXD,
                    weight="w",
                    revert_halflife=[INF, 40.0, 40.0, 10.0],
                )
                .over("group")
            ).unnest("y0")
        assert one.select(keep).equals(expr.select(keep), null_equal=True)

    def test_lazy_plan_equals_bank(self):
        df, _ = synthetic(seed=112, n_groups=2, n_rows=300, k=3, null_frac=0.0)
        spec = self._spec()
        want = po.ModelBank([spec]).fit_predict(df)
        got = df.lazy().online.fit_predict([spec]).collect()
        assert got.equals(want, null_equal=True)

    def test_cli_toml(self, tmp_path, online_cli):
        df, _ = synthetic(seed=113, n_groups=2, n_rows=300, k=3, null_frac=0.0)
        spec = self._spec()
        want = po.ModelBank([spec]).fit_predict(df)
        src = tmp_path / "in.parquet"
        dst = tmp_path / "out.parquet"
        df.write_parquet(src)
        (tmp_path / "bank.toml").write_text(
            f"""
input = "{src.as_posix()}"
output = "{dst.as_posix()}"

[[specs]]
name = "m"
targets = ["y0"]
features = ["x0", "x1", "x2"]
group = "group"
clock = "t"
max_dclock = {MAXD}
weight = "w"
halflife = 500.0
min_periods = 20.0

[specs.model]
type = "kalman"
coef_halflife = 100.0
revert_halflife = ["inf", 40.0, 40.0, 10.0]
""",
            encoding="utf-8",
        )
        subprocess.run([online_cli, "--config", str(tmp_path / "bank.toml")], check=True)
        got = pl.read_parquet(dst)
        assert got.select("m").equals(want.select("m"), null_equal=True)

    def test_output_index_is_unchanged(self):
        assert po.spec.output_index(self._spec()).equals(po.spec.output_index(_spec()))


# --- refusals -------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        ("value", "exc", "msg"),
        [
            (
                "10",
                TypeError,
                "revert_halflife must be a number or a list of numbers, got str '10'",
            ),
            (float("nan"), ValueError, "revert_halflife must not be NaN"),
            ([INF, float("nan"), 1.0, 1.0], ValueError, "revert_halflife must not be NaN"),
            (0.0, ValueError, 'revert_halflife must be > 0 ("inf" is the random walk)'),
            (-5.0, ValueError, 'revert_halflife must be > 0 ("inf" is the random walk)'),
            (
                [INF, 10.0, -1.0, 10.0],
                ValueError,
                'revert_halflife must be > 0 ("inf" is the random walk)',
            ),
            ([10.0, 10.0], ValueError, "revert_halflife must be scalar or length 4"),
            ([], ValueError, "revert_halflife must be scalar or length 4"),
        ],
    )
    def test_named_refusals(self, value, exc, msg):
        import re

        with pytest.raises(exc, match=re.escape(msg)):
            po.ModelBank([_spec(revert_halflife=value)])

    def test_inf_is_allowed_in_every_position(self):
        po.ModelBank([_spec(revert_halflife=INF)])
        po.ModelBank([_spec(revert_halflife=[INF, INF, 5.0, INF])])

    def test_other_models_reject_it(self):
        with pytest.raises(TypeError, match="revert_halflife"):
            po.spec.rls("r", targets=["y0"], features=["x0"], revert_halflife=10.0)
        with pytest.raises(TypeError, match="revert_halflife"):
            po.spec.ewridge("e", targets=["y0"], features=["x0"], revert_halflife=10.0)
