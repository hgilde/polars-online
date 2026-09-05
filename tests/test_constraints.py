"""E40: constrained coefficients on ``sgd`` and ``pa`` (docs/ENHANCEMENTS.md).

``coef_min`` / ``coef_max`` bound each slope, ``coef_sum`` fixes their total,
and after every update the slopes are replaced by the nearest feasible point
(the Euclidean projection). The intercept is never constrained.

Three layers:

* **Oracle** -- a Python replay of ``sgd`` and ``pa`` with the projection
  written out (breakpoint search, same arithmetic order), held to the bank's
  ``pred`` / ``n_eff`` / ``coef`` row by row, under nulls, zero and NaN
  weights, skipped rows and an irregular clock; plus an independent
  bisection-and-KKT check of the projection itself.
* **Large data** -- 200k rows: weights on the simplex recovered and feasible
  at every row; a sign constraint lands on the clamped truth; a sum alone is
  the hyperplane projection; a box clamps; the constraint holds in the
  caller's units under ``scale_features``.
* **Edge cases** -- pinned slopes, infinite bounds, list vs scalar bounds,
  several targets, zero-weight and null-target rows, the input bound, chunk
  invariance, save/load, ``predict``, groups, the expression, ``po.run``,
  the CLI, and every refusal by name.
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import polars as pl
import pytest

import polars_online as po
import reference

INF = float("inf")


# ------------------------------------------------------------ the projection


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def project(b: list[float], lo, hi, s, scales=None) -> list[float]:
    """Mirror of ``Constraint::project``: same breakpoints, same sums, same order."""
    k = len(b)

    def bound(i):
        sc = 1.0 if scales is None else scales[i]
        return 1.0 / sc, lo[i] * sc, hi[i] * sc

    if s is None:
        return [_clamp(b[i], *bound(i)[1:]) for i in range(k)]

    def t_hi(i):
        a, _, h = bound(i)
        return (b[i] - h) / a

    def t_lo(i):
        a, low, _ = bound(i)
        return (b[i] - low) / a

    breaks = sorted(t for i in range(k) for t in (t_hi(i), t_lo(i)) if math.isfinite(t))

    def g(mu):
        acc = 0.0
        for i in range(k):
            a, low, h = bound(i)
            acc += a * _clamp(b[i] - mu * a, low, h)
        return acc - s

    # partition_point: first index whose predicate is false.
    left, right = 0, len(breaks)
    while left < right:
        mid = (left + right) // 2
        if g(breaks[mid]) > 0.0:
            left = mid + 1
        else:
            right = mid
    if left < len(breaks):
        t, below = breaks[left], True
    else:
        t, below = (breaks[-1] if breaks else 0.0), False
    num = den = 0.0
    for i in range(k):
        a, low, h = bound(i)
        at_hi = below and t <= t_hi(i)
        at_lo = t > t_lo(i) if below else t >= t_lo(i)
        if at_hi:
            num += a * h
        elif at_lo:
            num += a * low
        else:
            num += a * b[i]
            den += a * a
    mu = (num - s) / den if den > 0.0 else t
    return [_clamp(b[i] - mu * bound(i)[0], *bound(i)[1:]) for i in range(k)]


def project_by_bisection(v, lo, hi, s, scales=None, tol=1e-13):
    """Independent of the breakpoint search: bisect mu on the sum."""
    k = len(v)
    a = [1.0 if scales is None else 1.0 / scales[i] for i in range(k)]
    lo_b = [lo[i] * (1.0 if scales is None else scales[i]) for i in range(k)]
    hi_b = [hi[i] * (1.0 if scales is None else scales[i]) for i in range(k)]
    if s is None:
        return [_clamp(v[i], lo_b[i], hi_b[i]) for i in range(k)]

    def at(mu):
        return [_clamp(v[i] - mu * a[i], lo_b[i], hi_b[i]) for i in range(k)]

    def total(mu):
        return sum(a[i] * x for i, x in enumerate(at(mu))) - s

    left, right = -1e6, 1e6
    assert total(left) >= 0.0 >= total(right)
    for _ in range(300):
        mid = 0.5 * (left + right)
        if total(mid) > 0.0:
            left = mid
        else:
            right = mid
        if right - left < tol:
            break
    return at(0.5 * (left + right))


class TestProjection:
    def test_matches_bisection_on_random_problems(self):
        rng = np.random.default_rng(0)
        for trial in range(400):
            k = int(rng.integers(1, 8))
            lo = [(-INF if rng.random() < 0.3 else float(rng.normal())) for _ in range(k)]
            hi = [
                (INF if rng.random() < 0.3 else float(max(lo[i], -3.0) + abs(rng.normal())))
                for i in range(k)
            ]
            scales = (
                None if rng.random() < 0.5 else [float(0.2 + 2 * rng.random()) for _ in range(k)]
            )
            inside = [
                max(lo[i], -3.0) + (min(hi[i], 3.0) - max(lo[i], -3.0)) * rng.random()
                for i in range(k)
            ]
            s = None if rng.random() < 0.3 else float(sum(inside))
            v = [float(4 * rng.normal()) for _ in range(k)]
            got = project(v, lo, hi, s, scales)
            want = project_by_bisection(v, lo, hi, s, scales)
            assert got == pytest.approx(want, abs=1e-9), (trial, v, lo, hi, s, scales)
            # Feasible, and a second projection changes nothing beyond rounding.
            for i in range(k):
                sc = 1.0 if scales is None else scales[i]
                assert lo[i] * sc - 1e-12 <= got[i] <= hi[i] * sc + 1e-12
            if s is not None:
                assert sum(
                    g / (1.0 if scales is None else scales[i]) for i, g in enumerate(got)
                ) == pytest.approx(s, abs=1e-9)
            assert project(got, lo, hi, s, scales) == pytest.approx(got, abs=1e-12)

    def test_the_simplex_matches_the_sorting_formula(self):
        rng = np.random.default_rng(1)
        for k in (1, 2, 3, 6, 11):
            for _ in range(50):
                v = [float(x) for x in 3 * rng.normal(size=k)]
                u = sorted(v, reverse=True)
                cum, theta = 0.0, 0.0
                for r, ur in enumerate(u, start=1):
                    cum += ur
                    cand = (cum - 1.0) / r
                    if ur - cand > 0.0:
                        theta = cand
                want = [max(x - theta, 0.0) for x in v]
                got = project(v, [0.0] * k, [INF] * k, 1.0)
                assert got == pytest.approx(want, abs=1e-12)


# ------------------------------------------------------------------ oracles


def _usable(v: float) -> bool:
    return math.isfinite(v) and abs(v) <= 1e100


def sgd_replay(
    X,
    y,
    *,
    lr=0.05,
    halflife=INF,
    min_periods=10.0,
    lo=None,
    hi=None,
    s=None,
    schedule="constant",
    power=0.5,
    l2=0.0,
    clip=1e3,
    t=None,
    w=None,
    max_dclock=INF,
):
    """``sgd`` (squared loss, no scaler) with the projection, row by row."""
    n, k = X.shape
    lo = [-INF] * k if lo is None else list(lo)
    hi = [INF] * k if hi is None else list(hi)
    constrained = s is not None or any(v != -INF for v in lo) or any(v != INF for v in hi)
    d_all, _ = reference.compute_dclock(t, None, n, max_dclock=max_dclock)
    beta = [0.0] * (k + 1)
    if constrained:
        beta[1:] = project(beta[1:], lo, hi, s)
    g2 = [0.0] * (k + 1)
    w_sum = 0.0
    pending = 0.0
    preds, neffs, coefs = [], [], []
    for i in range(n):
        wi = 1.0 if w is None else float(w[i])
        pending = pending + d_all[i] if i > 0 else 0.0
        if not all(_usable(v) for v in X[i]) or not math.isfinite(wi):
            preds.append(None)
            neffs.append(None)
            coefs.append(None)
            continue
        d = pending  # skipped rows fold their (capped) deltas in
        pending = 0.0
        lam = 1.0 if halflife == INF else math.exp2(-(d / halflife))
        z = [1.0, *map(float, X[i])]
        if lam != 1.0:
            g2 = [v * lam for v in g2]
        n_eff = w_sum
        ready = n_eff >= min_periods
        eta = 0.0
        for zi, bi in zip(z, beta, strict=True):
            eta += zi * bi
        preds.append(eta if ready else None)
        neffs.append(n_eff)
        yi = y[i]
        learned = False
        if yi is not None and not math.isnan(yi) and wi > 0.0 and math.isfinite(yi):
            dl = eta - yi
            for j in range(k + 1):
                g = dl * z[j] * wi
                if j >= 1:
                    g += l2 * beta[j]
                g = _clamp(g, -clip, clip)
                if schedule == "constant":
                    rate = lr
                elif schedule == "inv_scaling":
                    rate = lr / (1.0 + n_eff) ** power
                else:
                    g2[j] += g * g
                    rate = lr / (math.sqrt(g2[j]) + 1e-8)
                beta[j] -= rate * g
            learned = True
        if learned and constrained:
            beta[1:] = project(beta[1:], lo, hi, s)
        coefs.append(list(beta))
        w_sum = lam * w_sum + wi
    return preds, neffs, coefs


def pa_replay(
    X,
    y,
    *,
    mode="pa1",
    c=1.0,
    eps=0.1,
    halflife=INF,
    min_periods=10.0,
    lo=None,
    hi=None,
    s=None,
    t=None,
    w=None,
    max_dclock=INF,
):
    n, k = X.shape
    lo = [-INF] * k if lo is None else list(lo)
    hi = [INF] * k if hi is None else list(hi)
    constrained = s is not None or any(v != -INF for v in lo) or any(v != INF for v in hi)
    d_all, _ = reference.compute_dclock(t, None, n, max_dclock=max_dclock)
    beta = [0.0] * (k + 1)
    if constrained:
        beta[1:] = project(beta[1:], lo, hi, s)
    w_sum = 0.0
    pending = 0.0
    preds, neffs, coefs = [], [], []
    for i in range(n):
        wi = 1.0 if w is None else float(w[i])
        pending = pending + d_all[i] if i > 0 else 0.0
        if not all(_usable(v) for v in X[i]) or not math.isfinite(wi):
            preds.append(None)
            neffs.append(None)
            coefs.append(None)
            continue
        d = pending  # skipped rows fold their (capped) deltas in
        pending = 0.0
        lam = 1.0 if halflife == INF else math.exp2(-(d / halflife))
        z = [1.0, *map(float, X[i])]
        n_eff = w_sum
        ready = n_eff >= min_periods
        sq = 0.0
        for zi in z:
            sq += zi * zi
        p = 0.0
        for zi, bi in zip(z, beta, strict=True):
            p += zi * bi
        preds.append(p if ready else None)
        neffs.append(n_eff)
        yi = y[i]
        if (
            yi is not None
            and not math.isnan(yi)
            and wi > 0.0
            and math.isfinite(yi)
            and math.isfinite(p)
            and sq > 0.0
        ):
            err = yi - p
            loss = max(abs(err) - eps, 0.0)
            if loss != 0.0:
                if mode == "pa":
                    tau = loss / sq
                elif mode == "pa1":
                    tau = min(loss / sq, c)
                else:
                    tau = loss / (sq + 0.5 / c)
                tau *= min(wi, 1.0)
                step = tau * math.copysign(1.0, err)
                for j in range(k + 1):
                    beta[j] += step * z[j]
                if constrained:
                    beta[1:] = project(beta[1:], lo, hi, s)
        coefs.append(list(beta))
        w_sum = lam * w_sum + wi
    return preds, neffs, coefs


def _frame(X, y, t=None, w=None, **cols):
    k = X.shape[1]
    data = {f"x{i}": X[:, i] for i in range(k)}
    data["y"] = y
    if t is not None:
        data["t"] = t
    if w is not None:
        data["w"] = w
    data.update(cols)
    return pl.DataFrame(data)


def _same_as(out: pl.DataFrame, preds, neffs, coefs, what: str) -> None:
    got_p = out["m"].struct.field("pred_y").to_list()
    got_n = out["m"].struct.field("n_eff").to_list()
    got_c = out["m"].struct.field("coef").to_list()
    assert len(got_p) == len(preds)
    for i in range(len(preds)):
        assert got_n[i] == neffs[i], f"{what}: n_eff[{i}] {got_n[i]!r} vs {neffs[i]!r}"
        if preds[i] is None:
            assert got_p[i] is None, f"{what}: pred[{i}] {got_p[i]!r} should be null"
        else:
            assert got_p[i] is not None and math.isclose(
                got_p[i], preds[i], rel_tol=1e-12, abs_tol=1e-12
            ), f"{what}: pred[{i}] {got_p[i]!r} vs {preds[i]!r}"
        if got_c[i] is not None:
            assert coefs[i] is not None, f"{what}: coef[{i}] emitted on a skipped row"
            assert got_c[i] == pytest.approx(coefs[i], rel=1e-12, abs=1e-12), f"{what}: coef[{i}]"


def _stream(n, k, seed, truth=None, noise=0.1, missing=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    truth = np.array(truth if truth is not None else rng.normal(size=k))
    y = X @ truth + 0.25 + noise * rng.standard_normal(n)
    t = np.cumsum(rng.integers(1, 5, size=n)).astype(float)
    w = np.ones(n)
    if missing:
        w[::17] = 0.0
        w[::23] = np.nan
        y[::13] = np.nan
        X[::31, 0] = np.nan
        X[::41, 1] = 1e101
    return X, y, t, w


CONSTRAINTS = {
    "box": dict(coef_min=-0.5, coef_max=0.5),
    "sign": dict(coef_min=0.0),
    "simplex": dict(coef_min=0.0, coef_sum=1.0),
    "hyperplane": dict(coef_sum=0.5),
    "pinned": dict(coef_min=[0.3, -INF, -INF], coef_max=[0.3, INF, 0.2]),
    "mixed": dict(coef_min=[-1.0, 0.0, -INF], coef_max=[1.0, INF, INF], coef_sum=0.25),
}


def _bounds(k, kw):
    lo = kw.get("coef_min", -INF)
    hi = kw.get("coef_max", INF)
    lo = [lo] * k if not isinstance(lo, list) else lo
    hi = [hi] * k if not isinstance(hi, list) else hi
    return lo, hi, kw.get("coef_sum")


class TestOracle:
    @pytest.mark.parametrize("which", list(CONSTRAINTS))
    @pytest.mark.parametrize("schedule", ["constant", "inv_scaling", "adagrad"])
    def test_sgd_matches_the_replay(self, which, schedule):
        kw = CONSTRAINTS[which]
        X, y, t, w = _stream(3000, 3, seed=hash((which, schedule)) % 1000, truth=[0.4, 0.3, -0.2])
        lo, hi, s = _bounds(3, kw)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            halflife=50.0,
            min_periods=5.0,
            learning_rate=0.05,
            schedule=schedule,
            l2=0.01,
            clock="t",
            weight="w",
            max_dclock=3.0,
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(_frame(X, y, t, w))
        want = sgd_replay(
            X,
            y,
            lr=0.05,
            halflife=50.0,
            min_periods=5.0,
            lo=lo,
            hi=hi,
            s=s,
            schedule=schedule,
            l2=0.01,
            t=t,
            w=w,
            max_dclock=3.0,
        )
        _same_as(out, *want, what=f"sgd {which} {schedule}")

    @pytest.mark.parametrize("which", list(CONSTRAINTS))
    @pytest.mark.parametrize("mode", ["pa", "pa1", "pa2"])
    def test_pa_matches_the_replay(self, which, mode):
        kw = CONSTRAINTS[which]
        X, y, t, w = _stream(3000, 3, seed=hash((which, mode)) % 1000, truth=[0.4, 0.3, -0.2])
        lo, hi, s = _bounds(3, kw)
        spec = po.spec.pa(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            halflife=50.0,
            min_periods=5.0,
            mode=mode,
            c=0.2,
            eps=0.05,
            clock="t",
            weight="w",
            max_dclock=3.0,
            coef_every=1,
            **kw,
        )
        out = po.ModelBank([spec]).fit_predict(_frame(X, y, t, w))
        want = pa_replay(
            X,
            y,
            mode=mode,
            c=0.2,
            eps=0.05,
            halflife=50.0,
            min_periods=5.0,
            lo=lo,
            hi=hi,
            s=s,
            t=t,
            w=w,
            max_dclock=3.0,
        )
        _same_as(out, *want, what=f"pa {which} {mode}")

    def test_unconstrained_is_the_same_replay_with_nothing_projected(self):
        # The replay's projection is off when nothing is constrained; the
        # constrained path with infinite bounds is bit-identical to it.
        X, y, t, w = _stream(2000, 2, seed=5, truth=[0.7, -0.3])
        base = dict(
            targets=["y"], features=["x0", "x1"], halflife=INF, min_periods=5.0, coef_every=1
        )
        plain = po.ModelBank([po.spec.sgd("m", **base)]).fit_predict(_frame(X, y))
        explicit = po.ModelBank(
            [po.spec.sgd("m", coef_min=-INF, coef_max=INF, **base)]
        ).fit_predict(_frame(X, y))
        assert plain.equals(explicit, null_equal=True)
        _same_as(plain, *sgd_replay(X, y, min_periods=5.0, lr=0.01), what="plain sgd")


# --------------------------------------------------------------- large data


def _coefs(out: pl.DataFrame) -> np.ndarray:
    rows = [c for c in out["m"].struct.field("coef").to_list() if c is not None]
    return np.array(rows, dtype=float)


class TestLargeData:
    def test_simplex_weights_are_recovered_and_feasible_at_every_row(self):
        rng = np.random.default_rng(0)
        n, k = 200_000, 5
        truth = rng.dirichlet(np.ones(k))
        X = rng.standard_normal((n, k))
        y = X @ truth + 0.1 * rng.standard_normal(n)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=[f"x{i}" for i in range(k)],
            halflife=INF,
            min_periods=10.0,
            learning_rate=0.01,
            coef_min=0.0,
            coef_sum=1.0,
            coef_every=1,
        )
        out = po.ModelBank([spec]).fit_predict(_frame(X, y))
        c = _coefs(out)
        assert c.shape[0] == n
        slopes = c[:, 1:]
        assert slopes.min() >= 0.0
        assert np.abs(slopes.sum(axis=1) - 1.0).max() <= 1e-12
        # Row 0's coef is one step past the uniform start (a step is O(lr)).
        assert np.abs(slopes[0] - 1.0 / k).max() < 0.05
        # A constant rate jitters around the optimum by O(sqrt(lr)·sigma);
        # the iterate average over the tail is the estimate.
        assert slopes[-1] == pytest.approx(truth, abs=0.03), (slopes[-1], truth)
        assert slopes[-20_000:].mean(axis=0) == pytest.approx(truth, abs=0.005)
        assert abs(c[-20_000:, 0].mean()) < 0.005

    def test_a_sign_constraint_lands_on_the_clamped_truth(self):
        # Uncorrelated unit-scale features: the constrained least-squares
        # optimum is the truth with its negative entries set to zero.
        rng = np.random.default_rng(1)
        n = 200_000
        truth = np.array([0.8, -0.4, 0.3, -0.1])
        X = rng.standard_normal((n, 4))
        y = X @ truth + 0.2 * rng.standard_normal(n)
        base = dict(
            targets=["y"],
            features=["x0", "x1", "x2", "x3"],
            halflife=INF,
            min_periods=10.0,
            learning_rate=0.01,
            coef_every=1,
        )
        free = _coefs(po.ModelBank([po.spec.sgd("m", **base)]).fit_predict(_frame(X, y)))
        signed = _coefs(
            po.ModelBank([po.spec.sgd("m", coef_min=0.0, **base)]).fit_predict(_frame(X, y))
        )
        assert free[-1, 1:] == pytest.approx(truth, abs=0.02)
        assert (free[:, 1:] < 0).any()
        assert signed[:, 1:].min() >= 0.0
        tail = signed[-50_000:, 1:]
        # The residual now carries the -0.4 signal, so the jitter is wider;
        # the tail average is the estimate, and the slopes the truth would
        # pull negative sit on the wall most of the time.
        assert tail.mean(axis=0) == pytest.approx(np.maximum(truth, 0.0), abs=0.02)
        assert (tail[:, 1] == 0.0).mean() > 0.7 and tail[:, 1].max() < 0.02
        assert (tail[:, 3] == 0.0).mean() > 0.2 and tail[:, 3].max() < 0.1

    def test_a_sum_alone_is_the_hyperplane_projection(self):
        rng = np.random.default_rng(2)
        n = 200_000
        truth = np.array([0.5, 0.2, 0.1])
        X = rng.standard_normal((n, 3))
        y = X @ truth + 0.2 * rng.standard_normal(n)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            halflife=INF,
            min_periods=10.0,
            learning_rate=0.01,
            coef_sum=1.0,
            coef_every=1,
        )
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        assert np.abs(c[:, 1:].sum(axis=1) - 1.0).max() <= 1e-12
        want = truth + (1.0 - truth.sum()) / 3.0
        assert c[-50_000:, 1:].mean(axis=0) == pytest.approx(want, abs=0.01)

    def test_a_box_clamps_the_truth(self):
        rng = np.random.default_rng(3)
        n = 200_000
        truth = np.array([0.8, -0.4, 0.1])
        X = rng.standard_normal((n, 3))
        y = X @ truth + 0.2 * rng.standard_normal(n)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            halflife=INF,
            min_periods=10.0,
            learning_rate=0.01,
            coef_min=-0.25,
            coef_max=0.25,
            coef_every=1,
        )
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        assert c[:, 1:].min() >= -0.25 and c[:, 1:].max() <= 0.25
        tail = c[-50_000:, 1:]
        assert tail.mean(axis=0) == pytest.approx([0.25, -0.25, 0.1], abs=0.02)
        # Off the wall only for the step after a gradient points inward:
        # how often is set by how hard the truth pushes (0.55 vs 0.15 away).
        assert (tail[:, 0] == 0.25).mean() > 0.75 and (tail[:, 1] == -0.25).mean() > 0.25

    @pytest.mark.parametrize("schedule", ["inv_scaling", "adagrad"])
    def test_the_other_schedules_stay_on_the_simplex(self, schedule):
        rng = np.random.default_rng(4)
        n, k = 100_000, 4
        truth = rng.dirichlet(np.ones(k))
        X = rng.standard_normal((n, k))
        y = X @ truth + 0.1 * rng.standard_normal(n)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=[f"x{i}" for i in range(k)],
            halflife=INF,
            min_periods=10.0,
            learning_rate=0.05 if schedule == "adagrad" else 0.5,
            schedule=schedule,
            coef_min=0.0,
            coef_sum=1.0,
            coef_every=1,
        )
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        assert c[:, 1:].min() >= 0.0
        assert np.abs(c[:, 1:].sum(axis=1) - 1.0).max() <= 1e-12
        assert c[-20_000:, 1:].mean(axis=0) == pytest.approx(truth, abs=0.01), schedule

    def test_pa_recovers_simplex_weights(self):
        rng = np.random.default_rng(5)
        n, k = 200_000, 4
        truth = rng.dirichlet(np.ones(k))
        X = rng.standard_normal((n, k))
        y = X @ truth + 0.02 * rng.standard_normal(n)
        spec = po.spec.pa(
            "m",
            targets=["y"],
            features=[f"x{i}" for i in range(k)],
            halflife=INF,
            min_periods=10.0,
            mode="pa1",
            c=0.05,
            eps=0.05,
            coef_min=0.0,
            coef_sum=1.0,
            coef_every=1,
        )
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        assert c[:, 1:].min() >= 0.0
        assert np.abs(c[:, 1:].sum(axis=1) - 1.0).max() <= 1e-12
        assert c[-50_000:, 1:].mean(axis=0) == pytest.approx(truth, abs=0.01)

    def test_the_constraint_holds_in_the_callers_units_under_scaling(self):
        # Features a million apart in scale; the bound is on the coefficient
        # in the caller's units, and holds after every learned row.
        rng = np.random.default_rng(6)
        n = 100_000
        X = np.column_stack([1000.0 * rng.standard_normal(n), 0.001 * rng.standard_normal(n)])
        y = 0.002 * X[:, 0] + 900.0 * X[:, 1] + 0.1 * rng.standard_normal(n)
        spec = po.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            halflife=INF,
            min_periods=10.0,
            learning_rate=0.01,
            scale_features=True,
            coef_min=0.0,
            coef_max=100.0,
            coef_sum=100.0,
            coef_every=1,
        )
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        slopes = c[:, 1:]
        assert slopes.min() >= -1e-12 and slopes.max() <= 100.0 + 1e-10
        assert np.abs(slopes.sum(axis=1) - 100.0).max() <= 1e-9
        # The well-determined slope keeps its truth; the sum goes to the
        # other, whose unit is a million times cheaper in the standardized
        # metric.
        assert c[-1, 1] == pytest.approx(0.002, abs=2e-4)
        assert c[-1, 2] == pytest.approx(100.0 - 0.002, abs=2e-4)


# --------------------------------------------------------------- edge cases


def _base(**kw):
    d = dict(
        targets=["y"], features=["x0", "x1", "x2"], halflife=INF, min_periods=5.0, coef_every=1
    )
    d.update(kw)
    return d


def _clocked(**kw):
    d = dict(clock="t", max_dclock=3.0, weight="w", halflife=40.0)
    d.update(kw)
    return _base(**d)


class TestEdgeCases:
    def test_a_pinned_slope_is_exact_on_every_row(self):
        X, y, t, w = _stream(2000, 3, seed=7, truth=[0.4, 0.3, -0.2], missing=False)
        spec = po.spec.sgd("m", coef_min=[0.7, -INF, -INF], coef_max=[0.7, INF, INF], **_base())
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        assert (c[:, 1] == 0.7).all()
        assert c[-1, 2] == pytest.approx(0.3, abs=0.05)

    def test_everything_pinned_by_the_sum(self):
        # lo == hi on every slope and a sum equal to their total: legal, and
        # the slopes never move.
        X, y, t, w = _stream(500, 3, seed=8, missing=False)
        spec = po.spec.sgd(
            "m", coef_min=[0.1, 0.2, 0.3], coef_max=[0.1, 0.2, 0.3], coef_sum=0.6, **_base()
        )
        c = _coefs(po.ModelBank([spec]).fit_predict(_frame(X, y)))
        assert (c[:, 1:] == [0.1, 0.2, 0.3]).all()

    def test_a_scalar_bound_is_the_list_of_that_bound(self):
        X, y, t, w = _stream(1000, 3, seed=9)
        a = po.ModelBank([po.spec.sgd("m", coef_min=0.0, coef_max=0.5, **_base())]).fit_predict(
            _frame(X, y)
        )
        b = po.ModelBank(
            [po.spec.sgd("m", coef_min=[0.0] * 3, coef_max=[0.5] * 3, **_base())]
        ).fit_predict(_frame(X, y))
        assert a.equals(b, null_equal=True)

    def test_one_sided_bounds(self):
        X, y, t, w = _stream(2000, 3, seed=10, truth=[0.4, -0.6, 0.2], missing=False)
        below = _coefs(
            po.ModelBank([po.spec.sgd("m", coef_min=-0.3, **_base())]).fit_predict(_frame(X, y))
        )
        above = _coefs(
            po.ModelBank([po.spec.sgd("m", coef_max=0.3, **_base())]).fit_predict(_frame(X, y))
        )
        assert below[:, 1:].min() >= -0.3 and below[-1, 2] == -0.3
        assert above[:, 1:].max() <= 0.3 and above[-1, 1] == 0.3

    def test_each_target_is_projected_on_its_own(self):
        X, y, t, w = _stream(1500, 3, seed=11, truth=[0.4, 0.3, -0.2])
        y2 = X @ np.array([-0.5, 0.8, 0.1]) + 0.1 * np.random.default_rng(12).standard_normal(
            len(y)
        )
        df = _frame(X, y, y2=y2)
        kw = dict(coef_min=0.0, coef_sum=1.0)
        both = po.ModelBank([po.spec.sgd("m", **_base(targets=["y", "y2"]), **kw)]).fit_predict(df)
        one = po.ModelBank([po.spec.sgd("m", **_base(), **kw)]).fit_predict(df)
        two = po.ModelBank([po.spec.sgd("m", **_base(targets=["y2"]), **kw)]).fit_predict(df)
        assert both["m"].struct.field("pred_y").equals(one["m"].struct.field("pred_y"))
        assert both["m"].struct.field("pred_y2").equals(two["m"].struct.field("pred_y2"))
        c = _coefs(both)
        assert c.shape[1] == 8
        for slopes in (c[:, 1:4], c[:, 5:8]):
            assert slopes.min() >= 0.0
            assert np.abs(slopes.sum(axis=1) - 1.0).max() <= 1e-12

    def test_a_zero_weight_or_null_target_row_moves_nothing(self):
        X, y, t, w = _stream(300, 3, seed=13, missing=False)
        spec = po.spec.sgd("m", weight="w", coef_min=0.0, coef_sum=1.0, **_base())
        bank = po.ModelBank([spec])
        bank.fit_predict(_frame(X, y, w=np.ones(len(y))))
        before = bank.coef("m")["coef"].to_list()
        bank.fit_predict(
            pl.DataFrame({"x0": [0.3], "x1": [-0.2], "x2": [0.9], "y": [5.0], "w": [0.0]})
        )
        assert bank.coef("m")["coef"].to_list() == before
        bank.fit_predict(
            pl.DataFrame({"x0": [0.3], "x1": [-0.2], "x2": [0.9], "y": [None], "w": [1.0]})
        )
        assert bank.coef("m")["coef"].to_list() == before
        # A zero-weight first row: the clock advances, nothing is learned,
        # and the projected start is what predict sees.
        fresh = po.ModelBank(
            [po.spec.sgd("m", weight="w", coef_min=0.0, coef_sum=1.0, **_base(min_periods=0.0))]
        )
        out = fresh.fit_predict(
            pl.DataFrame({"x0": [1.0], "x1": [2.0], "x2": [3.0], "y": [1.0], "w": [0.0]})
        )
        assert out["m"].struct.field("pred_y")[0] == pytest.approx(2.0)
        assert fresh.coef("m")["coef"].to_list() == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3])

    def test_the_input_bound_skips_a_row_before_the_projection_sees_it(self):
        X, y, t, w = _stream(200, 3, seed=14, missing=False)
        X[100] = [1e101, 0.0, 0.0]
        spec = po.spec.sgd("m", coef_min=0.0, coef_sum=1.0, **_base())
        out = po.ModelBank([spec]).fit_predict(_frame(X, y))
        assert out["m"].struct.field("n_eff")[100] is None
        assert out["m"].struct.field("coef")[100] is None
        assert out["m"].struct.field("n_eff")[101] == 100.0

    def test_chunk_invariance(self):
        X, y, t, w = _stream(1000, 3, seed=15)
        df = _frame(X, y, t, w)
        for spec in (
            po.spec.sgd("m", coef_min=0.0, coef_sum=1.0, **_clocked()),
            po.spec.pa("m", coef_min=-0.3, coef_max=0.3, **_clocked()),
        ):
            one = po.ModelBank([spec]).fit_predict(df)
            for size in (1, 7, 97, 500):
                bank = po.ModelBank([spec])
                many = pl.concat(
                    [bank.fit_predict(df.slice(i, size)) for i in range(0, df.height, size)]
                )
                a, b = one.unnest("m"), many.unnest("m")
                assert a.drop("coef").equals(b.drop("coef"), null_equal=True), size
                has = a["coef"].is_not_null()
                assert a.filter(has)["coef"].equals(b.filter(has)["coef"]), size

    def test_save_load(self, tmp_path):
        X, y, t, w = _stream(800, 3, seed=16)
        df = _frame(X, y, t, w)
        for spec in (
            po.spec.sgd("m", coef_min=0.0, coef_sum=1.0, schedule="adagrad", **_clocked()),
            po.spec.pa("m", coef_min=[-0.2, 0.0, -INF], **_clocked()),
        ):
            for cut in (3, 100, 500):
                a = po.ModelBank([spec])
                a.fit_predict(df.slice(0, cut))
                path = tmp_path / f"{spec['model']['type']}{cut}.state"
                a.save(path)
                b = po.ModelBank.load(path, specs=[spec])
                rest = df.slice(cut, df.height - cut)
                assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut

    def test_predict_reads_the_projected_coefficients_and_moves_nothing(self):
        X, y, t, w = _stream(500, 3, seed=17, missing=False)
        spec = po.spec.sgd("m", coef_min=0.0, coef_sum=1.0, **_base())
        bank = po.ModelBank([spec])
        bank.fit_predict(_frame(X, y))
        c = np.array(bank.coef("m")["coef"].to_list())
        probe = pl.DataFrame({"x0": [1.0, -1.0], "x1": [0.5, 0.5], "x2": [0.0, 2.0]})
        p = bank.predict(probe)["m"].struct.field("pred_y").to_numpy()
        want = c[0] + np.column_stack([probe["x0"], probe["x1"], probe["x2"]]) @ c[1:]
        assert p == pytest.approx(want, rel=1e-12)
        assert bank.coef("m")["coef"].to_list() == c.tolist()

    def test_groups_are_projected_independently(self):
        X, y, t, w = _stream(1000, 3, seed=18, missing=False)
        df = _frame(X, y, g=["p", "q"] * 500)
        spec = po.spec.sgd("m", group="g", coef_min=0.0, coef_sum=1.0, **_base())
        both = po.ModelBank([spec]).fit_predict(df)
        solo = po.ModelBank([spec]).fit_predict(df.filter(pl.col("g") == "q"))
        assert both.filter(pl.col("g") == "q").equals(solo, null_equal=True)

    def test_expression_equals_bank(self):
        X, y, t, w = _stream(600, 3, seed=19, missing=False)
        df = _frame(X, y, g=["p", "q", "r"] * 200)
        for builder, method in ((po.spec.sgd, "sgd"), (po.spec.pa, "pa")):
            spec = builder("m", group="g", coef_min=0.0, coef_sum=1.0, **_base())
            bank = po.ModelBank([spec]).fit_predict(df).select("m").unnest("m")
            with pytest.warns(po.InMemoryExpressionWarning):
                expr = df.select(
                    getattr(pl.col("y").online, method)(
                        features=["x0", "x1", "x2"],
                        halflife=INF,
                        min_periods=5.0,
                        coef_every=1,
                        coef_min=0.0,
                        coef_sum=1.0,
                    ).over("g")
                ).unnest("y")
            assert bank.equals(expr, null_equal=True), method

    def test_lazy_and_runner_agree_with_the_bank(self, tmp_path):
        X, y, t, w = _stream(600, 3, seed=20)
        df = _frame(X, y, t, w)
        spec = po.spec.sgd("m", coef_min=0.0, coef_sum=1.0, **_clocked())
        want = po.ModelBank([spec]).fit_predict(df)
        lazy = df.lazy().online.fit_predict([spec]).collect()
        assert want.equals(lazy, null_equal=True)
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        df.write_parquet(src)
        po.run(input=str(src), output=str(dst), specs=[spec])
        assert want.equals(pl.read_parquet(dst), null_equal=True)

    def test_the_cli_runs_it(self, tmp_path, online_cli):
        X, y, t, w = _stream(400, 3, seed=21)
        df = _frame(X, y, t, w)
        src, dst, cfg = tmp_path / "in.parquet", tmp_path / "out.parquet", tmp_path / "bank.toml"
        df.write_parquet(src)
        cfg.write_text(
            "\n".join(
                [
                    f'input = "{src.as_posix()}"',
                    f'output = "{dst.as_posix()}"',
                    "[[specs]]",
                    'name = "m"',
                    'features = ["x0", "x1", "x2"]',
                    'targets = ["y"]',
                    'clock = "t"',
                    "max_dclock = 3.0",
                    'weight = "w"',
                    "halflife = 40.0",
                    "min_periods = 5.0",
                    "coef_every = 1",
                    "[specs.model]",
                    'type = "sgd"',
                    "coef_min = [0.0, 0.0, -0.1]",
                    'coef_max = "inf"',
                    "coef_sum = 1.0",
                ]
            )
        )
        subprocess.run([str(online_cli), "--config", str(cfg)], check=True, capture_output=True)
        spec = po.spec.sgd(
            "m", coef_min=[0.0, 0.0, -0.1], coef_max=INF, coef_sum=1.0, **_clocked(halflife=40.0)
        )
        want = po.ModelBank([spec]).fit_predict(df)
        assert pl.read_parquet(dst).equals(want, null_equal=True)

    def test_output_index_is_unchanged_by_a_constraint(self):
        spec = po.spec.sgd("m", coef_min=0.0, coef_sum=1.0, **_base())
        idx = po.spec.output_index(spec)
        assert idx["field"].to_list() == ["pred_y", "resid_y", "n_eff", "coef"]
        assert po.spec.coef_fields(spec)["name"].to_list() == [
            "coef_y_intercept",
            "coef_y_x0",
            "coef_y_x1",
            "coef_y_x2",
        ]


# ---------------------------------------------------------------- refusals


class TestRefusals:
    @pytest.mark.parametrize("builder", [po.spec.sgd, po.spec.pa])
    def test_shapes_and_values_are_named_in_python(self, builder):
        with pytest.raises(
            TypeError, match="coef_min must be a number or a list of numbers, got str 'a'"
        ):
            builder("m", coef_min="a", **_base())
        with pytest.raises(
            TypeError, match="coef_max must be a number or a list of numbers, got list"
        ):
            builder("m", coef_max=["a"], **_base())
        with pytest.raises(TypeError, match="coef_sum must be a number, got str"):
            builder("m", coef_sum="1", **_base())
        with pytest.raises(ValueError, match="coef_sum must be finite, got float inf"):
            builder("m", coef_sum=INF, **_base())
        with pytest.raises(ValueError, match="coef_min must not be NaN"):
            builder("m", coef_min=float("nan"), **_base())
        with pytest.raises(ValueError, match="coef_max must not be NaN"):
            builder("m", coef_max=[0.0, float("nan")], **_base())

    @pytest.mark.parametrize(
        ("kw", "msg"),
        [
            (dict(coef_min=[0.0, 0.0]), "coef_min lists 2 bounds for 3 features"),
            (dict(coef_max=[0.0] * 4), "coef_max lists 4 bounds for 3 features"),
            (dict(coef_min=1.0, coef_max=0.5), r"coef_min\[0\] = 1 is above coef_max\[0\] = 0.5"),
            (
                dict(coef_min=[0.0, 0.0, 2.0], coef_max=1.0),
                r"coef_min\[2\] = 2 is above coef_max\[2\] = 1",
            ),
            (
                dict(coef_min=0.0, coef_max=1.0, coef_sum=3.5),
                r"coef_sum = 3.5 is outside what the bounds allow, \[0, 3\]",
            ),
            (
                dict(coef_min=0.0, coef_sum=-1.0),
                r"coef_sum = -1 is outside what the bounds allow, \[0, inf\]",
            ),
            (dict(coef_min=INF), r"coef_min\[0\] is inf \(use -inf for no bound\)"),
            (dict(coef_max=-INF), r"coef_max\[0\] is -inf \(use inf for no bound\)"),
        ],
    )
    @pytest.mark.parametrize("builder", [po.spec.sgd, po.spec.pa])
    def test_the_rust_side_names_the_offence(self, builder, kw, msg):
        with pytest.raises(ValueError, match=f"{builder.__name__}: {msg}"):
            po.ModelBank([builder("m", **kw, **_base())])

    def test_other_builders_do_not_take_them(self):
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'coef_min'"):
            po.spec.ewridge("m", coef_min=0.0, **_base())

    def test_a_hand_built_spec_is_parsed_by_path(self):
        spec = po.spec.sgd("m", **_base())
        spec["model"]["coef_min"] = "0"
        with pytest.raises(
            ValueError,
            match=r'model: invalid value: string "0", expected a number or a list of numbers',
        ):
            po.ModelBank([spec])
        spec["model"]["coef_min"] = "-inf"
        spec["model"]["coef_sum"] = "inf"
        with pytest.raises(ValueError, match='model: invalid type: string "inf", expected f64'):
            po.ModelBank([spec])
