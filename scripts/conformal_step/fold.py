"""The conformal tracker and its candidate step rules, replayed over a bank's
served residuals in the bank's release order (docs/records/CONFORMAL-STEP.md).

The bank's model does not read its interval, so one run of the bank fixes
every served prediction; each rule here is then a fold over the same
residuals. `schedule` computes the release order the stream uses under
`label_clock` (a held row leaves before the first row whose clock reaches
its label time, in order of label time and then arrival), the decay of each
release from the label time released before it, and the residual spread
`sigma` each release reads. `bank_rule` is the bank's own tracker, and
`check_replica` holds it to the bank's `lo`/`hi` to the bit before any rule
is measured.

The rules (all read `sigma` and step in its units, as the bank does):

- ``fixed``: the bank today, a step of `rate * sigma` at each release,
  against the radius the row was shown;
- ``lag``: the same, with the rate divided by `1 + rate * g(alpha) * D / c`,
  `D` a measure of the predictions in flight and `g(alpha)` the density of
  `|Z|` at its `1 - alpha` quantile (the loop gain of the tracker on
  Gaussian residuals, per unit of rate);
- ``current``: the step taken against the radius the tracker holds at the
  release, not the one the row was shown (the coverage is still scored
  against the shown radius);
- ``agaci``: Zaffran et al. (2022)'s aggregation of several fixed-rate
  trackers by Bernstein online aggregation (Wintenberger 2017) of the pinball
  loss, with the gradient trick;
- ``saocp``: Bhatnagar et al. (2023)'s strongly adaptive tracker: a
  scale-free OGD expert started at every release, living `g * 2^v(t)`
  releases, combined by coin betting.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

# Acklam's rational approximation, as `online_core::conformal::norm_ppf` and
# tests/test_conformal.py evaluate it.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.38357751867269e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def norm_ppf(p: float) -> float:
    a, b, c, d = _A, _B, _C, _D

    def tail(q):
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if p < 0.02425:
        return tail(math.sqrt(-2.0 * math.log(p)))
    if p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return -tail(math.sqrt(-2.0 * math.log(1.0 - p)))


def gain(alpha: float) -> float:
    """g(alpha): the density of |Z|, Z standard normal, at its 1 - alpha
    quantile. On Gaussian residuals a rate `r` moves the expected error of
    the tracker by `r * g` per row and per unit of radius error in sigma."""
    z = norm_ppf(1.0 - 0.5 * alpha)
    return 2.0 * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def schedule(t, avail, resid, w, half_life, gap_cap):
    """The release order and what each release reads.

    Returns a dict of arrays over releases, in order: `row` (the released
    row), `at` (the row before whose serving it is released), `lam` (its
    decay), `sigma` (the spread before it is folded), and over rows:
    `released_before` (the releases done before the row's own batch of
    releases), `served_after` (the releases done when it is served),
    `held_before` (the rows held at the start of its batch)."""
    n = len(t)
    t = np.asarray(t, dtype=float)
    avail = np.asarray(avail, dtype=float)
    holds = np.isfinite(avail) & (w > 0)
    rows = np.nonzero(holds)[0]
    at = np.maximum(np.searchsorted(t, avail[rows], side="left"), rows + 1)
    keep = at < n
    rows, at = rows[keep], at[keep]
    order = np.lexsort((rows, avail[rows], at))
    rows, at = rows[order], at[order]
    lam, sigma = _spread(avail[rows], resid[rows], w[rows], float(half_life), float(gap_cap))
    released_before = np.searchsorted(at, np.arange(n), side="left")
    served_after = np.searchsorted(at, np.arange(n), side="right")
    held_before = np.cumsum(holds) - holds - released_before
    return {
        "row": rows,
        "at": at,
        "lam": lam,
        "sigma": sigma,
        "released_before": released_before,
        "held_before": held_before,
        "served_after": served_after,
    }


@nb.njit(cache=True)
def _spread(label, r, w, half_life, gap_cap):
    m = len(label)
    lam = np.empty(m)
    sig = np.empty(m)
    var = 0.0
    rw = 0.0
    last = np.nan
    for k in range(m):
        d = 0.0 if math.isnan(last) else min(label[k] - last, gap_cap)
        last = label[k]
        lam[k] = 2.0 ** (-(d / half_life))
        sig[k] = math.sqrt(max(var, 0.0)) if rw > 0.0 else np.nan
        if not math.isfinite(r[k]):
            rw *= lam[k]
            continue
        w_new = lam[k] * rw + w[k]
        var = (lam[k] * rw * var + w[k] * r[k] * r[k]) / w_new
        rw = w_new
    return lam, sig


# ---------------------------------------------------------------- the rules
# Each returns `shown`, the radius each row is shown (NaN before the warm
# start). `kind` for `_track`: 0 fixed, 1 lag on the rows held, 2 lag on the
# EW mean of the rows held, 3 lag on the row's own lag in releases, 4 its EW
# mean, 5 the step against the current radius, 6 a step only where the
# shown and the current radius agree on the miss.


@nb.njit(cache=True)
def _track(
    n,
    rows,
    at,
    lam,
    sigma,
    released_before,
    served_after,
    held_before,
    resid,
    w,
    alpha,
    rate,
    kind,
    c,
    g,
):
    shown = np.full(n, np.nan)
    m = len(rows)
    q = np.nan
    w_mean = 0.0
    w_rows = 0.0
    dbar = 0.0
    k = 0
    zq = _ppf(1.0 - 0.5 * alpha)
    for i in range(n):
        while k < m and at[k] == i:
            j = rows[k]
            r = resid[j]
            lm = lam[k]
            s = sigma[k]
            if kind == 1 or kind == 2:
                # the others in flight: the rows held, this one aside
                dk = held_before[i] - (k - released_before[i]) - 1
            else:
                dk = k - served_after[j]
            if not math.isfinite(r):
                w_rows *= lm
                k += 1
                continue
            w_rows = lm * w_rows + 1.0
            w_mean += (w[j] - w_mean) / w_rows
            dbar += (dk - dbar) / w_rows
            usable = math.isfinite(s) and s > 0.0
            if not math.isfinite(q):
                if usable:
                    q = s * zq
                k += 1
                continue
            against = shown[j]
            if kind == 5:
                against = q
            if not math.isfinite(against):
                k += 1
                continue
            miss = 1.0 if abs(r) > against else 0.0
            eta = rate * s if usable else 0.0
            if kind == 6 and miss != (1.0 if abs(r) > q else 0.0):
                # the shown and the current radius disagree: the step is
                # skipped (the radius has already moved past this score)
                eta = 0.0
            if kind == 1 or kind == 3:
                eta = eta / (1.0 + rate * g * dk / c)
            elif kind == 2 or kind == 4:
                eta = eta / (1.0 + rate * g * dbar / c)
            rel = w[j] / w_mean if w_mean > 0.0 else 1.0
            q = max(q + eta * rel * (miss - alpha), 0.0)
            k += 1
        shown[i] = q
    return shown


@nb.njit(cache=True)
def _ppf(p):
    a0, a1, a2, a3, a4, a5 = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.38357751867269e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b0, b1, b2, b3, b4 = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c0, c1, c2, c3, c4, c5 = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d0, d1, d2, d3 = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    if p < 0.02425:
        qq = math.sqrt(-2.0 * math.log(p))
        return (((((c0 * qq + c1) * qq + c2) * qq + c3) * qq + c4) * qq + c5) / (
            (((d0 * qq + d1) * qq + d2) * qq + d3) * qq + 1.0
        )
    if p <= 1.0 - 0.02425:
        qq = p - 0.5
        r = qq * qq
        return (
            (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5)
            * qq
            / (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
        )
    qq = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c0 * qq + c1) * qq + c2) * qq + c3) * qq + c4) * qq + c5) / (
        (((d0 * qq + d1) * qq + d2) * qq + d3) * qq + 1.0
    )


KINDS = {
    "fixed": 0,
    "lag_held": 1,
    "lag_held_ew": 2,
    "lag_own": 3,
    "lag_own_ew": 4,
    "current": 5,
    "agree": 6,
}


def track(sched, resid, w, coverage, rate=0.05, kind="fixed", c=1.0):
    n = len(resid)
    alpha = 1.0 - coverage
    return _track(
        n,
        sched["row"],
        sched["at"],
        sched["lam"],
        sched["sigma"],
        sched["released_before"],
        sched["served_after"],
        sched["held_before"],
        resid,
        w,
        alpha,
        rate,
        KINDS[kind],
        c,
        gain(alpha),
    )


# ------------------------------------------------------------------ AgACI


@nb.njit(cache=True)
def _agaci(n, rows, at, lam, sigma, resid, alpha, rates):
    """K trackers at fixed rates, each scored against the radius it showed,
    combined by BOA with adaptive per-expert rates on the pinball loss
    linearized at the combined radius (the gradient trick)."""
    K = len(rates)
    shown = np.full(n, np.nan)
    shown_k = np.full((n, K), np.nan)
    q = np.full(K, np.nan)
    R = np.zeros(K)  # regularized cumulative regret
    V = np.zeros(K)  # cumulative squared regret
    eta = np.zeros(K)
    E = 0.0
    logK = math.log(K)
    p = np.full(K, 1.0 / K)
    m = len(rows)
    k = 0
    zq = _ppf(1.0 - 0.5 * alpha)
    for i in range(n):
        while k < m and at[k] == i:
            j = rows[k]
            r = resid[j]
            s = sigma[k]
            k += 1
            if not math.isfinite(r):
                continue
            usable = math.isfinite(s) and s > 0.0
            if not math.isfinite(q[0]):
                if usable:
                    for e in range(K):
                        q[e] = s * zq
                continue
            if not math.isfinite(shown[j]):
                continue
            a = abs(r)
            grad = alpha - (1.0 if a > shown[j] else 0.0)
            for e in range(K):
                miss = 1.0 if a > shown_k[j, e] else 0.0
                if usable:
                    q[e] = max(q[e] + rates[e] * s * (miss - alpha), 0.0)
                reg = grad * (shown[j] - shown_k[j, e])  # the gain of the mix over expert e
                E = max(E, abs(reg))
                R[e] += reg - eta[e] * reg * reg
                V[e] += reg * reg
            if E > 0.0:
                top = -1e300
                for e in range(K):
                    eta[e] = (
                        min(1.0 / (2.0 * E), math.sqrt(logK / V[e]))
                        if V[e] > 0
                        else 1.0 / (2.0 * E)
                    )
                    top = max(top, eta[e] * -R[e])
                tot = 0.0
                for e in range(K):
                    p[e] = eta[e] * math.exp(eta[e] * -R[e] - top)
                    tot += p[e]
                for e in range(K):
                    p[e] /= tot
        if math.isfinite(q[0]):
            v = 0.0
            for e in range(K):
                v += p[e] * q[e]
                shown_k[i, e] = q[e]
            shown[i] = v
    return shown


def agaci(sched, resid, coverage, rates):
    return _agaci(
        len(resid),
        sched["row"],
        sched["at"],
        sched["lam"],
        sched["sigma"],
        resid,
        1.0 - coverage,
        np.asarray(rates, dtype=float),
    )


# ------------------------------------------------------------------ SAOCP

CAP = 4096


@nb.njit(cache=True)
def _pinball(score, s, alpha):
    return (1.0 - alpha) * (score - s) if score > s else alpha * (s - score)


@nb.njit(cache=True)
def _saocp(n, rows, at, sigma, resid, alpha, g, dmode, dscale):
    """dmode 0: D = sqrt(3) * the largest |resid| released so far (the
    paper's rule, with the stream's own past as the calibration split);
    1: D = sqrt(3) * dscale * sigma."""
    shown = np.full(n, np.nan)
    born = np.zeros(CAP, np.int64)
    life = np.zeros(CAP, np.int64)
    alive = np.zeros(CAP, np.bool_)
    s_e = np.zeros(CAP)  # each expert's radius
    g2 = np.zeros(CAP)  # each expert's sum of squared gradients
    gsum = np.zeros(CAP)  # sum of meta gradients
    wg = np.zeros(CAP)  # sum of w * meta gradient
    wt = np.zeros(CAP)  # current bet
    agg = np.nan
    rmax = 0.0
    t = 0
    m = len(rows)
    k = 0
    zq = _ppf(1.0 - 0.5 * alpha)
    for i in range(n):
        while k < m and at[k] == i:
            j = rows[k]
            r = resid[j]
            s = sigma[k]
            k += 1
            if not math.isfinite(r):
                continue
            a = abs(r)
            if not math.isfinite(agg):
                if math.isfinite(s) and s > 0.0:
                    agg = s * zq
                    rmax = max(rmax, a)
                continue
            rmax = max(rmax, a)
            if dmode == 0:
                D = math.sqrt(3.0) * rmax
            else:
                D = math.sqrt(3.0) * dscale * s if math.isfinite(s) and s > 0 else 0.0
            if D <= 0.0:
                continue
            t += 1
            # Start this step's expert at the current combined radius.
            v = 0
            tt = t
            while tt % 2 == 0:
                tt //= 2
                v += 1
            slot = -1
            for e in range(CAP):
                if alive[e] and t - born[e] >= life[e]:
                    alive[e] = False
                if slot < 0 and not alive[e]:
                    slot = e
            born[slot] = t
            life[slot] = g * (1 << v)
            alive[slot] = True
            s_e[slot] = agg
            g2[slot] = 0.0
            gsum[slot] = 0.0
            wg[slot] = 0.0
            wt[slot] = 0.0
            # Meta gradients and bets at the combined radius held now.
            lagg = _pinball(a, agg, alpha)
            for e in range(CAP):
                if not alive[e]:
                    continue
                ge = (lagg - _pinball(a, s_e[e], alpha)) / D
                if wt[e] <= 0.0:
                    ge = max(ge, 0.0)
                gsum[e] += ge
                wg[e] += wt[e] * ge
                age = t - born[e] + 1
                wt[e] = gsum[e] * (1.0 + wg[e]) / age
                # The expert's own step: scale-free OGD on the pinball loss.
                grad = alpha - (1.0 if a > s_e[e] else 0.0)
                g2[e] += grad * grad
                if g2[e] > 0.0:
                    s_e[e] = s_e[e] - (D / math.sqrt(3.0)) * grad / math.sqrt(g2[e])
            # Combine.
            tot = 0.0
            totp = 0.0
            num = 0.0
            nump = 0.0
            for e in range(CAP):
                if not alive[e] or t - born[e] + 1 > life[e]:
                    continue
                b = born[e]
                lb = 0
                bb = b
                while bb > 1:
                    bb //= 2
                    lb += 1
                pi = 1.0 / (b * b * (1.0 + lb))
                pe = pi * max(wt[e], 0.0)
                tot += pe
                num += pe * s_e[e]
                totp += pi
                nump += pi * s_e[e]
            agg = num / tot if tot > 0.0 else nump / totp
        shown[i] = agg
    return shown


def saocp(sched, resid, coverage, g=8, dmode=0, dscale=4.0):
    return _saocp(
        len(resid),
        sched["row"],
        sched["at"],
        sched["sigma"],
        resid,
        1.0 - coverage,
        g,
        dmode,
        dscale,
    )


def score(shown, resid, mask):
    ok = mask & np.isfinite(shown) & np.isfinite(resid)
    cov = float(np.mean(np.abs(resid[ok]) <= shown[ok]))
    half = float(np.mean(shown[ok]))
    # The worst coverage over blocks of 2,000 scored rows: a tracker that
    # oscillates covers well on average and badly in stretches.
    idx = np.nonzero(ok)[0]
    blocks = [idx[a : a + 2000] for a in range(0, len(idx) - 1999, 2000)]
    worst = (
        min(float(np.mean(np.abs(resid[b]) <= shown[b])) for b in blocks) if blocks else math.nan
    )
    # The second half of the scored rows alone: the tracker's steady state,
    # away from its start.
    late = idx[len(idx) // 2 :]
    return {
        "coverage": cov,
        "half_width": half,
        "worst_block": worst,
        "rows": int(ok.sum()),
        "late_coverage": float(np.mean(np.abs(resid[late]) <= shown[late])),
        "late_half_width": float(np.mean(shown[late])),
    }
