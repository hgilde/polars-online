"""Reference (oracle) implementations in numpy: deliberately slow and simple.

These define the semantics the Rust core must match to ~1e-9 (see docs/PLAN.md
section 9). Conventions, shared with the core:

- ``pred`` is out-of-sample: computed from the state *before* the current row's
  update, using the last solved coefficients (references solve every row).
- Feature null => the row is skipped: all outputs NaN, no update, but the clock
  still advances (its decay is folded into the next accepted row's delta).
- Target-j null => ``pred_j`` emitted, no update of ``r_j`` / ``sigma2_j``;
  the shared ``S`` still updates.
- Warmup: all outputs NaN (except nothing) while ``n_eff`` before the update is
  below ``min_periods``; additionally ``pred_j`` is NaN while target j has seen
  no data.
- Accumulators are EW *means* (stable under long runs): ``W = lam*W + w``,
  ``S = (lam*W_prev*S + w*x x^T)/W``. ``ridge`` is applied at solve time on the
  mean scale (per-observation, stable). With ``ridge_decay=True`` the ridge is a
  decaying prior on the *sum* scale, penalizing the intercept too -- exactly
  classic RLS regularization (used by the RLS agreement test).
- ``sigma2_j`` (EW residual variance) accumulates only on rows where ``pred_j``
  was emitted and ``y_j`` is present.
"""

from __future__ import annotations

import numpy as np


def compute_dclock(
    t: np.ndarray | None,
    session: np.ndarray | None,
    n: int,
    max_dclock: float = np.inf,
    on_clock_reset: str = "max",
    session_gap: float | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row clock deltas and state-reset flags (docs/PLAN.md section 3).

    Returns ``(dclock, reset)``. Row 0 has delta 0. A negative raw delta is
    handled per ``on_clock_reset``: "max" -> max_dclock, "zero" -> 0,
    "reset_state" -> reset. A session change overrides the delta with
    ``session_gap`` (or resets state if it is "reset").
    """
    d = np.zeros(n)
    reset = np.zeros(n, dtype=bool)
    if t is None:
        d[1:] = 1.0
    else:
        d[1:] = np.diff(t)
    for i in range(1, n):
        if session is not None and session[i] != session[i - 1]:
            if session_gap == "reset":
                reset[i] = True
                d[i] = 0.0
            else:
                d[i] = float(session_gap) if session_gap is not None else d[i]
                d[i] = min(max(d[i], 0.0), max_dclock)
            continue
        if d[i] < 0:
            if on_clock_reset == "max":
                d[i] = max_dclock
            elif on_clock_reset == "zero":
                d[i] = 0.0
            elif on_clock_reset == "reset_state":
                reset[i] = True
                d[i] = 0.0
            else:
                raise ValueError(on_clock_reset)
        else:
            d[i] = min(d[i], max_dclock)
    return d, reset


def _solve_ridge(
    S: np.ndarray,
    r: np.ndarray,
    W: float,
    ridge: float,
    add_intercept: bool,
    standardize: bool,
    ridge_decay: bool,
    prior_scale: float,
) -> np.ndarray:
    """Solve for coefficients from mean-form stats. Returns beta (len k_total)."""
    k_total = S.shape[0]
    if ridge_decay:
        # Sum-scale decaying prior, intercept penalized: (W*S + ps*ridge*I) b = W*r
        A = W * S + prior_scale * ridge * np.eye(k_total)
        return np.linalg.solve(A, W * r)
    if not standardize:
        D = np.eye(k_total)
        if add_intercept:
            D[0, 0] = 0.0
        return np.linalg.solve(S + ridge * D, r)
    # Standardize features (not the target) using S's own means/variances.
    if not add_intercept:
        s = np.sqrt(np.maximum(np.diag(S), 0.0))
        keep = s > 1e-12
        b = np.zeros(k_total)
        if keep.any():
            Ss = S[np.ix_(keep, keep)] / np.outer(s[keep], s[keep])
            b[keep] = np.linalg.solve(Ss + ridge * np.eye(keep.sum()), r[keep] / s[keep])
            b[keep] /= s[keep]
        return b
    m = S[0, 1:]
    ybar = r[0]
    C = S[1:, 1:] - np.outer(m, m)
    c = r[1:] - m * ybar
    v = np.diag(C).copy()
    s = np.sqrt(np.maximum(v, 0.0))
    keep = s > 1e-12
    beta = np.zeros(k_total)
    if keep.any():
        Cs = C[np.ix_(keep, keep)] / np.outer(s[keep], s[keep])
        b = np.linalg.solve(Cs + ridge * np.eye(keep.sum()), c[keep] / s[keep])
        beta[1:][keep] = b / s[keep]
    beta[0] = ybar - m @ beta[1:]
    return beta


def ewridge_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    reset: np.ndarray | None = None,
    halflife: float = 100.0,
    ridge: float = 1e-6,
    add_intercept: bool = True,
    min_periods: float | None = None,
    standardize: bool = False,
    ridge_decay: bool = False,
) -> dict[str, np.ndarray]:
    """EW-ridge oracle, solving every row. X: (n,k), Y: (n,m); NaN = null."""
    n, k = X.shape
    m = Y.shape[1]
    k_total = k + 1 if add_intercept else k
    if min_periods is None:
        min_periods = float(k_total)
    if reset is None:
        reset = np.zeros(n, dtype=bool)

    pred = np.full((n, m), np.nan)
    resid = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, k_total), np.nan)
    sig2_out = np.full((n, m), np.nan)

    def init():
        return {
            "W": 0.0,
            "Wj": np.zeros(m),
            "Wsig": np.zeros(m),
            "S": np.zeros((k_total, k_total)),
            "r": np.zeros((k_total, m)),
            "sig2": np.zeros(m),
            "beta": None,
            "prior_scale": 1.0,
            "pending": 0.0,
        }

    st = init()
    for i in range(n):
        if reset[i]:
            st = init()
        x_raw = X[i]
        if np.isnan(x_raw).any():
            st["pending"] += dclock[i]
            continue
        xi = np.concatenate(([1.0], x_raw)) if add_intercept else x_raw
        d = dclock[i] + st["pending"]
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife)

        # ---- predict (state before update) ----
        ready = st["W"] >= min_periods and st["beta"] is not None
        if ready:
            for j in range(m):
                if st["Wj"][j] > 0.0:
                    pred[i, j] = xi @ st["beta"][:, j]
                    if not np.isnan(Y[i, j]):
                        resid[i, j] = Y[i, j] - pred[i, j]
        n_eff[i] = st["W"]

        # ---- update ----
        W_new = lam * st["W"] + w[i]
        st["S"] = (lam * st["W"] * st["S"] + w[i] * np.outer(xi, xi)) / W_new
        for j in range(m):
            yij = Y[i, j]
            if not np.isnan(yij):
                Wj_new = lam * st["Wj"][j] + w[i]
                st["r"][:, j] = (lam * st["Wj"][j] * st["r"][:, j] + w[i] * xi * yij) / Wj_new
                st["Wj"][j] = Wj_new
                if not np.isnan(resid[i, j]):
                    Ws_new = lam * st["Wsig"][j] + w[i]
                    st["sig2"][j] = (
                        lam * st["Wsig"][j] * st["sig2"][j] + w[i] * resid[i, j] ** 2
                    ) / Ws_new
                    st["Wsig"][j] = Ws_new
            else:
                st["Wj"][j] *= lam
                st["Wsig"][j] *= lam
        st["W"] = W_new
        st["prior_scale"] *= lam
        sig2_out[i] = st["sig2"]

        # ---- solve (reference solves every row) ----
        beta = np.zeros((k_total, m))
        for j in range(m):
            if st["Wj"][j] > 0.0:
                beta[:, j] = _solve_ridge(
                    st["S"],
                    st["r"][:, j],
                    st["W"],
                    ridge,
                    add_intercept,
                    standardize,
                    ridge_decay,
                    st["prior_scale"],
                )
        st["beta"] = beta
        coef[i] = beta.T

    return {"pred": pred, "resid": resid, "n_eff": n_eff, "coef": coef, "sig2": sig2_out}


def rls_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    reset: np.ndarray | None = None,
    halflife: float = 100.0,
    ridge: float = 1.0,
    add_intercept: bool = True,
    min_periods: float | None = None,
) -> dict[str, np.ndarray]:
    """Classic RLS oracle via direct normal-equation solves (no Sherman-Morrison).

    A = decayed sum of w*x x^T plus the decaying prior ridge*I (intercept
    penalized); b_j = decayed sum of w*x*y_j. Rows with any NaN target are
    predict-only for all targets (RLS null-policy deviation, documented).
    """
    n, k = X.shape
    m = Y.shape[1]
    k_total = k + 1 if add_intercept else k
    if min_periods is None:
        min_periods = float(k_total)
    if reset is None:
        reset = np.zeros(n, dtype=bool)

    pred = np.full((n, m), np.nan)
    resid = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, k_total), np.nan)

    def init():
        return {
            "A": ridge * np.eye(k_total),
            "b": np.zeros((k_total, m)),
            "W": 0.0,
            "beta": None,
            "pending": 0.0,
            "seen": False,
        }

    st = init()
    for i in range(n):
        if reset[i]:
            st = init()
        x_raw = X[i]
        if np.isnan(x_raw).any():
            st["pending"] += dclock[i]
            continue
        xi = np.concatenate(([1.0], x_raw)) if add_intercept else x_raw
        d = dclock[i] + st["pending"]
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife)

        ready = st["W"] >= min_periods and st["beta"] is not None and st["seen"]
        if ready:
            pred[i] = xi @ st["beta"]
            for j in range(m):
                if not np.isnan(Y[i, j]):
                    resid[i, j] = Y[i, j] - pred[i, j]
        n_eff[i] = st["W"]

        st["A"] = lam * st["A"]
        st["b"] = lam * st["b"]
        st["W"] = lam * st["W"] + w[i]
        if not np.isnan(Y[i]).any():
            st["A"] = st["A"] + w[i] * np.outer(xi, xi)
            st["b"] = st["b"] + w[i] * np.outer(xi, Y[i])
            st["seen"] = True
        st["beta"] = np.linalg.solve(st["A"], st["b"])
        coef[i] = st["beta"].T

    return {"pred": pred, "resid": resid, "n_eff": n_eff, "coef": coef}


def kalman_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    reset: np.ndarray | None = None,
    halflife: float = 500.0,
    coef_halflife: float | list[float] = 100.0,
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float = 1.0,
    share_p: bool = False,
    add_intercept: bool = True,
    min_periods: float = 10.0,
) -> dict[str, np.ndarray]:
    """Kalman / random-walk-beta oracle (docs/PLAN.md section 4.4).

    Mirrors the core exactly, including the details that make it match:

    - features are standardized with the EW stats *before* the row's update,
      using scale 1 for the intercept slot and for near-zero-variance features
      (centered variance <= 1e-10 * raw second moment);
    - ``P += Q * d_clock`` happens before the gain, once per shared P;
    - innovation variance is ``z' P z + sigma2 / w`` (row weight scales the
      observation precision);
    - ``sigma2_j`` is the EW variance of the *out-of-sample* residual, updated
      only on rows where a prediction was emitted;
    - the EW stats update last, so this row's z used the prior stats.

    Coefficients come back in the ORIGINAL feature units.
    """
    n, k = X.shape
    m = Y.shape[1]
    off = 1 if add_intercept else 0
    kt = k + off
    if reset is None:
        reset = np.zeros(n, dtype=bool)
    hl = np.asarray(
        [coef_halflife] * kt if np.isscalar(coef_halflife) else coef_halflife, dtype=float
    )
    if hl.size == 1:
        hl = np.repeat(hl, kt)

    pred = np.full((n, m), np.nan)
    resid = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)

    def init():
        return {
            "W": 0.0,
            "mean": np.zeros(kt),
            "raw": np.zeros((kt, kt)),
            "beta": np.zeros((m, kt)),
            "P": [np.eye(kt) * p0 for _ in range(1 if share_p else m)],
            "sig2": np.zeros(m),
            "wsig": np.zeros(m),
            "wj": np.zeros(m),
            "pending": 0.0,
        }

    st = init()
    for i in range(n):
        if reset[i]:
            st = init()
        if np.isnan(X[i]).any():
            st["pending"] += dclock[i]
            continue
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i].copy()
        d = dclock[i] + st["pending"]
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife)

        # scales from the stats BEFORE this row
        scales = np.ones(kt)
        for j in range(off, kt):
            var = st["raw"][j, j] - st["mean"][j] ** 2
            raw = max(abs(st["raw"][j, j]), 1e-300)
            scales[j] = np.sqrt(var) if var > 1e-10 * raw else 1.0
        zs = np.ones(kt)
        for j in range(off, kt):
            zs[j] = (z[j] - st["mean"][j]) / scales[j]

        n_eff[i] = st["W"]
        ready = st["W"] >= min_periods
        if ready:
            for j in range(m):
                if st["wj"][j] > 0.0:
                    pred[i, j] = zs @ st["beta"][j]
                    if not np.isnan(Y[i, j]):
                        resid[i, j] = Y[i, j] - pred[i, j]

        for j in range(m):
            pi = 0 if share_p else j
            if obs_var is not None:
                sigma2 = obs_var
            else:
                s2 = st["sig2"].mean() if share_p else st["sig2"][j]
                sigma2 = s2 if s2 > 0.0 else 1.0
            if (not share_p) or j == 0:
                qv = (
                    np.asarray(q, dtype=float)
                    if q is not None
                    else np.where(np.isinf(hl), 0.0, sigma2 * (np.log(2.0) / hl) ** 2)
                )
                st["P"][pi] = st["P"][pi] + np.diag(qv * d)
            if np.isnan(Y[i, j]):
                st["wj"][j] *= lam
                st["wsig"][j] *= lam
                continue
            if w[i] <= 0.0:
                continue
            pz = st["P"][pi] @ zs
            s_inn = zs @ pz + sigma2 / w[i]
            if s_inn > 0.0:
                gain = pz / s_inn
                err = Y[i, j] - zs @ st["beta"][j]
                st["beta"][j] = st["beta"][j] + gain * err
                st["P"][pi] = st["P"][pi] - np.outer(gain, pz)
            if not np.isnan(pred[i, j]):
                r = Y[i, j] - pred[i, j]
                ws_new = lam * st["wsig"][j] + w[i]
                st["sig2"][j] = (lam * st["wsig"][j] * st["sig2"][j] + w[i] * r * r) / ws_new
                st["wsig"][j] = ws_new
            st["wj"][j] = lam * st["wj"][j] + w[i]

        # EW stats update last
        W_new = lam * st["W"] + w[i]
        a = lam * st["W"] / W_new
        b = w[i] / W_new
        st["mean"] = a * st["mean"] + b * z
        st["raw"] = a * st["raw"] + b * np.outer(z, z)
        st["W"] = W_new

        # coefficients back in original units
        for j in range(m):
            c = np.zeros(kt)
            c[off:] = st["beta"][j][off:] / scales[off:]
            if add_intercept:
                c[0] = st["beta"][j][0] - c[off:] @ st["mean"][off:]
            coef[i, j] = c

    return {"pred": pred, "resid": resid, "n_eff": n_eff, "coef": coef}
