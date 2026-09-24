"""Reference (oracle) implementations in numpy: deliberately slow and simple.

These define the semantics the Rust core must match to ~1e-9 (see docs/PLAN.md
section 9). Conventions, shared with the core:

- ``pred`` is out-of-sample: computed from the state *before* the current row's
  update, using the last solved coefficients (references solve every row,
  except ``lasso_ref``, which keeps the documented solve schedule).
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

    Per row only: the references below fold a skipped row's delta into the
    next accepted row's and cap that total at their own ``max_dclock``, as
    the stream does (review 2026-09-12, S3).
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
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """EW-ridge oracle, solving every row. X: (n,k), Y: (n,m); NaN = null.

    ``target_gaps="own_rows"`` (docs/PLAN.md task 81): each target's Gram is
    over the rows it is present on, and ages with its weight over the rest;
    ``n_eff`` is the weight over every row."""
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
            "S": np.zeros((m, k_total, k_total)),
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
        d = min(dclock[i] + st["pending"], max_dclock)
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife)

        # ---- predict (state before update) ----
        ready = st["W"] >= min_periods and st["beta"] is not None
        # The model's own prediction, which its own statistics fold; what is
        # emitted waits, besides, for each target's own weight (review
        # 2026-09-12, S2) -- a gate on the output, not on the model.
        p_own = np.full(m, np.nan)
        if ready:
            for j in range(m):
                if st["Wj"][j] > 0.0:
                    p_own[j] = xi @ st["beta"][:, j]
                    if st["Wj"][j] >= min_periods:
                        pred[i, j] = p_own[j]
                        if not np.isnan(Y[i, j]):
                            resid[i, j] = Y[i, j] - pred[i, j]
        n_eff[i] = st["W"]

        # ---- update ----
        W_new = lam * st["W"] + w[i]
        for j in range(m):
            yij = Y[i, j]
            if not np.isnan(yij):
                Wj_new = lam * st["Wj"][j] + w[i]
                if Wj_new > 0.0:
                    keep = lam * st["Wj"][j]
                    st["S"][j] = (keep * st["S"][j] + w[i] * np.outer(xi, xi)) / Wj_new
                    st["r"][:, j] = (keep * st["r"][:, j] + w[i] * xi * yij) / Wj_new
                st["Wj"][j] = Wj_new
                if not np.isnan(p_own[j]):
                    r_own = yij - p_own[j]
                    Ws_new = lam * st["Wsig"][j] + w[i]
                    st["sig2"][j] = (lam * st["Wsig"][j] * st["sig2"][j] + w[i] * r_own**2) / Ws_new
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
                    st["S"][j],
                    st["r"][:, j],
                    st["Wj"][j],
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
    max_dclock: float = np.inf,
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
        d = min(dclock[i] + st["pending"], max_dclock)
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


def _enet_descent(
    C: np.ndarray, c: np.ndarray, l1: float, l2: float, tol: float, max_sweeps: int = 10_000
) -> np.ndarray:
    """Minimise ``1/2 b'Cb - c'b + l1 |b|_1 + l2/2 |b|^2`` by cyclic coordinate
    descent (Friedman, Hastie & Tibshirani 2010), from zero, until no
    coefficient moves by ``tol``.

    Coordinate ``i`` with the rest held has the minimiser
    ``soft(c_i - sum_{j != i} C_ij b_j, l1) / (C_ii + l2)``, with
    ``soft(v, t) = sign(v) * max(|v| - t, 0)``. The caller has checked that
    the problem is well conditioned, so a descent that has not settled is a
    failure.
    """
    b = np.zeros(len(c))
    for _ in range(max_sweeps):
        moved = 0.0
        for i in range(len(c)):
            rho = c[i] - C[i] @ b + C[i, i] * b[i]
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / (C[i, i] + l2)
            moved = max(moved, abs(new - b[i]))
            b[i] = new
        if moved < tol:
            return b
    raise RuntimeError(f"coordinate descent did not settle to {tol} in {max_sweeps} sweeps")


def lasso_ref(
    X: np.ndarray,
    y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    lasso_path: list[float],
    l1_ratio: float = 1.0,
    halflife: float = np.inf,
    min_periods: float | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    max_dclock: float = np.inf,
    tol: float = 1e-14,
) -> dict[str, np.ndarray]:
    """Lasso / elastic-net path oracle (docs/PLAN.md section 4.3), one target
    with no nulls, written from the objective rather than ported.

    Each solve fits every path point ``l`` to the EW statistics of the rows
    learned so far. In standardized form it minimises::

        1/2 b'Cb - c'b + l1 |b|_1 + l2/2 |b|^2      l1 = l * l1_ratio, l2 = l * (1 - l1_ratio)

    ``C`` is the EW correlation matrix of the features and
    ``c_i = (E[x_i y] - m_i ybar) / s_i``: the target is centred but not
    scaled. ``s_i`` is the EW standard deviation of feature ``i``. Then
    ``coef_i = b_i / s_i`` and the intercept is ``ybar - m . coef``. ``l = 0``
    is the unpenalized EW least squares, with no ridge. The library drops a
    feature whose centred variance is exactly zero (``variance_is_usable``,
    since T-E9), from centred accumulators. This reference takes the variance
    from raw moments, where an exactly constant feature leaves rounding noise,
    so it drops one at 1e-10 of its raw second moment, which no feature in the
    tests comes near.

    :func:`_enet_descent` solves each problem from zero to ``tol``. So the
    library's warm start, along the path and from one solve to the next,
    cannot change the answer it is held to. A problem is held only where the
    smallest eigenvalue ``mu`` of ``C + l2 I`` is at least 1e-2. Below that
    it has no unique minimiser (fewer rows than features), or barely one, and
    a descent from zero takes about ``13 / mu`` sweeps to settle (measured).
    Such problems are the first solves of a stream, before ``min_periods``
    lets a row be scored. Their coefficients are NaN here, and a row scored
    with one raises.

    The schedule is ``ewridge``'s, which ``lasso`` takes:

    - a row is scored with the coefficients of the last solve before it, and
      only once ``n_eff``, the weight before the row, reaches ``min_periods``.
      Before the first solve nothing is scored;
    - after the row is learned, a solve runs when the clock since the last one
      has reached ``solve_every`` (the capped step counts, so a gap of at
      least ``solve_every`` forces one). One also runs when
      ``max_rows_between_solves`` rows have gone by since it (a zero-weight
      row is a row), or when there has been none yet and the weight has
      reached ``min_periods``;
    - ``solve_every`` defaults to ``halflife / 50``, which is every row for an
      infinite halflife; ``max_rows_between_solves`` is off by default.

    Feature null => the row is skipped. Nothing is scored or learned and it
    is not a row for the schedule. Its clock step folds into the next accepted
    row's, which is capped at ``max_dclock``.

    Returns ``pred`` (n, P) and ``n_eff`` (n,). ``coef`` (n, P, k+1) holds
    each row's last-solve coefficients. It is NaN before the first solve, on a
    skipped row, and for a problem the reference does not hold. ``solved``
    (n,) marks the rows a solve ran after.
    """
    n, k = X.shape
    kt = k + 1
    npath = len(lasso_path)
    if np.isnan(y).any():
        raise ValueError("lasso_ref takes a target with no nulls")
    if min_periods is None:
        min_periods = float(kt)
    if solve_every is None:
        solve_every = halflife / 50.0 if np.isfinite(halflife) else 0.0
    max_rows = np.inf if max_rows_between_solves is None else max_rows_between_solves

    pred = np.full((n, npath), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, npath, kt), np.nan)
    solved = np.zeros(n, dtype=bool)

    def solve(mean: np.ndarray, raw: np.ndarray, ry: np.ndarray) -> np.ndarray:
        cov = raw[1:, 1:] - np.outer(mean[1:], mean[1:])
        cxy = ry[1:] - mean[1:] * ry[0]
        var = np.diag(cov)
        kept = np.flatnonzero(var > 1e-10 * np.abs(np.diag(raw)[1:]))
        s = np.sqrt(var[kept])
        C = cov[np.ix_(kept, kept)] / np.outer(s, s)
        c = cxy[kept] / s
        floor = np.linalg.eigvalsh(C)[0] if kept.size else np.inf
        fit = np.zeros((npath, kt))
        for p, lam in enumerate(lasso_path):
            l1, l2 = lam * l1_ratio, lam * (1.0 - l1_ratio)
            if floor + l2 < 1e-2:
                fit[p] = np.nan
                continue
            fit[p, 1 + kept] = _enet_descent(C, c, l1, l2, tol) / s
            fit[p, 0] = ry[0] - mean[1:] @ fit[p, 1:]
        return fit

    # EW means of z = [1, x]: `mean`, `raw` = E[z z'], `ry` = E[z y].
    w_sum = 0.0
    mean, raw, ry = np.zeros(kt), np.zeros((kt, kt)), np.zeros(kt)
    fit = None
    since_clock, since_rows, pending = 0.0, 0, 0.0
    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        z = np.concatenate(([1.0], X[i]))
        d = min(dclock[i] + pending, max_dclock)
        pending = 0.0
        lam = 0.5 ** (d / halflife)

        # ---- score, from the state before the row ----
        n_eff[i] = w_sum
        if fit is not None and w_sum >= min_periods:
            if np.isnan(fit).any():
                raise ValueError(f"row {i} is scored with a fit the reference does not hold")
            pred[i] = fit @ z

        # ---- learn ----
        w_new = lam * w_sum + w[i]
        if w_new > 0.0:  # a zero-weight first row is 0/0 (hard rule 9)
            a, b = lam * w_sum / w_new, w[i] / w_new
            mean = a * mean + b * z
            raw = a * raw + b * np.outer(z, z)
            ry = a * ry + b * z * y[i]
        w_sum = w_new

        # ---- solve, on the schedule ----
        since_clock += d
        since_rows += 1
        if (
            solve_every <= 0.0
            or since_clock >= solve_every
            or since_rows >= max_rows
            or (fit is None and w_sum >= min_periods)
        ):
            fit = solve(mean, raw, ry)
            solved[i] = True
            since_clock, since_rows = 0.0, 0
        if fit is not None:
            coef[i] = fit

    return {"pred": pred, "n_eff": n_eff, "coef": coef, "solved": solved}


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
    revert_halflife: float | list[float] = float("inf"),
    standardize: bool = True,
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """Kalman / random-walk-beta oracle (docs/PLAN.md section 4.4).

    Mirrors the core exactly, including the details that make it match:

    - the transition ``b <- Phi b``, ``P <- Phi P Phi`` with
      ``Phi = diag(2^(-d / r_i))`` from ``revert_halflife`` runs first, before
      the prediction, on every accepted row (a null target or a zero weight
      still advances the clock); ``inf`` is ``Phi = I`` (E41);
    - features are standardized with the EW stats *before* the row's update,
      using scale 1 for the intercept slot and for near-zero-variance features
      (centered variance <= 1e-10 * raw second moment);
    - ``P += Q * d_clock`` happens after the transition and before the gain,
      once per shared P;
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
    rh = np.asarray(
        [revert_halflife] * kt if np.isscalar(revert_halflife) else revert_halflife, dtype=float
    )
    if rh.size == 1:
        rh = np.repeat(rh, kt)
    reverts = bool(np.isfinite(rh).any())

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
        d = min(dclock[i] + st["pending"], max_dclock)
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife)

        # transition first: the clock moved by d since the last row
        if reverts:
            phi = np.where(np.isinf(rh), 1.0, np.exp2(-(d / rh)))
            st["beta"] = st["beta"] * phi
            st["P"] = [P * np.outer(phi, phi) for P in st["P"]]

        # scales from the stats BEFORE this row (all ones, and no
        # centering, with `standardize` off: the state is the coefficient)
        scales = np.ones(kt)
        zs = z.copy()
        if standardize and off == 0:
            # No intercept: nothing to centre on, so each feature is scaled by
            # its raw second moment and not shifted. Centring here put a hidden
            # intercept into every prediction that `coef` had no slot for; the
            # core stopped doing it (review 2026-09-12, C10) and this oracle,
            # which had copied it, follows.
            for j in range(kt):
                raw = st["raw"][j, j]
                scales[j] = np.sqrt(raw) if raw > 0.0 else 1.0
                zs[j] = z[j] / scales[j]
        elif standardize:
            for j in range(off, kt):
                var = st["raw"][j, j] - st["mean"][j] ** 2
                raw = max(abs(st["raw"][j, j]), 1e-300)
                scales[j] = np.sqrt(var) if var > 1e-10 * raw else 1.0
            for j in range(off, kt):
                zs[j] = (z[j] - st["mean"][j]) / scales[j]

        n_eff[i] = st["W"]
        ready = st["W"] >= min_periods
        # The model's own prediction, which its own statistics fold; what is
        # emitted waits, besides, for each target's own weight (review
        # 2026-09-12, S2) -- a gate on the output, not on the model.
        p_own = np.full(m, np.nan)
        if ready:
            for j in range(m):
                if st["wj"][j] > 0.0:
                    p_own[j] = zs @ st["beta"][j]
                    if st["wj"][j] >= min_periods:
                        pred[i, j] = p_own[j]
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
            if np.isnan(Y[i, j]) or w[i] <= 0.0:
                # A null target and a zero weight alike: no update, and time
                # passes for both weights (review 2026-09-12, S9).
                st["wj"][j] *= lam
                st["wsig"][j] *= lam
                continue
            pz = st["P"][pi] @ zs
            s_inn = zs @ pz + sigma2 / w[i]
            if s_inn > 0.0:
                gain = pz / s_inn
                err = Y[i, j] - zs @ st["beta"][j]
                st["beta"][j] = st["beta"][j] + gain * err
                st["P"][pi] = st["P"][pi] - np.outer(gain, pz)
            if not np.isnan(p_own[j]):
                r = Y[i, j] - p_own[j]
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
            if not standardize:
                coef[i, j] = st["beta"][j]
                continue
            c = np.zeros(kt)
            c[off:] = st["beta"][j][off:] / scales[off:]
            if add_intercept:
                c[0] = st["beta"][j][0] - c[off:] @ st["mean"][off:]
            coef[i, j] = c

    return {"pred": pred, "resid": resid, "n_eff": n_eff, "coef": coef}


def _row_update(y, pred, scale, weight, present, aged, kt, loss, delta, tau, eps):
    """What a row does to a target's accumulators, mirroring `robust.rs`'s
    `row_update` (docs/PLAN.md section 4.5).

    Huber reweights the row, bounded by 1. The quantile loss takes one Newton
    step on the check loss smoothed by a uniform kernel of half-width
    ``eps * scale``, linearised at the fit the row was scored with: inside the
    band a least-squares row with target ``y + 2h(tau - 1/2)``, outside it
    ``2h * psi(r) * z`` into the cross-moment and no weight in the Gram at all.
    Under three rows per coefficient of the rows the target was present on
    (``present``) the fit is warming up and every row is an ordinary
    least-squares one (review 2026-09-12, N9), and past it the band is at
    least ``(kt / present) ** 0.4`` of ``scale`` (the second review's F3). A
    band holding under one row per coefficient (``aged``, its weight decayed
    to the row) is a fit the data has left behind, and takes least-squares
    rows until it holds rows again.

    Returns ``(kind, value, target)``: ``("fit", weight, target)`` or
    ``("nudge", nudge, None)``.
    """
    if loss == "huber":
        if np.isnan(pred):
            return "fit", weight, y
        cut = delta * scale
        a = abs(y - pred)
        return "fit", weight * (1.0 if (a <= cut or a == 0.0) else cut / a), y
    if np.isnan(pred) or present < 3.0 * kt or aged < kt:
        return "fit", weight, y
    h = scale * max(eps, (kt / present) ** 0.4)
    r = y - pred
    if abs(r) < h:
        return "fit", weight, y + 2.0 * h * (tau - 0.5)
    return "nudge", weight * 2.0 * h * (tau if r > 0.0 else tau - 1.0), None


def robust_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    reset: np.ndarray | None = None,
    halflife: float = 300.0,
    loss: str = "huber",
    huber_delta: float = 1.5,
    quantile: float = 0.5,
    quantile_eps: float = 0.2,
    ridge: float = 1e-6,
    standardize: bool = False,
    add_intercept: bool = True,
    min_periods: float | None = None,
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """Huber / quantile oracle (docs/PLAN.md section 4.5).

    Both losses read each row's *prior* residual, so both stay out-of-sample;
    what they do with it is :func:`_row_update`'s. Because the weights are per
    target, ``S`` is per target here (unlike ew_ridge, which shares one). Three
    details that matter for agreement:

    - the robust weight scales the accumulator update, but ``sigma2_j`` is
      updated with the *raw* row weight, so the scale estimate is not itself
      shrunk by the reweighting;
    - a row whose robust weight is zero still decays the accumulator;
    - a quantile row outside the band decays the accumulators and adds its
      nudge to the cross-moment, which is a mean, so the nudge enters divided
      by the target's decayed weight (review 2026-09-12, N9).
    """
    n, k = X.shape
    m = Y.shape[1]
    off = 1 if add_intercept else 0
    kt = k + off
    if min_periods is None:
        min_periods = float(kt)
    if reset is None:
        reset = np.zeros(n, dtype=bool)

    pred = np.full((n, m), np.nan)
    resid = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)

    def init():
        return {
            "W": np.zeros(m),
            "mean": [np.zeros(kt) for _ in range(m)],
            # Centred co-moments and cross-moments, by the weighted Welford
            # recursion `EwCov::update` and `Robust::step` take: `C = E[(z −
            # m)(z − m)']`, `c = E[(z − m)(y − ȳ)]`, `ȳ` beside them. They were
            # the raw `E[z z']` and `E[z·y]`, which the solve centred by
            # subtraction and so lost `level²·ε` (review 2026-09-18, S2).
            "C": [np.zeros((kt, kt)) for _ in range(m)],
            "c": [np.zeros(kt) for _ in range(m)],
            "ybar": np.zeros(m),
            "wj": np.zeros(m),
            # The rows each target was present on, at their raw weights: the
            # per-target gate's number (the second review's F1).
            "wobs": np.zeros(m),
            "sig2": np.zeros(m),
            "wsig": np.zeros(m),
            # EW count of observations under the RAW row weights: the
            # accumulators are scaled by the IRLS weights, the count is not.
            "w_raw": 0.0,
            "beta": None,
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
        d = min(dclock[i] + st["pending"], max_dclock)
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife)

        n_eff[i] = st["w_raw"]
        ready = st["w_raw"] >= min_periods and st["beta"] is not None
        # The model's own prediction, which its own statistics fold; what is
        # emitted waits, besides, for each target's own weight (review
        # 2026-09-12, S2) -- a gate on the output, not on the model.
        p_own = np.full(m, np.nan)
        if ready:
            for j in range(m):
                if st["wj"][j] > 0.0:
                    p_own[j] = z @ st["beta"][j]
                    if st["wobs"][j] >= min_periods:
                        pred[i, j] = p_own[j]
                        if not np.isnan(Y[i, j]):
                            resid[i, j] = Y[i, j] - pred[i, j]

        for j in range(m):
            present = lam * st["wobs"][j]
            if np.isnan(Y[i, j]):
                st["W"][j] *= lam
                st["wj"][j] *= lam
                st["wobs"][j] = present
                st["wsig"][j] *= lam
                continue
            st["wobs"][j] = present + (w[i] if w[i] > 0.0 else 0.0)
            sigma = np.sqrt(max(st["sig2"][j], 0.0))
            scale = sigma if sigma > 0.0 else 1.0
            aged_w, aged_wj = lam * st["W"][j], lam * st["wj"][j]
            kind, value, target = _row_update(
                Y[i, j],
                p_own[j],
                scale,
                w[i],
                present,
                aged_wj,
                kt,
                loss,
                huber_delta,
                quantile,
                quantile_eps,
            )
            if kind == "nudge":
                st["W"][j] = aged_w
                st["wj"][j] = aged_wj
                if aged_wj > 0.0:
                    # A sum's worth of nudge over the band's weight, centred:
                    # the raw `r += nudge·z/wj` is `ȳ += nudge/wj` and
                    # `c += nudge·(z − m)/wj`, exactly.
                    step = value / aged_wj
                    st["c"][j] = st["c"][j] + step * (z - st["mean"][j])
                    st["ybar"][j] += step
            else:
                ww = value
                if ww <= 0.0:
                    st["W"][j] = aged_w
                    st["wj"][j] = aged_wj
                    continue
                W_new = aged_w + ww
                a, b = aged_w / W_new, ww / W_new
                # Deviations from the means before the row.
                d = z - st["mean"][j]
                dy = target - st["ybar"][j]
                st["C"][j] = a * st["C"][j] + a * b * np.outer(d, d)
                st["c"][j] = a * st["c"][j] + a * b * d * dy
                st["mean"][j] = st["mean"][j] + b * d
                st["ybar"][j] += b * dy
                st["W"][j] = W_new
                st["wj"][j] = aged_wj + ww

            if not np.isnan(p_own[j]):
                rr = Y[i, j] - p_own[j]
                ws_new = lam * st["wsig"][j] + w[i]
                st["sig2"][j] = (lam * st["wsig"][j] * st["sig2"][j] + w[i] * rr * rr) / ws_new
                st["wsig"][j] = ws_new

        st["w_raw"] = lam * st["w_raw"] + w[i]

        beta = np.zeros((m, kt))
        for j in range(m):
            if st["wj"][j] > 0.0:
                beta[j] = _solve_centred(
                    st["mean"][j],
                    st["C"][j],
                    st["c"][j],
                    st["ybar"][j],
                    ridge,
                    add_intercept,
                    standardize,
                )
        st["beta"] = beta
        coef[i] = beta

    return {"pred": pred, "resid": resid, "n_eff": n_eff, "coef": coef}


def _solve_centred(
    mean: np.ndarray,
    C: np.ndarray,
    c: np.ndarray,
    ybar: float,
    ridge: float,
    add_intercept: bool,
    standardize: bool,
) -> np.ndarray:
    """`Robust::solve` on centred moments (review 2026-09-18, S2).

    With an intercept: the slopes from the centred system, `(C_ff/(s s') +
    ridge I) b = c_f/s`, `s` the feature standard deviations when
    standardized and 1 otherwise, then `beta_0 = ȳ − m_f · beta_f`. Through
    the origin nothing can absorb a level, so the raw system is the fit
    asked for: `E[z z'] = C + m m'` and `E[z·y] = c + m·ȳ`, scaled by the
    raw second-moment diagonals when standardized."""
    kt = len(mean)
    if add_intercept:
        Cf, cf, mf = C[1:, 1:], c[1:], mean[1:]
        s = np.sqrt(np.maximum(np.diag(Cf), 0.0)) if standardize else np.ones(kt - 1)
        keep = s > 1e-12
        beta = np.zeros(kt)
        if keep.any():
            A = Cf[np.ix_(keep, keep)] / np.outer(s[keep], s[keep]) + ridge * np.eye(keep.sum())
            beta[1:][keep] = np.linalg.solve(A, cf[keep] / s[keep]) / s[keep]
        beta[0] = ybar - mf @ beta[1:]
        return beta
    raw = C + np.outer(mean, mean)
    r = c + mean * ybar
    s = np.sqrt(np.maximum(np.diag(raw), 0.0)) if standardize else np.ones(kt)
    keep = s > 0.0
    beta = np.zeros(kt)
    if keep.any():
        A = raw[np.ix_(keep, keep)] / np.outer(s[keep], s[keep]) + ridge * np.eye(keep.sum())
        beta[keep] = np.linalg.solve(A, r[keep] / s[keep]) / s[keep]
    return beta


def ftrl_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    reset: np.ndarray | None = None,
    halflife: float = float("inf"),
    alpha: float = 0.1,
    beta: float = 1.0,
    l1: float = 0.0,
    l2: float = 1.0,
    add_intercept: bool = True,
    min_periods: float = 10.0,
    strict_binary: bool = False,
    loss: str = "logistic",
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """FTRL-proximal oracle (docs/PLAN.md section 4.6, McMahan 2013), for the
    logistic loss and the squared one (E18).

    The decay is applied to ``n``, ``z`` and the proximal sum ``d`` *before*
    the proximal weights are computed, so a row's prediction already reflects
    its own elapsed clock. Without decay the rate is river's ``(beta +
    sqrt(n)) / alpha``, which the proximal steps telescope to; under decay it
    is ``beta / alpha + d``, their own discounted sum. Decaying ``n`` inside
    the square root instead shrank every coefficient toward zero on every
    row, by a factor between ``lam`` and ``sqrt(lam)`` (review 2026-09-12,
    C24).
    """
    n, k = X.shape
    m = Y.shape[1]
    off = 1 if add_intercept else 0
    kt = k + off
    if reset is None:
        reset = np.zeros(n, dtype=bool)
    forgets = np.isfinite(halflife)

    pred = np.full((n, m), np.nan)
    resid = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)

    def init():
        return {
            "n": np.zeros((m, kt)),
            "z": np.zeros((m, kt)),
            "d": np.zeros((m, kt)),
            "w_sum": 0.0,
            "pending": 0.0,
        }

    def weights(st, j):
        out = np.zeros(kt)
        for i in range(kt):
            zi = st["z"][j, i]
            if abs(zi) > l1:
                if forgets:
                    rate = beta / alpha + st["d"][j, i]
                else:
                    rate = (beta + np.sqrt(st["n"][j, i])) / alpha
                out[i] = -(zi - np.sign(zi) * l1) / (rate + l2)
        return out

    st = init()
    for i in range(n):
        if reset[i]:
            st = init()
        if np.isnan(X[i]).any():
            st["pending"] += dclock[i]
            continue
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i].copy()
        d = min(dclock[i] + st["pending"], max_dclock)
        st["pending"] = 0.0
        lam = 0.5 ** (d / halflife) if forgets else 1.0

        if lam != 1.0:
            st["n"] *= lam
            st["z"] *= lam
            st["d"] *= lam

        n_eff[i] = st["w_sum"]
        ready = st["w_sum"] >= min_periods

        for j in range(m):
            b = weights(st, j)
            coef[i, j] = b
            p = z @ b if loss == "squared" else 1.0 / (1.0 + np.exp(-(z @ b)))
            if ready:
                pred[i, j] = p
                if not np.isnan(Y[i, j]):
                    resid[i, j] = Y[i, j] - p
            if np.isnan(Y[i, j]) or w[i] <= 0.0:
                continue
            yb = Y[i, j]
            if loss == "logistic":
                if strict_binary:
                    if yb not in (0.0, 1.0):
                        continue
                else:
                    yb = min(max(yb, 0.0), 1.0)
            err = p - yb
            for ii in range(kt):
                g = err * z[ii] * w[i]
                n_new = st["n"][j, ii] + g * g
                s = (np.sqrt(n_new) - np.sqrt(st["n"][j, ii])) / alpha
                st["z"][j, ii] += g - s * b[ii]
                st["n"][j, ii] = n_new
                st["d"][j, ii] += s
        st["w_sum"] = lam * st["w_sum"] + w[i]

    return {"pred": pred, "resid": resid, "n_eff": n_eff, "coef": coef}


def seqtest_ref(
    Y: np.ndarray,
    w: np.ndarray | None = None,
    reset: np.ndarray | None = None,
    min_periods: float = 0.0,
) -> dict[str, np.ndarray]:
    """Sequential sign test oracle (docs/ENHANCEMENTS.md E42): per target,
    two Kelly bettors with a Krichevsky-Trofimov stake on the sign counts.

    ``Y`` is ``(n, m)`` with NaN for a null. Per row and target, with the
    counts *before* the row and ``s`` the sign of ``Y[i, j]``::

        lam_pos = max(0, (n_pos - n_neg) / (n_pos + n_neg + 1))
        lam_neg = max(0, (n_neg - n_pos) / (n_pos + n_neg + 1))
        log_e_pos += log1p(lam_pos * s);  log_e_neg += log1p(-lam_neg * s)

    A null, zero (or NaN) value bets nothing and counts nothing; a weight of
    0 skips the row; any other weight is one trial and counts itself toward
    ``n_eff``. No decay, so no clock. ``reset[i]`` restarts the state before
    row ``i`` (a session change under ``session_gap="reset"``, a backwards
    clock under ``on_clock_reset="reset_state"``). Outputs are the state
    before the row, NaN while ``n_eff < min_periods`` (``n_eff`` always).
    """
    import math

    n, m = Y.shape
    if w is None:
        w = np.ones(n)
    if reset is None:
        reset = np.zeros(n, dtype=bool)
    log_e_pos = np.full((n, m), np.nan)
    log_e_neg = np.full((n, m), np.nan)
    n_pos = np.full((n, m), np.nan)
    n_neg = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)

    def init():
        return {
            "pos": np.zeros(m),
            "neg": np.zeros(m),
            "lp": np.zeros(m),
            "ln": np.zeros(m),
            "w": 0.0,
        }

    st = init()
    for i in range(n):
        if reset[i]:
            st = init()
        n_eff[i] = st["w"]
        if st["w"] >= min_periods:
            log_e_pos[i] = st["lp"]
            log_e_neg[i] = st["ln"]
            n_pos[i] = st["pos"]
            n_neg[i] = st["neg"]
        if not w[i] > 0.0:
            continue
        for j in range(m):
            y = Y[i, j]
            if np.isnan(y) or y == 0.0:
                continue
            s = 1.0 if y > 0.0 else -1.0
            n1 = st["pos"][j] + st["neg"][j] + 1.0
            lam_pos = max(0.0, (st["pos"][j] - st["neg"][j]) / n1)
            lam_neg = max(0.0, (st["neg"][j] - st["pos"][j]) / n1)
            st["lp"][j] += math.log1p(lam_pos * s)
            st["ln"][j] += math.log1p(-lam_neg * s)
            if s > 0.0:
                st["pos"][j] += 1.0
            else:
                st["neg"][j] += 1.0
        st["w"] += w[i]
    return {
        "log_e_pos": log_e_pos,
        "log_e_neg": log_e_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_eff": n_eff,
    }
