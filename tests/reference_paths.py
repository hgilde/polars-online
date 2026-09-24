"""Reference (oracle) implementations for model paths `tests/reference.py`
does not reach, written from the documented objective and semantics rather
than ported from the core.

The models whose state is a sum -- :func:`lasso_paths_ref`,
:func:`ewridge_paths_ref`, :func:`rls_paths_ref` -- are recomputed **from the
raw rows** at the moment each statistic is needed, where
`tests/reference.py` runs each model's own recursion longhand: each past row
at its weight ``w * 0.5 ** (age / halflife)``, the age read on the capped
clock the decay uses. That makes them independent of the mean-form
recursion as well as of the solver, and it is what lets a ``window`` be
written from its definition -- the rows whose age is at most ``window``,
nothing else -- instead of from the snapshot subtraction the core uses to get
there. The models that keep no sums -- :func:`pa_ref`, :func:`sgd_ref`,
:func:`holt_ref` -- are their builders' update equations, written out.

Conventions shared with the core (docs/PLAN.md sections 3 and 13, CLAUDE.md
hard rules 2, 8 and 9):

- A row with a null feature is skipped: every output NaN, nothing learned,
  and its clock step folds into the next accepted row's, which
  ``max_dclock`` caps.
- ``n_eff`` is the weight before the row's update and before its own decay,
  so its ages are counted from the last accepted row; under a ``window`` it
  is the weight inside the window, as seen from there.
- A zero-weight row advances the clock and teaches nothing.
"""

from __future__ import annotations

import numpy as np

from reference import _enet_descent


def _weighted_mean(v: np.ndarray, om: np.ndarray) -> np.ndarray:
    return (om[:, None] * v).sum(axis=0) / om.sum()


def lasso_paths_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    lasso_path: list[float],
    *,
    l1_ratio: float = 1.0,
    halflife: float = np.inf,
    select_halflife: float | None = None,
    min_periods: float | list[float] | None = None,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    max_dclock: float = np.inf,
    window: float | None = None,
    add_intercept: bool = True,
    target_gaps: str = "own_rows",
    tol: float = 1e-14,
) -> dict[str, np.ndarray]:
    """The ``lasso`` builder's path over several targets, null targets,
    ``target_gaps``, ``window``, ``add_intercept=False`` and
    ``lam_selected`` (docs/PLAN.md section 4.3 and task 81, the ``lasso`` and
    ``ewridge`` docstrings).

    **The fit.** Each solve fits every path point ``l`` of every target
    ``j`` from the rows learned so far, each at ``w * 0.5 ** (age /
    halflife)`` with the age counted from the solve's row, minimising::

        1/2 b'Cb - c'b + l1 |b|_1 + l2/2 |b|^2      l1 = l * l1_ratio, l2 = l * (1 - l1_ratio)

    by :func:`reference._enet_descent`, from zero to ``tol``. Which rows:

    - ``"own_rows"``: ``C``, ``c``, the means and the scales over the rows the
      target is present on -- the fit of the frame with its nulls dropped;
    - ``"pairwise"``: ``C`` and the scales over every row; ``c`` over the
      target's rows, centred at the target's own means (pandas'
      pairwise-complete covariance). The intercept reads the target's own
      means either way;
    - ``window``: of those, only the rows whose age is at most ``window``.

    With an intercept, ``C`` is the correlation matrix of the centred
    features, ``c_i = cov(x_i, y) / s_i``, ``coef_i = b_i / s_i`` and the
    intercept is ``ybar - m . coef``; a feature whose variance is zero is
    dropped with coefficient 0 (the ``ewridge`` docstring; the ``1e-10``
    threshold ``lasso_ref`` cites went with T-E9). Without one nothing is
    centred: ``C = E[x x'] / (r r')``, ``c_i = E[x_i y] / r_i`` with ``r_i``
    the root mean square, ``coef_i = b_i / r_i``, and a column that is all
    zero is dropped.

    A problem is held only where the smallest eigenvalue of ``C + l2 I`` is
    at least 1e-2, and a target with no weight has no problem to hold; either
    is NaN here, and a row scored with one raises (``lasso_ref``'s rule).
    Under a ``window`` a problem that drops a feature is not held either. The
    core rebuilds a window's Gram by subtracting the state at its boundary,
    so a variance that is zero inside the window -- one row left, say -- comes
    back as rounding noise of about 1e-14. The feature is then kept and the
    zero-penalty slope is noise over noise, where the documented rule drops
    it (found writing this reference, 2026-09-24; not held here, reported).

    **Scoring.** Row ``i`` is scored with the last solve before it, target
    ``j`` only once its own weight -- the rows it is present on, at their raw
    weights, decayed, inside the window -- reaches its ``min_periods``
    (scalar or one per target). ``n_eff`` is the weight of every row.

    **The schedule** is ``lasso_ref``'s: after the row is learned a solve
    runs when the clock since the last one reaches ``solve_every`` (default
    ``halflife / 50``, every row for an infinite halflife), when
    ``max_rows_between_solves`` rows have gone by, or when there has been
    none yet and ``n_eff`` has reached ``min_periods``. That last rule is
    written for one target and one threshold; with several the reference
    raises if it is ever the one that decides, so a test must let the cadence
    solve first.

    **``lam_selected``.** Per target, the path point with the least weighted
    sum of squared out-of-sample errors so far, each at ``w * 0.5 ** (age /
    select_halflife)`` (default the halflife) and, under a window, inside it
    -- as it stood before the row. The errors are the model's own
    predictions' wherever the target is present: the model predicts once
    ``n_eff`` reaches the smallest threshold (docs/ENHANCEMENTS.md E7) and a
    target once it has any weight, and what the target's own threshold
    withholds is the output, not the model (review 2026-09-12, S2, as
    ``ewridge_ref``'s ``sigma2`` folds it). NaN where there is no error yet,
    where the best two are within 1e-9 of each other (a tie is not something
    to hold a library to), and for good once an error came from a fit the
    reference does not hold.

    Returns ``pred`` (n, m, P), ``n_eff`` (n,), ``w_target`` (n, m), each
    target's weight before the row, ``coef`` (n, m, P, kt), the last solve's
    fit, NaN before the first and on skipped rows, ``lam_selected`` (n, m)
    and ``solved`` (n,).
    """
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if add_intercept else k
    npath = len(lasso_path)
    path = np.asarray(lasso_path, dtype=float)
    if target_gaps not in ("own_rows", "pairwise"):
        raise ValueError(target_gaps)
    if min_periods is None:
        min_periods = float(kt)
    mp = np.broadcast_to(np.asarray(min_periods, dtype=float), (m,))
    if solve_every is None:
        solve_every = halflife / 50.0 if np.isfinite(halflife) else 0.0
    if select_halflife is None:
        select_halflife = halflife
    max_rows = np.inf if max_rows_between_solves is None else max_rows_between_solves
    horizon = np.inf if window is None else window
    single = m == 1 and np.ndim(min_periods) == 0

    pred = np.full((n, m, npath), np.nan)
    n_eff = np.full(n, np.nan)
    w_target = np.full((n, m), np.nan)
    coef = np.full((n, m, npath, kt), np.nan)
    lam_selected = np.full((n, m), np.nan)
    solved = np.zeros(n, dtype=bool)

    def decay(ages: np.ndarray, h: float) -> np.ndarray:
        return np.ones_like(ages) if np.isinf(h) else 0.5 ** (ages / h)

    def solve_one(xg, omg, xo, yo, omo) -> np.ndarray:
        """One target's path: the Gram's rows ``xg`` at ``omg``, the target's
        own rows ``xo``, ``yo`` at ``omo``."""
        fit = np.full((npath, kt), np.nan)
        if omo.sum() <= 0.0:
            return fit
        if add_intercept:
            mg = _weighted_mean(xg, omg)
            cov = (omg[:, None, None] * np.einsum("ri,rj->rij", xg - mg, xg - mg)).sum(
                axis=0
            ) / omg.sum()
            mo = _weighted_mean(xo, omo)
            ybar = float(omo @ yo / omo.sum())
            cxy = ((omo * (yo - ybar))[:, None] * (xo - mo)).sum(axis=0) / omo.sum()
            var = np.diag(cov)
            kept = np.flatnonzero(var > 0.0)
            s = np.sqrt(var[kept])
            C = cov[np.ix_(kept, kept)] / np.outer(s, s)
            c = cxy[kept] / s
        else:
            raw = (omg[:, None, None] * np.einsum("ri,rj->rij", xg, xg)).sum(axis=0) / omg.sum()
            rxy = ((omo * yo)[:, None] * xo).sum(axis=0) / omo.sum()
            r = np.sqrt(np.diag(raw))
            kept = np.flatnonzero(r > 0.0)
            s = r[kept]
            C = raw[np.ix_(kept, kept)] / np.outer(s, s)
            c = rxy[kept] / s
        if window is not None and kept.size < k:
            return fit
        floor = np.linalg.eigvalsh(C)[0] if kept.size else np.inf
        off = 1 if add_intercept else 0
        for p, lam in enumerate(path):
            l1, l2 = lam * l1_ratio, lam * (1.0 - l1_ratio)
            if floor + l2 < 1e-2:
                continue
            b = np.zeros(k)
            b[kept] = _enet_descent(C, c, l1, l2, tol) / s
            fit[p, off:] = b
            if add_intercept:
                fit[p, 0] = ybar - mo @ b
        return fit

    def solve(t_now: float) -> np.ndarray:
        ta, wa, xa, ya = (np.asarray(v) for v in (T, Wr, Xr, Yr))
        ages = t_now - ta
        inside = ages <= horizon
        om = wa * decay(ages, halflife)
        fit = np.full((m, npath, kt), np.nan)
        for j in range(m):
            own = inside & ~np.isnan(ya[:, j])
            gram = own if target_gaps == "own_rows" else inside
            fit[j] = solve_one(xa[gram], om[gram], xa[own], ya[own, j], om[own])
        return fit

    # The rows learned so far, and the errors each target has been scored by.
    T: list[float] = []
    Wr: list[float] = []
    Xr: list[np.ndarray] = []
    Yr: list[np.ndarray] = []
    errs: list[list[tuple[float, float, np.ndarray]]] = [[] for _ in range(m)]
    unheld = np.zeros(m, dtype=bool)

    fit = None
    t_last, pending = 0.0, 0.0
    since_clock, since_rows = 0.0, 0
    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        d = min(dclock[i] + pending, max_dclock)
        pending = 0.0
        t_now = (t_last + d) if T else 0.0
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i]

        # ---- before the row: every weight is seen from the last accepted row ----
        if T:
            ages = t_last - np.asarray(T)
            inside = ages <= horizon
            om = np.where(inside, np.asarray(Wr) * decay(ages, halflife), 0.0)
            present = ~np.isnan(np.asarray(Yr))
            n_eff[i] = om.sum()
            w_target[i] = om @ present
        else:
            n_eff[i] = 0.0
            w_target[i] = 0.0
        own = np.full((m, npath), np.nan)
        for j in range(m):
            if fit is not None and w_target[i, j] > 0.0 and n_eff[i] >= float(np.min(mp)):
                own[j] = fit[j] @ z
                if w_target[i, j] >= mp[j]:
                    if np.isnan(own[j]).any():
                        raise ValueError(
                            f"row {i}, target {j} is scored with a fit the reference does not hold"
                        )
                    pred[i, j] = own[j]
            if errs[j] and not unheld[j]:
                te, we, ee = (np.asarray(v) for v in zip(*errs[j], strict=True))
                ages_e = t_last - te
                wt = np.where(ages_e <= horizon, we * decay(ages_e, select_halflife), 0.0)
                if wt.sum() > 0.0:
                    score = wt @ (ee**2)
                    lo, second = np.sort(score)[:2]
                    if second - lo > 1e-9 * second:
                        lam_selected[i, j] = path[int(np.argmin(score))]

        # ---- learn ----
        T.append(t_now)
        Wr.append(float(w[i]))
        Xr.append(X[i].copy())
        Yr.append(Y[i].copy())
        for j in range(m):
            if fit is None or np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            if w_target[i, j] > 0.0 and n_eff[i] >= float(np.min(mp)):
                if np.isnan(own[j]).any():
                    unheld[j] = True
                errs[j].append((t_now, float(w[i]), Y[i, j] - own[j]))
        t_last = t_now

        # ---- solve, on the schedule ----
        since_clock += d
        since_rows += 1
        cadence = solve_every <= 0.0 or since_clock >= solve_every or since_rows >= max_rows
        if not cadence and fit is None:
            ages = t_now - np.asarray(T)
            weight = (np.asarray(Wr) * decay(ages, halflife))[ages <= horizon].sum()
            if weight >= float(np.min(mp)):
                if not single:
                    raise ValueError(
                        f"row {i}: the forced first solve decides here, which the reference "
                        "defines for one target and one threshold only"
                    )
                cadence = True
        if cadence:
            fit = solve(t_now)
            solved[i] = True
            since_clock, since_rows = 0.0, 0
        if fit is not None:
            coef[i] = fit

    return {
        "pred": pred,
        "n_eff": n_eff,
        "w_target": w_target,
        "coef": coef,
        "lam_selected": lam_selected,
        "solved": solved,
    }


def _ridge_fit(xg, omg, xo, yo, omo, ridge, standardize, add_intercept, windowed):
    """One ridge problem, from the Gram's rows ``xg`` at ``omg`` and the
    target's own rows ``xo``, ``yo`` at ``omo`` (the ``ewridge`` docstring).

    With an intercept the slopes solve ``(C + ridge I) b = c``, ``C`` the
    centred feature covariance over the Gram's rows and ``c`` the target's
    cross-covariance over its own rows, centred at its own means, and the
    intercept is ``ybar - m . b`` at those means. ``standardize`` solves the
    same system in correlation form, ``(C / (s s') + ridge I) (s * b) = c /
    s``, and drops a feature whose variance is zero. Without an intercept
    nothing is centred: ``(E[x x'] + ridge I) b = E[x y]``, and
    ``standardize`` scales by each column's root mean square instead.

    Returns ``(intercept, slopes)``, or ``None`` where the reference does not
    hold the problem: a target with no weight, a feature set whose features
    are nearly collinear on these rows (the smallest eigenvalue of their
    correlation matrix below 1e-3), or, under a window, a dropped feature
    (``lasso_paths_ref`` says why)."""
    if omo.sum() <= 0.0 or omg.sum() <= 0.0:
        return None
    if add_intercept:
        mg = _weighted_mean(xg, omg)
        M = (omg[:, None, None] * np.einsum("ri,rj->rij", xg - mg, xg - mg)).sum(axis=0) / omg.sum()
        mo = _weighted_mean(xo, omo)
        ybar = float(omo @ yo / omo.sum())
        v = ((omo * (yo - ybar))[:, None] * (xo - mo)).sum(axis=0) / omo.sum()
    else:
        M = (omg[:, None, None] * np.einsum("ri,rj->rij", xg, xg)).sum(axis=0) / omg.sum()
        v = ((omo * yo)[:, None] * xo).sum(axis=0) / omo.sum()
    scale = np.sqrt(np.diag(M))
    kept = np.flatnonzero(scale > 0.0)
    if windowed and kept.size < len(scale):
        return None
    corr = M[np.ix_(kept, kept)] / np.outer(scale[kept], scale[kept])
    if kept.size and np.linalg.eigvalsh(corr)[0] < 1e-3:
        return None
    b = np.zeros(len(scale))
    if standardize:
        b[kept] = np.linalg.solve(corr + ridge * np.eye(kept.size), v[kept] / scale[kept])
        b[kept] /= scale[kept]
    else:
        b = np.linalg.solve(M + ridge * np.eye(len(scale)), v)
    return (ybar - mo @ b if add_intercept else 0.0), b


def ewridge_paths_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    *,
    halflife: float,
    ridge: float | list[float] = 1e-6,
    feature_sets: list[list[int]] | None = None,
    standardize: bool = False,
    add_intercept: bool = True,
    target_gaps: str = "own_rows",
    window: float | None = None,
    min_periods: float | list[float] = 0.0,
    solve_every: float | None = None,
    max_rows_between_solves: int | None = None,
    max_dclock: float = np.inf,
    session: np.ndarray | None = None,
    session_gap: float | str | None = None,
    session_shrink: float | None = None,
    long_halflife: float | None = None,
) -> dict[str, object]:
    """``ewridge`` on its documented schedule, over a ridge grid and feature
    sets, from the raw rows (the ``ewridge`` builder's docstring, docs/PLAN.md
    section 4.1 and task 81).

    Every past row of the current state enters a solve at its **effective
    weight**: ``w * 0.5 ** (age / halflife)``, with the age on the capped
    clock, inside the window under one. A ``session_shrink`` blend at ``f``
    replaces each row's weight by ``(1 - f)`` of it plus ``f`` of its weight
    in the slow twin at ``long_halflife``, as the rows stand before the first
    row of the new session; the mean-form sums are linear in these weights,
    so the blend is exact on them (``tests/test_second_opinion.py``
    ``TestSessionShrinkBlend`` pins that reading). The new session's first
    row is then scored from the blend, re-solved, and every row ages from
    there at ``halflife``. ``session_gap="reset"`` starts the state over.

    **Scoring.** Target ``j`` is scored with the last solve once its own
    weight (the rows it is present on) reaches its ``min_periods``, scalar or
    one per target. ``n_eff`` is the weight of every row. Both are seen from
    the last accepted row, before the row's own decay (hard rule 8).

    **The schedule** is ``lasso_paths_ref``'s: after the row is learned a
    solve runs when the clock since the last one reaches ``solve_every``
    (default ``halflife / 50``, every row for ``inf``), when
    ``max_rows_between_solves`` rows have gone by, or when there has been
    none yet and ``n_eff`` has reached the smallest ``min_periods`` -- which
    with ``ewridge``'s default of 0 is the first row.

    **The combinations** are every feature set (column indices; ``None`` is
    all of them) crossed with every ridge value, set-major. Each is solved by
    :func:`_ridge_fit` into a full-length coefficient vector, zero outside its
    set.

    Returns ``pred`` (n, m, C), ``n_eff`` (n,), ``coef`` (n, m, C, kt) --
    NaN before the first solve, on a skipped row, and for a problem the
    reference does not hold, which a scored row may not use -- ``solved``
    (n,), ``has_fit`` (n,), the rows after which the model has a solve (a
    reset takes it away), and ``combos``, the (feature set, ridge) of each
    C.
    """
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if add_intercept else k
    ridges = [ridge] if np.isscalar(ridge) else list(ridge)
    sets = [list(range(k))] if feature_sets is None else [list(s) for s in feature_sets]
    combos = [(s, r) for s in sets for r in ridges]
    mp = np.broadcast_to(np.asarray(min_periods, dtype=float), (m,))
    if solve_every is None:
        solve_every = halflife / 50.0 if np.isfinite(halflife) else 0.0
    max_rows = np.inf if max_rows_between_solves is None else max_rows_between_solves
    horizon = np.inf if window is None else window
    blend = session_shrink is not None
    if blend and long_halflife is None:
        raise ValueError("session_shrink needs long_halflife")

    pred = np.full((n, m, len(combos)), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, len(combos), kt), np.nan)
    solved = np.zeros(n, dtype=bool)
    has_fit = np.zeros(n, dtype=bool)

    def factor(d: float, h: float) -> float:
        return 1.0 if np.isinf(h) else 0.5 ** (d / h)

    def solve() -> np.ndarray:
        wa, xa, ya, ta = np.asarray(fast), np.asarray(Xr), np.asarray(Yr), np.asarray(T)
        inside = (t_last - ta) <= horizon
        fit = np.full((m, len(combos), kt), np.nan)
        for j in range(m):
            own = inside & ~np.isnan(ya[:, j])
            gram = own if target_gaps == "own_rows" else inside
            for c, (cols, r) in enumerate(combos):
                got = _ridge_fit(
                    xa[gram][:, cols],
                    wa[gram],
                    xa[own][:, cols],
                    ya[own, j],
                    wa[own],
                    r,
                    standardize,
                    add_intercept,
                    window is not None,
                )
                if got is None:
                    continue
                fit[j, c] = 0.0
                if add_intercept:
                    fit[j, c, 0] = got[0]
                fit[j, c, (1 if add_intercept else 0) + np.asarray(cols)] = got[1]
        return fit

    def restart():
        return [], [], [], [], [], None, 0.0, 0

    fast, slow, T, Xr, Yr, fit, since_clock, since_rows = restart()
    t_last, pending, prev_session, started = 0.0, 0.0, None, False
    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        changed = session is not None and started and session[i] != prev_session
        prev_session = None if session is None else session[i]
        if changed and session_gap == "reset":
            fast, slow, T, Xr, Yr, fit, since_clock, since_rows = restart()
            d = 0.0
        elif changed and session_gap is not None:
            d = min(float(session_gap), max_dclock)
        else:
            d = min(max(dclock[i], 0.0) + pending, max_dclock) if started else 0.0
        pending = 0.0
        started = True
        if changed and blend and fast:
            fast = list(
                (1.0 - session_shrink) * np.asarray(fast) + session_shrink * np.asarray(slow)
            )
            fit = solve()
            since_clock, since_rows = 0.0, 0
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i]

        # ---- before the row, seen from the last accepted row ----
        if fast:
            inside = (t_last - np.asarray(T)) <= horizon
            om = np.where(inside, np.asarray(fast), 0.0)
            n_eff[i] = om.sum()
            w_target = om @ ~np.isnan(np.asarray(Yr))
        else:
            n_eff[i], w_target = 0.0, np.zeros(m)
        for j in range(m):
            if fit is not None and w_target[j] > 0.0 and w_target[j] >= mp[j]:
                if np.isnan(fit[j]).any():
                    raise ValueError(f"row {i}, target {j} is scored with an unheld fit")
                pred[i, j] = fit[j] @ z

        # ---- learn: age every row by this row's step, then take it ----
        lam = factor(d, halflife)
        fast = [v * lam for v in fast] + [float(w[i])]
        if blend:
            lam_s = factor(d, long_halflife)
            slow = [v * lam_s for v in slow] + [float(w[i])]
        t_last = (t_last + d) if T else 0.0
        T.append(t_last)
        Xr.append(X[i].copy())
        Yr.append(Y[i].copy())

        # ---- solve, on the schedule ----
        since_clock += d
        since_rows += 1
        due = solve_every <= 0.0 or since_clock >= solve_every or since_rows >= max_rows
        if not due and fit is None:
            inside = (t_last - np.asarray(T)) <= horizon
            due = float(np.asarray(fast)[inside].sum()) >= float(np.min(mp))
        if due:
            fit = solve()
            solved[i] = True
            since_clock, since_rows = 0.0, 0
        if fit is not None:
            coef[i] = fit
            has_fit[i] = True

    return {
        "pred": pred,
        "n_eff": n_eff,
        "coef": coef,
        "solved": solved,
        "has_fit": has_fit,
        "combos": combos,
    }


def rls_paths_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    *,
    halflife: float,
    ridge: float = 1.0,
    coef_prior: np.ndarray | None = None,
    add_intercept: bool = True,
    min_periods: float | None = None,
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """``rls`` from its documented recursion (the ``rls`` builder's
    docstring), summed from the raw rows rather than rotated in::

        A <- lam * A + w * z z'        b_j <- lam * b_j + w * y_j * z        beta_j = A^-1 b_j
        A_0 = ridge * I                b_0 = ridge * coef_prior

    so after rows at ages ``a_r`` (the prior's age is the whole stream's),
    ``A = ridge * lam^T I + sum_r w_r lam^a_r z_r z_r'`` and ``b_j`` likewise.
    The factor ``R`` is shared, so a row with any null target is learned
    for none (it still ages the sums). Scoring: once ``n_eff``, the weight of
    every row, reaches ``min_periods`` (default the number of coefficients)
    and a row has been learned; the prior alone predicts nothing (review
    2026-09-12, S10). ``coef`` is ``beta`` after the row.

    Returns ``pred`` and ``coef`` per target, and ``n_eff``."""
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if add_intercept else k
    if min_periods is None:
        min_periods = float(kt)
    prior = np.zeros((m, kt)) if coef_prior is None else np.asarray(coef_prior, float)
    pred = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)

    t_now, pending, started = 0.0, 0.0, False
    T: list[float] = []
    Wr: list[float] = []
    learned: list[bool] = []
    Z: list[np.ndarray] = []
    Yr: list[np.ndarray] = []
    beta = None

    def solve(t: float) -> np.ndarray:
        age = t - np.asarray(T)
        om = np.asarray(Wr) * 0.5 ** (age / halflife) * np.asarray(learned)
        prior_w = ridge * 0.5 ** (t / halflife)
        za, ya = np.asarray(Z), np.nan_to_num(np.asarray(Yr))
        A = prior_w * np.eye(kt) + (om[:, None, None] * np.einsum("ri,rj->rij", za, za)).sum(axis=0)
        b = prior_w * prior.T + (om[:, None, None] * np.einsum("ri,rj->rij", za, ya)).sum(axis=0)
        return np.linalg.solve(A, b).T

    for i in range(n):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        d = min(dclock[i] + pending, max_dclock) if started else 0.0
        pending = 0.0
        if started:
            n_eff[i] = float((np.asarray(Wr) * 0.5 ** ((t_now - np.asarray(T)) / halflife)).sum())
        else:
            n_eff[i] = 0.0
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i].copy()
        if beta is not None and any(learned) and n_eff[i] >= min_periods:
            pred[i] = beta @ z
        t_now = t_now + d if started else 0.0
        started = True
        T.append(t_now)
        Wr.append(float(w[i]))
        learned.append(not np.isnan(Y[i]).any())
        Z.append(z)
        Yr.append(Y[i].copy())
        beta = solve(t_now)
        coef[i] = beta
    return {"pred": pred, "n_eff": n_eff, "coef": coef}


def _accepted_steps(X: np.ndarray, dclock: np.ndarray, max_dclock: float):
    """Yield ``(i, d)`` for each accepted row: a row with a null feature is
    skipped, its clock step folding into the next accepted row's, which
    ``max_dclock`` caps; the first accepted row's step is 0."""
    pending, started = 0.0, False
    for i in range(X.shape[0]):
        if np.isnan(X[i]).any():
            pending += dclock[i]
            continue
        d = min(dclock[i] + pending, max_dclock) if started else 0.0
        pending, started = 0.0, True
        yield i, d


def pa_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    *,
    mode: str = "pa1",
    c: float = 1.0,
    eps: float = 0.1,
    halflife: float = np.inf,
    add_intercept: bool = True,
    min_periods: float = 0.0,
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """Passive-aggressive regression, Crammer et al. (2006), as the ``pa``
    builder's docstring states it::

        p = z . b    loss = max(0, |y - p| - eps)    s = ||z||^2  (the intercept's 1 included)
        pa: tau = loss / s    pa1: tau = min(c, loss / s)    pa2: tau = loss / (s + 1 / (2c))
        b += min(w, 1) * tau * sign(y - p) * z

    per target, from zero. A null target or a zero weight moves nothing; the
    coefficients never decay, ``n_eff`` does. ``pred`` is null while
    ``n_eff`` is below ``min_periods``. Returns ``pred``, ``n_eff`` and
    ``coef`` (after the row)."""
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if add_intercept else k
    pred = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)
    b = np.zeros((m, kt))
    w_sum = 0.0
    for i, d in _accepted_steps(X, dclock, max_dclock):
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i]
        n_eff[i] = w_sum
        p = b @ z
        if w_sum >= min_periods:
            pred[i] = p
        s = z @ z
        for j in range(m):
            if np.isnan(Y[i, j]) or not w[i] > 0.0 or s <= 0.0:
                continue
            r = Y[i, j] - p[j]
            loss = max(0.0, abs(r) - eps)
            tau = {
                "pa": loss / s,
                "pa1": min(c, loss / s),
                "pa2": loss / (s + 1.0 / (2.0 * c)),
            }[mode]
            b[j] += min(w[i], 1.0) * tau * np.sign(r) * z
        lam = 1.0 if np.isinf(halflife) else 0.5 ** (d / halflife)
        w_sum = lam * w_sum + w[i]
        coef[i] = b
    return {"pred": pred, "n_eff": n_eff, "coef": coef}


def sgd_ref(
    X: np.ndarray,
    Y: np.ndarray,
    dclock: np.ndarray,
    w: np.ndarray,
    *,
    loss: str = "squared",
    learning_rate: float = 0.01,
    schedule: str = "constant",
    power: float = 0.5,
    huber_delta: float = 1.0,
    quantile: float = 0.5,
    eps: float = 0.1,
    halflife: float = np.inf,
    add_intercept: bool = True,
    min_periods: float = 0.0,
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """Stochastic gradient descent as the ``sgd`` builder's docstring states
    it, with ``l2 = 0`` and a gradient clip that never binds (both are
    worded too loosely there to be written from)::

        eta = z . b    p = link(eta)    d = dL / d eta
        squared: p - y                huber: clamp(p - y, +/- delta)
        quantile: 1{y < p} - tau      epsilon_insensitive: 0 inside the tube, else sign(p - y)
        poisson: exp(eta) - y         logistic: sigmoid(eta) - y
        g_i = d * z_i * w    b_i -= lr_i * g_i

    ``lr`` is ``learning_rate`` for ``"constant"``, ``learning_rate / (1 +
    n_eff) ** power`` for ``"inv_scaling"`` with ``n_eff`` the weight before
    the row, and ``learning_rate / (sqrt(G_i) + 1e-8)`` for ``"adagrad"``,
    ``G_i`` the sum of squared gradients with this row's in it. ``n_eff``
    and ``G`` decay on the clock; the coefficients do not. Per target, from
    zero; a null target or a zero weight moves nothing. ``pred`` is ``p``,
    null while ``n_eff`` is below ``min_periods``. Returns ``pred``,
    ``n_eff`` and ``coef`` (after the row)."""
    n, k = X.shape
    m = Y.shape[1]
    kt = k + 1 if add_intercept else k
    pred = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, kt), np.nan)
    b = np.zeros((m, kt))
    G = np.zeros((m, kt))
    w_sum = 0.0
    links = {
        "poisson": np.exp,
        "logistic": lambda e: 1.0 / (1.0 + np.exp(-e)),
    }
    for i, d in _accepted_steps(X, dclock, max_dclock):
        lam = 1.0 if np.isinf(halflife) else 0.5 ** (d / halflife)
        G *= lam
        z = np.concatenate(([1.0], X[i])) if add_intercept else X[i]
        n_eff[i] = w_sum
        p = links.get(loss, lambda e: e)(b @ z)
        if w_sum >= min_periods:
            pred[i] = p
        for j in range(m):
            if np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            e = p[j] - Y[i, j]
            dl = {
                "squared": e,
                "poisson": e,
                "logistic": e,
                "huber": min(max(e, -huber_delta), huber_delta),
                "quantile": float(Y[i, j] < p[j]) - quantile,
                "epsilon_insensitive": 0.0 if abs(e) <= eps else float(np.sign(e)),
            }[loss]
            g = dl * z * w[i]
            if schedule == "constant":
                lr = learning_rate
            elif schedule == "inv_scaling":
                lr = learning_rate / (1.0 + w_sum) ** power
            else:
                G[j] += g * g
                lr = learning_rate / (np.sqrt(G[j]) + 1e-8)
            b[j] -= lr * g
        w_sum = lam * w_sum + w[i]
        coef[i] = b
    return {"pred": pred, "n_eff": n_eff, "coef": coef}


def holt_ref(
    Y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    level_halflife: float,
    trend_halflife: float | None = None,
    min_periods: float = 0.0,
    max_dclock: float = np.inf,
) -> dict[str, np.ndarray]:
    """Holt's linear trend as the ``holt`` builder's docstring states it, per
    target, with ``s`` the clock since the target was last observed (this
    row's step included)::

        pred = l + b * s
        l'   = (lam_l * W * pred + w * y) / (lam_l * W + w)      W' = lam_l * W + w
        b'   = (lam_b * V * b + w * (l' - l) / s) / (lam_b * V + w)      V' = lam_b * V + w

    ``lam_l = 0.5 ** (s / level_halflife)`` and ``lam_b`` likewise at
    ``trend_halflife``, default four times the level's. The first
    observation sets the level; the second, at gain 1, the trend. A row at
    the last observation's clock (``s = 0``) is a second observation the
    level takes in, and the trend holds. A null target or a zero weight
    leaves ``l``, ``b``, ``W`` and ``V`` where they were and carries its
    clock to the next observation. ``pred`` is emitted from the second
    observation's row on, once the target's weight -- the rows it was
    present on, at their raw weights, decayed at the level halflife and seen
    before the row's own step -- reaches ``min_periods``; ``n_eff`` is every
    row's weight so decayed. Each clock step is capped at ``max_dclock``.
    Returns ``pred``, ``n_eff`` and ``coef`` (``[level, trend]`` after the
    row, NaN until the first observation)."""
    n, m = Y.shape
    th = 4.0 * level_halflife if trend_halflife is None else trend_halflife

    def lam(s: float, h: float) -> float:
        return 1.0 if np.isinf(h) else 0.5 ** (s / h)

    pred = np.full((n, m), np.nan)
    n_eff = np.full(n, np.nan)
    coef = np.full((n, m, 2), np.nan)
    level = np.full(m, np.nan)
    trend = np.zeros(m)
    W = np.zeros(m)
    V = np.zeros(m)
    since = np.zeros(m)
    seen = np.zeros(m, dtype=int)
    w_sum, w_own = 0.0, np.zeros(m)
    for i in range(n):
        d = 0.0 if i == 0 else min(max(t[i] - t[i - 1], 0.0), max_dclock)
        n_eff[i] = w_sum
        for j in range(m):
            since[j] += d
            s = since[j]
            if seen[j] >= 1 and w_own[j] >= min_periods and w_own[j] > 0.0:
                pred[i, j] = level[j] + trend[j] * s
            if np.isnan(Y[i, j]) or not w[i] > 0.0:
                continue
            if seen[j] == 0:
                level[j], W[j] = Y[i, j], w[i]
            else:
                p = level[j] + trend[j] * s
                ll = lam(s, level_halflife)
                new = (ll * W[j] * p + w[i] * Y[i, j]) / (ll * W[j] + w[i])
                W[j] = ll * W[j] + w[i]
                if s > 0.0:
                    lb = lam(s, th)
                    trend[j] = (lb * V[j] * trend[j] + w[i] * (new - level[j]) / s) / (
                        lb * V[j] + w[i]
                    )
                    V[j] = lb * V[j] + w[i]
                level[j] = new
            seen[j] += 1
            since[j] = 0.0
        step = lam(d, level_halflife)
        w_sum = step * w_sum + w[i]
        w_own = step * w_own + w[i] * ~np.isnan(Y[i])
        for j in range(m):
            if seen[j]:
                coef[i, j] = (level[j], trend[j])
    return {"pred": pred, "n_eff": n_eff, "coef": coef}
